use anyhow::{Context as _, Result};
use collections::FxHashMap;
use derive_more::{Deref, DerefMut};
use etagere::BucketedAtlasAllocator;
use gpui::{
    ATLAS_TILE_MAX_IDLE_FRAMES, AtlasKey, AtlasTextureId, AtlasTextureKind, AtlasTextureList,
    AtlasTile, Bounds, DevicePixels, PlatformAtlas, Point, Scene, Size, TileId,
};
use metal::Device;
use parking_lot::Mutex;
use std::borrow::Cow;

pub struct MetalAtlas(Mutex<MetalAtlasState>);

impl MetalAtlas {
    pub(crate) fn new(device: Device, is_apple_gpu: bool) -> Self {
        MetalAtlas(Mutex::new(MetalAtlasState {
            device: AssertSend(device),
            is_apple_gpu,
            monochrome_textures: Default::default(),
            polychrome_textures: Default::default(),
            tiles_by_key: Default::default(),
            frame: 0,
            retire_cursor: 0,
        }))
    }

    pub(crate) fn metal_texture(&self, id: AtlasTextureId) -> metal::Texture {
        self.0.lock().texture(id).metal_texture.clone()
    }

    /// The renderer's once-per-frame hook: marks every tile `scene` draws as used
    /// this frame, then lets one page's idle glyph and SVG tiles go. Marking must
    /// come first so a tile drawn this very frame can never be the one retired.
    pub fn on_frame_drawn(&self, scene: &Scene) {
        self.note_frame_drawn(scene);
        self.retire_unused(ATLAS_TILE_MAX_IDLE_FRAMES);
    }

    /// Marks `scene`'s tiles as used in the current frame without advancing
    /// it; see [`MetalRenderer::note_scene_tiles`].
    pub fn note_scene_tiles(&self, scene: &Scene) {
        self.0.lock().mark_scene_tiles(scene);
    }
}

impl MetalAtlas {
    /// Device bytes the atlas's textures hold, as (monochrome, polychrome).
    /// Read by the renderer's GPU stats logging.
    pub(crate) fn allocated_bytes(&self) -> (u64, u64) {
        fn total(list: &AtlasTextureList<MetalAtlasTexture>) -> u64 {
            list.textures
                .iter()
                .flatten()
                .map(|texture| texture.metal_texture.allocated_size())
                .sum()
        }
        let lock = self.0.lock();
        (
            total(&lock.monochrome_textures),
            total(&lock.polychrome_textures),
        )
    }

    #[cfg(test)]
    fn contains_key(&self, key: &AtlasKey) -> bool {
        self.0.lock().tiles_by_key.contains_key(key)
    }

    #[cfg(test)]
    fn texture_is_live(&self, id: AtlasTextureId) -> bool {
        let lock = self.0.lock();
        lock.textures(id.kind)
            .textures
            .get(id.index as usize)
            .is_some_and(|slot| slot.is_some())
    }
}

struct MetalAtlasState {
    device: AssertSend<Device>,
    is_apple_gpu: bool,
    monochrome_textures: AtlasTextureList<MetalAtlasTexture>,
    polychrome_textures: AtlasTextureList<MetalAtlasTexture>,
    tiles_by_key: FxHashMap<AtlasKey, AtlasTile>,
    /// Frames seen through `note_frame_drawn`. Shared across every window drawing
    /// with this atlas, which is why `Window` compares against it before replaying
    /// a retained scene.
    frame: u64,
    /// Position in the concatenated (monochrome, polychrome) page list that the
    /// next `retire_unused` call examines.
    retire_cursor: usize,
}

impl PlatformAtlas for MetalAtlas {
    fn get_or_insert_with<'a>(
        &self,
        key: &AtlasKey,
        build: &mut dyn FnMut() -> Result<Option<(Size<DevicePixels>, Cow<'a, [u8]>)>>,
    ) -> Result<Option<AtlasTile>> {
        let mut lock = self.0.lock();
        if let Some(tile) = lock.tiles_by_key.get(key).copied() {
            let frame = lock.frame;
            lock.touch(tile, frame);
            Ok(Some(tile))
        } else {
            let Some((size, bytes)) = build()? else {
                return Ok(None);
            };
            let tile = lock.allocate(size, key).context("failed to allocate")?;
            let texture = lock.texture(tile.texture_id);
            texture.upload(tile.bounds, &bytes);
            lock.tiles_by_key.insert(key.clone(), tile);
            Ok(Some(tile))
        }
    }

    fn remove(&self, key: &AtlasKey) {
        let mut lock = self.0.lock();
        let Some(tile) = lock.tiles_by_key.remove(key) else {
            return;
        };
        lock.release_tile(tile);
    }

    fn note_frame_drawn(&self, scene: &Scene) {
        let mut lock = self.0.lock();
        lock.frame += 1;
        lock.mark_scene_tiles(scene);
    }

    fn retire_unused(&self, max_idle_frames: u64) {
        let mut lock = self.0.lock();
        let monochrome_count = lock.monochrome_textures.textures.len();
        let page_count = monochrome_count + lock.polychrome_textures.textures.len();
        if page_count == 0 {
            return;
        }
        let cursor = lock.retire_cursor % page_count;
        lock.retire_cursor = cursor + 1;
        let (kind, index) = if cursor < monochrome_count {
            (AtlasTextureKind::Monochrome, cursor)
        } else {
            (AtlasTextureKind::Polychrome, cursor - monochrome_count)
        };

        let frame = lock.frame;
        let Some(Some(texture)) = lock.textures(kind).textures.get(index) else {
            return;
        };
        let idle_keys: Vec<AtlasKey> = texture
            .tiles
            .values()
            .filter(|record| {
                matches!(record.key, AtlasKey::Glyph(_) | AtlasKey::Svg(_))
                    && frame.saturating_sub(record.last_used_frame) >= max_idle_frames
            })
            .map(|record| record.key.clone())
            .collect();
        if idle_keys.is_empty() {
            return;
        }
        let retired = idle_keys.len();
        for key in idle_keys {
            if let Some(tile) = lock.tiles_by_key.remove(&key) {
                lock.release_tile(tile);
            }
        }
        let page_freed = lock
            .textures(kind)
            .textures
            .get(index)
            .is_some_and(|slot| slot.is_none());
        log::debug!(
            "[atlas] retired {retired} idle {kind:?} tile(s) from page {index} at frame {frame}{}",
            if page_freed { "; page freed" } else { "" }
        );
    }

    fn frame_index(&self) -> u64 {
        self.0.lock().frame
    }
}

impl MetalAtlasState {
    /// Records every tile `scene` references as used in the current frame.
    fn mark_scene_tiles(&mut self, scene: &Scene) {
        let frame = self.frame;
        // `Scene::finish` sorts each sprite list by tile id within a draw order, so
        // a glyph repeated across a line of text collapses to one lookup here.
        let mut previous: Option<(AtlasTextureId, TileId)> = None;
        let tiles = scene
            .monochrome_sprites
            .iter()
            .map(|sprite| sprite.tile)
            .chain(scene.subpixel_sprites.iter().map(|sprite| sprite.tile))
            .chain(scene.polychrome_sprites.iter().map(|sprite| sprite.tile));
        for tile in tiles {
            let identity = (tile.texture_id, tile.tile_id);
            if previous == Some(identity) {
                continue;
            }
            previous = Some(identity);
            self.touch(tile, frame);
        }
    }

    fn textures(&self, kind: AtlasTextureKind) -> &AtlasTextureList<MetalAtlasTexture> {
        match kind {
            AtlasTextureKind::Monochrome => &self.monochrome_textures,
            AtlasTextureKind::Polychrome => &self.polychrome_textures,
            AtlasTextureKind::Subpixel => unreachable!(),
        }
    }

    fn textures_mut(
        &mut self,
        kind: AtlasTextureKind,
    ) -> &mut AtlasTextureList<MetalAtlasTexture> {
        match kind {
            AtlasTextureKind::Monochrome => &mut self.monochrome_textures,
            AtlasTextureKind::Polychrome => &mut self.polychrome_textures,
            AtlasTextureKind::Subpixel => unreachable!(),
        }
    }

    fn touch(&mut self, tile: AtlasTile, frame: u64) {
        let id = tile.texture_id;
        // A scene can name a tile this atlas no longer holds (a retained scene
        // outliving the idle window before `Window` catches it); there is nothing
        // to mark then, and the draw already reads whatever the slot holds.
        if let Some(record) = self
            .textures_mut(id.kind)
            .textures
            .get_mut(id.index as usize)
            .and_then(|slot| slot.as_mut())
            .and_then(|texture| texture.tiles.get_mut(&tile.tile_id))
        {
            record.last_used_frame = frame;
        }
    }

    /// Returns a tile's space to its page and drops the page once it holds nothing.
    /// The caller removes the `tiles_by_key` entry.
    fn release_tile(&mut self, tile: AtlasTile) {
        let id = tile.texture_id;
        let textures = self.textures_mut(id.kind);
        let Some(texture_slot) = textures.textures.get_mut(id.index as usize) else {
            return;
        };
        if let Some(mut texture) = texture_slot.take() {
            if texture.tiles.remove(&tile.tile_id).is_some() {
                texture.allocator.deallocate(tile.tile_id.into());
            }
            if texture.tiles.is_empty() {
                textures.free_list.push(id.index as usize);
            } else {
                *texture_slot = Some(texture);
            }
        }
    }

    fn allocate(&mut self, size: Size<DevicePixels>, key: &AtlasKey) -> Option<AtlasTile> {
        let frame = self.frame;
        let texture_kind = key.texture_kind();
        if let Some(tile) = self
            .textures_mut(texture_kind)
            .iter_mut()
            .rev()
            .find_map(|texture| texture.allocate(size, key, frame))
        {
            return Some(tile);
        }

        let texture = self.push_texture(size, texture_kind);
        texture.allocate(size, key, frame)
    }

    fn push_texture(
        &mut self,
        min_size: Size<DevicePixels>,
        kind: AtlasTextureKind,
    ) -> &mut MetalAtlasTexture {
        const DEFAULT_ATLAS_SIZE: Size<DevicePixels> = Size {
            width: DevicePixels(1024),
            height: DevicePixels(1024),
        };
        // Max texture size on all modern Apple GPUs. Anything bigger than that crashes in validateWithDevice.
        const MAX_ATLAS_SIZE: Size<DevicePixels> = Size {
            width: DevicePixels(16384),
            height: DevicePixels(16384),
        };
        let size = min_size.min(&MAX_ATLAS_SIZE).max(&DEFAULT_ATLAS_SIZE);
        let texture_descriptor = metal::TextureDescriptor::new();
        texture_descriptor.set_width(size.width.into());
        texture_descriptor.set_height(size.height.into());
        let pixel_format;
        let usage;
        match kind {
            AtlasTextureKind::Monochrome => {
                pixel_format = metal::MTLPixelFormat::A8Unorm;
                usage = metal::MTLTextureUsage::ShaderRead;
            }
            AtlasTextureKind::Polychrome => {
                pixel_format = metal::MTLPixelFormat::BGRA8Unorm;
                usage = metal::MTLTextureUsage::ShaderRead;
            }
            AtlasTextureKind::Subpixel => unreachable!(),
        }
        texture_descriptor.set_pixel_format(pixel_format);
        texture_descriptor.set_usage(usage);
        // Shared memory mode can be used only on Apple GPU families
        // https://developer.apple.com/documentation/metal/mtlresourceoptions/storagemodeshared
        texture_descriptor.set_storage_mode(if self.is_apple_gpu {
            metal::MTLStorageMode::Shared
        } else {
            metal::MTLStorageMode::Managed
        });
        let metal_texture = self.device.new_texture(&texture_descriptor);

        let texture_list = self.textures_mut(kind);

        let index = texture_list.free_list.pop();

        let atlas_texture = MetalAtlasTexture {
            id: AtlasTextureId {
                index: index.unwrap_or(texture_list.textures.len()) as u32,
                kind,
            },
            allocator: etagere::BucketedAtlasAllocator::new(size_to_etagere(size)),
            metal_texture: AssertSend(metal_texture),
            tiles: FxHashMap::default(),
        };

        if let Some(ix) = index {
            texture_list.textures[ix] = Some(atlas_texture);
            texture_list.textures.get_mut(ix)
        } else {
            texture_list.textures.push(Some(atlas_texture));
            texture_list.textures.last_mut()
        }
        .unwrap()
        .as_mut()
        .unwrap()
    }

    fn texture(&self, id: AtlasTextureId) -> &MetalAtlasTexture {
        self.textures(id.kind)[id.index as usize].as_ref().unwrap()
    }
}

struct TileRecord {
    key: AtlasKey,
    last_used_frame: u64,
}

struct MetalAtlasTexture {
    id: AtlasTextureId,
    allocator: BucketedAtlasAllocator,
    metal_texture: AssertSend<metal::Texture>,
    /// Every live tile on this page. Empty means the page can be dropped.
    tiles: FxHashMap<TileId, TileRecord>,
}

impl MetalAtlasTexture {
    fn allocate(&mut self, size: Size<DevicePixels>, key: &AtlasKey, frame: u64) -> Option<AtlasTile> {
        let allocation = self.allocator.allocate(size_to_etagere(size))?;
        let tile = AtlasTile {
            texture_id: self.id,
            tile_id: allocation.id.into(),
            bounds: Bounds {
                origin: point_from_etagere(allocation.rectangle.min),
                size,
            },
            padding: 0,
        };
        self.tiles.insert(
            tile.tile_id,
            TileRecord {
                key: key.clone(),
                last_used_frame: frame,
            },
        );
        Some(tile)
    }

    fn upload(&self, bounds: Bounds<DevicePixels>, bytes: &[u8]) {
        let region = metal::MTLRegion::new_2d(
            bounds.origin.x.into(),
            bounds.origin.y.into(),
            bounds.size.width.into(),
            bounds.size.height.into(),
        );
        self.metal_texture.replace_region(
            region,
            0,
            bytes.as_ptr() as *const _,
            bounds.size.width.to_bytes(self.bytes_per_pixel()) as u64,
        );
    }

    fn bytes_per_pixel(&self) -> u8 {
        use metal::MTLPixelFormat::*;
        match self.metal_texture.pixel_format() {
            A8Unorm | R8Unorm => 1,
            RGBA8Unorm | BGRA8Unorm => 4,
            _ => unimplemented!(),
        }
    }
}

fn size_to_etagere(size: Size<DevicePixels>) -> etagere::Size {
    etagere::Size::new(size.width.into(), size.height.into())
}

fn point_from_etagere(value: etagere::Point) -> Point<DevicePixels> {
    Point {
        x: DevicePixels::from(value.x),
        y: DevicePixels::from(value.y),
    }
}

#[derive(Deref, DerefMut)]
struct AssertSend<T>(T);

unsafe impl<T> Send for AssertSend<T> {}

#[cfg(test)]
mod tests {
    use super::*;
    use gpui::PlatformAtlas;
    use std::borrow::Cow;

    fn create_atlas() -> Option<MetalAtlas> {
        let device = metal::Device::system_default()?;
        Some(MetalAtlas::new(device, true))
    }

    fn make_image_key(image_id: usize, frame_index: usize) -> AtlasKey {
        AtlasKey::Image(gpui::RenderImageParams {
            image_id: gpui::ImageId(image_id),
            frame_index,
        })
    }

    fn insert_tile(atlas: &MetalAtlas, key: &AtlasKey, size: Size<DevicePixels>) -> AtlasTile {
        atlas
            .get_or_insert_with(key, &mut || {
                let byte_count = (size.width.0 as usize) * (size.height.0 as usize) * 4;
                Ok(Some((size, Cow::Owned(vec![0u8; byte_count]))))
            })
            .expect("allocation should succeed")
            .expect("callback returns Some")
    }

    #[test]
    fn test_remove_clears_stale_keys_from_tiles_by_key() {
        let Some(atlas) = create_atlas() else {
            return;
        };

        let small = Size {
            width: DevicePixels(64),
            height: DevicePixels(64),
        };

        let key_a = make_image_key(1, 0);
        let key_b = make_image_key(2, 0);
        let key_c = make_image_key(3, 0);

        let tile_a = insert_tile(&atlas, &key_a, small);
        let tile_b = insert_tile(&atlas, &key_b, small);
        let tile_c = insert_tile(&atlas, &key_c, small);

        assert_eq!(tile_a.texture_id, tile_b.texture_id);
        assert_eq!(tile_b.texture_id, tile_c.texture_id);

        // Remove A: texture still has B and C, so it stays.
        // The key for A must be removed from tiles_by_key.
        atlas.remove(&key_a);

        // Remove B: texture still has C.
        atlas.remove(&key_b);

        // Remove C: texture becomes unreferenced and is deleted.
        atlas.remove(&key_c);

        // Re-inserting A must allocate a fresh tile on a new texture,
        // NOT return a stale tile referencing the deleted texture.
        let tile_a2 = insert_tile(&atlas, &key_a, small);

        // The texture must actually exist — this would panic before the fix.
        let _texture = atlas.metal_texture(tile_a2.texture_id);
    }

    #[test]
    fn test_remove_deallocates_tile_space_for_reuse() {
        let Some(atlas) = create_atlas() else {
            return;
        };

        let small = Size {
            width: DevicePixels(64),
            height: DevicePixels(64),
        };
        let big = Size {
            width: DevicePixels(700),
            height: DevicePixels(700),
        };

        let keeper_key = make_image_key(1, 0);
        let big_key_a = make_image_key(2, 0);
        let big_key_b = make_image_key(3, 0);

        let keeper_tile = insert_tile(&atlas, &keeper_key, small);
        let tile_a = insert_tile(&atlas, &big_key_a, big);
        assert_eq!(keeper_tile.texture_id, tile_a.texture_id);

        atlas.remove(&big_key_a);
        let tile_b = insert_tile(&atlas, &big_key_b, big);
        assert_eq!(tile_b.texture_id, keeper_tile.texture_id);
    }

    #[test]
    fn test_remove_nonexistent_key_is_noop() {
        let Some(atlas) = create_atlas() else {
            return;
        };
        let key = make_image_key(999, 0);
        atlas.remove(&key);
    }

    fn make_glyph_key(glyph_id: u32) -> AtlasKey {
        AtlasKey::Glyph(gpui::RenderGlyphParams {
            font_id: gpui::FontId(0),
            glyph_id: gpui::GlyphId(glyph_id),
            font_size: gpui::px(14.0),
            subpixel_variant: Default::default(),
            scale_factor: 2.0,
            is_emoji: false,
            subpixel_rendering: false,
            dilation: 0,
        })
    }

    fn scene_drawing(tiles: &[AtlasTile]) -> Scene {
        let mut scene = Scene::default();
        for tile in tiles {
            scene.monochrome_sprites.push(gpui::MonochromeSprite {
                order: 0,
                pad: 0,
                bounds: Default::default(),
                content_mask: Default::default(),
                color: gpui::transparent_black(),
                effect: Default::default(),
                tile: *tile,
                transformation: gpui::TransformationMatrix::unit(),
            });
        }
        scene
    }

    /// Runs `frames` frames drawing `drawn`, retiring after each like the renderer.
    fn run_frames(atlas: &MetalAtlas, frames: u64, drawn: &[AtlasTile]) {
        let scene = scene_drawing(drawn);
        for _ in 0..frames {
            atlas.on_frame_drawn(&scene);
        }
    }

    const IDLE: u64 = ATLAS_TILE_MAX_IDLE_FRAMES;

    #[test]
    fn test_tile_drawn_this_frame_is_not_retired() {
        let Some(atlas) = create_atlas() else {
            return;
        };
        let small = Size {
            width: DevicePixels(16),
            height: DevicePixels(16),
        };
        let hot_key = make_glyph_key(1);
        let hot = insert_tile(&atlas, &hot_key, small);

        // Well past the idle window, but the scene names the tile every frame.
        run_frames(&atlas, IDLE * 3, &[hot]);

        assert!(atlas.contains_key(&hot_key));
        assert!(atlas.texture_is_live(hot.texture_id));
        let mut rebuilt = false;
        let again = atlas
            .get_or_insert_with(&hot_key, &mut || {
                rebuilt = true;
                Ok(None)
            })
            .expect("lookup succeeds");
        assert!(!rebuilt, "a hot tile must be served from the atlas");
        assert_eq!(again, Some(hot));
    }

    #[test]
    fn test_idle_glyph_tile_is_retired_and_empty_page_freed() {
        let Some(atlas) = create_atlas() else {
            return;
        };
        let small = Size {
            width: DevicePixels(16),
            height: DevicePixels(16),
        };
        let hot_key = make_glyph_key(1);
        let cold_key = make_glyph_key(2);
        let hot = insert_tile(&atlas, &hot_key, small);
        let cold = insert_tile(&atlas, &cold_key, small);
        assert_eq!(hot.texture_id, cold.texture_id);

        // One frame short of the window: the cold tile survives.
        run_frames(&atlas, IDLE - 1, &[hot]);
        assert!(atlas.contains_key(&cold_key));

        // The page holds a live tile, so it stays even once `cold` goes.
        run_frames(&atlas, 2, &[hot]);
        assert!(!atlas.contains_key(&cold_key));
        assert!(atlas.contains_key(&hot_key));
        assert!(atlas.texture_is_live(hot.texture_id));

        // Stop drawing the hot tile too: it retires and the page is released.
        run_frames(&atlas, IDLE + 1, &[]);
        assert!(!atlas.contains_key(&hot_key));
        assert!(!atlas.texture_is_live(hot.texture_id));

        // The next glyph gets a fresh page rather than a stale tile.
        let reinserted = insert_tile(&atlas, &cold_key, small);
        assert!(atlas.texture_is_live(reinserted.texture_id));
        let _texture = atlas.metal_texture(reinserted.texture_id);
    }

    #[test]
    fn test_image_tiles_are_never_retired() {
        let Some(atlas) = create_atlas() else {
            return;
        };
        let small = Size {
            width: DevicePixels(16),
            height: DevicePixels(16),
        };
        let image_key = make_image_key(7, 0);
        let image = insert_tile(&atlas, &image_key, small);
        let glyph_key = make_glyph_key(3);
        let glyph = insert_tile(&atlas, &glyph_key, small);

        run_frames(&atlas, IDLE * 3, &[]);

        assert!(atlas.contains_key(&image_key));
        assert!(atlas.texture_is_live(image.texture_id));
        assert!(!atlas.contains_key(&glyph_key));
        assert!(!atlas.texture_is_live(glyph.texture_id));

        // Only `drop_image` releases it.
        atlas.remove(&image_key);
        assert!(!atlas.contains_key(&image_key));
        assert!(!atlas.texture_is_live(image.texture_id));
    }
}
