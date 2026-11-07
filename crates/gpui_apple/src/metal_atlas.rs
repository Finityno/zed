use anyhow::{Context as _, Result, ensure};
use etagere::BucketedAtlasAllocator;
use gpui::{
    ATLAS_TILE_MAX_IDLE_FRAMES, AtlasBackend, AtlasKey, AtlasState, AtlasTextureId,
    AtlasTextureKind, AtlasTextureList, AtlasTile, Bounds, DevicePixels, PlatformAtlas, Point,
    RenderMemoryGauge, RenderMemoryLedger, Scene, Size,
};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::{
    MTLDevice, MTLOrigin, MTLPixelFormat, MTLRegion, MTLResource, MTLSize, MTLStorageMode,
    MTLTexture, MTLTextureDescriptor, MTLTextureUsage,
};
use parking_lot::Mutex;
use std::{borrow::Cow, ptr::NonNull};

pub struct MetalAtlas(Mutex<AtlasState<MetalAtlasTextures>>);

impl MetalAtlas {
    pub(crate) fn new(
        device: Retained<ProtocolObject<dyn MTLDevice>>,
        supports_shared_storage: bool,
    ) -> Self {
        MetalAtlas(Mutex::new(AtlasState::new(MetalAtlasTextures {
            device,
            supports_shared_storage,
            monochrome_textures: AtlasTextureList::default(),
            polychrome_textures: AtlasTextureList::default(),
            render_memory: RenderMemoryLedger::default(),
        })))
    }

    /// Returns the GPU texture backing `id`, or `None` once every tile in it
    /// has been removed. A scene can still reference such a texture when a
    /// cached view replays a paint from before the image was dropped, so
    /// callers must skip those sprites rather than assume the texture exists.
    pub(crate) fn metal_texture(
        &self,
        id: AtlasTextureId,
    ) -> Option<Retained<ProtocolObject<dyn MTLTexture>>> {
        Some(self.0.lock().backend.texture(id)?.metal_texture.clone())
    }

    /// The renderer's once-per-frame hook: marks every tile `scene` draws as used
    /// this frame, then lets one page's idle tiles go. Marking must
    /// come first so a tile drawn this very frame can never be the one retired.
    pub fn on_frame_drawn(&self, scene: &Scene) {
        let mut lock = self.0.lock();
        lock.note_frame_drawn(scene);
        lock.retire_unused(ATLAS_TILE_MAX_IDLE_FRAMES);
    }

    /// Marks `scene`'s tiles as used in the current frame without advancing
    /// it; see [`crate::metal_renderer::MetalRenderer::note_scene_tiles`].
    pub fn note_scene_tiles(&self, scene: &Scene) {
        self.0.lock().mark_scene_tiles(scene);
    }

    /// Device bytes the atlas's textures hold, as (monochrome, polychrome).
    /// Read by the renderer's GPU stats logging.
    pub(crate) fn allocated_bytes(&self) -> (u64, u64) {
        self.0.lock().backend.allocated_bytes()
    }
}

struct MetalAtlasTextures {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    supports_shared_storage: bool,
    monochrome_textures: AtlasTextureList<MetalAtlasTexture>,
    polychrome_textures: AtlasTextureList<MetalAtlasTexture>,
    /// What this atlas has reported into the process-wide render memory
    /// gauges; refreshed whenever a page is created or freed.
    render_memory: RenderMemoryLedger,
}

impl PlatformAtlas for MetalAtlas {
    fn get_or_insert_with<'a>(
        &self,
        key: AtlasKey,
        build: &mut dyn FnMut() -> Result<Option<(Size<DevicePixels>, Cow<'a, [u8]>)>>,
    ) -> Result<Option<AtlasTile>> {
        self.0.lock().get_or_insert_with(key, build)
    }

    fn remove(&self, key: &AtlasKey) {
        self.0.lock().remove(key);
    }

    fn note_frame_drawn(&self, scene: &Scene) {
        self.0.lock().note_frame_drawn(scene);
    }

    fn retire_unused(&self, max_idle_frames: u64) {
        self.0.lock().retire_unused(max_idle_frames);
    }

    fn frame_index(&self) -> u64 {
        self.0.lock().frame_index()
    }

    #[cfg(any(test, feature = "test-support", feature = "bench-support"))]
    fn contains(&self, key: &AtlasKey) -> bool {
        self.0.lock().contains(key)
    }
}

impl AtlasBackend for MetalAtlasTextures {
    fn insert(
        &mut self,
        kind: AtlasTextureKind,
        size: Size<DevicePixels>,
        bytes: &[u8],
    ) -> Result<AtlasTile> {
        let upload = AtlasUpload::new(kind, size, bytes)?;
        let tile = self.allocate(size, kind)?;
        let texture = self
            .texture(tile.texture_id)
            .context("looking up allocated atlas texture")?;
        texture.upload(tile.bounds.origin, &upload)?;
        Ok(tile)
    }

    fn remove(&mut self, tile: AtlasTile) {
        let id = tile.texture_id;
        let Some(textures) = self.textures_mut(id.kind) else {
            return;
        };
        let Some(texture_slot) = textures.textures.get_mut(id.index as usize) else {
            return;
        };

        let mut page_freed = false;
        if let Some(mut texture) = texture_slot.take() {
            texture.allocator.deallocate(tile.tile_id.into());
            texture.decrement_ref_count();
            if texture.is_unreferenced() {
                textures.free_list.push(id.index as usize);
                page_freed = true;
            } else {
                *texture_slot = Some(texture);
            }
        }
        if page_freed {
            self.publish_memory();
        }
    }
}

impl MetalAtlasTextures {
    fn textures(&self, kind: AtlasTextureKind) -> Option<&AtlasTextureList<MetalAtlasTexture>> {
        match kind {
            AtlasTextureKind::Monochrome => Some(&self.monochrome_textures),
            AtlasTextureKind::Polychrome => Some(&self.polychrome_textures),
            AtlasTextureKind::Subpixel => None,
        }
    }

    fn textures_mut(
        &mut self,
        kind: AtlasTextureKind,
    ) -> Option<&mut AtlasTextureList<MetalAtlasTexture>> {
        match kind {
            AtlasTextureKind::Monochrome => Some(&mut self.monochrome_textures),
            AtlasTextureKind::Polychrome => Some(&mut self.polychrome_textures),
            AtlasTextureKind::Subpixel => None,
        }
    }

    /// Device bytes the pages hold, as (monochrome, polychrome).
    fn allocated_bytes(&self) -> (u64, u64) {
        fn total(list: &AtlasTextureList<MetalAtlasTexture>) -> u64 {
            list.textures
                .iter()
                .flatten()
                .map(|texture| texture.metal_texture.allocatedSize() as u64)
                .sum()
        }
        (
            total(&self.monochrome_textures),
            total(&self.polychrome_textures),
        )
    }

    fn publish_memory(&mut self) {
        let (monochrome, polychrome) = self.allocated_bytes();
        self.render_memory
            .publish(RenderMemoryGauge::AtlasMonochrome, monochrome);
        self.render_memory
            .publish(RenderMemoryGauge::AtlasPolychrome, polychrome);
    }

    fn allocate(
        &mut self,
        size: Size<DevicePixels>,
        texture_kind: AtlasTextureKind,
    ) -> Result<AtlasTile> {
        {
            let textures = self
                .textures_mut(texture_kind)
                .context("Metal does not render subpixel sprites")?;
            if let Some(tile) = textures
                .iter_mut()
                .rev()
                .find_map(|texture| texture.allocate(size))
            {
                return Ok(tile);
            }
        }

        let texture = self.push_texture(size, texture_kind)?;
        texture.allocate(size).context("allocating atlas tile")
    }

    fn push_texture(
        &mut self,
        min_size: Size<DevicePixels>,
        kind: AtlasTextureKind,
    ) -> Result<&mut MetalAtlasTexture> {
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
        let pixel_format = match kind {
            AtlasTextureKind::Monochrome => MTLPixelFormat::A8Unorm,
            AtlasTextureKind::Polychrome => MTLPixelFormat::BGRA8Unorm,
            AtlasTextureKind::Subpixel => anyhow::bail!("Metal does not render subpixel sprites"),
        };
        let texture_descriptor = MTLTextureDescriptor::new();
        // SAFETY: Both dimensions are clamped to the existing 1024..=16384 atlas size limits.
        unsafe {
            texture_descriptor.setWidth(size.width.0 as usize);
            texture_descriptor.setHeight(size.height.0 as usize);
        }
        texture_descriptor.setPixelFormat(pixel_format);
        texture_descriptor.setUsage(MTLTextureUsage::ShaderRead);
        // Shared memory mode can be used only on Apple GPU families
        // https://developer.apple.com/documentation/metal/mtlresourceoptions/storagemodeshared
        texture_descriptor.setStorageMode(if self.supports_shared_storage {
            MTLStorageMode::Shared
        } else {
            MTLStorageMode::Managed
        });
        let metal_texture = self
            .device
            .newTextureWithDescriptor(&texture_descriptor)
            .context("allocating atlas texture")?;

        let texture_list = self
            .textures_mut(kind)
            .context("Metal does not render subpixel sprites")?;
        let index = texture_list.free_list.pop().unwrap_or_else(|| {
            let index = texture_list.textures.len();
            texture_list.textures.push(None);
            index
        });
        texture_list.textures[index] = Some(MetalAtlasTexture {
            id: AtlasTextureId {
                index: index as u32,
                kind,
            },
            allocator: BucketedAtlasAllocator::new(size_to_etagere(size)),
            metal_texture,
            live_atlas_keys: 0,
        });
        self.publish_memory();
        self.textures_mut(kind)
            .and_then(|list| list.textures.get_mut(index))
            .and_then(Option::as_mut)
            .context("looking up the atlas texture just created")
    }

    fn texture(&self, id: AtlasTextureId) -> Option<&MetalAtlasTexture> {
        self.textures(id.kind)?
            .textures
            .get(id.index as usize)?
            .as_ref()
    }
}

struct MetalAtlasTexture {
    id: AtlasTextureId,
    allocator: BucketedAtlasAllocator,
    metal_texture: Retained<ProtocolObject<dyn MTLTexture>>,
    live_atlas_keys: u32,
}

impl MetalAtlasTexture {
    fn allocate(&mut self, size: Size<DevicePixels>) -> Option<AtlasTile> {
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
        self.live_atlas_keys += 1;
        Some(tile)
    }

    fn upload(&self, origin: Point<DevicePixels>, upload: &AtlasUpload<'_>) -> Result<()> {
        ensure!(
            self.id.kind == upload.kind,
            "atlas upload kind does not match texture"
        );
        ensure!(
            origin.x.0 >= 0
                && origin.y.0 >= 0
                && (origin.x.0 as usize)
                    .checked_add(upload.size.width.0 as usize)
                    .is_some_and(|end| end <= self.metal_texture.width())
                && (origin.y.0 as usize)
                    .checked_add(upload.size.height.0 as usize)
                    .is_some_and(|end| end <= self.metal_texture.height()),
            "atlas upload is outside texture bounds"
        );
        let region = MTLRegion {
            origin: MTLOrigin {
                x: origin.x.0 as usize,
                y: origin.y.0 as usize,
                z: 0,
            },
            size: MTLSize {
                width: upload.size.width.0 as usize,
                height: upload.size.height.0 as usize,
                depth: 1,
            },
        };
        // SAFETY: Validated bytes cover every row of the checked region in this CPU-accessible texture.
        // The borrowed bytes remain live for the copy; GPU ordering is a separate caller requirement.
        unsafe {
            self.metal_texture
                .replaceRegion_mipmapLevel_withBytes_bytesPerRow(
                    region,
                    0,
                    NonNull::from(upload.bytes).cast(),
                    upload.bytes_per_row,
                );
        }
        Ok(())
    }

    fn decrement_ref_count(&mut self) {
        self.live_atlas_keys = self.live_atlas_keys.saturating_sub(1);
    }

    fn is_unreferenced(&self) -> bool {
        self.live_atlas_keys == 0
    }
}

struct AtlasUpload<'a> {
    kind: AtlasTextureKind,
    size: Size<DevicePixels>,
    bytes: &'a [u8],
    bytes_per_row: usize,
}

impl<'a> AtlasUpload<'a> {
    fn new(kind: AtlasTextureKind, size: Size<DevicePixels>, bytes: &'a [u8]) -> Result<Self> {
        ensure!(
            size.width.0 > 0 && size.height.0 > 0,
            "atlas dimensions must be positive"
        );
        let bytes_per_pixel = match kind {
            AtlasTextureKind::Monochrome => 1,
            AtlasTextureKind::Polychrome => 4,
            AtlasTextureKind::Subpixel => anyhow::bail!("Metal does not render subpixel sprites"),
        };
        let bytes_per_row = (size.width.0 as usize)
            .checked_mul(bytes_per_pixel)
            .context("calculating atlas row byte count")?;
        let byte_count = bytes_per_row
            .checked_mul(size.height.0 as usize)
            .context("calculating atlas byte count")?;
        ensure!(
            bytes.len() >= byte_count,
            "atlas upload byte length is {}, expected at least {byte_count}",
            bytes.len()
        );
        Ok(Self {
            kind,
            size,
            bytes,
            bytes_per_row,
        })
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

#[cfg(test)]
mod tests {
    use super::*;
    use gpui::{PlatformAtlas, point, size};
    use objc2_metal::{MTLCreateSystemDefaultDevice, MTLGPUFamily};
    use std::borrow::Cow;

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

        let tile_a = insert_tile(&atlas, key_a.clone(), small);
        let tile_b = insert_tile(&atlas, key_b.clone(), small);
        let tile_c = insert_tile(&atlas, key_c.clone(), small);

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
        let tile_a2 = insert_tile(&atlas, key_a, small);
        assert_eq!(tile_a2.texture_id, tile_a.texture_id);

        // The texture must actually exist — this would panic before the fix.
        assert!(atlas.metal_texture(tile_a2.texture_id).is_some());
    }

    #[test]
    fn test_metal_texture_is_none_after_last_tile_removed() {
        let Some(atlas) = create_atlas() else {
            return;
        };

        let key = make_image_key(1, 0);
        let tile = insert_tile(
            &atlas,
            key.clone(),
            Size {
                width: DevicePixels(64),
                height: DevicePixels(64),
            },
        );
        assert!(atlas.metal_texture(tile.texture_id).is_some());

        // A scene built before the removal may still carry `tile`; looking its
        // texture up must report the gap instead of panicking.
        atlas.remove(&key);
        assert!(atlas.metal_texture(tile.texture_id).is_none());
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

        let keeper_tile = insert_tile(&atlas, keeper_key, small);
        let tile_a = insert_tile(&atlas, big_key_a.clone(), big);
        assert_eq!(keeper_tile.texture_id, tile_a.texture_id);

        atlas.remove(&big_key_a);
        let tile_b = insert_tile(&atlas, big_key_b, big);
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

    #[test]
    fn test_invalid_upload_does_not_allocate() {
        let Some(atlas) = create_atlas() else {
            return;
        };
        let mut state = atlas.0.lock();
        for (kind, short_bytes, short_error) in [
            (
                AtlasTextureKind::Monochrome,
                &[][..],
                "atlas upload byte length is 0, expected at least 1",
            ),
            (
                AtlasTextureKind::Polychrome,
                &[0][..],
                "atlas upload byte length is 1, expected at least 4",
            ),
        ] {
            for (width, height, bytes, expected_error) in [
                (1, 1, short_bytes, short_error),
                (0, 1, &[][..], "atlas dimensions must be positive"),
                (1, 0, &[][..], "atlas dimensions must be positive"),
                (i32::MIN, 1, &[][..], "atlas dimensions must be positive"),
                (1, i32::MIN, &[][..], "atlas dimensions must be positive"),
            ] {
                assert_eq!(
                    state
                        .backend
                        .insert(kind, size(DevicePixels(width), DevicePixels(height)), bytes)
                        .err()
                        .map(|error| error.to_string()),
                    Some(expected_error.to_owned())
                );
                for textures in [
                    &state.backend.monochrome_textures,
                    &state.backend.polychrome_textures,
                ] {
                    assert_eq!((textures.textures.len(), textures.free_list.len()), (0, 0));
                }
            }
        }
    }

    #[test]
    fn test_texture_upload_preserves_pixel_layout() -> Result<()> {
        let Some(atlas) = create_atlas() else {
            return Ok(());
        };
        let mut state = atlas.0.lock();
        for (kind, bytes_per_pixel) in [
            (AtlasTextureKind::Monochrome, 1),
            (AtlasTextureKind::Polychrome, 4),
        ] {
            let texture = state
                .backend
                .push_texture(size(DevicePixels(8), DevicePixels(8)), kind)?;
            texture.upload(
                Point::default(),
                &AtlasUpload::new(
                    kind,
                    size(DevicePixels(8), DevicePixels(8)),
                    &vec![0; 8 * 8 * bytes_per_pixel],
                )?,
            )?;
            let pixels = (1..=3 * 2 * bytes_per_pixel)
                .map(|value| value as u8)
                .collect::<Vec<_>>();
            texture.upload(
                point(DevicePixels(2), DevicePixels(3)),
                &AtlasUpload::new(kind, size(DevicePixels(3), DevicePixels(2)), &pixels)?,
            )?;
            let mut actual = vec![0; 8 * 8 * bytes_per_pixel];
            unsafe {
                texture
                    .metal_texture
                    .getBytes_bytesPerRow_fromRegion_mipmapLevel(
                        NonNull::from(actual.as_mut_slice()).cast(),
                        8 * bytes_per_pixel,
                        MTLRegion {
                            origin: MTLOrigin { x: 0, y: 0, z: 0 },
                            size: MTLSize {
                                width: 8,
                                height: 8,
                                depth: 1,
                            },
                        },
                        0,
                    );
            }
            let mut expected = vec![0; actual.len()];
            for row in 0..2 {
                let start = ((row + 3) * 8 + 2) * bytes_per_pixel;
                let source = row * 3 * bytes_per_pixel;
                expected[start..start + 3 * bytes_per_pixel]
                    .copy_from_slice(&pixels[source..source + 3 * bytes_per_pixel]);
            }
            assert_eq!(actual, expected);
        }
        Ok(())
    }

    impl MetalAtlas {
        fn published_bytes(&self) -> (u64, u64) {
            let lock = self.0.lock();
            (
                lock.backend
                    .render_memory
                    .published(RenderMemoryGauge::AtlasMonochrome),
                lock.backend
                    .render_memory
                    .published(RenderMemoryGauge::AtlasPolychrome),
            )
        }

        fn contains_key(&self, key: &AtlasKey) -> bool {
            self.0.lock().contains(key)
        }

        fn texture_is_live(&self, id: AtlasTextureId) -> bool {
            self.0.lock().backend.texture(id).is_some()
        }
    }

    fn create_atlas() -> Option<MetalAtlas> {
        let device = MTLCreateSystemDefaultDevice()?;
        let supports_shared_storage =
            cfg!(target_os = "ios") || device.supportsFamily(MTLGPUFamily::Apple1);
        Some(MetalAtlas::new(device, supports_shared_storage))
    }

    fn make_image_key(image_id: usize, frame_index: usize) -> AtlasKey {
        AtlasKey::Image(gpui::RenderImageParams {
            image_id: gpui::ImageId(image_id),
            frame_index,
        })
    }

    fn insert_tile(atlas: &MetalAtlas, key: AtlasKey, size: Size<DevicePixels>) -> AtlasTile {
        atlas
            .get_or_insert_with(key, &mut || {
                let byte_count = (size.width.0 as usize) * (size.height.0 as usize) * 4;
                Ok(Some((size, Cow::Owned(vec![0u8; byte_count]))))
            })
            .expect("inserting atlas tile")
            .expect("retrieving inserted atlas tile")
    }


    #[test]
    fn test_atlas_pages_are_reported_to_the_render_memory_gauges() {
        let Some(atlas) = create_atlas() else {
            return;
        };
        const MIB: u64 = 1024 * 1024;
        let key = make_image_key(1, 0);
        let small = Size {
            width: DevicePixels(64),
            height: DevicePixels(64),
        };
        insert_tile(&atlas, key.clone(), small);

        let (monochrome, polychrome) = atlas.allocated_bytes();
        assert_eq!(monochrome, 0);
        assert!(
            polychrome >= 4 * MIB,
            "a default 1024² BGRA8 page holds 4 MiB, got {polychrome}"
        );
        assert_eq!(atlas.published_bytes(), (0, polychrome));
        // The gauge sums every live atlas in the process, so it holds at
        // least this one's page whatever the other tests are doing.
        assert!(gpui::render_memory_gauges().atlas_polychrome_bytes >= polychrome);

        // Removing the only tile frees the page, and the report follows.
        atlas.remove(&key);
        assert_eq!(atlas.allocated_bytes(), (0, 0));
        assert_eq!(atlas.published_bytes(), (0, 0));
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
                rounded_clip: 0,
                clip_padding: 0,
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
        let hot = insert_tile(&atlas, hot_key.clone(), small);

        // Well past the idle window, but the scene names the tile every frame.
        run_frames(&atlas, IDLE * 3, &[hot]);

        assert!(atlas.contains_key(&hot_key));
        assert!(atlas.texture_is_live(hot.texture_id));
        let mut rebuilt = false;
        let again = atlas
            .get_or_insert_with(hot_key, &mut || {
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
        let hot = insert_tile(&atlas, hot_key.clone(), small);
        let cold = insert_tile(&atlas, cold_key.clone(), small);
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
        let reinserted = insert_tile(&atlas, cold_key, small);
        assert!(atlas.texture_is_live(reinserted.texture_id));
        assert!(atlas.metal_texture(reinserted.texture_id).is_some());
    }

    #[test]
    fn test_idle_image_tiles_are_retired() {
        let Some(atlas) = create_atlas() else {
            return;
        };
        let small = Size {
            width: DevicePixels(16),
            height: DevicePixels(16),
        };
        let image_key = make_image_key(7, 0);
        let image = insert_tile(&atlas, image_key.clone(), small);
        let glyph_key = make_glyph_key(3);
        let glyph = insert_tile(&atlas, glyph_key.clone(), small);

        run_frames(&atlas, IDLE * 3, &[]);

        // Nothing draws either: both retire, images like glyphs.
        assert!(!atlas.contains_key(&image_key));
        assert!(!atlas.texture_is_live(image.texture_id));
        assert!(!atlas.contains_key(&glyph_key));
        assert!(!atlas.texture_is_live(glyph.texture_id));
    }
}
