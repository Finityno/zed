use collections::FxHashMap;
use etagere::BucketedAtlasAllocator;
use parking_lot::Mutex;
use windows::Win32::Graphics::{
    Direct3D11::{
        D3D11_BIND_SHADER_RESOURCE, D3D11_BOX, D3D11_TEXTURE2D_DESC, D3D11_USAGE_DEFAULT,
        ID3D11Device, ID3D11DeviceContext, ID3D11ShaderResourceView, ID3D11Texture2D,
    },
    Dxgi::Common::*,
};

use gpui::{
    AtlasKey, AtlasTextureId, AtlasTextureKind, AtlasTextureList, AtlasTile, Bounds, DevicePixels,
    PlatformAtlas, Point, Size,
};

pub(crate) struct DirectXAtlas(Mutex<DirectXAtlasState>);

struct DirectXAtlasState {
    device: ID3D11Device,
    device_context: ID3D11DeviceContext,
    monochrome_textures: AtlasTextureList<DirectXAtlasTexture>,
    polychrome_textures: AtlasTextureList<DirectXAtlasTexture>,
    subpixel_textures: AtlasTextureList<DirectXAtlasTexture>,
    tiles_by_key: FxHashMap<AtlasKey, AtlasTile>,
}

struct DirectXAtlasTexture {
    id: AtlasTextureId,
    bytes_per_pixel: u32,
    allocator: BucketedAtlasAllocator,
    texture: ID3D11Texture2D,
    view: [Option<ID3D11ShaderResourceView>; 1],
    live_atlas_keys: u32,
}

impl DirectXAtlas {
    pub(crate) fn new(device: &ID3D11Device, device_context: &ID3D11DeviceContext) -> Self {
        DirectXAtlas(Mutex::new(DirectXAtlasState {
            device: device.clone(),
            device_context: device_context.clone(),
            monochrome_textures: Default::default(),
            polychrome_textures: Default::default(),
            subpixel_textures: Default::default(),
            tiles_by_key: Default::default(),
        }))
    }

    pub(crate) fn get_texture_view(
        &self,
        id: AtlasTextureId,
    ) -> [Option<ID3D11ShaderResourceView>; 1] {
        let lock = self.0.lock();
        let tex = lock.texture(id);
        tex.view.clone()
    }

    pub(crate) fn handle_device_lost(
        &self,
        device: &ID3D11Device,
        device_context: &ID3D11DeviceContext,
    ) {
        let mut lock = self.0.lock();
        lock.device = device.clone();
        lock.device_context = device_context.clone();
        lock.monochrome_textures = AtlasTextureList::default();
        lock.polychrome_textures = AtlasTextureList::default();
        lock.subpixel_textures = AtlasTextureList::default();
        lock.tiles_by_key.clear();
    }
}

impl PlatformAtlas for DirectXAtlas {
    fn get_or_insert_with<'a>(
        &self,
        key: &AtlasKey,
        build: &mut dyn FnMut() -> anyhow::Result<
            Option<(Size<DevicePixels>, std::borrow::Cow<'a, [u8]>)>,
        >,
    ) -> anyhow::Result<Option<AtlasTile>> {
        let mut lock = self.0.lock();
        if let Some(tile) = lock.tiles_by_key.get(key) {
            Ok(Some(*tile))
        } else {
            let Some((size, bytes)) = build()? else {
                return Ok(None);
            };
            // Creating a texture or uploading into one on a removed device
            // goes through the vendor driver, which is what the renderer is
            // parked to avoid (see `DirectXRenderer::quiesce_if_device_lost`).
            // Nothing is cached on this path, so the glyph is simply retried
            // once `handle_device_lost` has installed the new device.
            if lock.device_is_lost() {
                anyhow::bail!("sprite atlas upload skipped: the DirectX device has been removed");
            }
            let tile = lock
                .allocate(size, key.texture_kind())
                .ok_or_else(|| anyhow::anyhow!("failed to allocate"))?;
            let texture = lock.texture(tile.texture_id);
            let uploaded = texture.upload(&lock.device_context, tile.bounds, &bytes);
            if !uploaded {
                // The tile is on the GPU but blank. Caching it would make this glyph
                // invisible for the lifetime of the process while its advance stayed in the
                // layout — a character silently missing from the middle of a word. Give the
                // space back and report failure so the next frame retries instead.
                lock.deallocate_tile(tile);
                anyhow::bail!(
                    "failed to upload a {}x{} tile to the sprite atlas",
                    tile.bounds.size.width.0,
                    tile.bounds.size.height.0,
                );
            }
            lock.tiles_by_key.insert(key.clone(), tile);
            Ok(Some(tile))
        }
    }

    fn remove(&self, key: &AtlasKey) {
        let mut lock = self.0.lock();

        let Some(tile) = lock.tiles_by_key.remove(key) else {
            return;
        };
        lock.deallocate_tile(tile);
    }
}

impl DirectXAtlasState {
    /// See `DirectXRenderer::device_is_lost`: a runtime-level query that is
    /// safe on a removed device, unlike the texture calls below it.
    fn device_is_lost(&self) -> bool {
        unsafe { self.device.GetDeviceRemovedReason() }.is_err()
    }

    /// Returns a tile's space to its texture, freeing the texture itself once nothing
    /// references it. The caller is responsible for dropping the `tiles_by_key` entry.
    fn deallocate_tile(&mut self, tile: AtlasTile) {
        let id = tile.texture_id;
        let textures = match id.kind {
            AtlasTextureKind::Monochrome => &mut self.monochrome_textures,
            AtlasTextureKind::Polychrome => &mut self.polychrome_textures,
            AtlasTextureKind::Subpixel => &mut self.subpixel_textures,
        };

        let Some(texture_slot) = textures.textures.get_mut(id.index as usize) else {
            return;
        };

        if let Some(mut texture) = texture_slot.take() {
            texture.allocator.deallocate(tile.tile_id.into());
            texture.decrement_ref_count();
            if texture.is_unreferenced() {
                textures.free_list.push(texture.id.index as usize);
            } else {
                *texture_slot = Some(texture);
            }
        }
    }

    fn allocate(
        &mut self,
        size: Size<DevicePixels>,
        texture_kind: AtlasTextureKind,
    ) -> Option<AtlasTile> {
        {
            let textures = match texture_kind {
                AtlasTextureKind::Monochrome => &mut self.monochrome_textures,
                AtlasTextureKind::Polychrome => &mut self.polychrome_textures,
                AtlasTextureKind::Subpixel => &mut self.subpixel_textures,
            };

            if let Some(tile) = textures
                .iter_mut()
                .rev()
                .find_map(|texture| texture.allocate(size))
            {
                return Some(tile);
            }
        }

        let texture = self.push_texture(size, texture_kind)?;
        texture.allocate(size)
    }

    fn push_texture(
        &mut self,
        min_size: Size<DevicePixels>,
        kind: AtlasTextureKind,
    ) -> Option<&mut DirectXAtlasTexture> {
        const DEFAULT_ATLAS_SIZE: Size<DevicePixels> = Size {
            width: DevicePixels(1024),
            height: DevicePixels(1024),
        };
        // Max texture size for DirectX. See:
        // https://learn.microsoft.com/en-us/windows/win32/direct3d11/overviews-direct3d-11-resources-limits
        const MAX_ATLAS_SIZE: Size<DevicePixels> = Size {
            width: DevicePixels(16384),
            height: DevicePixels(16384),
        };
        let size = min_size.min(&MAX_ATLAS_SIZE).max(&DEFAULT_ATLAS_SIZE);
        let pixel_format;
        let bind_flag;
        let bytes_per_pixel;
        match kind {
            AtlasTextureKind::Monochrome => {
                pixel_format = DXGI_FORMAT_R8_UNORM;
                bind_flag = D3D11_BIND_SHADER_RESOURCE;
                bytes_per_pixel = 1;
            }
            AtlasTextureKind::Polychrome => {
                pixel_format = DXGI_FORMAT_B8G8R8A8_UNORM;
                bind_flag = D3D11_BIND_SHADER_RESOURCE;
                bytes_per_pixel = 4;
            }
            AtlasTextureKind::Subpixel => {
                pixel_format = DXGI_FORMAT_R8G8B8A8_UNORM;
                bind_flag = D3D11_BIND_SHADER_RESOURCE;
                bytes_per_pixel = 4;
            }
        }
        let texture_desc = D3D11_TEXTURE2D_DESC {
            Width: size.width.0 as u32,
            Height: size.height.0 as u32,
            MipLevels: 1,
            ArraySize: 1,
            Format: pixel_format,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Usage: D3D11_USAGE_DEFAULT,
            BindFlags: bind_flag.0 as u32,
            CPUAccessFlags: 0,
            MiscFlags: 0,
        };
        let mut texture: Option<ID3D11Texture2D> = None;
        unsafe {
            // This only returns None if the device is lost, which we will recreate later.
            // So it's ok to return None here.
            self.device
                .CreateTexture2D(&texture_desc, None, Some(&mut texture))
                .ok()?;
        }
        let texture = texture.unwrap();

        let texture_list = match kind {
            AtlasTextureKind::Monochrome => &mut self.monochrome_textures,
            AtlasTextureKind::Polychrome => &mut self.polychrome_textures,
            AtlasTextureKind::Subpixel => &mut self.subpixel_textures,
        };
        let index = texture_list.free_list.pop();
        let view = unsafe {
            let mut view = None;
            self.device
                .CreateShaderResourceView(&texture, None, Some(&mut view))
                .ok()?;
            [view]
        };
        let atlas_texture = DirectXAtlasTexture {
            id: AtlasTextureId {
                index: index.unwrap_or(texture_list.textures.len()) as u32,
                kind,
            },
            bytes_per_pixel,
            allocator: etagere::BucketedAtlasAllocator::new(device_size_to_etagere(size)),
            texture,
            view,
            live_atlas_keys: 0,
        };
        if let Some(ix) = index {
            texture_list.textures[ix] = Some(atlas_texture);
            texture_list.textures.get_mut(ix).unwrap().as_mut()
        } else {
            texture_list.textures.push(Some(atlas_texture));
            texture_list.textures.last_mut().unwrap().as_mut()
        }
    }

    fn texture(&self, id: AtlasTextureId) -> &DirectXAtlasTexture {
        match id.kind {
            AtlasTextureKind::Monochrome => &self.monochrome_textures[id.index as usize]
                .as_ref()
                .unwrap(),
            AtlasTextureKind::Polychrome => &self.polychrome_textures[id.index as usize]
                .as_ref()
                .unwrap(),
            AtlasTextureKind::Subpixel => {
                &self.subpixel_textures[id.index as usize].as_ref().unwrap()
            }
        }
    }
}

impl DirectXAtlasTexture {
    fn allocate(&mut self, size: Size<DevicePixels>) -> Option<AtlasTile> {
        let allocation = self.allocator.allocate(device_size_to_etagere(size))?;
        let tile = AtlasTile {
            texture_id: self.id,
            tile_id: allocation.id.into(),
            bounds: Bounds {
                origin: etagere_point_to_device(allocation.rectangle.min),
                size,
            },
            padding: 0,
        };
        self.live_atlas_keys += 1;
        Some(tile)
    }

    /// Returns whether the tile was actually written. A `false` return means the tile is
    /// still blank on the GPU and must not be cached — see `get_or_insert_with`.
    #[must_use]
    fn upload(
        &self,
        device_context: &ID3D11DeviceContext,
        bounds: Bounds<DevicePixels>,
        bytes: &[u8],
    ) -> bool {
        // `UpdateSubresource` reads `row_pitch * height` bytes from `bytes` based on the
        // `D3D11_BOX` below. If the caller hands us a slice shorter than that, the driver would
        // over-read past the end of the source buffer (potentially by multiple megabytes), so bail
        // out instead. This is a first-insert path rather than a per-frame one, so the check is
        // effectively free.
        let row_bytes = bounds.size.width.to_bytes(self.bytes_per_pixel as u8) as usize;
        let expected = row_bytes * bounds.size.height.0.max(0) as usize;
        if bytes.len() < expected {
            log::error!(
                "DirectXAtlasTexture::upload: source slice is {} bytes but the {}x{} region \
                 requires {} bytes; skipping upload to avoid a driver over-read",
                bytes.len(),
                bounds.size.width.0,
                bounds.size.height.0,
                expected,
            );
            return false;
        }
        unsafe {
            device_context.UpdateSubresource(
                &self.texture,
                0,
                Some(&D3D11_BOX {
                    left: bounds.left().0 as u32,
                    top: bounds.top().0 as u32,
                    front: 0,
                    right: bounds.right().0 as u32,
                    bottom: bounds.bottom().0 as u32,
                    back: 1,
                }),
                bytes.as_ptr() as _,
                bounds.size.width.to_bytes(self.bytes_per_pixel as u8),
                0,
            );
        }
        true
    }

    fn decrement_ref_count(&mut self) {
        self.live_atlas_keys -= 1;
    }

    fn is_unreferenced(&mut self) -> bool {
        self.live_atlas_keys == 0
    }
}

/// A CPU-side copy of an atlas texture, rows tightly packed.
#[cfg(test)]
pub(crate) struct TextureReadback {
    pub(crate) bytes: Vec<u8>,
    pub(crate) width: u32,
    pub(crate) bytes_per_pixel: u32,
}

#[cfg(test)]
impl TextureReadback {
    /// The bytes actually resident on the GPU for `bounds`, tightly packed, so a test can
    /// compare them against what was handed to `upload`.
    pub(crate) fn region(&self, bounds: Bounds<DevicePixels>) -> Vec<u8> {
        let bytes_per_pixel = self.bytes_per_pixel as usize;
        let row_bytes = self.width as usize * bytes_per_pixel;
        let tile_row_bytes = bounds.size.width.0 as usize * bytes_per_pixel;
        let left = bounds.origin.x.0 as usize * bytes_per_pixel;
        (0..bounds.size.height.0 as usize)
            .flat_map(|row| {
                let start = (bounds.origin.y.0 as usize + row) * row_bytes + left;
                self.bytes[start..start + tile_row_bytes].iter().copied()
            })
            .collect()
    }
}

#[cfg(test)]
impl DirectXAtlas {
    /// Whether a key currently resolves to a tile. `PlatformAtlas::contains` is gated behind
    /// gpui's `test-support` feature, which is not on for a plain `cargo test -p gpui_windows`,
    /// so this is an inherent method instead.
    pub(crate) fn contains_key(&self, key: &AtlasKey) -> bool {
        self.0.lock().tiles_by_key.contains_key(key)
    }

    /// Copies an atlas texture back to the CPU so tests can assert that what the GPU holds
    /// matches what was uploaded — the only way to catch a tile that was allocated and
    /// cached but never actually filled, or one that a later upload overwrote.
    pub(crate) fn read_back(&self, id: AtlasTextureId) -> Option<TextureReadback> {
        use windows::Win32::Graphics::Direct3D11::{
            D3D11_CPU_ACCESS_READ, D3D11_MAP_READ, D3D11_USAGE_STAGING,
        };

        let lock = self.0.lock();
        let texture = match id.kind {
            AtlasTextureKind::Monochrome => lock.monochrome_textures[id.index as usize].as_ref(),
            AtlasTextureKind::Polychrome => lock.polychrome_textures[id.index as usize].as_ref(),
            AtlasTextureKind::Subpixel => lock.subpixel_textures[id.index as usize].as_ref(),
        }?;

        let mut desc = D3D11_TEXTURE2D_DESC::default();
        unsafe { texture.texture.GetDesc(&mut desc) };
        let staging_desc = D3D11_TEXTURE2D_DESC {
            Usage: D3D11_USAGE_STAGING,
            BindFlags: 0,
            CPUAccessFlags: D3D11_CPU_ACCESS_READ.0 as u32,
            MiscFlags: 0,
            ..desc
        };

        if lock.device_is_lost() {
            return None;
        }
        let mut staging: Option<ID3D11Texture2D> = None;
        unsafe {
            lock.device
                .CreateTexture2D(&staging_desc, None, Some(&mut staging))
        }
        .ok()?;
        let staging = staging?;

        let bytes_per_pixel = texture.bytes_per_pixel;
        let row_bytes = desc.Width as usize * bytes_per_pixel as usize;
        let mut bytes = vec![0u8; row_bytes * desc.Height as usize];
        unsafe {
            lock.device_context.CopyResource(&staging, &texture.texture);
            let mut mapped = std::mem::zeroed();
            lock.device_context
                .Map(&staging, 0, D3D11_MAP_READ, 0, Some(&mut mapped))
                .ok()?;
            for row in 0..desc.Height as usize {
                std::ptr::copy_nonoverlapping(
                    (mapped.pData as *const u8).add(row * mapped.RowPitch as usize),
                    bytes.as_mut_ptr().add(row * row_bytes),
                    row_bytes,
                );
            }
            lock.device_context.Unmap(&staging, 0);
        }

        Some(TextureReadback {
            bytes,
            width: desc.Width,
            bytes_per_pixel,
        })
    }
}

fn device_size_to_etagere(size: Size<DevicePixels>) -> etagere::Size {
    etagere::Size::new(size.width.into(), size.height.into())
}

fn etagere_point_to_device(value: etagere::Point) -> Point<DevicePixels> {
    Point {
        x: DevicePixels::from(value.x),
        y: DevicePixels::from(value.y),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpui::{ImageId, RenderImageParams};
    use std::borrow::Cow;
    use windows::Win32::{
        Foundation::HMODULE,
        Graphics::{
            Direct3D::D3D_DRIVER_TYPE_WARP,
            Direct3D11::{D3D11_CREATE_DEVICE_BGRA_SUPPORT, D3D11_SDK_VERSION, D3D11CreateDevice},
        },
    };

    fn create_atlas() -> Option<DirectXAtlas> {
        let mut device: Option<ID3D11Device> = None;
        let mut device_context: Option<ID3D11DeviceContext> = None;
        unsafe {
            D3D11CreateDevice(
                None,
                D3D_DRIVER_TYPE_WARP,
                HMODULE::default(),
                D3D11_CREATE_DEVICE_BGRA_SUPPORT,
                None,
                D3D11_SDK_VERSION,
                Some(&mut device),
                None,
                Some(&mut device_context),
            )
        }
        .ok()?;
        Some(DirectXAtlas::new(&device?, &device_context?))
    }

    fn make_image_key(image_id: usize) -> AtlasKey {
        AtlasKey::Image(RenderImageParams {
            image_id: ImageId(image_id),
            frame_index: 0,
        })
    }

    fn insert_tile(atlas: &DirectXAtlas, key: &AtlasKey, size: Size<DevicePixels>) -> AtlasTile {
        atlas
            .get_or_insert_with(key, &mut || {
                let byte_count = (size.width.0 as usize) * (size.height.0 as usize) * 4;
                Ok(Some((size, Cow::Owned(vec![0u8; byte_count]))))
            })
            .expect("allocation should succeed")
            .expect("callback returns Some")
    }

    /// A tile whose upload was rejected must not be cached. Caching it would leave the glyph
    /// sampling a blank region of the atlas for the lifetime of the process — painting as a
    /// character silently missing from the middle of a word while its advance is preserved,
    /// which is the reported failure this guards against.
    #[test]
    fn test_rejected_upload_is_not_cached_and_frees_its_tile() {
        let Some(atlas) = create_atlas() else {
            return;
        };

        let size = Size {
            width: DevicePixels(64),
            height: DevicePixels(64),
        };
        let key = make_image_key(1);
        // Polychrome tiles are 4 bytes per pixel; one byte short has to be refused rather
        // than letting `UpdateSubresource` over-read the source buffer.
        let short_byte_count = (size.width.0 as usize) * (size.height.0 as usize) * 4 - 1;
        let mut short_upload = || Ok(Some((size, Cow::Owned(vec![0u8; short_byte_count]))));

        let result = atlas.get_or_insert_with(&key, &mut short_upload);
        assert!(
            result.is_err(),
            "an upload that was skipped must be reported as a failure, not silent success"
        );
        assert!(
            !atlas.contains_key(&key),
            "a tile that was never written must not be cached, or the glyph is blank forever"
        );

        // The very same key must still be insertable once the bytes are right.
        let tile = insert_tile(&atlas, &key, size);
        assert_eq!(tile.bounds.size, size);
        assert!(atlas.contains_key(&key));

        // Repeated rejections must hand their space back rather than leaking it.
        for id in 0..200 {
            let leak_key = make_image_key(100 + id);
            assert!(atlas.get_or_insert_with(&leak_key, &mut short_upload).is_err());
        }
        let big = Size {
            width: DevicePixels(700),
            height: DevicePixels(700),
        };
        let big_tile = insert_tile(&atlas, &make_image_key(2), big);
        assert_eq!(
            big_tile.texture_id, tile.texture_id,
            "rejected uploads leaked atlas space, forcing a new texture"
        );
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

        let keeper_key = make_image_key(1);
        let big_key_a = make_image_key(2);
        let big_key_b = make_image_key(3);

        let keeper_tile = insert_tile(&atlas, &keeper_key, small);
        let tile_a = insert_tile(&atlas, &big_key_a, big);
        assert_eq!(keeper_tile.texture_id, tile_a.texture_id);

        atlas.remove(&big_key_a);

        let tile_b = insert_tile(&atlas, &big_key_b, big);
        assert_eq!(tile_b.texture_id, keeper_tile.texture_id);
    }
}
