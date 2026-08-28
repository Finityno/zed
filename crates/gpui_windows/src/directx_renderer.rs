use std::{
    cell::Cell,
    ops::Range,
    rc::Rc,
    slice,
    sync::{Arc, OnceLock},
};

use anyhow::{Context, Result};
use gpui_util::ResultExt;
use windows::{
    Win32::{
        Foundation::HWND,
        Graphics::{
            Direct3D::*,
            Direct3D11::*,
            DirectComposition::*,
            DirectWrite::*,
            Dxgi::{Common::*, *},
        },
    },
    core::{HSTRING, Interface},
};

use crate::directx_renderer::shader_resources::{RawShaderBytes, ShaderModule, ShaderTarget};
use crate::*;
use gpui::*;

pub(crate) const DISABLE_DIRECT_COMPOSITION: &str = "GPUI_DISABLE_DIRECT_COMPOSITION";
const RENDER_TARGET_FORMAT: DXGI_FORMAT = DXGI_FORMAT_B8G8R8A8_UNORM;
// This configuration is used for MSAA rendering on paths only, and it's guaranteed to be supported by DirectX 11.
const PATH_MULTISAMPLE_COUNT: u32 = 4;
const MAX_INSTANCE_BUFFER_SIZE: usize = 256 * 1024 * 1024;
/// Drawn frames without a path batch before the path intermediates are
/// released. Frames, not time: an idle window stops drawing entirely, so a
/// time-based ttl would fire while the surfaces might be wanted again on the
/// very next redraw, while 300 *drawn* frames means sustained rendering that
/// needed no paths at all.
const PATH_INTERMEDIATE_IDLE_FRAMES: u32 = 300;
/// Paths rasterize through a fixed MSAA tile of this size instead of a
/// window-sized MSAA target. At 4K a window-sized 4-sample surface is 127.5MB
/// of committed memory for the window's life (D3D11 has no memoryless storage,
/// unlike Metal on Apple Silicon, so Windows pays real commit for it); a
/// 512x512 tile is 4MB. Each tile a batch touches is rasterized, resolved and
/// region-copied into the window-sized single-sample intermediate, and typical
/// frames touch a handful of tiles (corner masks, small indicators), so total
/// fill work goes down as well, not up.
const PATH_RASTER_TILE_SIZE: u32 = 512;

pub(crate) struct FontInfo {
    pub gamma_ratios: [f32; 4],
    pub grayscale_enhanced_contrast: f32,
    pub subpixel_enhanced_contrast: f32,
    pub is_bgr: bool,
}

pub(crate) struct DirectXRenderer {
    hwnd: HWND,
    atlas: Arc<DirectXAtlas>,
    devices: Option<DirectXRendererDevices>,
    resources: Option<DirectXResources>,
    overlay_resources: Option<OverlayResources>,
    globals: DirectXGlobalElements,
    pipelines: DirectXRenderPipelines,
    direct_composition: Option<DirectComposition>,
    font_info: &'static FontInfo,
    frame_scratch: FrameScratch,

    width: u32,
    height: u32,

    /// Render into an owned offscreen texture rather than the swap chain, so frames can be
    /// read back. Preserved across device-lost recovery.
    headless: bool,

    /// Whether we want to skip drwaing due to device lost events.
    ///
    /// In that case we want to discard the first frame that we draw as we got reset in the middle of a frame
    /// meaning we lost all the allocated gpu textures and scene resources.
    skip_draws: bool,

    /// How many times the swap chain has been resized, and whether the
    /// launch-storm summary (resize count at first present) was logged yet.
    /// Instrumentation for the Windows memory baseline work: launch is a known
    /// resize storm, and each resize used to stack a full generation of
    /// window-sized GPU surfaces.
    resize_count: u32,
    logged_first_present: bool,

    /// Path-batch census: what fraction of drawn frames carry any path batch
    /// decides how the path intermediates are managed (they are created
    /// lazily and released after [`PATH_INTERMEDIATE_IDLE_FRAMES`] path-free
    /// frames).
    frames_drawn: u64,
    frames_with_paths: u64,
    frames_since_paths: u32,
}

/// Direct3D objects
#[derive(Clone)]
pub(crate) struct DirectXRendererDevices {
    pub(crate) adapter: IDXGIAdapter1,
    pub(crate) dxgi_factory: IDXGIFactory6,
    pub(crate) device: ID3D11Device,
    pub(crate) device_context: ID3D11DeviceContext,
    dxgi_device: Option<IDXGIDevice>,
    annotation: Option<ID3DUserDefinedAnnotation>,
}

struct DirectXResources {
    // Direct3D rendering objects
    swap_chain: IDXGISwapChain1,
    render_target: Option<ID3D11Texture2D>,
    render_target_view: Option<ID3D11RenderTargetView>,
    /// Created on the first frame that carries a path batch and released after
    /// a stretch of path-free frames, not held for the window's life: these are
    /// the two largest textures in the process (a 4-sample MSAA target plus its
    /// resolve, ~160MB of commit at 4K), and D3D11 has no equivalent of the
    /// Metal renderer's memoryless storage that would make them free.
    path_intermediates: Option<PathIntermediates>,
    // `Option` so a resize can drop the outgoing texture before the
    // replacement is created; it is rebuilt in the same call.
    depth_stencil_texture: Option<ID3D11Texture2D>,
    depth_stencil_view: Option<ID3D11DepthStencilView>,
    viewport: D3D11_VIEWPORT,

    /// Render into an owned offscreen texture instead of the swap chain back buffer. A
    /// flip-model back buffer is not a dependable `CopyResource` source, so headless
    /// rendering needs a target it can actually read back.
    headless: bool,
}

struct OverlayResources {
    swap_chain: IDXGISwapChain1,
    render_target: Option<ID3D11Texture2D>,
    render_target_view: Option<ID3D11RenderTargetView>,
}

/// The intermediates path rasterization renders through. `texture` is the
/// window-sized single-sample surface the path-sprite pass samples;
/// `msaa_texture` is a fixed [`PATH_RASTER_TILE_SIZE`] MSAA tile the path
/// triangles are drawn into, one touched tile at a time, and `resolve_texture`
/// is the equally tile-sized single-sample surface each tile resolves to
/// before being region-copied into `texture` (D3D11's `ResolveSubresource`
/// has no partial form, so the copy is what places a tile at its window
/// position).
struct PathIntermediates {
    texture: ID3D11Texture2D,
    srv: Option<ID3D11ShaderResourceView>,
    /// For clearing `texture` before a mixed-draw-order batch, whose sprite
    /// pass samples a spanning rect that can cross tiles the batch never
    /// rewrites (see `draw_paths_to_intermediate`).
    render_target_view: Option<ID3D11RenderTargetView>,
    msaa_texture: ID3D11Texture2D,
    msaa_view: Option<ID3D11RenderTargetView>,
    resolve_texture: ID3D11Texture2D,
}

impl PathIntermediates {
    fn new(device: &ID3D11Device, width: u32, height: u32) -> Result<Self> {
        let (texture, srv, render_target_view) =
            create_path_intermediate_texture(device, width, height)?;
        let (msaa_texture, msaa_view) = create_path_intermediate_msaa_texture_and_view(
            device,
            PATH_RASTER_TILE_SIZE,
            PATH_RASTER_TILE_SIZE,
        )?;
        let resolve_texture = create_path_intermediate_resolve_texture(
            device,
            PATH_RASTER_TILE_SIZE,
            PATH_RASTER_TILE_SIZE,
        )?;
        Ok(Self {
            texture,
            srv,
            render_target_view,
            msaa_texture,
            msaa_view,
            resolve_texture,
        })
    }
}

struct DirectXRenderPipelines {
    shadow_pipeline: PipelineState<Shadow>,
    quad_pipeline: PipelineState<Quad>,
    /// Alpha-preserving blend for glass-content quads, used instead of
    /// `quad_pipeline`'s blend state for quads marked as glass content.
    quad_glass_blend_state: ID3D11BlendState,
    opaque_quad_pipeline: OpaqueQuadPipeline,
    path_rasterization_pipeline: PipelineState<PathRasterizationSprite>,
    path_sprite_pipeline: PipelineState<PathSprite>,
    underline_pipeline: PipelineState<Underline>,
    mono_sprites: PipelineState<MonochromeSprite>,
    subpixel_sprites: PipelineState<SubpixelSprite>,
    poly_sprites: PipelineState<PolychromeSprite>,
}

#[derive(Default)]
struct FrameScratch {
    path_vertices: Vec<PathRasterizationSprite>,
    path_sprites: Vec<PathSprite>,
    /// Raster-tile occupancy grid for the current path batch.
    path_tiles: Vec<bool>,
}

struct DirectXGlobalElements {
    global_params_buffer: Option<ID3D11Buffer>,
    batch_params_buffer: Option<ID3D11Buffer>,
    sampler: Option<ID3D11SamplerState>,
}

struct Annotation<'a>(&'a ID3DUserDefinedAnnotation);

impl<'a> Annotation<'a> {
    fn new(annotation: &'a ID3DUserDefinedAnnotation, label: HSTRING) -> Self {
        unsafe { annotation.BeginEvent(&label) };
        Self(annotation)
    }
}

impl Drop for Annotation<'_> {
    fn drop(&mut self) {
        unsafe { self.0.EndEvent() };
    }
}

struct DirectComposition {
    comp_device: IDCompositionDevice,
    // Keep these COM objects alive for the lifetime of the visual tree. They
    // are not otherwise read after the tree is attached to the target.
    #[allow(dead_code)]
    comp_target: IDCompositionTarget,
    #[allow(dead_code)]
    root_visual: IDCompositionVisual,
    base_visual: IDCompositionVisual,
    portal_container: IDCompositionVisual,
    overlay_visual: IDCompositionVisual,
}

struct DirectCompositionPortal {
    comp_device: IDCompositionDevice,
    container: IDCompositionVisual,
    visual: IDCompositionVisual,
    clip: IDCompositionRectangleClip,
    visible: Cell<bool>,
}

impl DirectXRendererDevices {
    pub(crate) fn new(
        directx_devices: &DirectXDevices,
        disable_direct_composition: bool,
    ) -> Result<Self> {
        let DirectXDevices {
            adapter,
            dxgi_factory,
            device,
            device_context,
        } = directx_devices;
        let dxgi_device = if disable_direct_composition {
            None
        } else {
            Some(device.cast().context("Creating DXGI device")?)
        };
        let annotation = device_context.cast().ok();

        Ok(Self {
            adapter: adapter.clone(),
            dxgi_factory: dxgi_factory.clone(),
            device: device.clone(),
            device_context: device_context.clone(),
            dxgi_device,
            annotation,
        })
    }
}

impl DirectXRenderer {
    pub(crate) fn new(
        hwnd: HWND,
        directx_devices: &DirectXDevices,
        disable_direct_composition: bool,
    ) -> Result<Self> {
        if disable_direct_composition {
            log::info!("Direct Composition is disabled.");
        }

        let devices = DirectXRendererDevices::new(directx_devices, disable_direct_composition)
            .context("Creating DirectX devices")?;
        let atlas = Arc::new(DirectXAtlas::new(&devices.device, &devices.device_context));

        let resources = DirectXResources::new(&devices, 1, 1, hwnd, disable_direct_composition, false)
            .context("Creating DirectX resources")?;
        let globals = DirectXGlobalElements::new(&devices.device)
            .context("Creating DirectX global elements")?;
        let pipelines = DirectXRenderPipelines::new(&devices.device)
            .context("Creating DirectX render pipelines")?;

        let direct_composition = if disable_direct_composition {
            None
        } else {
            let composition = DirectComposition::new(devices.dxgi_device.as_ref().unwrap(), hwnd)
                .context("Creating DirectComposition")?;
            composition
                .set_swap_chain(&resources.swap_chain)
                .context("Setting swap chain for DirectComposition")?;
            Some(composition)
        };

        Ok(DirectXRenderer {
            hwnd,
            atlas,
            devices: Some(devices),
            resources: Some(resources),
            overlay_resources: None,
            globals,
            pipelines,
            direct_composition,
            font_info: Self::get_font_info(),
            frame_scratch: FrameScratch::default(),
            width: 1,
            height: 1,
            headless: false,
            skip_draws: false,
            resize_count: 0,
            logged_first_present: false,
            frames_drawn: 0,
            frames_with_paths: 0,
            frames_since_paths: 0,
        })
    }

    /// A renderer with no window behind it, for rendering scenes to images in tests.
    ///
    /// Uses a composition swap chain, which unlike the HWND variant needs no window to exist,
    /// and skips DirectComposition entirely (that does need an HWND). Everything downstream —
    /// pipelines, shaders, blend states, the atlas — is the same as the windowed path, which is
    /// the point: an image produced here went through the real renderer.
    #[cfg(any(feature = "bench-support", feature = "test-support"))]
    pub(crate) fn new_headless(directx_devices: &DirectXDevices) -> Result<Self> {
        let devices = DirectXRendererDevices::new(directx_devices, false)
            .context("Creating DirectX devices")?;
        let atlas = Arc::new(DirectXAtlas::new(&devices.device, &devices.device_context));
        let resources = DirectXResources::new(&devices, 1, 1, HWND::default(), false, true)
            .context("Creating DirectX resources")?;
        let globals = DirectXGlobalElements::new(&devices.device)
            .context("Creating DirectX global elements")?;
        let pipelines = DirectXRenderPipelines::new(&devices.device)
            .context("Creating DirectX render pipelines")?;

        Ok(DirectXRenderer {
            hwnd: HWND::default(),
            atlas,
            devices: Some(devices),
            resources: Some(resources),
            globals,
            pipelines,
            direct_composition: None,
            // Overlays exist to compose native child views under the window's DirectComposition
            // tree, which a headless renderer has no window for. Rendering to an owned texture
            // never needs one.
            overlay_resources: None,
            font_info: Self::get_font_info(),
            frame_scratch: FrameScratch::default(),
            width: 1,
            height: 1,
            headless: true,
            skip_draws: false,
            resize_count: 0,
            logged_first_present: false,
            frames_drawn: 0,
            frames_with_paths: 0,
            frames_since_paths: 0,
        })
    }

    /// Copies the render target back to the CPU as RGBA.
    ///
    /// Must be called after `draw` and before any present. The render target is
    /// `DXGI_FORMAT_B8G8R8A8_UNORM`, so the channels are swapped on the way out.
    #[cfg(any(feature = "bench-support", feature = "test-support"))]
    pub(crate) fn read_back_render_target(&self) -> Result<image::RgbaImage> {
        use windows::Win32::Graphics::Direct3D11::{
            D3D11_CPU_ACCESS_READ, D3D11_MAP_READ, D3D11_USAGE_STAGING,
        };

        let devices = self.devices.as_ref().context("devices missing")?;
        let resources = self.resources.as_ref().context("resources missing")?;
        let render_target = resources
            .render_target
            .as_ref()
            .context("render target missing")?;

        let mut desc = D3D11_TEXTURE2D_DESC::default();
        unsafe { render_target.GetDesc(&mut desc) };
        let staging_desc = D3D11_TEXTURE2D_DESC {
            Usage: D3D11_USAGE_STAGING,
            BindFlags: 0,
            CPUAccessFlags: D3D11_CPU_ACCESS_READ.0 as u32,
            MiscFlags: 0,
            ..desc
        };

        let mut staging: Option<ID3D11Texture2D> = None;
        unsafe {
            devices
                .device
                .CreateTexture2D(&staging_desc, None, Some(&mut staging))
        }
        .context("Creating staging texture for readback")?;
        let staging = staging.context("staging texture missing")?;

        let mut pixels = vec![0u8; (desc.Width * desc.Height * 4) as usize];
        unsafe {
            devices.device_context.CopyResource(&staging, render_target);
            let mut mapped = std::mem::zeroed();
            devices
                .device_context
                .Map(&staging, 0, D3D11_MAP_READ, 0, Some(&mut mapped))
                .context("Mapping staging texture")?;
            for row in 0..desc.Height as usize {
                let source = (mapped.pData as *const u8).add(row * mapped.RowPitch as usize);
                let destination = row * desc.Width as usize * 4;
                for column in 0..desc.Width as usize {
                    let bgra = source.add(column * 4);
                    let rgba = &mut pixels[destination + column * 4..][..4];
                    rgba[0] = *bgra.add(2);
                    rgba[1] = *bgra.add(1);
                    rgba[2] = *bgra;
                    rgba[3] = *bgra.add(3);
                }
            }
            devices.device_context.Unmap(&staging, 0);
        }

        image::RgbaImage::from_raw(desc.Width, desc.Height, pixels)
            .context("Rendered pixels did not fit the image dimensions")
    }

    pub(crate) fn sprite_atlas(&self) -> Arc<dyn PlatformAtlas> {
        self.atlas.clone()
    }

    fn pre_draw(
        &self,
        render_target_view: &Option<ID3D11RenderTargetView>,
        clear_color: &[f32; 4],
    ) -> Result<()> {
        let resources = self.resources.as_ref().expect("resources missing");
        let device_context = &self
            .devices
            .as_ref()
            .expect("devices missing")
            .device_context;
        update_buffer(
            device_context,
            self.globals.global_params_buffer.as_ref().unwrap(),
            &[GlobalParams {
                gamma_ratios: self.font_info.gamma_ratios,
                viewport_size: [resources.viewport.Width, resources.viewport.Height],
                grayscale_enhanced_contrast: self.font_info.grayscale_enhanced_contrast,
                subpixel_enhanced_contrast: self.font_info.subpixel_enhanced_contrast,
                is_bgr: self.font_info.is_bgr as u32,
                _pad: [0; 3],
            }],
        )?;
        unsafe {
            device_context.ClearRenderTargetView(
                render_target_view
                    .as_ref()
                    .context("missing render target view")?,
                clear_color,
            );
            device_context.ClearDepthStencilView(
                resources
                    .depth_stencil_view
                    .as_ref()
                    .context("missing depth stencil view")?,
                D3D11_CLEAR_DEPTH.0,
                0.0,
                0,
            );
            device_context.OMSetRenderTargets(
                Some(slice::from_ref(render_target_view)),
                resources.depth_stencil_view.as_ref(),
            );
            device_context.RSSetViewports(Some(slice::from_ref(&resources.viewport)));
            device_context
                .VSSetConstantBuffers(0, Some(slice::from_ref(&self.globals.global_params_buffer)));
            device_context
                .VSSetConstantBuffers(1, Some(slice::from_ref(&self.globals.batch_params_buffer)));
            device_context
                .PSSetConstantBuffers(0, Some(slice::from_ref(&self.globals.global_params_buffer)));
        }
        Ok(())
    }

    #[inline]
    fn present(&mut self) -> Result<()> {
        let result = unsafe {
            self.resources
                .as_ref()
                .expect("resources missing")
                .swap_chain
                .Present(0, DXGI_PRESENT(0))
        };
        if !self.logged_first_present {
            self.logged_first_present = true;
            // The launch resize storm in one line: how many swap-chain
            // generations were built before anything reached the screen.
            log::info!(
                "[directx-mem] first present after {} swap-chain resizes (commit {})",
                self.resize_count,
                private_commit_label(),
            );
        }
        result.ok().context("Presenting swap chain failed")
    }

    /// Path-batch census, and the release half of the lazy path intermediates:
    /// after a sustained run of drawn frames with no path batch, the ~160MB of
    /// window-sized MSAA surfaces go away until some frame needs them again.
    fn note_frame_paths(&mut self, frame_had_paths: bool) {
        self.frames_drawn += 1;
        if frame_had_paths {
            self.frames_with_paths += 1;
            self.frames_since_paths = 0;
        } else {
            self.frames_since_paths = self.frames_since_paths.saturating_add(1);
            if self.frames_since_paths == PATH_INTERMEDIATE_IDLE_FRAMES
                && let Some(resources) = self.resources.as_mut()
                && resources.path_intermediates.take().is_some()
            {
                log::debug!(
                    "[directx-mem] released path intermediates after \
                     {PATH_INTERMEDIATE_IDLE_FRAMES} path-free frames (commit {})",
                    private_commit_label(),
                );
            }
        }
        if self.frames_drawn.is_multiple_of(1000) {
            log::debug!(
                "[directx-mem] path census: {}/{} drawn frames carried a path batch",
                self.frames_with_paths,
                self.frames_drawn,
            );
        }
    }

    pub(crate) fn handle_device_lost(&mut self, directx_devices: &DirectXDevices) -> Result<()> {
        try_to_recover_from_device_lost(|| {
            self.handle_device_lost_impl(directx_devices)
                .context("DirectXRenderer handling device lost")
        })
    }

    fn handle_device_lost_impl(&mut self, directx_devices: &DirectXDevices) -> Result<()> {
        let disable_direct_composition = self.direct_composition.is_none();
        let overlay_enabled = self.overlay_resources.is_some();

        unsafe {
            #[cfg(debug_assertions)]
            if let Some(devices) = &self.devices {
                report_live_objects(&devices.device)
                    .context("Failed to report live objects after device lost")
                    .log_err();
            }

            self.resources.take();
            self.overlay_resources.take();
            if let Some(devices) = &self.devices {
                devices.device_context.OMSetRenderTargets(None, None);
                devices.device_context.ClearState();
                devices.device_context.Flush();
                #[cfg(debug_assertions)]
                report_live_objects(&devices.device)
                    .context("Failed to report live objects after device lost")
                    .log_err();
            }

            self.direct_composition.take();
            self.devices.take();
        }

        let devices = DirectXRendererDevices::new(directx_devices, disable_direct_composition)
            .context("Recreating DirectX devices")?;
        let resources = DirectXResources::new(
            &devices,
            self.width,
            self.height,
            self.hwnd,
            disable_direct_composition,
            self.headless,
        )
        .context("Creating DirectX resources")?;
        let globals = DirectXGlobalElements::new(&devices.device)
            .context("Creating DirectXGlobalElements")?;
        let pipelines = DirectXRenderPipelines::new(&devices.device)
            .context("Creating DirectXRenderPipelines")?;

        let direct_composition = if disable_direct_composition {
            None
        } else {
            let composition =
                DirectComposition::new(devices.dxgi_device.as_ref().unwrap(), self.hwnd)?;
            composition.set_swap_chain(&resources.swap_chain)?;
            Some(composition)
        };
        let overlay_resources = if overlay_enabled {
            let overlay = OverlayResources::new(&devices, self.width, self.height)?;
            direct_composition
                .as_ref()
                .context("DirectComposition missing for overlay")?
                .set_overlay_swap_chain(&overlay.swap_chain)?;
            Some(overlay)
        } else {
            None
        };

        self.atlas
            .handle_device_lost(&devices.device, &devices.device_context);

        unsafe {
            devices.device_context.OMSetRenderTargets(
                Some(slice::from_ref(&resources.render_target_view)),
                resources.depth_stencil_view.as_ref(),
            );
        }
        self.devices = Some(devices);
        self.resources = Some(resources);
        self.overlay_resources = overlay_resources;
        self.globals = globals;
        self.pipelines = pipelines;
        self.direct_composition = direct_composition;
        self.skip_draws = true;
        Ok(())
    }

    /// Whether the Direct3D device has been removed (driver reset, TDR, adapter
    /// change). `GetDeviceRemovedReason` is answered by the D3D runtime, so it
    /// is safe to ask on a dead device — unlike every other call this renderer
    /// makes, which is forwarded into the vendor's user-mode driver.
    fn device_is_lost(&self) -> bool {
        self.devices
            .as_ref()
            .is_none_or(|devices| unsafe { devices.device.GetDeviceRemovedReason() }.is_err())
    }

    /// Stops the renderer touching the driver once the device is gone, and
    /// says whether it did.
    ///
    /// Recovery happens on the vsync thread (`handle_gpu_device_lost`), which
    /// deliberately waits a few hundred milliseconds before recreating the
    /// devices and only then hands them to this renderer through
    /// `handle_device_lost`. Until that arrives the main thread keeps
    /// receiving WM_PAINT, WM_SIZE and input-driven redraws, and each of
    /// those would otherwise drive a removed device through the user-mode
    /// driver. The D3D contract says such calls fail cleanly; in practice
    /// AMD's driver has faulted inside the process heap on exactly this
    /// path, taking the whole app down. So the first sign of removal parks
    /// the renderer in `skip_draws` and every later entry point bails
    /// before its first driver call. `handle_device_lost_impl` re-arms
    /// `skip_draws` on the fresh devices and `mark_drawable` lifts it.
    fn quiesce_if_device_lost(&mut self) -> bool {
        if !self.device_is_lost() {
            return false;
        }
        if !self.skip_draws {
            log::warn!(
                "DirectX device removed; skipping draws until the vsync thread recreates it"
            );
            self.skip_draws = true;
        }
        true
    }

    /// A draw that failed because the device went away mid-frame is not an
    /// error worth reporting — the vsync thread is already on its way with
    /// new devices — but it must park the renderer like a detected removal.
    fn absorb_device_lost(&mut self, result: Result<()>) -> Result<()> {
        if result.is_err() && self.quiesce_if_device_lost() {
            return Ok(());
        }
        result
    }

    pub(crate) fn draw(&mut self, scene: &Scene, clear_color: [f32; 4]) -> Result<()> {
        if self.skip_draws || self.quiesce_if_device_lost() {
            // skip drawing this frame, we just recovered from a device lost event
            // and so likely do not have the textures anymore that are required for drawing
            return Ok(());
        }
        let result = self.draw_inner(scene, clear_color);
        self.absorb_device_lost(result)
    }

    fn draw_inner(&mut self, scene: &Scene, clear_color: [f32; 4]) -> Result<()> {
        self.render(scene, clear_color)?;
        self.note_frame_paths(!scene.paths.is_empty());
        self.present()
    }

    /// Clear the base render target to `clear_color` and encode every
    /// primitive batch of `scene` into it, without presenting. Shared by
    /// [`draw`](Self::draw) (which then presents) and
    /// [`render_to_image`](Self::render_to_image) (which reads the target back
    /// instead), so the two cannot drift.
    fn render(&mut self, scene: &Scene, clear_color: [f32; 4]) -> Result<()> {
        let render_target_view = self
            .resources
            .as_ref()
            .context("resources missing")?
            .render_target_view
            .clone();
        self.pre_draw(&render_target_view, &clear_color)?;
        self.draw_scene(scene)
    }

    pub(crate) fn draw_layered(
        &mut self,
        scene: &Scene,
        overlay_start: usize,
        clear_color: [f32; 4],
    ) -> Result<()> {
        if self.overlay_resources.is_none() {
            return self.draw(scene, clear_color);
        }
        if self.skip_draws || self.quiesce_if_device_lost() {
            return Ok(());
        }
        let result = self.draw_layered_inner(scene, overlay_start, clear_color);
        self.absorb_device_lost(result)
    }

    fn draw_layered_inner(
        &mut self,
        scene: &Scene,
        overlay_start: usize,
        clear_color: [f32; 4],
    ) -> Result<()> {
        let split = overlay_start.min(scene.len());
        let mut base_scene = Scene::default();
        base_scene.replay(0..split, scene);
        base_scene.finish();
        let mut overlay_scene = Scene::default();
        overlay_scene.replay(split..scene.len(), scene);
        overlay_scene.finish();

        let base_view = self
            .resources
            .as_ref()
            .context("resources missing")?
            .render_target_view
            .clone();
        self.pre_draw(&base_view, &clear_color)?;
        self.draw_scene(&base_scene)?;

        let overlay_view = self
            .overlay_resources
            .as_ref()
            .context("overlay resources missing")?
            .render_target_view
            .clone();
        self.pre_draw(&overlay_view, &[0.0; 4])?;
        self.draw_scene(&overlay_scene)?;
        self.note_frame_paths(!scene.paths.is_empty());

        unsafe {
            self.resources
                .as_ref()
                .context("resources missing")?
                .swap_chain
                .Present(0, DXGI_PRESENT(0))
                .ok()
                .context("presenting base swap chain")?;
            self.overlay_resources
                .as_ref()
                .context("overlay resources missing")?
                .swap_chain
                .Present(0, DXGI_PRESENT(0))
                .ok()
                .context("presenting overlay swap chain")?;
        }
        Ok(())
    }

    pub(crate) fn enable_scene_overlay(&mut self) -> Result<()> {
        if self.overlay_resources.is_some() {
            return Ok(());
        }
        let devices = self.devices.as_ref().context("devices missing")?;
        let overlay = OverlayResources::new(devices, self.width, self.height)?;
        self.direct_composition
            .as_ref()
            .context("DirectComposition is disabled")?
            .set_overlay_swap_chain(&overlay.swap_chain)?;
        self.overlay_resources = Some(overlay);
        Ok(())
    }

    pub(crate) fn create_native_surface(&mut self) -> Result<Rc<dyn PlatformNativeSurface>> {
        self.enable_scene_overlay()?;
        Ok(Rc::new(
            self.direct_composition
                .as_ref()
                .context("DirectComposition is disabled")?
                .create_portal()?,
        ))
    }

    fn draw_scene(&mut self, scene: &Scene) -> Result<()> {
        self.upload_scene_buffers(scene)?;
        self.update_quad_indices(scene)?;

        let annotation = self
            .devices
            .as_ref()
            .and_then(|devices| devices.annotation.clone())
            .filter(|annotation| unsafe { annotation.GetStatus().as_bool() });

        if !scene.opaque_quad_indices.is_empty() {
            let _annotation = annotation.as_ref().map(|annotation| {
                Annotation::new(
                    annotation,
                    HSTRING::from(format!(
                        "opaque quads ({})",
                        scene.opaque_quad_indices.len()
                    )),
                )
            });
            self.draw_opaque_quads(
                scene.blended_quad_indices.len(),
                scene.opaque_quad_indices.len(),
            )?;
        }

        self.pipelines.opaque_quad_pipeline.enable_depth_test(
            &self
                .devices
                .as_ref()
                .context("devices missing")?
                .device_context,
        );

        let mut quad_cursor: u32 = 0;
        for batch in scene.batches() {
            let _annotation = annotation
                .as_ref()
                .map(|annotation| Annotation::new(annotation, HSTRING::from(batch.label())));

            // Quad shaders assign each quad its own depth.
            // In every other batch type, the depth is fixed to the position of the quad cursor.
            if matches!(&batch, PrimitiveBatch::Quads { .. }) {
                self.set_viewport_depth_range(0.0, 1.0)?;
            } else {
                let depth = quad_depth(quad_cursor);
                self.set_viewport_depth_range(depth, depth)?;
            }

            match batch {
                PrimitiveBatch::Shadows(range) => self.draw_shadows(range.start, range.len()),
                PrimitiveBatch::Quads {
                    range,
                    blended_range,
                } => {
                    quad_cursor += range.len() as u32;
                    self.draw_blended_quads_segmented(scene, blended_range)
                }
                PrimitiveBatch::Paths(range) => {
                    let paths = &scene.paths[range];
                    self.draw_paths_to_intermediate(paths)?;
                    self.draw_paths_from_intermediate(paths)
                }
                PrimitiveBatch::Underlines(range) => self.draw_underlines(range.start, range.len()),
                PrimitiveBatch::MonochromeSprites { texture_id, range } => {
                    self.draw_monochrome_sprites(texture_id, range.start, range.len())
                }
                PrimitiveBatch::SubpixelSprites { texture_id, range } => {
                    self.draw_subpixel_sprites(texture_id, range.start, range.len())
                }
                PrimitiveBatch::PolychromeSprites { texture_id, range } => {
                    self.draw_polychrome_sprites(texture_id, range.start, range.len())
                }
                PrimitiveBatch::Surfaces(range) => self.draw_surfaces(&scene.surfaces[range]),
            }
            .with_context(|| {
                format!(
                    "scene too large:\
                    {} paths, {} shadows, {} quads, {} underlines, {} mono, {} subpixel, {} poly, {} surfaces",
                    scene.paths.len(),
                    scene.shadows.len(),
                    scene.quads.len(),
                    scene.underlines.len(),
                    scene.monochrome_sprites.len(),
                    scene.subpixel_sprites.len(),
                    scene.polychrome_sprites.len(),
                    scene.surfaces.len(),
                )
            })?;
        }

        self.pipelines.opaque_quad_pipeline.disable_depth(
            &self
                .devices
                .as_ref()
                .context("devices missing")?
                .device_context,
        );

        // Presenting is the caller's job: `draw_layered` renders the base and
        // overlay scenes through here before presenting either swap chain.
        Ok(())
    }

    /// Render `scene` to an offscreen CPU image **without presenting** so
    /// the window need never be shown or visible (the macOS headless path
    /// goes through MetalRenderer; this is the Windows analogue). Draws into
    /// the existing render target, copies it into a `D3D11_USAGE_STAGING`
    /// texture, maps it, and converts BGRA to RGBA.
    #[cfg(any(test, feature = "test-support"))]
    pub(crate) fn render_to_image(
        &mut self,
        scene: &Scene,
        background_appearance: WindowBackgroundAppearance,
    ) -> Result<image::RgbaImage> {
        // A pending device-lost recovery (`skip_draws`) leaves the atlas holding
        // tile references from the previous device; drawing before the forced
        // re-render rebuilds them panics in `DirectXAtlasState::texture`.
        anyhow::ensure!(
            !self.skip_draws,
            "render_to_image unavailable while recovering from a lost device"
        );
        self.render(
            scene,
            match background_appearance {
                WindowBackgroundAppearance::Opaque => [1.0f32; 4],
                _ => [0.0f32; 4],
            },
        )?;

        let devices = self.devices.as_ref().context("devices missing")?;
        let device = &devices.device;
        let context = &devices.device_context;
        let resources = self.resources.as_ref().context("resources missing")?;
        let render_target = resources
            .render_target
            .as_ref()
            .context("render target missing")?;

        // A CPU-readable copy of the render target.
        let mut desc = D3D11_TEXTURE2D_DESC::default();
        unsafe { render_target.GetDesc(&mut desc) };
        let width = desc.Width;
        let height = desc.Height;
        let staging_desc = D3D11_TEXTURE2D_DESC {
            Usage: D3D11_USAGE_STAGING,
            BindFlags: 0,
            CPUAccessFlags: D3D11_CPU_ACCESS_READ.0 as u32,
            MiscFlags: 0,
            MipLevels: 1,
            ArraySize: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            ..desc
        };
        let mut staging: Option<ID3D11Texture2D> = None;
        unsafe { device.CreateTexture2D(&staging_desc, None, Some(&mut staging))? };
        let staging = staging.context("creating staging texture")?;
        unsafe { context.CopyResource(&staging, render_target) };

        let mut mapped = D3D11_MAPPED_SUBRESOURCE::default();
        unsafe { context.Map(&staging, 0, D3D11_MAP_READ, 0, Some(&mut mapped))? };
        let row_bytes = (width as usize) * 4;
        let mut pixels = vec![0u8; row_bytes * height as usize];
        // SAFETY: `Map` succeeded, so `pData` points at `RowPitch * height`
        // readable bytes for as long as the mapping is held, and `RowPitch >=
        // row_bytes` (it only ever adds trailing padding). `pixels` is sized
        // `row_bytes * height`, so every copy stays in bounds on both sides,
        // and the regions cannot overlap (`pixels` is a fresh allocation).
        unsafe {
            let src = mapped.pData as *const u8;
            for row in 0..height as usize {
                let s = src.add(row * mapped.RowPitch as usize);
                let d = pixels.as_mut_ptr().add(row * row_bytes);
                std::ptr::copy_nonoverlapping(s, d, row_bytes);
            }
            context.Unmap(&staging, 0);
        }
        // The render target is BGRA; image::RgbaImage expects RGBA.
        for px in pixels.chunks_exact_mut(4) {
            px.swap(0, 2);
        }
        image::RgbaImage::from_raw(width, height, pixels)
            .context("Failed to build RgbaImage from staging readback")
    }

    fn set_viewport_depth_range(&self, minimum_depth: f32, maximum_depth: f32) -> Result<()> {
        let devices = self.devices.as_ref().context("devices missing")?;
        let resources = self.resources.as_ref().context("resources missing")?;
        // Equal endpoints collapse every shader depth to the same value.
        let viewport = D3D11_VIEWPORT {
            MinDepth: minimum_depth,
            MaxDepth: maximum_depth,
            ..resources.viewport
        };
        unsafe {
            devices
                .device_context
                .RSSetViewports(Some(slice::from_ref(&viewport)))
        };
        Ok(())
    }

    fn update_quad_indices(&mut self, scene: &Scene) -> Result<()> {
        if scene.blended_quad_indices.is_empty() && scene.opaque_quad_indices.is_empty() {
            return Ok(());
        }
        let devices = self.devices.as_ref().context("devices missing")?;
        self.pipelines.opaque_quad_pipeline.update_quad_indices(
            &devices.device,
            &devices.device_context,
            &scene.blended_quad_indices,
            &scene.opaque_quad_indices,
        )
    }

    fn draw_opaque_quads(&self, quad_indices_start: usize, quad_count: usize) -> Result<()> {
        let devices = self.devices.as_ref().context("devices missing")?;
        self.pipelines.opaque_quad_pipeline.draw(
            &devices.device_context,
            &self.pipelines.quad_pipeline.view,
            self.globals
                .batch_params_buffer
                .as_ref()
                .context("batch params buffer missing")?,
            quad_indices_start,
            quad_count,
        )
    }

    pub(crate) fn resize(&mut self, new_size: Size<DevicePixels>) -> Result<()> {
        let width = new_size.width.0.max(1) as u32;
        let height = new_size.height.0.max(1) as u32;
        if self.width == width && self.height == height {
            return Ok(());
        }
        self.width = width;
        self.height = height;
        // The size is recorded first: `handle_device_lost_impl` rebuilds the
        // swap chain at `self.width`/`self.height`, so a resize that lands
        // while the device is gone is applied by the recovery instead.
        if self.quiesce_if_device_lost() {
            return Ok(());
        }

        // Clear the render target before resizing
        let devices = self.devices.as_ref().context("devices missing")?;
        unsafe { devices.device_context.OMSetRenderTargets(None, None) };
        let resources = self.resources.as_mut().context("resources missing")?;
        resources.render_target.take();
        resources.render_target_view.take();

        // Resizing the swap chain requires a call to the underlying DXGI adapter, which can return the device removed error.
        // The app might have moved to a monitor that's attached to a different graphics device.
        // When a graphics device is removed or reset, the desktop resolution often changes, resulting in a window size change.
        // But here we just return the error, because we are handling device lost scenarios elsewhere.
        unsafe {
            resources
                .swap_chain
                .ResizeBuffers(
                    BUFFER_COUNT as u32,
                    width,
                    height,
                    RENDER_TARGET_FORMAT,
                    DXGI_SWAP_CHAIN_FLAG(0),
                )
                .context("Failed to resize swap chain")?;
        }

        resources.recreate_resources(devices, width, height)?;
        self.resize_count += 1;
        log::debug!(
            "[directx-mem] resize #{} to {width}x{height} (commit {})",
            self.resize_count,
            private_commit_label(),
        );

        if let Some(overlay) = self.overlay_resources.as_mut() {
            overlay.resize(devices, width, height)?;
        }

        unsafe {
            devices.device_context.OMSetRenderTargets(
                Some(slice::from_ref(&resources.render_target_view)),
                resources.depth_stencil_view.as_ref(),
            );
        }

        Ok(())
    }

    fn upload_scene_buffers(&mut self, scene: &Scene) -> Result<()> {
        let devices = self.devices.as_ref().context("devices missing")?;

        if !scene.shadows.is_empty() {
            self.pipelines.shadow_pipeline.update_buffer(
                &devices.device,
                &devices.device_context,
                &scene.shadows,
            )?;
        }

        if !scene.quads.is_empty() {
            self.pipelines.quad_pipeline.update_buffer(
                &devices.device,
                &devices.device_context,
                &scene.quads,
            )?;
        }

        if !scene.underlines.is_empty() {
            self.pipelines.underline_pipeline.update_buffer(
                &devices.device,
                &devices.device_context,
                &scene.underlines,
            )?;
        }

        if !scene.monochrome_sprites.is_empty() {
            self.pipelines.mono_sprites.update_buffer(
                &devices.device,
                &devices.device_context,
                &scene.monochrome_sprites,
            )?;
        }

        if !scene.subpixel_sprites.is_empty() {
            self.pipelines.subpixel_sprites.update_buffer(
                &devices.device,
                &devices.device_context,
                &scene.subpixel_sprites,
            )?;
        }

        if !scene.polychrome_sprites.is_empty() {
            self.pipelines.poly_sprites.update_buffer(
                &devices.device,
                &devices.device_context,
                &scene.polychrome_sprites,
            )?;
        }

        Ok(())
    }

    fn draw_shadows(&mut self, start: usize, len: usize) -> Result<()> {
        if len == 0 {
            return Ok(());
        }
        let devices = self.devices.as_ref().context("devices missing")?;
        self.pipelines.shadow_pipeline.draw_range(
            &devices.device_context,
            self.globals
                .batch_params_buffer
                .as_ref()
                .context("batch params buffer missing")?,
            4,
            start as u32,
            len as u32,
            None,
        )
    }

    fn draw_blended_quads(
        &mut self,
        quad_indices_start: usize,
        quad_count: usize,
        glass: bool,
    ) -> Result<()> {
        if quad_count == 0 {
            return Ok(());
        }
        let devices = self.devices.as_ref().context("devices missing")?;
        let quad_pipeline = &self.pipelines.quad_pipeline;
        let opaque_quad_pipeline = &self.pipelines.opaque_quad_pipeline;
        let views = [
            quad_pipeline.view.clone(),
            opaque_quad_pipeline.quad_indices_range(quad_indices_start, quad_count)?,
        ];
        update_batch_start(
            &devices.device_context,
            self.globals
                .batch_params_buffer
                .as_ref()
                .context("batch params buffer missing")?,
            quad_indices_start as u32,
        )?;
        // Glass-content quads use the alpha-preserving blend so their
        // anti-aliased edges don't punch through a translucent glass surface.
        let blend_state = if glass {
            &self.pipelines.quad_glass_blend_state
        } else {
            &quad_pipeline.blend_state
        };
        set_pipeline_state(
            &devices.device_context,
            &views,
            D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP,
            &quad_pipeline.vertex,
            &quad_pipeline.fragment,
            blend_state,
        );
        unsafe {
            devices
                .device_context
                .DrawInstanced(4, quad_count as u32, 0, 0);
        }
        Ok(())
    }

    /// Draw a range of the frame's quad index buffer, splitting it into runs
    /// that share the same glass flag so glass-content quads use the
    /// alpha-preserving blend. Splitting by run keeps draw order intact.
    fn draw_blended_quads_segmented(
        &mut self,
        scene: &Scene,
        blended_range: Range<usize>,
    ) -> Result<()> {
        let mut start = blended_range.start;
        while start < blended_range.end {
            let is_glass = blended_quad_is_glass(scene, start);
            let mut end = start + 1;
            while end < blended_range.end && blended_quad_is_glass(scene, end) == is_glass {
                end += 1;
            }
            self.draw_blended_quads(start, end - start, is_glass)?;
            start = end;
        }
        Ok(())
    }

    fn draw_paths_to_intermediate(&mut self, paths: &[Path<ScaledPixels>]) -> Result<()> {
        if paths.is_empty() {
            return Ok(());
        }

        let devices = self.devices.as_ref().context("devices missing")?;
        let resources = self.resources.as_mut().context("resources missing")?;
        let intermediates = match &mut resources.path_intermediates {
            Some(intermediates) => intermediates,
            None => {
                let created = PathIntermediates::new(&devices.device, self.width, self.height)?;
                log::debug!(
                    "[directx-mem] created path intermediates at {}x{} (commit {})",
                    self.width,
                    self.height,
                    private_commit_label(),
                );
                resources.path_intermediates.insert(created)
            }
        };
        // A batch that mixes draw orders is sampled back through ONE spanning
        // rect (see `draw_paths_from_intermediate`), and that rect can cross
        // tiles no path in this batch touches. Those texels hold whatever an
        // earlier batch left in the intermediate — or driver-recycled garbage
        // on a texture rebuilt after a resize — and without this clear they
        // re-composite to the screen every frame: a dismissed selection
        // highlight or a moved corner mask stays visible as a ghost. A
        // uniform-order batch samples only per-path bounds, whose tiles are
        // all rewritten below, so it skips the window-sized clear.
        let mixed_orders = match (paths.first(), paths.last()) {
            (Some(first), Some(last)) => first.order != last.order,
            _ => false,
        };
        if mixed_orders {
            unsafe {
                devices.device_context.ClearRenderTargetView(
                    intermediates
                        .render_target_view
                        .as_ref()
                        .context("path intermediate render target view missing")?,
                    &[0.0; 4],
                );
            }
        }
        // Which raster tiles does this batch touch? Marked over a
        // ceil(width/T) x ceil(height/T) grid from each path's clipped bounds.
        let tile = PATH_RASTER_TILE_SIZE;
        let tiles_x = self.width.div_ceil(tile).max(1);
        let tiles_y = self.height.div_ceil(tile).max(1);
        let touched = &mut self.frame_scratch.path_tiles;
        touched.clear();
        touched.resize((tiles_x * tiles_y) as usize, false);
        for path in paths {
            let bounds = path.clipped_bounds();
            let left = ((bounds.origin.x.0.max(0.0) as u32) / tile).min(tiles_x - 1);
            let top = ((bounds.origin.y.0.max(0.0) as u32) / tile).min(tiles_y - 1);
            let right = ((bounds.bottom_right().x.0.ceil().max(0.0) as u32)
                .div_ceil(tile)
                .max(1))
            .min(tiles_x);
            let bottom = ((bounds.bottom_right().y.0.ceil().max(0.0) as u32)
                .div_ceil(tile)
                .max(1))
            .min(tiles_y);
            for tile_y in top..bottom {
                for tile_x in left..right {
                    touched[(tile_y * tiles_x + tile_x) as usize] = true;
                }
            }
        }

        // Collect all vertices for a single upload; each touched tile re-draws
        // the full set with a shifted viewport and the rasterizer clips to the
        // tile, so every covered pixel is still rasterized exactly once.
        let vertices = &mut self.frame_scratch.path_vertices;
        vertices.clear();
        for path in paths {
            vertices.extend(path.vertices.iter().map(|v| PathRasterizationSprite {
                xy_position: v.xy_position,
                st_position: v.st_position,
                color: path.color,
                bounds: path.clipped_bounds(),
            }));
        }

        self.pipelines.path_rasterization_pipeline.update_buffer(
            &devices.device,
            &devices.device_context,
            vertices,
        )?;

        // The batch's depth-collapsed viewport must come back for the sprite
        // pass that follows, so capture whatever is current rather than
        // rebuilding it from resources.viewport.
        let mut saved_viewports = [D3D11_VIEWPORT::default()];
        let mut saved_viewport_count = 1u32;
        unsafe {
            devices
                .device_context
                .RSGetViewports(&mut saved_viewport_count, Some(saved_viewports.as_mut_ptr()));
        }

        self.pipelines
            .opaque_quad_pipeline
            .disable_depth(&devices.device_context);

        for tile_y in 0..tiles_y {
            for tile_x in 0..tiles_x {
                if !touched[(tile_y * tiles_x + tile_x) as usize] {
                    continue;
                }
                let tile_left = tile_x * tile;
                let tile_top = tile_y * tile;
                unsafe {
                    devices.device_context.ClearRenderTargetView(
                        intermediates
                            .msaa_view
                            .as_ref()
                            .context("msaa view missing")?,
                        &[0.0; 4],
                    );
                    devices
                        .device_context
                        .OMSetRenderTargets(Some(slice::from_ref(&intermediates.msaa_view)), None);
                    // A window-sized viewport with a negative origin: window-space
                    // geometry keeps its NDC mapping and this tile's region lands
                    // at the tile texture's origin, with the rasterizer clipping
                    // to the tile's extent.
                    let viewport = D3D11_VIEWPORT {
                        TopLeftX: -(tile_left as f32),
                        TopLeftY: -(tile_top as f32),
                        Width: self.width as f32,
                        Height: self.height as f32,
                        MinDepth: 0.0,
                        MaxDepth: 1.0,
                    };
                    devices
                        .device_context
                        .RSSetViewports(Some(slice::from_ref(&viewport)));
                }

                self.pipelines.path_rasterization_pipeline.draw(
                    &devices.device_context,
                    D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST,
                    vertices.len() as u32,
                    1,
                )?;

                unsafe {
                    devices.device_context.ResolveSubresource(
                        &intermediates.resolve_texture,
                        0,
                        &intermediates.msaa_texture,
                        0,
                        RENDER_TARGET_FORMAT,
                    );
                    let copy_width = (self.width - tile_left).min(tile);
                    let copy_height = (self.height - tile_top).min(tile);
                    let source_box = D3D11_BOX {
                        left: 0,
                        top: 0,
                        front: 0,
                        right: copy_width,
                        bottom: copy_height,
                        back: 1,
                    };
                    devices.device_context.CopySubresourceRegion(
                        &intermediates.texture,
                        0,
                        tile_left,
                        tile_top,
                        0,
                        &intermediates.resolve_texture,
                        0,
                        Some(&source_box),
                    );
                }
            }
        }

        // Restore main render target and the batch's viewport
        unsafe {
            devices.device_context.OMSetRenderTargets(
                Some(slice::from_ref(&resources.render_target_view)),
                resources.depth_stencil_view.as_ref(),
            );
            if saved_viewport_count > 0 {
                devices
                    .device_context
                    .RSSetViewports(Some(&saved_viewports[..1]));
            }
            self.pipelines
                .opaque_quad_pipeline
                .enable_depth_test(&devices.device_context);
        }

        Ok(())
    }

    fn draw_paths_from_intermediate(&mut self, paths: &[Path<ScaledPixels>]) -> Result<()> {
        let Some(first_path) = paths.first() else {
            return Ok(());
        };

        // When copying paths from the intermediate texture to the drawable,
        // each pixel must only be copied once, in case of transparent paths.
        //
        // If all paths have the same draw order, then their bounds are all
        // disjoint, so we can copy each path's bounds individually. If this
        // batch combines different draw orders, we perform a single copy
        // for a minimal spanning rect.
        let sprites = &mut self.frame_scratch.path_sprites;
        sprites.clear();
        if paths.last().unwrap().order == first_path.order {
            sprites.extend(paths.iter().map(|path| PathSprite {
                bounds: path.clipped_bounds(),
            }));
        } else {
            let mut bounds = first_path.clipped_bounds();
            for path in paths.iter().skip(1) {
                bounds = bounds.union(&path.clipped_bounds());
            }
            sprites.push(PathSprite { bounds });
        }

        let devices = self.devices.as_ref().context("devices missing")?;
        let intermediates = self
            .resources
            .as_ref()
            .context("resources missing")?
            .path_intermediates
            .as_ref()
            .context("path intermediates missing")?;
        self.pipelines.path_sprite_pipeline.update_buffer(
            &devices.device,
            &devices.device_context,
            sprites,
        )?;

        // Draw the sprites with the path texture
        self.pipelines.path_sprite_pipeline.draw_with_texture(
            &devices.device_context,
            slice::from_ref(&intermediates.srv),
            slice::from_ref(&self.globals.sampler),
            sprites.len() as u32,
        )
    }

    fn draw_underlines(&mut self, start: usize, len: usize) -> Result<()> {
        if len == 0 {
            return Ok(());
        }
        let devices = self.devices.as_ref().context("devices missing")?;
        self.pipelines.underline_pipeline.draw_range(
            &devices.device_context,
            self.globals
                .batch_params_buffer
                .as_ref()
                .context("batch params buffer missing")?,
            4,
            start as u32,
            len as u32,
            None,
        )
    }

    fn draw_monochrome_sprites(
        &mut self,
        texture_id: AtlasTextureId,
        start: usize,
        len: usize,
    ) -> Result<()> {
        if len == 0 {
            return Ok(());
        }
        let devices = self.devices.as_ref().context("devices missing")?;
        let texture_view = self.atlas.get_texture_view(texture_id);
        self.pipelines.mono_sprites.draw_range_with_texture(
            &devices.device_context,
            &texture_view,
            self.globals
                .batch_params_buffer
                .as_ref()
                .context("batch params buffer missing")?,
            slice::from_ref(&self.globals.sampler),
            start as u32,
            len as u32,
        )
    }

    fn draw_subpixel_sprites(
        &mut self,
        texture_id: AtlasTextureId,
        start: usize,
        len: usize,
    ) -> Result<()> {
        if len == 0 {
            return Ok(());
        }
        let devices = self.devices.as_ref().context("devices missing")?;
        let texture_view = self.atlas.get_texture_view(texture_id);
        self.pipelines.subpixel_sprites.draw_range_with_texture(
            &devices.device_context,
            &texture_view,
            self.globals
                .batch_params_buffer
                .as_ref()
                .context("batch params buffer missing")?,
            slice::from_ref(&self.globals.sampler),
            start as u32,
            len as u32,
        )
    }

    fn draw_polychrome_sprites(
        &mut self,
        texture_id: AtlasTextureId,
        start: usize,
        len: usize,
    ) -> Result<()> {
        if len == 0 {
            return Ok(());
        }
        let devices = self.devices.as_ref().context("devices missing")?;
        let texture_view = self.atlas.get_texture_view(texture_id);
        self.pipelines.poly_sprites.draw_range_with_texture(
            &devices.device_context,
            &texture_view,
            self.globals
                .batch_params_buffer
                .as_ref()
                .context("batch params buffer missing")?,
            slice::from_ref(&self.globals.sampler),
            start as u32,
            len as u32,
        )
    }

    fn draw_surfaces(&mut self, surfaces: &[PaintSurface]) -> Result<()> {
        if surfaces.is_empty() {
            return Ok(());
        }
        Ok(())
    }

    pub(crate) fn gpu_specs(&self) -> Result<GpuSpecs> {
        let devices = self.devices.as_ref().context("devices missing")?;
        let desc = unsafe { devices.adapter.GetDesc1() }?;
        let is_software_emulated = (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32) != 0;
        let device_name = String::from_utf16_lossy(&desc.Description)
            .trim_matches(char::from(0))
            .to_string();
        let driver_name = match desc.VendorId {
            0x10DE => "NVIDIA Corporation".to_string(),
            0x1002 => "AMD Corporation".to_string(),
            0x8086 => "Intel Corporation".to_string(),
            id => format!("Unknown Vendor (ID: {:#X})", id),
        };
        let driver_version = match desc.VendorId {
            0x10DE => nvidia::get_driver_version(),
            0x1002 => amd::get_driver_version(),
            // For Intel and other vendors, we use the DXGI API to get the driver version.
            _ => dxgi::get_driver_version(&devices.adapter),
        }
        .context("Failed to get gpu driver info")
        .log_err()
        .unwrap_or("Unknown Driver".to_string());
        Ok(GpuSpecs {
            is_software_emulated,
            device_name,
            driver_name,
            driver_info: driver_version,
        })
    }

    pub(crate) fn get_font_info() -> &'static FontInfo {
        static CACHED_FONT_INFO: OnceLock<FontInfo> = OnceLock::new();
        CACHED_FONT_INFO.get_or_init(|| unsafe {
            let factory: IDWriteFactory5 = DWriteCreateFactory(DWRITE_FACTORY_TYPE_SHARED).unwrap();
            let render_params: IDWriteRenderingParams1 =
                factory.CreateRenderingParams().unwrap().cast().unwrap();
            FontInfo {
                gamma_ratios: gpui::get_gamma_correction_ratios(render_params.GetGamma()),
                grayscale_enhanced_contrast: render_params.GetGrayscaleEnhancedContrast(),
                subpixel_enhanced_contrast: render_params.GetEnhancedContrast(),
                is_bgr: render_params.GetPixelGeometry() == DWRITE_PIXEL_GEOMETRY_BGR,
            }
        })
    }

    pub(crate) fn mark_drawable(&mut self) {
        self.skip_draws = false;
    }
}

impl DirectXResources {
    pub fn new(
        devices: &DirectXRendererDevices,
        width: u32,
        height: u32,
        hwnd: HWND,
        disable_direct_composition: bool,
        headless: bool,
    ) -> Result<Self> {
        let swap_chain = if disable_direct_composition {
            create_swap_chain(&devices.dxgi_factory, &devices.device, hwnd, width, height)?
        } else {
            create_swap_chain_for_composition(
                &devices.dxgi_factory,
                &devices.device,
                width,
                height,
            )?
        };

        let (render_target, render_target_view, depth_stencil_texture, depth_stencil_view, viewport) =
            create_resources(devices, &swap_chain, width, height, headless)?;
        set_rasterizer_state(&devices.device, &devices.device_context)?;

        Ok(Self {
            swap_chain,
            headless,
            render_target: Some(render_target),
            render_target_view,
            path_intermediates: None,
            depth_stencil_texture: Some(depth_stencil_texture),
            depth_stencil_view,
            viewport,
        })
    }

    #[inline]
    fn recreate_resources(
        &mut self,
        devices: &DirectXRendererDevices,
        width: u32,
        height: u32,
    ) -> Result<()> {
        // Drop the outgoing generation before building its replacement, and
        // flush so the deferred destructions actually retire. Creating first
        // kept two full generations of window-sized surfaces committed across
        // every resize, and with no flush the retired ones lingered until some
        // later implicit flush — measured stacking up to a 2.1GB commit peak
        // during the launch resize storm.
        self.render_target = None;
        self.render_target_view = None;
        self.depth_stencil_view = None;
        self.depth_stencil_texture = None;
        // Not recreated here: the next frame that draws paths rebuilds them at
        // the new size, and a resize while no paths are on screen never pays
        // for them at all.
        self.path_intermediates = None;
        unsafe { devices.device_context.Flush() };

        let (render_target, render_target_view, depth_stencil_texture, depth_stencil_view, viewport) =
            create_resources(devices, &self.swap_chain, width, height, self.headless)?;
        self.render_target = Some(render_target);
        self.render_target_view = render_target_view;
        self.depth_stencil_texture = Some(depth_stencil_texture);
        self.depth_stencil_view = depth_stencil_view;
        self.viewport = viewport;
        Ok(())
    }
}

impl DirectXRenderPipelines {
    pub fn new(device: &ID3D11Device) -> Result<Self> {
        let shadow_pipeline = PipelineState::new(
            device,
            "shadow_pipeline",
            ShaderModule::Shadow,
            4,
            create_blend_state(device)?,
        )?;
        let quad_pipeline = PipelineState::new(
            device,
            "quad_pipeline",
            ShaderModule::Quad,
            64,
            create_blend_state(device)?,
        )?;
        let quad_glass_blend_state = create_glass_blend_state(device)?;
        let path_rasterization_pipeline = PipelineState::new(
            device,
            "path_rasterization_pipeline",
            ShaderModule::PathRasterization,
            32,
            create_blend_state_for_path_rasterization(device)?,
        )?;
        let path_sprite_pipeline = PipelineState::new(
            device,
            "path_sprite_pipeline",
            ShaderModule::PathSprite,
            4,
            create_blend_state_for_path_sprite(device)?,
        )?;
        let underline_pipeline = PipelineState::new(
            device,
            "underline_pipeline",
            ShaderModule::Underline,
            4,
            create_blend_state(device)?,
        )?;
        let mono_sprites = PipelineState::new(
            device,
            "monochrome_sprite_pipeline",
            ShaderModule::MonochromeSprite,
            512,
            create_blend_state(device)?,
        )?;
        let subpixel_sprites = PipelineState::new(
            device,
            "subpixel_sprite_pipeline",
            ShaderModule::SubpixelSprite,
            512,
            create_blend_state_for_subpixel_rendering(device)?,
        )?;
        let poly_sprites = PipelineState::new(
            device,
            "polychrome_sprite_pipeline",
            ShaderModule::PolychromeSprite,
            16,
            create_blend_state(device)?,
        )?;
        let opaque_quad_pipeline = OpaqueQuadPipeline::new(device)?;

        Ok(Self {
            shadow_pipeline,
            quad_pipeline,
            quad_glass_blend_state,
            opaque_quad_pipeline,
            path_rasterization_pipeline,
            path_sprite_pipeline,
            underline_pipeline,
            mono_sprites,
            subpixel_sprites,
            poly_sprites,
        })
    }
}

impl DirectComposition {
    pub fn new(dxgi_device: &IDXGIDevice, hwnd: HWND) -> Result<Self> {
        let comp_device = get_comp_device(dxgi_device)?;
        let comp_target = unsafe { comp_device.CreateTargetForHwnd(hwnd, true) }?;
        let root_visual = unsafe { comp_device.CreateVisual() }?;
        let base_visual = unsafe { comp_device.CreateVisual() }?;
        let portal_container = unsafe { comp_device.CreateVisual() }?;
        let overlay_visual = unsafe { comp_device.CreateVisual() }?;

        unsafe {
            root_visual.AddVisual(&base_visual, false, None)?;
            root_visual.AddVisual(&portal_container, true, &base_visual)?;
            root_visual.AddVisual(&overlay_visual, true, &portal_container)?;
            comp_target.SetRoot(&root_visual)?;
            comp_device.Commit()?;
        }

        Ok(Self {
            comp_device,
            comp_target,
            root_visual,
            base_visual,
            portal_container,
            overlay_visual,
        })
    }

    pub fn set_swap_chain(&self, swap_chain: &IDXGISwapChain1) -> Result<()> {
        unsafe {
            self.base_visual.SetContent(swap_chain)?;
            self.comp_device.Commit()?;
        }
        Ok(())
    }

    pub fn set_overlay_swap_chain(&self, swap_chain: &IDXGISwapChain1) -> Result<()> {
        unsafe {
            self.overlay_visual.SetContent(swap_chain)?;
            self.comp_device.Commit()?;
        }
        Ok(())
    }

    fn create_portal(&self) -> Result<DirectCompositionPortal> {
        let visual = unsafe { self.comp_device.CreateVisual() }?;
        let clip = unsafe { self.comp_device.CreateRectangleClip() }?;
        unsafe {
            visual.SetClip(&clip)?;
            self.portal_container.AddVisual(&visual, true, None)?;
            self.comp_device.Commit()?;
        }
        Ok(DirectCompositionPortal {
            comp_device: self.comp_device.clone(),
            container: self.portal_container.clone(),
            visual,
            clip,
            visible: Cell::new(true),
        })
    }
}

impl PlatformNativeSurface for DirectCompositionPortal {
    fn set_bounds(&self, bounds: Bounds<DevicePixels>) -> Result<()> {
        let x = bounds.origin.x.0 as f32;
        let y = bounds.origin.y.0 as f32;
        let width = bounds.size.width.0.max(0) as f32;
        let height = bounds.size.height.0.max(0) as f32;
        unsafe {
            self.visual.SetOffsetX2(x)?;
            self.visual.SetOffsetY2(y)?;
            self.clip.SetLeft2(0.0)?;
            self.clip.SetTop2(0.0)?;
            self.clip.SetRight2(width)?;
            self.clip.SetBottom2(height)?;
            self.comp_device.Commit()?;
        }
        Ok(())
    }

    fn set_visible(&self, visible: bool) -> Result<()> {
        if self.visible.get() != visible {
            unsafe {
                if visible {
                    self.container.AddVisual(&self.visual, true, None)?;
                } else {
                    self.container.RemoveVisual(&self.visual)?;
                }
                self.comp_device.Commit()?;
            }
            self.visible.set(visible);
        }
        Ok(())
    }

    fn platform_handle(&self) -> Box<dyn std::any::Any> {
        Box::new(
            self.visual
                .cast::<windows::core::IUnknown>()
                .expect("IDCompositionVisual must implement IUnknown"),
        )
    }
}

impl Drop for DirectCompositionPortal {
    fn drop(&mut self) {
        unsafe {
            self.container.RemoveVisual(&self.visual).ok();
            self.comp_device.Commit().ok();
        }
    }
}

impl OverlayResources {
    fn new(devices: &DirectXRendererDevices, width: u32, height: u32) -> Result<Self> {
        let swap_chain = create_swap_chain_for_composition(
            &devices.dxgi_factory,
            &devices.device,
            width,
            height,
        )?;
        // Never headless: an overlay always has a real composition swap chain to draw into, so
        // its render target comes from that swap chain's buffer rather than an owned texture.
        let (render_target, render_target_view) =
            create_render_target_and_its_view(&swap_chain, &devices.device, width, height, false)?;
        Ok(Self {
            swap_chain,
            render_target: Some(render_target),
            render_target_view,
        })
    }

    fn resize(&mut self, devices: &DirectXRendererDevices, width: u32, height: u32) -> Result<()> {
        self.render_target.take();
        self.render_target_view.take();
        unsafe {
            self.swap_chain.ResizeBuffers(
                BUFFER_COUNT as u32,
                width,
                height,
                RENDER_TARGET_FORMAT,
                DXGI_SWAP_CHAIN_FLAG(0),
            )?;
        }
        let (render_target, render_target_view) = create_render_target_and_its_view(
            &self.swap_chain,
            &devices.device,
            width,
            height,
            false,
        )?;
        self.render_target = Some(render_target);
        self.render_target_view = render_target_view;
        Ok(())
    }
}

impl DirectXGlobalElements {
    pub fn new(device: &ID3D11Device) -> Result<Self> {
        let global_params_buffer = create_constant_buffer::<GlobalParams>(device)?;
        let batch_params_buffer = create_constant_buffer::<BatchParams>(device)?;

        let sampler = unsafe {
            let desc = D3D11_SAMPLER_DESC {
                Filter: D3D11_FILTER_MIN_MAG_MIP_LINEAR,
                AddressU: D3D11_TEXTURE_ADDRESS_WRAP,
                AddressV: D3D11_TEXTURE_ADDRESS_WRAP,
                AddressW: D3D11_TEXTURE_ADDRESS_WRAP,
                MipLODBias: 0.0,
                MaxAnisotropy: 1,
                ComparisonFunc: D3D11_COMPARISON_ALWAYS,
                BorderColor: [0.0; 4],
                MinLOD: 0.0,
                MaxLOD: D3D11_FLOAT32_MAX,
            };
            let mut output = None;
            device.CreateSamplerState(&desc, Some(&mut output))?;
            output
        };

        Ok(Self {
            global_params_buffer,
            batch_params_buffer,
            sampler,
        })
    }
}

#[derive(Debug, Default)]
#[repr(C)]
struct GlobalParams {
    gamma_ratios: [f32; 4],
    viewport_size: [f32; 2],
    grayscale_enhanced_contrast: f32,
    subpixel_enhanced_contrast: f32,
    is_bgr: u32,
    _pad: [u32; 3],
}

#[derive(Clone, Copy, Debug, Default)]
#[repr(C, align(16))]
struct BatchParams {
    start_index: u32,
    _padding: [u32; 3],
}

const _: () = assert!(std::mem::size_of::<BatchParams>() == 16);

struct PipelineState<T> {
    label: &'static str,
    vertex: ID3D11VertexShader,
    fragment: ID3D11PixelShader,
    buffer: ID3D11Buffer,
    buffer_size: usize,
    view: Option<ID3D11ShaderResourceView>,
    blend_state: ID3D11BlendState,
    _marker: std::marker::PhantomData<T>,
}

impl<T> PipelineState<T> {
    fn new(
        device: &ID3D11Device,
        label: &'static str,
        shader_module: ShaderModule,
        buffer_size: usize,
        blend_state: ID3D11BlendState,
    ) -> Result<Self> {
        let vertex = {
            let raw_shader = RawShaderBytes::new(shader_module, ShaderTarget::Vertex)?;
            create_vertex_shader(device, raw_shader.as_bytes())?
        };
        let fragment = {
            let raw_shader = RawShaderBytes::new(shader_module, ShaderTarget::Fragment)?;
            create_fragment_shader(device, raw_shader.as_bytes())?
        };
        let buffer = create_buffer(device, std::mem::size_of::<T>(), buffer_size)?;
        let view = create_buffer_view(device, &buffer)?;

        Ok(PipelineState {
            label,
            vertex,
            fragment,
            buffer,
            buffer_size,
            view,
            blend_state,
            _marker: std::marker::PhantomData,
        })
    }

    fn update_buffer(
        &mut self,
        device: &ID3D11Device,
        device_context: &ID3D11DeviceContext,
        data: &[T],
    ) -> Result<()> {
        if self.buffer_size < data.len() {
            let element_size = std::mem::size_of::<T>();
            let required_size = std::mem::size_of_val(data);
            anyhow::ensure!(
                required_size <= MAX_INSTANCE_BUFFER_SIZE,
                "{} buffer needs {required_size} bytes, above the maximum of {MAX_INSTANCE_BUFFER_SIZE}",
                self.label
            );
            let new_buffer_size = data
                .len()
                .next_power_of_two()
                .min(MAX_INSTANCE_BUFFER_SIZE / element_size);
            log::debug!(
                "Updating {} buffer size from {} to {}",
                self.label,
                self.buffer_size,
                new_buffer_size
            );
            let buffer = create_buffer(device, std::mem::size_of::<T>(), new_buffer_size)?;
            let view = create_buffer_view(device, &buffer)?;
            self.buffer = buffer;
            self.view = view;
            self.buffer_size = new_buffer_size;
        }
        update_buffer(device_context, &self.buffer, data)
    }

    fn draw(
        &self,
        device_context: &ID3D11DeviceContext,
        topology: D3D_PRIMITIVE_TOPOLOGY,
        vertex_count: u32,
        instance_count: u32,
    ) -> Result<()> {
        set_pipeline_state(
            device_context,
            slice::from_ref(&self.view),
            topology,
            &self.vertex,
            &self.fragment,
            &self.blend_state,
        );
        unsafe {
            device_context.DrawInstanced(vertex_count, instance_count, 0, 0);
        }
        Ok(())
    }

    fn draw_with_texture(
        &self,
        device_context: &ID3D11DeviceContext,
        texture: &[Option<ID3D11ShaderResourceView>],
        sampler: &[Option<ID3D11SamplerState>],
        instance_count: u32,
    ) -> Result<()> {
        set_pipeline_state(
            device_context,
            slice::from_ref(&self.view),
            D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP,
            &self.vertex,
            &self.fragment,
            &self.blend_state,
        );
        unsafe {
            device_context.PSSetSamplers(0, Some(sampler));
            device_context.VSSetShaderResources(0, Some(texture));
            device_context.PSSetShaderResources(0, Some(texture));

            device_context.DrawInstanced(4, instance_count, 0, 0);
        }
        Ok(())
    }

    fn draw_range(
        &self,
        device_context: &ID3D11DeviceContext,
        batch_params_buffer: &ID3D11Buffer,
        vertex_count: u32,
        first_instance: u32,
        instance_count: u32,
        blend_override: Option<&ID3D11BlendState>,
    ) -> Result<()> {
        anyhow::ensure!(
            first_instance as usize + instance_count as usize <= self.buffer_size,
            "DirectX instance range exceeds the {} buffer",
            self.label
        );
        update_batch_start(device_context, batch_params_buffer, first_instance)?;
        set_pipeline_state(
            device_context,
            slice::from_ref(&self.view),
            D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP,
            &self.vertex,
            &self.fragment,
            blend_override.unwrap_or(&self.blend_state),
        );
        unsafe {
            device_context.DrawInstanced(vertex_count, instance_count, 0, 0);
        }
        Ok(())
    }

    fn draw_range_with_texture(
        &self,
        device_context: &ID3D11DeviceContext,
        texture: &[Option<ID3D11ShaderResourceView>],
        batch_params_buffer: &ID3D11Buffer,
        sampler: &[Option<ID3D11SamplerState>],
        first_instance: u32,
        instance_count: u32,
    ) -> Result<()> {
        anyhow::ensure!(
            first_instance as usize + instance_count as usize <= self.buffer_size,
            "DirectX instance range exceeds the {} buffer",
            self.label
        );
        update_batch_start(device_context, batch_params_buffer, first_instance)?;
        set_pipeline_state(
            device_context,
            slice::from_ref(&self.view),
            D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP,
            &self.vertex,
            &self.fragment,
            &self.blend_state,
        );
        unsafe {
            device_context.PSSetSamplers(0, Some(sampler));
            device_context.VSSetShaderResources(0, Some(texture));
            device_context.PSSetShaderResources(0, Some(texture));
            device_context.DrawInstanced(4, instance_count, 0, 0);
        }
        Ok(())
    }
}

struct OpaqueQuadPipeline {
    vertex: ID3D11VertexShader,
    fragment: ID3D11PixelShader,
    quad_indices_buffer: ID3D11Buffer,
    quad_indices_view: Option<ID3D11ShaderResourceView>,
    quad_indices_buffer_size: usize,
    quad_index_count: usize,
    blend_state: ID3D11BlendState,
    depth_write_state: ID3D11DepthStencilState,
    depth_test_state: ID3D11DepthStencilState,
    depth_disabled_state: ID3D11DepthStencilState,
}

impl OpaqueQuadPipeline {
    fn new(device: &ID3D11Device) -> Result<Self> {
        let vertex = {
            let raw_shader = RawShaderBytes::new(ShaderModule::OpaqueQuad, ShaderTarget::Vertex)?;
            create_vertex_shader(device, raw_shader.as_bytes())?
        };
        let fragment = {
            let raw_shader = RawShaderBytes::new(ShaderModule::OpaqueQuad, ShaderTarget::Fragment)?;
            create_fragment_shader(device, raw_shader.as_bytes())?
        };
        let quad_indices_buffer_size = 512;
        let quad_indices_buffer =
            create_buffer(device, std::mem::size_of::<u32>(), quad_indices_buffer_size)?;
        let quad_indices_view = create_buffer_view(device, &quad_indices_buffer)?;
        Ok(OpaqueQuadPipeline {
            vertex,
            fragment,
            quad_indices_buffer,
            quad_indices_view,
            quad_indices_buffer_size,
            quad_index_count: 0,
            blend_state: create_opaque_blend_state(device)?,
            depth_write_state: create_depth_stencil_state(
                device,
                true,
                D3D11_DEPTH_WRITE_MASK_ALL,
            )?,
            depth_test_state: create_depth_stencil_state(
                device,
                true,
                D3D11_DEPTH_WRITE_MASK_ZERO,
            )?,
            depth_disabled_state: create_depth_stencil_state(
                device,
                false,
                D3D11_DEPTH_WRITE_MASK_ZERO,
            )?,
        })
    }

    fn update_quad_indices(
        &mut self,
        device: &ID3D11Device,
        device_context: &ID3D11DeviceContext,
        blended_quad_indices: &[u32],
        opaque_quad_indices: &[u32],
    ) -> Result<()> {
        let quad_index_count = blended_quad_indices.len() + opaque_quad_indices.len();
        if self.quad_indices_buffer_size < quad_index_count {
            let element_size = std::mem::size_of::<u32>();
            let required_size = element_size
                .checked_mul(quad_index_count)
                .context("DirectX quad index buffer size overflow")?;
            anyhow::ensure!(
                required_size <= MAX_INSTANCE_BUFFER_SIZE,
                "quad index buffer needs {required_size} bytes, above the maximum of {MAX_INSTANCE_BUFFER_SIZE}"
            );
            let new_buffer_size = quad_index_count
                .checked_next_power_of_two()
                .context("DirectX quad index count is too large")?
                .min(MAX_INSTANCE_BUFFER_SIZE / element_size);
            self.quad_indices_buffer = create_buffer(device, element_size, new_buffer_size)?;
            self.quad_indices_view = create_buffer_view(device, &self.quad_indices_buffer)?;
            self.quad_indices_buffer_size = new_buffer_size;
        }
        self.quad_index_count = quad_index_count;
        unsafe {
            let mut destination = std::mem::zeroed();
            device_context.Map(
                &self.quad_indices_buffer,
                0,
                D3D11_MAP_WRITE_DISCARD,
                0,
                Some(&mut destination),
            )?;
            let destination = destination.pData as *mut u32;
            std::ptr::copy_nonoverlapping(
                blended_quad_indices.as_ptr(),
                destination,
                blended_quad_indices.len(),
            );
            std::ptr::copy_nonoverlapping(
                opaque_quad_indices.as_ptr(),
                destination.add(blended_quad_indices.len()),
                opaque_quad_indices.len(),
            );
            device_context.Unmap(&self.quad_indices_buffer, 0);
        }
        Ok(())
    }

    /// The whole index buffer stays bound; the batch's starting offset reaches the
    /// shader through the batch params constant buffer, so no ranged view — and
    /// therefore no `FirstElement` — is ever needed.
    fn quad_indices_range(
        &self,
        quad_indices_start: usize,
        quad_count: usize,
    ) -> Result<Option<ID3D11ShaderResourceView>> {
        let range_end = quad_indices_start
            .checked_add(quad_count)
            .context("DirectX quad index range overflow")?;
        anyhow::ensure!(
            range_end <= self.quad_index_count,
            "DirectX quad index range exceeds the {} uploaded quad indices",
            self.quad_index_count
        );
        Ok(self.quad_indices_view.clone())
    }

    fn draw(
        &self,
        device_context: &ID3D11DeviceContext,
        quad_instances_view: &Option<ID3D11ShaderResourceView>,
        batch_params_buffer: &ID3D11Buffer,
        quad_indices_start: usize,
        quad_count: usize,
    ) -> Result<()> {
        if quad_count == 0 {
            return Ok(());
        }
        let views = [
            quad_instances_view.clone(),
            self.quad_indices_range(quad_indices_start, quad_count)?,
        ];
        update_batch_start(device_context, batch_params_buffer, quad_indices_start as u32)?;
        unsafe { device_context.OMSetDepthStencilState(&self.depth_write_state, 0) };
        set_pipeline_state(
            device_context,
            &views,
            D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP,
            &self.vertex,
            &self.fragment,
            &self.blend_state,
        );
        unsafe { device_context.DrawInstanced(4, quad_count as u32, 0, 0) };
        Ok(())
    }

    fn enable_depth_test(&self, device_context: &ID3D11DeviceContext) {
        unsafe { device_context.OMSetDepthStencilState(&self.depth_test_state, 0) };
    }

    fn disable_depth(&self, device_context: &ID3D11DeviceContext) {
        unsafe { device_context.OMSetDepthStencilState(&self.depth_disabled_state, 0) };
    }
}

#[derive(Clone, Copy)]
#[repr(C)]
struct PathRasterizationSprite {
    xy_position: Point<ScaledPixels>,
    st_position: Point<f32>,
    color: Background,
    bounds: Bounds<ScaledPixels>,
}

#[derive(Clone, Copy)]
#[repr(C)]
struct PathSprite {
    bounds: Bounds<ScaledPixels>,
}

impl Drop for DirectXRenderer {
    fn drop(&mut self) {
        #[cfg(debug_assertions)]
        if let Some(devices) = &self.devices {
            report_live_objects(&devices.device).ok();
        }
    }
}

#[inline]
fn get_comp_device(dxgi_device: &IDXGIDevice) -> Result<IDCompositionDevice> {
    Ok(unsafe { DCompositionCreateDevice(dxgi_device)? })
}

fn create_swap_chain_for_composition(
    dxgi_factory: &IDXGIFactory6,
    device: &ID3D11Device,
    width: u32,
    height: u32,
) -> Result<IDXGISwapChain1> {
    let desc = DXGI_SWAP_CHAIN_DESC1 {
        Width: width,
        Height: height,
        Format: RENDER_TARGET_FORMAT,
        Stereo: false.into(),
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        BufferUsage: DXGI_USAGE_RENDER_TARGET_OUTPUT,
        BufferCount: BUFFER_COUNT as u32,
        // Composition SwapChains only support the DXGI_SCALING_STRETCH Scaling.
        Scaling: DXGI_SCALING_STRETCH,
        SwapEffect: DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL,
        AlphaMode: DXGI_ALPHA_MODE_PREMULTIPLIED,
        Flags: 0,
    };
    Ok(unsafe { dxgi_factory.CreateSwapChainForComposition(device, &desc, None)? })
}

fn create_swap_chain(
    dxgi_factory: &IDXGIFactory6,
    device: &ID3D11Device,
    hwnd: HWND,
    width: u32,
    height: u32,
) -> Result<IDXGISwapChain1> {
    use windows::Win32::Graphics::Dxgi::DXGI_MWA_NO_ALT_ENTER;

    let desc = DXGI_SWAP_CHAIN_DESC1 {
        Width: width,
        Height: height,
        Format: RENDER_TARGET_FORMAT,
        Stereo: false.into(),
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        BufferUsage: DXGI_USAGE_RENDER_TARGET_OUTPUT,
        BufferCount: BUFFER_COUNT as u32,
        Scaling: DXGI_SCALING_NONE,
        SwapEffect: DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL,
        AlphaMode: DXGI_ALPHA_MODE_IGNORE,
        Flags: 0,
    };
    let swap_chain =
        unsafe { dxgi_factory.CreateSwapChainForHwnd(device, hwnd, &desc, None, None) }?;
    unsafe { dxgi_factory.MakeWindowAssociation(hwnd, DXGI_MWA_NO_ALT_ENTER) }?;
    Ok(swap_chain)
}

#[inline]
fn create_resources(
    devices: &DirectXRendererDevices,
    swap_chain: &IDXGISwapChain1,
    width: u32,
    height: u32,
    headless: bool,
) -> Result<(
    ID3D11Texture2D,
    Option<ID3D11RenderTargetView>,
    ID3D11Texture2D,
    Option<ID3D11DepthStencilView>,
    D3D11_VIEWPORT,
)> {
    let (render_target, render_target_view) =
        create_render_target_and_its_view(swap_chain, &devices.device, width, height, headless)?;
    let (depth_stencil_texture, depth_stencil_view) =
        create_depth_stencil_texture_and_view(&devices.device, width, height)?;
    let viewport = D3D11_VIEWPORT {
        TopLeftX: 0.0,
        TopLeftY: 0.0,
        Width: width as f32,
        Height: height as f32,
        MinDepth: 0.0,
        MaxDepth: 1.0,
    };
    Ok((
        render_target,
        render_target_view,
        depth_stencil_texture,
        depth_stencil_view,
        viewport,
    ))
}

#[inline]
fn create_render_target_and_its_view(
    swap_chain: &IDXGISwapChain1,
    device: &ID3D11Device,
    width: u32,
    height: u32,
    headless: bool,
) -> Result<(ID3D11Texture2D, Option<ID3D11RenderTargetView>)> {
    let render_target: ID3D11Texture2D = if headless {
        let desc = D3D11_TEXTURE2D_DESC {
            Width: width,
            Height: height,
            MipLevels: 1,
            ArraySize: 1,
            Format: RENDER_TARGET_FORMAT,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Usage: D3D11_USAGE_DEFAULT,
            BindFlags: D3D11_BIND_RENDER_TARGET.0 as u32,
            CPUAccessFlags: 0,
            MiscFlags: 0,
        };
        let mut texture = None;
        unsafe { device.CreateTexture2D(&desc, None, Some(&mut texture))? };
        texture.context("Creating headless render target")?
    } else {
        unsafe { swap_chain.GetBuffer(0) }?
    };
    let mut render_target_view = None;
    unsafe { device.CreateRenderTargetView(&render_target, None, Some(&mut render_target_view))? };
    Ok((render_target, render_target_view))
}

#[inline]
fn create_depth_stencil_texture_and_view(
    device: &ID3D11Device,
    width: u32,
    height: u32,
) -> Result<(ID3D11Texture2D, Option<ID3D11DepthStencilView>)> {
    let texture = unsafe {
        let mut output = None;
        let desc = D3D11_TEXTURE2D_DESC {
            Width: width,
            Height: height,
            MipLevels: 1,
            ArraySize: 1,
            // `quad_depth` steps in 1/65535 increments and opaque-quad
            // partitioning is capped at 65534 quads, so 16 unorm bits resolve
            // every slot exactly at half the size of D32_FLOAT. Cleared to 0.0
            // with a GREATER test, both of which are format-independent.
            Format: DXGI_FORMAT_D16_UNORM,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Usage: D3D11_USAGE_DEFAULT,
            BindFlags: D3D11_BIND_DEPTH_STENCIL.0 as u32,
            CPUAccessFlags: 0,
            MiscFlags: 0,
        };
        device.CreateTexture2D(&desc, None, Some(&mut output))?;
        output.unwrap()
    };
    let mut view = None;
    unsafe { device.CreateDepthStencilView(&texture, None, Some(&mut view))? };
    Ok((texture, view))
}

#[inline]
fn create_path_intermediate_texture(
    device: &ID3D11Device,
    width: u32,
    height: u32,
) -> Result<(
    ID3D11Texture2D,
    Option<ID3D11ShaderResourceView>,
    Option<ID3D11RenderTargetView>,
)> {
    let texture = unsafe {
        let mut output = None;
        let desc = D3D11_TEXTURE2D_DESC {
            Width: width,
            Height: height,
            MipLevels: 1,
            ArraySize: 1,
            Format: RENDER_TARGET_FORMAT,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Usage: D3D11_USAGE_DEFAULT,
            BindFlags: (D3D11_BIND_RENDER_TARGET.0 | D3D11_BIND_SHADER_RESOURCE.0) as u32,
            CPUAccessFlags: 0,
            MiscFlags: 0,
        };
        device.CreateTexture2D(&desc, None, Some(&mut output))?;
        output.unwrap()
    };

    let mut shader_resource_view = None;
    unsafe { device.CreateShaderResourceView(&texture, None, Some(&mut shader_resource_view))? };
    let mut render_target_view = None;
    unsafe { device.CreateRenderTargetView(&texture, None, Some(&mut render_target_view))? };

    Ok((texture, Some(shader_resource_view.unwrap()), render_target_view))
}

/// Single-sample tile the MSAA tile resolves into before being region-copied
/// to its window position (`ResolveSubresource` has no partial form).
#[inline]
fn create_path_intermediate_resolve_texture(
    device: &ID3D11Device,
    width: u32,
    height: u32,
) -> Result<ID3D11Texture2D> {
    let texture = unsafe {
        let mut output = None;
        let desc = D3D11_TEXTURE2D_DESC {
            Width: width,
            Height: height,
            MipLevels: 1,
            ArraySize: 1,
            Format: RENDER_TARGET_FORMAT,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Usage: D3D11_USAGE_DEFAULT,
            BindFlags: D3D11_BIND_RENDER_TARGET.0 as u32,
            CPUAccessFlags: 0,
            MiscFlags: 0,
        };
        device.CreateTexture2D(&desc, None, Some(&mut output))?;
        output
    };
    texture.context("Creating path intermediate resolve texture")
}

#[inline]
fn create_path_intermediate_msaa_texture_and_view(
    device: &ID3D11Device,
    width: u32,
    height: u32,
) -> Result<(ID3D11Texture2D, Option<ID3D11RenderTargetView>)> {
    let msaa_texture = unsafe {
        let mut output = None;
        let desc = D3D11_TEXTURE2D_DESC {
            Width: width,
            Height: height,
            MipLevels: 1,
            ArraySize: 1,
            Format: RENDER_TARGET_FORMAT,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: PATH_MULTISAMPLE_COUNT,
                Quality: D3D11_STANDARD_MULTISAMPLE_PATTERN.0 as u32,
            },
            Usage: D3D11_USAGE_DEFAULT,
            BindFlags: D3D11_BIND_RENDER_TARGET.0 as u32,
            CPUAccessFlags: 0,
            MiscFlags: 0,
        };
        device.CreateTexture2D(&desc, None, Some(&mut output))?;
        output.unwrap()
    };
    let mut msaa_view = None;
    unsafe { device.CreateRenderTargetView(&msaa_texture, None, Some(&mut msaa_view))? };
    Ok((msaa_texture, Some(msaa_view.unwrap())))
}

/// Private commit of this process formatted for the `[directx-mem]` log
/// lines, or `"?"` when the counter read fails. Commit rather than working
/// set because the GPU sysmem shadows these lines exist to watch are
/// committed-never-touched: they never show up in the working set at all.
fn private_commit_label() -> String {
    use windows::Win32::System::ProcessStatus::{
        GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS, PROCESS_MEMORY_COUNTERS_EX,
    };
    use windows::Win32::System::Threading::GetCurrentProcess;

    let mut counters = PROCESS_MEMORY_COUNTERS_EX {
        cb: std::mem::size_of::<PROCESS_MEMORY_COUNTERS_EX>() as u32,
        ..Default::default()
    };
    // SAFETY: EX begins with the exact PROCESS_MEMORY_COUNTERS layout and the
    // API dispatches on cb, which names the EX size.
    let result = unsafe {
        GetProcessMemoryInfo(
            GetCurrentProcess(),
            std::ptr::from_mut(&mut counters).cast::<PROCESS_MEMORY_COUNTERS>(),
            counters.cb,
        )
    };
    match result {
        Ok(()) => format!(
            "{:.1} MB",
            counters.PrivateUsage as f64 / (1024.0 * 1024.0)
        ),
        Err(_) => "?".to_string(),
    }
}

#[inline]
fn set_rasterizer_state(device: &ID3D11Device, device_context: &ID3D11DeviceContext) -> Result<()> {
    let desc = D3D11_RASTERIZER_DESC {
        FillMode: D3D11_FILL_SOLID,
        CullMode: D3D11_CULL_NONE,
        FrontCounterClockwise: false.into(),
        DepthBias: 0,
        DepthBiasClamp: 0.0,
        SlopeScaledDepthBias: 0.0,
        DepthClipEnable: true.into(),
        ScissorEnable: false.into(),
        MultisampleEnable: true.into(),
        AntialiasedLineEnable: false.into(),
    };
    let rasterizer_state = unsafe {
        let mut state = None;
        device.CreateRasterizerState(&desc, Some(&mut state))?;
        state.unwrap()
    };
    unsafe { device_context.RSSetState(&rasterizer_state) };
    Ok(())
}

#[inline]
fn create_blend_state(device: &ID3D11Device) -> Result<ID3D11BlendState> {
    let mut desc = D3D11_BLEND_DESC::default();
    desc.RenderTarget[0].BlendEnable = true.into();
    desc.RenderTarget[0].BlendOp = D3D11_BLEND_OP_ADD;
    desc.RenderTarget[0].BlendOpAlpha = D3D11_BLEND_OP_ADD;
    desc.RenderTarget[0].SrcBlend = D3D11_BLEND_SRC_ALPHA;
    desc.RenderTarget[0].SrcBlendAlpha = D3D11_BLEND_ONE;
    desc.RenderTarget[0].DestBlend = D3D11_BLEND_INV_SRC_ALPHA;
    desc.RenderTarget[0].DestBlendAlpha = D3D11_BLEND_ONE;
    desc.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL.0 as u8;
    unsafe {
        let mut state = None;
        device.CreateBlendState(&desc, Some(&mut state))?;
        Ok(state.unwrap())
    }
}

/// Like [`create_blend_state`] but with `SrcBlendAlpha = ZERO`, so a quad's own
/// alpha is not accumulated into the framebuffer (the destination alpha is
/// preserved). Used for glass-content quads so their rounded, anti-aliased
/// edges don't punch through a translucent glass surface beneath them.
#[inline]
fn create_glass_blend_state(device: &ID3D11Device) -> Result<ID3D11BlendState> {
    let mut desc = D3D11_BLEND_DESC::default();
    desc.RenderTarget[0].BlendEnable = true.into();
    desc.RenderTarget[0].BlendOp = D3D11_BLEND_OP_ADD;
    desc.RenderTarget[0].BlendOpAlpha = D3D11_BLEND_OP_ADD;
    desc.RenderTarget[0].SrcBlend = D3D11_BLEND_SRC_ALPHA;
    desc.RenderTarget[0].SrcBlendAlpha = D3D11_BLEND_ZERO;
    desc.RenderTarget[0].DestBlend = D3D11_BLEND_INV_SRC_ALPHA;
    desc.RenderTarget[0].DestBlendAlpha = D3D11_BLEND_ONE;
    desc.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL.0 as u8;
    unsafe {
        let mut state = None;
        device.CreateBlendState(&desc, Some(&mut state))?;
        Ok(state.unwrap())
    }
}

#[inline]
fn create_opaque_blend_state(device: &ID3D11Device) -> Result<ID3D11BlendState> {
    let mut desc = D3D11_BLEND_DESC::default();
    desc.RenderTarget[0].BlendEnable = false.into();
    desc.RenderTarget[0].BlendOp = D3D11_BLEND_OP_ADD;
    desc.RenderTarget[0].BlendOpAlpha = D3D11_BLEND_OP_ADD;
    desc.RenderTarget[0].SrcBlend = D3D11_BLEND_ONE;
    desc.RenderTarget[0].SrcBlendAlpha = D3D11_BLEND_ONE;
    desc.RenderTarget[0].DestBlend = D3D11_BLEND_ZERO;
    desc.RenderTarget[0].DestBlendAlpha = D3D11_BLEND_ZERO;
    desc.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL.0 as u8;
    unsafe {
        let mut state = None;
        device.CreateBlendState(&desc, Some(&mut state))?;
        Ok(state.unwrap())
    }
}

#[inline]
fn create_depth_stencil_state(
    device: &ID3D11Device,
    depth_enable: bool,
    write_mask: D3D11_DEPTH_WRITE_MASK,
) -> Result<ID3D11DepthStencilState> {
    let stencil_op = D3D11_DEPTH_STENCILOP_DESC {
        StencilFailOp: D3D11_STENCIL_OP_KEEP,
        StencilDepthFailOp: D3D11_STENCIL_OP_KEEP,
        StencilPassOp: D3D11_STENCIL_OP_KEEP,
        StencilFunc: D3D11_COMPARISON_ALWAYS,
    };
    let desc = D3D11_DEPTH_STENCIL_DESC {
        DepthEnable: depth_enable.into(),
        DepthWriteMask: write_mask,
        DepthFunc: D3D11_COMPARISON_GREATER,
        StencilEnable: false.into(),
        StencilReadMask: D3D11_DEFAULT_STENCIL_READ_MASK as u8,
        StencilWriteMask: D3D11_DEFAULT_STENCIL_WRITE_MASK as u8,
        FrontFace: stencil_op,
        BackFace: stencil_op,
    };
    unsafe {
        let mut state = None;
        device.CreateDepthStencilState(&desc, Some(&mut state))?;
        Ok(state.unwrap())
    }
}

#[inline]
fn create_blend_state_for_subpixel_rendering(device: &ID3D11Device) -> Result<ID3D11BlendState> {
    let mut desc = D3D11_BLEND_DESC::default();
    desc.RenderTarget[0].BlendEnable = true.into();
    desc.RenderTarget[0].BlendOp = D3D11_BLEND_OP_ADD;
    desc.RenderTarget[0].BlendOpAlpha = D3D11_BLEND_OP_ADD;
    desc.RenderTarget[0].SrcBlend = D3D11_BLEND_SRC1_COLOR;
    desc.RenderTarget[0].DestBlend = D3D11_BLEND_INV_SRC1_COLOR;
    // It does not make sense to draw transparent subpixel-rendered text, since it cannot be meaningfully alpha-blended onto anything else.
    desc.RenderTarget[0].SrcBlendAlpha = D3D11_BLEND_ONE;
    desc.RenderTarget[0].DestBlendAlpha = D3D11_BLEND_ZERO;
    desc.RenderTarget[0].RenderTargetWriteMask =
        D3D11_COLOR_WRITE_ENABLE_ALL.0 as u8 & !D3D11_COLOR_WRITE_ENABLE_ALPHA.0 as u8;

    unsafe {
        let mut state = None;
        device.CreateBlendState(&desc, Some(&mut state))?;
        Ok(state.unwrap())
    }
}

#[inline]
fn create_blend_state_for_path_rasterization(device: &ID3D11Device) -> Result<ID3D11BlendState> {
    let mut desc = D3D11_BLEND_DESC::default();
    desc.RenderTarget[0].BlendEnable = true.into();
    desc.RenderTarget[0].BlendOp = D3D11_BLEND_OP_ADD;
    desc.RenderTarget[0].BlendOpAlpha = D3D11_BLEND_OP_ADD;
    desc.RenderTarget[0].SrcBlend = D3D11_BLEND_ONE;
    desc.RenderTarget[0].SrcBlendAlpha = D3D11_BLEND_ONE;
    desc.RenderTarget[0].DestBlend = D3D11_BLEND_INV_SRC_ALPHA;
    desc.RenderTarget[0].DestBlendAlpha = D3D11_BLEND_INV_SRC_ALPHA;
    desc.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL.0 as u8;
    unsafe {
        let mut state = None;
        device.CreateBlendState(&desc, Some(&mut state))?;
        Ok(state.unwrap())
    }
}

#[inline]
fn create_blend_state_for_path_sprite(device: &ID3D11Device) -> Result<ID3D11BlendState> {
    let mut desc = D3D11_BLEND_DESC::default();
    desc.RenderTarget[0].BlendEnable = true.into();
    desc.RenderTarget[0].BlendOp = D3D11_BLEND_OP_ADD;
    desc.RenderTarget[0].BlendOpAlpha = D3D11_BLEND_OP_ADD;
    desc.RenderTarget[0].SrcBlend = D3D11_BLEND_ONE;
    desc.RenderTarget[0].SrcBlendAlpha = D3D11_BLEND_ONE;
    desc.RenderTarget[0].DestBlend = D3D11_BLEND_INV_SRC_ALPHA;
    desc.RenderTarget[0].DestBlendAlpha = D3D11_BLEND_ONE;
    desc.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL.0 as u8;
    unsafe {
        let mut state = None;
        device.CreateBlendState(&desc, Some(&mut state))?;
        Ok(state.unwrap())
    }
}

#[inline]
fn create_vertex_shader(device: &ID3D11Device, bytes: &[u8]) -> Result<ID3D11VertexShader> {
    unsafe {
        let mut shader = None;
        device.CreateVertexShader(bytes, None, Some(&mut shader))?;
        Ok(shader.unwrap())
    }
}

#[inline]
fn create_fragment_shader(device: &ID3D11Device, bytes: &[u8]) -> Result<ID3D11PixelShader> {
    unsafe {
        let mut shader = None;
        device.CreatePixelShader(bytes, None, Some(&mut shader))?;
        Ok(shader.unwrap())
    }
}

#[inline]
fn create_constant_buffer<T>(device: &ID3D11Device) -> Result<Option<ID3D11Buffer>> {
    const { assert!(std::mem::size_of::<T>() != 0 && std::mem::size_of::<T>().is_multiple_of(16)) };
    let desc = D3D11_BUFFER_DESC {
        ByteWidth: std::mem::size_of::<T>() as u32,
        Usage: D3D11_USAGE_DYNAMIC,
        BindFlags: D3D11_BIND_CONSTANT_BUFFER.0 as u32,
        CPUAccessFlags: D3D11_CPU_ACCESS_WRITE.0 as u32,
        MiscFlags: 0,
        StructureByteStride: 0,
    };
    let mut buffer = None;
    unsafe { device.CreateBuffer(&desc, None, Some(&mut buffer)) }?;
    Ok(buffer)
}

#[inline]
fn create_buffer(
    device: &ID3D11Device,
    element_size: usize,
    buffer_size: usize,
) -> Result<ID3D11Buffer> {
    let desc = D3D11_BUFFER_DESC {
        ByteWidth: (element_size * buffer_size) as u32,
        Usage: D3D11_USAGE_DYNAMIC,
        BindFlags: D3D11_BIND_SHADER_RESOURCE.0 as u32,
        CPUAccessFlags: D3D11_CPU_ACCESS_WRITE.0 as u32,
        MiscFlags: D3D11_RESOURCE_MISC_BUFFER_STRUCTURED.0 as u32,
        StructureByteStride: element_size as u32,
    };
    let mut buffer = None;
    unsafe { device.CreateBuffer(&desc, None, Some(&mut buffer)) }?;
    Ok(buffer.unwrap())
}

#[inline]
fn create_buffer_view(
    device: &ID3D11Device,
    buffer: &ID3D11Buffer,
) -> Result<Option<ID3D11ShaderResourceView>> {
    let mut view = None;
    unsafe { device.CreateShaderResourceView(buffer, None, Some(&mut view)) }?;
    Ok(view)
}

#[inline]
fn update_buffer<T>(
    device_context: &ID3D11DeviceContext,
    buffer: &ID3D11Buffer,
    data: &[T],
) -> Result<()> {
    unsafe {
        let mut dest = std::mem::zeroed();
        device_context.Map(buffer, 0, D3D11_MAP_WRITE_DISCARD, 0, Some(&mut dest))?;
        std::ptr::copy_nonoverlapping(data.as_ptr(), dest.pData as _, data.len());
        device_context.Unmap(buffer, 0);
    }
    Ok(())
}

#[inline]
/// Whether the quad named at `position` in [`Scene::blended_quad_indices`] is
/// glass content. Out-of-range positions report `false` rather than panicking;
/// the caller only ever walks a range the batch iterator produced.
fn blended_quad_is_glass(scene: &Scene, position: usize) -> bool {
    scene
        .blended_quad_indices
        .get(position)
        .and_then(|&quad_id| scene.quads.get(quad_id as usize))
        .is_some_and(|quad| quad.background.is_glass_content())
}

fn update_batch_start(
    device_context: &ID3D11DeviceContext,
    buffer: &ID3D11Buffer,
    first_instance: u32,
) -> Result<()> {
    update_buffer(
        device_context,
        buffer,
        &[BatchParams {
            start_index: first_instance,
            _padding: [0; 3],
        }],
    )
}

#[inline]
fn set_pipeline_state(
    device_context: &ID3D11DeviceContext,
    buffer_view: &[Option<ID3D11ShaderResourceView>],
    topology: D3D_PRIMITIVE_TOPOLOGY,
    vertex_shader: &ID3D11VertexShader,
    fragment_shader: &ID3D11PixelShader,
    blend_state: &ID3D11BlendState,
) {
    unsafe {
        device_context.VSSetShaderResources(1, Some(buffer_view));
        device_context.PSSetShaderResources(1, Some(buffer_view));
        device_context.IASetPrimitiveTopology(topology);
        device_context.VSSetShader(vertex_shader, None);
        device_context.PSSetShader(fragment_shader, None);
        device_context.OMSetBlendState(blend_state, None, 0xFFFFFFFF);
    }
}

#[cfg(debug_assertions)]
fn report_live_objects(device: &ID3D11Device) -> Result<()> {
    let debug_device: ID3D11Debug = device.cast()?;
    unsafe {
        debug_device.ReportLiveDeviceObjects(D3D11_RLDO_DETAIL)?;
    }
    Ok(())
}

const BUFFER_COUNT: usize = 3;

pub(crate) mod shader_resources {
    use anyhow::Result;

    #[cfg(debug_assertions)]
    use windows::{
        Win32::Graphics::Direct3D::{Fxc::*, ID3DBlob},
        core::{HSTRING, PCSTR},
    };

    #[derive(Copy, Clone, Debug, Eq, PartialEq)]
    pub(crate) enum ShaderModule {
        Quad,
        OpaqueQuad,
        Shadow,
        Underline,
        PathRasterization,
        PathSprite,
        MonochromeSprite,
        SubpixelSprite,
        PolychromeSprite,
        EmojiRasterization,
    }

    #[derive(Copy, Clone, Debug, Eq, PartialEq)]
    pub(crate) enum ShaderTarget {
        Vertex,
        Fragment,
    }

    pub(crate) struct RawShaderBytes<'t> {
        inner: &'t [u8],

        #[cfg(debug_assertions)]
        _blob: ID3DBlob,
    }

    impl<'t> RawShaderBytes<'t> {
        pub(crate) fn new(module: ShaderModule, target: ShaderTarget) -> Result<Self> {
            #[cfg(not(debug_assertions))]
            {
                Ok(Self::from_bytes(module, target))
            }
            #[cfg(debug_assertions)]
            {
                let blob = build_shader_blob(module, target)?;
                let inner = unsafe {
                    std::slice::from_raw_parts(
                        blob.GetBufferPointer() as *const u8,
                        blob.GetBufferSize(),
                    )
                };
                Ok(Self { inner, _blob: blob })
            }
        }

        pub(crate) fn as_bytes(&'t self) -> &'t [u8] {
            self.inner
        }

        #[cfg(not(debug_assertions))]
        fn from_bytes(module: ShaderModule, target: ShaderTarget) -> Self {
            let bytes = match module {
                ShaderModule::Quad => match target {
                    ShaderTarget::Vertex => QUAD_VERTEX_BYTES,
                    ShaderTarget::Fragment => QUAD_FRAGMENT_BYTES,
                },
                ShaderModule::OpaqueQuad => match target {
                    ShaderTarget::Vertex => OPAQUE_QUAD_VERTEX_BYTES,
                    ShaderTarget::Fragment => OPAQUE_QUAD_FRAGMENT_BYTES,
                },
                ShaderModule::Shadow => match target {
                    ShaderTarget::Vertex => SHADOW_VERTEX_BYTES,
                    ShaderTarget::Fragment => SHADOW_FRAGMENT_BYTES,
                },
                ShaderModule::Underline => match target {
                    ShaderTarget::Vertex => UNDERLINE_VERTEX_BYTES,
                    ShaderTarget::Fragment => UNDERLINE_FRAGMENT_BYTES,
                },
                ShaderModule::PathRasterization => match target {
                    ShaderTarget::Vertex => PATH_RASTERIZATION_VERTEX_BYTES,
                    ShaderTarget::Fragment => PATH_RASTERIZATION_FRAGMENT_BYTES,
                },
                ShaderModule::PathSprite => match target {
                    ShaderTarget::Vertex => PATH_SPRITE_VERTEX_BYTES,
                    ShaderTarget::Fragment => PATH_SPRITE_FRAGMENT_BYTES,
                },
                ShaderModule::MonochromeSprite => match target {
                    ShaderTarget::Vertex => MONOCHROME_SPRITE_VERTEX_BYTES,
                    ShaderTarget::Fragment => MONOCHROME_SPRITE_FRAGMENT_BYTES,
                },
                ShaderModule::SubpixelSprite => match target {
                    ShaderTarget::Vertex => SUBPIXEL_SPRITE_VERTEX_BYTES,
                    ShaderTarget::Fragment => SUBPIXEL_SPRITE_FRAGMENT_BYTES,
                },
                ShaderModule::PolychromeSprite => match target {
                    ShaderTarget::Vertex => POLYCHROME_SPRITE_VERTEX_BYTES,
                    ShaderTarget::Fragment => POLYCHROME_SPRITE_FRAGMENT_BYTES,
                },
                ShaderModule::EmojiRasterization => match target {
                    ShaderTarget::Vertex => EMOJI_RASTERIZATION_VERTEX_BYTES,
                    ShaderTarget::Fragment => EMOJI_RASTERIZATION_FRAGMENT_BYTES,
                },
            };
            Self { inner: bytes }
        }
    }

    #[cfg(debug_assertions)]
    pub(super) fn build_shader_blob(entry: ShaderModule, target: ShaderTarget) -> Result<ID3DBlob> {
        unsafe {
            use windows::Win32::Graphics::{
                Direct3D::ID3DInclude, Hlsl::D3D_COMPILE_STANDARD_FILE_INCLUDE,
            };

            let shader_name = if matches!(entry, ShaderModule::EmojiRasterization) {
                "color_text_raster.hlsl"
            } else {
                "shaders.hlsl"
            };

            let entry = format!(
                "{}_{}\0",
                entry.as_str(),
                match target {
                    ShaderTarget::Vertex => "vertex",
                    ShaderTarget::Fragment => "fragment",
                }
            );
            let target = match target {
                ShaderTarget::Vertex => "vs_4_1\0",
                ShaderTarget::Fragment => "ps_4_1\0",
            };

            let mut compile_blob = None;
            let mut error_blob = None;
            let shader_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join(&format!("src/{}", shader_name))
                .canonicalize()?;

            let entry_point = PCSTR::from_raw(entry.as_ptr());
            let target_cstr = PCSTR::from_raw(target.as_ptr());

            // really dirty trick because winapi bindings are unhappy otherwise
            let include_handler = &std::mem::transmute::<usize, ID3DInclude>(
                D3D_COMPILE_STANDARD_FILE_INCLUDE as usize,
            );

            let ret = D3DCompileFromFile(
                &HSTRING::from(shader_path.to_str().unwrap()),
                None,
                include_handler,
                entry_point,
                target_cstr,
                D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION,
                0,
                &mut compile_blob,
                Some(&mut error_blob),
            );
            if ret.is_err() {
                let Some(error_blob) = error_blob else {
                    return Err(anyhow::anyhow!("{ret:?}"));
                };

                let error_string =
                    std::ffi::CStr::from_ptr(error_blob.GetBufferPointer() as *const i8)
                        .to_string_lossy();
                log::error!("Shader compile error: {}", error_string);
                return Err(anyhow::anyhow!("Compile error: {}", error_string));
            }
            Ok(compile_blob.unwrap())
        }
    }

    #[cfg(not(debug_assertions))]
    include!(concat!(env!("OUT_DIR"), "/shaders_bytes.rs"));

    #[cfg(debug_assertions)]
    impl ShaderModule {
        pub fn as_str(self) -> &'static str {
            match self {
                ShaderModule::Quad => "quad",
                ShaderModule::OpaqueQuad => "opaque_quad",
                ShaderModule::Shadow => "shadow",
                ShaderModule::Underline => "underline",
                ShaderModule::PathRasterization => "path_rasterization",
                ShaderModule::PathSprite => "path_sprite",
                ShaderModule::MonochromeSprite => "monochrome_sprite",
                ShaderModule::SubpixelSprite => "subpixel_sprite",
                ShaderModule::PolychromeSprite => "polychrome_sprite",
                ShaderModule::EmojiRasterization => "emoji_rasterization",
            }
        }
    }
}

mod nvidia {
    use std::{
        ffi::CStr,
        os::raw::{c_char, c_int, c_uint},
    };

    use anyhow::Result;
    use windows::{Win32::System::LibraryLoader::GetProcAddress, core::s};

    use crate::with_dll_library;

    // https://github.com/NVIDIA/nvapi/blob/7cb76fce2f52de818b3da497af646af1ec16ce27/nvapi_lite_common.h#L180
    const NVAPI_SHORT_STRING_MAX: usize = 64;

    // https://github.com/NVIDIA/nvapi/blob/7cb76fce2f52de818b3da497af646af1ec16ce27/nvapi_lite_common.h#L235
    #[allow(non_camel_case_types)]
    type NvAPI_ShortString = [c_char; NVAPI_SHORT_STRING_MAX];

    // https://github.com/NVIDIA/nvapi/blob/7cb76fce2f52de818b3da497af646af1ec16ce27/nvapi_lite_common.h#L447
    #[allow(non_camel_case_types)]
    type NvAPI_SYS_GetDriverAndBranchVersion_t = unsafe extern "C" fn(
        driver_version: *mut c_uint,
        build_branch_string: *mut NvAPI_ShortString,
    ) -> c_int;

    pub(super) fn get_driver_version() -> Result<String> {
        #[cfg(target_pointer_width = "64")]
        let nvidia_dll_name = s!("nvapi64.dll");
        #[cfg(target_pointer_width = "32")]
        let nvidia_dll_name = s!("nvapi.dll");

        with_dll_library(nvidia_dll_name, |nvidia_dll| unsafe {
            let nvapi_query_addr = GetProcAddress(nvidia_dll, s!("nvapi_QueryInterface"))
                .ok_or_else(|| anyhow::anyhow!("Failed to get nvapi_QueryInterface address"))?;
            let nvapi_query: extern "C" fn(u32) -> *mut () = std::mem::transmute(nvapi_query_addr);

            // https://github.com/NVIDIA/nvapi/blob/7cb76fce2f52de818b3da497af646af1ec16ce27/nvapi_interface.h#L41
            let nvapi_get_driver_version_ptr = nvapi_query(0x2926aaad);
            if nvapi_get_driver_version_ptr.is_null() {
                anyhow::bail!("Failed to get NVIDIA driver version function pointer");
            }
            let nvapi_get_driver_version: NvAPI_SYS_GetDriverAndBranchVersion_t =
                std::mem::transmute(nvapi_get_driver_version_ptr);

            let mut driver_version: c_uint = 0;
            let mut build_branch_string: NvAPI_ShortString = [0; NVAPI_SHORT_STRING_MAX];
            let result = nvapi_get_driver_version(
                &mut driver_version as *mut c_uint,
                &mut build_branch_string as *mut NvAPI_ShortString,
            );

            if result != 0 {
                anyhow::bail!(
                    "Failed to get NVIDIA driver version, error code: {}",
                    result
                );
            }
            let major = driver_version / 100;
            let minor = driver_version % 100;
            let branch_string = CStr::from_ptr(build_branch_string.as_ptr());
            Ok(format!(
                "{}.{} {}",
                major,
                minor,
                branch_string.to_string_lossy()
            ))
        })
    }
}

mod amd {
    use std::os::raw::{c_char, c_int, c_void};

    use anyhow::Result;
    use windows::{Win32::System::LibraryLoader::GetProcAddress, core::s};

    use crate::with_dll_library;

    // https://github.com/GPUOpen-LibrariesAndSDKs/AGS_SDK/blob/5d8812d703d0335741b6f7ffc37838eeb8b967f7/ags_lib/inc/amd_ags.h#L145
    const AGS_CURRENT_VERSION: i32 = (6 << 22) | (3 << 12);

    // https://github.com/GPUOpen-LibrariesAndSDKs/AGS_SDK/blob/5d8812d703d0335741b6f7ffc37838eeb8b967f7/ags_lib/inc/amd_ags.h#L204
    // This is an opaque type, using struct to represent it properly for FFI
    #[repr(C)]
    struct AGSContext {
        _private: [u8; 0],
    }

    #[repr(C)]
    pub struct AGSGPUInfo {
        pub driver_version: *const c_char,
        pub radeon_software_version: *const c_char,
        pub num_devices: c_int,
        pub devices: *mut c_void,
    }

    // https://github.com/GPUOpen-LibrariesAndSDKs/AGS_SDK/blob/5d8812d703d0335741b6f7ffc37838eeb8b967f7/ags_lib/inc/amd_ags.h#L429
    #[allow(non_camel_case_types)]
    type agsInitialize_t = unsafe extern "C" fn(
        version: c_int,
        config: *const c_void,
        context: *mut *mut AGSContext,
        gpu_info: *mut AGSGPUInfo,
    ) -> c_int;

    // https://github.com/GPUOpen-LibrariesAndSDKs/AGS_SDK/blob/5d8812d703d0335741b6f7ffc37838eeb8b967f7/ags_lib/inc/amd_ags.h#L436
    #[allow(non_camel_case_types)]
    type agsDeInitialize_t = unsafe extern "C" fn(context: *mut AGSContext) -> c_int;

    pub(super) fn get_driver_version() -> Result<String> {
        #[cfg(target_pointer_width = "64")]
        let amd_dll_name = s!("amd_ags_x64.dll");
        #[cfg(target_pointer_width = "32")]
        let amd_dll_name = s!("amd_ags_x86.dll");

        with_dll_library(amd_dll_name, |amd_dll| unsafe {
            let ags_initialize_addr = GetProcAddress(amd_dll, s!("agsInitialize"))
                .ok_or_else(|| anyhow::anyhow!("Failed to get agsInitialize address"))?;
            let ags_deinitialize_addr = GetProcAddress(amd_dll, s!("agsDeInitialize"))
                .ok_or_else(|| anyhow::anyhow!("Failed to get agsDeInitialize address"))?;

            let ags_initialize: agsInitialize_t = std::mem::transmute(ags_initialize_addr);
            let ags_deinitialize: agsDeInitialize_t = std::mem::transmute(ags_deinitialize_addr);

            let mut context: *mut AGSContext = std::ptr::null_mut();
            let mut gpu_info: AGSGPUInfo = AGSGPUInfo {
                driver_version: std::ptr::null(),
                radeon_software_version: std::ptr::null(),
                num_devices: 0,
                devices: std::ptr::null_mut(),
            };

            let result = ags_initialize(
                AGS_CURRENT_VERSION,
                std::ptr::null(),
                &mut context,
                &mut gpu_info,
            );
            if result != 0 {
                anyhow::bail!("Failed to initialize AMD AGS, error code: {}", result);
            }

            // Vulkan actually returns this as the driver version
            let software_version = if !gpu_info.radeon_software_version.is_null() {
                std::ffi::CStr::from_ptr(gpu_info.radeon_software_version)
                    .to_string_lossy()
                    .into_owned()
            } else {
                "Unknown Radeon Software Version".to_string()
            };

            let driver_version = if !gpu_info.driver_version.is_null() {
                std::ffi::CStr::from_ptr(gpu_info.driver_version)
                    .to_string_lossy()
                    .into_owned()
            } else {
                "Unknown Radeon Driver Version".to_string()
            };

            ags_deinitialize(context);
            Ok(format!("{} ({})", software_version, driver_version))
        })
    }
}

mod dxgi {
    use windows::{
        Win32::Graphics::Dxgi::{IDXGIAdapter1, IDXGIDevice},
        core::Interface,
    };

    pub(super) fn get_driver_version(adapter: &IDXGIAdapter1) -> anyhow::Result<String> {
        let number = unsafe { adapter.CheckInterfaceSupport(&IDXGIDevice::IID as _) }?;
        Ok(format!(
            "{}.{}.{}.{}",
            number >> 48,
            (number >> 32) & 0xFFFF,
            (number >> 16) & 0xFFFF,
            number & 0xFFFF
        ))
    }
}

/// Renders scenes to images with no window, so tests can assert on the pixels the real
/// DirectX pipeline actually produces rather than on the scene that was handed to it.
#[cfg(any(feature = "bench-support", feature = "test-support"))]
pub struct DirectXHeadlessRenderer {
    renderer: DirectXRenderer,
}

#[cfg(any(feature = "bench-support", feature = "test-support"))]
impl DirectXHeadlessRenderer {
    /// Returns `None` when no usable D3D11 adapter is available, so callers can skip rather
    /// than fail for a reason unrelated to what they are testing.
    pub fn new() -> Option<Self> {
        let devices = DirectXDevices::new().log_err()?;
        let renderer = DirectXRenderer::new_headless(&devices).log_err()?;
        Some(Self { renderer })
    }
}

#[cfg(any(feature = "bench-support", feature = "test-support"))]
impl gpui::PlatformHeadlessRenderer for DirectXHeadlessRenderer {
    fn render_scene_to_image(
        &mut self,
        scene: &Scene,
        size: Size<DevicePixels>,
    ) -> Result<image::RgbaImage> {
        self.renderer.resize(size)?;
        // Opaque black, so anti-aliased text has a stable background to be measured against.
        self.renderer.draw(scene, [0.0, 0.0, 0.0, 1.0])?;
        self.renderer.read_back_render_target()
    }

    fn render_scene(&mut self, scene: &Scene, size: Size<DevicePixels>) -> Result<()> {
        self.renderer.resize(size)?;
        self.renderer.draw(scene, [0.0, 0.0, 0.0, 1.0])
    }

    fn sprite_atlas(&self) -> Arc<dyn PlatformAtlas> {
        self.renderer.sprite_atlas()
    }
}


/// Pixel-level verification through the real DirectX pipeline.
///
/// Every other harness in this crate stops at the scene: they prove the right sprites, with the
/// right tiles, were handed to the renderer. This one renders that scene on the GPU with the
/// real shaders, blend states and atlas bindings, reads the framebuffer back, and asserts the
/// glyphs actually put ink on screen.
///
/// That closes the last gap in the "characters go missing" investigation — a glyph can be
/// present and correct in the scene and still never appear, if the draw path loses it.
#[cfg(all(test, feature = "test-support"))]
mod rendered_pixel_tests {
    use crate::{DirectWriteTextSystem, DirectXDevices, DirectXHeadlessRenderer};
    use gpui::{
        AppContext as _, ContentMask, Context, IntoElement, ParentElement as _, Pixels, Point,
        Render, Styled as _, TestAppContext, TestDispatcher, TextAlign, TextRun, Window, canvas,
        div, font, hsla, point, px, size,
    };
    use gpui_util::ResultExt as _;
    use std::sync::Arc;

    const TEXT: &str = "Changes in this project";
    const FONT_SIZE: Pixels = px(16.);
    const LINE_HEIGHT: Pixels = px(24.);

    struct TextView {
        origin: Point<Pixels>,
    }

    impl Render for TextView {
        fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
            let origin = self.origin;
            div().size_full().child(
                canvas(
                    |_, _, _| (),
                    move |bounds, _, window, cx| {
                        let runs = [TextRun {
                            len: TEXT.len(),
                            font: font(".SystemUIFont"),
                            // White on the renderer's black clear colour, so ink is unambiguous.
                            color: hsla(0., 0., 1., 1.),
                            background_color: None,
                            underline: None,
                            strikethrough: None,
                        }];
                        let line = window
                            .text_system()
                            .shape_line(TEXT.into(), FONT_SIZE, &runs, None);
                        window.with_content_mask(Some(ContentMask { bounds }), |window| {
                            line.paint(origin, LINE_HEIGHT, TextAlign::Left, None, window, cx)
                                .unwrap();
                        });
                    },
                )
                .size_full(),
            )
        }
    }

    /// Counts runs of columns containing ink. Adjacent glyphs can touch, so this is a lower
    /// bound on the number of characters drawn — but a glyph dropping out removes or splits a
    /// run, which is what a character missing from a word looks like.
    fn lit_column_runs(image: &image::RgbaImage) -> usize {
        let mut runs = 0;
        let mut in_run = false;
        for x in 0..image.width() {
            let lit = (0..image.height()).any(|y| {
                let pixel = image.get_pixel(x, y);
                // Anti-aliased edges are dim; anything clearly above the black clear colour
                // counts as ink.
                pixel[0] as u32 + pixel[1] as u32 + pixel[2] as u32 > 90
            });
            if lit && !in_run {
                runs += 1;
            }
            in_run = lit;
        }
        runs
    }

    /// Text that reached the scene has to reach the framebuffer. Anything lost between the two —
    /// instance buffer, batching, atlas binding, shader — shows up here as missing ink.
    ///
    /// Swept over sub-pixel origins, the axis the reported dropouts were sensitive to.
    ///
    /// `subpixel` picks which glyph pipeline to exercise. It matters more than it looks: the two
    /// share only the shaper. Subpixel glyphs go to a separate RGBA8 atlas, become
    /// `SubpixelSprite`s, and are drawn by a different shader under a dual-source blend state.
    /// Windows always takes that path in production and macOS can never take it.
    fn sweep_origins(subpixel: bool) -> Option<Vec<String>> {
        let devices = DirectXDevices::new().log_err()?;
        let text_system = DirectWriteTextSystem::new(&devices).log_err()?;
        DirectXHeadlessRenderer::new()?;

        gpui::TestWindow::set_subpixel_rendering_supported(subpixel);

        let mut cx = TestAppContext::build_with_platform(
            TestDispatcher::new(0),
            None,
            Arc::new(text_system),
            Some(Box::new(|| {
                DirectXHeadlessRenderer::new()
                    .map(|renderer| Box::new(renderer) as Box<dyn gpui::PlatformHeadlessRenderer>)
            })),
        );

        let mut failures = Vec::new();
        let mut reference_ink: Option<u64> = None;

        for origin_x in [0.0f32, 0.25, 0.5, 0.75, 1.3] {
            for origin_y in [4.0f32, 4.5, 11.25] {
                let origin = point(px(origin_x), px(origin_y));
                let window = cx.add_window(move |_, _| TextView { origin });
                cx.update_window(window.into(), |_, window, cx| {
                    window.resize(size(px(600.), px(80.)));
                    window.draw(cx).clear(cx);
                })
                .unwrap();

                let ((monochrome, subpixel_sprites), image) = cx
                    .update_window(window.into(), |_, window, _| {
                        (
                            window.rendered_glyph_sprite_counts(),
                            window.render_to_image(),
                        )
                    })
                    .unwrap();
                let image = image.expect("rendering the scene should succeed");
                let runs = lit_column_runs(&image);
                // Eyeballing the actual output is often faster than reading numbers, so allow
                // dumping it: GPUI_DUMP_RENDERED_TEXT=<dir> writes one PNG per origin.
                if let Ok(directory) = std::env::var("GPUI_DUMP_RENDERED_TEXT") {
                    let kind = if subpixel { "subpixel" } else { "grayscale" };
                    let path = std::path::Path::new(&directory)
                        .join(format!("text-{kind}-{origin_x}-{origin_y}.png"));
                    image.save(&path).log_err();
                }

                // Guard against the test silently covering the wrong pipeline.
                let (expected, wrong) = if subpixel {
                    (subpixel_sprites, monochrome)
                } else {
                    (monochrome, subpixel_sprites)
                };
                if expected == 0 || wrong != 0 {
                    failures.push(format!(
                        "  origin ({origin_x}, {origin_y}): wanted the {} pipeline but got \
                         {monochrome} monochrome and {subpixel_sprites} subpixel sprites",
                        if subpixel { "subpixel" } else { "grayscale" },
                    ));
                    continue;
                }
                let sprites = expected;
                if runs == 0 {
                    failures.push(format!(
                        "  origin ({origin_x}, {origin_y}): {sprites} sprites in the scene but \
                         the framebuffer is blank"
                    ));
                    continue;
                }
                // Total ink is the load-bearing measurement, not the run count. Run count moves
                // for a benign reason: at some sub-pixel offsets neighbouring glyphs touch and
                // merge into one run, which is why this string reports 19 runs at x=0.0 and 17
                // at x=0.25 on both pipelines while rendering identically. Ink mass does not
                // care about merging — a glyph that actually failed to draw removes its
                // coverage, and one missing character out of twenty is a multi-percent drop,
                // far outside the sub-0.01% wobble that anti-aliasing produces.
                let ink: u64 = image
                    .pixels()
                    .map(|p| (p[0] as u64 + p[1] as u64 + p[2] as u64) / 3)
                    .sum();
                match reference_ink {
                    None => reference_ink = Some(ink),
                    Some(reference) if ink * 100 < reference * 98 => failures.push(format!(
                        "  origin ({origin_x}, {origin_y}): {ink} ink on screen but {reference} \
                         at the reference origin, with {sprites} sprites in the scene — ink went \
                         missing in the draw path"
                    )),
                    Some(_) => {}
                }
                eprintln!(
                    "  origin ({origin_x}, {origin_y}): {sprites} sprites, {runs} lit runs, \
                     ink {ink}"
                );
            }
        }

        Some(failures)
    }

    #[test]
    fn shaped_text_reaches_the_framebuffer_grayscale() {
        let Some(failures) = sweep_origins(false) else {
            eprintln!("SKIPPED: no D3D11 adapter, DirectWrite or headless renderer");
            return;
        };
        assert!(
            failures.is_empty(),
            "{} origin(s) lost text between the scene and the framebuffer:\n{}",
            failures.len(),
            failures.join("\n")
        );
    }

    /// The pipeline fincode actually runs on Windows. Every earlier harness in this
    /// investigation went through `TestAppContext`, whose window reported subpixel rendering as
    /// unsupported, so they all covered the grayscale path — on Windows hardware, but never the
    /// Windows code.
    #[test]
    fn shaped_text_reaches_the_framebuffer_subpixel() {
        let Some(failures) = sweep_origins(true) else {
            eprintln!("SKIPPED: no D3D11 adapter, DirectWrite or headless renderer");
            return;
        };
        assert!(
            failures.is_empty(),
            "{} origin(s) lost text between the scene and the framebuffer:\n{}",
            failures.len(),
            failures.join("\n")
        );
    }

    struct PathView;

    impl Render for PathView {
        fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
            div().size_full().child(
                canvas(
                    |_, _, _| (),
                    move |_, _, window, _cx| {
                        // A filled rectangle spanning several 512px raster
                        // tiles in both axes, so every seam between tiles runs
                        // through its interior.
                        let mut builder = gpui::PathBuilder::fill();
                        builder.move_to(point(px(100.), px(100.)));
                        builder.line_to(point(px(1100.), px(100.)));
                        builder.line_to(point(px(1100.), px(700.)));
                        builder.line_to(point(px(100.), px(700.)));
                        builder.close();
                        if let Ok(path) = builder.build() {
                            window.paint_path(path, hsla(0., 0., 1., 1.));
                        }
                        // A second path in a tile the rectangle never touches,
                        // in the same batch — the far-tile case.
                        let mut builder = gpui::PathBuilder::fill();
                        builder.move_to(point(px(20.), px(730.)));
                        builder.line_to(point(px(80.), px(730.)));
                        builder.line_to(point(px(20.), px(790.)));
                        builder.close();
                        if let Ok(path) = builder.build() {
                            window.paint_path(path, hsla(0., 0., 1., 1.));
                        }
                    },
                )
                .size_full(),
            )
        }
    }

    /// Paths rasterize through a fixed 512px MSAA tile that is resolved and
    /// region-copied per touched tile (see `PATH_RASTER_TILE_SIZE`). This
    /// renders a filled rectangle whose interior crosses every tile seam of a
    /// 1200x800 window plus a far-tile triangle, and asserts full coverage
    /// inside, emptiness outside, and no seam artifacts along tile boundaries.
    #[test]
    fn paths_render_seamlessly_across_raster_tiles() {
        let Some(devices) = DirectXDevices::new().log_err() else {
            eprintln!("SKIPPED: no D3D11 adapter");
            return;
        };
        let Some(text_system) = DirectWriteTextSystem::new(&devices).log_err() else {
            eprintln!("SKIPPED: no DirectWrite");
            return;
        };
        if DirectXHeadlessRenderer::new().is_none() {
            eprintln!("SKIPPED: no headless renderer");
            return;
        }

        let mut cx = TestAppContext::build_with_platform(
            TestDispatcher::new(0),
            None,
            Arc::new(text_system),
            Some(Box::new(|| {
                DirectXHeadlessRenderer::new()
                    .map(|renderer| Box::new(renderer) as Box<dyn gpui::PlatformHeadlessRenderer>)
            })),
        );

        let window = cx.add_window(move |_, _| PathView);
        let image = cx
            .update_window(window.into(), |_, window, cx| {
                window.resize(size(px(1200.), px(800.)));
                window.draw(cx).clear(cx);
                window.render_to_image()
            })
            .unwrap()
            .expect("rendering the scene should succeed");

        if let Ok(directory) = std::env::var("GPUI_DUMP_RENDERED_TEXT") {
            let path = std::path::Path::new(&directory).join("path-tiles.png");
            image.save(&path).log_err();
        }

        // The image is device pixels; the paths were painted in logical
        // pixels. The scale factor is whatever the test window used.
        let scale = image.width() as f32 / 1200.0;
        let device = |logical: f32| (logical * scale).round() as u32;
        let lit = |x: u32, y: u32| {
            let pixel = image.get_pixel(x, y);
            pixel[0] as u32 + pixel[1] as u32 + pixel[2] as u32 > 380
        };
        // Interior: every device pixel well inside the rectangle must be lit.
        // The dense scan crosses every 512-multiple tile seam, so a tile that
        // failed to rasterize, resolved to the wrong place, or was skipped by
        // the occupancy grid shows up as a 512-aligned hole or dark seam.
        let mut dark_interior = 0u32;
        for y in (device(102.0)..device(698.0)).step_by(3) {
            for x in (device(102.0)..device(1098.0)).step_by(3) {
                if !lit(x, y) {
                    dark_interior += 1;
                }
            }
        }
        assert_eq!(
            dark_interior, 0,
            "dark pixels inside the filled rectangle (tile hole or seam)"
        );
        // Tile seams run at device-pixel multiples of 512; probe the columns
        // and rows around every seam the rectangle's interior crosses.
        let interior_x = device(105.0)..device(1095.0);
        let interior_y = device(105.0)..device(695.0);
        let mut seam = 512u32;
        while seam < image.width() {
            if interior_x.contains(&seam) {
                for probe in seam - 2..=seam + 2 {
                    assert!(lit(probe, device(400.0)), "seam column {probe} is dark");
                }
            }
            if interior_y.contains(&seam) {
                for probe in seam - 2..=seam + 2 {
                    assert!(lit(device(600.0), probe), "seam row {probe} is dark");
                }
            }
            seam += 512;
        }
        // Far tile: the triangle near (20..80, 730..790) landed.
        assert!(
            lit(device(30.0), device(740.0)),
            "far-tile triangle did not render"
        );
        // Outside: no ink beyond the shapes (a tile copied to the wrong
        // offset would smear ink into these).
        for (x, y) in [
            (device(50.0), device(50.0)),
            (device(1150.0), device(100.0)),
            (device(1150.0), device(750.0)),
            (device(600.0), device(750.0)),
            (device(97.0), device(97.0)),
        ] {
            assert!(!lit(x, y), "unexpected ink at ({x}, {y})");
        }
    }

    struct GhostView {
        second_frame: std::rc::Rc<std::cell::Cell<bool>>,
    }

    impl Render for GhostView {
        fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
            let second_frame = self.second_frame.get();
            div().size_full().child(
                canvas(
                    |_, _, _| (),
                    move |_, _, window, _cx| {
                        let square = |left: f32, top: f32, side: f32| {
                            let mut builder = gpui::PathBuilder::fill();
                            builder.move_to(point(px(left), px(top)));
                            builder.line_to(point(px(left + side), px(top)));
                            builder.line_to(point(px(left + side), px(top + side)));
                            builder.line_to(point(px(left), px(top + side)));
                            builder.close();
                            builder.build()
                        };
                        if second_frame {
                            // Two OVERLAPPING squares (the bounds tree gives the
                            // second a higher draw order, so the batch mixes
                            // orders and the sprite pass samples one spanning
                            // rect) plus a far square that stretches that rect
                            // across the tile where frame one's ink was left.
                            for path in [
                                square(100., 100., 80.),
                                square(140., 140., 80.),
                                square(1000., 700., 50.),
                            ]
                            .into_iter()
                            .flatten()
                            {
                                window.paint_path(path, hsla(0., 0., 1., 1.));
                            }
                        } else if let Ok(path) = square(580., 480., 40.) {
                            window.paint_path(path, hsla(0., 0., 1., 1.));
                        }
                    },
                )
                .size_full(),
            )
        }
    }

    /// A mixed-draw-order path batch is sampled from the intermediate through
    /// one spanning rect, which can cross raster tiles the batch never
    /// rewrites. Ink a previous frame left in those tiles must not be
    /// re-composited: frame one draws a lone square, frame two draws a
    /// mixed-order batch whose spanning rect covers the (now path-free) spot
    /// where that square was, and the spot must render dark.
    #[test]
    fn stale_path_ink_does_not_ghost_through_spanning_rect() {
        let Some(devices) = DirectXDevices::new().log_err() else {
            eprintln!("SKIPPED: no D3D11 adapter");
            return;
        };
        let Some(text_system) = DirectWriteTextSystem::new(&devices).log_err() else {
            eprintln!("SKIPPED: no DirectWrite");
            return;
        };
        if DirectXHeadlessRenderer::new().is_none() {
            eprintln!("SKIPPED: no headless renderer");
            return;
        }

        let mut cx = TestAppContext::build_with_platform(
            TestDispatcher::new(0),
            None,
            Arc::new(text_system),
            Some(Box::new(|| {
                DirectXHeadlessRenderer::new()
                    .map(|renderer| Box::new(renderer) as Box<dyn gpui::PlatformHeadlessRenderer>)
            })),
        );

        let second_frame = std::rc::Rc::new(std::cell::Cell::new(false));
        let window = cx.add_window({
            let second_frame = second_frame.clone();
            move |_, _| GhostView { second_frame }
        });

        let first_image = cx
            .update_window(window.into(), |_, window, cx| {
                window.resize(size(px(1200.), px(800.)));
                window.draw(cx).clear(cx);
                window.render_to_image()
            })
            .unwrap()
            .expect("rendering the first frame should succeed");

        second_frame.set(true);
        let second_image = cx
            .update_window(window.into(), |_, window, cx| {
                window.refresh();
                window.draw(cx).clear(cx);
                window.render_to_image()
            })
            .unwrap()
            .expect("rendering the second frame should succeed");

        if let Ok(directory) = std::env::var("GPUI_DUMP_RENDERED_TEXT") {
            let directory = std::path::Path::new(&directory);
            first_image.save(&directory.join("path-ghost-1.png")).log_err();
            second_image.save(&directory.join("path-ghost-2.png")).log_err();
        }

        let scale = first_image.width() as f32 / 1200.0;
        let device = |logical: f32| (logical * scale).round() as u32;
        let lit = |image: &image::RgbaImage, x: u32, y: u32| {
            let pixel = image.get_pixel(x, y);
            pixel[0] as u32 + pixel[1] as u32 + pixel[2] as u32 > 380
        };

        // Frame one: the lone square landed.
        assert!(
            lit(&first_image, device(600.0), device(500.0)),
            "frame one's square did not render"
        );
        // Frame two: its own paths landed...
        assert!(
            lit(&second_image, device(160.0), device(160.0)),
            "the overlapping pair did not render"
        );
        assert!(
            lit(&second_image, device(1025.0), device(725.0)),
            "the far square did not render"
        );
        // ...and frame one's square is gone, not ghosting through the
        // spanning rect from a raster tile this batch never rewrote.
        assert!(
            !lit(&second_image, device(600.0), device(500.0)),
            "stale path ink from the previous frame ghosted back"
        );
    }
}
