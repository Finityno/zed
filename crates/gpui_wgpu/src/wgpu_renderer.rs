use crate::{CompositorGpuHint, DeviceErrorState, WgpuAtlas, WgpuContext};
use anyhow::{Context as _, Result};
use bytemuck::{Pod, Zeroable};
use collections::FxHashMap;
use gpui::{
    AtlasTextureId, Background, Bounds, DevicePixels, GpuSpecs, Path, Point, PrimitiveBatch,
    RenderMemoryGauge, RenderMemoryLedger, ScaledPixels, Scene, Size, get_gamma_correction_ratios,
    quad_depth,
};
use log::warn;
#[cfg(not(target_family = "wasm"))]
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use smallvec::SmallVec;
use std::cell::RefCell;
use std::num::NonZeroU64;
use std::ops::Range;
use std::rc::Rc;
use std::sync::{Arc, OnceLock};

const MAX_INSTANCE_BUFFER_SIZE: u64 = 256 * 1024 * 1024;

const INSTANCE_TEXTURE_TEXEL_SIZE: u64 = 16;

/// Consecutive frames whose instance data fits in a quarter of the instance
/// buffer after which it is reallocated smaller. The buffer only grew before,
/// so one large frame (a huge scroll, a burst of paths) kept up to 256 MiB for
/// the life of the window.
const INSTANCE_SHRINK_FRAMES: u32 = 120;

/// Consecutive path-free frames after which the window-sized path targets are
/// released, the Metal renderer's count (DirectX uses 300). Frames, not time:
/// GPUI only draws on activity and the renderer has no clock, so a window that
/// goes idle keeps its targets for the very next redraw, while two seconds of
/// sustained 60 Hz drawing with no path means they are not wanted. A path that
/// comes back costs one allocation, the same as a resize step.
const PATH_TARGET_IDLE_FRAMES: u32 = 120;

/// Paths rasterize through a multisampled tile of at most this size, as on
/// DirectX, instead of a window-sized MSAA target: at 3200x2000 a 4-sample
/// window target is 102.4 MB and every path batch cleared and resolved all of
/// it, while a 512x512 tile is 4 MiB. Each batch rasterizes only the tiles its
/// sprites sample, resolves each and copies it into the window-sized
/// single-sample intermediate, so the work follows the path area (corner
/// masks, small indicators) instead of the window area.
const PATH_RASTER_TILE_SIZE: u32 = 512;

/// Shader variant for backends with storage buffer support: the shared shader
/// logic plus the storage-buffer instance transport.
const STORAGE_BUFFER_SHADERS: &str = concat!(
    include_str!("shaders.wgsl"),
    include_str!("shaders_storage.wgsl"),
);

/// Shader variant for WebGL2, which has no storage buffers: the shared shader
/// logic plus the texture-based instance transport.
const WEBGL_SHADERS: &str = concat!(
    include_str!("shaders.wgsl"),
    include_str!("shaders_webgl.wgsl"),
);

/// Subpixel text rendering requires dual-source blending, which WebGL2 lacks, so
/// this variant only ever runs with the storage-buffer transport. The `enable`
/// directive must precede all declarations.
const SUBPIXEL_SHADERS: &str = concat!(
    "enable dual_source_blending;\n",
    include_str!("shaders.wgsl"),
    include_str!("shaders_storage.wgsl"),
    include_str!("shaders_subpixel.wgsl"),
);

fn least_common_multiple(left: u64, right: u64) -> u64 {
    let mut first = left;
    let mut second = right;
    while second != 0 {
        let remainder = first % second;
        first = second;
        second = remainder;
    }
    left / first * right
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GlobalParams {
    viewport_size: [f32; 2],
    premultiplied_alpha: u32,
    pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PodBounds {
    origin: [f32; 2],
    size: [f32; 2],
}

impl From<Bounds<ScaledPixels>> for PodBounds {
    fn from(bounds: Bounds<ScaledPixels>) -> Self {
        Self {
            origin: [bounds.origin.x.0, bounds.origin.y.0],
            size: [bounds.size.width.0, bounds.size.height.0],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GammaParams {
    gamma_ratios: [f32; 4],
    grayscale_enhanced_contrast: f32,
    subpixel_enhanced_contrast: f32,
    is_bgr: u32,
    _pad: u32,
}

#[derive(Clone, Debug)]
#[repr(C)]
struct PathSprite {
    bounds: Bounds<ScaledPixels>,
}

#[derive(Clone, Debug)]
#[repr(C)]
struct PathRasterizationVertex {
    xy_position: Point<ScaledPixels>,
    st_position: Point<f32>,
    color: Background,
    bounds: Bounds<ScaledPixels>,
}

pub struct WgpuSurfaceConfig {
    pub size: Size<DevicePixels>,
    pub transparent: bool,
    /// Preferred presentation mode. When `Some`, the renderer will use this
    /// mode if supported by the surface, falling back to `Fifo`.
    /// When `None`, defaults to `Fifo` (VSync).
    ///
    /// Mobile platforms may prefer `Mailbox` (triple-buffering) to avoid
    /// blocking in `get_current_texture()` during lifecycle transitions.
    pub preferred_present_mode: Option<wgpu::PresentMode>,
}

/// `quad_depth` steps in 1/65535 increments and opaque-quad partitioning is
/// capped at 65534 quads, so 16 bits of unorm depth resolve every slot exactly
/// and the texture is half the size of `Depth32Float`. Chosen per adapter in
/// [`RenderingParameters::new`], which falls back to [`FALLBACK_DEPTH_FORMAT`]
/// on any backend that cannot attach it.
const PREFERRED_DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth16Unorm;
const FALLBACK_DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

struct WgpuPipelines {
    quads: wgpu::RenderPipeline,
    opaque_quads: wgpu::RenderPipeline,
    underlines: wgpu::RenderPipeline,
    mono_sprites: wgpu::RenderPipeline,
    subpixel_sprites: Option<wgpu::RenderPipeline>,
    // Compiling a pipeline is most of renderer creation, and a text-only
    // first frame draws none of these, so each is compiled the first time a
    // frame draws it. Replacing `WgpuPipelines` (transparency change, device
    // recovery) starts them over.
    shadows: OnceLock<wgpu::RenderPipeline>,
    path_rasterization: OnceLock<wgpu::RenderPipeline>,
    paths: OnceLock<wgpu::RenderPipeline>,
    poly_sprites: OnceLock<wgpu::RenderPipeline>,
    factory: PipelineFactory,
}

struct PipelineFactory {
    device: wgpu::Device,
    layouts: WgpuBindGroupLayouts,
    shader_module: wgpu::ShaderModule,
    color_target: wgpu::ColorTargetState,
    surface_format: wgpu::TextureFormat,
    depth_format: wgpu::TextureFormat,
    path_sample_count: u32,
}

impl PipelineFactory {
    fn depth_stencil(&self, depth_write_enabled: bool) -> Option<wgpu::DepthStencilState> {
        Some(wgpu::DepthStencilState {
            format: self.depth_format,
            depth_write_enabled: Some(depth_write_enabled),
            depth_compare: Some(wgpu::CompareFunction::Greater),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        })
    }

    fn create_pipeline(
        &self,
        name: &str,
        vs_entry: &str,
        fs_entry: &str,
        globals_layout: &wgpu::BindGroupLayout,
        data_layout: &wgpu::BindGroupLayout,
        texture_layout: Option<&wgpu::BindGroupLayout>,
        topology: wgpu::PrimitiveTopology,
        color_targets: &[Option<wgpu::ColorTargetState>],
        depth_stencil: Option<wgpu::DepthStencilState>,
        sample_count: u32,
        module: &wgpu::ShaderModule,
    ) -> wgpu::RenderPipeline {
        let mut group_layouts = vec![Some(globals_layout), Some(data_layout)];
        group_layouts.extend(texture_layout.map(Some));
        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(&format!("{name}_layout")),
                bind_group_layouts: &group_layouts,
                immediate_size: 0,
            });

        self.device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(name),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module,
                    entry_point: Some(vs_entry),
                    buffers: &[],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module,
                    entry_point: Some(fs_entry),
                    targets: color_targets,
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    unclipped_depth: false,
                    conservative: false,
                },
                depth_stencil,
                multisample: wgpu::MultisampleState {
                    count: sample_count,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                multiview_mask: None,
                cache: None,
            })
    }
}

impl WgpuPipelines {
    fn shadows(&self) -> &wgpu::RenderPipeline {
        self.shadows.get_or_init(|| {
            let factory = &self.factory;
            factory.create_pipeline(
                "shadows",
                "vs_shadow",
                "fs_shadow",
                &factory.layouts.globals,
                &factory.layouts.instances,
                None,
                wgpu::PrimitiveTopology::TriangleStrip,
                &[Some(factory.color_target.clone())],
                factory.depth_stencil(false),
                1,
                &factory.shader_module,
            )
        })
    }

    fn path_rasterization(&self) -> &wgpu::RenderPipeline {
        self.path_rasterization.get_or_init(|| {
            let factory = &self.factory;
            factory.create_pipeline(
                "path_rasterization",
                "vs_path_rasterization",
                "fs_path_rasterization",
                &factory.layouts.globals,
                &factory.layouts.instances,
                None,
                wgpu::PrimitiveTopology::TriangleList,
                &[Some(wgpu::ColorTargetState {
                    format: factory.surface_format,
                    blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                None,
                factory.path_sample_count,
                &factory.shader_module,
            )
        })
    }

    fn paths(&self) -> &wgpu::RenderPipeline {
        self.paths.get_or_init(|| {
            let factory = &self.factory;
            let paths_blend = wgpu::BlendState {
                color: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                    operation: wgpu::BlendOperation::Add,
                },
                alpha: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::One,
                    operation: wgpu::BlendOperation::Add,
                },
            };
            factory.create_pipeline(
                "paths",
                "vs_path",
                "fs_path",
                &factory.layouts.globals,
                &factory.layouts.instances,
                Some(&factory.layouts.texture),
                wgpu::PrimitiveTopology::TriangleStrip,
                &[Some(wgpu::ColorTargetState {
                    format: factory.surface_format,
                    blend: Some(paths_blend),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                factory.depth_stencil(false),
                1,
                &factory.shader_module,
            )
        })
    }

    fn poly_sprites(&self) -> &wgpu::RenderPipeline {
        self.poly_sprites.get_or_init(|| {
            let factory = &self.factory;
            factory.create_pipeline(
                "poly_sprites",
                "vs_poly_sprite",
                "fs_poly_sprite",
                &factory.layouts.globals,
                &factory.layouts.instances,
                Some(&factory.layouts.texture),
                wgpu::PrimitiveTopology::TriangleStrip,
                &[Some(factory.color_target.clone())],
                factory.depth_stencil(false),
                1,
                &factory.shader_module,
            )
        })
    }
}

/// One frame allocation of instance data, ready to bind.
struct InstanceBinding {
    bind_group: wgpu::BindGroup,
    /// Index of the allocation's first instance within the bound data. Always
    /// zero on the storage-buffer path, where the binding offset already
    /// positions the array; on the WebGL texture path the shader indexes the
    /// shared instance texture absolutely, so draws must offset their
    /// instance (or vertex) ranges by this value.
    first_instance: u32,
}

struct InstanceBindings {
    quads: InstanceBinding,
    shadows: InstanceBinding,
    underlines: InstanceBinding,
    monochrome_sprites: InstanceBinding,
    subpixel_sprites: InstanceBinding,
    polychrome_sprites: InstanceBinding,
}

#[derive(Clone)]
struct WgpuBindGroupLayouts {
    globals: wgpu::BindGroupLayout,
    instances: wgpu::BindGroupLayout,
    quads: wgpu::BindGroupLayout,
    texture: wgpu::BindGroupLayout,
}

/// Shared GPU context reference, used to coordinate device recovery across multiple windows.
pub type GpuContext = Rc<RefCell<Option<WgpuContext>>>;

enum InstanceData {
    Storage(wgpu::Buffer),
    // WebGL2 has no storage buffers. A uint texture keeps the records available to both shader
    // stages while preserving integer and floating-point bit patterns exactly.
    Texture {
        texture: wgpu::Texture,
        view: wgpu::TextureView,
        width: u32,
        height: u32,
    },
}

/// GPU resources that must be dropped together during device recovery.
struct WgpuResources {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    pipelines: WgpuPipelines,
    bind_group_layouts: WgpuBindGroupLayouts,
    atlas_sampler: wgpu::Sampler,
    atlas_texture_bind_groups: FxHashMap<AtlasTextureId, CachedTextureBindGroup>,
    globals_buffer: wgpu::Buffer,
    globals_bind_group: wgpu::BindGroup,
    path_globals_bind_group: wgpu::BindGroup,
    instance_data: InstanceData,
    depth_texture: Option<wgpu::Texture>,
    depth_view: Option<wgpu::TextureView>,
    path_intermediate_texture: Option<wgpu::Texture>,
    path_intermediate_view: Option<wgpu::TextureView>,
    path_msaa_texture: Option<wgpu::Texture>,
    path_msaa_view: Option<wgpu::TextureView>,
    path_tile_texture: Option<wgpu::Texture>,
    path_tile_view: Option<wgpu::TextureView>,
    path_free_frames: u32,
    render_memory: RenderMemoryLedger,
}

struct CachedTextureBindGroup {
    texture_generation: u64,
    bind_group: wgpu::BindGroup,
}

impl WgpuResources {
    fn invalidate_intermediate_textures(&mut self) {
        self.depth_texture = None;
        self.depth_view = None;
        self.release_path_targets();
    }

    /// Reported wherever the targets change, not only per frame, so a release
    /// that happens without a draw (resize, hidden window) is seen at once.
    fn publish_target_memory(&mut self) {
        let depth = texture_bytes(self.depth_texture.as_ref());
        let paths = [
            &self.path_intermediate_texture,
            &self.path_msaa_texture,
            &self.path_tile_texture,
        ]
        .into_iter()
        .map(|texture| texture_bytes(texture.as_ref()))
        .sum();
        self.render_memory
            .publish(RenderMemoryGauge::DepthTextures, depth);
        self.render_memory
            .publish(RenderMemoryGauge::PathTextures, paths);
    }

    /// Dropping is enough: wgpu keeps a texture alive until the submitted work
    /// that references it has finished.
    fn release_path_targets(&mut self) {
        self.path_intermediate_texture = None;
        self.path_intermediate_view = None;
        self.path_msaa_texture = None;
        self.path_msaa_view = None;
        self.path_tile_texture = None;
        self.path_tile_view = None;
        self.publish_target_memory();
    }
}

fn texture_bytes(texture: Option<&wgpu::Texture>) -> u64 {
    texture.map_or(0, |texture| {
        let bytes_per_texel = texture.format().block_copy_size(None).unwrap_or(4);
        u64::from(texture.width())
            * u64::from(texture.height())
            * u64::from(bytes_per_texel)
            * u64::from(texture.sample_count())
    })
}

/// Paths with different draw orders are composited through one rect spanning
/// all of them, so every tile under it is cleared and redrawn in this batch and
/// no tile content from an earlier batch can show through.
fn path_sprites(paths: &[Path<ScaledPixels>]) -> Vec<PathSprite> {
    let Some(first_path) = paths.first() else {
        return Vec::new();
    };
    if paths.last().map(|path| &path.order) == Some(&first_path.order) {
        paths
            .iter()
            .map(|path| PathSprite {
                bounds: path.clipped_bounds(),
            })
            .collect()
    } else {
        let mut bounds = first_path.clipped_bounds();
        for path in paths.iter().skip(1) {
            bounds = bounds.union(&path.clipped_bounds());
        }
        vec![PathSprite { bounds }]
    }
}

fn path_tile_size(size: Size<DevicePixels>) -> Size<u32> {
    Size {
        width: (size.width.0.max(1) as u32).min(PATH_RASTER_TILE_SIZE),
        height: (size.height.0.max(1) as u32).min(PATH_RASTER_TILE_SIZE),
    }
}

/// The tiles of the target a set of path sprites samples, each clamped to the
/// target, in row order and without repeats. Sprite bounds are widened by a
/// pixel so that a fractional edge, and any texel a filtered sample could
/// touch beside it, lands in a rasterized tile.
fn path_tiles(
    sprites: &[PathSprite],
    size: Size<DevicePixels>,
    tile_size: Size<u32>,
) -> Vec<Bounds<u32>> {
    let width = size.width.0.max(0) as u32;
    let height = size.height.0.max(0) as u32;
    let mut tiles = Vec::new();
    for sprite in sprites {
        let bounds = sprite.bounds;
        let left = ((bounds.origin.x.0).floor() - 1.).clamp(0., width as f32) as u32;
        let top = ((bounds.origin.y.0).floor() - 1.).clamp(0., height as f32) as u32;
        let right = ((bounds.origin.x.0 + bounds.size.width.0).ceil() + 1.)
            .clamp(0., width as f32) as u32;
        let bottom = ((bounds.origin.y.0 + bounds.size.height.0).ceil() + 1.)
            .clamp(0., height as f32) as u32;
        if bounds.size.width.0 <= 0. || bounds.size.height.0 <= 0. || right <= left || bottom <= top
        {
            continue;
        }
        for row in top / tile_size.height..=(bottom - 1) / tile_size.height {
            for column in left / tile_size.width..=(right - 1) / tile_size.width {
                tiles.push((row, column));
            }
        }
    }
    tiles.sort_unstable();
    tiles.dedup();
    tiles
        .into_iter()
        .map(|(row, column)| {
            let x = column * tile_size.width;
            let y = row * tile_size.height;
            Bounds {
                origin: Point { x, y },
                size: Size {
                    width: tile_size.width.min(width - x),
                    height: tile_size.height.min(height - y),
                },
            }
        })
        .collect()
}

/// The instance usage of the frames since the last frame that used a quarter
/// or more of the instance buffer.
#[derive(Debug, Default)]
struct InstanceUsageWindow {
    low_frames: u32,
    peak: u64,
}

impl InstanceUsageWindow {
    /// Records a frame that used `usage` bytes of a `capacity`-byte instance
    /// buffer and returns the capacity to reallocate to, once usage has stayed
    /// under a quarter of the capacity for `INSTANCE_SHRINK_FRAMES` frames:
    /// twice the window's peak, rounded up to a power of two so the next growth
    /// step lands where it started, and never below `initial`.
    fn record_frame(&mut self, usage: u64, capacity: u64, initial: u64) -> Option<u64> {
        if usage.saturating_mul(4) >= capacity {
            *self = Self::default();
            return None;
        }
        self.low_frames += 1;
        self.peak = self.peak.max(usage);
        if self.low_frames < INSTANCE_SHRINK_FRAMES {
            return None;
        }
        let target = self.peak.saturating_mul(2).next_power_of_two().max(initial);
        *self = Self::default();
        (target < capacity).then_some(target)
    }
}

struct WgpuRendererCore {
    resources: WgpuResources,
    atlas: Arc<WgpuAtlas>,
    path_globals_offset: u64,
    gamma_offset: u64,
    instance_data_capacity: u64,
    initial_instance_data_capacity: u64,
    instance_usage: InstanceUsageWindow,
    max_instance_data_size: u64,
    instance_data_alignment: u64,
    uses_webgl_instance_data: bool,
    rendering_params: RenderingParameters,
    is_bgr: bool,
    dual_source_blending: bool,
    adapter_info: wgpu::AdapterInfo,
    target_format: wgpu::TextureFormat,
    max_texture_size: u32,
}

/// GPU resources of a windowed renderer. A surface is only ever configured against the
/// device that owns `core`, so it cannot outlive it: there is no surface-only state.
enum RendererState {
    /// Frames can be drawn.
    Ready {
        surface: wgpu::Surface<'static>,
        core: WgpuRendererCore,
    },
    /// The native surface is gone (Android `TerminateWindow`, browser context loss) but
    /// the device, pipelines, and atlas remain so `replace_surface` can resume without
    /// re-uploading cached textures.
    Unconfigured { core: WgpuRendererCore },
    /// Released by `destroy`, or dropped for a device recovery that has not succeeded yet.
    Released,
}

const RECOVERY_INITIAL_DELAY: std::time::Duration = std::time::Duration::from_millis(350);
const RECOVERY_MAX_DELAY: std::time::Duration = std::time::Duration::from_secs(5);

/// Spaces out attempts to recreate a lost device. The platforms call
/// `recover` from every frame, and an attempt can block the foreground thread
/// (the settle sleep, adapter enumeration), so a device that stays
/// unavailable would otherwise stall each frame.
#[derive(Default)]
struct RecoveryBackoff {
    failures: u32,
    next_attempt: Option<std::time::Instant>,
}

#[cfg_attr(target_family = "wasm", allow(dead_code))]
impl RecoveryBackoff {
    fn is_due(&self, now: std::time::Instant) -> bool {
        self.next_attempt.is_none_or(|next_attempt| now >= next_attempt)
    }

    fn record_failure(&mut self, now: std::time::Instant) {
        let delay = RECOVERY_INITIAL_DELAY
            .saturating_mul(1 << self.failures.min(8))
            .min(RECOVERY_MAX_DELAY);
        self.failures = self.failures.saturating_add(1);
        self.next_attempt = Some(now + delay);
    }
}

pub struct WgpuRenderer {
    /// Shared GPU context for device recovery coordination (unused on WASM).
    #[allow(dead_code)]
    context: Option<GpuContext>,
    /// Compositor GPU hint for adapter selection (unused on WASM).
    #[allow(dead_code)]
    compositor_gpu: Option<CompositorGpuHint>,
    state: RendererState,
    surface_config: wgpu::SurfaceConfiguration,
    atlas: Arc<WgpuAtlas>,
    transparent_alpha_mode: wgpu::CompositeAlphaMode,
    opaque_alpha_mode: wgpu::CompositeAlphaMode,
    max_texture_size: u32,
    is_bgr: bool,
    failed_frame_count: u32,
    device_errors: Arc<DeviceErrorState>,
    observed_error_generation: u64,
    last_surface_error: Option<String>,
    needs_redraw: bool,
    recovery_backoff: RecoveryBackoff,
}

impl WgpuRenderer {
    fn core(&self) -> Option<&WgpuRendererCore> {
        match &self.state {
            RendererState::Ready { core, .. } | RendererState::Unconfigured { core } => Some(core),
            RendererState::Released => None,
        }
    }

    fn core_mut(&mut self) -> Option<&mut WgpuRendererCore> {
        match &mut self.state {
            RendererState::Ready { core, .. } | RendererState::Unconfigured { core } => Some(core),
            RendererState::Released => None,
        }
    }

    /// Creates a new WgpuRenderer from raw window handles.
    ///
    /// The `gpu_context` is a shared reference that coordinates GPU context across
    /// multiple windows. The first window to create a renderer will initialize the
    /// context; subsequent windows will share it.
    ///
    /// # Safety
    /// The caller must ensure that the window handle remains valid for the lifetime
    /// of the returned renderer.
    #[cfg(not(target_family = "wasm"))]
    pub fn new<W>(
        gpu_context: GpuContext,
        window: &W,
        config: WgpuSurfaceConfig,
        compositor_gpu: Option<CompositorGpuHint>,
    ) -> anyhow::Result<Self>
    where
        W: HasWindowHandle + HasDisplayHandle + std::fmt::Debug + Send + Sync + Clone + 'static,
    {
        let window_handle = window
            .window_handle()
            .map_err(|e| anyhow::anyhow!("Failed to get window handle: {e}"))?;

        let target = wgpu::SurfaceTargetUnsafe::RawHandle {
            // Fall back to the display handle already provided via InstanceDescriptor::display.
            raw_display_handle: None,
            raw_window_handle: window_handle.as_raw(),
        };

        // Use the existing context's instance if available, otherwise create a new one.
        // The surface must be created with the same instance that will be used for
        // adapter selection, otherwise wgpu will panic.
        let instance = gpu_context
            .borrow()
            .as_ref()
            .map(|ctx| ctx.instance.clone())
            .unwrap_or_else(|| WgpuContext::instance(Some(Box::new(window.clone()))));

        // Safety: The caller guarantees that the window handle is valid for the
        // lifetime of this renderer. In practice, the RawWindow struct is created
        // from the native window handles and the surface is dropped before the window.
        let surface = unsafe {
            instance
                .create_surface_unsafe(target)
                .map_err(|e| anyhow::anyhow!("Failed to create surface: {e}"))?
        };

        let mut ctx_ref = gpu_context.borrow_mut();
        let context = match ctx_ref.as_mut() {
            Some(context) => {
                context.check_compatible_with_surface(&surface)?;
                context
            }
            None => ctx_ref.insert(WgpuContext::new(instance, &surface, compositor_gpu)?),
        };

        let atlas = Arc::new(WgpuAtlas::from_context(context));

        Self::new_internal(
            Some(Rc::clone(&gpu_context)),
            context,
            surface,
            config,
            compositor_gpu,
            atlas,
        )
    }

    #[cfg(target_family = "wasm")]
    pub fn new_from_canvas(
        context: &WgpuContext,
        canvas: &web_sys::HtmlCanvasElement,
        config: WgpuSurfaceConfig,
    ) -> anyhow::Result<Self> {
        let surface = context
            .instance
            .create_surface(wgpu::SurfaceTarget::Canvas(canvas.clone()))
            .map_err(|e| anyhow::anyhow!("Failed to create surface: {e}"))?;
        Self::new_from_surface(context, surface, config)
    }

    #[cfg(target_family = "wasm")]
    #[allow(clippy::arc_with_non_send_sync)]
    pub fn new_from_surface(
        context: &WgpuContext,
        surface: wgpu::Surface<'static>,
        config: WgpuSurfaceConfig,
    ) -> anyhow::Result<Self> {
        let atlas = Arc::new(WgpuAtlas::from_context(context));
        Self::new_internal(None, context, surface, config, None, atlas)
    }

    fn new_internal(
        gpu_context: Option<GpuContext>,
        context: &WgpuContext,
        surface: wgpu::Surface<'static>,
        config: WgpuSurfaceConfig,
        compositor_gpu: Option<CompositorGpuHint>,
        atlas: Arc<WgpuAtlas>,
    ) -> anyhow::Result<Self> {
        let surface_caps = surface.get_capabilities(&context.adapter);
        let preferred_formats = [
            wgpu::TextureFormat::Bgra8Unorm,
            wgpu::TextureFormat::Rgba8Unorm,
        ];
        let surface_format = preferred_formats
            .iter()
            .find(|f| surface_caps.formats.contains(f))
            .copied()
            .or_else(|| surface_caps.formats.iter().find(|f| !f.is_srgb()).copied())
            .or_else(|| surface_caps.formats.first().copied())
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "Surface reports no supported texture formats for adapter {:?}",
                    context.adapter.get_info().name
                )
            })?;

        let pick_alpha_mode =
            |preferences: &[wgpu::CompositeAlphaMode]| -> anyhow::Result<wgpu::CompositeAlphaMode> {
                preferences
                    .iter()
                    .find(|p| surface_caps.alpha_modes.contains(p))
                    .copied()
                    .or_else(|| surface_caps.alpha_modes.first().copied())
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "Surface reports no supported alpha modes for adapter {:?}",
                            context.adapter.get_info().name
                        )
                    })
            };

        let transparent_alpha_mode = pick_alpha_mode(&[
            wgpu::CompositeAlphaMode::PreMultiplied,
            wgpu::CompositeAlphaMode::Inherit,
        ])?;

        let opaque_alpha_mode = pick_alpha_mode(&[
            wgpu::CompositeAlphaMode::Opaque,
            wgpu::CompositeAlphaMode::Inherit,
        ])?;

        let alpha_mode = if config.transparent {
            transparent_alpha_mode
        } else {
            opaque_alpha_mode
        };

        let device = Arc::clone(&context.device);
        let max_texture_size = device.limits().max_texture_dimension_2d;

        let requested_width = config.size.width.0 as u32;
        let requested_height = config.size.height.0 as u32;
        let clamped_width = requested_width.min(max_texture_size);
        let clamped_height = requested_height.min(max_texture_size);

        if clamped_width != requested_width || clamped_height != requested_height {
            warn!(
                "Requested surface size ({}, {}) exceeds maximum texture dimension {}. \
                 Clamping to ({}, {}). Window content may not fill the entire window.",
                requested_width, requested_height, max_texture_size, clamped_width, clamped_height
            );
        }

        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: clamped_width.max(1),
            height: clamped_height.max(1),
            present_mode: config
                .preferred_present_mode
                .filter(|mode| surface_caps.present_modes.contains(mode))
                .unwrap_or(wgpu::PresentMode::Fifo),
            desired_maximum_frame_latency: 2,
            alpha_mode,
            view_formats: vec![],
        };
        // Configure the surface immediately. The adapter selection process already validated
        // that this adapter can successfully configure this surface.
        surface.configure(&context.device, &surface_config);

        let core = WgpuRendererCore::new(context, atlas.clone(), surface_format, alpha_mode);

        Ok(Self {
            context: gpu_context,
            compositor_gpu,
            state: RendererState::Ready { surface, core },
            surface_config,
            atlas,
            transparent_alpha_mode,
            opaque_alpha_mode,
            max_texture_size,
            is_bgr: false,
            failed_frame_count: 0,
            device_errors: Arc::clone(context.errors()),
            observed_error_generation: 0,
            last_surface_error: None,
            needs_redraw: false,
            recovery_backoff: RecoveryBackoff::default(),
        })
    }
}

impl WgpuRendererCore {
    fn create_bind_group_layouts(
        device: &wgpu::Device,
        uses_webgl_instance_data: bool,
    ) -> WgpuBindGroupLayouts {
        let globals =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("globals_layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: NonZeroU64::new(
                                std::mem::size_of::<GlobalParams>() as u64
                            ),
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: NonZeroU64::new(
                                std::mem::size_of::<GammaParams>() as u64
                            ),
                        },
                        count: None,
                    },
                ],
            });

        let instance_data_entry = wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            ty: if uses_webgl_instance_data {
                wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Uint,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                }
            } else {
                wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                }
            },
            count: None,
        };

        let instances = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("instances_layout"),
            entries: &[instance_data_entry],
        });

        // The quad pipelines read the partition index table through a second
        // storage binding. The WebGL texture transport has no storage buffers
        // and no index table (partitioning is disabled there), so its quad
        // pipelines bind plain instance data.
        let storage_buffer_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let webgl_quad_entries = [instance_data_entry];
        let storage_quad_entries = [storage_buffer_entry(0), storage_buffer_entry(1)];
        let quads = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("quads_layout"),
            entries: if uses_webgl_instance_data {
                &webgl_quad_entries
            } else {
                &storage_quad_entries
            },
        });

        let texture = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("texture_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        WgpuBindGroupLayouts {
            globals,
            instances,
            quads,
            texture,
        }
    }

    fn create_instance_texture(
        device: &wgpu::Device,
        requested_capacity: u64,
        max_texture_dimension: u32,
    ) -> (InstanceData, u64) {
        let texel_count = requested_capacity.div_ceil(INSTANCE_TEXTURE_TEXEL_SIZE);
        let width = texel_count.min(u64::from(max_texture_dimension)).max(1) as u32;
        let height = texel_count
            .div_ceil(u64::from(width))
            .min(u64::from(max_texture_dimension))
            .max(1) as u32;
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("instance_texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Uint,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let capacity = u64::from(width) * u64::from(height) * INSTANCE_TEXTURE_TEXEL_SIZE;
        (
            InstanceData::Texture {
                texture,
                view,
                width,
                height,
            },
            capacity,
        )
    }

    fn create_pipelines(
        device: &wgpu::Device,
        layouts: &WgpuBindGroupLayouts,
        surface_format: wgpu::TextureFormat,
        alpha_mode: wgpu::CompositeAlphaMode,
        path_sample_count: u32,
        depth_format: wgpu::TextureFormat,
        dual_source_blending: bool,
        uses_webgl_instance_data: bool,
    ) -> WgpuPipelines {
        // Diagnostic guard: verify the device actually has
        // DUAL_SOURCE_BLENDING. We have a crash report (ZED-5G1) where a
        // feature mismatch caused a wgpu-hal abort, but we haven't
        // identified the code path that produces the mismatch. This
        // guard prevents the crash and logs more evidence.
        // Remove this check once:
        // a) We find and fix the root cause, or
        // b) There are no reports of this warning appearing for some time.
        let device_has_feature = device
            .features()
            .contains(wgpu::Features::DUAL_SOURCE_BLENDING);
        if dual_source_blending && !device_has_feature {
            log::error!(
                "BUG: dual_source_blending flag is true but device does not \
                 have DUAL_SOURCE_BLENDING enabled (device features: {:?}). \
                 Falling back to mono text rendering. Please report this at \
                 https://github.com/zed-industries/zed/issues",
                device.features(),
            );
        }
        let dual_source_blending =
            dual_source_blending && device_has_feature && !uses_webgl_instance_data;

        let shader_source = if uses_webgl_instance_data {
            WEBGL_SHADERS
        } else {
            STORAGE_BUFFER_SHADERS
        };
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gpui_shaders"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        let subpixel_shader_module = if dual_source_blending {
            Some(device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("gpui_subpixel_shaders"),
                source: wgpu::ShaderSource::Wgsl(SUBPIXEL_SHADERS.into()),
            }))
        } else {
            None
        };

        let blend_mode = match alpha_mode {
            wgpu::CompositeAlphaMode::PreMultiplied => {
                wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING
            }
            _ => wgpu::BlendState::ALPHA_BLENDING,
        };

        let factory = PipelineFactory {
            device: device.clone(),
            layouts: layouts.clone(),
            shader_module,
            color_target: wgpu::ColorTargetState {
                format: surface_format,
                blend: Some(blend_mode),
                write_mask: wgpu::ColorWrites::ALL,
            },
            surface_format,
            depth_format,
            path_sample_count,
        };

        let quads = factory.create_pipeline(
            "quads",
            "vs_quad",
            "fs_quad",
            &layouts.globals,
            &layouts.quads,
            None,
            wgpu::PrimitiveTopology::TriangleStrip,
            &[Some(factory.color_target.clone())],
            factory.depth_stencil(false),
            1,
            &factory.shader_module,
        );

        let opaque_quads = factory.create_pipeline(
            "opaque_quads",
            "vs_opaque_quad",
            "fs_opaque_quad",
            &layouts.globals,
            &layouts.quads,
            None,
            wgpu::PrimitiveTopology::TriangleStrip,
            &[Some(wgpu::ColorTargetState {
                format: surface_format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
            factory.depth_stencil(true),
            1,
            &factory.shader_module,
        );

        let underlines = factory.create_pipeline(
            "underlines",
            "vs_underline",
            "fs_underline",
            &layouts.globals,
            &layouts.instances,
            None,
            wgpu::PrimitiveTopology::TriangleStrip,
            &[Some(factory.color_target.clone())],
            factory.depth_stencil(false),
            1,
            &factory.shader_module,
        );

        let mono_sprites = factory.create_pipeline(
            "mono_sprites",
            "vs_mono_sprite",
            "fs_mono_sprite",
            &layouts.globals,
            &layouts.instances,
            Some(&layouts.texture),
            wgpu::PrimitiveTopology::TriangleStrip,
            &[Some(factory.color_target.clone())],
            factory.depth_stencil(false),
            1,
            &factory.shader_module,
        );

        let subpixel_sprites = if let Some(subpixel_module) = &subpixel_shader_module {
            let subpixel_blend = wgpu::BlendState {
                color: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::Src1,
                    dst_factor: wgpu::BlendFactor::OneMinusSrc1,
                    operation: wgpu::BlendOperation::Add,
                },
                alpha: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                    operation: wgpu::BlendOperation::Add,
                },
            };

            Some(factory.create_pipeline(
                "subpixel_sprites",
                "vs_subpixel_sprite",
                "fs_subpixel_sprite",
                &layouts.globals,
                &layouts.instances,
                Some(&layouts.texture),
                wgpu::PrimitiveTopology::TriangleStrip,
                &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(subpixel_blend),
                    write_mask: wgpu::ColorWrites::COLOR,
                })],
                factory.depth_stencil(false),
                1,
                subpixel_module,
            ))
        } else {
            None
        };

        WgpuPipelines {
            quads,
            opaque_quads,
            underlines,
            mono_sprites,
            subpixel_sprites,
            shadows: OnceLock::new(),
            path_rasterization: OnceLock::new(),
            paths: OnceLock::new(),
            poly_sprites: OnceLock::new(),
            factory,
        }
    }

    fn create_depth_texture(
        device: &wgpu::Device,
        depth_format: wgpu::TextureFormat,
        width: u32,
        height: u32,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("depth"),
            size: wgpu::Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: depth_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        (texture, view)
    }

    fn create_path_intermediate(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        width: u32,
        height: u32,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("path_intermediate"),
            size: wgpu::Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        (texture, view)
    }

    /// The single-sample tile a path tile is rasterized into (or resolved
    /// into, with MSAA) before being copied to its place in the intermediate.
    fn create_path_tile(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        tile_size: Size<u32>,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("path_tile"),
            size: wgpu::Extent3d {
                width: tile_size.width,
                height: tile_size.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        (texture, view)
    }

    fn create_msaa_if_needed(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        width: u32,
        height: u32,
        sample_count: u32,
    ) -> Option<(wgpu::Texture, wgpu::TextureView)> {
        if sample_count <= 1 {
            return None;
        }
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("path_msaa"),
            size: wgpu::Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        Some((texture, view))
    }
}

impl WgpuRenderer {
    /// Records the new drawable size and reconfigures the surface when one is present.
    /// The size is kept even without GPU resources so that recovery restores it.
    pub fn update_drawable_size(&mut self, size: Size<DevicePixels>) {
        let width = size.width.0 as u32;
        let height = size.height.0 as u32;

        if width == self.surface_config.width && height == self.surface_config.height {
            return;
        }

        let clamped_width = width.min(self.max_texture_size);
        let clamped_height = height.min(self.max_texture_size);
        if clamped_width != width || clamped_height != height {
            warn!(
                "Requested surface size ({}, {}) exceeds maximum texture dimension {}. \
                 Clamping to ({}, {}). Window content may not fill the entire window.",
                width, height, self.max_texture_size, clamped_width, clamped_height
            );
        }
        self.surface_config.width = clamped_width.max(1);
        self.surface_config.height = clamped_height.max(1);

        let Some(core) = self.core_mut() else {
            return;
        };
        let resources = &mut core.resources;

        // Wait for any in-flight GPU work to complete before destroying textures
        if let Err(e) = resources.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        }) {
            warn!("Failed to poll device during resize: {e:?}");
        }

        // Destroy old textures before allocating new ones to avoid GPU memory spikes
        if let Some(ref texture) = resources.depth_texture {
            texture.destroy();
        }
        if let Some(ref texture) = resources.path_intermediate_texture {
            texture.destroy();
        }
        if let Some(ref texture) = resources.path_msaa_texture {
            texture.destroy();
        }
        if let Some(ref texture) = resources.path_tile_texture {
            texture.destroy();
        }

        // Invalidate intermediate textures - they will be lazily recreated
        // in draw() after we confirm the surface is healthy. This avoids
        // panics when the device/surface is in an invalid state during resize.
        resources.invalidate_intermediate_textures();

        if let RendererState::Ready { surface, core } = &self.state {
            surface.configure(&core.resources.device, &self.surface_config);
        }
    }

    pub fn set_subpixel_layout(&mut self, is_bgr: bool) {
        self.is_bgr = is_bgr;
        if let Some(core) = self.core_mut() {
            core.is_bgr = is_bgr;
        }
    }

    pub fn update_transparency(&mut self, transparent: bool) {
        let new_alpha_mode = if transparent {
            self.transparent_alpha_mode
        } else {
            self.opaque_alpha_mode
        };
        if new_alpha_mode == self.surface_config.alpha_mode {
            return;
        }
        self.surface_config.alpha_mode = new_alpha_mode;
        let format = self.surface_config.format;

        let Some(core) = self.core_mut() else {
            return;
        };
        let resources = &mut core.resources;
        resources.pipelines = WgpuRendererCore::create_pipelines(
            &resources.device,
            &resources.bind_group_layouts,
            format,
            new_alpha_mode,
            core.rendering_params.path_sample_count,
            core.rendering_params.depth_format,
            core.dual_source_blending,
            core.uses_webgl_instance_data,
        );

        if let RendererState::Ready { surface, core } = &self.state {
            surface.configure(&core.resources.device, &self.surface_config);
        }
    }

    #[allow(dead_code)]
    pub fn viewport_size(&self) -> Size<DevicePixels> {
        Size {
            width: DevicePixels(self.surface_config.width as i32),
            height: DevicePixels(self.surface_config.height as i32),
        }
    }

    pub fn sprite_atlas(&self) -> &Arc<WgpuAtlas> {
        &self.atlas
    }

    pub fn supports_dual_source_blending(&self) -> bool {
        self.core().is_some_and(|core| core.dual_source_blending)
    }

    /// Returns `None` once GPU resources have been released by `destroy` or a pending
    /// device recovery.
    pub fn gpu_specs(&self) -> Option<GpuSpecs> {
        let adapter_info = &self.core()?.adapter_info;
        Some(GpuSpecs {
            is_software_emulated: adapter_info.device_type == wgpu::DeviceType::Cpu,
            device_name: adapter_info.name.clone(),
            driver_name: adapter_info.driver.clone(),
            driver_info: adapter_info.driver_info.clone(),
        })
    }

    pub fn max_texture_size(&self) -> u32 {
        self.max_texture_size
    }

    pub fn draw(&mut self, scene: &Scene) -> bool {
        #[cfg(target_family = "wasm")]
        if self.device_lost() {
            if matches!(self.state, RendererState::Ready { .. }) {
                log::error!(
                    "Browser graphics context was lost; rendering has stopped. Reload the page to recover."
                );
                self.unconfigure_surface();
            }
            return false;
        }

        // Bail out early if the surface has been unconfigured (e.g. during
        // Android background/rotation transitions).  Attempting to acquire
        // a texture from an unconfigured surface can block indefinitely on
        // some drivers (Adreno).
        let RendererState::Ready { surface, core } = &mut self.state else {
            return false;
        };

        if let Some(error) = self.last_surface_error.take().or_else(|| {
            self.device_errors
                .observe_error(&mut self.observed_error_generation)
        }) {
            self.failed_frame_count += 1;
            log::error!(
                "GPU error during frame (failure {} of 10): {error}",
                self.failed_frame_count
            );

            // TBD. Does retrying more actually help?
            if self.failed_frame_count > 10 {
                panic!("Too many consecutive GPU errors. Last error: {error}");
            } else if self.failed_frame_count > 5 {
                core.resources.invalidate_intermediate_textures();
                self.atlas.clear();
                self.needs_redraw = true;
                self.failed_frame_count = 0;
                return false;
            }
        } else {
            self.failed_frame_count = 0;
        }

        let frame = match surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(frame) => frame,
            wgpu::CurrentSurfaceTexture::Suboptimal(frame) => {
                // Textures must be destroyed before the surface can be reconfigured.
                drop(frame);
                surface.configure(&core.resources.device, &self.surface_config);
                return false;
            }
            wgpu::CurrentSurfaceTexture::Lost | wgpu::CurrentSurfaceTexture::Outdated => {
                surface.configure(&core.resources.device, &self.surface_config);
                return false;
            }
            wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded => {
                return false;
            }
            wgpu::CurrentSurfaceTexture::Validation => {
                self.last_surface_error = Some("Surface texture validation error".to_string());
                return false;
            }
        };

        // The acquired texture is the authority on frame dimensions; the surface
        // configuration is only a request.
        let size = Size {
            width: DevicePixels(frame.texture.width() as i32),
            height: DevicePixels(frame.texture.height() as i32),
        };
        let frame_view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let premultiplied_alpha =
            self.surface_config.alpha_mode == wgpu::CompositeAlphaMode::PreMultiplied;
        if let Err(error) = core.render_frame(
            scene,
            &frame_view,
            size,
            premultiplied_alpha,
            wgpu::Color::TRANSPARENT,
        ) {
            log::error!("{error:#}");
            return false;
        }

        frame.present();
        true
    }
}

impl WgpuRendererCore {
    fn new(
        context: &WgpuContext,
        atlas: Arc<WgpuAtlas>,
        target_format: wgpu::TextureFormat,
        alpha_mode: wgpu::CompositeAlphaMode,
    ) -> Self {
        let device = Arc::clone(&context.device);
        let queue = Arc::clone(&context.queue);
        let rendering_params = RenderingParameters::new(&context.adapter, target_format);
        let uses_webgl_instance_data = context.uses_webgl_instance_data();
        if uses_webgl_instance_data {
            // The texture transport cannot express the quad index indirection
            // the opaque depth prepass draws through, so scenes for this
            // process fall back to painter's-order quads (empty opaque bucket,
            // every quad on the blended pass), which renders identically.
            gpui::disable_opaque_quad_partitioning();
        }
        let dual_source_blending =
            context.supports_dual_source_blending() && !uses_webgl_instance_data;
        let bind_group_layouts = Self::create_bind_group_layouts(&device, uses_webgl_instance_data);
        let pipelines = Self::create_pipelines(
            &device,
            &bind_group_layouts,
            target_format,
            alpha_mode,
            rendering_params.path_sample_count,
            rendering_params.depth_format,
            dual_source_blending,
            uses_webgl_instance_data,
        );
        let atlas_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("atlas_sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let uniform_alignment = device.limits().min_uniform_buffer_offset_alignment as u64;
        let globals_size = std::mem::size_of::<GlobalParams>() as u64;
        let gamma_size = std::mem::size_of::<GammaParams>() as u64;
        let path_globals_offset = globals_size.next_multiple_of(uniform_alignment);
        let gamma_offset = (path_globals_offset + globals_size).next_multiple_of(uniform_alignment);
        let globals_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("globals_buffer"),
            size: gamma_offset + gamma_size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let (
            instance_data,
            instance_data_capacity,
            max_instance_data_size,
            instance_data_alignment,
        ) = if uses_webgl_instance_data {
            let max_texture_dimension = device.limits().max_texture_dimension_2d;
            let max_instance_data_size = (u64::from(max_texture_dimension).pow(2)
                * INSTANCE_TEXTURE_TEXEL_SIZE)
                .min(MAX_INSTANCE_BUFFER_SIZE);
            let initial_capacity = (2 * 1024 * 1024).min(max_instance_data_size);
            let (instance_data, capacity) =
                Self::create_instance_texture(&device, initial_capacity, max_texture_dimension);
            (
                instance_data,
                capacity,
                max_instance_data_size,
                INSTANCE_TEXTURE_TEXEL_SIZE,
            )
        } else {
            let max_buffer_size = device
                .limits()
                .max_buffer_size
                .min(device.limits().max_storage_buffer_binding_size)
                .min(MAX_INSTANCE_BUFFER_SIZE);
            let initial_capacity = (2 * 1024 * 1024).min(max_buffer_size);
            let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("instance_buffer"),
                size: initial_capacity,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            (
                InstanceData::Storage(buffer),
                initial_capacity,
                max_buffer_size,
                device.limits().min_storage_buffer_offset_alignment as u64,
            )
        };
        let globals_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("globals_bind_group"),
            layout: &bind_group_layouts.globals,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &globals_buffer,
                        offset: 0,
                        size: NonZeroU64::new(globals_size),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &globals_buffer,
                        offset: gamma_offset,
                        size: NonZeroU64::new(gamma_size),
                    }),
                },
            ],
        });
        let path_globals_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("path_globals_bind_group"),
            layout: &bind_group_layouts.globals,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &globals_buffer,
                        offset: path_globals_offset,
                        size: NonZeroU64::new(globals_size),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &globals_buffer,
                        offset: gamma_offset,
                        size: NonZeroU64::new(gamma_size),
                    }),
                },
            ],
        });
        let max_texture_size = device.limits().max_texture_dimension_2d;
        let mut render_memory = RenderMemoryLedger::default();
        render_memory.publish(RenderMemoryGauge::InstanceBuffers, instance_data_capacity);

        Self {
            resources: WgpuResources {
                device,
                queue,
                pipelines,
                bind_group_layouts,
                atlas_sampler,
                atlas_texture_bind_groups: FxHashMap::default(),
                globals_buffer,
                globals_bind_group,
                path_globals_bind_group,
                instance_data,
                depth_texture: None,
                depth_view: None,
                path_intermediate_texture: None,
                path_intermediate_view: None,
                path_msaa_texture: None,
                path_msaa_view: None,
                path_tile_texture: None,
                path_tile_view: None,
                path_free_frames: 0,
                render_memory,
            },
            atlas,
            path_globals_offset,
            gamma_offset,
            instance_data_capacity,
            initial_instance_data_capacity: instance_data_capacity,
            instance_usage: InstanceUsageWindow::default(),
            max_instance_data_size,
            instance_data_alignment,
            uses_webgl_instance_data,
            rendering_params,
            is_bgr: false,
            dual_source_blending,
            adapter_info: context.adapter.get_info(),
            target_format,
            max_texture_size,
        }
    }

    fn resources(&self) -> &WgpuResources {
        &self.resources
    }

    fn resources_mut(&mut self) -> &mut WgpuResources {
        &mut self.resources
    }

    /// The depth buffer and the path targets must match the frame target's
    /// dimensions, so they are keyed by size and rebuilt on mismatch. The path
    /// intermediate and its MSAA companion are only read by path batches, which
    /// come from `scene.paths`, so they are created only for a frame that has
    /// paths (before recording, so a path batch always finds them) and released
    /// after `PATH_TARGET_IDLE_FRAMES` path-free frames or a path-free resize.
    fn ensure_intermediate_textures(&mut self, size: Size<DevicePixels>, scene_has_paths: bool) {
        let width = size.width.0 as u32;
        let height = size.height.0 as u32;
        let matches_size =
            |texture: &wgpu::Texture| texture.width() == width && texture.height() == height;
        let format = self.target_format;
        let path_sample_count = self.rendering_params.path_sample_count;
        let depth_format = self.rendering_params.depth_format;
        let resources = &mut self.resources;

        if !resources.depth_texture.as_ref().is_some_and(matches_size) {
            let (depth_texture, depth_view) =
                Self::create_depth_texture(&resources.device, depth_format, width, height);
            resources.depth_texture = Some(depth_texture);
            resources.depth_view = Some(depth_view);
        }

        let path_targets_match = resources
            .path_intermediate_texture
            .as_ref()
            .is_some_and(matches_size);
        if !scene_has_paths {
            resources.path_free_frames = resources.path_free_frames.saturating_add(1);
            if !path_targets_match || resources.path_free_frames >= PATH_TARGET_IDLE_FRAMES {
                resources.release_path_targets();
            }
            return;
        }

        resources.path_free_frames = 0;
        if path_targets_match {
            return;
        }

        // Release the stale pair first so the old and new targets are never
        // resident together.
        resources.release_path_targets();
        let (texture, view) =
            Self::create_path_intermediate(&resources.device, format, width, height);
        resources.path_intermediate_texture = Some(texture);
        resources.path_intermediate_view = Some(view);

        let tile_size = path_tile_size(size);
        let (tile_texture, tile_view) =
            Self::create_path_tile(&resources.device, format, tile_size);
        resources.path_tile_texture = Some(tile_texture);
        resources.path_tile_view = Some(tile_view);

        let (path_msaa_texture, path_msaa_view) = Self::create_msaa_if_needed(
            &resources.device,
            format,
            tile_size.width,
            tile_size.height,
            path_sample_count,
        )
        .map(|(texture, view)| (Some(texture), Some(view)))
        .unwrap_or((None, None));
        resources.path_msaa_texture = path_msaa_texture;
        resources.path_msaa_view = path_msaa_view;
    }

    fn render_frame(
        &mut self,
        scene: &Scene,
        target_view: &wgpu::TextureView,
        size: Size<DevicePixels>,
        premultiplied_alpha: bool,
        clear_color: wgpu::Color,
    ) -> Result<wgpu::SubmissionIndex> {
        anyhow::ensure!(
            size.width.0 > 0 && size.height.0 > 0,
            "invalid render target size: {size:?}"
        );
        anyhow::ensure!(
            size.width.0 as u32 <= self.max_texture_size
                && size.height.0 as u32 <= self.max_texture_size,
            "render target size {size:?} exceeds maximum texture dimension {}",
            self.max_texture_size
        );

        self.atlas.before_frame();
        self.ensure_intermediate_textures(size, !scene.paths.is_empty());
        self.resources.publish_target_memory();

        let gamma_params = GammaParams {
            gamma_ratios: self.rendering_params.gamma_ratios,
            grayscale_enhanced_contrast: self.rendering_params.grayscale_enhanced_contrast,
            subpixel_enhanced_contrast: self.rendering_params.subpixel_enhanced_contrast,
            is_bgr: self.is_bgr as u32,
            _pad: 0,
        };
        let globals = GlobalParams {
            viewport_size: [size.width.0 as f32, size.height.0 as f32],
            premultiplied_alpha: premultiplied_alpha as u32,
            pad: 0,
        };
        let path_globals = GlobalParams {
            premultiplied_alpha: 0,
            ..globals
        };
        self.resources.queue.write_buffer(
            &self.resources.globals_buffer,
            0,
            bytemuck::bytes_of(&globals),
        );
        self.resources.queue.write_buffer(
            &self.resources.globals_buffer,
            self.path_globals_offset,
            bytemuck::bytes_of(&path_globals),
        );
        self.resources.queue.write_buffer(
            &self.resources.globals_buffer,
            self.gamma_offset,
            bytemuck::bytes_of(&gamma_params),
        );

        let depth_view = self
            .resources
            .depth_view
            .clone()
            .context("depth buffer missing after ensuring intermediate textures")?;

        let submission = self
            .record_frame(scene, target_view, &depth_view, size, clear_color)
            .inspect_err(|_| {
                // Queue writes are staged before encoding; flush them even if the frame fails.
                self.resources.queue.submit(std::iter::empty());
            })?;
        // Only a frame that was actually submitted counts its sprite tiles as
        // used and ages the atlas: `Window` reads the atlas frame to tell a
        // present that drew from one that bailed. Retiring after the submit is
        // safe for the GPU: wgpu keeps a dropped page alive until the work
        // that samples it has finished, and a re-used tile's upload is queued
        // ahead of the next submit, after this one.
        self.atlas.on_frame_drawn(scene);
        Ok(submission)
    }

    fn record_frame(
        &mut self,
        scene: &Scene,
        frame_view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        size: Size<DevicePixels>,
        clear_color: wgpu::Color,
    ) -> Result<wgpu::SubmissionIndex> {
        let mut instance_offset: u64 = 0;
        let starting_capacity = self.instance_data_capacity;
        let instance_bindings = self
            .write_instances(scene, &mut instance_offset)
            .with_context(|| {
                format!(
                    "scene too large: {} paths, {} shadows, {} quads, {} underlines, {} monochrome sprites, {} subpixel sprites, {} polychrome sprites",
                    scene.paths.len(),
                    scene.shadows.len(),
                    scene.quads.len(),
                    scene.underlines.len(),
                    scene.monochrome_sprites.len(),
                    scene.subpixel_sprites.len(),
                    scene.polychrome_sprites.len(),
                )
            })?;
        self.prepare_texture_bind_groups(scene);

        let mut encoder =
            self.resources()
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("main_encoder"),
                });

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("main_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: frame_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(clear_color),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(0.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });

            self.draw_instances(
                &instance_bindings.quads,
                |pipelines| &pipelines.opaque_quads,
                scene.blended_quad_indices.len() as u32
                    ..(scene.blended_quad_indices.len() + scene.opaque_quad_indices.len()) as u32,
                &mut pass,
            );

            let mut quad_cursor: u32 = 0;
            for batch in scene.batches() {
                // Quad shaders assign each quad its own depth. Every other batch
                // is flattened onto the depth of the quad cursor by collapsing
                // the viewport's depth range onto a single value.
                let batch_depth = quad_depth(quad_cursor);
                if matches!(batch, PrimitiveBatch::Quads { .. }) {
                    Self::set_pass_depth_range(&mut pass, size, 0.0, 1.0);
                } else {
                    Self::set_pass_depth_range(&mut pass, size, batch_depth, batch_depth);
                }

                match batch {
                    PrimitiveBatch::Quads {
                        range,
                        blended_range,
                    } => {
                        quad_cursor += range.len() as u32;
                        self.draw_instances(
                            &instance_bindings.quads,
                            |pipelines| &pipelines.quads,
                            instance_range(blended_range),
                            &mut pass,
                        );
                    }
                    PrimitiveBatch::Shadows(range) => self.draw_instances(
                        &instance_bindings.shadows,
                        WgpuPipelines::shadows,
                        instance_range(range),
                        &mut pass,
                    ),
                    PrimitiveBatch::Paths(range) => {
                        let paths = &scene.paths[range];
                        if paths.is_empty() {
                            continue;
                        }

                        drop(pass);
                        let sprites = path_sprites(paths);
                        let rasterized = self.draw_paths_to_intermediate(
                            &mut encoder,
                            paths,
                            &sprites,
                            size,
                            &mut instance_offset,
                        )?;

                        pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: Some("main_pass_continued"),
                            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                view: frame_view,
                                resolve_target: None,
                                ops: wgpu::Operations {
                                    load: wgpu::LoadOp::Load,
                                    store: wgpu::StoreOp::Store,
                                },
                                depth_slice: None,
                            })],
                            depth_stencil_attachment: Some(
                                wgpu::RenderPassDepthStencilAttachment {
                                    view: depth_view,
                                    depth_ops: Some(wgpu::Operations {
                                        load: wgpu::LoadOp::Load,
                                        store: wgpu::StoreOp::Store,
                                    }),
                                    stencil_ops: None,
                                },
                            ),
                            ..Default::default()
                        });
                        Self::set_pass_depth_range(&mut pass, size, batch_depth, batch_depth);

                        if rasterized {
                            self.draw_paths_from_intermediate(
                                &sprites,
                                &mut instance_offset,
                                &mut pass,
                            )?;
                        }
                    }
                    PrimitiveBatch::Underlines(range) => self.draw_instances(
                        &instance_bindings.underlines,
                        |pipelines| &pipelines.underlines,
                        instance_range(range),
                        &mut pass,
                    ),
                    PrimitiveBatch::MonochromeSprites { texture_id, range } => {
                        self.draw_sprites(
                            &instance_bindings.monochrome_sprites,
                            texture_id,
                            |pipelines| &pipelines.mono_sprites,
                            instance_range(range),
                            &mut pass,
                        )?;
                    }
                    PrimitiveBatch::SubpixelSprites { texture_id, range } => {
                        self.draw_sprites(
                            &instance_bindings.subpixel_sprites,
                            texture_id,
                            |pipelines| {
                                pipelines
                                    .subpixel_sprites
                                    .as_ref()
                                    .unwrap_or(&pipelines.mono_sprites)
                            },
                            instance_range(range),
                            &mut pass,
                        )?;
                    }
                    PrimitiveBatch::PolychromeSprites { texture_id, range } => {
                        self.draw_sprites(
                            &instance_bindings.polychrome_sprites,
                            texture_id,
                            WgpuPipelines::poly_sprites,
                            instance_range(range),
                            &mut pass,
                        )?;
                    }
                    // Surfaces are macOS-only for video playback and are not
                    // implemented by the WGPU renderer.
                    PrimitiveBatch::Surfaces(_surfaces) => {}
                }
            }
        }

        let submission = self
            .resources()
            .queue
            .submit(std::iter::once(encoder.finish()));

        // A frame that grew the buffer restarted its offsets in the new one,
        // so its final offset understates its usage; it counts as a full frame.
        let usage = if self.instance_data_capacity == starting_capacity {
            instance_offset
        } else {
            self.instance_data_capacity
        };
        if let Some(capacity) = self.instance_usage.record_frame(
            usage,
            self.instance_data_capacity,
            self.initial_instance_data_capacity,
        ) {
            log::debug!(
                "instance data shrunk from {} to {capacity}",
                self.instance_data_capacity
            );
            // The submitted frame's bind groups keep the old allocation alive
            // until the GPU has finished with it.
            self.reallocate_instance_data(capacity);
        }
        Ok(submission)
    }

    fn write_instances(
        &mut self,
        scene: &Scene,
        instance_offset: &mut u64,
    ) -> Result<InstanceBindings> {
        Ok(InstanceBindings {
            quads: self.write_quad_binding(instance_offset, scene)?,
            shadows: self.write_instance_binding(
                "shadows_bind_group",
                instance_offset,
                &scene.shadows,
            )?,
            underlines: self.write_instance_binding(
                "underlines_bind_group",
                instance_offset,
                &scene.underlines,
            )?,
            monochrome_sprites: self.write_instance_binding(
                "monochrome_sprites_bind_group",
                instance_offset,
                &scene.monochrome_sprites,
            )?,
            subpixel_sprites: self.write_instance_binding(
                "subpixel_sprites_bind_group",
                instance_offset,
                &scene.subpixel_sprites,
            )?,
            polychrome_sprites: self.write_instance_binding(
                "polychrome_sprites_bind_group",
                instance_offset,
                &scene.polychrome_sprites,
            )?,
        })
    }

    /// Uploads the frame's quads plus the partition index table the quad
    /// shaders resolve `instance_id` through, bound together as the quad
    /// pipelines' two-binding instance group.
    ///
    /// The WebGL texture transport cannot express the index indirection, so
    /// there the quads travel like every other primitive and `quad_index` in
    /// `shaders_webgl.wgsl` is the identity: partitioning is disabled under
    /// that transport (see [`WgpuRendererCore::new`]), which makes the blended
    /// walk cover every quad in paint order.
    fn write_quad_binding(
        &mut self,
        instance_offset: &mut u64,
        scene: &Scene,
    ) -> Result<InstanceBinding> {
        if self.uses_webgl_instance_data {
            return self.write_instance_binding("quads_bind_group", instance_offset, &scene.quads);
        }

        let quad_bytes = unsafe { Self::instance_bytes(&scene.quads) };
        let blended_bytes: &[u8] = bytemuck::cast_slice(&scene.blended_quad_indices);
        let opaque_bytes: &[u8] = bytemuck::cast_slice(&scene.opaque_quad_indices);

        // Both bindings live in one allocation so a mid-upload buffer growth
        // cannot strand the quad data in a discarded buffer. wgpu rejects
        // zero-sized bindings, so empty arrays still reserve the 16-byte
        // minimum, and the index binding's inner offset must satisfy the
        // storage binding alignment.
        let alignment = self.instance_data_alignment.max(1);
        let quads_size = (quad_bytes.len() as u64).max(16);
        let indices_size = ((blended_bytes.len() + opaque_bytes.len()) as u64).max(16);
        let indices_inner_offset = quads_size.next_multiple_of(alignment);
        let total_size = indices_inner_offset + indices_size;

        let mut offset = (*instance_offset).next_multiple_of(alignment);
        if offset + total_size > self.instance_data_capacity {
            self.grow_instance_data(total_size)?;
            offset = 0;
        }
        *instance_offset = offset + total_size;

        let resources = self.resources();
        let InstanceData::Storage(buffer) = &resources.instance_data else {
            anyhow::bail!("quad index upload requested on the WebGL texture transport");
        };
        if !quad_bytes.is_empty() {
            resources.queue.write_buffer(buffer, offset, quad_bytes);
        }
        let mut index_offset = offset + indices_inner_offset;
        for part in [blended_bytes, opaque_bytes] {
            if !part.is_empty() {
                resources.queue.write_buffer(buffer, index_offset, part);
                index_offset += part.len() as u64;
            }
        }

        let bind_group = resources
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("quads_bind_group"),
                layout: &resources.bind_group_layouts.quads,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer,
                            offset,
                            size: NonZeroU64::new(quads_size),
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer,
                            offset: offset + indices_inner_offset,
                            size: NonZeroU64::new(indices_size),
                        }),
                    },
                ],
            });
        Ok(InstanceBinding {
            bind_group,
            first_instance: 0,
        })
    }

    fn create_texture_bind_group(
        &self,
        label: &str,
        texture_view: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        let resources = self.resources();
        resources
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &resources.bind_group_layouts.texture,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(texture_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&resources.atlas_sampler),
                    },
                ],
            })
    }

    fn prepare_texture_bind_groups(&mut self, scene: &Scene) {
        let mut texture_ids = SmallVec::<[AtlasTextureId; 8]>::new();
        for batch in scene.batches() {
            let texture_id = match batch {
                PrimitiveBatch::MonochromeSprites { texture_id, .. }
                | PrimitiveBatch::SubpixelSprites { texture_id, .. }
                | PrimitiveBatch::PolychromeSprites { texture_id, .. } => texture_id,
                _ => continue,
            };
            if !texture_ids.contains(&texture_id) {
                texture_ids.push(texture_id);
            }
        }

        self.resources_mut()
            .atlas_texture_bind_groups
            .retain(|texture_id, _| texture_ids.contains(texture_id));

        for texture_id in texture_ids {
            let Some(texture_info) = self.atlas.get_texture_info(texture_id) else {
                self.resources_mut()
                    .atlas_texture_bind_groups
                    .remove(&texture_id);
                continue;
            };
            let is_current = self
                .resources()
                .atlas_texture_bind_groups
                .get(&texture_id)
                .is_some_and(|cached| cached.texture_generation == texture_info.generation);
            if is_current {
                continue;
            }

            let bind_group =
                self.create_texture_bind_group("atlas_texture_bind_group", &texture_info.view);
            self.resources_mut().atlas_texture_bind_groups.insert(
                texture_id,
                CachedTextureBindGroup {
                    texture_generation: texture_info.generation,
                    bind_group,
                },
            );
        }
    }

    fn set_pass_depth_range(
        pass: &mut wgpu::RenderPass<'_>,
        size: Size<DevicePixels>,
        min_depth: f32,
        max_depth: f32,
    ) {
        pass.set_viewport(
            0.0,
            0.0,
            size.width.0 as f32,
            size.height.0 as f32,
            min_depth,
            max_depth,
        );
    }

    /// The pipeline is requested only once the draw is known to happen, so a
    /// lazily compiled pipeline is not built for a batch that draws nothing.
    fn draw_instances(
        &self,
        instances: &InstanceBinding,
        pipeline: impl FnOnce(&WgpuPipelines) -> &wgpu::RenderPipeline,
        range: Range<u32>,
        pass: &mut wgpu::RenderPass<'_>,
    ) {
        if range.is_empty() {
            return;
        }
        pass.set_pipeline(pipeline(&self.resources().pipelines));
        pass.set_bind_group(0, &self.resources().globals_bind_group, &[]);
        pass.set_bind_group(1, &instances.bind_group, &[]);
        pass.draw(
            0..4,
            instances.first_instance + range.start..instances.first_instance + range.end,
        );
    }

    /// Like [`Self::draw_instances`], the pipeline is requested after the
    /// early returns.
    fn draw_sprites(
        &self,
        sprite_instances: &InstanceBinding,
        texture_id: AtlasTextureId,
        pipeline: impl FnOnce(&WgpuPipelines) -> &wgpu::RenderPipeline,
        range: Range<u32>,
        pass: &mut wgpu::RenderPass<'_>,
    ) -> Result<()> {
        if range.is_empty() {
            return Ok(());
        }
        let resources = self.resources();
        // The atlas has released this texture; the batch belongs to a stale
        // paint that will be replaced once its view re-renders.
        let Some(texture) = resources.atlas_texture_bind_groups.get(&texture_id) else {
            return Ok(());
        };
        pass.set_pipeline(pipeline(&resources.pipelines));
        pass.set_bind_group(0, &resources.globals_bind_group, &[]);
        pass.set_bind_group(1, &sprite_instances.bind_group, &[]);
        pass.set_bind_group(2, &texture.bind_group, &[]);
        pass.draw(
            0..4,
            sprite_instances.first_instance + range.start
                ..sprite_instances.first_instance + range.end,
        );
        Ok(())
    }

    unsafe fn instance_bytes<T>(instances: &[T]) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                instances.as_ptr() as *const u8,
                std::mem::size_of_val(instances),
            )
        }
    }

    fn draw_paths_from_intermediate(
        &mut self,
        sprites: &[PathSprite],
        instance_offset: &mut u64,
        pass: &mut wgpu::RenderPass<'_>,
    ) -> Result<()> {
        let Some(path_intermediate_view) = self.resources().path_intermediate_view.clone() else {
            return Ok(());
        };
        let instances =
            self.write_instance_binding("path_sprites_bind_group", instance_offset, sprites)?;
        let texture = self.create_texture_bind_group(
            "path_intermediate_texture_bind_group",
            &path_intermediate_view,
        );
        let resources = self.resources();
        pass.set_pipeline(resources.pipelines.paths());
        pass.set_bind_group(0, &resources.globals_bind_group, &[]);
        pass.set_bind_group(1, &instances.bind_group, &[]);
        pass.set_bind_group(2, &texture, &[]);
        pass.draw(
            0..4,
            instances.first_instance..instances.first_instance + sprites.len() as u32,
        );
        Ok(())
    }

    /// Rasterizes `paths` into every tile that `sprites` will sample and
    /// copies each tile into the intermediate. Tiles the sprites do not reach
    /// keep stale content from earlier batches, which nothing reads.
    fn draw_paths_to_intermediate(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        paths: &[Path<ScaledPixels>],
        sprites: &[PathSprite],
        size: Size<DevicePixels>,
        instance_offset: &mut u64,
    ) -> Result<bool> {
        if paths.iter().all(|path| path.vertices.is_empty()) {
            return Ok(false);
        }
        if self.resources().path_intermediate_texture.is_none()
            || self.resources().path_tile_view.is_none()
        {
            return Ok(false);
        }

        let mut vertices = Vec::new();
        for path in paths {
            let bounds = path.clipped_bounds();
            vertices.extend(path.vertices.iter().map(|vertex| PathRasterizationVertex {
                xy_position: vertex.xy_position,
                st_position: vertex.st_position,
                color: path.color,
                bounds,
            }));
        }
        // Uploaded once per batch: every tile draws the same window-space
        // vertices and only moves its viewport.
        let vertex_binding = self.write_instance_binding(
            "path_rasterization_bind_group",
            instance_offset,
            &vertices,
        )?;

        let tile_size = path_tile_size(size);
        for tile in path_tiles(sprites, size, tile_size) {
            let resources = self.resources();
            let (Some(intermediate), Some(tile_texture), Some(tile_view)) = (
                resources.path_intermediate_texture.as_ref(),
                resources.path_tile_texture.as_ref(),
                resources.path_tile_view.as_ref(),
            ) else {
                return Ok(false);
            };
            let (target_view, resolve_target) = match resources.path_msaa_view.as_ref() {
                Some(msaa_view) => (msaa_view, Some(tile_view)),
                None => (tile_view, None),
            };
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("path_rasterization_pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: target_view,
                        resolve_target,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: None,
                    ..Default::default()
                });

                // A window-sized viewport placed at minus the tile's origin
                // maps window clip space onto this tile, the same transform
                // the full-window target used shifted by whole pixels. wgpu
                // accepts viewport origins down to -2 * max_texture_dimension_2d,
                // which covers any tile of a window within that dimension.
                pass.set_viewport(
                    -(tile.origin.x as f32),
                    -(tile.origin.y as f32),
                    size.width.0 as f32,
                    size.height.0 as f32,
                    0.,
                    1.,
                );
                pass.set_pipeline(resources.pipelines.path_rasterization());
                pass.set_bind_group(0, &resources.path_globals_bind_group, &[]);
                pass.set_bind_group(1, &vertex_binding.bind_group, &[]);
                // The path rasterization shader loads records by vertex
                // index rather than instance index, so the allocation's
                // base shifts the vertex range here.
                pass.draw(
                    vertex_binding.first_instance
                        ..vertex_binding.first_instance + vertices.len() as u32,
                    0..1,
                );
            }
            encoder.copy_texture_to_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: tile_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyTextureInfo {
                    texture: intermediate,
                    mip_level: 0,
                    origin: wgpu::Origin3d {
                        x: tile.origin.x,
                        y: tile.origin.y,
                        z: 0,
                    },
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::Extent3d {
                    width: tile.size.width,
                    height: tile.size.height,
                    depth_or_array_layers: 1,
                },
            );
        }

        Ok(true)
    }

    fn write_instance_binding<T>(
        &mut self,
        label: &str,
        instance_offset: &mut u64,
        instances: &[T],
    ) -> Result<InstanceBinding> {
        let data = unsafe { Self::instance_bytes(instances) };
        // wgpu rejects zero-sized bindings, so empty primitive arrays still
        // reserve the 16-byte minimum.
        let size = (data.len() as u64).max(16);
        let stride = (std::mem::size_of::<T>() as u64).max(1);
        let (alignment, allocation_size) = if self.uses_webgl_instance_data {
            // The texture transport has no binding offset: the shader indexes
            // the instance texture absolutely, so each allocation must start on
            // a whole instance (a stride multiple) and a whole texel, and must
            // end on a texel boundary so the zero padding of its final partial
            // texel cannot overlap the next allocation.
            (
                least_common_multiple(self.instance_data_alignment, stride),
                size.next_multiple_of(INSTANCE_TEXTURE_TEXEL_SIZE),
            )
        } else {
            (self.instance_data_alignment.max(1), size)
        };
        let mut offset = (*instance_offset).next_multiple_of(alignment);
        if offset + allocation_size > self.instance_data_capacity {
            self.grow_instance_data(allocation_size)?;
            offset = 0;
        }
        *instance_offset = offset + allocation_size;

        let first_instance = if self.uses_webgl_instance_data {
            u32::try_from(offset / stride).context("instance index exceeds u32 range")?
        } else {
            0
        };

        let resources = self.resources();
        if !data.is_empty() {
            match &resources.instance_data {
                InstanceData::Storage(buffer) => resources.queue.write_buffer(buffer, offset, data),
                InstanceData::Texture { .. } => {
                    Self::write_instance_texture(resources, offset, data)
                }
            }
        }
        let bind_group = resources
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &resources.bind_group_layouts.instances,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: match &resources.instance_data {
                        InstanceData::Storage(buffer) => {
                            wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer,
                                offset,
                                size: NonZeroU64::new(size),
                            })
                        }
                        InstanceData::Texture { view, .. } => {
                            wgpu::BindingResource::TextureView(view)
                        }
                    },
                }],
            });
        Ok(InstanceBinding {
            bind_group,
            first_instance,
        })
    }

    fn write_instance_texture(resources: &WgpuResources, offset: u64, data: &[u8]) {
        let InstanceData::Texture {
            texture,
            width,
            height,
            ..
        } = &resources.instance_data
        else {
            return;
        };
        let mut byte_offset = 0usize;
        let mut texel_offset = offset / INSTANCE_TEXTURE_TEXEL_SIZE;
        while byte_offset < data.len() {
            let x = (texel_offset % u64::from(*width)) as u32;
            let y = (texel_offset / u64::from(*width)) as u32;
            if y >= *height {
                // The capacity check in write_instance_binding should make this
                // unreachable. Truncating silently would leave stale bytes in the
                // texture and draw garbage for the remaining instances.
                debug_assert!(
                    false,
                    "instance texture write out of bounds: row {y} >= height {}",
                    *height
                );
                log::error!(
                    "instance texture write out of bounds; dropping {} bytes of instance data",
                    data.len() - byte_offset
                );
                return;
            }
            let available_texels = u64::from(*width - x);
            let remaining_bytes = data.len() - byte_offset;
            let complete_texels = remaining_bytes as u64 / INSTANCE_TEXTURE_TEXEL_SIZE;
            let texels = complete_texels.min(available_texels);
            if texels > 0 {
                let byte_count = (texels * INSTANCE_TEXTURE_TEXEL_SIZE) as usize;
                resources.queue.write_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d { x, y, z: 0 },
                        aspect: wgpu::TextureAspect::All,
                    },
                    &data[byte_offset..byte_offset + byte_count],
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(byte_count as u32),
                        rows_per_image: None,
                    },
                    wgpu::Extent3d {
                        width: texels as u32,
                        height: 1,
                        depth_or_array_layers: 1,
                    },
                );
                byte_offset += byte_count;
                texel_offset += texels;
                continue;
            }

            let mut final_texel = [0; INSTANCE_TEXTURE_TEXEL_SIZE as usize];
            final_texel[..remaining_bytes].copy_from_slice(&data[byte_offset..]);
            resources.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d { x, y, z: 0 },
                    aspect: wgpu::TextureAspect::All,
                },
                &final_texel,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(INSTANCE_TEXTURE_TEXEL_SIZE as u32),
                    rows_per_image: None,
                },
                wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
            );
            break;
        }
    }

    fn grow_instance_data(&mut self, required: u64) -> Result<()> {
        let capacity = (self.instance_data_capacity * 2)
            .max(required.next_power_of_two())
            .min(self.max_instance_data_size);
        anyhow::ensure!(
            capacity >= required,
            "instance data needs {required} bytes, above the maximum of {}",
            self.max_instance_data_size
        );
        anyhow::ensure!(
            capacity > self.instance_data_capacity,
            "frame instance data exceeds the {}-byte maximum",
            self.max_instance_data_size
        );
        log::debug!(
            "instance data grown from {} to {capacity}",
            self.instance_data_capacity
        );
        // Bind groups created earlier in the frame keep the previous buffer or
        // texture alive, so allocations written before the grow remain valid;
        // only subsequent writes land in the new allocation.
        self.reallocate_instance_data(capacity);
        Ok(())
    }

    fn reallocate_instance_data(&mut self, capacity: u64) {
        let uses_webgl_instance_data = self.uses_webgl_instance_data;
        let resources = self.resources_mut();
        if uses_webgl_instance_data {
            let max_texture_dimension = resources.device.limits().max_texture_dimension_2d;
            let (instance_data, actual_capacity) =
                Self::create_instance_texture(&resources.device, capacity, max_texture_dimension);
            resources.instance_data = instance_data;
            self.instance_data_capacity = actual_capacity;
        } else {
            resources.instance_data =
                InstanceData::Storage(resources.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("instance_buffer"),
                    size: capacity,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }));
            self.instance_data_capacity = capacity;
        }
        self.resources.render_memory.publish(
            RenderMemoryGauge::InstanceBuffers,
            self.instance_data_capacity,
        );
    }
}

impl WgpuRenderer {
    /// Mark the surface as unconfigured so rendering is skipped until a new
    /// surface is provided via [`replace_surface`](Self::replace_surface).
    ///
    /// This does **not** drop the renderer — the device, queue, atlas, and
    /// pipelines stay alive.  Use this when the native window is destroyed
    /// (e.g. Android `TerminateWindow`) but you intend to re-create the
    /// surface later without losing cached atlas textures.
    pub fn unconfigure_surface(&mut self) {
        self.state = match std::mem::replace(&mut self.state, RendererState::Released) {
            RendererState::Ready { core, surface } => {
                drop(surface);
                RendererState::Unconfigured { core }
            }
            state @ (RendererState::Unconfigured { .. } | RendererState::Released) => state,
        };
        // Drop intermediate textures since they reference the old surface size.
        if let Some(core) = self.core_mut() {
            core.resources.invalidate_intermediate_textures();
        }
    }

    /// Replace the wgpu surface with a new one (e.g. after Android destroys
    /// and recreates the native window).  Keeps the device, queue, atlas, and
    /// all pipelines intact so cached `AtlasTextureId`s remain valid.
    ///
    /// The `instance` **must** be the same [`wgpu::Instance`] that was used to
    /// create the adapter and device (i.e. from the [`WgpuContext`]).  Using a
    /// different instance will cause a "Device does not exist" panic because
    /// the wgpu device is bound to its originating instance.
    #[cfg(not(target_family = "wasm"))]
    pub fn replace_surface<W: HasWindowHandle>(
        &mut self,
        window: &W,
        config: WgpuSurfaceConfig,
        instance: &wgpu::Instance,
    ) -> anyhow::Result<()> {
        let window_handle = window
            .window_handle()
            .map_err(|e| anyhow::anyhow!("Failed to get window handle: {e}"))?;

        let surface = create_surface(instance, window_handle.as_raw())?;

        let width = (config.size.width.0 as u32).max(1);
        let height = (config.size.height.0 as u32).max(1);

        let alpha_mode = if config.transparent {
            self.transparent_alpha_mode
        } else {
            self.opaque_alpha_mode
        };

        self.surface_config.width = width;
        self.surface_config.height = height;
        self.surface_config.alpha_mode = alpha_mode;
        if let Some(mode) = config.preferred_present_mode {
            self.surface_config.present_mode = mode;
        }

        let mut core = match std::mem::replace(&mut self.state, RendererState::Released) {
            RendererState::Ready {
                core,
                surface: old_surface,
            } => {
                drop(old_surface);
                core
            }
            RendererState::Unconfigured { core } => core,
            RendererState::Released => {
                anyhow::bail!("Cannot replace the surface: GPU resources have been released")
            }
        };
        surface.configure(&core.resources.device, &self.surface_config);
        core.resources.invalidate_intermediate_textures();
        self.state = RendererState::Ready { surface, core };

        Ok(())
    }

    /// Whether [`unconfigure_surface`](Self::unconfigure_surface) left the
    /// renderer waiting for a surface. Draws are no-ops until one is restored.
    pub fn is_unconfigured(&self) -> bool {
        matches!(self.state, RendererState::Unconfigured { .. })
    }

    /// Creates and configures a surface for `window` again after
    /// [`unconfigure_surface`](Self::unconfigure_surface), at the last
    /// recorded drawable size, transparency and present mode. Used to give a
    /// hidden window's swapchain and size-dependent targets back while it is
    /// not shown; the targets are rebuilt by the next draw.
    #[cfg(not(target_family = "wasm"))]
    pub fn restore_surface<W: HasWindowHandle>(&mut self, window: &W) -> anyhow::Result<()> {
        let instance = self
            .context
            .as_ref()
            .and_then(|context| {
                context
                    .borrow()
                    .as_ref()
                    .map(|context| context.instance.clone())
            })
            .context("Cannot restore the surface: no GPU context")?;
        let config = WgpuSurfaceConfig {
            size: self.viewport_size(),
            transparent: self.surface_config.alpha_mode != self.opaque_alpha_mode,
            preferred_present_mode: Some(self.surface_config.present_mode),
        };
        self.replace_surface(window, config, &instance)
    }

    pub fn destroy(&mut self) {
        // Release surface-bound GPU resources eagerly so the underlying native
        // window can be destroyed before the renderer itself is dropped.
        self.state = RendererState::Released;
    }

    /// Returns true if the GPU device was lost and recovery is needed.
    pub fn device_lost(&self) -> bool {
        self.device_errors.device_lost()
    }

    /// Returns true if a redraw is needed because GPU state was cleared.
    /// Calling this method clears the flag.
    pub fn needs_redraw(&mut self) -> bool {
        std::mem::take(&mut self.needs_redraw)
    }

    /// Recovers from a lost GPU device by recreating the renderer with a new context.
    ///
    /// Call this after detecting `device_lost()` returns true.
    ///
    /// This method coordinates recovery across multiple windows:
    /// - The first window to call this will recreate the shared context
    /// - Subsequent windows will adopt the already-recovered context
    ///
    /// After a failed attempt to recreate the context, further attempts wait
    /// out a backoff that doubles up to a few seconds. A call made before
    /// the next attempt is due returns `Ok(())` without doing anything and
    /// `device_lost()` stays true, so the caller keeps requesting frames and
    /// a later one retries.
    #[cfg(not(target_family = "wasm"))]
    pub fn recover<W>(&mut self, window: &W) -> anyhow::Result<()>
    where
        W: HasWindowHandle + HasDisplayHandle + std::fmt::Debug + Send + Sync + Clone + 'static,
    {
        self.recover_with_clock(window, std::time::Instant::now)
    }

    #[cfg(not(target_family = "wasm"))]
    fn recover_with_clock<W>(
        &mut self,
        window: &W,
        now: impl Fn() -> std::time::Instant,
    ) -> anyhow::Result<()>
    where
        W: HasWindowHandle + HasDisplayHandle + std::fmt::Debug + Send + Sync + Clone + 'static,
    {
        let gpu_context = self.context.as_ref().expect("recover requires gpu_context");

        // Check if another window already recovered the context
        let needs_new_context = gpu_context
            .borrow()
            .as_ref()
            .is_none_or(|ctx| ctx.device_lost());

        // Adopting a context another window recovered is cheap, so only
        // recreating one waits out the backoff.
        if needs_new_context && !self.recovery_backoff.is_due(now()) {
            return Ok(());
        }
        let result = self.try_recover(window, needs_new_context);
        // Measured from the end of the attempt, which can itself take long.
        if result.is_err() {
            self.recovery_backoff.record_failure(now());
        }
        result
    }

    #[cfg(not(target_family = "wasm"))]
    fn try_recover<W>(&mut self, window: &W, needs_new_context: bool) -> anyhow::Result<()>
    where
        W: HasWindowHandle + HasDisplayHandle + std::fmt::Debug + Send + Sync + Clone + 'static,
    {
        let gpu_context = self.context.as_ref().expect("recover requires gpu_context");

        let window_handle = window
            .window_handle()
            .map_err(|e| anyhow::anyhow!("Failed to get window handle: {e}"))?;

        let surface = if needs_new_context {
            log::warn!("GPU device lost, recreating context...");

            // Drop old resources to release Arc<Device>/Arc<Queue> and GPU resources
            self.state = RendererState::Released;
            *gpu_context.borrow_mut() = None;

            // Wait briefly for the GPU driver to stabilize, then try to
            // recreate the context without software renderers. If this fails
            // the caller should request another frame and retry — the real GPU
            // may need more time to come back (e.g. after suspend/resume).
            // Retries are already spaced out by the backoff, so only the first
            // attempt sleeps.
            if self.recovery_backoff.failures == 0 {
                std::thread::sleep(RECOVERY_INITIAL_DELAY);
            }

            let instance = WgpuContext::instance(Some(Box::new(window.clone())));
            let surface = create_surface(&instance, window_handle.as_raw())?;
            let new_context =
                WgpuContext::new_rejecting_software(instance, &surface, self.compositor_gpu)?;
            *gpu_context.borrow_mut() = Some(new_context);
            surface
        } else {
            let ctx_ref = gpu_context.borrow();
            let instance = &ctx_ref.as_ref().unwrap().instance;
            create_surface(instance, window_handle.as_raw())?
        };

        let config = WgpuSurfaceConfig {
            size: gpui::Size {
                width: gpui::DevicePixels(self.surface_config.width as i32),
                height: gpui::DevicePixels(self.surface_config.height as i32),
            },
            transparent: self.surface_config.alpha_mode != wgpu::CompositeAlphaMode::Opaque,
            preferred_present_mode: Some(self.surface_config.present_mode),
        };
        let gpu_context = Rc::clone(gpu_context);
        let ctx_ref = gpu_context.borrow();
        let context = ctx_ref.as_ref().expect("context should exist");

        self.state = RendererState::Released;
        self.atlas.handle_device_lost(context);

        let is_bgr = self.is_bgr;
        *self = Self::new_internal(
            Some(gpu_context.clone()),
            context,
            surface,
            config,
            self.compositor_gpu,
            self.atlas.clone(),
        )?;
        self.set_subpixel_layout(is_bgr);

        log::info!("GPU recovery complete");
        Ok(())
    }
}

#[cfg(all(
    not(target_family = "wasm"),
    any(test, feature = "bench-support", feature = "test-support")
))]
struct HeadlessRenderTarget {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
}

#[cfg(all(
    not(target_family = "wasm"),
    any(test, feature = "bench-support", feature = "test-support")
))]
impl HeadlessRenderTarget {
    fn size(&self) -> Size<DevicePixels> {
        Size {
            width: DevicePixels(self.texture.width() as i32),
            height: DevicePixels(self.texture.height() as i32),
        }
    }
}

#[cfg(all(
    not(target_family = "wasm"),
    any(test, feature = "bench-support", feature = "test-support")
))]
pub struct WgpuHeadlessRenderer {
    context: WgpuContext,
    core: WgpuRendererCore,
    render_target: Option<HeadlessRenderTarget>,
    observed_error_generation: u64,
}

#[cfg(all(
    not(target_family = "wasm"),
    any(test, feature = "bench-support", feature = "test-support")
))]
impl WgpuHeadlessRenderer {
    pub fn new() -> anyhow::Result<Self> {
        let (context, target_format) = WgpuContext::new_headless()?;
        let atlas = Arc::new(WgpuAtlas::from_context(&context));
        let core = WgpuRendererCore::new(
            &context,
            atlas,
            target_format,
            wgpu::CompositeAlphaMode::Opaque,
        );

        Ok(Self {
            context,
            core,
            render_target: None,
            observed_error_generation: 0,
        })
    }

    fn ensure_render_target(&mut self, size: Size<DevicePixels>) -> anyhow::Result<()> {
        anyhow::ensure!(
            size.width.0 > 0 && size.height.0 > 0,
            "invalid headless render target size: {size:?}"
        );
        anyhow::ensure!(
            size.width.0 as u32 <= self.core.max_texture_size
                && size.height.0 as u32 <= self.core.max_texture_size,
            "headless render target size {size:?} exceeds maximum texture dimension {}",
            self.core.max_texture_size
        );
        if self
            .render_target
            .as_ref()
            .is_some_and(|target| target.size() == size)
        {
            return Ok(());
        }

        let texture = self
            .core
            .resources
            .device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("headless_render_target"),
                size: wgpu::Extent3d {
                    width: size.width.0 as u32,
                    height: size.height.0 as u32,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: self.core.target_format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        self.render_target = Some(HeadlessRenderTarget { texture, view });
        Ok(())
    }

    fn check_gpu_errors(&mut self) -> anyhow::Result<()> {
        anyhow::ensure!(
            !self.context.device_lost(),
            "GPU device was lost during headless rendering"
        );
        if let Some(error) = self
            .context
            .errors()
            .observe_error(&mut self.observed_error_generation)
        {
            anyhow::bail!("GPU error during headless rendering: {error}");
        }
        Ok(())
    }

    fn render(&mut self, scene: &Scene, size: Size<DevicePixels>) -> anyhow::Result<()> {
        self.check_gpu_errors()?;
        self.ensure_render_target(size)?;
        let view = self
            .render_target
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Headless render target was not created"))?
            .view
            .clone();
        self.core
            .render_frame(scene, &view, size, false, wgpu::Color::BLACK)?;
        Ok(())
    }

    /// Copies the current render target back to the CPU. Dimensions come from the
    /// target texture itself, so the copy can never disagree with what was rendered.
    fn read_image(&mut self) -> anyhow::Result<image::RgbaImage> {
        let target = self
            .render_target
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Headless render target was not created"))?;
        let width = target.texture.width();
        let height = target.texture.height();
        let bytes_per_row = width
            .checked_mul(4)
            .ok_or_else(|| anyhow::anyhow!("Headless render target row size overflowed"))?;
        let padded_bytes_per_row = bytes_per_row
            .checked_next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
            .ok_or_else(|| anyhow::anyhow!("Headless padded row size overflowed"))?;
        let buffer_size = u64::from(padded_bytes_per_row)
            .checked_mul(u64::from(height))
            .ok_or_else(|| anyhow::anyhow!("Headless readback buffer size overflowed"))?;
        anyhow::ensure!(
            buffer_size <= self.core.resources.device.limits().max_buffer_size,
            "Headless readback buffer size {buffer_size} exceeds maximum buffer size {}",
            self.core.resources.device.limits().max_buffer_size
        );
        let readback_buffer = self
            .core
            .resources
            .device
            .create_buffer(&wgpu::BufferDescriptor {
                label: Some("headless_readback_buffer"),
                size: buffer_size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
        let mut encoder =
            self.core
                .resources
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("headless_readback_encoder"),
                });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &target.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &readback_buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bytes_per_row),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        let submission = self
            .core
            .resources
            .queue
            .submit(std::iter::once(encoder.finish()));
        let (sender, receiver) = std::sync::mpsc::channel();
        readback_buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                if sender.send(result).is_err() {
                    log::error!("Headless readback receiver was dropped before mapping completed");
                }
            });
        self.core
            .resources
            .device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: Some(std::time::Duration::from_secs(30)),
            })
            .map_err(|error| anyhow::anyhow!("Failed to wait for headless rendering: {error}"))?;
        receiver
            .recv_timeout(std::time::Duration::from_secs(30))
            .map_err(|error| anyhow::anyhow!("Failed to receive headless mapping result: {error}"))?
            .map_err(|error| anyhow::anyhow!("Failed to map headless readback buffer: {error}"))?;
        self.check_gpu_errors()?;

        let mapped_data = readback_buffer.slice(..).get_mapped_range();
        let pixel_capacity = usize::try_from(u64::from(bytes_per_row) * u64::from(height))
            .map_err(|_| anyhow::anyhow!("Headless image size exceeds addressable memory"))?;
        let mut pixels = Vec::with_capacity(pixel_capacity);
        for row in mapped_data
            .chunks_exact(padded_bytes_per_row as usize)
            .take(height as usize)
        {
            pixels.extend_from_slice(&row[..bytes_per_row as usize]);
        }
        drop(mapped_data);
        readback_buffer.unmap();

        if self.core.target_format == wgpu::TextureFormat::Bgra8Unorm {
            for pixel in pixels.chunks_exact_mut(4) {
                pixel.swap(0, 2);
            }
        }

        image::RgbaImage::from_raw(width, height, pixels)
            .ok_or_else(|| anyhow::anyhow!("Failed to create image from headless pixel data"))
    }
}

#[cfg(all(
    not(target_family = "wasm"),
    any(test, feature = "bench-support", feature = "test-support")
))]
impl gpui::PlatformHeadlessRenderer for WgpuHeadlessRenderer {
    fn render_scene_to_image(
        &mut self,
        scene: &Scene,
        size: Size<DevicePixels>,
    ) -> anyhow::Result<image::RgbaImage> {
        self.render(scene, size)?;
        self.read_image()
    }

    fn render_scene(&mut self, scene: &Scene, size: Size<DevicePixels>) -> anyhow::Result<()> {
        self.render(scene, size)
    }

    fn sprite_atlas(&self) -> Arc<dyn gpui::PlatformAtlas> {
        self.core.atlas.clone()
    }
}

fn instance_range(range: Range<usize>) -> Range<u32> {
    range.start as u32..range.end as u32
}

#[cfg(not(target_family = "wasm"))]
fn create_surface(
    instance: &wgpu::Instance,
    raw_window_handle: raw_window_handle::RawWindowHandle,
) -> anyhow::Result<wgpu::Surface<'static>> {
    unsafe {
        instance
            .create_surface_unsafe(wgpu::SurfaceTargetUnsafe::RawHandle {
                // Fall back to the display handle already provided via InstanceDescriptor::display.
                raw_display_handle: None,
                raw_window_handle,
            })
            .map_err(|e| anyhow::anyhow!("{e}"))
    }
}

struct RenderingParameters {
    path_sample_count: u32,
    depth_format: wgpu::TextureFormat,
    gamma_ratios: [f32; 4],
    grayscale_enhanced_contrast: f32,
    subpixel_enhanced_contrast: f32,
}

impl RenderingParameters {
    fn new(adapter: &wgpu::Adapter, surface_format: wgpu::TextureFormat) -> Self {
        use std::env;

        let format_features = adapter.get_texture_format_features(surface_format);
        let path_sample_count = [4, 2, 1]
            .into_iter()
            .find(|&n| format_features.flags.sample_count_supported(n))
            .unwrap_or(1);

        // Depth16Unorm is a core WebGPU format that wgpu reports as
        // depth-renderable on Vulkan, Metal, DX12 and GL (WebGL2 exposes it
        // as DEPTH_COMPONENT16), so the fallback is only a guard against an
        // adapter that omits it; the pipelines and the attachment are created
        // from this one value so they can never disagree.
        let depth_format = if adapter
            .get_texture_format_features(PREFERRED_DEPTH_FORMAT)
            .allowed_usages
            .contains(wgpu::TextureUsages::RENDER_ATTACHMENT)
        {
            PREFERRED_DEPTH_FORMAT
        } else {
            log::warn!(
                "adapter {:?} cannot render to {:?}; using {:?} for the depth buffer",
                adapter.get_info().name,
                PREFERRED_DEPTH_FORMAT,
                FALLBACK_DEPTH_FORMAT
            );
            FALLBACK_DEPTH_FORMAT
        };

        let gamma = env::var("ZED_FONTS_GAMMA")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1.8_f32)
            .clamp(1.0, 2.2);
        let gamma_ratios = get_gamma_correction_ratios(gamma);

        let grayscale_enhanced_contrast = env::var("ZED_FONTS_GRAYSCALE_ENHANCED_CONTRAST")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1.0_f32)
            .max(0.0);

        let subpixel_enhanced_contrast = env::var("ZED_FONTS_SUBPIXEL_ENHANCED_CONTRAST")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.5_f32)
            .max(0.0);

        Self {
            path_sample_count,
            depth_format,
            gamma_ratios,
            grayscale_enhanced_contrast,
            subpixel_enhanced_contrast,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpui::{
        BorderStyle, ColorSpace, ContentMask, Corners, Edges, Hsla, MonochromeSprite,
        PolychromeSprite, Quad, Shadow, Size, SubpixelSprite, Underline, linear_color_stop,
        linear_gradient,
    };
    #[cfg(target_os = "linux")]
    use gpui::{DevicePixels, PlatformAtlas, PlatformHeadlessRenderer, Scene};

    #[cfg(target_os = "linux")]
    fn device_size(width: i32, height: i32) -> Size<DevicePixels> {
        Size {
            width: DevicePixels(width),
            height: DevicePixels(height),
        }
    }

    #[cfg(target_os = "linux")]
    fn solid_quad(x: f32, y: f32, width: f32, height: f32, color: Hsla) -> Quad {
        let bounds = Bounds {
            origin: Point {
                x: x.into(),
                y: y.into(),
            },
            size: Size {
                width: width.into(),
                height: height.into(),
            },
        };
        Quad {
            order: 0,
            border_style: BorderStyle::Solid,
            bounds,
            content_mask: ContentMask { bounds },
            background: color.into(),
            border_color: color,
            corner_radii: Corners::default(),
            border_widths: Edges::default(),
        }
    }

    /// Channels are compared with a small tolerance so the assertions hold across
    /// drivers without pinning exact rasterizer output.
    #[cfg(target_os = "linux")]
    fn assert_pixel(image: &image::RgbaImage, x: u32, y: u32, expected: [u8; 4]) {
        let actual = image.get_pixel(x, y).0;
        assert!(
            actual
                .iter()
                .zip(expected)
                .all(|(actual, expected)| actual.abs_diff(expected) <= 3),
            "pixel ({x}, {y}) was {actual:?}, expected {expected:?}"
        );
    }

    #[cfg(target_os = "linux")]
    const RED: [u8; 4] = [255, 0, 0, 255];
    #[cfg(target_os = "linux")]
    const BLUE: [u8; 4] = [0, 0, 255, 255];
    #[cfg(target_os = "linux")]
    const BLACK: [u8; 4] = [0, 0, 0, 255];

    #[cfg(target_os = "linux")]
    #[test]
    fn headless_renderer_draws_quads_with_distinct_colors() -> anyhow::Result<()> {
        let mut renderer = WgpuHeadlessRenderer::new()?;
        let mut scene = Scene::default();
        scene.insert_primitive(solid_quad(0.0, 0.0, 32.0, 32.0, gpui::red()));
        scene.insert_primitive(solid_quad(32.0, 0.0, 32.0, 32.0, gpui::blue()));
        scene.finish();

        let image = renderer.render_scene_to_image(&scene, device_size(64, 32))?;
        assert_eq!(image.dimensions(), (64, 32));
        assert_pixel(&image, 8, 16, RED);
        assert_pixel(&image, 24, 16, RED);
        assert_pixel(&image, 40, 16, BLUE);
        assert_pixel(&image, 56, 16, BLUE);
        Ok(())
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn headless_renderer_captures_each_requested_size() -> anyhow::Result<()> {
        let mut renderer = WgpuHeadlessRenderer::new()?;
        let mut scene = Scene::default();
        scene.insert_primitive(solid_quad(2.0, 2.0, 4.0, 3.0, gpui::red()));
        scene.finish();

        // 13 px rows are 52 bytes, forcing readback to strip copy-row padding.
        let image = renderer.render_scene_to_image(&scene, device_size(13, 7))?;
        assert_eq!(image.dimensions(), (13, 7));
        assert_pixel(&image, 0, 0, BLACK);
        assert_pixel(&image, 3, 3, RED);
        assert_pixel(&image, 12, 6, BLACK);

        let image = renderer.render_scene_to_image(&scene, device_size(17, 9))?;
        assert_eq!(image.dimensions(), (17, 9));
        assert_pixel(&image, 3, 3, RED);
        assert_pixel(&image, 16, 8, BLACK);

        let image = renderer.render_scene_to_image(&Scene::default(), device_size(13, 7))?;
        assert_eq!(image.dimensions(), (13, 7));
        assert!(image.pixels().all(|pixel| pixel.0 == BLACK));
        Ok(())
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn headless_renderer_reuses_target_for_same_size() -> anyhow::Result<()> {
        let mut renderer = WgpuHeadlessRenderer::new()?;
        let target_texture = |renderer: &WgpuHeadlessRenderer| {
            renderer
                .render_target
                .as_ref()
                .map(|target| target.texture.clone())
        };

        renderer.render_scene(&Scene::default(), device_size(16, 16))?;
        let first = target_texture(&renderer);
        assert!(first.is_some());

        renderer.render_scene(&Scene::default(), device_size(16, 16))?;
        assert_eq!(target_texture(&renderer), first);

        renderer.render_scene(&Scene::default(), device_size(16, 17))?;
        assert_ne!(target_texture(&renderer), first);
        Ok(())
    }

    #[cfg(target_os = "linux")]
    fn red_square_path(size: f32) -> gpui::Path<ScaledPixels> {
        let mut path = gpui::Path::new(gpui::point(gpui::px(0.), gpui::px(0.)));
        path.line_to(gpui::point(gpui::px(size), gpui::px(0.)));
        path.line_to(gpui::point(gpui::px(size), gpui::px(size)));
        path.line_to(gpui::point(gpui::px(0.), gpui::px(size)));
        path.content_mask = ContentMask {
            bounds: Bounds {
                origin: gpui::point(gpui::px(0.), gpui::px(0.)),
                size: Size {
                    width: gpui::px(size),
                    height: gpui::px(size),
                },
            },
        };
        path.color = gpui::red().into();
        path.scale(1.0)
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn path_targets_exist_only_around_frames_with_paths() -> anyhow::Result<()> {
        let mut renderer = WgpuHeadlessRenderer::new()?;
        let path_targets_resident = |renderer: &WgpuHeadlessRenderer| {
            let resources = &renderer.core.resources;
            let has_msaa = renderer.core.rendering_params.path_sample_count > 1;
            assert_eq!(
                resources.path_intermediate_view.is_some(),
                resources.path_intermediate_texture.is_some()
            );
            assert_eq!(
                resources.path_msaa_texture.is_some(),
                has_msaa && resources.path_intermediate_texture.is_some()
            );
            assert_eq!(
                resources.path_tile_texture.is_some(),
                resources.path_intermediate_texture.is_some()
            );
            let published = resources
                .render_memory
                .published(RenderMemoryGauge::PathTextures);
            assert_eq!(
                published > 0,
                resources.path_intermediate_texture.is_some(),
                "path texture gauge reads {published} bytes"
            );
            resources.path_intermediate_texture.is_some()
        };
        let published = |renderer: &WgpuHeadlessRenderer, gauge| {
            renderer.core.resources.render_memory.published(gauge)
        };
        assert_eq!(
            published(&renderer, RenderMemoryGauge::InstanceBuffers),
            renderer.core.instance_data_capacity
        );
        let mut path_scene = Scene::default();
        path_scene.insert_primitive(red_square_path(16.0));
        path_scene.finish();
        let empty_scene = Scene::default();

        renderer.render_scene(&empty_scene, device_size(32, 32))?;
        assert!(!path_targets_resident(&renderer));
        assert!(renderer.core.resources.depth_texture.is_some());
        assert!(published(&renderer, RenderMemoryGauge::DepthTextures) >= 32 * 32 * 2);

        let image = renderer.render_scene_to_image(&path_scene, device_size(32, 32))?;
        assert!(path_targets_resident(&renderer));
        // The intermediate alone is 32x32 at four bytes per texel.
        let path_bytes = published(&renderer, RenderMemoryGauge::PathTextures);
        assert!(path_bytes >= 32 * 32 * 4);
        assert!(gpui::render_memory_gauges().path_texture_bytes >= path_bytes);
        assert_pixel(&image, 12, 4, RED);
        assert_pixel(&image, 4, 12, RED);
        assert_pixel(&image, 24, 24, BLACK);

        for _ in 1..PATH_TARGET_IDLE_FRAMES {
            renderer.render_scene(&empty_scene, device_size(32, 32))?;
        }
        assert!(path_targets_resident(&renderer));
        renderer.render_scene(&empty_scene, device_size(32, 32))?;
        assert!(!path_targets_resident(&renderer));

        let image = renderer.render_scene_to_image(&path_scene, device_size(32, 32))?;
        assert!(path_targets_resident(&renderer));
        assert_pixel(&image, 12, 4, RED);

        renderer.render_scene(&empty_scene, device_size(32, 33))?;
        assert!(!path_targets_resident(&renderer));

        // A release without a draw is seen at once.
        renderer.render_scene(&path_scene, device_size(32, 33))?;
        assert!(path_targets_resident(&renderer));
        renderer.core.resources.invalidate_intermediate_textures();
        assert!(!path_targets_resident(&renderer));
        assert_eq!(published(&renderer, RenderMemoryGauge::DepthTextures), 0);
        Ok(())
    }

    /// A hidden window's renderer: no surface, the core of `headless` moved in
    /// and a fresh one left behind. There is no window to create a real
    /// surface for, so the core is handed back to draw the re-shown frame.
    #[cfg(target_os = "linux")]
    fn unconfigured_renderer(headless: &mut WgpuHeadlessRenderer) -> WgpuRenderer {
        let atlas = headless.core.atlas.clone();
        let format = headless.core.target_format;
        let spare = WgpuRendererCore::new(
            &headless.context,
            atlas.clone(),
            format,
            wgpu::CompositeAlphaMode::Opaque,
        );
        let core = std::mem::replace(&mut headless.core, spare);
        WgpuRenderer {
            context: None,
            compositor_gpu: None,
            max_texture_size: core.max_texture_size,
            state: RendererState::Unconfigured { core },
            surface_config: wgpu::SurfaceConfiguration {
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                format,
                width: 32,
                height: 32,
                present_mode: wgpu::PresentMode::Fifo,
                desired_maximum_frame_latency: 2,
                alpha_mode: wgpu::CompositeAlphaMode::Opaque,
                view_formats: vec![],
            },
            atlas,
            transparent_alpha_mode: wgpu::CompositeAlphaMode::Opaque,
            opaque_alpha_mode: wgpu::CompositeAlphaMode::Opaque,
            is_bgr: false,
            failed_frame_count: 0,
            device_errors: Arc::clone(headless.context.errors()),
            observed_error_generation: 0,
            last_surface_error: None,
            needs_redraw: false,
            recovery_backoff: RecoveryBackoff::default(),
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn hidden_window_releases_targets_and_draws_again_when_shown() -> anyhow::Result<()> {
        let mut headless = WgpuHeadlessRenderer::new()?;
        let mut scene = Scene::default();
        scene.insert_primitive(solid_quad(16.0, 0.0, 16.0, 32.0, gpui::blue()));
        scene.insert_primitive(red_square_path(16.0));
        scene.finish();
        headless.render_scene(&scene, device_size(32, 32))?;

        let mut hidden = unconfigured_renderer(&mut headless);
        hidden.unconfigure_surface();
        assert!(hidden.is_unconfigured());
        let resident_bytes = |renderer: &WgpuRenderer| {
            let ledger = &renderer
                .core()
                .expect("core is kept")
                .resources
                .render_memory;
            ledger.published(RenderMemoryGauge::DepthTextures)
                + ledger.published(RenderMemoryGauge::PathTextures)
        };
        assert_eq!(resident_bytes(&hidden), 0);

        // A frame requested while hidden draws nothing and allocates nothing.
        assert!(!hidden.draw(&scene));
        assert_eq!(resident_bytes(&hidden), 0);
        let resources = &hidden.core().expect("core is kept").resources;
        assert!(resources.depth_texture.is_none());
        assert!(resources.path_intermediate_texture.is_none());

        let RendererState::Unconfigured { core } =
            std::mem::replace(&mut hidden.state, RendererState::Released)
        else {
            anyhow::bail!("the hidden renderer lost its core");
        };
        headless.core = core;
        let image = headless.render_scene_to_image(&scene, device_size(32, 32))?;
        assert_pixel(&image, 8, 8, RED);
        assert_pixel(&image, 24, 8, BLUE);
        assert_pixel(&image, 8, 24, BLACK);
        assert!(headless.core.resources.depth_texture.is_some());
        assert!(headless.core.resources.path_intermediate_texture.is_some());
        Ok(())
    }

    /// A window whose native handle cannot be had, standing in for one whose
    /// surface cannot be created: wgpu has no surface without a real window.
    #[cfg(target_os = "linux")]
    struct UnavailableWindow;

    #[cfg(target_os = "linux")]
    impl HasWindowHandle for UnavailableWindow {
        fn window_handle(
            &self,
        ) -> Result<raw_window_handle::WindowHandle<'_>, raw_window_handle::HandleError> {
            Err(raw_window_handle::HandleError::Unavailable)
        }
    }

    /// The platforms retry a failed restore on the next frame, which needs
    /// the renderer to stay unconfigured with its core, at the size recorded
    /// by a resize while hidden.
    #[cfg(target_os = "linux")]
    #[test]
    fn failed_restore_keeps_the_hidden_renderer_and_its_resized_config() -> anyhow::Result<()> {
        let mut headless = WgpuHeadlessRenderer::new()?;
        let mut hidden = unconfigured_renderer(&mut headless);
        assert!(hidden.restore_surface(&UnavailableWindow).is_err());
        assert!(hidden.is_unconfigured());

        let WgpuHeadlessRenderer { context, .. } = headless;
        hidden.context = Some(Rc::new(RefCell::new(Some(context))));
        hidden.update_drawable_size(device_size(48, 40));
        assert!(hidden.is_unconfigured());
        assert_eq!(hidden.viewport_size(), device_size(48, 40));
        let resources = &hidden.core().expect("core is kept").resources;
        assert!(resources.depth_texture.is_none());
        assert!(resources.path_intermediate_texture.is_none());

        assert!(hidden.restore_surface(&UnavailableWindow).is_err());
        assert!(hidden.is_unconfigured());
        assert!(hidden.core().is_some());
        assert_eq!(hidden.viewport_size(), device_size(48, 40));
        Ok(())
    }

    /// A window whose handle cannot be had, counting how often recovery asks
    /// for it: every attempt asks first, so the count is the attempt count.
    #[cfg(target_os = "linux")]
    #[derive(Clone, Debug, Default)]
    struct CountingWindow(Arc<std::sync::atomic::AtomicUsize>);

    #[cfg(target_os = "linux")]
    impl CountingWindow {
        fn attempts(&self) -> usize {
            self.0.load(std::sync::atomic::Ordering::Relaxed)
        }
    }

    #[cfg(target_os = "linux")]
    impl HasWindowHandle for CountingWindow {
        fn window_handle(
            &self,
        ) -> Result<raw_window_handle::WindowHandle<'_>, raw_window_handle::HandleError> {
            self.0.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Err(raw_window_handle::HandleError::Unavailable)
        }
    }

    #[cfg(target_os = "linux")]
    impl HasDisplayHandle for CountingWindow {
        fn display_handle(
            &self,
        ) -> Result<raw_window_handle::DisplayHandle<'_>, raw_window_handle::HandleError> {
            Err(raw_window_handle::HandleError::Unavailable)
        }
    }

    /// The platforms call `recover` from every frame while the device is
    /// lost. A failed attempt must not be repeated, nor its settle sleep
    /// paid, until the backoff has passed, and the backoff must stop growing
    /// at its cap so a device that comes back is picked up within seconds.
    #[cfg(target_os = "linux")]
    #[test]
    fn failed_recovery_waits_out_a_capped_backoff() -> anyhow::Result<()> {
        let mut headless = WgpuHeadlessRenderer::new()?;
        let mut renderer = unconfigured_renderer(&mut headless);
        // No shared context: the lost device's context was dropped by the
        // failed attempt, so recovery has to create a new one.
        renderer.context = Some(Rc::new(RefCell::new(None)));
        let window = CountingWindow::default();

        assert!(renderer.recover(&window).is_err());
        assert_eq!(window.attempts(), 1);
        let started = std::time::Instant::now();
        for _ in 0..10 {
            renderer.recover(&window)?;
        }
        assert_eq!(window.attempts(), 1, "a retry ran inside the backoff");
        assert!(started.elapsed() < RECOVERY_INITIAL_DELAY, "a deferred retry slept");

        let start = std::time::Instant::now();
        renderer.recovery_backoff = RecoveryBackoff::default();
        assert!(renderer.recover_with_clock(&window, || start).is_err());
        let mut due = start + RECOVERY_INITIAL_DELAY;
        let mut delay = RECOVERY_INITIAL_DELAY;
        for attempt in 2..=10 {
            let early = due - std::time::Duration::from_millis(1);
            renderer.recover_with_clock(&window, || early)?;
            assert_eq!(window.attempts(), attempt, "retried before the backoff passed");
            assert!(renderer.recover_with_clock(&window, || due).is_err());
            assert_eq!(window.attempts(), attempt + 1, "no retry once the backoff passed");
            delay = (delay * 2).min(RECOVERY_MAX_DELAY);
            due += delay;
        }
        assert_eq!(delay, RECOVERY_MAX_DELAY);
        Ok(())
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn instance_buffer_returns_to_its_initial_size_after_a_large_frame() -> anyhow::Result<()> {
        let mut renderer = WgpuHeadlessRenderer::new()?;
        let initial = renderer.core.instance_data_capacity;
        let quad_count = initial as usize / std::mem::size_of::<Quad>() + 1;
        let mut large_scene = Scene::default();
        for _ in 0..quad_count {
            large_scene.insert_primitive(solid_quad(0.0, 0.0, 16.0, 16.0, gpui::red()));
        }
        large_scene.finish();
        let image = renderer.render_scene_to_image(&large_scene, device_size(32, 32))?;
        assert_pixel(&image, 8, 8, RED);
        let grown = renderer.core.instance_data_capacity;
        assert!(grown > initial);

        let mut small_scene = Scene::default();
        small_scene.insert_primitive(solid_quad(0.0, 0.0, 16.0, 16.0, gpui::blue()));
        small_scene.finish();
        for _ in 1..INSTANCE_SHRINK_FRAMES {
            renderer.render_scene(&small_scene, device_size(32, 32))?;
        }
        assert_eq!(renderer.core.instance_data_capacity, grown);
        renderer.render_scene(&small_scene, device_size(32, 32))?;
        assert_eq!(renderer.core.instance_data_capacity, initial);
        assert_eq!(
            renderer
                .core
                .resources
                .render_memory
                .published(RenderMemoryGauge::InstanceBuffers),
            initial
        );

        let image = renderer.render_scene_to_image(&small_scene, device_size(32, 32))?;
        assert_pixel(&image, 8, 8, BLUE);
        assert_pixel(&image, 24, 24, BLACK);
        Ok(())
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn rare_pipelines_are_compiled_on_first_draw() -> anyhow::Result<()> {
        let mut renderer = WgpuHeadlessRenderer::new()?;
        let compiled = |renderer: &WgpuHeadlessRenderer| {
            let pipelines = &renderer.core.resources.pipelines;
            [
                pipelines.shadows.get().is_some(),
                pipelines.path_rasterization.get().is_some(),
                pipelines.paths.get().is_some(),
                pipelines.poly_sprites.get().is_some(),
            ]
        };
        assert_eq!(compiled(&renderer), [false; 4]);

        let mut quad_scene = Scene::default();
        quad_scene.insert_primitive(solid_quad(0.0, 0.0, 16.0, 16.0, gpui::blue()));
        quad_scene.finish();
        let image = renderer.render_scene_to_image(&quad_scene, device_size(32, 32))?;
        assert_pixel(&image, 8, 8, BLUE);
        assert_eq!(compiled(&renderer), [false; 4]);

        let mut path_scene = Scene::default();
        path_scene.insert_primitive(red_square_path(16.0));
        path_scene.finish();
        let image = renderer.render_scene_to_image(&path_scene, device_size(32, 32))?;
        assert_pixel(&image, 8, 8, RED);
        assert_pixel(&image, 24, 24, BLACK);
        assert_eq!(compiled(&renderer), [false, true, true, false]);

        let sprite = |tile: gpui::AtlasTile| {
            let bounds = Bounds {
                origin: gpui::point(gpui::px(16.), gpui::px(16.)),
                size: Size {
                    width: gpui::px(16.),
                    height: gpui::px(16.),
                },
            }
            .scale(1.0);
            PolychromeSprite {
                order: 0,
                pad: 0,
                grayscale: false.into(),
                opacity: 1.0,
                bounds,
                content_mask: ContentMask { bounds },
                corner_radii: Corners::default(),
                tile,
            }
        };

        // A batch from a stale paint whose texture the atlas has released
        // draws nothing, so it must not build the pipeline either.
        let mut stale_scene = Scene::default();
        stale_scene.insert_primitive(sprite(gpui::AtlasTile {
            texture_id: AtlasTextureId {
                index: 99,
                kind: gpui::AtlasTextureKind::Polychrome,
            },
            tile_id: gpui::TileId(0),
            padding: 0,
            bounds: Bounds::default(),
        }));
        stale_scene.finish();
        let image = renderer.render_scene_to_image(&stale_scene, device_size(32, 32))?;
        assert_pixel(&image, 24, 24, BLACK);
        assert_eq!(compiled(&renderer), [false, true, true, false]);

        // Polychrome uploads are BGRA; green reads the same either way.
        const GREEN: [u8; 4] = [0, 255, 0, 255];
        let tile_size = Size {
            width: DevicePixels(4),
            height: DevicePixels(4),
        };
        let tile = renderer
            .core
            .atlas
            .get_or_insert_with(
                gpui::AtlasKey::Image(gpui::RenderImageParams {
                    image_id: gpui::ImageId(1),
                    frame_index: 0,
                }),
                &mut || Ok(Some((tile_size, std::borrow::Cow::Owned(GREEN.repeat(16))))),
            )?
            .ok_or_else(|| anyhow::anyhow!("polychrome tile was not allocated"))?;
        let mut shadow_and_sprite_scene = Scene::default();
        let shadow_bounds = Bounds {
            origin: gpui::point(gpui::px(0.), gpui::px(16.)),
            size: Size {
                width: gpui::px(16.),
                height: gpui::px(16.),
            },
        }
        .scale(1.0);
        shadow_and_sprite_scene.insert_primitive(Shadow {
            order: 0,
            blur_radius: ScaledPixels(0.),
            bounds: shadow_bounds,
            corner_radii: Corners::default(),
            content_mask: ContentMask {
                bounds: shadow_bounds,
            },
            color: gpui::red(),
            element_bounds: shadow_bounds,
            element_corner_radii: Corners::default(),
            inset: 0,
            pad: 0,
        });
        shadow_and_sprite_scene.insert_primitive(sprite(tile));
        shadow_and_sprite_scene.finish();
        let image =
            renderer.render_scene_to_image(&shadow_and_sprite_scene, device_size(32, 32))?;
        assert_pixel(&image, 8, 24, RED);
        assert_pixel(&image, 24, 24, GREEN);
        assert_pixel(&image, 24, 8, BLACK);
        assert_eq!(compiled(&renderer), [true; 4]);
        Ok(())
    }

    /// A lens with curved, antialiased edges and a fractional origin, clipped
    /// to a 1100x700 window: three columns and two rows of path tiles.
    #[cfg(target_os = "linux")]
    fn lens_path(x: f32, y: f32, color: Background) -> gpui::Path<ScaledPixels> {
        let point_at = |offset_x: f32, offset_y: f32| {
            gpui::point(gpui::px(x + offset_x), gpui::px(y + offset_y))
        };
        let mut path = gpui::Path::new(point_at(0., 20.));
        path.line_to(point_at(20., 0.));
        path.curve_to(point_at(40., 20.), point_at(40., 0.));
        path.line_to(point_at(20., 40.));
        path.curve_to(point_at(0., 20.), point_at(0., 40.));
        path.content_mask = ContentMask {
            bounds: Bounds {
                origin: gpui::point(gpui::px(0.), gpui::px(0.)),
                size: Size {
                    width: gpui::px(TILED_WIDTH as f32),
                    height: gpui::px(TILED_HEIGHT as f32),
                },
            },
        };
        path.color = color;
        path.scale(1.0)
    }

    #[cfg(target_os = "linux")]
    const TILED_WIDTH: i32 = 1100;
    #[cfg(target_os = "linux")]
    const TILED_HEIGHT: i32 = 700;

    #[cfg(target_os = "linux")]
    fn render_lenses(
        renderer: &mut WgpuHeadlessRenderer,
        origins: &[(f32, f32)],
    ) -> anyhow::Result<image::RgbaImage> {
        let mut scene = Scene::default();
        for &(x, y) in origins {
            scene.insert_primitive(lens_path(x, y, gpui::red().into()));
        }
        scene.finish();
        renderer.render_scene_to_image(&scene, device_size(TILED_WIDTH, TILED_HEIGHT))
    }

    #[cfg(target_os = "linux")]
    fn assert_black_outside(image: &image::RgbaImage, boxes: &[(i64, i64)]) {
        for (x, y, pixel) in image.enumerate_pixels() {
            let inside = boxes.iter().any(|&(left, top)| {
                (left - 2..left + 43).contains(&(x as i64))
                    && (top - 2..top + 43).contains(&(y as i64))
            });
            if !inside {
                assert_eq!(pixel.0, BLACK, "stray path pixel at ({x}, {y})");
            }
        }
    }

    /// Paths rasterize tile by tile; a lens placed across tile seams, at the
    /// window's edges, or inside one tile must render the same pixels, and a
    /// tile left over from an earlier frame must never show through.
    #[cfg(target_os = "linux")]
    #[test]
    fn tiled_paths_render_the_same_across_tiles_and_edges() -> anyhow::Result<()> {
        let mut renderer = WgpuHeadlessRenderer::new()?;
        let (reference_x, reference_y) = (100, 100);
        let reference = render_lenses(
            &mut renderer,
            &[(reference_x as f32 + 0.3, reference_y as f32 + 0.6)],
        )?;
        assert_pixel(&reference, 120, 120, RED);
        assert_black_outside(&reference, &[(reference_x, reference_y)]);

        // Across the first column seam, the first corner, the last column's
        // seam, and clipped by each window edge.
        let placements = [(492, 100), (492, 492), (1004, 300), (1070, 660), (-15, 300), (600, -12)];
        let mut max_difference = 0;
        for (left, top) in placements {
            let image = render_lenses(&mut renderer, &[(left as f32 + 0.3, top as f32 + 0.6)])?;
            for offset_y in -2..43 {
                for offset_x in -2..43 {
                    let (x, y) = (left + offset_x, top + offset_y);
                    if !(0..TILED_WIDTH as i64).contains(&x) || !(0..TILED_HEIGHT as i64).contains(&y)
                    {
                        continue;
                    }
                    let actual = image.get_pixel(x as u32, y as u32).0;
                    let expected = reference
                        .get_pixel((reference_x + offset_x) as u32, (reference_y + offset_y) as u32)
                        .0;
                    for (actual, expected) in actual.iter().zip(expected) {
                        max_difference = max_difference.max(actual.abs_diff(expected));
                    }
                }
            }
            assert_black_outside(&image, &[(left, top)]);
        }
        // Each placement maps through a differently offset viewport; one level
        // of slack absorbs a driver whose rasterizer rounds that offset onto
        // the other side of a sample.
        assert!(
            max_difference <= 1,
            "lens pixels differ by {max_difference} between placements"
        );

        // A middle tile drawn in one frame, then a single mixed-order batch
        // whose spanning sprite covers it without drawing there.
        render_lenses(&mut renderer, &[(700.3, 200.6)])?;
        let image = render_lenses(&mut renderer, &[(100.3, 100.6), (1000.3, 600.6)])?;
        assert_pixel(&image, 120, 120, RED);
        assert_pixel(&image, 1020, 620, RED);
        assert_black_outside(&image, &[(100, 100), (1000, 600)]);

        // Two batches separated by a quad, in different tiles.
        let mut scene = Scene::default();
        scene.insert_primitive(lens_path(100.3, 100.6, gpui::red().into()));
        scene.insert_primitive(solid_quad(300.0, 600.0, 20.0, 20.0, gpui::blue()));
        scene.insert_primitive(lens_path(800.3, 400.6, gpui::red().into()));
        scene.finish();
        let image =
            renderer.render_scene_to_image(&scene, device_size(TILED_WIDTH, TILED_HEIGHT))?;
        assert_pixel(&image, 120, 120, RED);
        assert_pixel(&image, 820, 420, RED);
        assert_pixel(&image, 310, 610, BLUE);
        Ok(())
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn headless_renderer_rejects_invalid_sizes() -> anyhow::Result<()> {
        let mut renderer = WgpuHeadlessRenderer::new()?;
        let too_large = renderer.core.max_texture_size as i32 + 1;

        for size in [
            device_size(0, 8),
            device_size(8, 0),
            device_size(-1, 8),
            device_size(too_large, 8),
        ] {
            assert!(
                renderer.render_scene(&Scene::default(), size).is_err(),
                "{size:?} should be rejected"
            );
            assert!(
                renderer
                    .render_scene_to_image(&Scene::default(), size)
                    .is_err(),
                "{size:?} should be rejected"
            );
        }

        // Rejection must leave the renderer usable.
        let image = renderer.render_scene_to_image(&Scene::default(), device_size(4, 4))?;
        assert_eq!(image.dimensions(), (4, 4));
        Ok(())
    }

    #[test]
    fn instance_buffer_shrinks_after_a_window_of_low_usage() {
        const MIB: u64 = 1024 * 1024;
        let initial = 2 * MIB;
        let mut window = InstanceUsageWindow::default();

        // Usage at a quarter of the capacity or more never shrinks it.
        for _ in 0..INSTANCE_SHRINK_FRAMES * 2 {
            assert_eq!(window.record_frame(16 * MIB, 64 * MIB, initial), None);
        }

        // A frame back at a quarter restarts the window.
        for _ in 1..INSTANCE_SHRINK_FRAMES {
            assert_eq!(window.record_frame(3 * MIB, 64 * MIB, initial), None);
        }
        assert_eq!(window.record_frame(16 * MIB, 64 * MIB, initial), None);
        for _ in 1..INSTANCE_SHRINK_FRAMES {
            assert_eq!(window.record_frame(MIB, 64 * MIB, initial), None);
        }
        // Twice the window's peak (3 MiB), rounded up to a power of two.
        assert_eq!(
            window.record_frame(3 * MIB, 64 * MIB, initial),
            Some(8 * MIB)
        );

        // Never below the initial capacity, and no reallocation to the same size.
        for _ in 1..INSTANCE_SHRINK_FRAMES {
            assert_eq!(window.record_frame(1024, 16 * MIB, initial), None);
        }
        assert_eq!(window.record_frame(1024, 16 * MIB, initial), Some(initial));
        for _ in 0..INSTANCE_SHRINK_FRAMES * 2 {
            assert_eq!(window.record_frame(1024, initial, initial), None);
        }
    }

    #[test]
    fn webgl_shader_is_valid_wgsl_without_storage_buffers() {
        assert!(!WEBGL_SHADERS.contains("var<storage"));
        validate_wgsl(WEBGL_SHADERS, naga::valid::Capabilities::empty());
    }

    #[test]
    fn storage_buffer_shader_is_valid_wgsl() {
        validate_wgsl(STORAGE_BUFFER_SHADERS, naga::valid::Capabilities::empty());
    }

    #[test]
    fn subpixel_shader_is_valid_wgsl() {
        validate_wgsl(
            SUBPIXEL_SHADERS,
            naga::valid::Capabilities::DUAL_SOURCE_BLENDING,
        );
    }

    fn validate_wgsl(source: &str, capabilities: naga::valid::Capabilities) {
        let module = naga::front::wgsl::parse_str(source).expect("shader should parse");
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), capabilities)
            .validate(&module)
            .expect("shader should validate");
    }

    #[test]
    fn webgl_record_sizes_match_shader_word_strides() {
        assert_eq!(std::mem::size_of::<Quad>(), 40 * 4);
        assert_eq!(std::mem::size_of::<Shadow>(), 28 * 4);
        assert_eq!(std::mem::size_of::<PathRasterizationVertex>(), 26 * 4);
        assert_eq!(std::mem::size_of::<PathSprite>(), 4 * 4);
        assert_eq!(std::mem::size_of::<Underline>(), 16 * 4);
        assert_eq!(std::mem::size_of::<MonochromeSprite>(), 44 * 4);
        assert_eq!(std::mem::size_of::<SubpixelSprite>(), 44 * 4);
        assert_eq!(std::mem::size_of::<PolychromeSprite>(), 24 * 4);
    }

    #[test]
    fn webgl_quad_layout_matches_fixed_decoder() {
        let quad = Quad {
            order: 41,
            border_style: BorderStyle::Dashed,
            bounds: Bounds {
                origin: Point {
                    x: 1.0.into(),
                    y: 2.0.into(),
                },
                size: Size {
                    width: 3.0.into(),
                    height: 4.0.into(),
                },
            },
            content_mask: ContentMask {
                bounds: Bounds {
                    origin: Point {
                        x: 5.0.into(),
                        y: 6.0.into(),
                    },
                    size: Size {
                        width: 7.0.into(),
                        height: 8.0.into(),
                    },
                },
            },
            background: linear_gradient(
                11.0,
                linear_color_stop(
                    Hsla {
                        h: 12.0,
                        s: 13.0,
                        l: 14.0,
                        a: 15.0,
                    },
                    16.0,
                ),
                linear_color_stop(
                    Hsla {
                        h: 17.0,
                        s: 18.0,
                        l: 19.0,
                        a: 20.0,
                    },
                    21.0,
                ),
            )
            .color_space(ColorSpace::Oklab),
            border_color: Hsla {
                h: 22.0,
                s: 23.0,
                l: 24.0,
                a: 25.0,
            },
            corner_radii: Corners {
                top_left: 26.0.into(),
                top_right: 27.0.into(),
                bottom_right: 28.0.into(),
                bottom_left: 29.0.into(),
            },
            border_widths: Edges {
                top: 30.0.into(),
                right: 31.0.into(),
                bottom: 32.0.into(),
                left: 33.0.into(),
            },
        };

        let bytes = unsafe { WgpuRendererCore::instance_bytes(std::slice::from_ref(&quad)) };
        let words: &[u32] = bytemuck::cast_slice(bytes);
        assert_eq!(
            words,
            &[
                41,
                1,
                1.0_f32.to_bits(),
                2.0_f32.to_bits(),
                3.0_f32.to_bits(),
                4.0_f32.to_bits(),
                5.0_f32.to_bits(),
                6.0_f32.to_bits(),
                7.0_f32.to_bits(),
                8.0_f32.to_bits(),
                1,
                1,
                0,
                0,
                0,
                0,
                11.0_f32.to_bits(),
                12.0_f32.to_bits(),
                13.0_f32.to_bits(),
                14.0_f32.to_bits(),
                15.0_f32.to_bits(),
                16.0_f32.to_bits(),
                17.0_f32.to_bits(),
                18.0_f32.to_bits(),
                19.0_f32.to_bits(),
                20.0_f32.to_bits(),
                21.0_f32.to_bits(),
                0,
                22.0_f32.to_bits(),
                23.0_f32.to_bits(),
                24.0_f32.to_bits(),
                25.0_f32.to_bits(),
                26.0_f32.to_bits(),
                27.0_f32.to_bits(),
                28.0_f32.to_bits(),
                29.0_f32.to_bits(),
                30.0_f32.to_bits(),
                31.0_f32.to_bits(),
                32.0_f32.to_bits(),
                33.0_f32.to_bits(),
            ]
        );
    }

    /// The WGSL declarations are hand-written mirrors of the `#[repr(C)]`
    /// structs the renderer uploads, and nothing forces the two to agree.
    /// When they drift, wgpu either rejects the bind group ("bound with size N
    /// where the shader expects M", which invalidates the whole command buffer
    /// and blanks the window) or, for batches large enough to clear the
    /// minimum, silently strides through the instance buffer at the wrong
    /// pitch. WGSL's alignment rules are the trap: `vec3<u32>` aligns to 16
    /// where Rust's `[u32; 3]` aligns to 4, and one such field re-pads every
    /// struct that contains it.
    #[test]
    fn sprite_layouts_match_rust() {
        use naga::proc::Layouter;

        let module =
            naga::front::wgsl::parse_str(SUBPIXEL_SHADERS).expect("shaders parse");
        let mut layouter = Layouter::default();
        layouter
            .update(module.to_ctx())
            .expect("shader types have a layout");

        let expected: &[(&str, usize)] = &[
            ("Hsla", size_of::<gpui::Hsla>()),
            ("Background", size_of::<gpui::Background>()),
            ("AtlasTile", size_of::<gpui::AtlasTile>()),
            (
                "TransformationMatrix",
                size_of::<gpui::TransformationMatrix>(),
            ),
            ("SpriteEffect", size_of::<gpui::SpriteEffect>()),
            ("Quad", size_of::<Quad>()),
            ("Shadow", size_of::<Shadow>()),
            ("Underline", size_of::<Underline>()),
            ("MonochromeSprite", size_of::<MonochromeSprite>()),
            ("PolychromeSprite", size_of::<PolychromeSprite>()),
            ("SubpixelSprite", size_of::<SubpixelSprite>()),
        ];

        for (name, rust_size) in expected {
            let handle = module
                .types
                .iter()
                .find(|(_, ty)| ty.name.as_deref() == Some(name))
                .map(|(handle, _)| handle)
                .unwrap_or_else(|| panic!("no `{name}` struct in the WGSL modules"));
            let wgsl_size = layouter[handle].size as usize;
            assert_eq!(
                wgsl_size, *rust_size,
                "WGSL `{name}` is {wgsl_size} bytes but the Rust struct the \
                 renderer uploads is {rust_size}"
            );
        }
    }
}
