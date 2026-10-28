use crate::metal_atlas::MetalAtlas;
use etagere::BucketedAtlasAllocator;
use anyhow::{Context as _, Result};
use block2::RcBlock;
use core_graphics::geometry::CGSize;
use gpui::{
    AtlasTextureId, BackdropBlur, Background, Bounds, ContentMask, DevicePixels, PaintSurface, Path, Point,
    PresentBreakdown, PresentOutcome, PresentReport, PrimitiveBatch, RenderMemoryGauge,
    RenderMemoryLedger, ScaledPixels, Scene, Size, point, quad_depth, size,
};
#[cfg(any(test, feature = "bench-support", feature = "test-support"))]
use image::RgbaImage;
use objc2::runtime::AnyObject;

use core_graphics::color_space::{CGColorSpace, kCGColorSpaceDisplayP3};
use foreign_types::{ForeignType, ForeignTypeRef};
use metal::{
    CAMetalLayer, CommandQueue, MTLCompareFunction, MTLGPUFamily, MTLPixelFormat, MTLResourceOptions, NSRange,
    NSUInteger,
};
use objc2_core_foundation::CFRetained;
use objc2_core_video::{
    CVMetalTexture, CVMetalTextureCache, CVMetalTextureGetTexture, CVPixelBufferGetHeight,
    CVPixelBufferGetHeightOfPlane, CVPixelBufferGetPixelFormatType, CVPixelBufferGetWidth,
    CVPixelBufferGetWidthOfPlane, kCVPixelFormatType_420YpCbCr8BiPlanarFullRange, kCVReturnSuccess,
};
use parking_lot::Mutex;

use std::{
    cell::Cell,
    ffi::c_void,
    mem,
    mem::MaybeUninit,
    ops::Range,
    ptr, slice,
    sync::{
        Arc,
        atomic::{AtomicU32, Ordering},
    },
    time::Instant,
};

/// How many presents in a row may defer for want of a drawable before the
/// frame is dropped instead: about a second of display refreshes.
const MAX_CONSECUTIVE_DEFERRED_PRESENTS: u32 = 120;

// Exported to metal
pub(crate) type PointF = gpui::Point<f32>;

#[cfg(not(feature = "runtime_shaders"))]
const SHADERS_METALLIB: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/shaders.metallib"));
#[cfg(feature = "runtime_shaders")]
const SHADERS_SOURCE_FILE: &str = include_str!(concat!(env!("OUT_DIR"), "/stitched_shaders.metal"));
// Use 4x MSAA, all devices support it.
// https://developer.apple.com/documentation/metal/mtldevice/1433355-supportstexturesamplecount
const PATH_SAMPLE_COUNT: u32 = 4;
/// [`quad_depth`] spaces quads 1/65535 apart and the scene stops partitioning
/// beyond `MAX_DEPTH_PARTITIONED_QUADS`, so a 16-bit attachment resolves
/// every depth the prepass can write at half the memory of a 32-bit one
/// (2 bytes per drawable pixel).
const DEPTH_FORMAT: MTLPixelFormat = MTLPixelFormat::Depth16Unorm;
/// Metal requires the offset a buffer is bound at to be 256-byte aligned.
const INSTANCE_BUFFER_ALIGNMENT: usize = 256;
/// Frames without a path batch after which the drawable-sized path
/// intermediate is dropped. Two seconds at 60fps: a window whose paths
/// blink in and out (hover reveals, transient popups) keeps the texture;
/// one that stopped drawing paths gives back its 4 bytes per pixel.
const PATH_INTERMEDIATE_IDLE_FRAMES: u32 = 120;
/// A scratch pair that is more than this many times the area a blur needs
/// for [`BACKDROP_SCRATCH_SHRINK_AFTER_USES`] blurs in a row is replaced by
/// one sized to the need.
const BACKDROP_SCRATCH_OVERSIZE_FACTOR: u64 = 4;
const BACKDROP_SCRATCH_SHRINK_AFTER_USES: u32 = 120;
/// Scratch textures grow in steps of this many texels on each axis, so a
/// surface animating its size reuses one pair instead of allocating a pair
/// for every frame's size.
const BACKDROP_SCRATCH_GRANULE: u64 = 32;
/// Half floats, so the three passes a blur makes through the scratch do not
/// each round to eight bits; measured, eight-bit scratch doubled the share
/// of channels a step or more off an exact gaussian.
const BACKDROP_SCRATCH_FORMAT: MTLPixelFormat = MTLPixelFormat::RGBA16Float;
/// The fewest scratch texels a region's shorter side is downsampled to.
const BACKDROP_MIN_SCRATCH_EXTENT: u64 = 32;
/// The furthest a blur pass samples from its centre, in scratch texels: a
/// gaussian is sampled out to four standard deviations, and the downsample
/// keeps sigma near four texels, so this is never reached by a frosted
/// surface.
const BACKDROP_BLUR_MAX_RADIUS: u32 = 255;

const MAX_INSTANCE_BUFFER_SIZE: usize = 256 * 1024 * 1024;
// A drawable may be reused while Metal's completion callbacks are delayed.
// When targets are replaced, each unfinished submission can retain another full-size
// target, so the layer's three drawables do not bound resource retention.
const MAX_UNFINISHED_SUBMISSIONS: u32 = 3;

struct InFlightSubmission {
    unfinished: Arc<AtomicU32>,
}

impl InFlightSubmission {
    fn try_acquire(unfinished: &Arc<AtomicU32>) -> Option<Self> {
        unfinished
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |count| {
                (count < MAX_UNFINISHED_SUBMISSIONS).then_some(count + 1)
            })
            .ok()?;
        Some(Self {
            unfinished: Arc::clone(unfinished),
        })
    }
}

impl Drop for InFlightSubmission {
    fn drop(&mut self) {
        self.unfinished.fetch_sub(1, Ordering::Relaxed);
    }
}

pub type Context = Arc<Mutex<InstanceBufferPool>>;
pub type Renderer = MetalRenderer;

pub unsafe fn new_renderer(
    context: self::Context,
    _native_window: *mut c_void,
    _native_view: *mut c_void,
    _bounds: gpui::Size<f32>,
    transparent: bool,
    use_display_p3: bool,
) -> Renderer {
    MetalRenderer::new(context, transparent, use_display_p3)
}

pub fn new_overlay_renderer(context: self::Context, base: &Renderer) -> Renderer {
    base.new_sharing_atlas(context, true)
}

pub struct InstanceBufferPool {
    buffer_size: usize,
    buffers: Vec<metal::Buffer>,
    /// Frames reported in the current shrink window, and the most any one of
    /// them wrote. Drives the shrink in [`Self::record_frame_usage`].
    window_frames: u32,
    window_peak_bytes: usize,
    /// What the pool has reported into the process-wide render memory gauges:
    /// the buffers waiting here, refreshed as they come and go.
    render_memory: RenderMemoryLedger,
}

/// A pool buffer can never shrink below this; it is also the size every
/// window starts from.
const INSTANCE_BUFFER_FLOOR_SIZE: usize = 2 * 1024 * 1024;

/// How many consecutive low-use frames it takes to halve the buffer size.
/// Two seconds at 60fps: long enough that a burst of complex frames with
/// brief pauses never thrashes, short enough that one elaborate frame does
/// not set the allocation for the rest of the session.
const INSTANCE_BUFFER_SHRINK_AFTER_FRAMES: u32 = 120;

impl Default for InstanceBufferPool {
    fn default() -> Self {
        Self {
            buffer_size: INSTANCE_BUFFER_FLOOR_SIZE,
            buffers: Vec::new(),
            window_frames: 0,
            window_peak_bytes: 0,
            render_memory: RenderMemoryLedger::default(),
        }
    }
}

pub(crate) struct InstanceBuffer {
    metal_buffer: metal::Buffer,
    size: usize,
}

impl InstanceBufferPool {
    pub(crate) fn reset(&mut self, buffer_size: usize) {
        self.buffer_size = buffer_size;
        self.buffers.clear();
        self.publish_memory();
    }

    fn publish_memory(&mut self) {
        self.render_memory.publish(
            RenderMemoryGauge::InstanceBuffers,
            (self.buffers.len() * self.buffer_size) as u64,
        );
    }

    pub(crate) fn acquire(
        &mut self,
        device: &metal::Device,
        unified_memory: bool,
    ) -> InstanceBuffer {
        let buffer = self.buffers.pop().unwrap_or_else(|| {
            let options = if unified_memory {
                MTLResourceOptions::StorageModeShared
                    // Buffers are write only which can benefit from the combined cache
                    // https://developer.apple.com/documentation/metal/mtlresourceoptions/cpucachemodewritecombined
                    | MTLResourceOptions::CPUCacheModeWriteCombined
            } else {
                MTLResourceOptions::StorageModeManaged
            };

            device.new_buffer(self.buffer_size as u64, options)
        });
        self.publish_memory();
        InstanceBuffer {
            metal_buffer: buffer,
            size: self.buffer_size,
        }
    }

    pub(crate) fn release(&mut self, buffer: InstanceBuffer) {
        if buffer.size == self.buffer_size {
            self.buffers.push(buffer.metal_buffer);
            self.publish_memory();
        }
    }

    /// Notes how many instance bytes a finished frame actually wrote, and
    /// halves `buffer_size` once a whole window of frames has stayed under
    /// half of it.
    ///
    /// Growth is a ratchet without this: one elaborate frame doubles the
    /// buffer, and every frame after — plus every buffer the in-flight pool
    /// holds — pays that high-water size for the life of the window. The
    /// halving converges on the size the current workload needs (release()
    /// already drops returned buffers of the old size), and stops at the
    /// floor every window starts from. A frame that needs more space again
    /// grows exactly as before.
    ///
    /// The pool is shared by every renderer in the process and each reports
    /// its own frames here, so the window is judged by its peak across all of
    /// them: a small window animating at full rate cannot shrink the buffer
    /// out from under a large one that draws rarely, which made the large one
    /// regrow it — a fresh buffer and a split frame — every couple of seconds.
    pub(crate) fn record_frame_usage(&mut self, used_bytes: usize) {
        self.window_peak_bytes = self.window_peak_bytes.max(used_bytes);
        self.window_frames += 1;
        if self.window_frames < INSTANCE_BUFFER_SHRINK_AFTER_FRAMES {
            return;
        }
        let peak_bytes = self.window_peak_bytes;
        self.window_frames = 0;
        self.window_peak_bytes = 0;
        if self.buffer_size <= INSTANCE_BUFFER_FLOOR_SIZE
            || peak_bytes.saturating_mul(2) > self.buffer_size
        {
            return;
        }
        let buffer_size = (self.buffer_size / 2).max(INSTANCE_BUFFER_FLOOR_SIZE);
        log::info!("decreased instance buffer size to {buffer_size}");
        self.reset(buffer_size);
    }
}

pub struct MetalRenderer {
    device: metal::Device,
    layer: Option<metal::MetalLayer>,
    is_apple_gpu: bool,
    is_unified_memory: bool,
    presents_with_transaction: bool,
    /// Whether the layer renders in Display P3; an overlay plane's layer
    /// matches its window's.
    use_display_p3: bool,
    /// The size the window wants its drawables at, kept while
    /// [`Self::release_drawables`] has them shrunk.
    drawable_size: Size<DevicePixels>,
    drawables_released: bool,
    /// For headless rendering, tracks whether output should be opaque
    opaque: bool,
    command_queue: CommandQueue,
    unfinished_submissions: Arc<AtomicU32>,
    paths_rasterization_pipeline_state: metal::RenderPipelineState,
    path_sprites_pipeline_state: metal::RenderPipelineState,
    shadows_pipeline_state: metal::RenderPipelineState,
    quads_pipeline_state: metal::RenderPipelineState,
    // Same as `quads_pipeline_state` but preserves the destination alpha, used
    // for quads painted as glass content (see `Styled::glass`).
    quads_glass_pipeline_state: metal::RenderPipelineState,
    opaque_quads_pipeline_state: metal::RenderPipelineState,
    underlines_pipeline_state: metal::RenderPipelineState,
    monochrome_sprites_pipeline_state: metal::RenderPipelineState,
    polychrome_sprites_pipeline_state: metal::RenderPipelineState,
    surfaces_pipeline_state: metal::RenderPipelineState,
    /// `None` where the device cannot build a dual-source blend: frosted
    /// surfaces then paint their tint over an unblurred backdrop.
    backdrop_blurs_pipeline_state: Option<metal::RenderPipelineState>,
    backdrop_blur_downsample_pipeline_state: metal::RenderPipelineState,
    backdrop_blur_horizontal_pipeline_state: metal::RenderPipelineState,
    backdrop_blur_vertical_pipeline_state: metal::RenderPipelineState,
    /// The downsampled textures every backdrop blur of a frame passes
    /// through in turn, sized to the largest; see [`BackdropScratch`].
    backdrop_scratch: Option<BackdropScratch>,
    /// Frames drawn since the scratch was last used; once it reaches
    /// [`PATH_INTERMEDIATE_IDLE_FRAMES`] the scratch goes.
    backdrop_idle_frames: u32,
    /// Blurs in a row that needed less than a
    /// [`BACKDROP_SCRATCH_OVERSIZE_FACTOR`]th of the scratch.
    backdrop_oversized_uses: u32,
    /// Forces the downsample factor, for measuring each one's accuracy.
    #[cfg(test)]
    backdrop_downsample_override: Option<u32>,
    depth_write_state: metal::DepthStencilState,
    depth_test_state: metal::DepthStencilState,
    unit_vertices: metal::Buffer,
    #[allow(clippy::arc_with_non_send_sync)]
    instance_buffer_pool: Arc<Mutex<InstanceBufferPool>>,
    sprite_atlas: Arc<MetalAtlas>,
    core_video_texture_cache: CFRetained<CVMetalTextureCache>,
    depth_texture: Option<metal::Texture>,
    /// How many times `render_frame` found the intermediate textures sized
    /// for a stale drawable and recreated them; used to rate-limit the log.
    stale_texture_heals: u32,
    /// When GPU memory stats were last logged, while `FINCODE_GPU_STATS=1`.
    gpu_stats_last_log: Option<std::time::Instant>,
    /// Drawables handed to `present` whose presented handler has not run
    /// yet: queued behind the one on screen, or held by a compositor that is
    /// not showing the window. Shared with the handlers, which run on a
    /// Metal thread. A handler that never runs leaves it raised, so it is
    /// clamped when read ([`Self::read_drawables_in_flight`]) and starts
    /// afresh when the window is hidden ([`Self::forget_drawables_in_flight`])
    /// and with every new layer, since a layer is only ever made along with
    /// its renderer.
    drawables_in_flight: Arc<AtomicU32>,
    /// Set once `drawables_in_flight` has read above the layer's maximum,
    /// which only presented handlers that never ran can cause. Clamping
    /// removes only the excess over the maximum, so the count may go on
    /// overstating what is in flight; this stays set, and every report says
    /// so, until the count starts afresh.
    drawables_in_flight_overstated: bool,
    /// Presents deferred in a row because `nextDrawable` returned none; used
    /// to log a run of them once instead of once per display refresh, and to
    /// end a run that goes on too long (see `missing_drawable_outcome`).
    consecutive_deferred_presents: u32,
    path_intermediate_texture: Option<metal::Texture>,
    path_intermediate_msaa_texture: Option<metal::Texture>,
    /// Set by [`Self::release_intermediate_textures`] while the window is
    /// occluded: the intermediate textures stay dropped through drawable-size
    /// updates and are rebuilt by the next `render_frame` without being
    /// counted as a stale-texture heal.
    intermediate_textures_released: bool,
    /// Frames drawn since the path intermediate was last used; once it
    /// reaches [`PATH_INTERMEDIATE_IDLE_FRAMES`] the texture is dropped.
    path_intermediate_idle_frames: u32,
    /// An overlay renderer draws strictly after its window's base renderer
    /// and never concurrently with it, so it borrows the base renderer's
    /// intermediate textures for each draw ([`Self::take_intermediates`] /
    /// [`Self::lend_intermediates`]) instead of allocating its own copies,
    /// which would double the per-window drawable-sized memory.
    borrows_intermediates: bool,
    /// An overlay renderer shares the base renderer's atlas and draws in the
    /// same present, whose tiles the base draw has already marked (see
    /// [`Self::note_scene_tiles`]); it must not mark or age the atlas again.
    shares_atlas_frame: bool,
    /// A private depth texture for frames whose path batches did not all fit
    /// into the intermediate at once (see [`Self::plan_path_batches`]) and
    /// therefore split the main render pass; the regular depth texture is
    /// memoryless on Apple GPUs and cannot survive an encoder boundary.
    fallback_depth_texture: Option<metal::Texture>,
    fallback_depth_idle_frames: u32,
    /// The drawable presented last and the scene content it shows, so a
    /// present that only advances time animations can copy it and draw just
    /// what moved (see [`Scene::time_animation_damage`]).
    presented_frame: Option<PresentedFrame>,
    /// What the present being encoded redraws, reused across presents.
    damage_rects: Vec<Bounds<DevicePixels>>,
    /// Presents drawn whole and in part since the last `FINCODE_GPU_STATS`
    /// line.
    full_presents: u32,
    partial_presents: u32,
    /// The primitive kinds the render being encoded leaves out, as
    /// `ATTRIBUTED_*` bits; zero outside an attribution render.
    attribution_skips: u16,
    /// Set while an attribution render encodes, so it neither counts as a
    /// present nor ages the atlas.
    attributing: bool,
    /// `FINCODE_GPU_ATTRIBUTION=1`; see [`Self::maybe_attribute_gpu_time`].
    gpu_attribution: Option<GpuAttribution>,
    /// What this renderer has reported into the process-wide render memory
    /// gauges: the depth attachments it holds, refreshed by
    /// [`Self::publish_depth_memory`].
    render_memory: RenderMemoryLedger,
    path_sample_count: u32,
    /// Offscreen render target reused across `render_scene` calls when
    /// rendering headlessly without reading pixels back.
    #[cfg(any(test, feature = "bench-support", feature = "test-support"))]
    headless_render_target: Option<metal::Texture>,
}

/// The two textures a backdrop blur runs through, smaller than the region by
/// the blur's downsample factor on each axis: the region is box-averaged
/// into `primary`, blurred horizontally into `secondary` and vertically back
/// into `primary`, which the composite upsamples. One pair serves every blur
/// in a frame — Metal orders the passes that write and read it within and
/// across command buffers on the queue — so what a window holds for frosting
/// is one region's worth at a sixteenth or so of its pixels, however many
/// surfaces it frosts.
pub(crate) struct BackdropScratch {
    primary: metal::Texture,
    secondary: metal::Texture,
}

impl BackdropScratch {
    fn fits(&self, width: u64, height: u64) -> bool {
        self.primary.width() >= width && self.primary.height() >= height
    }

    fn texels(&self) -> u64 {
        self.primary.width() * self.primary.height()
    }

    fn allocated_bytes(&self) -> u64 {
        self.primary.allocated_size() + self.secondary.allocated_size()
    }
}

/// Per-pass parameters shared by the blur passes and the composite.
#[derive(Clone, Copy)]
#[repr(C)]
struct BackdropBlurPass {
    /// The region's top-left corner in the target, in device pixels.
    source_origin: PointF,
    /// The region's size, in device pixels.
    source_size: PointF,
    /// The part of the scratch textures the region occupies, in texels,
    /// pads included.
    scratch_size: PointF,
    /// Device pixels per scratch texel on each axis.
    downsample: f32,
    /// Paired taps on either side of the centre; the tap buffer holds one
    /// more, the centre.
    taps: u32,
}

/// How many device pixels one scratch texel stands for, for a gaussian of
/// `sigma` device pixels over a region whose shorter side is `extent`
/// pixels: the largest power of two that keeps sigma at four texels or more
/// and the region at [`BACKDROP_MIN_SCRATCH_EXTENT`] texels across, up to
/// eight. Box-averaging a block that small before a gaussian that wide
/// widens it by under a quarter of a percent; the floor on the extent keeps
/// the blocks at a small surface's edges, where the gaussian is clamped,
/// fine enough to follow the edge pixels it repeats.
fn backdrop_downsample(sigma: f32, extent: u64) -> u32 {
    let mut downsample = 1;
    while downsample < 8
        && sigma / (downsample * 2) as f32 >= 4.
        && extent / (downsample as u64 * 2) >= BACKDROP_MIN_SCRATCH_EXTENT
    {
        downsample *= 2;
    }
    downsample
}

/// The taps of a gaussian of `sigma` texels out to four standard deviations,
/// as (offset, weight): the centre first, then neighbouring texels paired
/// into one linear sample each, placed between them in proportion to their
/// weights, which is exact under linear filtering. The centre plus both sides
/// sums to one.
fn gaussian_taps(sigma: f32) -> Vec<[f32; 2]> {
    let sigma = sigma.max(0.01) as f64;
    let radius = ((sigma * 4.).ceil() as u32).clamp(1, BACKDROP_BLUR_MAX_RADIUS);
    let weight = |texel: u32| (-((texel * texel) as f64) / (2. * sigma * sigma)).exp();
    let total = weight(0) + 2. * (1..=radius).map(weight).sum::<f64>();
    let mut taps = vec![[0., (weight(0) / total) as f32]];
    let mut texel = 1;
    while texel <= radius {
        let near = weight(texel);
        let far = if texel < radius { weight(texel + 1) } else { 0. };
        let offset = (texel as f64 * near + (texel + 1) as f64 * far) / (near + far);
        taps.push([offset as f32, ((near + far) / total) as f32]);
        texel += 2;
    }
    taps
}

/// One path batch's region of the path intermediate for the current frame.
#[derive(Clone, Copy)]
struct PathBatchSlot {
    offset: Point<ScaledPixels>,
    scissor: metal::MTLScissorRect,
    empty: bool,
}

impl PathBatchSlot {
    const EMPTY: Self = Self {
        offset: Point {
            x: ScaledPixels(0.),
            y: ScaledPixels(0.),
        },
        scissor: metal::MTLScissorRect {
            x: 0,
            y: 0,
            width: 0,
            height: 0,
        },
        empty: true,
    };
}

/// The drawable-sized textures a frame renders through, moved between a
/// window's base and overlay renderer by [`MetalRenderer::take_intermediates`].
pub struct IntermediateTextures {
    depth: Option<metal::Texture>,
    path: Option<metal::Texture>,
    path_msaa: Option<metal::Texture>,
    path_idle_frames: u32,
    fallback_depth: Option<metal::Texture>,
    fallback_depth_idle_frames: u32,
    backdrop_scratch: Option<BackdropScratch>,
    backdrop_idle_frames: u32,
    backdrop_oversized_uses: u32,
}

#[repr(C)]
pub struct PathRasterizationVertex {
    pub xy_position: Point<ScaledPixels>,
    pub st_position: Point<f32>,
    pub color: Background,
    pub bounds: Bounds<ScaledPixels>,
    pub rounded_clip: u32,
    pub clip_padding: u32,
}

impl MetalRenderer {
    /// Creates a new MetalRenderer with a CAMetalLayer for window-based rendering.
    pub fn new(
        instance_buffer_pool: Arc<Mutex<InstanceBufferPool>>,
        transparent: bool,
        use_display_p3: bool,
    ) -> Self {
        let device = Self::create_device();
        // Support direct-to-display rendering if the window is not transparent
        // https://developer.apple.com/documentation/metal/managing-your-game-window-for-metal-in-macos
        let layer = Self::new_layer(&device, transparent, use_display_p3);

        let mut renderer = Self::new_internal(
            device,
            Some(layer),
            !transparent,
            instance_buffer_pool,
            None,
            None,
        );
        renderer.use_display_p3 = use_display_p3;
        renderer
    }

    /// Creates a renderer for a CAMetalLayer owned by a platform view.
    pub fn from_layer(
        instance_buffer_pool: Arc<Mutex<InstanceBufferPool>>,
        layer: &objc2_quartz_core::CAMetalLayer,
        transparent: bool,
    ) -> Self {
        let device = Self::create_device();
        // Both bindings describe the same class; to_owned retains the live layer.
        let layer = unsafe {
            metal::MetalLayerRef::from_ptr(ptr::from_ref(layer).cast_mut().cast::<CAMetalLayer>())
        }.to_owned();
        Self::configure_layer(&layer, &device, transparent, false);
        Self::new_internal(device, Some(layer), !transparent, instance_buffer_pool, None, None)
    }

    fn new_layer(device: &metal::Device, transparent: bool, use_display_p3: bool) -> metal::MetalLayer {
        let layer = metal::MetalLayer::new();
        Self::configure_layer(&layer, device, transparent, use_display_p3);
        layer
    }

    fn configure_layer(
        layer: &metal::MetalLayerRef,
        device: &metal::DeviceRef,
        transparent: bool,
        use_display_p3: bool,
    ) {
        layer.set_device(device);
        layer.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
        // metal-rs doesn't bind these setters, so view the same object through
        // objc2's typed CAMetalLayer binding.
        let objc2_layer: &objc2_quartz_core::CAMetalLayer = unsafe { &*layer.as_ptr().cast() };
        if use_display_p3 {
            // SAFETY: The constant is extern, from Core Graphics
            let display_p3_name = unsafe { kCGColorSpaceDisplayP3 };
            if let Some(display_p3_color_space) = CGColorSpace::create_with_name(display_p3_name) {
                // SAFETY: `layer` is a CAMetalLayer and the color space remains valid for the call.
                // 'metal-rs' does not expose CAMetalLayer's typed `colorspace` setter.
                // Could be replaced with a native binding via objc2-* in future
                unsafe {
                    let _: () = objc2::msg_send![
                        objc2_layer,
                        setColorspace: display_p3_color_space.as_ptr().cast::<c_void>()
                    ];
                }
            } else {
                log::error!("unable to create the Display P3 color space");
            }
        }
        // Support direct-to-display rendering if the window is not transparent
        // https://developer.apple.com/documentation/metal/managing-your-game-window-for-metal-in-macos
        layer.set_opaque(!transparent);
        // Core Animation's default three, deliberately.
        //
        // Two was tried, for one fewer drawable-sized IOSurface per window
        // (4 bytes per pixel, 29MB at retina full-screen) while the window is
        // painting, on the reasoning that the frame after a slow one could
        // wait on the previous present. It mattered, and not as a slow frame:
        // with `setAllowsNextDrawableTimeout: NO` below, a request made while
        // both drawables are in flight blocks the calling thread outright,
        // and the WindowServer does not hand one back promptly for a window
        // it is not compositing — occluded, just deactivated, or mid
        // app-switch. Sampled 2026-08-26 during app switching: 1195 of the
        // main thread's 8050 samples (~2.2s of 15s, 58% of every sample where
        // it was doing anything at all) sat in `nextDrawable` on
        // `semaphore_timedwait_trap`. Blocked there, the app cannot answer
        // AppKit's activation handshake, so macOS defers HID delivery
        // system-wide and the whole desktop freezes for about a second per
        // switch.
        //
        // The third drawable costs its IOSurface only while the window is
        // actually painting; the WindowServer purges idle ones. Apple
        // documents 2 and 3 as the only permitted values, so there is no
        // middle ground to take instead.
        layer.set_maximum_drawable_count(3);
        // Backdrop blurs sample the drawable they frost (the downsample pass
        // reads the region beneath a surface straight out of it), which Metal
        // allows only for a texture that is not framebuffer-only. Visual tests
        // and UI automation screenshots read it back too. A window that never
        // frosts pays the display path's lost optimisation as well; the app
        // frosts its composer in every window, so it would anyway.
        layer.set_framebuffer_only(false);
        objc2_layer.setAllowsNextDrawableTimeout(false);
        objc2_layer.setNeedsDisplayOnBoundsChange(true);
        // UIKit sizes a view's backing layer itself; only AppKit-hosted
        // layers need to track their superlayer's bounds.
        #[cfg(target_os = "macos")]
        objc2_layer.setAutoresizingMask(
            objc2_quartz_core::CAAutoresizingMask::LayerWidthSizable
                | objc2_quartz_core::CAAutoresizingMask::LayerHeightSizable,
        );
    }

    fn new_sharing_atlas(
        &self,
        instance_buffer_pool: Arc<Mutex<InstanceBufferPool>>,
        transparent: bool,
    ) -> Self {
        let device = self.device.clone();
        let layer = Self::new_layer(&device, transparent, self.use_display_p3);
        let mut renderer = Self::new_internal(
            device,
            Some(layer),
            !transparent,
            instance_buffer_pool,
            Some(self.sprite_atlas.clone()),
            Some(self.command_queue.clone()),
        );
        renderer.borrows_intermediates = true;
        renderer.shares_atlas_frame = true;
        renderer
    }

    /// Records the sprite tiles `scene` references as used in the frame about
    /// to be drawn, without advancing the atlas or retiring anything. A
    /// layered window calls this for its overlay scene before drawing the
    /// base scene, so the base draw's retirement has seen every tile the
    /// present as a whole still needs.
    pub fn note_scene_tiles(&self, scene: &Scene) {
        self.sprite_atlas.note_scene_tiles(scene);
    }

    /// Moves the drawable-sized intermediate textures out of this renderer so
    /// another renderer for the same drawable size can draw with them; pair
    /// with [`Self::lend_intermediates`] to hand them on and back. A frame
    /// that used them keeps its own references from its command buffer, so
    /// moving the handles never affects an in-flight draw.
    pub fn take_intermediates(&mut self) -> IntermediateTextures {
        let intermediates = IntermediateTextures {
            depth: self.depth_texture.take(),
            path: self.path_intermediate_texture.take(),
            path_msaa: self.path_intermediate_msaa_texture.take(),
            path_idle_frames: self.path_intermediate_idle_frames,
            fallback_depth: self.fallback_depth_texture.take(),
            fallback_depth_idle_frames: self.fallback_depth_idle_frames,
            backdrop_scratch: self.backdrop_scratch.take(),
            backdrop_idle_frames: self.backdrop_idle_frames,
            backdrop_oversized_uses: self.backdrop_oversized_uses,
        };
        self.publish_depth_memory();
        intermediates
    }

    /// Installs intermediate textures taken from another renderer; see
    /// [`Self::take_intermediates`].
    pub fn lend_intermediates(&mut self, intermediates: IntermediateTextures) {
        self.depth_texture = intermediates.depth;
        self.path_intermediate_texture = intermediates.path;
        self.path_intermediate_msaa_texture = intermediates.path_msaa;
        self.path_intermediate_idle_frames = intermediates.path_idle_frames;
        self.fallback_depth_texture = intermediates.fallback_depth;
        self.fallback_depth_idle_frames = intermediates.fallback_depth_idle_frames;
        self.backdrop_scratch = intermediates.backdrop_scratch;
        self.backdrop_idle_frames = intermediates.backdrop_idle_frames;
        self.backdrop_oversized_uses = intermediates.backdrop_oversized_uses;
        self.publish_depth_memory();
    }

    /// Reports the intermediate textures this renderer holds right now: the
    /// depth attachments, the path intermediates and the backdrop blur
    /// scratch. Called at the end of every frame and wherever the textures
    /// change hands or go away, so a hidden window's release and a
    /// base-to-overlay hand-off are seen without a draw. The gauges are
    /// device memory, so a memoryless attachment (tile memory on Apple GPUs)
    /// contributes nothing.
    fn publish_depth_memory(&mut self) {
        let depth = [&self.depth_texture, &self.fallback_depth_texture]
            .into_iter()
            .flatten()
            .map(|texture| texture.allocated_size())
            .sum();
        self.render_memory
            .publish(RenderMemoryGauge::DepthTextures, depth);
        let paths = [&self.path_intermediate_texture, &self.path_intermediate_msaa_texture]
            .into_iter()
            .flatten()
            .map(|texture| texture.allocated_size())
            .sum();
        self.render_memory
            .publish(RenderMemoryGauge::PathTextures, paths);
        let backdrop = self
            .backdrop_scratch
            .as_ref()
            .map_or(0, BackdropScratch::allocated_bytes);
        self.render_memory
            .publish(RenderMemoryGauge::BackdropTextures, backdrop);
        gpui::record_device_allocated_bytes(self.device.current_allocated_size());
    }

    /// Creates a new headless MetalRenderer for offscreen rendering without a window.
    ///
    /// This renderer can render scenes to images without requiring a CAMetalLayer,
    /// window, or AppKit. Use `render_scene_to_image()` to render scenes.
    #[cfg(any(test, feature = "bench-support", feature = "test-support"))]
    pub fn new_headless(instance_buffer_pool: Arc<Mutex<InstanceBufferPool>>) -> Self {
        let device = Self::create_device();
        Self::new_internal(device, None, true, instance_buffer_pool, None, None)
    }

    #[cfg(target_os = "macos")]
    fn create_device() -> metal::Device {
        // Prefer low‐power integrated GPUs on Intel Mac. On Apple
        // Silicon, there is only ever one GPU, so this is equivalent to
        // `metal::Device::system_default()`.
        if let Some(d) = metal::Device::all()
            .into_iter()
            .min_by_key(|d| (d.is_removable(), !d.is_low_power()))
        {
            d
        } else {
            // For some reason `all()` can return an empty list, see https://github.com/zed-industries/zed/issues/37689
            // In that case, we fall back to the system default device.
            log::error!(
                "Unable to enumerate Metal devices; attempting to use system default device"
            );
            Self::system_default_device()
        }
    }

    #[cfg(target_os = "ios")]
    fn create_device() -> metal::Device {
        Self::system_default_device()
    }

    fn system_default_device() -> metal::Device {
        metal::Device::system_default().unwrap_or_else(|| {
            log::error!("unable to access a compatible graphics device");
            std::process::exit(1);
        })
    }

    fn new_internal(
        device: metal::Device,
        layer: Option<metal::MetalLayer>,
        opaque: bool,
        instance_buffer_pool: Arc<Mutex<InstanceBufferPool>>,
        shared_sprite_atlas: Option<Arc<MetalAtlas>>,
        shared_command_queue: Option<CommandQueue>,
    ) -> Self {
        #[cfg(feature = "runtime_shaders")]
        let library = device
            .new_library_with_source(&SHADERS_SOURCE_FILE, &metal::CompileOptions::new())
            .expect("error building metal library");
        #[cfg(not(feature = "runtime_shaders"))]
        let library = device
            .new_library_with_data(SHADERS_METALLIB)
            .expect("error building metal library");

        fn to_float2_bits(point: PointF) -> u64 {
            let mut output = point.y.to_bits() as u64;
            output <<= 32;
            output |= point.x.to_bits() as u64;
            output
        }

        // Shared memory can be used only if CPU and GPU share the same memory space.
        // https://developer.apple.com/documentation/metal/setting-resource-storage-modes
        // iOS does not support managed resources. Its simulator may report a
        // non-unified host GPU even though resources must still use shared storage.
        let is_unified_memory = cfg!(target_os = "ios") || device.has_unified_memory();
        // Apple GPU families support memoryless textures, which can significantly reduce
        // memory usage by keeping render targets in on-chip tile memory instead of
        // allocating backing store in system memory.
        // https://developer.apple.com/documentation/metal/mtlgpufamily
        let is_apple_gpu = device.supports_family(MTLGPUFamily::Apple1);

        let unit_vertices = [
            to_float2_bits(point(0., 0.)),
            to_float2_bits(point(1., 0.)),
            to_float2_bits(point(0., 1.)),
            to_float2_bits(point(0., 1.)),
            to_float2_bits(point(1., 0.)),
            to_float2_bits(point(1., 1.)),
        ];
        let unit_vertices = device.new_buffer_with_data(
            unit_vertices.as_ptr() as *const c_void,
            mem::size_of_val(&unit_vertices) as u64,
            if is_unified_memory {
                MTLResourceOptions::StorageModeShared
                    | MTLResourceOptions::CPUCacheModeWriteCombined
            } else {
                MTLResourceOptions::StorageModeManaged
            },
        );

        let paths_rasterization_pipeline_state = build_path_rasterization_pipeline_state(
            &device,
            &library,
            "paths_rasterization",
            "path_rasterization_vertex",
            "path_rasterization_fragment",
            MTLPixelFormat::BGRA8Unorm,
            PATH_SAMPLE_COUNT,
        );
        let path_sprites_pipeline_state = build_path_sprite_pipeline_state(
            &device,
            &library,
            "path_sprites",
            "path_sprite_vertex",
            "path_sprite_fragment",
            MTLPixelFormat::BGRA8Unorm,
        );
        let shadows_pipeline_state = build_pipeline_state(
            &device,
            &library,
            "shadows",
            "shadow_vertex",
            "shadow_fragment",
            MTLPixelFormat::BGRA8Unorm,
        );
        let quads_pipeline_state = build_pipeline_state(
            &device,
            &library,
            "quads",
            "quad_vertex",
            "quad_fragment",
            MTLPixelFormat::BGRA8Unorm,
        );
        let quads_glass_pipeline_state = build_pipeline_state_with_blend(
            &device,
            &library,
            "quads_glass",
            "quad_vertex",
            "quad_fragment",
            MTLPixelFormat::BGRA8Unorm,
            metal::MTLBlendFactor::Zero,
            metal::MTLBlendFactor::One,
        );
        let opaque_quads_pipeline_state = build_opaque_quads_pipeline_state(
            &device,
            &library,
            "opaque_quads",
            "opaque_quad_vertex",
            "opaque_quad_fragment",
            MTLPixelFormat::BGRA8Unorm,
        );
        let underlines_pipeline_state = build_pipeline_state(
            &device,
            &library,
            "underlines",
            "underline_vertex",
            "underline_fragment",
            MTLPixelFormat::BGRA8Unorm,
        );
        let monochrome_sprites_pipeline_state = build_pipeline_state(
            &device,
            &library,
            "monochrome_sprites",
            "monochrome_sprite_vertex",
            "monochrome_sprite_fragment",
            MTLPixelFormat::BGRA8Unorm,
        );
        let polychrome_sprites_pipeline_state = build_pipeline_state(
            &device,
            &library,
            "polychrome_sprites",
            "polychrome_sprite_vertex",
            "polychrome_sprite_fragment",
            MTLPixelFormat::BGRA8Unorm,
        );
        let surfaces_pipeline_state = build_pipeline_state(
            &device,
            &library,
            "surfaces",
            "surface_vertex",
            "surface_fragment",
            MTLPixelFormat::BGRA8Unorm,
        );
        let backdrop_blurs_pipeline_state = build_backdrop_blur_pipeline_state(
            &device,
            &library,
            MTLPixelFormat::BGRA8Unorm,
        );
        let backdrop_blur_downsample_pipeline_state = build_backdrop_blur_pass_pipeline_state(
            &device,
            &library,
            "backdrop_blur_downsample",
            "backdrop_blur_downsample_fragment",
            BACKDROP_SCRATCH_FORMAT,
        );
        let backdrop_blur_horizontal_pipeline_state = build_backdrop_blur_pass_pipeline_state(
            &device,
            &library,
            "backdrop_blur_horizontal",
            "backdrop_blur_horizontal_fragment",
            BACKDROP_SCRATCH_FORMAT,
        );
        let backdrop_blur_vertical_pipeline_state = build_backdrop_blur_pass_pipeline_state(
            &device,
            &library,
            "backdrop_blur_vertical",
            "backdrop_blur_vertical_fragment",
            BACKDROP_SCRATCH_FORMAT,
        );

        let depth_write_state = build_depth_stencil_state(&device, true);
        let depth_test_state = build_depth_stencil_state(&device, false);

        // A renderer that borrows another's intermediate textures must also
        // queue behind its command buffers: Metal orders command buffers
        // within one queue, not across queues.
        let command_queue = shared_command_queue.unwrap_or_else(|| device.new_command_queue());
        let sprite_atlas = shared_sprite_atlas
            .unwrap_or_else(|| Arc::new(MetalAtlas::new(device.clone(), cfg!(target_os = "ios") || is_apple_gpu)));
        // metal and objc2-metal expose the same Objective-C device; the borrowed
        // protocol object remains valid while `device` owns it.
        let core_video_texture_cache = unsafe {
            let mut cache = ptr::null_mut();
            let status = CVMetalTextureCache::create(
                None,
                None,
                &*device
                    .as_ptr()
                    .cast::<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLDevice>>(),
                None,
                ptr::NonNull::from(&mut cache),
            );
            assert_eq!(
                status, kCVReturnSuccess,
                "failed to create CoreVideo Metal texture cache"
            );
            CFRetained::from_raw(ptr::NonNull::new(cache).expect("CoreVideo returned a null cache"))
        };

        Self {
            device,
            layer,
            presents_with_transaction: false,
            use_display_p3: false,
            drawable_size: Size::default(),
            drawables_released: false,
            is_apple_gpu,
            is_unified_memory,
            opaque,
            command_queue,
            unfinished_submissions: Arc::new(AtomicU32::new(0)),
            paths_rasterization_pipeline_state,
            path_sprites_pipeline_state,
            shadows_pipeline_state,
            quads_pipeline_state,
            quads_glass_pipeline_state,
            opaque_quads_pipeline_state,
            underlines_pipeline_state,
            monochrome_sprites_pipeline_state,
            polychrome_sprites_pipeline_state,
            surfaces_pipeline_state,
            backdrop_blurs_pipeline_state,
            backdrop_blur_downsample_pipeline_state,
            backdrop_blur_horizontal_pipeline_state,
            backdrop_blur_vertical_pipeline_state,
            backdrop_scratch: None,
            backdrop_idle_frames: 0,
            backdrop_oversized_uses: 0,
            #[cfg(test)]
            backdrop_downsample_override: None,
            depth_write_state,
            depth_test_state,
            unit_vertices,
            instance_buffer_pool,
            sprite_atlas,
            core_video_texture_cache,
            depth_texture: None,
            stale_texture_heals: 0,
            intermediate_textures_released: false,
            path_intermediate_idle_frames: 0,
            borrows_intermediates: false,
            shares_atlas_frame: false,
            fallback_depth_texture: None,
            fallback_depth_idle_frames: 0,
            presented_frame: None,
            damage_rects: Vec::new(),
            full_presents: 0,
            partial_presents: 0,
            attribution_skips: 0,
            attributing: false,
            gpu_attribution: GpuAttribution::from_env(),
            render_memory: RenderMemoryLedger::default(),
            gpu_stats_last_log: None,
            drawables_in_flight: Arc::new(AtomicU32::new(0)),
            drawables_in_flight_overstated: false,
            consecutive_deferred_presents: 0,
            path_intermediate_texture: None,
            path_intermediate_msaa_texture: None,
            path_sample_count: PATH_SAMPLE_COUNT,
            #[cfg(any(test, feature = "bench-support", feature = "test-support"))]
            headless_render_target: None,
        }
    }

    pub fn layer(&self) -> Option<&metal::MetalLayerRef> {
        self.layer.as_ref().map(|l| l.as_ref())
    }

    pub fn layer_ptr(&self) -> *mut CAMetalLayer {
        self.layer
            .as_ref()
            .map(|l| l.as_ptr())
            .unwrap_or(ptr::null_mut())
    }

    pub fn sprite_atlas(&self) -> &Arc<MetalAtlas> {
        &self.sprite_atlas
    }

    pub fn set_presents_with_transaction(&mut self, presents_with_transaction: bool) {
        self.presents_with_transaction = presents_with_transaction;
        if let Some(layer) = &self.layer {
            layer.set_presents_with_transaction(presents_with_transaction);
        }
    }

    pub fn update_drawable_size(&mut self, size: Size<DevicePixels>) {
        self.drawable_size = size;
        self.presented_frame = None;
        if let Some(layer) = &self.layer
            && !self.drawables_released
        {
            layer.set_drawable_size(CGSize::new(size.width.0 as f64, size.height.0 as f64));
        }
        self.update_intermediate_textures(size);
    }

    /// Shrinks the layer's drawables to a single pixel while the window is
    /// not visible. The layer keeps a pool of window-sized drawables (three at
    /// most, 29 MB each for a full-screen retina window) for as long as their
    /// size stands, whether or not anything presents; a new size retires the
    /// pool. What the window last showed stays as the layer's contents, so
    /// that one drawable remains and the window comes back showing it.
    /// Measured on a window hidden with Cmd-H and on one minimized to the
    /// Dock: two drawables went to one, and back to two once shown.
    /// [`Self::restore_drawables`] brings the size back before the next
    /// frame.
    pub fn release_drawables(&mut self) {
        self.presented_frame = None;
        if let Some(layer) = &self.layer {
            self.drawables_released = true;
            layer.set_drawable_size(CGSize::new(1., 1.));
        }
    }

    /// Undoes [`Self::release_drawables`] at the size the window has now.
    pub fn restore_drawables(&mut self) {
        if !self.drawables_released {
            return;
        }
        self.drawables_released = false;
        if let Some(layer) = &self.layer {
            let size = self.drawable_size;
            layer.set_drawable_size(CGSize::new(size.width.0 as f64, size.height.0 as f64));
        }
    }

    /// Drops the depth and path intermediate textures while the window is not
    /// visible; the next `render_frame` recreates them at the drawable's size.
    ///
    /// These are private, drawable-sized textures — tens of MB per window at
    /// retina full-screen — and unlike the layer's drawables, which the window
    /// server purges once a window stops painting, nothing reclaims them
    /// while the window sits minimized, on another Space, or behind another
    /// app. Rebuilding is a fraction of a millisecond, well inside a frame.
    ///
    /// The instance buffer pool is process-wide and shared with visible
    /// windows, so it is deliberately left alone.
    pub fn release_intermediate_textures(&mut self) {
        // A window the compositor is not showing may not keep what it last
        // showed, so the next present must not copy from it.
        self.presented_frame = None;
        if self.borrows_intermediates {
            return;
        }
        self.intermediate_textures_released = true;
        self.depth_texture = None;
        self.path_intermediate_texture = None;
        self.path_intermediate_msaa_texture = None;
        self.fallback_depth_texture = None;
        self.backdrop_scratch = None;
        self.publish_depth_memory();
    }

    fn update_intermediate_textures(&mut self, size: Size<DevicePixels>) {
        // A resize while occluded (display change, restore geometry) must not
        // rebuild what release_intermediate_textures dropped; the first draw
        // after the window is visible again does that.
        if self.intermediate_textures_released || self.borrows_intermediates {
            return;
        }
        self.recreate_intermediate_textures(size);
    }

    /// The creation half of [`Self::update_intermediate_textures`], without
    /// its guards. The render-time heal calls this on a borrowing overlay
    /// renderer as well: a lent texture that does not match the drawable is
    /// exactly the case the heal exists for, and the fresh textures travel
    /// back to the base renderer with the rest when the draw hands them over.
    fn recreate_intermediate_textures(&mut self, size: Size<DevicePixels>) {
        // We are uncertain when this happens, but sometimes size can be 0 here. Most likely before
        // the layout pass on window creation. Zero-sized texture creation causes SIGABRT.
        // https://github.com/zed-industries/zed/issues/36229
        if size.width.0 <= 0 || size.height.0 <= 0 {
            self.depth_texture = None;
            self.path_intermediate_texture = None;
            self.path_intermediate_msaa_texture = None;
            self.fallback_depth_texture = None;
            return;
        }

        // A depth texture of this size is kept: the headless path calls this
        // per render, and a fresh allocation per frame is pure churn.
        let depth_matches = self.depth_texture.as_ref().is_some_and(|texture| {
            texture.width() == size.width.0 as u64 && texture.height() == size.height.0 as u64
        });
        if depth_matches {
            if !self.path_intermediate_matches(size) {
                self.path_intermediate_texture = None;
                self.path_intermediate_msaa_texture = None;
            }
            return;
        }

        let depth_descriptor = metal::TextureDescriptor::new();
        depth_descriptor.set_width(size.width.0 as u64);
        depth_descriptor.set_height(size.height.0 as u64);
        depth_descriptor.set_pixel_format(DEPTH_FORMAT);
        // All path batches are rasterized before the main render pass, so the
        // depth attachment lives inside a single encoder and can be tile
        // memory on Apple GPUs: no system memory at all. Frames that cannot
        // be planned that way use `fallback_depth_texture` instead.
        depth_descriptor.set_storage_mode(if self.is_apple_gpu {
            metal::MTLStorageMode::Memoryless
        } else {
            metal::MTLStorageMode::Private
        });
        depth_descriptor.set_usage(metal::MTLTextureUsage::RenderTarget);
        self.depth_texture = Some(self.device.new_texture(&depth_descriptor));

        // The path intermediate is only needed by frames that contain a path
        // batch, so it is not recreated here: one sized for another drawable
        // is dropped and the next path batch allocates it on demand.
        if !self.path_intermediate_matches(size) {
            self.path_intermediate_texture = None;
            self.path_intermediate_msaa_texture = None;
        }
    }

    fn path_intermediate_matches(&self, size: Size<DevicePixels>) -> bool {
        self.path_intermediate_texture.as_ref().is_some_and(|texture| {
            texture.width() == size.width.0 as u64 && texture.height() == size.height.0 as u64
        })
    }

    /// Returns the drawable-sized path intermediate, creating it on first use
    /// and whenever the drawable size no longer matches.
    fn ensure_path_intermediate_textures(
        &mut self,
        size: Size<DevicePixels>,
    ) -> Result<metal::Texture> {
        self.path_intermediate_idle_frames = 0;
        if !self.path_intermediate_matches(size) {
            anyhow::ensure!(
                size.width.0 > 0 && size.height.0 > 0,
                "path intermediate requested for an empty {}x{} drawable",
                size.width.0,
                size.height.0
            );
            self.create_path_intermediate_textures(size);
        }
        self.path_intermediate_texture
            .clone()
            .context("missing path intermediate texture")
    }

    /// Counts a drawn frame against the path intermediate and drops it once
    /// enough frames have gone by without a path batch. The command buffer
    /// of a frame that used it holds its own reference, so dropping here
    /// never pulls a texture out from under an in-flight draw.
    fn retire_idle_path_intermediate(&mut self) {
        // Borrowed textures are aged by their owner once per present.
        if self.borrows_intermediates {
            return;
        }
        if self.path_intermediate_texture.is_some() {
            self.path_intermediate_idle_frames =
                self.path_intermediate_idle_frames.saturating_add(1);
            if self.path_intermediate_idle_frames > PATH_INTERMEDIATE_IDLE_FRAMES {
                self.path_intermediate_texture = None;
                self.path_intermediate_msaa_texture = None;
            }
        }
        if self.fallback_depth_texture.is_some() {
            self.fallback_depth_idle_frames = self.fallback_depth_idle_frames.saturating_add(1);
            if self.fallback_depth_idle_frames > PATH_INTERMEDIATE_IDLE_FRAMES {
                self.fallback_depth_texture = None;
            }
        }
    }

    /// Counts a drawn frame against the backdrop blur scratch and drops it
    /// once enough frames have gone by without a blur in either of the
    /// window's planes. In-flight command buffers hold their own references.
    fn retire_idle_backdrop_scratch(&mut self) {
        // Borrowed scratch is aged by its owner once per present; a blur on
        // the overlay plane resets the count it hands back.
        if self.borrows_intermediates || self.backdrop_scratch.is_none() {
            return;
        }
        self.backdrop_idle_frames = self.backdrop_idle_frames.saturating_add(1);
        if self.backdrop_idle_frames > PATH_INTERMEDIATE_IDLE_FRAMES {
            self.backdrop_scratch = None;
            self.backdrop_oversized_uses = 0;
        }
    }

    /// The scratch for a blur needing `width` × `height` texels: the one held
    /// when it is large enough, else a new one large enough for this blur and
    /// for everything the old one covered. One held long after the surfaces
    /// it was grown for have gone is replaced by one sized to the need.
    fn backdrop_scratch(&mut self, width: u64, height: u64) -> (metal::Texture, metal::Texture) {
        self.backdrop_idle_frames = 0;
        let needed = width * height;
        let reusable = match &self.backdrop_scratch {
            Some(scratch) if scratch.fits(width, height) => {
                if needed * BACKDROP_SCRATCH_OVERSIZE_FACTOR < scratch.texels() {
                    self.backdrop_oversized_uses = self.backdrop_oversized_uses.saturating_add(1);
                } else {
                    self.backdrop_oversized_uses = 0;
                }
                self.backdrop_oversized_uses <= BACKDROP_SCRATCH_SHRINK_AFTER_USES
            }
            _ => false,
        };
        if !reusable {
            let (mut grown_width, mut grown_height) = (width, height);
            if self.backdrop_oversized_uses <= BACKDROP_SCRATCH_SHRINK_AFTER_USES
                && let Some(scratch) = &self.backdrop_scratch
            {
                grown_width = grown_width.max(scratch.primary.width());
                grown_height = grown_height.max(scratch.primary.height());
            }
            let granule = |extent: u64| extent.div_ceil(BACKDROP_SCRATCH_GRANULE) * BACKDROP_SCRATCH_GRANULE;
            let descriptor = metal::TextureDescriptor::new();
            descriptor.set_width(granule(grown_width));
            descriptor.set_height(granule(grown_height));
            descriptor.set_pixel_format(BACKDROP_SCRATCH_FORMAT);
            descriptor.set_storage_mode(metal::MTLStorageMode::Private);
            descriptor.set_usage(
                metal::MTLTextureUsage::ShaderRead | metal::MTLTextureUsage::RenderTarget,
            );
            self.backdrop_scratch = Some(BackdropScratch {
                primary: self.device.new_texture(&descriptor),
                secondary: self.device.new_texture(&descriptor),
            });
            self.backdrop_oversized_uses = 0;
            self.publish_depth_memory();
        }
        let scratch = self
            .backdrop_scratch
            .as_ref()
            .expect("the scratch was just ensured");
        (scratch.primary.clone(), scratch.secondary.clone())
    }

    /// Runs `blur` between two batches: ends `command_encoder`, blurs the
    /// region beneath the blur's rect out of `texture` into the scratch (a
    /// downsample, a horizontal and a vertical pass), and returns a resumed encoder that has
    /// composited the result inside the blur's rounded rect. A blur that is
    /// entirely clipped keeps the encoder it was given.
    fn draw_backdrop_blur<'a>(
        &mut self,
        blur: &BackdropBlur,
        command_buffer: &'a metal::CommandBufferRef,
        command_encoder: &'a metal::RenderCommandEncoderRef,
        texture: &'a metal::TextureRef,
        depth_texture: &metal::TextureRef,
        store_depth: bool,
        rounded_clips: &BufferRegion,
        writer: &mut InstanceBufferWriter,
        viewport_size: Size<DevicePixels>,
        depth: f64,
    ) -> Result<&'a metal::RenderCommandEncoderRef> {
        let sigma = blur.blur_radius.0.max(0.5);
        // Only what lies under the surface is sampled, and the gaussian clamps
        // at the region's edges. Sampling the surroundings too would carry
        // whatever borders the surface (a line of text, a lighter panel) in
        // across its edge as a pale band just inside the outline.
        let visible = blur.bounds.intersect(&blur.content_mask.bounds);
        let texture_width = texture.width() as i64;
        let texture_height = texture.height() as i64;
        let left = (visible.origin.x.0.floor() as i64).max(0);
        let top = (visible.origin.y.0.floor() as i64).max(0);
        let right = ((visible.origin.x.0 + visible.size.width.0).ceil() as i64).min(texture_width);
        let bottom =
            ((visible.origin.y.0 + visible.size.height.0).ceil() as i64).min(texture_height);
        if right <= left || bottom <= top {
            return Ok(command_encoder);
        }
        let Some(composite) = self.backdrop_blurs_pipeline_state.clone() else {
            return Ok(command_encoder);
        };
        let composite = &composite;
        let width = (right - left) as u64;
        let height = (bottom - top) as u64;
        #[cfg(test)]
        let downsample = self
            .backdrop_downsample_override
            .unwrap_or_else(|| backdrop_downsample(sigma, width.min(height)));
        #[cfg(not(test))]
        let downsample = backdrop_downsample(sigma, width.min(height));
        // A block per `downsample` pixels, plus a pad of edge pixels each side.
        let scratch_width = width.div_ceil(downsample as u64) + 2;
        let scratch_height = height.div_ceil(downsample as u64) + 2;
        let (primary, secondary) = self.backdrop_scratch(scratch_width, scratch_height);

        let taps = gaussian_taps(sigma / downsample as f32);
        let pass = BackdropBlurPass {
            source_origin: point(left as f32, top as f32),
            source_size: point(width as f32, height as f32),
            scratch_size: point(scratch_width as f32, scratch_height as f32),
            downsample: downsample as f32,
            taps: taps.len() as u32 - 1,
        };

        command_encoder.end_encoding();
        for (pipeline_state, source, target) in [
            (&self.backdrop_blur_downsample_pipeline_state, texture, &*primary),
            (&self.backdrop_blur_horizontal_pipeline_state, &*primary, &*secondary),
            (&self.backdrop_blur_vertical_pipeline_state, &*secondary, &*primary),
        ] {
            let descriptor = metal::RenderPassDescriptor::new();
            let attachment = descriptor
                .color_attachments()
                .object_at(0)
                .context("a render pass has a first color attachment")?;
            attachment.set_texture(Some(target));
            attachment.set_load_action(metal::MTLLoadAction::DontCare);
            attachment.set_store_action(metal::MTLStoreAction::Store);
            let encoder = command_buffer.new_render_command_encoder(descriptor);
            encoder.set_viewport(metal::MTLViewport {
                originX: 0.,
                originY: 0.,
                width: scratch_width as f64,
                height: scratch_height as f64,
                znear: 0.,
                zfar: 1.,
            });
            encoder.set_render_pipeline_state(pipeline_state);
            encoder.set_vertex_buffer(
                BackdropBlurPassInputIndex::Vertices as u64,
                Some(&self.unit_vertices),
                0,
            );
            encoder.set_fragment_bytes(
                BackdropBlurPassInputIndex::Pass as u64,
                mem::size_of::<BackdropBlurPass>() as u64,
                &pass as *const BackdropBlurPass as *const _,
            );
            encoder.set_fragment_bytes(
                BackdropBlurPassInputIndex::Taps as u64,
                mem::size_of_val(taps.as_slice()) as u64,
                taps.as_ptr() as *const _,
            );
            encoder.set_fragment_texture(BackdropBlurPassInputIndex::Source as u64, Some(source));
            encoder.draw_primitives(metal::MTLPrimitiveType::Triangle, 0, 6);
            encoder.end_encoding();
        }

        let command_encoder = new_command_encoder_for_texture(
            command_buffer,
            texture,
            depth_texture,
            store_depth,
            viewport_size,
            None,
        );
        command_encoder.set_fragment_buffer(15, Some(&rounded_clips.buffer), rounded_clips.offset as u64);
        command_encoder.set_depth_stencil_state(&self.depth_test_state);
        set_viewport(command_encoder, viewport_size, depth, depth);

        let blurs = match writer.write(slice::from_ref(blur)) {
            Ok(blurs) => blurs,
            Err(error) => {
                command_encoder.end_encoding();
                return Err(error);
            }
        };
        command_encoder.set_render_pipeline_state(composite);
        command_encoder.set_vertex_buffer(
            BackdropBlurInputIndex::Vertices as u64,
            Some(&self.unit_vertices),
            0,
        );
        command_encoder.set_vertex_buffer(
            BackdropBlurInputIndex::Blurs as u64,
            Some(&blurs.buffer),
            blurs.offset as u64,
        );
        command_encoder.set_vertex_bytes(
            BackdropBlurInputIndex::ViewportSize as u64,
            mem::size_of_val(&viewport_size) as u64,
            &viewport_size as *const Size<DevicePixels> as *const _,
        );
        command_encoder.set_fragment_buffer(
            BackdropBlurInputIndex::Blurs as u64,
            Some(&blurs.buffer),
            blurs.offset as u64,
        );
        command_encoder.set_fragment_bytes(
            BackdropBlurInputIndex::Pass as u64,
            mem::size_of::<BackdropBlurPass>() as u64,
            &pass as *const BackdropBlurPass as *const _,
        );
        command_encoder
            .set_fragment_texture(BackdropBlurInputIndex::Blurred as u64, Some(&primary));
        command_encoder.draw_primitives(metal::MTLPrimitiveType::Triangle, 0, 6);
        Ok(command_encoder)
    }

    /// A private, drawable-sized depth texture for a frame whose main render
    /// pass has to be split around path batches; created on demand and
    /// retired like the path intermediate.
    fn ensure_fallback_depth_texture(&mut self, size: Size<DevicePixels>) -> metal::Texture {
        self.fallback_depth_idle_frames = 0;
        let matches = self.fallback_depth_texture.as_ref().is_some_and(|texture| {
            texture.width() == size.width.0 as u64 && texture.height() == size.height.0 as u64
        });
        if !matches {
            let descriptor = metal::TextureDescriptor::new();
            descriptor.set_width(size.width.0 as u64);
            descriptor.set_height(size.height.0 as u64);
            descriptor.set_pixel_format(DEPTH_FORMAT);
            descriptor.set_storage_mode(metal::MTLStorageMode::Private);
            descriptor.set_usage(metal::MTLTextureUsage::RenderTarget);
            self.fallback_depth_texture = Some(self.device.new_texture(&descriptor));
            log::debug!(
                "[renderer] path batches did not fit the {}x{} intermediate; split pass through a private depth texture",
                size.width.0,
                size.height.0
            );
        }
        self.fallback_depth_texture
            .clone()
            .expect("fallback depth texture was just ensured")
    }

    /// Assigns every path batch of the frame its own region of the path
    /// intermediate so all of them can be rasterized in one pass ahead of
    /// the main render pass. Returns `None` when they do not all fit, in
    /// which case the frame rasterizes each batch just before it is drawn,
    /// splitting the main pass around it.
    fn plan_path_batches(
        scene: &Scene,
        viewport_size: Size<DevicePixels>,
    ) -> Option<Vec<PathBatchSlot>> {
        let viewport_width = viewport_size.width.0.max(0);
        let viewport_height = viewport_size.height.0.max(0);
        let mut slots = Vec::new();
        let mut allocator = None;
        for batch in scene.batches() {
            let PrimitiveBatch::Paths(range) = batch else {
                continue;
            };
            let paths = &scene.paths[range];
            let Some(first_path) = paths.first() else {
                slots.push(PathBatchSlot::EMPTY);
                continue;
            };
            let mut bounds = first_path.clipped_bounds();
            for path in paths.iter().skip(1) {
                bounds = bounds.union(&path.clipped_bounds());
            }
            // One pixel of padding on each side keeps the sprite's linear
            // sampling from bleeding a neighbouring slot in.
            let left = (bounds.origin.x.0.floor() as i32 - 1).max(0);
            let top = (bounds.origin.y.0.floor() as i32 - 1).max(0);
            let right = ((bounds.origin.x.0 + bounds.size.width.0).ceil() as i32 + 1)
                .min(viewport_width);
            let bottom = ((bounds.origin.y.0 + bounds.size.height.0).ceil() as i32 + 1)
                .min(viewport_height);
            if right <= left || bottom <= top {
                slots.push(PathBatchSlot::EMPTY);
                continue;
            }
            let allocator = allocator.get_or_insert_with(|| {
                BucketedAtlasAllocator::new(etagere::size2(viewport_width, viewport_height))
            });
            let allocation = allocator.allocate(etagere::size2(right - left, bottom - top))?;
            let slot = allocation.rectangle;
            slots.push(PathBatchSlot {
                offset: point(
                    ScaledPixels((slot.min.x - left) as f32),
                    ScaledPixels((slot.min.y - top) as f32),
                ),
                scissor: metal::MTLScissorRect {
                    x: slot.min.x as u64,
                    y: slot.min.y as u64,
                    width: (right - left) as u64,
                    height: (bottom - top) as u64,
                },
                empty: false,
            });
        }
        Some(slots)
    }

    /// Rasterizes every path batch of the frame into its planned slot of the
    /// path intermediate, in a single render pass.
    fn rasterize_path_batches(
        &mut self,
        scene: &Scene,
        rounded_clips: &BufferRegion,
        slots: &[PathBatchSlot],
        writer: &mut InstanceBufferWriter,
        viewport_size: Size<DevicePixels>,
        command_buffer: &metal::CommandBufferRef,
    ) -> Result<()> {
        if slots.iter().all(|slot| slot.empty) {
            return Ok(());
        }
        let intermediate_texture = self.ensure_path_intermediate_textures(viewport_size)?;
        let command_encoder =
            self.new_path_rasterization_encoder(command_buffer, &intermediate_texture);
        command_encoder.set_fragment_buffer(15, Some(&rounded_clips.buffer), rounded_clips.offset as u64);
        command_encoder.set_vertex_bytes(
            PathRasterizationInputIndex::ViewportSize as u64,
            mem::size_of_val(&viewport_size) as u64,
            &viewport_size as *const Size<DevicePixels> as *const _,
        );

        let mut slots = slots.iter();
        let mut vertices = Vec::new();
        for batch in scene.batches() {
            let PrimitiveBatch::Paths(range) = batch else {
                continue;
            };
            let slot = slots
                .next()
                .context("path plan has fewer slots than the scene has path batches")?;
            if slot.empty {
                continue;
            }
            vertices.clear();
            for path in &scene.paths[range] {
                vertices.extend(path.vertices.iter().map(|v| PathRasterizationVertex {
                    rounded_clip: path.rounded_clip,
                    clip_padding: 0,
                    xy_position: v.xy_position,
                    st_position: v.st_position,
                    color: path.color,
                    bounds: path.bounds.intersect(&path.content_mask.bounds),
                }));
            }
            let vertex_instances = match writer.write(&vertices) {
                Ok(vertex_instances) => vertex_instances,
                Err(error) => {
                    command_encoder.end_encoding();
                    return Err(error);
                }
            };
            let slot_offset = [slot.offset.x.0, slot.offset.y.0];
            command_encoder.set_scissor_rect(slot.scissor);
            command_encoder.set_vertex_bytes(
                PathRasterizationInputIndex::SlotOffset as u64,
                mem::size_of_val(&slot_offset) as u64,
                slot_offset.as_ptr() as *const _,
            );
            command_encoder.set_vertex_buffer(
                PathRasterizationInputIndex::Vertices as u64,
                Some(&vertex_instances.buffer),
                vertex_instances.offset as u64,
            );
            command_encoder.set_fragment_buffer(
                PathRasterizationInputIndex::Vertices as u64,
                Some(&vertex_instances.buffer),
                vertex_instances.offset as u64,
            );
            command_encoder.draw_primitives(
                metal::MTLPrimitiveType::Triangle,
                0,
                vertices.len() as u64,
            );
        }
        command_encoder.end_encoding();
        Ok(())
    }

    fn new_path_rasterization_encoder<'a>(
        &self,
        command_buffer: &'a metal::CommandBufferRef,
        intermediate_texture: &metal::TextureRef,
    ) -> &'a metal::RenderCommandEncoderRef {
        let render_pass_descriptor = metal::RenderPassDescriptor::new();
        let color_attachment = render_pass_descriptor
            .color_attachments()
            .object_at(0)
            .unwrap();
        color_attachment.set_load_action(metal::MTLLoadAction::Clear);
        color_attachment.set_clear_color(metal::MTLClearColor::new(0., 0., 0., 0.));

        if let Some(msaa_texture) = &self.path_intermediate_msaa_texture {
            color_attachment.set_texture(Some(msaa_texture));
            color_attachment.set_resolve_texture(Some(intermediate_texture));
            color_attachment.set_store_action(metal::MTLStoreAction::MultisampleResolve);
        } else {
            color_attachment.set_texture(Some(intermediate_texture));
            color_attachment.set_store_action(metal::MTLStoreAction::Store);
        }

        let command_encoder = command_buffer.new_render_command_encoder(render_pass_descriptor);
        command_encoder.set_render_pipeline_state(&self.paths_rasterization_pipeline_state);
        command_encoder
    }

    fn create_path_intermediate_textures(&mut self, size: Size<DevicePixels>) {
        let texture_descriptor = metal::TextureDescriptor::new();
        texture_descriptor.set_width(size.width.0 as u64);
        texture_descriptor.set_height(size.height.0 as u64);
        texture_descriptor.set_pixel_format(metal::MTLPixelFormat::BGRA8Unorm);
        texture_descriptor.set_storage_mode(metal::MTLStorageMode::Private);
        texture_descriptor
            .set_usage(metal::MTLTextureUsage::RenderTarget | metal::MTLTextureUsage::ShaderRead);
        self.path_intermediate_texture = Some(self.device.new_texture(&texture_descriptor));

        if self.path_sample_count > 1 {
            // https://developer.apple.com/documentation/metal/choosing-a-resource-storage-mode-for-apple-gpus
            // Rendering MSAA textures are done in a single pass, so we can use memory-less storage on Apple Silicon
            let storage_mode = if self.is_apple_gpu {
                metal::MTLStorageMode::Memoryless
            } else {
                metal::MTLStorageMode::Private
            };

            let msaa_descriptor = texture_descriptor;
            msaa_descriptor.set_texture_type(metal::MTLTextureType::D2Multisample);
            msaa_descriptor.set_storage_mode(storage_mode);
            msaa_descriptor.set_sample_count(self.path_sample_count as _);
            self.path_intermediate_msaa_texture = Some(self.device.new_texture(&msaa_descriptor));
        } else {
            self.path_intermediate_msaa_texture = None;
        }
    }

    pub fn update_transparency(&mut self, transparent: bool) {
        self.opaque = !transparent;
        self.presented_frame = None;
        if let Some(layer) = &self.layer {
            layer.set_opaque(!transparent);
        }
    }

    pub fn destroy(&self) {
        // nothing to do
    }

    fn can_draw(&self) -> bool {
        self.unfinished_submissions.load(Ordering::Relaxed) < MAX_UNFINISHED_SUBMISSIONS
    }

    pub fn submission_queue_full_outcome(&mut self) -> Option<PresentOutcome> {
        if self.can_draw() {
            return None;
        }
        // Lost completion callbacks must not cause perpetual refresh retries.
        // Dropping the owed frame ends retries without forgetting permits for
        // resources Metal may still retain.
        Some(self.deferred_present_outcome(true))
    }

    pub fn draw(&mut self, scene: &Scene) -> PresentReport {
        // `nextDrawable` hands back an autoreleased drawable, and the command
        // buffer and pass descriptors are autoreleased too. Without a pool of
        // its own the frame's drawable stays retained until the main run
        // loop's pool drains, which a main thread busy for several frames in
        // a row puts off; Core Animation cannot reuse a drawable that is still
        // retained, so the layer's three-drawable pool empties faster and the
        // next `nextDrawable` waits. `present_drawable` retains the drawable
        // for as long as the command buffer needs it, and `render_frame`
        // returns an owned command buffer, so nothing used after the pool
        // outlives it.
        objc2::rc::autoreleasepool(|_| self.draw_in_pool(scene))
    }

    /// Presents `scene` to the layer and reports where the time went:
    /// `acquire` is `nextDrawable`, `encode` is `render_frame`, and `commit`
    /// is the hand-off to the GPU and compositor, including the
    /// in-transaction path's wait for the command buffer to be scheduled.
    /// They are split because a slow present has different causes in each:
    /// a compositor holding the layer's drawables, main-thread CPU (or page
    /// faults) while encoding, or a GPU backlog.
    fn draw_in_pool(&mut self, scene: &Scene) -> PresentReport {
        // Taken before anything can defer or drop this present: the window
        // advanced the scene's time animations for it, so a later present
        // cannot tell what changed since the frame on screen from the
        // scene's latest advance alone (a transition may have landed in
        // between). It goes back only once this present is encoded.
        let presented_frame = self.presented_frame.take();
        let mut breakdown = PresentBreakdown {
            layers: 1,
            ..PresentBreakdown::default()
        };
        let dropped = |breakdown| PresentReport {
            outcome: PresentOutcome::Dropped,
            breakdown,
        };
        let layer = match &self.layer {
            Some(l) => l.clone(),
            None => {
                log::error!(
                    "draw() called on headless renderer - use render_scene_to_image() instead"
                );
                return dropped(PresentBreakdown::default());
            }
        };
        // A frame drawn while the drawables are shrunk would present a single
        // pixel over what the window shows when it comes back; it stays owed
        // and is drawn once the size is restored.
        if self.drawables_released {
            return PresentReport {
                outcome: PresentOutcome::Deferred,
                breakdown,
            };
        }
        let viewport_size = layer.drawable_size();
        let viewport_size: Size<DevicePixels> = size(
            (viewport_size.width.ceil() as i32).into(),
            (viewport_size.height.ceil() as i32).into(),
        );
        // An empty layer has no drawable to give, and waiting for one would
        // retry every refresh for nothing. Resizing it to a real size redraws
        // the window anyway, so the frame is not owed.
        if viewport_size.width.0 <= 0 || viewport_size.height.0 <= 0 {
            // A dropped frame ends any run of deferrals, as it does when the
            // run is cut off, so the next missing drawable starts a new run.
            self.consecutive_deferred_presents = 0;
            return dropped(breakdown);
        }
        let Some(submission) = InFlightSubmission::try_acquire(&self.unfinished_submissions) else {
            return PresentReport {
                outcome: self.deferred_present_outcome(true),
                breakdown,
            };
        };
        let (drawables_in_flight, drawables_in_flight_clamped) =
            self.read_drawables_in_flight(&layer);
        breakdown.drawables_in_flight = drawables_in_flight;
        breakdown.drawables_in_flight_clamped = drawables_in_flight_clamped;
        let acquire_start = Instant::now();
        let drawable = layer.next_drawable();
        breakdown.acquire = acquire_start.elapsed();
        let Some(drawable) = drawable else {
            let objc2_layer: &objc2_quartz_core::CAMetalLayer = unsafe { &*layer.as_ptr().cast() };
            let allows_timeout = objc2_layer.allowsNextDrawableTimeout();
            let outcome = self.missing_drawable_outcome(allows_timeout);
            match outcome {
                PresentOutcome::Deferred if self.consecutive_deferred_presents == 1 => {
                    log::warn!(
                        "no drawable available (drawable size {viewport_size:?}, \
                         {} in flight); deferring the present",
                        breakdown.drawables_in_flight
                    );
                }
                PresentOutcome::Dropped => {
                    log::error!(
                        "no drawable available (drawable size {viewport_size:?}, \
                         {} in flight); dropping the frame",
                        breakdown.drawables_in_flight
                    );
                }
                _ => {}
            }
            return PresentReport { outcome, breakdown };
        };
        if self.consecutive_deferred_presents > 0 {
            log::info!(
                "drawable available again after {} deferred presents",
                self.consecutive_deferred_presents
            );
            self.consecutive_deferred_presents = 0;
        }

        let encode_start = Instant::now();
        let previous_frame =
            self.previous_frame_to_copy(presented_frame, scene, drawable.texture(), viewport_size);
        let command_buffer = self.render_frame(
            scene, drawable.texture(), viewport_size, Some(submission), previous_frame.as_deref(),
        );
        breakdown.encode = encode_start.elapsed();
        let command_buffer = match command_buffer {
            Ok(command_buffer) => command_buffer,
            Err(error) => {
                log::error!("failed to render: {error:#}");
                return dropped(breakdown);
            }
        };
        self.presented_frame = Some(PresentedFrame {
            texture: drawable.texture().to_owned(),
            content_id: scene.content_id(),
            viewport_size,
        });

        let commit_start = Instant::now();
        self.note_drawable_in_flight(drawable);
        if self.presents_with_transaction {
            command_buffer.commit();
            command_buffer.wait_until_scheduled();
            drawable.present();
        } else {
            command_buffer.present_drawable(drawable);
            command_buffer.commit();
        }
        breakdown.commit = commit_start.elapsed();
        self.maybe_log_gpu_stats();
        self.maybe_attribute_gpu_time(scene, viewport_size);
        PresentReport {
            outcome: PresentOutcome::Presented,
            breakdown,
        }
    }

    /// `presented`'s drawable, when `scene` is what it shows with only its
    /// time animations advanced once since; `damage_rects` then holds what
    /// changed, and the present copies the rest from it instead of drawing
    /// the whole window again for a shimmer or a spinner.
    ///
    /// Only the drawable last presented is taken: it is the one on screen or
    /// queued for it, so the compositor holds it and nothing writes to it.
    /// Earlier ones are back in the layer's pool for `nextDrawable` to hand
    /// out. `GPUI_DISABLE_PARTIAL_PRESENTS=1` draws every present in full.
    fn previous_frame_to_copy(
        &mut self,
        presented: Option<PresentedFrame>,
        scene: &Scene,
        texture: &metal::TextureRef,
        viewport_size: Size<DevicePixels>,
    ) -> Option<metal::Texture> {
        static DISABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let presented = presented?;
        let disabled = *DISABLED.get_or_init(|| {
            std::env::var("GPUI_DISABLE_PARTIAL_PRESENTS").is_ok_and(|value| value == "1")
        });
        // A drawable handed back again was released by the compositor, and a
        // copy onto itself is undefined.
        let usable = !disabled
            && presented.content_id == scene.content_id()
            && presented.viewport_size == viewport_size
            && presented.texture.as_ptr() != texture.as_ptr()
            // A surface's pixel buffer can change without the scene changing.
            && scene.surfaces.is_empty()
            && scene.time_animation_damage(viewport_size, &mut self.damage_rects);
        usable.then_some(presented.texture)
    }

    /// While `FINCODE_GPU_ATTRIBUTION=1`, every [`ATTRIBUTION_PRESENT_STRIDE`]th
    /// present draws its scene again into an offscreen target, once whole,
    /// once with nothing but the clear, and once without each primitive kind,
    /// timing each on the GPU. Every [`ATTRIBUTION_ROUNDS`] rounds it logs the
    /// medians: what a kind saves when left out is what it costs in this
    /// window's real scene, overdraw and early depth rejection included,
    /// which a Metal trace of the window cannot split out of one pass.
    fn maybe_attribute_gpu_time(&mut self, scene: &Scene, viewport_size: Size<DevicePixels>) {
        let Some(attribution) = self.gpu_attribution.as_mut() else {
            return;
        };
        attribution.presents_since_round += 1;
        if attribution.presents_since_round < ATTRIBUTION_PRESENT_STRIDE {
            return;
        }
        attribution.presents_since_round = 0;
        let (width, height) = (viewport_size.width.0 as u64, viewport_size.height.0 as u64);
        let target = match &attribution.target {
            Some(target) if target.width() == width && target.height() == height => target.clone(),
            _ => {
                let descriptor = metal::TextureDescriptor::new();
                descriptor.set_width(width);
                descriptor.set_height(height);
                descriptor.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
                descriptor.set_usage(metal::MTLTextureUsage::RenderTarget | metal::MTLTextureUsage::ShaderRead);
                descriptor.set_storage_mode(metal::MTLStorageMode::Private);
                let target = self.device.new_texture(&descriptor);
                attribution.target = Some(target.clone());
                attribution.samples.lock().clear();
                attribution.rounds = 0;
                target
            }
        };
        attribution.rounds += 1;
        let log_now = attribution.rounds >= ATTRIBUTION_ROUNDS;
        let round = attribution.rounds;
        let samples = attribution.samples.clone();
        let mut variants = [0; 2 + ATTRIBUTED_KINDS.len()];
        variants[1] = ATTRIBUTED_ALL;
        for (variant, (bit, _)) in variants[2..].iter_mut().zip(ATTRIBUTED_KINDS) {
            *variant = bit;
        }
        // The GPU clocks up through a round, so no variant always goes first.
        let count = variants.len();
        variants.rotate_left(attribution.rounds as usize % count);
        // Every render is encoded before any is committed, so the GPU runs
        // them back to back: each one's cost is then how much later it ends
        // than the one before, beginning with an empty fence. Their own
        // start-to-end spans overlap their neighbours' and overstate it.
        self.attributing = true;
        let mut renders = Vec::with_capacity(variants.len());
        for skips in variants {
            self.attribution_skips = skips;
            match self.render_frame(scene, &target, viewport_size, None, None) {
                Ok(command_buffer) => renders.push((skips, command_buffer)),
                Err(error) => {
                    log::warn!("[gpu-attribution] render failed: {error:#}");
                    break;
                }
            }
        }
        let fence = self.command_queue.new_command_buffer().to_owned();
        for (slot, (skips, command_buffer)) in
            std::iter::once((ATTRIBUTION_FENCE, &fence)).chain(renders.iter().map(|(skips, buffer)| (*skips, buffer))).enumerate()
        {
            let samples = samples.clone();
            let block = RcBlock::new(move |buffer: ptr::NonNull<AnyObject>| {
                let buffer = buffer.as_ptr();
                // SAFETY: `GPUEndTime` is a `CFTimeInterval` property of every
                // completed `MTLCommandBuffer`.
                let end: f64 = unsafe { objc2::msg_send![buffer, GPUEndTime] };
                samples.lock().push(AttributionSample { round, slot, skips, end });
            });
            // SAFETY: Both pointee types are opaque views of the same Objective-C block pointer ABI.
            unsafe {
                command_buffer.add_completed_handler(&*RcBlock::as_ptr(&block).cast());
            }
            command_buffer.commit();
        }
        self.attribution_skips = 0;
        self.attributing = false;
        if log_now && let Some(attribution) = self.gpu_attribution.as_mut() {
            attribution.rounds = 0;
            let samples = mem::take(&mut *attribution.samples.lock());
            if let Some(line) = gpu_attribution_line(&samples, scene, viewport_size) {
                log::info!("{line}");
            }
        }
    }

    /// What a present that got no drawable reports. The frame stays owed
    /// (`Deferred`) only while a nil drawable can mean the pool was merely
    /// empty: with `allowsNextDrawableTimeout` off, `nextDrawable` waits for
    /// a drawable instead, so a nil there is a fault (a lost device, a layer
    /// with no usable configuration) that would be retried every display
    /// refresh for as long as it lasts. A run of deferrals is also cut off
    /// after [`MAX_CONSECUTIVE_DEFERRED_PRESENTS`], so a pool that never
    /// refills costs one dropped frame per invalidation rather than a
    /// present attempt per refresh.
    fn missing_drawable_outcome(&mut self, layer_allows_timeout: bool) -> PresentOutcome {
        self.deferred_present_outcome(layer_allows_timeout)
    }

    fn deferred_present_outcome(&mut self, can_retry: bool) -> PresentOutcome {
        if can_retry
            && self.consecutive_deferred_presents < MAX_CONSECUTIVE_DEFERRED_PRESENTS
        {
            self.consecutive_deferred_presents += 1;
            PresentOutcome::Deferred
        } else {
            self.consecutive_deferred_presents = 0;
            PresentOutcome::Dropped
        }
    }

    /// How many drawables are in flight, for a present's report, and whether
    /// the count is known to overstate them. More than the layer's maximum
    /// drawable count can only mean presented handlers that never ran (a
    /// command buffer that failed after commit, say); the count is lowered to
    /// the maximum, which still bounds the drawables really in flight from
    /// above, so lost handlers cannot keep it climbing. That removes only the
    /// excess, so the bias below the maximum stays, and so does the flag,
    /// until [`Self::forget_drawables_in_flight`].
    fn read_drawables_in_flight(&mut self, layer: &metal::MetalLayerRef) -> (u8, bool) {
        let maximum = layer.maximum_drawable_count().min(u8::MAX as NSUInteger) as u32;
        let count = self
            .drawables_in_flight
            .fetch_min(maximum, Ordering::Relaxed);
        self.drawables_in_flight_overstated |= count > maximum;
        (count.min(maximum) as u8, self.drawables_in_flight_overstated)
    }

    /// Starts the in-flight count afresh, for a window that has just been
    /// hidden, so a count left raised by a presented handler that never ran
    /// does not read as a held pool for the rest of the window's life.
    /// Handlers of drawables presented before this still decrement the count
    /// they were registered with, not the new one; drawables the compositor
    /// holds from before the window was hidden therefore go uncounted once it
    /// is visible again, until it has presented past them.
    pub fn forget_drawables_in_flight(&mut self) {
        self.presented_frame = None;
        self.drawables_in_flight = Arc::new(AtomicU32::new(0));
        self.drawables_in_flight_overstated = false;
    }

    /// Counts `drawable` as in flight until Core Animation displays or
    /// discards it; the presented handler runs in both cases. Must be called
    /// before the drawable is presented, which is when handlers are fixed.
    fn note_drawable_in_flight(&self, drawable: &metal::MetalDrawableRef) {
        let drawables_in_flight = self.drawables_in_flight.clone();
        drawables_in_flight.fetch_add(1, Ordering::Relaxed);
        let block = RcBlock::new(move |_: ptr::NonNull<AnyObject>| {
            drawables_in_flight
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |count| {
                    count.checked_sub(1)
                })
                .ok();
        });
        // SAFETY: Both pointee types are opaque views of the same Objective-C block pointer ABI.
        unsafe {
            drawable.add_presented_handler(&*RcBlock::as_ptr(&block).cast());
        }
    }

    fn gpu_stats_enabled() -> bool {
        static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ENABLED
            .get_or_init(|| std::env::var("FINCODE_GPU_STATS").is_ok_and(|value| value == "1"))
    }

    /// Logs where the renderer's device memory sits, at most every ten
    /// seconds, while `FINCODE_GPU_STATS=1`.
    ///
    /// `current_allocated_size` is the device's own accounting of live
    /// allocations; the per-category numbers say which of atlas growth, the
    /// instance pool, the depth/path intermediates or the backdrop blur
    /// scratch is responsible when it climbs. Note the OS additionally charges driver-side arenas to the
    /// process ("owned graphics" in footprint output) that no number here
    /// includes.
    fn maybe_log_gpu_stats(&mut self) {
        if !Self::gpu_stats_enabled() {
            return;
        }
        let now = std::time::Instant::now();
        if self
            .gpu_stats_last_log
            .is_some_and(|last| now.duration_since(last) < std::time::Duration::from_secs(10))
        {
            return;
        }
        self.gpu_stats_last_log = Some(now);
        let (monochrome_bytes, polychrome_bytes) = self.sprite_atlas.allocated_bytes();
        let (pool_buffers, pool_buffer_size) = {
            let pool = self.instance_buffer_pool.lock();
            (pool.buffers.len(), pool.buffer_size)
        };
        let depth_bytes = self
            .depth_texture
            .as_ref()
            .map_or(0, |texture| texture.allocated_size());
        let path_intermediate_bytes = self
            .path_intermediate_texture
            .as_ref()
            .map_or(0, |texture| texture.allocated_size())
            + self
                .path_intermediate_msaa_texture
                .as_ref()
                .map_or(0, |texture| texture.allocated_size());
        let backdrop_bytes = self
            .backdrop_scratch
            .as_ref()
            .map_or(0, BackdropScratch::allocated_bytes);
        const MB: u64 = 1024 * 1024;
        log::info!(
            "[gpu-stats] device_allocated={}MB atlas_monochrome={}MB atlas_polychrome={}MB \
             instance_pool={}x{}KB depth={}MB path_intermediates={}MB backdrop_scratch={}KB \
             presents_full={} presents_partial={}",
            self.device.current_allocated_size() / MB,
            monochrome_bytes / MB,
            polychrome_bytes / MB,
            pool_buffers,
            pool_buffer_size / 1024,
            depth_bytes / MB,
            path_intermediate_bytes / MB,
            backdrop_bytes / 1024,
            mem::take(&mut self.full_presents),
            mem::take(&mut self.partial_presents),
        );
    }

    fn render_frame(
        &mut self,
        scene: &Scene,
        texture: &metal::TextureRef,
        viewport_size: Size<DevicePixels>,
        submission: Option<InFlightSubmission>,
        previous_frame: Option<&metal::TextureRef>,
    ) -> Result<metal::CommandBuffer> {
        // A window that lives through display, backing-scale, or backing-view
        // changes can be left with renderer textures sized for a stale
        // drawable. A depth attachment whose size disagrees with the color
        // attachment is undefined behavior — with the depth prepass it shows
        // up as blended fills failing arbitrary depth tests (translucent
        // overlays losing their fill) persistently for that window, while
        // fresh windows render correctly. Heal by recreating the textures
        // whenever they disagree with the drawable, and log the stale size so
        // the event that bypassed `update_drawable_size` can be identified.
        let depth_texture_matches_viewport = self.depth_texture.as_ref().is_some_and(|texture| {
            texture.width() == viewport_size.width.0 as u64
                && texture.height() == viewport_size.height.0 as u64
        });
        if self.intermediate_textures_released
            && viewport_size.width.0 > 0
            && viewport_size.height.0 > 0
        {
            self.intermediate_textures_released = false;
            self.update_intermediate_textures(viewport_size);
        } else if !depth_texture_matches_viewport
            && viewport_size.width.0 > 0
            && viewport_size.height.0 > 0
        {
            self.stale_texture_heals = self.stale_texture_heals.saturating_add(1);
            // In steady state this never fires: draw() reads the drawable
            // size that update_drawable_size set alongside the textures. A
            // recreated layer whose auto-derived drawable size tracks live
            // resizing can heal once per frame during a drag, so cap the log
            // rather than emit sixty lines a second.
            if self.stale_texture_heals == 1 || self.stale_texture_heals.is_multiple_of(100) {
                log::warn!(
                    "renderer textures were stale for a {}x{} drawable (depth texture {:?}, \
                     heal #{}) — recreating; a display or backing change likely bypassed \
                     update_drawable_size",
                    viewport_size.width.0,
                    viewport_size.height.0,
                    self.depth_texture
                        .as_ref()
                        .map(|texture| (texture.width(), texture.height())),
                    self.stale_texture_heals,
                );
            }
            self.recreate_intermediate_textures(viewport_size);
        }

        let mut writer = InstanceBufferWriter::new(
            &self.device,
            &self.instance_buffer_pool,
            self.is_unified_memory,
        );
        let instances = upload_frame_instances(scene, &mut writer).with_context(|| {
            format!(
                "scene too large: {} paths, {} shadows, {} quads, {} underlines, {} mono, {} poly, {} surfaces",
                scene.paths.len(),
                scene.shadows.len(),
                scene.quads.len(),
                scene.underlines.len(),
                scene.monochrome_sprites.len(),
                scene.polychrome_sprites.len(),
                scene.surfaces.len(),
            )
        })?;
        let command_buffer = self.draw_primitives_to_texture(
            scene,
            &instances,
            &mut writer,
            texture,
            viewport_size,
            previous_frame,
        )?;
        self.retire_idle_path_intermediate();
        self.retire_idle_backdrop_scratch();

        let instance_buffer_pool = self.instance_buffer_pool.clone();
        let instance_buffer = Cell::new(Some(writer.finish()));
        let submission = Cell::new(submission);
        let block = RcBlock::new(move |_: ptr::NonNull<AnyObject>| {
            if let Some(instance_buffer) = instance_buffer.take() {
                instance_buffer_pool.lock().release(instance_buffer);
            }
            drop(submission.take());
        });
        // SAFETY: Both pointee types are opaque views of the same Objective-C block pointer ABI.
        unsafe {
            command_buffer.add_completed_handler(&*RcBlock::as_ptr(&block).cast());
        }

        // Every present goes through here, so this is where a frame's sprite
        // tiles count as used; it also retires tiles idle for too long. Only
        // once the frame has actually been encoded: a frame that failed above
        // drew nothing, so it must not age the atlas. An overlay renderer's
        // scene was marked by its window's base draw and belongs to the same
        // present, so it neither marks nor advances (see `note_scene_tiles`).
        // An attribution render draws the frame on screen again; counting it
        // would age the atlas faster than the window presents.
        if !self.shares_atlas_frame && !self.attributing {
            self.sprite_atlas.on_frame_drawn(scene);
        }
        self.publish_depth_memory();

        Ok(command_buffer)
    }

    /// Renders the scene to a texture and returns the pixel data as an RGBA image.
    /// This does not present the frame to screen - useful for visual testing
    /// where we want to capture what would be rendered without displaying it.
    ///
    /// Note: This requires a layer-backed renderer. For headless rendering,
    /// use `render_scene_to_image()` instead.
    #[cfg(any(test, feature = "test-support"))]
    pub fn render_to_image(&mut self, scene: &Scene) -> Result<RgbaImage> {
        let layer = self
            .layer
            .clone()
            .ok_or_else(|| anyhow::anyhow!("render_to_image requires a layer-backed renderer"))?;
        let viewport_size = layer.drawable_size();
        let viewport_size: Size<DevicePixels> = size(
            (viewport_size.width.ceil() as i32).into(),
            (viewport_size.height.ceil() as i32).into(),
        );
        let drawable = layer
            .next_drawable()
            .ok_or_else(|| anyhow::anyhow!("Failed to get drawable for render_to_image"))?;

        let command_buffer = self.render_frame(scene, drawable.texture(), viewport_size, None, None)?;

        // Commit and wait for completion without presenting
        command_buffer.commit();
        command_buffer.wait_until_completed();

        read_texture_to_image(drawable.texture())
    }

    /// Renders a scene to an image without requiring a window or CAMetalLayer.
    ///
    /// This is the primary method for headless rendering. It creates an offscreen
    /// texture, renders the scene to it, and returns the pixel data as an RGBA image.
    #[cfg(any(test, feature = "bench-support", feature = "test-support"))]
    pub fn render_scene_to_image(
        &mut self,
        scene: &Scene,
        size: Size<DevicePixels>,
    ) -> Result<RgbaImage> {
        if size.width.0 <= 0 || size.height.0 <= 0 {
            anyhow::bail!("Invalid size for render_scene_to_image: {:?}", size);
        }

        // Headless callers do not have a Cocoa event-loop pool to release
        // autoreleased command buffers and render-pass descriptors.
        objc2::rc::autoreleasepool(|_| {
            // Update intermediate textures for this size
            self.update_intermediate_textures(size);

            // Create an offscreen texture as render target
            let texture_descriptor = metal::TextureDescriptor::new();
            texture_descriptor.set_width(size.width.0 as u64);
            texture_descriptor.set_height(size.height.0 as u64);
            texture_descriptor.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
            texture_descriptor.set_usage(
                metal::MTLTextureUsage::RenderTarget | metal::MTLTextureUsage::ShaderRead,
            );
            // Like the atlas, only Apple GPUs can create shared textures on macOS;
            // Intel Macs cannot, even with unified memory. iOS has no managed storage.
            let uses_shared_storage = cfg!(target_os = "ios") || self.is_apple_gpu;
            texture_descriptor.set_storage_mode(if uses_shared_storage {
                metal::MTLStorageMode::Shared
            } else {
                metal::MTLStorageMode::Managed
            });
            let target_texture = self.device.new_texture(&texture_descriptor);

            let command_buffer = self.render_frame(scene, &target_texture, size, None, None)?;

            // Managed textures require an explicit blit synchronize before the CPU
            // can read back the rendered data. Without this, get_bytes returns
            // stale zeros.
            if !uses_shared_storage {
                let blit = command_buffer.new_blit_command_encoder();
                blit.synchronize_resource(&target_texture);
                blit.end_encoding();
            }

            // Commit and wait for completion
            command_buffer.commit();
            command_buffer.wait_until_completed();

            read_texture_to_image(&target_texture)
        })
    }

    /// Renders a scene to a reused offscreen texture without reading pixels
    /// back or blocking on GPU completion.
    ///
    /// This mirrors the CPU cost of presenting a frame to a window (scene
    /// encoding, instance buffer writes, command submission) and is used by
    /// headless benchmark rendering, where the produced pixels are never
    /// inspected.
    #[cfg(any(test, feature = "bench-support", feature = "test-support"))]
    pub fn render_scene(&mut self, scene: &Scene, size: Size<DevicePixels>) -> Result<()> {
        if size.width.0 <= 0 || size.height.0 <= 0 {
            anyhow::bail!("Invalid size for render_scene: {:?}", size);
        }

        objc2::rc::autoreleasepool(|_| {
            self.update_intermediate_textures(size);

            let needs_new_target = self.headless_render_target.as_ref().is_none_or(|texture| {
                texture.width() != size.width.0 as u64 || texture.height() != size.height.0 as u64
            });
            if needs_new_target {
                let texture_descriptor = metal::TextureDescriptor::new();
                texture_descriptor.set_width(size.width.0 as u64);
                texture_descriptor.set_height(size.height.0 as u64);
                texture_descriptor.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
                texture_descriptor.set_usage(
                    metal::MTLTextureUsage::RenderTarget | metal::MTLTextureUsage::ShaderRead,
                );
                texture_descriptor.set_storage_mode(metal::MTLStorageMode::Private);
                self.headless_render_target = Some(self.device.new_texture(&texture_descriptor));
            }
            let target_texture = self
                .headless_render_target
                .clone()
                .expect("just ensured the render target exists");

            let command_buffer = self.render_frame(scene, &target_texture, size, None, None)?;

            // Commit without waiting, mirroring presentation to a real window where
            // the CPU doesn't block on the GPU.
            command_buffer.commit();
            Ok(())
        })
    }

    fn draw_primitives_to_texture(
        &mut self,
        scene: &Scene,
        instances: &FrameInstances,
        writer: &mut InstanceBufferWriter,
        texture: &metal::TextureRef,
        viewport_size: Size<DevicePixels>,
        previous_frame: Option<&metal::TextureRef>,
    ) -> Result<metal::CommandBuffer> {
        let path_plan = Self::plan_path_batches(scene, viewport_size);
        let depth_texture = if path_plan.is_some() || !self.is_apple_gpu {
            self.depth_texture
                .clone()
                .context("missing depth texture")?
        } else {
            self.ensure_fallback_depth_texture(viewport_size)
        };
        // Tile memory cannot be stored; the fallback texture must be, since
        // the main pass reloads it after every path batch.
        let store_depth = depth_texture.storage_mode() != metal::MTLStorageMode::Memoryless;
        let command_queue = self.command_queue.clone();
        let command_buffer = command_queue.new_command_buffer();
        if let Some(slots) = &path_plan
            && self.attribution_skips & ATTRIBUTED_PATHS == 0
        {
            self.rasterize_path_batches(scene, &instances.rounded_clips, slots, writer, viewport_size, command_buffer)?;
        }
        // A present drawn in part draws path batches only from the packed
        // slots: the split pass ends the encoder at each batch.
        let previous_frame = previous_frame.filter(|_| path_plan.is_some());
        let damage = previous_frame.map(|_| mem::take(&mut self.damage_rects));
        if !self.attributing {
            if damage.is_some() {
                self.partial_presents = self.partial_presents.saturating_add(1);
            } else {
                self.full_presents = self.full_presents.saturating_add(1);
            }
        }
        let alpha = if self.opaque { 1. } else { 0. };

        // The clear covers the whole target, so each damaged rectangle is
        // drawn from the clear colour up, as in a full frame, and the copy
        // that follows replaces everything outside them.
        let mut command_encoder = new_command_encoder_for_texture(
            command_buffer,
            texture,
            &depth_texture,
            store_depth,
            viewport_size,
            Some(metal::MTLClearColor::new(0., 0., 0., alpha)),
        );
        command_encoder.set_fragment_buffer(15, Some(&instances.rounded_clips.buffer), instances.rounded_clips.offset as u64);

        // Damage rectangles never overlap, so each one's depth prepass
        // writes only depth that its own draws test against.
        let regions = damage.as_ref().map_or(1, Vec::len);
        for region in 0..regions {
            let scissor = damage
                .as_ref()
                .and_then(|rects| rects.get(region))
                .map(scissor_rect);
            if let Some(scissor) = scissor {
                command_encoder.set_scissor_rect(scissor);
            }
            // A rectangle that reaches a blur holds all of it (see
            // `Scene::time_animation_damage`), so the blur runs again there
            // from what the rectangle redrew beneath it; the copy brings the
            // frosting of every other blur.
            let mut pending_blurs = scene
                .backdrop_blurs
                .iter()
                .filter(|blur| scissor.is_none_or(|scissor| scissor_reaches_blur(&scissor, blur)))
                .peekable();
            let mut planned_slots = path_plan.as_deref().map(|slots| slots.iter());
            let mut quad_cursor: u32 = 0;
            // The previous rectangle may have ended on a batch that collapsed
            // the depth range; the prepass gives each quad its own depth.
            set_viewport(command_encoder, viewport_size, 0., 1.);
            command_encoder.set_depth_stencil_state(&self.depth_write_state);
            let opaque_quad_indices = scene.blended_quad_indices.len()
                ..scene.blended_quad_indices.len() + scene.opaque_quad_indices.len();
            if self.attribution_skips & ATTRIBUTED_OPAQUE_QUADS == 0 {
                self.draw_quads(
                    &self.opaque_quads_pipeline_state,
                    opaque_quad_indices,
                    instances,
                    viewport_size,
                    command_encoder,
                );
            }
            command_encoder.set_depth_stencil_state(&self.depth_test_state);

            for batch in scene.batches() {
                let batch_order = scene.batch_order(&batch);
                while let Some(blur) = pending_blurs.next_if(|blur| blur.order <= batch_order) {
                    if self.attribution_skips & ATTRIBUTED_BACKDROP_BLURS != 0 {
                        continue;
                    }
                    command_encoder = self.draw_backdrop_blur(
                        blur,
                        command_buffer,
                        command_encoder,
                        texture,
                        &depth_texture,
                        store_depth,
                        &instances.rounded_clips,
                        writer,
                        viewport_size,
                        quad_depth(quad_cursor) as f64,
                    )?;
                    if let Some(scissor) = scissor {
                        command_encoder.set_scissor_rect(scissor);
                    }
                }
                // Quads carry their own depth. Every other batch is flattened onto
                // the depth of the quad cursor by collapsing the viewport's depth
                // range onto a single value.
                let batch_depth = quad_depth(quad_cursor) as f64;
                if matches!(batch, PrimitiveBatch::Quads { .. }) {
                    set_viewport(command_encoder, viewport_size, 0., 1.);
                } else {
                    set_viewport(command_encoder, viewport_size, batch_depth, batch_depth);
                }
                if self.attribution_skips & attributed_kind(&batch) != 0 {
                    match &batch {
                        PrimitiveBatch::Quads { range, .. } => quad_cursor += range.len() as u32,
                        PrimitiveBatch::Paths(_) => {
                            if let Some(slots) = planned_slots.as_mut() {
                                slots.next();
                            }
                        }
                        _ => {}
                    }
                    continue;
                }

                match batch {
                    PrimitiveBatch::Shadows(range) => {
                        self.draw_shadows(range, instances, viewport_size, command_encoder)
                    }
                    PrimitiveBatch::Quads {
                        range,
                        blended_range,
                    } => {
                        quad_cursor += range.len() as u32;
                        // Paint runs of consecutive quads that share the same glass
                        // flag, so glass-content quads use the alpha-preserving
                        // pipeline while ordinary quads use the standard one.
                        // Splitting by run keeps draw order intact.
                        let mut start = blended_range.start;
                        while start < blended_range.end {
                            let is_glass = blended_quad_is_glass(scene, start);
                            let mut end = start + 1;
                            while end < blended_range.end
                                && blended_quad_is_glass(scene, end) == is_glass
                            {
                                end += 1;
                            }

                            let pipeline = if is_glass {
                                &self.quads_glass_pipeline_state
                            } else {
                                &self.quads_pipeline_state
                            };
                            self.draw_quads(
                                pipeline,
                                start..end,
                                instances,
                                viewport_size,
                                command_encoder,
                            );
                            start = end;
                        }
                    }
                    PrimitiveBatch::Paths(range) => {
                        let paths = &scene.paths[range];
                        if let Some(slots) = planned_slots.as_mut() {
                            let Some(slot) = slots.next() else {
                                command_encoder.end_encoding();
                                anyhow::bail!("path plan has fewer slots than the scene has path batches");
                            };
                            if !slot.empty
                                && let Err(error) = self.draw_paths_from_intermediate(
                                    paths,
                                    slot.offset,
                                    writer,
                                    viewport_size,
                                    command_encoder,
                                )
                            {
                                command_encoder.end_encoding();
                                return Err(error);
                            }
                            continue;
                        }
                        command_encoder.end_encoding();

                        let did_draw = self.draw_paths_to_intermediate(
                            paths,
                            &instances.rounded_clips,
                            writer,
                            viewport_size,
                            command_buffer,
                        )?;

                        command_encoder = new_command_encoder_for_texture(
                            command_buffer,
                            texture,
                            &depth_texture,
                            store_depth,
                            viewport_size,
                            None,
                        );
                        command_encoder.set_fragment_buffer(15, Some(&instances.rounded_clips.buffer), instances.rounded_clips.offset as u64);
                        command_encoder.set_depth_stencil_state(&self.depth_test_state);
                        set_viewport(command_encoder, viewport_size, batch_depth, batch_depth);

                        if did_draw {
                            if let Err(error) = self.draw_paths_from_intermediate(
                                paths,
                                point(ScaledPixels(0.), ScaledPixels(0.)),
                                writer,
                                viewport_size,
                                command_encoder,
                            ) {
                                command_encoder.end_encoding();
                                return Err(error);
                            }
                        }
                    }
                    PrimitiveBatch::Underlines(range) => {
                        self.draw_underlines(range, instances, viewport_size, command_encoder)
                    }
                    PrimitiveBatch::MonochromeSprites { texture_id, range } => self
                        .draw_monochrome_sprites(
                            texture_id,
                            range,
                            instances,
                            viewport_size,
                            command_encoder,
                        ),
                    PrimitiveBatch::PolychromeSprites { texture_id, range } => self
                        .draw_polychrome_sprites(
                            texture_id,
                            range,
                            instances,
                            viewport_size,
                            command_encoder,
                        ),
                    PrimitiveBatch::Surfaces(range) => self.draw_surfaces(
                        &scene.surfaces[range.clone()],
                        range.start,
                        instances,
                        viewport_size,
                        command_encoder,
                    ),
                    PrimitiveBatch::SubpixelSprites { .. } => unreachable!(),
                }
            }
            // A blur above everything else still frosts what is beneath it.
            let skip_blurs = self.attribution_skips & ATTRIBUTED_BACKDROP_BLURS != 0;
            for blur in pending_blurs.filter(|_| !skip_blurs) {
                command_encoder = self.draw_backdrop_blur(
                    blur,
                    command_buffer,
                    command_encoder,
                    texture,
                    &depth_texture,
                    store_depth,
                    &instances.rounded_clips,
                    writer,
                    viewport_size,
                    quad_depth(quad_cursor) as f64,
                )?;
                if let Some(scissor) = scissor {
                    command_encoder.set_scissor_rect(scissor);
                }
            }
        }

        command_encoder.end_encoding();

        if let (Some(previous_frame), Some(damage)) = (previous_frame, damage) {
            let blit = command_buffer.new_blit_command_encoder();
            for rect in undamaged_rects(&damage, viewport_size) {
                let origin = metal::MTLOrigin {
                    x: rect.origin.x.0 as u64,
                    y: rect.origin.y.0 as u64,
                    z: 0,
                };
                blit.copy_from_texture(
                    previous_frame,
                    0,
                    0,
                    origin,
                    metal::MTLSize {
                        width: rect.size.width.0 as u64,
                        height: rect.size.height.0 as u64,
                        depth: 1,
                    },
                    texture,
                    0,
                    0,
                    origin,
                );
            }
            blit.end_encoding();
            self.damage_rects = damage;
        }

        Ok(command_buffer.to_owned())
    }

    fn draw_paths_to_intermediate(
        &mut self,
        paths: &[Path<ScaledPixels>],
        rounded_clips: &BufferRegion,
        writer: &mut InstanceBufferWriter,
        viewport_size: Size<DevicePixels>,
        command_buffer: &metal::CommandBufferRef,
    ) -> Result<bool> {
        if paths.is_empty() {
            return Ok(false);
        }
        let intermediate_texture = self.ensure_path_intermediate_textures(viewport_size)?;
        let command_encoder =
            self.new_path_rasterization_encoder(command_buffer, &intermediate_texture);
        command_encoder.set_fragment_buffer(15, Some(&rounded_clips.buffer), rounded_clips.offset as u64);
        let slot_offset = [0f32, 0f32];
        command_encoder.set_vertex_bytes(
            PathRasterizationInputIndex::SlotOffset as u64,
            mem::size_of_val(&slot_offset) as u64,
            slot_offset.as_ptr() as *const _,
        );

        let mut vertices = Vec::new();
        for path in paths {
            vertices.extend(path.vertices.iter().map(|v| PathRasterizationVertex {
                rounded_clip: path.rounded_clip,
                clip_padding: 0,
                xy_position: v.xy_position,
                st_position: v.st_position,
                color: path.color,
                bounds: path.bounds.intersect(&path.content_mask.bounds),
            }));
        }
        let vertex_instances = writer.write(&vertices)?;
        command_encoder.set_vertex_buffer(
            PathRasterizationInputIndex::Vertices as u64,
            Some(&vertex_instances.buffer),
            vertex_instances.offset as u64,
        );
        command_encoder.set_vertex_bytes(
            PathRasterizationInputIndex::ViewportSize as u64,
            mem::size_of_val(&viewport_size) as u64,
            &viewport_size as *const Size<DevicePixels> as *const _,
        );
        command_encoder.set_fragment_buffer(
            PathRasterizationInputIndex::Vertices as u64,
            Some(&vertex_instances.buffer),
            vertex_instances.offset as u64,
        );
        command_encoder.draw_primitives(
            metal::MTLPrimitiveType::Triangle,
            0,
            vertices.len() as u64,
        );

        command_encoder.end_encoding();
        Ok(true)
    }

    fn draw_shadows(
        &self,
        shadows: Range<usize>,
        instances: &FrameInstances,
        viewport_size: Size<DevicePixels>,
        command_encoder: &metal::RenderCommandEncoderRef,
    ) {
        if shadows.is_empty() {
            return;
        }

        command_encoder.set_render_pipeline_state(&self.shadows_pipeline_state);
        command_encoder.set_vertex_buffer(
            ShadowInputIndex::Vertices as u64,
            Some(&self.unit_vertices),
            0,
        );
        command_encoder.set_vertex_buffer(
            ShadowInputIndex::Shadows as u64,
            Some(&instances.shadows.buffer),
            instances.shadows.offset as u64,
        );
        command_encoder.set_fragment_buffer(
            ShadowInputIndex::Shadows as u64,
            Some(&instances.shadows.buffer),
            instances.shadows.offset as u64,
        );
        command_encoder.set_vertex_bytes(
            ShadowInputIndex::ViewportSize as u64,
            mem::size_of_val(&viewport_size) as u64,
            &viewport_size as *const Size<DevicePixels> as *const _,
        );

        command_encoder.draw_primitives_instanced_base_instance(
            metal::MTLPrimitiveType::Triangle,
            0,
            6,
            shadows.len() as u64,
            shadows.start as u64,
        );
    }

    /// Draws the quads named by `quad_indices`, a range of the frame's quad
    /// index buffer, through the blended or the opaque quad pipeline.
    fn draw_quads(
        &self,
        pipeline_state: &metal::RenderPipelineState,
        quad_indices: Range<usize>,
        instances: &FrameInstances,
        viewport_size: Size<DevicePixels>,
        command_encoder: &metal::RenderCommandEncoderRef,
    ) {
        if quad_indices.is_empty() {
            return;
        }

        command_encoder.set_render_pipeline_state(pipeline_state);
        command_encoder.set_vertex_buffer(
            QuadInputIndex::Vertices as u64,
            Some(&self.unit_vertices),
            0,
        );
        command_encoder.set_vertex_buffer(
            QuadInputIndex::Quads as u64,
            Some(&instances.quads.buffer),
            instances.quads.offset as u64,
        );
        command_encoder.set_vertex_buffer(
            QuadInputIndex::QuadIndices as u64,
            Some(&instances.quad_indices.buffer),
            instances.quad_indices.offset as u64,
        );
        command_encoder.set_fragment_buffer(
            QuadInputIndex::Quads as u64,
            Some(&instances.quads.buffer),
            instances.quads.offset as u64,
        );
        command_encoder.set_vertex_bytes(
            QuadInputIndex::ViewportSize as u64,
            mem::size_of_val(&viewport_size) as u64,
            &viewport_size as *const Size<DevicePixels> as *const _,
        );

        command_encoder.draw_primitives_instanced_base_instance(
            metal::MTLPrimitiveType::Triangle,
            0,
            6,
            quad_indices.len() as u64,
            quad_indices.start as u64,
        );
    }

    fn draw_paths_from_intermediate(
        &self,
        paths: &[Path<ScaledPixels>],
        slot_offset: Point<ScaledPixels>,
        writer: &mut InstanceBufferWriter,
        viewport_size: Size<DevicePixels>,
        command_encoder: &metal::RenderCommandEncoderRef,
    ) -> Result<()> {
        let Some(first_path) = paths.first() else {
            return Ok(());
        };
        let intermediate_texture = self
            .path_intermediate_texture
            .as_ref()
            .context("missing path intermediate texture")?;

        command_encoder.set_render_pipeline_state(&self.path_sprites_pipeline_state);
        command_encoder.set_vertex_buffer(
            SpriteInputIndex::Vertices as u64,
            Some(&self.unit_vertices),
            0,
        );
        command_encoder.set_vertex_bytes(
            SpriteInputIndex::ViewportSize as u64,
            mem::size_of_val(&viewport_size) as u64,
            &viewport_size as *const Size<DevicePixels> as *const _,
        );

        command_encoder.set_fragment_texture(
            SpriteInputIndex::AtlasTexture as u64,
            Some(intermediate_texture),
        );

        // When copying paths from the intermediate texture to the drawable,
        // each pixel must only be copied once, in case of transparent paths.
        //
        // If all paths have the same draw order, then their bounds are all
        // disjoint, so we can copy each path's bounds individually. If this
        // batch combines different draw orders, we perform a single copy
        // for a minimal spanning rect.
        let sprites;
        if paths.last().unwrap().order == first_path.order {
            sprites = paths
                .iter()
                .map(|path| PathSprite {
                    bounds: path.clipped_bounds(),
                    slot_offset,
                })
                .collect();
        } else {
            let mut bounds = first_path.clipped_bounds();
            for path in paths.iter().skip(1) {
                bounds = bounds.union(&path.clipped_bounds());
            }
            sprites = vec![PathSprite {
                bounds,
                slot_offset,
            }];
        }

        let sprite_instances = writer.write(&sprites)?;

        command_encoder.set_vertex_buffer(
            SpriteInputIndex::Sprites as u64,
            Some(&sprite_instances.buffer),
            sprite_instances.offset as u64,
        );

        command_encoder.draw_primitives_instanced(
            metal::MTLPrimitiveType::Triangle,
            0,
            6,
            sprites.len() as u64,
        );
        Ok(())
    }

    fn draw_underlines(
        &self,
        underlines: Range<usize>,
        instances: &FrameInstances,
        viewport_size: Size<DevicePixels>,
        command_encoder: &metal::RenderCommandEncoderRef,
    ) {
        if underlines.is_empty() {
            return;
        }

        command_encoder.set_render_pipeline_state(&self.underlines_pipeline_state);
        command_encoder.set_vertex_buffer(
            UnderlineInputIndex::Vertices as u64,
            Some(&self.unit_vertices),
            0,
        );
        command_encoder.set_vertex_buffer(
            UnderlineInputIndex::Underlines as u64,
            Some(&instances.underlines.buffer),
            instances.underlines.offset as u64,
        );
        command_encoder.set_fragment_buffer(
            UnderlineInputIndex::Underlines as u64,
            Some(&instances.underlines.buffer),
            instances.underlines.offset as u64,
        );
        command_encoder.set_vertex_bytes(
            UnderlineInputIndex::ViewportSize as u64,
            mem::size_of_val(&viewport_size) as u64,
            &viewport_size as *const Size<DevicePixels> as *const _,
        );

        command_encoder.draw_primitives_instanced_base_instance(
            metal::MTLPrimitiveType::Triangle,
            0,
            6,
            underlines.len() as u64,
            underlines.start as u64,
        );
    }

    fn draw_monochrome_sprites(
        &self,
        texture_id: AtlasTextureId,
        sprites: Range<usize>,
        instances: &FrameInstances,
        viewport_size: Size<DevicePixels>,
        command_encoder: &metal::RenderCommandEncoderRef,
    ) {
        if sprites.is_empty() {
            return;
        }

        let Some(texture) = self.sprite_atlas.metal_texture(texture_id) else {
            return;
        };
        let texture_size = size(
            DevicePixels(texture.width() as i32),
            DevicePixels(texture.height() as i32),
        );
        command_encoder.set_render_pipeline_state(&self.monochrome_sprites_pipeline_state);
        command_encoder.set_vertex_buffer(
            SpriteInputIndex::Vertices as u64,
            Some(&self.unit_vertices),
            0,
        );
        command_encoder.set_vertex_buffer(
            SpriteInputIndex::Sprites as u64,
            Some(&instances.monochrome_sprites.buffer),
            instances.monochrome_sprites.offset as u64,
        );
        command_encoder.set_vertex_bytes(
            SpriteInputIndex::ViewportSize as u64,
            mem::size_of_val(&viewport_size) as u64,
            &viewport_size as *const Size<DevicePixels> as *const _,
        );
        command_encoder.set_vertex_bytes(
            SpriteInputIndex::AtlasTextureSize as u64,
            mem::size_of_val(&texture_size) as u64,
            &texture_size as *const Size<DevicePixels> as *const _,
        );
        command_encoder.set_fragment_buffer(
            SpriteInputIndex::Sprites as u64,
            Some(&instances.monochrome_sprites.buffer),
            instances.monochrome_sprites.offset as u64,
        );
        command_encoder.set_fragment_texture(SpriteInputIndex::AtlasTexture as u64, Some(&texture));

        command_encoder.draw_primitives_instanced_base_instance(
            metal::MTLPrimitiveType::Triangle,
            0,
            6,
            sprites.len() as u64,
            sprites.start as u64,
        );
    }

    fn draw_polychrome_sprites(
        &self,
        texture_id: AtlasTextureId,
        sprites: Range<usize>,
        instances: &FrameInstances,
        viewport_size: Size<DevicePixels>,
        command_encoder: &metal::RenderCommandEncoderRef,
    ) {
        if sprites.is_empty() {
            return;
        }

        let Some(texture) = self.sprite_atlas.metal_texture(texture_id) else {
            return;
        };
        let texture_size = size(
            DevicePixels(texture.width() as i32),
            DevicePixels(texture.height() as i32),
        );
        command_encoder.set_render_pipeline_state(&self.polychrome_sprites_pipeline_state);
        command_encoder.set_vertex_buffer(
            SpriteInputIndex::Vertices as u64,
            Some(&self.unit_vertices),
            0,
        );
        command_encoder.set_vertex_buffer(
            SpriteInputIndex::Sprites as u64,
            Some(&instances.polychrome_sprites.buffer),
            instances.polychrome_sprites.offset as u64,
        );
        command_encoder.set_vertex_bytes(
            SpriteInputIndex::ViewportSize as u64,
            mem::size_of_val(&viewport_size) as u64,
            &viewport_size as *const Size<DevicePixels> as *const _,
        );
        command_encoder.set_vertex_bytes(
            SpriteInputIndex::AtlasTextureSize as u64,
            mem::size_of_val(&texture_size) as u64,
            &texture_size as *const Size<DevicePixels> as *const _,
        );
        command_encoder.set_fragment_buffer(
            SpriteInputIndex::Sprites as u64,
            Some(&instances.polychrome_sprites.buffer),
            instances.polychrome_sprites.offset as u64,
        );
        command_encoder.set_fragment_texture(SpriteInputIndex::AtlasTexture as u64, Some(&texture));

        command_encoder.draw_primitives_instanced_base_instance(
            metal::MTLPrimitiveType::Triangle,
            0,
            6,
            sprites.len() as u64,
            sprites.start as u64,
        );
    }

    fn draw_surfaces(
        &mut self,
        surfaces: &[PaintSurface],
        first_surface: usize,
        instances: &FrameInstances,
        viewport_size: Size<DevicePixels>,
        command_encoder: &metal::RenderCommandEncoderRef,
    ) {
        if surfaces.is_empty() {
            return;
        }

        command_encoder.set_render_pipeline_state(&self.surfaces_pipeline_state);
        command_encoder.set_vertex_buffer(
            SurfaceInputIndex::Vertices as u64,
            Some(&self.unit_vertices),
            0,
        );
        command_encoder.set_vertex_buffer(
            SurfaceInputIndex::Surfaces as u64,
            Some(&instances.surfaces.buffer),
            instances.surfaces.offset as u64,
        );
        command_encoder.set_vertex_bytes(
            SurfaceInputIndex::ViewportSize as u64,
            mem::size_of_val(&viewport_size) as u64,
            &viewport_size as *const Size<DevicePixels> as *const _,
        );

        for (index, surface) in surfaces.iter().enumerate() {
            let image_buffer = &surface.image_buffer;
            let texture_size = size(
                DevicePixels::from(CVPixelBufferGetWidth(image_buffer) as i32),
                DevicePixels::from(CVPixelBufferGetHeight(image_buffer) as i32),
            );

            assert_eq!(
                CVPixelBufferGetPixelFormatType(image_buffer),
                kCVPixelFormatType_420YpCbCr8BiPlanarFullRange
            );

            let plane_texture = |plane, format| unsafe {
                let mut texture = ptr::null_mut();
                let status = CVMetalTextureCache::create_texture_from_image(
                    None,
                    &self.core_video_texture_cache,
                    image_buffer,
                    None,
                    format,
                    CVPixelBufferGetWidthOfPlane(image_buffer, plane),
                    CVPixelBufferGetHeightOfPlane(image_buffer, plane),
                    plane,
                    ptr::NonNull::from(&mut texture),
                );
                assert_eq!(
                    status, kCVReturnSuccess,
                    "failed to create CoreVideo Metal texture"
                );
                CFRetained::<CVMetalTexture>::from_raw(
                    ptr::NonNull::new(texture).expect("CoreVideo returned a null texture"),
                )
            };
            let y_texture = plane_texture(0, objc2_metal::MTLPixelFormat::R8Unorm);
            let cb_cr_texture = plane_texture(1, objc2_metal::MTLPixelFormat::RG8Unorm);

            command_encoder.set_vertex_bytes(
                SurfaceInputIndex::TextureSize as u64,
                mem::size_of_val(&texture_size) as u64,
                &texture_size as *const Size<DevicePixels> as *const _,
            );
            command_encoder.set_fragment_texture(SurfaceInputIndex::YTexture as u64, unsafe {
                let texture = CVMetalTextureGetTexture(&y_texture)
                    .expect("CoreVideo returned no Metal texture");
                Some(metal::TextureRef::from_ptr(
                    objc2::rc::Retained::as_ptr(&texture) as *mut _,
                ))
            });
            command_encoder.set_fragment_texture(SurfaceInputIndex::CbCrTexture as u64, unsafe {
                let texture = CVMetalTextureGetTexture(&cb_cr_texture)
                    .expect("CoreVideo returned no Metal texture");
                Some(metal::TextureRef::from_ptr(
                    objc2::rc::Retained::as_ptr(&texture) as *mut _,
                ))
            });

            command_encoder.draw_primitives_instanced_base_instance(
                metal::MTLPrimitiveType::Triangle,
                0,
                6,
                1,
                (first_surface + index) as u64,
            );
        }
    }
}

fn new_command_encoder_for_texture<'a>(
    command_buffer: &'a metal::CommandBufferRef,
    texture: &'a metal::TextureRef,
    depth_texture: &metal::TextureRef,
    store_depth: bool,
    viewport_size: Size<DevicePixels>,
    clear_color: Option<metal::MTLClearColor>,
) -> &'a metal::RenderCommandEncoderRef {
    let render_pass_descriptor = metal::RenderPassDescriptor::new();
    let color_attachment = render_pass_descriptor
        .color_attachments()
        .object_at(0)
        .unwrap();
    color_attachment.set_texture(Some(texture));
    color_attachment.set_store_action(metal::MTLStoreAction::Store);

    let depth_attachment = render_pass_descriptor.depth_attachment().unwrap();
    depth_attachment.set_texture(Some(depth_texture));
    depth_attachment.set_store_action(if store_depth {
        metal::MTLStoreAction::Store
    } else {
        metal::MTLStoreAction::DontCare
    });

    if let Some(clear_color) = clear_color {
        color_attachment.set_load_action(metal::MTLLoadAction::Clear);
        color_attachment.set_clear_color(clear_color);
        depth_attachment.set_load_action(metal::MTLLoadAction::Clear);
        // Zero is reserved by `quad_depth` for the cleared buffer, so nothing
        // is occluded until an opaque quad writes over it.
        depth_attachment.set_clear_depth(0.0);
    } else {
        color_attachment.set_load_action(metal::MTLLoadAction::Load);
        if store_depth {
            depth_attachment.set_load_action(metal::MTLLoadAction::Load);
        } else {
            // A memoryless depth attachment has nothing to load. Only a
            // frame with a backdrop blur resumes its pass on one, and only
            // quads beneath its first blur take the opaque prepass, so
            // nothing drawn from here on can be occluded by what was lost.
            depth_attachment.set_load_action(metal::MTLLoadAction::Clear);
            depth_attachment.set_clear_depth(0.0);
        }
    }

    let command_encoder = command_buffer.new_render_command_encoder(render_pass_descriptor);
    set_viewport(command_encoder, viewport_size, 0.0, 1.0);
    command_encoder
}

/// The drawable a renderer presented last, for [`MetalRenderer::previous_frame_to_copy`].
struct PresentedFrame {
    texture: metal::Texture,
    content_id: u64,
    viewport_size: Size<DevicePixels>,
}

const ATTRIBUTED_OPAQUE_QUADS: u16 = 1 << 0;
const ATTRIBUTED_BLENDED_QUADS: u16 = 1 << 1;
const ATTRIBUTED_SHADOWS: u16 = 1 << 2;
const ATTRIBUTED_PATHS: u16 = 1 << 3;
const ATTRIBUTED_UNDERLINES: u16 = 1 << 4;
const ATTRIBUTED_GLYPHS: u16 = 1 << 5;
const ATTRIBUTED_IMAGES: u16 = 1 << 6;
const ATTRIBUTED_SURFACES: u16 = 1 << 7;
const ATTRIBUTED_BACKDROP_BLURS: u16 = 1 << 8;
const ATTRIBUTED_ALL: u16 = (1 << 9) - 1;
const ATTRIBUTED_KINDS: [(u16, &str); 9] = [
    (ATTRIBUTED_OPAQUE_QUADS, "opaque quads"),
    (ATTRIBUTED_BLENDED_QUADS, "blended quads"),
    (ATTRIBUTED_SHADOWS, "shadows"),
    (ATTRIBUTED_PATHS, "paths"),
    (ATTRIBUTED_UNDERLINES, "underlines"),
    (ATTRIBUTED_GLYPHS, "glyphs"),
    (ATTRIBUTED_IMAGES, "images"),
    (ATTRIBUTED_SURFACES, "surfaces"),
    (ATTRIBUTED_BACKDROP_BLURS, "backdrop blurs"),
];
/// Presents between attribution rounds: a round is eleven extra frames of
/// GPU work, so this keeps the app usable while it measures.
const ATTRIBUTION_PRESENT_STRIDE: u32 = 15;
/// Rounds per logged line.
const ATTRIBUTION_ROUNDS: u32 = 20;

/// What `FINCODE_GPU_ATTRIBUTION=1` keeps between rounds.
struct GpuAttribution {
    presents_since_round: u32,
    rounds: u32,
    target: Option<metal::Texture>,
    /// When each render of a round ended, pushed by completion handlers on
    /// a Metal thread.
    samples: Arc<Mutex<Vec<AttributionSample>>>,
}

/// The empty command buffer each round starts with, in place of a variant.
const ATTRIBUTION_FENCE: u16 = u16::MAX;

#[derive(Clone, Copy, Debug)]
struct AttributionSample {
    round: u32,
    /// Submission order within the round; the fence is slot 0.
    slot: usize,
    /// The `ATTRIBUTED_*` bits the render left out.
    skips: u16,
    /// `GPUEndTime`, in seconds.
    end: f64,
}

impl GpuAttribution {
    fn from_env() -> Option<Self> {
        std::env::var("FINCODE_GPU_ATTRIBUTION")
            .is_ok_and(|value| value == "1")
            .then(|| Self {
                presents_since_round: 0,
                rounds: 0,
                target: None,
                samples: Arc::default(),
            })
    }
}

fn attributed_kind(batch: &PrimitiveBatch) -> u16 {
    match batch {
        PrimitiveBatch::Shadows(_) => ATTRIBUTED_SHADOWS,
        PrimitiveBatch::Quads { .. } => ATTRIBUTED_BLENDED_QUADS,
        PrimitiveBatch::Paths(_) => ATTRIBUTED_PATHS,
        PrimitiveBatch::Underlines(_) => ATTRIBUTED_UNDERLINES,
        PrimitiveBatch::MonochromeSprites { .. } | PrimitiveBatch::SubpixelSprites { .. } => {
            ATTRIBUTED_GLYPHS
        }
        PrimitiveBatch::PolychromeSprites { .. } => ATTRIBUTED_IMAGES,
        PrimitiveBatch::Surfaces(_) => ATTRIBUTED_SURFACES,
    }
}

/// The `[gpu-attribution]` line: the median GPU time of the whole frame and
/// of the clear alone, then per kind what leaving it out saved, how many it
/// drew and how many window areas they cover between them.
fn gpu_attribution_line(
    samples: &[AttributionSample],
    scene: &Scene,
    viewport_size: Size<DevicePixels>,
) -> Option<String> {
    let mut ordered = samples.to_vec();
    ordered.sort_by_key(|sample| (sample.round, sample.slot));
    // A render's cost is how much later it ended than the one submitted
    // before it in its round.
    let costs: Vec<(u16, f64)> = ordered
        .windows(2)
        .filter(|pair| pair[0].round == pair[1].round && pair[0].slot + 1 == pair[1].slot)
        .map(|pair| (pair[1].skips, (pair[1].end - pair[0].end) * 1000.))
        .collect();
    let median = |skips: u16| {
        let mut times: Vec<f64> = costs
            .iter()
            .filter(|(sample_skips, _)| *sample_skips == skips)
            .map(|(_, milliseconds)| *milliseconds)
            .collect();
        times.sort_by(f64::total_cmp);
        times.get(times.len() / 2).copied()
    };
    let (Some(full), Some(clear)) = (median(0), median(ATTRIBUTED_ALL)) else {
        return None;
    };
    let area = |bounds: &Bounds<ScaledPixels>, mask: &ContentMask<ScaledPixels>| {
        let visible = bounds.intersect(&mask.bounds);
        (visible.size.width.0.max(0.) * visible.size.height.0.max(0.)) as f64
    };
    let quads = |indices: &[u32]| {
        indices
            .iter()
            .filter_map(|&index| scene.quads.get(index as usize))
            .map(|quad| area(&quad.bounds, &quad.content_mask))
            .collect::<Vec<_>>()
    };
    let coverage = |bit: u16| -> Vec<f64> {
        match bit {
            ATTRIBUTED_OPAQUE_QUADS => quads(&scene.opaque_quad_indices),
            ATTRIBUTED_BLENDED_QUADS => quads(&scene.blended_quad_indices),
            ATTRIBUTED_SHADOWS => scene.shadows.iter().map(|shadow| area(&shadow.bounds, &shadow.content_mask)).collect(),
            ATTRIBUTED_PATHS => scene.paths.iter().map(|path| area(&path.bounds, &path.content_mask)).collect(),
            ATTRIBUTED_UNDERLINES => scene.underlines.iter().map(|underline| area(&underline.bounds, &underline.content_mask)).collect(),
            ATTRIBUTED_GLYPHS => scene
                .monochrome_sprites
                .iter()
                .map(|sprite| area(&sprite.bounds, &sprite.content_mask))
                .chain(scene.subpixel_sprites.iter().map(|sprite| area(&sprite.bounds, &sprite.content_mask)))
                .collect(),
            ATTRIBUTED_IMAGES => scene.polychrome_sprites.iter().map(|sprite| area(&sprite.bounds, &sprite.content_mask)).collect(),
            ATTRIBUTED_SURFACES => scene.surfaces.iter().map(|surface| area(&surface.bounds, &surface.content_mask)).collect(),
            _ => scene.backdrop_blurs.iter().map(|blur| area(&blur.bounds, &blur.content_mask)).collect(),
        }
    };
    let window_area = (viewport_size.width.0.max(1) as f64) * (viewport_size.height.0.max(1) as f64);
    let mut line = format!(
        "[gpu-attribution] {}x{}: frame {full:.2} ms, clear alone {clear:.2} ms",
        viewport_size.width.0, viewport_size.height.0,
    );
    for (bit, name) in ATTRIBUTED_KINDS {
        let Some(without) = median(bit) else {
            continue;
        };
        let areas = coverage(bit);
        if areas.is_empty() {
            continue;
        }
        line.push_str(&format!(
            " | {name} {:.2} ms ({} over {:.1}x the window)",
            full - without,
            areas.len(),
            areas.iter().sum::<f64>() / window_area,
        ));
    }
    Some(line)
}

fn scissor_rect(rect: &Bounds<DevicePixels>) -> metal::MTLScissorRect {
    metal::MTLScissorRect {
        x: rect.origin.x.0.max(0) as u64,
        y: rect.origin.y.0.max(0) as u64,
        width: rect.size.width.0.max(0) as u64,
        height: rect.size.height.0.max(0) as u64,
    }
}

fn scissor_reaches_blur(scissor: &metal::MTLScissorRect, blur: &BackdropBlur) -> bool {
    let visible = blur.bounds.intersect(&blur.content_mask.bounds);
    let (left, top) = (scissor.x as f32, scissor.y as f32);
    let (right, bottom) = (left + scissor.width as f32, top + scissor.height as f32);
    visible.size.width.0 > 0.
        && visible.size.height.0 > 0.
        && visible.origin.x.0 < right
        && left < visible.origin.x.0 + visible.size.width.0
        && visible.origin.y.0 < bottom
        && top < visible.origin.y.0 + visible.size.height.0
}

/// The viewport outside `damage`, whose rectangles lie inside it and do not
/// overlap, as rectangles for a blit to copy: the rows between consecutive
/// damage edges, split around the damage those rows cross.
fn undamaged_rects(
    damage: &[Bounds<DevicePixels>],
    viewport: Size<DevicePixels>,
) -> Vec<Bounds<DevicePixels>> {
    let (width, height) = (viewport.width.0, viewport.height.0);
    let mut edges = Vec::with_capacity(damage.len() * 2 + 2);
    edges.extend([0, height]);
    for rect in damage {
        edges.extend([rect.top().0.clamp(0, height), rect.bottom().0.clamp(0, height)]);
    }
    edges.sort_unstable();
    edges.dedup();
    let mut undamaged = Vec::new();
    let mut crossed = Vec::with_capacity(damage.len());
    for rows in edges.windows(2) {
        let (top, bottom) = (rows[0], rows[1]);
        crossed.clear();
        crossed.extend(
            damage
                .iter()
                .filter(|rect| rect.top().0 <= top && rect.bottom().0 >= bottom)
                .map(|rect| (rect.left().0.clamp(0, width), rect.right().0.clamp(0, width))),
        );
        crossed.sort_unstable();
        let mut left = 0;
        let mut push = |left: i32, right: i32| {
            undamaged.push(Bounds {
                origin: point(DevicePixels(left), DevicePixels(top)),
                size: size(DevicePixels(right - left), DevicePixels(bottom - top)),
            });
        };
        for &(damage_left, damage_right) in &crossed {
            if damage_left > left {
                push(left, damage_left);
            }
            left = left.max(damage_right);
        }
        if left < width {
            push(left, width);
        }
    }
    undamaged
}

fn set_viewport(
    command_encoder: &metal::RenderCommandEncoderRef,
    viewport_size: Size<DevicePixels>,
    near_depth: f64,
    far_depth: f64,
) {
    command_encoder.set_viewport(metal::MTLViewport {
        originX: 0.0,
        originY: 0.0,
        width: i32::from(viewport_size.width) as f64,
        height: i32::from(viewport_size.height) as f64,
        znear: near_depth,
        zfar: far_depth,
    });
}

#[cfg(any(test, feature = "bench-support", feature = "test-support"))]
fn read_texture_to_image(texture: &metal::TextureRef) -> Result<RgbaImage> {
    let width = texture.width() as u32;
    let height = texture.height() as u32;
    let bytes_per_row = width as usize * 4;
    let mut pixels = vec![0u8; height as usize * bytes_per_row];

    let region = metal::MTLRegion {
        origin: metal::MTLOrigin { x: 0, y: 0, z: 0 },
        size: metal::MTLSize {
            width: width as u64,
            height: height as u64,
            depth: 1,
        },
    };
    texture.get_bytes(
        pixels.as_mut_ptr() as *mut std::ffi::c_void,
        bytes_per_row as u64,
        region,
        0,
    );

    // Convert BGRA to RGBA (swap B and R channels)
    for chunk in pixels.chunks_exact_mut(4) {
        chunk.swap(0, 2);
    }

    RgbaImage::from_raw(width, height, pixels).context("failed to create RgbaImage from pixel data")
}

fn build_pipeline_state(
    device: &metal::DeviceRef,
    library: &metal::LibraryRef,
    label: &str,
    vertex_fn_name: &str,
    fragment_fn_name: &str,
    pixel_format: metal::MTLPixelFormat,
) -> metal::RenderPipelineState {
    build_pipeline_state_with_blend(
        device,
        library,
        label,
        vertex_fn_name,
        fragment_fn_name,
        pixel_format,
        // The ordinary "over" blend. The shaders emit straight colour, so
        // `rgb = src * a + dst * (1 - a)` with `alpha = a + dst_a * (1 - a)` is
        // exactly premultiplied-over against the premultiplied drawable. The
        // alpha term only shows on a target that is not already opaque — the
        // transparent overlay layer above a native view, or a glass window's
        // translucent surfaces — where an additive destination factor let a
        // translucent surface over its own shadow saturate to alpha 1 while
        // its colour still carried only the surface's share: the surface over
        // black rather than over the page.
        metal::MTLBlendFactor::One,
        metal::MTLBlendFactor::OneMinusSourceAlpha,
    )
}

fn build_pipeline_state_with_blend(
    device: &metal::DeviceRef,
    library: &metal::LibraryRef,
    label: &str,
    vertex_fn_name: &str,
    fragment_fn_name: &str,
    pixel_format: metal::MTLPixelFormat,
    source_alpha_blend_factor: metal::MTLBlendFactor,
    destination_alpha_blend_factor: metal::MTLBlendFactor,
) -> metal::RenderPipelineState {
    let vertex_fn = library
        .get_function(vertex_fn_name, None)
        .expect("error locating vertex function");
    let fragment_fn = library
        .get_function(fragment_fn_name, None)
        .expect("error locating fragment function");

    let descriptor = metal::RenderPipelineDescriptor::new();
    descriptor.set_label(label);
    descriptor.set_vertex_function(Some(vertex_fn.as_ref()));
    descriptor.set_fragment_function(Some(fragment_fn.as_ref()));
    descriptor.set_depth_attachment_pixel_format(DEPTH_FORMAT);
    let color_attachment = descriptor.color_attachments().object_at(0).unwrap();
    color_attachment.set_pixel_format(pixel_format);
    color_attachment.set_blending_enabled(true);
    color_attachment.set_rgb_blend_operation(metal::MTLBlendOperation::Add);
    color_attachment.set_alpha_blend_operation(metal::MTLBlendOperation::Add);
    color_attachment.set_source_rgb_blend_factor(metal::MTLBlendFactor::SourceAlpha);
    // `(Zero, One)` keeps the destination alpha unchanged (glass content, so
    // its anti-aliased edge does not punch through the translucent glass
    // surface beneath it); `(One, OneMinusSourceAlpha)` is premultiplied-over.
    color_attachment.set_source_alpha_blend_factor(source_alpha_blend_factor);
    color_attachment.set_destination_rgb_blend_factor(metal::MTLBlendFactor::OneMinusSourceAlpha);
    color_attachment.set_destination_alpha_blend_factor(destination_alpha_blend_factor);

    device
        .new_render_pipeline_state(&descriptor)
        .expect("could not create render pipeline state")
}

fn build_opaque_quads_pipeline_state(
    device: &metal::DeviceRef,
    library: &metal::LibraryRef,
    label: &str,
    vertex_fn_name: &str,
    fragment_fn_name: &str,
    pixel_format: metal::MTLPixelFormat,
) -> metal::RenderPipelineState {
    let vertex_fn = library
        .get_function(vertex_fn_name, None)
        .expect("error locating vertex function");
    let fragment_fn = library
        .get_function(fragment_fn_name, None)
        .expect("error locating fragment function");

    let descriptor = metal::RenderPipelineDescriptor::new();
    descriptor.set_label(label);
    descriptor.set_vertex_function(Some(vertex_fn.as_ref()));
    descriptor.set_fragment_function(Some(fragment_fn.as_ref()));
    descriptor.set_depth_attachment_pixel_format(DEPTH_FORMAT);
    let color_attachment = descriptor.color_attachments().object_at(0).unwrap();
    color_attachment.set_pixel_format(pixel_format);
    color_attachment.set_blending_enabled(false);

    device
        .new_render_pipeline_state(&descriptor)
        .expect("could not create render pipeline state")
}

/// The backdrop blur composite: dual-source blending, `source + destination
/// * (1 - source1.a)`, which with the fragment's outputs is
/// `mix(destination, frosted, coverage)` on every channel. The blurred region
/// replaces what was there, alpha included, so a frosted region over a
/// translucent window keeps the window's translucency instead of adding to it.
/// `None`, logged, on a device whose driver refuses the dual-source blend.
fn build_backdrop_blur_pipeline_state(
    device: &metal::DeviceRef,
    library: &metal::LibraryRef,
    pixel_format: metal::MTLPixelFormat,
) -> Option<metal::RenderPipelineState> {
    let vertex_fn = library
        .get_function("backdrop_blur_vertex", None)
        .expect("error locating vertex function");
    let fragment_fn = library
        .get_function("backdrop_blur_fragment", None)
        .expect("error locating fragment function");

    let descriptor = metal::RenderPipelineDescriptor::new();
    descriptor.set_label("backdrop_blurs");
    descriptor.set_vertex_function(Some(vertex_fn.as_ref()));
    descriptor.set_fragment_function(Some(fragment_fn.as_ref()));
    descriptor.set_depth_attachment_pixel_format(DEPTH_FORMAT);
    let color_attachment = descriptor.color_attachments().object_at(0).unwrap();
    color_attachment.set_pixel_format(pixel_format);
    color_attachment.set_blending_enabled(true);
    color_attachment.set_rgb_blend_operation(metal::MTLBlendOperation::Add);
    color_attachment.set_alpha_blend_operation(metal::MTLBlendOperation::Add);
    color_attachment.set_source_rgb_blend_factor(metal::MTLBlendFactor::One);
    color_attachment.set_source_alpha_blend_factor(metal::MTLBlendFactor::One);
    color_attachment.set_destination_rgb_blend_factor(metal::MTLBlendFactor::OneMinusSource1Alpha);
    color_attachment.set_destination_alpha_blend_factor(metal::MTLBlendFactor::OneMinusSource1Alpha);

    device
        .new_render_pipeline_state(&descriptor)
        .map_err(|error| log::error!("backdrop blurs are unavailable on this device: {error}"))
        .ok()
}

/// One of the two backdrop blur passes: unblended, into a scratch texture,
/// with no depth attachment.
fn build_backdrop_blur_pass_pipeline_state(
    device: &metal::DeviceRef,
    library: &metal::LibraryRef,
    label: &str,
    fragment_fn_name: &str,
    pixel_format: metal::MTLPixelFormat,
) -> metal::RenderPipelineState {
    let vertex_fn = library
        .get_function("backdrop_blur_pass_vertex", None)
        .expect("error locating vertex function");
    let fragment_fn = library
        .get_function(fragment_fn_name, None)
        .expect("error locating fragment function");

    let descriptor = metal::RenderPipelineDescriptor::new();
    descriptor.set_label(label);
    descriptor.set_vertex_function(Some(vertex_fn.as_ref()));
    descriptor.set_fragment_function(Some(fragment_fn.as_ref()));
    let color_attachment = descriptor.color_attachments().object_at(0).unwrap();
    color_attachment.set_pixel_format(pixel_format);
    color_attachment.set_blending_enabled(false);

    device
        .new_render_pipeline_state(&descriptor)
        .expect("could not create render pipeline state")
}

fn build_depth_stencil_state(
    device: &metal::DeviceRef,
    write_depth: bool,
) -> metal::DepthStencilState {
    let descriptor = metal::DepthStencilDescriptor::new();
    descriptor.set_depth_compare_function(MTLCompareFunction::Greater);
    descriptor.set_depth_write_enabled(write_depth);
    device.new_depth_stencil_state(&descriptor)
}

fn build_path_sprite_pipeline_state(
    device: &metal::DeviceRef,
    library: &metal::LibraryRef,
    label: &str,
    vertex_fn_name: &str,
    fragment_fn_name: &str,
    pixel_format: metal::MTLPixelFormat,
) -> metal::RenderPipelineState {
    let vertex_fn = library
        .get_function(vertex_fn_name, None)
        .expect("error locating vertex function");
    let fragment_fn = library
        .get_function(fragment_fn_name, None)
        .expect("error locating fragment function");

    let descriptor = metal::RenderPipelineDescriptor::new();
    descriptor.set_label(label);
    descriptor.set_vertex_function(Some(vertex_fn.as_ref()));
    descriptor.set_fragment_function(Some(fragment_fn.as_ref()));
    descriptor.set_depth_attachment_pixel_format(DEPTH_FORMAT);
    let color_attachment = descriptor.color_attachments().object_at(0).unwrap();
    color_attachment.set_pixel_format(pixel_format);
    color_attachment.set_blending_enabled(true);
    color_attachment.set_rgb_blend_operation(metal::MTLBlendOperation::Add);
    color_attachment.set_alpha_blend_operation(metal::MTLBlendOperation::Add);
    // Path sprites arrive premultiplied from the intermediate, so this is the
    // same premultiplied-over as `build_pipeline_state`, alpha term included.
    color_attachment.set_source_rgb_blend_factor(metal::MTLBlendFactor::One);
    color_attachment.set_source_alpha_blend_factor(metal::MTLBlendFactor::One);
    color_attachment.set_destination_rgb_blend_factor(metal::MTLBlendFactor::OneMinusSourceAlpha);
    color_attachment.set_destination_alpha_blend_factor(metal::MTLBlendFactor::OneMinusSourceAlpha);

    device
        .new_render_pipeline_state(&descriptor)
        .expect("could not create render pipeline state")
}

fn build_path_rasterization_pipeline_state(
    device: &metal::DeviceRef,
    library: &metal::LibraryRef,
    label: &str,
    vertex_fn_name: &str,
    fragment_fn_name: &str,
    pixel_format: metal::MTLPixelFormat,
    path_sample_count: u32,
) -> metal::RenderPipelineState {
    let vertex_fn = library
        .get_function(vertex_fn_name, None)
        .expect("error locating vertex function");
    let fragment_fn = library
        .get_function(fragment_fn_name, None)
        .expect("error locating fragment function");

    let descriptor = metal::RenderPipelineDescriptor::new();
    descriptor.set_label(label);
    descriptor.set_vertex_function(Some(vertex_fn.as_ref()));
    descriptor.set_fragment_function(Some(fragment_fn.as_ref()));
    if path_sample_count > 1 {
        descriptor.set_raster_sample_count(path_sample_count as _);
        descriptor.set_alpha_to_coverage_enabled(false);
    }
    let color_attachment = descriptor.color_attachments().object_at(0).unwrap();
    color_attachment.set_pixel_format(pixel_format);
    color_attachment.set_blending_enabled(true);
    color_attachment.set_rgb_blend_operation(metal::MTLBlendOperation::Add);
    color_attachment.set_alpha_blend_operation(metal::MTLBlendOperation::Add);
    color_attachment.set_source_rgb_blend_factor(metal::MTLBlendFactor::One);
    color_attachment.set_source_alpha_blend_factor(metal::MTLBlendFactor::One);
    color_attachment.set_destination_rgb_blend_factor(metal::MTLBlendFactor::OneMinusSourceAlpha);
    color_attachment.set_destination_alpha_blend_factor(metal::MTLBlendFactor::OneMinusSourceAlpha);

    device
        .new_render_pipeline_state(&descriptor)
        .expect("could not create render pipeline state")
}

#[derive(Clone)]
struct BufferRegion {
    buffer: metal::Buffer,
    offset: usize,
}

struct FrameInstances {
    rounded_clips: BufferRegion,
    quads: BufferRegion,
    quad_indices: BufferRegion,
    shadows: BufferRegion,
    underlines: BufferRegion,
    monochrome_sprites: BufferRegion,
    polychrome_sprites: BufferRegion,
    surfaces: BufferRegion,
}

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

fn upload_frame_instances(
    scene: &Scene,
    writer: &mut InstanceBufferWriter,
) -> Result<FrameInstances> {
    Ok(FrameInstances {
        rounded_clips: if scene.rounded_clips.is_empty() {
            writer.write(&[gpui::RoundedClipNode::default()])?
        } else { writer.write(&scene.rounded_clips)? },
        quads: writer.write(&scene.quads)?,
        quad_indices: writer.write_slices(&[
            &scene.blended_quad_indices,
            &scene.opaque_quad_indices,
        ])?,
        shadows: writer.write(&scene.shadows)?,
        underlines: writer.write(&scene.underlines)?,
        monochrome_sprites: writer.write(&scene.monochrome_sprites)?,
        polychrome_sprites: writer.write(&scene.polychrome_sprites)?,
        surfaces: writer.write_iter(scene.surfaces.iter().map(|surface| SurfaceBounds {
            bounds: surface.bounds,
            content_mask: surface.content_mask,
            rounded_clip: surface.rounded_clip,
            clip_padding: 0,
        }))?,
    })
}

/// Packs a frame's instance data into pooled Metal buffers.
/// When the current buffer runs out of room the writer moves on to a larger one.
struct InstanceBufferWriter {
    device: metal::Device,
    pool: Arc<Mutex<InstanceBufferPool>>,
    unified_memory: bool,
    /// Buffers the frame has finished with, and how many bytes each holds.
    filled: Vec<(InstanceBuffer, usize)>,
    current: InstanceBuffer,
    offset: usize,
}

impl InstanceBufferWriter {
    fn new(
        device: &metal::Device,
        pool: &Arc<Mutex<InstanceBufferPool>>,
        unified_memory: bool,
    ) -> Self {
        let current = pool.lock().acquire(device, unified_memory);
        Self {
            device: device.clone(),
            pool: pool.clone(),
            unified_memory,
            filled: Vec::new(),
            current,
            offset: 0,
        }
    }

    fn allocate<T>(&mut self, count: usize) -> Result<(BufferRegion, &mut [MaybeUninit<T>])> {
        let size = mem::size_of::<T>() * count;
        let mut offset = self.offset.next_multiple_of(INSTANCE_BUFFER_ALIGNMENT);
        if offset + size > self.current.size {
            self.grow(size)?;
            offset = 0;
        }
        self.offset = offset + size;

        let region = BufferRegion {
            buffer: self.current.metal_buffer.clone(),
            offset,
        };
        // Safety: the reservation lies within a buffer this frame owns
        // exclusively, and never overlaps one handed out earlier.
        let values = unsafe {
            let start = (self.current.metal_buffer.contents() as *mut u8).add(offset);
            slice::from_raw_parts_mut(start.cast::<MaybeUninit<T>>(), count)
        };
        Ok((region, values))
    }

    fn write<T>(&mut self, values: &[T]) -> Result<BufferRegion> {
        let (region, destination) = self.allocate::<T>(values.len())?;
        unsafe {
            ptr::copy_nonoverlapping(
                values.as_ptr(),
                destination.as_mut_ptr().cast::<T>(),
                values.len(),
            );
        }
        Ok(region)
    }

    /// Packs several slices end to end into a single reservation, so the shader
    /// can index across all of them with one binding.
    fn write_slices<T>(&mut self, slices: &[&[T]]) -> Result<BufferRegion> {
        let count = slices.iter().map(|slice| slice.len()).sum();
        let (region, destination) = self.allocate::<T>(count)?;
        let mut written = 0;
        for slice in slices {
            unsafe {
                ptr::copy_nonoverlapping(
                    slice.as_ptr(),
                    destination.as_mut_ptr().add(written).cast::<T>(),
                    slice.len(),
                );
            }
            written += slice.len();
        }
        Ok(region)
    }

    fn write_iter<T>(&mut self, values: impl ExactSizeIterator<Item = T>) -> Result<BufferRegion> {
        let (region, destination) = self.allocate::<T>(values.len())?;
        for (slot, value) in destination.iter_mut().zip(values) {
            slot.write(value);
        }
        Ok(region)
    }

    /// Replaces the current buffer with one large enough for a `required`-byte reservation.
    fn grow(&mut self, required: usize) -> Result<()> {
        let mut pool = self.pool.lock();
        let buffer_size = (pool.buffer_size * 2)
            .max(required.next_power_of_two())
            .min(MAX_INSTANCE_BUFFER_SIZE);
        anyhow::ensure!(
            buffer_size >= required,
            "instance buffer needs {required} bytes, above the maximum of {MAX_INSTANCE_BUFFER_SIZE}"
        );
        log::info!("increased instance buffer size to {buffer_size}");

        pool.reset(buffer_size);
        let buffer = pool.acquire(&self.device, self.unified_memory);
        drop(pool);

        let filled = mem::replace(&mut self.current, buffer);
        self.filled.push((filled, self.offset));
        self.offset = 0;
        Ok(())
    }

    /// Flushes the frame's writes and returns the buffer worth recycling.
    fn finish(self) -> InstanceBuffer {
        let Self {
            pool,
            unified_memory,
            mut filled,
            current,
            offset,
            ..
        } = self;
        filled.push((current, offset));
        pool.lock()
            .record_frame_usage(filled.iter().map(|(_, written)| *written).sum());

        if !unified_memory {
            for (buffer, written) in &filled {
                // An allocation larger than the whole buffer retires it before
                // anything has been written to it.
                if *written == 0 {
                    continue;
                }
                buffer.metal_buffer.did_modify_range(NSRange {
                    location: 0,
                    length: *written as NSUInteger,
                });
            }
        }

        let (current, _) = filled.pop().expect("the current buffer was just pushed");
        current
    }
}

#[repr(C)]
enum ShadowInputIndex {
    Vertices = 0,
    Shadows = 1,
    ViewportSize = 2,
}

#[repr(C)]
enum BackdropBlurInputIndex {
    Vertices = 0,
    Blurs = 1,
    ViewportSize = 2,
    Pass = 3,
    Blurred = 4,
}

#[repr(C)]
enum BackdropBlurPassInputIndex {
    Vertices = 0,
    Pass = 1,
    Taps = 2,
    Source = 3,
}

#[repr(C)]
enum QuadInputIndex {
    Vertices = 0,
    Quads = 1,
    ViewportSize = 2,
    QuadIndices = 3,
}

#[repr(C)]
enum UnderlineInputIndex {
    Vertices = 0,
    Underlines = 1,
    ViewportSize = 2,
}

#[repr(C)]
enum SpriteInputIndex {
    Vertices = 0,
    Sprites = 1,
    ViewportSize = 2,
    AtlasTextureSize = 3,
    AtlasTexture = 4,
}

#[repr(C)]
enum SurfaceInputIndex {
    Vertices = 0,
    Surfaces = 1,
    ViewportSize = 2,
    TextureSize = 3,
    YTexture = 4,
    CbCrTexture = 5,
}

#[repr(C)]
enum PathRasterizationInputIndex {
    Vertices = 0,
    ViewportSize = 1,
    /// Translation from screen space into the batch's slot of the path
    /// intermediate, see [`MetalRenderer::plan_path_batches`].
    SlotOffset = 2,
}

#[derive(Clone, Debug, Eq, PartialEq)]
#[repr(C)]
pub struct PathSprite {
    pub bounds: Bounds<ScaledPixels>,
    /// Where this sprite's pixels sit in the path intermediate relative to
    /// their screen position.
    pub slot_offset: Point<ScaledPixels>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
#[repr(C)]
pub struct SurfaceBounds {
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub rounded_clip: u32,
    pub clip_padding: u32,
}

#[cfg(any(test, feature = "bench-support", feature = "test-support"))]
pub struct MetalHeadlessRenderer {
    renderer: MetalRenderer,
}

#[cfg(any(test, feature = "bench-support", feature = "test-support"))]
impl MetalHeadlessRenderer {
    pub fn new() -> Self {
        let instance_buffer_pool = Arc::new(Mutex::new(InstanceBufferPool::default()));
        let renderer = MetalRenderer::new_headless(instance_buffer_pool);
        Self { renderer }
    }
}

#[cfg(any(test, feature = "bench-support", feature = "test-support"))]
impl gpui::PlatformHeadlessRenderer for MetalHeadlessRenderer {
    fn render_scene_to_image(
        &mut self,
        scene: &Scene,
        size: Size<DevicePixels>,
    ) -> anyhow::Result<image::RgbaImage> {
        self.renderer.render_scene_to_image(scene, size)
    }

    fn render_scene(&mut self, scene: &Scene, size: Size<DevicePixels>) -> anyhow::Result<()> {
        self.renderer.render_scene(scene, size)
    }

    fn sprite_atlas(&self) -> Arc<dyn gpui::PlatformAtlas> {
        self.renderer.sprite_atlas().clone()
    }
}

/// The ordinary blend must be premultiplied-over in alpha as well as colour.
/// Over a transparent target (the overlay layer above a native view, a glass
/// window's translucent surfaces) an additive destination alpha saturates the
/// alpha of stacked translucent primitives while their colour still carries
/// only their own share, so the composited pixel reads as the top surface over
/// black — a popup on the overlay plane darker and less see-through than the
/// same popup on the base plane.
#[cfg(test)]
mod alpha_blend_tests {
    use super::{InstanceBufferPool, MetalRenderer};
    use gpui::{
        Background, Bounds, ContentMask, Corners, DevicePixels, Edges, Hsla, Point, Quad,
        ScaledPixels, Scene,
    };
    use parking_lot::Mutex;
    use std::sync::Arc;

    const EXTENT: f32 = 64.;

    fn translucent_quad(order: u32, color: Hsla) -> Quad {
        let bounds = Bounds {
            origin: Point {
                x: ScaledPixels(0.),
                y: ScaledPixels(0.),
            },
            size: gpui::size(ScaledPixels(EXTENT), ScaledPixels(EXTENT)),
        };
        Quad {
            rounded_clip: 0,
            clip_padding: 0,
            order,
            border_style: Default::default(),
            bounds,
            content_mask: ContentMask { bounds },
            background: Background::from(color),
            border_color: Hsla::transparent_black(),
            corner_radii: Corners::default(),
            border_widths: Edges::default(),
        }
    }

    #[test]
    fn translucent_quads_over_a_transparent_target_composite_as_over() {
        let mut renderer = MetalRenderer::new_headless(Arc::new(Mutex::new(
            InstanceBufferPool::default(),
        )));
        renderer.update_transparency(true);

        // A half-transparent black "shadow" with a half-transparent white
        // surface over it, the way a popup's box shadow sits under its fill.
        let mut scene = Scene::default();
        scene
            .quads
            .push(translucent_quad(0, Hsla::black().opacity(0.5)));
        scene
            .quads
            .push(translucent_quad(1, Hsla::white().opacity(0.5)));
        scene.finish();

        let image = renderer
            .render_scene_to_image(
                &scene,
                gpui::size(DevicePixels(EXTENT as i32), DevicePixels(EXTENT as i32)),
            )
            .expect("headless render succeeds");
        let pixel = image.get_pixel(EXTENT as u32 / 2, EXTENT as u32 / 2);

        // Premultiplied over: colour 0.5 (the white surface's own share), alpha
        // 0.5 + 0.5 * (1 - 0.5) = 0.75. The additive blend produced alpha 1.0
        // for the same colour: white at half strength over black, opaque.
        let expected_alpha = (0.75 * 255.0f32).round() as i32;
        let expected_color = (0.5 * 255.0f32).round() as i32;
        assert!(
            (pixel[3] as i32 - expected_alpha).abs() <= 2,
            "alpha must be the over-composite of the two quads, got {pixel:?}"
        );
        assert!(
            (0..3).all(|channel| (pixel[channel] as i32 - expected_color).abs() <= 2),
            "colour must be the white surface's premultiplied share, got {pixel:?}"
        );
    }
}

#[cfg(test)]
mod submission_retention_tests {
    use super::{InFlightSubmission, InstanceBufferPool, MAX_UNFINISHED_SUBMISSIONS, MetalRenderer};
    use gpui::{PlatformAtlas, PresentOutcome, Scene};
    use parking_lot::Mutex;
    use std::{
        cell::Cell,
        sync::{Arc, atomic::{AtomicU32, Ordering}},
    };

    #[test]
    fn unfinished_callbacks_hold_capacity_until_they_return_resources() {
        let unfinished = Arc::new(AtomicU32::new(0));
        let mut submissions = Vec::new();
        for _ in 0..MAX_UNFINISHED_SUBMISSIONS {
            submissions.push(InFlightSubmission::try_acquire(&unfinished).expect("capacity"));
        }
        assert!(InFlightSubmission::try_acquire(&unfinished).is_none());
        let callback = Cell::new(submissions.pop());
        assert!(InFlightSubmission::try_acquire(&unfinished).is_none());
        drop(callback.take());
        let replacement = InFlightSubmission::try_acquire(&unfinished).expect("returned capacity");
        drop(callback.take());
        assert_eq!(unfinished.load(Ordering::Relaxed), MAX_UNFINISHED_SUBMISSIONS);
        drop(replacement);
        drop(submissions);
        assert_eq!(unfinished.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn abandoned_submission_and_callback_return_capacity() {
        let unfinished = Arc::new(AtomicU32::new(0));
        let abandoned = InFlightSubmission::try_acquire(&unfinished);
        drop(abandoned);
        assert_eq!(unfinished.load(Ordering::Relaxed), 0);
        let callback = Cell::new(InFlightSubmission::try_acquire(&unfinished));
        drop(callback);
        assert_eq!(unfinished.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn full_submission_queue_defers_before_acquiring_a_drawable_or_rendering() {
        if metal::Device::system_default().is_none() {
            return;
        }
        let pool = Arc::new(Mutex::new(InstanceBufferPool::default()));
        let mut renderer = MetalRenderer::new(Arc::clone(&pool), false, false);
        renderer.update_drawable_size(gpui::size(gpui::DevicePixels(64), gpui::DevicePixels(64)));
        let submissions: Vec<_> = (0..MAX_UNFINISHED_SUBMISSIONS)
            .map(|_| InFlightSubmission::try_acquire(&renderer.unfinished_submissions))
            .collect();
        let atlas_frame = renderer.sprite_atlas.frame_index();
        assert!(!renderer.can_draw());
        assert_eq!(renderer.draw(&Scene::default()).outcome, PresentOutcome::Deferred);
        assert!(pool.lock().buffers.is_empty());
        assert_eq!(renderer.sprite_atlas.frame_index(), atlas_frame);
        for _ in 1..super::MAX_CONSECUTIVE_DEFERRED_PRESENTS {
            assert_eq!(renderer.draw(&Scene::default()).outcome, PresentOutcome::Deferred);
        }
        assert_eq!(renderer.draw(&Scene::default()).outcome, PresentOutcome::Dropped);
        assert!(!renderer.can_draw(), "dropping the frame must not retire retained resources");
        drop(submissions);
        assert!(renderer.can_draw());
    }

    #[test]
    fn layered_preflight_bounds_retries_without_retiring_unfinished_submissions() {
        if metal::Device::system_default().is_none() {
            return;
        }
        let mut renderer =
            MetalRenderer::new(Arc::new(Mutex::new(InstanceBufferPool::default())), false, false);
        let submissions: Vec<_> = (0..MAX_UNFINISHED_SUBMISSIONS)
            .map(|_| InFlightSubmission::try_acquire(&renderer.unfinished_submissions))
            .collect();
        for _ in 0..super::MAX_CONSECUTIVE_DEFERRED_PRESENTS {
            assert_eq!(renderer.submission_queue_full_outcome(), Some(PresentOutcome::Deferred));
        }
        assert_eq!(renderer.submission_queue_full_outcome(), Some(PresentOutcome::Dropped));
        assert!(!renderer.can_draw());
        drop(submissions);
        assert_eq!(renderer.submission_queue_full_outcome(), None);
        assert!(renderer.can_draw());
    }
}

#[cfg(test)]
mod instance_buffer_pool_tests {
    use super::{
        INSTANCE_BUFFER_FLOOR_SIZE, INSTANCE_BUFFER_SHRINK_AFTER_FRAMES, InstanceBufferPool,
    };
    use gpui::RenderMemoryGauge;

    fn pool_at(buffer_size: usize) -> InstanceBufferPool {
        let mut pool = InstanceBufferPool::default();
        pool.reset(buffer_size);
        pool
    }

    /// The pool reports the buffers waiting in it — not the ones a frame has
    /// out — and a size change retires the old ones from the report too.
    #[test]
    fn pooled_buffers_are_reported_to_the_render_memory_gauges() {
        let Some(device) = metal::Device::system_default() else {
            return;
        };
        let unified = device.has_unified_memory();
        let gauge = RenderMemoryGauge::InstanceBuffers;
        let floor = INSTANCE_BUFFER_FLOOR_SIZE as u64;
        let mut pool = pool_at(INSTANCE_BUFFER_FLOOR_SIZE);
        assert_eq!(pool.render_memory.published(gauge), 0);

        let first = pool.acquire(&device, unified);
        let second = pool.acquire(&device, unified);
        assert_eq!(pool.render_memory.published(gauge), 0);

        pool.release(first);
        assert_eq!(pool.render_memory.published(gauge), floor);
        assert!(gpui::render_memory_gauges().instance_buffer_bytes >= floor);

        // Growing drops the idle buffers of the old size, and one of that size
        // coming back is dropped rather than pooled.
        pool.reset(INSTANCE_BUFFER_FLOOR_SIZE * 2);
        assert_eq!(pool.render_memory.published(gauge), 0);
        pool.release(second);
        assert_eq!(pool.render_memory.published(gauge), 0);

        let third = pool.acquire(&device, unified);
        pool.release(third);
        assert_eq!(pool.render_memory.published(gauge), floor * 2);
    }

    /// One large frame inside the window — a big window drawing once while a
    /// small one animates — keeps the buffer at its size.
    #[test]
    fn a_single_large_frame_in_the_window_vetoes_the_shrink() {
        let size = INSTANCE_BUFFER_FLOOR_SIZE * 4;
        let mut pool = pool_at(size);
        for frame in 0..INSTANCE_BUFFER_SHRINK_AFTER_FRAMES * 3 {
            let used = if frame % INSTANCE_BUFFER_SHRINK_AFTER_FRAMES == 7 {
                size * 3 / 4
            } else {
                1024
            };
            pool.record_frame_usage(used);
            assert_eq!(pool.buffer_size, size, "frame {frame} shrank the buffer");
        }
    }

    /// A whole window of small frames halves it, and it never goes below the floor.
    #[test]
    fn a_window_of_small_frames_halves_the_buffer_down_to_the_floor() {
        let size = INSTANCE_BUFFER_FLOOR_SIZE * 4;
        let mut pool = pool_at(size);
        for _ in 0..INSTANCE_BUFFER_SHRINK_AFTER_FRAMES {
            pool.record_frame_usage(1024);
        }
        assert_eq!(pool.buffer_size, size / 2);
        for _ in 0..INSTANCE_BUFFER_SHRINK_AFTER_FRAMES * 4 {
            pool.record_frame_usage(1024);
        }
        assert_eq!(pool.buffer_size, INSTANCE_BUFFER_FLOOR_SIZE);
    }
}

#[cfg(test)]
mod render_memory_tests {
    use super::{InstanceBufferPool, MetalRenderer};
    use gpui::{DevicePixels, RenderMemoryGauge};
    use parking_lot::Mutex;
    use std::sync::Arc;

    fn headless() -> MetalRenderer {
        MetalRenderer::new_headless(Arc::new(Mutex::new(InstanceBufferPool::default())))
    }

    /// A renderer reports the depth attachments it holds, follows them as they
    /// are released and as they are lent to an overlay renderer and back, and
    /// the process-wide gauge carries at least what it reports.
    #[test]
    fn depth_attachments_are_reported_by_whichever_renderer_holds_them() {
        let gauge = RenderMemoryGauge::DepthTextures;
        let full_screen = gpui::size(DevicePixels(3456), DevicePixels(2168));
        let mut base = headless();
        assert_eq!(base.render_memory.published(gauge), 0);

        base.update_drawable_size(full_screen);
        base.ensure_fallback_depth_texture(full_screen);
        base.publish_depth_memory();
        let depth = base
            .depth_texture
            .as_ref()
            .map_or(0, |texture| texture.allocated_size());
        let fallback = base
            .fallback_depth_texture
            .as_ref()
            .map_or(0, |texture| texture.allocated_size());
        // The fallback is a private texture: two bytes per pixel, never tile memory.
        assert!(
            fallback >= 3456 * 2168 * 2,
            "a 16-bit private depth texture at 3456x2168 holds ~15 MB, got {fallback}"
        );
        let held = depth + fallback;
        println!(
            "depth attachments at {}x{}: depth {} bytes ({:?}), fallback {} bytes",
            full_screen.width.0,
            full_screen.height.0,
            depth,
            base.depth_texture.as_ref().map(|texture| texture.storage_mode()),
            fallback,
        );
        assert_eq!(base.render_memory.published(gauge), held);
        assert!(gpui::render_memory_gauges().depth_texture_bytes >= held);

        // Lent to an overlay renderer, the bytes move with the textures.
        let mut overlay = headless();
        overlay.lend_intermediates(base.take_intermediates());
        assert_eq!(base.render_memory.published(gauge), 0);
        assert_eq!(overlay.render_memory.published(gauge), held);
        base.lend_intermediates(overlay.take_intermediates());
        assert_eq!(base.render_memory.published(gauge), held);
        assert_eq!(overlay.render_memory.published(gauge), 0);

        // A hidden window's release is seen without a draw.
        base.release_intermediate_textures();
        assert_eq!(base.render_memory.published(gauge), 0);
    }
}

/// The stripe pattern (tag 4) reads its colour from the solid slot the
/// vertex stage prepares, and that stage only prepared it for the solid,
/// pattern-slash and checkerboard tags, so every stripe fill sampled an
/// unset colour. Each backend's shader mirrors this branch by hand.
#[cfg(test)]
mod stripe_pattern_tests {
    use super::{InstanceBufferPool, MetalRenderer};
    use gpui::{
        Bounds, ContentMask, Corners, DevicePixels, Edges, Hsla, Point, Quad, ScaledPixels, Scene,
        pattern_stripe,
    };
    use parking_lot::Mutex;
    use std::sync::Arc;

    #[test]
    fn stripes_are_painted_in_their_configured_color() {
        let mut renderer = MetalRenderer::new_headless(Arc::new(Mutex::new(
            InstanceBufferPool::default(),
        )));
        renderer.update_transparency(true);

        let extent = 32.;
        let bounds = Bounds {
            origin: Point {
                x: ScaledPixels(0.),
                y: ScaledPixels(0.),
            },
            size: gpui::size(ScaledPixels(extent), ScaledPixels(extent)),
        };
        let mut scene = Scene::default();
        scene.quads.push(Quad {
            rounded_clip: 0,
            clip_padding: 0,
            order: 0,
            border_style: Default::default(),
            bounds,
            content_mask: ContentMask { bounds },
            // 8px white stripes with 8px gaps, along the X axis.
            background: pattern_stripe(Hsla::white(), 8., 8.),
            border_color: Hsla::transparent_black(),
            corner_radii: Corners::default(),
            border_widths: Edges::default(),
        });
        scene.finish();

        let image = renderer
            .render_scene_to_image(
                &scene,
                gpui::size(DevicePixels(extent as i32), DevicePixels(extent as i32)),
            )
            .expect("headless render succeeds");

        let column = extent as u32 / 2;
        let lit_rows = (0..extent as u32)
            .filter(|&row| image.get_pixel(column, row)[3] > 200)
            .count();
        let dark_rows = (0..extent as u32)
            .filter(|&row| image.get_pixel(column, row)[3] < 30)
            .count();
        assert!(
            lit_rows >= 12 && dark_rows >= 12,
            "expected about half the rows striped white and half clear, got {lit_rows} lit and {dark_rows} clear"
        );
        let lit = (0..extent as u32)
            .map(|row| image.get_pixel(column, row))
            .find(|pixel| pixel[3] > 200)
            .expect("at least one striped row");
        assert!(
            lit[0] > 200 && lit[1] > 200 && lit[2] > 200,
            "the stripe must carry the configured white, got {lit:?}"
        );
    }
}

/// A window that lives through a display or backing change can keep
/// intermediate textures (depth above all) sized for a stale drawable, and a
/// mismatched depth attachment makes every depth test undefined — observed
/// live as translucent overlays losing their fill in aged windows while fresh
/// windows rendered correctly. `render_frame` must heal the mismatch instead
/// of rendering through it.
#[cfg(test)]
mod stale_texture_healing_tests {
    use super::{InstanceBufferPool, MetalRenderer, read_texture_to_image};
    use gpui::{
        Background, Bounds, ContentMask, Corners, DevicePixels, Edges, Hsla, Point, Quad,
        ScaledPixels, Scene,
    };
    use parking_lot::Mutex;
    use std::sync::Arc;

    pub(super) fn solid_quad_scene(extent: f32) -> Scene {
        let mut scene = Scene::default();
        let bounds = Bounds {
            origin: Point {
                x: ScaledPixels(0.),
                y: ScaledPixels(0.),
            },
            size: gpui::size(ScaledPixels(extent), ScaledPixels(extent)),
        };
        scene.quads.push(Quad {
            rounded_clip: 0,
            clip_padding: 0,
            order: 0,
            border_style: Default::default(),
            bounds,
            content_mask: ContentMask { bounds },
            background: Background::from(Hsla::red()),
            border_color: Hsla::transparent_black(),
            corner_radii: Corners::default(),
            border_widths: Edges::default(),
        });
        scene.finish();
        scene
    }

    #[test]
    fn render_frame_heals_stale_intermediate_textures() {
        let mut renderer = MetalRenderer::new_headless(Arc::new(Mutex::new(
            InstanceBufferPool::default(),
        )));

        // Size the intermediate textures for a small drawable.
        let small = gpui::size(DevicePixels(64), DevicePixels(64));
        let scene = solid_quad_scene(256.);
        renderer
            .render_scene_to_image(&scene, small)
            .expect("small render succeeds");

        // Now render a bigger drawable through `render_frame` directly, the
        // way a window draw does — without anything resizing the textures
        // first. Before the healing check this pairs a 128x128 color target
        // with a 64x64 depth texture.
        let large = gpui::size(DevicePixels(128), DevicePixels(128));
        let descriptor = metal::TextureDescriptor::new();
        descriptor.set_width(large.width.0 as u64);
        descriptor.set_height(large.height.0 as u64);
        descriptor.set_pixel_format(metal::MTLPixelFormat::BGRA8Unorm);
        descriptor
            .set_usage(metal::MTLTextureUsage::RenderTarget | metal::MTLTextureUsage::ShaderRead);
        descriptor.set_storage_mode(metal::MTLStorageMode::Managed);
        let target = renderer.device.new_texture(&descriptor);

        let command_buffer = renderer
            .render_frame(&scene, &target, large, None, None)
            .expect("stale-texture render heals and succeeds");
        if !renderer.is_unified_memory {
            let blit = command_buffer.new_blit_command_encoder();
            blit.synchronize_resource(&target);
            blit.end_encoding();
        }
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let depth_texture = renderer
            .depth_texture
            .as_ref()
            .expect("depth texture recreated");
        assert_eq!(
            (depth_texture.width(), depth_texture.height()),
            (large.width.0 as u64, large.height.0 as u64),
            "healing must resize the depth texture to the drawable"
        );

        let image = read_texture_to_image(&target).expect("read back");
        let center = image.get_pixel(64, 64);
        assert!(
            center[0] > 200 && center[1] < 60 && center[2] < 60,
            "the opaque red quad must survive the healed render, got {center:?}"
        );
    }

    /// An occluded window releases its drawable-sized intermediate textures,
    /// keeps them released across drawable-size updates, and rebuilds them on
    /// the next draw — without that rebuild registering as a stale-texture
    /// heal. Also prints the device-side bytes the release gives back for a
    /// retina full-screen drawable.
    #[test]
    fn release_intermediate_textures_reclaims_device_memory_until_next_draw() {
        let mut renderer = MetalRenderer::new_headless(Arc::new(Mutex::new(
            InstanceBufferPool::default(),
        )));
        const MB: u64 = 1024 * 1024;

        let full_screen = gpui::size(DevicePixels(3456), DevicePixels(2168));
        let baseline = renderer.device.current_allocated_size();
        renderer.update_drawable_size(full_screen);
        let with_textures = renderer.device.current_allocated_size();
        let held = with_textures.saturating_sub(baseline);
        assert!(renderer.depth_texture.is_some());
        assert!(
            renderer.path_intermediate_texture.is_none(),
            "the path intermediate is allocated by the first path batch, not by a resize"
        );

        renderer.release_intermediate_textures();
        let released = renderer.device.current_allocated_size();
        assert!(renderer.depth_texture.is_none());
        assert!(renderer.path_intermediate_texture.is_none());
        assert!(renderer.path_intermediate_msaa_texture.is_none());
        let reclaimed = with_textures.saturating_sub(released);
        println!(
            "intermediate textures for {}x{}: held {}MB, release reclaimed {}MB",
            full_screen.width.0,
            full_screen.height.0,
            held / MB,
            reclaimed / MB,
        );
        assert!(
            reclaimed * 10 >= held * 9,
            "release must give back the intermediate textures (held {held}, reclaimed {reclaimed})"
        );
        if renderer.is_apple_gpu {
            assert_eq!(held, 0, "tile-memory depth holds no device memory");
        }

        // A drawable-size update while occluded must not rebuild them.
        renderer.update_drawable_size(gpui::size(DevicePixels(1728), DevicePixels(1084)));
        assert!(
            renderer.depth_texture.is_none(),
            "resizing while released must keep the textures dropped"
        );

        // The next draw rebuilds them at the drawable's size, silently.
        let heals_before = renderer.stale_texture_heals;
        let small = gpui::size(DevicePixels(64), DevicePixels(64));
        renderer
            .render_scene_to_image(&solid_quad_scene(32.), small)
            .expect("render after release succeeds");
        let depth_texture = renderer
            .depth_texture
            .as_ref()
            .expect("depth texture rebuilt by the draw");
        assert_eq!(
            (depth_texture.width(), depth_texture.height()),
            (small.width.0 as u64, small.height.0 as u64)
        );
        assert!(!renderer.intermediate_textures_released);
        assert_eq!(
            renderer.stale_texture_heals, heals_before,
            "a rebuild after release is not a stale-texture heal"
        );
    }

    /// An overlay renderer never allocates drawable-sized intermediates of
    /// its own: it draws with the base renderer's, handed over per draw.
    #[test]
    fn borrowing_renderer_draws_with_lent_intermediates_and_allocates_none() {
        let pool = Arc::new(Mutex::new(InstanceBufferPool::default()));
        let mut base = MetalRenderer::new_headless(pool.clone());
        let mut overlay = MetalRenderer::new_headless(pool);
        overlay.borrows_intermediates = true;
        let size = gpui::size(DevicePixels(64), DevicePixels(64));

        overlay.update_drawable_size(size);
        assert!(
            overlay.depth_texture.is_none(),
            "a borrowing renderer must not allocate on resize"
        );
        overlay.release_intermediate_textures();
        assert!(!overlay.intermediate_textures_released);

        base.render_scene_to_image(&solid_quad_scene(32.), size)
            .expect("base render succeeds");
        let base_depth = base.depth_texture.clone().expect("base owns a depth texture");

        overlay.lend_intermediates(base.take_intermediates());
        assert!(base.depth_texture.is_none());
        let image = overlay
            .render_scene_to_image(&solid_quad_scene(32.), size)
            .expect("overlay render succeeds with lent textures");
        let center = image.get_pixel(16, 16);
        assert!(center[0] > 200 && center[1] < 60 && center[2] < 60, "got {center:?}");
        base.lend_intermediates(overlay.take_intermediates());

        assert!(overlay.depth_texture.is_none());
        let returned = base.depth_texture.as_ref().expect("textures returned to base");
        assert_eq!(
            returned.gpu_resource_id()._impl,
            base_depth.gpu_resource_id()._impl,
            "the same depth texture must come back, not a fresh allocation"
        );
    }

    fn square_path(
        origin: f32,
        extent: f32,
        color: Hsla,
        mask: f32,
    ) -> gpui::Path<ScaledPixels> {
        use gpui::{Path, px};
        let mut path = Path::new(Point {
            x: px(origin),
            y: px(origin),
        });
        path.line_to(Point {
            x: px(origin + extent),
            y: px(origin),
        });
        path.line_to(Point {
            x: px(origin + extent),
            y: px(origin + extent),
        });
        path.line_to(Point {
            x: px(origin),
            y: px(origin + extent),
        });
        path.color = Background::from(color);
        let mut path = path.scale(1.);
        path.content_mask = ContentMask {
            bounds: Bounds {
                origin: Point {
                    x: ScaledPixels(0.),
                    y: ScaledPixels(0.),
                },
                size: gpui::size(ScaledPixels(mask), ScaledPixels(mask)),
            },
        };
        path
    }

    fn triangle_path_scene(extent: f32) -> Scene {
        let mut scene = Scene::default();
        scene.insert_primitive(square_path(0., extent, Hsla::red(), extent));
        scene.finish();
        scene
    }

    /// Two path batches whose screen rectangles overlap must each keep their
    /// own pixels: with every batch rasterized up front into one intermediate,
    /// the later batch would otherwise overwrite the earlier one's region.
    /// Draw orders come from overlap: a quad overlapping the first path
    /// draws above it, and the second path overlapping that quad draws above
    /// both, which forces two path batches with the quad between them. The
    /// later batch must still win where the paths overlap, exactly as
    /// painter's order says.
    fn overlapping_path_batches_scene() -> Scene {
        let mut scene = Scene::default();
        scene.insert_primitive(square_path(0., 40., Hsla::red(), 64.));
        let bounds = Bounds {
            origin: Point {
                x: ScaledPixels(30.),
                y: ScaledPixels(30.),
            },
            size: gpui::size(ScaledPixels(4.), ScaledPixels(4.)),
        };
        scene.insert_primitive(Quad {
            rounded_clip: 0,
            clip_padding: 0,
            order: 0,
            border_style: Default::default(),
            bounds,
            content_mask: ContentMask { bounds },
            background: Background::from(Hsla::white()),
            border_color: Hsla::transparent_black(),
            corner_radii: Corners::default(),
            border_widths: Edges::default(),
        });
        scene.insert_primitive(square_path(24., 40., Hsla::blue(), 64.));
        scene.finish();
        assert_eq!(
            scene
                .batches()
                .filter(|batch| matches!(batch, gpui::PrimitiveBatch::Paths(_)))
                .count(),
            2,
            "the scene must produce two path batches"
        );
        scene
    }

    fn assert_overlapping_batches_rendered(image: &image::RgbaImage) {
        let red_only = image.get_pixel(8, 8);
        assert!(
            red_only[0] > 200 && red_only[2] < 60,
            "first batch's own region must be red, got {red_only:?}"
        );
        let overlap = image.get_pixel(38, 38);
        assert!(
            overlap[2] > 200 && overlap[0] < 60,
            "the later batch must win in the overlap, got {overlap:?}"
        );
        let blue_only = image.get_pixel(56, 56);
        assert!(
            blue_only[2] > 200 && blue_only[0] < 60,
            "second batch's own region must be blue, got {blue_only:?}"
        );
    }

    #[test]
    fn overlapping_path_batches_keep_their_own_slots_in_the_single_path_pass() {
        let mut renderer = MetalRenderer::new_headless(Arc::new(Mutex::new(
            InstanceBufferPool::default(),
        )));
        // Two 42x42 slots pack side by side into 128x128.
        let size = gpui::size(DevicePixels(128), DevicePixels(128));
        assert!(MetalRenderer::plan_path_batches(&overlapping_path_batches_scene(), size).is_some());
        let image = renderer
            .render_scene_to_image(&overlapping_path_batches_scene(), size)
            .expect("render succeeds");
        assert_overlapping_batches_rendered(&image);
        if renderer.is_apple_gpu {
            assert_eq!(
                renderer
                    .depth_texture
                    .as_ref()
                    .map(|texture| texture.storage_mode()),
                Some(metal::MTLStorageMode::Memoryless),
                "the planned frame renders through tile-memory depth"
            );
            assert!(
                renderer.fallback_depth_texture.is_none(),
                "a planned frame needs no private depth texture"
            );
        }
    }

    /// When the batches' slots do not fit into the intermediate the frame
    /// falls back to rasterizing each batch just before drawing it, which
    /// splits the main pass and needs a private depth texture — and the
    /// result must be pixel-identical.
    #[test]
    fn path_batches_that_do_not_fit_fall_back_to_a_split_pass() {
        let mut renderer = MetalRenderer::new_headless(Arc::new(Mutex::new(
            InstanceBufferPool::default(),
        )));
        let size = gpui::size(DevicePixels(64), DevicePixels(64));
        // Each quad overlaps the path before it and the path after it
        // overlaps the quad, so every path lands in its own batch.
        let mut scene = Scene::default();
        for index in 0..8 {
            let origin = index as f32 * 2.;
            scene.insert_primitive(square_path(origin, 40., Hsla::red(), 64.));
            let bounds = Bounds {
                origin: Point {
                    x: ScaledPixels(origin + 30.),
                    y: ScaledPixels(origin + 30.),
                },
                size: gpui::size(ScaledPixels(4.), ScaledPixels(4.)),
            };
            scene.insert_primitive(Quad {
                rounded_clip: 0,
                clip_padding: 0,
                order: 0,
                border_style: Default::default(),
                bounds,
                content_mask: ContentMask { bounds },
                background: Background::from(Hsla::white()),
                border_color: Hsla::transparent_black(),
                corner_radii: Corners::default(),
                border_widths: Edges::default(),
            });
        }
        scene.finish();
        assert_eq!(
            scene
                .batches()
                .filter(|batch| matches!(batch, gpui::PrimitiveBatch::Paths(_)))
                .count(),
            8
        );
        assert!(
            MetalRenderer::plan_path_batches(&scene, size).is_none(),
            "eight 42x42 slots cannot be packed into 64x64"
        );
        let image = renderer
            .render_scene_to_image(&scene, size)
            .expect("fallback render succeeds");
        let inside = image.get_pixel(20, 20);
        assert!(inside[0] > 200 && inside[2] < 60, "got {inside:?}");
        if renderer.is_apple_gpu {
            assert!(
                renderer.fallback_depth_texture.is_some(),
                "the split pass rendered through the private fallback depth texture"
            );
        }
        // Idle frames retire the fallback texture again.
        for _ in 0..=super::PATH_INTERMEDIATE_IDLE_FRAMES {
            renderer
                .render_scene_to_image(&solid_quad_scene(32.), size)
                .expect("quad-only render succeeds");
        }
        assert!(renderer.fallback_depth_texture.is_none());
    }

    /// The path intermediate is 4 bytes per drawable pixel that only frames
    /// with a path batch need: it is allocated by the first such frame at the
    /// drawable's size and dropped again after enough path-free frames.
    #[test]
    fn path_intermediate_is_allocated_on_demand_and_retired_when_idle() {
        let mut renderer = MetalRenderer::new_headless(Arc::new(Mutex::new(
            InstanceBufferPool::default(),
        )));
        let size = gpui::size(DevicePixels(64), DevicePixels(64));

        renderer
            .render_scene_to_image(&solid_quad_scene(32.), size)
            .expect("quad-only render succeeds");
        assert!(
            renderer.path_intermediate_texture.is_none(),
            "a frame without paths must not allocate the path intermediate"
        );

        renderer
            .render_scene_to_image(&triangle_path_scene(32.), size)
            .expect("path render succeeds");
        let intermediate = renderer
            .path_intermediate_texture
            .as_ref()
            .expect("the first path batch allocates the intermediate");
        assert_eq!(
            (intermediate.width(), intermediate.height()),
            (size.width.0 as u64, size.height.0 as u64)
        );

        for _ in 1..super::PATH_INTERMEDIATE_IDLE_FRAMES {
            renderer
                .render_scene_to_image(&solid_quad_scene(32.), size)
                .expect("quad-only render succeeds");
        }
        assert!(
            renderer.path_intermediate_texture.is_some(),
            "the intermediate survives until the idle window is used up"
        );
        renderer
            .render_scene_to_image(&solid_quad_scene(32.), size)
            .expect("quad-only render succeeds");
        assert!(
            renderer.path_intermediate_texture.is_none()
                && renderer.path_intermediate_msaa_texture.is_none(),
            "the PATH_INTERMEDIATE_IDLE_FRAMES-th path-free frame drops the intermediate"
        );

        // A later path frame simply allocates it again.
        renderer
            .render_scene_to_image(&triangle_path_scene(32.), size)
            .expect("path render succeeds after retirement");
        assert!(renderer.path_intermediate_texture.is_some());
    }
}

/// A layer-backed draw, the path every window present takes, reports where
/// its time went. The layer is in no window, so nothing reaches the screen,
/// but `nextDrawable`, encoding and commit all run for real.
#[cfg(test)]
mod present_report_tests {
    use super::{InstanceBufferPool, MetalRenderer, stale_texture_healing_tests::solid_quad_scene};
    use gpui::{DevicePixels, PresentOutcome};
    use parking_lot::Mutex;
    use std::{
        sync::{Arc, atomic::Ordering},
        time::Duration,
    };

    #[test]
    fn a_layer_backed_draw_reports_acquire_encode_and_commit_separately() {
        let mut renderer =
            MetalRenderer::new(Arc::new(Mutex::new(InstanceBufferPool::default())), false, false);
        renderer.update_drawable_size(gpui::size(DevicePixels(64), DevicePixels(64)));

        let report = renderer.draw(&solid_quad_scene(32.));

        assert_eq!(report.outcome, PresentOutcome::Presented);
        let breakdown = report.breakdown;
        assert_eq!(breakdown.layers, 1);
        assert!(
            breakdown.encode > Duration::ZERO,
            "encoding a quad takes measurable time: {breakdown:?}"
        );
        assert_eq!(
            breakdown.drawables_in_flight, 0,
            "nothing was presented before the first draw"
        );

        let report = renderer.draw(&solid_quad_scene(32.));
        assert_eq!(report.outcome, PresentOutcome::Presented);
        assert!(
            report.breakdown.drawables_in_flight <= 1,
            "only the first draw's drawable can still be in flight: {:?}",
            report.breakdown
        );
    }

    /// Noting a drawable counts it before it is presented, so before its
    /// presented handler can run: the count rises by one then, whatever the
    /// handler later does.
    #[test]
    fn noting_a_drawable_counts_it_in_flight_before_its_handler_can_run() {
        let mut renderer =
            MetalRenderer::new(Arc::new(Mutex::new(InstanceBufferPool::default())), false, false);
        renderer.update_drawable_size(gpui::size(DevicePixels(64), DevicePixels(64)));
        let layer = renderer.layer.clone().expect("a window renderer has a layer");

        objc2::rc::autoreleasepool(|_| {
            let drawable = layer
                .next_drawable()
                .expect("a layer in no window always has a drawable to give");
            assert_eq!(renderer.drawables_in_flight.load(Ordering::Relaxed), 0);
            renderer.note_drawable_in_flight(drawable);
            assert_eq!(
                renderer.drawables_in_flight.load(Ordering::Relaxed),
                1,
                "the drawable was counted and nothing has presented it yet"
            );
        });
    }

    /// A layer with no size has no drawable to give, and asking for one would
    /// only fail again on every refresh, so the present is dropped without
    /// asking; resizing the layer redraws the window anyway.
    #[test]
    fn a_zero_size_layer_drops_the_present_without_asking_for_a_drawable() {
        let mut renderer =
            MetalRenderer::new(Arc::new(Mutex::new(InstanceBufferPool::default())), false, false);
        let drawable_size = renderer
            .layer()
            .expect("a window renderer has a layer")
            .drawable_size();
        assert_eq!(
            (drawable_size.width, drawable_size.height),
            (0., 0.),
            "a layer that was never sized has an empty drawable"
        );
        renderer.consecutive_deferred_presents = 5;

        let report = renderer.draw(&solid_quad_scene(32.));

        assert_eq!(report.outcome, PresentOutcome::Dropped);
        assert_eq!(report.breakdown.layers, 1, "the layer exists, only empty");
        assert_eq!(
            report.breakdown.acquire,
            Duration::ZERO,
            "nextDrawable was never asked"
        );
        assert_eq!(
            renderer.consecutive_deferred_presents, 0,
            "a dropped present ends the run of deferrals before it"
        );
    }

    #[test]
    fn a_headless_renderer_reports_a_dropped_present() {
        let mut renderer =
            MetalRenderer::new_headless(Arc::new(Mutex::new(InstanceBufferPool::default())));

        let report = renderer.draw(&solid_quad_scene(32.));

        assert_eq!(report.outcome, PresentOutcome::Dropped);
        assert_eq!(report.breakdown.layers, 0);
    }

    /// A missing drawable defers the frame only while the layer can time out
    /// waiting for one, and only for a bounded run: anything else is a fault
    /// that would otherwise be retried every display refresh.
    #[test]
    fn a_missing_drawable_defers_only_a_bounded_run_on_a_layer_that_can_time_out() {
        let mut renderer =
            MetalRenderer::new(Arc::new(Mutex::new(InstanceBufferPool::default())), false, false);

        assert_eq!(
            renderer.missing_drawable_outcome(false),
            PresentOutcome::Dropped,
            "a layer that waits for drawables returns none only on a fault"
        );
        assert_eq!(renderer.consecutive_deferred_presents, 0);

        for _ in 0..super::MAX_CONSECUTIVE_DEFERRED_PRESENTS {
            assert_eq!(
                renderer.missing_drawable_outcome(true),
                PresentOutcome::Deferred
            );
        }
        assert_eq!(
            renderer.missing_drawable_outcome(true),
            PresentOutcome::Dropped,
            "a run that outlasts the bound drops the frame"
        );
        assert_eq!(
            renderer.missing_drawable_outcome(true),
            PresentOutcome::Deferred,
            "and the next present starts a fresh run"
        );
    }

    /// A layer never has more drawables out than its maximum, so a count
    /// above it can only come from presented handlers that never ran. It
    /// reads as the maximum, flagged, and is lowered so it stops climbing.
    /// Lowering it removes only the excess, so it stays flagged, however low
    /// it later reads, until hiding the window starts it afresh.
    #[test]
    fn an_in_flight_count_above_the_layer_maximum_is_clamped_flagged_and_forgotten() {
        let mut renderer =
            MetalRenderer::new(Arc::new(Mutex::new(InstanceBufferPool::default())), false, false);
        let layer = renderer.layer.clone().expect("a window renderer has a layer");
        let maximum = layer.maximum_drawable_count() as u8;
        assert_eq!(renderer.read_drawables_in_flight(&layer), (0, false));
        renderer.drawables_in_flight.store(7, Ordering::Relaxed);

        assert_eq!(renderer.read_drawables_in_flight(&layer), (maximum, true));
        assert_eq!(
            renderer.drawables_in_flight.load(Ordering::Relaxed),
            u32::from(maximum),
            "the count is lowered to the maximum"
        );
        assert_eq!(
            renderer.read_drawables_in_flight(&layer),
            (maximum, true),
            "the lost handlers' bias below the maximum is still in the count"
        );
        renderer.drawables_in_flight.store(1, Ordering::Relaxed);
        assert_eq!(
            renderer.read_drawables_in_flight(&layer),
            (1, true),
            "and still may be as it falls"
        );

        renderer.forget_drawables_in_flight();
        assert_eq!(renderer.read_drawables_in_flight(&layer), (0, false));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use objc2_core_foundation::{CFDictionary, CFType};
    use objc2_core_video::{
        CVPixelBufferCreate, CVPixelBufferGetBaseAddressOfPlane,
        CVPixelBufferGetBytesPerRowOfPlane, CVPixelBufferLockBaseAddress, CVPixelBufferLockFlags,
        CVPixelBufferUnlockBaseAddress, kCVPixelBufferIOSurfacePropertiesKey,
    };

    #[test]
    fn render_core_video_surface() {
        let attributes = CFDictionary::from_slices(
            &[unsafe { kCVPixelBufferIOSurfacePropertiesKey }],
            &[&*CFDictionary::<CFType, CFType>::empty()],
        );
        let mut image_buffer = ptr::null_mut();
        assert_eq!(
            unsafe {
                CVPixelBufferCreate(
                    None,
                    16,
                    16,
                    kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
                    Some(attributes.as_opaque()),
                    ptr::NonNull::from(&mut image_buffer),
                )
            },
            kCVReturnSuccess
        );
        let image_buffer =
            unsafe { CFRetained::from_raw(ptr::NonNull::new(image_buffer).unwrap()) };
        unsafe {
            assert_eq!(
                CVPixelBufferLockBaseAddress(&image_buffer, CVPixelBufferLockFlags::empty()),
                kCVReturnSuccess
            );
            for plane in 0..2 {
                let bytes = CVPixelBufferGetBaseAddressOfPlane(&image_buffer, plane).cast::<u8>();
                let length = CVPixelBufferGetBytesPerRowOfPlane(&image_buffer, plane)
                    * CVPixelBufferGetHeightOfPlane(&image_buffer, plane);
                slice::from_raw_parts_mut(bytes, length).fill(128);
            }
            assert_eq!(
                CVPixelBufferUnlockBaseAddress(&image_buffer, CVPixelBufferLockFlags::empty()),
                kCVReturnSuccess
            );
        }

        let bounds = Bounds {
            origin: Point::default(),
            size: size(ScaledPixels(16.0), ScaledPixels(16.0)),
        };
        let mut scene = Scene::default();
        scene.insert_primitive(PaintSurface {
            rounded_clip: 0,
            clip_padding: 0,
            order: 0,
            bounds,
            content_mask: ContentMask { bounds },
            image_buffer: image_buffer.clone(),
        });
        drop(image_buffer);
        scene.finish();

        let mut renderer = MetalRenderer::new_headless(Arc::new(Mutex::new(Default::default())));
        let image = renderer
            .render_scene_to_image(&scene, size(DevicePixels(16), DevicePixels(16)))
            .unwrap();
        for image::Rgba([red, green, blue, alpha]) in image.pixels() {
            assert!(red.abs_diff(128) <= 2);
            assert!(green.abs_diff(128) <= 2);
            assert!(blue.abs_diff(128) <= 2);
            assert_eq!(*alpha, 255);
        }
    }
}

/// What a frosted surface costs in device memory, measured with the device's
/// own accounting (`currentAllocatedSize`) rather than with the textures the
/// renderer believes it holds, so allocations made inside Metal Performance
/// Shaders show up too.
#[cfg(test)]
mod backdrop_blur_memory_tests {
    use super::{InstanceBufferPool, MetalRenderer};
    use gpui::{
        BackdropBlur, Background, Bounds, ContentMask, Corners, DevicePixels, Edges, Hsla, Point,
        Quad, ScaledPixels, Scene, Size,
    };
    use parking_lot::Mutex;
    use std::sync::Arc;

    const MIB: f64 = 1024. * 1024.;
    const FULL_SCREEN: Size<DevicePixels> = Size {
        width: DevicePixels(3456),
        height: DevicePixels(2168),
    };

    fn rect(x: f32, y: f32, width: f32, height: f32) -> Bounds<ScaledPixels> {
        Bounds {
            origin: Point {
                x: ScaledPixels(x),
                y: ScaledPixels(y),
            },
            size: gpui::size(ScaledPixels(width), ScaledPixels(height)),
        }
    }

    fn quad(bounds: Bounds<ScaledPixels>, color: Hsla) -> Quad {
        Quad {
            rounded_clip: 0,
            clip_padding: 0,
            order: 0,
            border_style: Default::default(),
            bounds,
            content_mask: ContentMask { bounds },
            background: Background::from(color),
            border_color: Hsla::transparent_black(),
            corner_radii: Corners::default(),
            border_widths: Edges::default(),
        }
    }

    /// Vertical stripes across the whole drawable, so a blur has edges to
    /// soften, then one frosted surface per entry of `blurs` with a tint over
    /// it, as `ui::frosted` paints a menu.
    fn frosted_scene(blurs: &[Bounds<ScaledPixels>]) -> Scene {
        let mut scene = Scene::default();
        let width = FULL_SCREEN.width.0 as f32;
        let height = FULL_SCREEN.height.0 as f32;
        let stripe = 24.;
        let mut x = 0.;
        let mut index = 0;
        while x < width {
            let hue = (index % 12) as f32 / 12.;
            scene.insert_primitive(quad(rect(x, 0., stripe, height), gpui::hsla(hue, 0.7, 0.5, 1.)));
            x += stripe;
            index += 1;
        }
        for bounds in blurs {
            scene.insert_backdrop_blur(BackdropBlur {
                order: 0,
                blur_radius: ScaledPixels(16.),
                saturation: 1.4,
                opacity: 1.,
                bounds: *bounds,
                content_mask: ContentMask { bounds: *bounds },
                corner_radii: Corners::all(ScaledPixels(16.)),
            });
            scene.insert_primitive(quad(*bounds, gpui::hsla(0., 0., 0.1, 0.93)));
        }
        scene.finish();
        scene
    }

    fn allocated(renderer: &MetalRenderer) -> f64 {
        renderer.device.current_allocated_size() as f64 / MIB
    }

    fn scratch_textures(renderer: &MetalRenderer) -> f64 {
        renderer
            .backdrop_scratch
            .as_ref()
            .map_or(0, super::BackdropScratch::allocated_bytes) as f64
            / MIB
    }

    fn render(renderer: &mut MetalRenderer, scene: &Scene) {
        renderer
            .render_scene_to_image(scene, FULL_SCREEN)
            .expect("headless render succeeds");
    }

    const ACCURACY_TARGET: Size<DevicePixels> = Size {
        width: DevicePixels(1600),
        height: DevicePixels(1000),
    };

    /// Coarse hue stripes with a band of one-device-pixel black and white
    /// columns through them: the hardest content for a blur to reproduce,
    /// since every bit of the fine band's detail has to average out exactly.
    fn detailed_scene(
        blurs: &[Bounds<ScaledPixels>],
        sigma: f32,
        saturation: f32,
        tint: Option<Hsla>,
    ) -> Scene {
        let mut scene = Scene::default();
        let width = ACCURACY_TARGET.width.0 as f32;
        let height = ACCURACY_TARGET.height.0 as f32;
        let mut x = 0.;
        let mut index = 0;
        while x < width {
            let hue = (index % 12) as f32 / 12.;
            scene.insert_primitive(quad(rect(x, 0., 24., height), gpui::hsla(hue, 0.7, 0.5, 1.)));
            x += 24.;
            index += 1;
        }
        let mut column = 0.;
        while column < width {
            scene.insert_primitive(quad(rect(column, 380., 1., 160.), gpui::hsla(0., 0., 1., 1.)));
            scene.insert_primitive(quad(rect(column + 1., 380., 1., 160.), gpui::hsla(0., 0., 0., 1.)));
            column += 2.;
        }
        for bounds in blurs {
            scene.insert_backdrop_blur(BackdropBlur {
                order: 0,
                blur_radius: ScaledPixels(sigma),
                saturation,
                opacity: 1.,
                bounds: *bounds,
                content_mask: ContentMask { bounds: *bounds },
                corner_radii: Corners::default(),
            });
            if let Some(tint) = tint {
                scene.insert_primitive(quad(*bounds, tint));
            }
        }
        scene.finish();
        scene
    }

    /// `frost` under an opaque-target "over" of a gray `tint_level` at
    /// `tint_alpha`, as the frosted material paints its tint.
    fn under_tint(frost: &[[f64; 4]], tint_level: f64, tint_alpha: f64) -> Vec<[f64; 4]> {
        frost
            .iter()
            .map(|pixel| {
                let mut output = *pixel;
                for channel in 0..3 {
                    output[channel] = tint_level * tint_alpha + pixel[channel] * (1. - tint_alpha);
                }
                output[3] = tint_alpha + pixel[3] * (1. - tint_alpha);
                output
            })
            .collect()
    }

    /// The blur a renderer should produce for `region` of `source`, computed
    /// on the CPU in double precision: a separable gaussian of `sigma` whose
    /// taps clamp at the region's edges, then the saturation boost.
    fn reference_frost(
        source: &image::RgbaImage,
        region: (u32, u32, u32, u32),
        sigma: f64,
        saturation: f64,
    ) -> Vec<[f64; 4]> {
        let (left, top, width, height) = region;
        let radius = (sigma * 4.).ceil() as i64;
        let weights: Vec<f64> = (-radius..=radius)
            .map(|offset| (-((offset * offset) as f64) / (2. * sigma * sigma)).exp())
            .collect();
        let total: f64 = weights.iter().sum();
        let pixel = |x: i64, y: i64| -> [f64; 4] {
            let x = x.clamp(0, width as i64 - 1) as u32 + left;
            let y = y.clamp(0, height as i64 - 1) as u32 + top;
            let image::Rgba(channels) = *source.get_pixel(x, y);
            channels.map(|channel| channel as f64 / 255.)
        };
        let mut horizontal = vec![[0.; 4]; (width * height) as usize];
        for y in 0..height as i64 {
            for x in 0..width as i64 {
                let mut sum = [0.; 4];
                for (tap, weight) in weights.iter().enumerate() {
                    let sample = pixel(x + tap as i64 - radius, y);
                    for channel in 0..4 {
                        sum[channel] += sample[channel] * weight / total;
                    }
                }
                horizontal[(y * width as i64 + x) as usize] = sum;
            }
        }
        let mut output = vec![[0.; 4]; (width * height) as usize];
        for y in 0..height as i64 {
            for x in 0..width as i64 {
                let mut sum = [0.; 4];
                for (tap, weight) in weights.iter().enumerate() {
                    let row = (y + tap as i64 - radius).clamp(0, height as i64 - 1);
                    let sample = horizontal[(row * width as i64 + x) as usize];
                    for channel in 0..4 {
                        sum[channel] += sample[channel] * weight / total;
                    }
                }
                let luma = 0.2126 * sum[0] + 0.7152 * sum[1] + 0.0722 * sum[2];
                for channel in 0..3 {
                    sum[channel] = (luma + (sum[channel] - luma) * saturation).clamp(0., sum[3]);
                }
                output[(y * width as i64 + x) as usize] = sum;
            }
        }
        output
    }

    /// Per-channel error of `rendered` against `reference` inside `region`,
    /// in 8-bit steps: (largest, mean, share of channels off by more than one).
    fn frost_error(
        rendered: &image::RgbaImage,
        reference: &[[f64; 4]],
        region: (u32, u32, u32, u32),
    ) -> (f64, f64, f64, (u32, u32)) {
        let (left, top, width, _) = region;
        let mut largest: f64 = 0.;
        let mut worst = (0, 0);
        let mut total = 0.;
        let mut over_one = 0usize;
        let mut count = 0usize;
        for (index, expected) in reference.iter().enumerate() {
            let x = index as u32 % width + left;
            let y = index as u32 / width + top;
            let image::Rgba(actual) = *rendered.get_pixel(x, y);
            for channel in 0..4 {
                let error = (actual[channel] as f64 - expected[channel] * 255.).abs();
                if error > largest {
                    largest = error;
                    worst = (x - left, y - top);
                }
                total += error;
                if error > 1. {
                    over_one += 1;
                }
                count += 1;
            }
        }
        (largest, total / count as f64, over_one as f64 / count as f64, worst)
    }

    /// The frost under the material's tint renders the bytes an exact
    /// gaussian would, to within rounding. The frost alone, before the tint,
    /// is held to what the downsampled blur measures on this region (largest
    /// 2.27 eight-bit steps, mean 0.29; the Metal Performance Shaders blur it
    /// replaced measured 1.78 and 0.28): its largest errors sit where a band
    /// of one-pixel columns meets the region's edge, which box-averaging the
    /// columns in pairs shifts by half a pixel.
    #[test]
    fn frost_matches_an_exact_gaussian() {
        let mut renderer = MetalRenderer::new_headless(Arc::new(Mutex::new(
            InstanceBufferPool::default(),
        )));
        let source = renderer
            .render_scene_to_image(&detailed_scene(&[], 0., 1., None), ACCURACY_TARGET)
            .expect("headless render succeeds");
        // Over the band of one-pixel columns and the stripes either side.
        let region = (700u32, 330u32, 200u32, 120u32);
        let (left, top, width, height) = region;
        let bounds = rect(left as f32, top as f32, width as f32, height as f32);
        let reference = reference_frost(&source, region, 16., 1.4);

        let frost = renderer
            .render_scene_to_image(&detailed_scene(&[bounds], 16., 1.4, None), ACCURACY_TARGET)
            .expect("headless render succeeds");
        let (largest, mean, _, worst) = frost_error(&frost, &reference, region);
        assert!(
            largest <= 2.5 && mean <= 0.35,
            "frost off an exact gaussian by up to {largest:.2} steps at {worst:?}, {mean:.3} on average"
        );

        let tint = gpui::hsla(0., 0., 0.1, 0.93);
        let tinted = renderer
            .render_scene_to_image(
                &detailed_scene(&[bounds], 16., 1.4, Some(tint)),
                ACCURACY_TARGET,
            )
            .expect("headless render succeeds");
        let (largest, _, over_one, worst) =
            frost_error(&tinted, &under_tint(&reference, 0.1, 0.93), region);
        assert!(
            largest <= 1. && over_one == 0.,
            "the tinted material is off an exact gaussian by up to {largest:.2} steps at {worst:?}"
        );
    }

    /// Opaque quads beneath a blur take the depth prepass, so they are in the
    /// framebuffer before the blur samples it and hide what they cover; an
    /// opaque quad above the blur paints after it and is never frosted into
    /// the surface beneath.
    #[test]
    fn a_frosted_frame_keeps_paint_order_around_the_blur() {
        let target = Size {
            width: DevicePixels(200),
            height: DevicePixels(100),
        };
        let mut renderer = MetalRenderer::new_headless(Arc::new(Mutex::new(
            InstanceBufferPool::default(),
        )));
        let mut scene = Scene::default();
        scene.insert_primitive(quad(rect(0., 0., 200., 100.), gpui::hsla(0., 1., 0.5, 1.)));
        scene.insert_primitive(quad(rect(0., 0., 100., 100.), gpui::hsla(1. / 3., 1., 0.5, 1.)));
        let frosted = rect(50., 0., 100., 100.);
        scene.insert_backdrop_blur(BackdropBlur {
            order: 0,
            blur_radius: ScaledPixels(4.),
            saturation: 1.,
            opacity: 1.,
            bounds: frosted,
            content_mask: ContentMask { bounds: frosted },
            corner_radii: Corners::default(),
        });
        scene.insert_primitive(quad(rect(120., 0., 80., 100.), gpui::hsla(2. / 3., 1., 0.5, 1.)));
        scene.finish();
        assert!(
            !scene.opaque_quad_indices.is_empty(),
            "the quads beneath the blur take the prepass"
        );

        let image = renderer
            .render_scene_to_image(&scene, target)
            .expect("headless render succeeds");
        let pixel = |x: u32| image.get_pixel(x, 50).0;
        assert_eq!(pixel(25)[..3], [0, 255, 0], "green over red, outside the blur");
        assert_eq!(pixel(130)[..3], [0, 0, 255], "the quad above the blur covers it");
        assert_eq!(pixel(175)[..3], [0, 0, 255]);
        assert!(pixel(101)[1] > 0, "green blurs across the frost's left half");
        for x in 100..120 {
            let [red, green, blue, _] = pixel(x);
            assert!(
                blue <= 1 && red > 0,
                "the frost at x={x} blurs red and green only, got {:?}",
                [red, green, blue]
            );
        }
    }

    /// Every frosted surface of a frame runs through one downsampled scratch
    /// pair, which is all the device memory frosting holds: nothing hides
    /// in a cache the gauges cannot see, as Metal Performance Shaders' did.
    #[test]
    fn frosted_surfaces_share_one_small_scratch_and_hold_nothing_else() {
        let mut renderer = MetalRenderer::new_headless(Arc::new(Mutex::new(
            InstanceBufferPool::default(),
        )));
        let plain = frosted_scene(&[]);
        render(&mut renderer, &plain);
        let baseline = renderer.device.current_allocated_size();

        let menu = rect(400., 300., 1200., 1600.);
        let many = frosted_scene(&[
            menu,
            rect(1700., 200., 700., 420.),
            rect(600., 1900., 2200., 240.),
            rect(2500., 700., 800., 1100.),
        ]);
        render(&mut renderer, &many);
        render(&mut renderer, &many);
        let scratch = renderer
            .backdrop_scratch
            .as_ref()
            .expect("a frosted frame holds the scratch");
        let scratch_bytes = scratch.allocated_bytes();
        // Sigma 16 is downsampled four times on each axis; the scratch
        // spans the largest region's blocks and pads, in 32-texel steps.
        let (widest, tallest) = (2200u64.div_ceil(4) + 2, 1600u64.div_ceil(4) + 2);
        assert!(
            scratch.primary.width() >= widest
                && scratch.primary.height() >= tallest
                && scratch.primary.width() < widest + 32
                && scratch.primary.height() < tallest + 32,
            "one pair sized to the largest region, got {}x{}",
            scratch.primary.width(),
            scratch.primary.height()
        );
        assert_eq!(
            renderer
                .render_memory
                .published(gpui::RenderMemoryGauge::BackdropTextures),
            scratch_bytes
        );
        let held = renderer.device.current_allocated_size() - baseline;
        assert!(
            held <= scratch_bytes + 256 * 1024,
            "frosting holds {held} bytes of device memory beyond its {scratch_bytes} bytes of scratch"
        );
        assert!(
            scratch_bytes < 4 * 1024 * 1024,
            "four frosted surfaces at 2x hold {scratch_bytes} bytes"
        );
    }

    /// The scratch goes once no blur has used it for the idle window, and a
    /// scratch grown for a large surface gives way to one sized to the small
    /// surfaces that follow.
    #[test]
    fn backdrop_scratch_retires_when_idle_and_shrinks_to_the_need() {
        let mut renderer = MetalRenderer::new_headless(Arc::new(Mutex::new(
            InstanceBufferPool::default(),
        )));
        let size = gpui::size(DevicePixels(640), DevicePixels(480));
        let large = {
            let mut scene = Scene::default();
            scene.insert_primitive(quad(rect(0., 0., 640., 480.), gpui::hsla(0.5, 0.5, 0.5, 1.)));
            scene.insert_backdrop_blur(BackdropBlur {
                order: 0,
                blur_radius: ScaledPixels(4.),
                saturation: 1.,
                opacity: 1.,
                bounds: rect(0., 0., 640., 480.),
                content_mask: ContentMask { bounds: rect(0., 0., 640., 480.) },
                corner_radii: Corners::default(),
            });
            scene.finish();
            scene
        };
        let small = {
            let mut scene = Scene::default();
            scene.insert_primitive(quad(rect(0., 0., 640., 480.), gpui::hsla(0.5, 0.5, 0.5, 1.)));
            scene.insert_backdrop_blur(BackdropBlur {
                order: 0,
                blur_radius: ScaledPixels(4.),
                saturation: 1.,
                opacity: 1.,
                bounds: rect(10., 10., 64., 48.),
                content_mask: ContentMask { bounds: rect(10., 10., 64., 48.) },
                corner_radii: Corners::default(),
            });
            scene.finish();
            scene
        };
        let plain = {
            let mut scene = Scene::default();
            scene.insert_primitive(quad(rect(0., 0., 640., 480.), gpui::hsla(0.5, 0.5, 0.5, 1.)));
            scene.finish();
            scene
        };
        let draw = |renderer: &mut MetalRenderer, scene: &Scene| {
            renderer.render_scene(scene, size).expect("headless render succeeds");
        };

        draw(&mut renderer, &large);
        let grown = renderer.backdrop_scratch.as_ref().map(|scratch| scratch.texels());
        for _ in 0..=super::BACKDROP_SCRATCH_SHRINK_AFTER_USES + 1 {
            draw(&mut renderer, &small);
        }
        let shrunk = renderer.backdrop_scratch.as_ref().map(|scratch| scratch.texels());
        assert!(
            shrunk.zip(grown).is_some_and(|(shrunk, grown)| shrunk * 16 <= grown),
            "the scratch shrinks to the small surface: {grown:?} -> {shrunk:?}"
        );

        for _ in 0..=super::PATH_INTERMEDIATE_IDLE_FRAMES {
            draw(&mut renderer, &plain);
        }
        assert!(renderer.backdrop_scratch.is_none(), "an idle window gives the scratch back");
        assert_eq!(
            renderer
                .render_memory
                .published(gpui::RenderMemoryGauge::BackdropTextures),
            0
        );
    }

    /// The overlay plane frosts through the base renderer's scratch, handed
    /// over with the other intermediates, and a blur there keeps the base
    /// renderer from retiring it.
    #[test]
    fn the_scratch_travels_with_the_lent_intermediates() {
        let pool = Arc::new(Mutex::new(InstanceBufferPool::default()));
        let mut base = MetalRenderer::new_headless(pool.clone());
        let size = gpui::size(DevicePixels(640), DevicePixels(480));
        let mut scene = Scene::default();
        scene.insert_primitive(quad(rect(0., 0., 640., 480.), gpui::hsla(0.5, 0.5, 0.5, 1.)));
        scene.insert_backdrop_blur(BackdropBlur {
            order: 0,
            blur_radius: ScaledPixels(16.),
            saturation: 1.,
            opacity: 1.,
            bounds: rect(20., 20., 200., 100.),
            content_mask: ContentMask { bounds: rect(20., 20., 200., 100.) },
            corner_radii: Corners::default(),
        });
        scene.finish();
        base.render_scene(&scene, size).expect("headless render succeeds");
        base.backdrop_idle_frames = 50;
        let intermediates = base.take_intermediates();
        assert!(base.backdrop_scratch.is_none());
        let mut overlay = MetalRenderer::new_headless(pool);
        overlay.borrows_intermediates = true;
        overlay.lend_intermediates(intermediates);
        assert!(overlay.backdrop_scratch.is_some());
        overlay.render_scene(&scene, size).expect("headless render succeeds");
        assert_eq!(overlay.backdrop_idle_frames, 0, "a blur on the overlay plane counts as use");
        base.lend_intermediates(overlay.take_intermediates());
        assert!(base.backdrop_scratch.is_some());
        assert_eq!(base.backdrop_idle_frames, 0);
    }

    #[test]
    #[ignore = "measurement"]
    fn measure_small_region_accuracy() {
        let mut renderer = MetalRenderer::new_headless(Arc::new(Mutex::new(
            InstanceBufferPool::default(),
        )));
        let source = renderer
            .render_scene_to_image(&detailed_scene(&[], 0., 1., None), ACCURACY_TARGET)
            .expect("headless render succeeds");
        for downsample in [None, Some(1)] {
        renderer.backdrop_downsample_override = downsample;
        for region in [(700u32, 330u32, 200u32, 120u32), (300, 420, 96, 64), (1000, 100, 420, 500)] {
            let (left, top, width, height) = region;
            let bounds = rect(left as f32, top as f32, width as f32, height as f32);
            let reference = reference_frost(&source, region, 16., 1.4);
            let frost = renderer
                .render_scene_to_image(&detailed_scene(&[bounds], 16., 1.4, None), ACCURACY_TARGET)
                .expect("headless render succeeds");
            let (largest, mean, over_one, worst) = frost_error(&frost, &reference, region);
            let tint = gpui::hsla(0., 0., 0.1, 0.93);
            let tinted = renderer
                .render_scene_to_image(&detailed_scene(&[bounds], 16., 1.4, Some(tint)), ACCURACY_TARGET)
                .expect("headless render succeeds");
            let (tinted_largest, _, tinted_over, _) = frost_error(&tinted, &under_tint(&reference, 0.1, 0.93), region);
            println!("[small-accuracy] {downsample:?} {region:?}: frost largest {largest:.2} at {worst:?} mean {mean:.3} over-one {:.3}%; tinted largest {tinted_largest:.2} over-one {:.3}%", over_one * 100., tinted_over * 100.);
        }
        }
    }

    /// Run with `--ignored --nocapture`; prints how far the renderer's frost
    /// is from an exact gaussian.
    #[test]
    #[ignore = "measurement, prints the frost's error against a CPU reference"]
    fn measure_backdrop_blur_accuracy() {
        let mut renderer = MetalRenderer::new_headless(Arc::new(Mutex::new(
            InstanceBufferPool::default(),
        )));
        let source = renderer
            .render_scene_to_image(&detailed_scene(&[], 0., 1., None), ACCURACY_TARGET)
            .expect("headless render succeeds");
        // The frosted material (sigma 8pt, saturation 1.4, a dark gray tint at
        // 93%) and the command menu (24pt, 1.0, about 65%), at 2x.
        for (sigma, saturation, tint_alpha, downsamples) in [
            (16., 1.4, 0.93, vec![None, Some(1), Some(2), Some(4)]),
            (48., 1.0, 0.65, vec![None, Some(4)]),
        ] {
            let tint = gpui::hsla(0., 0., 0.1, tint_alpha);
            for downsample in downsamples {
                renderer.backdrop_downsample_override = downsample;
                for region in [(500u32, 300u32, 600u32, 400u32), (1300, 600, 300, 400)] {
                    let (left, top, width, height) = region;
                    let bounds = rect(left as f32, top as f32, width as f32, height as f32);
                    let reference = reference_frost(&source, region, sigma as f64, saturation as f64);
                    for tinted in [false, true] {
                        let rendered = renderer
                            .render_scene_to_image(
                                &detailed_scene(&[bounds], sigma, saturation, tinted.then_some(tint)),
                                ACCURACY_TARGET,
                            )
                            .expect("headless render succeeds");
                        let expected = if tinted {
                            under_tint(&reference, 0.1, tint_alpha as f64)
                        } else {
                            reference.clone()
                        };
                        let (largest, mean, over_one, worst) =
                            frost_error(&rendered, &expected, region);
                        println!(
                            "[backdrop-accuracy] sigma {sigma} downsample {downsample:?} {} region {width}x{height} at ({left},{top}): largest {largest:.2} at {worst:?} in the region, mean {mean:.3}, over one step {:.4}%",
                            if tinted { "tinted" } else { "frost " },
                            over_one * 100.
                        );
                    }
                }
            }
        }
    }

    /// Run with `--ignored --nocapture`; prints the measurement.
    #[test]
    #[ignore = "measurement, prints device memory per frosted surface"]
    fn measure_backdrop_blur_device_memory() {
        let mut renderer = MetalRenderer::new_headless(Arc::new(Mutex::new(
            InstanceBufferPool::default(),
        )));
        let plain = frosted_scene(&[]);
        render(&mut renderer, &plain);
        render(&mut renderer, &plain);
        let baseline = allocated(&renderer);

        // A picker at 600x800 points on a 2x display.
        let menu = rect(400., 300., 1200., 1600.);
        let one_menu = frosted_scene(&[menu]);
        render(&mut renderer, &one_menu);
        let first = allocated(&renderer);
        let first_scratch = scratch_textures(&renderer);
        for _ in 0..10 {
            render(&mut renderer, &one_menu);
        }
        let steady = allocated(&renderer);

        // A menu, a hover card, a composer and a sidebar card at once: four
        // distinct region sizes, the most the scratch cache keeps.
        let many = frosted_scene(&[
            menu,
            rect(1700., 200., 700., 420.),
            rect(600., 1900., 2200., 240.),
            rect(2500., 700., 800., 1100.),
        ]);
        for _ in 0..3 {
            render(&mut renderer, &many);
        }
        let four = allocated(&renderer);
        let four_scratch = scratch_textures(&renderer);

        // The idle retirement after PATH_INTERMEDIATE_IDLE_FRAMES frames.
        for _ in 0..(super::PATH_INTERMEDIATE_IDLE_FRAMES + 2) {
            render(&mut renderer, &plain);
        }
        let retired = allocated(&renderer);

        renderer.release_intermediate_textures();
        let released = allocated(&renderer);

        println!("[backdrop-memory] baseline (plain frames)          {baseline:8.2} MiB");
        println!(
            "[backdrop-memory] one 1200x1600 blur, first frame  {first:8.2} MiB (+{:.2}; scratch textures {first_scratch:.2})",
            first - baseline
        );
        println!(
            "[backdrop-memory] one blur, after 10 more frames   {steady:8.2} MiB (+{:.2})",
            steady - baseline
        );
        println!(
            "[backdrop-memory] four blurs                       {four:8.2} MiB (+{:.2}; scratch textures {four_scratch:.2})",
            four - baseline
        );
        println!(
            "[backdrop-memory] {} plain frames later            {retired:8.2} MiB (+{:.2})",
            super::PATH_INTERMEDIATE_IDLE_FRAMES + 2,
            retired - baseline
        );
        println!(
            "[backdrop-memory] after release_intermediate_textures {released:8.2} MiB (+{:.2})",
            released - baseline
        );
    }
}

/// GPU time a frame spends frosting, from the command buffer's own GPU start
/// and end timestamps: the same frame drawn with and without its blurs.
#[cfg(test)]
mod backdrop_blur_timing_tests {
    use super::{InstanceBufferPool, MetalRenderer};
    use gpui::{
        BackdropBlur, Background, Bounds, ContentMask, Corners, DevicePixels, Edges, Hsla, Point,
        Quad, ScaledPixels, Scene, Size,
    };
    use foreign_types::ForeignType;
    use objc2::{msg_send, runtime::AnyObject};
    use parking_lot::Mutex;
    use std::sync::Arc;

    const TARGET: Size<DevicePixels> = Size {
        width: DevicePixels(3456),
        height: DevicePixels(2168),
    };

    fn rect(x: f32, y: f32, width: f32, height: f32) -> Bounds<ScaledPixels> {
        Bounds {
            origin: Point {
                x: ScaledPixels(x),
                y: ScaledPixels(y),
            },
            size: gpui::size(ScaledPixels(width), ScaledPixels(height)),
        }
    }

    fn quad(bounds: Bounds<ScaledPixels>, color: Hsla) -> Quad {
        Quad {
            rounded_clip: 0,
            clip_padding: 0,
            order: 0,
            border_style: Default::default(),
            bounds,
            content_mask: ContentMask { bounds },
            background: Background::from(color),
            border_color: Hsla::transparent_black(),
            corner_radii: Corners::default(),
            border_widths: Edges::default(),
        }
    }

    fn scene(blurs: &[(Bounds<ScaledPixels>, f32)]) -> Scene {
        let mut scene = Scene::default();
        let mut x = 0.;
        let mut index = 0;
        while x < TARGET.width.0 as f32 {
            let hue = (index % 12) as f32 / 12.;
            scene.insert_primitive(quad(
                rect(x, 0., 24., TARGET.height.0 as f32),
                gpui::hsla(hue, 0.7, 0.5, 1.),
            ));
            x += 24.;
            index += 1;
        }
        for (bounds, sigma) in blurs {
            scene.insert_backdrop_blur(BackdropBlur {
                order: 0,
                blur_radius: ScaledPixels(*sigma),
                saturation: 1.4,
                opacity: 1.,
                bounds: *bounds,
                content_mask: ContentMask { bounds: *bounds },
                corner_radii: Corners::all(ScaledPixels(16.)),
            });
            scene.insert_primitive(quad(*bounds, gpui::hsla(0., 0., 0.1, 0.93)));
        }
        scene.finish();
        scene
    }

    fn frame_gpu_milliseconds(
        renderer: &mut MetalRenderer,
        scene: &Scene,
        target: &metal::TextureRef,
    ) -> f64 {
        objc2::rc::autoreleasepool(|_| {
            renderer.update_intermediate_textures(TARGET);
            let command_buffer = renderer
                .render_frame(scene, target, TARGET, None, None)
                .expect("frame encodes");
            command_buffer.commit();
            command_buffer.wait_until_completed();
            let buffer = command_buffer.as_ptr().cast::<AnyObject>();
            // SAFETY: `GPUStartTime` and `GPUEndTime` are `CFTimeInterval`
            // properties of every completed `MTLCommandBuffer`.
            let (start, end): (f64, f64) =
                unsafe { (msg_send![buffer, GPUStartTime], msg_send![buffer, GPUEndTime]) };
            (end - start) * 1000.
        })
    }

    /// Median over `frames` pairs of the GPU milliseconds `frosted` takes
    /// beyond `plain`, the two drawn alternately so that anything else the
    /// GPU is doing weighs on both alike; and the median plain frame.
    fn gpu_milliseconds(
        renderer: &mut MetalRenderer,
        plain: &Scene,
        frosted: &Scene,
        frames: usize,
    ) -> (f64, f64) {
        let descriptor = metal::TextureDescriptor::new();
        descriptor.set_width(TARGET.width.0 as u64);
        descriptor.set_height(TARGET.height.0 as u64);
        descriptor.set_pixel_format(metal::MTLPixelFormat::BGRA8Unorm);
        descriptor.set_usage(
            metal::MTLTextureUsage::RenderTarget | metal::MTLTextureUsage::ShaderRead,
        );
        descriptor.set_storage_mode(metal::MTLStorageMode::Private);
        let target = renderer.device.new_texture(&descriptor);
        let mut differences = Vec::with_capacity(frames);
        let mut plains = Vec::with_capacity(frames);
        for _ in 0..frames {
            let plain_time = frame_gpu_milliseconds(renderer, plain, &target);
            let frosted_time = frame_gpu_milliseconds(renderer, frosted, &target);
            differences.push(frosted_time - plain_time);
            plains.push(plain_time);
        }
        differences.sort_by(f64::total_cmp);
        plains.sort_by(f64::total_cmp);
        (differences[frames / 2], plains[frames / 2])
    }

    /// Run with `--ignored --nocapture`; prints GPU time per frosted frame.
    #[test]
    #[ignore = "measurement, prints GPU time spent frosting"]
    fn measure_backdrop_blur_gpu_time() {
        let mut renderer = MetalRenderer::new_headless(Arc::new(Mutex::new(
            InstanceBufferPool::default(),
        )));
        let plain = scene(&[]);
        for (label, blurs) in [
            ("picker 1200x1600 sigma 16", vec![(rect(400., 300., 1200., 1600.), 16.)]),
            ("composer 1400x260 sigma 16", vec![(rect(1000., 1800., 1400., 260.), 16.)]),
            ("command menu 2400x1500 sigma 48", vec![(rect(500., 300., 2400., 1500.), 48.)]),
        ] {
            let frosted = scene(&blurs);
            gpu_milliseconds(&mut renderer, &plain, &frosted, 10);
            let (extra, plain_frame) = gpu_milliseconds(&mut renderer, &plain, &frosted, 200);
            println!(
                "[backdrop-gpu] {label}: {extra:.3} ms over the plain frame ({plain_frame:.3} ms), median of 200 alternating pairs"
            );
        }
    }

    /// A window's surfaces stacked the way an app paints them: a background,
    /// a frame, cards and panels each covering most of the window, rows of
    /// text-sized fills on top, and a frosted composer card near the bottom.
    fn stacked_scene(layers: usize, frost_composer: bool) -> Scene {
        let mut scene = Scene::default();
        let width = TARGET.width.0 as f32;
        let height = TARGET.height.0 as f32;
        for layer in 0..layers {
            let inset = layer as f32 * 8.;
            let lightness = 0.08 + layer as f32 * 0.01;
            scene.insert_primitive(quad(
                rect(inset, inset, width - inset * 2., height - inset * 2.),
                gpui::hsla(0.6, 0.05, lightness, 1.),
            ));
        }
        let mut y = 200.;
        while y < height - 400. {
            let mut x = 400.;
            while x < width - 400. {
                scene.insert_primitive(quad(rect(x, y, 14., 22.), gpui::hsla(0., 0., 0.9, 1.)));
                x += 18.;
            }
            y += 44.;
        }
        if frost_composer {
            let composer = rect(1000., 1800., 1400., 260.);
            scene.push_layer(composer);
            scene.insert_backdrop_blur(BackdropBlur {
                order: 0,
                blur_radius: ScaledPixels(16.),
                saturation: 1.4,
                opacity: 1.,
                bounds: composer,
                content_mask: ContentMask { bounds: composer },
                corner_radii: Corners::all(ScaledPixels(16.)),
            });
            scene.insert_primitive(quad(composer, gpui::hsla(0., 0., 0.1, 0.93)));
            scene.pop_layer();
        } else {
            scene.insert_primitive(quad(
                rect(1000., 1800., 1400., 260.),
                gpui::hsla(0., 0., 0.1, 0.93),
            ));
        }
        scene.finish();
        scene
    }

    /// Run with `--ignored --nocapture`; prints what one frosted surface costs
    /// a frame whose opaque surfaces overlap, against the same frame unfrosted.
    #[test]
    #[ignore = "measurement, prints GPU time of a frosted frame over stacked surfaces"]
    fn measure_backdrop_blur_gpu_time_over_stacked_surfaces() {
        let mut renderer = MetalRenderer::new_headless(Arc::new(Mutex::new(
            InstanceBufferPool::default(),
        )));
        for layers in [4, 12] {
            let plain = stacked_scene(layers, false);
            let frosted = stacked_scene(layers, true);
            gpu_milliseconds(&mut renderer, &plain, &frosted, 10);
            let (extra, plain_frame) = gpu_milliseconds(&mut renderer, &plain, &frosted, 200);
            println!(
                "[backdrop-gpu] {layers} stacked surfaces + frosted composer: {extra:.3} ms over the plain frame ({plain_frame:.3} ms), median of 200 alternating pairs"
            );
        }
    }
}

#[cfg(test)]
mod hidden_drawable_tests {
    use super::{InstanceBufferPool, MetalRenderer};
    use gpui::{DevicePixels, PresentOutcome, Scene, size};
    use parking_lot::Mutex;
    use std::sync::Arc;

    fn layer_size(renderer: &MetalRenderer) -> (f64, f64) {
        let layer = renderer.layer().expect("a window renderer has a layer");
        let drawable = layer.drawable_size();
        (drawable.width, drawable.height)
    }

    /// Frosting samples the drawable it blurs, so a window's drawables must
    /// be shader-readable in every build, not only in test builds.
    #[test]
    fn window_drawables_can_be_sampled() {
        let renderer = MetalRenderer::new(Arc::new(Mutex::new(InstanceBufferPool::default())), false, false);
        let layer = renderer.layer().expect("a window renderer has a layer");
        assert!(!layer.framebuffer_only());
    }

    /// A hidden window's layer holds single-pixel drawables, keeps the size
    /// the window wants through resizes made meanwhile, draws nothing, and
    /// gets that size back when shown.
    #[test]
    fn a_hidden_window_shrinks_its_drawables_and_restores_the_latest_size() {
        let mut renderer = MetalRenderer::new(Arc::new(Mutex::new(InstanceBufferPool::default())), false, false);
        renderer.update_drawable_size(size(DevicePixels(3456), DevicePixels(2168)));
        assert_eq!(layer_size(&renderer), (3456., 2168.));

        renderer.release_drawables();
        assert_eq!(layer_size(&renderer), (1., 1.));
        renderer.update_drawable_size(size(DevicePixels(1728), DevicePixels(1084)));
        assert_eq!(layer_size(&renderer), (1., 1.), "a resize while hidden waits");

        let mut scene = Scene::default();
        scene.finish();
        assert_eq!(
            renderer.draw(&scene).outcome,
            PresentOutcome::Deferred,
            "a frame drawn while hidden stays owed instead of presenting a pixel"
        );

        renderer.restore_drawables();
        assert_eq!(layer_size(&renderer), (1728., 1084.));
        renderer.restore_drawables();
        assert_eq!(layer_size(&renderer), (1728., 1084.), "restoring twice is harmless");
    }
}

/// A present that only advances time animations copies the frame presented
/// last and draws just what moved over the clear colour. Whatever the copy
/// brings, the result must be the frame a full draw gives: through a
/// translucent target, around a backdrop blur elsewhere in the window, and
/// with nothing of the previous frame left where the damage is.
#[cfg(test)]
mod partial_present_tests {
    use super::{InstanceBufferPool, MetalRenderer, read_texture_to_image, undamaged_rects};
    use gpui::{
        BackdropBlur, Background, Bounds, ContentMask, Corners, DevicePixels, Edges, Hsla,
        OpacityCycle, Point, Quad, ScaledPixels, Scene, Underline,
    };
    use parking_lot::Mutex;
    use std::sync::Arc;

    const SIZE: gpui::Size<DevicePixels> = gpui::Size {
        width: DevicePixels(96),
        height: DevicePixels(64),
    };

    fn rect(x: f32, y: f32, width: f32, height: f32) -> Bounds<ScaledPixels> {
        Bounds {
            origin: Point {
                x: ScaledPixels(x),
                y: ScaledPixels(y),
            },
            size: gpui::size(ScaledPixels(width), ScaledPixels(height)),
        }
    }

    fn quad(bounds: Bounds<ScaledPixels>, color: Hsla) -> Quad {
        Quad {
            rounded_clip: 0,
            clip_padding: 0,
            order: 0,
            border_style: Default::default(),
            bounds,
            content_mask: ContentMask { bounds },
            background: Background::from(color),
            border_color: Hsla::transparent_black(),
            corner_radii: Corners::all(ScaledPixels(3.)),
            border_widths: Edges::default(),
        }
    }

    fn target(renderer: &MetalRenderer) -> metal::Texture {
        let descriptor = metal::TextureDescriptor::new();
        descriptor.set_width(SIZE.width.0 as u64);
        descriptor.set_height(SIZE.height.0 as u64);
        descriptor.set_pixel_format(metal::MTLPixelFormat::BGRA8Unorm);
        descriptor.set_usage(metal::MTLTextureUsage::RenderTarget | metal::MTLTextureUsage::ShaderRead);
        descriptor.set_storage_mode(if renderer.is_apple_gpu {
            metal::MTLStorageMode::Shared
        } else {
            metal::MTLStorageMode::Managed
        });
        renderer.device.new_texture(&descriptor)
    }

    fn draw(
        renderer: &mut MetalRenderer,
        scene: &Scene,
        texture: &metal::TextureRef,
        previous_frame: Option<&metal::TextureRef>,
    ) -> image::RgbaImage {
        let command_buffer = renderer
            .render_frame(scene, texture, SIZE, None, previous_frame)
            .expect("headless render succeeds");
        if !renderer.is_apple_gpu {
            let blit = command_buffer.new_blit_command_encoder();
            blit.synchronize_resource(texture);
            blit.end_encoding();
        }
        command_buffer.commit();
        command_buffer.wait_until_completed();
        read_texture_to_image(texture).expect("readback succeeds")
    }

    fn pulsing_scene() -> Scene {
        let mut scene = Scene::default();
        for index in 0..8 {
            let alpha = if index % 2 == 0 { 1. } else { 0.6 };
            scene.insert_primitive(quad(
                rect(index as f32 * 12., 0., 12., 64.),
                gpui::hsla(index as f32 / 8., 0.7, 0.5, alpha),
            ));
        }
        // Opaque cards over opaque stripes under the lone pulses: the depth
        // prepass of each damaged rectangle has to keep them apart.
        scene.insert_primitive(quad(rect(2., 2., 16., 14.), gpui::hsla(0., 0., 0.4, 1.)));
        scene.insert_primitive(quad(rect(26., 36., 14., 14.), gpui::hsla(0., 0., 0.7, 1.)));
        let cycle = OpacityCycle::new(
            std::time::Duration::from_millis(900),
            [(0.0, 0.2), (0.5, 1.0), (0.75, 0.6), (1.0, 0.2)],
        );
        // Beneath the frosted surface, so the frosting changes as it pulses.
        scene.insert_opacity_cycled_quad(
            quad(rect(78., 48., 6., 6.), gpui::hsla(0.3, 0.9, 0.5, 1.)),
            cycle.phase(0.25),
        );
        let frosted = rect(60., 28., 32., 32.);
        scene.insert_backdrop_blur(BackdropBlur {
            order: 0,
            blur_radius: ScaledPixels(3.),
            saturation: 1.4,
            opacity: 1.,
            bounds: frosted,
            content_mask: ContentMask { bounds: frosted },
            corner_radii: Corners::all(ScaledPixels(6.)),
        });
        scene.insert_primitive(quad(frosted, gpui::hsla(0., 0., 0.1, 0.5)));
        // On the frosted surface, as a spinner in a frosted composer is.
        scene.insert_opacity_cycled_quad(
            quad(rect(66., 34., 8., 8.), gpui::hsla(0.8, 0.8, 0.6, 1.)),
            cycle.phase(0.75),
        );
        scene.insert_opacity_cycled_quad(quad(rect(6., 6., 10., 8.), gpui::hsla(0.6, 0.8, 0.6, 1.)), cycle);
        scene.insert_opacity_cycled_quad(
            quad(rect(30., 40., 8., 8.), gpui::hsla(0.1, 0.8, 0.6, 1.)),
            cycle.phase(0.5),
        );
        // A batch that collapses the depth range, ordered last because it
        // crosses the pulse and the frosted surface, ends every rectangle.
        let underlined = rect(0., 44., 96., 2.);
        scene.insert_primitive(Underline {
            order: 0,
            pad: 0,
            bounds: underlined,
            content_mask: ContentMask { bounds: underlined },
            color: gpui::hsla(0.5, 0.5, 0.5, 1.),
            thickness: ScaledPixels(2.),
            wavy: false.into(),
            rounded_clip: 0,
            clip_padding: 0,
        });
        scene.finish();
        scene
    }

    #[test]
    fn a_partial_present_draws_what_a_full_one_does() {
        let mut renderer = MetalRenderer::new_headless(Arc::new(Mutex::new(
            InstanceBufferPool::default(),
        )));
        renderer.update_transparency(true);
        renderer.update_intermediate_textures(SIZE);
        let mut scene = pulsing_scene();

        scene.advance_time_animations();
        let previous = target(&renderer);
        draw(&mut renderer, &scene, &previous, None);

        scene.advance_time_animations();
        assert!(scene.time_animation_damage(SIZE, &mut renderer.damage_rects));
        assert_eq!(
            renderer.damage_rects.len(),
            3,
            "one rectangle per lone pulse, and one holding the frosted surface and both pulses on it"
        );
        let magenta = [255u8, 0, 255, 255];
        for rect in &renderer.damage_rects {
            let width = rect.size.width.0 as usize;
            let height = rect.size.height.0 as usize;
            let pixels = magenta.repeat(width * height);
            previous.replace_region(
                metal::MTLRegion {
                    origin: metal::MTLOrigin {
                        x: rect.origin.x.0 as u64,
                        y: rect.origin.y.0 as u64,
                        z: 0,
                    },
                    size: metal::MTLSize {
                        width: width as u64,
                        height: height as u64,
                        depth: 1,
                    },
                },
                0,
                pixels.as_ptr().cast(),
                (width * 4) as u64,
            );
        }
        let partial_target = target(&renderer);
        let partial = draw(&mut renderer, &scene, &partial_target, Some(&previous));
        let full_target = target(&renderer);
        let full = draw(&mut renderer, &scene, &full_target, None);

        assert_eq!((renderer.partial_presents, renderer.full_presents), (1, 2));
        let mismatch = full
            .enumerate_pixels()
            .find(|(x, y, pixel)| partial.get_pixel(*x, *y) != *pixel);
        assert!(
            mismatch.is_none(),
            "partial present differs at {:?}: partial {:?}",
            mismatch,
            mismatch.map(|(x, y, _)| partial.get_pixel(x, y)),
        );
    }

    #[test]
    fn undamaged_rects_tile_the_rest_of_the_viewport() {
        let device_rect = |left: i32, top: i32, right: i32, bottom: i32| Bounds {
            origin: gpui::point(DevicePixels(left), DevicePixels(top)),
            size: gpui::size(DevicePixels(right - left), DevicePixels(bottom - top)),
        };
        for damage in [
            vec![],
            vec![device_rect(0, 0, 96, 64)],
            vec![device_rect(10, 10, 30, 30)],
            vec![device_rect(10, 10, 30, 30), device_rect(40, 5, 50, 55), device_rect(0, 50, 8, 64)],
        ] {
            let undamaged = undamaged_rects(&damage, SIZE);
            for y in 0..SIZE.height.0 {
                for x in 0..SIZE.width.0 {
                    let covers = |rect: &&Bounds<DevicePixels>| {
                        rect.origin.x.0 <= x
                            && x < rect.origin.x.0 + rect.size.width.0
                            && rect.origin.y.0 <= y
                            && y < rect.origin.y.0 + rect.size.height.0
                    };
                    let covered = damage.iter().filter(covers).count()
                        + undamaged.iter().filter(covers).count();
                    assert_eq!(covered, 1, "pixel ({x}, {y}) of {damage:?} covered {covered} times");
                }
            }
        }
    }
}

/// Run with `--ignored --nocapture`; prints the GPU time of a full-screen
/// frame drawn whole and of the same frame presented again for a pulse.
#[cfg(test)]
mod partial_present_timing_tests {
    use super::{InstanceBufferPool, MetalRenderer};
    use gpui::{
        BackdropBlur, Background, Bounds, ContentMask, Corners, DevicePixels, Edges, Hsla,
        OpacityCycle, Point, Quad, ScaledPixels, Scene, Size,
    };
    use foreign_types::ForeignType;
    use objc2::{msg_send, runtime::AnyObject};
    use parking_lot::Mutex;
    use std::sync::Arc;

    const TARGET: Size<DevicePixels> = Size {
        width: DevicePixels(3456),
        height: DevicePixels(2168),
    };

    fn quad(x: f32, y: f32, width: f32, height: f32, color: Hsla) -> Quad {
        let bounds = Bounds {
            origin: Point {
                x: ScaledPixels(x),
                y: ScaledPixels(y),
            },
            size: gpui::size(ScaledPixels(width), ScaledPixels(height)),
        };
        Quad {
            rounded_clip: 0,
            clip_padding: 0,
            order: 0,
            border_style: Default::default(),
            bounds,
            content_mask: ContentMask { bounds },
            background: Background::from(color),
            border_color: Hsla::transparent_black(),
            corner_radii: Corners::all(ScaledPixels(16.)),
            border_widths: Edges::default(),
        }
    }

    /// A translucent window's worth of stacked translucent panels, a frosted
    /// composer, and the two things that keep a turn's window presenting: a
    /// spinner pulse in the sidebar and one in the transcript.
    fn window_scene() -> Scene {
        let mut scene = Scene::default();
        let (width, height) = (TARGET.width.0 as f32, TARGET.height.0 as f32);
        scene.insert_primitive(quad(0., 0., width, height, gpui::hsla(0.6, 0.1, 0.12, 0.85)));
        for index in 0..12 {
            let inset = index as f32 * 40.;
            scene.insert_primitive(quad(
                inset,
                inset,
                width - inset * 2.,
                height - inset * 2.,
                gpui::hsla(index as f32 / 12., 0.3, 0.3, 0.4),
            ));
        }
        let composer = (900., 1800., 1700., 300.);
        let composer_bounds = Bounds {
            origin: Point {
                x: ScaledPixels(composer.0),
                y: ScaledPixels(composer.1),
            },
            size: gpui::size(ScaledPixels(composer.2), ScaledPixels(composer.3)),
        };
        scene.insert_backdrop_blur(BackdropBlur {
            order: 0,
            blur_radius: ScaledPixels(24.),
            saturation: 1.4,
            opacity: 1.,
            bounds: composer_bounds,
            content_mask: ContentMask {
                bounds: composer_bounds,
            },
            corner_radii: Corners::all(ScaledPixels(24.)),
        });
        scene.insert_primitive(quad(composer.0, composer.1, composer.2, composer.3, gpui::hsla(0., 0., 0.1, 0.8)));
        let cycle = OpacityCycle::new(
            std::time::Duration::from_millis(900),
            [(0.0, 0.2), (0.5, 1.0), (0.75, 0.6), (1.0, 0.2)],
        );
        scene.insert_opacity_cycled_quad(quad(60., 400., 24., 24., gpui::hsla(0.6, 0.8, 0.6, 1.)), cycle);
        scene.insert_opacity_cycled_quad(
            quad(1200., 1200., 36., 36., gpui::hsla(0.1, 0.8, 0.6, 1.)),
            cycle.phase(0.5),
        );
        scene.finish();
        scene
    }

    fn target(renderer: &MetalRenderer) -> metal::Texture {
        let descriptor = metal::TextureDescriptor::new();
        descriptor.set_width(TARGET.width.0 as u64);
        descriptor.set_height(TARGET.height.0 as u64);
        descriptor.set_pixel_format(metal::MTLPixelFormat::BGRA8Unorm);
        descriptor.set_usage(metal::MTLTextureUsage::RenderTarget | metal::MTLTextureUsage::ShaderRead);
        descriptor.set_storage_mode(metal::MTLStorageMode::Private);
        renderer.device.new_texture(&descriptor)
    }

    fn frame_gpu_milliseconds(
        renderer: &mut MetalRenderer,
        scene: &Scene,
        target: &metal::TextureRef,
        previous_frame: Option<&metal::TextureRef>,
    ) -> f64 {
        objc2::rc::autoreleasepool(|_| {
            let command_buffer = renderer
                .render_frame(scene, target, TARGET, None, previous_frame)
                .expect("frame encodes");
            command_buffer.commit();
            command_buffer.wait_until_completed();
            let buffer = command_buffer.as_ptr().cast::<AnyObject>();
            // SAFETY: `GPUStartTime` and `GPUEndTime` are `CFTimeInterval`
            // properties of every completed `MTLCommandBuffer`.
            let (start, end): (f64, f64) =
                unsafe { (msg_send![buffer, GPUStartTime], msg_send![buffer, GPUEndTime]) };
            (end - start) * 1000.
        })
    }

    /// One attribution round renders the frame whole, cleared and without
    /// each kind, and its line names what this scene draws.
    #[test]
    fn an_attribution_round_times_every_kind() {
        let mut renderer = MetalRenderer::new_headless(Arc::new(Mutex::new(
            InstanceBufferPool::default(),
        )));
        renderer.update_transparency(true);
        renderer.update_intermediate_textures(TARGET);
        renderer.gpu_attribution = Some(super::GpuAttribution {
            presents_since_round: 0,
            rounds: 0,
            target: None,
            samples: Arc::default(),
        });
        let mut scene = window_scene();
        scene.advance_time_animations();
        let rounds = 5;
        for _ in 0..super::ATTRIBUTION_PRESENT_STRIDE * rounds {
            renderer.maybe_attribute_gpu_time(&scene, TARGET);
        }
        // Command buffers complete in queue order.
        let fence = renderer.command_queue.new_command_buffer();
        fence.commit();
        fence.wait_until_completed();
        let samples = renderer
            .gpu_attribution
            .as_ref()
            .map(|attribution| attribution.samples.lock().clone())
            .unwrap_or_default();
        assert_eq!(samples.len(), (3 + super::ATTRIBUTED_KINDS.len()) * rounds as usize);
        assert_eq!((renderer.full_presents, renderer.attribution_skips, renderer.attributing), (0, 0, false));
        let line = super::gpu_attribution_line(&samples, &scene, TARGET).expect("a full and a clear render");
        println!("{line}");
        assert!(line.contains("blended quads") && line.contains("backdrop blurs"), "{line}");
        assert!(!line.contains("glyphs"), "a scene without text has no glyph entry: {line}");
    }

    #[test]
    #[ignore = "measurement, prints GPU time of full and partial presents"]
    fn measure_partial_present_gpu_time() {
        let mut renderer = MetalRenderer::new_headless(Arc::new(Mutex::new(
            InstanceBufferPool::default(),
        )));
        renderer.update_transparency(true);
        renderer.update_intermediate_textures(TARGET);
        let mut scene = window_scene();
        let targets = [target(&renderer), target(&renderer)];
        let frames = 60;
        let (mut full, mut partial) = (Vec::with_capacity(frames), Vec::with_capacity(frames));
        scene.advance_time_animations();
        frame_gpu_milliseconds(&mut renderer, &scene, &targets[0], None);
        for frame in 0..frames {
            let (previous, next) = (&targets[frame % 2], &targets[(frame + 1) % 2]);
            scene.advance_time_animations();
            full.push(frame_gpu_milliseconds(&mut renderer, &scene, next, None));
            assert!(scene.time_animation_damage(TARGET, &mut renderer.damage_rects));
            partial.push(frame_gpu_milliseconds(&mut renderer, &scene, next, Some(previous)));
        }
        full.sort_by(f64::total_cmp);
        partial.sort_by(f64::total_cmp);
        println!(
            "{}x{}: full present {:.3} ms, partial present {:.3} ms (median of {frames})",
            TARGET.width.0,
            TARGET.height.0,
            full[frames / 2],
            partial[frames / 2],
        );
    }
}
