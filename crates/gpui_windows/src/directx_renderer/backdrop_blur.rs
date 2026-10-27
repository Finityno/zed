use super::*;

/// Drawn frames without a backdrop blur after which the scratch textures are
/// released. Frames rather than time, for the reason given at
/// [`PATH_INTERMEDIATE_IDLE_FRAMES`].
const BACKDROP_BLUR_IDLE_FRAMES: u32 = 120;
/// Scratch texture sets kept, one per bucketed region size. A frame with more
/// blurs of distinct sizes than this recreates the least recently used set.
const MAX_BACKDROP_TEXTURES: usize = 4;
/// Region sizes are rounded up to this many pixels before looking for a
/// scratch set, so a popover animating its size reuses one set instead of
/// creating textures every frame.
const BACKDROP_TEXTURE_GRANULARITY: u32 = 64;
/// The largest gaussian standard deviation run, in device pixels. Beyond it
/// the frost no longer changes visibly while the work keeps growing.
const MAX_BACKDROP_BLUR_SIGMA: f32 = 64.;
/// `backdrop_taps` in shaders.hlsl: two folded tap pairs per vector, enough
/// for the 3-sigma reach of [`MAX_BACKDROP_BLUR_SIGMA`].
const BACKDROP_TAP_VECTORS: usize = 48;

/// `cbuffer BackdropBlurParams` (b2) in shaders.hlsl. The first three fields
/// each fill a 16-byte register, the paired fields after them two registers
/// each and the four scalars one more, so no field straddles a register.
#[derive(Clone, Copy)]
#[repr(C)]
struct BackdropBlurParams {
    bounds: [f32; 4],
    corner_radii: [f32; 4],
    content_mask: [f32; 4],
    region_origin: [f32; 2],
    region_size: [f32; 2],
    source_size: [f32; 2],
    source_extent: [f32; 2],
    direction: [f32; 2],
    scale: [f32; 2],
    saturation: f32,
    opacity: f32,
    center_weight: f32,
    tap_pair_count: u32,
    taps: [[f32; 4]; BACKDROP_TAP_VECTORS],
}

const _: () = assert!(std::mem::size_of::<BackdropBlurParams>() == (7 + BACKDROP_TAP_VECTORS) * 16);

fn bounds_to_array(bounds: &Bounds<ScaledPixels>) -> [f32; 4] {
    [
        bounds.origin.x.0,
        bounds.origin.y.0,
        bounds.size.width.0,
        bounds.size.height.0,
    ]
}

/// Writes the normalized gaussian for `sigma` as folded tap pairs: taps `k`
/// and `k + 1` (odd `k`) become one bilinear sample between them carrying
/// both weights, which halves the samples of every pass. Returns the centre
/// tap's weight and the number of pairs written.
fn gaussian_tap_pairs(sigma: f32, taps: &mut [[f32; 4]; BACKDROP_TAP_VECTORS]) -> (f32, u32) {
    let radius = ((sigma * 3.).ceil() as usize).min(BACKDROP_TAP_VECTORS * 4);
    let weight = |k: usize| {
        if k > radius {
            0.
        } else {
            let k = k as f32;
            (-(k * k) / (2. * sigma * sigma)).exp()
        }
    };
    let total = 1. + 2. * (1..=radius).map(weight).sum::<f32>();
    let pair_count = radius.div_ceil(2);
    for pair in 0..pair_count {
        let near = 2 * pair + 1;
        let near_weight = weight(near);
        let far_weight = weight(near + 1);
        let combined = near_weight + far_weight;
        let offset = near as f32 + far_weight / combined;
        let slot = &mut taps[pair / 2][(pair % 2) * 2..][..2];
        slot[0] = offset;
        slot[1] = combined / total;
    }
    (1. / total, pair_count as u32)
}

struct BackdropTexture {
    texture: ID3D11Texture2D,
    view: Option<ID3D11ShaderResourceView>,
    render_target_view: Option<ID3D11RenderTargetView>,
    width: u32,
    height: u32,
}

impl BackdropTexture {
    fn new(device: &ID3D11Device, width: u32, height: u32, render_target: bool) -> Result<Self> {
        let bind_flags = if render_target {
            D3D11_BIND_SHADER_RESOURCE.0 | D3D11_BIND_RENDER_TARGET.0
        } else {
            D3D11_BIND_SHADER_RESOURCE.0
        };
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
            BindFlags: bind_flags as u32,
            CPUAccessFlags: 0,
            MiscFlags: 0,
        };
        let mut texture = None;
        unsafe { device.CreateTexture2D(&desc, None, Some(&mut texture))? };
        let texture = texture.context("Creating backdrop blur texture")?;
        let mut view = None;
        unsafe { device.CreateShaderResourceView(&texture, None, Some(&mut view))? };
        let mut render_target_view = None;
        if render_target {
            unsafe { device.CreateRenderTargetView(&texture, None, Some(&mut render_target_view))? };
        }
        Ok(Self {
            texture,
            view,
            render_target_view,
            width,
            height,
        })
    }
}

/// The scratch textures one blur region goes through, sized for a bucketed
/// region of `width` × `height`: `source` holds the region as drawn so far,
/// `horizontal` and `blurred` the two gaussian passes, each `downsample`
/// times smaller along the axes blurred so far.
struct BackdropTextures {
    width: u32,
    height: u32,
    downsample: u32,
    source: BackdropTexture,
    horizontal: BackdropTexture,
    blurred: BackdropTexture,
    /// The overlay plane's own pixels under the region. On that plane
    /// `source` holds the base plane with the overlay composited over it,
    /// which is what the blur frosts, while the unblurred pixel the effect
    /// mixes with is the overlay's own.
    plane_copy: Option<BackdropTexture>,
}

impl BackdropTextures {
    fn new(device: &ID3D11Device, width: u32, height: u32, downsample: u32) -> Result<Self> {
        let reduced_width = width.div_ceil(downsample);
        Ok(Self {
            width,
            height,
            downsample,
            source: BackdropTexture::new(device, width, height, true)?,
            horizontal: BackdropTexture::new(device, reduced_width, height, true)?,
            blurred: BackdropTexture::new(
                device,
                reduced_width,
                height.div_ceil(downsample),
                true,
            )?,
            plane_copy: None,
        })
    }
}

struct BackdropBlurPipeline {
    pass_vertex: ID3D11VertexShader,
    pass_fragment: ID3D11PixelShader,
    composite_vertex: ID3D11VertexShader,
    composite_fragment: ID3D11PixelShader,
    params_buffer: Option<ID3D11Buffer>,
    sampler: Option<ID3D11SamplerState>,
    replace_blend_state: ID3D11BlendState,
    /// Premultiplied source-over, for compositing the overlay plane's pixels
    /// over the base plane's before blurring.
    over_blend_state: ID3D11BlendState,
}

impl BackdropBlurPipeline {
    fn new(device: &ID3D11Device) -> Result<Self> {
        let pass_vertex = {
            let raw_shader = RawShaderBytes::new(ShaderModule::BackdropBlurPass, ShaderTarget::Vertex)?;
            create_vertex_shader(device, raw_shader.as_bytes())?
        };
        let pass_fragment = {
            let raw_shader =
                RawShaderBytes::new(ShaderModule::BackdropBlurPass, ShaderTarget::Fragment)?;
            create_fragment_shader(device, raw_shader.as_bytes())?
        };
        let composite_vertex = {
            let raw_shader = RawShaderBytes::new(ShaderModule::BackdropBlur, ShaderTarget::Vertex)?;
            create_vertex_shader(device, raw_shader.as_bytes())?
        };
        let composite_fragment = {
            let raw_shader = RawShaderBytes::new(ShaderModule::BackdropBlur, ShaderTarget::Fragment)?;
            create_fragment_shader(device, raw_shader.as_bytes())?
        };
        // Clamped rather than the sprite sampler's wrap: a tap past the
        // render target's edge must repeat the edge, not the opposite side.
        let sampler_desc = D3D11_SAMPLER_DESC {
            Filter: D3D11_FILTER_MIN_MAG_MIP_LINEAR,
            AddressU: D3D11_TEXTURE_ADDRESS_CLAMP,
            AddressV: D3D11_TEXTURE_ADDRESS_CLAMP,
            AddressW: D3D11_TEXTURE_ADDRESS_CLAMP,
            MipLODBias: 0.0,
            MaxAnisotropy: 1,
            ComparisonFunc: D3D11_COMPARISON_ALWAYS,
            BorderColor: [0.0; 4],
            MinLOD: 0.0,
            MaxLOD: D3D11_FLOAT32_MAX,
        };
        let mut sampler = None;
        unsafe { device.CreateSamplerState(&sampler_desc, Some(&mut sampler))? };
        Ok(Self {
            pass_vertex,
            pass_fragment,
            composite_vertex,
            composite_fragment,
            params_buffer: create_constant_buffer::<BackdropBlurParams>(device)?,
            sampler,
            replace_blend_state: create_opaque_blend_state(device)?,
            over_blend_state: create_blend_state_for_path_rasterization(device)?,
        })
    }

    /// Draws a quad over the top-left `extent` texels of `target`, sampling
    /// `source` at t0.
    fn draw_pass(
        &self,
        device_context: &ID3D11DeviceContext,
        target: &BackdropTexture,
        extent: (u32, u32),
        source: &BackdropTexture,
        params: &BackdropBlurParams,
        blend_state: &ID3D11BlendState,
    ) -> Result<()> {
        update_buffer(
            device_context,
            self.params_buffer
                .as_ref()
                .context("backdrop blur params buffer missing")?,
            slice::from_ref(params),
        )?;
        let viewport = D3D11_VIEWPORT {
            TopLeftX: 0.0,
            TopLeftY: 0.0,
            Width: extent.0 as f32,
            Height: extent.1 as f32,
            MinDepth: 0.0,
            MaxDepth: 1.0,
        };
        unsafe {
            // A texture bound for sampling is silently unbound when it becomes
            // a render target, and the reverse, so the previous pass's source
            // is released before this pass's target is bound.
            device_context.PSSetShaderResources(0, Some(&[None]));
            device_context.OMSetRenderTargets(Some(slice::from_ref(&target.render_target_view)), None);
            device_context.PSSetShaderResources(0, Some(slice::from_ref(&source.view)));
            device_context.RSSetViewports(Some(slice::from_ref(&viewport)));
            device_context.OMSetBlendState(blend_state, None, 0xFFFFFFFF);
            device_context.Draw(4, 0);
        }
        Ok(())
    }
}

#[derive(Default)]
pub(super) struct BackdropBlurs {
    pipeline: Option<BackdropBlurPipeline>,
    /// Least recently used first.
    textures: Vec<BackdropTextures>,
    idle_frames: u32,
}

impl BackdropBlurs {
    /// Counts a drawn frame against the scratch textures, releasing them once
    /// enough frames have gone by without a blur. The pipeline is kept: it
    /// holds no window-sized memory.
    pub(super) fn note_frame(&mut self, frame_had_blurs: bool) {
        if frame_had_blurs {
            self.idle_frames = 0;
            return;
        }
        if self.textures.is_empty() {
            return;
        }
        self.idle_frames = self.idle_frames.saturating_add(1);
        if self.idle_frames >= BACKDROP_BLUR_IDLE_FRAMES {
            self.textures.clear();
        }
    }

    pub(super) fn release_textures(&mut self) {
        self.textures.clear();
    }

    /// The pipeline and a scratch set for a `width` × `height` region,
    /// created on demand; the set becomes the most recently used.
    fn prepare(
        &mut self,
        device: &ID3D11Device,
        width: u32,
        height: u32,
        downsample: u32,
        needs_plane_copy: bool,
    ) -> Result<(&BackdropBlurPipeline, &BackdropTextures)> {
        if self.pipeline.is_none() {
            self.pipeline = Some(BackdropBlurPipeline::new(device)?);
        }
        let width = width.next_multiple_of(BACKDROP_TEXTURE_GRANULARITY);
        let height = height.next_multiple_of(BACKDROP_TEXTURE_GRANULARITY);
        match self.textures.iter().position(|textures| {
            textures.width == width && textures.height == height && textures.downsample == downsample
        }) {
            Some(index) => {
                let textures = self.textures.remove(index);
                self.textures.push(textures);
            }
            None => {
                let textures = BackdropTextures::new(device, width, height, downsample)?;
                if self.textures.len() >= MAX_BACKDROP_TEXTURES {
                    self.textures.remove(0);
                }
                self.textures.push(textures);
            }
        }
        let textures = self
            .textures
            .last_mut()
            .context("backdrop blur textures missing")?;
        if needs_plane_copy && textures.plane_copy.is_none() {
            textures.plane_copy = Some(BackdropTexture::new(device, width, height, false)?);
        }
        let pipeline = self
            .pipeline
            .as_ref()
            .context("backdrop blur pipeline missing")?;
        let textures = self
            .textures
            .last()
            .context("backdrop blur textures missing")?;
        Ok((pipeline, textures))
    }
}

impl DirectXRenderer {
    /// Runs `blur` between two batches: copies the region it samples out of
    /// the plane being drawn, blurs the copy, and paints the result back
    /// inside the blur's rounded rect. On the overlay plane the region copied
    /// is the base plane with the overlay drawn so far composited over it,
    /// since the overlay's own target holds only what lies above the native
    /// surfaces. A blur that is entirely clipped draws nothing.
    pub(super) fn draw_backdrop_blur(&mut self, blur: &BackdropBlur) -> Result<()> {
        let target_bounds = Bounds::new(
            point(ScaledPixels(0.), ScaledPixels(0.)),
            size(
                ScaledPixels(self.width as f32),
                ScaledPixels(self.height as f32),
            ),
        );
        let visible = blur
            .bounds
            .intersect(&blur.content_mask.bounds)
            .intersect(&target_bounds);
        // Written so that NaN bounds are skipped too.
        if !(visible.size.width.0 > 0. && visible.size.height.0 > 0.) {
            return Ok(());
        }
        // `clamp` passes NaN through, and a NaN sigma would pad the region
        // out to the whole target.
        let sigma = if blur.blur_radius.0.is_nan() {
            0.5
        } else {
            blur.blur_radius.0.clamp(0.5, MAX_BACKDROP_BLUR_SIGMA)
        };
        // Only what lies under the surface is copied, and the blur clamps at
        // the copy's edges. Sampling the surroundings too would carry
        // whatever borders the surface (a line of text, a lighter panel) in
        // across its edge as a pale band just inside the outline.
        let left = visible.origin.x.0.floor().max(0.) as u32;
        let top = visible.origin.y.0.floor().max(0.) as u32;
        let right = visible.right().0.ceil().min(self.width as f32) as u32;
        let bottom = visible.bottom().0.ceil().min(self.height as f32) as u32;
        if right <= left || bottom <= top {
            return Ok(());
        }
        let width = right - left;
        let height = bottom - top;
        // A wide gaussian has nothing left at the sampling rate of a
        // downsampled texture, so wide blurs keep every Nth texel and the
        // composite upsamples them bilinearly; narrow ones stay full size.
        let downsample = ((sigma / 8.) as u32).clamp(1, 4);
        let blurred_width = width.div_ceil(downsample);
        let blurred_height = height.div_ceil(downsample);
        let on_overlay = self.active_plane == ScenePlane::Overlay;

        let devices = self.devices.as_ref().context("devices missing")?;
        let resources = self.resources.as_ref().context("resources missing")?;
        let (plane_render_target, plane_render_target_view) = match self.active_plane {
            ScenePlane::Base => (&resources.render_target, &resources.render_target_view),
            ScenePlane::Overlay => {
                let overlay = self
                    .overlay_resources
                    .as_ref()
                    .context("overlay resources missing")?;
                (&overlay.render_target, &overlay.render_target_view)
            }
        };
        let plane_render_target = plane_render_target
            .as_ref()
            .context("render target missing")?;
        let base_render_target = resources
            .render_target
            .as_ref()
            .context("render target missing")?;
        let (pipeline, textures) = self.backdrop_blurs.prepare(
            &devices.device,
            width,
            height,
            downsample,
            on_overlay,
        )?;
        let device_context = &devices.device_context;

        let mut params = BackdropBlurParams {
            bounds: bounds_to_array(&blur.bounds),
            corner_radii: [
                blur.corner_radii.top_left.0,
                blur.corner_radii.top_right.0,
                blur.corner_radii.bottom_right.0,
                blur.corner_radii.bottom_left.0,
            ],
            content_mask: bounds_to_array(&blur.content_mask.bounds),
            region_origin: [left as f32, top as f32],
            region_size: [width as f32, height as f32],
            source_size: [0.; 2],
            source_extent: [0.; 2],
            direction: [0.; 2],
            scale: [1.; 2],
            saturation: blur.saturation,
            opacity: blur.opacity,
            center_weight: 1.,
            tap_pair_count: 0,
            taps: [[0.; 4]; BACKDROP_TAP_VECTORS],
        };
        let (center_weight, tap_pair_count) = gaussian_tap_pairs(sigma, &mut params.taps);

        let region = D3D11_BOX {
            left,
            top,
            front: 0,
            right,
            bottom,
            back: 1,
        };
        unsafe {
            device_context.OMSetRenderTargets(None, None);
            device_context.CopySubresourceRegion(
                &textures.source.texture,
                0,
                0,
                0,
                0,
                base_render_target,
                0,
                Some(&region),
            );
        }
        let original = if on_overlay {
            let plane_copy = textures
                .plane_copy
                .as_ref()
                .context("backdrop blur plane copy missing")?;
            unsafe {
                device_context.CopySubresourceRegion(
                    &plane_copy.texture,
                    0,
                    0,
                    0,
                    0,
                    plane_render_target,
                    0,
                    Some(&region),
                );
            }
            plane_copy
        } else {
            &textures.source
        };

        self.pipelines
            .opaque_quad_pipeline
            .disable_depth(device_context);
        unsafe {
            device_context.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
            device_context.VSSetShader(&pipeline.pass_vertex, None);
            device_context.PSSetShader(&pipeline.pass_fragment, None);
            device_context.PSSetSamplers(1, Some(slice::from_ref(&pipeline.sampler)));
            device_context.VSSetConstantBuffers(2, Some(slice::from_ref(&pipeline.params_buffer)));
            device_context.PSSetConstantBuffers(2, Some(slice::from_ref(&pipeline.params_buffer)));
        }

        let result = (|| {
            if on_overlay {
                // With no taps the pass shader reads each texel exactly once.
                params.source_size = [original.width as f32, original.height as f32];
                params.source_extent = [width as f32, height as f32];
                pipeline.draw_pass(
                    device_context,
                    &textures.source,
                    (width, height),
                    original,
                    &params,
                    &pipeline.over_blend_state,
                )?;
            }

            params.center_weight = center_weight;
            params.tap_pair_count = tap_pair_count;
            params.source_size = [textures.source.width as f32, textures.source.height as f32];
            params.source_extent = [width as f32, height as f32];
            params.direction = [1., 0.];
            params.scale = [downsample as f32, 1.];
            pipeline.draw_pass(
                device_context,
                &textures.horizontal,
                (blurred_width, height),
                &textures.source,
                &params,
                &pipeline.replace_blend_state,
            )?;

            params.source_size = [
                textures.horizontal.width as f32,
                textures.horizontal.height as f32,
            ];
            params.source_extent = [blurred_width as f32, height as f32];
            params.direction = [0., 1.];
            params.scale = [1., downsample as f32];
            pipeline.draw_pass(
                device_context,
                &textures.blurred,
                (blurred_width, blurred_height),
                &textures.horizontal,
                &params,
                &pipeline.replace_blend_state,
            )?;

            params.source_size = [textures.blurred.width as f32, textures.blurred.height as f32];
            params.source_extent = [blurred_width as f32, blurred_height as f32];
            params.scale = [downsample as f32; 2];
            update_buffer(
                device_context,
                pipeline
                    .params_buffer
                    .as_ref()
                    .context("backdrop blur params buffer missing")?,
                slice::from_ref(&params),
            )?;
            unsafe {
                device_context.PSSetShaderResources(0, Some(&[None]));
                device_context.OMSetRenderTargets(
                    Some(slice::from_ref(plane_render_target_view)),
                    resources.depth_stencil_view.as_ref(),
                );
                device_context.PSSetShaderResources(0, Some(slice::from_ref(&original.view)));
                device_context.PSSetShaderResources(4, Some(slice::from_ref(&textures.blurred.view)));
                device_context.RSSetViewports(Some(slice::from_ref(&resources.viewport)));
                device_context.VSSetShader(&pipeline.composite_vertex, None);
                device_context.PSSetShader(&pipeline.composite_fragment, None);
                device_context.OMSetBlendState(&pipeline.replace_blend_state, None, 0xFFFFFFFF);
                device_context.Draw(4, 0);
            }
            anyhow::Ok(())
        })();

        // Whatever happened, the plane's target is bound again for the batches
        // that follow, and no scratch texture stays bound: a bound texture
        // would survive its release, and the next blur may render into it.
        unsafe {
            device_context.PSSetShaderResources(0, Some(&[None]));
            device_context.PSSetShaderResources(4, Some(&[None]));
            device_context.OMSetRenderTargets(
                Some(slice::from_ref(plane_render_target_view)),
                resources.depth_stencil_view.as_ref(),
            );
            device_context.RSSetViewports(Some(slice::from_ref(&resources.viewport)));
        }
        self.pipelines
            .opaque_quad_pipeline
            .enable_depth_test(device_context);
        result
    }
}

#[cfg(test)]
mod tests {
    use crate::{DirectXDevices, DirectXRenderer};
    use gpui::{
        BackdropBlur, Background, Bounds, ContentMask, Corners, DevicePixels, Edges, Hsla, Point,
        Quad, ScaledPixels, Scene, Size,
    };
    use gpui_util::ResultExt as _;

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

    fn headless_renderer() -> Option<DirectXRenderer> {
        let devices = DirectXDevices::new().log_err()?;
        DirectXRenderer::new_windowless(&devices, true).log_err()
    }

    /// Opaque quads beneath a blur take the depth prepass, so they are in the
    /// framebuffer before the blur copies it and hide what they cover; an
    /// opaque quad above the blur paints after it and is never frosted into
    /// the surface beneath.
    #[test]
    fn a_frosted_frame_keeps_paint_order_around_the_blur() {
        let Some(mut renderer) = headless_renderer() else {
            eprintln!("SKIPPED: no D3D11 adapter or headless renderer");
            return;
        };
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

        renderer
            .resize(gpui::size(DevicePixels(200), DevicePixels(100)))
            .expect("resizing the headless target succeeds");
        let image = renderer
            .render_to_image(&scene, [0.0, 0.0, 0.0, 1.0])
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

    const TARGET: Size<DevicePixels> = Size {
        width: DevicePixels(3456),
        height: DevicePixels(2168),
    };

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
        let composer = rect(1000., 1800., 1400., 260.);
        if frost_composer {
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
            scene.insert_primitive(quad(composer, gpui::hsla(0., 0., 0.1, 0.93)));
        }
        scene.finish();
        scene
    }

    /// GPU milliseconds `scene` takes to render, from a pair of timestamp
    /// queries around the frame.
    fn gpu_milliseconds(renderer: &mut DirectXRenderer, scene: &Scene) -> f64 {
        use windows::Win32::Foundation::S_OK;
        use windows::Win32::Graphics::Direct3D11::{
            D3D11_QUERY_DATA_TIMESTAMP_DISJOINT, D3D11_QUERY_DESC, D3D11_QUERY_TIMESTAMP,
            D3D11_QUERY_TIMESTAMP_DISJOINT, ID3D11DeviceContext, ID3D11Query,
        };
        use windows::core::Interface as _;

        // `GetData` maps S_FALSE, "not ready yet", to `Ok`, so the HRESULT is
        // read through the vtable instead.
        fn wait_for<T: Default>(context: &ID3D11DeviceContext, query: &ID3D11Query) -> T {
            let mut data = T::default();
            loop {
                let result = unsafe {
                    (windows::core::Interface::vtable(context).GetData)(
                        context.as_raw(),
                        query.as_raw(),
                        &mut data as *mut T as *mut _,
                        std::mem::size_of::<T>() as u32,
                        0,
                    )
                };
                if result == S_OK {
                    return data;
                }
                assert!(result.is_ok(), "GetData failed: {result:?}");
                std::hint::spin_loop();
            }
        }

        let devices = renderer.devices.as_ref().expect("devices");
        let device = devices.device.clone();
        let context = devices.device_context.clone();
        let create = |kind| {
            let mut query = None;
            unsafe {
                device
                    .CreateQuery(
                        &D3D11_QUERY_DESC {
                            Query: kind,
                            MiscFlags: 0,
                        },
                        Some(&mut query),
                    )
                    .expect("creating a query");
            }
            query.expect("query created")
        };
        let disjoint = create(D3D11_QUERY_TIMESTAMP_DISJOINT);
        let start = create(D3D11_QUERY_TIMESTAMP);
        let end = create(D3D11_QUERY_TIMESTAMP);
        unsafe {
            context.Begin(&disjoint);
            context.End(&start);
        }
        renderer.render(scene, [0.0, 0.0, 0.0, 1.0]).expect("frame renders");
        unsafe {
            context.End(&end);
            context.End(&disjoint);
        }
        let start_ticks: u64 = wait_for(&context, &start);
        let end_ticks: u64 = wait_for(&context, &end);
        let disjoint: D3D11_QUERY_DATA_TIMESTAMP_DISJOINT = wait_for(&context, &disjoint);
        if disjoint.Disjoint.as_bool() {
            return f64::NAN;
        }
        end_ticks.saturating_sub(start_ticks) as f64 * 1000. / disjoint.Frequency as f64
    }

    fn median(mut values: Vec<f64>) -> f64 {
        values.retain(|value| value.is_finite());
        values.sort_by(f64::total_cmp);
        values.get(values.len() / 2).copied().unwrap_or(f64::NAN)
    }

    /// Run with `--ignored --nocapture`; prints what a frame of overlapping
    /// opaque surfaces costs with a frosted composer over them, against the
    /// same frame unfrosted.
    #[test]
    #[ignore = "measurement, prints GPU time of a frosted frame over stacked surfaces"]
    fn measure_backdrop_blur_gpu_time_over_stacked_surfaces() {
        let Some(mut renderer) = headless_renderer() else {
            eprintln!("SKIPPED: no D3D11 adapter or headless renderer");
            return;
        };
        renderer
            .resize(TARGET)
            .expect("resizing the headless target succeeds");
        for layers in [4, 12] {
            let plain = stacked_scene(layers, false);
            let frosted = stacked_scene(layers, true);
            for _ in 0..10 {
                gpu_milliseconds(&mut renderer, &plain);
                gpu_milliseconds(&mut renderer, &frosted);
            }
            let mut plain_times = Vec::new();
            let mut frosted_times = Vec::new();
            for _ in 0..200 {
                plain_times.push(gpu_milliseconds(&mut renderer, &plain));
                frosted_times.push(gpu_milliseconds(&mut renderer, &frosted));
            }
            let plain_frame = median(plain_times);
            let frosted_frame = median(frosted_times);
            println!(
                "[backdrop-gpu] {layers} stacked surfaces + frosted composer: {frosted_frame:.3} ms, {:.3} ms over the plain frame ({plain_frame:.3} ms); {} of {} quads prepassed; median of 200 alternating pairs",
                frosted_frame - plain_frame,
                frosted.opaque_quad_indices.len(),
                frosted.quads.len(),
            );
        }
    }
}
