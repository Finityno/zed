// todo("windows"): remove
#![cfg_attr(windows, allow(dead_code))]

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{
    AtlasTextureId, AtlasTile, Background, Bounds, ContentMask, Corners, Edges, Hsla, Pixels,
    Point, Radians, ScaledPixels, Size, bounds_tree::BoundsTree, point, util::CapacityShrink,
};
use std::{
    fmt::Debug,
    iter::Peekable,
    ops::{Add, Range, Sub},
    slice,
    sync::OnceLock,
};

/// How glyph-drop reporting is configured by `GPUI_DEBUG_GLYPH_DROPS`.
#[derive(Clone, Copy, PartialEq, Eq)]
enum GlyphDropReporting {
    Off,
    /// Only glyphs that missed their content mask by a hair.
    NearMisses,
    /// Every dropped glyph, including ones legitimately scrolled offscreen.
    All,
}

/// Whether to report glyph sprites dropped by the content-mask cull in `insert_primitive`.
///
/// Off by default: the cull is a hot path and legitimately discards offscreen glyphs by the
/// thousand while scrolling. `GPUI_DEBUG_GLYPH_DROPS=1` reports only near misses, which is
/// what you want when diagnosing characters going missing from the middle of a word;
/// `GPUI_DEBUG_GLYPH_DROPS=all` reports every drop and will be extremely noisy.
fn glyph_drop_reporting() -> GlyphDropReporting {
    static MODE: OnceLock<GlyphDropReporting> = OnceLock::new();
    *MODE.get_or_init(
        || match std::env::var("GPUI_DEBUG_GLYPH_DROPS").as_deref() {
            Ok("1" | "true") => GlyphDropReporting::NearMisses,
            Ok("all") => GlyphDropReporting::All,
            _ => GlyphDropReporting::Off,
        },
    )
}

/// A glyph that missed its mask by less than this many scaled pixels is very unlikely to have
/// been scrolled away and very likely to have been rounded away.
const GLYPH_NEAR_MISS_PIXELS: f32 = 2.0;

/// Scaled pixels by which `bounds` misses `mask` on each axis; zero where they touch or
/// overlap.
///
/// The magnitude separates the two reasons a glyph is culled, which a log otherwise cannot
/// tell apart: content scrolled out of a viewport misses by tens or thousands of pixels, while
/// a glyph rounded off a clip edge misses by well under one.
///
/// The AXIS then separates the two reasons a glyph can miss by a hair, which matters just as
/// much. A vertical near miss is the ordinary case — the line above a scroll viewport sits a
/// few pixels above its mask and is supposed to be invisible, and a transcript produces
/// hundreds of those per second. A horizontal near miss is not ordinary: text is not scrolled
/// sideways, so a glyph a fraction of a pixel past the left or right edge means a clip edge
/// landed inside a label, which is the reported "character missing from the middle of a word".
/// Reporting only the larger of the two buries the interesting case under the boring one.
fn mask_miss_distances(bounds: &Bounds<ScaledPixels>, mask: &Bounds<ScaledPixels>) -> (f32, f32) {
    let horizontal = (mask.origin.x.0 - (bounds.origin.x.0 + bounds.size.width.0))
        .max(bounds.origin.x.0 - (mask.origin.x.0 + mask.size.width.0))
        .max(0.0);
    let vertical = (mask.origin.y.0 - (bounds.origin.y.0 + bounds.size.height.0))
        .max(bounds.origin.y.0 - (mask.origin.y.0 + mask.size.height.0))
        .max(0.0);
    (horizontal, vertical)
}

#[allow(non_camel_case_types, unused)]
#[expect(missing_docs)]
pub type PathVertex_ScaledPixels = PathVertex<ScaledPixels>;

#[expect(missing_docs)]
pub type DrawOrder = u32;

/// A boolean stored as a `u32` so that GPU-facing structs contain no
/// compiler-inserted padding bytes, which would be undefined behavior to
/// reinterpret as `&[u8]` when writing instance buffers. Guaranteed to be
/// `0` or `1` by construction; shaders read it as a `u32`/`uint`.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
#[repr(transparent)]
pub struct PaddedBool32(u32);

impl From<bool> for PaddedBool32 {
    fn from(value: bool) -> Self {
        PaddedBool32(value as u32)
    }
}

#[derive(Default)]
#[expect(missing_docs)]
pub struct Scene {
    pub(crate) paint_operations: Vec<PaintOperation>,
    primitive_bounds: BoundsTree<ScaledPixels>,
    layer_stack: Vec<DrawOrder>,
    pub shadows: Vec<Shadow>,
    pub quads: Vec<Quad>,
    pub paths: Vec<Path<ScaledPixels>>,
    pub underlines: Vec<Underline>,
    pub monochrome_sprites: Vec<MonochromeSprite>,
    pub subpixel_sprites: Vec<SubpixelSprite>,
    pub polychrome_sprites: Vec<PolychromeSprite>,
    pub surfaces: Vec<PaintSurface>,
    pub blended_quad_indices: Vec<u32>,
    pub opaque_quad_indices: Vec<u32>,
    /// One tracker per vector above, in the order `clear` destructures them.
    shrink: [CapacityShrink; 12],
}

#[expect(missing_docs)]
impl Scene {
    pub fn clear(&mut self) {
        let [
            paint_operations,
            layer_stack,
            paths,
            shadows,
            quads,
            underlines,
            monochrome_sprites,
            subpixel_sprites,
            polychrome_sprites,
            surfaces,
            blended_quad_indices,
            opaque_quad_indices,
        ] = &mut self.shrink;
        paint_operations.clear_vec(&mut self.paint_operations);
        self.primitive_bounds.clear();
        layer_stack.clear_vec(&mut self.layer_stack);
        paths.clear_vec(&mut self.paths);
        shadows.clear_vec(&mut self.shadows);
        quads.clear_vec(&mut self.quads);
        underlines.clear_vec(&mut self.underlines);
        monochrome_sprites.clear_vec(&mut self.monochrome_sprites);
        subpixel_sprites.clear_vec(&mut self.subpixel_sprites);
        polychrome_sprites.clear_vec(&mut self.polychrome_sprites);
        surfaces.clear_vec(&mut self.surfaces);
        blended_quad_indices.clear_vec(&mut self.blended_quad_indices);
        opaque_quad_indices.clear_vec(&mut self.opaque_quad_indices);
    }

    pub fn len(&self) -> usize {
        self.paint_operations.len()
    }

    /// Returns whether the scene contains no drawable primitives.
    ///
    /// A scene may have paint operations that only open and close empty layers,
    /// so `len() == 0` is not equivalent to having no visible/input-relevant
    /// overlay content.
    pub fn is_empty(&self) -> bool {
        self.shadows.is_empty()
            && self.quads.is_empty()
            && self.paths.is_empty()
            && self.underlines.is_empty()
            && self.monochrome_sprites.is_empty()
            && self.subpixel_sprites.is_empty()
            && self.polychrome_sprites.is_empty()
            && self.surfaces.is_empty()
    }

    pub fn push_layer(&mut self, bounds: Bounds<ScaledPixels>) {
        let order = self.primitive_bounds.insert(bounds);
        self.layer_stack.push(order);
        self.paint_operations
            .push(PaintOperation::StartLayer(bounds));
    }

    pub fn pop_layer(&mut self) {
        self.layer_stack.pop();
        self.paint_operations.push(PaintOperation::EndLayer);
    }

    pub fn insert_primitive(&mut self, primitive: impl Into<Primitive>) {
        let mut primitive = primitive.into();
        let clipped_bounds = primitive
            .bounds()
            .intersect(&primitive.content_mask().bounds);

        if clipped_bounds.is_empty() {
            // The second place a single glyph can vanish with nothing logged (the first is the
            // empty-raster-bounds path in `Window::paint_glyph`). Shaping has already run, so
            // the advance survives and the character is simply missing from the middle of a
            // word. `paint_line` pre-culls with the font's MAX bounding box at the pen
            // position, whereas this culls the glyph's actual quad, so a glyph can pass the
            // coarse check and still be dropped here — scattered singles, neighbours intact.
            // Kind first: this branch runs for every offscreen glyph while scrolling, and the
            // discriminant check is cheaper than the flag's atomic load.
            if matches!(
                primitive,
                Primitive::MonochromeSprite(_) | Primitive::SubpixelSprite(_)
            ) {
                let mode = glyph_drop_reporting();
                if mode != GlyphDropReporting::Off {
                    let bounds = primitive.bounds();
                    let mask = &primitive.content_mask().bounds;
                    let (horizontal, vertical) = mask_miss_distances(bounds, mask);
                    // A horizontal miss means a clip edge cut into a line of text, which is
                    // the suspicious case at any distance under a pixel. A purely vertical
                    // miss is the line above a scroll viewport and is expected.
                    let suspicious =
                        horizontal > 0.0 && horizontal < GLYPH_NEAR_MISS_PIXELS && vertical == 0.0;
                    if mode == GlyphDropReporting::All || suspicious {
                        let axis = if suspicious { "HORIZONTAL" } else { "vertical" };
                        log::warn!(
                            "dropped a glyph sprite ({axis}) {horizontal:.3}px x / \
                             {vertical:.3}px y outside its content mask: \
                             bounds {bounds:?} vs mask {mask:?}",
                        );
                    }
                }
            }
            return;
        }

        let order = self
            .layer_stack
            .last()
            .copied()
            .unwrap_or_else(|| self.primitive_bounds.insert(clipped_bounds));
        match &mut primitive {
            Primitive::Shadow(shadow) => {
                shadow.order = order;
                self.shadows.push(*shadow);
            }
            Primitive::Quad(quad) => {
                quad.order = order;
                self.quads.push(*quad);
            }
            Primitive::Path(path) => {
                path.order = order;
                path.id = PathId(self.paths.len());
                self.paths.push(path.clone());
            }
            Primitive::Underline(underline) => {
                underline.order = order;
                self.underlines.push(*underline);
            }
            Primitive::MonochromeSprite(sprite) => {
                sprite.order = order;
                self.monochrome_sprites.push(*sprite);
            }
            Primitive::SubpixelSprite(sprite) => {
                sprite.order = order;
                self.subpixel_sprites.push(*sprite);
            }
            Primitive::PolychromeSprite(sprite) => {
                sprite.order = order;
                self.polychrome_sprites.push(*sprite);
            }
            Primitive::Surface(surface) => {
                surface.order = order;
                self.surfaces.push(surface.clone());
            }
        }
        self.paint_operations
            .push(PaintOperation::Primitive(primitive));
    }

    pub fn replay(&mut self, range: Range<usize>, prev_scene: &Scene) {
        for operation in &prev_scene.paint_operations[range] {
            match operation {
                PaintOperation::Primitive(primitive) => self.insert_primitive(primitive.clone()),
                PaintOperation::StartLayer(bounds) => self.push_layer(*bounds),
                PaintOperation::EndLayer => self.pop_layer(),
            }
        }
    }

    pub fn finish(&mut self) {
        self.shadows.sort_by_key(|shadow| shadow.order);
        self.quads.sort_by_key(|quad| quad.order);
        self.paths.sort_by_key(|path| path.order);
        self.underlines.sort_by_key(|underline| underline.order);
        self.monochrome_sprites
            .sort_by_key(|sprite| (sprite.order, sprite.tile.tile_id));
        self.subpixel_sprites
            .sort_by_key(|sprite| (sprite.order, sprite.tile.tile_id));
        self.polychrome_sprites
            .sort_by_key(|sprite| (sprite.order, sprite.tile.tile_id));
        self.surfaces.sort_by_key(|surface| surface.order);
        self.partition_quads();
    }

    fn partition_quads(&mut self) {
        self.blended_quad_indices.clear();
        self.opaque_quad_indices.clear();
        let partitioning_enabled =
            opaque_quad_partitioning_enabled() && self.quads.len() <= MAX_DEPTH_PARTITIONED_QUADS;
        for (quad_id, quad) in self.quads.iter().enumerate() {
            let has_opaque_core = partitioning_enabled && quad.has_opaque_core();
            if has_opaque_core {
                self.opaque_quad_indices.push(quad_id as u32);
            }
            if !has_opaque_core || quad.has_rounded_corners() {
                self.blended_quad_indices.push(quad_id as u32);
            }
        }
        // Opaque quads are drawn front-to-back in the depth-writing pass.
        self.opaque_quad_indices.reverse();
    }

    /// Iterates the frame's batches in draw order. Only valid after `finish`.
    #[cfg_attr(
        all(
            any(target_os = "linux", target_os = "freebsd"),
            not(any(feature = "x11", feature = "wayland"))
        ),
        allow(dead_code)
    )]
    pub fn batches(&self) -> impl Iterator<Item = PrimitiveBatch> + '_ {
        BatchIterator {
            shadows_start: 0,
            shadows_iter: self.shadows.iter().peekable(),
            quads_start: 0,
            quads_iter: self.quads.iter().peekable(),
            blended_quad_indices: &self.blended_quad_indices,
            blended_quad_indices_start: 0,
            paths_start: 0,
            paths_iter: self.paths.iter().peekable(),
            underlines_start: 0,
            underlines_iter: self.underlines.iter().peekable(),
            monochrome_sprites_start: 0,
            monochrome_sprites_iter: self.monochrome_sprites.iter().peekable(),
            subpixel_sprites_start: 0,
            subpixel_sprites_iter: self.subpixel_sprites.iter().peekable(),
            polychrome_sprites_start: 0,
            polychrome_sprites_iter: self.polychrome_sprites.iter().peekable(),
            surfaces_start: 0,
            surfaces_iter: self.surfaces.iter().peekable(),
        }
    }
}

/// The most quads a frame can partition into the depth prepass: the depth
/// mapping gives each of them its own 16-bit depth value, so a 16-bit depth
/// attachment (half the memory of a 32-bit one) loses nothing. A frame with
/// more quads than this skips partitioning and paints everything blended in
/// painter's order, which needs no distinct depths at all.
pub const MAX_DEPTH_PARTITIONED_QUADS: usize = 65534;

/// Maps a quad's index in [`Scene::quads`] to depth, with greater values closer.
///
/// Zero is reserved for the cleared depth buffer. `quad_depth(n)` also
/// represents the cursor after the first `n` quads: with a strict greater-than
/// test, it is above quads before the cursor and ties with the quad after it.
/// Depth-based renderers must use this mapping in both CPU and shader code.
///
/// Steps are 1/65535 so that every partitioned quad (see
/// [`MAX_DEPTH_PARTITIONED_QUADS`]) lands on its own value of a 16-bit
/// unorm depth attachment as well as a 32-bit float one.
pub fn quad_depth(quad_id: u32) -> f32 {
    ((quad_id + 1) as f32 * (1.0 / 65535.0)).min(1.0)
}

static OPAQUE_QUAD_PARTITIONING_DISABLED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Routes every quad through the blended back-to-front pass, leaving the
/// opaque front-to-back depth prepass empty. Renderers whose instance
/// transport cannot express the quad index indirection (the WebGL2 texture
/// path) call this at startup; with the depth buffer cleared to zero and a
/// strict greater-than test, the blended-only walk paints in painter's order
/// and renders identically to the partitioned scheme.
pub fn disable_opaque_quad_partitioning() {
    OPAQUE_QUAD_PARTITIONING_DISABLED.store(true, std::sync::atomic::Ordering::Relaxed);
}

fn opaque_quad_partitioning_enabled() -> bool {
    !OPAQUE_QUAD_PARTITIONING_DISABLED.load(std::sync::atomic::Ordering::Relaxed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::{MIN_RETAINED_CAPACITY, SHRINK_AFTER_FRAMES};

    /// `quad_depth` is mirrored in every backend's shader by hand, and upstream
    /// spaces it differently. A sync that takes an upstream shader while
    /// keeping this file compiles clean and gives viewport-collapsed batches
    /// depths hundreds of times larger than the quads: text drawn before an
    /// opaque quad then passes the GREATER test over it.
    #[test]
    fn quad_depth_step_matches_every_shader() {
        let step = format!("1.0 / {}.0", MAX_DEPTH_PARTITIONED_QUADS + 1);
        assert_eq!(step, "1.0 / 65535.0");
        for (name, source) in [
            ("metal", include_str!("../../gpui_apple/src/shaders.metal")),
            ("hlsl", include_str!("../../gpui_windows/src/shaders.hlsl")),
            ("wgsl", include_str!("../../gpui_wgpu/src/shaders.wgsl")),
        ] {
            // The definition, not a forward declaration: any occurrence whose
            // next few hundred characters carry the step.
            let defined_with_step = source.match_indices("quad_depth(").any(|(index, _)| {
                source[index..(index + 400).min(source.len())].contains(&step)
            });
            assert!(
                defined_with_step,
                "the {name} shader's quad_depth must step by {step}, as scene.rs does"
            );
        }
    }

    fn paint_frame(scene: &mut Scene, quads: usize) {
        let bounds = Bounds {
            origin: Point::default(),
            size: Size {
                width: ScaledPixels::from(100.),
                height: ScaledPixels::from(100.),
            },
        };
        scene.push_layer(bounds);
        for _ in 0..quads {
            scene.insert_primitive(Quad {
                bounds,
                content_mask: ContentMask { bounds },
                ..Default::default()
            });
        }
        scene.pop_layer();
        scene.finish();
    }

    #[test]
    fn one_huge_frame_followed_by_small_ones_releases_its_capacity() {
        let mut scene = Scene::default();
        paint_frame(&mut scene, 10_000);
        scene.clear();
        let high_water = scene.quads.capacity();
        let operations_high_water = scene.paint_operations.capacity();
        assert!(high_water >= 10_000);

        for _ in 0..SHRINK_AFTER_FRAMES - 1 {
            paint_frame(&mut scene, 5);
            scene.clear();
            assert_eq!(scene.quads.capacity(), high_water);
            assert_eq!(scene.paint_operations.capacity(), operations_high_water);
        }
        paint_frame(&mut scene, 5);
        scene.clear();
        assert!(scene.quads.capacity() <= MIN_RETAINED_CAPACITY);
        assert!(scene.paint_operations.capacity() <= MIN_RETAINED_CAPACITY);
        assert!(scene.opaque_quad_indices.capacity() <= MIN_RETAINED_CAPACITY);
    }

    #[test]
    fn a_workload_that_alternates_big_and_small_frames_never_shrinks() {
        let mut scene = Scene::default();
        paint_frame(&mut scene, 10_000);
        scene.clear();
        let high_water = scene.quads.capacity();

        for frame in 0..SHRINK_AFTER_FRAMES * 5 {
            // Over half the high-water capacity, so every recurrence resets
            // the low-use count.
            let quads = if frame % 30 == 0 { 10_000 } else { 5 };
            paint_frame(&mut scene, quads);
            scene.clear();
            assert_eq!(scene.quads.capacity(), high_water);
        }
    }

    #[test]
    fn empty_layers_do_not_make_a_scene_drawable() {
        let mut scene = Scene::default();
        let bounds = Bounds {
            origin: Point::default(),
            size: Size {
                width: ScaledPixels::from(100.),
                height: ScaledPixels::from(100.),
            },
        };

        scene.push_layer(bounds);
        scene.pop_layer();

        assert_ne!(scene.len(), 0);
        assert!(scene.is_empty());
    }

    #[test]
    fn drawable_primitives_make_a_scene_non_empty() {
        let mut scene = Scene::default();
        let bounds = Bounds {
            origin: Point::default(),
            size: Size {
                width: ScaledPixels::from(100.),
                height: ScaledPixels::from(100.),
            },
        };

        scene.insert_primitive(Quad {
            bounds,
            content_mask: ContentMask { bounds },
            ..Default::default()
        });

        assert!(!scene.is_empty());
    }

    /// A 16-bit unorm depth attachment stores `round(z * 65535)`; every quad
    /// the prepass can partition must land on its own value there, and the
    /// cleared buffer's zero must stay reserved.
    #[test]
    fn quad_depths_stay_distinct_in_a_16_bit_depth_attachment() {
        let mut previous = 0u32;
        for quad_id in 0..=MAX_DEPTH_PARTITIONED_QUADS as u32 {
            let quantized = (quad_depth(quad_id) * 65535.0).round() as u32;
            assert!(
                quantized > previous,
                "quad {quad_id} quantizes to {quantized}, not above {previous}"
            );
            previous = quantized;
        }
        assert_eq!(quad_depth(u32::MAX - 1), 1.0);
    }

    #[test]
    fn frames_with_too_many_quads_for_the_depth_mapping_skip_partitioning() {
        let bounds = Bounds {
            origin: Point::default(),
            size: Size {
                width: ScaledPixels::from(100.),
                height: ScaledPixels::from(100.),
            },
        };
        let opaque_quad = || Quad {
            bounds,
            content_mask: ContentMask { bounds },
            background: Background::from(Hsla::black()),
            ..Default::default()
        };

        let mut scene = Scene::default();
        scene.insert_primitive(opaque_quad());
        scene.finish();
        assert_eq!(scene.opaque_quad_indices.len(), 1);

        let mut scene = Scene::default();
        for _ in 0..=MAX_DEPTH_PARTITIONED_QUADS {
            scene.insert_primitive(opaque_quad());
        }
        scene.finish();
        assert!(scene.opaque_quad_indices.is_empty());
        assert_eq!(scene.blended_quad_indices.len(), MAX_DEPTH_PARTITIONED_QUADS + 1);
    }

    #[test]
    fn replay_preserves_scene_emptiness() {
        let mut source = Scene::default();
        let bounds = Bounds {
            origin: Point::default(),
            size: Size {
                width: ScaledPixels::from(100.),
                height: ScaledPixels::from(100.),
            },
        };
        source.push_layer(bounds);
        source.pop_layer();

        let mut replayed = Scene::default();
        replayed.replay(0..source.len(), &source);

        assert!(replayed.is_empty());
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Default)]
#[cfg_attr(
    all(
        any(target_os = "linux", target_os = "freebsd"),
        not(any(feature = "x11", feature = "wayland"))
    ),
    allow(dead_code)
)]
pub(crate) enum PrimitiveKind {
    Shadow,
    #[default]
    Quad,
    Path,
    Underline,
    MonochromeSprite,
    SubpixelSprite,
    PolychromeSprite,
    Surface,
}

pub(crate) enum PaintOperation {
    Primitive(Primitive),
    StartLayer(Bounds<ScaledPixels>),
    EndLayer,
}

#[derive(Clone)]
#[expect(missing_docs)]
pub enum Primitive {
    Shadow(Shadow),
    Quad(Quad),
    Path(Path<ScaledPixels>),
    Underline(Underline),
    MonochromeSprite(MonochromeSprite),
    SubpixelSprite(SubpixelSprite),
    PolychromeSprite(PolychromeSprite),
    Surface(PaintSurface),
}

#[expect(missing_docs)]
impl Primitive {
    pub fn bounds(&self) -> &Bounds<ScaledPixels> {
        match self {
            Primitive::Shadow(shadow) => &shadow.bounds,
            Primitive::Quad(quad) => &quad.bounds,
            Primitive::Path(path) => &path.bounds,
            Primitive::Underline(underline) => &underline.bounds,
            Primitive::MonochromeSprite(sprite) => &sprite.bounds,
            Primitive::SubpixelSprite(sprite) => &sprite.bounds,
            Primitive::PolychromeSprite(sprite) => &sprite.bounds,
            Primitive::Surface(surface) => &surface.bounds,
        }
    }

    pub fn content_mask(&self) -> &ContentMask<ScaledPixels> {
        match self {
            Primitive::Shadow(shadow) => &shadow.content_mask,
            Primitive::Quad(quad) => &quad.content_mask,
            Primitive::Path(path) => &path.content_mask,
            Primitive::Underline(underline) => &underline.content_mask,
            Primitive::MonochromeSprite(sprite) => &sprite.content_mask,
            Primitive::SubpixelSprite(sprite) => &sprite.content_mask,
            Primitive::PolychromeSprite(sprite) => &sprite.content_mask,
            Primitive::Surface(surface) => &surface.content_mask,
        }
    }
}

#[cfg_attr(
    all(
        any(target_os = "linux", target_os = "freebsd"),
        not(any(feature = "x11", feature = "wayland"))
    ),
    allow(dead_code)
)]
struct BatchIterator<'a> {
    shadows_start: usize,
    shadows_iter: Peekable<slice::Iter<'a, Shadow>>,
    quads_start: usize,
    quads_iter: Peekable<slice::Iter<'a, Quad>>,
    blended_quad_indices: &'a [u32],
    blended_quad_indices_start: usize,
    paths_start: usize,
    paths_iter: Peekable<slice::Iter<'a, Path<ScaledPixels>>>,
    underlines_start: usize,
    underlines_iter: Peekable<slice::Iter<'a, Underline>>,
    monochrome_sprites_start: usize,
    monochrome_sprites_iter: Peekable<slice::Iter<'a, MonochromeSprite>>,
    subpixel_sprites_start: usize,
    subpixel_sprites_iter: Peekable<slice::Iter<'a, SubpixelSprite>>,
    polychrome_sprites_start: usize,
    polychrome_sprites_iter: Peekable<slice::Iter<'a, PolychromeSprite>>,
    surfaces_start: usize,
    surfaces_iter: Peekable<slice::Iter<'a, PaintSurface>>,
}

impl<'a> Iterator for BatchIterator<'a> {
    type Item = PrimitiveBatch;

    fn next(&mut self) -> Option<Self::Item> {
        let mut orders_and_kinds = [
            (
                self.shadows_iter.peek().map(|s| s.order),
                PrimitiveKind::Shadow,
            ),
            (self.quads_iter.peek().map(|q| q.order), PrimitiveKind::Quad),
            (self.paths_iter.peek().map(|q| q.order), PrimitiveKind::Path),
            (
                self.underlines_iter.peek().map(|u| u.order),
                PrimitiveKind::Underline,
            ),
            (
                self.monochrome_sprites_iter.peek().map(|s| s.order),
                PrimitiveKind::MonochromeSprite,
            ),
            (
                self.subpixel_sprites_iter.peek().map(|s| s.order),
                PrimitiveKind::SubpixelSprite,
            ),
            (
                self.polychrome_sprites_iter.peek().map(|s| s.order),
                PrimitiveKind::PolychromeSprite,
            ),
            (
                self.surfaces_iter.peek().map(|s| s.order),
                PrimitiveKind::Surface,
            ),
        ];
        orders_and_kinds.sort_by_key(|(order, kind)| (order.unwrap_or(u32::MAX), *kind));

        let first = orders_and_kinds[0];
        let second = orders_and_kinds[1];
        let (batch_kind, max_order_and_kind) = if first.0.is_some() {
            (first.1, (second.0.unwrap_or(u32::MAX), second.1))
        } else {
            return None;
        };

        match batch_kind {
            PrimitiveKind::Shadow => {
                let shadows_start = self.shadows_start;
                let mut shadows_end = shadows_start + 1;
                self.shadows_iter.next();
                while self
                    .shadows_iter
                    .next_if(|shadow| (shadow.order, batch_kind) < max_order_and_kind)
                    .is_some()
                {
                    shadows_end += 1;
                }
                self.shadows_start = shadows_end;
                Some(PrimitiveBatch::Shadows(shadows_start..shadows_end))
            }
            PrimitiveKind::Quad => {
                let quads_start = self.quads_start;
                let mut quads_end = quads_start + 1;
                self.quads_iter.next();
                while self
                    .quads_iter
                    .next_if(|quad| (quad.order, batch_kind) < max_order_and_kind)
                    .is_some()
                {
                    quads_end += 1;
                }
                self.quads_start = quads_end;

                let blended_quad_indices_start = self.blended_quad_indices_start;
                let mut blended_quad_indices_end = blended_quad_indices_start;
                while self
                    .blended_quad_indices
                    .get(blended_quad_indices_end)
                    .is_some_and(|&quad_id| (quad_id as usize) < quads_end)
                {
                    blended_quad_indices_end += 1;
                }
                self.blended_quad_indices_start = blended_quad_indices_end;

                Some(PrimitiveBatch::Quads {
                    range: quads_start..quads_end,
                    blended_range: blended_quad_indices_start..blended_quad_indices_end,
                })
            }
            PrimitiveKind::Path => {
                let paths_start = self.paths_start;
                let mut paths_end = paths_start + 1;
                self.paths_iter.next();
                while self
                    .paths_iter
                    .next_if(|path| (path.order, batch_kind) < max_order_and_kind)
                    .is_some()
                {
                    paths_end += 1;
                }
                self.paths_start = paths_end;
                Some(PrimitiveBatch::Paths(paths_start..paths_end))
            }
            PrimitiveKind::Underline => {
                let underlines_start = self.underlines_start;
                let mut underlines_end = underlines_start + 1;
                self.underlines_iter.next();
                while self
                    .underlines_iter
                    .next_if(|underline| (underline.order, batch_kind) < max_order_and_kind)
                    .is_some()
                {
                    underlines_end += 1;
                }
                self.underlines_start = underlines_end;
                Some(PrimitiveBatch::Underlines(underlines_start..underlines_end))
            }
            PrimitiveKind::MonochromeSprite => {
                let texture_id = self.monochrome_sprites_iter.peek().unwrap().tile.texture_id;
                let sprites_start = self.monochrome_sprites_start;
                let mut sprites_end = sprites_start + 1;
                self.monochrome_sprites_iter.next();
                while self
                    .monochrome_sprites_iter
                    .next_if(|sprite| {
                        (sprite.order, batch_kind) < max_order_and_kind
                            && sprite.tile.texture_id == texture_id
                    })
                    .is_some()
                {
                    sprites_end += 1;
                }
                self.monochrome_sprites_start = sprites_end;
                Some(PrimitiveBatch::MonochromeSprites {
                    texture_id,
                    range: sprites_start..sprites_end,
                })
            }
            PrimitiveKind::SubpixelSprite => {
                let texture_id = self.subpixel_sprites_iter.peek().unwrap().tile.texture_id;
                let sprites_start = self.subpixel_sprites_start;
                let mut sprites_end = sprites_start + 1;
                self.subpixel_sprites_iter.next();
                while self
                    .subpixel_sprites_iter
                    .next_if(|sprite| {
                        (sprite.order, batch_kind) < max_order_and_kind
                            && sprite.tile.texture_id == texture_id
                    })
                    .is_some()
                {
                    sprites_end += 1;
                }
                self.subpixel_sprites_start = sprites_end;
                Some(PrimitiveBatch::SubpixelSprites {
                    texture_id,
                    range: sprites_start..sprites_end,
                })
            }
            PrimitiveKind::PolychromeSprite => {
                let texture_id = self.polychrome_sprites_iter.peek().unwrap().tile.texture_id;
                let sprites_start = self.polychrome_sprites_start;
                let mut sprites_end = sprites_start + 1;
                self.polychrome_sprites_iter.next();
                while self
                    .polychrome_sprites_iter
                    .next_if(|sprite| {
                        (sprite.order, batch_kind) < max_order_and_kind
                            && sprite.tile.texture_id == texture_id
                    })
                    .is_some()
                {
                    sprites_end += 1;
                }
                self.polychrome_sprites_start = sprites_end;
                Some(PrimitiveBatch::PolychromeSprites {
                    texture_id,
                    range: sprites_start..sprites_end,
                })
            }
            PrimitiveKind::Surface => {
                let surfaces_start = self.surfaces_start;
                let mut surfaces_end = surfaces_start + 1;
                self.surfaces_iter.next();
                while self
                    .surfaces_iter
                    .next_if(|surface| (surface.order, batch_kind) < max_order_and_kind)
                    .is_some()
                {
                    surfaces_end += 1;
                }
                self.surfaces_start = surfaces_end;
                Some(PrimitiveBatch::Surfaces(surfaces_start..surfaces_end))
            }
        }
    }
}

#[derive(Debug)]
#[cfg_attr(
    all(
        any(target_os = "linux", target_os = "freebsd"),
        not(any(feature = "x11", feature = "wayland"))
    ),
    allow(dead_code)
)]
#[allow(missing_docs)]
pub enum PrimitiveBatch {
    Shadows(Range<usize>),
    Quads {
        range: Range<usize>,
        blended_range: Range<usize>,
    },
    Paths(Range<usize>),
    Underlines(Range<usize>),
    MonochromeSprites {
        texture_id: AtlasTextureId,
        range: Range<usize>,
    },
    #[cfg_attr(target_os = "macos", allow(dead_code))]
    SubpixelSprites {
        texture_id: AtlasTextureId,
        range: Range<usize>,
    },
    PolychromeSprites {
        texture_id: AtlasTextureId,
        range: Range<usize>,
    },
    Surfaces(Range<usize>),
}

impl PrimitiveBatch {
    #[expect(missing_docs)]
    pub fn label(&self) -> String {
        match self {
            Self::Shadows(range) => format!("shadows ({})", range.len()),
            Self::Quads { range, .. } => format!("quads ({})", range.len()),
            Self::Paths(range) => format!("paths ({})", range.len()),
            Self::Underlines(range) => format!("underlines ({})", range.len()),
            Self::MonochromeSprites { texture_id, range } => {
                format!(
                    "monochrome sprites ({}) on atlas {}",
                    range.len(),
                    texture_id.index
                )
            }
            Self::SubpixelSprites { texture_id, range } => {
                format!(
                    "subpixel sprites ({}) on atlas {}",
                    range.len(),
                    texture_id.index
                )
            }
            Self::PolychromeSprites { texture_id, range } => {
                format!(
                    "polychrome sprites ({}) on atlas {}",
                    range.len(),
                    texture_id.index
                )
            }
            Self::Surfaces(range) => format!("surfaces ({})", range.len()),
        }
    }
}

#[derive(Default, Debug, Copy, Clone)]
#[repr(C)]
#[expect(missing_docs)]
pub struct Quad {
    pub order: DrawOrder,
    pub border_style: BorderStyle,
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub background: Background,
    pub border_color: Hsla,
    pub corner_radii: Corners<ScaledPixels>,
    pub border_widths: Edges<ScaledPixels>,
}

impl Quad {
    fn has_opaque_core(&self) -> bool {
        let zero = ScaledPixels(0.);
        // Glass content deliberately leaves the destination alpha of the
        // translucent surface beneath it untouched. The opaque pass blends
        // nothing and so would write alpha 1 over that surface, so glass quads
        // never take it, however solid their color is.
        !self.background.is_glass_content()
            && self
                .background
                .as_solid()
                .is_some_and(|solid| solid.a >= 1.0)
            && self.border_widths.top == zero
            && self.border_widths.right == zero
            && self.border_widths.bottom == zero
            && self.border_widths.left == zero
    }

    fn has_rounded_corners(&self) -> bool {
        let zero = ScaledPixels(0.);
        self.corner_radii.top_left != zero
            || self.corner_radii.top_right != zero
            || self.corner_radii.bottom_right != zero
            || self.corner_radii.bottom_left != zero
    }
}

impl From<Quad> for Primitive {
    fn from(quad: Quad) -> Self {
        Primitive::Quad(quad)
    }
}

#[derive(Debug, Copy, Clone)]
#[repr(C)]
#[expect(missing_docs)]
pub struct Underline {
    pub order: DrawOrder,
    pub pad: u32, // align to 8 bytes
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub color: Hsla,
    pub thickness: ScaledPixels,
    pub wavy: PaddedBool32,
}

impl From<Underline> for Primitive {
    fn from(underline: Underline) -> Self {
        Primitive::Underline(underline)
    }
}

#[derive(Debug, Copy, Clone)]
#[repr(C)]
#[expect(missing_docs)]
pub struct Shadow {
    pub order: DrawOrder,
    pub blur_radius: ScaledPixels,
    pub bounds: Bounds<ScaledPixels>,
    pub corner_radii: Corners<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub color: Hsla,
    pub element_bounds: Bounds<ScaledPixels>,
    pub element_corner_radii: Corners<ScaledPixels>,
    /// 0 = drop shadow (rendered outside the element), 1 = inset shadow (rendered inside).
    pub inset: u32,
    pub pad: u32, // align to 8 bytes
}

impl From<Shadow> for Primitive {
    fn from(shadow: Shadow) -> Self {
        Primitive::Shadow(shadow)
    }
}

/// The style of a border.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[repr(C)]
pub enum BorderStyle {
    /// A solid border.
    #[default]
    Solid = 0,
    /// A dashed border.
    Dashed = 1,
}

/// A data type representing a 2 dimensional transformation that can be applied to an element.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct TransformationMatrix {
    /// 2x2 matrix containing rotation and scale,
    /// stored row-major
    pub rotation_scale: [[f32; 2]; 2],
    /// translation vector
    pub translation: [f32; 2],
}

impl Eq for TransformationMatrix {}

impl TransformationMatrix {
    /// The unit matrix, has no effect.
    pub fn unit() -> Self {
        Self {
            rotation_scale: [[1.0, 0.0], [0.0, 1.0]],
            translation: [0.0, 0.0],
        }
    }

    /// Move the origin by a given point
    pub fn translate(mut self, point: Point<ScaledPixels>) -> Self {
        self.compose(Self {
            rotation_scale: [[1.0, 0.0], [0.0, 1.0]],
            translation: [point.x.0, point.y.0],
        })
    }

    /// Clockwise rotation in radians around the origin
    pub fn rotate(self, angle: Radians) -> Self {
        self.compose(Self {
            rotation_scale: [
                [angle.0.cos(), -angle.0.sin()],
                [angle.0.sin(), angle.0.cos()],
            ],
            translation: [0.0, 0.0],
        })
    }

    /// Scale around the origin
    pub fn scale(self, size: Size<f32>) -> Self {
        self.compose(Self {
            rotation_scale: [[size.width, 0.0], [0.0, size.height]],
            translation: [0.0, 0.0],
        })
    }

    /// Perform matrix multiplication with another transformation
    /// to produce a new transformation that is the result of
    /// applying both transformations: first, `other`, then `self`.
    #[inline]
    pub fn compose(self, other: TransformationMatrix) -> TransformationMatrix {
        if other == Self::unit() {
            return self;
        }
        // Perform matrix multiplication
        TransformationMatrix {
            rotation_scale: [
                [
                    self.rotation_scale[0][0] * other.rotation_scale[0][0]
                        + self.rotation_scale[0][1] * other.rotation_scale[1][0],
                    self.rotation_scale[0][0] * other.rotation_scale[0][1]
                        + self.rotation_scale[0][1] * other.rotation_scale[1][1],
                ],
                [
                    self.rotation_scale[1][0] * other.rotation_scale[0][0]
                        + self.rotation_scale[1][1] * other.rotation_scale[1][0],
                    self.rotation_scale[1][0] * other.rotation_scale[0][1]
                        + self.rotation_scale[1][1] * other.rotation_scale[1][1],
                ],
            ],
            translation: [
                self.translation[0]
                    + self.rotation_scale[0][0] * other.translation[0]
                    + self.rotation_scale[0][1] * other.translation[1],
                self.translation[1]
                    + self.rotation_scale[1][0] * other.translation[0]
                    + self.rotation_scale[1][1] * other.translation[1],
            ],
        }
    }

    /// Apply transformation to a point, mainly useful for debugging
    pub fn apply(&self, point: Point<Pixels>) -> Point<Pixels> {
        let input = [point.x.0, point.y.0];
        let mut output = self.translation;
        for (i, output_cell) in output.iter_mut().enumerate() {
            for (k, input_cell) in input.iter().enumerate() {
                *output_cell += self.rotation_scale[i][k] * *input_cell;
            }
        }
        Point::new(output[0].into(), output[1].into())
    }
}

impl Default for TransformationMatrix {
    fn default() -> Self {
        Self::unit()
    }
}

/// Per-sprite highlight the text fragment shaders apply on top of the glyph
/// color. The struct is embedded in *every* [`MonochromeSprite`] and
/// [`SubpixelSprite`], so it is deliberately held to 64 bytes and the shaders
/// early-out on `kind == 0` before reading it.
#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct SpriteEffect {
    /// `0` for plain text, [`SpriteEffect::SHIMMER_KIND`] for the shimmer band.
    pub kind: u32,
    /// Where the band reaches full strength, as a fraction of `band_width`
    /// measured from the trailing edge. `0.5` is a symmetric band; values above
    /// it push the peak towards the leading edge, leaving a longer tail behind
    /// the sweep.
    pub peak: f32,
    /// Exponent applied to each side's smoothstep ramp. `1.0` is the plain
    /// smoothstep; larger values pull the highlight in towards the peak.
    pub falloff: f32,
    /// Fraction of the highlight withheld from the band's shoulders, so only
    /// the core reaches the full highlight color. `0.0` lights the whole band
    /// evenly.
    pub core_gain: f32,
    /// Point band offsets are measured from, in device pixels.
    pub origin: Point<ScaledPixels>,
    /// Half-width of the core, as a fraction of `band_width`.
    pub core_spread: f32,
    /// Padding to keep the struct at 64 bytes across all four shader backends.
    pub pad: u32,
    /// Color the band blends towards at full intensity.
    pub highlight_color: Hsla,
    /// Trailing edge of the highlight band, measured along `direction` from
    /// `origin`. Sweeping this past the projected extent of the text animates
    /// the shimmer.
    pub band_origin: f32,
    /// Width of the highlight band along `direction`.
    pub band_width: f32,
    /// Unit vector the band sweeps along, in y-down screen space. A CSS
    /// `linear-gradient(<angle>, ...)` maps to `(sin angle, -cos angle)`.
    pub direction: [f32; 2],
}

impl SpriteEffect {
    pub(crate) const SHIMMER_KIND: u32 = 1;
}

// Every glyph in the window carries one of these, and the four shader backends
// mirror the layout by hand, so growing it is a decision rather than an
// accident. `pad` exists to keep this assertion true.
const _: () = assert!(std::mem::size_of::<SpriteEffect>() == 64);

impl Default for SpriteEffect {
    fn default() -> Self {
        Self {
            kind: 0,
            peak: 0.5,
            falloff: 1.0,
            core_gain: 0.0,
            origin: Point::default(),
            core_spread: 0.5,
            pad: 0,
            highlight_color: Hsla::default(),
            band_origin: 0.0,
            band_width: 0.0,
            direction: [1.0, 0.0],
        }
    }
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
#[expect(missing_docs)]
pub struct MonochromeSprite {
    pub order: DrawOrder,
    pub pad: u32,
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub color: Hsla,
    pub effect: SpriteEffect,
    pub tile: AtlasTile,
    pub transformation: TransformationMatrix,
}

impl From<MonochromeSprite> for Primitive {
    fn from(sprite: MonochromeSprite) -> Self {
        Primitive::MonochromeSprite(sprite)
    }
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
#[expect(missing_docs)]
pub struct SubpixelSprite {
    pub order: DrawOrder,
    pub pad: u32, // align to 8 bytes
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub color: Hsla,
    pub effect: SpriteEffect,
    pub tile: AtlasTile,
    pub transformation: TransformationMatrix,
}

impl From<SubpixelSprite> for Primitive {
    fn from(sprite: SubpixelSprite) -> Self {
        Primitive::SubpixelSprite(sprite)
    }
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
#[expect(missing_docs)]
pub struct PolychromeSprite {
    pub order: DrawOrder,
    pub pad: u32,
    pub grayscale: PaddedBool32,
    pub opacity: f32,
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub corner_radii: Corners<ScaledPixels>,
    pub tile: AtlasTile,
}

impl From<PolychromeSprite> for Primitive {
    fn from(sprite: PolychromeSprite) -> Self {
        Primitive::PolychromeSprite(sprite)
    }
}

#[derive(Clone, Debug)]
#[allow(missing_docs)]
pub struct PaintSurface {
    pub order: DrawOrder,
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    #[cfg(target_os = "macos")]
    pub image_buffer: core_video::pixel_buffer::CVPixelBuffer,
}

impl From<PaintSurface> for Primitive {
    fn from(surface: PaintSurface) -> Self {
        Primitive::Surface(surface)
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[expect(missing_docs)]
pub struct PathId(pub usize);

/// A line made up of a series of vertices and control points.
#[derive(Clone, Debug)]
#[expect(missing_docs)]
pub struct Path<P: Clone + Debug + Default + PartialEq> {
    pub id: PathId,
    pub order: DrawOrder,
    pub bounds: Bounds<P>,
    pub content_mask: ContentMask<P>,
    pub vertices: Vec<PathVertex<P>>,
    pub color: Background,
    start: Point<P>,
    current: Point<P>,
    contour_count: usize,
}

impl Path<Pixels> {
    /// Create a new path with the given starting point.
    pub fn new(start: Point<Pixels>) -> Self {
        Self {
            id: PathId(0),
            order: DrawOrder::default(),
            vertices: Vec::new(),
            start,
            current: start,
            bounds: Bounds {
                origin: start,
                size: Default::default(),
            },
            content_mask: Default::default(),
            color: Default::default(),
            contour_count: 0,
        }
    }

    /// Scale this path by the given factor.
    pub fn scale(&self, factor: f32) -> Path<ScaledPixels> {
        Path {
            id: self.id,
            order: self.order,
            bounds: self.bounds.scale(factor),
            content_mask: self.content_mask.scale(factor),
            vertices: self
                .vertices
                .iter()
                .map(|vertex| vertex.scale(factor))
                .collect(),
            start: self.start.map(|start| start.scale(factor)),
            current: self.current.scale(factor),
            contour_count: self.contour_count,
            color: self.color,
        }
    }

    /// Move the start, current point to the given point.
    pub fn move_to(&mut self, to: Point<Pixels>) {
        self.contour_count += 1;
        self.start = to;
        self.current = to;
    }

    /// Draw a straight line from the current point to the given point.
    pub fn line_to(&mut self, to: Point<Pixels>) {
        self.contour_count += 1;
        if self.contour_count > 1 {
            self.push_triangle(
                (self.start, self.current, to),
                (point(0., 1.), point(0., 1.), point(0., 1.)),
            );
        }
        self.current = to;
    }

    /// Draw a curve from the current point to the given point, using the given control point.
    pub fn curve_to(&mut self, to: Point<Pixels>, ctrl: Point<Pixels>) {
        self.contour_count += 1;
        if self.contour_count > 1 {
            self.push_triangle(
                (self.start, self.current, to),
                (point(0., 1.), point(0., 1.), point(0., 1.)),
            );
        }

        self.push_triangle(
            (self.current, ctrl, to),
            (point(0., 0.), point(0.5, 0.), point(1., 1.)),
        );
        self.current = to;
    }

    /// Push a triangle to the Path.
    pub fn push_triangle(
        &mut self,
        xy: (Point<Pixels>, Point<Pixels>, Point<Pixels>),
        st: (Point<f32>, Point<f32>, Point<f32>),
    ) {
        self.bounds = self
            .bounds
            .union(&Bounds {
                origin: xy.0,
                size: Default::default(),
            })
            .union(&Bounds {
                origin: xy.1,
                size: Default::default(),
            })
            .union(&Bounds {
                origin: xy.2,
                size: Default::default(),
            });

        self.vertices.push(PathVertex {
            xy_position: xy.0,
            st_position: st.0,
            content_mask: Default::default(),
        });
        self.vertices.push(PathVertex {
            xy_position: xy.1,
            st_position: st.1,
            content_mask: Default::default(),
        });
        self.vertices.push(PathVertex {
            xy_position: xy.2,
            st_position: st.2,
            content_mask: Default::default(),
        });
    }
}

impl<T> Path<T>
where
    T: Clone + Debug + Default + PartialEq + PartialOrd + Add<T, Output = T> + Sub<Output = T>,
{
    #[allow(unused)]
    #[expect(missing_docs)]
    pub fn clipped_bounds(&self) -> Bounds<T> {
        self.bounds.intersect(&self.content_mask.bounds)
    }
}

impl From<Path<ScaledPixels>> for Primitive {
    fn from(path: Path<ScaledPixels>) -> Self {
        Primitive::Path(path)
    }
}

#[derive(Clone, Debug)]
#[repr(C)]
#[expect(missing_docs)]
pub struct PathVertex<P: Clone + Debug + Default + PartialEq> {
    pub xy_position: Point<P>,
    pub st_position: Point<f32>,
    pub content_mask: ContentMask<P>,
}

#[expect(missing_docs)]
impl PathVertex<Pixels> {
    pub fn scale(&self, factor: f32) -> PathVertex<ScaledPixels> {
        PathVertex {
            xy_position: self.xy_position.scale(factor),
            st_position: self.st_position,
            content_mask: self.content_mask.scale(factor),
        }
    }
}

#[cfg(test)]
mod glyph_drop_diagnostic_tests {
    use super::{GLYPH_NEAR_MISS_PIXELS, mask_miss_distances};
    use crate::{Bounds, ScaledPixels, point, size};

    fn bounds(x: f32, y: f32, width: f32, height: f32) -> Bounds<ScaledPixels> {
        Bounds {
            origin: point(ScaledPixels(x), ScaledPixels(y)),
            size: size(ScaledPixels(width), ScaledPixels(height)),
        }
    }

    /// The diagnostic is only useful if it separates a clip edge cutting into a line of text
    /// from the line above a scroll viewport. Both are "culled by the content mask", both can
    /// miss by well under a pixel, and only the first is a bug -- so distance alone is not
    /// enough, the axis carries the signal.
    #[test]
    fn horizontal_near_misses_are_distinguished_from_scroll_edges() {
        let mask = bounds(100., 100., 400., 200.);

        // A clip edge landing inside a line of text: overlaps vertically, misses horizontally
        // by a hair. This is the case worth waking someone up for.
        for glyph in [bounds(99.4, 150., 0.5, 10.), bounds(500.1, 150., 8., 10.)] {
            let (horizontal, vertical) = mask_miss_distances(&glyph, &mask);
            assert!(
                horizontal > 0.0 && horizontal < GLYPH_NEAR_MISS_PIXELS && vertical == 0.0,
                "{glyph:?} should read as a horizontal near miss, got {horizontal}x / {vertical}y"
            );
        }

        // The line just above or below a scroll viewport. Misses by a hair too, but vertically,
        // and is supposed to be invisible. A transcript emits hundreds of these per second, so
        // reporting them would bury the case above.
        for glyph in [bounds(150., 90., 8., 9.), bounds(150., 300.5, 8., 10.)] {
            let (horizontal, vertical) = mask_miss_distances(&glyph, &mask);
            assert_eq!(horizontal, 0.0, "{glyph:?} does not miss horizontally at all");
            assert!(vertical > 0.0, "{glyph:?} is a scroll-edge cull");
        }

        // The commonest shape in a real transcript: the line above the viewport ends exactly
        // on the mask's top edge, so the intersection is empty and it is dropped while missing
        // by zero on both axes. It must NOT read as horizontal, or every scrolling frame
        // reports hundreds of false positives.
        let (horizontal, vertical) = mask_miss_distances(&bounds(150., 91., 8., 9.), &mask);
        assert_eq!((horizontal, vertical), (0.0, 0.0));

        // Scrolled well out of the viewport: correctly culled, far away on both counts.
        let (_, vertical) = mask_miss_distances(&bounds(150., -400., 8., 10.), &mask);
        assert!(vertical >= GLYPH_NEAR_MISS_PIXELS);

        // Exactly touching an edge yields an empty intersection and so is dropped, with a zero
        // miss. That is the most suspicious geometry of all and must not read as "far away".
        assert_eq!(mask_miss_distances(&bounds(92., 150., 8., 10.), &mask).0, 0.0);
        assert_eq!(mask_miss_distances(&bounds(150., 150., 8., 10.), &mask), (0.0, 0.0));
    }
}
