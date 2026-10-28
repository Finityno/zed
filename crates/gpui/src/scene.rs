// todo("windows"): remove
#![cfg_attr(windows, allow(dead_code))]

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{
    AtlasTextureId, AtlasTile, Background, Bounds, ContentMask, Corners, DevicePixels, Edges, Hsla, Pixels,
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

/// Logs a glyph sprite culled by its content mask, when reporting is on.
fn report_dropped_glyph(bounds: &Bounds<ScaledPixels>, mask: &Bounds<ScaledPixels>) {
    let mode = glyph_drop_reporting();
    if mode != GlyphDropReporting::Off {
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
    /// Filled by `finish`, in draw order, from the painted sprites below.
    pub monochrome_sprites: Vec<MonochromeSprite>,
    pub subpixel_sprites: Vec<SubpixelSprite>,
    /// Glyph sprites in the order they were painted, which paint operations
    /// name by index rather than keeping a copy of each as well. Unlike the
    /// draw-order vectors, presenting never changes them, so they are what a
    /// later frame replays.
    painted_monochrome_sprites: Vec<MonochromeSprite>,
    painted_subpixel_sprites: Vec<SubpixelSprite>,
    pub polychrome_sprites: Vec<PolychromeSprite>,
    pub surfaces: Vec<PaintSurface>,
    /// Backdrop blurs, in draw order once `finish` has run. They are kept out
    /// of the primitive batches: a renderer ends its pass at each one's order
    /// to blur what is already in the framebuffer beneath it.
    pub backdrop_blurs: Vec<BackdropBlur>,
    /// Rounded masks referenced by primitives, one-based so zero needs no mask.
    pub rounded_clips: Vec<RoundedClipNode>,
    pub(crate) current_rounded_clip: u32,
    pub blended_quad_indices: Vec<u32>,
    pub opaque_quad_indices: Vec<u32>,
    /// Sweeps referenced by `SpriteEffect::animation`.
    shimmer_animations: Vec<ShimmerAnimation>,
    /// Turns referenced by `SpriteEffect::animation` with
    /// [`ROTATION_ANIMATION`] set.
    rotation_animations: Vec<RotationAnimation>,
    /// The monochrome sprites that turn, by draw-order index, with the
    /// transformation each was painted with and where it was painted,
    /// gathered by `finish`: every present rotates that transformation, so
    /// turning never compounds.
    rotating_monochrome_sprites: Vec<(u32, TransformationMatrix, Point<ScaledPixels>)>,
    /// Indices of the sprites whose effect animates, gathered by `finish`
    /// (sorting moves sprites, so they cannot be recorded on insertion).
    animated_monochrome_sprites: Vec<u32>,
    animated_subpixel_sprites: Vec<u32>,
    /// Opacity cycles referenced by `Background::time_animation`.
    quad_animations: Vec<QuadAnimation>,
    /// The animated quads that move with an [`OffsetCycle`], by draw-order
    /// index, with the origin each was painted at, gathered by `finish`, and
    /// where the latest advance put it at rest, transitions included: every
    /// present offsets from there, so moving never compounds.
    offset_quads: Vec<OffsetQuad>,
    animated_quads: Vec<u32>,
    /// Moves referenced by each primitive's transition id (`pad`, or the
    /// background's transition bits for quads).
    pub(crate) transitions: Vec<SceneTransition>,
    /// The transition primitives inserted now are stamped with, set by
    /// [`crate::Window::with_time_transition`].
    current_transition: u32,
    /// Gathered by `finish`, like the animated sprite indices.
    transitioned: Vec<TransitionedPrimitive>,
    /// When the last transition this scene carries lands.
    transitions_end_at: Option<std::time::Instant>,
    /// Each transition's composed offset and opacity for the present being
    /// prepared, reused across presents.
    transition_states: Vec<(Point<ScaledPixels>, f32)>,
    /// Which `finish` produced what this scene holds, unique across scenes,
    /// so a renderer can tell a present of the frame it drew last from a
    /// present of a new one. Zero until the first `finish`.
    content_id: u64,
    /// When [`Self::advance_time_animations`] last ran on this content.
    last_advanced_at: Option<std::time::Instant>,
    /// Whether the transitions this scene carries had already landed when
    /// the advance before the latest one ran, so the latest moved nothing.
    transitions_unmoved: bool,
    /// One tracker per vector above, in the order `clear` destructures them.
    shrink: [CapacityShrink; 15],
    /// Room `finish` sorts in, kept across frames.
    draw_order_scratch: DrawOrderScratch,
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
            painted_monochrome_sprites,
            painted_subpixel_sprites,
            rounded_clips,
        ] = &mut self.shrink;
        paint_operations.clear_vec(&mut self.paint_operations);
        self.primitive_bounds.clear();
        rounded_clips.clear_vec(&mut self.rounded_clips);
        self.current_rounded_clip = 0;
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
        painted_monochrome_sprites.clear_vec(&mut self.painted_monochrome_sprites);
        painted_subpixel_sprites.clear_vec(&mut self.painted_subpixel_sprites);
        self.backdrop_blurs.clear();
        self.shimmer_animations.clear();
        self.rotation_animations.clear();
        self.rotating_monochrome_sprites.clear();
        self.animated_monochrome_sprites.clear();
        self.animated_subpixel_sprites.clear();
        self.quad_animations.clear();
        self.offset_quads.clear();
        self.animated_quads.clear();
        self.transitions.clear();
        self.current_transition = 0;
        self.transitioned.clear();
        self.transitions_end_at = None;
        self.last_advanced_at = None;
        self.transitions_unmoved = false;
    }

    /// Shrinks this scene's vectors to twice the fill of `rendered`, the
    /// scene still on screen, once the window has stopped drawing; see
    /// [`CapacityShrink::idle_target`]. With `None` this is the scene on
    /// screen, and keeps twice its own fill with its contents as they are.
    pub(crate) fn shrink_idle(&mut self, rendered: Option<&Scene>) {
        let fill = |scene: &Scene| {
            [
                scene.paint_operations.len(),
                scene.layer_stack.len(),
                scene.paths.len(),
                scene.shadows.len(),
                scene.quads.len(),
                scene.underlines.len(),
                scene.monochrome_sprites.len(),
                scene.subpixel_sprites.len(),
                scene.polychrome_sprites.len(),
                scene.surfaces.len(),
                scene.blended_quad_indices.len(),
                scene.opaque_quad_indices.len(),
                scene.painted_monochrome_sprites.len(),
                scene.painted_subpixel_sprites.len(),
                scene.rounded_clips.len(),
                scene.largest_primitive_count(),
            ]
        };
        let [
            paint_operations_len,
            layer_stack_len,
            paths_len,
            shadows_len,
            quads_len,
            underlines_len,
            monochrome_sprites_len,
            subpixel_sprites_len,
            polychrome_sprites_len,
            surfaces_len,
            blended_quad_indices_len,
            opaque_quad_indices_len,
            painted_monochrome_sprites_len,
            painted_subpixel_sprites_len,
            rounded_clips_len,
            largest_primitive_count,
        ] = fill(rendered.unwrap_or(&*self));
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
            painted_monochrome_sprites,
            painted_subpixel_sprites,
            rounded_clips,
        ] = &mut self.shrink;
        paint_operations.shrink_vec_idle(&mut self.paint_operations, paint_operations_len);
        self.primitive_bounds
            .shrink_idle(rendered.map(|rendered| &rendered.primitive_bounds));
        rounded_clips.shrink_vec_idle(&mut self.rounded_clips, rounded_clips_len);
        layer_stack.shrink_vec_idle(&mut self.layer_stack, layer_stack_len);
        paths.shrink_vec_idle(&mut self.paths, paths_len);
        shadows.shrink_vec_idle(&mut self.shadows, shadows_len);
        quads.shrink_vec_idle(&mut self.quads, quads_len);
        underlines.shrink_vec_idle(&mut self.underlines, underlines_len);
        monochrome_sprites.shrink_vec_idle(&mut self.monochrome_sprites, monochrome_sprites_len);
        subpixel_sprites.shrink_vec_idle(&mut self.subpixel_sprites, subpixel_sprites_len);
        polychrome_sprites.shrink_vec_idle(&mut self.polychrome_sprites, polychrome_sprites_len);
        surfaces.shrink_vec_idle(&mut self.surfaces, surfaces_len);
        blended_quad_indices
            .shrink_vec_idle(&mut self.blended_quad_indices, blended_quad_indices_len);
        opaque_quad_indices.shrink_vec_idle(&mut self.opaque_quad_indices, opaque_quad_indices_len);
        painted_monochrome_sprites.shrink_vec_idle(
            &mut self.painted_monochrome_sprites,
            painted_monochrome_sprites_len,
        );
        painted_subpixel_sprites
            .shrink_vec_idle(&mut self.painted_subpixel_sprites, painted_subpixel_sprites_len);
        self.draw_order_scratch.shrink_idle(largest_primitive_count);
    }

    fn largest_primitive_count(&self) -> usize {
        self.shadows
            .len()
            .max(self.quads.len())
            .max(self.paths.len())
            .max(self.underlines.len())
            .max(self.painted_monochrome_sprites.len())
            .max(self.painted_subpixel_sprites.len())
            .max(self.polychrome_sprites.len())
            .max(self.surfaces.len())
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
            && self.painted_monochrome_sprites.is_empty()
            && self.painted_subpixel_sprites.is_empty()
            && self.polychrome_sprites.is_empty()
            && self.surfaces.is_empty()
            && self.backdrop_blurs.is_empty()
    }

    pub fn push_layer(&mut self, bounds: Bounds<ScaledPixels>) {
        self.push_clipped_layer(bounds, bounds, bounds);
    }

    /// Pushes a layer of `bounds`, what is left of `extent` once `mask`
    /// clipped it, covered to whole device pixels.
    pub(crate) fn push_clipped_layer(
        &mut self,
        bounds: Bounds<ScaledPixels>,
        extent: Bounds<ScaledPixels>,
        mask: Bounds<ScaledPixels>,
    ) {
        let order = self.primitive_bounds.insert(bounds);
        self.layer_stack.push(order);
        self.paint_operations.push(PaintOperation::StartLayer(Layer {
            bounds,
            extent,
            mask,
        }));
    }

    pub fn pop_layer(&mut self) {
        self.layer_stack.pop();
        self.paint_operations.push(PaintOperation::EndLayer);
    }

    /// [`Self::insert_primitive`] for a glyph sprite, without first moving the
    /// sprite into a [`Primitive`]: text paints one of these per visible glyph,
    /// and the detour through the enum costs an extra copy of every sprite.
    pub(crate) fn insert_monochrome_sprite(&mut self, mut sprite: MonochromeSprite) {
        let Some(order) =
            self.glyph_sprite_order(&sprite.bounds, &sprite.content_mask.bounds, &mut sprite.pad)
        else {
            return;
        };
        sprite.order = order;
        sprite.rounded_clip = self.current_rounded_clip;
        let index = self.painted_monochrome_sprites.len() as u32;
        self.painted_monochrome_sprites.push(sprite);
        self.paint_operations
            .push(PaintOperation::MonochromeSprite(index));
    }

    /// See [`Self::insert_monochrome_sprite`].
    pub(crate) fn insert_subpixel_sprite(&mut self, mut sprite: SubpixelSprite) {
        let Some(order) =
            self.glyph_sprite_order(&sprite.bounds, &sprite.content_mask.bounds, &mut sprite.pad)
        else {
            return;
        };
        sprite.order = order;
        sprite.rounded_clip = self.current_rounded_clip;
        let index = self.painted_subpixel_sprites.len() as u32;
        self.painted_subpixel_sprites.push(sprite);
        self.paint_operations
            .push(PaintOperation::SubpixelSprite(index));
    }

    /// The number of paint operations and of painted glyph sprites so far,
    /// which tell what a line painted when taken before and after it.
    pub(crate) fn glyph_watermark(&self) -> GlyphWatermark {
        GlyphWatermark {
            operations: self.paint_operations.len(),
            sprites: self.painted_monochrome_sprites.len() + self.painted_subpixel_sprites.len(),
        }
    }

    /// The paint operations since `start`, if every one of them is a glyph
    /// sprite.
    pub(crate) fn glyph_sprites_since(&self, start: GlyphWatermark) -> Option<Range<usize>> {
        let end = self.glyph_watermark();
        (end.operations - start.operations == end.sprites - start.sprites)
            .then_some(start.operations..end.operations)
    }

    /// Paints again the glyph sprites `operations` names in `prev_scene`, all
    /// of which [`Self::glyph_sprites_since`] found to be glyph sprites, into
    /// the layer being painted, as painting the same glyphs at the same place
    /// would: each takes the layer's draw order and the current transition.
    /// Returns false, having painted nothing, outside a layer, where each
    /// sprite would take a draw order of its own.
    pub(crate) fn replay_glyph_sprites(
        &mut self,
        prev_scene: &Scene,
        operations: Range<usize>,
    ) -> bool {
        let Some(&order) = self.layer_stack.last() else {
            return false;
        };
        let Some(prev_operations) = prev_scene.paint_operations.get(operations) else {
            return false;
        };
        let pad = self.current_transition;
        self.paint_operations.reserve(prev_operations.len());
        // A line's sprites were pushed one after another, so they are
        // replayed a run of consecutive indices at a time, which names each
        // run's paint operations as a range rather than sprite by sprite.
        let mut position = 0;
        while let Some(operation) = prev_operations.get(position) {
            let (subpixel, first) = match *operation {
                PaintOperation::MonochromeSprite(index) => (false, index),
                PaintOperation::SubpixelSprite(index) => (true, index),
                PaintOperation::Primitive(_)
                | PaintOperation::BackdropBlur(_)
                | PaintOperation::StartLayer(_)
                | PaintOperation::EndLayer => {
                    debug_assert!(false, "only glyph sprites are replayed");
                    position += 1;
                    continue;
                }
            };
            let mut run = 1;
            while let Some(next) = prev_operations.get(position + run) {
                let next_index = first + run as u32;
                let continues = match *next {
                    PaintOperation::MonochromeSprite(index) => !subpixel && index == next_index,
                    PaintOperation::SubpixelSprite(index) => subpixel && index == next_index,
                    _ => false,
                };
                if !continues {
                    break;
                }
                run += 1;
            }
            let sources = first as usize..first as usize + run;
            if subpixel {
                let start = append_replayed_sprites(
                    &mut self.painted_subpixel_sprites,
                    &prev_scene.painted_subpixel_sprites[sources],
                    |sprite| {
                        sprite.order = order;
                        sprite.pad = pad;
                        sprite.rounded_clip = self.current_rounded_clip;
                    },
                );
                self.paint_operations
                    .extend((start..start + run as u32).map(PaintOperation::SubpixelSprite));
            } else {
                let start = append_replayed_sprites(
                    &mut self.painted_monochrome_sprites,
                    &prev_scene.painted_monochrome_sprites[sources],
                    |sprite| {
                        sprite.order = order;
                        sprite.pad = pad;
                        sprite.rounded_clip = self.current_rounded_clip;
                    },
                );
                self.paint_operations
                    .extend((start..start + run as u32).map(PaintOperation::MonochromeSprite));
            }
            position += run;
        }
        true
    }

    /// The cull, transition and draw order steps of [`Self::insert_primitive`]
    /// for a glyph sprite; `None` when the sprite is culled.
    fn glyph_sprite_order(
        &mut self,
        bounds: &Bounds<ScaledPixels>,
        mask: &Bounds<ScaledPixels>,
        pad: &mut u32,
    ) -> Option<DrawOrder> {
        let clipped_bounds = bounds.intersect(mask);
        if clipped_bounds.is_empty() {
            report_dropped_glyph(bounds, mask);
            return None;
        }
        if self.current_transition != 0 && *pad == 0 {
            *pad = self.current_transition;
        }
        Some(
            self.layer_stack
                .last()
                .copied()
                .unwrap_or_else(|| self.primitive_bounds.insert(clipped_bounds)),
        )
    }

    /// Records a backdrop blur at the current layer's draw order, so that
    /// everything the layer draws composites over the blurred backdrop.
    pub fn insert_backdrop_blur(&mut self, mut blur: BackdropBlur) {
        let clipped_bounds = blur.bounds.intersect(&blur.content_mask.bounds);
        if clipped_bounds.is_empty() || blur.opacity <= 0. {
            return;
        }
        blur.order = self
            .layer_stack
            .last()
            .copied()
            .unwrap_or_else(|| self.primitive_bounds.insert(clipped_bounds));
        self.backdrop_blurs.push(blur);
        self.paint_operations
            .push(PaintOperation::BackdropBlur(blur));
    }

    /// The draw order of the first primitive in `batch`.
    pub fn batch_order(&self, batch: &PrimitiveBatch) -> DrawOrder {
        match batch {
            PrimitiveBatch::Shadows(range) => self.shadows[range.start].order,
            PrimitiveBatch::Quads { range, .. } => self.quads[range.start].order,
            PrimitiveBatch::Paths(range) => self.paths[range.start].order,
            PrimitiveBatch::Underlines(range) => self.underlines[range.start].order,
            PrimitiveBatch::MonochromeSprites { range, .. } => {
                self.monochrome_sprites[range.start].order
            }
            PrimitiveBatch::SubpixelSprites { range, .. } => {
                self.subpixel_sprites[range.start].order
            }
            PrimitiveBatch::PolychromeSprites { range, .. } => {
                self.polychrome_sprites[range.start].order
            }
            PrimitiveBatch::Surfaces(range) => self.surfaces[range.start].order,
        }
    }

    pub fn insert_primitive(&mut self, primitive: impl Into<Primitive>) {
        let mut primitive = primitive.into();
        if primitive.rounded_clip() == 0 {
            primitive.set_rounded_clip(self.current_rounded_clip);
        }
        // A turning sprite reaches the square it sweeps, not just its own
        // rectangle, so that is what it is culled and ordered by.
        let reach = match &primitive {
            Primitive::MonochromeSprite(sprite) => monochrome_sprite_extent(sprite),
            Primitive::Quad(quad) => self.quad_extent(quad),
            primitive => *primitive.bounds(),
        };
        let clipped_bounds = reach.intersect(&primitive.content_mask().bounds);

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
                report_dropped_glyph(primitive.bounds(), &primitive.content_mask().bounds);
            }
            return;
        }

        if self.current_transition != 0 {
            let transition = self.current_transition;
            match &mut primitive {
                Primitive::MonochromeSprite(MonochromeSprite { pad, .. })
                | Primitive::SubpixelSprite(SubpixelSprite { pad, .. })
                | Primitive::PolychromeSprite(PolychromeSprite { pad, .. })
                | Primitive::Underline(Underline { pad, .. })
                | Primitive::Shadow(Shadow { pad, .. })
                    if *pad == 0 =>
                {
                    *pad = transition;
                }
                Primitive::Quad(quad) if quad.background.time_transition() == 0 => {
                    quad.background = quad.background.with_time_transition(transition);
                }
                _ => {}
            }
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
                let index = self.painted_monochrome_sprites.len() as u32;
                self.painted_monochrome_sprites.push(*sprite);
                self.paint_operations
                    .push(PaintOperation::MonochromeSprite(index));
                return;
            }
            Primitive::SubpixelSprite(sprite) => {
                sprite.order = order;
                let index = self.painted_subpixel_sprites.len() as u32;
                self.painted_subpixel_sprites.push(*sprite);
                self.paint_operations
                    .push(PaintOperation::SubpixelSprite(index));
                return;
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

    /// Registers an opacity cycle or an offset cycle for one quad, returning
    /// the value its background's time animation should carry.
    pub(crate) fn push_quad_animation(&mut self, animation: QuadAnimation) -> u32 {
        self.quad_animations.push(animation);
        self.quad_animations.len() as u32
    }

    /// Inserts `quad` with its opacity driven by `cycle`, as a quad painted
    /// inside [`crate::Window::with_opacity_cycle`] is.
    pub fn insert_opacity_cycled_quad(&mut self, quad: Quad, cycle: OpacityCycle) {
        let animation = self.push_quad_animation(QuadAnimation {
            opacity: Some(cycle),
            offset: None,
            background: quad.background,
            border_color: quad.border_color,
        });
        self.insert_primitive(Quad {
            background: quad.background.with_time_animation(animation),
            ..quad
        });
    }

    /// Registers a continuous turn for one sprite, returning the value its
    /// [`SpriteEffect::animation`] should carry.
    pub(crate) fn push_rotation_animation(&mut self, animation: RotationAnimation) -> u32 {
        self.rotation_animations.push(animation);
        self.rotation_animations.len() as u32 | ROTATION_ANIMATION
    }

    /// Registers a sweep for sprites painted after this call, returning the
    /// value their [`SpriteEffect::animation`] should carry.
    pub(crate) fn push_shimmer_animation(&mut self, animation: ShimmerAnimation) -> u32 {
        self.shimmer_animations.push(animation);
        self.shimmer_animations.len() as u32
    }

    /// Registers a transition for primitives inserted until the next
    /// [`Self::set_current_transition`], returning its id, or `None` once the
    /// scene holds more than a quad's background can name.
    pub(crate) fn push_transition(&mut self, transition: SceneTransition) -> Option<u32> {
        let id = u32::try_from(self.transitions.len() + 1).ok()?;
        if id > Background::MAX_TIME_TRANSITION {
            return None;
        }
        self.transitions.push(transition);
        Some(id)
    }

    /// Copies `transition` and the transitions it was pushed inside from
    /// `prev_scene`, parents first, so a replayed id always names a later
    /// entry than its parent's. `0` when the scene has run out of ids.
    ///
    /// `remapped` starts out holding the transition the replayed range was
    /// painted inside, mapped to the one it is replayed inside, when those
    /// differ; see [`Self::replay_inside`].
    ///
    /// With `landed_before` given, a transition that has landed by then and
    /// rests where it started (no offset, full opacity) is left out, its
    /// primitives taking its parent's: it no longer moves or fades anything,
    /// and content drawn again frame after frame would otherwise copy it
    /// forever, against the scene's budget of transitions.
    fn remap_transition(
        &mut self,
        prev_scene: &Scene,
        transition: u32,
        remapped: &mut Vec<(u32, u32)>,
        landed_before: Option<std::time::Instant>,
    ) -> u32 {
        if let Some(&(_, id)) = remapped.iter().find(|(source, _)| *source == transition) {
            return id;
        }
        let mut entry = prev_scene.transitions[transition as usize - 1];
        if let Some(&(_, id)) = remapped.iter().find(|(source, _)| *source == entry.parent) {
            entry.parent = id;
        } else if entry.parent != 0 {
            entry.parent =
                self.remap_transition(prev_scene, entry.parent, remapped, landed_before);
        }
        if landed_before.is_some_and(|now| entry.transition.rests_at_identity_by(now)) {
            remapped.push((transition, entry.parent));
            return entry.parent;
        }
        let id = self.push_transition(entry).unwrap_or(0);
        remapped.push((transition, id));
        id
    }

    /// The transition primitives are being stamped with, `0` for none.
    pub(crate) fn current_transition(&self) -> u32 {
        self.current_transition
    }

    /// Stamps primitives inserted from now on with `transition` (`0` for
    /// none), returning the one that was current.
    pub(crate) fn set_current_transition(&mut self, transition: u32) -> u32 {
        std::mem::replace(&mut self.current_transition, transition)
    }

    fn remap_rounded_clip(
        &mut self,
        previous: &Scene,
        index: u32,
        remapped: &mut Vec<(u32, u32)>,
        moved: Option<&SceneMove>,
        transitions: &mut Vec<(u32, u32)>,
        landed_before: Option<std::time::Instant>,
        inherited_clip: Option<(usize, u32)>,
    ) -> u32 {
        if index == 0 { return 0; }
        if let Some((_, current)) = remapped.iter().find(|(old, _)| *old == index) {
            return *current;
        }
        let inherited_clip = inherited_clip.or_else(|| moved.map(|moved| (moved.inherited_clip_depth, moved.inherited_clip)));
        if let Some((inherited_depth, inherited_index)) = inherited_clip {
            let mut depth = 0;
            let mut ancestor = index;
            while ancestor != 0 {
                depth += 1;
                ancestor = previous.rounded_clips[ancestor as usize - 1].parent;
            }
            if depth <= inherited_depth { return inherited_index; }
        }
        let mut mask = previous.rounded_clips[index as usize - 1];
        mask.parent = self.remap_rounded_clip(previous, mask.parent, remapped, moved, transitions, landed_before, inherited_clip);
        if let Some((_, current)) = transitions.iter().find(|(old, _)| *old == mask.transition) {
            mask.transition = *current;
        } else if mask.transition != 0 {
            mask.transition = self.remap_transition(previous, mask.transition, transitions, landed_before);
        }
        if let Some(moved) = moved { mask.base_origin = mask.base_origin + moved.delta; }
        mask.bounds.origin = mask.base_origin;
        self.rounded_clips.push(mask);
        let current = self.rounded_clips.len() as u32;
        remapped.push((index, current));
        current
    }

    pub fn replay(&mut self, range: Range<usize>, prev_scene: &Scene) {
        self.replay_inside(range, prev_scene, None, None, None);
    }

    /// Splits this finished scene at `split` into the two planes a window
    /// with a scene overlay presents separately, each finished and moved to
    /// now.
    ///
    /// Replaying copies primitives as they were painted, while the scene
    /// moves its time-driven primitives in their drawn copies before each
    /// present (see [`Self::advance_time_animations`]). A present that redraws
    /// nothing, which is how a shimmer, a pulse or a transition advances,
    /// would otherwise show both planes at the phase they were painted with.
    pub fn split_planes_into(&self, split: usize, base: &mut Scene, overlay: &mut Scene) {
        let split = split.min(self.len());
        for (plane, range) in [(base, 0..split), (overlay, split..self.len())] {
            plane.clear();
            plane.replay(range, self);
            plane.finish();
            plane.advance_time_animations();
        }
    }

    /// Whether every primitive and layer in `range` that reached past its
    /// content mask was clipped by a mask other than `outer`, on each side
    /// it reached past; see [`SceneMove`].
    pub(crate) fn clipped_only_inside(&self, range: Range<usize>, outer: [f32; 4]) -> bool {
        use crate::window::view_retention::moving::{clipped_only_inside, scaled_edges};
        let clipped = |extent: &Bounds<ScaledPixels>, mask: &ContentMask<ScaledPixels>| {
            clipped_only_inside(scaled_edges(extent), scaled_edges(&mask.bounds), outer)
        };
        self.paint_operations[range].iter().all(|operation| match operation {
            PaintOperation::Primitive(Primitive::Quad(quad)) => {
                clipped(&self.quad_extent(quad), &quad.content_mask)
            }
            PaintOperation::Primitive(primitive) => {
                clipped(&primitive_extent(primitive), primitive.content_mask())
            }
            PaintOperation::MonochromeSprite(index) => {
                let sprite = &self.painted_monochrome_sprites[*index as usize];
                clipped(&monochrome_sprite_extent(sprite), &sprite.content_mask)
            }
            PaintOperation::SubpixelSprite(index) => {
                let sprite = &self.painted_subpixel_sprites[*index as usize];
                clipped(&sprite.bounds, &sprite.content_mask)
            }
            PaintOperation::BackdropBlur(blur) => clipped(&blur.bounds, &blur.content_mask),
            // A layer is clipped again where it is drawn moved (see
            // `replay_inside`).
            PaintOperation::StartLayer(_) | PaintOperation::EndLayer => true,
        })
    }

    /// Replays `range` of `prev_scene`, as [`Self::replay`] does, into a scene
    /// where what it was painted inside differs: `rebase` maps the transition
    /// the range was painted inside (`0` for none) to the one it is replayed
    /// inside. Primitives that carried the old one carry the new one, and
    /// transitions pushed inside the range are parented to it, so content
    /// replayed into a transition that started since moves with it rather
    /// than with a copy of the one it was painted in.
    ///
    /// With `moved`, the range is replayed moved, as a view drawn again
    /// elsewhere is: see [`SceneMove`].
    pub(crate) fn replay_inside(
        &mut self,
        range: Range<usize>,
        prev_scene: &Scene,
        rebase: Option<(u32, u32)>,
        moved: Option<&SceneMove>,
        inherited_clip: Option<(usize, u32)>,
    ) {
        // Every glyph of one shimmering label names the same sweep, so the
        // label's glyphs share one remapped entry rather than one each.
        let mut remapped_clips = Vec::new();
        let mut remapped_animation = (0, 0);
        let mut remapped_transitions: Vec<(u32, u32)> = Vec::new();
        // Only content drawn again from a view's record is compacted; a
        // cached view's replay stays as it was.
        let landed_before = rebase.map(|_| std::time::Instant::now());
        let rebase = rebase.filter(|(from, to)| from != to);
        remapped_transitions.extend(rebase);
        // Layers moved out from under the mask around them are left out, as
        // painting them there would have; so is the end of each.
        let mut layers_left_out: Vec<bool> = Vec::new();
        for operation in &prev_scene.paint_operations[range] {
            match operation {
                PaintOperation::Primitive(_)
                | PaintOperation::MonochromeSprite(_)
                | PaintOperation::SubpixelSprite(_) => {
                    let mut primitive = match operation {
                        PaintOperation::MonochromeSprite(index) => Primitive::MonochromeSprite(
                            prev_scene.painted_monochrome_sprites[*index as usize],
                        ),
                        PaintOperation::SubpixelSprite(index) => Primitive::SubpixelSprite(
                            prev_scene.painted_subpixel_sprites[*index as usize],
                        ),
                        PaintOperation::Primitive(primitive) => primitive.clone(),
                        PaintOperation::BackdropBlur(_)
                        | PaintOperation::StartLayer(_)
                        | PaintOperation::EndLayer => continue,
                    };
                    let clip = self.remap_rounded_clip(
                        prev_scene, primitive.rounded_clip(), &mut remapped_clips, moved,
                        &mut remapped_transitions, landed_before, inherited_clip,
                    );
                    primitive.set_rounded_clip(clip);
                    if let Some(moved) = moved {
                        moved.move_primitive(&mut primitive);
                    }
                    let transition = primitive_transition(&primitive);
                    if let Some((from, to)) = rebase
                        && transition == from
                    {
                        set_primitive_transition(&mut primitive, to);
                    } else if transition != 0 {
                        let remapped = self.remap_transition(
                            prev_scene,
                            transition,
                            &mut remapped_transitions,
                            landed_before,
                        );
                        set_primitive_transition(&mut primitive, remapped);
                    }
                    if let Primitive::Quad(quad) = &mut primitive
                        && quad.background.time_animation() != 0
                    {
                        let animation = self.push_quad_animation(
                            prev_scene.quad_animations
                                [quad.background.time_animation() as usize - 1],
                        );
                        quad.background = quad.background.with_time_animation(animation);
                    }
                    if let Primitive::MonochromeSprite(MonochromeSprite { effect, .. })
                    | Primitive::SubpixelSprite(SubpixelSprite { effect, .. }) = &mut primitive
                        && effect.animation != 0
                    {
                        if remapped_animation.0 != effect.animation {
                            let remapped = if effect.animation & ROTATION_ANIMATION != 0 {
                                self.push_rotation_animation(
                                    prev_scene.rotation_animations
                                        [(effect.animation & !ROTATION_ANIMATION) as usize - 1],
                                )
                            } else {
                                self.push_shimmer_animation(
                                    prev_scene.shimmer_animations[effect.animation as usize - 1],
                                )
                            };
                            remapped_animation = (effect.animation, remapped);
                        }
                        effect.animation = remapped_animation.1;
                    }
                    self.insert_primitive(primitive)
                }
                PaintOperation::BackdropBlur(blur) => {
                    let mut blur = *blur;
                    if let Some(moved) = moved {
                        blur.bounds = moved.bounds(blur.bounds);
                        blur.content_mask = moved.mask(&blur.content_mask);
                    }
                    self.insert_backdrop_blur(blur);
                }
                PaintOperation::StartLayer(layer) => match moved {
                    Some(moved) => {
                        // Clipped again by the mask it is in now, as painting
                        // it there would.
                        let extent = moved.bounds(layer.extent);
                        let mask = moved.mask(&ContentMask { bounds: layer.mask }).bounds;
                        let clipped = extent.intersect(&mask);
                        let left_out = clipped.is_empty();
                        if !left_out {
                            let bounds = Bounds::from_corners(
                                point(
                                    ScaledPixels(clipped.origin.x.0.floor()),
                                    ScaledPixels(clipped.origin.y.0.floor()),
                                ),
                                point(
                                    ScaledPixels(clipped.bottom_right().x.0.ceil()),
                                    ScaledPixels(clipped.bottom_right().y.0.ceil()),
                                ),
                            );
                            self.push_clipped_layer(bounds, extent, mask);
                        }
                        layers_left_out.push(left_out);
                    }
                    None => {
                        self.push_clipped_layer(layer.bounds, layer.extent, layer.mask);
                        layers_left_out.push(false);
                    }
                },
                PaintOperation::EndLayer => {
                    if !layers_left_out.pop().unwrap_or(false) {
                        self.pop_layer();
                    }
                }
            }
        }
    }

    pub fn finish(&mut self) {
        let scratch = &mut self.draw_order_scratch;
        scratch.sort(&mut self.shadows, |shadow| shadow.order);
        scratch.sort(&mut self.quads, |quad| quad.order);
        scratch.sort(&mut self.paths, |path| path.order);
        scratch.sort(&mut self.underlines, |underline| underline.order);
        scratch.sort(&mut self.surfaces, |surface| surface.order);
        // A batch draws from one atlas texture, and tile ids restart at zero
        // in every texture, so sprites of one order group by texture first.
        // Sprites of one order were already drawn in tile order rather than
        // the order they were painted in (a layer, such as a line of text,
        // gives everything in it one order), so grouping by texture does not
        // reorder anything that was in paint order before.
        scratch.sort_sprites_into(
            &self.painted_monochrome_sprites,
            &mut self.monochrome_sprites,
            |sprite| (sprite.order, sprite.tile.texture_id.index, sprite.tile.tile_id.0),
        );
        scratch.sort_sprites_into(
            &self.painted_subpixel_sprites,
            &mut self.subpixel_sprites,
            |sprite| (sprite.order, sprite.tile.texture_id.index, sprite.tile.tile_id.0),
        );
        scratch.sort_sprites(&mut self.polychrome_sprites, |sprite| {
            (sprite.order, sprite.tile.texture_id.index, sprite.tile.tile_id.0)
        });
        self.backdrop_blurs.sort_by_key(|blur| blur.order);
        self.partition_quads();
        self.animated_monochrome_sprites.clear();
        self.rotating_monochrome_sprites.clear();
        for (index, sprite) in self.monochrome_sprites.iter().enumerate() {
            if sprite.effect.animation & ROTATION_ANIMATION != 0 {
                self.rotating_monochrome_sprites
                    .push((index as u32, sprite.transformation, sprite.bounds.origin));
            } else if sprite.effect.animation != 0 {
                self.animated_monochrome_sprites.push(index as u32);
            }
        }
        self.animated_subpixel_sprites.clear();
        self.animated_subpixel_sprites.extend(
            (0..self.subpixel_sprites.len() as u32)
                .filter(|&index| self.subpixel_sprites[index as usize].effect.animation != 0),
        );
        self.animated_quads.clear();
        self.offset_quads.clear();
        for (index, quad) in self.quads.iter().enumerate() {
            let animation = quad.background.time_animation();
            if animation == 0 {
                continue;
            }
            self.animated_quads.push(index as u32);
            if self
                .quad_animations
                .get(animation as usize - 1)
                .is_some_and(|animation| animation.offset.is_some())
            {
                self.offset_quads.push(OffsetQuad {
                    index: index as u32,
                    painted_origin: quad.bounds.origin,
                    rest_origin: quad.bounds.origin,
                });
            }
        }
        self.gather_transitioned();
        self.content_id = NEXT_SCENE_CONTENT_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.last_advanced_at = None;
        self.transitions_unmoved = false;
    }

    fn gather_transitioned(&mut self) {
        self.transitioned.clear();
        self.transitions_end_at = None;
        if self.transitions.is_empty() {
            return;
        }
        let transitioned = &mut self.transitioned;
        let mut push = |transition: u32, target: TransitionTarget| {
            if transition != 0 {
                transitioned.push(TransitionedPrimitive { transition, target });
            }
        };
        for (index, sprite) in self.monochrome_sprites.iter().enumerate() {
            push(sprite.pad, TransitionTarget::MonochromeSprite {
                index: index as u32,
                origin: sprite.bounds.origin,
                alpha: sprite.color.a,
            });
        }
        for (index, sprite) in self.subpixel_sprites.iter().enumerate() {
            push(sprite.pad, TransitionTarget::SubpixelSprite {
                index: index as u32,
                origin: sprite.bounds.origin,
                alpha: sprite.color.a,
            });
        }
        for (index, sprite) in self.polychrome_sprites.iter().enumerate() {
            push(sprite.pad, TransitionTarget::PolychromeSprite {
                index: index as u32,
                origin: sprite.bounds.origin,
                opacity: sprite.opacity,
            });
        }
        for (index, underline) in self.underlines.iter().enumerate() {
            push(underline.pad, TransitionTarget::Underline {
                index: index as u32,
                origin: underline.bounds.origin,
                alpha: underline.color.a,
            });
        }
        for (index, shadow) in self.shadows.iter().enumerate() {
            push(shadow.pad, TransitionTarget::Shadow {
                index: index as u32,
                origin: shadow.bounds.origin,
                element_origin: shadow.element_bounds.origin,
                alpha: shadow.color.a,
            });
        }
        for (index, quad) in self.quads.iter().enumerate() {
            push(quad.background.time_transition(), TransitionTarget::Quad {
                index: index as u32,
                origin: quad.bounds.origin,
                background: quad.background,
                border_color: quad.border_color,
            });
        }
        self.transitions_end_at = self
            .transitions
            .iter()
            .map(|transition| transition.transition.ends_at())
            .max();
    }

    /// Whether a transition this scene carries has yet to land, so presenting
    /// it again shows something new every frame rather than at the slower
    /// time-animation rate.
    pub fn transitions_in_flight(&self) -> bool {
        !self.transitioned.is_empty()
            && self
                .transitions_end_at
                .is_some_and(|end| std::time::Instant::now() < end)
    }

    /// Whether anything in this finished scene moves with time on its own, so
    /// presenting it again later shows something new.
    pub fn has_time_animations(&self) -> bool {
        !self.animated_monochrome_sprites.is_empty()
            || !self.rotating_monochrome_sprites.is_empty()
            || !self.animated_subpixel_sprites.is_empty()
            || !self.animated_quads.is_empty()
            || self.transitions_in_flight()
    }

    /// Whether a time animation of this finished scene moves something (an
    /// offset cycle), which a window presents at its faster motion rate.
    pub fn has_motion_animations(&self) -> bool {
        !self.offset_quads.is_empty()
    }

    /// Moves every time-driven primitive of this finished scene to where it is
    /// now. Called before each present, so a frame replayed from a cached view
    /// shows the current phase rather than the one it was painted at.
    pub fn advance_time_animations(&mut self) {
        let now = std::time::Instant::now();
        self.transitions_unmoved = self.transitions.is_empty()
            || self
                .transitions_end_at
                .zip(self.last_advanced_at)
                .is_some_and(|(end, last_advanced_at)| last_advanced_at >= end);
        self.last_advanced_at = Some(now);
        if !self.has_time_animations() && self.transitioned.is_empty() {
            return;
        }
        let animations = &self.shimmer_animations;
        let mut current = (0, 0.0);
        let mut band_origin = |animation: u32| {
            if current.0 != animation {
                current = (animation, animations[animation as usize - 1].band_origin());
            }
            current.1
        };
        for &index in &self.animated_monochrome_sprites {
            let effect = &mut self.monochrome_sprites[index as usize].effect;
            effect.band_origin = band_origin(effect.animation);
        }
        for &index in &self.animated_subpixel_sprites {
            let effect = &mut self.subpixel_sprites[index as usize].effect;
            effect.band_origin = band_origin(effect.animation);
        }
        for &index in &self.animated_quads {
            let quad = &mut self.quads[index as usize];
            let animation_id = quad.background.time_animation();
            let animation = &self.quad_animations[animation_id as usize - 1];
            // An offset-only quad is reset too, at full opacity: a transition
            // fading it starts from the colors reset here, and would compound
            // its fade on every present otherwise.
            let opacity = animation.opacity.map_or(1.0, |cycle| cycle.current_opacity());
            quad.background = animation
                .background
                .opacity(opacity)
                .with_time_animation(animation_id);
            quad.border_color = animation.border_color.opacity(opacity);
        }
        // Back to where they were painted before any transition moves them,
        // so the cycle's offset below lands on top of the transition's.
        for offset_quad in &self.offset_quads {
            self.quads[offset_quad.index as usize].bounds.origin = offset_quad.painted_origin;
        }
        self.advance_transitions(now);
        for offset_quad in &mut self.offset_quads {
            let quad = &mut self.quads[offset_quad.index as usize];
            offset_quad.rest_origin = quad.bounds.origin;
            let animation_id = quad.background.time_animation() as usize;
            let Some(offset) = self
                .quad_animations
                .get(animation_id.wrapping_sub(1))
                .and_then(|animation| animation.offset)
            else {
                continue;
            };
            quad.bounds.origin = quad.bounds.origin + offset.current_displacement();
        }
        // After the transitions, which move a sprite's bounds and with them
        // the centre it turns about.
        for &(index, painted, painted_origin) in &self.rotating_monochrome_sprites {
            let sprite = &mut self.monochrome_sprites[index as usize];
            let animation_id = (sprite.effect.animation & !ROTATION_ANIMATION) as usize;
            let Some(animation) = self.rotation_animations.get(animation_id.wrapping_sub(1)) else {
                continue;
            };
            sprite.transformation = turned(
                painted,
                sprite.bounds.origin - painted_origin,
                sprite.bounds.center(),
                animation.current_angle(),
            );
        }
    }

    /// Which finished frame this scene holds; see [`Self::content_id`]'s
    /// field. Advancing time animations keeps it.
    pub fn content_id(&self) -> u64 {
        self.content_id
    }

    /// Where the latest [`Self::advance_time_animations`] can have changed
    /// what this scene draws, as device-pixel rectangles inside `viewport`
    /// that do not overlap. A renderer still showing this content from its
    /// previous present needs to draw only those and can copy the rest of
    /// the frame over, which for a shimmering label or a pulsing spinner is a
    /// sliver of the window instead of all of it.
    ///
    /// A rectangle that reaches a backdrop blur covers all of it, so the blur
    /// can frost it again from what the rectangle redraws beneath it.
    ///
    /// `false` when the change cannot be bounded that way: no advance has
    /// run on this content yet, a transition was still moving, a moving
    /// sprite is transformed, or the rectangles cover so much of the viewport
    /// that drawing everything costs about the same.
    pub fn time_animation_damage(
        &self,
        viewport: Size<DevicePixels>,
        damage: &mut Vec<Bounds<DevicePixels>>,
    ) -> bool {
        damage.clear();
        if self.last_advanced_at.is_none() || !self.transitions_unmoved {
            return false;
        }
        let mut rects = Vec::new();
        for sprite in self
            .animated_monochrome_sprites
            .iter()
            .map(|&index| &self.monochrome_sprites[index as usize])
        {
            if sprite.transformation != TransformationMatrix::unit() {
                return false;
            }
            rects.extend(DamageRect::covering(&sprite.bounds, &sprite.content_mask, viewport));
        }
        for sprite in self
            .animated_subpixel_sprites
            .iter()
            .map(|&index| &self.subpixel_sprites[index as usize])
        {
            if sprite.transformation != TransformationMatrix::unit() {
                return false;
            }
            rects.extend(DamageRect::covering(&sprite.bounds, &sprite.content_mask, viewport));
        }
        for &(index, painted, _) in &self.rotating_monochrome_sprites {
            // Turning about its centre, the sprite stays inside the square
            // its corners sweep.
            if painted != TransformationMatrix::unit() {
                return false;
            }
            let sprite = &self.monochrome_sprites[index as usize];
            rects.extend(DamageRect::covering(
                &swept_by_turning(&sprite.bounds, painted),
                &sprite.content_mask,
                viewport,
            ));
        }
        for &index in &self.animated_quads {
            let quad = &self.quads[index as usize];
            if self.quad_offset(quad).is_none() {
                rects.extend(DamageRect::covering(&quad.bounds, &quad.content_mask, viewport));
            }
        }
        // From where the latest advance put each one at rest, so a landed
        // transition's offset moves the span along with the quad.
        for offset_quad in &self.offset_quads {
            let quad = &self.quads[offset_quad.index as usize];
            if let Some(offset) = self.quad_offset(quad) {
                let swept = offset.swept(&quad.bounds, Some(offset_quad.rest_origin));
                rects.extend(DamageRect::covering(&swept, &quad.content_mask, viewport));
            }
        }
        let mut merged = DamageRect::merge(rects);
        // A blur frosts what is beneath it within its bounds, so damage that
        // reaches one takes all of it in: the rectangle then holds everything
        // the blur samples, and the blur can run again inside it.
        let frosted: Vec<DamageRect> = self
            .backdrop_blurs
            .iter()
            .map(|blur| DamageRect::from_scaled(&blur.bounds.intersect(&blur.content_mask.bounds)))
            .collect();
        loop {
            let mut grown = false;
            for rect in &mut merged {
                for blur in &frosted {
                    let covering = rect.union(blur);
                    if rect.intersects(blur) && covering != *rect {
                        *rect = covering;
                        grown = true;
                    }
                }
            }
            if !grown {
                break;
            }
            merged = DamageRect::merge(merged);
        }
        for rect in &mut merged {
            *rect = rect.clamped_to(viewport);
        }
        let viewport_area = viewport.width.0.max(0) as i64 * viewport.height.0.max(0) as i64;
        let damaged_area: i64 = merged.iter().map(DamageRect::area).sum();
        if damaged_area * MAX_DAMAGE_SHARE_DENOMINATOR > viewport_area {
            return false;
        }
        damage.extend(merged.iter().map(DamageRect::to_bounds));
        true
    }

    /// The offset cycle `quad` moves with, if any.
    fn quad_offset(&self, quad: &Quad) -> Option<ScaledOffsetCycle> {
        let animation = quad.background.time_animation() as usize;
        self.quad_animations
            .get(animation.wrapping_sub(1))
            .and_then(|animation| animation.offset)
    }

    /// Where `quad`, as painted, can draw: its rectangle, or for one that an
    /// offset cycle moves, every rectangle it passes through.
    fn quad_extent(&self, quad: &Quad) -> Bounds<ScaledPixels> {
        match self.quad_offset(quad) {
            Some(offset) => offset.swept(&quad.bounds, Some(quad.bounds.origin)),
            None => quad.bounds,
        }
    }

    fn advance_transitions(&mut self, now: std::time::Instant) {
        if self.transitioned.is_empty() && self.rounded_clips.iter().all(|clip| clip.transition == 0) {
            return;
        }
        // Parents are always pushed before the transitions inside them, so
        // one pass in order composes every chain.
        self.transition_states.clear();
        for transition in &self.transitions {
            let own = (transition.offset_at(now), transition.transition.opacity_at(now));
            let composed = match transition.parent {
                0 => own,
                parent => {
                    let (offset, opacity) = self.transition_states[parent as usize - 1];
                    (own.0 + offset, own.1 * opacity)
                }
            };
            self.transition_states.push(composed);
        }
        for clip in &mut self.rounded_clips {
            if clip.transition != 0 {
                clip.bounds.origin = clip.base_origin + self.transition_states[clip.transition as usize - 1].0;
            }
        }
        for primitive in &self.transitioned {
            let (offset, opacity) = self.transition_states[primitive.transition as usize - 1];
            match primitive.target {
                TransitionTarget::MonochromeSprite { index, origin, alpha } => {
                    let sprite = &mut self.monochrome_sprites[index as usize];
                    sprite.bounds.origin = origin + offset;
                    sprite.color.a = alpha * opacity;
                }
                TransitionTarget::SubpixelSprite { index, origin, alpha } => {
                    let sprite = &mut self.subpixel_sprites[index as usize];
                    sprite.bounds.origin = origin + offset;
                    sprite.color.a = alpha * opacity;
                }
                TransitionTarget::PolychromeSprite { index, origin, opacity: painted } => {
                    let sprite = &mut self.polychrome_sprites[index as usize];
                    sprite.bounds.origin = origin + offset;
                    sprite.opacity = painted * opacity;
                }
                TransitionTarget::Underline { index, origin, alpha } => {
                    let underline = &mut self.underlines[index as usize];
                    underline.bounds.origin = origin + offset;
                    underline.color.a = alpha * opacity;
                }
                TransitionTarget::Shadow { index, origin, element_origin, alpha } => {
                    let shadow = &mut self.shadows[index as usize];
                    shadow.bounds.origin = origin + offset;
                    shadow.element_bounds.origin = element_origin + offset;
                    shadow.color.a = alpha * opacity;
                }
                TransitionTarget::Quad { index, origin, background, border_color } => {
                    let quad = &mut self.quads[index as usize];
                    quad.bounds.origin = origin + offset;
                    // An opacity cycle has just reset this quad's colors from
                    // its own rest state; anything else starts from the colors
                    // it was painted with, so neither compounds per present.
                    let (base_background, base_border) = if quad.background.time_animation() != 0 {
                        (quad.background, quad.border_color)
                    } else {
                        (background, border_color)
                    };
                    quad.background = base_background
                        .opacity(opacity)
                        .with_time_transition(primitive.transition);
                    quad.border_color = base_border.opacity(opacity);
                }
            }
        }
    }

    fn partition_quads(&mut self) {
        self.blended_quad_indices.clear();
        self.opaque_quad_indices.clear();
        // The depth prepass draws every opaque quad before anything else. A
        // quad above a backdrop blur would then be in the framebuffer when the
        // blur samples it, so only quads beneath the first blur take the
        // prepass; everything from the first blur up paints back to front.
        // Frosted surfaces sit near the top of the window, so this keeps the
        // prepass for nearly every opaque surface a frosted frame paints:
        // without it, each surface stacked under a frosted composer is shaded
        // in full on every frame.
        let partitioning_enabled = opaque_quad_partitioning_enabled()
            && self.quads.len() <= MAX_DEPTH_PARTITIONED_QUADS;
        let prepass_below = self
            .backdrop_blurs
            .first()
            .map_or(DrawOrder::MAX, |blur| blur.order);
        for (quad_id, quad) in self.quads.iter().enumerate() {
            // A quad whose opacity animates may be solid in this frame and
            // translucent in the next without the scene being rebuilt, so it
            // always takes the blended pass; so does one a transition moves
            // or fades.
            let has_opaque_core = partitioning_enabled
                && quad.order < prepass_below
                && quad.background.time_animation() == 0
                && quad.background.time_transition() == 0
                && quad.has_opaque_core();
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
            backdrop_blurs_iter: self.backdrop_blurs.iter().peekable(),
        }
    }
}

/// Room to put a finished scene in draw order, kept from frame to frame.
///
/// Primitives are 100 to 170 bytes, and a comparison sort moves them as often
/// as it compares them, through a scratch buffer a stable sort allocates on
/// every call. Sorting packed integer keys that carry each primitive's index,
/// then permuting the primitives in place along the cycles of that order,
/// costs one swap per primitive and allocates nothing once the keys have
/// room. A kind already in order, as a static frame's often is, costs a scan.
#[derive(Default)]
struct DrawOrderScratch {
    keys: Vec<u64>,
    sprite_keys: Vec<u128>,
    radix_scratch: Vec<u64>,
    permutation: Vec<u32>,
}

impl DrawOrderScratch {
    /// Sorts `items` by `key`, keeping equal keys in the order they came in.
    fn sort<T>(&mut self, items: &mut [T], key: impl Fn(&T) -> u32) {
        if items.is_sorted_by_key(&key) {
            return;
        }
        self.keys.clear();
        self.keys.extend(
            items
                .iter()
                .enumerate()
                .map(|(index, item)| (u64::from(key(item)) << 32) | index as u64),
        );
        self.keys.sort_unstable();
        self.permutation.clear();
        self.permutation.extend(self.keys.iter().map(|&key| key as u32));
        permute_in_place(items, &mut self.permutation);
    }

    /// Sorts sprites by `(order, texture, tile)`, keeping equal keys in the
    /// order they came in.
    fn sort_sprites<T>(&mut self, items: &mut [T], key: impl Fn(&T) -> (u32, u32, u32)) {
        if items.is_sorted_by_key(&key) {
            return;
        }
        self.sort_sprite_indices(items, key);
        permute_in_place(items, &mut self.permutation);
    }

    /// Replaces `sorted` with `items` sorted as [`Self::sort_sprites`] sorts
    /// them, copying each sprite once rather than swapping it into place.
    fn sort_sprites_into<T: Copy>(
        &mut self,
        items: &[T],
        sorted: &mut Vec<T>,
        key: impl Fn(&T) -> (u32, u32, u32),
    ) {
        sorted.clear();
        if items.is_sorted_by_key(&key) {
            sorted.extend_from_slice(items);
            return;
        }
        self.sort_sprite_indices(items, key);
        sorted.extend(self.permutation.iter().map(|&index| items[index as usize]));
    }

    /// Leaves the indices of `items` in `permutation`, sorted as
    /// [`Self::sort_sprites`] sorts them.
    fn sort_sprite_indices<T>(&mut self, items: &[T], key: impl Fn(&T) -> (u32, u32, u32)) {
        let (mut orders, mut textures, mut tiles) = (0, 0, 0);
        for item in items {
            let (order, texture, tile) = key(item);
            orders |= order;
            textures |= texture;
            tiles |= tile;
        }
        let width = |bits: u32| u32::BITS - bits.leading_zeros();
        let tile_shift = usize::BITS - items.len().leading_zeros();
        let texture_shift = tile_shift + width(tiles);
        let order_shift = texture_shift + width(textures);
        let key_width = order_shift + width(orders);
        self.permutation.clear();
        // Each part packed only as wide as this frame needs usually fits all
        // four in 64 bits, which sort in a few linear passes instead of a
        // comparison sort of 128-bit keys.
        if key_width < u64::BITS {
            self.keys.clear();
            self.keys.extend(items.iter().enumerate().map(|(index, item)| {
                let (order, texture, tile) = key(item);
                (u64::from(order) << order_shift)
                    | (u64::from(texture) << texture_shift)
                    | (u64::from(tile) << tile_shift)
                    | index as u64
            }));
            // The keys start in index order and the sort is stable, so the
            // index bits need no pass of their own.
            radix_sort(&mut self.keys, &mut self.radix_scratch, tile_shift..key_width);
            let index_mask = (1 << tile_shift) - 1;
            self.permutation
                .extend(self.keys.iter().map(|&key| (key & index_mask) as u32));
        } else {
            self.sprite_keys.clear();
            self.sprite_keys.extend(items.iter().enumerate().map(|(index, item)| {
                let (order, texture, tile) = key(item);
                (u128::from(order) << 96)
                    | (u128::from(texture) << 64)
                    | (u128::from(tile) << 32)
                    | index as u128
            }));
            self.sprite_keys.sort_unstable();
            self.permutation
                .extend(self.sprite_keys.iter().map(|&key| key as u32));
        }
    }

    fn shrink_idle(&mut self, largest_primitive_count: usize) {
        let capacity = largest_primitive_count
            .saturating_mul(2)
            .max(crate::util::MIN_RETAINED_CAPACITY);
        self.keys.shrink_to(capacity);
        self.sprite_keys.shrink_to(capacity);
        self.radix_scratch.shrink_to(capacity);
        self.permutation.shrink_to(capacity);
    }
}

const RADIX_DIGIT_BITS_MAX: u32 = 11;

/// Sorts `keys` stably by their bits in `bits`, a digit at a time, through
/// `scratch`.
///
/// Each pass costs two sweeps of the keys plus one of the digit's buckets, so
/// digits are as wide as there are keys to share the buckets among, up to
/// [`RADIX_DIGIT_BITS_MAX`], and the bits are split evenly between as few
/// passes as that allows: a frame's few thousand sprites sort in three passes
/// rather than the four a byte at a time takes.
fn radix_sort(keys: &mut Vec<u64>, scratch: &mut Vec<u64>, bits: Range<u32>) {
    let width = bits.end.saturating_sub(bits.start);
    if width == 0 {
        return;
    }
    let widest_digit =
        (usize::BITS - keys.len().leading_zeros()).clamp(u8::BITS, RADIX_DIGIT_BITS_MAX);
    let passes = width.div_ceil(widest_digit);
    let digit_bits = width.div_ceil(passes);
    let mask = (1u64 << digit_bits) - 1;
    scratch.clear();
    scratch.resize(keys.len(), 0);
    let mut buckets = [0usize; 1 << RADIX_DIGIT_BITS_MAX];
    let offsets = &mut buckets[..1 << digit_bits];
    for pass in 0..passes {
        let shift = bits.start + pass * digit_bits;
        offsets.fill(0);
        for &key in keys.iter() {
            offsets[((key >> shift) & mask) as usize] += 1;
        }
        // A digit every key shares, such as the texture of a frame drawn
        // from one atlas page, moves nothing.
        if offsets.contains(&keys.len()) {
            continue;
        }
        let mut offset = 0;
        for slot in offsets.iter_mut() {
            offset += std::mem::replace(slot, offset);
        }
        for &key in keys.iter() {
            let slot = &mut offsets[((key >> shift) & mask) as usize];
            scratch[*slot] = key;
            *slot += 1;
        }
        std::mem::swap(keys, scratch);
    }
}

fn append_replayed_sprites<T: Copy>(
    painted: &mut Vec<T>,
    sources: &[T],
    mut stamp: impl FnMut(&mut T),
) -> u32 {
    let start = painted.len();
    painted.reserve(sources.len());
    for sprite in sources {
        // Stamped where it lands rather than on the way: changing a copy in
        // between keeps the compiler from copying the sprite straight into
        // place, costing a second copy of every glyph.
        painted.push(*sprite);
        if let Some(pushed) = painted.last_mut() {
            stamp(pushed);
        }
    }
    start as u32
}

/// Rearranges `items` so position `i` holds what was at `sources[i]`,
/// following each cycle of the permutation with swaps. `sources` must be a
/// permutation of `0..items.len()`; it is left as the identity.
fn permute_in_place<T>(items: &mut [T], sources: &mut [u32]) {
    debug_assert_eq!(items.len(), sources.len());
    for start in 0..sources.len() {
        let mut position = start;
        loop {
            let source = sources[position] as usize;
            sources[position] = position as u32;
            if source == start {
                break;
            }
            items.swap(position, source);
            position = source;
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

static NEXT_SCENE_CONTENT_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// Past this many rectangles, [`Scene::time_animation_damage`] draws the one
/// bounding them all: every rectangle repeats the frame's draw calls.
const MAX_DAMAGE_RECTS: usize = 8;
/// Rectangles nearer than this, in device pixels, are drawn as one, so a line
/// of shimmering glyphs is one rectangle rather than one per glyph.
const DAMAGE_MERGE_GAP: i32 = 16;
/// Damage over more than one part in this many of the viewport draws the
/// whole frame: the copy of the rest is no longer the cheaper half.
const MAX_DAMAGE_SHARE_DENOMINATOR: i64 = 2;

/// A device-pixel rectangle of [`Scene::time_animation_damage`], half-open.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct DamageRect {
    left: i32,
    top: i32,
    right: i32,
    bottom: i32,
}

impl DamageRect {
    fn from_scaled(bounds: &Bounds<ScaledPixels>) -> Self {
        // A pixel of slack on each side covers antialiasing that a
        // primitive's coverage spreads past its bounds.
        Self {
            left: bounds.left().0.floor() as i32 - 1,
            top: bounds.top().0.floor() as i32 - 1,
            right: bounds.right().0.ceil() as i32 + 1,
            bottom: bounds.bottom().0.ceil() as i32 + 1,
        }
    }

    /// What of `bounds` its content mask lets onto the viewport, if any.
    fn covering(
        bounds: &Bounds<ScaledPixels>,
        mask: &ContentMask<ScaledPixels>,
        viewport: Size<DevicePixels>,
    ) -> Option<Self> {
        let rect = Self::from_scaled(&bounds.intersect(&mask.bounds)).clamped_to(viewport);
        (rect.left < rect.right && rect.top < rect.bottom).then_some(rect)
    }

    fn clamped_to(&self, viewport: Size<DevicePixels>) -> Self {
        Self {
            left: self.left.max(0),
            top: self.top.max(0),
            right: self.right.min(viewport.width.0),
            bottom: self.bottom.min(viewport.height.0),
        }
    }

    fn intersects(&self, other: &Self) -> bool {
        self.left < other.right
            && other.left < self.right
            && self.top < other.bottom
            && other.top < self.bottom
    }

    fn near(&self, other: &Self) -> bool {
        self.left - DAMAGE_MERGE_GAP < other.right
            && other.left - DAMAGE_MERGE_GAP < self.right
            && self.top - DAMAGE_MERGE_GAP < other.bottom
            && other.top - DAMAGE_MERGE_GAP < self.bottom
    }

    fn union(&self, other: &Self) -> Self {
        Self {
            left: self.left.min(other.left),
            top: self.top.min(other.top),
            right: self.right.max(other.right),
            bottom: self.bottom.max(other.bottom),
        }
    }

    fn area(&self) -> i64 {
        (self.right - self.left) as i64 * (self.bottom - self.top) as i64
    }

    fn to_bounds(&self) -> Bounds<DevicePixels> {
        Bounds {
            origin: point(DevicePixels(self.left), DevicePixels(self.top)),
            size: Size {
                width: DevicePixels(self.right - self.left),
                height: DevicePixels(self.bottom - self.top),
            },
        }
    }

    /// Folds `rects` into rectangles no two of which are near each other,
    /// and so never overlap, at most [`MAX_DAMAGE_RECTS`] of them.
    fn merge(rects: Vec<Self>) -> Vec<Self> {
        let mut merged: Vec<Self> = Vec::with_capacity(rects.len().min(MAX_DAMAGE_RECTS + 1));
        for mut rect in rects {
            // A union can reach rectangles neither half was near, so keep
            // folding until it stands alone.
            while let Some(index) = merged.iter().position(|other| other.near(&rect)) {
                rect = rect.union(&merged.swap_remove(index));
            }
            merged.push(rect);
        }
        if merged.len() > MAX_DAMAGE_RECTS
            && let Some(bounding) = merged.iter().copied().reduce(|bounding, rect| bounding.union(&rect))
        {
            merged.clear();
            merged.push(bounding);
        }
        merged
    }
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
    use crate::px;
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

    fn unit_bounds() -> Bounds<ScaledPixels> {
        Bounds {
            origin: Point::default(),
            size: Size {
                width: ScaledPixels::from(10.),
                height: ScaledPixels::from(10.),
            },
        }
    }

    fn opaque_quad() -> Quad {
        Quad {
            bounds: unit_bounds(),
            content_mask: ContentMask {
                bounds: unit_bounds(),
            },
            background: Background::from(Hsla::black()),
            ..Default::default()
        }
    }

    fn unit_backdrop_blur() -> BackdropBlur {
        BackdropBlur {
            order: 0,
            blur_radius: ScaledPixels(8.),
            saturation: 1.,
            opacity: 1.,
            bounds: unit_bounds(),
            content_mask: ContentMask {
                bounds: unit_bounds(),
            },
            corner_radii: Corners::default(),
        }
    }

    /// A frosted card: the quad under it, then a layer holding the blur and
    /// the card's own fill.
    fn frosted_card_scene() -> Scene {
        let mut scene = Scene::default();
        scene.insert_primitive(opaque_quad());
        scene.push_layer(unit_bounds());
        scene.insert_backdrop_blur(unit_backdrop_blur());
        scene.insert_primitive(opaque_quad());
        scene.pop_layer();
        scene
    }

    #[test]
    fn a_backdrop_blur_ends_the_batch_beneath_it() {
        let mut scene = frosted_card_scene();
        scene.finish();

        let blur_order = scene.backdrop_blurs[0].order;
        let orders: Vec<_> = scene
            .batches()
            .map(|batch| scene.batch_order(&batch))
            .collect();
        assert_eq!(orders.len(), 2, "one quad batch either side of the blur");
        assert!(orders[0] < blur_order);
        assert_eq!(orders[1], blur_order);
    }

    #[test]
    fn only_quads_beneath_the_first_backdrop_blur_take_the_opaque_prepass() {
        let mut scene = frosted_card_scene();
        scene.finish();
        let blur_order = scene.backdrop_blurs[0].order;
        let prepassed: Vec<_> = scene
            .opaque_quad_indices
            .iter()
            .map(|&index| scene.quads[index as usize].order)
            .collect();
        assert_eq!(prepassed.len(), 1, "the quad under the frosted card");
        assert!(prepassed[0] < blur_order);
        let blended: Vec<_> = scene
            .blended_quad_indices
            .iter()
            .map(|&index| scene.quads[index as usize].order)
            .collect();
        assert_eq!(
            blended,
            vec![blur_order],
            "the card's own fill paints over the blur, back to front"
        );
    }

    #[test]
    fn replay_keeps_a_backdrop_blur_under_its_card() {
        let mut painted = frosted_card_scene();
        painted.finish();
        let mut replayed = Scene::default();
        replayed.replay(0..painted.len(), &painted);
        replayed.finish();

        assert_eq!(replayed.backdrop_blurs.len(), 1);
        let blur_order = replayed.backdrop_blurs[0].order;
        assert_eq!(replayed.quads[1].order, blur_order);
        assert!(!replayed.is_empty());
    }

    /// What replaying a glyph sprite keeps (its kind, tile and place) and
    /// what it takes from the scene it is replayed into (draw order and
    /// transition), for each glyph sprite among `operations`.
    fn painted_glyphs(
        scene: &Scene,
        operations: Range<usize>,
    ) -> Vec<(bool, u32, ScaledPixels, DrawOrder, u32)> {
        scene.paint_operations[operations]
            .iter()
            .map(|operation| match *operation {
                PaintOperation::MonochromeSprite(index) => {
                    let sprite = &scene.painted_monochrome_sprites[index as usize];
                    (false, sprite.tile.tile_id.0, sprite.bounds.origin.x, sprite.order, sprite.pad)
                }
                PaintOperation::SubpixelSprite(index) => {
                    let sprite = &scene.painted_subpixel_sprites[index as usize];
                    (true, sprite.tile.tile_id.0, sprite.bounds.origin.x, sprite.order, sprite.pad)
                }
                _ => panic!("only glyph sprites are replayed"),
            })
            .collect()
    }

    /// Replaying a line copies its sprites a run of consecutive indices at a
    /// time. A line whose monochrome and subpixel glyphs interleave, replayed
    /// into a scene whose sprite vectors already hold other glyphs, still
    /// comes back glyph for glyph in paint order, each taking the replaying
    /// layer's draw order and transition.
    #[test]
    fn replayed_glyph_sprites_come_back_in_paint_order() {
        let monochrome = |tile: u32, x: f32| {
            let mut sprite = shimmering_glyph(0);
            sprite.effect = SpriteEffect::default();
            sprite.tile.tile_id = crate::TileId(tile);
            sprite.bounds.origin.x = ScaledPixels::from(x);
            sprite.bounds.size.width = ScaledPixels::from(1.);
            sprite
        };
        let subpixel = |tile: u32, x: f32| {
            let sprite = monochrome(tile, x);
            SubpixelSprite {
                rounded_clip: 0,
                clip_padding: 0,
                order: 0,
                pad: 0,
                bounds: sprite.bounds,
                content_mask: sprite.content_mask,
                color: sprite.color,
                effect: sprite.effect,
                tile: sprite.tile,
                transformation: sprite.transformation,
            }
        };

        let mut source = Scene::default();
        source.insert_monochrome_sprite(monochrome(90, 9.));
        let mut lines = Vec::new();
        for line in [
            &[(false, 1), (false, 2), (true, 3), (true, 4), (false, 5)][..],
            &[(true, 6), (false, 7), (false, 8)][..],
        ] {
            source.push_layer(unit_bounds());
            let start = source.glyph_watermark();
            for &(is_subpixel, tile) in line {
                if is_subpixel {
                    source.insert_subpixel_sprite(subpixel(tile, tile as f32));
                } else {
                    source.insert_monochrome_sprite(monochrome(tile, tile as f32));
                }
            }
            let Some(operations) = source.glyph_sprites_since(start) else {
                panic!("a line of glyphs paints only glyph sprites");
            };
            lines.push(operations);
            source.pop_layer();
        }

        let started_at = std::time::Instant::now();
        let mut replayed = Scene::default();
        replayed.insert_subpixel_sprite(subpixel(91, 9.));
        replayed.insert_monochrome_sprite(monochrome(92, 9.));
        let Some(transition) = replayed.push_transition(rolling_in(started_at)) else {
            panic!("a first transition always fits");
        };
        replayed.set_current_transition(transition);
        // In reverse, so neither line lands at the indices it came from.
        for operations in lines.iter().rev() {
            replayed.push_layer(unit_bounds());
            let order = *replayed.layer_stack.last().unwrap_or(&0);
            let start = replayed.glyph_watermark();
            assert!(replayed.replay_glyph_sprites(&source, operations.clone()));
            let Some(replayed_operations) = replayed.glyph_sprites_since(start) else {
                panic!("replaying paints only glyph sprites");
            };
            let expected: Vec<_> = painted_glyphs(&source, operations.clone())
                .into_iter()
                .map(|(is_subpixel, tile, x, _, _)| (is_subpixel, tile, x, order, transition))
                .collect();
            assert_eq!(painted_glyphs(&replayed, replayed_operations), expected);
            replayed.pop_layer();
        }
        assert_eq!(replayed.painted_monochrome_sprites.len(), 1 + 5);
        assert_eq!(replayed.painted_subpixel_sprites.len(), 1 + 3);
    }

    fn shimmering_glyph(animation: u32) -> MonochromeSprite {
        MonochromeSprite {
            rounded_clip: 0,
            clip_padding: 0,
            order: 0,
            pad: 0,
            bounds: unit_bounds(),
            content_mask: ContentMask {
                bounds: unit_bounds(),
            },
            color: crate::white(),
            effect: SpriteEffect {
                kind: SpriteEffect::SHIMMER_KIND,
                animation,
                ..SpriteEffect::default()
            },
            tile: AtlasTile {
                texture_id: crate::AtlasTextureId {
                    index: 0,
                    kind: crate::AtlasTextureKind::Monochrome,
                },
                tile_id: crate::TileId(0),
                padding: 0,
                bounds: Bounds::default(),
            },
            transformation: TransformationMatrix::unit(),
        }
    }

    /// Sorting keys and permuting in place gives what a stable sort by the
    /// same key gives, for inputs with many equal keys, already sorted runs,
    /// and fixed points.
    #[test]
    fn draw_order_sort_matches_a_stable_sort() {
        use rand::{Rng, SeedableRng};
        let mut scratch = DrawOrderScratch::default();
        for seed in 0..200u64 {
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
            let len = rng.random_range(0..300);
            let distinct_orders = rng.random_range(1..20);
            let items: Vec<(u32, usize)> = (0..len)
                .map(|index| (rng.random_range(0..distinct_orders), index))
                .collect();
            let mut expected = items.clone();
            expected.sort_by_key(|item| item.0);
            let mut sorted = items;
            scratch.sort(&mut sorted, |item| item.0);
            assert_eq!(sorted, expected, "seed {seed}");
            assert!(
                scratch
                    .permutation
                    .iter()
                    .enumerate()
                    .all(|(index, &source)| source as usize == index)
            );
        }
    }

    /// Sprites sort as a stable sort by `(order, texture, tile)` would, both
    /// when the parts pack into 64 bits and when they are too wide to, and
    /// with enough of them that the radix sort takes digits wider than a byte.
    #[test]
    fn sprite_sort_matches_a_stable_sort() {
        use rand::{Rng, SeedableRng};
        let mut scratch = DrawOrderScratch::default();
        for seed in 0..200u64 {
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
            let len = if seed % 5 == 1 {
                rng.random_range(1024..5000)
            } else {
                rng.random_range(0..600)
            };
            let widest = if seed % 4 == 0 { u32::MAX } else { 1 << rng.random_range(0..12) };
            let items: Vec<(u32, u32, u32, usize)> = (0..len)
                .map(|index| {
                    (
                        rng.random_range(0..=widest),
                        rng.random_range(0..3),
                        rng.random_range(0..=widest),
                        index,
                    )
                })
                .collect();
            let mut expected = items.clone();
            expected.sort_by_key(|item| (item.0, item.1, item.2));
            let mut sorted = items.clone();
            scratch.sort_sprites(&mut sorted, |item| (item.0, item.1, item.2));
            assert_eq!(sorted, expected, "seed {seed}");
            let mut copied = Vec::new();
            scratch.sort_sprites_into(&items, &mut copied, |item| (item.0, item.1, item.2));
            assert_eq!(copied, expected, "seed {seed}");
        }
    }

    /// Sprites of one order are grouped by the texture they come from, even
    /// where tile ids of different textures interleave, so each order draws
    /// one batch per texture rather than one per change of texture.
    #[test]
    fn sprites_of_one_order_batch_by_texture() {
        let mut scene = Scene::default();
        let mut sprite = shimmering_glyph(0);
        sprite.effect = SpriteEffect::default();
        // Side by side, so every sprite gets the same order.
        for index in 0..8u32 {
            sprite.bounds.origin.x = ScaledPixels::from(index as f32 * 20.);
            sprite.content_mask.bounds = sprite.bounds;
            sprite.tile.texture_id.index = index % 2;
            sprite.tile.tile_id = crate::TileId(index / 2);
            scene.insert_primitive(sprite);
        }
        scene.finish();
        let textures: Vec<u32> = scene
            .monochrome_sprites
            .iter()
            .map(|sprite| sprite.tile.texture_id.index)
            .collect();
        assert_eq!(textures, [0, 0, 0, 0, 1, 1, 1, 1]);
        let sprite_batches = scene
            .batches()
            .filter(|batch| matches!(batch, PrimitiveBatch::MonochromeSprites { .. }))
            .count();
        assert_eq!(sprite_batches, 2);
    }

    /// A shimmer's band is moved by the scene before each present; a view
    /// cached around it replays its glyphs into the next frame, and they must
    /// still name a sweep there, or the band freezes at the phase it was
    /// painted with.
    #[test]
    fn replayed_shimmer_glyphs_keep_their_sweep() {
        let mut source = Scene::default();
        let animation = source.push_shimmer_animation(ShimmerAnimation {
            band_start: -40.0,
            travel: 1000.0,
            period: std::time::Duration::from_millis(1000),
            hold: 0.0,
        });
        source.insert_primitive(shimmering_glyph(animation));
        source.insert_primitive(shimmering_glyph(animation));
        source.insert_primitive(shimmering_glyph(0));
        source.finish();
        assert!(source.has_time_animations());

        let mut replayed = Scene::default();
        // An unrelated sweep first, so the replayed index has to be remapped.
        replayed.push_shimmer_animation(ShimmerAnimation {
            band_start: 0.0,
            travel: 0.0,
            period: std::time::Duration::from_millis(1),
            hold: 0.0,
        });
        replayed.replay(0..source.len(), &source);
        replayed.finish();
        assert!(replayed.has_time_animations());
        assert_eq!(replayed.shimmer_animations.len(), 2, "one entry per label");

        replayed.advance_time_animations();
        let animated: Vec<_> = replayed
            .monochrome_sprites
            .iter()
            .filter(|sprite| sprite.effect.animation != 0)
            .collect();
        assert_eq!(animated.len(), 2);
        for sprite in animated {
            assert_eq!(sprite.effect.animation, 2);
            assert!((-40.0..=960.0).contains(&sprite.effect.band_origin));
        }
    }

    /// A window with a scene overlay presents the two planes split from its
    /// scene. A present that redraws nothing only moves the scene's drawn
    /// copies, so the planes have to be moved again after the split, or the
    /// shimmer band and the pulse stay where they were painted.
    #[test]
    fn split_planes_show_time_animations_where_they_are_now() {
        let mut source = Scene::default();
        let shimmer = source.push_shimmer_animation(ShimmerAnimation {
            band_start: -40.0,
            travel: 1000.0,
            period: std::time::Duration::from_millis(1000),
            hold: 0.0,
        });
        let mut glyph = shimmering_glyph(shimmer);
        glyph.effect.band_origin = -1000.0;
        source.insert_primitive(glyph);
        let quad = opaque_quad();
        let pulse = source.push_quad_animation(QuadAnimation {
            opacity: Some(OpacityCycle::new(
                std::time::Duration::from_millis(750),
                [(0.0, 0.5), (0.5, 0.5), (0.75, 0.5), (1.0, 0.5)],
            )),
            offset: None,
            background: quad.background,
            border_color: quad.border_color,
        });
        source.insert_primitive(Quad {
            background: quad.background.with_time_animation(pulse),
            ..quad
        });
        source.finish();
        source.advance_time_animations();

        let mut base = Scene::default();
        let mut overlay = Scene::default();
        source.split_planes_into(1, &mut base, &mut overlay);

        assert_eq!(base.monochrome_sprites.len(), 1);
        assert!((-40.0..=960.0).contains(&base.monochrome_sprites[0].effect.band_origin));
        assert_eq!(overlay.quads.len(), 1);
        assert!((overlay.quads[0].background.solid.a - quad.background.solid.a * 0.5).abs() < 1e-6);
    }

    /// What a transition covers moves in both planes too, from where it was
    /// painted: the split copies primitives at rest, so a present that
    /// redraws nothing would otherwise show a label's outgoing and incoming
    /// copies stacked at rest until it settles.
    #[test]
    fn split_planes_show_transitions_where_they_are_now() {
        let duration = std::time::Duration::from_secs(600);
        let mut source = Scene::default();
        let Some(transition) = source.push_transition(SceneTransition {
            transition: TimeTransition::new(std::time::Instant::now() - duration / 2, duration)
                .opacity(1.0, 0.0),
            scale_factor: 2.0,
            parent: 0,
        }) else {
            panic!("a first transition always fits");
        };
        source.set_current_transition(transition);
        source.insert_primitive(shimmering_glyph(0));
        source.insert_primitive(opaque_quad());
        source.set_current_transition(0);
        source.finish();
        source.advance_time_animations();

        let mut base = Scene::default();
        let mut overlay = Scene::default();
        source.split_planes_into(1, &mut base, &mut overlay);

        let faded = |alpha: f32| (0.3..0.7).contains(&alpha);
        assert!(faded(base.monochrome_sprites[0].color.a));
        assert!(overlay.quads[0].background.as_solid().is_some_and(|color| faded(color.a)));
        assert!(base.transitions_in_flight() && overlay.transitions_in_flight());
    }

    fn rolling_in(started_at: std::time::Instant) -> SceneTransition {
        SceneTransition {
            transition: TimeTransition::new(started_at, std::time::Duration::from_millis(100))
                .offset(point(px(0.), px(10.)), Point::default())
                .opacity(0.0, 1.0),
            scale_factor: 2.0,
            parent: 0,
        }
    }

    /// A slide along x with no fade, pushed inside `parent`.
    fn sliding(started_at: std::time::Instant, parent: u32) -> SceneTransition {
        SceneTransition {
            transition: TimeTransition::new(started_at, std::time::Duration::from_millis(100))
                .offset(point(px(-6.), px(0.)), Point::default()),
            scale_factor: 2.0,
            parent,
        }
    }

    #[test]
    fn rounded_clip_move_keeps_ancestor_masks_stationary() {
        let mut source = Scene::default();
        let outer = RoundedClipNode {
            bounds: unit_bounds(),
            corner_radii: Corners::all(ScaledPixels(0.25)),
            ..Default::default()
        };
        source.rounded_clips.push(outer);
        source.rounded_clips.push(RoundedClipNode { parent: 1, ..outer });
        source.current_rounded_clip = 2;
        source.insert_primitive(shimmering_glyph(0));
        let mut replayed = Scene::default();
        replayed.rounded_clips.push(outer);
        replayed.replay_inside(0..source.len(), &source, None, Some(&SceneMove {
            delta: point(ScaledPixels(1.), ScaledPixels(0.)),
            old_outer: [0., 0., 10., 10.],
            new_outer: [0., 0., 10., 10.],
            inherited_clip_depth: 1,
            inherited_clip: 1,
        }), None);
        replayed.finish();
        assert_eq!(replayed.rounded_clips.len(), 2);
        assert_eq!(replayed.rounded_clips[0].bounds.origin.x, ScaledPixels(0.));
        assert_eq!(replayed.rounded_clips[1].bounds.origin.x, ScaledPixels(1.));
        assert_eq!(replayed.rounded_clips[1].parent, 1);
        assert_eq!(replayed.monochrome_sprites[0].rounded_clip, 2);
    }

    #[test]
    fn rounded_clip_cached_subtree_joins_a_new_transition_without_moving_its_ancestor() {
        let started_at = std::time::Instant::now();
        let mut source = Scene::default();
        let outer = RoundedClipNode {
            bounds: unit_bounds(),
            corner_radii: Corners::all(ScaledPixels(0.25)),
            ..Default::default()
        };
        source.rounded_clips.push(outer);
        source.rounded_clips.push(RoundedClipNode { parent: 1, ..outer });
        source.current_rounded_clip = 2;
        source.insert_primitive(shimmering_glyph(0));
        let mut replayed = Scene::default();
        replayed.rounded_clips.push(outer);
        let Some(transition) = replayed.push_transition(rolling_in(started_at)) else {
            panic!("a first transition fits");
        };
        replayed.replay_inside(0..source.len(), &source, Some((0, transition)), None, Some((1, 1)));
        replayed.finish();
        assert_eq!(replayed.rounded_clips.len(), 2);
        assert_eq!(replayed.rounded_clips[0].transition, 0);
        assert_eq!(replayed.rounded_clips[1].transition, transition);
        replayed.advance_transitions(started_at + std::time::Duration::from_millis(50));
        assert_eq!(replayed.rounded_clips[0].bounds.origin.y, ScaledPixels(0.));
        assert_eq!(replayed.rounded_clips[1].bounds.origin.y, ScaledPixels(10.));
        assert_eq!(replayed.monochrome_sprites[0].bounds.origin.y, ScaledPixels(10.));
    }

    #[test]
    fn rounded_clip_replay_remaps_parents_and_moves_only_transition_local_masks() {
        let started_at = std::time::Instant::now();
        let mut source = Scene::default();
        let Some(transition) = source.push_transition(rolling_in(started_at)) else {
            panic!("a first transition fits");
        };
        let outer = RoundedClipNode {
            bounds: Bounds::new(Point::default(), crate::size(ScaledPixels(100.), ScaledPixels(80.))),
            corner_radii: Corners::all(ScaledPixels(20.)),
            ..Default::default()
        };
        source.rounded_clips.push(outer);
        source.rounded_clips.push(RoundedClipNode { parent: 1, transition, ..outer });
        source.current_rounded_clip = 2;
        source.set_current_transition(transition);
        source.insert_primitive(shimmering_glyph(0));
        source.set_current_transition(0);
        source.finish();
        source.advance_transitions(started_at + std::time::Duration::from_millis(50));
        assert_eq!(source.rounded_clips[0].bounds.origin.y, ScaledPixels(0.));
        assert_eq!(source.rounded_clips[1].bounds.origin.y, ScaledPixels(10.));

        let mut replayed = Scene::default();
        replayed.rounded_clips.push(outer);
        replayed.push_transition(sliding(started_at, 0));
        replayed.replay(0..source.len(), &source);
        replayed.finish();
        assert_eq!(replayed.monochrome_sprites[0].rounded_clip, 3);
        assert_eq!(replayed.rounded_clips[2].parent, 2);
        assert_eq!(replayed.rounded_clips[2].transition, 2);
        replayed.advance_transitions(started_at + std::time::Duration::from_millis(50));
        assert_eq!(replayed.rounded_clips[1].bounds.origin.y, ScaledPixels(0.));
        assert_eq!(replayed.rounded_clips[2].bounds.origin.y, ScaledPixels(10.));
        replayed.advance_transitions(started_at + std::time::Duration::from_millis(100));
        assert_eq!(replayed.rounded_clips[2].bounds.origin.y, ScaledPixels(0.));
    }

    #[test]
    fn a_nested_transition_moves_with_the_one_around_it() {
        let started_at = std::time::Instant::now();
        let mut scene = Scene::default();
        let Some(outer) = scene.push_transition(rolling_in(started_at)) else {
            panic!("a first transition always fits");
        };
        let Some(inner) = scene.push_transition(sliding(started_at, outer)) else {
            panic!("a second transition always fits");
        };
        scene.set_current_transition(inner);
        scene.insert_primitive(shimmering_glyph(0));
        scene.set_current_transition(0);
        scene.finish();

        scene.advance_transitions(started_at + std::time::Duration::from_millis(50));
        let sprite = &scene.monochrome_sprites[0];
        // Halfway through both: the outer's 5 logical pixels down, the
        // inner's 3 left, at scale factor 2; only the outer fades.
        assert_eq!(sprite.bounds.origin.y, ScaledPixels::from(10.));
        assert_eq!(sprite.bounds.origin.x, ScaledPixels::from(-6.));
        assert!((sprite.color.a - 0.5).abs() < 1e-6);
    }

    #[test]
    fn replaying_a_nested_transition_brings_its_parent_along() {
        let started_at = std::time::Instant::now();
        let mut source = Scene::default();
        let Some(outer) = source.push_transition(rolling_in(started_at)) else {
            panic!("a first transition always fits");
        };
        let Some(inner) = source.push_transition(sliding(started_at, outer)) else {
            panic!("a second transition always fits");
        };
        source.set_current_transition(inner);
        source.insert_primitive(shimmering_glyph(0));
        source.set_current_transition(0);
        source.finish();

        let mut replayed = Scene::default();
        replayed.push_transition(rolling_in(started_at - std::time::Duration::from_secs(5)));
        replayed.replay(0..source.len(), &source);
        replayed.finish();
        assert_eq!(replayed.transitions.len(), 3);
        let stamped = replayed.monochrome_sprites[0].pad;
        let parent = replayed.transitions[stamped as usize - 1].parent;
        assert!(parent != 0 && parent < stamped, "the parent is copied first");
        replayed.advance_transitions(started_at + std::time::Duration::from_millis(50));
        assert_eq!(replayed.monochrome_sprites[0].bounds.origin.x, ScaledPixels::from(-6.));
        assert_eq!(replayed.monochrome_sprites[0].bounds.origin.y, ScaledPixels::from(10.));
    }

    #[test]
    fn a_transition_moves_and_fades_what_it_covers_from_rest() {
        let started_at = std::time::Instant::now();
        let mut scene = Scene::default();
        let transition = scene.push_transition(rolling_in(started_at));
        let Some(transition) = transition else {
            panic!("a first transition always fits");
        };
        scene.set_current_transition(transition);
        scene.insert_primitive(shimmering_glyph(0));
        scene.insert_primitive(opaque_quad());
        scene.set_current_transition(0);
        scene.insert_primitive(shimmering_glyph(0));
        scene.finish();
        assert!(scene.transitions_in_flight());
        assert!(scene.has_time_animations());
        assert_eq!(scene.transitioned.len(), 2, "only what was painted inside it");
        // Moved content never takes the opaque depth pass.
        assert!(scene.opaque_quad_indices.is_empty());

        scene.advance_transitions(started_at + std::time::Duration::from_millis(50));
        let moved: Vec<_> = scene.monochrome_sprites.iter().filter(|sprite| sprite.pad != 0).collect();
        assert_eq!(moved.len(), 1);
        // Halfway, linear: half of 10 logical pixels at scale factor 2.
        assert_eq!(moved[0].bounds.origin.y, ScaledPixels::from(10.));
        assert!((moved[0].color.a - 0.5).abs() < 1e-6);
        let still = scene.monochrome_sprites.iter().find(|sprite| sprite.pad == 0);
        assert!(still.is_some_and(|sprite| sprite.bounds.origin.y == ScaledPixels::from(0.)
            && sprite.color.a == 1.0));
        assert_eq!(scene.quads[0].bounds.origin.y, ScaledPixels::from(10.));
        assert!(scene.quads[0].background.as_solid().is_some_and(|color| (color.a - 0.5).abs() < 1e-6));
        assert_eq!(scene.quads[0].background.time_transition(), transition);

        // Every present starts from rest, so advancing twice lands the same.
        for _ in 0..2 {
            scene.advance_transitions(started_at + std::time::Duration::from_millis(100));
        }
        assert_eq!(moved_origin(&scene), ScaledPixels::from(0.));
        assert!(scene.quads[0].background.as_solid().is_some_and(|color| color.a == 1.0));
    }

    fn moved_origin(scene: &Scene) -> ScaledPixels {
        scene
            .monochrome_sprites
            .iter()
            .find(|sprite| sprite.pad != 0)
            .map_or(ScaledPixels::from(-1.), |sprite| sprite.bounds.origin.y)
    }

    /// A landed transition rests where its content was painted only when
    /// its easing and its fade span both reach their ends.
    #[test]
    fn a_landed_transition_rests_only_where_it_ends_at_identity() {
        let now = std::time::Instant::now();
        let landed = TimeTransition::new(
            now - std::time::Duration::from_secs(1),
            std::time::Duration::from_millis(200),
        )
        .offset(point(px(0.), px(20.)), Point::default())
        .opacity(0., 1.);
        assert!(landed.rests_at_identity_by(now));
        assert!(!landed.easing(|progress| progress * 0.5).rests_at_identity_by(now));
        assert!(!landed.opacity_span(0., 2.).rests_at_identity_by(now));
        assert!(!landed.rests_at_identity_by(now - std::time::Duration::from_secs(1)));
    }

    #[test]
    fn a_landed_transition_stops_asking_for_frames() {
        let started_at = std::time::Instant::now() - std::time::Duration::from_secs(1);
        let mut scene = Scene::default();
        let Some(transition) = scene.push_transition(rolling_in(started_at)) else {
            panic!("a first transition always fits");
        };
        scene.set_current_transition(transition);
        scene.insert_primitive(shimmering_glyph(0));
        scene.finish();
        assert!(!scene.transitions_in_flight());
        assert!(!scene.has_time_animations());
        scene.advance_time_animations();
        assert_eq!(moved_origin(&scene), ScaledPixels::from(0.), "painted where it landed");
    }

    #[test]
    fn replayed_transitions_are_remapped_into_the_new_scene() {
        let started_at = std::time::Instant::now();
        let mut source = Scene::default();
        let Some(transition) = source.push_transition(rolling_in(started_at)) else {
            panic!("a first transition always fits");
        };
        source.set_current_transition(transition);
        source.insert_primitive(shimmering_glyph(0));
        source.insert_primitive(shimmering_glyph(0));
        source.set_current_transition(0);
        source.finish();

        let mut replayed = Scene::default();
        // An unrelated transition first, so the replayed id has to be remapped.
        replayed.push_transition(rolling_in(started_at - std::time::Duration::from_secs(5)));
        replayed.replay(0..source.len(), &source);
        replayed.finish();
        assert_eq!(replayed.transitions.len(), 2, "one entry per transition, not per glyph");
        assert!(replayed.monochrome_sprites.iter().all(|sprite| sprite.pad == 2));
        assert!(replayed.transitions_in_flight());
    }

    #[test]
    fn a_background_keeps_glass_animation_and_transition_apart() {
        let background = Background::from(Hsla::black())
            .glass_content()
            .with_time_animation(0x1234)
            .with_time_transition(0x456);
        assert!(background.is_glass_content());
        assert_eq!(background.time_animation(), 0x1234);
        assert_eq!(background.time_transition(), 0x456);
        let background = background.with_time_animation(7);
        assert_eq!(background.time_transition(), 0x456);
        let background = background.with_time_transition(Background::MAX_TIME_TRANSITION + 1);
        assert_eq!(background.time_transition(), 0x456, "an id that does not fit is not stored");
        assert_eq!(background.with_time_transition(0).time_animation(), 7);
    }

    #[test]
    fn a_fade_can_run_over_part_of_the_transition() {
        let started_at = std::time::Instant::now();
        let transition = TimeTransition::new(started_at, std::time::Duration::from_millis(100))
            .opacity(0.0, 1.0)
            .opacity_span(0.5, 1.0);
        let at = |millis| started_at + std::time::Duration::from_millis(millis);
        assert_eq!(transition.opacity_at(at(25)), 0.0);
        assert!((transition.opacity_at(at(75)) - 0.5).abs() < 1e-6);
        assert_eq!(transition.opacity_at(at(100)), 1.0);
    }

    #[test]
    fn a_scene_without_moving_parts_does_not_animate() {
        let mut scene = Scene::default();
        scene.insert_primitive(shimmering_glyph(0));
        scene.insert_primitive(opaque_quad());
        scene.finish();
        assert!(!scene.has_time_animations());
    }

    #[test]
    fn opacity_cycle_interpolates_between_keyframes_and_wraps() {
        let cycle = OpacityCycle::new(
            std::time::Duration::from_millis(750),
            [(0.0, 1.0), (0.5, 0.2), (0.75, 0.2), (1.0, 1.0)],
        );
        assert!((cycle.opacity_at(0.0) - 1.0).abs() < 1e-6);
        assert!((cycle.opacity_at(0.25) - 0.6).abs() < 1e-6);
        assert!((cycle.opacity_at(0.6) - 0.2).abs() < 1e-6);
        assert!((cycle.opacity_at(0.875) - 0.6).abs() < 1e-6);
        assert!((cycle.opacity_at(1.25) - 0.6).abs() < 1e-6);
    }

    /// An animated quad may be solid now and translucent at the next present,
    /// which the opaque depth pass would paint as solid.
    #[test]
    fn opacity_cycled_quads_take_the_blended_pass() {
        let mut scene = Scene::default();
        let quad = opaque_quad();
        let animation = scene.push_quad_animation(QuadAnimation {
            opacity: Some(OpacityCycle::new(
                std::time::Duration::from_millis(750),
                [(0.0, 0.5), (0.5, 0.5), (0.75, 0.5), (1.0, 0.5)],
            )),
            offset: None,
            background: quad.background,
            border_color: quad.border_color,
        });
        scene.insert_primitive(Quad {
            background: quad.background.with_time_animation(animation),
            ..quad
        });
        scene.finish();
        assert!(scene.opaque_quad_indices.is_empty());
        assert_eq!(scene.blended_quad_indices, vec![0]);

        scene.advance_time_animations();
        let background = scene.quads[0].background;
        assert_eq!(background.time_animation(), animation);
        assert!((background.solid.a - quad.background.solid.a * 0.5).abs() < 1e-6);
    }

    fn shimmer_scene_with_glyphs_at(origins: &[(f32, f32)]) -> Scene {
        let mut scene = Scene::default();
        let animation = scene.push_shimmer_animation(ShimmerAnimation {
            band_start: -40.0,
            travel: 1000.0,
            period: std::time::Duration::from_millis(1000),
            hold: 0.0,
        });
        for &(x, y) in origins {
            let bounds = Bounds {
                origin: point(ScaledPixels(x), ScaledPixels(y)),
                size: unit_bounds().size,
            };
            scene.insert_primitive(MonochromeSprite {
                bounds,
                content_mask: ContentMask { bounds },
                ..shimmering_glyph(animation)
            });
        }
        scene
    }

    fn device_viewport(width: i32, height: i32) -> Size<DevicePixels> {
        Size {
            width: DevicePixels(width),
            height: DevicePixels(height),
        }
    }

    fn device_rect(left: i32, top: i32, right: i32, bottom: i32) -> Bounds<DevicePixels> {
        Bounds {
            origin: point(DevicePixels(left), DevicePixels(top)),
            size: device_viewport(right - left, bottom - top),
        }
    }

    /// Presenting a shimmer again changes only its glyphs, so only they are
    /// damaged, each with a pixel of slack, while the scene keeps the content
    /// id a renderer recognizes its previous present by.
    #[test]
    fn a_shimmer_present_damages_only_its_glyphs() {
        let mut scene = shimmer_scene_with_glyphs_at(&[(100., 100.), (500., 600.)]);
        scene.insert_primitive(opaque_quad());
        scene.finish();
        let content_id = scene.content_id();
        let viewport = device_viewport(1000, 1000);
        let mut damage = Vec::new();
        assert!(
            !scene.time_animation_damage(viewport, &mut damage),
            "nothing to compare against before the first advance"
        );

        scene.advance_time_animations();
        assert!(scene.time_animation_damage(viewport, &mut damage));
        damage.sort_by_key(|rect| rect.origin.y.0);
        assert_eq!(
            damage,
            vec![device_rect(99, 99, 111, 111), device_rect(499, 599, 511, 611)]
        );
        assert_eq!(scene.content_id(), content_id);

        scene.finish();
        assert_ne!(scene.content_id(), content_id);
    }

    /// A line of shimmering glyphs is damaged as one rectangle, not one per
    /// glyph.
    #[test]
    fn neighbouring_glyphs_share_one_damage_rect() {
        let mut scene = shimmer_scene_with_glyphs_at(&[(0., 0.), (12., 0.), (24., 0.), (36., 20.)]);
        scene.finish();
        scene.advance_time_animations();
        let mut damage = Vec::new();
        assert!(scene.time_animation_damage(device_viewport(1000, 1000), &mut damage));
        assert_eq!(damage, vec![device_rect(0, 0, 47, 31)]);
    }

    #[test]
    fn damage_reaching_a_backdrop_blur_takes_all_of_it() {
        let blur_at = |x: f32, y: f32| {
            let bounds = Bounds {
                origin: point(ScaledPixels(x), ScaledPixels(y)),
                size: Size {
                    width: ScaledPixels(40.),
                    height: ScaledPixels(20.),
                },
            };
            BackdropBlur {
                bounds,
                content_mask: ContentMask { bounds },
                ..unit_backdrop_blur()
            }
        };
        let mut scene = shimmer_scene_with_glyphs_at(&[(400., 400.)]);
        scene.insert_backdrop_blur(blur_at(405., 405.));
        scene.insert_backdrop_blur(blur_at(700., 700.));
        scene.finish();
        scene.advance_time_animations();
        let mut damage = Vec::new();
        assert!(scene.time_animation_damage(device_viewport(1000, 1000), &mut damage));
        assert_eq!(
            damage,
            vec![device_rect(399, 399, 446, 426)],
            "the blur the glyph reaches is redrawn whole, the other one not at all"
        );
    }

    #[test]
    fn damage_over_half_the_viewport_is_not_worth_bounding() {
        let mut scene = shimmer_scene_with_glyphs_at(&[(0., 0.)]);
        scene.finish();
        scene.advance_time_animations();
        let mut damage = Vec::new();
        assert!(scene.time_animation_damage(device_viewport(100, 100), &mut damage));
        assert!(!scene.time_animation_damage(device_viewport(14, 14), &mut damage));
    }

    /// A moving transition shifts whatever it covers, so its present cannot
    /// be bounded; once it has landed before an advance, the next one moves
    /// nothing and the shimmer alone is damaged again.
    #[test]
    fn damage_is_unbounded_until_transitions_have_landed() {
        let viewport = device_viewport(1000, 1000);
        let mut damage = Vec::new();

        let mut moving = shimmer_scene_with_glyphs_at(&[(100., 100.)]);
        let Some(transition) = moving.push_transition(rolling_in(std::time::Instant::now())) else {
            panic!("a first transition always fits");
        };
        moving.set_current_transition(transition);
        moving.insert_primitive(shimmering_glyph(0));
        moving.finish();
        moving.advance_time_animations();
        moving.advance_time_animations();
        assert!(!moving.time_animation_damage(viewport, &mut damage));

        let landed_at = std::time::Instant::now() - std::time::Duration::from_secs(1);
        let mut landed = shimmer_scene_with_glyphs_at(&[(100., 100.)]);
        let Some(transition) = landed.push_transition(rolling_in(landed_at)) else {
            panic!("a first transition always fits");
        };
        landed.set_current_transition(transition);
        landed.insert_primitive(shimmering_glyph(0));
        landed.finish();
        landed.advance_time_animations();
        assert!(
            !landed.time_animation_damage(viewport, &mut damage),
            "the first advance after a draw may still have moved it"
        );
        landed.advance_time_animations();
        assert!(landed.time_animation_damage(viewport, &mut damage));
        assert_eq!(damage, vec![device_rect(99, 99, 111, 111)]);
    }

    fn turning_sprite(scene: &mut Scene, period_ms: u64) -> MonochromeSprite {
        let animation = scene.push_rotation_animation(RotationAnimation {
            period: std::time::Duration::from_millis(period_ms),
        });
        let bounds = Bounds {
            origin: point(ScaledPixels(100.), ScaledPixels(200.)),
            size: Size {
                width: ScaledPixels(20.),
                height: ScaledPixels(20.),
            },
        };
        MonochromeSprite {
            bounds,
            content_mask: ContentMask { bounds },
            effect: SpriteEffect {
                kind: SpriteEffect::MASK_KIND,
                animation,
                ..SpriteEffect::default()
            },
            ..shimmering_glyph(0)
        }
    }

    fn turned_point(sprite: &MonochromeSprite, x: f32, y: f32) -> (f32, f32) {
        let matrix = sprite.transformation;
        (
            matrix.rotation_scale[0][0] * x + matrix.rotation_scale[0][1] * y + matrix.translation[0],
            matrix.rotation_scale[1][0] * x + matrix.rotation_scale[1][1] * y + matrix.translation[1],
        )
    }

    /// A turning sprite is rotated about its own centre from the
    /// transformation it was painted with on every advance, so the turn
    /// never compounds, and a replay of the frame keeps it turning.
    #[test]
    fn a_turning_sprite_rotates_about_its_centre_on_every_advance() {
        let mut scene = Scene::default();
        let sprite = turning_sprite(&mut scene, 1000);
        scene.insert_primitive(sprite);
        scene.finish();
        assert!(scene.has_time_animations());
        assert!(scene.animated_monochrome_sprites.is_empty(), "a turn is not a shimmer");

        for _ in 0..3 {
            scene.advance_time_animations();
            let sprite = &scene.monochrome_sprites[0];
            let (x, y) = turned_point(sprite, 110., 210.);
            assert!((x - 110.).abs() < 1e-3 && (y - 210.).abs() < 1e-3, "the centre stays put");
            let (x, y) = turned_point(sprite, 120., 210.);
            let radius = ((x - 110.).powi(2) + (y - 210.).powi(2)).sqrt();
            assert!((radius - 10.).abs() < 1e-3, "a turn keeps distances from the centre: {radius}");
            let determinant = sprite.transformation.rotation_scale[0][0]
                * sprite.transformation.rotation_scale[1][1]
                - sprite.transformation.rotation_scale[0][1]
                    * sprite.transformation.rotation_scale[1][0];
            assert!((determinant - 1.).abs() < 1e-4, "turns never compound into a scale");
        }
        assert_eq!(
            scene.painted_monochrome_sprites[0].transformation,
            TransformationMatrix::unit(),
            "the painted copy keeps the transformation it was painted with"
        );

        let mut replayed = Scene::default();
        replayed.push_shimmer_animation(ShimmerAnimation {
            band_start: 0.0,
            travel: 0.0,
            period: std::time::Duration::from_millis(1),
            hold: 0.0,
        });
        replayed.replay(0..scene.len(), &scene);
        replayed.finish();
        assert_eq!(replayed.rotating_monochrome_sprites.len(), 1);
        assert_eq!(replayed.rotation_animations.len(), 1);
        assert_eq!(replayed.shimmer_animations.len(), 1, "the turn is not copied as a sweep");
        replayed.advance_time_animations();
        assert_ne!(replayed.monochrome_sprites[0].transformation, TransformationMatrix::unit());
    }

    /// A sprite painted translated, or moved by a transition, turns in place
    /// where it is shown rather than orbiting the spot it was laid out at.
    #[test]
    fn a_turning_sprite_turns_where_it_is_shown() {
        let painted = TransformationMatrix::unit().translate(point(ScaledPixels(10.), ScaledPixels(0.)));
        let apply = |matrix: TransformationMatrix, x: f32, y: f32| {
            (
                matrix.rotation_scale[0][0] * x + matrix.rotation_scale[0][1] * y + matrix.translation[0],
                matrix.rotation_scale[1][0] * x + matrix.rotation_scale[1][1] * y + matrix.translation[1],
            )
        };
        for angle in [0.0, 1.0, std::f32::consts::FRAC_PI_2, std::f32::consts::PI, 4.0] {
            let still = turned(painted, Point::default(), point(ScaledPixels(110.), ScaledPixels(210.)), angle);
            let (x, y) = apply(still, 110., 210.);
            assert!((x - 120.).abs() < 1e-3 && (y - 210.).abs() < 1e-3, "orbited to ({x}, {y}) at {angle}");
            let moved = turned(
                painted,
                point(ScaledPixels(5.), ScaledPixels(-7.)),
                point(ScaledPixels(115.), ScaledPixels(203.)),
                angle,
            );
            let (x, y) = apply(moved, 115., 203.);
            assert!((x - 125.).abs() < 1e-3 && (y - 203.).abs() < 1e-3, "moved one orbited to ({x}, {y})");
            // A scaled matrix does not commute with the move: unless it is
            // carried along, the sprite is scaled about where it was laid
            // out and shows at (120, 196).
            let doubled = TransformationMatrix::unit()
                .translate(point(ScaledPixels(110.), ScaledPixels(210.)))
                .scale(Size { width: 2., height: 2. })
                .translate(point(ScaledPixels(-110.), ScaledPixels(-210.)));
            let moved = turned(
                doubled,
                point(ScaledPixels(5.), ScaledPixels(-7.)),
                point(ScaledPixels(115.), ScaledPixels(203.)),
                angle,
            );
            let (x, y) = apply(moved, 115., 203.);
            assert!((x - 115.).abs() < 1e-3 && (y - 203.).abs() < 1e-3, "moved scaled one showed at ({x}, {y})");
        }
    }

    /// A turning sprite is culled and ordered by the square it sweeps: a quad
    /// it only reaches while turned still draws over it, and one whose own
    /// rectangle misses its mask is kept for the corners that do not.
    #[test]
    fn a_turning_sprite_is_ordered_by_the_square_it_sweeps() {
        let mut scene = Scene::default();
        let sprite = turning_sprite(&mut scene, 800);
        scene.insert_primitive(MonochromeSprite {
            content_mask: ContentMask {
                bounds: Bounds {
                    origin: point(ScaledPixels(0.), ScaledPixels(0.)),
                    size: Size {
                        width: ScaledPixels(1000.),
                        height: ScaledPixels(1000.),
                    },
                },
            },
            ..sprite
        });
        let reached = Bounds {
            origin: point(ScaledPixels(122.), ScaledPixels(208.)),
            size: Size {
                width: ScaledPixels(4.),
                height: ScaledPixels(4.),
            },
        };
        scene.insert_primitive(Quad {
            bounds: reached,
            content_mask: ContentMask { bounds: reached },
            ..opaque_quad()
        });
        let mut corner_only = turning_sprite(&mut scene, 800);
        corner_only.content_mask = ContentMask {
            bounds: Bounds {
                origin: point(ScaledPixels(121.), ScaledPixels(200.)),
                size: Size {
                    width: ScaledPixels(10.),
                    height: ScaledPixels(20.),
                },
            },
        };
        scene.insert_primitive(corner_only);
        scene.finish();
        assert_eq!(scene.monochrome_sprites.len(), 2, "the sprite reaching its mask only turned was kept");
        assert!(
            scene.quads[0].order > scene.monochrome_sprites[0].order,
            "the quad the turning sprite reaches must draw over it"
        );
    }

    /// The square a sprite sweeps follows the transformation it is painted
    /// with: one painted at twice its size reaches twice as far.
    #[test]
    fn a_scaled_turning_sprite_sweeps_its_scaled_corners() {
        let bounds = Bounds {
            origin: point(ScaledPixels(100.), ScaledPixels(200.)),
            size: Size {
                width: ScaledPixels(20.),
                height: ScaledPixels(20.),
            },
        };
        let center = bounds.center();
        let doubled = TransformationMatrix::unit()
            .translate(center)
            .scale(Size { width: 2., height: 2. })
            .translate(point(ScaledPixels(-center.x.0), ScaledPixels(-center.y.0)));
        let swept = swept_by_turning(&bounds, doubled);
        let reach = 2. * (200f32).sqrt();
        assert!((swept.origin.x.0 - (110. - reach)).abs() < 1e-3);
        assert!((swept.size.height.0 - 2. * reach).abs() < 1e-3);
        assert!(swept.bottom().0 > 238.28, "a 45 degree corner at y = 238.28 must be inside");
    }

    /// A retained view moved within its mask refuses the move when a turning
    /// sprite's swept corners would cross an edge its own rectangle only
    /// touches.
    #[test]
    fn a_turning_sprite_is_clipped_by_the_square_it_sweeps_when_moved() {
        let mut scene = Scene::default();
        let mut sprite = turning_sprite(&mut scene, 800);
        sprite.content_mask = ContentMask {
            bounds: Bounds {
                origin: point(ScaledPixels(0.), ScaledPixels(0.)),
                size: Size {
                    width: ScaledPixels(120.),
                    height: ScaledPixels(400.),
                },
            },
        };
        scene.insert_primitive(sprite);
        let outer = [0.0, 0.0, 120.0, 400.0];
        assert!(
            !scene.clipped_only_inside(0..scene.paint_operations.len(), outer),
            "the swept corners reach past the right edge the rectangle only touches"
        );
    }

    /// A present that only turns a spinner damages the square its corners
    /// sweep, and nothing else.
    #[test]
    fn a_turning_sprite_damages_the_square_it_sweeps() {
        let mut scene = Scene::default();
        let mut sprite = turning_sprite(&mut scene, 800);
        sprite.content_mask = ContentMask {
            bounds: Bounds {
                origin: point(ScaledPixels(0.), ScaledPixels(0.)),
                size: Size {
                    width: ScaledPixels(1000.),
                    height: ScaledPixels(1000.),
                },
            },
        };
        scene.insert_primitive(sprite);
        scene.insert_primitive(opaque_quad());
        scene.finish();
        scene.advance_time_animations();
        let mut damage = Vec::new();
        assert!(scene.time_animation_damage(device_viewport(1000, 1000), &mut damage));
        // Centre (110, 210), half-diagonal 14.14, a pixel of slack.
        assert_eq!(damage, vec![device_rect(94, 194, 126, 226)]);
    }

    fn bouncing_quad(scene: &mut Scene, phase: f32) -> Quad {
        let animation = scene.push_quad_animation(QuadAnimation {
            opacity: None,
            offset: Some(
                OffsetCycle::bounce(
                    std::time::Duration::from_millis(1000),
                    0.55,
                    point(crate::px(0.), crate::px(-9.)),
                )
                .phase(phase)
                .scaled(2.0),
            ),
            background: opaque_quad().background,
            border_color: opaque_quad().border_color,
        });
        let bounds = Bounds {
            origin: point(ScaledPixels(100.), ScaledPixels(200.)),
            size: Size {
                width: ScaledPixels(8.),
                height: ScaledPixels(8.),
            },
        };
        Quad {
            bounds,
            content_mask: ContentMask {
                bounds: Bounds {
                    origin: point(ScaledPixels(0.), ScaledPixels(0.)),
                    size: Size {
                        width: ScaledPixels(1000.),
                        height: ScaledPixels(1000.),
                    },
                },
            },
            background: opaque_quad().background.with_time_animation(animation),
            ..opaque_quad()
        }
    }

    #[test]
    fn an_offset_cycle_rests_outside_its_active_window_and_peaks_inside_it() {
        let cycle = OffsetCycle::bounce(std::time::Duration::from_secs(1), 0.5, point(crate::px(0.), crate::px(-4.)));
        assert!(cycle.rise_at(0.0).abs() < 1e-6);
        assert!((cycle.rise_at(0.25) - 1.0).abs() < 1e-6);
        assert!(cycle.rise_at(0.5).abs() < 1e-6);
        assert!(cycle.rise_at(0.8).abs() < 1e-6, "at rest for the rest of the period");
        let late = cycle.phase(-0.25);
        assert!((late.rise_at(0.5) - 1.0).abs() < 1e-6, "a negative phase starts later");
    }

    /// A bouncing quad is moved from where it was painted on every advance,
    /// by whole device pixels, never compounding, and only ever between its
    /// rest and its peak; its damage covers that whole span.
    #[test]
    fn an_offset_quad_moves_from_where_it_was_painted() {
        let mut scene = Scene::default();
        let quad = bouncing_quad(&mut scene, 0.0);
        scene.insert_primitive(quad);
        scene.finish();
        assert!(scene.has_time_animations());
        for _ in 0..5 {
            scene.advance_time_animations();
            let origin = scene.quads[0].bounds.origin;
            assert_eq!(origin.x, ScaledPixels(100.));
            assert!(origin.y.0 <= 200. && origin.y.0 >= 182., "out of its span: {origin:?}");
            assert_eq!(origin.y.0.fract(), 0., "moves by whole device pixels");
        }
        let mut damage = Vec::new();
        assert!(scene.time_animation_damage(device_viewport(1000, 1000), &mut damage));
        assert_eq!(damage, vec![device_rect(99, 181, 109, 209)]);
    }

    /// A bouncing quad inside a transition takes the transition's fade once
    /// per present, not compounded, and once the transition has landed away
    /// from where it was painted its damage follows it there.
    #[test]
    fn an_offset_quad_in_a_transition_neither_compounds_nor_loses_its_damage() {
        let landed_at = std::time::Instant::now() - std::time::Duration::from_secs(1);
        let mut scene = Scene::default();
        let Some(transition) = scene.push_transition(SceneTransition {
            transition: TimeTransition::new(landed_at, std::time::Duration::from_millis(100))
                .offset(point(px(50.), px(0.)), point(px(50.), px(0.)))
                .opacity(0.5, 0.5),
            scale_factor: 2.0,
            parent: 0,
        }) else {
            panic!("a first transition always fits");
        };
        scene.set_current_transition(transition);
        let quad = bouncing_quad(&mut scene, 0.0);
        let painted_alpha = quad.background.solid.a;
        scene.insert_primitive(quad);
        scene.finish();
        for _ in 0..4 {
            scene.advance_time_animations();
            let alpha = scene.quads[0].background.solid.a;
            assert!((alpha - painted_alpha * 0.5).abs() < 1e-5, "the fade compounded to {alpha}");
            assert!(scene.quads[0].bounds.origin.x.0 >= 199.9, "moved with the transition");
        }
        let mut damage = Vec::new();
        assert!(scene.time_animation_damage(device_viewport(1000, 1000), &mut damage));
        assert_eq!(damage, vec![device_rect(199, 181, 209, 209)]);
    }

    /// A bouncing quad is ordered by every rectangle it passes through: a
    /// quad painted after it, which it only reaches at its peak, draws over
    /// it.
    #[test]
    fn an_offset_quad_is_ordered_by_the_span_it_moves_through() {
        let mut scene = Scene::default();
        let quad = bouncing_quad(&mut scene, 0.0);
        scene.insert_primitive(quad);
        let above = Bounds {
            origin: point(ScaledPixels(100.), ScaledPixels(183.)),
            size: Size {
                width: ScaledPixels(8.),
                height: ScaledPixels(4.),
            },
        };
        scene.insert_primitive(Quad {
            bounds: above,
            content_mask: ContentMask { bounds: above },
            ..opaque_quad()
        });
        scene.finish();
        let order = |y: f32| {
            scene
                .quads
                .iter()
                .find(|quad| quad.bounds.origin.y.0 == y)
                .map(|quad| quad.order)
        };
        assert!(order(183.) > order(200.), "the later quad must draw over the bounce's peak");
    }

    /// The glyph sprite inserts build the same scene `insert_primitive` builds
    /// from the same sprites: culling, draw order, transition tagging and paint
    /// operations alike.
    #[test]
    fn glyph_sprite_inserts_match_insert_primitive() {
        fn subpixel(sprite: MonochromeSprite) -> SubpixelSprite {
            SubpixelSprite {
                rounded_clip: 0,
                clip_padding: 0,
                order: sprite.order,
                pad: sprite.pad,
                bounds: sprite.bounds,
                content_mask: sprite.content_mask,
                color: sprite.color,
                effect: sprite.effect,
                tile: sprite.tile,
                transformation: sprite.transformation,
            }
        }
        let started_at = std::time::Instant::now();
        let offscreen = MonochromeSprite {
            bounds: Bounds {
                origin: point(ScaledPixels::from(50.), ScaledPixels::from(0.)),
                ..unit_bounds()
            },
            ..shimmering_glyph(0)
        };
        let shifted = MonochromeSprite {
            bounds: Bounds {
                origin: point(ScaledPixels::from(4.), ScaledPixels::from(2.)),
                ..unit_bounds()
            },
            ..shimmering_glyph(0)
        };
        let build = |fast: bool| {
            let mut scene = Scene::default();
            let mut insert = |scene: &mut Scene, sprite: MonochromeSprite| {
                if fast {
                    scene.insert_monochrome_sprite(sprite);
                    scene.insert_subpixel_sprite(subpixel(sprite));
                } else {
                    scene.insert_primitive(sprite);
                    scene.insert_primitive(subpixel(sprite));
                }
            };
            insert(&mut scene, shimmering_glyph(0));
            insert(&mut scene, offscreen);
            scene.insert_primitive(opaque_quad());
            let Some(transition) = scene.push_transition(rolling_in(started_at)) else {
                panic!("a first transition always fits");
            };
            scene.set_current_transition(transition);
            insert(&mut scene, shifted);
            scene.set_current_transition(0);
            scene.push_layer(unit_bounds());
            insert(&mut scene, shifted);
            scene.pop_layer();
            scene.finish();
            scene
        };
        let primitive = build(false);
        let fast = build(true);
        assert_eq!(fast.paint_operations.len(), primitive.paint_operations.len());
        assert_eq!(fast.monochrome_sprites.len(), 3);
        assert_eq!(fast.subpixel_sprites.len(), 3);
        let key = |order: DrawOrder, pad: u32, bounds: Bounds<ScaledPixels>| (order, pad, bounds);
        let monochrome = |scene: &Scene| {
            scene
                .monochrome_sprites
                .iter()
                .map(|sprite| key(sprite.order, sprite.pad, sprite.bounds))
                .collect::<Vec<_>>()
        };
        let subpixel_keys = |scene: &Scene| {
            scene
                .subpixel_sprites
                .iter()
                .map(|sprite| key(sprite.order, sprite.pad, sprite.bounds))
                .collect::<Vec<_>>()
        };
        assert_eq!(monochrome(&fast), monochrome(&primitive));
        assert_eq!(subpixel_keys(&fast), subpixel_keys(&primitive));
        assert_eq!(fast.quads[0].order, primitive.quads[0].order);
        assert_eq!(fast.transitioned.len(), primitive.transitioned.len());
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

/// How far primitives replayed from the last frame move, and the content
/// mask around them then and now: what a view drawn again elsewhere drew
/// moves with it, the mask edges it pushed itself move with it, and the
/// edges of the mask around it are the new mask's. See
/// [`crate::window::view_retention::moving::ViewMove`].
#[derive(Clone, Debug)]
pub(crate) struct SceneMove {
    pub(crate) delta: Point<ScaledPixels>,
    /// The edges of the mask around what moved, then and now.
    pub(crate) old_outer: [f32; 4],
    pub(crate) new_outer: [f32; 4],
    pub(crate) inherited_clip_depth: usize,
    pub(crate) inherited_clip: u32,
}

impl SceneMove {
    pub(crate) fn bounds(&self, bounds: Bounds<ScaledPixels>) -> Bounds<ScaledPixels> {
        Bounds {
            origin: bounds.origin + self.delta,
            size: bounds.size,
        }
    }

    fn mask(&self, mask: &ContentMask<ScaledPixels>) -> ContentMask<ScaledPixels> {
        use crate::window::view_retention::moving::{rebase_edges, scaled_edges};
        let [left, top, right, bottom] = rebase_edges(
            scaled_edges(&mask.bounds),
            self.old_outer,
            self.new_outer,
            [self.delta.x.0, self.delta.y.0],
        );
        ContentMask {
            bounds: Bounds {
                origin: point(ScaledPixels(left), ScaledPixels(top)),
                size: crate::size(ScaledPixels(right - left), ScaledPixels(bottom - top)),
            },
        }
    }

    fn move_primitive(&self, primitive: &mut Primitive) {
        match primitive {
            Primitive::Shadow(shadow) => {
                shadow.bounds = self.bounds(shadow.bounds);
                shadow.element_bounds = self.bounds(shadow.element_bounds);
                shadow.content_mask = self.mask(&shadow.content_mask);
            }
            Primitive::Quad(quad) => {
                quad.bounds = self.bounds(quad.bounds);
                quad.content_mask = self.mask(&quad.content_mask);
            }
            Primitive::Path(path) => {
                path.bounds = self.bounds(path.bounds);
                path.content_mask = self.mask(&path.content_mask);
                for vertex in &mut path.vertices {
                    vertex.xy_position = vertex.xy_position + self.delta;
                    vertex.content_mask = self.mask(&vertex.content_mask);
                }
            }
            Primitive::Underline(underline) => {
                underline.bounds = self.bounds(underline.bounds);
                underline.content_mask = self.mask(&underline.content_mask);
            }
            Primitive::MonochromeSprite(sprite) => {
                sprite.bounds = self.bounds(sprite.bounds);
                sprite.content_mask = self.mask(&sprite.content_mask);
                if sprite.effect.kind == SpriteEffect::SHIMMER_KIND {
                    sprite.effect.origin = sprite.effect.origin + self.delta;
                }
                sprite.transformation = self.transformation(sprite.transformation);
            }
            Primitive::SubpixelSprite(sprite) => {
                sprite.bounds = self.bounds(sprite.bounds);
                sprite.content_mask = self.mask(&sprite.content_mask);
                if sprite.effect.kind == SpriteEffect::SHIMMER_KIND {
                    sprite.effect.origin = sprite.effect.origin + self.delta;
                }
                sprite.transformation = self.transformation(sprite.transformation);
            }
            Primitive::PolychromeSprite(sprite) => {
                sprite.bounds = self.bounds(sprite.bounds);
                sprite.content_mask = self.mask(&sprite.content_mask);
            }
            Primitive::Surface(surface) => {
                surface.bounds = self.bounds(surface.bounds);
                surface.content_mask = self.mask(&surface.content_mask);
            }
        }
    }

    /// A sprite's transformation, which maps scene positions, moved: it is
    /// conjugated with the move, `T(d) * M * T(-d)`.
    fn transformation(&self, matrix: TransformationMatrix) -> TransformationMatrix {
        if matrix == TransformationMatrix::unit() {
            return matrix;
        }
        let [dx, dy] = [self.delta.x.0, self.delta.y.0];
        let [[a, b], [c, d]] = matrix.rotation_scale;
        let [tx, ty] = matrix.translation;
        TransformationMatrix {
            rotation_scale: matrix.rotation_scale,
            translation: [tx + dx - (a * dx + b * dy), ty + dy - (c * dx + d * dy)],
        }
    }
}

/// The box a primitive can draw into: its bounds, and for a shadow the blur
/// around them.
pub(crate) fn primitive_extent(primitive: &Primitive) -> Bounds<ScaledPixels> {
    match primitive {
        Primitive::Shadow(shadow) => {
            let margin = shadow.blur_radius * 3.;
            shadow.bounds.dilate(margin)
        }
        Primitive::MonochromeSprite(sprite) => monochrome_sprite_extent(sprite),
        primitive => *primitive.bounds(),
    }
}

/// `painted` carried along with a sprite a transition moved by `moved`, then
/// turned by `angle` about where that puts the sprite's `center`: the sprite
/// turns in place, wherever its painted transformation and the transition
/// put it, rather than orbiting the spot it was laid out at.
fn turned(
    painted: TransformationMatrix,
    moved: Point<ScaledPixels>,
    center: Point<ScaledPixels>,
    angle: f32,
) -> TransformationMatrix {
    let negate = |point: Point<ScaledPixels>| Point::new(ScaledPixels(-point.x.0), ScaledPixels(-point.y.0));
    let carried = TransformationMatrix::unit()
        .translate(moved)
        .compose(painted)
        .translate(negate(moved));
    let [[a, b], [c, d]] = carried.rotation_scale;
    let shown = Point::new(
        ScaledPixels(a * center.x.0 + b * center.y.0 + carried.translation[0]),
        ScaledPixels(c * center.x.0 + d * center.y.0 + carried.translation[1]),
    );
    TransformationMatrix::unit()
        .translate(shown)
        .rotate(Radians(angle))
        .translate(negate(shown))
        .compose(carried)
}

/// Where a monochrome sprite can draw: its rectangle, or for one that turns,
/// the square it sweeps.
fn monochrome_sprite_extent(sprite: &MonochromeSprite) -> Bounds<ScaledPixels> {
    if sprite.effect.animation & ROTATION_ANIMATION != 0 {
        swept_by_turning(&sprite.bounds, sprite.transformation)
    } else {
        sprite.bounds
    }
}

/// The square a sprite of `bounds`, drawn through `painted`, stays inside
/// while turning about its centre as shown: centred there, as far out as
/// its farthest transformed corner.
pub(crate) fn swept_by_turning(
    bounds: &Bounds<ScaledPixels>,
    painted: TransformationMatrix,
) -> Bounds<ScaledPixels> {
    let [[a, b], [c, d]] = painted.rotation_scale;
    let shown = |x: f32, y: f32| {
        (
            a * x + b * y + painted.translation[0],
            c * x + d * y + painted.translation[1],
        )
    };
    let center = bounds.center();
    let (center_x, center_y) = shown(center.x.0, center.y.0);
    let (left, top) = (bounds.left().0, bounds.top().0);
    let (right, bottom) = (bounds.right().0, bounds.bottom().0);
    let reach = [(left, top), (right, top), (left, bottom), (right, bottom)]
        .into_iter()
        .map(|(x, y)| {
            let (x, y) = shown(x, y);
            ((x - center_x).powi(2) + (y - center_y).powi(2)).sqrt()
        })
        .fold(0f32, f32::max);
    Bounds {
        origin: Point::new(ScaledPixels(center_x - reach), ScaledPixels(center_y - reach)),
        size: Size {
            width: ScaledPixels(reach * 2.0),
            height: ScaledPixels(reach * 2.0),
        },
    }
}

/// See [`Scene::glyph_watermark`].
#[derive(Clone, Copy, Debug)]
pub(crate) struct GlyphWatermark {
    operations: usize,
    sprites: usize,
}

/// A layer a paint operation starts: the bounds it orders what is in it by,
/// and the extent and content mask they were clipped from.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Layer {
    bounds: Bounds<ScaledPixels>,
    extent: Bounds<ScaledPixels>,
    mask: Bounds<ScaledPixels>,
}

pub(crate) enum PaintOperation {
    Primitive(Primitive),
    /// Indices into [`Scene::painted_monochrome_sprites`] and
    /// [`Scene::painted_subpixel_sprites`]: glyphs are most of a frame, and
    /// copying each into a paint operation as well cost as much as the rest
    /// of inserting it.
    MonochromeSprite(u32),
    SubpixelSprite(u32),
    BackdropBlur(BackdropBlur),
    StartLayer(Layer),
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
    fn rounded_clip(&self) -> u32 {
        match self {
            Self::Shadow(value) => value.rounded_clip,
            Self::Quad(value) => value.rounded_clip,
            Self::Path(value) => value.rounded_clip,
            Self::Underline(value) => value.rounded_clip,
            Self::MonochromeSprite(value) => value.rounded_clip,
            Self::SubpixelSprite(value) => value.rounded_clip,
            Self::PolychromeSprite(value) => value.rounded_clip,
            Self::Surface(value) => value.rounded_clip,
        }
    }

    fn set_rounded_clip(&mut self, index: u32) {
        match self {
            Self::Shadow(value) => value.rounded_clip = index,
            Self::Quad(value) => value.rounded_clip = index,
            Self::Path(value) => value.rounded_clip = index,
            Self::Underline(value) => value.rounded_clip = index,
            Self::MonochromeSprite(value) => value.rounded_clip = index,
            Self::SubpixelSprite(value) => value.rounded_clip = index,
            Self::PolychromeSprite(value) => value.rounded_clip = index,
            Self::Surface(value) => value.rounded_clip = index,
        }
    }

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
    backdrop_blurs_iter: Peekable<slice::Iter<'a, BackdropBlur>>,
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
        let (batch_kind, mut max_order_and_kind) = if let Some(first_order) = first.0 {
            // A blur at or below this batch's order runs before it; the next
            // one ends the batch, so nothing at or above its order is drawn
            // before the blur samples the framebuffer.
            while self
                .backdrop_blurs_iter
                .next_if(|blur| blur.order <= first_order)
                .is_some()
            {}
            (first.1, (second.0.unwrap_or(u32::MAX), second.1))
        } else {
            return None;
        };
        if let Some(blur) = self.backdrop_blurs_iter.peek() {
            max_order_and_kind = max_order_and_kind.min((blur.order, PrimitiveKind::Shadow));
        }

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

/// One intersection in a rounded clip chain, in device coordinates.
#[derive(Default, Debug, Copy, Clone)]
#[repr(C)]
pub struct RoundedClipNode {
    /// The rectangle before rounding, independent of rectangular overflow masks.
    pub bounds: Bounds<ScaledPixels>,
    /// Radii clamped to the rectangle, in clockwise order.
    pub corner_radii: Corners<ScaledPixels>,
    /// One-based parent index, or zero for the outermost clip.
    pub parent: u32,
    /// The transition containing this clip, independent of its children.
    pub transition: u32,
    /// The unanimated origin, so advancing a transition never compounds offsets.
    pub base_origin: Point<ScaledPixels>,
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
    pub rounded_clip: u32,
    pub clip_padding: u32,
}

impl Quad {
    fn has_opaque_core(&self) -> bool {
        let zero = ScaledPixels(0.);
        // Glass content deliberately leaves the destination alpha of the
        // translucent surface beneath it untouched. The opaque pass blends
        // nothing and so would write alpha 1 over that surface, so glass quads
        // never take it, however solid their color is.
        self.rounded_clip == 0
            && !self.background.is_glass_content()
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
    pub rounded_clip: u32,
    pub clip_padding: u32,
}

impl From<Underline> for Primitive {
    fn from(underline: Underline) -> Self {
        Primitive::Underline(underline)
    }
}

/// A frosted backdrop: what is drawn beneath `bounds` at the time the
/// renderer reaches `order`, blurred, its saturation scaled, and painted back
/// inside the rounded rect, the way a within-window `NSVisualEffectView`
/// treats the content behind it. Painted with
/// [`crate::Window::paint_backdrop_blur`].
#[derive(Debug, Copy, Clone)]
#[repr(C)]
pub struct BackdropBlur {
    /// The draw order the blur runs at, before every primitive of this order.
    pub order: DrawOrder,
    /// The gaussian's standard deviation.
    pub blur_radius: ScaledPixels,
    /// `1.0` keeps the backdrop's colour; above it the blurred backdrop is
    /// more vivid, as system materials make it.
    pub saturation: f32,
    /// How much of the effect replaces the unblurred backdrop, so that an
    /// element fading in or out fades its frost with it.
    pub opacity: f32,
    /// The rect the frosted backdrop fills.
    pub bounds: Bounds<ScaledPixels>,
    /// The mask the rect is clipped to.
    pub content_mask: ContentMask<ScaledPixels>,
    /// The rect's rounding, matching the surface painted over it.
    pub corner_radii: Corners<ScaledPixels>,
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
    pub rounded_clip: u32,
    pub clip_padding: u32,
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
/// color, and what kind of mask the sprite's tile holds. The struct is embedded
/// in *every* [`MonochromeSprite`] and [`SubpixelSprite`], so it is
/// deliberately held to 64 bytes and the shaders early-out on a `kind` without
/// a band before reading it.
#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct SpriteEffect {
    /// `0` for plain text, [`SpriteEffect::SHIMMER_KIND`] for the shimmer band,
    /// [`SpriteEffect::MASK_KIND`] for a tile that is not text at all.
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
    /// One more than the index of the [`ShimmerAnimation`] in the owning
    /// scene that moves `band_origin` with time, or `0` for a still band; or,
    /// with [`ROTATION_ANIMATION`] set, one more than the index of the
    /// [`RotationAnimation`] that turns the sprite. The shaders never read it
    /// (they see it as padding); the scene rewrites `band_origin`, or the
    /// sprite's transformation, from it before every present, so a sweeping
    /// shimmer or a turning spinner animates without its view drawing again.
    pub animation: u32,
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
    /// The tile is an alpha mask that is not a glyph (today: a rasterized SVG).
    /// Windows renders text through DirectWrite's contrast and gamma ramp and
    /// its fragment shader reproduces that for every monochrome sprite; an
    /// icon's mask must be sampled as authored instead, or every
    /// partial-coverage pixel in it is lifted. The other backends apply no
    /// ramp, so for them this kind only means "no shimmer band".
    pub(crate) const MASK_KIND: u32 = 2;
}

/// How a sweeping shimmer band moves with time, kept beside the scene rather
/// than in [`SpriteEffect`] so every glyph in the window does not pay for it.
///
/// The band's trailing edge is `band_start + travel * sweep(t)`, where the
/// sweep runs from 0 to 1 over `period`, then holds past the end for the
/// `hold` fraction of the cycle.
#[derive(Copy, Clone, Debug, PartialEq)]
pub(crate) struct ShimmerAnimation {
    /// Trailing edge at the start of a sweep, in device pixels along the
    /// effect's `direction`.
    pub band_start: f32,
    /// Distance the trailing edge covers in one sweep, in device pixels.
    pub travel: f32,
    pub period: std::time::Duration,
    pub hold: f32,
}

/// Set in a [`SpriteEffect::animation`] that names a [`RotationAnimation`]
/// rather than a [`ShimmerAnimation`]. The shaders never read the field.
pub(crate) const ROTATION_ANIMATION: u32 = 1 << 31;

/// A continuous clockwise turn of one sprite about its centre, evaluated by
/// the scene before every present so a spinner turns without its view
/// drawing again. Painted inside [`crate::Window::with_rotation_animation`].
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct RotationAnimation {
    /// How long one full turn takes.
    pub period: std::time::Duration,
}

impl RotationAnimation {
    fn current_angle(&self) -> f32 {
        time_animation_phase(self.period) * std::f32::consts::TAU
    }
}

/// A repeating opacity curve for quads, evaluated by the scene before every
/// present so the quads animate without their view drawing again.
///
/// `keyframes` are `(phase, opacity)` points over one cycle, in increasing
/// phase from `0.0` to `1.0`; opacity is interpolated linearly between them.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct OpacityCycle {
    period: std::time::Duration,
    phase: f32,
    keyframes: [(f32, f32); 4],
}

impl OpacityCycle {
    /// A cycle of `period` through four `(phase, opacity)` keyframes.
    pub fn new(period: std::time::Duration, keyframes: [(f32, f32); 4]) -> Self {
        Self {
            period,
            phase: 0.0,
            keyframes,
        }
    }

    /// Offsets this cycle by a fraction of its period, so several quads can
    /// run the same curve out of step.
    pub fn phase(mut self, phase: f32) -> Self {
        self.phase = phase;
        self
    }

    /// Opacity at `phase` through the cycle (wrapped into `0..1`).
    pub fn opacity_at(&self, phase: f32) -> f32 {
        let phase = phase.rem_euclid(1.0);
        for pair in self.keyframes.windows(2) {
            let ((start, from), (end, to)) = (pair[0], pair[1]);
            if phase <= end {
                let progress = if end > start {
                    ((phase - start) / (end - start)).clamp(0.0, 1.0)
                } else {
                    1.0
                };
                return from + (to - from) * progress;
            }
        }
        self.keyframes[3].1
    }

    /// Opacity now, on the clock every time-animated primitive shares.
    pub fn current_opacity(&self) -> f32 {
        self.opacity_at(time_animation_phase(self.period) + self.phase)
    }
}

static TIME_ANIMATION_EPOCH: std::sync::LazyLock<std::time::Instant> =
    std::sync::LazyLock::new(std::time::Instant::now);

/// How far through a cycle of `period` the shared animation clock is, in
/// `0..1`. Computed in `f64` so the phase does not step after long uptimes.
fn time_animation_phase(period: std::time::Duration) -> f32 {
    let period = period.as_secs_f64();
    if period <= 0.0 {
        return 0.0;
    }
    (TIME_ANIMATION_EPOCH.elapsed().as_secs_f64() / period).rem_euclid(1.0) as f32
}

/// A one-shot move and fade, evaluated by the scene before every present so
/// what it applies to animates without its view drawing again.
///
/// Everything painted inside [`crate::Window::with_time_transition`] is laid
/// out and painted at rest; the transition offsets it from there by
/// `from_offset` → `to_offset` and scales its opacity `from_opacity` →
/// `to_opacity`, both along `easing` over `duration` from `started_at`. The
/// fade can run over a sub-span of that progress (see
/// [`Self::opacity_span`]). Clipping stays where the element was painted,
/// so content can travel into or out of a clipped slot.
#[derive(Copy, Clone, Debug)]
pub struct TimeTransition {
    started_at: std::time::Instant,
    duration: std::time::Duration,
    from_offset: Point<Pixels>,
    to_offset: Point<Pixels>,
    from_opacity: f32,
    to_opacity: f32,
    opacity_span: (f32, f32),
    easing: fn(f32) -> f32,
}

impl TimeTransition {
    /// A transition of `duration` from `started_at` that, until configured,
    /// neither moves nor fades anything.
    pub fn new(started_at: std::time::Instant, duration: std::time::Duration) -> Self {
        Self {
            started_at,
            duration,
            from_offset: Point::default(),
            to_offset: Point::default(),
            from_opacity: 1.0,
            to_opacity: 1.0,
            opacity_span: (0.0, 1.0),
            easing: |progress| progress,
        }
    }

    /// Travel from `from` to `to`, relative to where the content was painted.
    pub fn offset(mut self, from: Point<Pixels>, to: Point<Pixels>) -> Self {
        self.from_offset = from;
        self.to_offset = to;
        self
    }

    /// Fade from `from` to `to`, multiplied into the painted opacity.
    pub fn opacity(mut self, from: f32, to: f32) -> Self {
        self.from_opacity = from;
        self.to_opacity = to;
        self
    }

    /// Run the fade over this fraction of the transition's progress rather
    /// than all of it, so an outgoing copy can be gone before an incoming one
    /// arrives.
    pub fn opacity_span(mut self, start: f32, end: f32) -> Self {
        self.opacity_span = (start, end);
        self
    }

    /// The curve both the offset and the fade follow, from linear progress in
    /// `0..=1` to eased progress.
    pub fn easing(mut self, easing: fn(f32) -> f32) -> Self {
        self.easing = easing;
        self
    }

    /// Whether the transition has landed by `now` and rests where it was
    /// painted: no offset, full opacity. An easing that does not end at 1,
    /// or a fade span that ends past the transition, leaves the content short
    /// of its endpoints, so it does not rest there.
    pub(crate) fn rests_at_identity_by(&self, now: std::time::Instant) -> bool {
        self.ends_at() <= now
            && self.to_offset == Point::default()
            && self.to_opacity == 1.0
            && self.opacity_span.1 <= 1.0
            && (self.easing)(1.0) == 1.0
    }

    /// When the transition lands.
    pub fn ends_at(&self) -> std::time::Instant {
        self.started_at + self.duration
    }

    /// Linear progress at `now`, in `0..=1`.
    pub fn progress_at(&self, now: std::time::Instant) -> f32 {
        let duration = self.duration.as_secs_f32();
        if duration <= 0.0 {
            return 1.0;
        }
        (now.saturating_duration_since(self.started_at).as_secs_f32() / duration).clamp(0.0, 1.0)
    }

    /// Offset from the painted position at `now`.
    pub fn offset_at(&self, now: std::time::Instant) -> Point<Pixels> {
        let eased = (self.easing)(self.progress_at(now));
        self.from_offset + (self.to_offset - self.from_offset) * eased
    }

    /// Opacity factor at `now`.
    pub fn opacity_at(&self, now: std::time::Instant) -> f32 {
        let (start, end) = self.opacity_span;
        let progress = self.progress_at(now);
        let span_progress = if end > start {
            ((progress - start) / (end - start)).clamp(0.0, 1.0)
        } else if progress >= end {
            1.0
        } else {
            0.0
        };
        let eased = (self.easing)(span_progress);
        self.from_opacity + (self.to_opacity - self.from_opacity) * eased
    }
}

/// A [`TimeTransition`] as the scene applies it: offsets in device pixels.
#[derive(Copy, Clone, Debug)]
pub(crate) struct SceneTransition {
    pub transition: TimeTransition,
    pub scale_factor: f32,
    /// The transition this one was pushed inside, or `0`. Its offset adds to
    /// this one's and its opacity multiplies in, so a label rolling inside a
    /// row that slides moves with both.
    pub parent: u32,
}

impl SceneTransition {
    fn offset_at(&self, now: std::time::Instant) -> Point<ScaledPixels> {
        self.transition.offset_at(now).scale(self.scale_factor)
    }
}

/// The transition id a primitive carries, or `0`.
fn primitive_transition(primitive: &Primitive) -> u32 {
    match primitive {
        Primitive::MonochromeSprite(MonochromeSprite { pad, .. })
        | Primitive::SubpixelSprite(SubpixelSprite { pad, .. })
        | Primitive::PolychromeSprite(PolychromeSprite { pad, .. })
        | Primitive::Underline(Underline { pad, .. })
        | Primitive::Shadow(Shadow { pad, .. }) => *pad,
        Primitive::Quad(quad) => quad.background.time_transition(),
        Primitive::Path(_) | Primitive::Surface(_) => 0,
    }
}

fn set_primitive_transition(primitive: &mut Primitive, transition: u32) {
    match primitive {
        Primitive::MonochromeSprite(MonochromeSprite { pad, .. })
        | Primitive::SubpixelSprite(SubpixelSprite { pad, .. })
        | Primitive::PolychromeSprite(PolychromeSprite { pad, .. })
        | Primitive::Underline(Underline { pad, .. })
        | Primitive::Shadow(Shadow { pad, .. }) => *pad = transition,
        Primitive::Quad(quad) => {
            quad.background = quad.background.with_time_transition(transition);
        }
        Primitive::Path(_) | Primitive::Surface(_) => {}
    }
}

/// A primitive moved by a transition, with what it was painted as so every
/// present computes its state from rest rather than from the last present.
#[derive(Copy, Clone, Debug)]
struct TransitionedPrimitive {
    transition: u32,
    target: TransitionTarget,
}

#[derive(Copy, Clone, Debug)]
enum TransitionTarget {
    MonochromeSprite { index: u32, origin: Point<ScaledPixels>, alpha: f32 },
    SubpixelSprite { index: u32, origin: Point<ScaledPixels>, alpha: f32 },
    PolychromeSprite { index: u32, origin: Point<ScaledPixels>, opacity: f32 },
    Underline { index: u32, origin: Point<ScaledPixels>, alpha: f32 },
    Shadow {
        index: u32,
        origin: Point<ScaledPixels>,
        element_origin: Point<ScaledPixels>,
        alpha: f32,
    },
    Quad {
        index: u32,
        origin: Point<ScaledPixels>,
        background: Background,
        border_color: Hsla,
    },
}

/// A quad whose background and border fade with an [`OpacityCycle`], whose
/// position moves with an [`OffsetCycle`], or both; the colors are the ones
/// it was painted with at full cycle opacity.
#[derive(Copy, Clone, Debug)]
pub(crate) struct QuadAnimation {
    pub opacity: Option<OpacityCycle>,
    pub offset: Option<ScaledOffsetCycle>,
    pub background: Background,
    pub border_color: Hsla,
}

/// A repeating move out to a peak offset and back: a raised cosine over the
/// first `active` fraction of each `period`, at rest for the remainder. A
/// bouncing loader's dots are each one, out of step by their `phase`.
/// Painted inside [`crate::Window::with_offset_cycle`], it moves quads
/// without their view drawing again.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct OffsetCycle {
    period: std::time::Duration,
    phase: f32,
    active: f32,
    peak: Point<Pixels>,
}

impl OffsetCycle {
    /// Out to `peak` and back over the first `active` fraction (clamped to
    /// `0..=1`) of every `period`.
    pub fn bounce(period: std::time::Duration, active: f32, peak: Point<Pixels>) -> Self {
        Self {
            period,
            phase: 0.0,
            active: active.clamp(0.0, 1.0),
            peak,
        }
    }

    /// Offsets this cycle by a fraction of its period: with a negative
    /// phase it starts that much later.
    pub fn phase(mut self, phase: f32) -> Self {
        self.phase = phase;
        self
    }

    /// How far out the cycle is at `phase` of its period, from `0.0` at rest
    /// to `1.0` at the peak.
    pub fn rise_at(&self, phase: f32) -> f32 {
        let phase = (phase + self.phase).rem_euclid(1.0);
        if self.active <= 0.0 || phase >= self.active {
            return 0.0;
        }
        (1.0 - (phase / self.active * std::f32::consts::TAU).cos()) * 0.5
    }

    pub(crate) fn scaled(self, scale_factor: f32) -> ScaledOffsetCycle {
        ScaledOffsetCycle {
            cycle: self,
            peak: self.peak.scale(scale_factor),
        }
    }
}

/// A quad an [`OffsetCycle`] moves; see `Scene::offset_quads`.
#[derive(Copy, Clone, Debug)]
struct OffsetQuad {
    index: u32,
    painted_origin: Point<ScaledPixels>,
    rest_origin: Point<ScaledPixels>,
}

/// An [`OffsetCycle`] with its peak in device pixels, as a scene keeps it.
#[derive(Copy, Clone, Debug, PartialEq)]
pub(crate) struct ScaledOffsetCycle {
    cycle: OffsetCycle,
    peak: Point<ScaledPixels>,
}

impl ScaledOffsetCycle {
    /// Where the cycle has moved its quads now, snapped to whole device
    /// pixels so a moving quad keeps the size it rasterizes at.
    fn current_displacement(&self) -> Point<ScaledPixels> {
        let rise = self.cycle.rise_at(time_animation_phase(self.cycle.period));
        Point::new(
            ScaledPixels((self.peak.x.0 * rise).round()),
            ScaledPixels((self.peak.y.0 * rise).round()),
        )
    }

    /// Every rectangle `bounds` passes through, moved from `painted_origin`
    /// (or where `bounds` is, when that is not known) out to the peak.
    fn swept(
        &self,
        bounds: &Bounds<ScaledPixels>,
        painted_origin: Option<Point<ScaledPixels>>,
    ) -> Bounds<ScaledPixels> {
        let rest = Bounds {
            origin: painted_origin.unwrap_or(bounds.origin),
            size: bounds.size,
        };
        let peak = Bounds {
            origin: rest.origin
                + Point::new(
                    ScaledPixels(self.peak.x.0.round()),
                    ScaledPixels(self.peak.y.0.round()),
                ),
            size: rest.size,
        };
        rest.union(&peak)
    }
}

impl ShimmerAnimation {
    fn band_origin(&self) -> f32 {
        self.band_start + self.travel * crate::elements::shimmer_delta(self.period, self.hold)
    }
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
            animation: 0,
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
    pub tile: AtlasTile,
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub color: Hsla,
    pub effect: SpriteEffect,
    pub transformation: TransformationMatrix,
    pub rounded_clip: u32,
    pub clip_padding: u32,
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
    pub tile: AtlasTile,
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub color: Hsla,
    pub effect: SpriteEffect,
    pub transformation: TransformationMatrix,
    pub rounded_clip: u32,
    pub clip_padding: u32,
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
    pub tile: AtlasTile,
    pub grayscale: PaddedBool32,
    pub opacity: f32,
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub corner_radii: Corners<ScaledPixels>,
    pub rounded_clip: u32,
    pub clip_padding: u32,
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
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    pub image_buffer: objc2_core_foundation::CFRetained<objc2_core_video::CVPixelBuffer>,
    pub rounded_clip: u32,
    pub clip_padding: u32,
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
    pub rounded_clip: u32,
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
            rounded_clip: 0,
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
            rounded_clip: self.rounded_clip,
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
