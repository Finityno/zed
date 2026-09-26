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
    /// Sweeps referenced by `SpriteEffect::animation`.
    shimmer_animations: Vec<ShimmerAnimation>,
    /// Indices of the sprites whose effect animates, gathered by `finish`
    /// (sorting moves sprites, so they cannot be recorded on insertion).
    animated_monochrome_sprites: Vec<u32>,
    animated_subpixel_sprites: Vec<u32>,
    /// Opacity cycles referenced by `Background::time_animation`.
    quad_animations: Vec<QuadOpacityAnimation>,
    animated_quads: Vec<u32>,
    /// Moves referenced by each primitive's transition id (`pad`, or the
    /// background's transition bits for quads).
    transitions: Vec<SceneTransition>,
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
        self.shimmer_animations.clear();
        self.animated_monochrome_sprites.clear();
        self.animated_subpixel_sprites.clear();
        self.quad_animations.clear();
        self.animated_quads.clear();
        self.transitions.clear();
        self.current_transition = 0;
        self.transitioned.clear();
        self.transitions_end_at = None;
    }

    /// Shrinks this cleared scene's vectors to twice the fill of `rendered`,
    /// the scene still on screen, once the window has stopped drawing; see
    /// [`CapacityShrink::idle_target`].
    pub(crate) fn shrink_idle(&mut self, rendered: &Scene) {
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
        paint_operations.shrink_vec_idle(&mut self.paint_operations, rendered.paint_operations.len());
        self.primitive_bounds.shrink_idle(&rendered.primitive_bounds);
        layer_stack.shrink_vec_idle(&mut self.layer_stack, rendered.layer_stack.len());
        paths.shrink_vec_idle(&mut self.paths, rendered.paths.len());
        shadows.shrink_vec_idle(&mut self.shadows, rendered.shadows.len());
        quads.shrink_vec_idle(&mut self.quads, rendered.quads.len());
        underlines.shrink_vec_idle(&mut self.underlines, rendered.underlines.len());
        monochrome_sprites
            .shrink_vec_idle(&mut self.monochrome_sprites, rendered.monochrome_sprites.len());
        subpixel_sprites.shrink_vec_idle(&mut self.subpixel_sprites, rendered.subpixel_sprites.len());
        polychrome_sprites
            .shrink_vec_idle(&mut self.polychrome_sprites, rendered.polychrome_sprites.len());
        surfaces.shrink_vec_idle(&mut self.surfaces, rendered.surfaces.len());
        blended_quad_indices
            .shrink_vec_idle(&mut self.blended_quad_indices, rendered.blended_quad_indices.len());
        opaque_quad_indices
            .shrink_vec_idle(&mut self.opaque_quad_indices, rendered.opaque_quad_indices.len());
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

    /// Registers an opacity cycle for one quad, returning the value its
    /// background's time animation should carry.
    pub(crate) fn push_quad_animation(&mut self, animation: QuadOpacityAnimation) -> u32 {
        self.quad_animations.push(animation);
        self.quad_animations.len() as u32
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
    fn remap_transition(
        &mut self,
        prev_scene: &Scene,
        transition: u32,
        remapped: &mut Vec<(u32, u32)>,
    ) -> u32 {
        if let Some(&(_, id)) = remapped.iter().find(|(source, _)| *source == transition) {
            return id;
        }
        let mut entry = prev_scene.transitions[transition as usize - 1];
        if entry.parent != 0 {
            entry.parent = self.remap_transition(prev_scene, entry.parent, remapped);
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

    pub fn replay(&mut self, range: Range<usize>, prev_scene: &Scene) {
        // Every glyph of one shimmering label names the same sweep, so the
        // label's glyphs share one remapped entry rather than one each.
        let mut remapped_animation = (0, 0);
        let mut remapped_transitions: Vec<(u32, u32)> = Vec::new();
        for operation in &prev_scene.paint_operations[range] {
            match operation {
                PaintOperation::Primitive(primitive) => {
                    let mut primitive = primitive.clone();
                    let transition = primitive_transition(&primitive);
                    if transition != 0 {
                        let remapped =
                            self.remap_transition(prev_scene, transition, &mut remapped_transitions);
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
                            let animation =
                                prev_scene.shimmer_animations[effect.animation as usize - 1];
                            remapped_animation =
                                (effect.animation, self.push_shimmer_animation(animation));
                        }
                        effect.animation = remapped_animation.1;
                    }
                    self.insert_primitive(primitive)
                }
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
        self.animated_monochrome_sprites.clear();
        self.animated_monochrome_sprites.extend(
            (0..self.monochrome_sprites.len() as u32)
                .filter(|&index| self.monochrome_sprites[index as usize].effect.animation != 0),
        );
        self.animated_subpixel_sprites.clear();
        self.animated_subpixel_sprites.extend(
            (0..self.subpixel_sprites.len() as u32)
                .filter(|&index| self.subpixel_sprites[index as usize].effect.animation != 0),
        );
        self.animated_quads.clear();
        self.animated_quads.extend(
            (0..self.quads.len() as u32)
                .filter(|&index| self.quads[index as usize].background.time_animation() != 0),
        );
        self.gather_transitioned();
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
            || !self.animated_subpixel_sprites.is_empty()
            || !self.animated_quads.is_empty()
            || self.transitions_in_flight()
    }

    /// Moves every time-driven primitive of this finished scene to where it is
    /// now. Called before each present, so a frame replayed from a cached view
    /// shows the current phase rather than the one it was painted at.
    pub(crate) fn advance_time_animations(&mut self) {
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
            let opacity = animation.cycle.current_opacity();
            quad.background = animation
                .background
                .opacity(opacity)
                .with_time_animation(animation_id);
            quad.border_color = animation.border_color.opacity(opacity);
        }
        self.advance_transitions(std::time::Instant::now());
    }

    fn advance_transitions(&mut self, now: std::time::Instant) {
        if self.transitioned.is_empty() {
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
        let partitioning_enabled =
            opaque_quad_partitioning_enabled() && self.quads.len() <= MAX_DEPTH_PARTITIONED_QUADS;
        for (quad_id, quad) in self.quads.iter().enumerate() {
            // A quad whose opacity animates may be solid in this frame and
            // translucent in the next without the scene being rebuilt, so it
            // always takes the blended pass; so does one a transition moves
            // or fades.
            let has_opaque_core = partitioning_enabled
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

    fn shimmering_glyph(animation: u32) -> MonochromeSprite {
        MonochromeSprite {
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
        let animation = scene.push_quad_animation(QuadOpacityAnimation {
            cycle: OpacityCycle::new(
                std::time::Duration::from_millis(750),
                [(0.0, 0.5), (0.5, 0.5), (0.75, 0.5), (1.0, 0.5)],
            ),
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
    /// One more than the index of the [`ShimmerAnimation`] in the owning
    /// scene that moves `band_origin` with time, or `0` for a still band. The
    /// shaders never read it (they see it as padding); the scene rewrites
    /// `band_origin` from it before every present, so a sweeping shimmer
    /// animates without its view drawing again.
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

/// A quad whose background and border fade with an [`OpacityCycle`]; the
/// colors are the ones it was painted with at full cycle opacity.
#[derive(Copy, Clone, Debug)]
pub(crate) struct QuadOpacityAnimation {
    pub cycle: OpacityCycle,
    pub background: Background,
    pub border_color: Hsla,
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
