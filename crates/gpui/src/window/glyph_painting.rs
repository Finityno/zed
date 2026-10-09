//! Painting glyphs a line at a time. What a glyph's sprite depends on that is
//! fixed for its line (the snapped content mask, the text effect, the element
//! opacity) is worked out once per line, and what is fixed for its run (the
//! subpixel decision and the dilation of its colour) once per run. The window
//! also keeps the raster bounds of recently painted glyphs, their atlas tiles
//! for the draw they were looked up in, and the extents of recent fonts, so a
//! glyph does not lock and hash its way through the text system and the atlas.

use super::Window;
use crate::{
    AtlasTile, Bounds, ContentMask, DecorationRun, DevicePixels, FontId, GlyphId, Hsla, IsZero,
    LineLayout, MonochromeSprite, Pixels, Point, RenderGlyphParams, ScaledPixels, SpriteEffect,
    SubpixelSprite, TextAlign, TextRenderingMode, TransformationMatrix,
    WindowBackgroundAppearance, WrappedLineLayout, scene::GlyphWatermark,
};
use anyhow::Result;
use collections::{FxBuildHasher, FxHashMap};
use smallvec::SmallVec;
use std::{borrow::Cow, hash::BuildHasher, mem, ops::Range, sync::Arc};

/// How the glyphs of one run are rasterized: the part of a glyph's
/// [`RenderGlyphParams`] that depends on its font, size and colour rather
/// than on the glyph.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct GlyphRunRendering {
    subpixel_rendering: bool,
    dilation: u8,
}

/// Paints the glyphs of one line, working out what they share once: the
/// snapped content mask, text effect and element opacity for the line, and
/// the rendering of each run while its font, size and colour stay the same.
pub(crate) struct LineGlyphPainter {
    content_mask: ContentMask<ScaledPixels>,
    effect: SpriteEffect,
    element_opacity: f32,
    run: Option<GlyphRun>,
    operation_floor: Option<usize>,
}

/// What the glyphs of a run share while its font, size and colour stay the
/// same.
#[derive(Clone, Copy)]
struct GlyphRun {
    font_id: FontId,
    font_size: Pixels,
    color: Hsla,
    rendering: GlyphRunRendering,
    slot_seed: u64,
    /// `color` with the element opacity applied, as its sprites are painted.
    sprite_color: Hsla,
}

impl LineGlyphPainter {
    /// Takes what the window is painting with now; nothing painting a line
    /// changes it until the line is done.
    pub(crate) fn new(window: &Window) -> Self {
        Self {
            content_mask: window.snapped_content_mask(),
            effect: window.current_text_effect(),
            element_opacity: window.element_opacity(),
            run: None,
            operation_floor: None,
        }
    }

    /// Starts a line segment in which no callback can publish a paint index.
    pub(crate) fn for_line(window: &Window) -> Self {
        let mut painter = Self::new(window);
        painter.operation_floor = Some(window.next_frame.scene.len());
        painter
    }

    /// A decoration callback may capture retained indices without painting
    /// anything; records it observed must never grow after the callback.
    pub(crate) fn resume_after_callback(&mut self, window: &Window) {
        self.operation_floor = Some(window.next_frame.scene.len());
    }

    /// Paints a monochrome glyph, as [`Window::paint_glyph`] does.
    pub(crate) fn paint_glyph(
        &mut self,
        window: &mut Window,
        origin: Point<Pixels>,
        font_id: FontId,
        glyph_id: GlyphId,
        font_size: Pixels,
        color: Hsla,
    ) -> Result<()> {
        let run = match self.run {
            Some(run)
                if run.font_id == font_id && run.font_size == font_size && run.color == color =>
            {
                run
            }
            _ => {
                let rendering = window.glyph_run_rendering(font_id, font_size, color);
                let run = GlyphRun {
                    font_id,
                    font_size,
                    color,
                    rendering,
                    slot_seed: GlyphRasterCache::slot_seed(
                        font_id,
                        font_size,
                        window.scale_factor(),
                        false,
                        rendering,
                    ),
                    sprite_color: color.opacity(self.element_opacity),
                };
                self.run = Some(run);
                run
            }
        };
        window.paint_glyph_in_run(self, &run, origin, glyph_id)
    }
}

/// A font's bounding box and descent at one size.
#[derive(Clone, Copy, Debug)]
pub(crate) struct FontExtents {
    pub(crate) bounding_box: Bounds<Pixels>,
    pub(crate) descent: Pixels,
}

/// How many glyphs' raster bounds a window keeps, as a power of two.
const GLYPH_SLOT_BITS: u32 = 11;

/// How many fonts' extents a window keeps, most recently used last.
const FONT_EXTENTS_KEPT: usize = 16;

/// What a window keeps of the glyphs and fonts it painted lately.
///
/// A glyph's raster bounds depend only on its [`RenderGlyphParams`], and the
/// text system remembers them while they are drawn (except the empty bounds of a glyph
/// that has ink, which it asks again; those are not kept here either). Each
/// glyph has one slot, chosen by a hash of its parameters; a glyph needing a
/// slot another holds takes it over.
///
/// A slot also keeps the glyph's atlas tile, but only for the draw it was
/// looked up in: the same glyph is painted hundreds of times a frame, and
/// the atlas never drops a tile while a window paints, but it may retire idle
/// tiles, or lose them all with its device, when a frame is presented.
#[derive(Default)]
pub(crate) struct GlyphRasterCache {
    /// Empty until the window paints its first glyph.
    slots: Vec<Option<GlyphSlot>>,
    /// Counts finished draws, telling a tile looked up in this draw from one
    /// looked up in an earlier one.
    draw: u64,
    font_extents: Vec<(FontId, Pixels, FontExtents)>,
    /// Counts glyphs whose raster bounds the text system did not remember,
    /// which are asked again every time they are painted.
    unremembered_glyphs: u64,
}

#[derive(Clone)]
struct GlyphSlot {
    params: RenderGlyphParams,
    raster_bounds: Bounds<DevicePixels>,
    /// The glyph's tile, and the draw it was looked up in.
    tile: Option<(u64, AtlasTile)>,
}

impl GlyphRasterCache {
    /// Hashes what a glyph's parameters share with the rest of its run, so
    /// that finding a glyph's slot only mixes in its id and subpixel variant.
    fn slot_seed(
        font_id: FontId,
        font_size: Pixels,
        scale_factor: f32,
        is_emoji: bool,
        rendering: GlyphRunRendering,
    ) -> u64 {
        FxBuildHasher.hash_one((
            font_id.0,
            font_size.0.to_bits(),
            scale_factor.to_bits(),
            is_emoji,
            rendering.subpixel_rendering,
            rendering.dilation,
        ))
    }

    /// Fibonacci hashing of the glyph's id and variant over its run's seed:
    /// a multiply and a shift, where hashing every field of the parameters
    /// cost a round per field, for every glyph painted.
    fn slot_in_run(seed: u64, glyph_id: GlyphId, subpixel_variant: Point<u8>) -> usize {
        let glyph = (glyph_id.0 as u64) << 16
            | (subpixel_variant.x as u64) << 8
            | subpixel_variant.y as u64;
        ((seed ^ glyph).wrapping_mul(0x9e37_79b9_7f4a_7c15) >> (u64::BITS - GLYPH_SLOT_BITS))
            as usize
    }

    #[cfg(test)]
    fn slot(params: &RenderGlyphParams) -> usize {
        Self::slot_in_run(
            Self::slot_seed(
                params.font_id,
                params.font_size,
                params.scale_factor,
                params.is_emoji,
                GlyphRunRendering {
                    subpixel_rendering: params.subpixel_rendering,
                    dilation: params.dilation,
                },
            ),
            params.glyph_id,
            params.subpixel_variant,
        )
    }

    /// The glyph's raster bounds if kept in `slot`, with its tile if it was
    /// looked up in this draw.
    fn lookup(
        &self,
        slot: usize,
        params: &RenderGlyphParams,
    ) -> Option<(Bounds<DevicePixels>, Option<AtlasTile>)> {
        match self.slots.get(slot) {
            Some(Some(slot)) if slot.params == *params => Some((
                slot.raster_bounds,
                slot.tile
                    .and_then(|(draw, tile)| (draw == self.draw).then_some(tile)),
            )),
            _ => None,
        }
    }

    fn insert(
        &mut self,
        slot: usize,
        params: &RenderGlyphParams,
        raster_bounds: Bounds<DevicePixels>,
    ) {
        if self.slots.is_empty() {
            self.slots.resize(1 << GLYPH_SLOT_BITS, None);
        }
        self.slots[slot] = Some(GlyphSlot {
            params: params.clone(),
            raster_bounds,
            tile: None,
        });
    }

    fn insert_tile(&mut self, slot: usize, params: &RenderGlyphParams, tile: AtlasTile) {
        if let Some(Some(slot)) = self.slots.get_mut(slot)
            && slot.params == *params
        {
            slot.tile = Some((self.draw, tile));
        }
    }

    /// Ends the draw the kept tiles were looked up in.
    pub(crate) fn finish_draw(&mut self) {
        self.draw += 1;
    }
}

/// The layout a painted line was shaped into, held so that a line painted
/// again is told apart from another that took over its allocation.
#[derive(Clone)]
pub(crate) enum LineGlyphsLayout {
    Shaped(Arc<LineLayout>),
    Wrapped(Arc<WrappedLineLayout>),
}

impl LineGlyphsLayout {
    fn address(&self) -> usize {
        match self {
            Self::Shaped(layout) => Arc::as_ptr(layout) as usize,
            Self::Wrapped(layout) => Arc::as_ptr(layout) as usize,
        }
    }

    fn same(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Shaped(this), Self::Shaped(other)) => Arc::ptr_eq(this, other),
            (Self::Wrapped(this), Self::Wrapped(other)) => Arc::ptr_eq(this, other),
            _ => false,
        }
    }
}

/// Everything the glyph sprites of a line without underlines or
/// strikethroughs depend on: its layout (and wrapping), where and how it is
/// placed, the colour of each run, and what the window paints glyphs with.
/// Two lines that agree on all of it paint the same sprites.
pub(crate) struct LineGlyphsKey {
    layout: LineGlyphsLayout,
    origin: (u32, u32),
    line_height: u32,
    align: TextAlign,
    align_width: Option<u32>,
    colors: SmallVec<[(u32, Hsla); 8]>,
    /// As painting asked for it, before snapping: painting culls by both.
    content_mask: ContentMask<Pixels>,
    element_opacity: u32,
    scale_factor: u32,
    rendering: (WindowBackgroundAppearance, bool, TextRenderingMode),
}

impl LineGlyphsKey {
    fn slot(&self) -> (usize, (u32, u32)) {
        (self.layout.address(), self.origin)
    }

    fn matches(&self, other: &Self) -> bool {
        self.layout.same(&other.layout)
            && self.origin == other.origin
            && self.line_height == other.line_height
            && self.align == other.align
            && self.align_width == other.align_width
            && self.content_mask == other.content_mask
            && self.element_opacity == other.element_opacity
            && self.scale_factor == other.scale_factor
            && self.rendering == other.rendering
            && self.colors.len() == other.colors.len()
            && self
                .colors
                .iter()
                .zip(&other.colors)
                .all(|((len, color), (other_len, other_color))| {
                    len == other_len && same_color(color, other_color)
                })
    }
}

fn same_color(this: &Hsla, other: &Hsla) -> bool {
    this.h.to_bits() == other.h.to_bits()
        && this.s.to_bits() == other.s.to_bits()
        && this.l.to_bits() == other.l.to_bits()
        && this.a.to_bits() == other.a.to_bits()
}

/// A line whose glyphs are being painted, to be kept for the next frame once
/// they are.
pub(crate) struct LineGlyphsRecording {
    key: LineGlyphsKey,
    start: GlyphWatermark,
    unremembered_glyphs: u64,
}

struct PaintedLineGlyphs {
    key: LineGlyphsKey,
    /// The line's sprites among the paint operations of the frame it was
    /// painted in.
    operations: Range<usize>,
}

/// The glyph sprites of the lines painted in the last frame, by layout and
/// origin, and those of the lines painted so far in this one.
///
/// Painting a line again where it was, unchanged, in a window painting glyphs
/// as it was, paints the same sprites, so they are copied from the frame the
/// window last drew rather than worked out a glyph at a time. That frame's
/// tiles are as live as a cached view's replayed ones: the window draws
/// without either once its rendered frame is old enough for the atlas to
/// have retired them.
#[derive(Default)]
pub(crate) struct LineGlyphCache {
    previous: FxHashMap<(usize, (u32, u32)), PaintedLineGlyphs>,
    current: FxHashMap<(usize, (u32, u32)), PaintedLineGlyphs>,
    /// The atlas generation the last frame's tiles were handed out in.
    atlas_generation: u64,
    #[cfg(test)]
    pub(crate) replayed_lines: usize,
}

impl LineGlyphCache {
    /// Ends a draw: the lines it painted become those the next one can
    /// paint again.
    pub(crate) fn finish_draw(&mut self) {
        mem::swap(&mut self.previous, &mut self.current);
        self.current.clear();
    }

    /// Forgets the lines of the last frame, whose sprites may name tiles the
    /// atlas has retired.
    pub(crate) fn forget_previous(&mut self) {
        self.previous.clear();
    }

    /// Starts a draw with the atlas at `atlas_generation`, forgetting the
    /// lines of the last frame if the atlas dropped their tiles since.
    pub(crate) fn start_draw(&mut self, atlas_generation: u64) {
        if self.atlas_generation != atlas_generation {
            self.atlas_generation = atlas_generation;
            self.forget_previous();
        }
    }
}

impl Window {
    /// What the glyph sprites of a line depend on, or `None` when it cannot
    /// be painted again from the last frame: one with an underline or
    /// strikethrough, or one painted with a text effect.
    pub(crate) fn line_glyphs_key(
        &self,
        layout: LineGlyphsLayout,
        origin: Point<Pixels>,
        line_height: Pixels,
        align: TextAlign,
        align_width: Option<Pixels>,
        decoration_runs: &[DecorationRun],
    ) -> Option<LineGlyphsKey> {
        if !self.text_shimmer_stack.is_empty() && !super::text_shimmer_disabled() {
            return None;
        }
        let mut colors = SmallVec::new();
        for run in decoration_runs {
            if run.underline.is_some() || run.strikethrough.is_some() {
                return None;
            }
            colors.push((run.len, run.color));
        }
        Some(LineGlyphsKey {
            layout,
            origin: (origin.x.0.to_bits(), origin.y.0.to_bits()),
            line_height: line_height.0.to_bits(),
            align,
            align_width: align_width.map(|width| width.0.to_bits()),
            colors,
            content_mask: self.content_mask(),
            element_opacity: self.element_opacity().to_bits(),
            scale_factor: self.scale_factor().to_bits(),
            rendering: (
                self.platform_window.background_appearance(),
                self.platform_window.is_subpixel_rendering_supported(),
                self.text_rendering_mode.get(),
            ),
        })
    }

    /// Paints the glyphs of the line `key` describes by copying the sprites
    /// the last frame painted for it, if it painted that line. Returns false,
    /// having painted nothing, when it did not.
    pub(crate) fn replay_line_glyphs(&mut self, key: &LineGlyphsKey) -> bool {
        let slot = key.slot();
        let Some(mut painted) = self.line_glyph_cache.previous.remove(&slot) else {
            return false;
        };
        if !painted.key.matches(key) {
            return false;
        }
        let start = self.next_frame.scene.glyph_watermark();
        if !self
            .next_frame
            .scene
            .replay_glyph_sprites(&self.rendered_frame.scene, painted.operations.clone())
        {
            return false;
        }
        let Some(operations) = self.next_frame.scene.glyph_sprites_since(start) else {
            return true;
        };
        painted.operations = operations;
        self.line_glyph_cache.current.insert(slot, painted);
        #[cfg(test)]
        {
            self.line_glyph_cache.replayed_lines += 1;
        }
        true
    }

    /// Starts painting the glyphs of the line `key` describes glyph by glyph.
    pub(crate) fn record_line_glyphs(&self, key: LineGlyphsKey) -> LineGlyphsRecording {
        LineGlyphsRecording {
            key,
            start: self.next_frame.scene.glyph_watermark(),
            unremembered_glyphs: self.glyph_raster_cache.unremembered_glyphs,
        }
    }

    /// Keeps the sprites a line painted since [`Self::record_line_glyphs`]
    /// for the next frame to paint again, unless it painted something other
    /// than glyph sprites, or a glyph whose raster bounds are asked again
    /// each time it is painted.
    pub(crate) fn finish_line_glyphs(&mut self, recording: LineGlyphsRecording) {
        if recording.unremembered_glyphs != self.glyph_raster_cache.unremembered_glyphs {
            return;
        }
        let Some(operations) = self.next_frame.scene.glyph_sprites_since(recording.start) else {
            return;
        };
        self.line_glyph_cache.current.insert(
            recording.key.slot(),
            PaintedLineGlyphs {
                key: recording.key,
                operations,
            },
        );
    }

    /// The extents of `font_id` at `font_size`, which painting a line asks
    /// for once a run: the text system takes a lock and hashes the font to
    /// find them, where a window paints in a handful of fonts and sizes.
    pub(crate) fn font_extents(&mut self, font_id: FontId, font_size: Pixels) -> FontExtents {
        let kept = &mut self.glyph_raster_cache.font_extents;
        if let Some(position) = kept
            .iter()
            .rposition(|(id, size, _)| *id == font_id && *size == font_size)
        {
            let extents = kept[position].2;
            // Most recently used last, so the one pushed out is the one used
            // least recently.
            if position + 1 != kept.len() {
                let entry = kept.remove(position);
                kept.push(entry);
            }
            return extents;
        }
        let text_system = self.text_system();
        let extents = FontExtents {
            bounding_box: text_system.bounding_box(font_id, font_size),
            descent: text_system.descent(font_id, font_size),
        };
        let kept = &mut self.glyph_raster_cache.font_extents;
        if kept.len() == FONT_EXTENTS_KEPT {
            kept.remove(0);
        }
        kept.push((font_id, font_size, extents));
        extents
    }

    fn glyph_run_rendering(
        &self,
        font_id: FontId,
        font_size: Pixels,
        color: Hsla,
    ) -> GlyphRunRendering {
        GlyphRunRendering {
            subpixel_rendering: self.should_use_subpixel_rendering(font_id, font_size),
            dilation: self.text_system().glyph_dilation_for_color(color),
        }
    }

    fn should_use_subpixel_rendering(&self, font_id: FontId, font_size: Pixels) -> bool {
        if self.platform_window.background_appearance() != WindowBackgroundAppearance::Opaque {
            return false;
        }

        if !self.platform_window.is_subpixel_rendering_supported() {
            return false;
        }

        let mode = match self.text_rendering_mode.get() {
            TextRenderingMode::PlatformDefault => self
                .text_system()
                .recommended_rendering_mode(font_id, font_size),
            mode => mode,
        };

        mode == TextRenderingMode::Subpixel
    }

    fn paint_glyph_in_run(
        &mut self,
        line: &LineGlyphPainter,
        run: &GlyphRun,
        origin: Point<Pixels>,
        glyph_id: GlyphId,
    ) -> Result<()> {
        self.invalidator.debug_assert_paint();

        let scale_factor = self.scale_factor();
        let (integer_origin, subpixel_variant) =
            super::quantize_glyph_origin(origin.scale(scale_factor));
        let GlyphRunRendering {
            subpixel_rendering,
            dilation,
        } = run.rendering;
        let params = RenderGlyphParams {
            font_id: run.font_id,
            glyph_id,
            font_size: run.font_size,
            subpixel_variant,
            scale_factor,
            is_emoji: false,
            subpixel_rendering,
            dilation,
        };

        let slot = GlyphRasterCache::slot_in_run(run.slot_seed, glyph_id, subpixel_variant);
        let (raster_bounds, tile) = match self.glyph_raster_cache.lookup(slot, &params) {
            Some(kept) => kept,
            None => {
                let (raster_bounds, remembered) =
                    self.text_system().remembered_raster_bounds(&params)?;
                if remembered {
                    self.glyph_raster_cache.insert(slot, &params, raster_bounds);
                } else {
                    self.glyph_raster_cache.unremembered_glyphs += 1;
                }
                (raster_bounds, None)
            }
        };
        if raster_bounds.is_zero() {
            return Ok(());
        }
        let tile = match tile {
            Some(tile) => tile,
            None => {
                let tile = self
                    .sprite_atlas
                    .get_or_insert_with(params.clone().into(), &mut || {
                        let (size, bytes) = self.text_system().rasterize_glyph(&params)?;
                        Ok(Some((size, Cow::Owned(bytes))))
                    })?
                    .expect("Callback above only errors or returns Some");
                self.glyph_raster_cache.insert_tile(slot, &params, tile);
                tile
            }
        };
        let bounds = Bounds {
            origin: integer_origin + raster_bounds.origin.map(Into::into),
            size: tile.bounds.size.map(Into::into),
        };
        let color = run.sprite_color;
        if subpixel_rendering {
            let sprite = SubpixelSprite {
                order: 0,
                pad: 0,
                bounds,
                content_mask: line.content_mask,
                color,
                effect: line.effect,
                tile,
                transformation: TransformationMatrix::unit(),
            };
            if let Some(floor) = line.operation_floor {
                self.next_frame.scene.insert_line_subpixel_sprite(sprite, floor);
            } else {
                self.next_frame.scene.insert_subpixel_sprite(sprite);
            }
        } else {
            let sprite = MonochromeSprite {
                order: 0,
                pad: 0,
                bounds,
                content_mask: line.content_mask,
                color,
                effect: line.effect,
                tile,
                transformation: TransformationMatrix::unit(),
            };
            if let Some(floor) = line.operation_floor {
                self.next_frame.scene.insert_line_monochrome_sprite(sprite, floor);
            } else {
                self.next_frame.scene.insert_monochrome_sprite(sprite);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AtlasTextureId, AtlasTextureKind, Size, TileId, point, px};

    fn params(glyph: u32) -> RenderGlyphParams {
        RenderGlyphParams {
            font_id: FontId(1),
            glyph_id: GlyphId(glyph),
            font_size: px(13.),
            subpixel_variant: point(1, 0),
            scale_factor: 2.,
            is_emoji: false,
            subpixel_rendering: false,
            dilation: 0,
        }
    }

    fn tile(index: u32) -> AtlasTile {
        AtlasTile {
            texture_id: AtlasTextureId {
                index,
                kind: AtlasTextureKind::Monochrome,
            },
            tile_id: TileId(index),
            padding: 0,
            bounds: Bounds::default(),
        }
    }

    fn bounds(width: i32) -> Bounds<DevicePixels> {
        Bounds {
            origin: Point::default(),
            size: Size {
                width: DevicePixels(width),
                height: DevicePixels(10),
            },
        }
    }

    /// A glyph's raster bounds are kept across draws, its tile only for the
    /// draw it was looked up in, and a glyph is told apart from one whose
    /// parameters differ only in how it is rendered.
    #[test]
    fn raster_bounds_outlive_a_draw_and_tiles_do_not() {
        let mut cache = GlyphRasterCache::default();
        let glyph = params(7);
        let slot = GlyphRasterCache::slot(&glyph);
        assert!(cache.lookup(slot, &glyph).is_none());

        cache.insert(slot, &glyph, bounds(5));
        cache.insert_tile(slot, &glyph, tile(3));
        assert_eq!(cache.lookup(slot, &glyph), Some((bounds(5), Some(tile(3)))));

        let mut dilated = glyph.clone();
        dilated.dilation = 1;
        assert!(cache.lookup(GlyphRasterCache::slot(&dilated), &dilated).is_none());
        // Even in the slot the other glyph holds.
        assert!(cache.lookup(slot, &dilated).is_none());

        cache.finish_draw();
        assert_eq!(cache.lookup(slot, &glyph), Some((bounds(5), None)));
    }

    /// A glyph whose slot another took over is looked up afresh rather than
    /// answered with the other's bounds.
    #[test]
    fn a_glyph_whose_slot_was_taken_is_not_found() {
        let mut cache = GlyphRasterCache::default();
        let first = params(1);
        let Some(second) = (2..100_000)
            .map(params)
            .find(|other| GlyphRasterCache::slot(other) == GlyphRasterCache::slot(&first))
        else {
            panic!("some glyph shares a slot among so many");
        };
        let slot = GlyphRasterCache::slot(&first);
        cache.insert(slot, &first, bounds(1));
        cache.insert(slot, &second, bounds(2));
        assert!(cache.lookup(slot, &first).is_none());
        assert_eq!(cache.lookup(slot, &second), Some((bounds(2), None)));
    }
}
