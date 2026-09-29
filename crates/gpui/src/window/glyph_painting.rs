//! Painting glyphs a line at a time. What a glyph's sprite depends on that is
//! fixed for its line (the snapped content mask, the text effect, the element
//! opacity) is worked out once per line, and what is fixed for its run (the
//! subpixel decision and the dilation of its colour) once per run. The window
//! also keeps the raster bounds of recently painted glyphs, their atlas tiles
//! for the draw they were looked up in, and the extents of recent fonts, so a
//! glyph does not lock and hash its way through the text system and the atlas.

use super::Window;
use crate::{
    AtlasTile, Bounds, ContentMask, DevicePixels, FontId, GlyphId, Hsla, IsZero, MonochromeSprite,
    Pixels, Point, RenderGlyphParams, ScaledPixels, SpriteEffect, SubpixelSprite,
    TransformationMatrix, TextRenderingMode, WindowBackgroundAppearance,
};
use anyhow::Result;
use collections::FxBuildHasher;
use std::{borrow::Cow, hash::BuildHasher};

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
    run: Option<(FontId, Pixels, Hsla, GlyphRunRendering)>,
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
        }
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
        let rendering = match self.run {
            Some((run_font_id, run_font_size, run_color, rendering))
                if run_font_id == font_id && run_font_size == font_size && run_color == color =>
            {
                rendering
            }
            _ => {
                let rendering = window.glyph_run_rendering(font_id, font_size, color);
                self.run = Some((font_id, font_size, color, rendering));
                rendering
            }
        };
        window.paint_glyph_in_run(self, rendering, origin, font_id, glyph_id, font_size, color)
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
/// text system remembers them forever (except the empty bounds of a glyph
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
}

#[derive(Clone)]
struct GlyphSlot {
    params: RenderGlyphParams,
    raster_bounds: Bounds<DevicePixels>,
    /// The glyph's tile, and the draw it was looked up in.
    tile: Option<(u64, AtlasTile)>,
}

impl GlyphRasterCache {
    fn slot(params: &RenderGlyphParams) -> usize {
        (FxBuildHasher.hash_one(params) >> (u64::BITS - GLYPH_SLOT_BITS)) as usize
    }

    /// The glyph's raster bounds if kept, with its tile if it was looked up
    /// in this draw.
    fn lookup(
        &self,
        params: &RenderGlyphParams,
    ) -> Option<(Bounds<DevicePixels>, Option<AtlasTile>)> {
        match self.slots.get(Self::slot(params)) {
            Some(Some(slot)) if slot.params == *params => Some((
                slot.raster_bounds,
                slot.tile
                    .and_then(|(draw, tile)| (draw == self.draw).then_some(tile)),
            )),
            _ => None,
        }
    }

    fn insert(&mut self, params: &RenderGlyphParams, raster_bounds: Bounds<DevicePixels>) {
        if self.slots.is_empty() {
            self.slots.resize(1 << GLYPH_SLOT_BITS, None);
        }
        self.slots[Self::slot(params)] = Some(GlyphSlot {
            params: params.clone(),
            raster_bounds,
            tile: None,
        });
    }

    fn insert_tile(&mut self, params: &RenderGlyphParams, tile: AtlasTile) {
        if let Some(Some(slot)) = self.slots.get_mut(Self::slot(params))
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

impl Window {
    /// The extents of `font_id` at `font_size`, which painting a line asks
    /// for once a run: the text system takes a lock and hashes the font to
    /// find them, where a window paints in a handful of fonts and sizes.
    pub(crate) fn font_extents(&mut self, font_id: FontId, font_size: Pixels) -> FontExtents {
        let kept = &mut self.glyph_raster_cache.font_extents;
        if let Some((_, _, extents)) = kept
            .iter()
            .rev()
            .find(|(id, size, _)| *id == font_id && *size == font_size)
        {
            return *extents;
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

    #[allow(clippy::too_many_arguments)]
    fn paint_glyph_in_run(
        &mut self,
        line: &LineGlyphPainter,
        rendering: GlyphRunRendering,
        origin: Point<Pixels>,
        font_id: FontId,
        glyph_id: GlyphId,
        font_size: Pixels,
        color: Hsla,
    ) -> Result<()> {
        self.invalidator.debug_assert_paint();

        let scale_factor = self.scale_factor();
        let (integer_origin, subpixel_variant) =
            super::quantize_glyph_origin(origin.scale(scale_factor));
        let GlyphRunRendering {
            subpixel_rendering,
            dilation,
        } = rendering;
        let params = RenderGlyphParams {
            font_id,
            glyph_id,
            font_size,
            subpixel_variant,
            scale_factor,
            is_emoji: false,
            subpixel_rendering,
            dilation,
        };

        let (raster_bounds, tile) = match self.glyph_raster_cache.lookup(&params) {
            Some(kept) => kept,
            None => {
                let (raster_bounds, remembered) =
                    self.text_system().remembered_raster_bounds(&params)?;
                if remembered {
                    self.glyph_raster_cache.insert(&params, raster_bounds);
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
                self.glyph_raster_cache.insert_tile(&params, tile);
                tile
            }
        };
        let bounds = Bounds {
            origin: integer_origin + raster_bounds.origin.map(Into::into),
            size: tile.bounds.size.map(Into::into),
        };
        let color = color.opacity(line.element_opacity);
        if subpixel_rendering {
            self.next_frame.scene.insert_primitive(SubpixelSprite {
                order: 0,
                pad: 0,
                bounds,
                content_mask: line.content_mask,
                color,
                effect: line.effect,
                tile,
                transformation: TransformationMatrix::unit(),
            });
        } else {
            self.next_frame.scene.insert_primitive(MonochromeSprite {
                order: 0,
                pad: 0,
                bounds,
                content_mask: line.content_mask,
                color,
                effect: line.effect,
                tile,
                transformation: TransformationMatrix::unit(),
            });
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
        assert!(cache.lookup(&glyph).is_none());

        cache.insert(&glyph, bounds(5));
        cache.insert_tile(&glyph, tile(3));
        assert_eq!(cache.lookup(&glyph), Some((bounds(5), Some(tile(3)))));

        let mut dilated = glyph.clone();
        dilated.dilation = 1;
        assert!(cache.lookup(&dilated).is_none());

        cache.finish_draw();
        assert_eq!(cache.lookup(&glyph), Some((bounds(5), None)));
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
        cache.insert(&first, bounds(1));
        cache.insert(&second, bounds(2));
        assert!(cache.lookup(&first).is_none());
        assert_eq!(cache.lookup(&second), Some((bounds(2), None)));
    }
}
