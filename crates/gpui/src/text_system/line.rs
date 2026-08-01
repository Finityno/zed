use crate::{
    App, Bounds, DevicePixels, Half, Hsla, LineLayout, Pixels, Point, RenderGlyphParams, Result,
    SharedString, StrikethroughStyle, TextAlign, UnderlineStyle, Window, WrapBoundary,
    WrappedLineLayout, black, fill, point, px, size,
};
use derive_more::{Deref, DerefMut};
use smallvec::SmallVec;
use std::sync::Arc;

/// Pre-computed glyph data for efficient painting without per-glyph cache lookups.
///
/// This is produced by `ShapedLine::compute_glyph_raster_data` during prepaint
/// and consumed by `ShapedLine::paint_with_raster_data` during paint.
#[derive(Clone, Debug)]
pub struct GlyphRasterData {
    /// The raster bounds for each glyph, in paint order.
    pub bounds: Vec<Bounds<DevicePixels>>,
    /// The render params for each glyph (needed for sprite atlas lookup).
    pub params: Vec<RenderGlyphParams>,
}

/// Set the text decoration for a run of text.
#[derive(Debug, Clone)]
pub struct DecorationRun {
    /// The length of the run in utf-8 bytes.
    pub len: u32,

    /// The color for this run
    pub color: Hsla,

    /// The background color for this run
    pub background_color: Option<Hsla>,

    /// The underline style for this run
    pub underline: Option<UnderlineStyle>,

    /// The strikethrough style for this run
    pub strikethrough: Option<StrikethroughStyle>,
}

/// A line of text that has been shaped and decorated.
#[derive(Clone, Default, Debug, Deref, DerefMut)]
pub struct ShapedLine {
    #[deref]
    #[deref_mut]
    pub(crate) layout: Arc<LineLayout>,
    /// The text that was shaped for this line.
    pub text: SharedString,
    pub(crate) decoration_runs: SmallVec<[DecorationRun; 32]>,
}

impl ShapedLine {
    /// The length of the line in utf-8 bytes.
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.layout.len
    }

    /// The width of the shaped line in pixels.
    ///
    /// This is the glyph advance width computed by the text shaping system and is useful for
    /// incrementally advancing a "pen" when painting multiple fragments on the same row.
    pub fn width(&self) -> Pixels {
        self.layout.width
    }

    /// Override the len, useful if you're rendering text a
    /// as text b (e.g. rendering invisibles).
    pub fn with_len(mut self, len: usize) -> Self {
        let layout = self.layout.as_ref();
        self.layout = Arc::new(LineLayout {
            font_size: layout.font_size,
            width: layout.width,
            ascent: layout.ascent,
            descent: layout.descent,
            runs: layout.runs.clone(),
            len,
        });
        self
    }

    /// Paint the line of text to the window.
    pub fn paint(
        &self,
        origin: Point<Pixels>,
        line_height: Pixels,
        align: TextAlign,
        align_width: Option<Pixels>,
        window: &mut Window,
        cx: &mut App,
    ) -> Result<()> {
        paint_line(
            origin,
            &self.layout,
            line_height,
            align,
            align_width,
            &self.decoration_runs,
            &[],
            window,
            cx,
        )?;

        Ok(())
    }

    /// Paint the background of the line to the window.
    pub fn paint_background(
        &self,
        origin: Point<Pixels>,
        line_height: Pixels,
        align: TextAlign,
        align_width: Option<Pixels>,
        window: &mut Window,
        cx: &mut App,
    ) -> Result<()> {
        paint_line_background(
            origin,
            &self.layout,
            line_height,
            align,
            align_width,
            &self.decoration_runs,
            &[],
            window,
            cx,
        )?;

        Ok(())
    }

    /// Split this shaped line at a byte index, returning `(prefix, suffix)`.
    ///
    /// - `prefix` contains glyphs for bytes `[0, byte_index)` with original positions.
    ///   Its width equals the x-advance up to the split point.
    /// - `suffix` contains glyphs for bytes `[byte_index, len)` with positions
    ///   shifted left so the first glyph starts at x=0, and byte indices rebased to 0.
    /// - Decoration runs are partitioned at the boundary; a run that straddles it is
    ///   split into two with adjusted lengths.
    /// - `font_size`, `ascent`, and `descent` are copied to both halves.
    pub fn split_at(&self, byte_index: usize) -> (ShapedLine, ShapedLine) {
        let (left_layout, right_layout) = self.layout.split_at(byte_index);

        // Partition decoration runs. A run straddling the boundary is split into two.
        let mut left_decorations = SmallVec::new();
        let mut right_decorations = SmallVec::new();
        let mut decoration_offset = 0u32;
        let split_point = byte_index as u32;

        for decoration in &self.decoration_runs {
            let run_end = decoration_offset + decoration.len;

            if run_end <= split_point {
                left_decorations.push(decoration.clone());
            } else if decoration_offset >= split_point {
                right_decorations.push(decoration.clone());
            } else {
                let left_len = split_point - decoration_offset;
                let right_len = run_end - split_point;
                left_decorations.push(DecorationRun {
                    len: left_len,
                    color: decoration.color,
                    background_color: decoration.background_color,
                    underline: decoration.underline,
                    strikethrough: decoration.strikethrough,
                });
                right_decorations.push(DecorationRun {
                    len: right_len,
                    color: decoration.color,
                    background_color: decoration.background_color,
                    underline: decoration.underline,
                    strikethrough: decoration.strikethrough,
                });
            }

            decoration_offset = run_end;
        }

        // Split text
        let left_text = if byte_index == self.text.len() {
            self.text.clone()
        } else {
            SharedString::new(&self.text[..byte_index])
        };
        let right_text = if byte_index == 0 {
            self.text.clone()
        } else {
            SharedString::new(&self.text[byte_index..])
        };

        let left = ShapedLine {
            layout: Arc::new(left_layout),
            text: left_text,
            decoration_runs: left_decorations,
        };

        let right = ShapedLine {
            layout: Arc::new(right_layout),
            text: right_text,
            decoration_runs: right_decorations,
        };

        (left, right)
    }
}

impl LineLayout {
    /// Paint this layout to the window, using the given decoration runs to color
    /// glyphs and draw underlines and strikethroughs.
    ///
    /// This is a lower-level alternative to [`ShapedLine::paint`] for callers that
    /// hold a bare layout and track decorations themselves.
    pub fn paint(
        &self,
        origin: Point<Pixels>,
        line_height: Pixels,
        align: TextAlign,
        align_width: Option<Pixels>,
        decoration_runs: &[DecorationRun],
        window: &mut Window,
        cx: &mut App,
    ) -> Result<()> {
        paint_line(
            origin,
            self,
            line_height,
            align,
            align_width,
            decoration_runs,
            &[],
            window,
            cx,
        )
    }

    /// Paint the background of this layout to the window, using the given
    /// decoration runs to determine background colors.
    ///
    /// This is a lower-level alternative to [`ShapedLine::paint_background`] for
    /// callers that hold a bare layout and track decorations themselves.
    pub fn paint_background(
        &self,
        origin: Point<Pixels>,
        line_height: Pixels,
        align: TextAlign,
        align_width: Option<Pixels>,
        decoration_runs: &[DecorationRun],
        window: &mut Window,
        cx: &mut App,
    ) -> Result<()> {
        paint_line_background(
            origin,
            self,
            line_height,
            align,
            align_width,
            decoration_runs,
            &[],
            window,
            cx,
        )
    }
}

/// A line of text that has been shaped, decorated, and wrapped by the text layout system.
#[derive(Default, Debug, Deref, DerefMut)]
pub struct WrappedLine {
    #[deref]
    #[deref_mut]
    pub(crate) layout: Arc<WrappedLineLayout>,
    /// The text that was shaped for this line.
    pub text: SharedString,
    pub(crate) decoration_runs: Vec<DecorationRun>,
}

impl WrappedLine {
    /// The length of the underlying, unwrapped layout, in utf-8 bytes.
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.layout.len()
    }

    /// Paint this line of text to the window.
    pub fn paint(
        &self,
        origin: Point<Pixels>,
        line_height: Pixels,
        align: TextAlign,
        bounds: Option<Bounds<Pixels>>,
        window: &mut Window,
        cx: &mut App,
    ) -> Result<()> {
        let align_width = match bounds {
            Some(bounds) => Some(bounds.size.width),
            None => self.layout.wrap_width,
        };

        paint_line(
            origin,
            &self.layout.unwrapped_layout,
            line_height,
            align,
            align_width,
            &self.decoration_runs,
            &self.wrap_boundaries,
            window,
            cx,
        )?;

        Ok(())
    }

    /// Paint the background of line of text to the window.
    pub fn paint_background(
        &self,
        origin: Point<Pixels>,
        line_height: Pixels,
        align: TextAlign,
        bounds: Option<Bounds<Pixels>>,
        window: &mut Window,
        cx: &mut App,
    ) -> Result<()> {
        let align_width = match bounds {
            Some(bounds) => Some(bounds.size.width),
            None => self.layout.wrap_width,
        };

        paint_line_background(
            origin,
            &self.layout.unwrapped_layout,
            line_height,
            align,
            align_width,
            &self.decoration_runs,
            &self.wrap_boundaries,
            window,
            cx,
        )?;

        Ok(())
    }
}

fn paint_line(
    origin: Point<Pixels>,
    layout: &LineLayout,
    line_height: Pixels,
    align: TextAlign,
    align_width: Option<Pixels>,
    decoration_runs: &[DecorationRun],
    wrap_boundaries: &[WrapBoundary],
    window: &mut Window,
    cx: &mut App,
) -> Result<()> {
    let line_bounds = Bounds::new(
        origin,
        size(
            layout.width,
            line_height * (wrap_boundaries.len() as f32 + 1.),
        ),
    );
    window.paint_layer(line_bounds, |window| {
        let padding_top = (line_height - layout.ascent - layout.descent) / 2.;
        let baseline_offset = point(px(0.), padding_top + layout.ascent);
        let mut decoration_runs = decoration_runs.iter();
        let mut wraps = wrap_boundaries.iter().peekable();
        let mut run_end = 0;
        let mut color = black();
        let mut current_underline: Option<(Point<Pixels>, UnderlineStyle)> = None;
        let mut current_strikethrough: Option<(Point<Pixels>, StrikethroughStyle)> = None;
        let text_system = cx.text_system().clone();
        let mut glyph_origin = point(
            aligned_origin_x(
                origin,
                align_width.unwrap_or(layout.width),
                px(0.0),
                &align,
                layout,
                wraps.peek(),
            ),
            origin.y,
        );
        let mut prev_glyph_position = Point::default();
        let mut max_glyph_size = size(px(0.), px(0.));
        // The font's bounding box, which contains every glyph's ink by construction. It is
        // expressed relative to the BASELINE with y pointing up, so it has to be flipped and
        // positioned on the baseline to say where ink can actually land on screen.
        let mut max_glyph_box = Bounds::default();
        let mut first_glyph_x = origin.x;
        for (run_ix, run) in layout.runs.iter().enumerate() {
            max_glyph_box = text_system.bounding_box(run.font_id, layout.font_size);
            max_glyph_size = max_glyph_box.size;

            for (glyph_ix, glyph) in run.glyphs.iter().enumerate() {
                glyph_origin.x += glyph.position.x - prev_glyph_position.x;
                if glyph_ix == 0 && run_ix == 0 {
                    first_glyph_x = glyph_origin.x;
                }

                if wraps.peek() == Some(&&WrapBoundary { run_ix, glyph_ix }) {
                    wraps.next();
                    if let Some((underline_origin, underline_style)) = current_underline.as_mut() {
                        if glyph_origin.x == underline_origin.x {
                            underline_origin.x -= max_glyph_size.width.half();
                        };
                        window.paint_underline(
                            *underline_origin,
                            glyph_origin.x - underline_origin.x,
                            underline_style,
                        );
                        if glyph.index < run_end {
                            underline_origin.x = origin.x;
                            underline_origin.y += line_height;
                        } else {
                            current_underline = None;
                        }
                    }
                    if let Some((strikethrough_origin, strikethrough_style)) =
                        current_strikethrough.as_mut()
                    {
                        if glyph_origin.x == strikethrough_origin.x {
                            strikethrough_origin.x -= max_glyph_size.width.half();
                        };
                        window.paint_strikethrough(
                            *strikethrough_origin,
                            glyph_origin.x - strikethrough_origin.x,
                            strikethrough_style,
                        );
                        if glyph.index < run_end {
                            strikethrough_origin.x = origin.x;
                            strikethrough_origin.y += line_height;
                        } else {
                            current_strikethrough = None;
                        }
                    }

                    glyph_origin.x = aligned_origin_x(
                        origin,
                        align_width.unwrap_or(layout.width),
                        glyph.position.x,
                        &align,
                        layout,
                        wraps.peek(),
                    );
                    glyph_origin.y += line_height;
                }
                prev_glyph_position = glyph.position;

                let mut finished_underline: Option<(Point<Pixels>, UnderlineStyle)> = None;
                let mut finished_strikethrough: Option<(Point<Pixels>, StrikethroughStyle)> = None;
                if glyph.index >= run_end {
                    let mut style_run = decoration_runs.next();

                    // ignore style runs that apply to a partial glyph
                    while let Some(run) = style_run {
                        if glyph.index < run_end + (run.len as usize) {
                            break;
                        }
                        run_end += run.len as usize;
                        style_run = decoration_runs.next();
                    }

                    if let Some(style_run) = style_run {
                        if let Some((_, underline_style)) = &mut current_underline
                            && style_run.underline.as_ref() != Some(underline_style)
                        {
                            finished_underline = current_underline.take();
                        }
                        if let Some(run_underline) = style_run.underline.as_ref() {
                            current_underline.get_or_insert((
                                point(
                                    glyph_origin.x,
                                    glyph_origin.y + baseline_offset.y + (layout.descent * 0.618),
                                ),
                                UnderlineStyle {
                                    color: Some(run_underline.color.unwrap_or(style_run.color)),
                                    thickness: run_underline.thickness,
                                    wavy: run_underline.wavy,
                                },
                            ));
                        }
                        if let Some((_, strikethrough_style)) = &mut current_strikethrough
                            && style_run.strikethrough.as_ref() != Some(strikethrough_style)
                        {
                            finished_strikethrough = current_strikethrough.take();
                        }
                        if let Some(run_strikethrough) = style_run.strikethrough.as_ref() {
                            current_strikethrough.get_or_insert((
                                point(
                                    glyph_origin.x,
                                    glyph_origin.y
                                        + (((layout.ascent * 0.5) + baseline_offset.y) * 0.5),
                                ),
                                StrikethroughStyle {
                                    color: Some(run_strikethrough.color.unwrap_or(style_run.color)),
                                    thickness: run_strikethrough.thickness,
                                },
                            ));
                        }

                        run_end += style_run.len as usize;
                        color = style_run.color;
                    } else {
                        run_end = layout.len;
                        finished_underline = current_underline.take();
                        finished_strikethrough = current_strikethrough.take();
                    }
                }

                if let Some((mut underline_origin, underline_style)) = finished_underline {
                    if underline_origin.x == glyph_origin.x {
                        underline_origin.x -= max_glyph_size.width.half();
                    };
                    window.paint_underline(
                        underline_origin,
                        glyph_origin.x - underline_origin.x,
                        &underline_style,
                    );
                }

                if let Some((mut strikethrough_origin, strikethrough_style)) =
                    finished_strikethrough
                {
                    if strikethrough_origin.x == glyph_origin.x {
                        strikethrough_origin.x -= max_glyph_size.width.half();
                    };
                    window.paint_strikethrough(
                        strikethrough_origin,
                        glyph_origin.x - strikethrough_origin.x,
                        &strikethrough_style,
                    );
                }

                // Conservative pre-cull: this exists only to skip rasterizing glyphs that are
                // obviously offscreen, and the exact cull happens later in
                // `Scene::insert_primitive` against the glyph's real quad. So it must never
                // discard a glyph that would have been visible.
                //
                // It previously used the font's max box anchored at `glyph_origin` — the pen
                // position at the TOP of the line — while the glyph is painted down at the
                // baseline. The box therefore described a region the glyph is not in, and near
                // a clip edge that culls glyphs that are still visible, one at a time, leaving
                // their advances behind: characters missing from the middle of a word.
                //
                // Anchor it to the baseline instead. `max_glyph_box` is in font space (y up,
                // relative to the baseline), so flipping it onto the baseline the glyph is
                // actually painted on gives a span that provably contains the ink, descenders
                // included. Union with the line row so the box is never smaller than the row,
                // and allow a glyph box of horizontal overhang for negative side bearings.
                let vertical_offset = point(px(0.0), glyph.position.y);
                let baseline_y = glyph_origin.y + baseline_offset.y + vertical_offset.y;
                let ink_top = baseline_y - (max_glyph_box.origin.y + max_glyph_box.size.height);
                let ink_bottom = baseline_y - max_glyph_box.origin.y;
                let cull_top = ink_top.min(glyph_origin.y);
                let cull_bottom = ink_bottom.max(glyph_origin.y + line_height);
                let max_glyph_bounds = Bounds {
                    origin: point(glyph_origin.x - max_glyph_size.width, cull_top),
                    size: size(max_glyph_size.width * 3., cull_bottom - cull_top),
                };

                let content_mask = window.content_mask();
                if max_glyph_bounds.intersects(&content_mask.bounds) {
                    if glyph.is_emoji {
                        window.paint_emoji(
                            glyph_origin + baseline_offset + vertical_offset,
                            run.font_id,
                            glyph.id,
                            layout.font_size,
                        )?;
                    } else {
                        window.paint_glyph(
                            glyph_origin + baseline_offset + vertical_offset,
                            run.font_id,
                            glyph.id,
                            layout.font_size,
                            color,
                        )?;
                    }
                }
            }
        }

        let mut last_line_end_x = first_glyph_x + layout.width;
        if let Some(boundary) = wrap_boundaries.last() {
            let run = &layout.runs[boundary.run_ix];
            let glyph = &run.glyphs[boundary.glyph_ix];
            last_line_end_x -= glyph.position.x;
        }

        if let Some((mut underline_start, underline_style)) = current_underline.take() {
            if last_line_end_x == underline_start.x {
                underline_start.x -= max_glyph_size.width.half()
            };
            window.paint_underline(
                underline_start,
                last_line_end_x - underline_start.x,
                &underline_style,
            );
        }

        if let Some((mut strikethrough_start, strikethrough_style)) = current_strikethrough.take() {
            if last_line_end_x == strikethrough_start.x {
                strikethrough_start.x -= max_glyph_size.width.half()
            };
            window.paint_strikethrough(
                strikethrough_start,
                last_line_end_x - strikethrough_start.x,
                &strikethrough_style,
            );
        }

        Ok(())
    })
}

fn paint_line_background(
    origin: Point<Pixels>,
    layout: &LineLayout,
    line_height: Pixels,
    align: TextAlign,
    align_width: Option<Pixels>,
    decoration_runs: &[DecorationRun],
    wrap_boundaries: &[WrapBoundary],
    window: &mut Window,
    cx: &mut App,
) -> Result<()> {
    let line_bounds = Bounds::new(
        origin,
        size(
            layout.width,
            line_height * (wrap_boundaries.len() as f32 + 1.),
        ),
    );
    window.paint_layer(line_bounds, |window| {
        let mut decoration_runs = decoration_runs.iter();
        let mut wraps = wrap_boundaries.iter().peekable();
        let mut run_end = 0;
        let mut current_background: Option<(Point<Pixels>, Hsla)> = None;
        let text_system = cx.text_system().clone();
        let mut glyph_origin = point(
            aligned_origin_x(
                origin,
                align_width.unwrap_or(layout.width),
                px(0.0),
                &align,
                layout,
                wraps.peek(),
            ),
            origin.y,
        );
        let mut prev_glyph_position = Point::default();
        let mut max_glyph_size = size(px(0.), px(0.));
        for (run_ix, run) in layout.runs.iter().enumerate() {
            max_glyph_size = text_system.bounding_box(run.font_id, layout.font_size).size;

            for (glyph_ix, glyph) in run.glyphs.iter().enumerate() {
                glyph_origin.x += glyph.position.x - prev_glyph_position.x;

                if wraps.peek() == Some(&&WrapBoundary { run_ix, glyph_ix }) {
                    wraps.next();
                    if let Some((background_origin, background_color)) = current_background.as_mut()
                    {
                        if glyph_origin.x == background_origin.x {
                            background_origin.x -= max_glyph_size.width.half()
                        }
                        window.paint_quad(fill(
                            Bounds {
                                origin: *background_origin,
                                size: size(glyph_origin.x - background_origin.x, line_height),
                            },
                            *background_color,
                        ));
                        if glyph.index < run_end {
                            background_origin.x = origin.x;
                            background_origin.y += line_height;
                        } else {
                            current_background = None;
                        }
                    }

                    glyph_origin.x = aligned_origin_x(
                        origin,
                        align_width.unwrap_or(layout.width),
                        glyph.position.x,
                        &align,
                        layout,
                        wraps.peek(),
                    );
                    glyph_origin.y += line_height;
                }
                prev_glyph_position = glyph.position;

                let mut finished_background: Option<(Point<Pixels>, Hsla)> = None;
                if glyph.index >= run_end {
                    let mut style_run = decoration_runs.next();

                    // ignore style runs that apply to a partial glyph
                    while let Some(run) = style_run {
                        if glyph.index < run_end + (run.len as usize) {
                            break;
                        }
                        run_end += run.len as usize;
                        style_run = decoration_runs.next();
                    }

                    if let Some(style_run) = style_run {
                        if let Some((_, background_color)) = &mut current_background
                            && style_run.background_color.as_ref() != Some(background_color)
                        {
                            finished_background = current_background.take();
                        }
                        if let Some(run_background) = style_run.background_color {
                            current_background.get_or_insert((
                                point(glyph_origin.x, glyph_origin.y),
                                run_background,
                            ));
                        }
                        run_end += style_run.len as usize;
                    } else {
                        run_end = layout.len;
                        finished_background = current_background.take();
                    }
                }

                if let Some((mut background_origin, background_color)) = finished_background {
                    let mut width = glyph_origin.x - background_origin.x;
                    if background_origin.x == glyph_origin.x {
                        background_origin.x -= max_glyph_size.width.half();
                    };
                    window.paint_quad(fill(
                        Bounds {
                            origin: background_origin,
                            size: size(width, line_height),
                        },
                        background_color,
                    ));
                }
            }
        }

        let mut last_line_end_x = origin.x + layout.width;
        if let Some(boundary) = wrap_boundaries.last() {
            let run = &layout.runs[boundary.run_ix];
            let glyph = &run.glyphs[boundary.glyph_ix];
            last_line_end_x -= glyph.position.x;
        }

        if let Some((mut background_origin, background_color)) = current_background.take() {
            if last_line_end_x == background_origin.x {
                background_origin.x -= max_glyph_size.width.half()
            };
            window.paint_quad(fill(
                Bounds {
                    origin: background_origin,
                    size: size(last_line_end_x - background_origin.x, line_height),
                },
                background_color,
            ));
        }

        Ok(())
    })
}

fn aligned_origin_x(
    origin: Point<Pixels>,
    align_width: Pixels,
    last_glyph_x: Pixels,
    align: &TextAlign,
    layout: &LineLayout,
    wrap_boundary: Option<&&WrapBoundary>,
) -> Pixels {
    let end_of_line = if let Some(WrapBoundary { run_ix, glyph_ix }) = wrap_boundary {
        layout.runs[*run_ix].glyphs[*glyph_ix].position.x
    } else {
        layout.width
    };

    let line_width = end_of_line - last_glyph_x;

    match align {
        TextAlign::Left => origin.x,
        TextAlign::Center => (origin.x * 2.0 + align_width - line_width) / 2.0,
        TextAlign::Right => origin.x + align_width - line_width,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FontId, GlyphId, ShapedGlyph, ShapedRun};

    /// Helper: build a ShapedLine from glyph descriptors without the platform text system.
    /// Each glyph is described as (byte_index, x_position).
    fn make_shaped_line(
        text: &str,
        glyphs: &[(usize, f32)],
        width: f32,
        decorations: &[DecorationRun],
    ) -> ShapedLine {
        let shaped_glyphs: Vec<ShapedGlyph> = glyphs
            .iter()
            .map(|&(index, x)| ShapedGlyph {
                id: GlyphId(0),
                position: point(px(x), px(0.0)),
                index,
                is_emoji: false,
            })
            .collect();

        ShapedLine {
            layout: Arc::new(LineLayout {
                font_size: px(16.0),
                width: px(width),
                ascent: px(12.0),
                descent: px(4.0),
                runs: vec![ShapedRun {
                    font_id: FontId(0),
                    glyphs: shaped_glyphs,
                }],
                len: text.len(),
            }),
            text: SharedString::new(text),
            decoration_runs: SmallVec::from(decorations.to_vec()),
        }
    }

    #[test]
    fn test_split_at_invariants() {
        // Split "abcdef" at every possible byte index and verify structural invariants.
        let line = make_shaped_line(
            "abcdef",
            &[
                (0, 0.0),
                (1, 10.0),
                (2, 20.0),
                (3, 30.0),
                (4, 40.0),
                (5, 50.0),
            ],
            60.0,
            &[],
        );

        for i in 0..=6 {
            let (left, right) = line.split_at(i);

            assert_eq!(
                left.width() + right.width(),
                line.width(),
                "widths must sum at split={i}"
            );
            assert_eq!(
                left.len() + right.len(),
                line.len(),
                "lengths must sum at split={i}"
            );
            assert_eq!(
                format!("{}{}", left.text.as_ref(), right.text.as_ref()),
                "abcdef",
                "text must concatenate at split={i}"
            );
            assert_eq!(left.font_size, line.font_size, "font_size at split={i}");
            assert_eq!(right.ascent, line.ascent, "ascent at split={i}");
            assert_eq!(right.descent, line.descent, "descent at split={i}");
        }

        // Edge: split at 0 produces no left runs, full content on right
        let (left, right) = line.split_at(0);
        assert_eq!(left.runs.len(), 0);
        assert_eq!(right.runs[0].glyphs.len(), 6);

        // Edge: split at end produces full content on left, no right runs
        let (left, right) = line.split_at(6);
        assert_eq!(left.runs[0].glyphs.len(), 6);
        assert_eq!(right.runs.len(), 0);
    }

    #[test]
    fn test_split_at_glyph_rebasing() {
        // Two font runs (simulating a font fallback boundary at byte 3):
        //   run A (FontId 0): glyphs at bytes 0,1,2  positions 0,10,20
        //   run B (FontId 1): glyphs at bytes 3,4,5  positions 30,40,50
        // Successive splits simulate the incremental splitting done during wrap.
        let line = ShapedLine {
            layout: Arc::new(LineLayout {
                font_size: px(16.0),
                width: px(60.0),
                ascent: px(12.0),
                descent: px(4.0),
                runs: vec![
                    ShapedRun {
                        font_id: FontId(0),
                        glyphs: vec![
                            ShapedGlyph {
                                id: GlyphId(0),
                                position: point(px(0.0), px(0.0)),
                                index: 0,
                                is_emoji: false,
                            },
                            ShapedGlyph {
                                id: GlyphId(0),
                                position: point(px(10.0), px(0.0)),
                                index: 1,
                                is_emoji: false,
                            },
                            ShapedGlyph {
                                id: GlyphId(0),
                                position: point(px(20.0), px(0.0)),
                                index: 2,
                                is_emoji: false,
                            },
                        ],
                    },
                    ShapedRun {
                        font_id: FontId(1),
                        glyphs: vec![
                            ShapedGlyph {
                                id: GlyphId(0),
                                position: point(px(30.0), px(0.0)),
                                index: 3,
                                is_emoji: false,
                            },
                            ShapedGlyph {
                                id: GlyphId(0),
                                position: point(px(40.0), px(0.0)),
                                index: 4,
                                is_emoji: false,
                            },
                            ShapedGlyph {
                                id: GlyphId(0),
                                position: point(px(50.0), px(0.0)),
                                index: 5,
                                is_emoji: false,
                            },
                        ],
                    },
                ],
                len: 6,
            }),
            text: "abcdef".into(),
            decoration_runs: SmallVec::new(),
        };

        // First split at byte 2 — mid-run in run A
        let (first, remainder) = line.split_at(2);
        assert_eq!(first.text.as_ref(), "ab");
        assert_eq!(first.runs.len(), 1);
        assert_eq!(first.runs[0].font_id, FontId(0));

        // Remainder "cdef" should have two runs: tail of A (1 glyph) + all of B (3 glyphs)
        assert_eq!(remainder.text.as_ref(), "cdef");
        assert_eq!(remainder.runs.len(), 2);
        assert_eq!(remainder.runs[0].font_id, FontId(0));
        assert_eq!(remainder.runs[0].glyphs.len(), 1);
        assert_eq!(remainder.runs[0].glyphs[0].index, 0);
        assert_eq!(remainder.runs[0].glyphs[0].position.x, px(0.0));
        assert_eq!(remainder.runs[1].font_id, FontId(1));
        assert_eq!(remainder.runs[1].glyphs[0].index, 1);
        assert_eq!(remainder.runs[1].glyphs[0].position.x, px(10.0));

        // Second split at byte 2 within remainder — crosses the run boundary
        let (second, final_part) = remainder.split_at(2);
        assert_eq!(second.text.as_ref(), "cd");
        assert_eq!(final_part.text.as_ref(), "ef");
        assert_eq!(final_part.runs[0].glyphs[0].index, 0);
        assert_eq!(final_part.runs[0].glyphs[0].position.x, px(0.0));

        // Widths must sum across all three pieces
        assert_eq!(
            first.width() + second.width() + final_part.width(),
            line.width()
        );
    }

    #[test]
    fn test_split_at_decorations() {
        // Three decoration runs: red [0..2), green [2..5), blue [5..6).
        // Split at byte 3 — red goes entirely left, green straddles, blue goes entirely right.
        let red = Hsla {
            h: 0.0,
            s: 1.0,
            l: 0.5,
            a: 1.0,
        };
        let green = Hsla {
            h: 0.3,
            s: 1.0,
            l: 0.5,
            a: 1.0,
        };
        let blue = Hsla {
            h: 0.6,
            s: 1.0,
            l: 0.5,
            a: 1.0,
        };

        let line = make_shaped_line(
            "abcdef",
            &[
                (0, 0.0),
                (1, 10.0),
                (2, 20.0),
                (3, 30.0),
                (4, 40.0),
                (5, 50.0),
            ],
            60.0,
            &[
                DecorationRun {
                    len: 2,
                    color: red,
                    background_color: None,
                    underline: None,
                    strikethrough: None,
                },
                DecorationRun {
                    len: 3,
                    color: green,
                    background_color: None,
                    underline: None,
                    strikethrough: None,
                },
                DecorationRun {
                    len: 1,
                    color: blue,
                    background_color: None,
                    underline: None,
                    strikethrough: None,
                },
            ],
        );

        let (left, right) = line.split_at(3);

        // Left: red(2) + green(1) — green straddled, left portion has len 1
        assert_eq!(left.decoration_runs.len(), 2);
        assert_eq!(left.decoration_runs[0].len, 2);
        assert_eq!(left.decoration_runs[0].color, red);
        assert_eq!(left.decoration_runs[1].len, 1);
        assert_eq!(left.decoration_runs[1].color, green);

        // Right: green(2) + blue(1) — green straddled, right portion has len 2
        assert_eq!(right.decoration_runs.len(), 2);
        assert_eq!(right.decoration_runs[0].len, 2);
        assert_eq!(right.decoration_runs[0].color, green);
        assert_eq!(right.decoration_runs[1].len, 1);
        assert_eq!(right.decoration_runs[1].color, blue);
    }
}

/// End-to-end reproduction for glyphs vanishing from the middle of a word while their advance
/// survives (fincode `docs/text-flicker-root-cause-and-plan.md`, root cause 7).
///
/// `paint_line` pre-culls each glyph with a cheap box before rasterizing it, and the exact cull
/// happens later in `Scene::insert_primitive` against the glyph's real quad. The cheap box used
/// to be the font's max bounding box anchored at `glyph_origin` — the pen position at the TOP of
/// the line — while the glyph is painted down at the baseline. At generous line heights the box
/// and the ink are disjoint, so near a clip edge the cheap cull throws away glyphs that are
/// plainly visible, leaving their advances behind.
///
/// This drives the real paint path (shape -> `paint_line` -> pre-cull -> `paint_glyph` ->
/// `insert_primitive` -> scene) and counts the glyph sprites that actually reached the scene.
#[cfg(test)]
mod pre_cull_regression_tests {
    use crate::{
        AppContext as _, Bounds, ContentMask, Context, DevicePixels, Font, FontId,
        FontMetrics, FontRun,
        GlyphId, Hsla, IntoElement, LineLayout, NoopTextSystem, Pixels, PlatformTextSystem, Point,
        ParentElement as _, Render, RenderGlyphParams, Size, Styled as _, TestAppContext,
        TestDispatcher,
        TextAlign,
        TextRenderingMode, TextRun, Window, black, canvas, div, font, point, px, size,
    };
    use anyhow::Result;
    use std::{borrow::Cow, cell::Cell, cell::RefCell, rc::Rc, sync::Arc};

    const TEXT: &str = "Changes in this project";
    const FONT_SIZE: Pixels = Pixels(16.);
    /// Roomy leading, as the transcript and review panel use. This is what pushes the baseline
    /// far below the pen position and separates the ink from the old cull box.
    const LINE_HEIGHT: Pixels = Pixels(64.);

    /// Reports glyphs whose ink sits at the baseline, like a real font: `origin.y` is negative
    /// (up from the baseline) and the ink is roughly cap height. `NoopTextSystem` reports empty
    /// raster bounds for everything, which would make `paint_glyph` skip every glyph and hide
    /// the very behaviour under test.
    struct InkedTextSystem(NoopTextSystem);

    /// Ink height in device pixels, varied per glyph so the line is not one uniform box.
    fn ink_height(glyph_id: GlyphId, font_size: Pixels) -> i32 {
        (font_size.0 * 0.62).round() as i32 + (glyph_id.0 % 3) as i32
    }

    impl PlatformTextSystem for InkedTextSystem {
        fn glyph_raster_bounds(&self, params: &RenderGlyphParams) -> Result<Bounds<DevicePixels>> {
            let height = ink_height(params.glyph_id, params.font_size);
            let width = (params.font_size.0 * 0.5).round() as i32;
            Ok(Bounds {
                origin: point(DevicePixels(0), DevicePixels(-height)),
                size: size(DevicePixels(width), DevicePixels(height)),
            })
        }

        fn rasterize_glyph(
            &self,
            _params: &RenderGlyphParams,
            raster_bounds: Bounds<DevicePixels>,
        ) -> Result<(Size<DevicePixels>, Vec<u8>)> {
            let byte_count =
                (raster_bounds.size.width.0 * raster_bounds.size.height.0).max(0) as usize;
            Ok((raster_bounds.size, vec![255; byte_count]))
        }

        fn add_fonts(&self, fonts: Vec<Cow<'static, [u8]>>) -> Result<()> {
            self.0.add_fonts(fonts)
        }
        fn all_font_names(&self) -> Vec<String> {
            self.0.all_font_names()
        }
        fn font_id(&self, descriptor: &Font) -> Result<FontId> {
            self.0.font_id(descriptor)
        }
        fn font_metrics(&self, font_id: FontId) -> FontMetrics {
            self.0.font_metrics(font_id)
        }
        fn typographic_bounds(&self, font_id: FontId, glyph_id: GlyphId) -> Result<Bounds<f32>> {
            self.0.typographic_bounds(font_id, glyph_id)
        }
        fn advance(&self, font_id: FontId, glyph_id: GlyphId) -> Result<Size<f32>> {
            self.0.advance(font_id, glyph_id)
        }
        fn glyph_for_char(&self, font_id: FontId, ch: char) -> Option<GlyphId> {
            self.0.glyph_for_char(font_id, ch)
        }
        fn layout_line(&self, text: &str, font_size: Pixels, runs: &[FontRun]) -> LineLayout {
            self.0.layout_line(text, font_size, runs)
        }
        fn recommended_rendering_mode(
            &self,
            font_id: FontId,
            font_size: Pixels,
        ) -> TextRenderingMode {
            self.0.recommended_rendering_mode(font_id, font_size)
        }
        fn glyph_dilation_for_color(&self, color: Hsla) -> u8 {
            self.0.glyph_dilation_for_color(color)
        }
    }

    struct TextUnderMask {
        mask: Bounds<Pixels>,
        origin: Point<Pixels>,
        shaped_glyphs: Rc<Cell<usize>>,
    }

    impl Render for TextUnderMask {
        fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
            let mask = self.mask;
            let origin = self.origin;
            let shaped_glyphs = self.shaped_glyphs.clone();
            div().child(canvas(
                |_, _, _| (),
                move |_bounds, _, window, cx| {
                    let runs = [TextRun {
                        len: TEXT.len(),
                        font: font("test"),
                        color: black(),
                        background_color: None,
                        underline: None,
                        strikethrough: None,
                    }];
                    let line = window
                        .text_system()
                        .shape_line(TEXT.into(), FONT_SIZE, &runs, None);
                    shaped_glyphs.set(line.runs.iter().map(|run| run.glyphs.len()).sum::<usize>());
                    window.with_content_mask(Some(ContentMask { bounds: mask }), |window| {
                        line.paint(origin, LINE_HEIGHT, TextAlign::Left, None, window, cx)
                            .unwrap();
                    });
                },
            ))
        }
    }

    /// Every glyph whose ink lands inside the content mask has to reach the scene. The mask here
    /// is a band that contains the painted ink but sits entirely BELOW the old pre-cull box, so
    /// the old box and the mask do not intersect at all: the old code discarded the whole line
    /// while every glyph in it was visible.
    #[test]
    fn glyphs_inside_the_content_mask_are_not_dropped_by_the_pre_cull() {
        let platform_text_system = Arc::new(InkedTextSystem(NoopTextSystem));
        let metrics = platform_text_system.font_metrics(FontId(0));
        let mut cx = TestAppContext::build_with_text_system(
            TestDispatcher::new(0),
            None,
            platform_text_system,
        );

        let origin = point(px(0.), px(0.));
        // Mirrors `paint_line`: the baseline sits `padding_top + ascent` below the pen position.
        let ascent = FONT_SIZE * (metrics.ascent / metrics.units_per_em as f32);
        let descent = FONT_SIZE * (metrics.descent / metrics.units_per_em as f32);
        let baseline = (LINE_HEIGHT - ascent - descent) / 2. + ascent;
        let old_cull_bottom = metrics.bounding_box(FONT_SIZE).size.height;

        // A band that covers the ink (which spans about `baseline - ink_height ..= baseline`)
        // while starting below the old cull box, which was `origin.y .. origin.y + font_box`.
        let mask_top = baseline - px(14.);
        let mask = Bounds {
            origin: point(px(-10.), mask_top),
            size: size(px(1000.), px(60.)),
        };
        assert!(
            mask_top > old_cull_bottom,
            "the mask has to start below the old cull box for this to exercise the defect \
             (mask top {mask_top:?}, old box bottom {old_cull_bottom:?})"
        );

        let shaped_glyphs = Rc::new(Cell::new(0));
        let window = cx.add_window({
            let shaped_glyphs = shaped_glyphs.clone();
            move |_, _| TextUnderMask {
                mask,
                origin,
                shaped_glyphs,
            }
        });

        let painted = cx
            .update_window(window.into(), |_, window, cx| {
                window.draw(cx).clear(cx);
                window.rendered_frame.scene.monochrome_sprites.len()
                    + window.rendered_frame.scene.subpixel_sprites.len()
            })
            .unwrap();

        let expected = shaped_glyphs.get();
        assert!(
            expected > 0,
            "the line shaped no glyphs, so this proved nothing"
        );
        assert_eq!(
            painted, expected,
            "{} of {expected} glyphs never reached the scene even though their ink is inside \
             the content mask, so they were visible: the pre-cull box discarded them",
            expected - painted
        );
    }

    /// Where a glyph's ink lands, in window coordinates, derived from the same numbers
    /// `paint_line`/`paint_glyph` use. `InkedTextSystem` puts the ink directly above the
    /// baseline, so the vertical span is `baseline - ink_height ..= baseline`.
    fn expected_ink(
        glyph: (GlyphId, Pixels),
        origin: Point<Pixels>,
        line_height: Pixels,
        metrics: &FontMetrics,
    ) -> Bounds<Pixels> {
        let (glyph_id, advance_x) = glyph;
        let ascent = FONT_SIZE * (metrics.ascent / metrics.units_per_em as f32);
        let descent = FONT_SIZE * (metrics.descent / metrics.units_per_em as f32);
        let baseline = (line_height - ascent - descent) / 2. + ascent;
        let height = Pixels(ink_height(glyph_id, FONT_SIZE) as f32);
        let width = Pixels((FONT_SIZE.0 * 0.5).round());
        Bounds {
            origin: point(origin.x + advance_x, origin.y + baseline - height),
            size: size(width, height),
        }
    }

    /// Paints the line once under `mask` and returns how many glyph sprites reached the scene,
    /// along with the shaped glyphs so the caller can work out how many should have.
    fn paint_once(
        line_height: Pixels,
        origin: Point<Pixels>,
        mask: Bounds<Pixels>,
        redraws: usize,
    ) -> (usize, Vec<(GlyphId, Pixels)>) {
        let platform_text_system = Arc::new(InkedTextSystem(NoopTextSystem));
        let mut cx = TestAppContext::build_with_text_system(
            TestDispatcher::new(0),
            None,
            platform_text_system,
        );

        let glyphs = Rc::new(RefCell::new(Vec::new()));
        let window = cx.add_window({
            let glyphs = glyphs.clone();
            move |_, _| SweepView {
                mask,
                origin,
                line_height,
                glyphs,
            }
        });

        let mut painted = 0;
        for _ in 0..redraws.max(1) {
            painted = cx
                .update_window(window.into(), |_, window, cx| {
                    window.draw(cx).clear(cx);
                    window.rendered_frame.scene.monochrome_sprites.len()
                        + window.rendered_frame.scene.subpixel_sprites.len()
                })
                .unwrap();
        }
        let glyphs = glyphs.borrow().clone();
        (painted, glyphs)
    }

    struct SweepView {
        mask: Bounds<Pixels>,
        origin: Point<Pixels>,
        line_height: Pixels,
        glyphs: Rc<RefCell<Vec<(GlyphId, Pixels)>>>,
    }

    impl Render for SweepView {
        fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
            let mask = self.mask;
            let origin = self.origin;
            let line_height = self.line_height;
            let glyphs = self.glyphs.clone();
            div().child(canvas(
                |_, _, _| (),
                move |_bounds, _, window, cx| {
                    let runs = [TextRun {
                        len: TEXT.len(),
                        font: font("test"),
                        color: black(),
                        background_color: None,
                        underline: None,
                        strikethrough: None,
                    }];
                    let line = window
                        .text_system()
                        .shape_line(TEXT.into(), FONT_SIZE, &runs, None);
                    *glyphs.borrow_mut() = line
                        .runs
                        .iter()
                        .flat_map(|run| run.glyphs.iter())
                        .map(|glyph| (glyph.id, glyph.position.x))
                        .collect();
                    window.with_content_mask(Some(ContentMask { bounds: mask }), |window| {
                        line.paint(origin, line_height, TextAlign::Left, None, window, cx)
                            .unwrap();
                    });
                },
            ))
        }
    }

    /// The correctness property the whole cull chain has to satisfy: a glyph is painted if and
    /// only if its ink intersects the content mask. Anything else is either a character missing
    /// from visible text, or work done on something that cannot be seen.
    ///
    /// Swept across line heights, mask edges that slice through the text at every offset, and
    /// fractional origins that move the glyphs between subpixel variants. Glyphs straddling the
    /// mask edge are excluded from both bounds, so boundary rounding is never disputed.
    #[test]
    fn painted_glyphs_match_the_content_mask_across_configurations() {
        let metrics = InkedTextSystem(NoopTextSystem).font_metrics(FontId(0));
        let mut failures = Vec::new();
        let mut configurations = 0;

        for line_height in [16., 20., 24., 32., 48., 64.] {
            let line_height = Pixels(line_height);
            for origin_y in [0., 0.25, 0.5, 7.3] {
                let origin = point(px(0.), px(origin_y));
                // Slide a band down through the line so its edges cut the text everywhere.
                for mask_top in [-20., -5., 0., 4., 8., 12., 16., 20., 24., 30., 40., 60.] {
                    for mask_height in [6., 12., 24., 60.] {
                        let mask = Bounds {
                            origin: point(px(-50.), px(mask_top)),
                            size: size(px(2000.), px(mask_height)),
                        };
                        configurations += 1;

                        let (painted, glyphs) = paint_once(line_height, origin, mask, 1);

                        let mut must_paint = 0;
                        let mut may_paint = 0;
                        for glyph in glyphs {
                            let ink = expected_ink(glyph, origin, line_height, &metrics);
                            // Shrink and grow by a pixel so glyphs sitting exactly on the edge
                            // land in neither bucket.
                            let inside = ink.origin.y >= mask.origin.y + px(1.)
                                && ink.origin.y + ink.size.height
                                    <= mask.origin.y + mask.size.height - px(1.);
                            // Grown by a pixel, so "definitely outside" never claims a glyph
                            // that overlaps the mask by a sliver.
                            let outside = ink.origin.y + ink.size.height
                                <= mask.origin.y - px(1.)
                                || ink.origin.y >= mask.origin.y + mask.size.height + px(1.);
                            if inside {
                                must_paint += 1;
                            }
                            if !outside {
                                may_paint += 1;
                            }
                        }

                        if painted < must_paint {
                            failures.push(format!(
                                "  line-height {line_height:?} origin.y {origin_y} mask \
                                 {mask_top}..{}: {painted} painted but {must_paint} glyphs are \
                                 fully inside the mask ({} dropped while visible)",
                                mask_top + mask_height,
                                must_paint - painted
                            ));
                        } else if painted > may_paint {
                            failures.push(format!(
                                "  line-height {line_height:?} origin.y {origin_y} mask \
                                 {mask_top}..{}: {painted} painted but only {may_paint} glyphs \
                                 touch the mask at all",
                                mask_top + mask_height
                            ));
                        }
                    }
                }
            }
        }

        eprintln!("swept {configurations} mask/line-height/origin configurations");
        assert!(
            failures.is_empty(),
            "{} configuration(s) painted a different set of glyphs than the content mask \
             allows:\n{}",
            failures.len(),
            failures
                .iter()
                .take(30)
                .cloned()
                .collect::<Vec<_>>()
                .join("\n")
        );
    }

    /// Drawing the same unchanged view repeatedly takes the cached-paint reuse path from the
    /// second frame on: prepaint returns no element and `Window::reuse_paint` replays the
    /// previous frame's recorded slice of paint operations through `Scene::replay`.
    ///
    /// If a replayed range were ever stale or misaligned, the replay would silently re-insert a
    /// wrong slice — dropping or duplicating arbitrary primitives, which for text means
    /// scattered characters going missing while their neighbours survive. So every redraw has to
    /// paint exactly what the first one did.
    #[test]
    fn cached_replay_paints_the_same_glyphs_every_frame() {
        let line_height = Pixels(24.);
        let origin = point(px(0.), px(0.));
        let mask = Bounds {
            origin: point(px(-50.), px(-50.)),
            size: size(px(2000.), px(400.)),
        };

        let (first, glyphs) = paint_once(line_height, origin, mask, 1);
        assert!(!glyphs.is_empty(), "the line shaped no glyphs");
        assert_eq!(
            first,
            glyphs.len(),
            "the baseline frame already dropped glyphs, so the replay comparison is meaningless"
        );

        for redraws in [2usize, 3, 5, 8] {
            let (painted, _) = paint_once(line_height, origin, mask, redraws);
            assert_eq!(
                painted, first,
                "after {redraws} redraws the scene held {painted} glyph sprites but the first \
                 frame painted {first}; cached paint reuse changed what was drawn"
            );
        }
    }

    /// Timing harness for the pre-cull change, not a correctness test.
    ///
    /// The pre-cull exists to avoid rasterizing glyphs that cannot be seen, and the fix made its
    /// box larger, so it now admits some glyphs that the exact cull in `Scene::insert_primitive`
    /// rejects a moment later. This paints a realistic scrolled block of text under a viewport
    /// mask — most rows clipped away, a band visible — and reports the wall time plus how many
    /// glyphs survived, so the extra work can be compared against the old box by reverting it.
    ///
    /// Run with: cargo test -p gpui --release --lib pre_cull_paint_cost -- --nocapture --ignored
    #[test]
    #[ignore = "timing harness, run manually"]
    fn pre_cull_paint_cost() {
        const ROWS: usize = 200;
        const ITERATIONS: usize = 200;
        const BATCHES: usize = 7;
        let line_height = Pixels(24.);
        // A viewport showing roughly 25 rows out of 200: the rest must be culled.
        let mask = Bounds {
            origin: point(px(0.), px(1200.)),
            size: size(px(1200.), px(600.)),
        };

        let platform_text_system = Arc::new(InkedTextSystem(NoopTextSystem));
        let mut cx = TestAppContext::build_with_text_system(
            TestDispatcher::new(0),
            None,
            platform_text_system,
        );
        let painted = Rc::new(Cell::new(0usize));
        let window = cx.add_window({
            let painted = painted.clone();
            move |_, _| ScrolledTextView {
                mask,
                line_height,
                rows: ROWS,
                painted,
            }
        });

        // Warm the glyph caches so the measurement is paint work, not first-touch rasterization.
        for _ in 0..3 {
            cx.update_window(window.into(), |_, window, cx| window.draw(cx).clear(cx))
                .unwrap();
        }

        // Report the best of several batches. The minimum is the robust estimator here, since
        // scheduling noise can only ever make a batch slower, never faster.
        let mut sprites = 0;
        let mut best_ms = f64::INFINITY;
        let mut worst_ms: f64 = 0.;
        for _ in 0..BATCHES {
            let started = std::time::Instant::now();
            for _ in 0..ITERATIONS {
                sprites = cx
                    .update_window(window.into(), |_, window, cx| {
                        window.draw(cx).clear(cx);
                        window.rendered_frame.scene.monochrome_sprites.len()
                            + window.rendered_frame.scene.subpixel_sprites.len()
                    })
                    .unwrap();
            }
            let per_frame = started.elapsed().as_secs_f64() * 1000.0 / ITERATIONS as f64;
            best_ms = best_ms.min(per_frame);
            worst_ms = worst_ms.max(per_frame);
        }

        eprintln!(
            "pre-cull paint cost: {ROWS} rows, {BATCHES}x{ITERATIONS} frames; best \
             {best_ms:.4} ms/frame, worst {worst_ms:.4}; {sprites} sprites reached the scene \
             out of {} glyphs painted",
            painted.get(),
        );
    }

    struct ScrolledTextView {
        mask: Bounds<Pixels>,
        line_height: Pixels,
        rows: usize,
        painted: Rc<Cell<usize>>,
    }

    impl Render for ScrolledTextView {
        fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
            let mask = self.mask;
            let line_height = self.line_height;
            let rows = self.rows;
            let painted = self.painted.clone();
            div().size_full().child(
                canvas(
                    |_, _, _| (),
                    move |bounds, _, window, cx| {
                        let runs = [TextRun {
                            len: TEXT.len(),
                            font: font("test"),
                            color: black(),
                            background_color: None,
                            underline: None,
                            strikethrough: None,
                        }];
                        let line = window
                            .text_system()
                            .shape_line(TEXT.into(), FONT_SIZE, &runs, None);
                        let glyphs_per_row: usize =
                            line.runs.iter().map(|run| run.glyphs.len()).sum();
                        painted.set(glyphs_per_row * rows);
                        // The viewport is the canvas itself, so the mask is guaranteed to be
                        // inside the window; `with_content_mask` intersects, and a mask outside
                        // the window would cull everything and measure nothing.
                        let _ = mask;
                        window.with_content_mask(
                            Some(ContentMask { bounds }),
                            |window| {
                                // Scroll most of the rows off the top so the cull has real work.
                                let scrolled = bounds.origin.y - line_height * (rows as f32 / 4.);
                                for row in 0..rows {
                                    let origin =
                                        point(bounds.origin.x, scrolled + line_height * row as f32);
                                    line.paint(
                                        origin,
                                        line_height,
                                        TextAlign::Left,
                                        None,
                                        window,
                                        cx,
                                    )
                                    .unwrap();
                                }
                            },
                        );
                    },
                )
                .size_full(),
            )
        }
    }
}
