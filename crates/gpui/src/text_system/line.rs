use crate::{
    App, Bounds, DevicePixels, Half, Hsla, LineLayout, Pixels, Point, RenderGlyphParams, Result,
    SharedString, StrikethroughStyle, TextAlign, UnderlineStyle, Window, WrapBoundary,
    WrappedLineLayout, black, fill, point, px, size, underline_y_offset,
};
use derive_more::{Deref, DerefMut};
use smallvec::SmallVec;
use std::{ops::Range, sync::Arc};

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
    /// Returns a forward-only cursor for this shaped line.
    pub fn cursor(&self) -> ShapedLineCursor<'_> {
        assert_eq!(
            self.len(),
            self.text.len(),
            "cannot split a shaped line with an adjusted length"
        );
        let byte_ordered = self
            .layout
            .runs
            .iter()
            .flat_map(|run| run.glyphs.iter().map(|glyph| glyph.index))
            .is_sorted();
        ShapedLineCursor {
            line: self,
            unordered_remainder: (!byte_ordered).then(|| self.clone()),
            byte_index: 0,
            run_index: 0,
            glyph_index: 0,
            decoration_index: 0,
            decoration_offset: 0,
            x_offset: px(0.),
        }
    }

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
        self.paint_with_underline_handler(
            origin,
            line_height,
            align,
            align_width,
            window,
            cx,
            |_, origin, width, style, window| window.paint_underline(origin, width, style),
        )
    }

    /// Paint the line with a handler for each underline.
    pub fn paint_with_underline_handler(
        &self,
        origin: Point<Pixels>,
        line_height: Pixels,
        align: TextAlign,
        align_width: Option<Pixels>,
        window: &mut Window,
        cx: &mut App,
        mut paint_underline: impl FnMut(
            Range<usize>,
            Point<Pixels>,
            Pixels,
            &UnderlineStyle,
            &mut Window,
        ),
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
            &mut paint_underline,
        )
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

/// Incrementally splits a [`ShapedLine`] at increasing UTF-8 byte boundaries.
///
/// Each piece preserves the original glyphs and decorations, rebased as in
/// [`ShapedLine::split_at`]. Byte-ordered glyphs are advanced in linear time,
/// copying each glyph and byte at most once; visually reordered glyphs fall
/// back to the existing split operation.
pub struct ShapedLineCursor<'a> {
    line: &'a ShapedLine,
    /// Bidirectional shaping can put glyphs out of byte order.
    unordered_remainder: Option<ShapedLine>,
    byte_index: usize,
    run_index: usize,
    glyph_index: usize,
    decoration_index: usize,
    decoration_offset: u32,
    x_offset: Pixels,
}

impl<'a> ShapedLineCursor<'a> {
    /// Takes the bytes since the previous boundary.
    ///
    /// Panics if the boundary precedes the previous one, exceeds the line's
    /// length, or falls inside a UTF-8 character.
    pub fn take_until(&mut self, byte_index: usize) -> ShapedLine {
        assert!(
            byte_index >= self.byte_index,
            "split boundary moved backwards"
        );
        assert!(
            byte_index <= self.line.len(),
            "split boundary exceeds line length"
        );
        assert!(
            self.line.text.is_char_boundary(byte_index),
            "split boundary is not a UTF-8 character boundary"
        );
        let previous_index = self.byte_index;
        let previous_x = self.x_offset;
        if let Some(remainder) = &mut self.unordered_remainder {
            let (piece, rest) = remainder.split_at(byte_index - previous_index);
            *remainder = rest;
            self.byte_index = byte_index;
            self.x_offset = self.line.layout.x_for_index(byte_index);
            return piece;
        }
        let mut runs = Vec::new();
        let mut next_x = self.line.layout.width;
        while let Some(run) = self.line.layout.runs.get(self.run_index) {
            let start = self.glyph_index;
            while let Some(glyph) = run.glyphs.get(self.glyph_index) {
                if glyph.index >= byte_index {
                    break;
                }
                self.glyph_index += 1;
            }
            let end = self.glyph_index;
            if start < end {
                runs.push(crate::ShapedRun {
                    font_id: run.font_id,
                    glyphs: run.glyphs[start..end]
                        .iter()
                        .map(|glyph| crate::ShapedGlyph {
                            id: glyph.id,
                            position: point(glyph.position.x - previous_x, glyph.position.y),
                            index: glyph.index - previous_index,
                            is_emoji: glyph.is_emoji,
                        })
                        .collect(),
                });
            }
            if let Some(glyph) = run.glyphs.get(self.glyph_index) {
                next_x = glyph.position.x;
                break;
            }
            self.run_index += 1;
            self.glyph_index = 0;
        }
        let mut decorations = SmallVec::new();
        while let Some(decoration) = self.line.decoration_runs.get(self.decoration_index)
            && (self.decoration_offset < byte_index as u32
                || (decoration.len == 0 && self.decoration_offset == byte_index as u32))
        {
            let end = self.decoration_offset + decoration.len;
            let start = self.decoration_offset.max(previous_index as u32);
            let len = end.min(byte_index as u32) - start;
            if len > 0 || decoration.len == 0 {
                decorations.push(DecorationRun {
                    len,
                    color: decoration.color,
                    background_color: decoration.background_color,
                    underline: decoration.underline,
                    strikethrough: decoration.strikethrough,
                });
            }
            if end <= byte_index as u32 {
                self.decoration_index += 1;
                self.decoration_offset = end;
            } else {
                break;
            }
        }
        self.byte_index = byte_index;
        self.x_offset = next_x;
        ShapedLine {
            layout: Arc::new(LineLayout {
                font_size: self.line.layout.font_size,
                width: next_x - previous_x,
                ascent: self.line.layout.ascent,
                descent: self.line.layout.descent,
                runs,
                len: byte_index - previous_index,
            }),
            text: SharedString::new(&self.line.text[previous_index..byte_index]),
            decoration_runs: decorations,
        }
    }

    /// Returns the original line's x position at the current boundary.
    pub fn x_offset(&self) -> Pixels {
        self.x_offset
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
            &mut |_, origin, width, style, window| window.paint_underline(origin, width, style),
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
            &mut |_, origin, width, style, window| window.paint_underline(origin, width, style),
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
    paint_underline: &mut dyn FnMut(
        Range<usize>,
        Point<Pixels>,
        Pixels,
        &UnderlineStyle,
        &mut Window,
    ),
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
        let underline_y_offset = underline_y_offset(line_height, layout.ascent, layout.descent);
        let mut decoration_runs = decoration_runs.iter();
        let mut wraps = wrap_boundaries.iter().peekable();
        let mut run_end = 0;
        let mut color = black();
        let mut current_underline: Option<(Point<Pixels>, UnderlineStyle, Range<usize>)> = None;
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
                    if let Some((underline_origin, underline_style, underline_range)) =
                        current_underline.as_mut()
                    {
                        if glyph_origin.x == underline_origin.x {
                            underline_origin.x -= max_glyph_size.width.half();
                        };
                        paint_underline(
                            underline_range.clone(),
                            *underline_origin,
                            glyph_origin.x - underline_origin.x,
                            underline_style,
                            window,
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

                let mut finished_underline: Option<(Point<Pixels>, UnderlineStyle, Range<usize>)> =
                    None;
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
                        let style_run_start = run_end;
                        if let Some((_, underline_style, underline_range)) = &mut current_underline
                        {
                            if style_run.underline.as_ref() != Some(underline_style) {
                                finished_underline = current_underline.take();
                            } else {
                                underline_range.end = style_run_start + style_run.len as usize;
                            }
                        }
                        if let Some(run_underline) = style_run.underline.as_ref() {
                            current_underline.get_or_insert((
                                point(glyph_origin.x, glyph_origin.y + underline_y_offset),
                                UnderlineStyle {
                                    color: Some(run_underline.color.unwrap_or(style_run.color)),
                                    thickness: run_underline.thickness,
                                    wavy: run_underline.wavy,
                                },
                                style_run_start..style_run_start + style_run.len as usize,
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

                if let Some((mut underline_origin, underline_style, underline_range)) =
                    finished_underline
                {
                    if underline_origin.x == glyph_origin.x {
                        underline_origin.x -= max_glyph_size.width.half();
                    };
                    paint_underline(
                        underline_range,
                        underline_origin,
                        glyph_origin.x - underline_origin.x,
                        &underline_style,
                        window,
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

        if let Some((mut underline_start, underline_style, underline_range)) =
            current_underline.take()
        {
            if last_line_end_x == underline_start.x {
                underline_start.x -= max_glyph_size.width.half()
            };
            paint_underline(
                underline_range,
                underline_start,
                last_line_end_x - underline_start.x,
                &underline_style,
                window,
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
    use crate::{
        AppContext as _, Context, FontId, GlyphId, IntoElement, Render, ShapedGlyph, ShapedRun,
        Styled, TestAppContext, TextRun, Underline, canvas, font, hsla,
    };
    use std::rc::Rc;

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

    #[gpui::test]
    fn test_underline_handler_matches_default_paint(cx: &mut TestAppContext) {
        test_underline_handler_at_scales(cx, |window, cx| {
            let first_style = UnderlineStyle {
                thickness: px(1.),
                color: Some(hsla(0., 1., 0.5, 1.)),
                wavy: true,
            };
            let last_style = UnderlineStyle {
                color: Some(hsla(0.5, 1., 0.5, 1.)),
                wavy: false,
                ..first_style
            };
            let fallback_style = UnderlineStyle {
                color: Some(black()),
                ..first_style
            };
            let decoration = DecorationRun {
                len: 1,
                color: black(),
                background_color: None,
                underline: Some(first_style),
                strikethrough: None,
            };
            let line = underline_test_line(
                "aébcde",
                &[
                    decoration.clone(),
                    DecorationRun {
                        len: 2,
                        ..decoration.clone()
                    },
                    DecorationRun {
                        underline: None,
                        ..decoration.clone()
                    },
                    DecorationRun {
                        underline: Some(UnderlineStyle {
                            color: None,
                            ..first_style
                        }),
                        ..decoration.clone()
                    },
                    DecorationRun {
                        underline: Some(last_style),
                        ..decoration.clone()
                    },
                    DecorationRun {
                        underline: Some(last_style),
                        ..decoration
                    },
                ],
                false,
                window,
            );
            assert_eq!(line.width(), px(48.));
            for origin_x in [-3.25, 0., 4.25] {
                for (align, align_width, offset) in [
                    (TextAlign::Left, None, 0.),
                    (TextAlign::Center, None, 0.),
                    (TextAlign::Right, None, 0.),
                    (TextAlign::Left, Some(px(96.)), 0.),
                    (TextAlign::Center, Some(px(96.)), 24.),
                    (TextAlign::Right, Some(px(96.)), 48.),
                ] {
                    let origin = point(px(origin_x), px(12.25));
                    let line_height = px(20.);
                    window.next_frame.scene.clear();
                    line.paint(origin, line_height, align, align_width, window, cx)
                        .unwrap();
                    let original = window.next_frame.scene.underlines.clone();
                    window.next_frame.scene.clear();
                    line.layout
                        .paint(
                            origin,
                            line_height,
                            align,
                            align_width,
                            &line.decoration_runs,
                            window,
                            cx,
                        )
                        .unwrap();
                    assert_underline_primitives_eq(&window.next_frame.scene.underlines, &original);

                    window.next_frame.scene.clear();
                    let mut strokes = Vec::new();
                    line.paint_with_underline_handler(
                        origin,
                        line_height,
                        align,
                        align_width,
                        window,
                        cx,
                        |range, origin, width, style, window| {
                            strokes.push((range, origin, width, *style));
                            window.paint_underline(origin, width, style);
                        },
                    )
                    .unwrap();
                    let start = px(origin_x + offset);
                    let y = origin.y + underline_y_offset(line_height, line.ascent, line.descent);
                    assert_eq!(
                        strokes,
                        [
                            (0..3, point(start, y), px(16.), first_style),
                            (4..5, point(start + px(24.), y), px(8.), fallback_style),
                            (5..7, point(start + px(32.), y), px(16.), last_style),
                        ]
                    );
                    assert_underline_primitives_eq(&window.next_frame.scene.underlines, &original);

                    window.next_frame.scene.clear();
                    let mut captured = Vec::new();
                    line.paint_with_underline_handler(
                        origin,
                        line_height,
                        align,
                        align_width,
                        window,
                        cx,
                        |range, origin, width, style, _| {
                            captured.push((range, origin, width, *style))
                        },
                    )
                    .unwrap();
                    assert_eq!(captured, strokes);
                    assert_eq!(window.next_frame.scene.underlines.len(), 0);
                }
            }
            for text in ["", "abc"] {
                let line = underline_test_line(text, &[], false, window);
                let mut calls = 0;
                line.paint_with_underline_handler(
                    point(px(4.), px(10.)),
                    px(20.),
                    TextAlign::Left,
                    None,
                    window,
                    cx,
                    |_, _, _, _, _| calls += 1,
                )
                .unwrap();
                assert_eq!(calls, 0);
            }
        });
    }

    #[gpui::test]
    fn test_underline_handler_reports_zero_advance_geometry(cx: &mut TestAppContext) {
        test_underline_handler_at_scales(cx, |window, cx| {
            let first_style = UnderlineStyle {
                thickness: px(1.),
                color: Some(black()),
                wavy: true,
            };
            let last_style = UnderlineStyle {
                wavy: false,
                ..first_style
            };
            let decoration = DecorationRun {
                len: 1,
                color: black(),
                background_color: None,
                underline: Some(first_style),
                strikethrough: None,
            };
            let line = underline_test_line(
                "ab",
                &[
                    decoration.clone(),
                    DecorationRun {
                        underline: Some(last_style),
                        ..decoration
                    },
                ],
                true,
                window,
            );
            let half_width = cx
                .text_system()
                .bounding_box(line.runs[0].font_id, line.font_size)
                .size
                .width
                / 2.;
            let origin = point(px(40.25), px(10.25));
            let line_height = px(20.);
            let y = origin.y + underline_y_offset(line_height, line.ascent, line.descent);
            for (align, offset) in [
                (TextAlign::Left, 0.),
                (TextAlign::Center, 16.),
                (TextAlign::Right, 32.),
            ] {
                window.next_frame.scene.clear();
                line.paint(origin, line_height, align, Some(px(32.)), window, cx)
                    .unwrap();
                let original = window.next_frame.scene.underlines.clone();
                window.next_frame.scene.clear();
                let mut strokes = Vec::new();
                line.paint_with_underline_handler(
                    origin,
                    line_height,
                    align,
                    Some(px(32.)),
                    window,
                    cx,
                    |range, origin, width, style, window| {
                        strokes.push((range, origin, width, *style));
                        window.paint_underline(origin, width, style);
                    },
                )
                .unwrap();
                let end = origin.x + px(offset);
                let start = point(end - half_width, y);
                let width = end - start.x;
                assert_eq!(
                    strokes,
                    [
                        (0..1, start, width, first_style),
                        (1..2, start, width, last_style),
                    ]
                );
                assert_underline_primitives_eq(&window.next_frame.scene.underlines, &original);
            }
        });
    }

    #[gpui::test]
    fn test_underline_handler_matches_wrapped_paint(cx: &mut TestAppContext) {
        test_underline_handler_at_scales(cx, |window, cx| {
            let style = UnderlineStyle {
                thickness: px(1.),
                color: Some(black()),
                wavy: true,
            };
            for zero_advance in [false, true] {
                let line = underline_test_line(
                    "abcd",
                    &[DecorationRun {
                        len: 4,
                        color: black(),
                        background_color: None,
                        underline: Some(style),
                        strikethrough: None,
                    }],
                    zero_advance,
                    window,
                );
                let half_width = cx
                    .text_system()
                    .bounding_box(line.runs[0].font_id, line.font_size)
                    .size
                    .width
                    / 2.;
                let origin = point(px(40.25), px(10.25));
                let line_height = px(20.);
                let y = origin.y + underline_y_offset(line_height, line.ascent, line.descent);
                let wrapped = WrappedLine {
                    layout: Arc::new(WrappedLineLayout {
                        unwrapped_layout: line.layout,
                        wrap_boundaries: SmallVec::from_buf([WrapBoundary {
                            run_ix: 0,
                            glyph_ix: 2,
                        }]),
                        wrap_width: Some(px(16.)),
                    }),
                    text: line.text,
                    decoration_runs: line.decoration_runs.into_vec(),
                };
                window.next_frame.scene.clear();
                wrapped
                    .paint(origin, line_height, TextAlign::Left, None, window, cx)
                    .unwrap();
                let original = window.next_frame.scene.underlines.clone();
                window.next_frame.scene.clear();
                let mut strokes = Vec::new();
                paint_line(
                    origin,
                    &wrapped.unwrapped_layout,
                    line_height,
                    TextAlign::Left,
                    Some(px(16.)),
                    &wrapped.decoration_runs,
                    &wrapped.wrap_boundaries,
                    window,
                    cx,
                    &mut |range, origin, width, style, window| {
                        strokes.push((range, origin, width, *style));
                        window.paint_underline(origin, width, style);
                    },
                )
                .unwrap();
                let (start, width) = if zero_advance {
                    let start = origin.x - half_width;
                    (start, origin.x - start)
                } else {
                    (origin.x, px(16.))
                };
                assert_eq!(
                    strokes,
                    [
                        (0..4, point(start, y), width, style),
                        (0..4, point(start, y + line_height), width, style),
                    ]
                );
                assert_underline_primitives_eq(&window.next_frame.scene.underlines, &original);
            }
        });
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

    struct UnderlineHandlerTestView(Rc<dyn Fn(&mut Window, &mut App)>);

    impl Render for UnderlineHandlerTestView {
        fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
            let paint = self.0.clone();
            canvas(
                |_, _, _| {},
                move |_, _, window, cx| {
                    window.with_element_opacity(Some(0.5), |window| paint(window, cx));
                },
            )
            .size_full()
        }
    }

    fn test_underline_handler_at_scales(
        cx: &mut TestAppContext,
        paint: impl Fn(&mut Window, &mut App) + 'static,
    ) {
        let window = cx.add_window(move |_, _| UnderlineHandlerTestView(Rc::new(paint)));
        for scale in [1., 1.25, 1.5, 2., 3.] {
            cx.simulate_window_scale_factor_change(window.into(), scale);
            cx.update_window(window.into(), |_, window, cx| window.draw(cx).clear(cx))
                .unwrap();
        }
    }

    fn underline_test_line(
        text: &str,
        decorations: &[DecorationRun],
        zero_advance: bool,
        window: &Window,
    ) -> ShapedLine {
        let mut line = window.text_system().shape_line(
            SharedString::new(text),
            px(16.),
            &[TextRun {
                len: text.len(),
                font: font(".ZedMono"),
                color: black(),
                ..TextRun::default()
            }],
            None,
        );
        line.decoration_runs = SmallVec::from(decorations.to_vec());
        let layout = &line.layout;
        let mut runs = layout.runs.clone();
        let advance = if zero_advance { px(0.) } else { px(8.) };
        let mut width = px(0.);
        for glyph in runs.iter_mut().flat_map(|run| &mut run.glyphs) {
            glyph.position.x = width;
            width += advance;
        }
        line.layout = Arc::new(LineLayout {
            font_size: layout.font_size,
            width,
            ascent: layout.ascent,
            descent: layout.descent,
            runs,
            len: layout.len,
        });
        line
    }

    fn assert_underline_primitives_eq(actual: &[Underline], expected: &[Underline]) {
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            assert_eq!(actual.bounds, expected.bounds);
            assert_eq!(actual.content_mask, expected.content_mask);
            assert_eq!(actual.color, expected.color);
            assert_eq!(actual.thickness, expected.thickness);
            assert_eq!(actual.wavy, expected.wavy);
            assert_eq!(actual.order, expected.order);
            assert_eq!(actual.pad, expected.pad);
        }
    }

    #[test]
    fn test_cursor_preserves_shaping_metadata_across_runs() {
        let line = ShapedLine {
            layout: Arc::new(LineLayout {
                font_size: px(16.0),
                width: px(50.0),
                ascent: px(12.0),
                descent: px(4.0),
                runs: vec![
                    ShapedRun {
                        font_id: FontId(3),
                        glyphs: vec![
                            ShapedGlyph {
                                id: GlyphId(11),
                                position: point(px(0.0), px(1.0)),
                                index: 0,
                                is_emoji: true,
                            },
                            ShapedGlyph {
                                id: GlyphId(12),
                                position: point(px(17.0), px(1.0)),
                                index: 1,
                                is_emoji: false,
                            },
                            ShapedGlyph {
                                id: GlyphId(13),
                                position: point(px(19.0), px(-1.0)),
                                index: 1,
                                is_emoji: false,
                            },
                        ],
                    },
                    ShapedRun {
                        font_id: FontId(8),
                        glyphs: vec![
                            ShapedGlyph {
                                id: GlyphId(21),
                                position: point(px(25.0), px(1.0)),
                                index: 5,
                                is_emoji: true,
                            },
                            ShapedGlyph {
                                id: GlyphId(22),
                                position: point(px(41.0), px(1.0)),
                                index: 7,
                                is_emoji: false,
                            },
                        ],
                    },
                ],
                len: 10,
            }),
            text: "a😀bcdef".into(),
            decoration_runs: SmallVec::new(),
        };
        let mut cursor = line.cursor();
        let first = cursor.take_until(5);
        assert_eq!(first.text.as_ref(), "a😀");
        assert_eq!(first.runs[0].font_id, FontId(3));
        assert_eq!(first.runs[0].glyphs[0].id, GlyphId(11));
        assert!(first.runs[0].glyphs[0].is_emoji);
        assert_eq!(first.runs[0].glyphs[1].index, 1);
        assert_eq!(first.runs[0].glyphs[1].position, point(px(17.0), px(1.0)));
        assert_eq!(first.runs[0].glyphs.len(), 3);
        assert_eq!(first.runs[0].glyphs[2].index, 1);
        assert_eq!(first.runs[0].glyphs[2].position, point(px(19.0), px(-1.0)));
        assert_eq!(cursor.x_offset(), px(25.0));

        let second = cursor.take_until(7);
        assert_eq!(second.text.as_ref(), "bc");
        assert_eq!(second.runs[0].font_id, FontId(8));
        assert_eq!(second.runs[0].glyphs[0].id, GlyphId(21));
        assert_eq!(second.runs[0].glyphs[0].index, 0);
        assert_eq!(second.runs[0].glyphs[0].position, point(px(0.0), px(1.0)));
        assert_eq!(cursor.x_offset(), px(41.0));

        let final_part = cursor.take_until(10);
        assert_eq!(final_part.text.as_ref(), "def");
        assert_eq!(final_part.runs[0].font_id, FontId(8));
        assert_eq!(final_part.runs[0].glyphs[0].id, GlyphId(22));
        assert_eq!(final_part.runs[0].glyphs[0].index, 0);
        assert_eq!(
            final_part.runs[0].glyphs[0].position,
            point(px(0.0), px(1.0))
        );
    }

    #[test]
    fn test_cursor_preserves_existing_visual_order_splitting() {
        let line = make_shaped_line("abc", &[(0, 0.0), (2, 10.0), (1, 20.0)], 30.0, &[]);
        let mut cursor = line.cursor();
        let mut remainder = line.clone();
        let mut previous_boundary = 0;
        for boundary in [0, 1, 2, 3] {
            let (expected, rest) = remainder.split_at(boundary - previous_boundary);
            let actual = cursor.take_until(boundary);
            assert_eq!(actual.text, expected.text);
            assert_eq!(actual.width(), expected.width());
            assert_eq!(actual.runs.len(), expected.runs.len());
            for (actual, expected) in actual.runs.iter().zip(&expected.runs) {
                assert_eq!(actual.font_id, expected.font_id);
                assert_eq!(actual.glyphs.len(), expected.glyphs.len());
                for (actual, expected) in actual.glyphs.iter().zip(&expected.glyphs) {
                    assert_eq!(actual.id, expected.id);
                    assert_eq!(actual.index, expected.index);
                    assert_eq!(actual.position, expected.position);
                }
            }
            assert_eq!(cursor.x_offset(), line.x_for_index(boundary));
            remainder = rest;
            previous_boundary = boundary;
        }
    }

    #[test]
    fn test_cursor_partitions_one_decoration_across_three_chunks() {
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
            &[DecorationRun {
                len: 6,
                color: Hsla {
                    h: 0.2,
                    s: 0.4,
                    l: 0.6,
                    a: 1.0,
                },
                background_color: None,
                underline: None,
                strikethrough: None,
            }],
        );
        let mut cursor = line.cursor();
        assert_eq!(cursor.take_until(2).decoration_runs[0].len, 2);
        assert_eq!(cursor.take_until(4).decoration_runs[0].len, 2);
        assert_eq!(cursor.take_until(6).decoration_runs[0].len, 2);
    }

    #[test]
    fn test_cursor_matches_successive_splits_at_ordered_boundaries() {
        let decorations: Vec<_> = [2, 0, 3, 1]
            .into_iter()
            .map(|len| DecorationRun {
                len,
                color: Hsla {
                    h: len as f32 / 10.0,
                    s: 0.5,
                    l: 0.5,
                    a: 1.0,
                },
                background_color: Some(black()),
                underline: None,
                strikethrough: None,
            })
            .collect();
        let line = make_shaped_line(
            "abcdef",
            &[(0, 5.0), (0, 5.0), (2, 15.0), (4, 25.0), (5, 35.0)],
            45.0,
            &decorations,
        );
        for first in 0..=line.len() {
            for second in first..=line.len() {
                let mut cursor = line.cursor();
                let mut remainder = line.clone();
                let mut previous_boundary = 0;
                let mut total_width = px(0.0);
                let mut text = String::new();
                for boundary in [first, second, line.len(), line.len()] {
                    let (expected, rest) = remainder.split_at(boundary - previous_boundary);
                    let actual = cursor.take_until(boundary);
                    assert_eq!(actual.text, expected.text);
                    assert_eq!(actual.len(), expected.len());
                    assert_eq!(actual.width(), expected.width());
                    assert_eq!(actual.runs.len(), expected.runs.len());
                    for (actual, expected) in actual.runs.iter().zip(&expected.runs) {
                        assert_eq!(actual.font_id, expected.font_id);
                        assert_eq!(actual.glyphs.len(), expected.glyphs.len());
                        for (actual, expected) in actual.glyphs.iter().zip(&expected.glyphs) {
                            assert_eq!(actual.id, expected.id);
                            assert_eq!(actual.index, expected.index);
                            assert_eq!(actual.position, expected.position);
                        }
                    }
                    assert_eq!(actual.decoration_runs.len(), expected.decoration_runs.len());
                    for (actual, expected) in
                        actual.decoration_runs.iter().zip(&expected.decoration_runs)
                    {
                        assert_eq!(actual.len, expected.len);
                        assert_eq!(actual.color, expected.color);
                        assert_eq!(actual.background_color, expected.background_color);
                    }
                    total_width += actual.width();
                    text.push_str(&actual.text);
                    remainder = rest;
                    previous_boundary = boundary;
                }
                assert_eq!(total_width, line.width());
                assert_eq!(text, line.text.as_ref());
            }
        }
    }

    #[test]
    fn test_cursor_empty_chunks_and_repeated_boundaries() {
        let line = make_shaped_line("ab", &[(0, 5.0), (1, 15.0)], 20.0, &[]);
        let mut cursor = line.cursor();
        assert_eq!(cursor.take_until(0).text.as_ref(), "");
        assert_eq!(cursor.take_until(0).text.as_ref(), "");
        assert_eq!(cursor.take_until(1).text.as_ref(), "a");
        assert_eq!(cursor.take_until(2).text.as_ref(), "b");
        assert_eq!(cursor.take_until(2).text.as_ref(), "");
        let empty = make_shaped_line("", &[], 0.0, &[]);
        let piece = empty.cursor().take_until(0);
        assert!(piece.text.is_empty());
        assert!(piece.runs.is_empty());
        assert_eq!(piece.width(), px(0.0));
    }

    #[test]
    fn test_cursor_rejects_invalid_boundaries() {
        let line = make_shaped_line("é", &[(0, 0.0)], 10.0, &[]);
        let mut cursor = line.cursor();
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                cursor.take_until(1);
            }))
            .is_err()
        );
        let mut cursor = line.cursor();
        cursor.take_until(2);
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                cursor.take_until(0);
            }))
            .is_err()
        );
        let mut cursor = line.cursor();
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                cursor.take_until(3);
            }))
            .is_err()
        );
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
