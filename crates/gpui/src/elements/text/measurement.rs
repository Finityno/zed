//! Measuring a text element, and carrying the measurement over to the text
//! element that takes its layout node on the next frame.
//!
//! A kept layout node given a new measurement closure has to be dirtied, and
//! with it every node above it, since nothing says the closure measures what
//! the old one did. Text says so: the element hands its text, runs and style
//! to its measurement, and the next frame's element at the same node compares
//! its own against them. When they would shape the same way, the measurement
//! is carried over (repainted with the new decorations if only those changed)
//! and the node stays clean.

use super::{TextLayout, TextLayoutInner};
use crate::{
    App, AvailableSpace, DecorationRun, Hsla, LayoutId, Pixels, SharedString, Size,
    StrikethroughStyle, TextOverflow, TextRun, TextStyle, TruncateFrom, UnderlineStyle,
    WhiteSpace, Window, WrappedLine, taffy::Adopted,
};
use gpui_util::ResultExt as _;
use std::{any::Any, borrow::Cow, cell::RefCell, cmp, rc::Rc};

/// Everything a text element's measurement is taken from, and the layout the
/// measurement is kept in.
pub(super) struct TextMeasureInputs {
    text: SharedString,
    /// The runs the element was given, or none for plain text, which is one
    /// run in the text style, made only when the text has to be shaped.
    runs: Vec<TextRun>,
    text_style: TextStyle,
    font_size: Pixels,
    line_height: Pixels,
    /// The text system's font generation: fonts added since can shape the
    /// same text differently.
    font_generation: usize,
    /// The layout of the element whose inputs these are, or of a later one
    /// that took the measurement over while leaving the node with these.
    layout: RefCell<TextLayout>,
}

type Decoration = (
    Hsla,
    Option<Hsla>,
    Option<UnderlineStyle>,
    Option<StrikethroughStyle>,
);

fn decoration_of(run: &TextRun) -> Decoration {
    (
        run.color,
        run.background_color,
        run.underline,
        run.strikethrough,
    )
}

fn style_decoration(style: &TextStyle) -> Decoration {
    (
        style.color,
        style.background_color,
        style.underline,
        style.strikethrough,
    )
}

/// Whether two text styles make the same font, without making either.
fn same_font(style: &TextStyle, other: &TextStyle) -> bool {
    style.font_family == other.font_family
        && style.font_features == other.font_features
        && style.font_fallbacks == other.font_fallbacks
        && style.font_weight == other.font_weight
        && style.font_style == other.font_style
}

fn nonempty_runs(runs: &[TextRun]) -> impl Iterator<Item = &TextRun> {
    runs.iter().filter(|run| run.len > 0)
}

/// For each run after the first, whether it has the decorations of the run
/// before it: shaping splits font runs wherever decorations change.
fn joins_previous<'a>(runs: impl Iterator<Item = &'a TextRun>) -> impl Iterator<Item = bool> {
    let mut previous = None;
    runs.map(move |run| {
        previous
            .replace(decoration_of(run))
            .is_some_and(|previous| previous == decoration_of(run))
    })
}

impl TextMeasureInputs {
    pub(super) fn new(
        text: SharedString,
        runs: Option<Vec<TextRun>>,
        layout: &TextLayout,
        window: &Window,
    ) -> Self {
        let text_style = window.text_style();
        let font_size = text_style.font_size.to_pixels(window.rem_size());
        let line_height = window.pixel_snap(
            text_style
                .line_height
                .to_pixels(font_size.into(), window.rem_size()),
        );
        Self {
            text,
            runs: runs.unwrap_or_default(),
            text_style,
            font_size,
            line_height,
            font_generation: window.text_system().font_generation(),
            layout: RefCell::new(layout.clone()),
        }
    }

    /// Whether the text truncates, in which case it is shaped from a
    /// rewritten string whose runs no longer line up with these.
    fn truncates(&self) -> bool {
        self.text_style.text_overflow.is_some()
    }

    fn plain(&self) -> bool {
        self.runs.is_empty()
    }

    fn runs(&self) -> Cow<'_, [TextRun]> {
        if self.plain() {
            Cow::Owned(vec![self.text_style.to_run(self.text.len())])
        } else {
            Cow::Borrowed(&self.runs)
        }
    }

    /// Whether `self` is shaped as `other` is: the same text, sizes, fonts and
    /// wrapping, and decorations changing in the same places. What it is
    /// painted with may differ; see [`Self::decorated_as`].
    fn shapes_as(&self, other: &Self) -> bool {
        let (style, other_style) = (&self.text_style, &other.text_style);
        if !(self.text == other.text
            && self.font_size == other.font_size
            && self.line_height == other.line_height
            && self.font_generation == other.font_generation
            && style.white_space == other_style.white_space
            && style.line_clamp == other_style.line_clamp
            && style.text_overflow == other_style.text_overflow
            // Only truncation reads the style's own font.
            && (!self.truncates() || same_font(style, other_style)))
        {
            return false;
        }
        if self.plain() && other.plain() {
            return self.text.is_empty() || same_font(style, other_style);
        }
        let (runs, other_runs) = (self.runs(), other.runs());
        nonempty_runs(&runs).count() == nonempty_runs(&other_runs).count()
            && nonempty_runs(&runs)
                .zip(nonempty_runs(&other_runs))
                .all(|(run, other)| run.len == other.len && run.font == other.font)
            && joins_previous(nonempty_runs(&runs)).eq(joins_previous(nonempty_runs(&other_runs)))
    }

    /// Whether `self` is painted with what `other` is.
    fn decorated_as(&self, other: &Self) -> bool {
        if self.plain() && other.plain() {
            return self.text.is_empty()
                || style_decoration(&self.text_style) == style_decoration(&other.text_style);
        }
        let (runs, other_runs) = (self.runs(), other.runs());
        nonempty_runs(&runs)
            .map(decoration_of)
            .eq(nonempty_runs(&other_runs).map(decoration_of))
    }
}

/// Requests the layout of a text element whose measurement is kept in
/// `layout`, carrying last frame's measurement over when it still stands.
pub(super) fn layout_text(
    layout: &TextLayout,
    text: SharedString,
    runs: Option<Vec<TextRun>>,
    window: &mut Window,
    cx: &mut App,
) -> LayoutId {
    let inputs = TextMeasureInputs::new(text, runs, layout, window);
    window.request_carried_measured_layout(inputs, adopt_measurement, measure_text, cx)
}

/// Takes over the measurement `previous` left, if it stands for `inputs`.
///
/// When only the decorations differ, the measurement is repainted with this
/// element's and the node is given this element's inputs, to measure from
/// when Taffy measures it under other constraints. Otherwise the node keeps
/// last frame's inputs, which measure it the same way, and only its layout is
/// pointed at this element's, where the measurement is kept from now on.
fn adopt_measurement(inputs: &TextMeasureInputs, previous: &dyn Any) -> Adopted {
    let Some(previous) = previous.downcast_ref::<TextMeasureInputs>() else {
        return Adopted::No;
    };
    if !previous.shapes_as(inputs) {
        return Adopted::No;
    }
    let recolored = !previous.decorated_as(inputs);
    if recolored && inputs.truncates() {
        return Adopted::No;
    }
    let Some(mut inner) = carry_measurement(&previous.layout.borrow()) else {
        return Adopted::No;
    };
    let layout = inputs.layout.borrow();
    if recolored {
        update_decoration_runs(&mut inner.lines, &inputs.runs());
        *layout.0.borrow_mut() = Some(inner);
        Adopted::Measurement
    } else {
        *layout.0.borrow_mut() = Some(inner);
        *previous.layout.borrow_mut() = layout.clone();
        Adopted::Node
    }
}

/// What the measurement kept in `layout` left, without where it was painted.
///
/// Last frame's element is usually gone, and its layout held only by what it
/// left for this frame's, in which case the measurement is moved rather than
/// copied line by line. Something may still hold it (a caller's clone of a
/// `StyledText`'s layout, say), and then it is copied, so it still answers.
fn carry_measurement(layout: &TextLayout) -> Option<TextLayoutInner> {
    if Rc::strong_count(&layout.0) == 1 {
        let mut inner = layout.0.borrow_mut().take()?;
        inner.bounds = None;
        Some(inner)
    } else {
        layout.0.borrow().as_ref().map(|inner| TextLayoutInner {
            text_align: inner.text_align,
            len: inner.len,
            lines: inner
                .lines
                .iter()
                .map(|line| WrappedLine {
                    layout: line.layout.clone(),
                    text: line.text.clone(),
                    decoration_runs: line.decoration_runs.clone(),
                })
                .collect(),
            line_height: inner.line_height,
            wrap_width: inner.wrap_width,
            truncate_width: inner.truncate_width,
            size: inner.size,
            bounds: None,
        })
    }
}

/// Rewrites the decorations of lines already shaped, leaving the shaping
/// alone. `runs` must split the lines as the runs they were shaped with did;
/// see [`TextMeasureInputs::shapes_as`].
fn update_decoration_runs(lines: &mut [WrappedLine], runs: &[TextRun]) {
    let mut runs = runs.iter().filter(|run| run.len > 0).cloned().peekable();
    for line in lines.iter_mut() {
        let line_len = line.text.len();
        line.decoration_runs.clear();
        let mut offset = 0;
        while offset < line_len {
            let Some(run) = runs.peek_mut() else {
                log::warn!("`TextRun`s do not cover the entire shaped text");
                break;
            };
            let len_within_line = cmp::min(line_len - offset, run.len);
            if let Some(last_run) = line.decoration_runs.last_mut()
                && last_run.color == run.color
                && last_run.underline == run.underline
                && last_run.strikethrough == run.strikethrough
                && last_run.background_color == run.background_color
            {
                last_run.len += len_within_line as u32;
            } else {
                line.decoration_runs.push(DecorationRun {
                    len: len_within_line as u32,
                    color: run.color,
                    background_color: run.background_color,
                    underline: run.underline,
                    strikethrough: run.strikethrough,
                });
            }
            run.len -= len_within_line;
            if run.len == 0 {
                runs.next();
            }
            offset += len_within_line;
        }
        // Skip the `\n` that separated this line from the next.
        if let Some(run) = runs.peek_mut() {
            run.len -= 1;
            if run.len == 0 {
                runs.next();
            }
        }
    }
}

/// Measures text under the constraints Taffy offers, keeping the result in
/// its layout.
fn measure_text(
    inputs: &TextMeasureInputs,
    known_dimensions: Size<Option<Pixels>>,
    available_space: Size<AvailableSpace>,
    window: &mut Window,
    cx: &mut App,
) -> Size<Pixels> {
    let TextMeasureInputs {
        text,
        text_style,
        font_size,
        line_height,
        layout,
        ..
    } = inputs;
    let element_state = &*layout.borrow();
    let (font_size, line_height) = (*font_size, *line_height);
    let wrap_width = if text_style.white_space == WhiteSpace::Normal {
        known_dimensions.width.or(match available_space.width {
            AvailableSpace::Definite(x) => Some(x),
            _ => None,
        })
    } else {
        None
    };

    let truncate_width = text_style.text_overflow.as_ref().and_then(|_| {
        known_dimensions.width.or(match available_space.width {
            AvailableSpace::Definite(x) => match text_style.line_clamp {
                Some(max_lines) => Some(x * max_lines),
                None => Some(x),
            },
            _ => None,
        })
    });

    // Only use cached layout if:
    // 1. We have a cached size
    // 2. wrap_width matches EXACTLY (including None == None), or the
    //    cached layout was shaped unwrapped and already fits the
    //    width now offered (see below)
    // 3. truncate_width matches (a layout computed without truncation
    //    cannot answer for one with it, nor the other way round)
    //
    // FINCODE FORK: upstream's check here is
    // `wrap_width.is_none() || wrap_width == text_layout.wrap_width`,
    // which answers intrinsic-sizing probes (Taffy min-/max-content,
    // wrap_width == None) with a size shaped at whatever definite
    // width happened to be cached. That poisons flex sizing: if one
    // transient frame shaped this text at a collapsed width (the
    // cached size is then ~one glyph wide and very tall), the next
    // probe is answered with that collapsed size, the parent flex
    // node is sized around it, and the definite width handed back
    // re-creates the collapsed shape — a self-perpetuating fixpoint
    // that paints transcript text one character per line until some
    // unrelated change rebuilds the element. Probes must therefore
    // only reuse a layout produced under the same wrap_width.
    //
    // The other direction is exact: Taffy probes a wrapping leaf
    // unconstrained before laying it out at a definite width, and
    // text whose unwrapped shape is no wider than that width wraps
    // nowhere, so the unwrapped lines are the wrapped lines. Only
    // text the width actually bites is shaped a second time.
    if let Some(text_layout) = element_state.0.borrow().as_ref()
        && let Some(size) = text_layout.size
        && truncate_width == text_layout.truncate_width
        && (wrap_width == text_layout.wrap_width
            || (text_layout.wrap_width.is_none()
                && truncate_width.is_none()
                && wrap_width.is_some_and(|wrap_width| size.width <= wrap_width)))
    {
        return size;
    }

    // FINCODE FORK: when an intrinsic probe (wrap_width == None)
    // misses the cache above, it shapes unwrapped below — the
    // correct intrinsic answer. But it must NOT overwrite a cached
    // definite-width layout: paint() draws whatever lines are
    // stored here into the assigned bounds without re-wrapping, so
    // clobbering the wrapped lines with an unwrapped probe shape
    // would paint the text as one long unwrapped line. Probes
    // compute their answer and leave the stored layout untouched.
    let preserve_cached_layout = wrap_width.is_none()
        && element_state
            .0
            .borrow()
            .as_ref()
            .is_some_and(|cached| cached.wrap_width.is_some() && cached.size.is_some());

    let runs = inputs.runs();
    let runs = &*runs;
    // Only truncation needs a line wrapper and an affix: taking the
    // wrapper resolves the font and locks the wrapper pool.
    let (text, runs) = if let Some(truncate_width) = truncate_width
        && let Some(text_overflow) = text_style.text_overflow.as_ref()
    {
        let (truncation_affix, truncate_from) = match text_overflow {
            TextOverflow::Truncate(affix) => (affix, TruncateFrom::End),
            TextOverflow::TruncateStart(affix) => (affix, TruncateFrom::Start),
            TextOverflow::TruncateMiddle(affix) => (affix, TruncateFrom::Middle),
        };
        let mut line_wrapper = cx.text_system().line_wrapper(text_style.font(), font_size);
        if let Some(max_lines) = text_style.line_clamp
            && let Some(wrap_width) = wrap_width
        {
            line_wrapper.truncate_wrapped_line(
                text.clone(),
                wrap_width,
                max_lines,
                truncation_affix,
                runs,
                truncate_from,
            )
        } else if let Some(unclipped) = window
            .text_system()
            .shape_text(text.clone(), font_size, runs, None, None)
            .log_err()
            && unclipped
                .iter()
                .all(|line| line.size(line_height).width <= truncate_width)
        {
            // The truncation decision below sums per-character advances,
            // which overestimates the shaped width (no kerning), truncating
            // text that fits exactly in its measured width. Skip truncation
            // whenever the honestly-shaped text fits; the shaping result
            // comes from the line layout cache when the same text was
            // already measured untruncated this frame.
            (text.clone(), Cow::Borrowed(runs))
        } else {
            line_wrapper.truncate_line(
                text.clone(),
                truncate_width,
                truncation_affix,
                runs,
                truncate_from,
            )
        }
    } else {
        (text.clone(), Cow::Borrowed(runs))
    };
    let len = text.len();

    let Some(lines) = window
        .text_system()
        .shape_text(text, font_size, &runs, wrap_width, text_style.line_clamp)
        .log_err()
    else {
        if !preserve_cached_layout {
            element_state.0.borrow_mut().replace(TextLayoutInner {
                text_align: text_style.text_align,
                lines: Default::default(),
                len: 0,
                line_height,
                wrap_width,
                truncate_width,
                size: Some(Size::default()),
                bounds: None,
            });
        }
        return Size::default();
    };

    let mut size: Size<Pixels> = Size::default();
    for line in &lines {
        let line_size = line.size(line_height);
        size.height += line_size.height;
        size.width = size.width.max(line_size.width).ceil();
    }

    if !preserve_cached_layout {
        element_state.0.borrow_mut().replace(TextLayoutInner {
            text_align: text_style.text_align,
            lines,
            len,
            line_height,
            wrap_width,
            truncate_width,
            size: Some(size),
            bounds: None,
        });
    }

    size
}
