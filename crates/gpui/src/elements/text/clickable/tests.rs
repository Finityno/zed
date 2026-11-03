//! Where an interactive text shows the hand, and what the pointer moving over
//! it renders again.

use super::line_stretches;
use crate::{
    AnyWindowHandle, App, AppContext as _, Bounds, Context, CursorStyle, FontId, GlyphId,
    InteractiveElement as _, InteractiveText, IntoElement, Keystroke, LineLayout, Modifiers,
    MouseButton, MouseDownEvent, MouseMoveEvent, MouseUpEvent, ParentElement as _, Pixels,
    PlatformInput, Point, Render, ScrollDelta, ScrollEvent, ScrollHandle, ShapedGlyph,
    ShapedRun, SharedString, StatefulInteractiveElement as _, Styled as _, StyledText, TestAppContext,
    TextAlign, TextLayout, TouchPhase, Window, WindowHandle, WrapBoundary, WrappedLineLayout, div,
    point, px,
};
use std::{
    cell::{Cell, RefCell},
    ops::Range,
    rc::Rc,
    sync::Arc,
};

/// How wide a glyph is: the test text system advances each by 0.6 of the
/// font size, and the texts are set at 10px.
const GLYPH: f32 = 6.;
/// How tall a row of text is.
const ROW: f32 = 20.;
/// Where a [`TextPage`]'s column begins in the window.
const LEFT: f32 = 30.;
const TOP: f32 = 40.;
/// How wide a [`TextPage`]'s column is.
const COLUMN: f32 = 100.;

fn with_window<R>(
    cx: &mut TestAppContext,
    window: AnyWindowHandle,
    f: impl FnOnce(&mut Window, &mut App) -> R,
) -> R {
    cx.update_window(window, |_, window, cx| f(window, cx)).unwrap()
}

fn draw(cx: &mut TestAppContext, window: AnyWindowHandle) {
    with_window(cx, window, |window, cx| window.draw(cx).clear(cx));
}

/// Moves the pointer to `position`, drawing no frame of its own.
fn move_to(cx: &mut TestAppContext, window: AnyWindowHandle, position: Point<Pixels>) {
    with_window(cx, window, |window, cx| {
        window.dispatch_event(
            PlatformInput::MouseMove(MouseMoveEvent {
                position,
                pressed_button: None,
                modifiers: Modifiers::default(),
            }),
            cx,
        );
    });
}

/// Whether the platform shows the hand.
fn hand(cx: &TestAppContext) -> bool {
    cx.test_platform().active_cursor_style() == CursorStyle::PointingHand
}

/// Makes `window` the active, hovered window, which the window sets the
/// platform's cursor for.
fn hover_window(cx: &mut TestAppContext, window: AnyWindowHandle) {
    let test_window = cx.test_window(window);
    test_window.simulate_active_status_change(true);
    test_window.simulate_hover_status_change(true);
}

/// The middle of a [`TextPage`]'s glyph `column` of row `row`, for left
/// aligned text.
fn glyph(column: usize, row: usize) -> Point<Pixels> {
    point(
        px(LEFT + (column as f32 + 0.5) * GLYPH),
        px(TOP + (row as f32 + 0.5) * ROW),
    )
}

/// Somewhere away from a [`TextPage`]'s text.
fn away(step: usize) -> Point<Pixels> {
    point(px(300. + 10. * step as f32), px(300.))
}

/// Whether the hand belongs at `position`: within the text's bounds, where
/// the index [`TextLayout::index_for_position`] finds, which the clicks use,
/// is in one of `ranges`.
fn clickable_at(layout: &TextLayout, ranges: &[Range<usize>], position: Point<Pixels>) -> bool {
    layout.bounds().contains(&position)
        && layout
            .index_for_position(position)
            .is_ok_and(|ix| ranges.iter().any(|range| range.contains(&ix)))
}

/// A view showing one interactive text in a column 100px wide, with rows
/// 20px tall, counting its renders and the ranges clicked.
struct TextPage {
    text: &'static str,
    ranges: Vec<Range<usize>>,
    wrap: bool,
    align: TextAlign,
    hover_listener: bool,
    renders: Rc<Cell<usize>>,
    clicks: Rc<RefCell<Vec<usize>>>,
    layout: Rc<RefCell<TextLayout>>,
}

impl Render for TextPage {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        self.renders.set(self.renders.get() + 1);
        let styled = StyledText::new(self.text);
        *self.layout.borrow_mut() = styled.layout().clone();
        let clicks = self.clicks.clone();
        let mut text = InteractiveText::new("text", styled).on_click(
            self.ranges.clone(),
            move |range_ix, _, _| clicks.borrow_mut().push(range_ix),
        );
        if self.hover_listener {
            text = text.on_hover(|_, _, _, _| {});
        }
        let column = div()
            .w(px(COLUMN))
            .text_size(px(10.))
            .line_height(px(ROW))
            .text_align(self.align)
            .child(text);
        let column = if self.wrap {
            column
        } else {
            column.whitespace_nowrap()
        };
        div().size_full().pl(px(LEFT)).pt(px(TOP)).child(column)
    }
}

/// A window showing a [`TextPage`], drawn and hovered, and what the test
/// reads of it.
struct Shown {
    window: WindowHandle<TextPage>,
    renders: Rc<Cell<usize>>,
    clicks: Rc<RefCell<Vec<usize>>>,
    layout: Rc<RefCell<TextLayout>>,
}

impl Shown {
    fn any(&self) -> AnyWindowHandle {
        self.window.into()
    }

    /// The text's layout, as last laid out and prepainted.
    fn layout(&self) -> TextLayout {
        self.layout.borrow().clone()
    }
}

struct Page {
    text: &'static str,
    ranges: Vec<Range<usize>>,
    wrap: bool,
    align: TextAlign,
    hover_listener: bool,
}

impl Page {
    fn new(text: &'static str, ranges: Vec<Range<usize>>) -> Self {
        Page {
            text,
            ranges,
            wrap: false,
            align: TextAlign::Left,
            hover_listener: false,
        }
    }
}

/// Shows `page` with view retention on, as the app runs, so that a view is
/// rendered again only for what it read.
fn show(cx: &mut TestAppContext, page: Page) -> Shown {
    cx.update(|cx| cx.set_view_retention(true));
    let renders = Rc::new(Cell::new(0));
    let clicks = Rc::new(RefCell::new(Vec::new()));
    let layout = Rc::new(RefCell::new(TextLayout::default()));
    let window = cx.add_window({
        let renders = renders.clone();
        let clicks = clicks.clone();
        let layout = layout.clone();
        move |_, _| TextPage {
            text: page.text,
            ranges: page.ranges,
            wrap: page.wrap,
            align: page.align,
            hover_listener: page.hover_listener,
            renders,
            clicks,
            layout,
        }
    });
    let shown = Shown {
        window,
        renders,
        clicks,
        layout,
    };
    hover_window(cx, shown.any());
    draw(cx, shown.any());
    shown
}

/// On a grid of points over a text and around it, the hand shows where the
/// index under the pointer is clickable, within the text's bounds, for text
/// aligned left, centered and right. One text wraps in its column, with a
/// range wrapped across two rows and a second line; the other does not wrap
/// and runs past its column, with ranges out of order, overlapping, and past
/// the column. The grid keeps off glyph and row edges, and so off the y where
/// two lines meet; the next test takes that.
#[test]
fn the_hand_shows_where_the_index_under_the_pointer_is_clickable() {
    for align in [TextAlign::Left, TextAlign::Center, TextAlign::Right] {
        for (text, ranges, wrap) in [
            (
                "click this linked text that wraps\nand a second line with a link",
                vec![6..27, 59..63],
                true,
            ),
            (
                "a not wrapping text running past its column",
                vec![20..30, 2..5, 4..9, 14..18],
                false,
            ),
        ] {
            let mut cx = TestAppContext::single();
            let shown = show(
                &mut cx,
                Page {
                    wrap,
                    align,
                    ..Page::new(text, ranges.clone())
                },
            );
            let layout = shown.layout();
            let bounds = layout.bounds();
            let rows = (bounds.size.height / px(ROW)).round() as usize;
            if wrap {
                assert_eq!(layout.line_layouts().len(), 2, "{text:?}: two lines");
                assert!(rows >= 4, "{text:?}: each line wraps");
                assert_ne!(
                    layout.position_for_index(6).map(|position| position.y),
                    layout.position_for_index(26).map(|position| position.y),
                    "{text:?}: the first range wraps across rows"
                );
            } else {
                assert_eq!(bounds.size.width, px(COLUMN), "{text:?}: its column");
                assert!(
                    layout.line_layouts()[0].width() > px(COLUMN),
                    "{text:?}: runs past its column"
                );
            }

            let mut ranges_with_the_hand = vec![false; ranges.len()];
            let mut without_the_hand_within = 0;
            for row in -1..=rows as i32 {
                for quarter in [0.25, 0.75] {
                    // Steps of a third of a glyph, never on a glyph edge
                    // whichever whole pixel a row's alignment starts it at.
                    for third in -9..135 {
                        let position = point(
                            px(LEFT + (third as f32 + 0.5) * GLYPH / 3.),
                            px(TOP + (row as f32 + quarter) * ROW),
                        );
                        move_to(&mut cx, shown.any(), position);
                        let expected = clickable_at(&layout, &ranges, position);
                        assert_eq!(hand(&cx), expected, "{align:?} {text:?} at {position:?}");
                        if expected && let Ok(ix) = layout.index_for_position(position) {
                            for (range, seen) in ranges.iter().zip(&mut ranges_with_the_hand) {
                                *seen |= range.contains(&ix);
                            }
                        } else if bounds.contains(&position) {
                            without_the_hand_within += 1;
                        }
                    }
                }
            }
            assert!(without_the_hand_within > 0, "{align:?} {text:?}: no hand somewhere");
            if wrap {
                assert_eq!(ranges_with_the_hand, vec![true, true], "{align:?} {text:?}");
            } else if align == TextAlign::Left {
                // The first range is past the column.
                assert_eq!(
                    ranges_with_the_hand,
                    vec![false, true, true, true],
                    "{align:?} {text:?}"
                );
            }
        }
    }
}

/// Where one line ends and the next begins, `index_for_position` takes the y
/// for the line above, below its last row, and finds no index; the line
/// below's hitbox holds that y, and the hand shows. Just above and below, it
/// finds an index in a range, and the hand shows.
#[test]
fn the_line_below_takes_the_y_where_two_lines_meet() {
    let mut cx = TestAppContext::single();
    let shown = show(
        &mut cx,
        Page {
            wrap: true,
            ..Page::new("link\nlink", vec![0..4, 5..9])
        },
    );
    let layout = shown.layout();
    let x = px(LEFT + 1.5 * GLYPH);
    for (y, index) in [
        (TOP + ROW - 2., Ok(1)),
        (TOP + ROW, Err(0)),
        (TOP + ROW + 2., Ok(6)),
    ] {
        let position = point(x, px(y));
        assert_eq!(layout.index_for_position(position), index, "at {position:?}");
        move_to(&mut cx, shown.any(), position);
        assert!(hand(&cx), "the hand at {position:?}");
    }
}

/// A line laid out by hand: `glyphs`, as their index and x in the runs'
/// order, split into runs of `run_lengths`, and wrapped before the glyphs
/// `wraps` (in the same order).
fn line(
    glyphs: &[(usize, f32)],
    run_lengths: &[usize],
    wraps: &[usize],
    width: f32,
    len: usize,
) -> WrappedLineLayout {
    let mut glyphs = glyphs.iter();
    let runs = run_lengths
        .iter()
        .map(|&length| ShapedRun {
            font_id: FontId(0),
            glyphs: glyphs
                .by_ref()
                .take(length)
                .map(|&(index, x)| ShapedGlyph {
                    id: GlyphId(1),
                    position: point(px(x), px(0.)),
                    index,
                    is_emoji: false,
                })
                .collect(),
        })
        .collect();
    let wrap_boundaries = wraps
        .iter()
        .map(|&glyph| {
            let mut run_ix = 0;
            let mut glyph_ix = glyph;
            while glyph_ix >= run_lengths[run_ix] {
                glyph_ix -= run_lengths[run_ix];
                run_ix += 1;
            }
            WrapBoundary { run_ix, glyph_ix }
        })
        .collect();
    WrappedLineLayout {
        unwrapped_layout: Arc::new(LineLayout {
            font_size: px(10.),
            width: px(width),
            ascent: px(8.),
            descent: px(-2.),
            runs,
            len,
        }),
        wrap_boundaries,
        wrap_width: Some(px(width)),
    }
}

/// The stretches the hitboxes are inserted over are where
/// `WrappedLineLayout::index_for_position`, given each row's alignment
/// offset as `TextLayout::index_for_position` gives it, finds an index the
/// text takes, on a grid between glyph and row edges and on them, for lines
/// laid out by hand, aligned left, centered and right: one wrapped twice
/// across two runs, and ones the test text system does not make, running
/// right to left (its glyphs held from the right or from the left), in both
/// directions, with a zero-width wrap boundary past the line's width, and
/// with characters of several bytes.
#[test]
fn the_stretches_are_where_the_index_under_the_pointer_is_clickable() {
    let left_to_right: Vec<_> = (0..12).map(|ix| (ix, ix as f32 * GLYPH)).collect();
    let right_to_left: Vec<_> = (0..6).map(|ix| (ix, (5 - ix) as f32 * GLYPH)).collect();
    let right_to_left_from_the_left: Vec<_> = right_to_left.iter().rev().copied().collect();
    let lines = [
        (
            "wrapped twice across two runs",
            line(&left_to_right, &[5, 7], &[4, 9], 72., 12),
        ),
        ("right to left", line(&right_to_left, &[6], &[], 36., 6)),
        (
            "right to left, wrapped",
            line(&right_to_left, &[6], &[3], 36., 6),
        ),
        (
            "right to left, held from the left, wrapped",
            line(&right_to_left_from_the_left, &[6], &[3], 36., 6),
        ),
        (
            "both directions",
            line(
                &[(0, 0.), (1, 6.), (2, 12.), (3, 30.), (4, 24.), (5, 18.)],
                &[3, 3],
                &[],
                36.,
                6,
            ),
        ),
        (
            "a zero-width wrap boundary past the line's width",
            line(
                &[(0, 0.), (1, 6.), (2, 12.), (3, 18.), (4, 24.), (5, 31.)],
                &[6],
                &[5],
                30.,
                6,
            ),
        ),
        (
            "characters of several bytes",
            line(
                &[(0, 0.), (1, 6.), (3, 12.), (4, 18.), (7, 24.)],
                &[5],
                &[2],
                30.,
                8,
            ),
        ),
    ];
    let clickables: [(&str, fn(usize) -> bool); 3] = [
        ("every index", |_| true),
        ("some indices", |ix| [1, 3, 4, 9].contains(&ix)),
        ("even indices", |ix| ix.is_multiple_of(2)),
    ];
    let width = px(80.);
    for align in [TextAlign::Left, TextAlign::Center, TextAlign::Right] {
        for (name, line) in &lines {
            let rows = line.wrap_boundaries.len() + 1;
            let (mut clickable_points, mut other_points) = (0, 0);
            for (clickable_name, clickable) in clickables {
                let mut stretches: Vec<Bounds<Pixels>> = Vec::new();
                line_stretches(line, px(ROW), align, width, clickable, |stretch| {
                    stretches.push(stretch)
                });
                for pair in stretches.windows(2) {
                    assert!(
                        pair[0].origin.y != pair[1].origin.y || pair[0].right() < pair[1].left(),
                        "{align:?} {name}, {clickable_name}: stretches that meet are merged: {stretches:?}"
                    );
                }
                // From the line's top down: `TextLayout::index_for_position`
                // finds no index above the text before it hands a line a y.
                for y in (0..=(rows as i32 + 1) * 8 + 2).map(|step| step as f32 * 2.5) {
                    let row = (y / ROW) as usize;
                    let offset = line.alignment_offset(row, align, width);
                    for x in (-12..=(width.0 as i32 + 6) * 2).map(|step| step as f32 * 0.5) {
                        let position = point(px(x), px(y));
                        let expected = line
                            .index_for_position(point(px(x) - offset, px(y)), px(ROW))
                            .is_ok_and(clickable);
                        let actual = stretches.iter().any(|stretch| stretch.contains(&position));
                        assert_eq!(
                            actual, expected,
                            "{align:?} {name}, {clickable_name}: at {position:?}, in {stretches:?}"
                        );
                        if expected {
                            clickable_points += 1;
                        } else {
                            other_points += 1;
                        }
                    }
                }
            }
            assert!(clickable_points > 0 && other_points > 0, "{align:?} {name}");
        }
    }
}

/// The pointer moving into the range and out of it sets the platform's
/// cursor to the hand and back to the arrow, as the window picks the cursor
/// again when its hit test changes. No frame is drawn, and the view is not
/// rendered again.
#[test]
fn the_hand_follows_the_pointer_without_a_frame() {
    let mut cx = TestAppContext::single();
    let shown = show(&mut cx, Page::new("a link here", vec![2..6]));
    move_to(&mut cx, shown.any(), away(0));
    draw(&mut cx, shown.any());
    with_window(&mut cx, shown.any(), |window, _| {
        window.reset_frame_work_stats(false)
    });
    let renders = shown.renders.get();
    for (column, expected) in [
        (0, false),
        (2, true),
        (3, true),
        (5, true),
        (7, false),
        (8, false),
    ] {
        move_to(&mut cx, shown.any(), glyph(column, 0));
        assert_eq!(hand(&cx), expected, "the hand over column {column}");
    }
    cx.run_until_parked();
    let frames = with_window(&mut cx, shown.any(), |window, _| {
        window.frame_work_stats().frames
    });
    assert_eq!(frames, 0, "no frame is drawn");
    assert_eq!(shown.renders.get(), renders, "the view is not rendered again");
}

/// The pointer moving away from the text and over it, from character to
/// character, into and out of the range, does not render the view again,
/// frames drawn or not; the hand shows over the range.
#[test]
fn the_pointer_moving_does_not_render_the_view_again() {
    let mut cx = TestAppContext::single();
    let shown = show(&mut cx, Page::new("a link here", vec![2..6]));
    move_to(&mut cx, shown.any(), away(0));
    draw(&mut cx, shown.any());
    let renders = shown.renders.get();
    for step in 1..4 {
        move_to(&mut cx, shown.any(), away(step));
        draw(&mut cx, shown.any());
        assert_eq!(shown.renders.get(), renders, "the view rendered again away");
        assert!(!hand(&cx), "no hand away");
    }
    for column in 0..11 {
        move_to(&mut cx, shown.any(), glyph(column, 0));
        draw(&mut cx, shown.any());
        assert_eq!(
            shown.renders.get(),
            renders,
            "the view rendered again over column {column}"
        );
        assert_eq!(
            hand(&cx),
            (2..6).contains(&column),
            "the hand over column {column}"
        );
    }
}

/// After the pointer moved over the text without the view rendering again,
/// a press and release over a range calls the click listener with that
/// range.
#[test]
fn a_click_over_a_range_calls_the_click_listener() {
    let mut cx = TestAppContext::single();
    let shown = show(
        &mut cx,
        Page::new("a link and another", vec![2..6, 11..18]),
    );
    let renders = shown.renders.get();
    for column in [0, 3, 8, 12] {
        move_to(&mut cx, shown.any(), glyph(column, 0));
        draw(&mut cx, shown.any());
    }
    assert_eq!(shown.renders.get(), renders, "the view rendered again");
    let position = glyph(12, 0);
    with_window(&mut cx, shown.any(), |window, cx| {
        window.dispatch_event(
            PlatformInput::MouseDown(MouseDownEvent {
                button: MouseButton::Left,
                position,
                modifiers: Modifiers::default(),
                click_count: 1,
                first_mouse: false,
            }),
            cx,
        );
    });
    draw(&mut cx, shown.any());
    with_window(&mut cx, shown.any(), |window, cx| {
        window.dispatch_event(
            PlatformInput::MouseUp(MouseUpEvent {
                button: MouseButton::Left,
                position,
                modifiers: Modifiers::default(),
                click_count: 1,
            }),
            cx,
        );
    });
    assert_eq!(*shown.clicks.borrow(), vec![1], "the second range clicked");
}

/// With a hover listener, a change of the index under the pointer still
/// renders the view again: what the listener did with it may change what
/// the view renders.
#[test]
fn a_hover_listener_has_the_view_rendered_again_as_the_index_changes() {
    let mut cx = TestAppContext::single();
    let shown = show(
        &mut cx,
        Page {
            hover_listener: true,
            ..Page::new("a link here", vec![2..6])
        },
    );
    move_to(&mut cx, shown.any(), glyph(0, 0));
    draw(&mut cx, shown.any());
    for column in [1, 2, 3, 7] {
        let renders = shown.renders.get();
        move_to(&mut cx, shown.any(), glyph(column, 0));
        draw(&mut cx, shown.any());
        assert_eq!(
            shown.renders.get(),
            renders + 1,
            "the view rendered again for column {column}"
        );
    }
}

/// After a key press, which hides hover, the hand still shows over the range
/// under a still pointer, in a frame painting the text again.
#[test]
fn the_hand_stays_after_keyboard_input() {
    let mut cx = TestAppContext::single();
    let shown = show(&mut cx, Page::new("a link here", vec![2..6]));
    move_to(&mut cx, shown.any(), glyph(3, 0));
    assert!(hand(&cx), "the hand over the range");
    with_window(&mut cx, shown.any(), |window, cx| {
        if let Ok(keystroke) = Keystroke::parse("x") {
            window.dispatch_keystroke(keystroke, cx);
        }
    });
    let renders = shown.renders.get();
    shown.window.update(&mut cx, |_, _, cx| cx.notify()).unwrap();
    draw(&mut cx, shown.any());
    assert!(
        with_window(&mut cx, shown.any(), |window, _| window.last_input_was_keyboard()),
        "the key press hides hover"
    );
    assert!(shown.renders.get() > renders, "the text painted again");
    assert!(hand(&cx), "the hand after the key press");
}

/// How many rows a [`RowsPage`] holds.
const ROWS: usize = 60;
/// How wide and tall a [`RowsPage`]'s viewport is.
const VIEWPORT_WIDTH: f32 = 200.;
const VIEWPORT_HEIGHT: f32 = 100.;
/// Where the pointer stays while the wheel scrolls a [`RowsPage`].
const POINTER: (f32, f32) = (20., 30.);

/// A row of interactive text, 20px tall, clickable over all its text when it
/// is an even one.
fn text_row(ix: usize) -> impl IntoElement {
    let label = SharedString::from(format!("row {ix} of the list"));
    let len = label.len();
    let text = InteractiveText::new(("text", ix), StyledText::new(label));
    let text = if ix.is_multiple_of(2) {
        text.on_click(vec![0..len], |_, _, _| {})
    } else {
        text
    };
    div().w(px(VIEWPORT_WIDTH)).h(px(ROW)).child(text)
}

/// [`ROWS`] rows of interactive text in a scroll container.
struct RowsPage {
    scroll: ScrollHandle,
}

impl Render for RowsPage {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        div().size_full().text_size(px(10.)).line_height(px(ROW)).child(
            div()
                .id("scroller")
                .overflow_y_scroll()
                .track_scroll(&self.scroll)
                .w(px(VIEWPORT_WIDTH))
                .h(px(VIEWPORT_HEIGHT))
                .children((0..ROWS).map(text_row)),
        )
    }
}

/// The wheel scrolls rows, every other one clickable, under a still pointer,
/// with view retention on and off: after each scroll, the hand shows over a
/// clickable row and not over the others.
#[test]
fn the_hand_follows_rows_the_wheel_scrolls_under_a_still_pointer() {
    for retention in [false, true] {
        let mut cx = TestAppContext::single();
        cx.update(|cx| cx.set_view_retention(retention));
        let window = cx.add_window(|_, _| RowsPage {
            scroll: ScrollHandle::new(),
        });
        let any: AnyWindowHandle = window.into();
        hover_window(&mut cx, any);
        draw(&mut cx, any);
        let pointer = point(px(POINTER.0), px(POINTER.1));
        move_to(&mut cx, any, pointer);
        draw(&mut cx, any);
        let mut rows_seen = Vec::new();
        for frame in 0..6 {
            with_window(&mut cx, any, |window, cx| {
                window.dispatch_event(
                    PlatformInput::Scroll(ScrollEvent {
                        position: pointer,
                        delta: ScrollDelta::Pixels(point(px(0.), px(-9.))),
                        modifiers: Modifiers::default(),
                        touch_phase: TouchPhase::Moved,
                        ..Default::default()
                    }),
                    cx,
                );
            });
            draw(&mut cx, any);
            let offset = window.update(&mut cx, |page, _, _| page.scroll.offset().y).unwrap();
            let row = ((pointer.y - offset) / px(ROW)).floor() as usize;
            rows_seen.push(row);
            assert_eq!(
                hand(&cx),
                row.is_multiple_of(2),
                "retention {retention}, frame {frame}: row {row} under the pointer"
            );
        }
        assert!(
            rows_seen.iter().any(|row| row.is_multiple_of(2))
                && rows_seen.iter().any(|row| !row.is_multiple_of(2)),
            "retention {retention}: rows of both kinds under the pointer: {rows_seen:?}"
        );
    }
}
