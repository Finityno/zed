//! A list's rows drawn again from the last frame, with row replay on, must
//! draw what building them draws.
//!
//! Every test drives two windows of one app through the same changes: the
//! first draws what it can again from the last frame, the second is
//! refreshed before every frame, so that everything in it is built. Their
//! frames must show the same primitives in the same places and leave the
//! same hitboxes.

use super::{describe_frame, first_difference};
use crate::{
    App, Context, Entity, FrameWorkStats, Hsla, IntoElement, ListAlignment, ListState, Modifiers,
    Pixels, Point, Render, SharedString, TestAppContext, Window, WindowHandle, div, hsla, list,
    point, prelude::*, px,
};
use rand::{Rng as _, SeedableRng as _, rngs::StdRng};
use std::{
    cell::{Cell, RefCell},
    rc::Rc,
};

const PALETTE: [Hsla; 4] = [
    hsla(0.0, 0.0, 0.1, 1.0),
    hsla(0.6, 0.7, 0.5, 1.0),
    hsla(0.3, 0.6, 0.4, 1.0),
    hsla(0.0, 0.8, 0.6, 1.0),
];

const WORDS: [&str; 6] = [
    "a",
    "row",
    "a longer label that wraps past the edge of the row",
    "42",
    "sit amet dolor",
    "x",
];

const ROWS: usize = 40;
const LIST_TOP: f32 = 20.;
const LIST_HEIGHT: f32 = 230.;

/// What one row shows, read by the row closure.
struct Label(usize);

/// A view drawn in every third row, counting its renders.
struct Badge {
    ix: usize,
    renders: Rc<Cell<usize>>,
}

impl Render for Badge {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        self.renders.set(self.renders.get() + 1);
        div()
            .id("badge")
            .w(px(30.))
            .h(px(12.))
            .bg(PALETTE[self.ix % PALETTE.len()])
            .hover(|style| style.bg(PALETTE[0]))
    }
}

/// A view drawing a list of rows of plain elements, every third with a
/// [`Badge`] view in it, beside a line telling whether the list is at its
/// end.
struct Host {
    list_state: ListState,
    labels: Vec<Entity<Label>>,
    badges: Vec<Entity<Badge>>,
    /// Read by `render` and handed to the row closure: rows show it, so a
    /// change to it changes every row.
    accent: usize,
    /// The rows the row closure was called for.
    built: Rc<RefCell<Vec<usize>>>,
    /// Asks the list whether it is at its end as it renders.
    asks_end: bool,
    renders: Rc<Cell<usize>>,
}

impl Render for Host {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        self.renders.set(self.renders.get() + 1);
        let at_end = self
            .asks_end
            .then(|| self.list_state.is_scrolled_to_end())
            .flatten();
        let labels = self.labels.clone();
        let badges = self.badges.clone();
        let accent = self.accent;
        let built = self.built.clone();
        div()
            .size_full()
            .child(
                div()
                    .h(px(LIST_TOP))
                    .child(SharedString::from(format!("at end: {at_end:?}"))),
            )
            .child(
                list(self.list_state.clone(), move |ix, _, cx: &mut App| {
                    built.borrow_mut().push(ix);
                    let label = labels.get(ix).map_or(0, |label| label.read(cx).0);
                    div()
                        .id(("row", ix))
                        .flex()
                        .flex_col()
                        .w_full()
                        // Tall enough for the test text system's glyphs, which hang
                        // far below their lines: what overflows a row and is culled
                        // out of view is not drawn when the row moves into view (as
                        // for any view drawn moved).
                        .min_h(px(40. + (ix % 4) as f32 * 9.))
                        .line_height(px(36.))
                        .bg(PALETTE[(ix + accent) % PALETTE.len()])
                        .hover(|style| style.bg(PALETTE[3]))
                        .child(SharedString::from(format!(
                            "row {ix}: {}",
                            WORDS[label % WORDS.len()]
                        )))
                        .when_some(
                            badges.get(ix).filter(|_| ix % 3 == 0).cloned(),
                            |row, badge| row.child(badge),
                        )
                        .when(ix % 5 == 1, |row| row.child(sticky_marker()))
                        .into_any_element()
                })
                .h(px(LIST_HEIGHT))
                .w(px(180.)),
            )
    }
}

/// A marker kept at the top of what shows of its row, as a sticky header
/// is: it reads the content mask as it prepaints, so where it is drawn
/// depends on how much of the row shows.
fn sticky_marker() -> impl IntoElement {
    crate::canvas(
        |bounds, window, _| bounds.top().max(window.content_mask().bounds.top()),
        |bounds, top, window, _| {
            window.paint_quad(crate::fill(
                crate::Bounds {
                    origin: point(bounds.left(), top),
                    size: crate::size(px(8.), px(4.)),
                },
                PALETTE[2],
            ));
        },
    )
    .absolute()
    .top_0()
    .left_0()
    .size_full()
}

struct Fixture {
    windows: [WindowHandle<Host>; 2],
    list_states: [ListState; 2],
    built: [Rc<RefCell<Vec<usize>>>; 2],
    badge_renders: [Rc<Cell<usize>>; 2],
    host_renders: [Rc<Cell<usize>>; 2],
}

fn fixture(cx: &mut TestAppContext, asks_end: bool) -> Fixture {
    cx.update(|cx| {
        cx.set_view_retention(true);
        cx.set_list_row_replay(true);
    });
    let mut list_states = Vec::new();
    let mut built = Vec::new();
    let mut badge_renders = Vec::new();
    let mut host_renders = Vec::new();
    let windows = [(); 2].map(|_| {
        let list_state = ListState::new(ROWS, ListAlignment::Top, px(0.)).measure_all();
        let rows_built = Rc::new(RefCell::new(Vec::new()));
        let badges_rendered = Rc::new(Cell::new(0));
        let hosts_rendered = Rc::new(Cell::new(0));
        list_states.push(list_state.clone());
        built.push(rows_built.clone());
        badge_renders.push(badges_rendered.clone());
        host_renders.push(hosts_rendered.clone());
        cx.add_window(move |_, cx| Host {
            list_state,
            labels: (0..ROWS).map(|ix| cx.new(|_| Label(ix))).collect(),
            badges: (0..ROWS)
                .map(|ix| {
                    let renders = badges_rendered.clone();
                    cx.new(|_| Badge { ix, renders })
                })
                .collect(),
            accent: 0,
            built: rows_built,
            asks_end,
            renders: hosts_rendered,
        })
    });
    Fixture {
        windows,
        list_states: list_states.try_into().ok().unwrap(),
        built: built.try_into().ok().unwrap(),
        badge_renders: badge_renders.try_into().ok().unwrap(),
        host_renders: host_renders.try_into().ok().unwrap(),
    }
}

impl Fixture {
    /// Draws both windows, the second from scratch, returning what each shows
    /// and the first's work since the last call: the test platform draws a
    /// window as soon as input dirties it, and that frame's work counts too.
    fn draw(&self, cx: &mut TestAppContext) -> ([Vec<String>; 2], FrameWorkStats) {
        let (retained, work) = cx
            .update_window(self.windows[0].into(), |_, window, cx| {
                window.draw(cx).clear(cx);
                let work = window.frame_work_stats();
                window.reset_frame_work_stats(false);
                (describe_frame(window), work)
            })
            .unwrap();
        let from_scratch = cx
            .update_window(self.windows[1].into(), |_, window, cx| {
                window.refresh();
                window.draw(cx).clear(cx);
                describe_frame(window)
            })
            .unwrap();
        ([retained, from_scratch], work)
    }

    /// Draws both windows and checks they show the same, returning the
    /// first's work and the rows its row closure was called for.
    fn draw_matching(&self, cx: &mut TestAppContext, what: &str) -> (FrameWorkStats, Vec<usize>) {
        let ([retained, from_scratch], work) = self.draw(cx);
        if let Some(difference) = first_difference(&retained, &from_scratch) {
            let only = |a: &[String], b: &[String]| -> Vec<String> {
                a.iter().filter(|line| !b.contains(line)).cloned().collect()
            };
            panic!(
                "{what}: {difference}\nonly drawn again: {:#?}\nonly from scratch: {:#?}",
                only(&retained, &from_scratch),
                only(&from_scratch, &retained)
            );
        }
        let built = std::mem::take(&mut *self.built[0].borrow_mut());
        (work, built)
    }

    fn wheel(&self, cx: &mut TestAppContext, delta: f32) {
        for (window, state) in self.windows.into_iter().zip(&self.list_states) {
            let position = state.viewport_bounds().center();
            cx.update_window(window.into(), |_, window, cx| {
                window.dispatch_event(
                    crate::PlatformInput::Scroll(crate::ScrollEvent {
                        position,
                        delta: crate::ScrollDelta::Pixels(point(px(0.), px(delta))),
                        modifiers: Modifiers::default(),
                        touch_phase: crate::TouchPhase::Moved,
                        is_momentum: false,
                        source: crate::ScrollSource::Pointer,
                        wheel_notches: None,
                    }),
                    cx,
                );
            })
            .unwrap();
        }
    }

    fn move_mouse(&self, cx: &mut TestAppContext, position: Point<Pixels>) {
        for window in self.windows {
            cx.update_window(window.into(), |_, window, cx| {
                window.dispatch_event(
                    crate::PlatformInput::MouseMove(crate::MouseMoveEvent {
                        position,
                        pressed_button: None,
                        modifiers: Modifiers::default(),
                    }),
                    cx,
                );
            })
            .unwrap();
        }
    }

    fn update_hosts(&self, cx: &mut TestAppContext, update: impl Fn(&mut Host, &mut Context<Host>)) {
        for window in self.windows {
            window.update(cx, |host, _, cx| update(host, cx)).unwrap();
        }
    }

    fn set_label(&self, cx: &mut TestAppContext, ix: usize, label: usize) {
        for window in self.windows {
            let entity = window.read_with(cx, |host, _| host.labels[ix].clone()).unwrap();
            entity.update(cx, |entity, cx| {
                entity.0 = label;
                cx.notify();
            });
        }
    }

    fn notify_badge(&self, cx: &mut TestAppContext, ix: usize) {
        for window in self.windows {
            let badge = window.read_with(cx, |host, _| host.badges[ix].clone()).unwrap();
            badge.update(cx, |_, cx| cx.notify());
        }
    }
}

/// A wheel scroll draws the rows nothing changed for again, moved, without
/// calling the row closure for them: only rows coming into view, and rows
/// cut by the list's edge, are built.
#[test]
fn a_scrolled_list_draws_its_rows_again_moved() {
    let mut cx = super::super::layout_retention_tests::text_system_context_with_ink_in_font_bounds(0);
    let fixture = fixture(&mut cx, false);
    fixture.move_mouse(&mut cx, point(px(500.), px(500.)));
    fixture.draw_matching(&mut cx, "first frame");
    fixture.draw_matching(&mut cx, "second frame");
    let mut replayed = 0;
    for (step, delta) in [-12., -12., -50., 13., -7., 40., -10.].into_iter().enumerate() {
        fixture.wheel(&mut cx, delta);
        let (work, built) = fixture.draw_matching(&mut cx, &format!("scroll {step}"));
        replayed += work.list_rows_replayed;
        assert!(
            built.len() as u64 <= work.list_rows_built,
            "scroll {step} called the row closure only for rows built: {built:?} {work:?}"
        );
    }
    assert!(replayed >= 6, "{replayed}");
}

/// With row replay off, rows are built on every scroll, as without it.
#[test]
fn with_row_replay_off_every_row_is_built() {
    let mut cx = super::super::layout_retention_tests::text_system_context_with_ink_in_font_bounds(0);
    let fixture = fixture(&mut cx, false);
    cx.update(|cx| cx.set_list_row_replay(false));
    fixture.draw_matching(&mut cx, "first frame");
    fixture.draw_matching(&mut cx, "second frame");
    fixture.wheel(&mut cx, -40.);
    let (work, built) = fixture.draw_matching(&mut cx, "scrolled");
    assert_eq!(work.list_rows_replayed, 0, "{work:?}");
    assert_eq!(work.list_rows_built, 0, "{work:?}");
    assert!(!built.is_empty());
}

/// A change to what one row read builds that row, and draws the others
/// again; a change to what the view drawing the list read itself builds
/// every row, since it hands the list another row closure.
#[test]
fn a_change_builds_the_rows_that_read_it() {
    let mut cx = super::super::layout_retention_tests::text_system_context_with_ink_in_font_bounds(0);
    let fixture = fixture(&mut cx, false);
    fixture.move_mouse(&mut cx, point(px(500.), px(500.)));
    fixture.draw_matching(&mut cx, "first frame");
    fixture.draw_matching(&mut cx, "second frame");
    fixture.wheel(&mut cx, -30.);
    fixture.draw_matching(&mut cx, "scrolled");

    fixture.set_label(&mut cx, 3, 2);
    fixture.wheel(&mut cx, -6.);
    let (work, built) = fixture.draw_matching(&mut cx, "label of row 3 changed");
    assert!(built.contains(&3), "{built:?}");
    assert!(work.list_rows_replayed > 0, "{work:?}");

    fixture.update_hosts(&mut cx, |host, cx| {
        host.accent += 1;
        cx.notify();
    });
    let (work, built) = fixture.draw_matching(&mut cx, "accent changed");
    assert_eq!(work.list_rows_replayed, 0, "{work:?}");
    assert!(!built.is_empty());

    fixture.wheel(&mut cx, -6.);
    let (work, _) = fixture.draw_matching(&mut cx, "scrolled after the accent changed");
    assert!(work.list_rows_replayed > 0, "{work:?}");
}

/// A view drawn in a row that is notified is built with its row; the other
/// rows are drawn again.
#[test]
fn a_notified_view_in_a_row_builds_its_row() {
    let mut cx = super::super::layout_retention_tests::text_system_context_with_ink_in_font_bounds(0);
    let fixture = fixture(&mut cx, false);
    fixture.move_mouse(&mut cx, point(px(500.), px(500.)));
    fixture.draw_matching(&mut cx, "first frame");
    fixture.draw_matching(&mut cx, "second frame");
    fixture.wheel(&mut cx, -20.);
    fixture.draw_matching(&mut cx, "scrolled");
    let badges_before = fixture.badge_renders[0].get();
    fixture.notify_badge(&mut cx, 3);
    fixture.wheel(&mut cx, -5.);
    let (work, built) = fixture.draw_matching(&mut cx, "badge 3 notified");
    assert!(built.contains(&3), "{built:?}");
    assert!(fixture.badge_renders[0].get() > badges_before, "the badge notified was built");
    assert!(work.list_rows_replayed > 0, "{work:?}");
}

/// A splice builds every row: rows are known by index.
#[test]
fn a_splice_builds_every_row() {
    let mut cx = super::super::layout_retention_tests::text_system_context_with_ink_in_font_bounds(0);
    let fixture = fixture(&mut cx, false);
    fixture.draw_matching(&mut cx, "first frame");
    fixture.draw_matching(&mut cx, "second frame");
    for state in &fixture.list_states {
        state.splice(0..1, 1);
    }
    fixture.update_hosts(&mut cx, |_, cx| cx.notify());
    let (work, _) = fixture.draw_matching(&mut cx, "spliced");
    assert_eq!(work.list_rows_replayed, 0, "{work:?}");
}

/// A scrollbar setting the list's offset scrolls it as the wheel does: the
/// view drawing it is built without being notified, and draws its rows
/// again.
#[test]
fn a_list_dragged_by_its_scrollbar_draws_its_rows_again() {
    let mut cx = super::super::layout_retention_tests::text_system_context_with_ink_in_font_bounds(0);
    let fixture = fixture(&mut cx, false);
    fixture.move_mouse(&mut cx, point(px(500.), px(500.)));
    fixture.draw_matching(&mut cx, "first frame");
    fixture.draw_matching(&mut cx, "second frame");
    let renders_before = fixture.host_renders[0].get();
    for offset in [40., 52., 90.] {
        for state in &fixture.list_states {
            state.scrollbar_drag_started();
            state.set_offset_from_scrollbar(point(px(0.), px(-offset)));
            state.scrollbar_drag_ended();
        }
        let (work, _) = fixture.draw_matching(&mut cx, &format!("dragged to {offset}"));
        assert!(work.list_rows_replayed > 0, "dragged to {offset}: {work:?}");
    }
    assert_eq!(
        fixture.host_renders[0].get(),
        renders_before + 3,
        "the view drawing the list was built once per drag step"
    );
}

/// A view asking whether the list is at its end depends on the answer, not
/// on the scroll position: it is built only once the answer changes, and the
/// rows are drawn again meanwhile.
#[test]
fn asking_whether_a_list_is_at_its_end_reads_only_the_answer() {
    let mut cx = super::super::layout_retention_tests::text_system_context_with_ink_in_font_bounds(0);
    let fixture = fixture(&mut cx, true);
    fixture.move_mouse(&mut cx, point(px(500.), px(500.)));
    fixture.draw_matching(&mut cx, "first frame");
    fixture.draw_matching(&mut cx, "second frame");
    fixture.wheel(&mut cx, -30.);
    let (work, _) = fixture.draw_matching(&mut cx, "scrolled");
    assert!(work.list_rows_replayed > 0, "{work:?}");
    // Far past the end: the answer flips, and the line reading it changes.
    fixture.wheel(&mut cx, -5000.);
    let (_, _) = fixture.draw_matching(&mut cx, "scrolled to the end");
    assert_eq!(fixture.list_states[0].is_scrolled_to_end(), Some(true));
    fixture.wheel(&mut cx, 20.);
    fixture.draw_matching(&mut cx, "scrolled back from the end");
    assert_eq!(fixture.list_states[0].is_scrolled_to_end(), Some(false));
}

/// Random histories of scrolls by whole and fractional pixels, pointer
/// moves, row and view changes, notifications and splices draw what drawing
/// from scratch draws, at scale factors 1, 1.25 and 2.
#[test]
fn random_histories_match_frames_drawn_from_scratch() {
    for seed in 0..12u64 {
        let mut cx = super::super::layout_retention_tests::text_system_context_with_ink_in_font_bounds(seed);
        let fixture = fixture(&mut cx, seed % 2 == 0);
        let scale = [1.0, 1.25, 2.0][seed as usize % 3];
        for window in fixture.windows {
            cx.test_window(window.into())
                .simulate_scale_factor_change(scale);
        }
        let mut rng = StdRng::seed_from_u64(seed);
        let mut replayed = 0;
        fixture.draw_matching(&mut cx, "first frame");
        for step in 0..60 {
            let change = rng.random_range(0..10);
            let what = match change {
                0..=4 => {
                    let delta = if rng.random_bool(0.7) {
                        rng.random_range(-60..60) as f32
                    } else {
                        rng.random_range(-600..600) as f32 / 10.
                    };
                    fixture.wheel(&mut cx, delta);
                    format!("wheel {delta}")
                }
                5 => {
                    let position = point(
                        px(rng.random_range(0..200) as f32),
                        px(rng.random_range(0..280) as f32),
                    );
                    fixture.move_mouse(&mut cx, position);
                    format!("mouse to {position:?}")
                }
                6 => {
                    let ix = rng.random_range(0..ROWS);
                    let label = rng.random_range(0..WORDS.len());
                    fixture.set_label(&mut cx, ix, label);
                    format!("label {ix} to {label}")
                }
                7 => {
                    let ix = rng.random_range(0..ROWS / 3) * 3;
                    fixture.notify_badge(&mut cx, ix);
                    format!("badge {ix} notified")
                }
                8 => {
                    fixture.update_hosts(&mut cx, |host, cx| {
                        host.accent += 1;
                        cx.notify();
                    });
                    "accent changed".to_string()
                }
                _ => {
                    let at = rng.random_range(0..ROWS - 1);
                    for state in &fixture.list_states {
                        state.splice(at..at + 1, 1);
                    }
                    fixture.update_hosts(&mut cx, |_, cx| cx.notify());
                    format!("row {at} spliced")
                }
            };
            let (work, _) =
                fixture.draw_matching(&mut cx, &format!("seed {seed} step {step}: {what}"));
            replayed += work.list_rows_replayed;
        }
        assert!(replayed > 0, "seed {seed} drew no row again");
    }
}
