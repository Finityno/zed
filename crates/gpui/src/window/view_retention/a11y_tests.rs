//! With accessibility active, views drawn again from the last frame must
//! leave the accessibility tree the frame drawn from scratch builds: the same
//! nodes in the same places, the same focus, and the same action listeners.
//! See [`crate::window::a11y::replay`].
//!
//! [`describe_frame`] describes the tree along with what a frame paints, so
//! every comparison here, and the random histories of `tests.rs` run with
//! accessibility active, compare trees too.

use super::{ViewRebuildReason, describe_frame, first_difference};
use crate::{
    Context, Entity, FocusHandle, FrameWorkStats, IntoElement, ListAlignment, ListState, Render,
    TestAppContext, Window, WindowHandle, div, list, prelude::*, px,
};
use accesskit::{Action, ActionRequest, Role};
use std::{cell::Cell, rc::Rc};

/// Two windows of the same root, the first drawing views again from the last
/// frame and the second drawn from scratch, both with accessibility active.
fn accessible_windows<V: 'static + Render>(
    cx: &mut TestAppContext,
    build: impl Fn(&mut Context<V>) -> V,
) -> [WindowHandle<V>; 2] {
    cx.update(|cx| cx.set_view_retention(true));
    [(); 2].map(|_| {
        let window = cx.add_window(|_, cx| build(cx));
        cx.update_window(window.into(), |_, window, cx| {
            window.force_a11y_active();
            window.draw(cx).clear(cx);
            assert!(window.a11y.is_active(), "accessibility is active");
        })
        .unwrap();
        window
    })
}

/// Draws both windows, the second from scratch and the first only if no
/// change drew it since it was last drawn here, returning both frames and
/// the first's work since then.
pub(super) fn draw_both<V: 'static>(
    cx: &mut TestAppContext,
    windows: [WindowHandle<V>; 2],
) -> (Vec<String>, Vec<String>, FrameWorkStats) {
    let (retained, work) = cx
        .update_window(windows[0].into(), |_, window, cx| {
            if window.frame_work_stats().frames == 0 {
                window.draw(cx).clear(cx);
            }
            let work = window.frame_work_stats();
            window.reset_frame_work_stats(false);
            (describe_frame(window), work)
        })
        .unwrap();
    let from_scratch = cx
        .update_window(windows[1].into(), |_, window, cx| {
            window.refresh();
            window.draw(cx).clear(cx);
            describe_frame(window)
        })
        .unwrap();
    (retained, from_scratch, work)
}

#[track_caller]
pub(super) fn assert_same(retained: &[String], from_scratch: &[String]) {
    if let Some(difference) = first_difference(retained, from_scratch) {
        let only_in = |these: &[String], those: &[String]| {
            these
                .iter()
                .filter(|line| !those.contains(line))
                .take(8)
                .map(|line| format!("    {line}"))
                .collect::<Vec<_>>()
                .join("\n")
        };
        panic!(
            "the frame drawing views again differs from the frame drawn from scratch at \
             {difference}\nonly drawing views again:\n{}\nonly from scratch:\n{}",
            only_in(retained, from_scratch),
            only_in(from_scratch, retained)
        );
    }
}

fn a11y_lines(frame: &[String]) -> usize {
    frame.iter().filter(|line| line.starts_with("a11y r")).count()
}

/// An item with a node, a button node listening for clicks, and a synthetic
/// node, optionally with bounds of its own.
struct Item {
    label: usize,
    placed: bool,
    clicks: Rc<Cell<usize>>,
}

impl Render for Item {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        let clicks = self.clicks.clone();
        let placed = self.placed;
        div()
            .id("item")
            .role(Role::ListItem)
            .aria_label(format!("item {}", self.label))
            .when(self.label % 4 == 1, |this| this.aria_active_descendant())
            .h(px(20.))
            .child(
                div()
                    .id("button")
                    .role(Role::Button)
                    .on_a11y_action(Action::Click, move |_, _, _| clicks.set(clicks.get() + 1))
                    .size(px(10.))
                    .a11y_synthetic_children(move |builder| {
                        let id = builder.synthetic_node_id(0);
                        let mut node = accesskit::Node::new(Role::Image);
                        if placed {
                            node.set_bounds(accesskit::Rect {
                                x0: 1.,
                                y0: 2.,
                                x1: 3.,
                                y1: 4.,
                            });
                        }
                        builder.push_child(id, node);
                    }),
            )
    }
}

/// A focusable list of items under a header whose height can change, which
/// moves the items.
struct ItemList {
    items: Vec<Entity<Item>>,
    focus: FocusHandle,
    title: usize,
    header: f32,
}

impl ItemList {
    fn new(placed: bool, clicks: &Rc<Cell<usize>>, cx: &mut Context<Self>) -> Self {
        Self {
            items: (0..4)
                .map(|label| {
                    let clicks = clicks.clone();
                    cx.new(|_| Item {
                        label,
                        placed,
                        clicks,
                    })
                })
                .collect(),
            focus: cx.focus_handle(),
            title: 0,
            header: 0.,
        }
    }
}

impl Render for ItemList {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        div()
            .size_full()
            .flex()
            .flex_col()
            .child(div().w(px(10.)).h(px(self.header)))
            .child(
                div()
                    .id("list")
                    .role(Role::List)
                    .track_focus(&self.focus)
                    .aria_label(format!("list {}", self.title))
                    .flex()
                    .flex_col()
                    .children(self.items.iter().cloned()),
            )
    }
}

fn update_lists(
    cx: &mut TestAppContext,
    windows: [WindowHandle<ItemList>; 2],
    update: impl Fn(&mut ItemList, &mut Context<ItemList>),
) {
    for window in windows {
        window.update(cx, |view, _, cx| update(view, cx)).unwrap();
    }
}

fn focus_lists(cx: &mut TestAppContext, windows: [WindowHandle<ItemList>; 2]) {
    for window in windows {
        cx.update_window(window.into(), |root, window, cx| {
            let focus = root.downcast::<ItemList>().unwrap().read(cx).focus.clone();
            window.focus(&focus, cx);
        })
        .unwrap();
    }
}

/// Views drawn again whole, and around a nested view that was built, leave
/// the tree drawing from scratch does, focus and active descendant included.
#[test]
fn views_drawn_again_keep_their_accessibility_nodes() {
    let mut cx = TestAppContext::single();
    let clicks = Rc::new(Cell::new(0));
    let windows = accessible_windows(&mut cx, |cx| ItemList::new(true, &clicks, cx));
    // The first frame with accessibility active has no tree to draw again
    // from; the second does.
    let (retained, from_scratch, _) = draw_both(&mut cx, windows);
    assert_same(&retained, &from_scratch);
    // The window, the list, and each item's node, button and image.
    assert_eq!(a11y_lines(&retained), 2 + 4 * 3, "{retained:#?}");
    let (retained, from_scratch, work) = draw_both(&mut cx, windows);
    assert_same(&retained, &from_scratch);
    assert_eq!(work.view_rebuilds.accessibility, 0, "{work:?}");

    // Nothing changed but the list: the items are drawn again.
    update_lists(&mut cx, windows, |_, cx| cx.notify());
    let (retained, from_scratch, work) = draw_both(&mut cx, windows);
    assert_same(&retained, &from_scratch);
    assert!(work.views_reused >= 4, "{work:?}");

    // One item changed: the list is drawn again around it.
    update_lists(&mut cx, windows, |list, cx| {
        list.items[2].update(cx, |item, cx| {
            item.label = 6;
            cx.notify();
        })
    });
    let (retained, from_scratch, work) = draw_both(&mut cx, windows);
    assert_same(&retained, &from_scratch);
    assert!(work.views_spliced >= 1, "{work:?}");

    // The list focused: one item claims to be its active descendant.
    focus_lists(&mut cx, windows);
    let (retained, from_scratch, _) = draw_both(&mut cx, windows);
    assert_same(&retained, &from_scratch);
    assert!(
        retained.iter().any(|line| line == "a11y focus r.0.1"),
        "the item claiming to be the active descendant is focused: {retained:#?}"
    );
    update_lists(&mut cx, windows, |list, cx| {
        list.items[1].update(cx, |item, cx| {
            item.label = 7;
            cx.notify();
        })
    });
    let (retained, from_scratch, work) = draw_both(&mut cx, windows);
    assert_same(&retained, &from_scratch);
    assert!(work.views_spliced >= 1, "{work:?}");
    // The active descendant was spliced out; the list itself is focused.
    assert!(retained.iter().any(|line| line == "a11y focus r.0"), "{retained:#?}");
    update_lists(&mut cx, windows, |list, cx| {
        list.title += 1;
        cx.notify();
    });
    let (retained, from_scratch, _) = draw_both(&mut cx, windows);
    assert_same(&retained, &from_scratch);
}

/// Items pushed down by a header that grew are drawn again moved, their
/// nodes moved with them.
#[test]
fn views_drawn_again_moved_move_their_accessibility_nodes() {
    let mut cx = TestAppContext::single();
    let clicks = Rc::new(Cell::new(0));
    let windows = accessible_windows(&mut cx, |cx| ItemList::new(false, &clicks, cx));
    draw_both(&mut cx, windows);
    draw_both(&mut cx, windows);
    for header in [10., 25., 5.25, 7., 0.] {
        update_lists(&mut cx, windows, |list, cx| {
            list.header = header;
            cx.notify();
        });
        let (retained, from_scratch, work) = draw_both(&mut cx, windows);
        assert_same(&retained, &from_scratch);
        assert!(work.views_moved >= 4, "header {header}: {work:?}");
    }
}

/// An item whose synthetic node has bounds of its own is built rather than
/// drawn again moved: the code that placed the node may not place it where
/// its element is.
#[test]
fn a_view_with_a_placed_synthetic_node_is_built_where_it_moved() {
    let mut cx = TestAppContext::single();
    let clicks = Rc::new(Cell::new(0));
    let windows = accessible_windows(&mut cx, |cx| ItemList::new(true, &clicks, cx));
    draw_both(&mut cx, windows);
    draw_both(&mut cx, windows);
    update_lists(&mut cx, windows, |list, cx| {
        list.header = 10.;
        cx.notify();
    });
    let (retained, from_scratch, work) = draw_both(&mut cx, windows);
    assert_same(&retained, &from_scratch);
    assert_eq!(work.views_moved, 0, "{work:?}");
}

/// The action listeners of a view drawn again answer the actions asked of
/// its nodes, as those of a view built would.
#[test]
fn views_drawn_again_answer_accessibility_actions() {
    let mut cx = TestAppContext::single();
    let clicks = Rc::new(Cell::new(0));
    let windows = accessible_windows(&mut cx, |cx| ItemList::new(true, &clicks, cx));
    draw_both(&mut cx, windows);
    for _ in 0..3 {
        update_lists(&mut cx, windows, |_, cx| cx.notify());
        let (retained, from_scratch, work) = draw_both(&mut cx, windows);
        assert_same(&retained, &from_scratch);
        assert!(work.views_reused >= 4, "{work:?}");
    }
    let before = clicks.get();
    cx.update_window(windows[0].into(), |_, window, cx| {
        let buttons: Vec<_> = window
            .a11y
            .action_listeners
            .iter()
            .filter(|(_, listeners)| listeners.iter().any(|(action, _)| *action == Action::Click))
            .map(|(node, _)| *node)
            .collect();
        assert_eq!(buttons.len(), 4, "every button listens for clicks");
        for node in buttons {
            window.handle_a11y_action(
                ActionRequest {
                    action: Action::Click,
                    target_tree: accesskit::TreeId::ROOT,
                    target_node: node,
                    data: None,
                },
                cx,
            );
        }
    })
    .unwrap();
    assert_eq!(clicks.get(), before + 4);
}

/// The first frame drawn with accessibility active builds every view, since
/// the frame before logged no nodes to draw again; the next draws them again.
#[test]
fn turning_accessibility_on_builds_every_view_once() {
    let mut cx = TestAppContext::single();
    cx.update(|cx| cx.set_view_retention(true));
    let clicks = Rc::new(Cell::new(0));
    let window = cx.add_window(|_, cx| ItemList::new(true, &clicks, cx));
    // The list is notified before each frame, so that its items are drawn
    // again one by one rather than the whole window at once.
    let draw = |cx: &mut TestAppContext| {
        cx.update_window(window.into(), |_, window, _| window.reset_frame_work_stats(false))
            .unwrap();
        window.update(cx, |_, _, cx| cx.notify()).unwrap();
        cx.run_until_parked();
        cx.update_window(window.into(), |_, window, cx| {
            // Drawn here unless the notification drew it.
            if window.frame_work_stats().frames == 0 {
                window.draw(cx).clear(cx);
            }
            (window.frame_work_stats(), describe_frame(window))
        })
        .unwrap()
    };
    draw(&mut cx);
    let (work, _) = draw(&mut cx);
    assert!(work.views_reused >= 4, "{work:?}");
    // Activated the way the platform does it: the flag is set and the window
    // asked to refresh, which can come a frame later.
    cx.update_window(window.into(), |_, window, _| window.a11y.force_active())
        .unwrap();
    let (work, first) = draw(&mut cx);
    assert_eq!(work.views_reused, 0, "{work:?}");
    assert!(work.view_rebuilds.accessibility >= 4, "{work:?}");
    assert_eq!(a11y_lines(&first), 2 + 4 * 3, "{first:#?}");
    let (work, second) = draw(&mut cx);
    assert!(work.views_reused >= 4, "{work:?}");
    assert_eq!(work.view_rebuilds.accessibility, 0, "{work:?}");
    assert_same(&second, &first);
    let reasons = cx
        .update_window(window.into(), |_, window, _| window.view_rebuild_reasons().to_vec())
        .unwrap();
    assert!(
        !reasons
            .iter()
            .any(|(_, reason)| *reason == ViewRebuildReason::Accessibility),
        "{reasons:?}"
    );
}

/// A row of a list that asks to be scrolled into view has the list lay its
/// rows out again, rolling back what it prepainted: the nodes the rolled-back
/// prepaint pushed are pushed once more, not twice.
struct Revealing {
    reveal: bool,
}

impl Render for Revealing {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        div()
            .id("row")
            .role(Role::Row)
            .h(px(30.))
            .child(div().id("cell").role(Role::Cell).size(px(10.)))
            .when(self.reveal, |this| {
                this.child(
                    crate::canvas(
                        |bounds, window, _| window.request_autoscroll(bounds),
                        |_, _, _, _| {},
                    )
                    .size(px(4.)),
                )
            })
    }
}

struct RevealingList {
    rows: Vec<Entity<Revealing>>,
    state: ListState,
}

impl Render for RevealingList {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        let rows = self.rows.clone();
        div().size_full().child(
            list(self.state.clone(), move |ix, _, _| rows[ix].clone().into_any_element())
                .w(px(100.))
                .h(px(60.)),
        )
    }
}

#[test]
fn a_rolled_back_prepaint_pushes_its_nodes_once() {
    let mut cx = TestAppContext::single();
    let windows = accessible_windows(&mut cx, |cx| RevealingList {
        rows: (0..12).map(|_| cx.new(|_| Revealing { reveal: false })).collect(),
        state: ListState::new(12, ListAlignment::Top, px(20.)),
    });
    draw_both(&mut cx, windows);
    draw_both(&mut cx, windows);
    for ix in [8, 2, 11] {
        for window in windows {
            let row = window
                .read_with(&cx, |view, _| view.rows[ix].clone())
                .unwrap();
            row.update(&mut cx, |row, cx| {
                row.reveal = true;
                cx.notify();
            });
        }
        let (retained, from_scratch, _) = draw_both(&mut cx, windows);
        assert_same(&retained, &from_scratch);
        let (retained, from_scratch, _) = draw_both(&mut cx, windows);
        assert_same(&retained, &from_scratch);
        for window in windows {
            let row = window
                .read_with(&cx, |view, _| view.rows[ix].clone())
                .unwrap();
            row.update(&mut cx, |row, cx| {
                row.reveal = false;
                cx.notify();
            });
        }
        let (retained, from_scratch, _) = draw_both(&mut cx, windows);
        assert_same(&retained, &from_scratch);
    }
}

/// A view drawn again inside a deferred draw (a popover) keeps its nodes,
/// which hang off the window's root as the deferred draw's do.
struct Popover {
    open: bool,
    item: Entity<Item>,
    label: usize,
}

impl Render for Popover {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        div()
            .size_full()
            .child(
                div()
                    .id("anchor")
                    .role(Role::Group)
                    .aria_label(format!("anchor {}", self.label))
                    .size(px(20.)),
            )
            .when(self.open, |this| {
                this.child(
                    crate::deferred(
                        div()
                            .id("popover")
                            .role(Role::Dialog)
                            .absolute()
                            .top(px(30.))
                            .w(px(60.))
                            .child(self.item.clone()),
                    )
                    .with_priority(1),
                )
            })
    }
}

#[test]
fn a_deferred_draw_drawn_again_keeps_its_accessibility_nodes() {
    let mut cx = TestAppContext::single();
    let clicks = Rc::new(Cell::new(0));
    let windows = accessible_windows(&mut cx, |cx| {
        let clicks = clicks.clone();
        Popover {
            open: true,
            item: cx.new(|_| Item {
                label: 1,
                placed: false,
                clicks,
            }),
            label: 0,
        }
    });
    draw_both(&mut cx, windows);
    for step in 0..4 {
        for window in windows {
            window
                .update(&mut cx, |popover, _, cx| {
                    match step {
                        1 => popover.item.update(cx, |item, cx| {
                            item.label += 4;
                            cx.notify();
                        }),
                        2 => popover.open = !popover.open,
                        _ => popover.label += 1,
                    }
                    cx.notify();
                })
                .unwrap();
        }
        let (retained, from_scratch, _) = draw_both(&mut cx, windows);
        assert_same(&retained, &from_scratch);
    }
}

/// Without accessibility, nothing is logged for drawing nodes again, and no
/// view is built for it.
#[test]
fn without_accessibility_nothing_is_logged() {
    let mut cx = TestAppContext::single();
    cx.update(|cx| cx.set_view_retention(true));
    let clicks = Rc::new(Cell::new(0));
    let window = cx.add_window(|_, cx| ItemList::new(true, &clicks, cx));
    for _ in 0..3 {
        window.update(&mut cx, |_, _, cx| cx.notify()).unwrap();
        cx.update_window(window.into(), |_, window, cx| {
            window.reset_frame_work_stats(false);
            window.draw(cx).clear(cx);
            assert!(!window.a11y.is_active());
            assert_eq!(window.a11y.nodes.log.ops_len(), 0);
            assert_eq!(window.a11y.nodes.log.actions_len(), 0);
            assert_eq!(window.frame_work_stats().view_rebuilds.accessibility, 0);
        })
        .unwrap();
    }
}

/// The random histories of `tests.rs`, with both windows building their
/// accessibility trees: views are drawn again, moved and spliced as often
/// as without accessibility, and every frame's tree is the one drawing from
/// scratch builds. `GPUI_RETAINED_VIEWS_ORACLE_SEEDS` sets how many seeds
/// run.
#[test]
fn histories_with_accessibility_match_frames_drawn_from_scratch() {
    let seeds = std::env::var("GPUI_RETAINED_VIEWS_ORACLE_SEEDS")
        .ok()
        .and_then(|seeds| seeds.parse().ok())
        .unwrap_or(8);
    let mut reuse = super::tests::Reuse::default();
    for seed in 0..seeds {
        let run = super::tests::run_with(seed, 50, true);
        reuse.reused += run.reused;
        reuse.moved += run.moved;
        reuse.spliced += run.spliced;
    }
    let super::tests::Reuse {
        reused,
        moved,
        spliced,
    } = reuse;
    assert!(reused > 400, "views were drawn again {reused} times");
    assert!(moved > 40, "views were drawn again moved {moved} times of {reused}");
    assert!(spliced > 40, "views were drawn again around others {spliced} times of {reused}");
}
