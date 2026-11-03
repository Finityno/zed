//! Scenarios that broke other retained-view designs, each checked against
//! this one: every frame of a window drawing views again from the last frame
//! must be the frame a twin window draws from scratch, with every view built
//! and, where the scenario is about layout, every layout node made afresh.
//!
//! - a view built in a gap of a view drawn again around it (see
//!   [`super::splice`]) while it is laid out outside its host's layout tree
//!   (a root of its own, a list item, a container query), inside a deferred
//!   or anchored draw, together with a deferred view, under a host whose own
//!   reads changed, and while it adds or drops layout nodes;
//! - a cached view whose path from the root changes as an anonymous wrapper
//!   is put around it or taken away;
//! - a view moved from one parent to another and back;
//! - text given new text that wraps otherwise than the old, measured at a
//!   width other than the one it is placed at, and asked for its unwrapped
//!   width after it was measured narrow.

use super::a11y_tests::{assert_same, draw_both};
use crate::{
    AnyElement, App, Bounds, Context, Element, ElementId, Entity, FrameWorkStats, GlobalElementId,
    InspectorElementId, IntoElement, LayoutId, ListAlignment, ListState, Pixels, Render,
    ScrollHandle, SharedString, Style, StyleRefinement, TestAppContext, Window, WindowHandle,
    anchored, container_query, deferred, div, list, point, prelude::*, px, relative,
    uniform_list,
};
use std::{cell::Cell, rc::Rc};

/// A window drawing views again and its twin drawn from scratch, with the
/// twin making every layout node afresh when `fresh_layout`.
fn twin_windows<V: 'static + Render>(
    cx: &mut TestAppContext,
    fresh_layout: bool,
    build: impl Fn(&mut Context<V>) -> V,
) -> [WindowHandle<V>; 2] {
    cx.update(|cx| cx.set_view_retention(true));
    let windows = [(); 2].map(|_| cx.add_window(|_, cx| build(cx)));
    if fresh_layout {
        cx.update_window(windows[1].into(), |_, window, _| {
            window.layout_keys.set_enabled(false)
        })
        .unwrap();
    }
    windows
}

/// Draws both windows and checks they agree, returning the first's frame and
/// work.
#[track_caller]
fn check<V: 'static>(
    cx: &mut TestAppContext,
    windows: [WindowHandle<V>; 2],
) -> (Vec<String>, FrameWorkStats) {
    let (retained, from_scratch, work) = draw_both(cx, windows);
    assert_same(&retained, &from_scratch);
    (retained, work)
}

fn update_both<V: 'static + Render>(
    cx: &mut TestAppContext,
    windows: [WindowHandle<V>; 2],
    update: impl Fn(&mut V, &mut Context<V>),
) {
    for window in windows {
        window.update(cx, |view, _, cx| update(view, cx)).unwrap();
    }
}

/// Updates the entity `of` picks out of each window's root.
fn update_entity<V: 'static + Render, E: 'static>(
    cx: &mut TestAppContext,
    windows: [WindowHandle<V>; 2],
    of: impl Fn(&V) -> Entity<E>,
    update: impl Fn(&mut E, &mut Context<E>),
) {
    for window in windows {
        let entity = window.read_with(cx, |view, _| of(view)).unwrap();
        entity.update(cx, |entity, cx| update(entity, cx));
    }
}

/// Lays its child out as a root of its own as it prepaints, as a list lays
/// out its items: outside the layout tree of the view around it.
struct RootPlacer(Option<AnyElement>);

impl IntoElement for RootPlacer {
    type Element = Self;

    fn into_element(self) -> Self {
        self
    }
}

impl Element for RootPlacer {
    type RequestLayoutState = ();
    type PrepaintState = Option<AnyElement>;

    fn id(&self) -> Option<ElementId> {
        None
    }

    fn source_location(&self) -> Option<&'static std::panic::Location<'static>> {
        None
    }

    fn request_layout(
        &mut self,
        _: Option<&GlobalElementId>,
        _: Option<&InspectorElementId>,
        window: &mut Window,
        cx: &mut App,
    ) -> (LayoutId, ()) {
        let mut style = Style::default();
        style.size.width = relative(1.).into();
        style.size.height = relative(1.).into();
        (window.request_layout(style, None, cx), ())
    }

    fn prepaint(
        &mut self,
        _: Option<&GlobalElementId>,
        _: Option<&InspectorElementId>,
        bounds: Bounds<Pixels>,
        _: &mut (),
        window: &mut Window,
        cx: &mut App,
    ) -> Option<AnyElement> {
        let mut child = self.0.take()?;
        child.prepaint_as_root(bounds.origin, bounds.size.into(), window, cx);
        Some(child)
    }

    fn paint(
        &mut self,
        _: Option<&GlobalElementId>,
        _: Option<&InspectorElementId>,
        _: Bounds<Pixels>,
        _: &mut (),
        child: &mut Option<AnyElement>,
        window: &mut Window,
        cx: &mut App,
    ) {
        if let Some(child) = child {
            child.paint(window, cx);
        }
    }
}

/// A status line whose text changes length.
struct Status(SharedString);

impl Render for Status {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        div()
            .size_full()
            .flex()
            .items_center()
            .justify_center()
            .child(self.0.clone())
    }
}

#[derive(Clone, Copy, Debug)]
enum Placement {
    Flow,
    Root,
    UniformList,
    List,
    ContainerQuery,
    Deferred,
    Anchored,
}

/// Places a status view, under a header.
struct StatusHost {
    status: Entity<Status>,
    placement: Placement,
    list: ListState,
}

impl Render for StatusHost {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        let status = self.status.clone();
        let child = div().size_full().child(status.clone()).into_any_element();
        let body = match self.placement {
            Placement::Flow => child,
            Placement::Root => RootPlacer(Some(child)).into_any_element(),
            Placement::UniformList => uniform_list("status-list", 1, move |_, _, _| {
                vec![div().h(px(40.)).child(status.clone()).into_any_element()]
            })
            .size_full()
            .into_any_element(),
            Placement::List => list(self.list.clone(), move |_, _, _| {
                div().h(px(40.)).child(status.clone()).into_any_element()
            })
            .size_full()
            .into_any_element(),
            Placement::ContainerQuery => {
                container_query(move |_, _, _| child).into_any_element()
            }
            Placement::Deferred => deferred(child).into_any_element(),
            Placement::Anchored => anchored()
                .position(point(px(30.), px(40.)))
                .child(div().w(px(200.)).h(px(40.)).child(child))
                .into_any_element(),
        };
        div()
            .size_full()
            .flex()
            .flex_col()
            .child("header")
            .child(div().flex_1().child(body))
    }
}

/// Holds a view the way an application's window holds its workspace: a
/// view drawn again around the view drawn again around the status.
struct Shell {
    host: Entity<StatusHost>,
}

impl Render for Shell {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        div().size_full().p_1().child(self.host.clone())
    }
}

fn status_changes_at(placement: Placement) {
    let mut cx = super::super::layout_retention_tests::text_system_context(0);
    let windows = twin_windows(&mut cx, true, |cx| Shell {
        host: cx.new(|cx| StatusHost {
            status: cx.new(|_| Status("Starting…".into())),
            placement,
            list: ListState::new(1, ListAlignment::Top, px(0.)),
        }),
    });
    let (first, _) = check(&mut cx, windows);
    check(&mut cx, windows);
    for text in [
        "Session unavailable: a longer status",
        "Ready",
        "Another longer status",
    ] {
        for window in windows {
            let status = window
                .read_with(&cx, |shell, cx| shell.host.read(cx).status.clone())
                .unwrap();
            status.update(&mut cx, |status, cx| {
                status.0 = text.into();
                cx.notify();
            });
        }
        // The frame the notification draws, with nothing drawn after it.
        let (changed, _) = check(&mut cx, windows);
        assert_ne!(first, changed, "{placement:?}: the status changed");
        check(&mut cx, windows);
    }
}

#[test]
fn a_status_changed_in_flow_is_laid_out_again() {
    status_changes_at(Placement::Flow);
}

#[test]
fn a_status_changed_under_a_root_of_its_own_is_laid_out_again() {
    status_changes_at(Placement::Root);
}

#[test]
fn a_status_changed_in_a_uniform_list_is_laid_out_again() {
    status_changes_at(Placement::UniformList);
}

#[test]
fn a_status_changed_in_a_list_is_laid_out_again() {
    status_changes_at(Placement::List);
}

#[test]
fn a_status_changed_under_a_container_query_is_laid_out_again() {
    status_changes_at(Placement::ContainerQuery);
}

#[test]
fn a_status_changed_in_a_deferred_draw_is_laid_out_again() {
    status_changes_at(Placement::Deferred);
}

#[test]
fn a_status_changed_in_an_anchored_draw_is_laid_out_again() {
    status_changes_at(Placement::Anchored);
}

/// A view painting a tint, and counting how often it was built.
struct Tinted {
    tint: usize,
    builds: Rc<Cell<usize>>,
}

impl Render for Tinted {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        self.builds.set(self.builds.get() + 1);
        let colors = [crate::black(), crate::white(), crate::red()];
        div()
            .w(px(60.))
            .h(px(20.))
            .bg(colors[self.tint % colors.len()])
            .child(format!("tint {}", self.tint))
    }
}

/// A view in flow and one in an anchored overlay deferred over it.
struct FlowAndOverlay {
    flow: Entity<Tinted>,
    overlay: Entity<Tinted>,
}

impl Render for FlowAndOverlay {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        div().size_full().child(self.flow.clone()).child(deferred(
            anchored()
                .position(point(px(10.), px(10.)))
                .child(div().w(px(200.)).h(px(40.)).child(self.overlay.clone())),
        ))
    }
}

struct FlowShell {
    content: Entity<FlowAndOverlay>,
}

impl Render for FlowShell {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        div().size_full().child(self.content.clone())
    }
}

/// A view in flow and a view in a deferred draw notified together are both
/// built on the frame the notification draws: the deferred one is not drawn
/// again from the last frame around the one in flow.
fn flow_and_overlay_notified_together(idle_first: bool) {
    let mut cx = super::super::layout_retention_tests::text_system_context(0);
    let builds = [(); 2].map(|_| (Rc::new(Cell::new(0)), Rc::new(Cell::new(0))));
    let next = Cell::new(0);
    let windows = twin_windows(&mut cx, false, |cx| {
        let (flow_builds, overlay_builds) = builds[next.get()].clone();
        next.set(next.get() + 1);
        FlowShell {
            content: cx.new(|cx| FlowAndOverlay {
                flow: cx.new(|_| Tinted {
                    tint: 0,
                    builds: flow_builds,
                }),
                overlay: cx.new(|_| Tinted {
                    tint: 0,
                    builds: overlay_builds,
                }),
            }),
        }
    });
    check(&mut cx, windows);
    if idle_first {
        check(&mut cx, windows);
    }
    let (flow_builds, overlay_builds) = &builds[0];
    for tint in [1, 0, 1] {
        let before = (flow_builds.get(), overlay_builds.get());
        for window in windows {
            let (flow, overlay) = window
                .read_with(&cx, |shell, cx| {
                    let content = shell.content.read(cx);
                    (content.flow.clone(), content.overlay.clone())
                })
                .unwrap();
            cx.update(|cx| {
                for view in [&flow, &overlay] {
                    view.update(cx, |view, cx| {
                        view.tint = tint;
                        cx.notify();
                    });
                }
            });
        }
        check(&mut cx, windows);
        assert_eq!(flow_builds.get(), before.0 + 1, "the view in flow is built");
        assert_eq!(overlay_builds.get(), before.1 + 1, "the deferred view is built");
        check(&mut cx, windows);
    }
}

#[test]
fn views_in_flow_and_deferred_notified_together_are_both_built() {
    flow_and_overlay_notified_together(false);
}

#[test]
fn views_in_flow_and_deferred_drawn_again_then_notified_together_are_both_built() {
    flow_and_overlay_notified_together(true);
}

/// A host that paints by where its scroll handle is scrolled to, around a
/// child view.
struct OffsetReader {
    scroll: ScrollHandle,
    child: Entity<Tinted>,
    builds: Rc<Cell<usize>>,
}

impl Render for OffsetReader {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        self.builds.set(self.builds.get() + 1);
        let color = if self.scroll.offset().y < px(0.) {
            crate::white()
        } else {
            crate::black()
        };
        div()
            .size_full()
            .bg(color)
            .child(
                div()
                    .id("scroller")
                    .overflow_y_scroll()
                    .track_scroll(&self.scroll)
                    .h(px(40.))
                    .child(div().h(px(200.)).w(px(20.))),
            )
            .child(self.child.clone())
    }
}

/// A host whose own read of a scroll offset changed is built, not drawn
/// again around a child notified in the same frame.
#[test]
fn a_host_whose_scroll_offset_read_changed_is_built_around_a_notified_child() {
    let mut cx = super::super::layout_retention_tests::text_system_context(0);
    let builds = [Rc::new(Cell::new(0)), Rc::new(Cell::new(0))];
    let scrolls = [ScrollHandle::new(), ScrollHandle::new()];
    let next = Cell::new(0);
    let windows = twin_windows(&mut cx, false, |cx| {
        let ix = next.get();
        next.set(ix + 1);
        OffsetReader {
            scroll: scrolls[ix].clone(),
            child: cx.new(|_| Tinted {
                tint: 0,
                builds: Rc::new(Cell::new(0)),
            }),
            builds: builds[ix].clone(),
        }
    });
    check(&mut cx, windows);
    check(&mut cx, windows);
    let before = builds[0].get();
    for (ix, window) in windows.into_iter().enumerate() {
        scrolls[ix].set_offset(point(px(0.), px(-20.)));
        let child = window.read_with(&cx, |host, _| host.child.clone()).unwrap();
        child.update(&mut cx, |child, cx| {
            child.tint = 1;
            cx.notify();
        });
    }
    check(&mut cx, windows);
    assert_eq!(builds[0].get(), before + 1, "the host's own read changed");
}

/// A view of fixed size that adds an absolutely placed box, with a layout
/// node of its own, and takes it away.
struct Growing {
    extra: Rc<Cell<bool>>,
}

impl Render for Growing {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        div()
            .relative()
            .w(px(100.))
            .h(px(40.))
            .child(div().id("base").size_full())
            .when(self.extra.get(), |this| {
                this.child(
                    div()
                        .id("extra")
                        .absolute()
                        .top_0()
                        .left_0()
                        .size(px(10.))
                        .bg(crate::white()),
                )
            })
    }
}

struct GrowingHost {
    child: Entity<Growing>,
    builds: Rc<Cell<usize>>,
}

impl Render for GrowingHost {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        self.builds.set(self.builds.get() + 1);
        div().size_full().p_2().child(self.child.clone())
    }
}

/// Layout nodes a view built in a gap adds, or drops, are kept with the
/// view drawn again around it, through the frames that draw both again.
#[test]
fn layout_nodes_added_in_a_gap_outlive_the_frames_after() {
    let mut cx = super::super::layout_retention_tests::text_system_context(0);
    let extra = Rc::new(Cell::new(false));
    let host_builds = [Rc::new(Cell::new(0)), Rc::new(Cell::new(0))];
    let next = Cell::new(0);
    let windows = twin_windows(&mut cx, true, |cx| {
        let ix = next.get();
        next.set(ix + 1);
        GrowingHost {
            child: cx.new(|_| Growing {
                extra: extra.clone(),
            }),
            builds: host_builds[ix].clone(),
        }
    });
    check(&mut cx, windows);
    check(&mut cx, windows);
    for shown in [true, false, true] {
        extra.set(shown);
        update_entity(&mut cx, windows, |host| host.child.clone(), |_, cx| cx.notify());
        let (changed, _) = check(&mut cx, windows);
        for _ in 0..3 {
            update_both(&mut cx, windows, |_, cx| cx.notify());
            let (idle, work) = check(&mut cx, windows);
            assert_eq!(idle, changed, "the frames after draw what the change drew");
            assert!(work.views_reused >= 1, "{work:?}");
        }
    }
}

/// A frame shown inside a cached view, which is drawn again from the last
/// frame while its parent puts an anonymous wrapper around it or takes it
/// away, and then changes.
struct Framed {
    label: usize,
}

impl Render for Framed {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        div().size_full().child(format!("frame {}", self.label))
    }
}

struct Workspace {
    framed: Entity<Framed>,
    wrapped: bool,
}

impl Render for Workspace {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        let framed = self
            .framed
            .clone()
            .cached(StyleRefinement::default().size_full())
            .into_any_element();
        let leaf = if self.wrapped {
            div().size_full().child(framed).into_any_element()
        } else {
            framed
        };
        div().size_full().child(leaf)
    }
}

struct WorkspaceRoot {
    workspace: Entity<Workspace>,
}

impl Render for WorkspaceRoot {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        div().size_full().child(self.workspace.clone())
    }
}

fn wrapper_changes_around_a_cached_view(initially: bool, then: bool) {
    let mut cx = super::super::layout_retention_tests::text_system_context(0);
    let windows = twin_windows(&mut cx, true, |cx| WorkspaceRoot {
        workspace: cx.new(|cx| Workspace {
            framed: cx.new(|_| Framed { label: 0 }),
            wrapped: initially,
        }),
    });
    check(&mut cx, windows);
    check(&mut cx, windows);
    update_entity(
        &mut cx,
        windows,
        |root| root.workspace.clone(),
        |workspace, cx| {
            workspace.wrapped = then;
            cx.notify();
        },
    );
    check(&mut cx, windows);
    for label in 1..=3 {
        for window in windows {
            let framed = window
                .read_with(&cx, |root, cx| root.workspace.read(cx).framed.clone())
                .unwrap();
            framed.update(&mut cx, |framed, cx| {
                framed.label = label;
                cx.notify();
            });
        }
        check(&mut cx, windows);
        check(&mut cx, windows);
    }
}

#[test]
fn a_cached_view_unwrapped_then_changed_is_laid_out_where_it_is() {
    wrapper_changes_around_a_cached_view(true, false);
}

#[test]
fn a_cached_view_wrapped_then_changed_is_laid_out_where_it_is() {
    wrapper_changes_around_a_cached_view(false, true);
}

#[test]
fn a_cached_view_kept_in_its_wrapper_then_changed_is_laid_out_where_it_is() {
    wrapper_changes_around_a_cached_view(true, true);
}

/// Two anonymous parents of fixed widths, one of which holds a child view:
/// the child's id is the same under either, its layout path is not.
struct TwoParents {
    child: Entity<Tinted>,
    under_second: bool,
    label: usize,
}

impl Render for TwoParents {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        let child = self.child.clone();
        div()
            .size_full()
            .flex()
            .flex_col()
            .child(format!("parents {}", self.label))
            .child(
                div()
                    .w(px(120.))
                    .p_1()
                    .when(!self.under_second, |this| this.child(child.clone())),
            )
            .child(
                div()
                    .w(px(200.))
                    .p_2()
                    .when(self.under_second, |this| this.child(child)),
            )
    }
}

/// A child view moved to another parent and back is laid out under the
/// parent it is in, and the parent it left is laid out without it, while
/// the child itself is drawn again from the last frame wherever it can be.
#[test]
fn a_view_moved_to_another_parent_and_back_is_laid_out_where_it_is() {
    let mut cx = super::super::layout_retention_tests::text_system_context(0);
    let windows = twin_windows(&mut cx, true, |cx| TwoParents {
        child: cx.new(|_| Tinted {
            tint: 0,
            builds: Rc::new(Cell::new(0)),
        }),
        under_second: false,
        label: 0,
    });
    check(&mut cx, windows);
    check(&mut cx, windows);
    for step in 0..6 {
        update_both(&mut cx, windows, |parents, cx| {
            if step % 3 == 2 {
                parents.label += 1;
            } else {
                parents.under_second = !parents.under_second;
            }
            cx.notify();
        });
        check(&mut cx, windows);
        if step == 3 {
            update_entity(&mut cx, windows, |parents| parents.child.clone(), |child, cx| {
                child.tint += 1;
                cx.notify();
            });
            check(&mut cx, windows);
        }
        check(&mut cx, windows);
    }
}

/// Text in a box that is either as wide as its text or of a fixed width.
struct Fitted {
    text: &'static str,
    width: Option<Pixels>,
}

impl Render for Fitted {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        div().flex().items_start().child(
            div()
                .when_some(self.width, |this, width| this.w(width))
                .child(self.text),
        )
    }
}

/// New text on a text node kept from the last frame, which the last text
/// was wrapped on in a narrow box, is laid out as text never measured: it
/// does not answer for its unwrapped width with the box's width once the box
/// lets it be as wide as it likes.
#[test]
fn new_text_after_text_wrapped_narrow_is_laid_out_afresh() {
    // The same length unwrapped, breaking into lines of other lengths in a
    // narrow box.
    const WRAPPED: &str = "aaaa bbbb cccc dddd";
    const NEW: &str = "aaaaaaa bbbbbbbbbbb";
    let mut cx = super::super::layout_retention_tests::text_system_context(0);
    let windows = twin_windows(&mut cx, true, |_| Fitted {
        text: WRAPPED,
        width: Some(px(40.)),
    });
    check(&mut cx, windows);
    check(&mut cx, windows);
    update_both(&mut cx, windows, |fitted, cx| {
        fitted.text = NEW;
        fitted.width = None;
        cx.notify();
    });
    check(&mut cx, windows);
    check(&mut cx, windows);
    update_both(&mut cx, windows, |fitted, cx| {
        fitted.text = WRAPPED;
        fitted.width = Some(px(40.));
        cx.notify();
    });
    check(&mut cx, windows);
}

const FLEXED_TEXT: &str =
    "a heading long enough to wrap onto several lines in the narrow column it is laid out in";

/// Text taking the rest of a row (`flex_1`, free to shrink to nothing) in a
/// column capped at a width, beside a line that changes every frame: the row
/// is sized to its content again on every frame, which measures the text at
/// width zero after it was placed.
struct FlexedText {
    tick: usize,
}

impl Render for FlexedText {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        let content = div()
            .flex()
            .flex_col()
            .child(div().flex().child(div().flex_1().min_w_0().child(FLEXED_TEXT)))
            .child(SharedString::from(format!("frame {}", self.tick)));
        div().flex().child(
            div().w_full().max_w(px(300.)).child(
                div().flex().flex_col().child(
                    div()
                        .flex()
                        .flex_col()
                        .w_full()
                        .child(div().flex().flex_col().min_w_0().w_full().child(content)),
                ),
            ),
        )
    }
}

/// A text leaf whose node is placed from Taffy's cache is painted in the
/// lines of the box it is placed in, not in those of a narrower measurement
/// taken after it was placed.
#[test]
fn text_placed_from_the_cache_paints_the_lines_of_its_box() {
    let mut cx = super::super::layout_retention_tests::text_system_context(0);
    let windows = twin_windows(&mut cx, true, |_| FlexedText { tick: 0 });
    check(&mut cx, windows);
    for tick in 1..5 {
        update_both(&mut cx, windows, |text, cx| {
            text.tick = tick;
            cx.notify();
        });
        check(&mut cx, windows);
    }
}

/// Text in a `min_w_0` child of a row as wide as its content, in a column
/// whose width changes: narrowed, the text is measured wrapped; widened
/// again, the row asks the text for its unwrapped width, which a wrapped
/// measurement must not answer.
struct ContentSizedRow {
    width: f32,
    tick: usize,
}

impl Render for ContentSizedRow {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        div()
            .w(px(self.width))
            .flex()
            .flex_col()
            .items_start()
            .child(
                div()
                    .max_w_full()
                    .flex()
                    .child(div().min_w_0().child("the quick brown fox jumps over the lazy dog")),
            )
            .child(SharedString::from(format!("tick {}", self.tick)))
    }
}

#[test]
fn text_measured_narrow_answers_for_its_unwrapped_width_unwrapped() {
    let mut cx = super::super::layout_retention_tests::text_system_context(0);
    let windows = twin_windows(&mut cx, true, |_| ContentSizedRow {
        width: 600.,
        tick: 0,
    });
    check(&mut cx, windows);
    for (tick, width) in [(1, 600.), (2, 40.), (3, 40.), (4, 600.), (5, 600.), (6, 90.), (7, 600.)] {
        update_both(&mut cx, windows, |row, cx| {
            row.tick = tick;
            row.width = width;
            cx.notify();
        });
        check(&mut cx, windows);
    }
}
