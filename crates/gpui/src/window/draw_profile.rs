//! Per-draw accounting behind [`crate::DrawBreakdown`]: where a window draw's
//! time went, phase by phase.
//!
//! The fast path is a handful of `Instant::now()` calls per draw plus two per
//! taffy layout pass; everything costlier runs only for slow draws.

use std::time::Duration;

use scheduler::Instant;

use crate::DrawBreakdown;

/// The draw phase the clock is currently charging.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub(crate) enum DrawClockPhase {
    /// Setup, frame finish, focus listeners and the accessibility update.
    /// Not accumulated: the breakdown derives it as the remainder.
    Other,
    RequestLayout,
    Prepaint,
    Paint,
}

/// Charges elapsed draw time to phases. Taffy layout passes are carved out
/// of whichever phase they run in (see [`Self::begin_layout`]).
pub(crate) struct DrawClock {
    active: bool,
    phase: DrawClockPhase,
    phase_started_at: Instant,
    request_layout: Duration,
    layout: Duration,
    prepaint: Duration,
    paint: Duration,
    layout_passes: u16,
    views_rendered: u16,
    views_reused: u16,
}

impl DrawClock {
    pub(crate) fn new() -> Self {
        Self {
            active: false,
            phase: DrawClockPhase::Other,
            phase_started_at: Instant::now(),
            request_layout: Duration::ZERO,
            layout: Duration::ZERO,
            prepaint: Duration::ZERO,
            paint: Duration::ZERO,
            layout_passes: 0,
            views_rendered: 0,
            views_reused: 0,
        }
    }

    /// Starts charging a draw that began at `draw_start`, in
    /// [`DrawClockPhase::Other`].
    pub(crate) fn begin(&mut self, draw_start: Instant) {
        *self = Self {
            active: true,
            phase_started_at: draw_start,
            ..Self::new()
        };
    }

    fn charge(&mut self, until: Instant) {
        let elapsed = until.saturating_duration_since(self.phase_started_at);
        match self.phase {
            DrawClockPhase::Other => {}
            DrawClockPhase::RequestLayout => self.request_layout += elapsed,
            DrawClockPhase::Prepaint => self.prepaint += elapsed,
            DrawClockPhase::Paint => self.paint += elapsed,
        }
        self.phase_started_at = until;
    }

    /// Charges the time since the last mark to the current phase and
    /// switches to `next`. Returns the mark's timestamp.
    pub(crate) fn mark(&mut self, next: DrawClockPhase) -> Instant {
        let now = Instant::now();
        if self.active {
            self.charge(now);
            self.phase = next;
        }
        now
    }

    /// Called as a taffy layout pass starts: charges the current phase up to
    /// now, so the pass itself goes to `layout`. Returns `None` outside a
    /// profiled draw. Layout passes do not nest (the window's layout engine
    /// is taken for the duration of one), so no depth tracking is needed.
    pub(crate) fn begin_layout(&mut self) -> Option<Instant> {
        if !self.active {
            return None;
        }
        let now = Instant::now();
        self.charge(now);
        Some(now)
    }

    /// Called as a taffy layout pass ends, with [`Self::begin_layout`]'s
    /// result.
    pub(crate) fn end_layout(&mut self, started_at: Option<Instant>) {
        let Some(started_at) = started_at else {
            return;
        };
        let now = Instant::now();
        self.layout += now.saturating_duration_since(started_at);
        self.layout_passes = self.layout_passes.saturating_add(1);
        self.phase_started_at = now;
    }

    /// An entity view called `render`.
    pub(crate) fn count_render(&mut self) {
        if self.active {
            self.views_rendered = self.views_rendered.saturating_add(1);
        }
    }

    /// A cached view replayed its previous prepaint instead of rendering.
    pub(crate) fn count_reuse(&mut self) {
        if self.active {
            self.views_reused = self.views_reused.saturating_add(1);
        }
    }

    /// Ends the draw at `now` and returns its breakdown. `other` is left for
    /// [`crate::WindowProfiler::end_draw`], which knows the draw's end.
    pub(crate) fn finish(&mut self, now: Instant) -> DrawBreakdown {
        if self.active {
            self.charge(now);
        }
        self.active = false;
        fn micros(duration: Duration) -> u32 {
            duration.as_micros().min(u32::MAX as u128) as u32
        }
        DrawBreakdown {
            request_layout_us: micros(self.request_layout),
            layout_us: micros(self.layout),
            prepaint_us: micros(self.prepaint),
            paint_us: micros(self.paint),
            layout_passes: self.layout_passes,
            views_rendered: self.views_rendered,
            views_reused: self.views_reused,
            ..DrawBreakdown::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{cell::Cell, rc::Rc, time::Duration};

    use scheduler::Instant;

    use crate::{
        AppContext as _, Context, Entity, FrameTiming, IntoElement, ListAlignment, ListState,
        ParentElement as _, Render, RequestFrameOptions, Style, Styled as _, TestAppContext,
        TestWindow, Window, WindowHandle, WindowOptions, div, list, px, size,
    };

    const SPIN: Duration = Duration::from_millis(3);

    fn spin(duration: Duration) {
        let started_at = Instant::now();
        while started_at.elapsed() < duration {
            std::hint::spin_loop();
        }
    }

    #[derive(Clone, Copy, PartialEq, Eq)]
    enum SpinIn {
        Nowhere,
        Render,
        Measure,
    }

    /// Renders nothing but a leaf that is measured by a closure, spinning
    /// in its render or its measure closure when asked to.
    struct Worker {
        spin_in: Rc<Cell<SpinIn>>,
    }

    impl Render for Worker {
        fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
            if self.spin_in.get() == SpinIn::Render {
                spin(SPIN);
            }
            let spin_in = self.spin_in.clone();
            div()
                .w(px(10.))
                .h(px(10.))
                .child(MeasuredLeaf { spin_in })
        }
    }

    struct MeasuredLeaf {
        spin_in: Rc<Cell<SpinIn>>,
    }

    impl IntoElement for MeasuredLeaf {
        type Element = Self;

        fn into_element(self) -> Self {
            self
        }
    }

    impl crate::Element for MeasuredLeaf {
        type RequestLayoutState = ();
        type PrepaintState = ();

        fn id(&self) -> Option<crate::ElementId> {
            None
        }

        fn source_location(&self) -> Option<&'static core::panic::Location<'static>> {
            None
        }

        fn request_layout(
            &mut self,
            _: Option<&crate::GlobalElementId>,
            _: Option<&crate::InspectorElementId>,
            window: &mut Window,
            _: &mut crate::App,
        ) -> (crate::LayoutId, ()) {
            let spin_in = self.spin_in.clone();
            let layout_id =
                window.request_measured_layout(Style::default(), move |_, _, _, _| {
                    if spin_in.get() == SpinIn::Measure {
                        spin(SPIN);
                    }
                    size(px(5.), px(5.))
                });
            (layout_id, ())
        }

        fn prepaint(
            &mut self,
            _: Option<&crate::GlobalElementId>,
            _: Option<&crate::InspectorElementId>,
            _: crate::Bounds<crate::Pixels>,
            _: &mut (),
            _: &mut Window,
            _: &mut crate::App,
        ) {
        }

        fn paint(
            &mut self,
            _: Option<&crate::GlobalElementId>,
            _: Option<&crate::InspectorElementId>,
            _: crate::Bounds<crate::Pixels>,
            _: &mut (),
            _: &mut (),
            _: &mut Window,
            _: &mut crate::App,
        ) {
        }
    }

    struct Cached;

    impl Render for Cached {
        fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
            div().size_full().child("cached")
        }
    }

    /// A root holding an uncached worker view, a cached view and a list.
    struct Root {
        worker: Entity<Worker>,
        cached: Entity<Cached>,
        list: ListState,
    }

    impl Render for Root {
        fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
            div()
                .size_full()
                .child(self.worker.clone())
                .child(self.cached.clone().cached(
                    crate::StyleRefinement::default().w(px(50.)).h(px(50.)),
                ))
                .child(
                    list(self.list.clone(), |index, _, _| {
                        div().h(px(10.)).child(format!("item {index}")).into_any_element()
                    })
                    .h(px(100.)),
                )
        }
    }

    struct Fixture {
        window: WindowHandle<Root>,
        test_window: TestWindow,
        spin_in: Rc<Cell<SpinIn>>,
        worker: Entity<Worker>,
        cached: Entity<Cached>,
    }

    fn open_window(cx: &mut TestAppContext) -> Fixture {
        let spin_in = Rc::new(Cell::new(SpinIn::Nowhere));
        let window = cx.update(|cx| {
            cx.open_window(WindowOptions::default(), {
                let spin_in = spin_in.clone();
                move |_, cx| {
                    let worker = cx.new(|_| Worker { spin_in });
                    let cached = cx.new(|_| Cached);
                    cx.new(|_| Root {
                        worker,
                        cached,
                        list: ListState::new(5, ListAlignment::Top, px(0.)),
                    })
                }
            })
            .unwrap()
        });
        let (worker, cached) = window
            .read_with(cx, |root, _| (root.worker.clone(), root.cached.clone()))
            .unwrap();
        let test_window = cx.test_window(window.into());
        test_window.simulate_active_status_change(true);
        test_window.simulate_frame_request(RequestFrameOptions::default());
        Fixture {
            window,
            test_window,
            spin_in,
            worker,
            cached,
        }
    }

    impl Fixture {
        fn last_draw(&self, cx: &mut TestAppContext) -> FrameTiming {
            self.window
                .update(cx, |_, window, _| window.window_profiler.last_draw())
                .unwrap()
                .expect("a draw was recorded")
        }

        /// Notifies `entity` and draws the resulting frame.
        fn redraw<T: 'static>(&self, entity: &Entity<T>, cx: &mut TestAppContext) -> FrameTiming {
            cx.update(|cx| entity.update(cx, |_, cx| cx.notify()));
            self.test_window
                .simulate_frame_request(RequestFrameOptions::default());
            self.last_draw(cx)
        }
    }

    fn assert_parts_add_up(timing: &FrameTiming) {
        let breakdown = timing.breakdown;
        let parts = breakdown.request_layout()
            + breakdown.layout()
            + breakdown.prepaint()
            + breakdown.paint()
            + breakdown.other();
        let total = timing.draw_duration();
        assert!(
            parts <= total && total - parts <= Duration::from_micros(5),
            "the five parts ({parts:?}) add up to the draw ({total:?}): {breakdown:?}"
        );
    }

    #[gpui::test]
    fn a_draw_is_split_into_phases_that_add_up(cx: &mut TestAppContext) {
        let fixture = open_window(cx);
        let first = fixture.last_draw(cx);
        assert_parts_add_up(&first);
        let breakdown = first.breakdown;
        assert!(
            breakdown.layout_passes() >= 2,
            "the root and each list item and cache-missed view are laid out: {breakdown:?}"
        );
        assert_eq!(
            (breakdown.views_rendered(), breakdown.views_reused()),
            (3, 0),
            "the first draw renders the root, the worker and the cached view"
        );

        let worker_notified = fixture.redraw(&fixture.worker, cx);
        assert_parts_add_up(&worker_notified);
        assert_eq!(
            (
                worker_notified.breakdown.views_rendered(),
                worker_notified.breakdown.views_reused()
            ),
            (2, 1),
            "the root re-renders with its uncached worker, and the cached view is reused"
        );

        let cached_notified = fixture.redraw(&fixture.cached, cx);
        assert_eq!(
            (
                cached_notified.breakdown.views_rendered(),
                cached_notified.breakdown.views_reused()
            ),
            (3, 0),
            "notifying the cached view renders it and its ancestors"
        );
    }

    #[gpui::test]
    fn render_work_counts_as_request_layout_and_measure_work_as_layout(
        cx: &mut TestAppContext,
    ) {
        let fixture = open_window(cx);

        fixture.spin_in.set(SpinIn::Render);
        let rendering = fixture.redraw(&fixture.worker, cx);
        assert_parts_add_up(&rendering);
        let breakdown = rendering.breakdown;
        assert!(
            breakdown.request_layout() >= SPIN,
            "a slow render is charged to request_layout: {breakdown:?}"
        );
        assert!(breakdown.layout() < SPIN, "{breakdown:?}");

        fixture.spin_in.set(SpinIn::Measure);
        let measuring = fixture.redraw(&fixture.worker, cx);
        assert_parts_add_up(&measuring);
        let breakdown = measuring.breakdown;
        assert!(
            breakdown.layout() >= SPIN,
            "a slow measure closure is charged to layout: {breakdown:?}"
        );
        assert!(breakdown.request_layout() < SPIN, "{breakdown:?}");
        assert!(breakdown.prepaint() < SPIN, "{breakdown:?}");
    }
}
