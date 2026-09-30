//! Per-draw accounting behind [`crate::DrawBreakdown`]: where a window draw's
//! time went, phase by phase.
//!
//! The fast path is a handful of `Instant::now()` calls per draw, two per
//! taffy layout pass and, under the default sampling policy, one read of the
//! thread's CPU time (a system call); everything costlier runs only after a
//! quiet gap or for slow draws.

use std::time::Duration;

use scheduler::Instant;

use collections::FxHashMap;

use crate::{
    App, DRAW_QUIET_GAP, DrawBreakdown, DrawResourceSampling, DrawResources, EntityId,
    PlatformDispatcher, ResourceSample, SLOW_DRAW_VIEW_COUNT, SLOW_DRAW_VIEW_MIN, SlowDrawDetail,
    ViewRenderTime, ViewTiming, ViewTimingStart, Window, profiler, view::ViewName,
};

impl Window {
    /// Starts the profiler's record of a draw: the phase clock and, under the
    /// sampling policy, the thread's resource counters.
    pub(super) fn begin_draw_profile(&mut self, cx: &App) {
        let draw_start = self.window_profiler.begin_draw();
        let after_quiet_or_slow = self.draw_clock.previous_draw_slow
            || self.draw_clock.last_draw_end.is_none_or(|last_draw_end| {
                draw_start.saturating_duration_since(last_draw_end) >= DRAW_QUIET_GAP
            });
        let time_views = match profiler::view_timing() {
            ViewTiming::Off => false,
            ViewTiming::OnSlowDraws => self.draw_clock.previous_draw_slow,
            ViewTiming::Always => true,
        };
        self.draw_clock.begin(draw_start);
        self.draw_resources
            .begin(after_quiet_or_slow, cx.background_executor().dispatcher().as_ref());
        // Also disarms a timer a draw that unwound part way left armed.
        self.view_timer
            .reset(time_views.then_some(ViewTimingStart::Start));
    }

    /// Moves the draw clock to `phase`. Entering prepaint or paint may also
    /// start timing views part way through the draw (see
    /// [`Self::time_views_from_now_if_slow`]), reusing the mark's timestamp.
    pub(super) fn mark_draw_phase(&mut self, phase: DrawClockPhase) {
        let now = self.draw_clock.mark(phase);
        let timed_from = match phase {
            DrawClockPhase::Prepaint => ViewTimingStart::Prepaint,
            DrawClockPhase::Paint => ViewTimingStart::Paint,
            DrawClockPhase::Other
            | DrawClockPhase::RequestLayout
            | DrawClockPhase::Finish
            | DrawClockPhase::Focus => return,
        };
        self.time_views_from_now_if_slow(now, timed_from);
    }

    /// Ends a taffy layout pass [`DrawClock::begin_layout`] started. A slow
    /// draw that did not start out timing its views starts here, part way
    /// through prepaint, where lists and cache-missed views render: a
    /// single slow item then leaves the rest of prepaint timed rather than
    /// only paint. The check reuses the pass's end timestamp.
    #[inline]
    pub(super) fn end_draw_layout(&mut self, started_at: Option<Instant>) {
        if let Some(now) = self.draw_clock.end_layout(started_at) {
            self.time_views_from_now_if_slow(now, ViewTimingStart::Layout);
        }
    }

    /// Under [`ViewTiming::OnSlowDraws`], starts timing views once the draw
    /// has run for half the detail threshold. The views being drawn at that
    /// moment are timed from `now` on, so the view whose prepaint turned
    /// the draw slow is still charged for the rest of it.
    #[inline]
    fn time_views_from_now_if_slow(&mut self, now: Instant, timed_from: ViewTimingStart) {
        if !self.view_timer.is_armed()
            && self.draw_clock.active
            && self.draw_clock.elapsed(now) >= profiler::draw_detail_threshold() / 2
            && profiler::view_timing() == ViewTiming::OnSlowDraws
        {
            self.view_timer
                .arm_partway(timed_from, now, self.rendered_entity_stack.len());
        }
    }

    /// A view with an identity is about to call `render`. Recorded where
    /// the view element calls it, so hand-written [`crate::View`]s count
    /// as well as entities.
    #[inline]
    pub(crate) fn record_view_render(&mut self, entity_id: EntityId, view_name: ViewName) {
        self.draw_clock.count_render();
        if self.view_timer.is_armed() {
            self.view_timer.record_render(entity_id, view_name.0);
        }
    }

    /// Ends the profiler's record of a draw and returns its duration. A draw
    /// that took at least [`profiler::draw_detail_threshold`] also records
    /// what it cost the thread and, when they were timed, its slowest views.
    pub(super) fn end_draw_profile(
        &mut self,
        dirty_at: Option<Instant>,
        invalidations: u64,
        cx: &App,
    ) -> Duration {
        let now = Instant::now();
        let duration = self.draw_clock.elapsed(now);
        let slow = duration >= profiler::draw_detail_threshold();
        let resources = self
            .draw_resources
            .finish(slow, cx.background_executor().dispatcher().as_ref());
        let draw_start = self.draw_clock.draw_start;
        let mut breakdown = self.draw_clock.finish(now, slow);
        let views_timed_from = self.view_timer.finish();
        if slow && (resources.is_some() || views_timed_from.is_some()) {
            let recorded = profiler::record_slow_draw_detail(SlowDrawDetail {
                window_id: self.handle.window_id(),
                draw_start,
                duration,
                resources: resources.unwrap_or_default(),
                views_timed_from,
                views: self.view_timer.slowest_views(),
            });
            if recorded {
                breakdown.flags |= DrawBreakdown::DETAIL_RECORDED;
            }
        }
        if views_timed_from.is_some() {
            self.view_timer.clear();
        }
        self.window_profiler
            .end_draw(dirty_at, invalidations, breakdown)
    }
}

/// The draw phase the clock is currently charging.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub(crate) enum DrawClockPhase {
    /// Setup before the tree is built, and the bookkeeping after focus
    /// listeners. Not accumulated: the breakdown derives it as the
    /// remainder.
    Other,
    RequestLayout,
    Prepaint,
    Paint,
    Finish,
    Focus,
}

/// Charges elapsed draw time to phases. Taffy layout passes are carved out
/// of whichever phase they run in (see [`Self::begin_layout`]).
pub(crate) struct DrawClock {
    active: bool,
    draw_start: Instant,
    /// When the window's previous draw ended; kept across draws.
    last_draw_end: Option<Instant>,
    /// Whether the window's previous draw reached the detail threshold;
    /// kept across draws.
    previous_draw_slow: bool,
    phase: DrawClockPhase,
    phase_started_at: Instant,
    request_layout: Duration,
    layout: Duration,
    prepaint: Duration,
    paint: Duration,
    finish: Duration,
    focus: Duration,
    layout_passes: u16,
    views_rendered: u16,
    views_reused: u16,
}

impl DrawClock {
    pub(crate) fn new() -> Self {
        let now = Instant::now();
        Self {
            active: false,
            draw_start: now,
            last_draw_end: None,
            previous_draw_slow: false,
            phase: DrawClockPhase::Other,
            phase_started_at: now,
            request_layout: Duration::ZERO,
            layout: Duration::ZERO,
            prepaint: Duration::ZERO,
            paint: Duration::ZERO,
            finish: Duration::ZERO,
            focus: Duration::ZERO,
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
            draw_start,
            last_draw_end: self.last_draw_end,
            previous_draw_slow: self.previous_draw_slow,
            phase: DrawClockPhase::Other,
            phase_started_at: draw_start,
            request_layout: Duration::ZERO,
            layout: Duration::ZERO,
            prepaint: Duration::ZERO,
            paint: Duration::ZERO,
            finish: Duration::ZERO,
            focus: Duration::ZERO,
            layout_passes: 0,
            views_rendered: 0,
            views_reused: 0,
        };
    }

    /// How long the current draw has run.
    pub(crate) fn elapsed(&self, now: Instant) -> Duration {
        now.saturating_duration_since(self.draw_start)
    }

    fn charge(&mut self, until: Instant) {
        let elapsed = until.saturating_duration_since(self.phase_started_at);
        match self.phase {
            DrawClockPhase::Other => {}
            DrawClockPhase::RequestLayout => self.request_layout += elapsed,
            DrawClockPhase::Prepaint => self.prepaint += elapsed,
            DrawClockPhase::Paint => self.paint += elapsed,
            DrawClockPhase::Finish => self.finish += elapsed,
            DrawClockPhase::Focus => self.focus += elapsed,
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
    /// result. Returns the pass's end inside a profiled draw.
    pub(crate) fn end_layout(&mut self, started_at: Option<Instant>) -> Option<Instant> {
        let started_at = started_at?;
        let now = Instant::now();
        self.layout += now.saturating_duration_since(started_at);
        self.layout_passes = self.layout_passes.saturating_add(1);
        self.phase_started_at = now;
        Some(now)
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
    /// `slow` records whether the draw reached the detail threshold, for
    /// the next draw's sampling decisions.
    pub(crate) fn finish(&mut self, now: Instant, slow: bool) -> DrawBreakdown {
        let measured = self.active;
        if measured {
            self.charge(now);
        }
        self.active = false;
        self.last_draw_end = Some(now);
        self.previous_draw_slow = slow;
        fn micros(duration: Duration) -> u32 {
            duration.as_micros().min(u32::MAX as u128) as u32
        }
        DrawBreakdown {
            request_layout_us: micros(self.request_layout),
            layout_us: micros(self.layout),
            prepaint_us: micros(self.prepaint),
            paint_us: micros(self.paint),
            finish_us: micros(self.finish),
            focus_us: micros(self.focus),
            layout_passes: self.layout_passes,
            views_rendered: self.views_rendered,
            views_reused: self.views_reused,
            flags: if measured {
                DrawBreakdown::PHASES_MEASURED
            } else {
                0
            },
            ..DrawBreakdown::default()
        }
    }
}

/// Times each view's `render`, prepaint and paint, net of the views nested
/// inside it, for the draws [`crate::ViewTiming`] selects.
pub(crate) struct ViewTimer {
    armed: Option<ViewTimingStart>,
    stack: Vec<TimedView>,
    views: FxHashMap<EntityId, ViewTotals>,
}

struct TimedView {
    started_at: Instant,
    nested: Duration,
}

/// One view's share of a timed draw so far.
#[derive(Default)]
struct ViewTotals {
    self_time: Duration,
    /// Set by any of the view's phases, not only its render: a view timed
    /// from part way through a draw, or a cached view replaying its last
    /// prepaint, is still named.
    type_name: Option<&'static str>,
    renders: u16,
}

impl ViewTimer {
    pub(crate) fn new() -> Self {
        Self {
            armed: None,
            stack: Vec::new(),
            views: FxHashMap::default(),
        }
    }

    /// Starts a draw: timing from its start when `armed` is set, and
    /// otherwise disarmed with nothing left over from an earlier draw.
    fn reset(&mut self, armed: Option<ViewTimingStart>) {
        self.armed = armed;
        self.stack.clear();
        if !self.views.is_empty() {
            self.clear();
        }
    }

    /// Starts timing part way through a draw, at `now`, while `depth` views
    /// are being drawn. Those views are timed from `now` on, so the timer's
    /// stack mirrors the window's stack of views being drawn, and every
    /// [`Self::exit`] from here on pairs with an entry.
    fn arm_partway(&mut self, from: ViewTimingStart, now: Instant, depth: usize) {
        self.armed = Some(from);
        self.stack.clear();
        self.stack.extend((0..depth).map(|_| TimedView {
            started_at: now,
            nested: Duration::ZERO,
        }));
    }

    #[inline]
    pub(crate) fn is_armed(&self) -> bool {
        self.armed.is_some()
    }

    /// A view is being entered. While timing, every view entered is left
    /// through [`Self::exit`], and a timer that starts mid-view mirrors the
    /// views already entered, so enters and exits always pair.
    #[inline]
    pub(crate) fn enter(&mut self) {
        if self.armed.is_some() {
            self.stack.push(TimedView {
                started_at: Instant::now(),
                nested: Duration::ZERO,
            });
        }
    }

    /// The view most recently entered is being left. `view_name` is its
    /// type's name where the caller knows it.
    #[inline]
    pub(crate) fn exit(&mut self, entity_id: EntityId, view_name: ViewName) {
        if self.armed.is_some() {
            self.exit_timed(entity_id, view_name.0);
        }
    }

    fn exit_timed(&mut self, entity_id: EntityId, type_name: Option<&'static str>) {
        let Some(view) = self.stack.pop() else {
            debug_assert!(false, "a timed view was left that was never entered");
            return;
        };
        let total = view.started_at.elapsed();
        if let Some(parent) = self.stack.last_mut() {
            parent.nested += total;
        }
        let totals = self.views.entry(entity_id).or_default();
        totals.self_time += total.saturating_sub(view.nested);
        if type_name.is_some() {
            totals.type_name = type_name;
        }
    }

    fn record_render(&mut self, entity_id: EntityId, type_name: Option<&'static str>) {
        let totals = self.views.entry(entity_id).or_default();
        if type_name.is_some() {
            totals.type_name = type_name;
        }
        totals.renders = totals.renders.saturating_add(1);
    }

    /// Stops timing, returning where the draw's timing started, if it was
    /// timed. The times stay readable until [`Self::clear`].
    fn finish(&mut self) -> Option<ViewTimingStart> {
        self.armed.take()
    }

    /// The views with the most time of their own, longest first.
    fn slowest_views(&self) -> heapless::Vec<ViewRenderTime, SLOW_DRAW_VIEW_COUNT> {
        let mut slowest = heapless::Vec::<ViewRenderTime, SLOW_DRAW_VIEW_COUNT>::new();
        for totals in self.views.values() {
            if totals.self_time < SLOW_DRAW_VIEW_MIN {
                continue;
            }
            let view = ViewRenderTime {
                type_name: totals.type_name.unwrap_or("<unnamed view>"),
                self_time: totals.self_time,
                renders: totals.renders,
            };
            if let Err(view) = slowest.push(view)
                && let Some(fastest) = slowest.iter_mut().min_by_key(|view| view.self_time)
                && fastest.self_time < view.self_time
            {
                *fastest = view;
            }
        }
        slowest.sort_unstable_by_key(|view| std::cmp::Reverse(view.self_time));
        slowest
    }

    /// Forgets a timed draw's times, keeping the maps' capacity for the
    /// next one.
    fn clear(&mut self) {
        self.stack.clear();
        self.views.clear();
    }
}

/// Reads the thread's resource counters at the start of a draw, and again
/// at the end of a slow one.
pub(crate) struct DrawResourceSampler {
    start: Option<ResourceSample>,
    process_counters: bool,
}

impl DrawResourceSampler {
    pub(crate) fn new() -> Self {
        Self {
            start: None,
            process_counters: false,
        }
    }

    /// Takes the start sample under the sampling policy. `after_quiet_or_slow`
    /// says whether the window's last draw ended at least
    /// [`DRAW_QUIET_GAP`] ago or was slow.
    fn begin(&mut self, after_quiet_or_slow: bool, dispatcher: &dyn PlatformDispatcher) {
        self.process_counters = match profiler::draw_resource_sampling() {
            DrawResourceSampling::Off => {
                self.start = None;
                return;
            }
            DrawResourceSampling::ThreadCpu => false,
            DrawResourceSampling::AfterQuiet => after_quiet_or_slow,
            DrawResourceSampling::Always => true,
        };
        self.start = dispatcher.sample_draw_resources(self.process_counters);
    }

    /// For a slow draw with a start sample, takes the end sample and returns
    /// the difference. A fast draw costs nothing here.
    fn finish(
        &mut self,
        slow: bool,
        dispatcher: &dyn PlatformDispatcher,
    ) -> Option<DrawResources> {
        let start = self.start.take()?;
        if !slow {
            return None;
        }
        let end = dispatcher.sample_draw_resources(self.process_counters)?;
        Some(DrawResources::between(&start, &end))
    }
}

#[cfg(test)]
mod tests {
    use std::{cell::Cell, rc::Rc, time::Duration};

    use scheduler::Instant;

    use super::ViewTimer;
    use crate::{
        AppContext as _, Context, DRAW_QUIET_GAP, DrawResourceSampling, DrawResources, Entity,
        FaultScope, FocusHandle, FrameTiming, InteractiveElement as _, IntoElement, ListAlignment,
        ListState, ParentElement as _, Render, RequestFrameOptions, ResourceSample,
        SLOW_DRAW_VIEW_MIN, SlowDrawDetail, Style, Styled as _, TestAppContext, TestWindow,
        ViewTiming, ViewTimingStart, Window, WindowHandle, WindowId, WindowOptions, div, list,
        profiler, px, size, view::ViewName,
    };

    const SPIN: Duration = Duration::from_millis(3);

    /// Serializes the tests that change the process-wide draw profiling
    /// knobs, and restores the defaults when dropped.
    struct Knobs {
        _lock: std::sync::MutexGuard<'static, ()>,
    }

    static KNOBS_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    impl Knobs {
        fn set(detail_threshold: Duration, sampling: DrawResourceSampling) -> Self {
            let lock = KNOBS_LOCK
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            profiler::set_draw_detail_threshold(detail_threshold);
            profiler::set_draw_resource_sampling(sampling);
            Self { _lock: lock }
        }

        /// Holds the knobs at their defaults, for tests whose draws another
        /// test's settings would otherwise change.
        fn defaults() -> Self {
            let knobs = Self::set(Duration::from_millis(8), DrawResourceSampling::AfterQuiet);
            profiler::set_view_timing(ViewTiming::OnSlowDraws);
            knobs
        }
    }

    impl Drop for Knobs {
        fn drop(&mut self) {
            profiler::set_draw_detail_threshold(Duration::from_millis(8));
            profiler::set_draw_resource_sampling(DrawResourceSampling::AfterQuiet);
            profiler::set_view_timing(ViewTiming::OnSlowDraws);
        }
    }

    const EVERY_DRAW_IS_SLOW: Duration = Duration::ZERO;
    const NO_DRAW_IS_SLOW: Duration = Duration::from_secs(3600);

    fn sample(user_ms: u64, system_ms: u64, faults: u64) -> ResourceSample {
        ResourceSample {
            user: Duration::from_millis(user_ms),
            system: Duration::from_millis(system_ms),
            faults: Some(faults),
            major_faults: Some(faults / 100),
            decompressions: Some(faults / 2),
            fault_scope: FaultScope::Process,
        }
    }

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
        RenderAndPaint,
    }

    /// Renders nothing but a leaf that is measured by a closure, spinning
    /// in its render, its measure closure or its paint when asked to.
    struct Worker {
        spin_in: Rc<Cell<SpinIn>>,
    }

    impl Render for Worker {
        fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
            if matches!(self.spin_in.get(), SpinIn::Render | SpinIn::RenderAndPaint) {
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
            if self.spin_in.get() == SpinIn::RenderAndPaint {
                spin(SPIN);
            }
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
        assert!(breakdown.phases_measured(), "{breakdown:?}");
        let parts = breakdown.request_layout()
            + breakdown.layout()
            + breakdown.prepaint()
            + breakdown.paint()
            + breakdown.finish()
            + breakdown.focus()
            + breakdown.other();
        let total = timing.draw_duration();
        assert!(
            parts <= total && total - parts <= Duration::from_micros(5),
            "the seven parts ({parts:?}) add up to the draw ({total:?}): {breakdown:?}"
        );
    }

    #[gpui::test]
    fn a_draw_is_split_into_phases_that_add_up(cx: &mut TestAppContext) {
        let _knobs = Knobs::defaults();
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
        let _knobs = Knobs::defaults();
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

    #[gpui::test]
    fn a_draw_that_skips_drawing_builds_no_tree(cx: &mut TestAppContext) {
        let _knobs = Knobs::defaults();
        cx.skip_drawing();
        let fixture = open_window(cx);
        let skipped = fixture.redraw(&fixture.worker, cx);
        assert_parts_add_up(&skipped);
        let breakdown = skipped.breakdown;
        assert_eq!(
            [
                breakdown.request_layout(),
                breakdown.layout(),
                breakdown.prepaint(),
                breakdown.paint(),
            ],
            [Duration::ZERO; 4],
            "{breakdown:?}"
        );
        assert_eq!(
            (breakdown.layout_passes(), breakdown.views_rendered()),
            (0, 0),
            "{breakdown:?}"
        );
    }

    struct Focusable {
        handle: FocusHandle,
    }

    impl Render for Focusable {
        fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
            div().track_focus(&self.handle).size_full()
        }
    }

    #[gpui::test]
    fn focus_listeners_count_as_focus(cx: &mut TestAppContext) {
        let _knobs = Knobs::defaults();
        let window = cx.update(|cx| {
            cx.open_window(WindowOptions::default(), |_, cx| {
                cx.new(|cx| Focusable {
                    handle: cx.focus_handle(),
                })
            })
            .unwrap()
        });
        let test_window = cx.test_window(window.into());
        test_window.simulate_active_status_change(true);
        test_window.simulate_frame_request(RequestFrameOptions::default());
        let _subscription = window
            .update(cx, |root, window, cx| {
                let subscription = window.on_focus_in(&root.handle, cx, |_, _| spin(SPIN));
                window.focus(&root.handle, cx);
                subscription
            })
            .unwrap();
        test_window.simulate_frame_request(RequestFrameOptions::default());
        let focusing = window
            .update(cx, |_, window, _| window.window_profiler.last_draw())
            .unwrap()
            .expect("a draw was recorded");
        assert_parts_add_up(&focusing);
        let breakdown = focusing.breakdown;
        assert!(
            breakdown.focus() >= SPIN,
            "a slow focus listener is charged to focus: {breakdown:?}"
        );
        assert!(breakdown.other() < SPIN, "{breakdown:?}");
        assert!(breakdown.finish() < SPIN, "{breakdown:?}");
    }

    #[gpui::test]
    fn a_slow_draw_records_what_it_cost_the_thread(cx: &mut TestAppContext) {
        let _knobs = Knobs::set(EVERY_DRAW_IS_SLOW, DrawResourceSampling::Always);
        let fixture = open_window(cx);
        cx.dispatcher.take_draw_resource_requests();

        cx.dispatcher
            .script_draw_resource_samples([sample(1, 2, 1000), sample(4, 7, 5100)]);
        let measured = fixture.redraw(&fixture.worker, cx);
        let resources = fixture.resources(&measured);
        assert_eq!(cx.dispatcher.take_draw_resource_requests(), [true, true]);
        assert_eq!(resources.user_cpu(), Some(Duration::from_millis(3)));
        assert_eq!(resources.system_cpu(), Some(Duration::from_millis(5)));
        assert_eq!(resources.faults(), Some(4100));
        assert_eq!(resources.major_faults(), Some(41));
        assert_eq!(resources.decompressions(), Some(2050));
        assert!(resources.faults_process_wide());

        let unmeasured = fixture.redraw(&fixture.worker, cx);
        let unmeasured = fixture.resources(&unmeasured);
        assert_eq!(
            unmeasured,
            Default::default(),
            "a platform that returns no sample leaves every counter unmeasured"
        );
        assert_eq!(unmeasured.user_cpu(), None);
        assert_eq!(unmeasured.faults(), None);
    }

    #[gpui::test]
    fn a_fast_draw_takes_no_end_sample(cx: &mut TestAppContext) {
        let _knobs = Knobs::set(NO_DRAW_IS_SLOW, DrawResourceSampling::ThreadCpu);
        let fixture = open_window(cx);
        cx.dispatcher.take_draw_resource_requests();

        cx.dispatcher
            .script_draw_resource_samples([sample(1, 2, 1000), sample(4, 7, 5100)]);
        let fast = fixture.redraw(&fixture.worker, cx);
        assert_eq!(
            cx.dispatcher.take_draw_resource_requests(),
            [false],
            "only the thread-CPU start sample is read"
        );
        assert_eq!(
            cx.dispatcher.take_unread_draw_resource_samples(),
            [sample(4, 7, 5100)],
            "the end sample is left unread"
        );
        assert_eq!(fixture.detail(&fast), None, "and nothing is recorded");
    }

    #[gpui::test]
    fn after_quiet_reads_process_counters_after_a_gap_or_a_slow_draw(cx: &mut TestAppContext) {
        let knobs = Knobs::set(NO_DRAW_IS_SLOW, DrawResourceSampling::AfterQuiet);
        let fixture = open_window(cx);
        cx.dispatcher.take_draw_resource_requests();
        cx.dispatcher
            .script_draw_resource_samples(std::iter::repeat_n(sample(1, 1, 1), 16));

        fixture.redraw(&fixture.worker, cx);
        assert_eq!(
            cx.dispatcher.take_draw_resource_requests(),
            [false],
            "a draw right after another reads the thread's CPU time only"
        );

        std::thread::sleep(DRAW_QUIET_GAP + Duration::from_millis(10));
        fixture.redraw(&fixture.worker, cx);
        assert_eq!(
            cx.dispatcher.take_draw_resource_requests(),
            [true],
            "the first draw after a quiet gap reads the process counters"
        );

        profiler::set_draw_detail_threshold(EVERY_DRAW_IS_SLOW);
        fixture.redraw(&fixture.worker, cx);
        assert_eq!(
            cx.dispatcher.take_draw_resource_requests(),
            [false, false],
            "a slow draw right after another reads its end sample like its start"
        );
        profiler::set_draw_detail_threshold(NO_DRAW_IS_SLOW);
        fixture.redraw(&fixture.worker, cx);
        assert_eq!(
            cx.dispatcher.take_draw_resource_requests(),
            [true],
            "the draw after a slow one reads the process counters"
        );

        profiler::set_draw_resource_sampling(DrawResourceSampling::Off);
        fixture.redraw(&fixture.worker, cx);
        assert_eq!(cx.dispatcher.take_draw_resource_requests(), [false; 0]);
        drop(knobs);
    }

    fn recorded_detail(window: WindowId, timing: &FrameTiming) -> Option<SlowDrawDetail> {
        let detail = profiler::slow_draw_detail(window, timing.draw_start);
        assert_eq!(
            detail.is_some(),
            timing.breakdown.detail_recorded(),
            "the breakdown's flag matches the recorded detail"
        );
        detail
    }

    impl Fixture {
        fn detail(&self, timing: &FrameTiming) -> Option<SlowDrawDetail> {
            recorded_detail(self.window.window_id(), timing)
        }

        fn resources(&self, timing: &FrameTiming) -> DrawResources {
            self.detail(timing)
                .map(|detail| detail.resources)
                .unwrap_or_default()
        }

        fn views_timed_from(&self, timing: &FrameTiming) -> Option<ViewTimingStart> {
            self.detail(timing)
                .and_then(|detail| detail.views_timed_from)
        }
    }

    #[gpui::test]
    fn a_slow_draw_names_its_slowest_view(cx: &mut TestAppContext) {
        let _knobs = Knobs::set(Duration::from_millis(1), DrawResourceSampling::Off);
        profiler::set_view_timing(ViewTiming::Always);
        let fixture = open_window(cx);

        fixture.spin_in.set(SpinIn::Render);
        let slow = fixture.redraw(&fixture.worker, cx);
        let recorded = fixture.detail(&slow).expect("the slow draw's views");
        assert_eq!(recorded.views_timed_from, Some(ViewTimingStart::Start));
        let slowest = recorded.views.first().expect("a view spent 3 ms");
        assert!(
            slowest.type_name.ends_with("::Worker"),
            "the spinning view ranks first: {recorded:?}"
        );
        assert!(slowest.self_time >= SPIN);
        assert_eq!(slowest.renders, 1);
        assert!(
            recorded
                .views
                .iter()
                .all(|view| !view.type_name.ends_with("::Root")),
            "the root's own time excludes the worker nested in it: {recorded:?}"
        );
    }

    #[gpui::test]
    fn on_slow_draws_times_views_after_a_slow_draw_or_partway_through_one(
        cx: &mut TestAppContext,
    ) {
        let _knobs = Knobs::set(NO_DRAW_IS_SLOW, DrawResourceSampling::Off);
        profiler::set_view_timing(ViewTiming::OnSlowDraws);
        let fixture = open_window(cx);

        let fast = fixture.redraw(&fixture.worker, cx);
        assert_eq!(fixture.detail(&fast), None, "a fast draw records nothing");

        profiler::set_draw_detail_threshold(Duration::from_millis(1));
        fixture.spin_in.set(SpinIn::Render);
        let first_slow = fixture.redraw(&fixture.worker, cx);
        let first_slow = fixture.detail(&first_slow).expect("the slow draw's detail");
        assert_eq!(
            first_slow.views_timed_from,
            Some(ViewTimingStart::Prepaint),
            "a draw that ran long while building the tree is timed from prepaint"
        );
        assert_eq!(
            first_slow.views.len(),
            0,
            "which misses the render that made it slow"
        );

        let repeat = fixture.redraw(&fixture.worker, cx);
        assert_eq!(
            fixture.views_timed_from(&repeat),
            Some(ViewTimingStart::Start),
            "the draw after a slow one is timed from its start"
        );
        let recorded = fixture.detail(&repeat).expect("the repeat's views");
        assert!(recorded.views[0].type_name.ends_with("::Worker"), "{recorded:?}");

        fixture.spin_in.set(SpinIn::Nowhere);
        profiler::set_draw_detail_threshold(NO_DRAW_IS_SLOW);
        fixture.redraw(&fixture.worker, cx);
        let after = fixture.redraw(&fixture.worker, cx);
        assert_eq!(
            fixture.views_timed_from(&after),
            None,
            "once draws are fast again nothing is timed"
        );
    }

    #[gpui::test]
    fn a_view_that_rendered_before_timing_started_is_named_by_its_paint(cx: &mut TestAppContext) {
        let _knobs = Knobs::set(NO_DRAW_IS_SLOW, DrawResourceSampling::Off);
        profiler::set_view_timing(ViewTiming::OnSlowDraws);
        let fixture = open_window(cx);
        fixture.redraw(&fixture.worker, cx);

        profiler::set_draw_detail_threshold(Duration::from_millis(1));
        fixture.spin_in.set(SpinIn::RenderAndPaint);
        let slow = fixture.redraw(&fixture.worker, cx);
        let recorded = fixture.detail(&slow).expect("the slow paint's view");
        assert_eq!(recorded.views_timed_from, Some(ViewTimingStart::Prepaint));
        let slowest = recorded.views.first().expect("a view spent 3 ms painting");
        assert!(
            slowest.type_name.ends_with("::Worker") && slowest.self_time >= SPIN,
            "{recorded:?}"
        );
        assert_eq!(slowest.renders, 0, "its render was not timed");
    }

    #[test]
    fn a_timed_view_is_named_by_whichever_phase_names_it() {
        let mut timer = ViewTimer::new();
        timer.reset(Some(ViewTimingStart::Start));
        let replayed = crate::EntityId::from(1u64);
        let unnamed = crate::EntityId::from(2u64);
        for (entity_id, view_name) in [
            (replayed, ViewName(Some("app::Replayed"))),
            (unnamed, ViewName::default()),
        ] {
            timer.enter();
            spin(SLOW_DRAW_VIEW_MIN);
            timer.exit(entity_id, view_name);
        }
        let mut names = timer
            .slowest_views()
            .iter()
            .map(|view| (view.type_name, view.renders))
            .collect::<Vec<_>>();
        names.sort_unstable();
        assert_eq!(names, [("<unnamed view>", 0), ("app::Replayed", 0)]);
    }

    const ROW_SPIN: Duration = Duration::from_millis(6);

    /// A list row that spins in its render when asked to.
    struct Row {
        spin: Rc<Cell<bool>>,
    }

    impl Render for Row {
        fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
            if self.spin.get() {
                spin(ROW_SPIN);
            }
            div().h(px(10.)).child("row")
        }
    }

    /// A list of rows, rendered and laid out one by one during prepaint.
    struct Rows {
        rows: Vec<Entity<Row>>,
        list: ListState,
    }

    impl Render for Rows {
        fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
            let rows = self.rows.clone();
            div().size_full().child(
                list(self.list.clone(), move |index, _, _| {
                    rows[index].clone().into_any_element()
                })
                .h(px(100.)),
            )
        }
    }

    #[gpui::test]
    fn a_draw_slowed_by_a_list_row_times_the_rest_of_prepaint(cx: &mut TestAppContext) {
        let _knobs = Knobs::set(NO_DRAW_IS_SLOW, DrawResourceSampling::Off);
        profiler::set_view_timing(ViewTiming::OnSlowDraws);
        let spin_rows = Rc::new(Cell::new(false));
        let window = cx.update(|cx| {
            cx.open_window(WindowOptions::default(), {
                let spin_rows = spin_rows.clone();
                move |_, cx| {
                    let rows = (0..3)
                        .map(|_| {
                            let spin = spin_rows.clone();
                            cx.new(|_| Row { spin })
                        })
                        .collect();
                    cx.new(|_| Rows {
                        rows,
                        list: ListState::new(3, ListAlignment::Top, px(0.)),
                    })
                }
            })
            .unwrap()
        });
        let root = window.entity(cx).unwrap();
        let test_window = cx.test_window(window.into());
        test_window.simulate_active_status_change(true);
        let draw = |cx: &mut TestAppContext| {
            cx.update(|cx| root.update(cx, |_, cx| cx.notify()));
            test_window.simulate_frame_request(RequestFrameOptions::default());
            window
                .update(cx, |_, window, _| window.window_profiler.last_draw())
                .unwrap()
                .expect("a draw was recorded")
        };
        draw(cx);
        draw(cx);

        // Half the threshold is under one row's render, and far more than
        // the draw takes to reach prepaint.
        profiler::set_draw_detail_threshold(ROW_SPIN + ROW_SPIN / 2);
        spin_rows.set(true);
        let slow = draw(cx);
        let recorded = recorded_detail(window.window_id(), &slow).expect("the slow draw's views");
        assert_eq!(
            recorded.views_timed_from,
            Some(ViewTimingStart::Layout),
            "timing starts after the first row's layout pass: {recorded:?}"
        );
        let slowest = recorded.views.first().expect("a row spent 6 ms");
        assert!(
            slowest.type_name.ends_with("::Row") && slowest.self_time >= ROW_SPIN,
            "the rows rendered after timing started are named: {recorded:?}"
        );
        assert_eq!(
            recorded
                .views
                .iter()
                .filter(|view| view.type_name.ends_with("::Row"))
                .count(),
            2,
            "the first row rendered before timing started: {recorded:?}"
        );
    }

    /// A hand-written view: props from its parent, identity from an entity.
    struct Handwritten {
        identity: Entity<Cached>,
    }

    impl crate::View for Handwritten {
        fn entity_id(&self) -> Option<crate::EntityId> {
            Some(self.identity.entity_id())
        }

        fn render(self, _: &mut Window, _: &mut crate::App) -> impl IntoElement {
            spin(SPIN);
            div().w(px(10.)).h(px(10.))
        }
    }

    struct HandwrittenHost {
        identity: Entity<Cached>,
    }

    impl Render for HandwrittenHost {
        fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
            div().size_full().child(crate::ViewElement::new(Handwritten {
                identity: self.identity.clone(),
            }))
        }
    }

    #[gpui::test]
    fn a_hand_written_view_is_counted_and_named(cx: &mut TestAppContext) {
        let _knobs = Knobs::set(Duration::from_millis(1), DrawResourceSampling::Off);
        profiler::set_view_timing(ViewTiming::Always);
        let window = cx.update(|cx| {
            cx.open_window(WindowOptions::default(), |_, cx| {
                let identity = cx.new(|_| Cached);
                cx.new(|_| HandwrittenHost { identity })
            })
            .unwrap()
        });
        let test_window = cx.test_window(window.into());
        test_window.simulate_active_status_change(true);
        test_window.simulate_frame_request(RequestFrameOptions::default());
        let draw = window
            .update(cx, |_, window, _| window.window_profiler.last_draw())
            .unwrap()
            .expect("a draw was recorded");
        assert_eq!(
            draw.breakdown.views_rendered(),
            2,
            "the host and the hand-written view render: {:?}",
            draw.breakdown
        );
        let recorded = recorded_detail(window.window_id(), &draw).expect("the slow draw's views");
        let slowest = recorded.views.first().expect("a view spent 3 ms");
        assert!(
            slowest.type_name.ends_with("::Handwritten") && slowest.renders == 1,
            "{recorded:?}"
        );
    }
}
