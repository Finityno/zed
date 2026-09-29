//! Counts of the work a window does to draw its frames, for benchmarks and
//! embedders that need to tell a change that did less work from one that
//! only moved it around.

use super::Window;
use std::time::{Duration, Instant};

/// The work a window did drawing frames since
/// [`Window::reset_frame_work_stats`] was last called, or since it opened.
///
/// The counts are a few integer increments per frame, element, layout node,
/// measurement and shaped line, and are always kept. The durations take a
/// clock read per frame phase, layout computation, measurement and shaped
/// line, so they are kept only once [`Window::reset_frame_work_stats`] has
/// asked for them, and are zero until then.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FrameWorkStats {
    /// Frames drawn.
    pub frames: u64,
    /// Elements whose layout was requested: every element built.
    pub elements: u64,
    /// Views rendered, whether plain or cached ones that could not be reused.
    pub views_rendered: u64,
    /// Cached views whose previous frame's output was reused rather than
    /// rendered.
    pub views_reused: u64,
    /// Layout nodes requested from the layout engine.
    pub layout_nodes: u64,
    /// Layout nodes kept from an earlier frame and handed out again.
    pub layout_nodes_reused: u64,
    /// Layout nodes made anew, because no node was kept for the element or
    /// layout retention is off (`GPUI_RETAINED_LAYOUT=0`).
    pub layout_nodes_created: u64,
    /// Layout nodes released at the end of a frame: the ones made without a
    /// key, and the kept ones no element asked for.
    pub layout_nodes_released: u64,
    /// Styles written to kept layout nodes because they changed. Each write
    /// dirties the node and its ancestors.
    pub layout_style_writes: u64,
    /// Child lists written to kept layout nodes because they changed.
    pub layout_children_writes: u64,
    /// Kept self-measuring nodes given a new measurement and dirtied, so that
    /// the layout engine measures them again.
    pub measured_nodes_dirtied: u64,
    /// Text nodes whose element took last frame's measurement over, because
    /// it would shape its text the same way, leaving the node clean.
    pub measurements_carried: u64,
    /// Text nodes whose text changed but measured to every size the layout
    /// engine had taken of them, leaving the node clean.
    pub measurements_replayed: u64,
    /// Measurements taken to find those out, outside the layout engine, and
    /// not counted in [`Self::measure_calls`].
    pub replay_measure_calls: u64,
    /// Layout computations: the root's, and every one an element asked for
    /// on its own, such as a cached view or a list item.
    pub compute_layout_calls: u64,
    /// Times the layout engine asked a measured node for its size. A node is
    /// usually measured more than once per computation, for its intrinsic
    /// size and then for its final one.
    pub measure_calls: u64,
    /// Lines of text handed to the platform to shape, which the line layout
    /// cache held from neither this frame nor the one before.
    pub lines_shaped: u64,
    /// Time spent in the root's request-layout walk, which renders every view
    /// reached from the root and builds its elements.
    pub build_time: Duration,
    /// Time spent from the end of the build walk to the start of painting:
    /// computing layout, placing elements, and rendering the cached views
    /// that could not be reused, the deferred draws and the overlays.
    pub prepaint_time: Duration,
    /// Time spent painting the frame's elements into its scene.
    pub paint_time: Duration,
    /// Time spent computing layout, measurements included.
    pub compute_layout_time: Duration,
    /// Time spent in measurements.
    pub measure_time: Duration,
    /// Time spent shaping [`Self::lines_shaped`].
    pub shape_time: Duration,
}

/// What a window keeps to fill in [`FrameWorkStats`]; the shaping counts are
/// kept by its text system.
#[derive(Default)]
pub(crate) struct FrameWorkCounters {
    pub(crate) stats: FrameWorkStats,
    timed: bool,
}

impl FrameWorkCounters {
    /// Now, if durations are being kept.
    #[inline]
    pub(crate) fn clock(&self) -> Option<Instant> {
        self.timed.then(Instant::now)
    }
}

/// Adds the time since `started_at` to `total`, when there is a start.
#[inline]
pub(crate) fn add_elapsed(total: &mut Duration, started_at: Option<Instant>) {
    if let Some(started_at) = started_at {
        *total += started_at.elapsed();
    }
}

impl Window {
    /// The work this window did drawing frames since the last call to
    /// [`Self::reset_frame_work_stats`].
    pub fn frame_work_stats(&self) -> FrameWorkStats {
        let (lines_shaped, shape_time) = self.text_system().shaping_stats();
        let retention = self
            .layout_engine
            .as_ref()
            .map(|engine| engine.retention_counts())
            .unwrap_or_default();
        FrameWorkStats {
            lines_shaped,
            shape_time,
            layout_nodes_reused: retention.nodes_reused,
            layout_nodes_created: retention.nodes_created,
            layout_nodes_released: retention.nodes_released,
            layout_style_writes: retention.style_writes,
            layout_children_writes: retention.children_writes,
            measured_nodes_dirtied: retention.measured_nodes_dirtied,
            measurements_carried: retention.measurements_carried,
            measurements_replayed: retention.measurements_replayed,
            replay_measure_calls: retention.replay_measure_calls,
            ..self.frame_work.stats
        }
    }

    /// Zeroes the counts [`Self::frame_work_stats`] reports. With `timed`,
    /// the durations are kept from now on too.
    pub fn reset_frame_work_stats(&mut self, timed: bool) {
        self.frame_work = FrameWorkCounters {
            stats: FrameWorkStats::default(),
            timed,
        };
        self.text_system().reset_shaping_stats(timed);
        if let Some(engine) = self.layout_engine.as_mut() {
            engine.reset_retention_counts();
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        AppContext as _, Context, IntoElement, ParentElement as _, Render, Styled as _,
        TestAppContext, Window, div, px,
    };
    use std::time::Duration;

    struct Labels {
        count: usize,
    }

    impl Render for Labels {
        fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
            div()
                .flex()
                .flex_col()
                .w(px(300.))
                .children((0..self.count).map(|index| div().child(format!("label {index}"))))
        }
    }

    /// A frame counts the elements it built, the nodes it laid out and the
    /// lines it shaped; drawing the same frame again shapes nothing new; and
    /// durations are kept only once they are asked for.
    #[test]
    fn frames_report_the_work_they_did() {
        let mut cx = TestAppContext::single();
        let window = cx.add_window(|_, _| Labels { count: 5 });
        let draw = |cx: &mut TestAppContext| {
            cx.update_window(window.into(), |_, window, cx| {
                window.refresh();
                window.draw(cx).clear(cx);
                window.frame_work_stats()
            })
            .unwrap()
        };

        cx.update_window(window.into(), |_, window, _| {
            window.reset_frame_work_stats(false)
        })
        .unwrap();
        window
            .update(&mut cx, |labels, _, cx| {
                labels.count += 1;
                cx.notify();
            })
            .unwrap();
        // Notifying may draw the window once already, before this does.
        let first = draw(&mut cx);
        assert!(first.frames >= 1, "{first:?}");
        assert!(first.elements >= 13, "{first:?}");
        assert!(first.views_rendered >= 1, "{first:?}");
        assert!(first.layout_nodes >= 13, "{first:?}");
        assert!(first.measure_calls >= 1, "the new label is measured: {first:?}");
        assert!(
            first.measure_calls + first.measurements_carried >= 6,
            "every label is measured or carries its measurement: {first:?}"
        );
        assert!(first.compute_layout_calls >= 1, "{first:?}");
        assert_eq!(first.lines_shaped, 1, "only the new label is shaped: {first:?}");
        assert_eq!(first.build_time, Duration::ZERO);
        assert_eq!(first.shape_time, Duration::ZERO);

        cx.update_window(window.into(), |_, window, _| {
            window.reset_frame_work_stats(true)
        })
        .unwrap();
        let again = draw(&mut cx);
        assert_eq!(again.frames, 1);
        assert_eq!(again.elements * first.frames, first.elements);
        assert_eq!(again.lines_shaped, 0, "the line cache holds last frame's lines");
        assert!(
            again.build_time + again.prepaint_time + again.paint_time > Duration::ZERO,
            "{again:?}"
        );
    }
}
