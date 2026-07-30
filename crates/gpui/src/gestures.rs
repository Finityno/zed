//! Touch gesture recognition vocabulary.
//!
//! GPUI recognizes gestures from raw [`TouchEvent`](crate::TouchEvent)s in a
//! single, portable arena in gpui core: recognizers compete for in-flight
//! touches, winners claim them, and losers are cancelled. Recognized gestures
//! are surfaced through *existing* semantic events wherever possible, a tap
//! becomes [`ClickEvent::Touch`](crate::ClickEvent), a pan becomes
//! [`ScrollWheelEvent`](crate::ScrollWheelEvent)s carrying a
//! [`TouchPhase`](crate::TouchPhase), and a pinch becomes
//! [`PinchEvent`](crate::PinchEvent)s — so components written against
//! `on_click` and scroll containers work untouched on mobile.

use std::time::{Duration, Instant};

use crate::{Axis, IsZero, Pixels, Point, TouchPhase, px};

/// How eagerly a scroll gesture commits to one axis, and how hard the user has to push to
/// get it back off that axis.
///
/// Surfaces differ enough here that one set of numbers does not serve all of them. A small
/// horizontally-scrolling region embedded in a long vertical document has to pick up sideways
/// intent almost immediately or it feels stuck, while a viewer that fills the pane must not let
/// an incidental sideways drift derail vertical reading. Pick the tuning per scroll container
/// with [`Styled::scroll_axis_lock`](crate::Styled::scroll_axis_lock); users notice changes to
/// these numbers, so prefer a named constant over an inline literal.
#[derive(
    Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize, schemars::JsonSchema,
)]
pub struct ScrollAxisLock {
    /// Gap after which an event begins a new gesture instead of continuing the current one.
    ///
    /// Only a fallback: platforms that report [`TouchPhase`] delimit gestures exactly.
    pub gesture_separation: Duration,
    /// Deltas below this on both axes are noise — too small to pick an axis or to reconsider
    /// the one already picked.
    pub unlock_lower_bound: Pixels,
    /// How much the horizontal component must dominate the vertical one for a *new* gesture to
    /// start out horizontal. Below `1.0` a gesture claims the horizontal axis while still
    /// drifting slightly downward, which is what a real sideways trackpad swipe looks like.
    pub start_percent: f32,
    /// How much the other axis must dominate for an *in-flight* gesture to break its lock.
    /// Larger values make the lock stickier.
    pub unlock_percent: f32,
}

impl Default for ScrollAxisLock {
    fn default() -> Self {
        Self::BALANCED
    }
}

impl ScrollAxisLock {
    /// Sticky enough that vertical reading is never derailed by sideways drift. The right
    /// default for a scroll container that fills its pane.
    pub const BALANCED: Self = Self {
        gesture_separation: Duration::from_millis(28),
        unlock_lower_bound: px(6.),
        start_percent: 1.0,
        unlock_percent: 1.9,
    };

    /// Picks up sideways intent eagerly. For a small horizontally-scrollable region embedded in
    /// a vertically scrolling document, where the horizontal axis is the whole point and a
    /// hard-to-reach lock reads as the region being stuck.
    pub const EAGER_HORIZONTAL: Self = Self {
        gesture_separation: Duration::from_millis(28),
        unlock_lower_bound: px(3.),
        start_percent: 0.85,
        unlock_percent: 1.05,
    };

    /// Whether `major` beats `minor` by enough to claim the gesture.
    ///
    /// Takes magnitudes: every caller passes `abs()` values, and folding the `abs()` in here is
    /// what stops a hard leftward flick (a large *negative* delta) reporting "no intent".
    fn dominates(&self, major: Pixels, minor: Pixels, percent: f32) -> bool {
        let major = major.abs();
        let minor = minor.abs();
        major >= self.unlock_lower_bound
            && (minor < self.unlock_lower_bound || major >= minor * percent)
    }
}

/// Tracks the dominant axis across the events in a scroll gesture.
#[derive(Clone, Copy, Debug, Default)]
pub struct OngoingScroll {
    last_event: Option<Instant>,
    axis: Option<Axis>,
}

impl OngoingScroll {
    /// The axis this gesture is currently locked to, if it has committed to one.
    pub fn axis(&self) -> Option<Axis> {
        self.axis
    }

    /// Pin the gesture to `axis` for the rest of its duration, bypassing the usual dominance
    /// test. Used for input that names its axis outright, such as a Shift-modified wheel.
    pub fn lock_to(&mut self, axis: Axis) {
        self.last_event = Some(Instant::now());
        self.axis = Some(axis);
    }

    /// Filters the given delta to the dominant axis of the current scroll gesture.
    ///
    /// Gestures are delimited by their touch phase when available, with a timeout
    /// fallback for platforms that only emit [`TouchPhase::Moved`].
    pub fn filter(
        &mut self,
        tuning: &ScrollAxisLock,
        delta: &mut Point<Pixels>,
        touch_phase: TouchPhase,
    ) {
        self.filter_at(tuning, delta, touch_phase, Instant::now())
    }

    fn filter_at(
        &mut self,
        tuning: &ScrollAxisLock,
        delta: &mut Point<Pixels>,
        touch_phase: TouchPhase,
        now: Instant,
    ) {
        if matches!(touch_phase, TouchPhase::Ended | TouchPhase::Cancelled) {
            self.last_event = None;
            self.axis = None;
            return;
        }

        let x = delta.x.abs();
        let y = delta.y.abs();
        if x.is_zero() && y.is_zero() {
            if touch_phase == TouchPhase::Started {
                self.last_event = None;
                self.axis = None;
            }
            return;
        }

        let starts_new_gesture = touch_phase == TouchPhase::Started
            || self.last_event.is_none_or(|last_event| {
                now.duration_since(last_event) >= tuning.gesture_separation
            });
        let mut axis = self.axis;
        if starts_new_gesture {
            axis = if tuning.dominates(x, y, tuning.start_percent) {
                Some(Axis::Horizontal)
            } else {
                Some(Axis::Vertical)
            };
        } else if x.max(y) >= tuning.unlock_lower_bound {
            match axis {
                Some(Axis::Vertical) if tuning.dominates(x, y, tuning.unlock_percent) => {
                    axis = None;
                }
                Some(Axis::Horizontal) if tuning.dominates(y, x, tuning.unlock_percent) => {
                    axis = None;
                }
                _ => {}
            }
        }

        self.last_event = Some(now);
        self.axis = axis;
        match axis {
            Some(Axis::Vertical) => delta.x = Pixels::ZERO,
            Some(Axis::Horizontal) => delta.y = Pixels::ZERO,
            None => {}
        }
    }
}

/// Feel constants consumed by gesture recognizers. Provided on a best-effort
/// basis, depending on each platform's support, defaulting to GPUI's own
/// (iOS flavored) values
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GestureTuning {
    /// Distance a touch may travel before it stops being a potential tap and
    /// becomes a pan/drag.
    pub touch_slop: Pixels,
    /// Maximum interval between taps for them to accumulate a tap count.
    pub multi_tap_interval: Duration,
    /// Maximum distance between taps for them to accumulate a tap count.
    pub multi_tap_slop: Pixels,
    /// How long a touch must remain within [`Self::touch_slop`] to be
    /// recognized as a long press.
    pub long_press_duration: Duration,
    /// Per-millisecond decay factor applied to scroll momentum after a fling.
    /// (`UIScrollView` uses `0.998` per millisecond for its normal
    /// deceleration rate.)
    pub momentum_decay_per_ms: f32,
    /// Minimum release velocity, in pixels per second, required to start
    /// scroll momentum.
    pub min_fling_velocity: f32,
}

impl Default for GestureTuning {
    fn default() -> Self {
        Self {
            touch_slop: px(8.),
            multi_tap_interval: Duration::from_millis(400),
            multi_tap_slop: px(16.),
            long_press_duration: Duration::from_millis(500),
            momentum_decay_per_ms: 0.998,
            min_fling_velocity: 50.,
        }
    }
}

/// The set of gesture kinds that participate in recognition.
///
/// Used by [`PlatformGestures::native_recognizers`] to declare which gestures
/// the platform recognizes natively rather than leaving to gpui core's
/// portable recognizers.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GestureKinds {
    /// Tap (and multi-tap), surfaced as [`ClickEvent::Touch`](crate::ClickEvent).
    pub tap: bool,
    /// Long press, surfaced as [`LongPressEvent`].
    pub long_press: bool,
    /// Pan/scroll (including fling momentum), surfaced as
    /// [`ScrollWheelEvent`](crate::ScrollWheelEvent)s.
    pub pan: bool,
    /// Pinch to zoom, surfaced as [`PinchEvent`](crate::PinchEvent)s.
    pub pinch: bool,
}

impl GestureKinds {
    /// No gestures; gpui core's portable recognizers handle everything.
    pub const NONE: Self = Self {
        tap: false,
        long_press: false,
        pan: false,
        pinch: false,
    };

    /// All gesture kinds.
    pub const ALL: Self = Self {
        tap: true,
        long_press: true,
        pan: true,
        pinch: true,
    };
}

/// A long-press gesture, mobile's context-menu trigger.
///
/// A bare long press is surfaced as a [`ClickEvent`](crate::ClickEvent) with
/// `long_press: true`, delivered to aux-click listeners alongside right
/// clicks. This event is the raw hook for elements that need the gesture
/// itself (e.g. long-press to start a drag); the registration API ships
/// together with the gesture arena.
#[derive(Clone, Debug, Default)]
pub struct LongPressEvent {
    /// The position of the touch that was recognized as a long press.
    pub position: Point<Pixels>,
}

/// Platform gesture recognition services.
///
/// If your mobile platform supports native gesture recognition, use this
/// to share it with GPUI.
pub trait PlatformGestures {
    /// Feel constants for the portable recognizers on this platform.
    fn tuning(&self) -> GestureTuning {
        GestureTuning::default()
    }

    /// The gesture kinds this platform recognizes natively.
    fn native_recognizers(&self) -> GestureKinds {
        GestureKinds::NONE
    }
}

/// A no-op [`PlatformGestures`] implementation: no native recognizers and
/// default tuning. Suitable for desktop platforms and tests.
pub struct NullPlatformGestures;

impl PlatformGestures for NullPlatformGestures {}

#[cfg(test)]
mod tests {
    use super::*;

    const TUNING: ScrollAxisLock = ScrollAxisLock::BALANCED;
    use crate::point;

    #[test]
    fn ongoing_scroll_locks_to_dominant_axis() {
        let now = Instant::now();
        let mut ongoing_scroll = OngoingScroll::default();
        let mut horizontal_delta = point(px(10.), px(2.));
        ongoing_scroll.filter_at(&TUNING, &mut horizontal_delta, TouchPhase::Started, now);
        assert_eq!(ongoing_scroll.axis, Some(Axis::Horizontal));
        assert_eq!(horizontal_delta, point(px(10.), px(0.)));

        let mut continued_delta = point(px(3.), px(2.));
        ongoing_scroll.filter_at(
            &TUNING,
            &mut continued_delta,
            TouchPhase::Moved,
            now + Duration::from_millis(1),
        );
        assert_eq!(ongoing_scroll.axis, Some(Axis::Horizontal));
        assert_eq!(continued_delta, point(px(3.), px(0.)));
    }

    #[test]
    fn ongoing_scroll_unlocks_when_direction_changes() {
        let now = Instant::now();
        let mut ongoing_scroll = OngoingScroll::default();
        let mut horizontal_delta = point(px(10.), px(2.));
        ongoing_scroll.filter_at(&TUNING, &mut horizontal_delta, TouchPhase::Started, now);

        let mut vertical_delta = point(px(2.), px(10.));
        ongoing_scroll.filter_at(
            &TUNING,
            &mut vertical_delta,
            TouchPhase::Moved,
            now + Duration::from_millis(1),
        );
        assert_eq!(ongoing_scroll.axis, None);
        assert_eq!(vertical_delta, point(px(2.), px(10.)));
    }

    #[test]
    fn ongoing_scroll_starts_new_gesture_at_timeout_boundary() {
        let now = Instant::now();
        let mut ongoing_scroll = OngoingScroll::default();
        let mut horizontal_delta = point(px(10.), px(2.));
        ongoing_scroll.filter_at(&TUNING, &mut horizontal_delta, TouchPhase::Moved, now);

        let mut vertical_delta = point(px(2.), px(10.));
        ongoing_scroll.filter_at(
            &TUNING,
            &mut vertical_delta,
            TouchPhase::Moved,
            now + TUNING.gesture_separation,
        );
        assert_eq!(ongoing_scroll.axis, Some(Axis::Vertical));
        assert_eq!(vertical_delta, point(px(0.), px(10.)));
    }

    #[test]
    fn ongoing_scroll_ignores_zero_delta_and_resets_when_ended() {
        let now = Instant::now();
        let mut ongoing_scroll = OngoingScroll::default();
        let mut horizontal_delta = point(px(10.), px(2.));
        ongoing_scroll.filter_at(&TUNING, &mut horizontal_delta, TouchPhase::Started, now);

        let mut zero_delta = Point::default();
        ongoing_scroll.filter_at(
            &TUNING,
            &mut zero_delta,
            TouchPhase::Ended,
            now + Duration::from_millis(1),
        );
        assert_eq!(ongoing_scroll.axis, None);

        let mut vertical_delta = point(px(2.), px(3.));
        ongoing_scroll.filter_at(
            &TUNING,
            &mut vertical_delta,
            TouchPhase::Moved,
            now + Duration::from_millis(2),
        );
        assert_eq!(ongoing_scroll.axis, Some(Axis::Vertical));
        assert_eq!(vertical_delta, point(px(0.), px(3.)));
    }

    #[test]
    fn ongoing_scroll_ignores_zero_delta_movement() {
        let now = Instant::now();
        let mut ongoing_scroll = OngoingScroll::default();
        let mut horizontal_delta = point(px(10.), px(2.));
        ongoing_scroll.filter_at(&TUNING, &mut horizontal_delta, TouchPhase::Started, now);

        let mut zero_delta = Point::default();
        ongoing_scroll.filter_at(
            &TUNING,
            &mut zero_delta,
            TouchPhase::Moved,
            now + TUNING.gesture_separation,
        );

        let mut vertical_delta = point(px(2.), px(10.));
        ongoing_scroll.filter_at(
            &TUNING,
            &mut vertical_delta,
            TouchPhase::Moved,
            now + TUNING.gesture_separation,
        );
        assert_eq!(ongoing_scroll.axis, Some(Axis::Vertical));
        assert_eq!(vertical_delta, point(px(0.), px(10.)));
    }

    #[test]
    fn ongoing_scroll_supports_moved_only_platforms() {
        let now = Instant::now();
        let mut ongoing_scroll = OngoingScroll::default();
        let mut horizontal_delta = point(px(10.), px(2.));
        ongoing_scroll.filter_at(&TUNING, &mut horizontal_delta, TouchPhase::Moved, now);
        assert_eq!(ongoing_scroll.axis, Some(Axis::Horizontal));
        assert_eq!(horizontal_delta, point(px(10.), px(0.)));
    }

    /// The balanced tuning treats sub-`unlock_lower_bound` movement as noise and reads it as
    /// vertical, rather than committing to horizontal on a 1px sideways wobble.
    #[test]
    fn balanced_tuning_reads_tiny_deltas_as_vertical() {
        let now = Instant::now();
        let mut ongoing_scroll = OngoingScroll::default();
        let mut jitter = point(px(2.), px(1.));

        ongoing_scroll.filter_at(&TUNING, &mut jitter, TouchPhase::Started, now);

        assert_eq!(ongoing_scroll.axis, Some(Axis::Vertical));
    }

    /// The eager tuning is the one a small embedded horizontal region wants: it claims a gesture
    /// that is still drifting downward, which the balanced tuning reads as vertical.
    #[test]
    fn eager_tuning_claims_a_downward_drifting_sideways_swipe() {
        let now = Instant::now();
        let mut balanced = OngoingScroll::default();
        let mut balanced_delta = point(px(8.), px(9.));
        balanced.filter_at(&TUNING, &mut balanced_delta, TouchPhase::Started, now);
        assert_eq!(balanced.axis, Some(Axis::Vertical));

        let mut eager = OngoingScroll::default();
        let mut eager_delta = point(px(8.), px(9.));
        eager.filter_at(
            &ScrollAxisLock::EAGER_HORIZONTAL,
            &mut eager_delta,
            TouchPhase::Started,
            now,
        );
        assert_eq!(eager.axis, Some(Axis::Horizontal));
        assert_eq!(eager_delta, point(px(8.), px(0.)));
    }

    /// The eager tuning also breaks a vertical lock on much weaker sideways intent.
    #[test]
    fn eager_tuning_unlocks_a_vertical_gesture_sooner() {
        let now = Instant::now();
        let mut vertical = point(px(0.), px(30.));
        let later = now + Duration::from_millis(1);

        let mut balanced = OngoingScroll::default();
        balanced.filter_at(&TUNING, &mut vertical.clone(), TouchPhase::Started, now);
        balanced.filter_at(
            &TUNING,
            &mut point(px(9.), px(8.)),
            TouchPhase::Moved,
            later,
        );
        assert_eq!(
            balanced.axis,
            Some(Axis::Vertical),
            "9 vs 8 is well under the balanced 1.9x unlock ratio"
        );

        let eager_tuning = ScrollAxisLock::EAGER_HORIZONTAL;
        let mut eager = OngoingScroll::default();
        eager.filter_at(&eager_tuning, &mut vertical, TouchPhase::Started, now);
        eager.filter_at(
            &eager_tuning,
            &mut point(px(9.), px(8.)),
            TouchPhase::Moved,
            later,
        );
        assert_eq!(eager.axis, None, "the eager 1.05x ratio releases the lock");
    }

    /// Signs must not decide intent: a hard flick left/up is as much intent as right/down.
    #[test]
    fn dominance_uses_magnitudes_not_signs() {
        let now = Instant::now();
        let mut ongoing_scroll = OngoingScroll::default();
        let mut leftward = point(px(-24.), px(-6.));

        ongoing_scroll.filter_at(&TUNING, &mut leftward, TouchPhase::Started, now);

        assert_eq!(ongoing_scroll.axis, Some(Axis::Horizontal));
        assert_eq!(leftward, point(px(-24.), px(0.)));
    }

    /// `lock_to` names the axis outright, bypassing the dominance test, and counts as an event
    /// so the following delta continues the same gesture rather than starting a new one.
    #[test]
    fn lock_to_pins_the_axis_without_a_dominant_delta() {
        let mut ongoing_scroll = OngoingScroll::default();
        ongoing_scroll.lock_to(Axis::Horizontal);
        assert_eq!(ongoing_scroll.axis(), Some(Axis::Horizontal));

        // Vertically dominant, but under the noise floor, so the lock stands and the vertical
        // component is filtered out. A fresh gesture would have read this as vertical.
        let mut drifting_delta = point(px(3.), px(4.));
        ongoing_scroll.filter(&TUNING, &mut drifting_delta, TouchPhase::Moved);

        assert_eq!(ongoing_scroll.axis(), Some(Axis::Horizontal));
        assert_eq!(drifting_delta, point(px(3.), px(0.)));
    }

    /// ...but the lock is not permanent: clear intent on the other axis still releases it, which
    /// is what lets a Shift-locked gesture hand control back once Shift is no longer driving it.
    #[test]
    fn lock_to_still_releases_on_clear_opposing_intent() {
        let mut ongoing_scroll = OngoingScroll::default();
        ongoing_scroll.lock_to(Axis::Horizontal);

        let mut vertical_delta = point(px(1.), px(9.));
        ongoing_scroll.filter(&TUNING, &mut vertical_delta, TouchPhase::Moved);

        assert_eq!(ongoing_scroll.axis(), None);
        assert_eq!(vertical_delta, point(px(1.), px(9.)));
    }
}
