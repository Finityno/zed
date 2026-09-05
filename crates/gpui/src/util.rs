use crate::{BackgroundExecutor, Task};
use std::{
    future::Future,
    pin::Pin,
    sync::atomic::{AtomicUsize, Ordering::SeqCst},
    task,
    time::Duration,
};

/// A helper trait for building complex objects with imperative conditionals in a fluent style.
pub trait FluentBuilder {
    /// Imperatively modify self with the given closure.
    fn map<U>(self, f: impl FnOnce(Self) -> U) -> U
    where
        Self: Sized,
    {
        f(self)
    }

    /// Conditionally modify self with the given closure.
    fn when(self, condition: bool, then: impl FnOnce(Self) -> Self) -> Self
    where
        Self: Sized,
    {
        self.map(|this| if condition { then(this) } else { this })
    }

    /// Conditionally modify self with the given closure.
    fn when_else(
        self,
        condition: bool,
        then: impl FnOnce(Self) -> Self,
        else_fn: impl FnOnce(Self) -> Self,
    ) -> Self
    where
        Self: Sized,
    {
        self.map(|this| if condition { then(this) } else { else_fn(this) })
    }

    /// Conditionally unwrap and modify self with the given closure, if the given option is Some.
    fn when_some<T>(self, option: Option<T>, then: impl FnOnce(Self, T) -> Self) -> Self
    where
        Self: Sized,
    {
        self.map(|this| {
            if let Some(value) = option {
                then(this, value)
            } else {
                this
            }
        })
    }
    /// Conditionally unwrap and modify self with the given closure, if the given option is None.
    fn when_none<T>(self, option: &Option<T>, then: impl FnOnce(Self) -> Self) -> Self
    where
        Self: Sized,
    {
        self.map(|this| if option.is_some() { this } else { then(this) })
    }
}

/// Extensions for Future types that provide additional combinators and utilities.
pub trait FutureExt {
    /// Requires a Future to complete before the specified duration has elapsed.
    /// Similar to tokio::timeout.
    fn with_timeout(self, timeout: Duration, executor: &BackgroundExecutor) -> WithTimeout<Self>
    where
        Self: Sized;
}

impl<T: Future> FutureExt for T {
    fn with_timeout(self, timeout: Duration, executor: &BackgroundExecutor) -> WithTimeout<Self>
    where
        Self: Sized,
    {
        WithTimeout {
            future: self,
            timer: executor.timer(timeout),
        }
    }
}

#[pin_project::pin_project]
pub struct WithTimeout<T> {
    #[pin]
    future: T,
    #[pin]
    timer: Task<()>,
}

#[derive(Debug, thiserror::Error)]
#[error("Timed out before future resolved")]
/// Error returned by with_timeout when the timeout duration elapsed before the future resolved
pub struct Timeout;

impl<T: Future> Future for WithTimeout<T> {
    type Output = Result<T::Output, Timeout>;

    fn poll(self: Pin<&mut Self>, cx: &mut task::Context) -> task::Poll<Self::Output> {
        let this = self.project();

        if let task::Poll::Ready(output) = this.future.poll(cx) {
            task::Poll::Ready(Ok(output))
        } else if this.timer.poll(cx).is_ready() {
            task::Poll::Ready(Err(Timeout))
        } else {
            task::Poll::Pending
        }
    }
}

/// Increment the given atomic counter if it is not zero.
/// Return the new value of the counter.
pub(crate) fn atomic_incr_if_not_zero(counter: &AtomicUsize) -> usize {
    let mut loaded = counter.load(SeqCst);
    loop {
        if loaded == 0 {
            return 0;
        }
        match counter.compare_exchange_weak(loaded, loaded + 1, SeqCst, SeqCst) {
            Ok(x) => return x + 1,
            Err(actual) => loaded = actual,
        }
    }
}

/// Rounds to the nearest integer with 0.5 ties toward zero.
#[inline]
pub(crate) fn round_half_toward_zero(value: f32) -> f32 {
    (value.abs() - 0.5).ceil().copysign(value)
}

#[inline]
pub(crate) fn round_half_toward_zero_f64(value: f64) -> f64 {
    (value.abs() - 0.5).ceil().copysign(value)
}

#[inline]
pub(crate) fn round_to_device_pixel(logical: f32, scale_factor: f32) -> f32 {
    round_half_toward_zero(logical * scale_factor)
}

#[inline]
pub(crate) fn round_stroke_to_device_pixel(logical: f32, scale_factor: f32) -> f32 {
    if logical == 0.0 {
        0.0
    } else {
        round_to_device_pixel(logical.max(0.0), scale_factor).max(1.0)
    }
}

#[inline]
pub(crate) fn floor_to_device_pixel(logical: f32, scale_factor: f32) -> f32 {
    (logical * scale_factor).floor()
}

#[inline]
pub(crate) fn ceil_to_device_pixel(logical: f32, scale_factor: f32) -> f32 {
    (logical * scale_factor).ceil()
}

#[cfg(test)]
mod tests {
    use crate::TestAppContext;

    use super::*;

    #[test]
    fn test_round_half_toward_zero() {
        // Midpoint ties go toward zero
        assert_eq!(round_half_toward_zero(0.5), 0.0);
        assert_eq!(round_half_toward_zero(1.5), 1.0);
        assert_eq!(round_half_toward_zero(2.5), 2.0);
        assert_eq!(round_half_toward_zero(-0.5), 0.0);
        assert_eq!(round_half_toward_zero(-1.5), -1.0);
        assert_eq!(round_half_toward_zero(-2.5), -2.0);

        // Non-midpoint values round to nearest
        assert_eq!(round_half_toward_zero(1.5001), 2.0);
        assert_eq!(round_half_toward_zero(1.4999), 1.0);
        assert_eq!(round_half_toward_zero(-1.5001), -2.0);
        assert_eq!(round_half_toward_zero(-1.4999), -1.0);

        // Integers are unchanged
        assert_eq!(round_half_toward_zero(0.0), 0.0);
        assert_eq!(round_half_toward_zero(3.0), 3.0);
        assert_eq!(round_half_toward_zero(-3.0), -3.0);
    }

    #[test]
    fn test_device_pixel_helpers() {
        // Snap uses half-toward-zero: 1.0 * 1.5 = 1.5 ties toward 1.0.
        assert_eq!(round_to_device_pixel(1.0, 1.5), 1.0);
        // Below the tie rounds down, above rounds up.
        assert_eq!(round_to_device_pixel(0.3, 2.0), 1.0);
        assert_eq!(round_to_device_pixel(1.4, 1.0), 1.0);
        assert_eq!(round_to_device_pixel(1.6, 1.0), 2.0);

        // Stroke uses snap, but clamps non-zero input up to at least 1dp.
        assert_eq!(round_stroke_to_device_pixel(0.0, 1.0), 0.0);
        assert_eq!(round_stroke_to_device_pixel(0.4, 1.0), 1.0);
        assert_eq!(round_stroke_to_device_pixel(0.5, 1.0), 1.0);
        assert_eq!(round_stroke_to_device_pixel(1.0, 1.5), 1.0);
        assert_eq!(round_stroke_to_device_pixel(1.6, 1.0), 2.0);

        // Cover's near edge floors, far edge ceils. Together they form a strict superset.
        assert_eq!(floor_to_device_pixel(0.3, 2.0), 0.0);
        assert_eq!(ceil_to_device_pixel(0.3, 2.0), 1.0);
        assert_eq!(floor_to_device_pixel(2.1, 1.0), 2.0);
        assert_eq!(ceil_to_device_pixel(2.1, 1.0), 3.0);

        // Integer device-pixel inputs are stable under all three.
        assert_eq!(round_to_device_pixel(2.0, 2.0), 4.0);
        assert_eq!(floor_to_device_pixel(2.0, 2.0), 4.0);
        assert_eq!(ceil_to_device_pixel(2.0, 2.0), 4.0);
    }

    #[test]
    fn test_round_half_toward_zero_f64() {
        assert_eq!(round_half_toward_zero_f64(0.5), 0.0);
        assert_eq!(round_half_toward_zero_f64(-0.5), 0.0);
        assert_eq!(round_half_toward_zero_f64(1.5), 1.0);
        assert_eq!(round_half_toward_zero_f64(-1.5), -1.0);
        assert_eq!(round_half_toward_zero_f64(2.5001), 3.0);
    }

    #[gpui::test]
    async fn test_with_timeout(cx: &mut TestAppContext) {
        Task::ready(())
            .with_timeout(Duration::from_secs(1), &cx.executor())
            .await
            .expect("Timeout should be noop");

        let long_duration = Duration::from_secs(6000);
        let short_duration = Duration::from_secs(1);
        cx.executor()
            .timer(long_duration)
            .with_timeout(short_duration, &cx.executor())
            .await
            .expect_err("timeout should have triggered");

        let fut = cx
            .executor()
            .timer(long_duration)
            .with_timeout(short_duration, &cx.executor());
        cx.executor().advance_clock(short_duration * 2);
        futures::FutureExt::now_or_never(fut)
            .unwrap_or_else(|| panic!("timeout should have triggered"))
            .expect_err("timeout");
    }
}

/// How many consecutive low-use frames it takes to shrink a per-frame
/// collection back down. Two seconds at 60fps, matching the renderer's
/// instance-buffer pool: long enough that a burst of complex frames with brief
/// pauses never thrashes, short enough that one elaborate frame does not set
/// the allocation for the rest of the session.
pub(crate) const SHRINK_AFTER_FRAMES: u32 = 120;

/// A per-frame collection is never shrunk below this many elements.
pub(crate) const MIN_RETAINED_CAPACITY: usize = 32;

/// Hysteresis for collections that are refilled every frame and cleared with
/// their capacity retained.
///
/// `Vec::clear` keeps the high-water capacity, so one elaborate frame — a long
/// chat transcript scrolled past, a big review diff — pins that memory for the
/// life of the window, and a window holds two frames of it. This notes each
/// frame's fill against the capacity and, after [`SHRINK_AFTER_FRAMES`]
/// consecutive frames that filled at most half of it, shrinks to twice the
/// largest fill seen in that window. The doubled target leaves headroom so the
/// next frame does not immediately regrow, and any frame that needs more than
/// half the capacity resets the count, so a workload that alternates between
/// big and small frames never shrinks at all.
#[derive(Debug, Default)]
pub(crate) struct CapacityShrink {
    low_use_frames: u32,
    max_recent_len: usize,
}

impl CapacityShrink {
    /// Notes one frame's `len` against `capacity` and returns the capacity to
    /// shrink to, if this frame completes the low-use window.
    pub(crate) fn record(&mut self, len: usize, capacity: usize) -> Option<usize> {
        if capacity <= MIN_RETAINED_CAPACITY || len.saturating_mul(2) > capacity {
            self.low_use_frames = 0;
            self.max_recent_len = 0;
            return None;
        }
        self.max_recent_len = self.max_recent_len.max(len);
        self.low_use_frames += 1;
        if self.low_use_frames < SHRINK_AFTER_FRAMES {
            return None;
        }
        self.low_use_frames = 0;
        let target = self
            .max_recent_len
            .saturating_mul(2)
            .max(MIN_RETAINED_CAPACITY);
        self.max_recent_len = 0;
        Some(target)
    }

    /// Clears `vec` for the next frame, shrinking it once it has been
    /// over-provisioned for long enough.
    pub(crate) fn clear_vec<T>(&mut self, vec: &mut Vec<T>) {
        let target = self.record(vec.len(), vec.capacity());
        vec.clear();
        if let Some(target) = target {
            vec.shrink_to(target);
        }
    }

    /// Clears `map` for the next frame, shrinking it once it has been
    /// over-provisioned for long enough.
    pub(crate) fn clear_map<K, V, S>(&mut self, map: &mut std::collections::HashMap<K, V, S>)
    where
        K: Eq + std::hash::Hash,
        S: std::hash::BuildHasher,
    {
        let target = self.record(map.len(), map.capacity());
        map.clear();
        if let Some(target) = target {
            map.shrink_to(target);
        }
    }

    /// The capacity to shrink to once the window has stopped drawing, given
    /// the fill of the frame still on screen.
    ///
    /// [`Self::record`] only advances on draws, so a heavy scene followed by a
    /// quiet or hidden window would keep its high-water capacity for as long
    /// as it stays quiet. Here the low-use run is treated as complete now:
    /// the target is twice the largest fill seen since the last busy frame,
    /// `last_len` included, and `None` when the collection is not
    /// over-provisioned against that.
    pub(crate) fn idle_target(&mut self, last_len: usize, capacity: usize) -> Option<usize> {
        let target = self
            .max_recent_len
            .max(last_len)
            .saturating_mul(2)
            .max(MIN_RETAINED_CAPACITY);
        if target >= capacity {
            return None;
        }
        self.low_use_frames = 0;
        self.max_recent_len = 0;
        Some(target)
    }

    /// Shrinks the cleared `vec` to twice `last_len`, the fill of its
    /// counterpart in the frame still on screen, if it holds more than that.
    pub(crate) fn shrink_vec_idle<T>(&mut self, vec: &mut Vec<T>, last_len: usize) {
        if let Some(target) = self.idle_target(last_len, vec.capacity()) {
            vec.shrink_to(target);
        }
    }

    /// Shrinks the cleared `map` to twice `last_len`, the fill of its
    /// counterpart in the frame still on screen, if it holds more than that.
    pub(crate) fn shrink_map_idle<K, V, S>(
        &mut self,
        map: &mut std::collections::HashMap<K, V, S>,
        last_len: usize,
    ) where
        K: Eq + std::hash::Hash,
        S: std::hash::BuildHasher,
    {
        if let Some(target) = self.idle_target(last_len, map.capacity()) {
            map.shrink_to(target);
        }
    }
}

#[cfg(test)]
mod capacity_shrink_tests {
    use super::*;

    #[test]
    fn shrinks_after_a_full_window_of_low_use_frames() {
        let mut vec: Vec<u64> = Vec::with_capacity(10_000);
        let mut shrink = CapacityShrink::default();
        for _ in 0..SHRINK_AFTER_FRAMES - 1 {
            vec.extend(0..10);
            shrink.clear_vec(&mut vec);
            assert_eq!(vec.capacity(), 10_000);
        }
        vec.extend(0..10);
        shrink.clear_vec(&mut vec);
        assert!(vec.capacity() <= MIN_RETAINED_CAPACITY.max(20));
    }

    #[test]
    fn a_busy_frame_resets_the_window() {
        let mut vec: Vec<u64> = Vec::with_capacity(10_000);
        let mut shrink = CapacityShrink::default();
        for frame in 0..SHRINK_AFTER_FRAMES * 4 {
            let fill = if frame % 100 == 0 { 6_000 } else { 10 };
            vec.extend(0..fill);
            shrink.clear_vec(&mut vec);
            assert_eq!(vec.capacity(), 10_000);
        }
    }

    #[test]
    fn never_shrinks_below_the_floor() {
        let mut shrink = CapacityShrink::default();
        for _ in 0..SHRINK_AFTER_FRAMES * 2 {
            assert_eq!(shrink.record(0, MIN_RETAINED_CAPACITY), None);
        }
        for _ in 0..SHRINK_AFTER_FRAMES - 1 {
            assert_eq!(shrink.record(1, 1_000), None);
        }
        assert_eq!(shrink.record(1, 1_000), Some(MIN_RETAINED_CAPACITY));
    }

    #[test]
    fn idle_shrinks_before_the_frame_window_completes() {
        let mut vec: Vec<u64> = Vec::with_capacity(10_000);
        let mut shrink = CapacityShrink::default();
        for _ in 0..5 {
            vec.extend(0..10);
            shrink.clear_vec(&mut vec);
        }
        assert_eq!(vec.capacity(), 10_000);
        shrink.shrink_vec_idle(&mut vec, 10);
        assert!(vec.capacity() <= MIN_RETAINED_CAPACITY.max(20));
    }

    #[test]
    fn idle_keeps_what_the_frame_on_screen_fills() {
        let mut shrink = CapacityShrink::default();
        assert_eq!(shrink.record(10, 1_000), None);
        assert_eq!(shrink.idle_target(600, 1_000), None);
        assert_eq!(shrink.idle_target(10, 1_000), Some(MIN_RETAINED_CAPACITY));
    }

    #[test]
    fn idle_target_covers_the_largest_recent_fill() {
        let mut shrink = CapacityShrink::default();
        assert_eq!(shrink.record(300, 10_000), None);
        assert_eq!(shrink.record(10, 10_000), None);
        assert_eq!(shrink.idle_target(10, 10_000), Some(600));
        // The run is consumed: the next idle sizes against the frame on screen
        // alone, where the stale 300 would have called 600 a fit.
        assert_eq!(shrink.idle_target(10, 600), Some(MIN_RETAINED_CAPACITY));
    }
}
