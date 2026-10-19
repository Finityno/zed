use crate::{App, EntityId, Global, Pixels, ScrollDelta, ScrollWheelEvent, TouchPhase, Window, px};
use scheduler::Instant;
use std::{cell::RefCell, rc::Rc, time::Duration};

/// App-wide opt-in for elastic feedback at vertical scroll boundaries.
#[derive(Default)]
pub struct ScrollEdgeBounce(pub bool);

impl Global for ScrollEdgeBounce {}

impl ScrollEdgeBounce {
    fn enabled(window: &Window, cx: &App) -> bool {
        cx.try_global::<Self>().is_some_and(|setting| setting.0)
            && !cx.reduce_motion()
            && window.is_window_active()
    }
}

/// Persistent elastic feedback for a custom vertical scroll surface.
#[derive(Clone, Debug, Default)]
pub struct ScrollBounce(Rc<RefCell<Bounce>>);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ScrollInput {
    Direct,
    Momentum,
    Wheel,
}

#[derive(Debug, Default)]
struct Bounce {
    position: f32,
    velocity: f32,
    limit: f32,
    last_update: Option<Instant>,
    momentum_consumed: bool,
    wheel_consumed: bool,
    last_wheel: Option<Instant>,
}

impl Bounce {
    // The reference's six edge impacts fit x = A * e * w * t * exp(-w * t).
    // This peaks after 80 ms, then returns halfway in another 134 ms.
    const SPRING_FREQUENCY: f32 = 12.5;

    fn sample(&self, now: Instant) -> (f32, f32) {
        let Some(last_update) = self.last_update else { return (0., 0.); };
        let time = now.saturating_duration_since(last_update).as_secs_f32();
        let damping = Self::SPRING_FREQUENCY;
        let decay = (-damping * time).exp();
        let coefficient = self.velocity + damping * self.position;
        (
            (self.position + coefficient * time) * decay,
            (self.velocity - damping * coefficient * time) * decay,
        )
    }

    fn displacement(&self, now: Instant) -> f32 {
        let (position, velocity) = self.sample(now);
        if position.abs() < 0.5 && velocity.abs() < 5. { 0. } else { position }
    }

    fn clear_motion(&mut self) {
        // Keep consumed-input latches after settling: a long native momentum
        // tail must not start another bounce on the next packet.
        self.position = 0.;
        self.velocity = 0.;
        self.last_update = None;
    }

    fn input(&mut self, excess: f32, extent: f32, phase: TouchPhase, input: ScrollInput, now: Instant) {
        if phase == TouchPhase::Cancelled {
            self.clear_motion();
            self.momentum_consumed = true;
            self.wheel_consumed = true;
            return;
        }
        if phase == TouchPhase::Ended { return; }
        if input == ScrollInput::Direct {
            self.momentum_consumed = false;
            self.wheel_consumed = false;
        }
        if input == ScrollInput::Wheel {
            if self.last_wheel.is_none_or(|last| now.saturating_duration_since(last) >= Duration::from_millis(150)) {
                self.wheel_consumed = false;
            }
            self.last_wheel = Some(now);
        }
        if excess == 0. {
            self.clear_motion();
            return;
        }
        if (input == ScrollInput::Momentum && self.momentum_consumed)
            || (input == ScrollInput::Wheel && self.wheel_consumed) {
            return;
        }
        let (position, velocity) = self.sample(now);
        self.limit = (extent * 0.06).clamp(6., 80.);
        if input != ScrollInput::Direct {
            if input == ScrollInput::Momentum { self.momentum_consumed = true; }
            else { self.wheel_consumed = true; }
            if position.abs() >= 0.5 && position.signum() == excess.signum() {
                return;
            }
            let pull = excess * 0.55;
            let peak = self.limit * pull / (self.limit + pull.abs());
            self.position = 0.;
            self.velocity = peak * std::f32::consts::E * Self::SPRING_FREQUENCY;
        } else {
            if phase == TouchPhase::Started || position.signum() != excess.signum() {
                self.position = 0.;
                self.velocity = 0.;
            } else {
                self.position = position;
                self.velocity = velocity;
            }
            let pull = self.position * self.limit
                / (self.limit - self.position.abs()).max(self.limit / 9.);
            let pull = (pull + excess * 0.55).clamp(-self.limit * 8., self.limit * 8.);
            self.position = self.limit * pull / (self.limit + pull.abs());
            let maximum_velocity = Self::SPRING_FREQUENCY * self.position.abs();
            self.velocity = self.velocity.clamp(-maximum_velocity, maximum_velocity);
        }
        self.last_update = Some(now);
    }
}

impl ScrollBounce {
    /// Feed the part of a wheel event a scrollable surface could not consume.
    /// Call after normal scrolling, including zero-delta end/cancel events.
    /// `view` must be the entity rendering the surface, captured before dispatch.
    pub fn scroll(
        &self,
        excess: Pixels,
        viewport_height: Pixels,
        event: &ScrollWheelEvent,
        view: EntityId,
        window: &mut Window,
        cx: &mut App,
    ) {
        window.queue_scroll_bounce(self, excess, viewport_height, event, view, cx);
    }

    /// Clear feedback after a programmatic scroll, scrollbar drag, or content reset.
    pub fn clear(&self) -> bool {
        let mut state = self.0.borrow_mut();
        let changed = state.position != 0. || state.velocity != 0.;
        state.clear_motion();
        changed
    }

    /// Return the visual displacement and schedule its next animation frame.
    /// `offset` is the clamped, negative content offset; `maximum` is its positive extent.
    pub fn offset(
        &self,
        offset: Pixels,
        maximum: Pixels,
        window: &mut Window,
        cx: &App,
    ) -> Pixels {
        if !ScrollEdgeBounce::enabled(window, cx) || maximum <= px(0.) {
            self.clear();
            return px(0.);
        }
        let state = self.0.borrow();
        let now = Instant::now();
        let displacement = state.displacement(now);
        let velocity = state.sample(now).1;
        let direction = if displacement == 0. { velocity } else { displacement };
        let active = state.last_update.is_some() && (displacement != 0. || velocity.abs() >= 5.);
        drop(state);
        if (direction > 0. && offset < px(-0.5))
            || (direction < 0. && offset > -maximum + px(0.5))
            || !active
        {
            self.clear();
            return px(0.);
        }
        window.request_animation_frame();
        px(displacement)
    }
}

pub(crate) struct ScrollBounceRequest {
    bounce: ScrollBounce,
    excess: Pixels,
    extent: Pixels,
    phase: TouchPhase,
    input: ScrollInput,
    view: EntityId,
}

impl Window {
    pub(crate) fn queue_scroll_bounce(
        &mut self,
        bounce: &ScrollBounce,
        excess: Pixels,
        extent: Pixels,
        event: &ScrollWheelEvent,
        view: EntityId,
        cx: &mut App,
    ) {
        if !ScrollEdgeBounce::enabled(self, cx) {
            if bounce.clear() { cx.notify(view); }
            return;
        }
        // Let normal bubbling decide which scroller owns the feedback. An inner
        // scroller at its boundary must still hand the gesture to its ancestor.
        if let Some(previous) = self.pending_scroll_bounce.take()
            && !Rc::ptr_eq(&previous.bounce.0, &bounce.0)
            && previous.bounce.clear()
        {
            cx.notify(previous.view);
        }
        self.pending_scroll_bounce = Some(ScrollBounceRequest {
            bounce: bounce.clone(), excess, extent, phase: event.touch_phase,
            input: if event.is_momentum { ScrollInput::Momentum }
                else if matches!(event.delta, ScrollDelta::Lines(_)) { ScrollInput::Wheel }
                else { ScrollInput::Direct },
            view,
        });
    }

    pub(crate) fn apply_scroll_bounce(&mut self, cx: &mut App) {
        if let Some(request) = self.pending_scroll_bounce.take() {
            request.bounce.0.borrow_mut().input(
                request.excess.into(), request.extent.into(), request.phase, request.input, Instant::now(),
            );
            cx.notify(request.view);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scroll_bounce_matches_measured_momentum_curve() {
        let now = Instant::now();
        // Bottom impact at video frame 408, tracked through the helper text.
        let reference = [(1, 14. / 31.), (2, 23. / 31.), (4, 29. / 31.),
            (5, 1.), (9, 26. / 31.), (12, 18. / 31.), (17, 9. / 31.),
            (22, 4. / 31.), (29, 1. / 31.)];
        for direction in [-1., 1.] {
            let mut bounce = Bounce::default();
            bounce.input(direction * 200., 540., TouchPhase::Moved, ScrollInput::Momentum, now);
            let peak = bounce.displacement(now + Duration::from_millis(80)).abs();
            assert!(peak <= 540. * 0.06);
            assert_eq!(bounce.displacement(now), 0.);
            for (frame, fraction) in reference {
                let actual = bounce.displacement(now + Duration::from_secs_f32(frame as f32 / 60.));
                assert_eq!(actual.signum(), direction);
                assert!((actual.abs() / peak - fraction).abs() < 0.065,
                    "frame {frame}: {} vs {fraction}", actual.abs() / peak);
            }
        }
    }

    #[test]
    fn scroll_bounce_absorbs_momentum_even_after_settling() {
        let now = Instant::now();
        for direction in [-1., 1.] {
            let mut bounce = Bounce::default();
            let mut unforced = Bounce::default();
            for state in [&mut bounce, &mut unforced] {
                state.input(direction * 200., 540., TouchPhase::Moved, ScrollInput::Momentum, now);
            }
            for frame in 1..=120 {
                let time = now + Duration::from_secs_f32(frame as f32 / 60.);
                bounce.input(direction * 100., 540., TouchPhase::Moved, ScrollInput::Momentum, time);
                assert_eq!(bounce.displacement(time), unforced.displacement(time));
                if bounce.displacement(time) == 0. { bounce.clear_motion(); }
            }
            let next_gesture = now + Duration::from_secs(3);
            bounce.input(direction * 80., 540., TouchPhase::Started, ScrollInput::Direct, next_gesture);
            assert_eq!(bounce.displacement(next_gesture).signum(), direction);
        }
    }

    #[test]
    fn scroll_bounce_wheel_burst_returns_and_rearms_after_a_pause() {
        let now = Instant::now();
        let mut bounce = Bounce::default();
        bounce.input(200., 540., TouchPhase::Moved, ScrollInput::Wheel, now);
        let peak = bounce.displacement(now + Duration::from_millis(80));
        for frame in 1..=40 {
            bounce.input(100., 540., TouchPhase::Moved, ScrollInput::Wheel, now + Duration::from_millis(frame * 16));
        }
        assert_eq!(bounce.displacement(now + Duration::from_secs(1)), 0.);
        let next = now + Duration::from_secs(2);
        bounce.input(200., 540., TouchPhase::Moved, ScrollInput::Wheel, next);
        assert!((bounce.displacement(next + Duration::from_millis(80)) - peak).abs() < 0.01);
    }

    #[test]
    fn scroll_bounce_waits_for_momentum_to_reach_the_edge() {
        let now = Instant::now();
        let mut bounce = Bounce::default();
        bounce.input(0., 540., TouchPhase::Moved, ScrollInput::Momentum, now);
        bounce.input(-200., 540., TouchPhase::Moved, ScrollInput::Momentum, now + Duration::from_millis(100));
        assert!(bounce.displacement(now + Duration::from_millis(180)) < 0.);
    }

    #[test]
    fn scroll_bounce_release_keeps_the_existing_return() {
        let now = Instant::now();
        let mut bounce = Bounce::default();
        bounce.input(200., 540., TouchPhase::Moved, ScrollInput::Direct, now);
        let later = now + Duration::from_millis(200);
        let expected = bounce.displacement(later);
        bounce.input(0., 540., TouchPhase::Ended, ScrollInput::Direct, now + Duration::from_millis(48));
        bounce.input(100., 540., TouchPhase::Moved, ScrollInput::Momentum, now + Duration::from_millis(64));
        assert_eq!(bounce.displacement(later), expected);
        bounce.input(0., 540., TouchPhase::Cancelled, ScrollInput::Direct, later);
        assert_eq!(bounce.displacement(later), 0.);
        bounce.input(100., 540., TouchPhase::Moved, ScrollInput::Momentum, later);
        assert_eq!(bounce.displacement(later), 0.);
    }
}
