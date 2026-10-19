use crate::{App, EntityId, Global, Pixels, TouchPhase, Window, px};
use scheduler::Instant;
use std::{cell::RefCell, rc::Rc};

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

#[derive(Debug, Default)]
struct Bounce {
    position: f32,
    velocity: f32,
    limit: f32,
    last_update: Option<Instant>,
}

impl Bounce {
    const SPRING_FREQUENCY: f32 = 32.;

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
        let displacement = self.sample(now).0;
        if displacement.abs() < 0.1 { 0. } else { displacement }
    }

    fn input(&mut self, excess: f32, extent: f32, phase: TouchPhase, now: Instant) {
        if phase == TouchPhase::Cancelled {
            *self = Self::default();
        } else if phase == TouchPhase::Ended {
            // The spring is already returning; restarting it here would lose its velocity.
            return;
        } else if excess != 0. {
            let (position, velocity) = self.sample(now);
            if phase == TouchPhase::Started || position.abs() < 0.1
                || position.signum() != excess.signum() {
                self.position = 0.;
                self.velocity = 0.;
            } else {
                // Momentum packets must not reset the return to a stationary pull.
                self.position = position;
                self.velocity = velocity;
            }
            self.limit = (extent * 0.15).clamp(12., 80.);
            let pull = self.position * self.limit
                / (self.limit - self.position.abs()).max(self.limit / 9.);
            let pull = (pull + excess * 0.55).clamp(-self.limit * 8., self.limit * 8.);
            self.position = self.limit * pull / (self.limit + pull.abs());
            let maximum_velocity = Self::SPRING_FREQUENCY * self.position.abs();
            self.velocity = self.velocity.clamp(-maximum_velocity, maximum_velocity);
            self.last_update = Some(now);
        } else {
            *self = Self::default();
        }
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
        phase: TouchPhase,
        view: EntityId,
        window: &mut Window,
        cx: &mut App,
    ) {
        window.queue_scroll_bounce(self, excess, viewport_height, phase, view, cx);
    }

    /// Clear feedback after a programmatic scroll, scrollbar drag, or content reset.
    pub fn clear(&self) -> bool {
        let mut state = self.0.borrow_mut();
        let changed = state.position != 0.;
        *state = Bounce::default();
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
        let displacement = self.0.borrow().displacement(Instant::now());
        if (displacement > 0. && offset < px(-0.5))
            || (displacement < 0. && offset > -maximum + px(0.5))
            || displacement == 0.
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
    view: EntityId,
}

impl Window {
    pub(crate) fn queue_scroll_bounce(
        &mut self,
        bounce: &ScrollBounce,
        excess: Pixels,
        extent: Pixels,
        phase: TouchPhase,
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
            bounce: bounce.clone(), excess, extent, phase, view,
        });
    }

    pub(crate) fn apply_scroll_bounce(&mut self, cx: &mut App) {
        if let Some(request) = self.pending_scroll_bounce.take() {
            request.bounce.0.borrow_mut().input(
                request.excess.into(), request.extent.into(), request.phase, Instant::now(),
            );
            cx.notify(request.view);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn scroll_bounce_resists_both_edges_and_settles() {
        let now = Instant::now();
        for direction in [-1., 1.] {
            let mut bounce = Bounce::default();
            bounce.input(direction * 10000., 600., TouchPhase::Moved, now);
            let pulled = bounce.displacement(now);
            assert_eq!(pulled.signum(), direction);
            assert!(pulled.abs() < 80.);
            assert!(bounce.displacement(now + Duration::from_millis(16)).abs() < pulled.abs() * 0.95);
            bounce.input(0., 600., TouchPhase::Ended, now);
            assert!((bounce.displacement(now) - pulled).abs() < 0.01);
            assert!(bounce.displacement(now + Duration::from_millis(100)).abs() < pulled.abs() * 0.2);
            assert_eq!(bounce.displacement(now + Duration::from_secs(1)), 0.);
        }
    }

    #[test]
    fn scroll_bounce_keeps_returning_through_a_momentum_tail() {
        let now = Instant::now();
        for direction in [-1., 1.] {
            let mut bounce = Bounce::default();
            bounce.input(direction * 200., 600., TouchPhase::Started, now);
            let initial = bounce.displacement(now).abs();
            for frame in 1..=20 {
                let time = now + Duration::from_millis(frame * 16);
                let excess = direction * 4. * (-0.25 * frame as f32).exp();
                let before = bounce.displacement(time).abs();
                bounce.input(excess, 600., TouchPhase::Moved, time);
                let after = bounce.displacement(time).abs();
                assert!(after - before < excess.abs() * 0.55 + 0.1);
            }
            assert!(bounce.displacement(now + Duration::from_millis(320)).abs() < initial * 0.02);
            assert_eq!(bounce.displacement(now + Duration::from_millis(600)), 0.);
        }
    }

    #[test]
    fn scroll_bounce_end_does_not_restart_the_return() {
        let now = Instant::now();
        let mut bounce = Bounce::default();
        bounce.input(200., 600., TouchPhase::Moved, now);
        let ended = now + Duration::from_millis(48);
        let later = now + Duration::from_millis(80);
        let position_at_end = bounce.displacement(ended);
        let position_later = bounce.displacement(later);
        bounce.input(0., 600., TouchPhase::Ended, ended);
        assert_eq!(bounce.displacement(ended), position_at_end);
        assert_eq!(bounce.displacement(later), position_later);
    }

    #[test]
    fn scroll_bounce_returns_without_an_end_event_and_cancels() {
        let now = Instant::now();
        let mut bounce = Bounce::default();
        bounce.input(100., 400., TouchPhase::Moved, now);
        assert!(bounce.displacement(now + Duration::from_millis(50)) > 0.);
        assert_eq!(bounce.displacement(now + Duration::from_secs(1)), 0.);
        bounce.input(100., 400., TouchPhase::Moved, now);
        bounce.input(0., 400., TouchPhase::Cancelled, now);
        assert_eq!(bounce.displacement(now), 0.);
        bounce.input(-100., 400., TouchPhase::Moved, now);
        bounce.input(0., 400., TouchPhase::Moved, now);
        assert_eq!(bounce.displacement(now), 0.);
    }
}
