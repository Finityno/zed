use crate::{App, EntityId, Global, Pixels, TouchPhase, Window, px};
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

#[derive(Clone, Debug, Default)]
pub(crate) struct ScrollBounce(Rc<RefCell<Bounce>>);

#[derive(Debug, Default)]
struct Bounce {
    pull: f32,
    limit: f32,
    last_input: Option<Instant>,
    released: bool,
}

impl Bounce {
    fn displacement(&self, now: Instant) -> f32 {
        let Some(last_input) = self.last_input else { return 0.; };
        if self.pull == 0. || self.limit <= 0. { return 0.; }
        let elapsed = now.saturating_duration_since(last_input);
        let delay = if self.released { Duration::ZERO } else { Duration::from_millis(80) };
        let time = elapsed.saturating_sub(delay).as_secs_f32();
        let spring = (1. + 22. * time) * (-22. * time).exp();
        let displacement = self.limit * self.pull / (self.limit + self.pull.abs()) * spring;
        if displacement.abs() < 0.1 { 0. } else { displacement }
    }

    fn input(&mut self, excess: f32, extent: f32, phase: TouchPhase, now: Instant) {
        if phase == TouchPhase::Cancelled {
            *self = Self::default();
        } else if phase == TouchPhase::Ended {
            // Start the return from the current position, even after a gap in input.
            let displacement = self.displacement(now);
            self.pull = if self.limit > displacement.abs() {
                displacement * self.limit / (self.limit - displacement.abs())
            } else { 0. };
            self.last_input = Some(now);
            self.released = true;
        } else if excess != 0. {
            if phase == TouchPhase::Started || self.released || self.displacement(now) == 0.
                || self.pull.signum() != excess.signum()
            {
                self.pull = 0.;
            }
            self.limit = (extent * 0.15).clamp(12., 80.);
            self.pull = (self.pull + excess * 0.55).clamp(-self.limit * 8., self.limit * 8.);
            self.last_input = Some(now);
            self.released = false;
        } else {
            *self = Self::default();
        }
    }
}

impl ScrollBounce {
    pub(crate) fn clear(&self) -> bool {
        let mut state = self.0.borrow_mut();
        let changed = state.pull != 0.;
        *state = Bounce::default();
        changed
    }

    pub(crate) fn offset(
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

    #[test]
    fn scroll_bounce_resists_both_edges_and_settles() {
        let now = Instant::now();
        for direction in [-1., 1.] {
            let mut bounce = Bounce::default();
            bounce.input(direction * 10000., 600., TouchPhase::Moved, now);
            let pulled = bounce.displacement(now);
            assert_eq!(pulled.signum(), direction);
            assert!(pulled.abs() < 80.);
            bounce.input(0., 600., TouchPhase::Ended, now);
            assert!((bounce.displacement(now) - pulled).abs() < 0.01);
            assert!(bounce.displacement(now + Duration::from_millis(100)).abs() < pulled.abs());
            assert_eq!(bounce.displacement(now + Duration::from_secs(1)), 0.);
        }
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
