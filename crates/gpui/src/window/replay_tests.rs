//! What a view drawn again from the last frame has to bring along.

use crate::{
    Context, Entity, IntoElement, Render,
    StyleRefinement, TestAppContext, Window, WindowControlArea, div, prelude::*, px,
};

struct Chrome;

impl Render for Chrome {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        div()
            .id("caption")
            .size_full()
            .window_control_area(WindowControlArea::Drag)
    }
}

struct Shell {
    chrome: Entity<Chrome>,
    renders: usize,
}

impl Render for Shell {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        self.renders += 1;
        div().size_full().child(
            self.chrome
                .clone()
                .cached(StyleRefinement::default().w(px(200.)).h(px(30.))),
        )
    }
}

/// A cached view drawn again from the last frame keeps the window-control
/// areas it painted: the platform hit-tests the caption, drag strips and
/// caption buttons against them.
#[test]
fn a_reused_cached_view_keeps_its_window_control_areas() {
    let mut cx = TestAppContext::single();
    let window = cx.add_window(|_, cx| Shell {
        chrome: cx.new(|_| Chrome),
        renders: 0,
    });
    let controls = |cx: &mut TestAppContext| {
        cx.update_window(window.into(), |_, window, cx| {
            window.draw(cx).clear(cx);
            window
                .rendered_frame
                .window_control_hitboxes
                .iter()
                .map(|(area, hitbox)| (*area, hitbox.bounds))
                .collect::<Vec<_>>()
        })
        .unwrap()
    };
    let first = controls(&mut cx);
    assert_eq!(first.len(), 1);

    // Only the shell is notified, so the chrome is drawn from last frame.
    window
        .update(&mut cx, |_, window, cx| {
            window.reset_frame_work_stats(false);
            cx.notify()
        })
        .unwrap();
    let reused = controls(&mut cx);
    assert!(window.read_with(&cx, |shell, _| shell.renders).unwrap() >= 2);
    let views_reused = window
        .update(&mut cx, |_, window, _| window.frame_work_stats().views_reused)
        .unwrap();
    assert!(views_reused > 0, "the chrome was built again, not reused");
    assert_eq!(reused, first);
}
