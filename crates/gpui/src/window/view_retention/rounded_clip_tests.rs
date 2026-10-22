use crate::{
    AppContext, Context, Entity, FocusHandle, FrameWorkStats, IntoElement, Render, TestAppContext,
    Window, WindowHandle, div, point, prelude::*, px,
};
use std::{cell::Cell, rc::Rc};

struct ClippedChild {
    focus: FocusHandle,
    revision: usize,
    width: f32,
}

impl Render for ClippedChild {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        div()
            .id("clipped-child")
            .track_focus(&self.focus)
            .w(px(self.width))
            .h(px(120.))
            .rounded(px(8.))
            .overflow_hidden()
            .child(div().size_full().bg(if self.revision % 2 == 0 { crate::red() } else { crate::blue() }).occlude())
    }
}

struct ClippedSiblings {
    children: [Entity<ClippedChild>; 2],
    renders: Rc<Cell<usize>>,
}

impl Render for ClippedSiblings {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        self.renders.set(self.renders.get() + 1);
        div().flex().gap(px(10.)).children(self.children.iter().map(|child| {
            div().flex_none().w(px(100.)).h(px(120.)).rounded(px(30.)).overflow_hidden()
                .child(div().size_full().rounded(px(18.)).overflow_hidden().child(child.clone()))
        }))
    }
}

fn snapshot(window: &Window) -> Vec<String> {
    let frame = &window.rendered_frame;
    let mut result: Vec<_> = frame.scene.quads.iter().map(|quad| {
        let mut chain = Vec::new();
        let mut index = quad.rounded_clip;
        while index != 0 {
            let clip = &frame.scene.rounded_clips[index as usize - 1];
            chain.push(format!("{:?} {:?}", clip.bounds, clip.corner_radii));
            index = clip.parent;
        }
        let mut quad = *quad;
        quad.rounded_clip = 0;
        format!("quad {quad:?} clips {chain:?}")
    }).collect();
    result.sort_unstable();
    result.extend(frame.hitboxes.iter().map(|hitbox| {
        format!("hitbox {:?} {:?} {:?}", hitbox.bounds, hitbox.content_mask, hitbox.rounded_clip)
    }));
    for horizontal in [0., 110.] {
        for (x, y, expected) in [(1., 1., 0), (7., 7., 0), (50., 1., 1), (50., 60., 1), (99., 119., 0)] {
            let hits = frame.hit_test(point(px(horizontal + x), px(y))).ids.len();
            assert_eq!(hits, expected, "hit at ({}, {y})", horizontal + x);
            result.push(format!("probe {horizontal} {x} {y}: {hits}"));
        }
    }
    result
}

fn draw(
    cx: &mut TestAppContext,
    handle: WindowHandle<ClippedSiblings>,
    refresh: bool,
) -> anyhow::Result<(Vec<String>, FrameWorkStats)> {
    cx.update_window(handle.into(), |_, window, cx| {
        window.reset_frame_work_stats(false);
        if refresh { window.refresh(); }
        window.draw(cx).clear(cx);
        (snapshot(window), window.frame_work_stats())
    })
}

#[test]
fn spliced_sibling_views_keep_nested_rounded_paint_and_hit_clips() -> anyhow::Result<()> {
    let mut cx = TestAppContext::single();
    cx.update(|cx| cx.set_view_retention(true));
    let renders = [Rc::new(Cell::new(0)), Rc::new(Cell::new(0))];
    let windows = renders.each_ref().map(|renders| {
        let renders = Rc::clone(renders);
        cx.add_window(move |_, cx| ClippedSiblings {
            children: [(); 2].map(|_| cx.new(|cx| ClippedChild {
                focus: cx.focus_handle(), revision: 0, width: 100.,
            })),
            renders,
        })
    });
    for _ in 0..2 {
        assert_eq!(draw(&mut cx, windows[0], false)?.0, draw(&mut cx, windows[1], true)?.0);
    }
    let parent_renders = renders[0].get();
    for child_index in [1, 0, 1, 0] {
        for handle in windows {
            let child = handle.read_with(&cx, |siblings, _| siblings.children[child_index].clone())?;
            child.update(&mut cx, |child, cx| { child.revision += 1; cx.notify(); });
            handle.update(&mut cx, |_, window, cx| {
                let focus = child.read(cx).focus.clone();
                focus.focus(window, cx);
            })?;
        }
        let (retained, work) = draw(&mut cx, windows[0], false)?;
        assert_eq!(retained, draw(&mut cx, windows[1], true)?.0);
        assert!(work.views_spliced > 0, "{work:?}");
        assert_eq!(renders[0].get(), parent_renders, "the parent must stay retained");
    }
    for handle in windows {
        let child = handle.read_with(&cx, |siblings, _| siblings.children[0].clone())?;
        child.update(&mut cx, |child, cx| { child.width = 104.; cx.notify(); });
    }
    let (retained, _) = draw(&mut cx, windows[0], false)?;
    assert_eq!(retained, draw(&mut cx, windows[1], true)?.0);
    assert!(renders[0].get() > parent_renders, "changed child layout must roll the splice back");
    Ok(())
}
