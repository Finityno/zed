use std::sync::Arc;

use crate::{Bounds, Corners, Pixels, Point, RoundedClipNode, Window};

#[derive(Debug, PartialEq)]
pub(crate) struct RoundedClipRegion {
    pub bounds: Bounds<Pixels>,
    pub corners: Corners<Pixels>,
    pub parent: Option<Arc<Self>>,
}

impl RoundedClipRegion {
    pub fn contains(&self, position: Point<Pixels>) -> bool {
        if !self.bounds.contains(&position) { return false; }
        let offset = position - self.bounds.origin;
        let width = self.bounds.size.width;
        let height = self.bounds.size.height;
        let (radius, horizontal, vertical) = if offset.x < width / 2.0 {
            if offset.y < height / 2.0 {
                (self.corners.top_left, offset.x, offset.y)
            } else {
                (self.corners.bottom_left, offset.x, height - offset.y)
            }
        } else if offset.y < height / 2.0 {
            (self.corners.top_right, width - offset.x, offset.y)
        } else {
            (self.corners.bottom_right, width - offset.x, height - offset.y)
        };
        let inside_corner = horizontal >= radius || vertical >= radius
            || (f32::from(horizontal - radius).powi(2)
                + f32::from(vertical - radius).powi(2)) <= f32::from(radius).powi(2);
        inside_corner && self.parent.as_ref().is_none_or(|parent| parent.contains(position))
    }

    pub(crate) fn depth(&self) -> usize {
        1 + self.parent.as_ref().map_or(0, |parent| parent.depth())
    }

    pub(crate) fn moved_inside(self: &Arc<Self>, delta: Point<Pixels>, inherited_depth: usize) -> Arc<Self> {
        if self.depth() <= inherited_depth { return Arc::clone(self); }
        Arc::new(Self {
            bounds: Bounds { origin: self.bounds.origin + delta, ..self.bounds },
            corners: self.corners,
            parent: self.parent.as_ref().map(|parent| parent.moved_inside(delta, inherited_depth)),
        })
    }

}

impl Window {
    pub(crate) fn install_rounded_clip(&mut self, clip: Option<Arc<RoundedClipRegion>>) {
        fn insert(window: &mut Window, clip: &RoundedClipRegion) -> u32 {
            let parent = clip.parent.as_ref().map_or(0, |parent| insert(window, parent));
            let scale_factor = window.scale_factor();
            let bounds = clip.bounds.scale(scale_factor);
            window.next_frame.scene.rounded_clips.push(RoundedClipNode {
                bounds,
                corner_radii: clip.corners.map(|radius| radius.scale(scale_factor)),
                parent,
                transition: 0,
                base_origin: bounds.origin,
            });
            window.next_frame.scene.rounded_clips.len() as u32
        }
        if self.invalidator.is_painting() {
            self.next_frame.scene.current_rounded_clip = clip.as_ref().map_or(0, |clip| insert(self, clip));
        }
        self.rounded_clip = clip;
    }

    /// Clip a subtree to rounded bounds during both prepaint and paint.
    ///
    /// Nested clips intersect. The same shape clips hitboxes, and its device-space
    /// mask applies to all scene primitives without painting a surrounding color.
    pub fn with_rounded_clip<R>(
        &mut self,
        bounds: Bounds<Pixels>,
        corners: impl Into<Corners<Pixels>>,
        draw: impl FnOnce(&mut Self) -> R,
    ) -> R {
        self.invalidator.debug_assert_paint_or_prepaint();
        let corners = corners.into().map(|radius| (*radius).max(Pixels::ZERO))
            .clamp_radii_for_quad_size(bounds.size);
        if corners == Corners::default() {
            return self.with_content_mask(Some(crate::ContentMask { bounds }), draw);
        }
        let previous = self.rounded_clip.take();
        self.rounded_clip = Some(Arc::new(RoundedClipRegion {
            bounds, corners, parent: previous.clone(),
        }));
        let parent = self.next_frame.scene.current_rounded_clip;
        if self.invalidator.is_painting() {
            let scale_factor = self.scale_factor();
            let device_bounds = bounds.scale(scale_factor);
            self.next_frame.scene.rounded_clips.push(RoundedClipNode {
                bounds: device_bounds,
                corner_radii: corners.map(|radius| radius.scale(scale_factor)),
                parent,
                transition: self.next_frame.scene.current_transition(),
                base_origin: device_bounds.origin,
            });
            self.next_frame.scene.current_rounded_clip = self.next_frame.scene.rounded_clips.len() as u32;
        }
        let result = self.with_content_mask(Some(crate::ContentMask { bounds }), draw);
        self.next_frame.scene.current_rounded_clip = parent;
        self.rounded_clip = previous;
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AppContext, Context, InteractiveElement, IntoElement, ParentElement, Render, Styled, TestAppContext, div, point, px, size};

    struct ClippedCard { radius: Pixels }

    impl Render for ClippedCard {
        fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
            div().size(px(100.0)).rounded(self.radius).overflow_hidden().occlude()
                .child(div().size_full().bg(crate::red()).occlude())
        }
    }

    #[test]
    fn rounded_clip_style_applies_to_paint_and_hitboxes_and_updates_with_radius() -> anyhow::Result<()> {
        let mut cx = TestAppContext::single();
        let window = cx.add_window(|_, _| ClippedCard { radius: px(20.0) });
        cx.update_window(window.into(), |_, window, cx| {
            window.draw(cx).clear(cx);
            let frame = &window.rendered_frame;
            assert!(frame.scene.quads.iter().any(|quad| quad.rounded_clip != 0));
            assert!(frame.hit_test(point(px(1.0), px(1.0))).ids.is_empty());
            assert!(!frame.hit_test(point(px(50.0), px(50.0))).ids.is_empty());
        })?;
        window.update(&mut cx, |card, _, cx| {
            card.radius = px(0.0);
            cx.notify();
        })?;
        cx.update_window(window.into(), |_, window, cx| {
            window.draw(cx).clear(cx);
            assert!(window.rendered_frame.scene.rounded_clips.is_empty());
            assert!(!window.rendered_frame.hit_test(point(px(1.0), px(1.0))).ids.is_empty());
        })?;
        Ok(())
    }

    #[test]
    fn rounded_clip_hit_testing_intersects_nested_corners() {
        let outer = Arc::new(RoundedClipRegion {
            bounds: Bounds::new(point(px(0.0), px(0.0)), size(px(100.0), px(80.0))),
            corners: Corners::all(px(20.0)),
            parent: None,
        });
        assert!(!outer.contains(point(px(1.0), px(1.0))));
        assert!(outer.contains(point(px(20.0), px(1.0))));
        assert!(outer.contains(point(px(10.0), px(10.0))));
        let inner = RoundedClipRegion {
            bounds: Bounds::new(point(px(0.0), px(0.0)), size(px(50.0), px(40.0))),
            corners: Corners { top_left: px(0.0), top_right: px(10.0),
                bottom_left: px(0.0), bottom_right: px(0.0) },
            parent: Some(outer),
        };
        assert!(!inner.contains(point(px(1.0), px(1.0))));
        assert!(!inner.contains(point(px(49.0), px(1.0))));
        assert!(inner.contains(point(px(40.0), px(1.0))));
        assert!(inner.contains(point(px(1.0), px(39.0))));
    }
}
