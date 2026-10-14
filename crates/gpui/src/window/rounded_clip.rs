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

    pub(crate) fn moved(&self, delta: Point<Pixels>) -> Arc<Self> {
        Arc::new(Self {
            bounds: Bounds { origin: self.bounds.origin + delta, ..self.bounds },
            corners: self.corners,
            parent: self.parent.as_ref().map(|parent| parent.moved(delta)),
        })
    }
}

impl Window {
    pub(crate) fn install_rounded_clip(&mut self, clip: Option<Arc<RoundedClipRegion>>) {
        fn insert(window: &mut Window, clip: &RoundedClipRegion) -> u32 {
            let parent = clip.parent.as_ref().map_or(0, |parent| insert(window, parent));
            window.next_frame.scene.rounded_clips.push(RoundedClipNode {
                bounds: clip.bounds.scale(window.scale_factor()),
                corner_radii: clip.corners.map(|radius| radius.scale(window.scale_factor())),
                parent,
                padding: [0; 3],
            });
            window.next_frame.scene.rounded_clips.len() as u32
        }
        self.next_frame.scene.current_rounded_clip = clip.as_ref().map_or(0, |clip| insert(self, clip));
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
        let corners = corners.into().map(|radius| radius.max(Pixels::ZERO))
            .clamp_radii_for_quad_size(bounds.size);
        if corners == Corners::default() {
            return self.with_content_mask(Some(crate::ContentMask { bounds }), draw);
        }
        let previous = self.rounded_clip.take();
        self.rounded_clip = Some(Arc::new(RoundedClipRegion {
            bounds, corners, parent: previous.clone(),
        }));
        let parent = self.next_frame.scene.current_rounded_clip;
        self.next_frame.scene.rounded_clips.push(RoundedClipNode {
            bounds: bounds.scale(self.scale_factor()),
            corner_radii: corners.map(|radius| radius.scale(self.scale_factor())),
            parent,
            padding: [0; 3],
        });
        self.next_frame.scene.current_rounded_clip = self.next_frame.scene.rounded_clips.len() as u32;
        let result = self.with_content_mask(Some(crate::ContentMask { bounds }), draw);
        self.next_frame.scene.current_rounded_clip = parent;
        self.rounded_clip = previous;
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{point, px, size};

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
