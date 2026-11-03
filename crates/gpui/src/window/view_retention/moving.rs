//! Views drawn again from the last frame somewhere else.
//!
//! A view whose record matches in everything but where it is drawn (it
//! scrolled, or something above it grew) is drawn again from the last frame
//! moved, instead of being built: its hitboxes and primitives are copied
//! shifted by how far it moved and clipped by the content mask it is drawn
//! in now, the records nested in it move with it, and so does what its
//! elements wrote of their position as they prepainted (see
//! [`PositionedState`]) and what was registered to move with it while it
//! prepainted or painted (see [`Window::on_replayed_at_offset`]).
//!
//! The copy is the frame drawn from scratch when:
//! - the view is the same size, and inherits the same text style, opacity,
//!   rem size, image cache and groups;
//! - it moved by a whole number of device pixels, so that glyphs land on the
//!   same subpixel offsets and pixel snapping comes out the same;
//! - if it drew anything by whether a hitbox was hovered, the pointer is
//!   over neither where it was nor where it is, so moving it hovers nothing
//!   anew;
//! - it lies wholly inside the content mask around it, in the last frame and
//!   in this one. Whatever was culled against the mask, or laid out by how
//!   much of it was visible (lines of a long text, a sticky header), would
//!   otherwise be missing or misplaced once moved;
//! - nothing it drew was clipped by the mask around it, only by masks of its
//!   own, so that every mask moves with what it clips: on each side where a
//!   hitbox or primitive reached past its mask, the mask's edge is not the
//!   one around the view. With list row replay on (see [`super::list_rows`]),
//!   a view partly out of view, before or now, is drawn moved too where no
//!   more of it shows than showed, moved, if nothing in it read the content
//!   mask as it prepainted ([`Window::content_mask`]): what was culled stays
//!   out of view, and every mask edge that was the one around it is moved to
//!   the one around it now;
//! - it deferred nothing (an anchored popover places itself against the
//!   window, which does not move with it), holds no input handler (the
//!   platform asks the focused input for bounds), was not painted inside a
//!   text shimmer from around it (whose band an ancestor placed), and wrote
//!   no entity, global or versioned state as it was prepainted or painted,
//!   which could be where it was.
//!
//! Mouse listeners are not moved. They were registered with the bounds the
//! view had when it was built, and a listener may compare those with an
//! event's position, or keep an event's position for later, so neither
//! moving the bounds nor the event is right for every listener. A view drawn
//! moved is instead unsettled until it is built again: before the window
//! dispatches any input but a scroll wheel or a key, it draws a frame in
//! which every unsettled view is built ([`Window::settle_moved_views`]), and
//! it asks for such a frame once views stop moving, so that state outside
//! the frame they wrote their position into catches up.

use super::{ViewContext, ViewRecord};
use crate::{App, Bounds, ContentMask, Hitbox, Pixels, Point, ScaledPixels, Window, scene::SceneMove};
use gpui_util::ResultExt;
use std::{rc::Rc, time::Duration};

/// How long after the last frame that drew a view moved the views drawn
/// moved are built, catching up what they wrote outside the frame.
const SETTLE_AFTER: Duration = Duration::from_millis(200);

/// State outside the frame an element writes its position into as it
/// prepaints, for others to read later: a scroll handle's bounds, a list's,
/// a text layout's. A view drawn again elsewhere from the last frame is not
/// prepainted, so this is moved with it. Noted with
/// [`Window::note_positioned_state`].
pub(crate) trait PositionedState {
    /// Moves the position the state holds by `by`.
    fn translate(&self, by: Point<Pixels>);
}

/// How far a view drawn again from the last frame moved, and the content
/// mask around it then and now.
///
/// A mask something in the view was clipped by is the mask around the view
/// intersected with masks the view pushed itself. Moved, the edges the view
/// pushed move with it and the edges of the mask around it are the new
/// mask's; an edge that is both, by chance, is taken for the mask around
/// the view, which clips nothing the view drew (it lies inside that mask).
#[derive(Clone, Copy, Debug)]
pub(crate) struct ViewMove {
    pub(crate) delta: Point<Pixels>,
    pub(crate) old_outer: ContentMask<Pixels>,
    pub(crate) new_outer: ContentMask<Pixels>,
    pub(crate) inherited_clip_depth: usize,
}

impl ViewMove {
    /// `mask`, which clipped something the view drew, as it clips it now.
    pub(crate) fn mask_of(&self, mask: &ContentMask<Pixels>) -> ContentMask<Pixels> {
        let [left, top, right, bottom] = rebase_edges(
            edges(&mask.bounds),
            edges(&self.old_outer.bounds),
            edges(&self.new_outer.bounds),
            [self.delta.x.0, self.delta.y.0],
        );
        ContentMask {
            bounds: Bounds {
                origin: crate::point(Pixels(left), Pixels(top)),
                size: crate::size(Pixels(right - left), Pixels(bottom - top)),
            },
        }
    }

    pub(crate) fn hitbox(&self, hitbox: &Hitbox) -> Hitbox {
        Hitbox {
            id: hitbox.id,
            bounds: Bounds {
                origin: hitbox.bounds.origin + self.delta,
                size: hitbox.bounds.size,
            },
            content_mask: self.mask_of(&hitbox.content_mask),
            rounded_clip: hitbox.rounded_clip.as_ref().map(|clip| clip.moved_inside(self.delta, self.inherited_clip_depth)),
            behavior: hitbox.behavior,
        }
    }

    /// The move in device pixels, for primitives, whose masks are the
    /// window's masks grown out to whole device pixels.
    pub(crate) fn scaled(&self, window: &Window) -> SceneMove {
        SceneMove {
            delta: {
                let delta = self.delta.scale(window.scale_factor());
                crate::point(ScaledPixels(delta.x.0.round()), ScaledPixels(delta.y.0.round()))
            },
            old_outer: scaled_edges(&window.cover_bounds(self.old_outer.bounds)),
            new_outer: scaled_edges(&window.cover_bounds(self.new_outer.bounds)),
            inherited_clip_depth: self.inherited_clip_depth,
            inherited_clip: window.next_frame.scene.current_rounded_clip,
        }
    }

    fn context(&self, context: &ViewContext) -> ViewContext {
        ViewContext {
            bounds: Bounds {
                origin: context.bounds.origin + self.delta,
                size: context.bounds.size,
            },
            content_mask: self.mask_of(&context.content_mask),
            rounded_clip: context.rounded_clip.as_ref().map(|clip| clip.moved_inside(self.delta, self.inherited_clip_depth)),
            text_style: context.text_style.clone(),
            opacity: context.opacity,
            rem_size: context.rem_size,
            image_cache: context.image_cache,
            glass_content: context.glass_content,
        }
    }

    /// A record copied along with a view drawn moved, as it is in this frame.
    pub(super) fn move_record(&self, record: &ViewRecord) -> Rc<ViewContext> {
        Rc::new(self.context(&record.context))
    }
}

/// The edges of a mask that clipped something a view drew, as they are once
/// the view moved by `delta` from inside `old_outer` to inside `new_outer`:
/// see [`ViewMove`]. An empty result has its far edges at its near ones.
pub(crate) fn rebase_edges(
    mask: [f32; 4],
    old_outer: [f32; 4],
    new_outer: [f32; 4],
    [dx, dy]: [f32; 2],
) -> [f32; 4] {
    const EPSILON: f32 = 1e-3;
    let side = |index: usize, delta: f32| {
        if (mask[index] - old_outer[index]).abs() < EPSILON {
            new_outer[index]
        } else {
            mask[index] + delta
        }
    };
    let left = side(0, dx).max(new_outer[0]);
    let top = side(1, dy).max(new_outer[1]);
    let right = side(2, dx).min(new_outer[2]).max(left);
    let bottom = side(3, dy).min(new_outer[3]).max(top);
    [left, top, right, bottom]
}

/// Whether a box whose part inside `mask` was drawn was clipped only by masks
/// of the view's own, not by `outer`, the mask around the view: on each side
/// where the box reaches past `mask`, `mask`'s edge is not `outer`'s.
pub(crate) fn clipped_only_inside(extent: [f32; 4], mask: [f32; 4], outer: [f32; 4]) -> bool {
    const EPSILON: f32 = 1e-3;
    let [left, top, right, bottom] = extent;
    let [mask_left, mask_top, mask_right, mask_bottom] = mask;
    let [outer_left, outer_top, outer_right, outer_bottom] = outer;
    let outer_edge = |mask: f32, outer: f32| (mask - outer).abs() < EPSILON;
    !((left < mask_left - EPSILON && outer_edge(mask_left, outer_left))
        || (top < mask_top - EPSILON && outer_edge(mask_top, outer_top))
        || (right > mask_right + EPSILON && outer_edge(mask_right, outer_right))
        || (bottom > mask_bottom + EPSILON && outer_edge(mask_bottom, outer_bottom)))
}

/// The edges of `bounds`, left, top, right and bottom.
pub(crate) fn edges(bounds: &Bounds<Pixels>) -> [f32; 4] {
    let (left, top) = (bounds.origin.x.0, bounds.origin.y.0);
    [left, top, left + bounds.size.width.0, top + bounds.size.height.0]
}

/// The edges of scaled `bounds`, as [`edges`].
pub(crate) fn scaled_edges(bounds: &Bounds<ScaledPixels>) -> [f32; 4] {
    let (left, top) = (bounds.origin.x.0, bounds.origin.y.0);
    [left, top, left + bounds.size.width.0, top + bounds.size.height.0]
}

fn inside(inner: [f32; 4], outer: [f32; 4]) -> bool {
    const EPSILON: f32 = 1e-3;
    inner[0] >= outer[0] - EPSILON
        && inner[1] >= outer[1] - EPSILON
        && inner[2] <= outer[2] + EPSILON
        && inner[3] <= outer[3] + EPSILON
}

/// Whether `position` hits `hitbox`, as the window's hit test asks.
fn hitbox_contains(hitbox: &Hitbox, position: Point<Pixels>) -> bool {
    hitbox.bounds.intersect(&hitbox.content_mask.bounds).contains(&position)
        && hitbox
            .rounded_clip
            .as_ref()
            .is_none_or(|clip| clip.contains(position))
}

fn intersection(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    let left = a[0].max(b[0]);
    let top = a[1].max(b[1]);
    [left, top, a[2].min(b[2]).max(left), a[3].min(b[3]).max(top)]
}

fn is_empty(extent: [f32; 4]) -> bool {
    const EPSILON: f32 = 1e-3;
    extent[2] - extent[0] < EPSILON || extent[3] - extent[1] < EPSILON
}

fn whole_device_pixels(delta: f32) -> bool {
    (delta - delta.round()).abs() < 1e-3
}

/// Why a view whose record did not match where it is drawn now was not
/// drawn again moved, for [`super::culprits`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum MoveRefusal {
    /// The frame builds every view drawn moved since it was last built.
    Settling,
    /// `GPUI_RETAINED_VIEW_MOVES=0`.
    MovesOff,
    /// It said so, wrote state as it was drawn, or a view nested in it did.
    StaysPut,
    /// It was painted inside a text shimmer from around it.
    InheritedShimmer,
    InheritedRoundedClip,
    SizeChanged,
    OpacityChanged,
    RemSizeChanged,
    ImageCacheChanged,
    GlassModeChanged,
    TextStyleChanged,
    /// A group it resolved outside it resolves to another hitbox.
    GroupsChanged,
    /// It moved by a fraction of a device pixel.
    NotWholeDevicePixels,
    /// It drew by hovers, and the pointer is over where it was or is (with
    /// list row replay on: moving it moves a hitbox it drew by under the
    /// pointer, or out from under it).
    UnderPointer,
    /// With list row replay on: it drew by a hover that was not answered by
    /// whether the pointer was over the hitbox (a hitbox above it, keyboard
    /// input, a captured pointer).
    HoveredOtherwise,
    /// It reached past the content mask around it in the last frame.
    OutsideMaskBefore,
    /// It reaches past the content mask around it now.
    OutsideMaskNow,
    /// Partly out of view, before or now, it read the content mask as it
    /// prepainted, and may have drawn by how much of it showed.
    ReadMask,
    /// Partly out of view, it drew past its bounds, where what was culled
    /// may come into view.
    DrawnOutside,
    DeferredDraw,
    InputHandler,
    /// A hitbox of it was clipped by the mask around it.
    HitboxClippedByOuterMask,
    /// It reached past the mask it was painted in.
    OutsidePaintMask,
    /// A primitive of it was clipped by the mask around it.
    PrimitiveClippedByOuterMask,
    /// Accessibility is active and it added a synthetic accessibility node
    /// with bounds of its own, which its element's code placed and nothing
    /// says moves with the element.
    PlacedAccessibilityNode,
}

/// A callback registered with [`Window::on_replayed_at_offset`].
struct ReplayedAtOffset(Box<dyn Fn(Point<Pixels>)>);

impl PositionedState for ReplayedAtOffset {
    fn translate(&self, by: Point<Pixels>) {
        (self.0)(by)
    }
}

impl Window {
    /// Registers `moved` to be called with how far the view being drawn
    /// moved, whenever it is drawn again from this frame somewhere else
    /// rather than built: for state outside the frame that holds window
    /// positions of what the view draws (a registry of text segments for
    /// selection, the bounds of an anchor), recorded as the view prepaints
    /// or paints, to move with it instead of going stale.
    ///
    /// Call it while prepainting or painting, next to where the positions
    /// are recorded. It is called once for each frame the view is drawn
    /// moved, with how far it moved since the frame before, and called
    /// again with the opposite offset when a prepaint that moved it is
    /// rolled back. A view built again is prepainted and painted afresh,
    /// and registers anew; one drawn again where it was calls nothing.
    /// Without view retention, or outside an entity view, nothing is
    /// registered. Mouse listeners are not moved; a view drawn moved is
    /// built again before the window dispatches input other than a scroll
    /// wheel or a key.
    pub fn on_replayed_at_offset(&mut self, moved: impl Fn(Point<Pixels>) + 'static) {
        self.invalidator.debug_assert_paint_or_prepaint();
        if self.view_retention.view_stack.is_empty() {
            return;
        }
        let state: Rc<dyn PositionedState> = Rc::new(ReplayedAtOffset(Box::new(moved)));
        if self.invalidator.is_painting() {
            self.next_frame.painted_positions.push(state);
        } else {
            self.next_frame
                .positioned_states
                .push((state, Point::default()));
        }
    }

    /// Notes, as an element prepaints, that `state` holds where it is. Only
    /// kept inside a retained view, which is what can be drawn moved.
    pub(crate) fn note_positioned_state<S: PositionedState + 'static>(
        &mut self,
        state: impl FnOnce() -> Rc<S>,
    ) {
        if !self.view_retention.view_stack.is_empty() {
            self.next_frame
                .positioned_states
                .push((state(), Point::default()));
        }
    }

    /// How the view last frame's record `previous` stands for can be drawn
    /// again at `bounds` from that record, moved, if it can.
    pub(super) fn view_move(
        &self,
        previous: usize,
        bounds: Bounds<Pixels>,
    ) -> Result<ViewMove, MoveRefusal> {
        let retention = &self.view_retention;
        if retention.settling {
            return Err(MoveRefusal::Settling);
        }
        if !retention.moves_enabled {
            return Err(MoveRefusal::MovesOff);
        }
        let record = &self.rendered_frame.retained_views.records[previous];
        let context = &record.context;
        let refusal = if record.stays_put {
            Some(MoveRefusal::StaysPut)
        } else if record.paint_context.shimmer.is_some() {
            Some(MoveRefusal::InheritedShimmer)
        } else if context.bounds.size != bounds.size {
            Some(MoveRefusal::SizeChanged)
        } else if context.opacity != self.element_opacity {
            Some(MoveRefusal::OpacityChanged)
        } else if context.rem_size != self.rem_size() {
            Some(MoveRefusal::RemSizeChanged)
        } else if context.image_cache != self.inherited_image_cache() {
            Some(MoveRefusal::ImageCacheChanged)
        } else if context.glass_content != self.glass_content {
            Some(MoveRefusal::GlassModeChanged)
        } else if context.text_style != self.text_style() {
            Some(MoveRefusal::TextStyleChanged)
        } else if !self.groups_unchanged(record) {
            Some(MoveRefusal::GroupsChanged)
        } else {
            None
        };
        if let Some(refusal) = refusal {
            return Err(refusal);
        }
        let delta = bounds.origin - context.bounds.origin;
        let scale_factor = self.scale_factor();
        let scaled = delta.scale(scale_factor);
        if !whole_device_pixels(scaled.x.0) || !whole_device_pixels(scaled.y.0) {
            return Err(MoveRefusal::NotWholeDevicePixels);
        }
        // With list row replay on, rows are scrolled by whole device pixels
        // but may sit between them, where what they draw snaps to the nearest
        // one: moved by a whole number of them, a view between two lands
        // where laying it out there might round the other way.
        if retention.shrinking_moves {
            let old = context.bounds.origin.scale(scale_factor);
            let new = bounds.origin.scale(scale_factor);
            if ![old.x.0, old.y.0, new.x.0, new.y.0]
                .into_iter()
                .all(whole_device_pixels)
            {
                return Err(MoveRefusal::NotWholeDevicePixels);
            }
        }
        // Exactly the whole device pixels it moved by: two layouts that place
        // it by different sums of the same sizes can differ in the last bit,
        // and moved by that, what it drew would overlap what is drawn beside
        // it where, built, it only touches.
        let delta = crate::point(
            Pixels(scaled.x.0.round() / scale_factor),
            Pixels(scaled.y.0.round() / scale_factor),
        );
        if self.rounded_clip != context.rounded_clip {
            return Err(MoveRefusal::InheritedRoundedClip);
        }
        let mask = self.current_content_mask();
        let outer = edges(&context.content_mask.bounds);
        let outside = if !inside(edges(&context.bounds), outer) {
            Some(MoveRefusal::OutsideMaskBefore)
        } else if !inside(edges(&bounds), edges(&mask.bounds)) {
            Some(MoveRefusal::OutsideMaskNow)
        } else {
            None
        };
        // Partly out of view, it is drawn moved only where no more of it
        // shows than showed: what was culled stays out of view, and what was
        // clipped by the mask around it is clipped by the one around it now.
        let shrinking = match outside {
            None => false,
            Some(refusal) => {
                let showed = intersection(
                    edges(&context.bounds),
                    intersection(outer, edges(&record.paint_mask.bounds)),
                );
                let shows = intersection(edges(&bounds), edges(&mask.bounds));
                let showed_moved = [
                    showed[0] + delta.x.0,
                    showed[1] + delta.y.0,
                    showed[2] + delta.x.0,
                    showed[3] + delta.y.0,
                ];
                let shrinks = retention.shrinking_moves
                    && !record.read_mask
                    && (is_empty(shows) || (!is_empty(showed) && inside(shows, showed_moved)));
                if !shrinks {
                    return Err(if record.read_mask && retention.shrinking_moves {
                        MoveRefusal::ReadMask
                    } else {
                        refusal
                    });
                }
                true
            }
        };
        // What it drew by whether its hitboxes were hovered was checked
        // against where they were; moved under the pointer, or out from
        // under it, they may be hovered otherwise.
        if !record.hovers.is_empty() {
            let pointer = self.mouse_position;
            let hitboxes = &self.rendered_frame.hitboxes
                [record.prepaint_range.start.hitboxes_index..record.prepaint_range.end.hitboxes_index];
            let under_pointer = if retention.shrinking_moves {
                // With list row replay on, only a hover that would come out
                // otherwise keeps it in place: one whose hitbox it moves under
                // or out from under the pointer, or which was not answered
                // by whether the pointer was over it (another hitbox above
                // it, keyboard input). What is drawn above it now is checked
                // again as it paints.
                let moving = ViewMove {
                    delta,
                    old_outer: context.content_mask,
                    new_outer: mask,
                    inherited_clip_depth: self.rounded_clip.as_ref().map_or(0, |clip| clip.depth()),
                };
                let mut refusal = None;
                for hover in record.hovers.iter() {
                    let Some(hitbox) = hitboxes.iter().find(|hitbox| hitbox.id == hover.hitbox)
                    else {
                        continue;
                    };
                    let before = hitbox_contains(hitbox, pointer);
                    if before != hover.hovered {
                        refusal = Some(MoveRefusal::HoveredOtherwise);
                        break;
                    }
                    if hitbox_contains(&moving.hitbox(hitbox), pointer) != before {
                        refusal = Some(MoveRefusal::UnderPointer);
                        break;
                    }
                }
                if let Some(refusal) = refusal {
                    return Err(refusal);
                }
                false
            } else {
                hitboxes.iter().any(|hitbox| {
                    let moved = Bounds {
                        origin: hitbox.bounds.origin + delta,
                        size: hitbox.bounds.size,
                    };
                    hitbox.bounds.contains(&pointer) || moved.contains(&pointer)
                })
            };
            if under_pointer {
                return Err(MoveRefusal::UnderPointer);
            }
        }
        let prepaint = &record.prepaint_range;
        let paint = &record.paint_range;
        if prepaint.start.deferred_draws_index != prepaint.end.deferred_draws_index {
            return Err(MoveRefusal::DeferredDraw);
        }
        if paint.start.input_handlers_index != paint.end.input_handlers_index {
            return Err(MoveRefusal::InputHandler);
        }
        if self.a11y_placed_leaf_in(prepaint.start.a11y_ops_index..prepaint.end.a11y_ops_index) {
            return Err(MoveRefusal::PlacedAccessibilityNode);
        }
        let frame = &self.rendered_frame;
        if shrinking
            && !frame.scene.drawn_inside(
                paint.start.scene_index..paint.end.scene_index,
                scaled_edges(&self.cover_bounds(context.bounds)),
            )
        {
            return Err(MoveRefusal::DrawnOutside);
        }
        if shrinking {
            return Ok(ViewMove {
                delta,
                old_outer: context.content_mask,
                new_outer: mask,
                inherited_clip_depth: self.rounded_clip.as_ref().map_or(0, |clip| clip.depth()),
            });
        }
        let hitboxes_clipped_inside = frame.hitboxes
            [prepaint.start.hitboxes_index..prepaint.end.hitboxes_index]
            .iter()
            .all(|hitbox| {
                clipped_only_inside(
                    edges(&hitbox.bounds),
                    edges(&hitbox.content_mask.bounds),
                    outer,
                )
            });
        if !hitboxes_clipped_inside {
            return Err(MoveRefusal::HitboxClippedByOuterMask);
        }
        if !inside(edges(&context.bounds), edges(&record.paint_mask.bounds)) {
            return Err(MoveRefusal::OutsidePaintMask);
        }
        if !frame.scene.clipped_only_inside(
            paint.start.scene_index..paint.end.scene_index,
            scaled_edges(&self.cover_bounds(record.paint_mask.bounds)),
        ) {
            return Err(MoveRefusal::PrimitiveClippedByOuterMask);
        }
        Ok(ViewMove {
            delta,
            old_outer: context.content_mask,
            new_outer: mask,
            inherited_clip_depth: self.rounded_clip.as_ref().map_or(0, |clip| clip.depth()),
        })
    }

    /// Draws a frame in which every view drawn moved since it was last built
    /// is built, if there is one, before the window dispatches `input`; see
    /// the module documentation for why.
    pub(crate) fn settle_moved_views(&mut self, cx: &mut App) {
        if !self.rendered_frame.retained_views.unsettled {
            return;
        }
        self.view_retention.settle_requested = true;
        self.draw(cx).clear(cx);
    }

    /// Asks, once views stopped moving for a while, for a frame that builds
    /// the views drawn moved. One task waits for that, rearming itself while
    /// views keep moving, rather than one per frame.
    pub(super) fn schedule_settle_frame(&mut self, cx: &App) {
        let retention = &mut self.view_retention;
        if !std::mem::take(&mut retention.moved_this_frame) {
            return;
        }
        let now = cx.background_executor().now();
        retention.last_moved_at = now;
        if retention.settle_frame.is_some() {
            return;
        }
        let task = self.spawn(cx, async move |cx| {
            let mut wait = SETTLE_AFTER;
            loop {
                cx.background_executor().timer(wait).await;
                let rearm = cx
                    .update(|window, cx| {
                        let still_for = cx
                            .background_executor()
                            .now()
                            .saturating_duration_since(window.view_retention.last_moved_at);
                        if still_for < SETTLE_AFTER {
                            return Some(SETTLE_AFTER - still_for);
                        }
                        window.view_retention.settle_frame = None;
                        if window.rendered_frame.retained_views.unsettled {
                            window.view_retention.settle_requested = true;
                            window.invalidator.set_dirty(true);
                        }
                        None
                    })
                    .log_err()
                    .flatten();
                match rearm {
                    Some(remaining) => wait = remaining,
                    None => break,
                }
            }
        });
        self.view_retention.settle_frame = Some(task);
    }
}
