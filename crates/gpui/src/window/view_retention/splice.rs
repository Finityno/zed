//! Drawing a view again from the last frame around the views nested in it
//! that have to be built.
//!
//! Notifying a view marks every view around it dirty, because they have to be
//! walked to reach it, and a change to something a view read is a change to
//! what every view around it depends on. A view that has to be built only for
//! those reasons (it was not notified itself, and nothing it read itself
//! changed) is not built: it is
//! drawn again from the last frame stretch by stretch, with the nested views
//! that have to be built (notified, or something they read changed) built in
//! the gaps where they were, at the layout nodes they kept and with what they
//! inherited there. The views nested in it that do not have to be built are
//! copied along, as they are when a view is drawn again whole.
//!
//! What a gap inherited, it gets again from its record: where it was, the
//! content mask, text style, rem size and opacity around it, the group
//! containers around it, the layout key its element was laid out under, and
//! the dispatch node its element hung off, which is copied with the stretch
//! before it.
//!
//! A view is only spliced where that is what building it would draw:
//! - its layout is the one it had, so it is drawn where it was, and each gap
//!   asks for the layout it had, or for one that, laid out again in its tree
//!   (the window's, or a list item's), moves nothing outside the gap;
//!   otherwise what was built of the splice is rolled back and the view is
//!   built instead;
//! - it deferred nothing (a deferred draw of a view nested in it is drawn
//!   after the frame, out of the stretches);
//! - each gap can be built on its own: it is an entity or an [`AnyView`],
//!   was laid out with nodes it kept (or is a cached view, laid out at its
//!   bounds), is not under an image cache, a text shimmer or a time
//!   transition of its own.

use super::{
    HoverRead, PaintStatus, ViewPrepaint, ViewRebuildReason, ViewRecord, ViewSource,
    dependencies::DependencyChange,
};
use crate::{
    App, ContentMask, ElementId, EntityId, GlobalElementId, GroupHitboxes, LayoutId, Pixels,
    Point, PrepaintStateIndex, TextStyle, TextStyleRefinement, Window,
    key_dispatch::DispatchNodeId, view::ViewName,
};
use smallvec::SmallVec;
use std::{mem, ops::Range};

/// A view laid out as it was last frame, to be drawn again around the nested
/// views at `gaps`, all records of the last frame.
pub(crate) struct Splice {
    pub(super) previous: usize,
    pub(super) gaps: SmallVec<[usize; 4]>,
}

/// What a spliced view's prepaint leaves for its paint.
pub(crate) struct SplicedPrepaint {
    previous: usize,
    /// Its record in this frame.
    index: usize,
    gaps: Vec<PrepaintedGap>,
    /// The records copied along with it, in this frame, each with the gap it
    /// was copied before, or the number of gaps for after the last.
    copied: Vec<(usize, usize)>,
}

struct PrepaintedGap {
    /// Its record last frame.
    previous: usize,
    entity: EntityId,
    id: GlobalElementId,
    /// Its element's dispatch node in this frame.
    node: DispatchNodeId,
    prepaint: ViewPrepaint,
}

/// Last frame's dispatch nodes copied stretch by stretch, with the nodes the
/// copy is inside of left open between stretches.
#[derive(Default)]
struct OpenDispatchCopy {
    /// Last frame's nodes the copy is inside of, innermost last.
    open: Vec<DispatchNodeId>,
    /// Each stretch copied, and where it landed.
    stretches: Vec<(Range<usize>, usize)>,
}

impl OpenDispatchCopy {
    /// Where last frame's node `node` is now, if it was copied.
    fn copied(&self, node: DispatchNodeId) -> Option<DispatchNodeId> {
        let (range, start) = self
            .stretches
            .iter()
            .find(|(range, _)| range.contains(&node.0))?;
        Some(DispatchNodeId(node.0 - range.start + start))
    }
}

/// What the window inherits where a gap is built, set aside meanwhile.
struct Inherited {
    element_id_stack: SmallVec<[ElementId; 32]>,
    text_style_stack: Vec<TextStyleRefinement>,
    content_mask_stack: Vec<ContentMask<Pixels>>,
    element_offset_stack: Vec<Point<Pixels>>,
    rem_size_override_stack: SmallVec<[Pixels; 8]>,
    element_opacity: f32,
    /// Restored after a gap's prepaint; its paint sets glass mode with the
    /// rest of what it painted inside.
    glass_content: Option<bool>,
    groups: GroupHitboxes,
}

/// A text style as the one refinement that gives it.
fn text_style_refinement(style: &TextStyle) -> TextStyleRefinement {
    TextStyleRefinement {
        color: Some(style.color),
        font_family: Some(style.font_family.clone()),
        font_features: Some(style.font_features.clone()),
        font_fallbacks: style.font_fallbacks.clone(),
        font_size: Some(style.font_size),
        line_height: Some(style.line_height),
        font_weight: Some(style.font_weight),
        font_style: Some(style.font_style),
        background_color: style.background_color,
        underline: style.underline,
        strikethrough: style.strikethrough,
        white_space: Some(style.white_space),
        text_overflow: style.text_overflow.clone(),
        text_align: Some(style.text_align),
        line_clamp: style.line_clamp,
    }
}

/// The entity of the view whose element has the id `id`.
pub(super) fn view_entity(id: &GlobalElementId) -> Option<EntityId> {
    match id.0.last()? {
        ElementId::View(entity) => Some(*entity),
        _ => None,
    }
}

impl Window {
    /// Why the view last frame's `record` stands for has to be built, if it
    /// does, leaving out what is around it: it was notified (or a view
    /// nested in it was), or something it read itself, or a hover it was
    /// drawn by itself, changed.
    fn own_rebuild_reason(&self, record: &ViewRecord, cx: &App) -> Option<ViewRebuildReason> {
        let entity = view_entity(&record.id)?;
        if self.dirty_views.contains(&entity) {
            return Some(ViewRebuildReason::Notified);
        }
        self.rebuild_reason_besides_dirty(record, entity, cx)
    }

    /// Why the view `record` stands for has to be built, if it does, besides
    /// a notification of it or of a view nested in it.
    fn rebuild_reason_besides_dirty(
        &self,
        record: &ViewRecord,
        entity: EntityId,
        cx: &App,
    ) -> Option<ViewRebuildReason> {
        if cx.non_retainable_views.contains(&entity) {
            return Some(ViewRebuildReason::OptedOut);
        }
        if record.layout_blocked
            || (self.view_retention.settling && record.unsettled)
            || !matches!(record.paint, PaintStatus::Painted { .. })
            || record.asked_for_autoscroll()
        {
            return Some(ViewRebuildReason::ContextChanged);
        }
        let now = cx.background_executor().now();
        match cx.dependencies_changed_except(
            &record.own_dependencies,
            self.inside_notified_view(),
            now,
            &self.view_retention.every_frame,
        ) {
            Some(DependencyChange::Entity) => return Some(ViewRebuildReason::EntityChanged),
            Some(DependencyChange::Global) => return Some(ViewRebuildReason::GlobalChanged),
            Some(DependencyChange::State) => return Some(ViewRebuildReason::StateChanged),
            Some(DependencyChange::Deadline) => return Some(ViewRebuildReason::Deadline),
            None => {}
        }
        if !record.own_hovers.iter().all(|hover| hover.unchanged(self)) {
            return Some(ViewRebuildReason::HoverChanged);
        }
        None
    }

    /// Whether the view `record` stands for can be built on its own, inside
    /// `around`, drawn again around it.
    fn buildable_on_its_own(record: &ViewRecord, around: &ViewRecord) -> bool {
        record.source.any_view.is_some()
            && (record.source.cached || record.layout.is_some())
            && matches!(record.paint, PaintStatus::Painted { .. })
            && record.context.image_cache.is_none()
            && record.paint_context.shimmer.is_none()
            && record.paint_context.transition == around.paint_context.transition
    }

    /// Lays out the view `id`, to be built only because views nested in it
    /// have to be, as it was laid out last frame, if it can be drawn again
    /// around them; see the module documentation.
    pub(super) fn splice_layout(
        &mut self,
        id: &GlobalElementId,
        entity: EntityId,
        cx: &mut App,
    ) -> Option<(LayoutId, Splice)> {
        if !self.view_retention.splices_enabled || self.view_retention.notified.contains(&entity) {
            return None;
        }
        let previous = self.rendered_frame.retained_views.find(id)?;
        let record = &self.rendered_frame.retained_views.records[previous];
        // Dirty only because of what is nested in it: nothing else of its
        // own has it built.
        if self
            .rebuild_reason_besides_dirty(record, entity, cx)
            .is_some()
        {
            return None;
        }
        let splice = self.splice_gaps(previous, cx)?;
        // A rebuilt gap records its current states; replaying its old states
        // would keep removed elements alive even when the splice is abandoned.
        let layout_id = self.reuse_view_layout(previous, &splice.gaps)?;
        Some((layout_id, splice))
    }

    /// The views nested in last frame's record `previous` that have to be
    /// built, the outermost of each, if there are some and each can be
    /// built on its own.
    fn splice_gaps(&self, previous: usize, cx: &App) -> Option<Splice> {
        let records = &self.rendered_frame.retained_views.records;
        let record = &records[previous];
        let prepaint = &record.prepaint_range;
        if prepaint.start.deferred_draws_index != prepaint.end.deferred_draws_index {
            return None;
        }
        let mut gaps = SmallVec::new();
        let mut index = previous + 1;
        while index <= previous + record.nested {
            let nested = &records[index];
            // A view with something to build inside it is a gap even when
            // it has nothing of its own to build: it is spliced in turn.
            // Copied whole, its record would span a gap built at another
            // length than the one it had.
            let subtree = index..=index + nested.nested;
            if !subtree
                .into_iter()
                .any(|inner| self.own_rebuild_reason(&records[inner], cx).is_some())
            {
                index += nested.nested + 1;
                continue;
            }
            if !Self::buildable_on_its_own(nested, record) {
                return None;
            }
            gaps.push(index);
            index += nested.nested + 1;
        }
        (!gaps.is_empty()).then_some(Splice { previous, gaps })
    }

    /// Copies last frame's prepaint of `range` as [`Window::reuse_prepaint`]
    /// does, leaving the dispatch nodes it enters open in `dispatch` for
    /// what follows to hang off. A spliced view defers nothing.
    fn copy_prepaint_stretch(
        &mut self,
        range: Range<PrepaintStateIndex>,
        dispatch: &mut OpenDispatchCopy,
    ) {
        let rendered = &mut self.rendered_frame;
        let next = &mut self.next_frame;
        next.hitboxes.extend(
            rendered.hitboxes[range.start.hitboxes_index..range.end.hitboxes_index]
                .iter()
                .cloned(),
        );
        // Copied rather than taken: a splice rolled back must leave the last
        // frame as it was.
        next.tooltip_requests.extend(
            rendered.tooltip_requests[range.start.tooltips_index..range.end.tooltips_index]
                .iter()
                .cloned(),
        );
        next.accessed_element_states.extend(
            rendered.accessed_element_states[range.start.accessed_element_states_index
                ..range.end.accessed_element_states_index]
                .iter()
                .cloned(),
        );
        next.autoscroll_requests.extend_from_slice(
            &rendered.autoscroll_requests
                [range.start.autoscroll_requests_index..range.end.autoscroll_requests_index],
        );
        next.positioned_states.extend(
            rendered.positioned_states
                [range.start.positioned_states_index..range.end.positioned_states_index]
                .iter()
                .map(|(state, _)| (state.clone(), Point::default())),
        );
        self.text_system
            .reuse_layouts(range.start.line_layout_index..range.end.line_layout_index);
        let nodes = range.start.dispatch_tree_index..range.end.dispatch_tree_index;
        let start = self.next_frame.dispatch_tree.len();
        dispatch.stretches.push((nodes.clone(), start));
        let contains_focus = self.next_frame.dispatch_tree.copy_stretch(
            nodes,
            &self.rendered_frame.dispatch_tree,
            &mut dispatch.open,
            self.focus,
        );
        if contains_focus {
            self.next_frame.focus = self.focus;
        }
    }

    /// Copies last frame's records `records`, whose prepaint was copied from
    /// `from` to `to`, into this frame inside the spliced view `anchor`,
    /// noting in `copied` where each went and the gap it comes before.
    fn copy_records(
        &mut self,
        records: Range<usize>,
        (from, to): (&PrepaintStateIndex, &PrepaintStateIndex),
        (anchor, gap): (usize, usize),
        copied: &mut Vec<(usize, usize)>,
        cx: &App,
    ) {
        let writes_now = cx.entities.write_generation();
        let source = &self.rendered_frame.retained_views;
        let target = &mut self.next_frame.retained_views;
        for index in records {
            let record = &source.records[index];
            let paint = match record.paint {
                PaintStatus::Painted { .. } => PaintStatus::Pending { anchor },
                _ => PaintStatus::Unpainted,
            };
            target.unsettled |= record.unsettled;
            let copy = target.push(record.copied(from, to, paint, writes_now));
            copied.push((copy, gap));
        }
    }

    /// Puts the window where the view of last frame's record `gap` was
    /// prepainted (`painting` false) or painted, returning what to put back
    /// with [`Window::leave_gap`].
    fn enter_gap(&mut self, gap: usize, painting: bool) -> Inherited {
        let record = &self.rendered_frame.retained_views.records[gap];
        let context = &record.context;
        let groups = GroupHitboxes::of_tops(&record.inherited_groups);
        let mask = if painting {
            record.paint_mask
        } else {
            context.content_mask
        };
        let inherited = Inherited {
            element_id_stack: mem::replace(
                &mut self.element_id_stack,
                record.id.0.iter().cloned().collect(),
            ),
            text_style_stack: mem::replace(
                &mut self.text_style_stack,
                vec![text_style_refinement(&context.text_style)],
            ),
            content_mask_stack: mem::replace(&mut self.content_mask_stack, vec![mask]),
            element_offset_stack: mem::take(&mut self.element_offset_stack),
            rem_size_override_stack: mem::replace(
                &mut self.rem_size_override_stack,
                SmallVec::from_slice(&[context.rem_size]),
            ),
            element_opacity: mem::replace(&mut self.element_opacity, context.opacity),
            glass_content: (!painting)
                .then(|| mem::replace(&mut self.glass_content, context.glass_content)),
            groups: if painting {
                mem::replace(&mut self.group_hitboxes, groups)
            } else {
                mem::replace(&mut self.view_retention.prepaint_groups, groups)
            },
        };
        inherited
    }

    fn leave_gap(&mut self, inherited: Inherited, painting: bool) {
        self.element_id_stack = inherited.element_id_stack;
        self.text_style_stack = inherited.text_style_stack;
        self.content_mask_stack = inherited.content_mask_stack;
        self.element_offset_stack = inherited.element_offset_stack;
        self.rem_size_override_stack = inherited.rem_size_override_stack;
        self.element_opacity = inherited.element_opacity;
        if let Some(glass_content) = inherited.glass_content {
            self.glass_content = glass_content;
        }
        if painting {
            self.group_hitboxes = inherited.groups;
        } else {
            self.view_retention.prepaint_groups = inherited.groups;
        }
    }

    /// Builds the view of last frame's record `gap` on its own, where it was
    /// prepainted, as far as its prepaint goes; `None` if it asks for another
    /// layout than the one it had.
    fn prepaint_gap(&mut self, gap: usize, cx: &mut App) -> Option<(EntityId, ViewPrepaint)> {
        let record = &self.rendered_frame.retained_views.records[gap];
        let id = record.id.clone();
        let entity = view_entity(&id).expect("a gap is a view");
        let bounds = record.context.bounds;
        let layout_scope = record.layout_scope;
        let source = &record.source;
        let cached = source.cached;
        let any_view = source.any_view.clone().expect("a gap can be built on its own");
        let reason = self
            .own_rebuild_reason(record, cx)
            .unwrap_or(ViewRebuildReason::Notified);
        // Where its root has to be offset to land where it was, from where
        // the layout it kept puts it.
        let root = record.layout.as_ref().map(|layout| layout.root);
        let offset = root.and_then(|root| {
            let scale_factor = self.scale_factor();
            let laid_out = self
                .layout_engine
                .as_mut()?
                .layout_bounds(root, scale_factor);
            Some(bounds.origin - laid_out.origin.map(Into::into))
        });

        let inherited = self.enter_gap(gap, false);
        let scope = self.layout_keys.enter_prepaint_scope(layout_scope);
        self.set_view_id(entity);
        let untracked = cx.push_untracked_reads(entity);
        // A gap dirty only because views nested in it are is drawn again
        // around them in turn; its layout nodes were kept with the view
        // around it.
        let spliced = (reason == ViewRebuildReason::Notified
            && !self.view_retention.notified.contains(&entity)
            && self.rebuild_reason_besides_dirty(
                &self.rendered_frame.retained_views.records[gap],
                entity,
                cx,
            )
            .is_none())
        .then(|| self.splice_gaps(gap, cx))
        .flatten()
        .and_then(|splice| {
            self.with_named_view(entity, ViewName::default(), |window| {
                window.splice_prepaint(splice, cx)
            })
        });
        if let Some(spliced) = spliced {
            cx.pop_untracked_reads(untracked);
            self.layout_keys.exit_prepaint_scope(scope);
            self.leave_gap(inherited, false);
            return Some((entity, spliced));
        }
        self.note_rebuild(entity, reason);
        let mut render = |window: &mut Window, cx: &mut App| any_view.render_any(window, cx);
        let source = ViewSource {
            any_view: Some(any_view.clone()),
            cached,
        };
        let prepaint = self.with_named_view(entity, ViewName::default(), |window| {
            match (cached, offset) {
                (false, Some(offset)) => window.with_absolute_element_offset(offset, |window| {
                    window.try_build_at_kept_layout(
                        gap,
                        bounds,
                        &id,
                        source,
                        &mut render,
                        true,
                        cx,
                    )
                }),
                _ => Some(window.build_view_at(bounds, &id, source, &mut render, cx)),
            }
        });
        cx.pop_untracked_reads(untracked);
        self.layout_keys.exit_prepaint_scope(scope);
        self.leave_gap(inherited, false);
        Some((entity, prepaint?))
    }

    /// Prepaints the view that [`Window::splice_layout`] laid out: last
    /// frame's prepaint, copied around the gaps, and the gaps built where
    /// they were. `None`, with nothing of it left, if a gap asked for another
    /// layout than the one it had: the view is built instead.
    pub(super) fn splice_prepaint(&mut self, splice: Splice, cx: &mut App) -> Option<ViewPrepaint> {
        let focus = self.next_frame.focus;
        let rebuilds = self.view_retention.rebuilds.len();
        match self.transact(|window| window.splice_prepaint_inner(splice, cx).ok_or(())) {
            Ok(prepaint) => {
                self.frame_work.stats.views_reused += 1;
                self.frame_work.stats.views_spliced += 1;
                Some(prepaint)
            }
            Err(()) => {
                self.next_frame.focus = focus;
                self.view_retention.rebuilds.truncate(rebuilds);
                // The gaps worked out where their nodes were, which the view
                // built instead lays out again.
                if let Some(engine) = self.layout_engine.as_mut() {
                    engine.forget_layout_bounds();
                }
                None
            }
        }
    }

    fn splice_prepaint_inner(&mut self, splice: Splice, cx: &mut App) -> Option<ViewPrepaint> {
        let Splice { previous, gaps } = splice;
        let (prepaint_range, last, dependencies, own_hovers) = {
            let source = &self.rendered_frame.retained_views;
            let record = &source.records[previous];
            // The nodes the copied views laid out as they prepainted, and
            // their layouts' nodes; the gaps claim theirs as they are built.
            if let Some(engine) = self.layout_engine.as_mut() {
                let keys_before = engine.claimed_keys_len();
                let mut index = previous;
                while index <= previous + record.nested {
                    if gaps.contains(&index) {
                        index += source.records[index].nested + 1;
                        continue;
                    }
                    let nested = &source.records[index];
                    engine.keep_retained(&nested.prepaint_layout_keys);
                    if let Some(layout) = nested.layout.as_deref() {
                        engine.keep_retained(&layout.keys);
                    }
                    index += 1;
                }
                let keys_after = engine.claimed_keys_len();
                let retention = &mut self.view_retention;
                if retention.open_recordings > 0 {
                    retention.nested_keys.push(keys_before..keys_after);
                }
            }
            (
                record.prepaint_range.clone(),
                previous + record.nested,
                record.dependencies.clone(),
                record.own_hovers.clone(),
            )
        };
        if let Some(deadline) = dependencies.rebuild_at {
            self.rebuild_at(deadline);
        }
        // What the view and the views copied with it read is read again; the
        // gaps read theirs as they are built.
        cx.replay_dependencies(&dependencies);
        self.take_hover_reads();
        let hovers_start = self.view_retention.hovers.len();
        self.view_retention.hovers.extend_from_slice(&own_hovers);

        let writes_now = cx.entities.write_generation();
        let start = self.prepaint_index();
        let index = {
            let record = &self.rendered_frame.retained_views.records[previous];
            let copy = ViewRecord {
                id: record.id.clone(),
                prepaint_range: start.clone()..start.clone(),
                paint_range: record.paint_range.clone(),
                paint: PaintStatus::Unpainted,
                nested: 0,
                context: record.context.clone(),
                paint_context: record.paint_context.clone(),
                dependencies: record.own_dependencies.written_up_to(writes_now),
                own_dependencies: record.own_dependencies.written_up_to(writes_now),
                hovers: own_hovers.clone(),
                own_hovers: own_hovers.clone(),
                groups: record.groups.clone(),
                fresh_hitboxes: 0..0,
                prepaint_layout_keys: record.prepaint_layout_keys.clone(),
                layout: record.layout.clone(),
                unsettled: false,
                stays_put: record.stays_put,
                moved: None,
                paint_mask: record.paint_mask,
                source: record.source.clone(),
                layout_scope: record.layout_scope,
                inherited_groups: record.inherited_groups.clone(),
                layout_blocked: false,
            };
            let target = &mut self.next_frame.retained_views;
            target.reused_any = true;
            let index = target.push(copy);
            target.open.push(index);
            index
        };
        let id = self.next_frame.retained_views.records[index].id.clone();
        self.view_retention.view_stack.push(id);

        let mut dispatch = OpenDispatchCopy::default();
        let mut copied = Vec::new();
        let mut prepainted = Vec::with_capacity(gaps.len());
        let mut cursor = prepaint_range.start.clone();
        let mut next_record = previous + 1;
        for (position, &gap) in gaps.iter().enumerate() {
            let (gap_range, gap_nested) = {
                let gap_record = &self.rendered_frame.retained_views.records[gap];
                (gap_record.prepaint_range.clone(), gap_record.nested)
            };
            let stretch_start = self.prepaint_index();
            self.copy_prepaint_stretch(cursor.clone()..gap_range.start.clone(), &mut dispatch);
            self.copy_records(
                next_record..gap,
                (&cursor, &stretch_start),
                (index, position),
                &mut copied,
                cx,
            );
            // The gap's element pushed its dispatch node just before the
            // gap began, which was copied with the stretch before it: the
            // gap hangs off it again.
            let element_node = gap_range
                .start
                .dispatch_tree_index
                .checked_sub(1)
                .map(DispatchNodeId);
            self.next_frame
                .dispatch_tree
                .close_copied_to(&mut dispatch.open, element_node);
            let node = match element_node.and_then(|node| dispatch.copied(node)) {
                Some(node) if self.next_frame.dispatch_tree.active_node_id() == Some(node) => node,
                _ => {
                    let node = self.next_frame.dispatch_tree.push_node();
                    dispatch.open.push(DispatchNodeId(usize::MAX));
                    node
                }
            };
            let Some((entity, prepaint)) = self.prepaint_gap(gap, cx) else {
                self.next_frame
                    .dispatch_tree
                    .close_copied_to(&mut dispatch.open, None);
                self.take_hover_reads();
                self.view_retention.view_stack.pop();
                return None;
            };
            prepainted.push(PrepaintedGap {
                previous: gap,
                entity,
                id: self.rendered_frame.retained_views.records[gap].id.clone(),
                node,
                prepaint,
            });
            cursor = gap_range.end.clone();
            next_record = gap + gap_nested + 1;
        }
        let stretch_start = self.prepaint_index();
        self.copy_prepaint_stretch(cursor.clone()..prepaint_range.end, &mut dispatch);
        self.copy_records(
            next_record..last + 1,
            (&cursor, &stretch_start),
            (index, gaps.len()),
            &mut copied,
            cx,
        );
        self.next_frame
            .dispatch_tree
            .close_copied_to(&mut dispatch.open, None);
        let end = self.prepaint_index();

        self.take_hover_reads();
        self.view_retention.view_stack.pop();
        let views = &mut self.next_frame.retained_views;
        views.open.retain(|open| *open != index);
        let nested = views.records.len() - index - 1;
        // What it read, and the hovers it was drawn by: its own, and those
        // of the views nested in it, copied or built.
        let mut dependencies = views.records[index].own_dependencies.clone();
        let mut all_hovers: Vec<HoverRead> = own_hovers.to_vec();
        let mut unsettled = false;
        let mut stays_put = views.records[index].stays_put;
        let mut nested_index = index + 1;
        while nested_index <= index + nested {
            let record = &views.records[nested_index];
            dependencies = dependencies.union(&record.dependencies);
            all_hovers.extend_from_slice(&record.hovers);
            unsettled |= record.unsettled;
            stays_put |= record.stays_put;
            nested_index += record.nested + 1;
        }
        let record = &mut views.records[index];
        record.prepaint_range = start..end;
        record.nested = nested;
        record.dependencies = dependencies;
        record.hovers = all_hovers.into();
        record.unsettled = unsettled;
        record.stays_put = stays_put;
        let retention = &mut self.view_retention;
        let hovers_end = retention.hovers.len();
        retention.hover_nested.push(hovers_start..hovers_end);
        Some(ViewPrepaint::Spliced(SplicedPrepaint {
            previous,
            index,
            gaps: prepainted,
            copied,
        }))
    }

    /// Paints the view [`Window::splice_prepaint`] prepainted: last frame's
    /// paint, copied around the gaps, and the gaps painted where they were.
    pub(super) fn splice_paint(&mut self, spliced: &mut SplicedPrepaint, cx: &mut App) {
        let SplicedPrepaint {
            previous,
            index,
            gaps,
            copied,
        } = spliced;
        let (paint_range, recorded) = {
            let record = &self.rendered_frame.retained_views.records[*previous];
            (record.paint_range.clone(), record.paint_context.clone())
        };
        let current = self.paint_context();
        let paint_mask = self.content_mask();
        let id = self.next_frame.retained_views.records[*index].id.clone();
        self.take_hover_reads();
        self.view_retention.view_stack.push(id);
        let groups_start = self.view_retention.group_reads.len();
        let hitboxes = {
            let range = &self.next_frame.retained_views.records[*index].prepaint_range;
            range.start.hitboxes_index..range.end.hitboxes_index
        };

        let start = self.paint_index();
        let mut stretches: Vec<(crate::window::PaintIndex, crate::window::PaintIndex)> =
            Vec::with_capacity(gaps.len() + 1);
        let mut cursor = paint_range.start.clone();
        for gap in gaps.iter_mut() {
            let gap_range = self.rendered_frame.retained_views.records[gap.previous]
                .paint_range
                .clone();
            stretches.push((cursor.clone(), self.paint_index()));
            self.reuse_paint_moved(
                cursor.clone()..gap_range.start.clone(),
                recorded.transition,
                current.transition,
                None,
            );
            let inherited = self.enter_gap(gap.previous, true);
            let (glass, opacity_cycles) = {
                let record = &self.rendered_frame.retained_views.records[gap.previous];
                (
                    record.paint_context.glass_content,
                    record.paint_context.opacity_cycle,
                )
            };
            let glass_before = mem::replace(&mut self.glass_content, glass);
            let cycles_before = mem::replace(
                &mut self.opacity_cycle_stack,
                opacity_cycles.into_iter().collect(),
            );
            let shimmers_before = mem::take(&mut self.text_shimmer_stack);
            self.next_frame.dispatch_tree.set_active_node(gap.node);
            self.paint_retained_view(
                gap.entity,
                ViewName::default(),
                &gap.id,
                &mut gap.prepaint,
                cx,
            );
            self.text_shimmer_stack = shimmers_before;
            self.opacity_cycle_stack = cycles_before;
            self.glass_content = glass_before;
            self.leave_gap(inherited, true);
            cursor = gap_range.end.clone();
        }
        stretches.push((cursor.clone(), self.paint_index()));
        self.reuse_paint_moved(
            cursor..paint_range.end,
            recorded.transition,
            current.transition,
            None,
        );
        let end = self.paint_index();

        // The records copied along land where their stretch did.
        for &(copy, stretch) in copied.iter() {
            let (from, to) = &stretches[stretch];
            let record = &mut self.next_frame.retained_views.records[copy];
            if let PaintStatus::Pending { .. } = record.paint {
                record.paint_range =
                    record.paint_range.start.shifted(from, to)..record.paint_range.end.shifted(from, to);
                record.paint = PaintStatus::Painted { source: None };
            }
        }

        self.take_hover_reads();
        self.view_retention.view_stack.pop();
        // The groups the gaps resolved outside the view, besides its own.
        let new_groups: Vec<_> = self.view_retention.group_reads[groups_start..]
            .iter()
            .filter(|read| {
                read.hitbox.is_none_or(|hitbox| {
                    !self.next_frame.hitboxes[hitboxes.clone()]
                        .iter()
                        .any(|inside| inside.id == hitbox)
                })
            })
            .cloned()
            .collect();
        let own_hovers;
        {
            let record = &mut self.next_frame.retained_views.records[*index];
            record.paint_range = start..end;
            record.paint = PaintStatus::Painted { source: None };
            record.paint_context = current.clone();
            record.paint_mask = paint_mask;
            if !new_groups.is_empty() {
                let mut groups = record.groups.to_vec();
                for read in new_groups {
                    if !groups.contains(&read) {
                        groups.push(read);
                    }
                }
                record.groups = groups.into();
            }
            own_hovers = record.own_hovers.clone();
        }
        // As for a view drawn again whole, what it inherits at paint and the
        // hovers checked against last frame's hitboxes are only known now:
        // either changed builds it on the next frame.
        if !own_hovers.iter().all(|hover| hover.unchanged(self))
            || !recorded.effects_match(&current)
        {
            self.request_animation_frame();
        }
    }
}
