//! A `list`'s rows drawn again from the last frame on frames that only
//! scrolled the list.
//!
//! A wheel scroll builds the view drawing a `list`, which builds the list:
//! the list calls its row closure for every row it shows and lays each one
//! out, though the views inside are then drawn again moved. With row replay
//! on (off unless an application turns it on, see
//! [`App::set_list_row_replay`], and only with view retention on), every row
//! a list draws leaves a record of its own in the frame, as a view does: what
//! it read while it was rendered, laid out, prepainted and painted, where its
//! hitboxes, listeners and primitives went, and the layout nodes it claimed.
//! On a frame where the view drawing the list was built only because the list
//! scrolled, a row whose record says nothing it read changed is drawn again
//! from that record, where it was or moved by the scroll, without calling the
//! row closure, laying the row out, prepainting or painting it.
//!
//! The row closure is handed to the list anew by every render of the view
//! drawing it, and may hold anything that render read: a row is only drawn
//! again on frames where nothing that view read itself changed but the
//! list's scroll (see [`Window::list_rows_replayable`]). What a row read is
//! the row's own, besides being the view's around it, so a change to it
//! builds that row, and the view around it, again. A splice, reset or
//! remeasure of the list builds every row again: rows are known by index.
//!
//! Whether a row can be drawn again where it is now is decided as it is for a
//! view drawn moved (see [`super::moving`]): a row that moved by a fraction
//! of a device pixel, reaches past the list's edge, holds a focused input,
//! draws by hovers under the pointer, or a view nested in which was notified,
//! is built instead, by the row closure, at the size it had.

use super::{
    LayoutRecording, RetainedLayout, ViewSource, culprits,
    dependencies::{Recorded, StateVersion},
    splice::view_entity,
};
use crate::{
    AnyElement, App, Bounds, ElementId, GlobalElementId, LayoutId, Pixels, Point, SharedString,
    Size, Window,
};
use std::{
    rc::Rc,
    sync::{
        OnceLock,
        atomic::{AtomicU64, Ordering},
    },
};

/// Whether list row replay starts out on: `GPUI_LIST_ROW_REPLAY=1` (or
/// `true`).
pub(crate) fn list_row_replay_from_environment() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        matches!(
            std::env::var("GPUI_LIST_ROW_REPLAY").as_deref(),
            Ok("1" | "true")
        )
    })
}

/// A number no other list state has, for its rows' records to be told apart
/// from another list's drawn in the same view.
pub(crate) fn next_list_serial() -> u64 {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    NEXT.fetch_add(1, Ordering::Relaxed)
}

/// How many frames a list row's layout nodes are kept through after it was
/// last drawn: a few seconds' worth, for a reader scrolling past a row and
/// back to find it laid out, its text measured.
const ROW_NODES_LINGER_FRAMES: u64 = 240;

/// What laying a list row out left for its prepaint to record: the layout
/// nodes it claimed, and what it read.
pub(crate) struct RowLayout {
    layout: Option<Rc<RetainedLayout>>,
    dependencies: Recorded,
}

/// A list row's layout being recorded as a [`RowLayout`].
pub(crate) struct RowLayoutRecording(LayoutRecording);

impl App {
    /// Turns list row replay on or off: whether a `list` whose view was built
    /// only because the list scrolled draws the rows nothing changed for
    /// again from the last frame, moved by the scroll, instead of calling its
    /// row closure for them and laying them out, prepainting and painting
    /// them. Only takes effect with view retention on
    /// ([`App::set_view_retention`]).
    ///
    /// Off by default, and on from startup with `GPUI_LIST_ROW_REPLAY=1`.
    /// With it on, a list's row closure must draw a row from what it reads
    /// (entities, globals, versioned state), and from what the render of the
    /// view handing the list the closure read: a row whose closure draws it
    /// from something else (the clock, a `Rc<RefCell<..>>` written behind
    /// the list's back) is drawn as it was until it is notified, or the
    /// list is spliced or remeasured, as a view would be; nor may it draw a
    /// row by where the list is scrolled to. Whatever else the closure does
    /// for a row (noting it was shown, say) it does only on the frames that
    /// build it; see [`crate::List::on_rows_replayed`] for the others.
    pub fn set_list_row_replay(&mut self, enabled: bool) {
        if self.list_row_replay != enabled {
            self.list_row_replay = enabled;
            self.refresh_windows();
        }
    }

    /// Whether list row replay is on. See [`App::set_list_row_replay`].
    pub fn list_row_replay(&self) -> bool {
        self.list_row_replay
    }

    /// Whether lists record their rows for them to be drawn again: row
    /// replay and view retention are both on.
    pub(crate) fn records_list_rows(&self) -> bool {
        self.list_row_replay && self.view_retention()
    }
}

impl Window {
    /// The id the record of row `index` of the list `list` (see
    /// [`next_list_serial`]) is known by, under the element being drawn.
    pub(crate) fn list_row_id(&mut self, list: u64, index: usize) -> GlobalElementId {
        self.with_id(
            ElementId::NamedInteger(SharedString::new_static("list-rows"), list),
            |window| {
                window.with_global_id(ElementId::Integer(index as u64), |id, _| id.clone())
            },
        )
    }

    /// Whether the list being prepainted may draw its rows again from the last
    /// frame: the view drawing it is being built, and it was not notified,
    /// and nothing it read itself (outside its rows and the views in them)
    /// changed but the list's own state, `list_state` (where it is scrolled
    /// to and what it answers of it, which a view drawing a list reads to
    /// place what is drawn beside it). The row closure it handed the list is
    /// then the one it handed the last frame's list, and draws the rows it
    /// drew then: a list's rows are drawn by what they read, not by where the
    /// list is scrolled to. What changes the rows themselves (a splice, a
    /// reset, a remeasure, another width) is read by every row's record.
    pub(crate) fn list_rows_replayable(&self, list_state: &[&StateVersion], cx: &App) -> bool {
        if !cx.records_list_rows() {
            return false;
        }
        let views = &self.next_frame.retained_views;
        let Some(&open) = views.open.last() else {
            return false;
        };
        let id = &views.records[open].id;
        // A view drawn twice in a frame keeps no record, so the open one
        // can be a view around the one drawing the list.
        if self.view_retention.view_stack.last() != Some(id) {
            return false;
        }
        if view_entity(id).is_some_and(|view| self.view_retention.notified.contains(&view)) {
            culprits::list_row_built(|| "replay off: the view drawing the list was notified".into());
            return false;
        }
        let Some(previous) = self.rendered_frame.retained_views.find(id) else {
            return false;
        };
        let record = &self.rendered_frame.retained_views.records[previous];
        let now = cx.background_executor().now();
        if let Some(change) = cx.dependencies_changed_except(
            &record.own_dependencies,
            self.inside_notified_view(),
            now,
            list_state,
        ) {
            culprits::list_row_built(|| {
                format!("replay off: the view drawing the list read a change ({change:?})")
            });
            return false;
        }
        if !record.own_hovers.iter().all(|hover| hover.unchanged(self)) {
            culprits::list_row_built(|| "replay off: the view drawing the list hovers otherwise".into());
            return false;
        }
        true
    }

    /// Last frame's record of the list row `id`, if it can be drawn again from
    /// it at `size`: nothing it read or was drawn by changed, and no view in
    /// it was notified. Where it is drawn is checked as it prepaints.
    pub(crate) fn replayable_list_row(
        &self,
        id: &GlobalElementId,
        size: Size<Pixels>,
        cx: &App,
    ) -> Option<usize> {
        if let Some(reason) = self.frame_builds_every_record(cx) {
            culprits::list_row_built(|| format!("{reason:?}"));
            return None;
        }
        let previous = match self.reusable_record(id, cx) {
            Ok(previous) => previous,
            Err(reason) => {
                culprits::list_row_built(|| format!("{reason:?}"));
                return None;
            }
        };
        let records = &self.rendered_frame.retained_views.records;
        let record = &records[previous];
        if record.context.bounds.size != size {
            culprits::list_row_built(|| "measured at another size".into());
            return None;
        }
        // The element states its layout used, which the frame keeps only if
        // something uses them, are known only from the layout it kept.
        if record.layout.is_none() {
            culprits::list_row_built(|| "its layout was not kept".into());
            return None;
        }
        // Notifying a view marks the views around it dirty, not the rows
        // around it.
        let notified = records[previous..=previous + record.nested]
            .iter()
            .filter_map(|record| view_entity(&record.id))
            .any(|view| self.dirty_views.contains(&view));
        if notified {
            culprits::list_row_built(|| "a view in it was notified".into());
            return None;
        }
        Some(previous)
    }

    /// Starts recording the layout of a list row about to be rendered and
    /// laid out, which reads `rows`, the version the list bumps when its rows
    /// are spliced, reset or remeasured.
    pub(crate) fn begin_list_row_layout(
        &mut self,
        rows: &StateVersion,
        cx: &mut App,
    ) -> RowLayoutRecording {
        let recording = self.begin_view_layout(cx);
        super::dependencies::note_state_read(rows);
        RowLayoutRecording(recording)
    }

    /// Ends the recording of a list row's layout, `element` laid out.
    pub(crate) fn finish_list_row_layout(
        &mut self,
        recording: RowLayoutRecording,
        element: &AnyElement,
        cx: &mut App,
    ) -> RowLayout {
        let root: Option<LayoutId> = element.layout_id();
        let (layout, dependencies) = self.finish_layout_recording(recording.0, root, cx);
        RowLayout {
            layout,
            dependencies,
        }
    }

    /// Prepaints a list row built this frame at `origin`, recording it under
    /// `id`, and returns its record.
    pub(crate) fn prepaint_list_row(
        &mut self,
        id: &GlobalElementId,
        layout: RowLayout,
        origin: Point<Pixels>,
        size: Size<Pixels>,
        element: &mut AnyElement,
        cx: &mut App,
    ) -> Option<usize> {
        self.frame_work.stats.list_rows_built += 1;
        let recording = self.begin_record_prepaint(id, ViewSource::default(), cx);
        element.prepaint_at(origin, self, cx);
        let record = self.finish_view_prepaint(
            recording,
            Bounds { origin, size },
            layout.layout,
            Some(layout.dependencies),
            cx,
        );
        if let Some(index) = record {
            self.linger_row_nodes(index);
        }
        record
    }

    /// Has the layout nodes of the row recorded at `index` in this frame, and
    /// of the views in it, outlive it by [`ROW_NODES_LINGER_FRAMES`] once it
    /// is no longer drawn.
    fn linger_row_nodes(&mut self, index: usize) {
        let records = &self.next_frame.retained_views.records;
        let (Some(record), Some(engine)) = (records.get(index), self.layout_engine.as_mut())
        else {
            return;
        };
        for record in &records[index..=index + record.nested] {
            engine.linger(&record.prepaint_layout_keys, ROW_NODES_LINGER_FRAMES);
            if let Some(layout) = record.layout.as_deref() {
                engine.linger(&layout.keys, ROW_NODES_LINGER_FRAMES);
            }
        }
    }

    /// Draws last frame's list row `previous` again at `bounds`, as far as its
    /// prepaint goes, where it was or moved, if it can be drawn there, and
    /// returns its record in this frame.
    pub(crate) fn prepaint_replayed_list_row(
        &mut self,
        previous: usize,
        bounds: Bounds<Pixels>,
        cx: &mut App,
    ) -> Option<usize> {
        let moved = if self.view_context_matches(previous, bounds) {
            None
        } else {
            match self.view_move(previous, bounds) {
                Ok(moved) => Some(moved),
                Err(refusal) => {
                    culprits::list_row_built(|| format!("not drawn moved: {refusal:?}"));
                    return None;
                }
            }
        };
        self.frame_work.stats.list_rows_replayed += 1;
        // What the row and the views in it used as they were laid out, as a
        // view drawn again uses it as it is laid out (outside its prepaint,
        // which copies what was used there).
        let records = &self.rendered_frame.retained_views.records;
        let subtree = &records[previous..=previous + records[previous].nested];
        let states = &mut self.next_frame.accessed_element_states;
        for layout in subtree.iter().filter_map(|record| record.layout.as_deref()) {
            states.extend(layout.element_states.iter().cloned());
        }
        Some(self.reuse_record_prepaint(previous, moved, cx))
    }

    /// Paints a list row built this frame, into its record if it has one.
    pub(crate) fn paint_list_row(
        &mut self,
        record: Option<usize>,
        id: &GlobalElementId,
        element: &mut AnyElement,
        cx: &mut App,
    ) {
        match record {
            Some(index) => {
                let recording = self.begin_view_paint(index, id, cx);
                element.paint(self, cx);
                self.finish_view_paint(recording, cx);
            }
            None => element.paint(self, cx),
        }
    }

    /// Paints a list row drawn again from the last frame, into its record
    /// `index` in this frame.
    pub(crate) fn paint_replayed_list_row(&mut self, index: usize) {
        self.reuse_view_paint(index);
    }
}
