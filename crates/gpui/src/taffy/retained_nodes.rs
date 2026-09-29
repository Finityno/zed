//! Layout nodes kept from one frame to the next.
//!
//! Taffy caches what it computed for each node and throws that away for a
//! node and its ancestors whenever the node is written to. Every write dirties
//! unconditionally, so keeping nodes is only worth something if they are not
//! written to: each node remembers what it was last asked for, and a request
//! that matches it leaves the node, and Taffy's cache for it and everything
//! above it, alone.
//!
//! An element finds its node by its layout key (see `window::layout_keys`).
//! A node's style, children and measurement are always compared against what
//! the element asks for, so a key that matches the wrong node costs writes,
//! never a wrong layout. A node unclaimed for a whole frame is released.

use super::{EXPECT_MESSAGE, LayoutId, MeasureFn, NodeContext, TaffyLayoutEngine, ToTaffy as _};
use crate::{
    AbsoluteLength, AlignItems, App, AvailableSpace, DefiniteLength, Edges, GridTemplate, Length,
    Pixels, Size, Style, Window, util::round_to_device_pixel,
};
use collections::{FxHashMap, FxHasher};
use smallvec::SmallVec;
use std::{
    any::Any,
    cell::{Cell, RefCell},
    fmt::Debug,
    hash::{Hash as _, Hasher as _},
    mem,
    rc::Rc,
};
use taffy::TaffyTree;

/// The nodes a [`TaffyLayoutEngine`] keeps across frames.
#[derive(Default)]
pub(crate) struct LayoutRetention {
    retained: FxHashMap<u64, RetainedNode>,
    /// Nodes made this frame without a key, or with one another element
    /// already claimed. Released at the end of the frame.
    transient: Vec<LayoutId>,
    /// The styles elements asked for, for the nodes whose style Taffy holds
    /// stretched to fill the window instead (see
    /// [`TaffyLayoutEngine::stretch_auto_size_to_fill`]), with the frame it
    /// was last stretched in. Compared against the stretched style, every
    /// request would differ and dirty the root.
    unstretched_styles: FxHashMap<LayoutId, (taffy::style::Style, u64)>,
    frame: u64,
    /// Retained nodes claimed this frame. When every retained node was, the
    /// end of the frame has nothing to sweep.
    claimed_this_frame: usize,
    /// Keys claimed since the outermost open transaction began, for a
    /// transaction rolled back to hand back. See
    /// [`TaffyLayoutEngine::begin_transaction`].
    transaction_claims: Vec<u64>,
    open_transactions: usize,
    /// Keys claimed while a recording is open, for a view to claim again on
    /// a frame it is drawn without being laid out. See
    /// [`TaffyLayoutEngine::record_claimed_keys`].
    claimed_keys: Vec<u64>,
    open_key_recordings: usize,
    /// The fingerprint of the default style, which every text leaf asks for,
    /// under the rem size and scale factor it was taken at.
    default_fingerprint: Option<(Pixels, f32, u64)>,
    pub(crate) counts: RetentionCounts,
}

/// What keeping layout nodes did, for [`crate::FrameWorkStats`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct RetentionCounts {
    pub(crate) nodes_reused: u64,
    pub(crate) nodes_kept: u64,
    pub(crate) nodes_created: u64,
    pub(crate) nodes_released: u64,
    pub(crate) style_writes: u64,
    pub(crate) children_writes: u64,
    pub(crate) measured_nodes_dirtied: u64,
    pub(crate) measurements_carried: u64,
    pub(crate) measurements_replayed: u64,
    pub(crate) replay_measure_calls: u64,
}

struct RetainedNode {
    id: LayoutId,
    claimed_in_frame: u64,
    /// The children last written, kept because reading them back out of
    /// Taffy allocates.
    children: SmallVec<[LayoutId; 8]>,
    measured: bool,
    /// [`layout_fingerprint`] of the style last asked for.
    style_fingerprint: u64,
    /// What the element measuring the node left for the next frame's element
    /// to take its measurement over from.
    measurement: Option<Rc<dyn Any>>,
    /// The measurements Taffy took of the node since it was last dirtied.
    measure_log: Option<Rc<MeasureLog>>,
}

/// Removes a node, and releases what its measurement captured: Taffy keeps a
/// removed node's context until another node takes its slot, so a text
/// element's text and shaped lines would otherwise outlive it. Replacing the
/// closure, rather than clearing the context, dirties nothing on the way out.
fn remove_node(taffy: &mut TaffyTree<NodeContext>, id: LayoutId) {
    if let Some(context) = taffy.get_node_context_mut(id.into()) {
        let released: Box<MeasureFn> = Box::new(|_, _, _, _| Size::default());
        #[cfg(feature = "stacker")]
        let released = super::StackSafe::new(released);
        context.measure = released;
    }
    taffy.remove(id.into()).expect(EXPECT_MESSAGE);
}

/// Marks every ancestor of `id` dirty, after a write to it.
///
/// Taffy stops dirtying ancestors at the first node whose cache is already
/// empty, assuming its ancestors were dirtied along with it. A tree rebuilt
/// every frame never tests that; one kept across frames holds nodes whose
/// caches were emptied without their ancestors' (the descendants of a
/// `display: none` node, which hidden layout empties while the hidden node
/// itself caches its result). The whole chain is walked instead of relying
/// on it: after the first write, every step is a check of an empty cache.
fn dirty_ancestors(taffy: &mut TaffyTree<NodeContext>, id: LayoutId) {
    let mut node = taffy.parent(id.into());
    while let Some(ancestor) = node {
        taffy.mark_dirty(ancestor).expect(EXPECT_MESSAGE);
        node = taffy.parent(ancestor);
    }
}

/// Whether Taffy lays out the children of the node, or of its parent, in full
/// while only sizing it.
///
/// A flex container aligning by baseline has to place its children to find
/// their baselines, so a probe of its size leaves them laid out for the
/// probe's constraints. A tree built every frame lays the container out
/// again afterwards; a kept one may answer that from the container's cache
/// and leave the children where the probe put them. Such a node is laid out
/// every frame, as if it were new.
fn aligns_by_baseline(style: &Style) -> bool {
    style.align_items == Some(AlignItems::Baseline) || style.align_self == Some(AlignItems::Baseline)
}

/// The measurements Taffy took of one node since it was last dirtied: the
/// constraints of each and the size it gave, in the order last taken.
///
/// Any size Taffy holds for the node came from one of these, so a new
/// measurement giving every one of them again leaves Taffy's cache for the
/// node, and for the nodes above it, exactly right. The log gives up once it
/// holds more entries than a node measured under steady constraints needs.
#[derive(Default)]
pub(crate) struct MeasureLog {
    entries: RefCell<SmallVec<[MeasureLogEntry; 4]>>,
    overflowed: Cell<bool>,
}

type MeasureLogEntry = (Size<Option<Pixels>>, Size<AvailableSpace>, Size<Pixels>);

/// Taffy caches up to ten results per node.
const MEASURE_LOG_CAPACITY: usize = 16;

impl MeasureLog {
    fn record(&self, known: Size<Option<Pixels>>, available: Size<AvailableSpace>, size: Size<Pixels>) {
        if self.overflowed.get() {
            return;
        }
        let mut entries = self.entries.borrow_mut();
        if let Some(index) = entries
            .iter()
            .position(|(logged_known, logged_available, _)| {
                *logged_known == known && *logged_available == available
            })
        {
            entries.remove(index);
        } else if entries.len() == MEASURE_LOG_CAPACITY {
            self.overflowed.set(true);
            entries.clear();
            return;
        }
        entries.push((known, available, size));
    }

    /// Whether `measure` gives every logged size under the constraints it
    /// was logged under, measured again in the order they were last taken so
    /// that what `measure` keeps is what the last one left.
    fn replays(
        &self,
        measure: &mut dyn FnMut(
            Size<Option<Pixels>>,
            Size<AvailableSpace>,
            &mut Window,
            &mut App,
        ) -> Size<Pixels>,
        counts: &mut RetentionCounts,
        window: &mut Window,
        cx: &mut App,
    ) -> bool {
        if self.overflowed.get() {
            return false;
        }
        let entries = self.entries.borrow().clone();
        !entries.is_empty()
            && entries.iter().all(|(known, available, size)| {
                counts.replay_measure_calls += 1;
                measure(*known, *available, window, cx) == *size
            })
    }

    fn logged(
        log: &Rc<MeasureLog>,
        mut measure: impl FnMut(
            Size<Option<Pixels>>,
            Size<AvailableSpace>,
            &mut Window,
            &mut App,
        ) -> Size<Pixels>
        + 'static,
    ) -> Box<MeasureFn> {
        let log = log.clone();
        Box::new(move |known, available, window: &mut Window, cx: &mut App| {
            let size = measure(known, available, window, cx);
            log.record(known, available, size);
            size
        })
    }
}

/// What an element made of the measurement its node was left with.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Adopted {
    /// The measurement does not stand for this element.
    No,
    /// The element took the measurement over and measures the node from now
    /// on: it differs from the element before it in something measuring it
    /// again would use, though not in what it measures to.
    Measurement,
    /// The element took the measurement over, and the node's closure measures
    /// as its own would: the node is left exactly as it is.
    Node,
}

enum Claim {
    Reused(u64, LayoutId),
    Vacant(u64),
    Unkeyed,
}

fn boxed_measure(
    measure: impl FnMut(Size<Option<Pixels>>, Size<AvailableSpace>, &mut Window, &mut App) -> Size<Pixels>
    + 'static,
) -> super::NodeMeasureFn {
    let measure = Box::new(measure) as Box<MeasureFn>;
    #[cfg(feature = "stacker")]
    let measure = super::StackSafe::new(measure);
    measure
}

impl LayoutRetention {
    fn note_claim(&mut self, key: u64) {
        if self.open_transactions > 0 {
            self.transaction_claims.push(key);
        }
        if self.open_key_recordings > 0 {
            self.claimed_keys.push(key);
        }
    }

    /// Drops every retained node's record, for a tree being replaced.
    ///
    /// The collections are replaced rather than cleared: this runs to give
    /// back what the largest frame made them hold.
    pub(crate) fn forget_all(&mut self) {
        self.retained = FxHashMap::default();
        self.transient = Vec::new();
        self.unstretched_styles = FxHashMap::default();
        self.transaction_claims = Vec::new();
        self.claimed_keys = Vec::new();
        self.claimed_this_frame = 0;
    }
}

impl TaffyLayoutEngine {
    fn claim(&mut self, key: Option<u64>) -> Claim {
        let Some(key) = key else {
            return Claim::Unkeyed;
        };
        let retention = &mut self.retention;
        let frame = retention.frame;
        let Some(node) = retention.retained.get_mut(&key) else {
            return Claim::Vacant(key);
        };
        if node.claimed_in_frame == frame {
            // Two elements resolved to one key: the second makes a node of
            // its own rather than share or strand the first's.
            return Claim::Unkeyed;
        }
        node.claimed_in_frame = frame;
        let id = node.id;
        retention.claimed_this_frame += 1;
        retention.counts.nodes_reused += 1;
        retention.note_claim(key);
        Claim::Reused(key, id)
    }

    /// Claims every node kept under `keys` without asking anything of them,
    /// as [`Self::keep_retained`] does, if every one is still kept and not yet
    /// claimed this frame; otherwise claims none. For a view laid out as it
    /// was last frame without being built, whose layout stands only if all
    /// of its nodes do.
    pub(crate) fn try_keep_retained(&mut self, keys: &[u64]) -> bool {
        let retention = &self.retention;
        let frame = retention.frame;
        let all_kept = keys.iter().all(|key| {
            retention
                .retained
                .get(key)
                .is_some_and(|node| node.claimed_in_frame != frame)
        });
        if all_kept {
            self.keep_retained(keys);
        }
        all_kept
    }

    /// Hands back the claims [`Self::keep_retained`] made on `keys`, for the
    /// element that asked for them to claim them again as it lays out.
    pub(crate) fn release_kept(&mut self, keys: &[u64]) {
        let retention = &mut self.retention;
        let frame = retention.frame;
        for key in keys {
            if let Some(node) = retention.retained.get_mut(key)
                && node.claimed_in_frame == frame
            {
                node.claimed_in_frame = frame.wrapping_sub(1);
                retention.claimed_this_frame -= 1;
                retention.counts.nodes_kept = retention.counts.nodes_kept.saturating_sub(1);
            }
        }
    }

    /// Writes made to kept nodes and nodes made so far: layout requests that
    /// asked for something other than what the nodes held.
    pub(crate) fn layout_writes(&self) -> u64 {
        let counts = &self.retention.counts;
        counts.style_writes + counts.children_writes + counts.nodes_created + counts.measured_nodes_dirtied
    }

    /// How many nodes made this frame will be released at its end, having no
    /// key or one another element took.
    pub(crate) fn transient_count(&self) -> usize {
        self.retention.transient.len()
    }

    /// Whether the node kept under `key` was claimed this frame.
    pub(crate) fn claimed_this_frame(&self, key: u64) -> bool {
        self.retention
            .retained
            .get(&key)
            .is_some_and(|node| node.claimed_in_frame == self.retention.frame)
    }

    /// Starts recording the keys claimed from here on, returning where the
    /// recording begins. Recordings nest.
    pub(crate) fn record_claimed_keys(&mut self) -> usize {
        let retention = &mut self.retention;
        retention.open_key_recordings += 1;
        retention.claimed_keys.len()
    }

    /// Ends the recording begun at `start`, returning the keys claimed while
    /// it was open.
    pub(crate) fn finish_recording_claimed_keys(&mut self, start: usize) -> Vec<u64> {
        let retention = &mut self.retention;
        let mut keys = retention.claimed_keys[start..].to_vec();
        retention.open_key_recordings -= 1;
        if retention.open_key_recordings == 0 {
            retention.claimed_keys.clear();
        }
        // A transaction rolled back hands its claims back, and the requests
        // made again claim the same keys a second time.
        keys.sort_unstable();
        keys.dedup();
        keys
    }

    /// Claims the nodes kept under `keys` that are still kept and not yet
    /// claimed this frame, without asking anything of them: the nodes of a
    /// view drawn again from the last frame, which does not lay them out,
    /// kept for the frame that builds it again.
    pub(crate) fn keep_retained(&mut self, keys: &[u64]) {
        let retention = &mut self.retention;
        let frame = retention.frame;
        for &key in keys {
            if let Some(node) = retention.retained.get_mut(&key)
                && node.claimed_in_frame != frame
            {
                node.claimed_in_frame = frame;
                retention.claimed_this_frame += 1;
                retention.counts.nodes_kept += 1;
                retention.note_claim(key);
            }
        }
    }

    /// Begins a stretch of layout requests that may be rolled back, as
    /// [`crate::Window::transact`] rolls back a prepaint that has to be done
    /// again. Returns where the stretch begins.
    pub(crate) fn begin_transaction(&mut self) -> usize {
        let retention = &mut self.retention;
        retention.open_transactions += 1;
        retention.transaction_claims.len()
    }

    /// Ends the stretch begun at `start`. Rolled back, the nodes it claimed
    /// are handed back, so the requests made again claim them rather than
    /// finding them taken and making nodes of their own.
    pub(crate) fn end_transaction(&mut self, start: usize, rolled_back: bool) {
        let retention = &mut self.retention;
        if rolled_back {
            let frame = retention.frame;
            for key in retention.transaction_claims.drain(start..) {
                if let Some(node) = retention.retained.get_mut(&key)
                    && node.claimed_in_frame == frame
                {
                    node.claimed_in_frame = frame.wrapping_sub(1);
                    retention.claimed_this_frame -= 1;
                }
            }
        }
        retention.open_transactions -= 1;
        if retention.open_transactions == 0 {
            retention.transaction_claims.clear();
        }
    }

    fn retain(
        &mut self,
        key: Option<u64>,
        id: LayoutId,
        children: &[LayoutId],
        measured: bool,
        style_fingerprint: u64,
    ) {
        let retention = &mut self.retention;
        retention.counts.nodes_created += 1;
        let Some(key) = key else {
            retention.transient.push(id);
            return;
        };
        retention.note_claim(key);
        retention.retained.insert(
            key,
            RetainedNode {
                id,
                claimed_in_frame: retention.frame,
                children: SmallVec::from_slice(children),
                measured,
                style_fingerprint,
                measurement: None,
                measure_log: None,
            },
        );
        retention.claimed_this_frame += 1;
    }

    fn retained_node(&mut self, key: u64) -> &mut RetainedNode {
        self.retention
            .retained
            .get_mut(&key)
            .expect("a claimed key is always retained")
    }

    /// Brings a retained node's style up to date, converting and comparing
    /// only when the request is not the one the node was last given.
    fn apply_requested_style(
        &mut self,
        key: u64,
        id: LayoutId,
        style: &Style,
        style_fingerprint: u64,
        rem_size: Pixels,
        scale_factor: f32,
    ) {
        let node = self.retained_node(key);
        if node.style_fingerprint == style_fingerprint {
            // A field the fingerprint missed would leave the node with a
            // stale style whenever only that field changed; debug builds
            // compare in full to catch one.
            debug_assert!(
                self.retention
                    .unstretched_styles
                    .get(&id)
                    .map(|(requested, _)| requested)
                    .unwrap_or_else(|| self.taffy.style(id.into()).expect(EXPECT_MESSAGE))
                    == &style.to_taffy(rem_size, scale_factor),
                "layout_fingerprint matched a style that converts differently"
            );
            return;
        }
        node.style_fingerprint = style_fingerprint;
        let style = style.to_taffy(rem_size, scale_factor);
        let retention = &mut self.retention;
        let previous = retention
            .unstretched_styles
            .get(&id)
            .map(|(requested, _)| requested)
            .unwrap_or_else(|| self.taffy.style(id.into()).expect(EXPECT_MESSAGE));
        if previous == &style {
            return;
        }
        retention.unstretched_styles.remove(&id);
        retention.counts.style_writes += 1;
        self.taffy
            .set_style(id.into(), style)
            .expect(EXPECT_MESSAGE);
        dirty_ancestors(&mut self.taffy, id);
    }

    fn apply_children(&mut self, key: u64, id: LayoutId, children: &[LayoutId]) {
        let node = self.retained_node(key);
        if node.children.as_slice() == children {
            return;
        }
        node.children.clear();
        node.children.extend_from_slice(children);
        self.retention.counts.children_writes += 1;
        self.taffy
            .set_children(id.into(), LayoutId::to_taffy_slice(children))
            .expect(EXPECT_MESSAGE);
        dirty_ancestors(&mut self.taffy, id);
    }

    fn style_fingerprint(&mut self, style: Option<&Style>, rem_size: Pixels, scale_factor: f32) -> u64 {
        match style {
            Some(style) => layout_fingerprint(style, rem_size, scale_factor),
            None => match self.retention.default_fingerprint {
                Some((rem, scale, fingerprint)) if rem == rem_size && scale == scale_factor => {
                    fingerprint
                }
                _ => {
                    let fingerprint =
                        layout_fingerprint(&Style::default(), rem_size, scale_factor);
                    self.retention.default_fingerprint =
                        Some((rem_size, scale_factor, fingerprint));
                    fingerprint
                }
            },
        }
    }

    /// Adds a node to the tree, reusing the one retained under `key` when
    /// there is one, and writing to it only what differs from last frame.
    pub(crate) fn request_keyed_layout(
        &mut self,
        key: Option<u64>,
        style: &Style,
        rem_size: Pixels,
        scale_factor: f32,
        children: &[LayoutId],
    ) -> LayoutId {
        let key = match self.claim(key) {
            Claim::Reused(key, id) => {
                let fingerprint = layout_fingerprint(style, rem_size, scale_factor);
                self.apply_requested_style(key, id, style, fingerprint, rem_size, scale_factor);
                self.apply_children(key, id, children);
                if aligns_by_baseline(style) {
                    self.taffy.mark_dirty(id.into()).expect(EXPECT_MESSAGE);
                    dirty_ancestors(&mut self.taffy, id);
                }
                let node = self.retained_node(key);
                if node.measured {
                    node.measured = false;
                    node.measurement = None;
                    node.measure_log = None;
                    self.taffy
                        .set_node_context(id.into(), None)
                        .expect(EXPECT_MESSAGE);
                    dirty_ancestors(&mut self.taffy, id);
                }
                return id;
            }
            Claim::Vacant(key) => Some(key),
            Claim::Unkeyed => None,
        };

        let fingerprint = if key.is_some() {
            layout_fingerprint(style, rem_size, scale_factor)
        } else {
            0
        };
        let id: LayoutId = self
            .taffy
            .new_leaf(style.to_taffy(rem_size, scale_factor))
            .expect(EXPECT_MESSAGE)
            .into();
        if !children.is_empty() {
            // A retained child can arrive still listed under the parent it
            // had last frame. `new_with_children` would leave it listed there,
            // and that parent rewriting its children later would cut the
            // child's parent link, which its position is added up along.
            // `set_children` detaches each child from wherever it was.
            self.taffy
                .set_children(id.into(), LayoutId::to_taffy_slice(children))
                .expect(EXPECT_MESSAGE);
        }
        self.retain(key, id, children, false, fingerprint);
        id
    }

    /// Adds a self-measuring leaf, reusing the node retained under `key`.
    /// Nothing says what the measurement depends on, so a reused node is
    /// given the new closure and dirtied, and `measure` is sure to run.
    pub(crate) fn request_keyed_measured_layout(
        &mut self,
        key: Option<u64>,
        style: Option<&Style>,
        rem_size: Pixels,
        scale_factor: f32,
        measure: Box<MeasureFn>,
    ) -> LayoutId {
        #[cfg(feature = "stacker")]
        let measure = super::StackSafe::new(measure);
        match self.claim(key) {
            Claim::Reused(key, id) => {
                let fingerprint = self.style_fingerprint(style, rem_size, scale_factor);
                let default_style;
                let style = match style {
                    Some(style) => style,
                    None => {
                        default_style = Style::default();
                        &default_style
                    }
                };
                self.apply_requested_style(key, id, style, fingerprint, rem_size, scale_factor);
                self.apply_children(key, id, &[]);
                self.retention.counts.measured_nodes_dirtied += 1;
                if let Some(context) = self.taffy.get_node_context_mut(id.into()) {
                    context.measure = measure;
                } else {
                    self.taffy
                        .set_node_context(id.into(), Some(NodeContext { measure }))
                        .expect(EXPECT_MESSAGE);
                }
                self.taffy.mark_dirty(id.into()).expect(EXPECT_MESSAGE);
                dirty_ancestors(&mut self.taffy, id);
                let node = self.retained_node(key);
                node.measured = true;
                node.measurement = None;
                node.measure_log = None;
                id
            }
            claim => {
                let key = match claim {
                    Claim::Vacant(key) => Some(key),
                    _ => None,
                };
                let fingerprint = if key.is_some() {
                    self.style_fingerprint(style, rem_size, scale_factor)
                } else {
                    0
                };
                let taffy_style = match style {
                    Some(style) => style.to_taffy(rem_size, scale_factor),
                    None => Style::default().to_taffy(rem_size, scale_factor),
                };
                let id: LayoutId = self
                    .taffy
                    .new_leaf_with_context(taffy_style, NodeContext { measure })
                    .expect(EXPECT_MESSAGE)
                    .into();
                self.retain(key, id, &[], true, fingerprint);
                id
            }
        }
    }

    /// Adds a self-measuring leaf whose measurement can be carried over from
    /// the element that measured its node last frame, rather than taken
    /// again.
    ///
    /// That element left `state`'s counterpart on the node; `adopt` is given
    /// both, and takes the measurement over if it still stands for this
    /// element, in which case the node stays clean and keeps what Taffy
    /// cached for it and the nodes above it. [`Adopted::Node`] leaves the
    /// node's closure and state as they are, and this element's `state` is
    /// dropped; [`Adopted::Measurement`] gives the node this element's.
    ///
    /// When the measurement does not stand, the node is still left clean if
    /// `measure` gives every size Taffy took of the node since it was last
    /// dirtied, under the same constraints; see [`MeasureLog`]. Otherwise it
    /// is measured again as [`Self::request_keyed_measured_layout`] would.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn request_carried_measured_layout<S: 'static>(
        &mut self,
        key: Option<u64>,
        style: Option<&Style>,
        rem_size: Pixels,
        scale_factor: f32,
        state: S,
        adopt: impl FnOnce(&Rc<S>, &Rc<dyn Any>) -> Adopted,
        measure: impl Fn(&Rc<S>, Size<Option<Pixels>>, Size<AvailableSpace>, &mut Window, &mut App) -> Size<Pixels>
        + 'static,
        window: &mut Window,
        cx: &mut App,
    ) -> LayoutId {
        let state = Rc::new(state);
        let fingerprint = self.style_fingerprint(style, rem_size, scale_factor);
        let frame = self.retention.frame;
        let reusable = key
            .and_then(|key| self.retention.retained.get(&key))
            .filter(|node| {
                node.claimed_in_frame != frame
                    && node.measured
                    && node.style_fingerprint == fingerprint
                    && !style.is_some_and(aligns_by_baseline)
            })
            .map(|node| (node.measurement.clone(), node.measure_log.clone()));

        if let Some((previous, log)) = reusable {
            let adopted = previous.map_or(Adopted::No, |previous| adopt(&state, &previous));
            let kept = if adopted != Adopted::No {
                self.retention.counts.measurements_carried += 1;
                true
            } else if let Some(log) = &log
                && log.replays(
                    &mut |known, available, window: &mut Window, cx: &mut App| {
                        measure(&state, known, available, window, cx)
                    },
                    &mut self.retention.counts,
                    window,
                    cx,
                )
            {
                self.retention.counts.measurements_replayed += 1;
                true
            } else {
                false
            };
            if kept && let Claim::Reused(key, id) = self.claim(key) {
                if adopted == Adopted::Node {
                    return id;
                }
                // A measured node always has a context; one without is
                // measured afresh below rather than left without one.
                if self.taffy.get_node_context(id.into()).is_some() {
                    let measurement: Rc<dyn Any> = state.clone();
                    let log = log.unwrap_or_default();
                    let measure = boxed_measure(MeasureLog::logged(
                        &log,
                        move |known, available, window, cx| {
                            measure(&state, known, available, window, cx)
                        },
                    ));
                    if let Some(context) = self.taffy.get_node_context_mut(id.into()) {
                        context.measure = measure;
                    }
                    let node = self.retained_node(key);
                    node.measurement = Some(measurement);
                    node.measure_log = Some(log);
                    return id;
                }
                self.release_claim(key);
            }
        }

        let measurement: Rc<dyn Any> = state.clone();
        let log = Rc::<MeasureLog>::default();
        let measure = MeasureLog::logged(&log, move |known, available, window, cx| {
            measure(&state, known, available, window, cx)
        });
        let id = self.request_keyed_measured_layout(key, style, rem_size, scale_factor, measure);
        if let Some(key) = key
            && let Some(node) = self.retention.retained.get_mut(&key)
            && node.id == id
        {
            node.measurement = Some(measurement);
            node.measure_log = Some(log);
        }
        id
    }

    fn release_claim(&mut self, key: u64) {
        let retention = &mut self.retention;
        if let Some(node) = retention.retained.get_mut(&key)
            && node.claimed_in_frame == retention.frame
        {
            node.claimed_in_frame = retention.frame.wrapping_sub(1);
            retention.claimed_this_frame -= 1;
            retention.counts.nodes_reused = retention.counts.nodes_reused.saturating_sub(1);
        }
    }

    /// Ends the frame for the retained nodes: releases the transient ones and
    /// those no element claimed, and keeps the rest with their caches.
    pub(crate) fn release_unclaimed_nodes(&mut self) {
        let retention = &mut self.retention;
        if retention.retained.is_empty() {
            // Nothing is kept, as when retention is turned off: clearing the
            // tree at once is cheaper than removing its nodes one by one.
            retention.counts.nodes_released += retention.transient.len() as u64;
            retention.transient.clear();
            retention.unstretched_styles.clear();
            self.taffy.clear();
        } else {
            for id in retention.transient.drain(..) {
                retention.unstretched_styles.remove(&id);
                remove_node(&mut self.taffy, id);
                retention.counts.nodes_released += 1;
            }
            if retention.retained.len() != retention.claimed_this_frame {
                let frame = retention.frame;
                let taffy = &mut self.taffy;
                let unstretched_styles = &mut retention.unstretched_styles;
                let released = &mut retention.counts.nodes_released;
                retention.retained.retain(|_, node| {
                    if node.claimed_in_frame == frame {
                        return true;
                    }
                    unstretched_styles.remove(&node.id);
                    remove_node(taffy, node.id);
                    *released += 1;
                    false
                });
            }
            // A node stretched on an earlier frame and not on this one (no
            // longer a window's root, say) still holds the stretched style in
            // Taffy, which no element asked for: it gets back the style that
            // was asked for.
            let frame = retention.frame;
            let taffy = &mut self.taffy;
            retention
                .unstretched_styles
                .retain(|&id, (requested, stretched_in)| {
                    if *stretched_in == frame {
                        return true;
                    }
                    taffy
                        .set_style(id.into(), requested.clone())
                        .expect(EXPECT_MESSAGE);
                    dirty_ancestors(taffy, id);
                    false
                });
        }
        retention.claimed_this_frame = 0;
        retention.frame += 1;
    }

    /// Treats any `auto` dimension of the node's style as filling `size`,
    /// keeping the style the element asked for aside: a stretched `auto` looks
    /// exactly like an explicit length, and the next frame has to compare its
    /// request against the request, not against the stretch.
    pub(crate) fn stretch_retained_auto_size_to_fill(
        &mut self,
        id: LayoutId,
        size: Size<Pixels>,
        scale_factor: f32,
    ) {
        let retention = &mut self.retention;
        let requested = match retention.unstretched_styles.get(&id) {
            Some((requested, _)) => requested,
            None => self.taffy.style(id.into()).expect(EXPECT_MESSAGE),
        };
        let stretch_width = requested.size.width.is_auto();
        let stretch_height = requested.size.height.is_auto();
        if !stretch_width && !stretch_height {
            return;
        }
        let requested = requested.clone();
        let mut style = requested.clone();
        if stretch_width {
            style.size.width =
                taffy::style::Dimension::length(round_to_device_pixel(size.width.0, scale_factor));
        }
        if stretch_height {
            style.size.height =
                taffy::style::Dimension::length(round_to_device_pixel(size.height.0, scale_factor));
        }
        if self.taffy.style(id.into()).expect(EXPECT_MESSAGE) != &style {
            retention.counts.style_writes += 1;
            self.taffy
                .set_style(id.into(), style)
                .expect(EXPECT_MESSAGE);
            dirty_ancestors(&mut self.taffy, id);
        }
        retention.unstretched_styles.insert(id, (requested, retention.frame));
    }
}

/// A hash of everything in `style` its conversion to a Taffy style reads, and
/// of what lengths are resolved against.
///
/// It has to read exactly the fields [`ToTaffy`](super::ToTaffy) does: a field
/// it missed would be a change a retained node never receives. A test changes
/// each field in turn, and debug builds check every match against a full
/// conversion.
pub(crate) fn layout_fingerprint(style: &Style, rem_size: Pixels, scale_factor: f32) -> u64 {
    fn absolute(hasher: &mut FxHasher, length: &AbsoluteLength) {
        match length {
            AbsoluteLength::Pixels(pixels) => (0u8, pixels.0.to_bits()).hash(hasher),
            AbsoluteLength::Rems(rems) => (1u8, rems.0.to_bits()).hash(hasher),
        }
    }
    fn definite(hasher: &mut FxHasher, length: &DefiniteLength) {
        match length {
            DefiniteLength::Absolute(length) => {
                0u8.hash(hasher);
                absolute(hasher, length);
            }
            DefiniteLength::Fraction(fraction) => (1u8, fraction.to_bits()).hash(hasher),
        }
    }
    fn length(hasher: &mut FxHasher, length: &Length) {
        match length {
            Length::Definite(length) => {
                0u8.hash(hasher);
                definite(hasher, length);
            }
            Length::Auto => 1u8.hash(hasher),
        }
    }
    fn edges<T: Clone + Debug + Default + PartialEq>(
        hasher: &mut FxHasher,
        edges: &Edges<T>,
        each: fn(&mut FxHasher, &T),
    ) {
        each(hasher, &edges.top);
        each(hasher, &edges.right);
        each(hasher, &edges.bottom);
        each(hasher, &edges.left);
    }
    fn sizes<T: Clone + Debug + Default + PartialEq>(
        hasher: &mut FxHasher,
        size: &Size<T>,
        each: fn(&mut FxHasher, &T),
    ) {
        each(hasher, &size.width);
        each(hasher, &size.height);
    }
    fn placement(hasher: &mut FxHasher, placement: &crate::GridPlacement) {
        match placement {
            crate::GridPlacement::Line(line) => (0u8, *line).hash(hasher),
            crate::GridPlacement::Span(span) => (1u8, *span).hash(hasher),
            crate::GridPlacement::Auto => 2u8.hash(hasher),
        }
    }
    fn template(hasher: &mut FxHasher, template: &Option<GridTemplate>) {
        match template {
            Some(template) => {
                (1u8, template.repeat).hash(hasher);
                mem::discriminant(&template.min_size).hash(hasher);
            }
            None => 0u8.hash(hasher),
        }
    }

    let mut hasher = FxHasher::default();
    let hasher = &mut hasher;
    rem_size.0.to_bits().hash(hasher);
    scale_factor.to_bits().hash(hasher);

    mem::discriminant(&style.display).hash(hasher);
    mem::discriminant(&style.overflow.x).hash(hasher);
    mem::discriminant(&style.overflow.y).hash(hasher);
    absolute(hasher, &style.scrollbar_width);
    mem::discriminant(&style.position).hash(hasher);
    edges(hasher, &style.inset, length);
    sizes(hasher, &style.size, length);
    sizes(hasher, &style.min_size, length);
    sizes(hasher, &style.max_size, length);
    style.aspect_ratio.map(f32::to_bits).hash(hasher);
    edges(hasher, &style.margin, length);
    edges(hasher, &style.padding, definite);
    edges(hasher, &style.border_widths, absolute);
    style
        .align_items
        .map(|align| mem::discriminant(&align))
        .hash(hasher);
    style
        .align_self
        .map(|align| mem::discriminant(&align))
        .hash(hasher);
    style
        .align_content
        .map(|align| mem::discriminant(&align))
        .hash(hasher);
    style
        .justify_content
        .map(|align| mem::discriminant(&align))
        .hash(hasher);
    sizes(hasher, &style.gap, definite);
    mem::discriminant(&style.flex_direction).hash(hasher);
    mem::discriminant(&style.flex_wrap).hash(hasher);
    length(hasher, &style.flex_basis);
    style.flex_grow.to_bits().hash(hasher);
    style.flex_shrink.to_bits().hash(hasher);
    template(hasher, &style.grid_rows);
    template(hasher, &style.grid_cols);
    match &style.grid_location {
        Some(location) => {
            1u8.hash(hasher);
            for line in [&location.row, &location.column] {
                placement(hasher, &line.start);
                placement(hasher, &line.end);
            }
        }
        None => 0u8.hash(hasher),
    }
    hasher.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::px;

    /// Every field the conversion to a Taffy style reads has to reach the
    /// fingerprint: each case changes one such field, and both have to see it.
    #[test]
    fn the_layout_fingerprint_sees_every_field_the_taffy_style_is_made_from() {
        use crate::{
            AlignContent, AlignItems, Display, FlexDirection, FlexWrap, GridLocation,
            GridPlacement, GridTemplateMinSize, Overflow, Position, relative, rems,
        };

        let (rem_size, scale_factor) = (px(16.), 2.);
        let base = Style::default();
        let cases: Vec<(&str, Box<dyn Fn(&mut Style)>)> = vec![
            ("display", Box::new(|style| style.display = Display::Grid)),
            ("overflow.x", Box::new(|style| style.overflow.x = Overflow::Hidden)),
            ("overflow.y", Box::new(|style| style.overflow.y = Overflow::Scroll)),
            (
                "scrollbar_width",
                Box::new(|style| style.scrollbar_width = px(7.).into()),
            ),
            ("position", Box::new(|style| style.position = Position::Absolute)),
            ("inset", Box::new(|style| style.inset.left = px(3.).into())),
            ("size", Box::new(|style| style.size.width = px(40.).into())),
            ("size in rems", Box::new(|style| style.size.width = rems(2.).into())),
            (
                "size as a fraction",
                Box::new(|style| style.size.width = relative(0.5).into()),
            ),
            ("min_size", Box::new(|style| style.min_size.height = px(5.).into())),
            ("max_size", Box::new(|style| style.max_size.width = px(90.).into())),
            ("aspect_ratio", Box::new(|style| style.aspect_ratio = Some(1.5))),
            ("margin", Box::new(|style| style.margin.top = px(2.).into())),
            ("padding", Box::new(|style| style.padding.bottom = px(4.).into())),
            (
                "border_widths",
                Box::new(|style| style.border_widths.right = px(1.).into()),
            ),
            (
                "align_items",
                Box::new(|style| style.align_items = Some(AlignItems::Center)),
            ),
            (
                "align_self",
                Box::new(|style| style.align_self = Some(AlignItems::End)),
            ),
            (
                "align_content",
                Box::new(|style| style.align_content = Some(AlignContent::End)),
            ),
            (
                "justify_content",
                Box::new(|style| style.justify_content = Some(AlignContent::Center)),
            ),
            ("gap", Box::new(|style| style.gap.width = px(6.).into())),
            (
                "flex_direction",
                Box::new(|style| style.flex_direction = FlexDirection::Column),
            ),
            ("flex_wrap", Box::new(|style| style.flex_wrap = FlexWrap::Wrap)),
            ("flex_basis", Box::new(|style| style.flex_basis = px(12.).into())),
            ("flex_grow", Box::new(|style| style.flex_grow = 1.)),
            ("flex_shrink", Box::new(|style| style.flex_shrink = 0.)),
            (
                "grid_rows",
                Box::new(|style| {
                    style.grid_rows = Some(GridTemplate {
                        repeat: 3,
                        min_size: GridTemplateMinSize::Zero,
                    })
                }),
            ),
            (
                "grid_cols",
                Box::new(|style| {
                    style.grid_cols = Some(GridTemplate {
                        repeat: 2,
                        min_size: GridTemplateMinSize::MinContent,
                    })
                }),
            ),
            (
                "grid_location",
                Box::new(|style| {
                    style.grid_location = Some(GridLocation {
                        row: GridPlacement::Line(1)..GridPlacement::Span(2),
                        column: GridPlacement::Auto..GridPlacement::Auto,
                    })
                }),
            ),
        ];

        let base_fingerprint = layout_fingerprint(&base, rem_size, scale_factor);
        let base_taffy = base.to_taffy(rem_size, scale_factor);
        for (field, change) in &cases {
            let mut style = base.clone();
            change(&mut style);
            assert_ne!(
                style.to_taffy(rem_size, scale_factor),
                base_taffy,
                "changing {field} should change the Taffy style, or this case tests nothing"
            );
            assert_ne!(
                layout_fingerprint(&style, rem_size, scale_factor),
                base_fingerprint,
                "changing {field} changes the Taffy style but not the fingerprint"
            );
        }

        let mut in_rems = base;
        in_rems.size.width = rems(2.).into();
        assert_ne!(
            layout_fingerprint(&in_rems, rem_size, scale_factor),
            layout_fingerprint(&in_rems, px(20.), scale_factor)
        );
        assert_ne!(
            layout_fingerprint(&in_rems, rem_size, scale_factor),
            layout_fingerprint(&in_rems, rem_size, 1.)
        );
    }

    fn sized(width: f32) -> Style {
        let mut style = Style::default();
        style.size.width = px(width).into();
        style.size.height = px(10.).into();
        style
    }

    /// A retained node hidden with `display: none` and shown again is laid
    /// out again, at its size.
    #[test]
    fn a_retained_node_hidden_and_shown_again_is_laid_out_again() {
        let mut engine = TaffyLayoutEngine::new();
        let frame = |engine: &mut TaffyLayoutEngine, hidden: bool| {
            let mut child_style = sized(30.);
            if hidden {
                child_style.display = crate::Display::None;
            }
            let child = engine.request_keyed_layout(Some(2), &child_style, px(16.), 1., &[]);
            let mut parent_style = Style::default();
            parent_style.display = crate::Display::Flex;
            let parent = engine.request_keyed_layout(Some(1), &parent_style, px(16.), 1., &[child]);
            let root = engine.request_keyed_layout(Some(0), &Style::default(), px(16.), 1., &[parent]);
            engine.taffy
                .compute_layout(
                    root.into(),
                    taffy::geometry::Size {
                        width: taffy::AvailableSpace::Definite(100.),
                        height: taffy::AvailableSpace::Definite(100.),
                    },
                )
                .expect(EXPECT_MESSAGE);
            let width = engine.taffy.layout(child.into()).expect(EXPECT_MESSAGE).size.width;
            engine.clear();
            width
        };
        assert_eq!(frame(&mut engine, false), 30.);
        assert_eq!(frame(&mut engine, true), 0.);
        assert_eq!(frame(&mut engine, true), 0.);
        assert_eq!(frame(&mut engine, false), 30.);
        assert_eq!(engine.retention.counts.nodes_created, 3);
    }

    /// Requests rolled back with the prepaint they were made in hand their
    /// nodes back, and the requests made again find them.
    #[test]
    fn requests_made_again_after_a_rollback_find_their_nodes() {
        let mut engine = TaffyLayoutEngine::new();
        let frame = |engine: &mut TaffyLayoutEngine| {
            let transaction = engine.begin_transaction();
            let child = engine.request_keyed_layout(Some(2), &sized(5.), px(16.), 1., &[]);
            engine.request_keyed_layout(Some(1), &Style::default(), px(16.), 1., &[child]);
            engine.end_transaction(transaction, true);
            let child = engine.request_keyed_layout(Some(2), &sized(5.), px(16.), 1., &[]);
            engine.request_keyed_layout(Some(1), &Style::default(), px(16.), 1., &[child]);
            engine.clear();
        };
        frame(&mut engine);
        assert_eq!(engine.retention.counts.nodes_created, 2);
        frame(&mut engine);
        assert_eq!(engine.retention.counts.nodes_created, 2);
        assert_eq!(engine.taffy.total_node_count(), 2);
    }

    /// A frame asking for what the last one asked for writes nothing, and
    /// nodes no element claims are released.
    #[test]
    fn unchanged_requests_write_nothing_and_unclaimed_nodes_are_released() {
        let mut engine = TaffyLayoutEngine::new();
        let frame = |engine: &mut TaffyLayoutEngine, children: usize| {
            let ids: Vec<LayoutId> = (0..children)
                .map(|index| {
                    engine.request_keyed_layout(Some(10 + index as u64), &sized(5.), px(16.), 1., &[])
                })
                .collect();
            engine.request_keyed_layout(Some(1), &Style::default(), px(16.), 1., &ids);
            engine.clear();
        };
        frame(&mut engine, 4);
        let created = engine.retention.counts.nodes_created;
        frame(&mut engine, 4);
        assert_eq!(engine.retention.counts.nodes_created, created);
        assert_eq!(engine.retention.counts.style_writes, 0);
        assert_eq!(engine.retention.counts.children_writes, 0);
        assert_eq!(engine.taffy.total_node_count(), 5);

        frame(&mut engine, 2);
        assert_eq!(engine.retention.counts.children_writes, 1);
        assert_eq!(engine.taffy.total_node_count(), 3);
    }
}
