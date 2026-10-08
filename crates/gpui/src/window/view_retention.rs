//! Views drawn again from what they drew on the last frame.
//!
//! With view retention on (it is off unless an application turns it on, see
//! [`App::set_view_retention`]), every entity view drawn in a frame leaves a
//! record there: where its hitboxes, dispatch nodes, listeners and
//! primitives went, what it read while it was built (see [`dependencies`]),
//! the hovers it was drawn by, what it inherited where it was drawn, and the
//! layout nodes it holds. On the next frame, a view whose record says nothing
//! it depends on changed is drawn again by copying those stretches of the
//! last frame, instead of being rendered, laid out, prepainted and painted.
//!
//! The records live in the frame rather than in element state because the
//! stretches they point to belong to one frame. A view drawn again from the
//! last frame is not visited, so drawing it again copies its record and the
//! records nested in it, shifted to where the copy landed, so that each can
//! be drawn again on its own later, when what is around it has to be built.
//!
//! A view is built instead when it was notified (a view around one notified
//! is drawn again around it, see below), when something it read changed, when a hover it was drawn by changed, when
//! it inherits something else where it is drawn (a group it hovers by, say,
//! whose container was built), while the window refreshes, while something is
//! dragged, while the inspector picks, while accessibility is active, and
//! when it opted out ([`crate::Context::set_view_retainable`]).
//!
//! A view drawn somewhere else than it was (scrolled, or pushed down by
//! something above it that grew) is drawn again moved, its primitives,
//! hitboxes and nested records shifted with it, where that is what building
//! it would draw; otherwise it is built at the layout nodes it kept. See
//! [`moving`].
//!
//! A view dirty only because a view nested in it was notified, or has to be
//! built for something it read, is drawn again from the last frame around
//! the nested views that are built, rather than built with them. See
//! [`splice`].
//!
//! What a view inherits at paint (the text shimmer, the opacity cycle and the
//! time transition its primitives are stamped with) is only known once the
//! frame is painting, too late to build the view instead. A transition is
//! rebased onto the one the view is painted in as its primitives are copied;
//! a different shimmer or opacity cycle asks for the next frame, on which the
//! view is built.

pub(crate) mod culprits;
pub(crate) mod dependencies;
pub(crate) mod moving;
pub(crate) mod splice;

pub use dependencies::DrawDependency;
pub(crate) use moving::{PositionedState, ViewMove};

#[cfg(test)]
mod tests;

use super::{ArenaClearNeeded, PaintIndex, PrepaintStateIndex, Window};
use crate::{
    AnyElement, AnyView, App, AvailableSpace, Bounds, ContentMask, EntityId, GlobalElementId,
    GroupHitboxes, HitboxId, LayoutId, OpacityCycle, Pixels, SharedString, Size, Style,
    StyleRefinement, Task, TextShimmerStyle, TextStyle, view::ViewName,
};
use collections::FxHashMap;
use dependencies::{
    AmbientReads, DependencyChange, DependencyRecording, Recorded, RenderDependencies,
};
use gpui_util::ResultExt;
use refineable::Refineable;
use smallvec::SmallVec;
use std::{any::TypeId, cell::RefCell, ops::Range, rc::Rc, sync::OnceLock, time::Instant};

/// Whether view retention starts out on: `GPUI_RETAINED_VIEWS=1` (or `true`).
pub(crate) fn view_retention_from_environment() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        matches!(
            std::env::var("GPUI_RETAINED_VIEWS").as_deref(),
            Ok("1" | "true")
        )
    })
}

/// How often a frame that drew views again from the last one is drawn again
/// from scratch and compared, from `GPUI_RETAINED_VIEWS_VERIFY`: every such
/// frame for `1`, every `n`th for `n`. See [`Window::verify_retained_frame`].
fn verification_interval() -> Option<u64> {
    static INTERVAL: OnceLock<Option<u64>> = OnceLock::new();
    *INTERVAL.get_or_init(|| {
        std::env::var("GPUI_RETAINED_VIEWS_VERIFY")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .filter(|interval| *interval > 0)
    })
}

/// Whether views may be drawn again moved: unless `GPUI_RETAINED_VIEW_MOVES`
/// is `0` (or `false`).
fn moves_from_environment() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        !matches!(
            std::env::var("GPUI_RETAINED_VIEW_MOVES").as_deref(),
            Ok("0" | "false")
        )
    })
}

/// Whether views dirty only because views nested in them are may be drawn
/// again around them: unless `GPUI_RETAINED_VIEW_SPLICES` is `0` (or `false`).
fn splices_from_environment() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        !matches!(
            std::env::var("GPUI_RETAINED_VIEW_SPLICES").as_deref(),
            Ok("0" | "false")
        )
    })
}

/// Why a view was built rather than drawn again from the last frame, with
/// view retention on. See [`Window::view_rebuild_reasons`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ViewRebuildReason {
    /// The view has no record from the last frame: it was not drawn then, it
    /// was drawn twice in one frame, or the record was dropped.
    FirstDraw,
    /// The window refreshed, something is being dragged or the inspector is
    /// picking, all of which build every view.
    WindowRefresh,
    /// Accessibility is active, which builds every view so that its
    /// accessibility nodes are built.
    Accessibility,
    /// The view, or a view nested in it, was notified since the last frame.
    Notified,
    /// An entity the view read changed: updated and notified, or written
    /// while the window drew.
    EntityChanged,
    /// A global the view read was written, or the pointer or modifier keys it
    /// read changed.
    GlobalChanged,
    /// Versioned state the view read changed: a scroll handle, a list state
    /// or a [`DrawDependency`].
    StateChanged,
    /// A time the view said it would look different at has passed. See
    /// [`Window::rebuild_at`].
    Deadline,
    /// A hitbox whose hover the view was drawn by is hovered differently.
    HoverChanged,
    /// The view is drawn at other bounds, or inherits another content mask,
    /// text style, opacity or rem size there.
    ContextChanged,
    /// The view opted out of being drawn again from the last frame.
    OptedOut,
}

/// A hover a view was drawn by: whether `hitbox` was hovered, as
/// [`HitboxId::is_hovered`] (or, `ignoring_modality`,
/// [`HitboxId::is_hovered_ignoring_last_input`]) answered.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct HoverRead {
    hitbox: HitboxId,
    ignoring_modality: bool,
    hovered: bool,
}

impl HoverRead {
    fn unchanged(&self, window: &Window) -> bool {
        self.hitbox.hovered_now(window, self.ignoring_modality) == self.hovered
    }
}

/// A group a view resolved as it painted (see
/// [`crate::InteractiveElement::group`]): the hitbox of the innermost group
/// container of that name around it, if any.
///
/// Only groups resolved outside the view are kept: a container inside it is
/// drawn again with it. Hitbox ids are new whenever a container is built, so
/// a view drawn again around a container that was built would hover by a
/// hitbox no longer drawn, and is built instead.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct GroupRead {
    name: SharedString,
    hitbox: Option<HitboxId>,
}

/// What a view element knows of its view, for its record to build it again
/// on its own, inside a view drawn again around it. See [`splice`].
#[derive(Default)]
pub(crate) struct ViewSource {
    /// The view, when it is an entity or an [`AnyView`].
    pub(crate) any_view: Option<AnyView>,
    /// Whether it is a cached view, laid out at its style.
    pub(crate) cached: bool,
}

/// The retained views drawn in one frame, in the order they began
/// prepainting, which puts a view's nested views right after it.
#[derive(Default)]
pub(crate) struct RetainedViews {
    records: Vec<ViewRecord>,
    by_id: FxHashMap<GlobalElementId, usize>,
    /// The records whose prepaint is under way, innermost last.
    open: Vec<usize>,
    /// Whether any view was drawn again from the frame before.
    reused_any: bool,
    /// Whether a view drawn in this frame is unsettled: drawn again moved
    /// since it was last built. See [`moving`].
    pub(crate) unsettled: bool,
}

struct ViewRecord {
    id: GlobalElementId,
    prepaint_range: Range<PrepaintStateIndex>,
    paint_range: Range<PaintIndex>,
    paint: PaintStatus,
    /// How many of the records following this one are nested inside it.
    nested: usize,
    context: Rc<ViewContext>,
    paint_context: PaintContext,
    /// Everything the view read, nested views and what they deferred
    /// included.
    dependencies: RenderDependencies,
    /// What the view read itself, outside the views nested in it, and what
    /// it deferred.
    own_dependencies: RenderDependencies,
    /// The hovers the view was drawn by, likewise, and its own.
    hovers: Rc<[HoverRead]>,
    own_hovers: Rc<[HoverRead]>,
    /// The groups it, and the views nested in it, resolved outside it.
    groups: Rc<[GroupRead]>,
    /// The ids of the hitboxes inserted as it was prepainted this frame,
    /// built: groups resolved to one of those are inside it.
    fresh_hitboxes: Range<u64>,
    /// The layout nodes the view claimed while it was prepainted (list items,
    /// say), kept while it is drawn again so that building it again later
    /// finds them.
    prepaint_layout_keys: Rc<[u64]>,
    layout: Option<Rc<RetainedLayout>>,
    /// Drawn again moved since it, or a view nested in it, was last built:
    /// its mouse listeners still answer for where it was. See [`moving`].
    unsettled: bool,
    /// Cannot be drawn again moved: it, or a view nested in it, wrote state
    /// as it was prepainted or painted, or opted out.
    stays_put: bool,
    /// How it moved this frame, drawn again from the last one, for its paint
    /// to move what it copies.
    moved: Option<ViewMove>,
    /// The content mask around it as it was painted, which what it painted
    /// was clipped by; usually the one around it as it was prepainted.
    paint_mask: ContentMask<Pixels>,
    /// How to build it again on its own; see [`splice`].
    source: Rc<ViewSource>,
    /// The layout key of its element, which what it lays out hangs off.
    layout_scope: u64,
    /// The group containers around it as it was painted, innermost of each
    /// name, for building it again on its own.
    inherited_groups: Rc<[(SharedString, HitboxId)]>,
    /// A view built inside it asked for another layout than the one it was
    /// laid out at, so its own layout is out of date: it is built on the
    /// next frame.
    layout_blocked: bool,
}

impl ViewRecord {
    /// Whether something drawn in the view, or in a view nested in it,
    /// asked to be scrolled into view as it prepainted. The request is for
    /// that frame; drawn again, the view would not make it, and what scrolls
    /// around it would not answer it.
    fn asked_for_autoscroll(&self) -> bool {
        self.prepaint_range.start.autoscroll_requests_index
            != self.prepaint_range.end.autoscroll_requests_index
    }

    /// This record copied into the next frame along with the stretch of
    /// prepaint it was made in, drawn again from `from` to `to`.
    fn copied(
        &self,
        from: &PrepaintStateIndex,
        to: &PrepaintStateIndex,
        paint: PaintStatus,
        writes_now: u64,
    ) -> ViewRecord {
        ViewRecord {
            id: self.id.clone(),
            prepaint_range: self.prepaint_range.start.shifted(from, to)
                ..self.prepaint_range.end.shifted(from, to),
            paint_range: self.paint_range.clone(),
            paint,
            nested: self.nested,
            context: self.context.clone(),
            paint_context: self.paint_context.clone(),
            dependencies: self.dependencies.written_up_to(writes_now),
            own_dependencies: self.own_dependencies.written_up_to(writes_now),
            hovers: self.hovers.clone(),
            own_hovers: self.own_hovers.clone(),
            groups: self.groups.clone(),
            fresh_hitboxes: 0..0,
            prepaint_layout_keys: self.prepaint_layout_keys.clone(),
            layout: self.layout.clone(),
            unsettled: self.unsettled,
            stays_put: self.stays_put,
            moved: None,
            paint_mask: self.paint_mask,
            source: self.source.clone(),
            layout_scope: self.layout_scope,
            inherited_groups: self.inherited_groups.clone(),
            layout_blocked: self.layout_blocked,
        }
    }
}

#[derive(Clone)]
enum PaintStatus {
    /// Not painted, so `paint_range` means nothing.
    Unpainted,
    /// Painted this frame into `paint_range`; when that was drawn from the
    /// last frame, `source` is where it started there, for the records
    /// copied along with it to shift their ranges by.
    Painted { source: Option<PaintIndex> },
    /// Copied along with the record at `anchor` and still holding last
    /// frame's `paint_range`, which is shifted once that one is painted.
    Pending { anchor: usize },
}

/// What a view's prepaint depended on besides what it read: where it was
/// drawn and what it inherited there.
#[derive(PartialEq)]
struct ViewContext {
    bounds: Bounds<Pixels>,
    content_mask: ContentMask<Pixels>,
    text_style: TextStyle,
    opacity: f32,
    rem_size: Pixels,
    /// The image cache images without their own load through.
    image_cache: Option<EntityId>,
    /// Whether it is inside a glass surface, which its quads are stamped
    /// with as they paint.
    glass_content: bool,
}

/// What a view's paint inherited that turns into what its primitives hold:
/// the text effect glyphs are stamped with, the opacity cycle and glass mode
/// quads are stamped with, and the time transition everything is moved with.
#[derive(Clone, Default)]
struct PaintContext {
    shimmer: Option<TextShimmerStyle>,
    opacity_cycle: Option<OpacityCycle>,
    glass_content: bool,
    transition: u32,
}

impl PaintContext {
    fn effects_match(&self, other: &Self) -> bool {
        self.shimmer == other.shimmer
            && self.opacity_cycle == other.opacity_cycle
            && self.glass_content == other.glass_content
    }
}

/// How to lay a view out as it was laid out last frame without building it:
/// a view is laid out by its content, so its layout is only known from the
/// nodes its content left.
pub(crate) struct RetainedLayout {
    root: LayoutId,
    /// The layout key of the view, which its content's keys hang off: the
    /// layout stands for the view only where it has the same key.
    view_key: Option<u64>,
    /// Every node its content claimed while its layout was requested.
    keys: Vec<u64>,
    /// The element states its content used while its layout was requested,
    /// kept alive while it is not built.
    element_states: Vec<(GlobalElementId, TypeId)>,
}

/// A layout request being recorded as a [`RetainedLayout`].
struct LayoutRecording {
    /// Where the recording of claimed keys began, when there is a layout
    /// engine to record them.
    keys: Option<usize>,
    transient: usize,
    element_states: usize,
    /// Where this recording's nested stretches begin in `nested_keys` and
    /// `nested_states`.
    nested_keys: usize,
    nested_states: usize,
    dependencies: DependencyRecording,
}

/// A retained view being prepainted.
struct ViewRecording {
    index: Option<usize>,
    dependencies: DependencyRecording,
    layout_keys: usize,
    nested_keys: usize,
    hovers_start: usize,
    hover_nested: usize,
    hitboxes_start: u64,
    /// The view being prepainted.
    view: Option<EntityId>,
    writes_start: Writes,
}

/// Where the writes to entities and globals stood as a view's prepaint or
/// paint began: a view that wrote any while it prepainted or painted may
/// have written where it was, and is not drawn moved. Versioned state the
/// framework keeps its position in (scroll handles, list states) is moved
/// with it instead; see [`moving::PositionedState`].
#[derive(Clone, Copy, PartialEq)]
struct Writes {
    entities: u64,
}

impl Writes {
    fn now(cx: &App) -> Self {
        Writes {
            entities: cx.entities.pinning_writes(),
        }
    }
}

/// A retained view being painted.
struct ViewPaintRecording {
    index: Option<usize>,
    start: PaintIndex,
    hovers_start: usize,
    hover_nested: usize,
    groups_start: usize,
    dependencies: DependencyRecording,
    writes_start: Writes,
}

/// A window's state for drawing views again, besides the records its frames
/// hold.
pub(crate) struct ViewRetention {
    /// The retained views being built or painted, innermost last, while
    /// hovers are noted for them.
    view_stack: Vec<GlobalElementId>,
    /// The hovers read this frame by retained views, in the order read.
    hovers: Vec<HoverRead>,
    /// Hovers read through [`HitboxId::is_hovered`] since `hovers` last took
    /// them in.
    hover_reads: RefCell<Vec<HoverRead>>,
    /// The views notified since the last frame was drawn, without the views
    /// around them.
    notified: collections::FxHashSet<EntityId>,
    /// Records reads of the pointer and modifier keys while views are drawn.
    pub(crate) ambient_reads: AmbientReads,
    /// The stretches of `hovers` that views nested in the ones being drawn
    /// took for themselves, in order: a view's own hovers leave them out.
    hover_nested: Vec<Range<usize>>,
    /// The group containers around what is being prepainted, which a view
    /// drawn again must find where it found them as it was painted.
    pub(crate) prepaint_groups: GroupHitboxes,
    /// The groups resolved by retained views as they painted this frame, in
    /// the order resolved.
    group_reads: Vec<GroupRead>,
    /// Why each view built in the last frame was built.
    rebuilds: Vec<(EntityId, ViewRebuildReason)>,
    /// The earliest time something drawn in this frame said it would look
    /// different at, and the task asking for a frame at the earliest one.
    deadline: Option<Instant>,
    deadline_frame: Option<(Instant, Task<()>)>,
    /// The stretches of the layout engine's claimed-key log, and of the
    /// frame's accessed element states, that views nested in the ones being
    /// recorded took for themselves, in order: a record holds only its own
    /// keys and states, and a view drawn again gathers its nested views'
    /// from their records, so that what a view built costs does not grow
    /// with how much is nested in it.
    nested_keys: Vec<Range<usize>>,
    nested_states: Vec<Range<usize>>,
    /// How many retained view recordings are open.
    open_recordings: usize,
    /// Whether the deferred draw being drawn was deferred from inside a view
    /// notified since the last frame; see [`EnclosingViews`].
    deferred_inside_notified: bool,
    /// [`crate::key_dispatch::DispatchTree::action_fingerprint`] of the frame
    /// last drawn, to tell when which actions are available changed, if it
    /// was worked out: only frames with a view that read the actions need it.
    actions_fingerprint: Option<u64>,
    /// The text system's font generation as the frame last drawn began, and
    /// as the one being drawn began: views recorded before fonts were added
    /// shaped their text without them, so none is drawn again.
    drawn_font_generation: usize,
    drawing_font_generation: usize,
    /// The retained views a deferred draw being drawn now counts as part of,
    /// for what it defers in turn to count as theirs too.
    deferring_views: SmallVec<[usize; 4]>,
    /// Bumped as every frame begins. A view that opted out reads it, so that
    /// the views around it depend on it too and are never drawn again whole,
    /// which would copy it along.
    every_frame: dependencies::StateVersion,
    /// Frames that drew views again since the last one checked against a
    /// frame drawn from scratch, and how many to let pass between checks.
    frames_since_verification: u64,
    verification_interval: Option<u64>,
    /// Whether views may be drawn again moved (off with
    /// `GPUI_RETAINED_VIEW_MOVES=0`), whether the frame being drawn builds
    /// every unsettled view instead, whether the next one should, whether
    /// one was drawn moved in this frame, and the task asking for a frame
    /// that settles them once they stop moving. See [`moving`].
    pub(crate) moves_enabled: bool,
    /// Whether views dirty only because views nested in them are drawn again
    /// around them (off with `GPUI_RETAINED_VIEW_SPLICES=0`); see [`splice`].
    pub(crate) splices_enabled: bool,
    pub(crate) settling: bool,
    pub(crate) settle_requested: bool,
    pub(crate) moved_this_frame: bool,
    pub(crate) last_moved_at: Instant,
    pub(crate) settle_frame: Option<Task<()>>,
}

impl ViewRetention {
    /// Whether no retained view is being drawn.
    pub(crate) fn view_stack_is_empty(&self) -> bool {
        self.view_stack.is_empty()
    }

    pub(crate) fn new(cx: &App) -> Self {
        Self {
            view_stack: Vec::new(),
            hovers: Vec::new(),
            hover_reads: RefCell::default(),
            notified: Default::default(),
            ambient_reads: cx.ambient_reads(),
            hover_nested: Vec::new(),
            prepaint_groups: GroupHitboxes::default(),
            group_reads: Vec::new(),
            rebuilds: Vec::new(),
            deadline: None,
            deadline_frame: None,
            every_frame: Default::default(),
            nested_keys: Vec::new(),
            nested_states: Vec::new(),
            open_recordings: 0,
            deferred_inside_notified: false,
            actions_fingerprint: None,
            drawn_font_generation: 0,
            drawing_font_generation: 0,
            deferring_views: SmallVec::new(),
            frames_since_verification: 0,
            verification_interval: verification_interval(),
            moves_enabled: moves_from_environment(),
            splices_enabled: splices_from_environment(),
            settling: false,
            settle_requested: false,
            moved_this_frame: false,
            last_moved_at: cx.background_executor().now(),
            settle_frame: None,
        }
    }
}

/// The retained views, as indices into this frame's records, being prepainted
/// when something was deferred: what drawing it reads and the hovers it is
/// drawn by are theirs too, though it is drawn after them.
#[derive(Clone, Default)]
pub(crate) struct EnclosingViews {
    views: SmallVec<[usize; 4]>,
    /// Whether one of the views around it was notified since the last
    /// frame: drawn later, it has only the view it was deferred from around
    /// it, and would otherwise not know.
    inside_notified: bool,
    /// The entities whose reads the views around it leave out (see
    /// [`crate::Context::untrack_reads_of`]), which it leaves out too.
    untracked: SmallVec<[EntityId; 2]>,
}

/// Something deferred from retained views, being drawn.
pub(crate) struct DeferredViewRecording {
    enclosing: EnclosingViews,
    /// What [`ViewRetention::deferring_views`] held before, restored after.
    deferring_before: SmallVec<[usize; 4]>,
    dependencies: DependencyRecording,
    hovers_start: usize,
    untracked: usize,
}

/// Where a transaction began in the retained records. See
/// [`Window::transact`].
pub(crate) struct RetainedTransaction {
    records: usize,
    hovers: usize,
    hover_nested: usize,
}

impl PrepaintStateIndex {
    /// This index, taken from a range that started at `from`, as it falls in
    /// a copy of that range starting at `to`.
    fn shifted(&self, from: &Self, to: &Self) -> Self {
        PrepaintStateIndex {
            hitboxes_index: self.hitboxes_index - from.hitboxes_index + to.hitboxes_index,
            tooltips_index: self.tooltips_index - from.tooltips_index + to.tooltips_index,
            deferred_draws_index: self.deferred_draws_index - from.deferred_draws_index
                + to.deferred_draws_index,
            positioned_states_index: self.positioned_states_index
                - from.positioned_states_index
                + to.positioned_states_index,
            autoscroll_requests_index: self.autoscroll_requests_index
                - from.autoscroll_requests_index
                + to.autoscroll_requests_index,
            dispatch_tree_index: self.dispatch_tree_index - from.dispatch_tree_index
                + to.dispatch_tree_index,
            accessed_element_states_index: self.accessed_element_states_index
                - from.accessed_element_states_index
                + to.accessed_element_states_index,
            line_layout_index: self
                .line_layout_index
                .shifted(&from.line_layout_index, &to.line_layout_index),
        }
    }

    fn same_place(&self, other: &Self) -> bool {
        self.hitboxes_index == other.hitboxes_index
            && self.tooltips_index == other.tooltips_index
            && self.deferred_draws_index == other.deferred_draws_index
            && self.positioned_states_index == other.positioned_states_index
            && self.autoscroll_requests_index == other.autoscroll_requests_index
            && self.dispatch_tree_index == other.dispatch_tree_index
            && self.accessed_element_states_index == other.accessed_element_states_index
            && self.line_layout_index == other.line_layout_index
    }
}

impl PaintIndex {
    fn shifted(&self, from: &Self, to: &Self) -> Self {
        PaintIndex {
            scene_index: self.scene_index - from.scene_index + to.scene_index,
            window_control_hitboxes_index: self.window_control_hitboxes_index
                - from.window_control_hitboxes_index
                + to.window_control_hitboxes_index,
            #[cfg(any(test, feature = "test-support"))]
            debug_bounds_index: self.debug_bounds_index - from.debug_bounds_index
                + to.debug_bounds_index,
            mouse_listeners_index: self.mouse_listeners_index - from.mouse_listeners_index
                + to.mouse_listeners_index,
            input_handlers_index: self.input_handlers_index - from.input_handlers_index
                + to.input_handlers_index,
            cursor_styles_index: self.cursor_styles_index - from.cursor_styles_index
                + to.cursor_styles_index,
            accessed_element_states_index: self.accessed_element_states_index
                - from.accessed_element_states_index
                + to.accessed_element_states_index,
            tab_handle_index: self.tab_handle_index - from.tab_handle_index + to.tab_handle_index,
            painted_positions_index: self.painted_positions_index - from.painted_positions_index
                + to.painted_positions_index,
            line_layout_index: self
                .line_layout_index
                .shifted(&from.line_layout_index, &to.line_layout_index),
        }
    }

    fn same_place(&self, other: &Self) -> bool {
        self.scene_index == other.scene_index
            && self.window_control_hitboxes_index == other.window_control_hitboxes_index
            && self.mouse_listeners_index == other.mouse_listeners_index
            && self.input_handlers_index == other.input_handlers_index
            && self.cursor_styles_index == other.cursor_styles_index
            && self.accessed_element_states_index == other.accessed_element_states_index
            && self.tab_handle_index == other.tab_handle_index
            && self.painted_positions_index == other.painted_positions_index
            && self.line_layout_index == other.line_layout_index
    }
}

impl RetainedViews {
    pub(crate) fn clear(&mut self) {
        self.records.clear();
        self.by_id.clear();
        self.open.clear();
        self.reused_any = false;
        self.unsettled = false;
    }

    /// Whether a view drawn in this frame, built or drawn again, read which
    /// actions are available or bound (see [`dependencies::ambient::Actions`]).
    /// A record holds what the records nested in it read, so only the
    /// outermost ones are looked at.
    fn read_actions(&self) -> bool {
        let actions = TypeId::of::<dependencies::ambient::Actions>();
        let mut index = 0;
        while let Some(record) = self.records.get(index) {
            if record
                .dependencies
                .globals
                .binary_search_by_key(&actions, |(global, _)| *global)
                .is_ok()
            {
                return true;
            }
            index += record.nested + 1;
        }
        false
    }

    fn find(&self, id: &GlobalElementId) -> Option<usize> {
        let index = *self.by_id.get(id)?;
        matches!(self.records[index].paint, PaintStatus::Painted { .. }).then_some(index)
    }

    fn push(&mut self, record: ViewRecord) -> usize {
        let index = self.records.len();
        match self.by_id.entry(record.id.clone()) {
            collections::hash_map::Entry::Vacant(entry) => {
                entry.insert(index);
            }
            // A view drawn twice in one frame can be drawn again from neither
            // record, since a record found by its id could be either.
            collections::hash_map::Entry::Occupied(entry) => {
                let first = *entry.get();
                self.records[first].paint = PaintStatus::Unpainted;
            }
        }
        self.records.push(record);
        index
    }

    fn add_dependencies(&mut self, records: &[usize], dependencies: &RenderDependencies) {
        for &index in records {
            let record = &mut self.records[index];
            record.dependencies = record.dependencies.union(dependencies);
            record.own_dependencies = record.own_dependencies.union(dependencies);
        }
    }

    fn add_hovers(&mut self, records: &[usize], hovers: &[HoverRead]) {
        if hovers.is_empty() {
            return;
        }
        for &index in records {
            let record = &mut self.records[index];
            record.hovers = record.hovers.iter().chain(hovers).copied().collect();
            record.own_hovers = record.own_hovers.iter().chain(hovers).copied().collect();
        }
    }

    /// Forgets the layout nodes the records name, once the layout engine
    /// replaced its tree (see [`crate::taffy::TaffyLayoutEngine::reclaim_idle_capacity`]):
    /// the ids would name nodes of the old tree, or other nodes of the new
    /// one. The views are laid out afresh when they are next built.
    pub(crate) fn forget_layouts(&mut self) {
        for record in &mut self.records {
            record.layout = None;
            record.prepaint_layout_keys = Rc::new([]);
        }
    }

    /// Shifts the paint ranges of records copied along with a view drawn from
    /// the last frame, now that it was painted, and forgets those not painted.
    fn finish_frame(&mut self) {
        for index in 0..self.records.len() {
            let PaintStatus::Pending { anchor } = self.records[index].paint else {
                continue;
            };
            let shift = match &self.records[anchor].paint {
                PaintStatus::Painted {
                    source: Some(source),
                } => Some((
                    source.clone(),
                    self.records[anchor].paint_range.start.clone(),
                )),
                _ => None,
            };
            let record = &mut self.records[index];
            match shift {
                Some((from, to)) => {
                    record.paint_range = record.paint_range.start.shifted(&from, &to)
                        ..record.paint_range.end.shifted(&from, &to);
                    record.paint = PaintStatus::Painted { source: None };
                }
                None => record.paint = PaintStatus::Unpainted,
            }
        }
    }

    /// Forgets the records made since `transaction` began, whose ranges point
    /// into what the rolled-back transaction drew.
    fn roll_back(&mut self, transaction: &RetainedTransaction) {
        let kept = transaction.records;
        if self.records.len() <= kept {
            return;
        }
        self.records.truncate(kept);
        self.by_id.retain(|_, index| *index < kept);
        self.open.retain(|index| *index < kept);
    }
}

impl App {
    /// Turns view retention on or off in every window: whether a view that
    /// was not notified since the last frame, and read nothing that changed,
    /// is drawn again from what it drew then instead of being built again.
    ///
    /// Off by default, and on from startup with `GPUI_RETAINED_VIEWS=1`.
    /// With it on, a view depends on every entity and global it read while
    /// it was drawn, on versioned state such as scroll handles and list
    /// states, on the pointer and modifier keys if it read them, on the
    /// hovers it was drawn by, and on what it inherits where it is drawn; a
    /// view that only moved is drawn again moved (see
    /// [`crate::Context::set_view_movable`]). Anything else its
    /// render reads (a `Rc<RefCell<..>>`, the clock, a thread-local) it has to
    /// be notified of, or declare: see [`DrawDependency`],
    /// [`Window::rebuild_at`] and [`crate::Context::set_view_retainable`].
    ///
    /// A follow-up frame for a view is asked for by notifying it by id
    /// ([`App::notify`], [`crate::Context::notify`] or
    /// [`Window::request_animation_frame`]), which builds that view again
    /// and nothing that read it; the views around it are drawn again around
    /// it where they can be. Updating an entity and notifying it builds every
    /// view that read it. `GPUI_RETAINED_VIEW_SPLICES=0` builds the views
    /// around a notified view instead.
    ///
    /// With `GPUI_RETAINED_VIEWS_VERIFY=n`, every `n`th frame that drew a
    /// view again is drawn again from scratch, and where the two differ is
    /// logged as an error. The frame drawn from scratch is the one kept and
    /// shown; the work counted ([`Window::frame_work_stats`], the rebuild
    /// reasons) is the first draw's. Every view is rendered a second time
    /// in such a frame, so whatever a render does besides describing the
    /// view (writing a model, asking for an animation frame) is done twice. `GPUI_RETAINED_VIEW_MOVES=0` builds every view
    /// that moved instead of drawing it again moved.
    pub fn set_view_retention(&mut self, enabled: bool) {
        if self.entities.access_log.enabled != enabled {
            self.entities.access_log.enabled = enabled;
            if !enabled {
                self.entities.access_log.forget_all();
            }
            self.refresh_windows();
        }
    }

    /// Whether view retention is on. See [`App::set_view_retention`].
    #[inline]
    pub fn view_retention(&self) -> bool {
        self.entities.access_log.enabled
    }

    /// Starts leaving out of the dependencies being recorded the reads of the
    /// entities `view` said it does not depend on (see
    /// [`crate::Context::untrack_reads_of`]), returning how many to stop
    /// leaving out once it is drawn.
    pub(crate) fn push_untracked_reads(&self, view: EntityId) -> usize {
        match self.untracked_reads.get(&view) {
            Some(untracked) => self.entities.access_log.push_untracked(untracked),
            None => 0,
        }
    }

    /// Stops leaving out the last `count` entities pushed.
    pub(crate) fn pop_untracked_reads(&self, count: usize) {
        self.entities.access_log.pop_untracked(count);
    }

    /// Asks every window for a frame without refreshing it, so that the views
    /// whose dependencies changed are built and the rest drawn again.
    pub(crate) fn request_frame_in_every_window(&mut self) {
        for window in self.windows.values().flatten() {
            window.invalidator.set_dirty(true);
        }
    }
}

impl Window {
    /// Shows a change an element made to state it keeps behind a version,
    /// such as a list scrolled by the wheel. With view retention on, the
    /// views that read the version are built again on their own, so the
    /// window only needs a frame, and only if `changed`; notifying `view`
    /// would also build it and wake its observers. Without retention, `view`
    /// is notified, as it always was.
    pub(crate) fn show_state_change(&mut self, changed: bool, view: EntityId, cx: &mut App) {
        if !cx.view_retention() {
            cx.notify(view);
        } else if changed {
            self.invalidator.set_dirty(true);
        }
    }

    /// Declares that what is being drawn depends on `dependency`: with view
    /// retention on, the view drawing it is built again, rather than drawn
    /// from the last frame, once [`DrawDependency::changed`] is called.
    pub fn depend_on(&self, dependency: &DrawDependency) {
        dependencies::note_state_read(dependency.version());
    }

    /// Declares that what is being drawn will look different at `deadline`,
    /// as a relative time ("5 minutes ago") or an elapsed-time label does.
    /// The window asks for a frame then, and with view retention on, the
    /// view drawing it is drawn again from the last frame until then and
    /// built on the first frame after it.
    pub fn rebuild_at(&mut self, deadline: Instant) {
        dependencies::note_deadline(deadline);
        let retention = &mut self.view_retention;
        if retention
            .deadline
            .is_none_or(|earliest| deadline < earliest)
        {
            retention.deadline = Some(deadline);
        }
    }

    /// Asks for a frame at the earliest time something drawn in the frame
    /// just drawn said it would look different at.
    fn schedule_deadline_frame(&mut self, cx: &App) {
        let Some(deadline) = self.view_retention.deadline.take() else {
            return;
        };
        if let Some((scheduled, _)) = &self.view_retention.deadline_frame
            && *scheduled <= deadline
            && *scheduled > cx.background_executor().now()
        {
            return;
        }
        let delay = deadline.saturating_duration_since(cx.background_executor().now());
        let task = self.spawn(cx, async move |cx| {
            cx.background_executor().timer(delay).await;
            cx.update(|window, _| {
                window.view_retention.deadline_frame = None;
                window.invalidator.set_dirty(true);
            })
            .log_err();
        });
        self.view_retention.deadline_frame = Some((deadline, task));
    }

    /// Why each view built in the last frame was built rather than drawn from
    /// the frame before, with view retention on. Views drawn again are not
    /// listed; [`crate::FrameWorkStats::views_reused`] counts them.
    pub fn view_rebuild_reasons(&self) -> &[(EntityId, ViewRebuildReason)] {
        &self.view_retention.rebuilds
    }

    fn note_rebuild(&mut self, entity: EntityId, reason: ViewRebuildReason) {
        culprits::rebuilt(entity, reason);
        self.view_retention.rebuilds.push((entity, reason));
        let counts = &mut self.frame_work.stats.view_rebuilds;
        let count = match reason {
            ViewRebuildReason::FirstDraw => &mut counts.first_draw,
            ViewRebuildReason::WindowRefresh => &mut counts.window_refresh,
            ViewRebuildReason::Accessibility => &mut counts.accessibility,
            ViewRebuildReason::Notified => &mut counts.notified,
            ViewRebuildReason::EntityChanged => &mut counts.entity_changed,
            ViewRebuildReason::GlobalChanged => &mut counts.global_changed,
            ViewRebuildReason::StateChanged => &mut counts.state_changed,
            ViewRebuildReason::Deadline => &mut counts.deadline,
            ViewRebuildReason::HoverChanged => &mut counts.hover_changed,
            ViewRebuildReason::ContextChanged => &mut counts.context_changed,
            ViewRebuildReason::OptedOut => &mut counts.opted_out,
        };
        *count += 1;
    }

    /// Notes the views notified since the last frame, as the frame begins.
    pub(crate) fn begin_retained_views_frame(
        &mut self,
        notified: &collections::FxHashSet<EntityId>,
    ) {
        let font_generation = self.text_system().font_generation();
        let retention = &mut self.view_retention;
        retention.drawn_font_generation =
            std::mem::replace(&mut retention.drawing_font_generation, font_generation);
        retention.rebuilds.clear();
        retention.notified.clone_from(notified);
        retention.every_frame.bump();
        retention.settling = std::mem::take(&mut retention.settle_requested);
    }

    /// Ends the retained bookkeeping of the frame being drawn.
    pub(crate) fn finish_retained_views_frame(&mut self, cx: &mut App) {
        culprits::frame_finished();
        self.view_retention.actions_fingerprint =
            if cx.view_retention() && self.next_frame.retained_views.read_actions() {
                Some(self.note_changed_actions(cx))
            } else {
                // No view drawn in this frame read the actions, so none is
                // drawn again from an answer they gave. A view that reads them
                // in a later frame is built then, from that frame's actions;
                // a record lives one frame, so it must have been drawn here to
                // be drawn again then.
                None
            };
        let retention = &mut self.view_retention;
        retention.hovers.clear();
        retention.hover_nested.clear();
        retention.group_reads.clear();
        retention.hover_reads.get_mut().clear();
        retention.notified.clear();
        retention.view_stack.clear();
        retention.deferred_inside_notified = false;
        self.next_frame.retained_views.finish_frame();
        self.schedule_deadline_frame(cx);
        self.schedule_settle_frame(cx);
    }

    /// Stamps a change to the actions when the frame being drawn changed
    /// them, returning its fingerprint. The views drawn in it answered from
    /// the frame before's; those that read the actions, of which there is
    /// one, are built again in a follow-up frame, asked for once this draw
    /// has returned.
    fn note_changed_actions(&mut self, cx: &mut App) -> u64 {
        let fingerprint = self.next_frame.dispatch_tree.action_fingerprint();
        let previous = match self.view_retention.actions_fingerprint {
            Some(previous) => previous,
            None => self.rendered_frame.dispatch_tree.action_fingerprint(),
        };
        if fingerprint != previous {
            dependencies::ambient_changed::<dependencies::ambient::Actions>(cx);
            self.spawn(cx, async move |cx| {
                cx.update(|window, _| window.invalidator.set_dirty(true))
                    .log_err();
            })
            .detach();
        }
        fingerprint
    }

    /// Whether the view being drawn, or one around it, was notified since the
    /// last frame.
    fn inside_notified_view(&self) -> bool {
        let notified = &self.view_retention.notified;
        self.view_retention.deferred_inside_notified
            || (!notified.is_empty()
                && self
                    .rendered_entity_stack
                    .iter()
                    .any(|entity| notified.contains(entity)))
    }

    /// The record `id` left last frame, if nothing about this frame rules out
    /// drawing the view again from it. Where it is drawn is checked later.
    fn reusable_view(
        &self,
        id: &GlobalElementId,
        entity: EntityId,
        cx: &App,
    ) -> Result<usize, ViewRebuildReason> {
        let opted_out = cx.non_retainable_views.contains(&entity);
        if opted_out {
            dependencies::note_state_read(&self.view_retention.every_frame);
        }
        if self.refreshing
            || cx.has_active_drag()
            || self.is_inspector_picking(cx)
            || self.view_retention.drawn_font_generation != self.text_system().font_generation()
        {
            return Err(ViewRebuildReason::WindowRefresh);
        }
        if self.a11y.is_active() {
            return Err(ViewRebuildReason::Accessibility);
        }
        if opted_out {
            return Err(ViewRebuildReason::OptedOut);
        }
        if self.dirty_views.contains(&entity) {
            return Err(ViewRebuildReason::Notified);
        }
        if self.next_frame.retained_views.by_id.contains_key(id) {
            return Err(ViewRebuildReason::FirstDraw);
        }
        let index = self
            .rendered_frame
            .retained_views
            .find(id)
            .ok_or(ViewRebuildReason::FirstDraw)?;
        let record = &self.rendered_frame.retained_views.records[index];
        if record.layout_blocked
            || (self.view_retention.settling && record.unsettled)
            || record.asked_for_autoscroll()
        {
            return Err(ViewRebuildReason::ContextChanged);
        }
        let now = cx.background_executor().now();
        match cx.dependencies_changed(&record.dependencies, self.inside_notified_view(), now) {
            Some(DependencyChange::Entity) => return Err(ViewRebuildReason::EntityChanged),
            Some(DependencyChange::Global) => return Err(ViewRebuildReason::GlobalChanged),
            Some(DependencyChange::State) => return Err(ViewRebuildReason::StateChanged),
            Some(DependencyChange::Deadline) => return Err(ViewRebuildReason::Deadline),
            None => {}
        }
        if !record.hovers.iter().all(|hover| hover.unchanged(self)) {
            return Err(ViewRebuildReason::HoverChanged);
        }
        Ok(index)
    }

    fn view_context(&self, bounds: Bounds<Pixels>) -> ViewContext {
        ViewContext {
            bounds,
            content_mask: self.content_mask(),
            text_style: self.text_style(),
            opacity: self.element_opacity,
            rem_size: self.rem_size(),
            image_cache: self.inherited_image_cache(),
            glass_content: self.glass_content,
        }
    }

    fn inherited_image_cache(&self) -> Option<EntityId> {
        self.image_cache_stack.last().map(|cache| cache.entity_id())
    }

    fn view_context_matches(&self, previous: usize, bounds: Bounds<Pixels>) -> bool {
        let record = &self.rendered_frame.retained_views.records[previous];
        let context = &record.context;
        self.groups_unchanged(record)
            && context.bounds == bounds
            && context.opacity == self.element_opacity
            && context.rem_size == self.rem_size()
            && context.content_mask == self.content_mask()
            && context.text_style == self.text_style()
            && context.image_cache == self.inherited_image_cache()
            && context.glass_content == self.glass_content
    }

    /// Whether the groups `record` resolved outside its view resolve to the
    /// same hitboxes where it is being drawn now.
    fn groups_unchanged(&self, record: &ViewRecord) -> bool {
        let groups = &self.view_retention.prepaint_groups;
        record
            .groups
            .iter()
            .all(|read| groups.top(&read.name) == read.hitbox)
    }

    /// Lays out the view last frame's record `previous` stands for as it was
    /// laid out then, without building it, if its nodes are all still there.
    fn reuse_view_layout(&mut self, previous: usize) -> Option<LayoutId> {
        let records = &self.rendered_frame.retained_views.records;
        let record = &records[previous];
        let layout = record.layout.as_ref()?;
        let inherited_matches = record.context.text_style == self.text_style()
            && record.context.rem_size == self.rem_size()
            && record.context.image_cache == self.inherited_image_cache();
        if layout.view_key.is_none()
            || layout.view_key != self.layout_keys.current()
            || !inherited_matches
        {
            return None;
        }
        let root = layout.root;
        // The view's own nodes and those of the views nested in it, each of
        // which recorded its own.
        let subtree = &records[previous..=previous + record.nested];
        let key_sets = subtree
            .iter()
            .filter_map(|record| record.layout.as_deref())
            .map(|layout| layout.keys.as_slice());
        let engine = self.layout_engine.as_mut()?;
        let keys_before = engine.claimed_keys_len();
        if !engine.try_keep_retained_sets(key_sets, root) {
            return None;
        }
        let keys_after = engine.claimed_keys_len();
        let states = &mut self.next_frame.accessed_element_states;
        let states_before = states.len();
        for layout in subtree.iter().filter_map(|record| record.layout.as_deref()) {
            states.extend(layout.element_states.iter().cloned());
        }
        let states_after = states.len();
        self.note_nested(keys_before..keys_after, states_before..states_after);
        Some(root)
    }

    /// Marks stretches of the claimed-key log and of the accessed element
    /// states as a nested view's, which the records being made leave out.
    fn note_nested(&mut self, keys: Range<usize>, states: Range<usize>) {
        let retention = &mut self.view_retention;
        if retention.open_recordings > 0 {
            retention.nested_keys.push(keys);
            retention.nested_states.push(states);
        }
    }

    /// Ends a recording of a view's own layout keys begun at `start`, whose
    /// nested stretches begin at `nested` in `nested_keys`, handing the
    /// whole recording to the one around it as nested.
    fn finish_own_keys(&mut self, start: usize, nested: usize) -> Vec<u64> {
        let retention = &mut self.view_retention;
        retention.open_recordings = retention.open_recordings.saturating_sub(1);
        let Some(engine) = self.layout_engine.as_mut() else {
            return Vec::new();
        };
        let nested_from = nested.min(retention.nested_keys.len());
        let (keys, end) = engine.finish_recording_own_keys(start, &retention.nested_keys[nested_from..]);
        retention.nested_keys.truncate(nested);
        if retention.open_recordings == 0 {
            retention.nested_keys.clear();
            retention.nested_states.clear();
        } else {
            retention.nested_keys.push(start..end);
        }
        keys
    }

    /// The element states accessed since `start`, leaving out the stretches
    /// nested views took from `nested` on in `nested_states`, handing the
    /// whole stretch to the recording around it as nested.
    fn own_element_states(&mut self, start: usize, nested: usize) -> Vec<(GlobalElementId, TypeId)> {
        let states = &self.next_frame.accessed_element_states;
        let retention = &mut self.view_retention;
        let end = states.len();
        let mut own = Vec::new();
        let mut from = start;
        for range in &retention.nested_states[nested.min(retention.nested_states.len())..] {
            if range.start > from {
                own.extend(states[from..range.start.min(end)].iter().cloned());
            }
            from = from.max(range.end);
        }
        if from < end {
            own.extend(states[from..end].iter().cloned());
        }
        retention.nested_states.truncate(nested);
        if retention.open_recordings > 0 {
            retention.nested_states.push(start..end);
        }
        own
    }

    fn begin_view_layout(&mut self, cx: &mut App) -> LayoutRecording {
        let (keys, transient) = match self.layout_engine.as_mut() {
            Some(engine) => (Some(engine.record_claimed_keys()), engine.transient_count()),
            None => (None, 0),
        };
        let retention = &mut self.view_retention;
        if keys.is_some() {
            retention.open_recordings += 1;
        }
        LayoutRecording {
            keys,
            transient,
            element_states: self.next_frame.accessed_element_states.len(),
            nested_keys: retention.nested_keys.len(),
            nested_states: retention.nested_states.len(),
            dependencies: cx.begin_recording_dependencies(),
        }
    }

    fn finish_view_layout(
        &mut self,
        recording: LayoutRecording,
        root: LayoutId,
        cx: &mut App,
    ) -> (Option<Rc<RetainedLayout>>, Recorded) {
        let dependencies = cx.finish_recording_dependencies(recording.dependencies);
        let Some(keys) = recording.keys else {
            return (None, dependencies);
        };
        // The states first: the recording around this one takes the keys
        // and states it hands over as nested once both are handed over.
        let element_states =
            self.own_element_states(recording.element_states, recording.nested_states);
        let keys = self.finish_own_keys(keys, recording.nested_keys);
        // A node made without a key is gone at the end of the frame, so the
        // layout cannot be taken again without building the view.
        if self
            .layout_engine
            .as_ref()
            .is_none_or(|engine| engine.transient_count() != recording.transient)
        {
            return (None, dependencies);
        }
        let layout = RetainedLayout {
            root,
            view_key: self.layout_keys.current(),
            keys,
            element_states,
        };
        (Some(Rc::new(layout)), dependencies)
    }

    /// Draws the view last frame's record `previous` stands for again, as far
    /// as its prepaint goes, returning its record in this frame: where it was,
    /// or `moved`.
    fn reuse_view_prepaint(
        &mut self,
        previous: usize,
        moved: Option<ViewMove>,
        cx: &mut App,
    ) -> usize {
        let (prepaint_range, dependencies, hovers) = {
            let records = &self.rendered_frame.retained_views.records;
            let record = &records[previous];
            // The nodes the view and the views nested in it laid out as they
            // prepainted, and their layouts' nodes where those were not kept
            // as the view was laid out.
            if let Some(engine) = self.layout_engine.as_mut() {
                let keys_before = engine.claimed_keys_len();
                for nested in &records[previous..=previous + record.nested] {
                    engine.keep_retained(&nested.prepaint_layout_keys);
                    if let Some(layout) = nested.layout.as_deref() {
                        engine.keep_retained(&layout.keys);
                    }
                }
                let keys_after = engine.claimed_keys_len();
                let retention = &mut self.view_retention;
                if retention.open_recordings > 0 {
                    retention.nested_keys.push(keys_before..keys_after);
                }
            }
            (
                record.prepaint_range.clone(),
                record.dependencies.clone(),
                record.hovers.clone(),
            )
        };
        let hovers_start = self.view_retention.hovers.len();
        self.frame_work.stats.views_reused += 1;
        if moved.is_some() {
            self.frame_work.stats.views_moved += 1;
            self.view_retention.moved_this_frame = true;
        }
        if let Some(deadline) = dependencies.rebuild_at {
            self.rebuild_at(deadline);
        }
        // What the view read is read again: by the window, which draws again
        // when one of those entities is notified, and by the views around it.
        cx.replay_dependencies(&dependencies);
        self.take_hover_reads();
        let hovers_start = hovers_start.max(self.view_retention.hovers.len());
        self.view_retention.hovers.extend_from_slice(&hovers);
        let hovers_end = self.view_retention.hovers.len();
        self.view_retention
            .hover_nested
            .push(hovers_start..hovers_end);

        let start = self.prepaint_index();
        self.reuse_prepaint_moved(prepaint_range.clone(), moved.as_ref());
        let end = self.prepaint_index();

        // The nested records can be shifted into this frame only if the copy
        // is what it was copied from, entry for entry; the text system skips
        // a range shaped with fonts that have since changed.
        let copied_whole =
            end.same_place(&prepaint_range.end.shifted(&prepaint_range.start, &start));
        // Nothing written since the records were made changed what they read,
        // or the view would not be drawn again: they are up to date as of now.
        let writes_now = cx.entities.write_generation();
        let source = &self.rendered_frame.retained_views;
        let target = &mut self.next_frame.retained_views;
        target.reused_any = true;
        let anchor = target.records.len();
        let mut unsettled = false;
        let nested = if copied_whole {
            source.records[previous].nested
        } else {
            0
        };
        for index in previous..=previous + nested {
            let record = &source.records[index];
            let paint = match record.paint {
                PaintStatus::Painted { .. } => PaintStatus::Pending { anchor },
                _ => PaintStatus::Unpainted,
            };
            let prepaint_range = if index == previous {
                start.clone()..end.clone()
            } else {
                record
                    .prepaint_range
                    .start
                    .shifted(&prepaint_range.start, &start)
                    ..record
                        .prepaint_range
                        .end
                        .shifted(&prepaint_range.start, &start)
            };
            target.push(ViewRecord {
                id: record.id.clone(),
                prepaint_range,
                paint_range: record.paint_range.clone(),
                paint,
                nested: if index == previous {
                    nested
                } else {
                    record.nested
                },
                context: match &moved {
                    Some(moved) => moved.move_record(record),
                    None => record.context.clone(),
                },
                paint_context: record.paint_context.clone(),
                dependencies: record.dependencies.written_up_to(writes_now),
                own_dependencies: record.own_dependencies.written_up_to(writes_now),
                hovers: record.hovers.clone(),
                own_hovers: record.own_hovers.clone(),
                groups: record.groups.clone(),
                fresh_hitboxes: 0..0,
                prepaint_layout_keys: record.prepaint_layout_keys.clone(),
                layout: record.layout.clone(),
                unsettled: record.unsettled || moved.is_some(),
                stays_put: record.stays_put,
                moved: if index == previous { moved } else { None },
                paint_mask: match &moved {
                    Some(moved) if index != previous => moved.mask_of(&record.paint_mask),
                    _ => record.paint_mask,
                },
                source: record.source.clone(),
                layout_scope: record.layout_scope,
                inherited_groups: record.inherited_groups.clone(),
                layout_blocked: record.layout_blocked,
            });
            unsettled |= record.unsettled || moved.is_some();
        }
        target.unsettled |= unsettled;
        anchor
    }

    /// Draws the view whose prepaint [`Self::reuse_view_prepaint`] drew again
    /// as far as its paint goes.
    fn reuse_view_paint(&mut self, index: usize) {
        let record = &self.next_frame.retained_views.records[index];
        if !matches!(record.paint, PaintStatus::Pending { .. }) {
            return;
        }
        let source = record.paint_range.clone();
        let hovers = record.hovers.clone();
        let groups = record.groups.clone();
        let recorded = record.paint_context.clone();
        let current = self.paint_context();
        let paint_mask = self.content_mask();
        // Moved with what it painted from the mask around it then to the one
        // around it now, which can differ from the ones around its prepaint.
        let moved = record.moved.as_ref().map(|moved| ViewMove {
            delta: moved.delta,
            old_outer: record.paint_mask,
            new_outer: paint_mask,
        });

        let start = self.paint_index();
        self.reuse_paint_moved(
            source.clone(),
            recorded.transition,
            current.transition,
            moved.as_ref(),
        );
        let end = self.paint_index();
        let copied_whole = end.same_place(&source.end.shifted(&source.start, &start));
        let record = &mut self.next_frame.retained_views.records[index];
        record.paint_range = start..end;
        record.paint = PaintStatus::Painted {
            source: copied_whole.then_some(source.start),
        };
        record.paint_context = current.clone();
        record.paint_mask = paint_mask;

        // The hovers were checked against the last frame's hitboxes, and this
        // frame's may put something over the view; the text effect, opacity
        // cycle and glass mode it inherits are only known now. Either is too late
        // to build the view in this frame, so it is notified for the next.
        if !hovers.iter().all(|hover| hover.unchanged(self)) || !recorded.effects_match(&current) {
            self.request_animation_frame();
        }
        self.take_hover_reads();
        let hovers_start = self.view_retention.hovers.len();
        self.view_retention.hovers.extend_from_slice(&hovers);
        let hovers_end = self.view_retention.hovers.len();
        self.view_retention
            .hover_nested
            .push(hovers_start..hovers_end);
        self.view_retention.group_reads.extend_from_slice(&groups);
    }

    fn paint_context(&self) -> PaintContext {
        PaintContext {
            shimmer: self.text_shimmer_stack.last().copied(),
            opacity_cycle: self.opacity_cycle_stack.last().copied(),
            glass_content: self.glass_content,
            transition: self.next_frame.scene.current_transition(),
        }
    }

    fn begin_view_prepaint(
        &mut self,
        id: &GlobalElementId,
        source: ViewSource,
        cx: &mut App,
    ) -> ViewRecording {
        let layout_scope = self.layout_keys.prepaint_scope();
        self.frame_work.stats.views_rendered += 1;
        let start = self.prepaint_index();
        let views = &mut self.next_frame.retained_views;
        let index = (!views.by_id.contains_key(id)).then(|| {
            let index = views.push(ViewRecord {
                id: id.clone(),
                prepaint_range: start.clone()..start,
                paint_range: PaintIndex::default()..PaintIndex::default(),
                paint: PaintStatus::Unpainted,
                nested: 0,
                context: Rc::new(ViewContext {
                    bounds: Bounds::default(),
                    content_mask: ContentMask::default(),
                    text_style: TextStyle::default(),
                    opacity: 1.,
                    rem_size: Pixels::ZERO,
                    image_cache: None,
                    glass_content: false,
                }),
                paint_context: PaintContext::default(),
                dependencies: RenderDependencies::default(),
                own_dependencies: RenderDependencies::default(),
                hovers: Rc::new([]),
                own_hovers: Rc::new([]),
                groups: Rc::new([]),
                fresh_hitboxes: 0..0,
                prepaint_layout_keys: Rc::new([]),
                layout: None,
                unsettled: false,
                stays_put: false,
                moved: None,
                paint_mask: ContentMask::default(),
                source: Rc::new(source),
                layout_scope,
                inherited_groups: Rc::new([]),
                layout_blocked: false,
            });
            views.open.push(index);
            index
        });
        self.take_hover_reads();
        self.view_retention.view_stack.push(id.clone());
        let layout_keys = self
            .layout_engine
            .as_mut()
            .map_or(0, |engine| engine.record_claimed_keys());
        let retention = &mut self.view_retention;
        retention.open_recordings += 1;
        ViewRecording {
            index,
            dependencies: cx.begin_recording_dependencies(),
            layout_keys,
            nested_keys: retention.nested_keys.len(),
            hovers_start: retention.hovers.len(),
            hover_nested: retention.hover_nested.len(),
            hitboxes_start: self.next_hitbox_id.0,
            view: self.rendered_entity_stack.last().copied(),
            writes_start: Writes::now(cx),
        }
    }

    fn finish_view_prepaint(
        &mut self,
        recording: ViewRecording,
        bounds: Bounds<Pixels>,
        layout: Option<Rc<RetainedLayout>>,
        layout_dependencies: Option<Recorded>,
        cx: &mut App,
    ) -> Option<usize> {
        let prepaint_layout_keys = self.finish_own_keys(recording.layout_keys, recording.nested_keys);
        let dependencies = cx.finish_recording_dependencies(recording.dependencies);
        self.take_hover_reads();
        self.view_retention.view_stack.pop();
        let index = recording.index?;
        let context = self.view_context(bounds);
        let end = self.prepaint_index();
        let hitboxes_end = self.next_hitbox_id.0;
        let hovers: Rc<[HoverRead]> = self.view_retention.hovers[recording.hovers_start..].into();
        let own_hovers: Rc<[HoverRead]> = self
            .own_hovers(recording.hovers_start, recording.hover_nested)
            .into();
        let views = &mut self.next_frame.retained_views;
        views.open.retain(|open| *open != index);
        let nested = views.records.len() - index - 1;
        let (unsettled, stays_put) = views.records[index + 1..]
            .iter()
            .fold((false, false), |(unsettled, stays_put), record| {
                (unsettled || record.unsettled, stays_put || record.stays_put)
            });
        let wrote = recording.writes_start != Writes::now(cx);
        let fixed = recording
            .view
            .is_some_and(|view| cx.fixed_views.contains(&view));
        if culprits::enabled()
            && let Some(view) = recording.view
        {
            culprits::note_stays_put(view, fixed, wrote.then_some("prepainting"), stays_put);
        }
        let record = &mut views.records[index];
        record.unsettled = unsettled;
        record.stays_put = stays_put || wrote || fixed;
        record.prepaint_range.end = end;
        record.nested = nested;
        record.context = Rc::new(context);
        // A recording is as of the generation it began at, so only one that
        // was made counts: an empty one would date everything back to none.
        (record.dependencies, record.own_dependencies) = match layout_dependencies {
            Some(layout) => (
                layout.all.union(&dependencies.all),
                layout.own.union(&dependencies.own),
            ),
            None => (dependencies.all, dependencies.own),
        };
        record.hovers = hovers;
        record.own_hovers = own_hovers;
        record.fresh_hitboxes = recording.hitboxes_start..hitboxes_end;
        record.prepaint_layout_keys = prepaint_layout_keys.into();
        record.layout = layout;
        Some(index)
    }

    fn begin_view_paint(
        &mut self,
        index: usize,
        id: &GlobalElementId,
        cx: &mut App,
    ) -> ViewPaintRecording {
        self.take_hover_reads();
        self.view_retention.view_stack.push(id.clone());
        let paint_context = self.paint_context();
        let paint_mask = self.content_mask();
        let inherited_groups = self.group_hitboxes.tops();
        let record = &mut self.next_frame.retained_views.records[index];
        record.paint_context = paint_context;
        record.paint_mask = paint_mask;
        if !inherited_groups.is_empty() || !record.inherited_groups.is_empty() {
            record.inherited_groups = inherited_groups.into();
        }
        ViewPaintRecording {
            index: Some(index),
            start: self.paint_index(),
            hovers_start: self.view_retention.hovers.len(),
            hover_nested: self.view_retention.hover_nested.len(),
            groups_start: self.view_retention.group_reads.len(),
            dependencies: cx.begin_recording_dependencies(),
            writes_start: Writes::now(cx),
        }
    }

    fn finish_view_paint(&mut self, recording: ViewPaintRecording, cx: &mut App) {
        self.take_hover_reads();
        self.view_retention.view_stack.pop();
        let dependencies = cx.finish_recording_dependencies(recording.dependencies);
        let Some(index) = recording.index else {
            return;
        };
        let end = self.paint_index();
        let own_painted_hovers = self.own_hovers(recording.hovers_start, recording.hover_nested);
        let painted_hovers = &self.view_retention.hovers[recording.hovers_start..];
        let record = &mut self.next_frame.retained_views.records[index];
        let fresh = record.fresh_hitboxes.clone();
        let mut groups: Vec<GroupRead> = Vec::new();
        for read in &self.view_retention.group_reads[recording.groups_start..] {
            let outside = read.hitbox.is_none_or(|hitbox| !fresh.contains(&hitbox.0));
            if outside && !groups.contains(read) {
                groups.push(read.clone());
            }
        }
        record.groups = groups.into();
        let wrote = recording.writes_start != Writes::now(cx);
        if wrote
            && culprits::enabled()
            && let Some(view) = splice::view_entity(&record.id)
        {
            culprits::note_stays_put(view, false, Some("painting"), false);
        }
        record.stays_put |= wrote;
        record.paint_range = recording.start..end;
        record.paint = PaintStatus::Painted { source: None };
        if !painted_hovers.is_empty() {
            record.hovers = record.hovers.iter().chain(painted_hovers).copied().collect();
        }
        if !own_painted_hovers.is_empty() {
            record.own_hovers = record
                .own_hovers
                .iter()
                .chain(&own_painted_hovers)
                .copied()
                .collect();
        }
        record.dependencies = record.dependencies.union(&dependencies.all);
        record.own_dependencies = record.own_dependencies.union(&dependencies.own);
    }

    /// The hovers read since `start` outside the stretches views nested in
    /// the one being finished took from `nested` on, which become one
    /// stretch, for the view around it to leave out.
    fn own_hovers(&mut self, start: usize, nested: usize) -> Vec<HoverRead> {
        let retention = &mut self.view_retention;
        let end = retention.hovers.len();
        let mut own = Vec::new();
        let mut from = start;
        for range in &retention.hover_nested[nested.min(retention.hover_nested.len())..] {
            if range.start > from {
                own.extend_from_slice(&retention.hovers[from..range.start.min(end)]);
            }
            from = from.max(range.end);
        }
        if from < end {
            own.extend_from_slice(&retention.hovers[from..end]);
        }
        retention.hover_nested.truncate(nested);
        retention.hover_nested.push(start..end);
        own
    }

    /// Takes the hovers read since the last call into `hovers`.
    fn take_hover_reads(&mut self) {
        let retention = &mut self.view_retention;
        let reads = retention.hover_reads.get_mut();
        if !reads.is_empty() {
            retention.hovers.append(reads);
        }
    }

    /// The retained views being prepainted right now, for something deferred
    /// from them to count as theirs.
    pub(crate) fn enclosing_views(&self) -> EnclosingViews {
        let mut views: SmallVec<[usize; 4]> =
            self.next_frame.retained_views.open.iter().copied().collect();
        for &index in &self.view_retention.deferring_views {
            if !views.contains(&index) {
                views.push(index);
            }
        }
        EnclosingViews {
            views,
            inside_notified: self.inside_notified_view(),
            untracked: self.view_retention.ambient_reads.untracked(),
        }
    }

    /// Starts drawing something deferred as a part of the retained views it
    /// was deferred from, in its prepaint or its paint.
    pub(crate) fn begin_deferred_view(
        &mut self,
        enclosing: &EnclosingViews,
        cx: &mut App,
    ) -> Option<DeferredViewRecording> {
        self.view_retention.deferred_inside_notified = enclosing.inside_notified;
        if enclosing.views.is_empty() {
            return None;
        }
        let untracked = cx.entities.access_log.push_untracked(&enclosing.untracked);
        let views = &self.next_frame.retained_views;
        let ids: SmallVec<[GlobalElementId; 4]> = enclosing
            .views
            .iter()
            .map(|&index| views.records[index].id.clone())
            .collect();
        self.take_hover_reads();
        self.view_retention.view_stack.extend(ids);
        let deferring_before = std::mem::replace(
            &mut self.view_retention.deferring_views,
            enclosing.views.clone(),
        );
        Some(DeferredViewRecording {
            enclosing: enclosing.clone(),
            deferring_before,
            dependencies: cx.begin_recording_dependencies(),
            hovers_start: self.view_retention.hovers.len(),
            untracked,
        })
    }

    /// Ends `recording`, adding what the deferred draw read, and the hovers it
    /// was drawn by, to the retained views it was deferred from.
    pub(crate) fn finish_deferred_view(
        &mut self,
        recording: Option<DeferredViewRecording>,
        cx: &mut App,
    ) {
        let Some(recording) = recording else {
            return;
        };
        self.take_hover_reads();
        let dependencies = cx.finish_recording_dependencies(recording.dependencies);
        cx.entities.access_log.pop_untracked(recording.untracked);
        self.view_retention.view_stack.clear();
        self.view_retention.deferring_views = recording.deferring_before;
        let enclosing = &recording.enclosing.views;
        let views = &mut self.next_frame.retained_views;
        // What a view deferred is part of drawing it, so it is its own.
        views.add_dependencies(enclosing, &dependencies.all);
        views.add_hovers(enclosing, &self.view_retention.hovers[recording.hovers_start..]);
    }

    /// The records made from here on, for a deferred draw to note which
    /// views it drew.
    pub(crate) fn retained_records_len(&self) -> usize {
        self.next_frame.retained_views.records.len()
    }

    /// Copies the records of the views a deferred draw drew last frame,
    /// `records` there, along with its prepaint drawn again from `from` to
    /// `to`, returning where they landed. Without them, the views it drew
    /// would be built afresh once the views around them are built again.
    pub(crate) fn copy_deferred_records(
        &mut self,
        records: Range<usize>,
        from: &PrepaintStateIndex,
        to: &PrepaintStateIndex,
        cx: &App,
    ) -> Range<usize> {
        let writes_now = cx.entities.write_generation();
        let source = &self.rendered_frame.retained_views;
        let Some(records) = source.records.get(records) else {
            return 0..0;
        };
        if let Some(engine) = self.layout_engine.as_mut() {
            for record in records {
                engine.keep_retained(&record.prepaint_layout_keys);
                if let Some(layout) = record.layout.as_deref() {
                    engine.keep_retained(&layout.keys);
                }
            }
        }
        let target = &mut self.next_frame.retained_views;
        let start = target.records.len();
        for record in records {
            let index = target.records.len();
            let paint = match record.paint {
                // Shifted once the deferred draw is painted again; see
                // [`Self::paint_deferred_records`].
                PaintStatus::Painted { .. } => PaintStatus::Pending { anchor: index },
                _ => PaintStatus::Unpainted,
            };
            target.unsettled |= record.unsettled;
            target.push(record.copied(from, to, paint, writes_now));
        }
        start..target.records.len()
    }

    /// Shifts the paint ranges of the records [`Self::copy_deferred_records`]
    /// copied, now that their deferred draw's paint was drawn again from
    /// `from` to `to`.
    pub(crate) fn paint_deferred_records(
        &mut self,
        records: Range<usize>,
        from: &PaintIndex,
        to: &PaintIndex,
    ) {
        let target = &mut self.next_frame.retained_views;
        for record in target.records.get_mut(records).into_iter().flatten() {
            if let PaintStatus::Pending { .. } = record.paint {
                record.paint_range =
                    record.paint_range.start.shifted(from, to)..record.paint_range.end.shifted(from, to);
                record.paint = PaintStatus::Painted { source: None };
            }
        }
    }

    /// Where a transaction begins, for its records to be rolled back.
    pub(crate) fn begin_retained_transaction(&self) -> RetainedTransaction {
        RetainedTransaction {
            records: self.next_frame.retained_views.records.len(),
            hovers: self.view_retention.hovers.len(),
            hover_nested: self.view_retention.hover_nested.len(),
        }
    }

    /// Forgets the records a rolled-back transaction made, which point into
    /// the ranges it truncated.
    pub(crate) fn roll_back_retained_transaction(&mut self, transaction: RetainedTransaction) {
        self.next_frame.retained_views.roll_back(&transaction);
        self.view_retention.hovers.truncate(transaction.hovers);
        self.view_retention
            .hover_nested
            .truncate(transaction.hover_nested);
    }

    /// Draws the frame just drawn again from scratch, and logs where it
    /// differs from the one drawn with views drawn again from the frame
    /// before, when `GPUI_RETAINED_VIEWS_VERIFY` asks for it. The frame drawn
    /// from scratch is the one kept.
    pub(crate) fn verify_retained_frame(
        &mut self,
        arena_clear_needed: ArenaClearNeeded,
        cx: &mut App,
    ) -> ArenaClearNeeded {
        let Some(interval) = self.view_retention.verification_interval else {
            return arena_clear_needed;
        };
        if !self.rendered_frame.retained_views.reused_any {
            return arena_clear_needed;
        }
        let retention = &mut self.view_retention;
        retention.frames_since_verification += 1;
        if retention.frames_since_verification < interval {
            return arena_clear_needed;
        }
        retention.frames_since_verification = 0;
        arena_clear_needed.clear(cx);
        let retained = describe_frame(self);
        // The work counted is the frame's that drew views again, which is
        // what verifying is meant to leave as it is.
        let stats = self.frame_work.stats;
        let shaping = self.text_system().shaping_stats();
        let layout_counts = self
            .layout_engine
            .as_ref()
            .map(|engine| engine.retention_counts());
        let rebuilds = std::mem::take(&mut self.view_retention.rebuilds);
        culprits::suspend(true);
        self.refreshing = true;
        let arena_clear_needed = self.draw_frame(cx);
        culprits::suspend(false);
        self.frame_work.stats = stats;
        self.text_system().restore_shaping_stats(shaping);
        if let Some((engine, counts)) = self.layout_engine.as_mut().zip(layout_counts) {
            engine.restore_retention_counts(counts);
        }
        self.view_retention.rebuilds = rebuilds;
        let from_scratch = describe_frame(self);
        if let Some(difference) = first_difference(&retained, &from_scratch) {
            log::error!(
                "a frame that drew views again from the frame before differs from the same \
                 frame drawn from scratch at {difference}"
            );
        }
        arena_clear_needed
    }
}

/// What a frame shows and where it can be hit, as lines two frames can be
/// compared by. Atlas tiles are left out, since the same glyph can land in
/// another tile.
///
/// A content mask is described by what it leaves visible of what it clips
/// (see [`crate::scene::primitive_extent`]), not as it is: masks that cut the
/// same pixels out of a primitive or a hitbox are the same to whoever looks.
/// A view drawn again moved clips what it drew with masks moved with it,
/// which differ from the ones a frame drawn from scratch pushes wherever
/// they clip nothing. Positions are rounded to 1/64 of a device pixel, the
/// rounding a moved copy and a fresh layout can differ by. Primitives are
/// compared as a set, each with its draw order; hitboxes in order, which is
/// which one is on top.
pub(crate) fn describe_frame(window: &Window) -> Vec<String> {
    use crate::scene::{Primitive, primitive_extent};
    fn round(value: f32) -> f32 {
        (value * 64.).round() / 64. + 0.
    }
    fn rounded(bounds: Bounds<crate::ScaledPixels>) -> Bounds<crate::ScaledPixels> {
        use crate::ScaledPixels;
        Bounds {
            origin: crate::point(
                ScaledPixels(round(bounds.origin.x.0)),
                ScaledPixels(round(bounds.origin.y.0)),
            ),
            size: crate::size(
                ScaledPixels(round(bounds.size.width.0)),
                ScaledPixels(round(bounds.size.height.0)),
            ),
        }
    }
    let scale_factor = window.scale_factor();
    let rounded_pixels = |bounds: Bounds<Pixels>| rounded(bounds.scale(scale_factor));
    fn visible(primitive: impl Into<Primitive>) -> Primitive {
        let mut primitive = primitive.into();
        let extent = primitive_extent(&primitive);
        let clip = rounded(extent.intersect(&primitive.content_mask().bounds));
        let bounds = rounded(*primitive.bounds());
        match &mut primitive {
            Primitive::Shadow(shadow) => {
                shadow.bounds = bounds;
                shadow.content_mask.bounds = clip;
            }
            Primitive::Quad(quad) => {
                quad.bounds = bounds;
                quad.content_mask.bounds = clip;
            }
            Primitive::Underline(underline) => {
                underline.bounds = bounds;
                underline.content_mask.bounds = clip;
            }
            Primitive::MonochromeSprite(sprite) => {
                sprite.bounds = bounds;
                sprite.content_mask.bounds = clip;
            }
            Primitive::SubpixelSprite(sprite) => {
                sprite.bounds = bounds;
                sprite.content_mask.bounds = clip;
            }
            Primitive::PolychromeSprite(sprite) => {
                sprite.bounds = bounds;
                sprite.content_mask.bounds = clip;
            }
            Primitive::Path(path) => {
                path.bounds = bounds;
                path.content_mask.bounds = clip;
            }
            Primitive::Surface(surface) => {
                surface.bounds = bounds;
                surface.content_mask.bounds = clip;
            }
        }
        primitive
    }
    let scene = &window.rendered_frame.scene;
    let mut lines = Vec::new();
    for primitive in scene
        .shadows
        .iter()
        .map(|shadow| visible(*shadow))
        .chain(scene.quads.iter().map(|quad| visible(*quad)))
        .chain(scene.underlines.iter().map(|underline| visible(*underline)))
        .chain(
            scene
                .monochrome_sprites
                .iter()
                .map(|sprite| visible(*sprite)),
        )
        .chain(scene.subpixel_sprites.iter().map(|sprite| visible(*sprite)))
        .chain(
            scene
                .polychrome_sprites
                .iter()
                .map(|sprite| visible(*sprite)),
        )
        .chain(scene.paths.iter().map(|path| visible(path.clone())))
    {
        lines.push(match primitive {
            Primitive::Shadow(shadow) => format!("{shadow:?}"),
            Primitive::Quad(quad) => format!("{quad:?}"),
            Primitive::Underline(underline) => format!("{underline:?}"),
            Primitive::MonochromeSprite(sprite) => format!(
                "monochrome {} {:?} {:?} {:?} {:?}",
                sprite.order, sprite.bounds, sprite.content_mask, sprite.color, sprite.effect
            ),
            Primitive::SubpixelSprite(sprite) => format!(
                "subpixel {} {:?} {:?} {:?} {:?}",
                sprite.order, sprite.bounds, sprite.content_mask, sprite.color, sprite.effect
            ),
            Primitive::PolychromeSprite(sprite) => format!(
                "polychrome {} {:?} {:?}",
                sprite.order, sprite.bounds, sprite.content_mask
            ),
            Primitive::Path(path) => {
                format!("path {} {:?} {:?}", path.order, path.bounds, path.content_mask)
            }
            Primitive::Surface(surface) => format!("surface {:?}", surface.bounds),
        });
    }
    // Primitives of one draw order do not overlap, so the order they come in
    // within it is not seen; sprites of one order are sorted by their atlas
    // tile, which depends on the order glyphs were first rasterized in.
    lines.sort_unstable();
    let frame = &window.rendered_frame;
    lines.extend(frame.hitboxes.iter().map(|hitbox| {
        format!(
            "hitbox {:?} {:?} {:?}",
            rounded_pixels(hitbox.bounds),
            rounded_pixels(hitbox.bounds.intersect(&hitbox.content_mask.bounds)),
            hitbox.behavior
        )
    }));
    lines.extend(frame.window_control_hitboxes.iter().map(|(area, hitbox)| {
        format!("window control {area:?} {:?}", rounded_pixels(hitbox.bounds))
    }));
    lines.push(format!("overlay starts at {}", frame.overlay_scene_start));
    lines
}

/// Where two frames' descriptions first differ, if they do.
pub(crate) fn first_difference(retained: &[String], from_scratch: &[String]) -> Option<String> {
    if retained == from_scratch {
        return None;
    }
    let first = retained
        .iter()
        .zip(from_scratch)
        .position(|(retained, from_scratch)| retained != from_scratch)
        .unwrap_or(retained.len().min(from_scratch.len()));
    Some(format!(
        "line {first} of {} against {}: {:?} against {:?}",
        retained.len(),
        from_scratch.len(),
        retained.get(first),
        from_scratch.get(first)
    ))
}

/// How a view was laid out, for its prepaint to follow up on. This is the
/// layout state of `impl Element for ViewElement`, so it is `pub`, in a
/// module nothing outside the crate can name.
#[doc(hidden)]
pub struct ViewLayoutState(pub(crate) ViewLayout);

/// What a view's prepaint left for its paint; see [`ViewLayoutState`].
#[doc(hidden)]
pub struct ViewPrepaintState(pub(crate) ViewPrepaint);

pub(crate) enum ViewLayout {
    /// Laid out as views are without retention: the element, if it was
    /// built.
    Unretained(Option<AnyElement>),
    /// A cached view, laid out by its style, and built at prepaint if at all.
    Cached,
    /// Built, and laid out by its content.
    Built {
        element: AnyElement,
        layout: Option<Rc<RetainedLayout>>,
        dependencies: Recorded,
    },
    /// Laid out as it was last frame without being built, from the record
    /// at this index, which it is drawn again from if nothing moved it.
    Retained { previous: usize },
    /// Laid out as it was last frame, to be drawn again around the views
    /// nested in it that are built; see [`splice`].
    Spliced(splice::Splice),
}

pub(crate) enum ViewPrepaint {
    /// Prepainted as views are without retention.
    Unretained(Option<AnyElement>),
    /// Built this frame, into the record at this index if it has one.
    Built {
        element: AnyElement,
        record: Option<usize>,
    },
    /// Drawn again from the last frame, as the record at this index.
    Reused(usize),
    /// Drawn again from the last frame around the views nested in it that
    /// were built.
    Spliced(splice::SplicedPrepaint),
}

impl Window {
    /// Lays out an entity view with view retention on.
    pub(crate) fn request_retained_view_layout(
        &mut self,
        entity: EntityId,
        view_name: ViewName,
        cached_style: Option<&StyleRefinement>,
        global_id: &GlobalElementId,
        render: &mut dyn FnMut(&mut Window, &mut App) -> AnyElement,
        cx: &mut App,
    ) -> (LayoutId, ViewLayout) {
        let untracked = cx.push_untracked_reads(entity);
        let layout = self.with_named_view(entity, view_name, |window| {
            if let Some(style) = cached_style
                && !window.is_inspector_picking(cx)
            {
                let mut root_style = Style::default();
                root_style.refine(style);
                let layout_id = window.request_layout(root_style, None, cx);
                return (layout_id, ViewLayout::Cached);
            }
            match window.reusable_view(global_id, entity, cx) {
                Ok(previous) => {
                    if let Some(layout_id) = window.reuse_view_layout(previous) {
                        return (layout_id, ViewLayout::Retained { previous });
                    }
                    window.note_rebuild(entity, ViewRebuildReason::ContextChanged);
                }
                // Dirty because a view nested in it was notified, or because
                // something changed that it or a view nested in it read: it
                // is spliced when only nested views have to be built.
                Err(
                    reason @ (ViewRebuildReason::Notified
                    | ViewRebuildReason::EntityChanged
                    | ViewRebuildReason::GlobalChanged
                    | ViewRebuildReason::StateChanged
                    | ViewRebuildReason::Deadline
                    | ViewRebuildReason::HoverChanged),
                ) => {
                    if let Some((layout_id, splice)) = window.splice_layout(global_id, entity, cx)
                    {
                        return (layout_id, ViewLayout::Spliced(splice));
                    }
                    window.note_rebuild(entity, reason);
                }
                Err(reason) => window.note_rebuild(entity, reason),
            }
            let recording = window.begin_view_layout(cx);
            #[cfg(feature = "profiler")]
            window.record_view_render(entity, view_name);
            let mut element = render(window, cx);
            let layout_id = element.request_layout(window, cx);
            let (layout, dependencies) = window.finish_view_layout(recording, layout_id, cx);
            (
                layout_id,
                ViewLayout::Built {
                    element,
                    layout,
                    dependencies,
                },
            )
        });
        cx.pop_untracked_reads(untracked);
        layout
    }

    /// Prepaints an entity view with view retention on, following up on how
    /// it was laid out.
    pub(crate) fn prepaint_retained_view(
        &mut self,
        entity: EntityId,
        view_name: ViewName,
        global_id: &GlobalElementId,
        bounds: Bounds<Pixels>,
        layout: ViewLayout,
        source: ViewSource,
        render: &mut dyn FnMut(&mut Window, &mut App) -> AnyElement,
        cx: &mut App,
    ) -> ViewPrepaint {
        self.set_view_id(entity);
        let untracked = cx.push_untracked_reads(entity);
        let prepaint = self.with_named_view(entity, view_name, |window| match layout {
            ViewLayout::Unretained(element) => {
                ViewPrepaint::Unretained(element.map(|mut element| {
                    element.prepaint(window, cx);
                    element
                }))
            }
            ViewLayout::Built {
                mut element,
                layout,
                dependencies,
            } => {
                let recording = window.begin_view_prepaint(global_id, source, cx);
                element.prepaint(window, cx);
                let record =
                    window.finish_view_prepaint(recording, bounds, layout, Some(dependencies), cx);
                ViewPrepaint::Built { element, record }
            }
            ViewLayout::Retained { previous } => {
                if window.view_context_matches(previous, bounds) {
                    #[cfg(feature = "profiler")]
                    window.draw_clock.count_reuse();
                    return ViewPrepaint::Reused(window.reuse_view_prepaint(previous, None, cx));
                }
                match window.view_move(previous, bounds) {
                    Ok(moved) => {
                        #[cfg(feature = "profiler")]
                        window.draw_clock.count_reuse();
                        return ViewPrepaint::Reused(window.reuse_view_prepaint(
                            previous,
                            Some(moved),
                            cx,
                        ));
                    }
                    Err(refusal) => culprits::blame_refused_move(entity, refusal),
                }
                window.note_rebuild(entity, ViewRebuildReason::ContextChanged);
                #[cfg(feature = "profiler")]
                window.record_view_render(entity, view_name);
                window.build_at_kept_layout(previous, bounds, global_id, source, render, cx)
            }
            ViewLayout::Spliced(splice) => {
                let previous = splice.previous;
                let context_matches = window.view_context_matches(previous, bounds);
                if context_matches && let Some(spliced) = window.splice_prepaint(splice, cx) {
                    return spliced;
                }
                if culprits::enabled() {
                    if context_matches {
                        culprits::blame("not drawn again around its nested views: one asked for \
                                         another layout"
                            .into());
                    } else if let Err(refusal) = window.view_move(previous, bounds) {
                        culprits::blame_refused_move(entity, refusal);
                    }
                }
                window.note_rebuild(entity, ViewRebuildReason::ContextChanged);
                #[cfg(feature = "profiler")]
                window.record_view_render(entity, view_name);
                window.build_at_kept_layout(previous, bounds, global_id, source, render, cx)
            }
            ViewLayout::Cached => {
                match window.reusable_view(global_id, entity, cx) {
                    Ok(previous) if window.view_context_matches(previous, bounds) => {
                        #[cfg(feature = "profiler")]
                        window.draw_clock.count_reuse();
                        return ViewPrepaint::Reused(window.reuse_view_prepaint(
                            previous, None, cx,
                        ));
                    }
                    Ok(previous) => {
                        match window.view_move(previous, bounds) {
                            Ok(moved) => {
                                #[cfg(feature = "profiler")]
                                window.draw_clock.count_reuse();
                                return ViewPrepaint::Reused(window.reuse_view_prepaint(
                                    previous,
                                    Some(moved),
                                    cx,
                                ));
                            }
                            Err(refusal) => culprits::blame_refused_move(entity, refusal),
                        }
                        window.note_rebuild(entity, ViewRebuildReason::ContextChanged)
                    }
                    Err(reason) => window.note_rebuild(entity, reason),
                }
                #[cfg(feature = "profiler")]
                window.record_view_render(entity, view_name);
                window.build_view_at(bounds, global_id, source, render, cx)
            }
        });
        cx.pop_untracked_reads(untracked);
        prepaint
    }

    /// Builds a view whose layout was taken from the last frame, at the
    /// nodes it kept, which the frame's layout has placed: it moved, or what
    /// it inherits changed, so it cannot be drawn again. Its content asks
    /// for its layout as it did when the view was last built, from where the
    /// view's own layout was requested, and so finds the nodes it kept.
    ///
    /// Built from what it read then, it asks for the same layout, and the
    /// layout computed for the frame stands. Should it ask for another (it
    /// read something it was not notified of), it is laid out on its own at
    /// its bounds for this frame, and notified for the next.
    fn build_at_kept_layout(
        &mut self,
        previous: usize,
        bounds: Bounds<Pixels>,
        global_id: &GlobalElementId,
        source: ViewSource,
        render: &mut dyn FnMut(&mut Window, &mut App) -> AnyElement,
        cx: &mut App,
    ) -> ViewPrepaint {
        self.try_build_at_kept_layout(previous, bounds, global_id, source, render, false, cx)
            .unwrap_or_else(|| unreachable!("only a strict build gives up"))
    }

    /// Builds a view at the nodes it kept, as [`Self::build_at_kept_layout`]
    /// does, unless `strict` and it asks for another layout, or its kept
    /// layout is out of date: then it gives up, without prepainting it, for
    /// what is drawn around it to be built instead (see [`splice`]).
    pub(super) fn try_build_at_kept_layout(
        &mut self,
        previous: usize,
        bounds: Bounds<Pixels>,
        global_id: &GlobalElementId,
        source: ViewSource,
        render: &mut dyn FnMut(&mut Window, &mut App) -> AnyElement,
        strict: bool,
        cx: &mut App,
    ) -> Option<ViewPrepaint> {
        let Some(kept) = self.rendered_frame.retained_views.records[previous]
            .layout
            .clone()
        else {
            return Some(self.build_view_at(bounds, global_id, source, render, cx));
        };
        {
            let records = &self.rendered_frame.retained_views.records;
            let nested = records[previous].nested;
            if let Some(engine) = self.layout_engine.as_mut() {
                for record in &records[previous..=previous + nested] {
                    if let Some(layout) = record.layout.as_deref() {
                        engine.release_kept(&layout.keys);
                    }
                }
            }
        }
        let writes_before = self
            .layout_engine
            .as_ref()
            .map_or(0, |engine| engine.layout_writes());
        let recording = self.begin_view_prepaint(global_id, source, cx);
        let layout_recording = self.begin_view_layout(cx);
        let view_key = kept.view_key.unwrap_or_else(|| self.layout_keys.prepaint_scope());
        self.layout_keys.push_key(view_key);
        let mut element = render(self, cx);
        let layout_id = element.request_layout(self, cx);
        let (layout, layout_dependencies) = self.finish_view_layout(layout_recording, layout_id, cx);
        self.layout_keys.pop();
        // A kept layout stands if the view asked for the nodes it had, as they
        // were, and they were laid out since they last changed: a tree laid
        // out on its own (a list item) that nothing laid out again this frame
        // still holds the layout it had then. The root of such a tree also
        // stands if, laid out again, it comes out the size it had.
        let unchanged = layout_id == kept.root
            && (self.layout_engine.as_ref().is_some_and(|engine| {
                engine.layout_writes() == writes_before && !engine.needs_layout(layout_id)
            }) || self.lay_out_again_in_place(layout_id, bounds, cx));
        if unchanged {
            element.prepaint(self, cx);
        } else if strict {
            // Given up on: what was drawn around it is built instead, which
            // lays it out again with the rest.
            self.finish_view_prepaint(recording, bounds, None, Some(layout_dependencies), cx);
            return None;
        } else {
            // Laid out on its own at the size it was given: what is around it
            // was laid out with the layout it had, and may have stretched or
            // grown it.
            if let Some(mut engine) = self.layout_engine.take() {
                engine.lay_out_at_size(layout_id, bounds.size, self, cx);
                self.layout_engine = Some(engine);
                self.frame_work.stats.compute_layout_calls += 1;
            }
            element.prepaint_at(bounds.origin, self, cx);
            self.request_animation_frame();
            // The views around it were laid out with the layout it had: they
            // are built on the next frame, at the one it asks for now.
            let views = &mut self.next_frame.retained_views;
            for &open in &views.open {
                views.records[open].layout_blocked = true;
            }
        }
        let record = self.finish_view_prepaint(
            recording,
            bounds,
            layout.filter(|_| unchanged),
            Some(layout_dependencies),
            cx,
        );
        Some(ViewPrepaint::Built { element, record })
    }

    /// Lays out again, in the space it was last laid out in, the tree rooted
    /// at `root` (a list item) whose nodes changed, and returns whether it
    /// came out where it was, filling `bounds` as it did. Whatever laid it
    /// out last placed it, and nothing else, by its size: what was laid out
    /// around it stands, and what changed inside is laid out anew.
    ///
    /// Only a root is laid out again: a node inside a tree is placed by the
    /// layout of the rest of the tree, which is not laid out again during
    /// prepaint because views built at their bounds this frame were laid out
    /// there as roots of their own.
    fn lay_out_again_in_place(
        &mut self,
        root: LayoutId,
        bounds: Bounds<Pixels>,
        cx: &mut App,
    ) -> bool {
        let Some(mut engine) = self.layout_engine.take() else {
            return false;
        };
        let unchanged = engine.root_space(root).is_some_and(|space| {
            let before = engine.laid_out_at(root);
            engine.lay_out_again(root, space, self, cx);
            self.frame_work.stats.compute_layout_calls += 1;
            let scale_factor = self.scale_factor();
            engine.laid_out_at(root) == before
                && engine.layout_bounds(root, scale_factor).size == bounds.size
        });
        self.layout_engine = Some(engine);
        unchanged
    }

    /// Builds a view whose layout was not requested from its content (a
    /// cached view, or one whose kept layout is gone) and lays it out on its
    /// own at `bounds`, as a cached view is without retention.
    fn build_view_at(
        &mut self,
        bounds: Bounds<Pixels>,
        global_id: &GlobalElementId,
        source: ViewSource,
        render: &mut dyn FnMut(&mut Window, &mut App) -> AnyElement,
        cx: &mut App,
    ) -> ViewPrepaint {
        let recording = self.begin_view_prepaint(global_id, source, cx);
        let mut element = render(self, cx);
        element.layout_as_root(Size::<AvailableSpace>::from(bounds.size), self, cx);
        element.prepaint_at(bounds.origin, self, cx);
        let record = self.finish_view_prepaint(recording, bounds, None, None, cx);
        ViewPrepaint::Built { element, record }
    }

    /// Paints an entity view with view retention on.
    pub(crate) fn paint_retained_view(
        &mut self,
        entity: EntityId,
        view_name: ViewName,
        global_id: &GlobalElementId,
        prepaint: &mut ViewPrepaint,
        cx: &mut App,
    ) {
        let untracked = cx.push_untracked_reads(entity);
        self.with_named_view(entity, view_name, |window| match prepaint {
            ViewPrepaint::Unretained(element) => {
                if let Some(element) = element {
                    element.paint(window, cx);
                }
            }
            ViewPrepaint::Reused(index) => window.reuse_view_paint(*index),
            ViewPrepaint::Spliced(spliced) => window.splice_paint(spliced, cx),
            ViewPrepaint::Built {
                element,
                record: None,
            } => element.paint(window, cx),
            ViewPrepaint::Built {
                element,
                record: Some(record),
            } => {
                let recording = window.begin_view_paint(*record, global_id, cx);
                element.paint(window, cx);
                window.finish_view_paint(recording, cx);
            }
        });
        cx.pop_untracked_reads(untracked);
    }
}

/// Notes that what is being drawn inside a retained view looks the way it
/// does because `hitbox` is, or is not, hovered. Returns the answer when it
/// is noted, and `None` when no retained view is being drawn.
#[inline]
pub(crate) fn note_hover_read(
    window: &Window,
    hitbox: HitboxId,
    ignoring_modality: bool,
) -> Option<bool> {
    if window.view_retention.view_stack.is_empty() {
        return None;
    }
    let hovered = hitbox.hovered_now(window, ignoring_modality);
    window
        .view_retention
        .hover_reads
        .borrow_mut()
        .push(HoverRead {
            hitbox,
            ignoring_modality,
            hovered,
        });
    Some(hovered)
}

/// Notes that what is being painted inside a retained view resolved the
/// group `name` to `hitbox`. Groups are only resolved as views paint; the
/// lookups made as they prepaint find no container and are not noted.
pub(crate) fn note_group_read(window: &mut Window, name: &SharedString, hitbox: Option<HitboxId>) {
    let retention = &mut window.view_retention;
    if retention.view_stack.is_empty() || !window.invalidator.is_painting() {
        return;
    }
    let reads = &mut retention.group_reads;
    if reads
        .last()
        .is_some_and(|last| last.hitbox == hitbox && last.name == *name)
    {
        return;
    }
    reads.push(GroupRead {
        name: name.clone(),
        hitbox,
    });
}
