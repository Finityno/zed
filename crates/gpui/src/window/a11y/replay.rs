//! Accessibility nodes drawn again from the last frame.
//!
//! While assistive technology is attached, every frame builds the whole
//! AccessKit tree: each element with an id and a role pushes a node as it
//! prepaints, the nodes of what it contains hang off it, a focusable element
//! says whether it holds the focus, and an element registers its action
//! listeners as it paints. A view drawn again from the last frame (see
//! [`crate::window::view_retention`]) is neither prepainted nor painted, so
//! none of that would happen for it.
//!
//! Instead, a frame logs what building its tree did, in order (a node
//! pushed, a leaf added, a node finished, an element made focusable, an
//! active descendant claimed), and [`crate::window::PrepaintStateIndex`]
//! keeps where in that log a stretch of prepaint starts and ends, as it does
//! for hitboxes. Drawing a stretch again does again what its part of last
//! frame's log did, through the same builder:
//!
//! - a node is pushed where the stretch lands, so it hangs off whatever node
//!   is open there;
//! - a finished node is last frame's node, moved with the view when it was
//!   drawn again elsewhere, with the children this frame gave it: a view
//!   spliced around a nested view that was built (see
//!   [`crate::window::view_retention::splice`]) hangs the built view's nodes
//!   off its own, and finishes its nodes after them;
//! - the focus is asked again, so a node is focused if its element's focus
//!   handle is focused now, and an active descendant is claimed under the
//!   node focused now.
//!
//! Last frame's finished nodes are read from the tree last sent, which the
//! window keeps for its debug dump. A stretch drawn again logs what it did as
//! any frame does, so it can be drawn again on the next frame.
//!
//! The action listeners an element registers as it paints are logged too,
//! and a stretch of paint drawn again moves them from last frame's into this
//! frame's, as mouse listeners are moved.
//!
//! Nothing can be drawn again from a frame that logged nothing: accessibility
//! was off while it was drawn, or a stretch drawn again in it logged other
//! ops than it had (a node id pushed twice is not pushed), which leaves the
//! indices of the stretches after it pointing past their ops. On the frame
//! after such a frame, every view is built
//! ([`crate::window::view_retention::ViewRebuildReason::Accessibility`]).

use crate::{
    Bounds, FocusId, Pixels, Point, Window,
    window::a11y::{A11y, A11yActionListener},
};
use accesskit::{Action, NodeId};
use collections::FxHashMap;
use std::{mem, ops::Range};

/// One thing building a frame's accessibility tree did.
#[derive(Clone, Copy, Debug)]
pub(crate) enum A11yOp {
    /// A node pushed: it hangs off the node open then, and is open until its
    /// [`A11yOp::Pop`].
    Push(NodeId),
    /// A leaf added under the node open, finished at this index of the
    /// frame's nodes.
    Leaf(usize),
    /// The node open finished, at this index of the frame's nodes.
    Pop(usize),
    /// A node made focusable with a focus handle, focused if the handle is.
    Focusable(NodeId, FocusId),
    /// A node claiming to be the active descendant of the focused node.
    ActiveDescendant(NodeId),
}

/// What building a window's accessibility tree logs, and the last frame's
/// log, to draw stretches of it again from.
#[derive(Default)]
pub(crate) struct A11yLog {
    /// Whether the frame being drawn logs: accessibility was active as it
    /// began.
    recording: bool,
    /// Whether the last frame was logged in full, so that it can be drawn
    /// again from.
    previous_recorded: bool,
    /// Whether a stretch drawn again in this frame logged other ops than it
    /// did last frame.
    diverged: bool,
    ops: Vec<A11yOp>,
    previous_ops: Vec<A11yOp>,
    /// The action listeners registered while painting, in order, by the
    /// node they listen on.
    actions: Vec<(NodeId, Action)>,
    /// Last frame's, each with where its listener is among its node's.
    previous_actions: Vec<(NodeId, Action, usize)>,
    previous_listeners: FxHashMap<NodeId, Vec<Option<(Action, A11yActionListener)>>>,
    previous_bounds: FxHashMap<NodeId, Bounds<Pixels>>,
    /// For each node pushed again and not yet finished, whether it was
    /// pushed. Kept between stretches: a view spliced around a nested view
    /// that is built finishes its nodes after the nested view's.
    replay_open: Vec<bool>,
}

impl A11yLog {
    #[inline]
    pub(crate) fn push(&mut self, id: NodeId) {
        if self.recording {
            self.ops.push(A11yOp::Push(id));
        }
    }

    #[inline]
    pub(crate) fn leaf(&mut self, at: usize) {
        if self.recording {
            self.ops.push(A11yOp::Leaf(at));
        }
    }

    #[inline]
    pub(crate) fn pop(&mut self, at: usize) {
        if self.recording {
            self.ops.push(A11yOp::Pop(at));
        }
    }

    #[inline]
    pub(crate) fn focusable(&mut self, node: NodeId, focus: FocusId) {
        if self.recording {
            self.ops.push(A11yOp::Focusable(node, focus));
        }
    }

    #[inline]
    pub(crate) fn active_descendant(&mut self, node: NodeId) {
        if self.recording {
            self.ops.push(A11yOp::ActiveDescendant(node));
        }
    }

    #[inline]
    pub(crate) fn action(&mut self, node: NodeId, action: Action) {
        if self.recording {
            self.actions.push((node, action));
        }
    }

    /// Where the frame being drawn stands in its log of tree building.
    #[inline]
    pub(crate) fn ops_len(&self) -> usize {
        self.ops.len()
    }

    /// Where the frame being drawn stands in its log of action listeners.
    #[inline]
    pub(crate) fn actions_len(&self) -> usize {
        self.actions.len()
    }

    /// Starts a frame, whether accessibility is active in it or not.
    pub(crate) fn new_frame(&mut self) {
        self.previous_recorded = self.recording && !mem::take(&mut self.diverged);
        self.recording = false;
    }

    /// Ends a logged frame, whose log becomes the last frame's.
    pub(crate) fn end_frame(&mut self) {
        mem::swap(&mut self.ops, &mut self.previous_ops);
        self.ops.clear();
        self.replay_open.clear();
    }
}

/// Where a transaction began in the accessibility tree being built, for it
/// to be rolled back. See [`Window::transact`].
pub(crate) struct A11ySavepoint {
    nodes: usize,
    depth: usize,
    children: usize,
    ops: usize,
    replay_open: usize,
    focus: Option<NodeId>,
    active_descendant: Option<NodeId>,
}

impl A11y {
    /// Starts logging a frame in which accessibility is active, setting aside
    /// what the last frame left that this one is about to clear: the bounds
    /// of its nodes and its action listeners.
    pub(crate) fn begin_replay_frame(&mut self) {
        let log = &mut self.nodes.log;
        log.recording = true;
        log.ops.clear();
        log.replay_open.clear();
        mem::swap(&mut log.previous_bounds, &mut self.node_bounds);
        log.previous_actions.clear();
        let mut counts: FxHashMap<NodeId, usize> = FxHashMap::default();
        for (node, action) in log.actions.drain(..) {
            let count = counts.entry(node).or_default();
            log.previous_actions.push((node, action, *count));
            *count += 1;
        }
        log.previous_listeners.clear();
        log.previous_listeners.extend(
            self.action_listeners
                .drain()
                .map(|(node, listeners)| (node, listeners.into_iter().map(Some).collect())),
        );
    }

    /// Whether accessibility is active but nothing can be drawn again from
    /// the last frame's tree: every view has to be built for its nodes to be.
    #[inline]
    pub(crate) fn replay_unavailable(&self) -> bool {
        self.is_active() && !self.nodes.log.previous_recorded
    }

    pub(crate) fn savepoint(&self) -> Option<A11ySavepoint> {
        if !self.is_active() {
            return None;
        }
        let nodes = &self.nodes;
        Some(A11ySavepoint {
            nodes: nodes.all_nodes.len(),
            depth: nodes.ids_stack.len(),
            children: nodes
                .nodes_stack
                .last()
                .map_or(0, |node| node.children().len()),
            ops: nodes.log.ops.len(),
            replay_open: nodes.log.replay_open.len(),
            focus: nodes.focus,
            active_descendant: nodes.active_descendant,
        })
    }

    /// Forgets the nodes pushed since `savepoint`, as the rolled-back
    /// prepaint that pushed them is forgotten: tried again, it pushes them
    /// again.
    pub(crate) fn roll_back(&mut self, savepoint: A11ySavepoint) {
        let nodes = &mut self.nodes;
        if nodes.ids_stack.len() < savepoint.depth || nodes.all_nodes.len() < savepoint.nodes {
            log::error!(
                "a11y: a rolled-back prepaint finished nodes it did not push; \
                 the tree may hold stale nodes this frame"
            );
            return;
        }
        let mut forgotten: Vec<NodeId> = nodes
            .all_nodes
            .drain(savepoint.nodes..)
            .map(|(id, _)| id)
            .collect();
        forgotten.extend(nodes.ids_stack.drain(savepoint.depth..));
        nodes.nodes_stack.truncate(savepoint.depth);
        for id in forgotten {
            nodes.seen_ids.remove(&id);
            self.node_bounds.remove(&id);
            self.focus_ids.remove(&id);
            #[cfg(debug_assertions)]
            nodes.node_info.remove(&id);
        }
        if let Some(open) = nodes.nodes_stack.last_mut()
            && open.children().len() > savepoint.children
        {
            let children = open.children()[..savepoint.children].to_vec();
            open.set_children(children);
        }
        nodes.focus = savepoint.focus;
        nodes.active_descendant = savepoint.active_descendant;
        nodes.log.ops.truncate(savepoint.ops);
        nodes.log.replay_open.truncate(savepoint.replay_open);
    }
}

/// What a stretch is drawn again from: last frame's log, the finished nodes
/// its indices point into, and the bounds of its nodes.
struct Source {
    ops: Vec<A11yOp>,
    update: Option<accesskit::TreeUpdate>,
    bounds: FxHashMap<NodeId, Bounds<Pixels>>,
}

impl Window {
    /// Does again what building the accessibility tree did over `ops` of the
    /// last frame's log, its nodes moved by `delta`, as a stretch of
    /// prepaint is drawn again.
    #[inline]
    pub(crate) fn replay_a11y_prepaint(&mut self, ops: Range<usize>, delta: Point<Pixels>) {
        if ops.is_empty() || !self.a11y.is_active() {
            return;
        }
        self.replay_a11y_ops(ops, delta);
    }

    #[inline(never)]
    fn replay_a11y_ops(&mut self, range: Range<usize>, delta: Point<Pixels>) {
        if !self.a11y.nodes.log.previous_recorded {
            debug_assert!(false, "drew again from a frame that logged no accessibility");
            return;
        }
        let source = Source {
            ops: mem::take(&mut self.a11y.nodes.log.previous_ops),
            update: self.a11y.debug.take_last_tree_update(),
            bounds: mem::take(&mut self.a11y.nodes.log.previous_bounds),
        };
        match source.ops.get(range) {
            Some(ops) => self.replay_a11y(ops, &source, delta),
            None => {
                debug_assert!(false, "a stretch reaches past last frame's accessibility log");
                self.a11y.nodes.log.diverged = true;
            }
        }
        self.a11y.debug.restore_last_tree_update(source.update);
        let log = &mut self.a11y.nodes.log;
        log.previous_ops = source.ops;
        log.previous_bounds = source.bounds;
    }

    fn replay_a11y(&mut self, ops: &[A11yOp], source: &Source, delta: Point<Pixels>) {
        let scale = self.scale_factor();
        let shift = (delta != Point::default())
            .then(|| (f64::from(delta.x.0 * scale), f64::from(delta.y.0 * scale)));
        let moved = |node: &accesskit::Node| {
            let mut node = node.clone();
            if let Some((dx, dy)) = shift
                && let Some(rect) = node.bounds()
            {
                node.set_bounds(accesskit::Rect {
                    x0: rect.x0 + dx,
                    y0: rect.y0 + dy,
                    x1: rect.x1 + dx,
                    y1: rect.y1 + dy,
                });
            }
            node
        };
        let finished: &[(NodeId, accesskit::Node)] = source
            .update
            .as_ref()
            .map_or(&[], |update| update.nodes.as_slice());
        let logged_before = self.a11y.nodes.log.ops.len();
        for op in ops {
            match *op {
                A11yOp::Push(id) => {
                    let pushed = self.a11y.nodes.push(id, accesskit::Node::default());
                    self.a11y.nodes.log.replay_open.push(pushed);
                    if pushed {
                        if let Some(bounds) = source.bounds.get(&id) {
                            self.a11y
                                .node_bounds
                                .insert(id, Bounds::new(bounds.origin + delta, bounds.size));
                        }
                        self.replay_a11y_node_info(id);
                    }
                }
                A11yOp::Leaf(at) => {
                    let Some((id, node)) = finished.get(at) else {
                        debug_assert!(false, "a leaf missing from last frame's tree");
                        self.a11y.nodes.log.diverged = true;
                        continue;
                    };
                    if self.a11y.nodes.push_leaf(*id, moved(node)) {
                        self.replay_a11y_node_info(*id);
                    }
                }
                A11yOp::Pop(at) => {
                    if self.a11y.nodes.log.replay_open.pop() != Some(true) {
                        continue;
                    }
                    let mut node = match finished.get(at) {
                        Some((_, node)) => moved(node),
                        None => {
                            debug_assert!(false, "a node missing from last frame's tree");
                            self.a11y.nodes.log.diverged = true;
                            accesskit::Node::default()
                        }
                    };
                    if let Some(open) = self.a11y.nodes.nodes_stack.last_mut() {
                        // Its children are this frame's: those pushed again,
                        // and those of nested views built in a splice.
                        if open.children() != node.children() {
                            if open.children().is_empty() {
                                node.clear_children();
                            } else {
                                node.set_children(open.children().to_vec());
                            }
                        }
                        *open = node;
                    }
                    self.a11y.nodes.pop();
                }
                A11yOp::Focusable(node, focus) => {
                    self.a11y.set_focusable(node, focus);
                    if focus.is_focused(self) {
                        self.a11y.set_focus(node);
                    }
                }
                A11yOp::ActiveDescendant(node) => self.a11y.set_active_descendant(node),
            }
        }
        let log = &mut self.a11y.nodes.log;
        log.diverged |= log.ops.len() - logged_before != ops.len();
    }

    /// Whether `ops` of the last frame's log add a synthetic node with bounds
    /// of its own, while accessibility is active. Drawn again moved, the node
    /// would move with its view, where the code that placed it may not have
    /// put it.
    pub(crate) fn a11y_placed_leaf_in(&self, ops: Range<usize>) -> bool {
        if ops.is_empty() || !self.a11y.is_active() {
            return false;
        }
        let Some(update) = self.a11y.debug.last_tree_update() else {
            return false;
        };
        self.a11y
            .nodes
            .log
            .previous_ops
            .get(ops)
            .is_none_or(|ops| {
                ops.iter().any(|op| {
                    matches!(op, A11yOp::Leaf(at)
                        if update.nodes.get(*at).is_some_and(|(_, node)| node.bounds().is_some()))
                })
            })
    }

    #[cfg(debug_assertions)]
    fn replay_a11y_node_info(&mut self, id: NodeId) {
        if let Some(info) = self.a11y.debug.last_node_info(id) {
            self.a11y.nodes.record_node_info(id, info);
        }
    }

    #[cfg(not(debug_assertions))]
    #[inline]
    fn replay_a11y_node_info(&mut self, _id: NodeId) {}

    /// Moves the action listeners that last frame's paint registered over
    /// `actions` of its log into the frame being drawn, as a stretch of paint
    /// is drawn again.
    #[inline]
    pub(crate) fn replay_a11y_paint(&mut self, actions: Range<usize>) {
        if actions.is_empty() || !self.a11y.is_active() {
            return;
        }
        self.replay_a11y_actions(actions);
    }

    #[inline(never)]
    fn replay_a11y_actions(&mut self, range: Range<usize>) {
        let a11y = &mut self.a11y;
        let log = &mut a11y.nodes.log;
        let Some(actions) = log.previous_actions.get(range) else {
            debug_assert!(false, "a stretch reaches past last frame's action listeners");
            return;
        };
        for &(node, action, index) in actions {
            let listener = log
                .previous_listeners
                .get_mut(&node)
                .and_then(|listeners| listeners.get_mut(index))
                .and_then(Option::take);
            log.actions.push((node, action));
            match listener {
                Some(listener) => a11y
                    .action_listeners
                    .entry(node)
                    .or_default()
                    .push(listener),
                None => debug_assert!(false, "an action listener drawn again twice"),
            }
        }
    }

    /// Makes the window build its accessibility tree from the next frame on
    /// as though assistive technology had asked for it. For harnesses that
    /// measure frames drawn with accessibility active; nothing tells the
    /// platform about it, so the tree is sent to an adapter nobody reads.
    #[doc(hidden)]
    pub fn force_a11y_active(&mut self) {
        self.a11y.force_active();
        self.refresh();
    }
}

/// A window's last accessibility tree as lines two windows drawing the same
/// views can be compared by: nodes named by where they are in the tree
/// rather than by their ids (two windows drawing the same views have
/// different view ids), depth first, each with its bounds rounded to 1/64 of
/// a device pixel (the rounding a view drawn again moved and a fresh layout
/// can differ by), whether it is focusable, where a click on it lands and
/// which actions it listens for; and which node is focused.
pub(crate) fn describe_a11y_tree(window: &Window) -> Vec<String> {
    use std::fmt::Write as _;
    let Some(tree) = window.a11y.debug.last_tree_update() else {
        return Vec::new();
    };
    fn round(value: f64) -> f64 {
        (value * 64.).round() / 64. + 0.
    }
    let nodes: FxHashMap<NodeId, &accesskit::Node> =
        tree.nodes.iter().map(|(id, node)| (*id, node)).collect();
    let mut lines = Vec::with_capacity(tree.nodes.len() + 2);
    if nodes.len() != tree.nodes.len() {
        lines.push(format!(
            "a11y: {} nodes, {} ids",
            tree.nodes.len(),
            nodes.len()
        ));
    }
    let mut focus = String::from("unreachable");
    let mut reached = 0;
    let mut stack = vec![(super::ROOT_NODE_ID, String::from("r"))];
    while let Some((id, path)) = stack.pop() {
        let Some(node) = nodes.get(&id) else {
            lines.push(format!("a11y {path}: missing"));
            continue;
        };
        reached += 1;
        if id == tree.focus {
            focus.clone_from(&path);
        }
        for (index, child) in node.children().iter().enumerate().rev() {
            stack.push((*child, format!("{path}.{index}")));
        }
        let mut bare = (*node).clone();
        bare.clear_children();
        bare.clear_bounds();
        let mut line = format!("a11y {path} {bare:?}");
        if let Some(rect) = node.bounds() {
            write!(
                line,
                " rect ({}, {})-({}, {})",
                round(rect.x0),
                round(rect.y0),
                round(rect.x1),
                round(rect.y1)
            )
            .ok();
        }
        if let Some(bounds) = window.a11y.node_bounds.get(&id) {
            let scaled = bounds.scale(window.scale_factor());
            write!(
                line,
                " click ({}, {}) {}x{}",
                round(f64::from(scaled.origin.x.0)),
                round(f64::from(scaled.origin.y.0)),
                round(f64::from(scaled.size.width.0)),
                round(f64::from(scaled.size.height.0))
            )
            .ok();
        }
        if window.a11y.focus_ids.contains_key(&id) {
            line.push_str(" focusable");
        }
        if let Some(listeners) = window.a11y.action_listeners.get(&id) {
            let mut actions: Vec<String> = listeners
                .iter()
                .map(|(action, _)| format!("{action:?}"))
                .collect();
            actions.sort_unstable();
            write!(line, " actions {actions:?}").ok();
        }
        lines.push(line);
    }
    if reached != tree.nodes.len() {
        lines.push(format!(
            "a11y: {} of {} nodes reachable from the root",
            reached,
            tree.nodes.len()
        ));
    }
    lines.push(format!("a11y focus {focus}"));
    lines
}
