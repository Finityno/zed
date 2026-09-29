//! The keys that match an element to the layout node it had last frame.
//!
//! An element's key is a hash of its path from the root of the element tree:
//! each step is the element's [`ElementId`] when it has one, or its position
//! among the siblings that have none. Identified elements therefore keep
//! their node when siblings are inserted or reordered around them, and
//! unidentified ones are matched by position, the same bargain element state
//! makes. Because every step mixes in the parent's key, a node is never
//! matched to an element that moved to another parent.
//!
//! Only the request-layout walk has a path. Elements laid out during an
//! element's prepaint (list items, a cached view's contents, anything laid
//! out with `layout_as_root`) are keyed under the element being prepainted,
//! and list items additionally by their index, so a list that scrolls keeps
//! the nodes of the items still in view.
//!
//! A key only decides which node an element is offered. Whether the node's
//! style, children and measurement still stand is always checked against
//! what the element asks for, so a key that lands on the wrong node costs
//! work, never correctness.

use super::Window;
use crate::{AnyElement, App, AvailableSpace, ElementId, Pixels, Size};
use collections::FxBuildHasher;
use smallvec::SmallVec;
use std::{hash::BuildHasher, mem, sync::OnceLock};

/// Whether windows keep layout nodes from one frame to the next. On unless
/// `GPUI_RETAINED_LAYOUT` is `0` or `false`, which lays every frame out from
/// scratch, for comparing the two or for ruling retention out in the field.
pub(crate) fn layout_retention_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        !matches!(
            std::env::var("GPUI_RETAINED_LAYOUT").as_deref(),
            Ok("0" | "false")
        )
    })
}

/// A window's position in the element tree, as layout keys see it.
pub(crate) struct LayoutKeys {
    /// Keys of the elements whose layout is being requested, root first.
    stack: SmallVec<[LayoutKeyFrame; 32]>,
    /// How many unidentified trees were laid out under the current prepaint
    /// scope: the window root, prompts, drags, tooltips, deferred draws, list
    /// items and anything else laid out with `layout_as_root`.
    root_index: u32,
    /// The key of the element being prepainted, which what it lays out hangs
    /// off.
    prepaint_scope: u64,
    /// Whether keys are handed out at all. Without them every node is made
    /// afresh and released at the end of the frame.
    enabled: bool,
}

struct LayoutKeyFrame {
    key: u64,
    /// How many children without an [`ElementId`] were entered so far.
    next_unidentified_child: u32,
}

/// Where key paths start.
const ROOT_SEED: u64 = 0x9E37_79B9_7F4A_7C15;

/// Keeps what an element lays out during its prepaint apart from its own
/// children, which hang off the same key.
const PREPAINT_SALT: u64 = 0x5BF0_3635_931A_2E77;

/// Keeps the three kinds of step (an id, a position, a list index) apart,
/// so that a step of one kind never lands on another kind's key.
const IDENTIFIED_STEP: u64 = 1;
const POSITIONAL_STEP: u64 = 2;
const LIST_ITEM_STEP: u64 = 3;

/// Mixes `value` into `state`: the SplitMix64 finalizer over the two.
fn mix(state: u64, value: u64) -> u64 {
    let mut z = state
        .rotate_left(27)
        .wrapping_add(value)
        .wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

impl Default for LayoutKeys {
    fn default() -> Self {
        Self {
            stack: SmallVec::new(),
            root_index: 0,
            prepaint_scope: ROOT_SEED,
            enabled: layout_retention_enabled(),
        }
    }
}

/// What [`LayoutKeys::enter_prepaint_scope`] replaced.
pub(crate) struct PrepaintScope {
    prepaint_scope: u64,
    root_index: u32,
}

impl LayoutKeys {
    /// The key of the element whose layout is being requested, if any, and
    /// if keys are handed out.
    #[inline]
    pub(crate) fn current(&self) -> Option<u64> {
        if !self.enabled {
            return None;
        }
        self.stack.last().map(|frame| frame.key)
    }

    #[cfg(test)]
    pub(crate) fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Begins an element, identified by `id` if it has one, returning its key.
    #[inline]
    pub(crate) fn push(&mut self, id: Option<&ElementId>) -> u64 {
        let step = match id {
            Some(id) => mix(FxBuildHasher.hash_one(id), IDENTIFIED_STEP),
            None => {
                let index = match self.stack.last_mut() {
                    Some(parent) => &mut parent.next_unidentified_child,
                    None => &mut self.root_index,
                };
                let step = mix(u64::from(*index), POSITIONAL_STEP);
                *index += 1;
                step
            }
        };
        self.push_step(step)
    }

    fn push_step(&mut self, step: u64) -> u64 {
        let parent = self
            .stack
            .last()
            .map(|parent| parent.key)
            .unwrap_or_else(|| mix(self.prepaint_scope, PREPAINT_SALT));
        let key = mix(parent, step);
        self.stack.push(LayoutKeyFrame {
            key,
            next_unidentified_child: 0,
        });
        key
    }

    /// Ends the element most recently begun.
    #[inline]
    pub(crate) fn pop(&mut self) {
        self.stack.pop();
    }

    /// Hangs what is laid out from here on off the element whose key is
    /// `key`, which is being prepainted.
    #[inline]
    pub(crate) fn enter_prepaint_scope(&mut self, key: u64) -> PrepaintScope {
        PrepaintScope {
            prepaint_scope: mem::replace(&mut self.prepaint_scope, key),
            root_index: mem::replace(&mut self.root_index, 0),
        }
    }

    #[inline]
    pub(crate) fn exit_prepaint_scope(&mut self, scope: PrepaintScope) {
        self.prepaint_scope = scope.prepaint_scope;
        self.root_index = scope.root_index;
    }

    /// Starts the next frame's paths from the root again.
    pub(crate) fn end_frame(&mut self) {
        debug_assert!(self.stack.is_empty());
        self.stack.clear();
        self.root_index = 0;
        self.prepaint_scope = ROOT_SEED;
    }
}

impl Window {
    /// Lays `element` out as the item at `index` of a list, as
    /// [`AnyElement::layout_as_root`] does.
    ///
    /// A list lays out only the items in view, so an item without an
    /// [`ElementId`] would otherwise be matched to last frame's nodes by
    /// where it came among the items laid out this frame, and scrolling by
    /// one row would hand every item its neighbour's nodes. Keyed by its
    /// index, an item keeps its nodes while it stays in view. An item with an
    /// id keeps being matched by that, so one identified by its data keeps
    /// its nodes when items are inserted ahead of it. Only the layout is
    /// keyed; the element id stack and element state are untouched.
    pub(crate) fn layout_as_list_item(
        &mut self,
        element: &mut AnyElement,
        index: usize,
        available_space: Size<AvailableSpace>,
        cx: &mut App,
    ) -> Size<Pixels> {
        if element.element_id().is_some() {
            return element.layout_as_root(available_space, self, cx);
        }
        self.layout_keys
            .push_step(mix(index as u64, LIST_ITEM_STEP));
        let size = element.layout_as_root(available_space, self, cx);
        self.layout_keys.pop();
        size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Identified children keep their keys when siblings are inserted ahead
    /// of them; unidentified ones are keyed by position; and the same step
    /// under another parent is another key.
    #[test]
    fn keys_follow_ids_and_positions_under_their_parent() {
        let mut keys = LayoutKeys::default();
        keys.enabled = true;
        let key_of = |keys: &mut LayoutKeys, children: &[Option<&str>]| {
            keys.push(None);
            let child_keys: Vec<u64> = children
                .iter()
                .map(|child| {
                    let id = child.map(|name| ElementId::Name(name.into()));
                    let key = keys.push(id.as_ref());
                    keys.pop();
                    key
                })
                .collect();
            keys.pop();
            keys.end_frame();
            child_keys
        };

        let before = key_of(&mut keys, &[Some("a"), None, Some("b")]);
        let after = key_of(&mut keys, &[None, Some("a"), None, Some("b")]);
        assert_eq!(before[0], after[1], "an id keeps its key");
        assert_eq!(before[2], after[3], "an id keeps its key");
        assert_eq!(before[1], after[0], "the first unidentified child keeps its key");
        assert_ne!(after[0], after[2]);

        keys.push(None);
        keys.push(None);
        let nested = keys.push(Some(&ElementId::Name("a".into())));
        assert_ne!(nested, before[0], "the same id under another parent");
    }
}
