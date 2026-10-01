//! Why retained views were built, for finding what keeps a view from being
//! drawn again.
//!
//! With `GPUI_REBUILD_CULPRITS=1`, every view built is counted under its
//! type and the reason it was built, and one built because an entity or a
//! global it read changed is counted under that entity's or global's type
//! too, with the stack that last changed it. The most frequent are logged
//! every 300 frames as `[rebuild-culprit]` lines. Capturing a stack on
//! every change is slow; this is for diagnosis, not for shipping. Off, each
//! hook costs one load of a cached flag.

use super::ViewRebuildReason;
use crate::EntityId;
use collections::FxHashMap;
use std::{any::TypeId, backtrace::Backtrace, cell::RefCell, sync::OnceLock};

/// How often, in frames, the counts are logged.
const LOG_EVERY: u64 = 300;
/// How many of the most frequent lines are logged.
const LOG_LINES: usize = 40;
/// How many frames of a change's stack name where it was made.
const STACK_FRAMES: usize = 16;

#[inline]
pub(crate) fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    #[cfg(test)]
    if FORCED.get() {
        return true;
    }
    *ENABLED.get_or_init(|| std::env::var("GPUI_REBUILD_CULPRITS").is_ok_and(|value| value == "1"))
}

#[cfg(test)]
thread_local! {
    static FORCED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Turns the counts on for this thread, for a test to read with
/// [`counts`].
#[cfg(test)]
pub(crate) fn force_on() {
    FORCED.set(true);
}

/// The lines counted so far on this thread, most frequent first.
#[cfg(test)]
pub(crate) fn counts() -> Vec<(String, u64)> {
    with(|culprits| {
        let mut lines: Vec<_> = culprits
            .counts
            .iter()
            .map(|(line, count)| (line.clone(), *count))
            .collect();
        lines.sort_by_key(|(_, count)| std::cmp::Reverse(*count));
        lines
    })
}

#[derive(Default)]
struct Culprits {
    entity_types: FxHashMap<EntityId, &'static str>,
    global_types: FxHashMap<TypeId, &'static str>,
    /// How each entity was last changed, and where.
    entity_changes: FxHashMap<EntityId, (&'static str, Backtrace)>,
    global_changes: FxHashMap<TypeId, (&'static str, Backtrace)>,
    /// What the last dependency check that found a change blamed, for the
    /// rebuild it causes to name.
    blamed: Option<String>,
    counts: FxHashMap<String, u64>,
    frames: u64,
}

thread_local! {
    static CULPRITS: RefCell<Culprits> = RefCell::default();
    /// Whether the views built now are left out, as a frame drawn again
    /// from scratch to verify the one drawn with views drawn again is.
    static SUSPENDED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Leaves the views built while `suspended` out of the counts.
pub(crate) fn suspend(suspended: bool) {
    SUSPENDED.set(suspended);
}

fn with<R>(f: impl FnOnce(&mut Culprits) -> R) -> R {
    CULPRITS.with_borrow_mut(f)
}

/// Notes the type of an entity as it is made.
#[inline]
pub(crate) fn note_entity_type(entity: EntityId, name: &'static str) {
    if enabled() {
        with(|culprits| culprits.entity_types.insert(entity, name));
    }
}

/// Notes the type of a global as it is written.
#[inline]
pub(crate) fn note_global_type(global: TypeId, name: &'static str) {
    if enabled() {
        with(|culprits| culprits.global_types.insert(global, name));
    }
}

/// Notes that `entity` changed, how, and where.
#[inline]
pub(crate) fn note_entity_change(entity: EntityId, how: &'static str) {
    if enabled() {
        let stack = Backtrace::force_capture();
        with(|culprits| culprits.entity_changes.insert(entity, (how, stack)));
    }
}

/// Notes that `global` changed, and where.
#[inline]
pub(crate) fn note_global_change(global: TypeId, drawing: bool) {
    if enabled() {
        let how = if drawing {
            "written while drawing"
        } else {
            "written"
        };
        let stack = Backtrace::force_capture();
        with(|culprits| culprits.global_changes.insert(global, (how, stack)));
    }
}

/// The frames of `stack` outside the standard library and this module,
/// innermost first.
fn site(stack: &Backtrace) -> String {
    let text = stack.to_string();
    let frames: Vec<&str> = text
        .lines()
        .map(str::trim)
        .filter(|line| !line.starts_with("at "))
        .filter_map(|line| line.split_once(": ").map(|(_, symbol)| symbol))
        .filter(|symbol| {
            !["std::", "core::", "<core::", "<std::", "alloc::", "<alloc::", "__rust"]
                .iter()
                .any(|prefix| symbol.starts_with(prefix))
                && !symbol.contains("culprits")
                && !symbol.contains("view_retention::dependencies")
        })
        .take(STACK_FRAMES)
        .collect();
    frames.join(" <- ")
}

/// Blames the rebuild about to be noted on `entity`, which changed `how`.
pub(crate) fn blame_entity(entity: EntityId) {
    with(|culprits| {
        let name = culprits.entity_types.get(&entity).copied().unwrap_or("?");
        let change = culprits
            .entity_changes
            .get(&entity)
            .map(|(how, stack)| format!("{how} at {}", site(stack)))
            .unwrap_or_default();
        culprits.blamed = Some(format!("entity {name} {change}"));
    });
}

/// Blames the rebuild about to be noted on `global`.
pub(crate) fn blame_global(global: TypeId) {
    with(|culprits| {
        let name = culprits
            .global_types
            .get(&global)
            .copied()
            .unwrap_or("? (window state, or whether a global is set)");
        let change = culprits
            .global_changes
            .get(&global)
            .map(|(how, stack)| format!("{how} at {}", site(stack)))
            .unwrap_or_default();
        culprits.blamed = Some(format!("global {name} {change}"));
    });
}

/// Counts a view built, under its type and why, and what was blamed.
pub(crate) fn rebuilt(view: EntityId, reason: ViewRebuildReason) {
    if !enabled() || SUSPENDED.get() {
        return;
    }
    with(|culprits| {
        let name = culprits.entity_types.get(&view).copied().unwrap_or("?");
        let blamed = culprits.blamed.take();
        let line = match (reason, blamed) {
            (ViewRebuildReason::EntityChanged | ViewRebuildReason::GlobalChanged, Some(blamed)) => {
                format!("{name} {reason:?} <- {blamed}")
            }
            _ => format!("{name} {reason:?}"),
        };
        *culprits.counts.entry(line).or_default() += 1;
    });
}

/// Counts a frame, logging the most frequent lines every [`LOG_EVERY`].
pub(crate) fn frame_finished() {
    if !enabled() {
        return;
    }
    with(|culprits| {
        culprits.frames += 1;
        if culprits.frames % LOG_EVERY != 0 {
            return;
        }
        let mut lines: Vec<_> = culprits.counts.iter().collect();
        lines.sort_by(|a, b| b.1.cmp(a.1));
        for (line, count) in lines.into_iter().take(LOG_LINES) {
            log::info!(
                "[rebuild-culprit] frames={} count={count} {line}",
                culprits.frames
            );
        }
    });
}

/// Forgets a released entity.
pub(crate) fn forget(entity: EntityId) {
    if enabled() {
        with(|culprits| {
            culprits.entity_types.remove(&entity);
            culprits.entity_changes.remove(&entity);
        });
    }
}
