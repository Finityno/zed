//! A notification for when a GPUI-owned thread has run out of work.
//!
//! GPUI has no interest in what an embedder does with this. The motivating
//! consumer is an allocator: mimalloc-style allocators can only return the free
//! space inside a still-used page from the thread that owns it, and only while
//! that thread is not allocating — so a thread that never announces its idle
//! points keeps its slack for the life of the process.
//!
//! The hook is global, set once before any executor starts, and defaults to
//! doing nothing, so an embedder that does not care pays one relaxed atomic
//! load per idle point.
//!
//! Deliberately **unpaired**. An earlier version of this had a park/unpark pair
//! so an embedder could hold something for the duration of the block, and that
//! shape is a trap: it is only sound where GPUI controls every instruction
//! between the two calls, and at most of these sites it does not. The macOS run
//! loop runs AppKit and Core Animation observers after ours; a dispatched task
//! returns into GCD. An embedder that allocates on those threads would be doing
//! so while its "we are parked" state said otherwise. One synchronous call that
//! holds nothing is correct at every site, so that is the only shape offered.

use std::sync::OnceLock;

/// Called on a thread that has just run out of work, before it blocks.
pub type ThreadIdleHook = fn();

static HOOK: OnceLock<ThreadIdleHook> = OnceLock::new();

/// Install the idle hook.
///
/// Call once, before [`Application::new`](crate::Application::new) — worker
/// threads start with the first executor, and a hook installed after that
/// misses every idle point those threads have already passed.
///
/// Returns `false` if a hook was already installed, in which case it is kept.
/// There is one hook per process, not per `Application`.
///
/// It fires often — every trip round an idle run loop, and after every
/// dispatched background task on macOS — so it must be cheap and
/// self-rate-limiting. It must not block, panic, or re-enter GPUI. No site
/// holds an executor lock across it, so a slow hook delays only the thread it
/// runs on — but that thread may be the main thread about to sleep before the
/// next frame.
pub fn set_thread_idle_hook(hook: ThreadIdleHook) -> bool {
    HOOK.set(hook).is_ok()
}

/// Whether an idle hook has been installed, so a site can skip the setup
/// around a call — releasing a lock, say — when there is nothing to call.
///
/// Not part of the embedder-facing API: its one caller is the worker park in
/// `queue.rs`, which no macOS thread reaches (GCD owns the workers there), so
/// on that platform the function is dead code by design.
#[cfg_attr(target_os = "macos", allow(dead_code))]
pub(crate) fn is_installed() -> bool {
    HOOK.get().is_some()
}

/// Run the idle hook, if one is installed.
///
/// Public because the platform backends are separate crates; not part of the
/// embedder-facing API.
#[doc(hidden)]
pub fn thread_idle() {
    if let Some(hook) = HOOK.get() {
        hook();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static CALLS: AtomicUsize = AtomicUsize::new(0);

    /// The hook is a process-wide `OnceLock`, so this test installs it for the
    /// rest of the test binary and cannot assume it was first to do so: every
    /// assertion below is conditional on which hook actually won.
    #[test]
    fn the_hook_runs_and_installs_only_once() {
        let installed = set_thread_idle_hook(|| {
            CALLS.fetch_add(1, Ordering::Relaxed);
        });
        assert!(is_installed());
        if installed {
            let before = CALLS.load(Ordering::Relaxed);
            thread_idle();
            assert_eq!(CALLS.load(Ordering::Relaxed), before + 1);
        }

        // A second install keeps the first hook rather than racing.
        assert!(!set_thread_idle_hook(|| {
            CALLS.fetch_add(100, Ordering::Relaxed);
        }));
        if installed {
            let before = CALLS.load(Ordering::Relaxed);
            thread_idle();
            assert_eq!(CALLS.load(Ordering::Relaxed), before + 1);
        }
    }
}
