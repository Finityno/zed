//! Notifications for when a GPUI-owned thread is about to block, and when it
//! wakes again.
//!
//! GPUI has no interest in what an embedder does with these. The motivating
//! consumer is an allocator: mimalloc-style allocators can only return the free
//! space inside a still-used page from the thread that owns it, and only while
//! that thread is not allocating — so a thread that never announces its idle
//! points keeps its slack for the life of the process. A park is exactly that
//! announcement.
//!
//! Hooks are global and set once, before any executor starts. Both default to
//! doing nothing, so an embedder that does not care pays one relaxed atomic
//! load per park.

use std::sync::OnceLock;

/// Called with no arguments on the thread that is about to block, and again on
/// that same thread when it wakes.
pub type ThreadParkHook = fn();

/// The pair an embedder installs with [`set_thread_park_hooks`].
///
/// Whether these are ever read depends on which platform backend is linked:
/// the Linux and Windows dispatchers own their worker pools and park in
/// `queue.rs`, while macOS hands work to GCD and owns no worker threads at all.
/// So on a macOS build this and [`park`] below are genuinely dead, which is a
/// property of the backend rather than a mistake.
#[allow(dead_code)]
#[derive(Clone, Copy)]
struct ThreadParkHooks {
    on_park: ThreadParkHook,
    on_unpark: ThreadParkHook,
}

static HOOKS: OnceLock<ThreadParkHooks> = OnceLock::new();

/// Install the park/unpark pair.
///
/// Call once, before [`Application::new`](crate::Application::new) — worker
/// threads start with the first executor, and a hook installed after that
/// misses every park those threads have already entered.
///
/// Returns `false` if hooks were already installed, in which case the existing
/// pair is kept. There is one set of hooks per process, not per `Application`.
///
/// `on_park` runs immediately before the thread blocks and `on_unpark`
/// immediately after it wakes, on that same thread, always paired. Neither may
/// block, panic, or re-enter GPUI: a park hook runs while the executor holds
/// its queue lock, so anything slow here is contention every other worker pays
/// for.
pub fn set_thread_park_hooks(on_park: ThreadParkHook, on_unpark: ThreadParkHook) -> bool {
    HOOKS.set(ThreadParkHooks { on_park, on_unpark }).is_ok()
}

static ON_IDLE: OnceLock<ThreadParkHook> = OnceLock::new();

/// Install a hook called on the main thread each time it is about to sleep in
/// the platform run loop.
///
/// Unpaired, unlike [`set_thread_park_hooks`], and that difference is the whole
/// point. A paired hook asks the embedder to do something for the *duration* of
/// the block, which is only sound where GPUI controls everything that runs in
/// between. On the main thread it does not: AppKit, Core Animation and any
/// other run-loop observer get their turn after this fires and before the
/// thread actually sleeps, and they allocate. So this hook must do its work
/// synchronously and return, holding nothing across the sleep.
///
/// Fires often — every trip round an idle run loop — so it must be cheap and
/// self-rate-limiting.
///
/// Returns `false` if a hook was already installed, in which case it is kept.
pub fn set_main_thread_idle_hook(on_idle: ThreadParkHook) -> bool {
    ON_IDLE.set(on_idle).is_ok()
}

/// Run the main-thread idle hook, if one is installed.
///
/// Public because the platform backends are separate crates; not part of the
/// embedder-facing API.
#[doc(hidden)]
pub fn main_thread_idle() {
    if let Some(on_idle) = ON_IDLE.get() {
        on_idle();
    }
}

/// Guard returned by [`park`]. Runs the unpark hook when dropped, so a wait
/// that unwinds or returns early still pairs.
#[allow(dead_code)]
pub(crate) struct Parked {
    on_unpark: Option<ThreadParkHook>,
}

impl Drop for Parked {
    fn drop(&mut self) {
        if let Some(on_unpark) = self.on_unpark {
            on_unpark();
        }
    }
}

/// Announce that the calling thread is about to block. Drop the guard when it
/// wakes.
#[allow(dead_code)]
pub(crate) fn park() -> Parked {
    let Some(hooks) = HOOKS.get() else {
        return Parked { on_unpark: None };
    };
    (hooks.on_park)();
    Parked {
        on_unpark: Some(hooks.on_unpark),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static PARKS: AtomicUsize = AtomicUsize::new(0);
    static UNPARKS: AtomicUsize = AtomicUsize::new(0);

    // The whole contract is that the two always pair, including when the code
    // between them unwinds — an allocator hook that leaks a park leaves this
    // thread's heaps lent out forever, which is invisible until the process
    // grows. So exercise the panic path too, not just the happy one.
    #[test]
    fn park_and_unpark_are_paired_even_when_the_wait_unwinds() {
        assert!(
            set_thread_park_hooks(
                || {
                    PARKS.fetch_add(1, Ordering::Relaxed);
                },
                || {
                    UNPARKS.fetch_add(1, Ordering::Relaxed);
                }
            ),
            "hooks were already installed in this test process"
        );

        {
            let _parked = park();
        }
        assert_eq!(PARKS.load(Ordering::Relaxed), 1);
        assert_eq!(UNPARKS.load(Ordering::Relaxed), 1);

        let unwound = std::panic::catch_unwind(|| {
            let _parked = park();
            panic!("the wait failed");
        });
        assert!(unwound.is_err());
        assert_eq!(
            UNPARKS.load(Ordering::Relaxed),
            2,
            "the unpark hook did not run when the parked scope unwound"
        );

        // Installing twice keeps the first pair rather than racing.
        assert!(!set_thread_park_hooks(|| {}, || {}));
    }
}
