//! Asks a layer for its next drawable off the main thread.
//!
//! The layers are made with `allowsNextDrawableTimeout` off, so
//! `nextDrawable` blocks until Core Animation hands a drawable back: for as
//! long as the compositor holds all of the layer's drawables, however long
//! that is. It holds them for a window it is not compositing (just
//! deactivated, mid app-switch, on its way to occluded) and behind a GPU
//! backlog. Asked on the main thread, that wait froze the whole app: 72% of
//! the field's present hangs, and nearly all of those over a second, were
//! spent there. Asked here, the main thread waits only as long as it chooses
//! to and defers the present when no drawable came in time.

use metal::{MetalDrawable, MetalLayer};
use parking_lot::{Condvar, Mutex, MutexGuard};
use std::{sync::Arc, thread, time::Instant};

/// What [`DrawableAcquirer::take`] found.
pub(crate) enum Acquired {
    /// What `nextDrawable` returned. `None` only from a layer that gives up
    /// without one; with the timeout off that is a fault, not a busy pool.
    Ready(Option<MetalDrawable>),
    /// The layer is still waiting for a drawable to come back.
    Pending,
}

#[derive(Default)]
struct State {
    /// The generation of the request the thread is serving, or is about to.
    requested: Option<u64>,
    /// The answer to the last request, until the renderer takes it.
    ready: Option<Option<MetalDrawable>>,
    /// Bumped by [`DrawableAcquirer::discard`], so a drawable that comes back
    /// for a request made before it is released instead of kept.
    generation: u64,
    shutdown: bool,
}

struct Shared {
    state: Mutex<State>,
    changed: Condvar,
    #[cfg(test)]
    gate: Arc<AcquireGate>,
}

/// Holds the thread back before it asks for a drawable, standing in for a
/// compositor that holds every drawable: a layer in no window allocates a new
/// one for every request and never runs out.
#[cfg(test)]
#[derive(Default)]
pub(crate) struct AcquireGate {
    closed: Mutex<bool>,
    opened: Condvar,
}

#[cfg(test)]
impl AcquireGate {
    pub(crate) fn close(&self) {
        *self.closed.lock() = true;
    }

    pub(crate) fn open(&self) {
        *self.closed.lock() = false;
        self.opened.notify_all();
    }

    fn pass(&self) {
        let mut closed = self.closed.lock();
        while *closed {
            self.opened.wait(&mut closed);
        }
    }
}

pub(crate) struct DrawableAcquirer {
    shared: Arc<Shared>,
    layer: MetalLayer,
    thread_started: bool,
}

impl DrawableAcquirer {
    pub(crate) fn new(layer: MetalLayer) -> Self {
        Self {
            shared: Arc::new(Shared {
                state: Mutex::new(State::default()),
                changed: Condvar::new(),
                #[cfg(test)]
                gate: Arc::default(),
            }),
            layer,
            thread_started: false,
        }
    }

    /// Asks for the next drawable, unless a request is already out or its
    /// answer is waiting to be taken. Returns whether this made a new
    /// request: a present that finds one already out was deferred before,
    /// and waiting on it again would block the main thread for a share of
    /// every display refresh while the compositor holds the drawables.
    pub(crate) fn request(&mut self) -> bool {
        self.start_thread();
        let mut state = self.shared.state.lock();
        if state.requested.is_some() || state.ready.is_some() {
            return false;
        }
        state.requested = Some(state.generation);
        self.shared.changed.notify_all();
        true
    }

    /// Waits until the requested drawable has come back or `deadline` has
    /// passed, without taking it; `None` waits for as long as it takes.
    /// Returns whether it came back.
    pub(crate) fn wait(&self, deadline: Option<Instant>) -> bool {
        let mut state = self.shared.state.lock();
        while state.ready.is_none() && state.requested.is_some() {
            match deadline {
                Some(deadline) => {
                    if self.shared.changed.wait_until(&mut state, deadline).timed_out() {
                        break;
                    }
                }
                None => self.shared.changed.wait(&mut state),
            }
        }
        state.ready.is_some()
    }

    /// Whether no request is out and no drawable is waiting to be taken.
    pub(crate) fn is_idle(&self) -> bool {
        let state = self.shared.state.lock();
        state.requested.is_none() && state.ready.is_none()
    }

    /// Takes the requested drawable if it has come back.
    pub(crate) fn take(&self) -> Acquired {
        match self.shared.state.lock().ready.take() {
            Some(drawable) => Acquired::Ready(drawable),
            None => Acquired::Pending,
        }
    }

    /// Releases a drawable that came back and was not used, and one still
    /// on its way, so neither is held while the window is not painting: a
    /// drawable held unpresented is one the layer cannot hand out again.
    pub(crate) fn discard(&self) {
        let discarded = {
            let mut state = self.shared.state.lock();
            state.generation += 1;
            state.ready.take()
        };
        drop(discarded);
    }

    #[cfg(test)]
    pub(crate) fn gate(&self) -> Arc<AcquireGate> {
        Arc::clone(&self.shared.gate)
    }

    fn start_thread(&mut self) {
        if self.thread_started {
            return;
        }
        self.thread_started = true;
        let shared = Arc::clone(&self.shared);
        let layer = self.layer.clone();
        let spawned = thread::Builder::new()
            .name("gpui-drawable-acquire".into())
            .spawn(move || serve_requests(&shared, &layer));
        if let Err(error) = spawned {
            // Without the thread every request stays pending and every present
            // defers, so fall back to asking on the calling thread.
            log::error!("failed to start the drawable acquire thread: {error}");
            self.thread_started = false;
            let drawable =
                objc2::rc::autoreleasepool(|_| self.layer.next_drawable().map(ToOwned::to_owned));
            let mut state = self.shared.state.lock();
            state.requested = None;
            state.ready = Some(drawable);
        }
    }
}

impl Drop for DrawableAcquirer {
    fn drop(&mut self) {
        let discarded = {
            let mut state = self.shared.state.lock();
            state.shutdown = true;
            state.generation += 1;
            self.shared.changed.notify_all();
            state.ready.take()
        };
        drop(discarded);
        // Not joined: the thread may be inside `nextDrawable`, and it ends by
        // itself once that returns.
    }
}

fn serve_requests(shared: &Shared, layer: &MetalLayer) {
    let mut state = shared.state.lock();
    loop {
        if state.shutdown {
            return;
        }
        let Some(generation) = state.requested else {
            MutexGuard::unlocked(&mut state, gpui::thread_idle);
            if state.requested.is_none() && !state.shutdown {
                shared.changed.wait(&mut state);
            }
            continue;
        };
        let drawable = MutexGuard::unlocked(&mut state, || {
            #[cfg(test)]
            shared.gate.pass();
            // The drawable comes back autoreleased; it is retained before
            // this thread's pool drains, and released wherever it is dropped.
            objc2::rc::autoreleasepool(|_| layer.next_drawable().map(ToOwned::to_owned))
        });
        state.requested = None;
        shared.changed.notify_all();
        if state.generation == generation && !state.shutdown {
            state.ready = Some(drawable);
        } else {
            MutexGuard::unlocked(&mut state, || drop(drawable));
        }
    }
}
