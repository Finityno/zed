//! Demand-driven frames. Each clock thread (a `MonitorVBlank` thread per
//! monitor, and the `VSyncProvider` compositor clock for every other window)
//! sleeps until a window it ticks asks for a frame, so an idle app costs no
//! wakeups: neither on the clocks (240 per second per 240 Hz monitor when
//! they waited for every vblank) nor on the UI thread. The compositor clock
//! also wakes when the GPU device is removed, since it recovers the devices.
//!
//! Each window owns a [`FrameRequest`]: GPUI's frame waker sets its demand,
//! and the clock that ticks the window consumes it at its next vblank by
//! invalidating the window. GPUI re-arms the demand after every frame for as
//! long as it still wants frames (dirty views, next-frame callbacks,
//! animations, deferred presents, sustained presentation after high-rate
//! input), so nothing here re-arms on the app's behalf.
//!
//! A request wakes the clock that last served its window (see
//! [`FrameClocks`]); a clock that finds demand for a window another clock
//! ticks now (the window moved, a monitor clock stopped or started) hands
//! the demand on, so a request never waits on a clock that will not serve it.

use std::{
    collections::HashMap,
    sync::{
        Arc, LazyLock,
        atomic::{AtomicBool, AtomicIsize, AtomicU32, AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};

use anyhow::{Context as _, Result};
use gpui_util::ResultExt;
use windows::{
    Win32::{
        Foundation::{CloseHandle, HANDLE, WAIT_EVENT, WAIT_OBJECT_0, WAIT_TIMEOUT},
        Graphics::Direct3D11::{ID3D11Device, ID3D11Device4},
        System::Threading::{
            CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, CancelWaitableTimer, CreateEventW,
            CreateWaitableTimerExW, INFINITE, SetEvent, SetWaitableTimer, TIMER_ALL_ACCESS,
            WaitForMultipleObjects, WaitForSingleObject,
        },
    },
    core::Interface,
};

/// How often a parked compositor clock checks for device loss when Windows
/// cannot notify it (`ID3D11Device4` is unavailable).
const DEVICE_POLL_INTERVAL: Duration = Duration::from_secs(1);

/// Wakes one clock thread (the worker). Each has its own, so stopping one can
/// never be consumed by its replacement, and a request wakes only the clock
/// that ticks its window.
pub(crate) struct FrameWake {
    /// Auto-reset: one wait consumes every signal raised since the last one.
    event: HANDLE,
    /// Wakes the worker when a deferred frame falls due. High resolution
    /// where available: the default timer tick (15.6 ms) would turn a 33 ms
    /// frame cap into anything up to 49 ms.
    timer: HANDLE,
    /// Whether the worker is parked, and until when; see [`Parked`].
    parked: parking_lot::Mutex<Parked>,
}

/// What a request finds the worker doing. Held under a lock so that the
/// deadline it records and the timer it is set to never disagree: either
/// side may set the timer, the worker as it parks and a request for an
/// earlier frame while it is parked.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Parked {
    /// Awake: the worker looks at the demand of every window before it
    /// parks, so a request needs nothing more.
    Awake,
    /// Parked until a request or, if set, the timer at this encoded deadline
    /// (see [`to_deadline`]).
    Until(Option<u64>),
}

// SAFETY: the event and timer handles are kernel objects usable from any
// thread; this type only signals, sets, waits on and finally closes them.
unsafe impl Send for FrameWake {}
unsafe impl Sync for FrameWake {}

impl FrameWake {
    pub(crate) fn new() -> Result<Arc<Self>> {
        let event = unsafe { CreateEventW(None, false, false, None) }
            .context("creating the frame wake event")?;
        let timer = unsafe {
            CreateWaitableTimerExW(
                None,
                None,
                CREATE_WAITABLE_TIMER_HIGH_RESOLUTION,
                TIMER_ALL_ACCESS.0,
            )
            .or_else(|_| CreateWaitableTimerExW(None, None, 0, TIMER_ALL_ACCESS.0))
        };
        let timer = match timer {
            Ok(timer) => timer,
            Err(error) => {
                unsafe { CloseHandle(event) }.log_err();
                return Err(error).context("creating the frame deadline timer");
            }
        };
        Ok(Arc::new(Self {
            event,
            timer,
            parked: parking_lot::Mutex::new(Parked::Awake),
        }))
    }

    /// Says the worker is about to park until `deadline`, setting the timer
    /// for it. The worker then looks at the demand of the windows once more
    /// before it waits: a request made before this saw it awake and did not
    /// wake it.
    pub(crate) fn arm(&self, deadline: Option<Instant>) {
        let mut parked = self.parked.lock();
        let deadline = deadline.map(to_deadline);
        if deadline.is_none() {
            // A timer a request set while the worker was last parked would
            // otherwise still fire, for nothing.
            unsafe { CancelWaitableTimer(self.timer) }
                .context("stopping the frame deadline timer")
                .log_err();
        }
        *parked = Parked::Until(deadline.filter(|deadline| self.set_timer(*deadline)));
    }

    /// Drops signals raised while the worker was awake: a request that found
    /// it parked a moment before it woke, or a deadline timer that fired
    /// meanwhile (stopping a timer leaves it signalled). Called before
    /// [`Self::arm`]: what such a signal stood for is seen as the worker
    /// looks at the windows after arming, and keeping it would only wake the
    /// worker at once, for nothing.
    pub(crate) fn clear_signal(&self) {
        unsafe {
            WaitForSingleObject(self.event, 0);
            WaitForSingleObject(self.timer, 0);
        }
    }

    /// Says the worker is awake again.
    pub(crate) fn disarm(&self) {
        *self.parked.lock() = Parked::Awake;
    }

    /// Sets the timer for an encoded deadline, or stops it for none.
    fn set_timer(&self, deadline: u64) -> bool {
        let remaining = from_deadline(deadline).saturating_duration_since(Instant::now());
        // Relative due times are negative, in 100 ns units.
        let due = -(i64::try_from(remaining.as_nanos() / 100).unwrap_or(i64::MAX)).max(1);
        unsafe { SetWaitableTimer(self.timer, &due, 0, None, None, false) }
            .context("arming the frame deadline timer")
            .log_err()
            .is_some()
    }

    /// Wakes a parked worker for a frame due now, returning whether it did.
    fn wake_if_parked(&self) -> bool {
        if *self.parked.lock() == Parked::Awake {
            return false;
        }
        self.wake();
        true
    }

    /// Makes a parked worker wake by the encoded `deadline`, returning
    /// whether it had to be told. Its timer is moved earlier from here: the
    /// worker sleeps on until then rather than waking to set it itself.
    fn wake_parked_by(&self, deadline: u64) -> bool {
        let mut parked = self.parked.lock();
        match *parked {
            Parked::Awake => false,
            Parked::Until(Some(armed)) if armed <= deadline => false,
            Parked::Until(_) => {
                if self.set_timer(deadline) {
                    *parked = Parked::Until(Some(deadline));
                } else {
                    drop(parked);
                    self.wake();
                }
                true
            }
        }
    }

    pub(crate) fn wake(&self) {
        count(Count::WakeSignal);
        unsafe { SetEvent(self.event) }
            .context("signalling the frame wake event")
            .log_err();
    }

    /// Blocks until [`Self::wake`] is called, the device is removed (for the
    /// clock that owns device loss), or the timer set by [`Self::arm`] (or a
    /// request since) fires.
    ///
    /// Announces an idle point (see `gpui::set_thread_idle_hook`) only when it
    /// is actually about to sleep, never between two frames of an animation
    /// whose next request is already waiting.
    pub(crate) fn wait(&self, device_loss: DeviceLoss<'_>) -> WakeReason {
        let handles;
        let handles: &[HANDLE] = match device_loss {
            DeviceLoss::Watched(watch) => {
                handles = [self.event, self.timer, watch.event];
                &handles
            }
            DeviceLoss::Polled | DeviceLoss::NotOwned => {
                handles = [self.event, self.timer, self.timer];
                &handles[..2]
            }
        };
        let timeout = match device_loss {
            DeviceLoss::Polled => DEVICE_POLL_INTERVAL.as_millis() as u32,
            DeviceLoss::Watched(_) | DeviceLoss::NotOwned => INFINITE,
        };
        let mut result = unsafe { WaitForMultipleObjects(handles, false, 0) };
        if result == WAIT_TIMEOUT {
            gpui::thread_idle();
            result = unsafe { WaitForMultipleObjects(handles, false, timeout) };
        }
        count(Count::WorkerWake);
        let reason = WakeReason::from_wait(result);
        if reason == WakeReason::DeadlinePassed {
            count(Count::DeadlineWakeup);
        }
        reason
    }
}

impl Drop for FrameWake {
    fn drop(&mut self) {
        unsafe {
            CloseHandle(self.event)
                .context("closing the frame wake event")
                .log_err();
            CloseHandle(self.timer)
                .context("closing the frame deadline timer")
                .log_err();
        }
    }
}

/// How a parked clock learns that the GPU device was removed.
#[derive(Clone, Copy)]
pub(crate) enum DeviceLoss<'a> {
    /// Direct3D signals this watch's event.
    Watched(&'a DeviceRemovedWatch),
    /// Windows cannot notify; the clock wakes periodically to check.
    Polled,
    /// Another clock recovers the devices: this one sleeps until woken.
    NotOwned,
}

/// Every clock thread's wake, and which clock ticks which monitor: what a
/// window's request uses to wake the one clock that will serve it.
///
/// The compositor clock ticks every window whose monitor has no clock of its
/// own (no DXGI output, plugged in after the clocks were enumerated, or its
/// clock stopped), and recovers lost devices.
pub(crate) struct FrameClocks {
    compositor: Arc<FrameWake>,
    monitors: parking_lot::RwLock<HashMap<isize, Arc<FrameWake>>>,
    stopped: AtomicBool,
}

/// What [`FrameRequest`] records for the compositor clock. No monitor handle
/// is null.
pub(crate) const COMPOSITOR_CLOCK: isize = 0;

impl FrameClocks {
    pub(crate) fn new() -> Result<Arc<Self>> {
        Ok(Arc::new(Self {
            compositor: FrameWake::new()?,
            monitors: Default::default(),
            stopped: AtomicBool::new(false),
        }))
    }

    pub(crate) fn compositor(&self) -> &Arc<FrameWake> {
        &self.compositor
    }

    /// Registers `monitor`'s clock, which then ticks the windows on it.
    /// Refused once the clocks are stopped: [`Self::stop`] could not wake it.
    pub(crate) fn add_monitor(&self, monitor: isize, wake: Arc<FrameWake>) -> bool {
        let mut monitors = self.monitors.write();
        if self.is_stopped() {
            return false;
        }
        monitors.insert(monitor, wake);
        true
    }

    /// `monitor`'s clock stopped, or never started: the compositor clock
    /// ticks its windows from now on, and is woken for demand a request sent
    /// to the stopped clock.
    pub(crate) fn remove_monitor(&self, monitor: isize) {
        self.monitors.write().remove(&monitor);
        // After the removal: a request that still found the stopped clock
        // set its demand before then, so the compositor clock sees it.
        self.compositor.wake();
    }

    /// The clock that ticks a window whose fastest monitor is `monitor`.
    pub(crate) fn clock_for(&self, monitor: isize) -> isize {
        if self.monitors.read().contains_key(&monitor) {
            monitor
        } else {
            COMPOSITOR_CLOCK
        }
    }

    fn with_clock<R>(&self, clock: isize, f: impl FnOnce(&FrameWake) -> R) -> R {
        let monitors = self.monitors.read();
        f(monitors.get(&clock).unwrap_or(&self.compositor))
    }

    /// Stops every clock: each exits once it wakes.
    pub(crate) fn stop(&self) {
        self.stopped.store(true, Ordering::SeqCst);
        let monitors = self.monitors.read();
        self.compositor.wake();
        for wake in monitors.values() {
            wake.wake();
        }
    }

    pub(crate) fn is_stopped(&self) -> bool {
        self.stopped.load(Ordering::SeqCst)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum WakeReason {
    Requested,
    /// A deferred frame fell due.
    DeadlinePassed,
    DeviceRemoved,
    /// Nothing signalled: the device-loss poll is due.
    TimedOut,
}

impl WakeReason {
    fn from_wait(result: WAIT_EVENT) -> Self {
        match result.0.wrapping_sub(WAIT_OBJECT_0.0) {
            0 => Self::Requested,
            1 => Self::DeadlinePassed,
            2 => Self::DeviceRemoved,
            _ if result == WAIT_TIMEOUT => Self::TimedOut,
            _ => {
                // WAIT_FAILED: the handles are owned for the worker's lifetime, so
                // this is not expected. Degrade to polling at about 60 Hz instead
                // of spinning or never waking again.
                log::error!(
                    "waiting for frame demand failed: {}",
                    std::io::Error::last_os_error()
                );
                std::thread::sleep(Duration::from_millis(16));
                Self::Requested
            }
        }
    }
}

/// One window's frame demand, shared between the window (UI thread) and the
/// platform's window list, which the clocks read.
///
/// Requests are tied to this value rather than to the raw `HWND`: once the
/// window is destroyed its requests are ignored, even if Windows hands the
/// same handle value to a window created later.
pub(crate) struct FrameRequest {
    demand: AtomicBool,
    closed: AtomicBool,
    /// While set, the demand is not due before this time (see
    /// [`Self::request_at`]), in nanoseconds since [`EPOCH`] plus one, so
    /// that zero can mean no deadline.
    not_before: AtomicU64,
    /// The clock that last served or was handed this window's demand (a
    /// monitor handle, or [`COMPOSITOR_CLOCK`]): the one a request wakes.
    clock: AtomicIsize,
    clocks: Arc<FrameClocks>,
}

/// What [`FrameRequest::take`] found.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FrameDemand {
    /// Nothing asked for a frame.
    None,
    /// A frame is wanted now; the demand was consumed.
    Due,
    /// A frame is wanted at this time, not before; the demand is kept.
    NotBefore(Instant),
}

static EPOCH: LazyLock<Instant> = LazyLock::new(Instant::now);

fn to_deadline(at: Instant) -> u64 {
    u64::try_from(at.saturating_duration_since(*EPOCH).as_nanos())
        .unwrap_or(u64::MAX - 1)
        .saturating_add(1)
}

fn from_deadline(deadline: u64) -> Instant {
    *EPOCH + Duration::from_nanos(deadline - 1)
}

impl FrameRequest {
    pub(crate) fn new(clocks: Arc<FrameClocks>) -> Arc<Self> {
        Arc::new(Self {
            demand: AtomicBool::new(false),
            closed: AtomicBool::new(false),
            not_before: AtomicU64::new(0),
            clock: AtomicIsize::new(COMPOSITOR_CLOCK),
            clocks,
        })
    }

    /// Runs `f` with the wake of the clock this window's requests go to.
    /// Called after the demand is set: a clock that stops after this lookup
    /// wakes the compositor clock, which then sees the demand.
    fn with_clock<R>(&self, f: impl FnOnce(&FrameWake) -> R) -> R {
        self.clocks.with_clock(self.clock.load(Ordering::SeqCst), f)
    }

    /// Records that `clock` serves this window, so its requests wake it.
    pub(crate) fn served_by(&self, clock: isize) {
        if self.clock.load(Ordering::Relaxed) != clock {
            self.clock.store(clock, Ordering::SeqCst);
        }
    }

    /// Hands demand that a clock found, but does not tick, to `clock`, which
    /// does, waking it as a request would have.
    pub(crate) fn hand_to(&self, clock: isize, demand: FrameDemand) {
        self.served_by(clock);
        let woke = self.clocks.with_clock(clock, |wake| match demand {
            FrameDemand::None => false,
            FrameDemand::Due => wake.wake_if_parked(),
            FrameDemand::NotBefore(at) => wake.wake_parked_by(to_deadline(at)),
        });
        count(Count::HandOff);
        if woke {
            count(Count::HandOffWake);
        }
    }

    /// Asks for a frame at the next vsync.
    ///
    /// Only the transition to demanded wakes the worker: a demand already set
    /// has woken it, and it will consume that demand after its next vsync.
    /// A deadline set by [`Self::request_at`] is superseded, and the worker,
    /// which may be sleeping until it, is woken.
    pub(crate) fn request(&self) {
        if self.closed.load(Ordering::Acquire) {
            return;
        }
        let was_deferred = self.not_before.swap(0, Ordering::SeqCst) != 0;
        if (!self.demand.swap(true, Ordering::SeqCst) || was_deferred)
            && self.with_clock(FrameWake::wake_if_parked)
        {
            count(Count::RequestWake);
        }
    }

    /// Asks for a frame no earlier than `at`: demand GPUI knows it will not
    /// serve sooner (a frame its frame-rate cap refused, or a time
    /// animation's next step). The worker sleeps until then instead of
    /// painting the window every vsync only for GPUI to decline.
    pub(crate) fn request_at(&self, at: Instant) {
        if self.closed.load(Ordering::Acquire) {
            return;
        }
        let deadline = to_deadline(at);
        self.not_before.store(deadline, Ordering::SeqCst);
        self.demand.store(true, Ordering::SeqCst);
        // Only a worker parked with no deadline, or a later one, needs to
        // know, and only its timer is moved: an awake one sets its timer for
        // this before it parks.
        if self.with_clock(|wake| wake.wake_parked_by(deadline)) {
            count(Count::DeadlineRequestWake);
        }
    }

    /// Drops a deadline set by [`Self::request_at`], for window activation,
    /// which can lift the cap that deferred the frame, so the frame is asked
    /// for at the next vsync again, as it would be without one.
    pub(crate) fn expedite(&self) {
        if self.not_before.swap(0, Ordering::SeqCst) != 0
            && self.demand.load(Ordering::SeqCst)
            && self.with_clock(FrameWake::wake_if_parked)
        {
            count(Count::ExpediteWake);
        }
    }

    /// Sets demand and wakes the worker even if demand was already set.
    ///
    /// Used when the window joins the platform's window list: demand raised
    /// before that (while the window was being created) woke a worker that
    /// could not see the window yet, and later requests would not wake it
    /// again because the demand never cleared.
    pub(crate) fn request_and_wake(&self) {
        if self.closed.load(Ordering::Acquire) {
            return;
        }
        self.demand.store(true, Ordering::SeqCst);
        self.with_clock(FrameWake::wake);
    }

    /// Wakes the compositor clock without asking for a frame, for the work
    /// it does on its own behalf (recreating devices after a failed resize).
    pub(crate) fn wake_device_owner(&self) {
        self.clocks.compositor.wake();
    }

    /// What [`Self::take`] would find at `now`, consuming nothing.
    pub(crate) fn peek(&self, now: Instant) -> FrameDemand {
        if self.closed.load(Ordering::Acquire) || !self.demand.load(Ordering::SeqCst) {
            return FrameDemand::None;
        }
        match self.not_before.load(Ordering::SeqCst) {
            0 => FrameDemand::Due,
            not_before => {
                let due_at = from_deadline(not_before);
                if due_at > now {
                    FrameDemand::NotBefore(due_at)
                } else {
                    FrameDemand::Due
                }
            }
        }
    }

    /// Consumes demand that is due at `now`; the worker calls this once per
    /// vsync per window.
    pub(crate) fn take(&self, now: Instant) -> FrameDemand {
        if self.closed.load(Ordering::Acquire) || !self.demand.load(Ordering::Acquire) {
            return FrameDemand::None;
        }
        let not_before = self.not_before.load(Ordering::Acquire);
        if not_before != 0 {
            let due_at = from_deadline(not_before);
            if due_at > now {
                return FrameDemand::NotBefore(due_at);
            }
            // Served now. Only this deadline: a newer one is kept.
            self.not_before
                .compare_exchange(not_before, 0, Ordering::AcqRel, Ordering::Acquire)
                .ok();
        }
        if self.demand.swap(false, Ordering::AcqRel) {
            FrameDemand::Due
        } else {
            FrameDemand::None
        }
    }

    pub(crate) fn close(&self) {
        self.closed.store(true, Ordering::Release);
        self.demand.store(false, Ordering::Release);
        self.not_before.store(0, Ordering::Release);
    }

    pub(crate) fn is_closed(&self) -> bool {
        self.closed.load(Ordering::Acquire)
    }
}

/// Lets a parked worker sleep without polling for device loss: Direct3D
/// signals the event when the device is removed.
pub(crate) struct DeviceRemovedWatch {
    device: ID3D11Device4,
    event: HANDLE,
    cookie: u32,
}

impl DeviceRemovedWatch {
    /// `None` (logged) when Windows cannot notify; the worker then polls.
    pub(crate) fn new(device: &ID3D11Device) -> Option<Self> {
        Self::register(device)
            .context("registering for device-removed notifications")
            .log_err()
    }

    fn register(device: &ID3D11Device) -> Result<Self> {
        let device: ID3D11Device4 = device.cast()?;
        let event = unsafe { CreateEventW(None, false, false, None) }?;
        match unsafe { device.RegisterDeviceRemovedEvent(event) } {
            Ok(cookie) => Ok(Self {
                device,
                event,
                cookie,
            }),
            Err(error) => {
                unsafe { CloseHandle(event) }.log_err();
                Err(error.into())
            }
        }
    }
}

impl Drop for DeviceRemovedWatch {
    fn drop(&mut self) {
        unsafe {
            self.device.UnregisterDeviceRemoved(self.cookie);
            CloseHandle(self.event).log_err();
        }
    }
}

/// What the frame counters count; see [`FrameCounters`] for each.
#[derive(Clone, Copy)]
pub(crate) enum Count {
    WakeSignal,
    RequestWake,
    DeadlineRequestWake,
    ExpediteWake,
    WorkerWake,
    DeadlineWakeup,
    VsyncWait,
    MonitorVsyncWait,
    VsyncWithoutPaint,
    VsyncWaitSkipped,
    Invalidation,
    HandOff,
    HandOffWake,
    Paint,
    Present,
}

const COUNT_KINDS: usize = Count::Present as usize + 1;

#[allow(clippy::declare_interior_mutable_const, reason = "only used to initialize COUNTERS")]
const ZERO: AtomicU64 = AtomicU64::new(0);
static COUNTERS: [AtomicU64; COUNT_KINDS] = [ZERO; COUNT_KINDS];
static VSYNC_THREAD_ID: AtomicU32 = AtomicU32::new(0);

pub(crate) fn count(what: Count) {
    COUNTERS[what as usize].fetch_add(1, Ordering::Relaxed);
}

pub(crate) fn set_vsync_thread_id(thread_id: u32) {
    VSYNC_THREAD_ID.store(thread_id, Ordering::Relaxed);
}

/// Process-wide frame scheduling counters, for benchmarks and diagnostics.
/// Each is a relaxed atomic bumped once per event, so they cost nothing
/// measurable to keep.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FrameCounters {
    /// Times anything signalled the worker's wake event.
    pub wake_signals: u64,
    /// Of those, a window's demand going from clear to set (or superseding
    /// a deadline).
    pub request_wakes: u64,
    /// Of those, a frame asked for at a deadline.
    pub deadline_request_wakes: u64,
    /// Of those, activation pulling a deferred frame forward.
    pub expedite_wakes: u64,
    /// Times the worker returned from waiting for demand.
    pub worker_wakes: u64,
    /// Of those, a deferred frame's deadline passing.
    pub deadline_wakeups: u64,
    /// Times a clock waited for a vblank.
    pub vsync_waits: u64,
    /// Of those, a monitor's own clock (`MonitorVBlank`) rather than the
    /// compositor clock.
    pub monitor_vsync_waits: u64,
    /// Vsyncs after which the worker painted no window.
    pub vsyncs_without_paint: u64,
    /// Wakes after which the worker skipped the vsync: nothing was due.
    pub vsync_waits_skipped: u64,
    /// Windows invalidated by a clock (`RedrawWindow`).
    pub invalidations: u64,
    /// Demand a clock found for a window another clock ticks, and handed on.
    pub hand_offs: u64,
    /// Of those, the clock handed to was parked and had to be woken.
    pub hand_off_wakes: u64,
    /// `draw_window` calls: `WM_PAINT`, size-move timer ticks and forced draws.
    pub paints: u64,
    /// DXGI `Present` calls, counting each swap chain of a layered present.
    pub presents: u64,
}

#[doc(hidden)]
pub fn frame_counters() -> FrameCounters {
    let get = |what: Count| COUNTERS[what as usize].load(Ordering::Relaxed);
    FrameCounters {
        wake_signals: get(Count::WakeSignal),
        request_wakes: get(Count::RequestWake),
        deadline_request_wakes: get(Count::DeadlineRequestWake),
        expedite_wakes: get(Count::ExpediteWake),
        worker_wakes: get(Count::WorkerWake),
        deadline_wakeups: get(Count::DeadlineWakeup),
        vsync_waits: get(Count::VsyncWait),
        monitor_vsync_waits: get(Count::MonitorVsyncWait),
        vsyncs_without_paint: get(Count::VsyncWithoutPaint),
        vsync_waits_skipped: get(Count::VsyncWaitSkipped),
        invalidations: get(Count::Invalidation),
        hand_offs: get(Count::HandOff),
        hand_off_wakes: get(Count::HandOffWake),
        paints: get(Count::Paint),
        presents: get(Count::Present),
    }
}

/// The OS thread id of the most recently started `VSyncProvider` (compositor
/// clock) thread.
#[doc(hidden)]
pub fn vsync_thread_id() -> Option<u32> {
    match VSYNC_THREAD_ID.load(Ordering::Relaxed) {
        0 => None,
        id => Some(id),
    }
}

#[cfg(test)]
mod tests {
    use super::{DeviceLoss, FrameClocks, FrameDemand, FrameRequest, FrameWake, WakeReason};
    use std::{
        sync::{
            Arc,
            atomic::{AtomicBool, Ordering},
            mpsc,
        },
        thread,
        time::{Duration, Instant},
    };
    use windows::Win32::{
        Foundation::{HANDLE, WAIT_OBJECT_0},
        System::Threading::{SetEvent, WaitForSingleObject},
    };

    fn is_signalled(wake: &FrameWake) -> bool {
        // Consumes the signal, as the worker's wait would.
        unsafe { WaitForSingleObject(wake.event, 0) == WAIT_OBJECT_0 }
    }

    #[test]
    fn requests_coalesce_until_the_worker_takes_them() {
        let clocks = FrameClocks::new().unwrap();
        let wake = clocks.compositor().clone();
        wake.arm(None);
        let first = FrameRequest::new(clocks.clone());
        let second = FrameRequest::new(clocks);

        first.request();
        first.request();
        assert!(is_signalled(&wake));
        assert!(!is_signalled(&wake), "a repeated request must not signal again");

        assert_eq!(first.take(Instant::now()), FrameDemand::Due);
        assert_eq!(first.take(Instant::now()), FrameDemand::None);
        assert_eq!(
            second.take(Instant::now()),
            FrameDemand::None,
            "a request must not invalidate other windows"
        );

        first.request();
        assert!(is_signalled(&wake), "a consumed demand wakes again");
    }

    #[test]
    fn a_closed_window_ignores_pending_and_new_requests() {
        let clocks = FrameClocks::new().unwrap();
        let wake = clocks.compositor().clone();
        wake.arm(None);
        let request = FrameRequest::new(clocks);

        request.request();
        assert!(is_signalled(&wake));
        request.close();
        assert_eq!(
            request.take(Instant::now()),
            FrameDemand::None,
            "closing discards the pending demand"
        );

        request.request();
        request.request_and_wake();
        assert!(!is_signalled(&wake));
        assert_eq!(request.take(Instant::now()), FrameDemand::None);
    }

    #[test]
    fn a_reused_hwnd_does_not_inherit_a_closed_windows_demand() {
        let clocks = FrameClocks::new().unwrap();
        let closed_window = FrameRequest::new(clocks.clone());
        let new_window = FrameRequest::new(clocks);

        closed_window.request();
        closed_window.close();
        new_window.request();

        assert_eq!(closed_window.take(Instant::now()), FrameDemand::None);
        assert_eq!(new_window.take(Instant::now()), FrameDemand::Due);
    }

    #[test]
    fn joining_the_window_list_wakes_even_with_demand_already_set() {
        let clocks = FrameClocks::new().unwrap();
        let wake = clocks.compositor().clone();
        wake.arm(None);
        let request = FrameRequest::new(clocks);

        // Demand raised during creation, consumed by a worker that could not
        // see the window yet.
        request.request();
        assert!(is_signalled(&wake));

        request.request_and_wake();
        assert!(is_signalled(&wake));
        assert_eq!(request.take(Instant::now()), FrameDemand::Due);
    }

    #[test]
    fn the_worker_sleeps_until_woken() {
        let wake = FrameWake::new().unwrap();
        let (sender, receiver) = mpsc::channel();
        let worker = {
            let wake = wake.clone();
            thread::spawn(move || sender.send(wake.wait(DeviceLoss::NotOwned)).unwrap())
        };
        assert!(matches!(
            receiver.recv_timeout(Duration::from_millis(50)),
            Err(mpsc::RecvTimeoutError::Timeout)
        ));
        wake.wake();
        assert_eq!(
            receiver.recv_timeout(Duration::from_secs(5)).unwrap(),
            WakeReason::Requested
        );
        worker.join().unwrap();
    }

    #[test]
    fn stopping_one_worker_cannot_be_consumed_by_its_replacement() {
        let old_wake = FrameWake::new().unwrap();
        let new_wake = FrameWake::new().unwrap();
        let old_stop = Arc::new(AtomicBool::new(false));
        let (sender, receiver) = mpsc::channel();
        let workers = [(true, old_wake.clone()), (false, new_wake.clone())].map(
            |(old, wake)| {
                let sender = sender.clone();
                let stop = old_stop.clone();
                thread::spawn(move || {
                    wake.wait(DeviceLoss::NotOwned);
                    sender.send((old, old && stop.load(Ordering::Acquire))).unwrap();
                })
            },
        );

        old_stop.store(true, Ordering::Release);
        old_wake.wake();
        assert_eq!(
            receiver.recv_timeout(Duration::from_secs(5)).unwrap(),
            (true, true)
        );
        assert!(matches!(
            receiver.recv_timeout(Duration::from_millis(50)),
            Err(mpsc::RecvTimeoutError::Timeout)
        ));
        new_wake.wake();
        assert_eq!(
            receiver.recv_timeout(Duration::from_secs(5)).unwrap(),
            (false, false)
        );
        for worker in workers {
            worker.join().unwrap();
        }
    }

    #[test]
    fn a_device_removal_wakes_a_parked_worker() {
        let devices = crate::DirectXDevices::new().unwrap();
        let watch = super::DeviceRemovedWatch::new(&devices.device)
            .expect("Direct3D 11.4 device-removed notifications are available on Windows 10");
        let wake = FrameWake::new().unwrap();
        // Direct3D signals this event on removal; signal it by hand, since a
        // real removal would reset the GPU. The signaller waits first so the
        // worker is parked (not polling) when it arrives.
        let event = watch.event.0 as usize;
        let signaller = thread::spawn(move || {
            thread::sleep(Duration::from_millis(50));
            unsafe { SetEvent(HANDLE(event as _)) }.unwrap();
        });
        assert_eq!(wake.wait(DeviceLoss::Watched(&watch)), WakeReason::DeviceRemoved);
        signaller.join().unwrap();

        wake.wake();
        assert_eq!(wake.wait(DeviceLoss::Watched(&watch)), WakeReason::Requested);
    }

    #[test]
    fn deferred_demand_is_kept_until_due() {
        let clocks = FrameClocks::new().unwrap();
        let wake = clocks.compositor().clone();
        wake.arm(None);
        let request = FrameRequest::new(clocks);
        let now = Instant::now();
        let due_at = now + Duration::from_millis(30);

        request.request_at(due_at);
        assert!(!is_signalled(&wake), "a parked worker sleeps on");
        assert_eq!(
            *wake.parked.lock(),
            super::Parked::Until(Some(super::to_deadline(due_at))),
            "until the deadline, its timer set by the request"
        );
        assert_eq!(request.take(now), FrameDemand::NotBefore(due_at));
        assert_eq!(request.take(now), FrameDemand::NotBefore(due_at), "kept");
        assert_eq!(request.take(due_at), FrameDemand::Due);
        assert_eq!(request.take(due_at), FrameDemand::None);
    }

    #[test]
    fn a_request_supersedes_a_deadline_and_wakes_the_worker() {
        let clocks = FrameClocks::new().unwrap();
        let wake = clocks.compositor().clone();
        wake.arm(None);
        let request = FrameRequest::new(clocks);
        let now = Instant::now();

        request.request_at(now + Duration::from_secs(3600));
        assert!(!is_signalled(&wake));
        // Demand is already set, but the worker sleeps until the deadline,
        // so a notify must still wake it.
        request.request();
        assert!(is_signalled(&wake));
        assert_eq!(request.take(now), FrameDemand::Due);
    }

    #[test]
    fn expediting_moves_a_deferred_frame_to_the_next_vsync() {
        let clocks = FrameClocks::new().unwrap();
        let wake = clocks.compositor().clone();
        wake.arm(None);
        let request = FrameRequest::new(clocks);
        let now = Instant::now();

        request.expedite();
        assert!(!is_signalled(&wake), "nothing deferred, nothing to wake");

        request.request_at(now + Duration::from_secs(3600));
        assert!(!is_signalled(&wake));
        request.expedite();
        assert!(is_signalled(&wake));
        assert_eq!(request.take(now), FrameDemand::Due);
        request.expedite();
        assert!(!is_signalled(&wake));
    }

    #[test]
    fn closing_drops_a_deferred_frame() {
        let request = FrameRequest::new(FrameClocks::new().unwrap());
        let now = Instant::now();
        request.request_at(now);
        request.close();
        assert_eq!(request.take(now + Duration::from_secs(1)), FrameDemand::None);
        request.request_at(now);
        assert_eq!(request.take(now + Duration::from_secs(1)), FrameDemand::None);
    }

    #[test]
    fn a_parked_worker_wakes_for_a_deadline() {
        let clocks = FrameClocks::new().unwrap();
        let wake = clocks.compositor().clone();
        let start = Instant::now();
        let deadline = start + Duration::from_millis(30);
        wake.arm(Some(deadline));
        assert_eq!(wake.wait(DeviceLoss::NotOwned), WakeReason::DeadlinePassed);
        let woke_at = Instant::now();
        assert!(woke_at >= deadline, "woke {:?} early", deadline - woke_at);
        // Generous: a loaded machine, not the 15.6 ms default timer tick.
        assert!(woke_at - deadline < Duration::from_millis(200));

        wake.arm(Some(start));
        assert_eq!(
            wake.wait(DeviceLoss::NotOwned),
            WakeReason::DeadlinePassed,
            "a past deadline returns at once"
        );

        // A wake before the deadline wins.
        wake.arm(Some(Instant::now() + Duration::from_secs(3600)));
        wake.wake();
        assert_eq!(wake.wait(DeviceLoss::NotOwned), WakeReason::Requested);

        // A request for an earlier frame moves the timer of a parked worker
        // without signalling it.
        let request = FrameRequest::new(clocks);
        let soon = Instant::now() + Duration::from_millis(20);
        request.request_at(soon);
        assert!(!is_signalled(&wake));
        assert_eq!(wake.wait(DeviceLoss::NotOwned), WakeReason::DeadlinePassed);
        assert!(Instant::now() >= soon);
    }

    #[test]
    fn only_a_parked_worker_is_woken() {
        let clocks = FrameClocks::new().unwrap();
        let wake = clocks.compositor().clone();
        let request = FrameRequest::new(clocks);
        let now = Instant::now();

        // Awake, the worker looks at every window before it parks.
        request.request();
        assert!(!is_signalled(&wake), "an awake worker is not signalled");
        assert_eq!(request.take(now), FrameDemand::Due);

        // Parked until a deadline: a later frame does not wake it, an
        // earlier one and a frame due now do.
        wake.arm(Some(now + Duration::from_secs(30)));
        let parked_until = |wake: &FrameWake| *wake.parked.lock();
        request.request_at(now + Duration::from_secs(60));
        assert!(!is_signalled(&wake), "the timer is armed for earlier");
        assert_eq!(
            parked_until(&wake),
            super::Parked::Until(Some(super::to_deadline(now + Duration::from_secs(30))))
        );
        request.request_at(now + Duration::from_secs(10));
        assert!(!is_signalled(&wake), "an earlier deadline moves the timer");
        assert_eq!(
            parked_until(&wake),
            super::Parked::Until(Some(super::to_deadline(now + Duration::from_secs(10))))
        );
        request.request();
        assert!(is_signalled(&wake), "a frame due now wakes it");

        wake.disarm();
        request.request_at(now + Duration::from_millis(10));
        assert!(!is_signalled(&wake));
    }

    /// A clock for `monitor`, registered with `clocks` and parked.
    fn monitor_clock(clocks: &FrameClocks, monitor: isize) -> Arc<FrameWake> {
        let wake = FrameWake::new().unwrap();
        assert!(clocks.add_monitor(monitor, wake.clone()));
        wake.arm(None);
        wake
    }

    #[test]
    fn a_request_wakes_only_the_clock_that_serves_its_window() {
        let clocks = FrameClocks::new().unwrap();
        clocks.compositor().arm(None);
        let monitor = monitor_clock(&clocks, 7);
        let other_monitor = monitor_clock(&clocks, 8);
        let request = FrameRequest::new(clocks.clone());

        request.request();
        assert!(
            is_signalled(clocks.compositor()),
            "a new window's requests go to the compositor clock, which routes them"
        );
        assert_eq!(request.take(Instant::now()), FrameDemand::Due);

        request.served_by(7);
        request.request();
        assert!(is_signalled(&monitor));
        assert!(!is_signalled(clocks.compositor()));
        assert!(!is_signalled(&other_monitor));
    }

    #[test]
    fn demand_handed_to_another_clock_wakes_it_as_a_request_would() {
        let clocks = FrameClocks::new().unwrap();
        let monitor = monitor_clock(&clocks, 7);
        let request = FrameRequest::new(clocks.clone());
        let now = Instant::now();

        // The compositor clock is awake, so the request signals nothing; the
        // compositor finds the demand and hands it to the window's clock.
        request.request();
        assert!(!is_signalled(clocks.compositor()));
        request.hand_to(7, request.peek(now));
        assert!(is_signalled(&monitor), "the parked clock is woken");
        assert_eq!(request.take(now), FrameDemand::Due);

        request.request();
        assert!(is_signalled(&monitor), "and the window's requests go to it");
        assert_eq!(request.take(now), FrameDemand::Due);

        let due_at = now + Duration::from_secs(30);
        request.request_at(due_at);
        assert!(!is_signalled(&monitor));
        assert_eq!(
            *monitor.parked.lock(),
            super::Parked::Until(Some(super::to_deadline(due_at))),
            "a deferred frame moves the parked clock's timer"
        );
        monitor.arm(None);
        request.hand_to(7, request.peek(now));
        assert!(!is_signalled(&monitor), "handed on, it moves the timer too");
        assert_eq!(
            *monitor.parked.lock(),
            super::Parked::Until(Some(super::to_deadline(due_at)))
        );
    }

    #[test]
    fn a_stopped_monitor_clocks_windows_fall_to_the_compositor_clock() {
        let clocks = FrameClocks::new().unwrap();
        clocks.compositor().arm(None);
        let monitor = monitor_clock(&clocks, 7);
        let request = FrameRequest::new(clocks.clone());
        request.served_by(7);

        // A request that reached the clock as it stopped, before it could
        // serve the demand.
        request.request();
        assert!(is_signalled(&monitor));
        clocks.remove_monitor(7);
        assert!(
            is_signalled(clocks.compositor()),
            "the compositor clock is woken for demand the stopped clock left"
        );
        assert_eq!(request.peek(Instant::now()), FrameDemand::Due);
        assert_eq!(clocks.clock_for(7), super::COMPOSITOR_CLOCK);
        assert_eq!(request.take(Instant::now()), FrameDemand::Due);

        request.request();
        assert!(
            is_signalled(clocks.compositor()),
            "and its windows' requests wake the compositor clock from then on"
        );
        assert!(!is_signalled(&monitor));
    }

    #[test]
    fn stopping_wakes_every_clock_and_refuses_new_ones() {
        let clocks = FrameClocks::new().unwrap();
        let monitor = FrameWake::new().unwrap();
        assert!(clocks.add_monitor(7, monitor.clone()));
        assert_eq!(clocks.clock_for(7), 7);

        clocks.stop();
        assert!(is_signalled(clocks.compositor()));
        assert!(is_signalled(&monitor));
        assert!(
            !clocks.add_monitor(8, FrameWake::new().unwrap()),
            "a clock registered after the stop would never be woken to exit"
        );
        assert_eq!(clocks.clock_for(8), super::COMPOSITOR_CLOCK);
    }
}
