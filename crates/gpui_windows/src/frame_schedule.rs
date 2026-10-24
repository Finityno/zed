//! Demand-driven frames. The `VSyncProvider` worker sleeps until a window asks
//! for a frame or the GPU device is removed, so an idle app costs no wakeups:
//! neither on the worker (240 per second on a 240 Hz display when it waited
//! for every vsync) nor on the UI thread.
//!
//! Each window owns a [`FrameRequest`]: GPUI's frame waker sets its demand,
//! and the worker consumes it at the next vsync by invalidating the window.
//! GPUI re-arms the demand after every frame for as long as it still wants
//! frames (dirty views, next-frame callbacks, animations, deferred presents,
//! sustained presentation after high-rate input), so nothing here re-arms on
//! the app's behalf.

use std::{
    sync::{
        Arc, LazyLock,
        atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering},
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
            CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, CreateEventW, CreateWaitableTimerExW,
            INFINITE, SetEvent, SetWaitableTimer, TIMER_ALL_ACCESS, WaitForMultipleObjects,
        },
    },
    core::Interface,
};

/// How often a parked worker checks for device loss when Windows cannot
/// notify it (`ID3D11Device4` is unavailable).
const DEVICE_POLL_INTERVAL: Duration = Duration::from_secs(1);

/// Wakes one `VSyncProvider` worker. Each worker has its own, so stopping one
/// can never be consumed by its replacement.
pub(crate) struct FrameWake {
    /// Auto-reset: one wait consumes every signal raised since the last one.
    event: HANDLE,
    /// Wakes the worker when a deferred frame falls due. High resolution
    /// where available: the default timer tick (15.6 ms) would turn a 33 ms
    /// frame cap into anything up to 49 ms.
    timer: HANDLE,
}

// SAFETY: an event handle is a kernel object usable from any thread; this
// type only signals, waits on and finally closes it.
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
        Ok(Arc::new(Self { event, timer }))
    }

    pub(crate) fn wake(&self) {
        COUNTERS.wake_signals.fetch_add(1, Ordering::Relaxed);
        unsafe { SetEvent(self.event) }
            .context("signalling the frame wake event")
            .log_err();
    }

    /// Blocks until [`Self::wake`] is called, the device is removed, or
    /// `deadline` (a deferred frame falling due) passes.
    ///
    /// Announces an idle point (see `gpui::set_thread_idle_hook`) only when it
    /// is actually about to sleep, never between two frames of an animation
    /// whose next request is already waiting.
    pub(crate) fn wait(
        &self,
        device_removed: Option<&DeviceRemovedWatch>,
        deadline: Option<Instant>,
    ) -> WakeReason {
        let mut handles = [self.event, self.event, self.event];
        let mut count = 1;
        if let Some(watch) = device_removed {
            handles[count] = watch.event;
            count += 1;
        }
        let timer_index = count;
        if let Some(deadline) = deadline {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return WakeReason::DeadlinePassed;
            }
            // Relative due times are negative, in 100 ns units.
            let due = -(i64::try_from(remaining.as_nanos() / 100).unwrap_or(i64::MAX)).max(1);
            match unsafe { SetWaitableTimer(self.timer, &due, 0, None, None, false) } {
                Ok(()) => {
                    handles[count] = self.timer;
                    count += 1;
                }
                // Without the timer the deadline is polled at the coarse
                // wait timeout below instead.
                Err(error) => log::error!("arming the frame deadline timer: {error}"),
            }
        }
        let handles = &handles[..count];
        let poll = device_removed.is_none().then_some(DEVICE_POLL_INTERVAL);
        let fallback_deadline = (count == timer_index)
            .then(|| deadline.map(|deadline| deadline.saturating_duration_since(Instant::now())))
            .flatten();
        let timeout = match (poll, fallback_deadline) {
            (Some(poll), Some(deadline)) => poll.min(deadline).as_millis() as u32,
            (Some(timeout), None) | (None, Some(timeout)) => timeout.as_millis() as u32,
            (None, None) => INFINITE,
        };
        let mut result = unsafe { WaitForMultipleObjects(handles, false, 0) };
        if result == WAIT_TIMEOUT {
            gpui::thread_idle();
            result = unsafe { WaitForMultipleObjects(handles, false, timeout) };
        }
        COUNTERS.worker_wakes.fetch_add(1, Ordering::Relaxed);
        WakeReason::from_wait(result, device_removed.is_some(), count > timer_index)
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum WakeReason {
    Requested,
    DeviceRemoved,
    /// A deferred frame fell due.
    DeadlinePassed,
    /// Nothing signalled: the device-loss poll (or, without a timer, the
    /// deadline) is due.
    TimedOut,
}

impl WakeReason {
    fn from_wait(result: WAIT_EVENT, has_device_watch: bool, has_timer: bool) -> Self {
        let index = result.0.wrapping_sub(WAIT_OBJECT_0.0);
        if index == 0 {
            Self::Requested
        } else if index == 1 && has_device_watch {
            Self::DeviceRemoved
        } else if (index == 1 || index == 2) && has_timer {
            Self::DeadlinePassed
        } else if result == WAIT_TIMEOUT {
            Self::TimedOut
        } else {
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

/// One window's frame demand, shared between the window (UI thread) and the
/// platform's window list, which the worker reads.
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
    wake: Arc<FrameWake>,
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
    pub(crate) fn new(wake: Arc<FrameWake>) -> Arc<Self> {
        Arc::new(Self {
            demand: AtomicBool::new(false),
            closed: AtomicBool::new(false),
            not_before: AtomicU64::new(0),
            wake,
        })
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
        let was_deferred = self.not_before.swap(0, Ordering::AcqRel) != 0;
        if !self.demand.swap(true, Ordering::AcqRel) || was_deferred {
            self.wake.wake();
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
        self.not_before.store(to_deadline(at), Ordering::Release);
        self.demand.store(true, Ordering::Release);
        // The worker may be asleep with no deadline, or a later one.
        self.wake.wake();
    }

    /// Drops a deadline set by [`Self::request_at`], for events that can
    /// change why GPUI deferred the frame (input, activation), so the frame
    /// is asked for at the next vsync again, as it would be without one.
    pub(crate) fn expedite(&self) {
        if self.not_before.swap(0, Ordering::AcqRel) != 0
            && self.demand.load(Ordering::Acquire)
        {
            self.wake.wake();
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
        self.demand.store(true, Ordering::Release);
        self.wake.wake();
    }

    /// Wakes the worker without asking for a frame, for work it does on its
    /// own behalf (recreating devices after a failed resize).
    pub(crate) fn wake_worker(&self) {
        self.wake.wake();
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

struct Counters {
    wake_signals: AtomicU64,
    worker_wakes: AtomicU64,
    vsync_waits: AtomicU64,
    invalidations: AtomicU64,
    paints: AtomicU64,
    presents: AtomicU64,
    vsync_thread_id: AtomicU32,
}

static COUNTERS: Counters = Counters {
    wake_signals: AtomicU64::new(0),
    worker_wakes: AtomicU64::new(0),
    vsync_waits: AtomicU64::new(0),
    invalidations: AtomicU64::new(0),
    paints: AtomicU64::new(0),
    presents: AtomicU64::new(0),
    vsync_thread_id: AtomicU32::new(0),
};

pub(crate) fn count_vsync_wait() {
    COUNTERS.vsync_waits.fetch_add(1, Ordering::Relaxed);
}

pub(crate) fn count_invalidation() {
    COUNTERS.invalidations.fetch_add(1, Ordering::Relaxed);
}

pub(crate) fn count_paint() {
    COUNTERS.paints.fetch_add(1, Ordering::Relaxed);
}

pub(crate) fn count_present() {
    COUNTERS.presents.fetch_add(1, Ordering::Relaxed);
}

pub(crate) fn set_vsync_thread_id(thread_id: u32) {
    COUNTERS.vsync_thread_id.store(thread_id, Ordering::Relaxed);
}

/// Process-wide frame scheduling counters, for benchmarks and diagnostics.
/// Each is a relaxed atomic bumped once per event, so they cost nothing
/// measurable to keep.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FrameCounters {
    /// Times the UI thread (or a stop) signalled the worker's wake event.
    pub wake_signals: u64,
    /// Times the worker returned from waiting for demand.
    pub worker_wakes: u64,
    /// Times the worker waited for a vsync.
    pub vsync_waits: u64,
    /// Windows invalidated by the worker (`RedrawWindow`).
    pub invalidations: u64,
    /// `draw_window` calls: `WM_PAINT`, size-move timer ticks and forced draws.
    pub paints: u64,
    /// DXGI `Present` calls, counting each swap chain of a layered present.
    pub presents: u64,
}

#[doc(hidden)]
pub fn frame_counters() -> FrameCounters {
    FrameCounters {
        wake_signals: COUNTERS.wake_signals.load(Ordering::Relaxed),
        worker_wakes: COUNTERS.worker_wakes.load(Ordering::Relaxed),
        vsync_waits: COUNTERS.vsync_waits.load(Ordering::Relaxed),
        invalidations: COUNTERS.invalidations.load(Ordering::Relaxed),
        paints: COUNTERS.paints.load(Ordering::Relaxed),
        presents: COUNTERS.presents.load(Ordering::Relaxed),
    }
}

/// The OS thread id of the most recently started `VSyncProvider` worker.
#[doc(hidden)]
pub fn vsync_thread_id() -> Option<u32> {
    match COUNTERS.vsync_thread_id.load(Ordering::Relaxed) {
        0 => None,
        id => Some(id),
    }
}

#[cfg(test)]
mod tests {
    use super::{FrameDemand, FrameRequest, FrameWake, WakeReason};
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
        let wake = FrameWake::new().unwrap();
        let first = FrameRequest::new(wake.clone());
        let second = FrameRequest::new(wake.clone());

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
        let wake = FrameWake::new().unwrap();
        let request = FrameRequest::new(wake.clone());

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
        let wake = FrameWake::new().unwrap();
        let closed_window = FrameRequest::new(wake.clone());
        let new_window = FrameRequest::new(wake);

        closed_window.request();
        closed_window.close();
        new_window.request();

        assert_eq!(closed_window.take(Instant::now()), FrameDemand::None);
        assert_eq!(new_window.take(Instant::now()), FrameDemand::Due);
    }

    #[test]
    fn joining_the_window_list_wakes_even_with_demand_already_set() {
        let wake = FrameWake::new().unwrap();
        let request = FrameRequest::new(wake.clone());

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
            thread::spawn(move || sender.send(wake.wait(None, None)).unwrap())
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
                    wake.wait(None, None);
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
        assert_eq!(wake.wait(Some(&watch), None), WakeReason::DeviceRemoved);
        signaller.join().unwrap();

        wake.wake();
        assert_eq!(wake.wait(Some(&watch), None), WakeReason::Requested);
    }

    #[test]
    fn deferred_demand_is_kept_until_due() {
        let wake = FrameWake::new().unwrap();
        let request = FrameRequest::new(wake.clone());
        let now = Instant::now();
        let due_at = now + Duration::from_millis(30);

        request.request_at(due_at);
        assert!(is_signalled(&wake), "a parked worker learns the deadline");
        assert_eq!(request.take(now), FrameDemand::NotBefore(due_at));
        assert_eq!(request.take(now), FrameDemand::NotBefore(due_at), "kept");
        assert_eq!(request.take(due_at), FrameDemand::Due);
        assert_eq!(request.take(due_at), FrameDemand::None);
    }

    #[test]
    fn a_request_supersedes_a_deadline_and_wakes_the_worker() {
        let wake = FrameWake::new().unwrap();
        let request = FrameRequest::new(wake.clone());
        let now = Instant::now();

        request.request_at(now + Duration::from_secs(3600));
        assert!(is_signalled(&wake));
        // Demand is already set, but the worker sleeps until the deadline,
        // so a notify must still wake it.
        request.request();
        assert!(is_signalled(&wake));
        assert_eq!(request.take(now), FrameDemand::Due);
    }

    #[test]
    fn expediting_moves_a_deferred_frame_to_the_next_vsync() {
        let wake = FrameWake::new().unwrap();
        let request = FrameRequest::new(wake.clone());
        let now = Instant::now();

        request.expedite();
        assert!(!is_signalled(&wake), "nothing deferred, nothing to wake");

        request.request_at(now + Duration::from_secs(3600));
        assert!(is_signalled(&wake));
        request.expedite();
        assert!(is_signalled(&wake));
        assert_eq!(request.take(now), FrameDemand::Due);
        request.expedite();
        assert!(!is_signalled(&wake));
    }

    #[test]
    fn closing_drops_a_deferred_frame() {
        let request = FrameRequest::new(FrameWake::new().unwrap());
        let now = Instant::now();
        request.request_at(now);
        request.close();
        assert_eq!(request.take(now + Duration::from_secs(1)), FrameDemand::None);
        request.request_at(now);
        assert_eq!(request.take(now + Duration::from_secs(1)), FrameDemand::None);
    }

    #[test]
    fn a_parked_worker_wakes_for_a_deadline() {
        let wake = FrameWake::new().unwrap();
        let start = Instant::now();
        let deadline = start + Duration::from_millis(30);
        assert_eq!(wake.wait(None, Some(deadline)), WakeReason::DeadlinePassed);
        let woke_at = Instant::now();
        assert!(woke_at >= deadline, "woke {:?} early", deadline - woke_at);
        // Generous: a loaded machine, not the 15.6 ms default timer tick.
        assert!(woke_at - deadline < Duration::from_millis(200));

        assert_eq!(
            wake.wait(None, Some(start)),
            WakeReason::DeadlinePassed,
            "a past deadline returns at once"
        );

        // A wake before the deadline wins.
        wake.wake();
        assert_eq!(
            wake.wait(None, Some(Instant::now() + Duration::from_secs(3600))),
            WakeReason::Requested
        );
    }
}
