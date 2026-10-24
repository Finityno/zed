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
        Arc,
        atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering},
    },
    time::Duration,
};

use anyhow::{Context as _, Result};
use gpui_util::ResultExt;
use windows::{
    Win32::{
        Foundation::{CloseHandle, HANDLE, WAIT_EVENT, WAIT_OBJECT_0, WAIT_TIMEOUT},
        Graphics::Direct3D11::{ID3D11Device, ID3D11Device4},
        System::Threading::{CreateEventW, INFINITE, SetEvent, WaitForMultipleObjects},
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
}

// SAFETY: an event handle is a kernel object usable from any thread; this
// type only signals, waits on and finally closes it.
unsafe impl Send for FrameWake {}
unsafe impl Sync for FrameWake {}

impl FrameWake {
    pub(crate) fn new() -> Result<Arc<Self>> {
        let event = unsafe { CreateEventW(None, false, false, None) }
            .context("creating the frame wake event")?;
        Ok(Arc::new(Self { event }))
    }

    pub(crate) fn wake(&self) {
        COUNTERS.wake_signals.fetch_add(1, Ordering::Relaxed);
        unsafe { SetEvent(self.event) }
            .context("signalling the frame wake event")
            .log_err();
    }

    /// Blocks until [`Self::wake`] is called or the device is removed.
    ///
    /// Announces an idle point (see `gpui::set_thread_idle_hook`) only when it
    /// is actually about to sleep, never between two frames of an animation
    /// whose next request is already waiting.
    pub(crate) fn wait(&self, device_removed: Option<&DeviceRemovedWatch>) -> WakeReason {
        let handles;
        let handles: &[HANDLE] = match device_removed {
            Some(watch) => {
                handles = [self.event, watch.event];
                &handles
            }
            None => std::slice::from_ref(&self.event),
        };
        let timeout = match device_removed {
            Some(_) => INFINITE,
            None => DEVICE_POLL_INTERVAL.as_millis() as u32,
        };
        let mut result = unsafe { WaitForMultipleObjects(handles, false, 0) };
        if result == WAIT_TIMEOUT {
            gpui::thread_idle();
            result = unsafe { WaitForMultipleObjects(handles, false, timeout) };
        }
        COUNTERS.worker_wakes.fetch_add(1, Ordering::Relaxed);
        WakeReason::from_wait(result)
    }
}

impl Drop for FrameWake {
    fn drop(&mut self) {
        unsafe { CloseHandle(self.event) }
            .context("closing the frame wake event")
            .log_err();
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum WakeReason {
    Requested,
    DeviceRemoved,
    /// Nothing signalled: the device-loss poll is due.
    TimedOut,
}

impl WakeReason {
    fn from_wait(result: WAIT_EVENT) -> Self {
        if result == WAIT_OBJECT_0 {
            Self::Requested
        } else if result.0 == WAIT_OBJECT_0.0 + 1 {
            Self::DeviceRemoved
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
    wake: Arc<FrameWake>,
}

impl FrameRequest {
    pub(crate) fn new(wake: Arc<FrameWake>) -> Arc<Self> {
        Arc::new(Self {
            demand: AtomicBool::new(false),
            closed: AtomicBool::new(false),
            wake,
        })
    }

    /// Asks for a frame at the next vsync.
    ///
    /// Only the transition to demanded wakes the worker: a demand already set
    /// has woken it, and it will consume that demand after its next vsync.
    pub(crate) fn request(&self) {
        if self.closed.load(Ordering::Acquire) {
            return;
        }
        if !self.demand.swap(true, Ordering::AcqRel) {
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

    /// Consumes the demand; the worker calls this once per vsync per window.
    pub(crate) fn take(&self) -> bool {
        !self.closed.load(Ordering::Acquire) && self.demand.swap(false, Ordering::AcqRel)
    }

    pub(crate) fn close(&self) {
        self.closed.store(true, Ordering::Release);
        self.demand.store(false, Ordering::Release);
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
    use super::{FrameRequest, FrameWake, WakeReason};
    use std::{
        sync::{
            Arc,
            atomic::{AtomicBool, Ordering},
            mpsc,
        },
        thread,
        time::Duration,
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

        assert!(first.take());
        assert!(!first.take());
        assert!(!second.take(), "a request must not invalidate other windows");

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
        assert!(!request.take(), "closing discards the pending demand");

        request.request();
        request.request_and_wake();
        assert!(!is_signalled(&wake));
        assert!(!request.take());
    }

    #[test]
    fn a_reused_hwnd_does_not_inherit_a_closed_windows_demand() {
        let wake = FrameWake::new().unwrap();
        let closed_window = FrameRequest::new(wake.clone());
        let new_window = FrameRequest::new(wake);

        closed_window.request();
        closed_window.close();
        new_window.request();

        assert!(!closed_window.take());
        assert!(new_window.take());
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
        assert!(request.take());
    }

    #[test]
    fn the_worker_sleeps_until_woken() {
        let wake = FrameWake::new().unwrap();
        let (sender, receiver) = mpsc::channel();
        let worker = {
            let wake = wake.clone();
            thread::spawn(move || sender.send(wake.wait(None)).unwrap())
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
                    wake.wait(None);
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
        assert_eq!(wake.wait(Some(&watch)), WakeReason::DeviceRemoved);
        signaller.join().unwrap();

        wake.wake();
        assert_eq!(wake.wait(Some(&watch)), WakeReason::Requested);
    }
}
