//! Frame pacing for macOS windows.
//!
//! On macOS 14 and later a visible window is paced by a `CAMetalDisplayLink`
//! on its layer ([`gpui_apple::display_link::MetalFrameSource`], shared with
//! GPUI's frame waker through [`MetalFrames`]), unless that is turned off
//! (see [`metal_display_link_enabled`]). Earlier systems, and windows that
//! keep rendering while hidden, use the shared `CVDisplayLink` registry
//! described below ([`WindowFrameSource`]).
//!
//! CVDisplayLink has no safe teardown: `CVDisplayLinkStop` merely flags the
//! link's io thread to exit "soon" and returns immediately, and there is no
//! way to learn when (or whether) a final output callback has finished. Two
//! crash classes came from ignoring this:
//!
//! * releasing the link after `stop` let the still-running io thread read the
//!   freed link's internals (segfault in `CVHWTime::reset`, fixed by leaking
//!   the link in #32116);
//! * releasing the dispatch source that the output callback dereferenced via
//!   its context pointer let a straggler callback read freed memory (segfault
//!   in `dispatch_source_merge_data`, Sentry issue ZED-7XR).
//!
//! Instead of leaking one link and one source per teardown (and teardown
//! happens on every resize tick, activation, and occlusion change), we keep a
//! single immortal `CVDisplayLink` per display in a static registry and have
//! windows subscribe to it:
//!
//! * Links are created lazily and never released, so the release race cannot
//!   occur. The output callback's context is the display id (an integer, not
//!   a pointer), and the only memory the callback dereferences is the static
//!   registry, so a straggler callback after `stop` is a harmless no-op.
//! * Each window owns one dispatch source for the life of the window. On
//!   window close it is removed from the registry under the lock (making it
//!   unreachable from the callback), cancelled, and genuinely released.
//! * A display's link runs iff it has subscribers; windows never start or
//!   stop links directly, so interleaved starts/stops of windows sharing a
//!   display cannot conflict.
//!
//! One tradeoff of immortal entries: a link created for a given
//! `CGDirectDisplayID` is reused forever, including after the display is
//! unplugged and one reappears with the same id, or after mode/refresh-rate
//! changes. `CVDisplayLink` looks up display timing dynamically, so a cached
//! link keeps pacing correctly; if that ever proves untrue the fix is to
//! also refresh entries from a display-reconfiguration callback, not to
//! release links (which would reintroduce the teardown race).
//!
//! Lock ordering: the output callback runs on the link's io thread and takes
//! the registry lock, possibly while holding CVDisplayLink-internal locks. To
//! avoid a lock cycle through those (undocumented) internals, we never call a
//! CoreVideo function while holding the registry lock. Registry mutations and
//! link start/stop happen on the main thread only, which keeps the `running`
//! flag and the link's actual state consistent without holding the lock
//! across the calls.
//!
//! `std::sync::Mutex` rather than `parking_lot` is deliberate: on macOS it is
//! currently backed by `os_unfair_lock`, whose priority donation resolves
//! inversions between the high-priority io thread and the main thread. (That
//! is a std implementation detail, not a guarantee; if it changes, the cost
//! is added latency under contention, not incorrectness.)

use crate::main_thread_marker;
use anyhow::Result;
use core_graphics::display::CGDirectDisplayID;
use dispatch2::{
    _dispatch_source_type_data_add, DispatchObject, DispatchQueue, DispatchRetained, DispatchSource,
};
use gpui::{FrameRequestSource, PlatformFrameSignal};
use gpui_apple::display_link::MetalFrameSource;
use gpui_util::ResultExt;
use scheduler::Instant;
use std::{
    cell::{Cell, RefCell},
    collections::{BTreeMap, btree_map},
    ffi::c_void,
    sync::{
        Arc, Mutex, MutexGuard, OnceLock, PoisonError,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    time::Duration,
};

static REGISTRY: Mutex<Registry> = Mutex::new(Registry::new());

struct Registry {
    displays: BTreeMap<CGDirectDisplayID, DisplayEntry>,
    next_subscriber_id: u64,
}

impl Registry {
    const fn new() -> Self {
        Registry {
            displays: BTreeMap::new(),
            next_subscriber_id: 0,
        }
    }
}

struct DisplayEntry {
    link: sys::DisplayLink,
    running: bool,
    subscribers: Vec<DisplayLinkSubscriber>,
}

struct DisplayLinkSubscriber {
    id: SubscriberId,
    frame_requests: DispatchRetained<DispatchSource>,
    signal: Arc<PlatformFrameSignal>,
    frame_demand: Arc<AtomicBool>,
    frame_deadline: Arc<AtomicU64>,
}

// SAFETY: Both fields wrapping raw pointers are refcounted handles to
// thread-safe objects that are valid on any thread: `sys::DisplayLink` to a
// CoreVideo object, and each subscriber's `DispatchRetained<DispatchSource>`
// to a GCD object (which the display's io thread really does use, calling
// `merge_data` from the output callback). All mutation of the entry itself
// is serialized by the registry lock.
unsafe impl Send for DisplayEntry {}

#[derive(Copy, Clone, PartialEq, Eq)]
struct SubscriberId(u64);

fn lock_registry() -> MutexGuard<'static, Registry> {
    // Proceeding past poison is safe here (the map's invariants hold after
    // any partial mutation), and panicking instead would abort the process
    // when this is reached from the extern "C" output callback.
    REGISTRY.lock().unwrap_or_else(PoisonError::into_inner)
}

fn debug_assert_main_thread() {
    debug_assert!(
        main_thread_marker().is_some(),
        "display link registry mutations must happen on the main thread; \
         the registry's lock ordering and state consistency depend on it"
    );
}

unsafe extern "C" fn display_link_output_callback(
    _display_link_out: *mut sys::CVDisplayLink,
    current_time: *const sys::CVTimeStamp,
    output_time: *const sys::CVTimeStamp,
    _flags_in: i64,
    _flags_out: *mut i64,
    display_id: *mut c_void,
) -> i32 {
    let signal_at = PlatformFrameSignal::capture(|| {
        // Instant cannot be constructed from Mach ticks, so sample both clock domains.
        let current_host_time = objc2_core_video::CVGetCurrentHostTime();
        let received_at = Instant::now();
        let host_clock_frequency = objc2_core_video::CVGetHostClockFrequency();
        // SAFETY: CoreVideo owns this timestamp for the lifetime of the callback.
        let current_time = unsafe { current_time.as_ref() };
        frame_request_instant(
            current_time,
            current_host_time,
            host_clock_frequency,
            received_at,
        )
    });
    // SAFETY: CoreVideo owns this timestamp for the lifetime of the callback.
    let output_time = unsafe { output_time.as_ref() };
    let frame_deadline =
        frame_deadline_host_time(output_time, objc2_core_video::CVGetHostClockFrequency())
            .unwrap_or(0);
    let display_id = display_id as usize as CGDirectDisplayID;
    let registry = lock_registry();
    if let Some(entry) = registry.displays.get(&display_id) {
        for subscriber in &entry.subscribers {
            if subscriber.frame_demand.swap(false, Ordering::AcqRel) {
                if let Some(signal_at) = signal_at {
                    subscriber
                        .signal
                        .record(signal_at, FrameRequestSource::NativeCallback);
                }
                subscriber.frame_deadline.store(frame_deadline, Ordering::Release);
                subscriber.frame_requests.merge_data(1);
            }
        }
    }
    0
}

/// Maps a CoreVideo host timestamp using a paired clock sample.
///
/// Invalid timestamps or clock conversions fall back to callback receipt time.
fn frame_request_instant(
    timestamp: Option<&sys::CVTimeStamp>,
    current_host_time: u64,
    host_clock_frequency: f64,
    received_at: Instant,
) -> Instant {
    let Some(timestamp) = timestamp.filter(|timestamp| {
        timestamp.version == 0 && timestamp.flags & sys::kCVTimeStampHostTimeValid != 0
    }) else {
        return received_at;
    };
    if !host_clock_frequency.is_finite() || host_clock_frequency <= 0.0 {
        return received_at;
    }
    current_host_time
        .checked_sub(timestamp.host_time)
        .and_then(|elapsed_ticks| {
            Duration::try_from_secs_f64(elapsed_ticks as f64 / host_clock_frequency).ok()
        })
        .and_then(|elapsed| received_at.checked_sub(elapsed))
        .unwrap_or(received_at)
}

/// The host time by which a frame requested on this callback has to be
/// drawn. CoreVideo's output time is the refresh that shows a frame drawn
/// now; the compositor puts it on screen during the refresh before, so the
/// frame is due when that one starts, as Core Animation expects of an app's
/// frames. `None` when CoreVideo gave no usable timing.
fn frame_deadline_host_time(
    output_time: Option<&sys::CVTimeStamp>,
    host_clock_frequency: f64,
) -> Option<u64> {
    const PERIOD_VALID: u64 =
        sys::kCVTimeStampHostTimeValid | sys::kCVTimeStampVideoRefreshPeriodValid;
    let timestamp = output_time.filter(|timestamp| {
        timestamp.version == 0
            && timestamp.flags & PERIOD_VALID == PERIOD_VALID
            && timestamp.video_time_scale > 0
            && timestamp.video_refresh_period > 0
    })?;
    if !host_clock_frequency.is_finite() || host_clock_frequency <= 0.0 {
        return None;
    }
    let period_seconds =
        timestamp.video_refresh_period as f64 / timestamp.video_time_scale as f64;
    // No display refreshes faster than 1000 Hz or slower than once a second;
    // a period outside that is a timestamp to ignore, not a deadline.
    if !(0.001..=1.0).contains(&period_seconds) {
        return None;
    }
    timestamp
        .host_time
        .checked_sub((period_seconds * host_clock_frequency).round() as u64)
        .filter(|deadline| *deadline != 0)
}

fn subscribe(
    display_id: CGDirectDisplayID,
    frame_requests: DispatchRetained<DispatchSource>,
    frame_demand: Arc<AtomicBool>,
    frame_deadline: Arc<AtomicU64>,
    signal: Arc<PlatformFrameSignal>,
) -> Result<SubscriberId> {
    debug_assert_main_thread();

    let needs_link = !lock_registry().displays.contains_key(&display_id);
    let new_link = if needs_link {
        // Created outside the registry lock; see the lock ordering note above.
        Some(unsafe {
            sys::DisplayLink::new(
                display_id,
                display_link_output_callback,
                display_id as usize as *mut c_void,
            )?
        })
    } else {
        None
    };

    let (subscriber_id, link_to_start) = {
        let mut registry = lock_registry();
        let registry = &mut *registry;
        let subscriber_id = SubscriberId(registry.next_subscriber_id);
        registry.next_subscriber_id += 1;
        let entry = match (registry.displays.entry(display_id), new_link) {
            // If an entry appeared since the check above, dropping `new_link`
            // is safe: a never-started link has no io thread, so releasing it
            // doesn't race.
            (btree_map::Entry::Occupied(entry), _) => entry.into_mut(),
            (btree_map::Entry::Vacant(vacant), Some(link)) => vacant.insert(DisplayEntry {
                link,
                running: false,
                subscribers: Vec::new(),
            }),
            (btree_map::Entry::Vacant(_), None) => {
                // Entries are never removed, so an entry observed above cannot
                // have disappeared while subscriptions stay on the main thread.
                anyhow::bail!("display link registry entry vanished for display {display_id}");
            }
        };
        entry.subscribers.push(DisplayLinkSubscriber {
            id: subscriber_id,
            frame_requests,
            signal,
            frame_demand,
            frame_deadline,
        });
        let link_to_start = if entry.running {
            None
        } else {
            entry.running = true;
            // Clone the refcounted handle so the CVDisplayLinkStart call can
            // happen after the lock is released.
            Some(entry.link.clone())
        };
        (subscriber_id, link_to_start)
    };

    if let Some(mut link) = link_to_start {
        if let Err(error) = unsafe { link.start() } {
            let mut registry = lock_registry();
            if let Some(entry) = registry.displays.get_mut(&display_id) {
                entry.running = false;
                entry.subscribers.retain(|subscriber| {
                    if subscriber.id == subscriber_id {
                        subscriber.signal.take();
                        false
                    } else {
                        true
                    }
                });
            }
            return Err(error);
        }
    }

    Ok(subscriber_id)
}

fn unsubscribe(display_id: CGDirectDisplayID, subscriber_id: SubscriberId) {
    debug_assert_main_thread();

    let link_to_stop = {
        let mut registry = lock_registry();
        let Some(entry) = registry.displays.get_mut(&display_id) else {
            return;
        };
        entry
            .subscribers
            .retain(|subscriber| subscriber.id != subscriber_id);
        if entry.subscribers.is_empty() && entry.running {
            entry.running = false;
            Some(entry.link.clone())
        } else {
            None
        }
    };

    if let Some(mut link) = link_to_stop {
        // A final output callback can still fire after this returns; it finds
        // no subscribers for this display and does nothing.
        unsafe { link.stop().log_err() };
    }
}

/// A per-window source of frame requests, paced by the display the window is
/// on. The wrapped dispatch source coalesces vsync ticks from the display's
/// io thread and invokes `callback(data)` on the main queue.
///
/// Requests are demand-driven: a refresh only reaches the window when its
/// `frame_demand` flag was set since the last delivered request. The link itself keeps running
/// while the window is subscribed; starting and stopping it per idle period
/// would reintroduce the teardown races described above.
pub struct WindowFrameSource {
    frame_requests: DispatchRetained<DispatchSource>,
    frame_demand: Arc<AtomicBool>,
    frame_deadline: Arc<AtomicU64>,
    registration: Option<(CGDirectDisplayID, SubscriberId)>,
    signal: Arc<PlatformFrameSignal>,
}

impl WindowFrameSource {
    /// `frame_demand` is shared with whoever asks for frames; setting it
    /// (from any thread) asks for a request on the display's next refresh.
    pub fn new(
        data: *mut c_void,
        callback: extern "C" fn(*mut c_void),
        frame_demand: Arc<AtomicBool>,
    ) -> Self {
        let frame_requests = unsafe {
            let frame_requests = DispatchSource::new(
                &raw const _dispatch_source_type_data_add as *mut _,
                0,
                0,
                Some(DispatchQueue::main()),
            );
            frame_requests.set_context(data);
            frame_requests.set_event_handler_f(callback);
            // Resume before this source can ever be dropped: destroying a
            // suspended dispatch source is undefined behavior (#50875).
            frame_requests.resume();
            frame_requests
        };
        Self {
            frame_requests,
            frame_demand,
            frame_deadline: Arc::new(AtomicU64::new(0)),
            registration: None,
            signal: Arc::new(PlatformFrameSignal::new()),
        }
    }

    pub fn start(&mut self, display_id: CGDirectDisplayID) -> Result<()> {
        self.stop();
        // A (re)subscription follows an occlusion, screen, resize or
        // activation change; deliver a frame on the first refresh so the
        // window catches up even if nothing was pending.
        self.frame_demand.store(true, Ordering::Release);
        let subscriber_id = subscribe(
            display_id,
            self.frame_requests.clone(),
            self.frame_demand.clone(),
            self.frame_deadline.clone(),
            self.signal.clone(),
        )?;
        self.registration = Some((display_id, subscriber_id));
        Ok(())
    }

    pub fn debug_state(&self) -> serde_json::Value {
        serde_json::json!({
            "kind": "CVDisplayLink",
            "attached": self.registration.is_some(),
            "display_id": self.registration.map(|(display_id, _)| display_id),
        })
    }

    pub fn take_signal(&self) -> Option<(Instant, FrameRequestSource)> {
        self.signal.take()
    }

    /// The host time by which the frame this source last requested has to
    /// be drawn, once: `None` when the display gave no timing, or the
    /// request was already taken.
    pub fn take_frame_deadline(&self) -> Option<u64> {
        Some(self.frame_deadline.swap(0, Ordering::AcqRel)).filter(|deadline| *deadline != 0)
    }

    pub fn stop(&mut self) {
        if let Some((display_id, subscriber_id)) = self.registration.take() {
            unsubscribe(display_id, subscriber_id);
        }
        self.signal.take();
        self.frame_deadline.store(0, Ordering::Release);
    }

    /// Whether this source is subscribed to a display's link, and so
    /// delivers a frame request on that display's next refresh.
    pub fn is_running(&self) -> bool {
        self.registration.is_some()
    }
}

/// Whether visible windows are paced by a `CAMetalDisplayLink`: on a system
/// that has one, unless `GPUI_METAL_DISPLAY_LINK=0` asks for the
/// `CVDisplayLink` pacing instead.
///
/// The link keeps a drawable pool of its own, five drawables deep at 120 Hz
/// and four at 60 Hz on macOS 27, whatever the layer's
/// `maximumDrawableCount` (which raises an exception if set while a link
/// drives the layer) or the link's `preferredFrameLatency` say. So a window
/// that draws steadily holds one or two more full-size drawables than the
/// three the `CVDisplayLink` pacing keeps.
pub(crate) fn metal_display_link_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        !std::env::var("GPUI_METAL_DISPLAY_LINK").is_ok_and(|value| value == "0")
            && MetalFrameSource::is_available()
    })
}

/// The share of the time to a `CAMetalDisplayLink` update's target that an
/// interactive frame reports as its deadline; see [`metal_frame_deadline`].
const METAL_FRAME_DEADLINE_SHARE: f64 = 0.3;

/// The deadline an interactive frame drawn for a `CAMetalDisplayLink`
/// update reports to the frame work interval, as host times: the slack a
/// `CVDisplayLink` frame has, not the link's.
///
/// CoreVideo calls back about 3.3 ms before a refresh at 120 Hz, and a frame
/// it paces is due at that refresh, so the main thread starts it about
/// 2.5 ms (0.3 of the period) before its deadline. The link's update comes
/// at a refresh and targets the next one, a whole period later. Reporting
/// that target let the performance controller run scroll frames on the
/// slower cores, with 21% more CPU time and a 74% longer 90th percentile
/// draw for the same instructions (measured on macOS 27, 2026-11).
pub(crate) fn metal_frame_deadline(now: u64, target: u64) -> u64 {
    match target.checked_sub(now) {
        Some(slack) if slack > 0 => {
            now + (slack as f64 * METAL_FRAME_DEADLINE_SHARE).round().max(1.0) as u64
        }
        _ => target,
    }
}

/// A window's `CAMetalDisplayLink`, shared with GPUI's frame waker so that a
/// wake reaches the current link without taking the window's lock.
///
/// The link pauses after any update in which GPUI neither drew, presented,
/// nor asked for another frame, so an idle window acquires no drawables. GPUI
/// decides when frames are needed (including presenting after high-rate
/// input); this only stops the link between those decisions.
#[derive(Default)]
pub struct MetalFrames {
    source: RefCell<Option<MetalFrameSource>>,
    /// Whether GPUI asked for a frame since the current update began.
    woken: Cell<bool>,
}

impl MetalFrames {
    pub fn replace(&self, source: Option<MetalFrameSource>) {
        *self.source.borrow_mut() = source;
    }

    pub fn is_attached(&self) -> bool {
        self.source.borrow().is_some()
    }

    pub fn wake(&self) {
        self.woken.set(true);
        // A source being replaced starts unpaused, so skipping it is safe.
        if let Ok(source) = self.source.try_borrow()
            && let Some(source) = source.as_ref()
        {
            source.set_paused(false);
        }
    }

    pub fn begin_frame(&self) {
        self.woken.set(false);
    }

    pub fn debug_state(&self) -> serde_json::Value {
        let paused = self
            .source
            .try_borrow()
            .ok()
            .and_then(|source| source.as_ref().map(MetalFrameSource::is_paused));
        serde_json::json!({
            "kind": "CAMetalDisplayLink",
            "attached": self.is_attached(),
            "paused": paused,
            "woken": self.woken.get(),
        })
    }

    /// Ends an update, pausing the link if GPUI neither used its drawable nor
    /// asked for another frame while it ran. Returns whether it paused.
    pub fn end_frame(&self, used_drawable: bool) -> bool {
        let pause = !used_drawable && !self.woken.get();
        if pause && let Some(source) = self.source.borrow().as_ref() {
            source.set_paused(true);
        }
        pause
    }
}

/// The host time (in `mach_absolute_time` ticks, at `host_clock_frequency`
/// ticks per second) of a Core Animation media time, which counts seconds on
/// the same clock. `None` for a time that does not convert to a usable
/// deadline.
pub(crate) fn media_time_to_host_time(media_time: f64, host_clock_frequency: f64) -> Option<u64> {
    if !media_time.is_finite()
        || media_time <= 0.0
        || !host_clock_frequency.is_finite()
        || host_clock_frequency <= 0.0
    {
        return None;
    }
    let host_time = (media_time * host_clock_frequency).round();
    (host_time >= 1.0 && host_time < u64::MAX as f64).then_some(host_time as u64)
}

impl Drop for WindowFrameSource {
    fn drop(&mut self) {
        self.stop();
        // Unsubscribing makes this source unreachable from the output
        // callback, so unlike before (ZED-7XR) it is safe to actually release
        // it. Cancelling first guarantees the event handler never runs again;
        // its context points at the window's native view, which may be
        // deallocated after this.
        self.frame_requests.cancel();
    }
}

mod sys {
    //! Derived from display-link crate under the following license:
    //! <https://github.com/BrainiumLLC/display-link/blob/master/LICENSE-MIT>
    //! Apple docs: [CVDisplayLink](https://developer.apple.com/documentation/corevideo/cvdisplaylinkoutputcallback?language=objc)
    #![allow(dead_code, non_upper_case_globals)]

    use anyhow::Result;
    use core_graphics::display::CGDirectDisplayID;
    use foreign_types::{ForeignType, foreign_type};
    use std::{
        ffi::c_void,
        fmt::{self, Debug, Formatter},
    };

    #[derive(Debug)]
    pub enum CVDisplayLink {}

    foreign_type! {
        pub unsafe type DisplayLink {
            type CType = CVDisplayLink;
            fn drop = CVDisplayLinkRelease;
            fn clone = CVDisplayLinkRetain;
        }
    }

    impl Debug for DisplayLink {
        fn fmt(&self, formatter: &mut Formatter) -> fmt::Result {
            formatter
                .debug_tuple("DisplayLink")
                .field(&self.as_ptr())
                .finish()
        }
    }

    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    pub(crate) struct CVTimeStamp {
        pub version: u32,
        pub video_time_scale: i32,
        pub video_time: i64,
        pub host_time: u64,
        pub rate_scalar: f64,
        pub video_refresh_period: i64,
        pub smpte_time: CVSMPTETime,
        pub flags: u64,
        pub reserved: u64,
    }

    pub type CVTimeStampFlags = u64;

    pub const kCVTimeStampVideoTimeValid: CVTimeStampFlags = 1 << 0;
    pub const kCVTimeStampHostTimeValid: CVTimeStampFlags = 1 << 1;
    pub const kCVTimeStampSMPTETimeValid: CVTimeStampFlags = 1 << 2;
    pub const kCVTimeStampVideoRefreshPeriodValid: CVTimeStampFlags = 1 << 3;
    pub const kCVTimeStampRateScalarValid: CVTimeStampFlags = 1 << 4;
    pub const kCVTimeStampTopField: CVTimeStampFlags = 1 << 16;
    pub const kCVTimeStampBottomField: CVTimeStampFlags = 1 << 17;
    pub const kCVTimeStampVideoHostTimeValid: CVTimeStampFlags =
        kCVTimeStampVideoTimeValid | kCVTimeStampHostTimeValid;
    pub const kCVTimeStampIsInterlaced: CVTimeStampFlags =
        kCVTimeStampTopField | kCVTimeStampBottomField;

    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    pub(crate) struct CVSMPTETime {
        pub subframes: i16,
        pub subframe_divisor: i16,
        pub counter: u32,
        pub time_type: u32,
        pub flags: u32,
        pub hours: i16,
        pub minutes: i16,
        pub seconds: i16,
        pub frames: i16,
    }

    pub type CVSMPTETimeType = u32;

    pub const kCVSMPTETimeType24: CVSMPTETimeType = 0;
    pub const kCVSMPTETimeType25: CVSMPTETimeType = 1;
    pub const kCVSMPTETimeType30Drop: CVSMPTETimeType = 2;
    pub const kCVSMPTETimeType30: CVSMPTETimeType = 3;
    pub const kCVSMPTETimeType2997: CVSMPTETimeType = 4;
    pub const kCVSMPTETimeType2997Drop: CVSMPTETimeType = 5;
    pub const kCVSMPTETimeType60: CVSMPTETimeType = 6;
    pub const kCVSMPTETimeType5994: CVSMPTETimeType = 7;

    pub type CVSMPTETimeFlags = u32;

    pub const kCVSMPTETimeValid: CVSMPTETimeFlags = 1 << 0;
    pub const kCVSMPTETimeRunning: CVSMPTETimeFlags = 1 << 1;

    pub type CVDisplayLinkOutputCallback = unsafe extern "C" fn(
        display_link_out: *mut CVDisplayLink,
        // A pointer to the current timestamp. This represents the timestamp when the callback is called.
        current_time: *const CVTimeStamp,
        // A pointer to the output timestamp. This represents the timestamp for when the frame will be displayed.
        output_time: *const CVTimeStamp,
        // Unused
        flags_in: i64,
        // Unused
        flags_out: *mut i64,
        // A pointer to app-defined data.
        display_link_context: *mut c_void,
    ) -> i32;

    #[link(name = "CoreFoundation", kind = "framework")]
    #[link(name = "CoreVideo", kind = "framework")]
    #[allow(improper_ctypes, unknown_lints, clippy::duplicated_attributes)]
    unsafe extern "C" {
        pub fn CVDisplayLinkCreateWithActiveCGDisplays(
            display_link_out: *mut *mut CVDisplayLink,
        ) -> i32;
        pub fn CVDisplayLinkSetCurrentCGDisplay(
            display_link: &mut DisplayLinkRef,
            display_id: u32,
        ) -> i32;
        pub fn CVDisplayLinkSetOutputCallback(
            display_link: &mut DisplayLinkRef,
            callback: CVDisplayLinkOutputCallback,
            user_info: *mut c_void,
        ) -> i32;
        pub fn CVDisplayLinkStart(display_link: &mut DisplayLinkRef) -> i32;
        pub fn CVDisplayLinkStop(display_link: &mut DisplayLinkRef) -> i32;
        pub fn CVDisplayLinkRelease(display_link: *mut CVDisplayLink);
        pub fn CVDisplayLinkRetain(display_link: *mut CVDisplayLink) -> *mut CVDisplayLink;
    }

    impl DisplayLink {
        /// Apple docs: [CVDisplayLinkCreateWithCGDisplay](https://developer.apple.com/documentation/corevideo/1456981-cvdisplaylinkcreatewithcgdisplay?language=objc)
        pub unsafe fn new(
            display_id: CGDirectDisplayID,
            callback: CVDisplayLinkOutputCallback,
            user_info: *mut c_void,
        ) -> Result<Self> {
            unsafe {
                let mut display_link: *mut CVDisplayLink = 0 as _;

                let code = CVDisplayLinkCreateWithActiveCGDisplays(&mut display_link);
                anyhow::ensure!(code == 0, "could not create display link, code: {}", code);

                let mut display_link = DisplayLink::from_ptr(display_link);

                let code = CVDisplayLinkSetOutputCallback(&mut display_link, callback, user_info);
                anyhow::ensure!(code == 0, "could not set output callback, code: {}", code);

                let code = CVDisplayLinkSetCurrentCGDisplay(&mut display_link, display_id);
                anyhow::ensure!(
                    code == 0,
                    "could not assign display to display link, code: {}",
                    code
                );

                Ok(display_link)
            }
        }
    }

    impl DisplayLinkRef {
        /// Apple docs: [CVDisplayLinkStart](https://developer.apple.com/documentation/corevideo/1457193-cvdisplaylinkstart?language=objc)
        pub unsafe fn start(&mut self) -> Result<()> {
            unsafe {
                let code = CVDisplayLinkStart(self);
                anyhow::ensure!(code == 0, "could not start display link, code: {}", code);
                Ok(())
            }
        }

        /// Apple docs: [CVDisplayLinkStop](https://developer.apple.com/documentation/corevideo/1457281-cvdisplaylinkstop?language=objc)
        pub unsafe fn stop(&mut self) -> Result<()> {
            unsafe {
                let code = CVDisplayLinkStop(self);
                anyhow::ensure!(code == 0, "could not stop display link, code: {}", code);
                Ok(())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use objc2::AnyThread;
    use objc2_quartz_core::CAMetalDisplayLink;

    #[test]
    fn metal_frames_pause_only_when_gpui_neither_draws_nor_asks_again() {
        let frames = MetalFrames::default();

        frames.begin_frame();
        assert!(
            !frames.end_frame(true),
            "a frame that drew keeps the link running"
        );

        frames.begin_frame();
        frames.wake();
        assert!(
            !frames.end_frame(false),
            "a wake during the update, like a throttled frame's retry, keeps it running"
        );

        frames.wake();
        frames.begin_frame();
        assert!(
            frames.end_frame(false),
            "a wake before the update was served by it"
        );

        frames.begin_frame();
        assert!(frames.end_frame(false), "an idle update pauses the link");
    }

    #[test]
    fn a_renderer_draws_normally_after_its_display_link_is_invalidated() -> anyhow::Result<()> {
        if !MetalFrameSource::is_available() {
            return Ok(());
        }
        let mut renderer = crate::renderer::MetalRenderer::new(Default::default(), false, false)?;
        renderer.update_drawable_size(gpui::size(gpui::DevicePixels(64), gpui::DevicePixels(64)));
        let layer = renderer
            .layer()
            .map(objc2::Message::retain)
            .ok_or_else(|| anyhow::anyhow!("a window renderer has a layer"))?;
        // The link is never added to a run loop: attaching it to the layer is
        // the state under test.
        let link = CAMetalDisplayLink::initWithMetalLayer(CAMetalDisplayLink::alloc(), &layer);
        assert!(renderer.can_attach_display_link());
        renderer.set_display_link_attached(true);
        assert_eq!(
            renderer.draw(&gpui::Scene::default()).outcome,
            gpui::PresentOutcome::Deferred,
            "a present outside the link's update has no drawable"
        );

        link.invalidate();
        renderer.set_display_link_attached(false);
        assert_eq!(
            renderer.draw(&gpui::Scene::default()).outcome,
            gpui::PresentOutcome::Presented
        );
        Ok(())
    }

    #[test]
    fn metal_frame_deadline_reports_a_cv_display_link_frames_slack() {
        // 120 Hz at 24 MHz: a period is 200_000 ticks.
        assert_eq!(metal_frame_deadline(1_000_000, 1_200_000), 1_060_000);
        assert_eq!(
            metal_frame_deadline(1_000_000, 1_000_001),
            1_000_001,
            "a deadline never lands on the frame's own start"
        );
        assert_eq!(
            metal_frame_deadline(1_000_000, 1_000_000),
            1_000_000,
            "a target already reached is reported as it is"
        );
        assert_eq!(metal_frame_deadline(1_000_000, 900_000), 900_000);
    }

    #[test]
    fn media_time_converts_to_host_ticks() {
        assert_eq!(media_time_to_host_time(2.0, 24e6), Some(48_000_000));
        assert_eq!(media_time_to_host_time(0.5, 1e9), Some(500_000_000));
        for (media_time, frequency) in [
            (0.0, 24e6),
            (-1.0, 24e6),
            (f64::NAN, 24e6),
            (f64::INFINITY, 24e6),
            (1.0, 0.0),
            (1.0, f64::NAN),
            (f64::MAX, 24e6),
        ] {
            assert_eq!(media_time_to_host_time(media_time, frequency), None);
        }
    }

    #[test]
    fn host_timestamp_preserves_request_age_at_different_clock_rates() {
        let received_at = Instant::now();
        for (host_time, current_host_time, frequency, age) in [
            (4_000_000, 10_000_000, 1e9, Duration::from_millis(6)),
            (1_760_000, 2_000_000, 24e6, Duration::from_millis(10)),
        ] {
            let timestamp = timestamp(host_time);
            assert_eq!(
                frame_request_instant(Some(&timestamp), current_host_time, frequency, received_at),
                received_at - age,
            );
        }
    }

    #[test]
    fn invalid_host_timestamps_fall_back_to_receipt_time() {
        let received_at = Instant::now();
        let valid = timestamp(1_000);
        let mut missing_host_time = valid;
        missing_host_time.flags = 0;
        let mut unknown_version = valid;
        unknown_version.version = 1;
        for timestamp in [None, Some(&missing_host_time), Some(&unknown_version)] {
            assert_eq!(
                frame_request_instant(timestamp, 2_000, 1e9, received_at),
                received_at,
            );
        }
        assert_eq!(
            frame_request_instant(Some(&valid), 999, 1e9, received_at),
            received_at,
        );
        for frequency in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::MIN_POSITIVE] {
            assert_eq!(
                frame_request_instant(Some(&valid), 2_000, frequency, received_at),
                received_at,
            );
        }
    }

    #[test]
    fn frame_deadline_is_the_refresh_before_the_output_time() {
        // CoreVideo's own numbers on a 120 Hz panel: a 24 MHz host clock and
        // a refresh period of 200000 in a 24000000 time scale.
        let output = refresh_timestamp(1_000_000, 200_000, 24_000_000);
        assert_eq!(frame_deadline_host_time(Some(&output), 24e6), Some(800_000));
        let at_60_hz = refresh_timestamp(1_000_000, 1, 60);
        assert_eq!(frame_deadline_host_time(Some(&at_60_hz), 24e6), Some(600_000));
    }

    #[test]
    fn frame_deadline_needs_a_valid_refresh_period() {
        let valid = refresh_timestamp(1_000_000, 200_000, 24_000_000);
        let mut no_period = valid;
        no_period.flags = sys::kCVTimeStampHostTimeValid;
        let mut no_host_time = valid;
        no_host_time.flags = sys::kCVTimeStampVideoRefreshPeriodValid;
        let mut unknown_version = valid;
        unknown_version.version = 1;
        let mut zero_scale = valid;
        zero_scale.video_time_scale = 0;
        let mut too_long = valid;
        too_long.video_refresh_period = 48_000_000;
        let mut too_short = valid;
        too_short.video_refresh_period = 1;
        for timestamp in [
            None,
            Some(&no_period),
            Some(&no_host_time),
            Some(&unknown_version),
            Some(&zero_scale),
            Some(&too_long),
            Some(&too_short),
        ] {
            assert_eq!(frame_deadline_host_time(timestamp, 24e6), None);
        }
        for frequency in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert_eq!(frame_deadline_host_time(Some(&valid), frequency), None);
        }
        for host_time in [0, 200_000] {
            let before_any_refresh = refresh_timestamp(host_time, 200_000, 24_000_000);
            assert_eq!(frame_deadline_host_time(Some(&before_any_refresh), 24e6), None);
        }
    }

    fn refresh_timestamp(host_time: u64, period: i64, scale: i32) -> sys::CVTimeStamp {
        sys::CVTimeStamp {
            host_time,
            video_refresh_period: period,
            video_time_scale: scale,
            flags: sys::kCVTimeStampHostTimeValid | sys::kCVTimeStampVideoRefreshPeriodValid,
            ..Default::default()
        }
    }

    fn timestamp(host_time: u64) -> sys::CVTimeStamp {
        sys::CVTimeStamp {
            host_time,
            flags: sys::kCVTimeStampHostTimeValid,
            ..Default::default()
        }
    }
}
