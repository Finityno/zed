use std::{
    ffi::c_void,
    ptr::NonNull,
    sync::atomic::{AtomicBool, AtomicU64, Ordering},
    thread::{ThreadId, current},
    time::Duration,
};

use anyhow::Context;
use gpui_util::ResultExt;
use windows::Win32::{
    Foundation::{FILETIME, LPARAM, WPARAM},
    Media::{timeBeginPeriod, timeEndPeriod},
    System::Threading::{
        CloseThreadpoolTimer, CreateThreadpoolTimer, GetCurrentThread, PTP_CALLBACK_INSTANCE,
        PTP_TIMER, SetThreadPriority, SetThreadpoolTimer, THREAD_PRIORITY_TIME_CRITICAL,
        TP_CALLBACK_ENVIRON_V3, TP_CALLBACK_PRIORITY, TP_CALLBACK_PRIORITY_HIGH,
        TP_CALLBACK_PRIORITY_LOW, TP_CALLBACK_PRIORITY_NORMAL, TrySubmitThreadpoolCallback,
    },
    UI::WindowsAndMessaging::PostMessageW,
};

use crate::{HWND, SafeHwnd, WM_GPUI_TASK_DISPATCHED_ON_MAIN_THREAD};
use gpui::{
    PlatformDispatcher, Priority, PriorityQueueSender, RunnableVariant, TimerResolutionGuard,
};

/// Foreground-queue depth past which each doubling is logged as an error. A
/// healthy queue stays in the tens even under load; six figures of undrained
/// runnables only happens when the main thread produces without draining.
const MAIN_THREAD_QUEUE_ALARM_DEPTH: usize = 128 * 1024;

/// How many times to try posting the foreground wake-up before giving up on
/// it. See [`WindowsDispatcher::wake_main_thread`].
const WAKE_POST_ATTEMPTS: usize = 3;

const QUEUE_ALARM_GENERATION_STEP: u64 = 1 << 32;
const QUEUE_ALARM_DEPTH_MASK: u64 = QUEUE_ALARM_GENERATION_STEP - 1;

pub(crate) struct WindowsDispatcher {
    pub(crate) wake_posted: AtomicBool,
    main_sender: PriorityQueueSender<RunnableVariant>,
    main_thread_id: ThreadId,
    pub(crate) platform_window_handle: SafeHwnd,
    validation_number: usize,
    /// Runaway-queue alarm state: a re-arm generation in the high 32 bits, the
    /// depth at which the next alarm fires in the low 32 bits (a depth past
    /// `u32::MAX` would need more runnables than the address space holds).
    ///
    /// The bar is stateful rather than a `depth.is_power_of_two()` test so a
    /// queue merely *hovering* at the threshold does not log on every push
    /// that lands back on it — an error line per dispatch from the hot path,
    /// on a thread that is already starved.
    ///
    /// The generation is what makes the bar safe to raise from an arbitrary
    /// producer thread. Without it, a producer that reads the bar and then
    /// stalls while the main thread drains to empty and re-arms would land its
    /// compare-exchange on the *re-armed* value, raising the bar on a queue
    /// that is now empty — precisely the ratchet the re-arm exists to undo.
    /// Bumping the generation makes that stale exchange fail instead.
    queue_alarm: AtomicU64,
    /// Set while wake-up posts are failing so a failure streak logs one error,
    /// not one per producer: once the platform window is destroyed at shutdown
    /// every post fails with ERROR_INVALID_WINDOW_HANDLE, and logging each one
    /// emitted dozens of identical error reports in milliseconds. Cleared by
    /// the first successful post, which logs how many failures went unlogged.
    wake_post_failing: AtomicBool,
    suppressed_wake_post_failures: AtomicU64,
}

impl WindowsDispatcher {
    pub(crate) fn new(
        main_sender: PriorityQueueSender<RunnableVariant>,
        platform_window_handle: HWND,
        validation_number: usize,
    ) -> Self {
        let main_thread_id = current().id();
        let platform_window_handle = platform_window_handle.into();

        WindowsDispatcher {
            main_sender,
            main_thread_id,
            platform_window_handle,
            validation_number,
            wake_posted: AtomicBool::new(false),
            queue_alarm: AtomicU64::new(MAIN_THREAD_QUEUE_ALARM_DEPTH as u64),
            wake_post_failing: AtomicBool::new(false),
            suppressed_wake_post_failures: AtomicU64::new(0),
        }
    }

    /// Lower the alarm back to its floor once the main queue has fully
    /// drained. Without this the bar only ever ratchets up, so the first
    /// episode to be caught would also be the last: a hang that peaked at 5M
    /// entries leaves the next alarm at 10M, and every later hang below that
    /// is silent even though the queue recovered in between.
    ///
    /// Called only from the drain on the main thread, so a plain load/store
    /// needs no compare-exchange of its own: a producer that raises the bar
    /// between the two is meant to lose, and the generation bump makes it.
    pub(crate) fn rearm_queue_alarm(&self) {
        let generation = self
            .queue_alarm
            .load(Ordering::Relaxed)
            .wrapping_add(QUEUE_ALARM_GENERATION_STEP)
            & !QUEUE_ALARM_DEPTH_MASK;
        self.queue_alarm.store(
            generation | MAIN_THREAD_QUEUE_ALARM_DEPTH as u64,
            Ordering::Relaxed,
        );
    }

    /// Raises the runaway-queue alarm for a push that landed at `queued`,
    /// returning whether this caller is the one that should log it.
    ///
    /// The compare-exchange covers the generation as well as the bar, so it
    /// both picks a single winner among concurrent producers crossing the same
    /// threshold and fails outright if the queue drained and re-armed while
    /// this caller was deciding.
    fn crossed_queue_alarm(&self, queued: usize) -> bool {
        let alarm = self.queue_alarm.load(Ordering::Relaxed);
        if (queued as u64) < (alarm & QUEUE_ALARM_DEPTH_MASK) {
            return false;
        }
        let next_depth = queued.saturating_mul(2).min(u32::MAX as usize) as u64;
        self.queue_alarm
            .compare_exchange(
                alarm,
                (alarm & !QUEUE_ALARM_DEPTH_MASK) | next_depth,
                Ordering::Relaxed,
                Ordering::Relaxed,
            )
            .is_ok()
    }

    /// Posts the message that drives the foreground drain, claiming
    /// `wake_posted` first so only one wake is ever in flight.
    ///
    /// `PostMessageW` fails when the target thread's message queue is at its
    /// quota or USER handles are exhausted, and neither reaction to that is
    /// sufficient on its own. Leaving the flag set strands the queue forever:
    /// nothing but the drain that wake triggers ever clears it, so no producer
    /// would post again. Clearing it and walking away strands the queue too,
    /// just less permanently — producers that enqueued while this claim was
    /// held skipped their own post on the strength of it, so their runnables
    /// are already queued with nothing scheduled to run them. So: re-claim and
    /// retry until a post lands or another producer takes the claim, and with
    /// it the duty to post.
    pub(crate) fn wake_main_thread(&self) {
        let mut last_error = None;
        for _ in 0..WAKE_POST_ATTEMPTS {
            if self.wake_posted.swap(true, Ordering::AcqRel) {
                // A wake is already in flight, or a drain is running and will
                // re-check the queue after it clears the flag.
                return;
            }
            match self.post_wake_message() {
                Ok(()) => {
                    self.note_wake_post_success();
                    return;
                }
                Err(error) => last_error = Some(error),
            }
            self.wake_posted.store(false, Ordering::Release);
        }
        if self.claim_wake_post_failure_log() {
            log::error!(
                "failed to post the main-thread wake-up {WAKE_POST_ATTEMPTS} times ({last_error:?}); \
                 queued runnables will not run until the next dispatch posts one; \
                 suppressing further failures until a post succeeds"
            );
        }
    }

    /// Re-arms the wake from inside a drain that is yielding with work still
    /// queued. The flag is already this drain's, so there is nothing to claim,
    /// but a failed post strands that work just the same — retry before
    /// releasing the flag to whichever producer enqueues next.
    pub(crate) fn repost_wake(&self) {
        let mut last_error = None;
        for _ in 0..WAKE_POST_ATTEMPTS {
            match self.post_wake_message() {
                Ok(()) => {
                    self.note_wake_post_success();
                    return;
                }
                Err(error) => last_error = Some(error),
            }
        }
        self.wake_posted.store(false, Ordering::Release);
        if self.claim_wake_post_failure_log() {
            log::error!(
                "failed to re-post the main-thread wake-up {WAKE_POST_ATTEMPTS} times ({last_error:?}); \
                 the remaining foreground work will not run until the next dispatch posts one; \
                 suppressing further failures until a post succeeds"
            );
        }
    }

    /// Returns whether this failure starts a streak and should be the one that
    /// logs it. Every wake-up post fails for the same reason once it starts —
    /// the message queue is at quota, USER handles are exhausted, or the
    /// window is gone at shutdown — so the repeats are counted for the
    /// recovery line instead of each producing an identical error.
    fn claim_wake_post_failure_log(&self) -> bool {
        if self.wake_post_failing.swap(true, Ordering::Relaxed) {
            self.suppressed_wake_post_failures
                .fetch_add(1, Ordering::Relaxed);
            return false;
        }
        true
    }

    /// Ends a failure streak, surfacing how many failures went unlogged. The
    /// load-before-swap keeps the successful-post hot path to a read.
    fn note_wake_post_success(&self) {
        if self.wake_post_failing.load(Ordering::Relaxed)
            && self.wake_post_failing.swap(false, Ordering::Relaxed)
        {
            let suppressed = self
                .suppressed_wake_post_failures
                .swap(0, Ordering::Relaxed);
            log::warn!(
                "main-thread wake-up posts recovered after {suppressed} suppressed failures"
            );
        }
    }

    fn post_wake_message(&self) -> windows::core::Result<()> {
        unsafe {
            PostMessageW(
                Some(self.platform_window_handle.as_raw()),
                WM_GPUI_TASK_DISPATCHED_ON_MAIN_THREAD,
                WPARAM(self.validation_number),
                LPARAM(0),
            )
        }
    }

    fn dispatch_on_threadpool(&self, priority: TP_CALLBACK_PRIORITY, runnable: RunnableVariant) {
        let environ = TP_CALLBACK_ENVIRON_V3 {
            Version: 3,
            CallbackPriority: priority,
            Size: size_of::<TP_CALLBACK_ENVIRON_V3>() as u32,
            ..Default::default()
        };

        // If the thread pool never runs our callback, the matching `from_raw` is never called, which leaks the runnable.
        // Dropping the scheduled runnable would cancel its task and make the next poll of any awaiter panic. Since we expect
        // the scenario to usually happen during shutdown, this leak is acceptable.
        let context = runnable.into_raw().as_ptr() as *mut c_void;

        unsafe {
            TrySubmitThreadpoolCallback(Some(run_work_callback), Some(context), Some(&environ))
                .log_err();
        }
    }

    fn dispatch_on_threadpool_after(&self, runnable: RunnableVariant, duration: Duration) {
        let context = runnable.into_raw().as_ptr() as *mut c_void;

        unsafe {
            if let Ok(timer) = CreateThreadpoolTimer(Some(run_timer_callback), Some(context), None)
            {
                // Negative FILETIME expresses a relative delay in 100ns ticks
                let ticks = (duration.as_nanos() / 100).min(i64::MAX as u128) as i64;
                let due = (-ticks) as u64;
                let due_time = FILETIME {
                    dwLowDateTime: due as u32,
                    dwHighDateTime: (due >> 32) as u32,
                };
                SetThreadpoolTimer(timer, Some(&due_time), 0, None);
            }
        }
    }

    #[inline(always)]
    pub(crate) fn execute_runnable(runnable: RunnableVariant) {
        let location = runnable.metadata().location;
        let spawned = runnable.metadata().spawned;
        gpui::profiler::update_running_task(spawned, location);
        runnable.run();
        gpui::profiler::save_task_timing();
    }
}

impl PlatformDispatcher for WindowsDispatcher {
    fn is_main_thread(&self) -> bool {
        current().id() == self.main_thread_id
    }

    fn dispatch(&self, runnable: RunnableVariant, priority: Priority) {
        let priority = match priority {
            Priority::RealtimeAudio => {
                panic!("RealtimeAudio priority should use spawn_realtime, not dispatch")
            }
            Priority::High => TP_CALLBACK_PRIORITY_HIGH,
            Priority::Medium => TP_CALLBACK_PRIORITY_NORMAL,
            Priority::Low => TP_CALLBACK_PRIORITY_LOW,
        };
        self.dispatch_on_threadpool(priority, runnable);
    }

    fn dispatch_on_main_thread(&self, runnable: RunnableVariant, priority: Priority) {
        match self.main_sender.send_and_len(priority, runnable) {
            Ok(queued) => {
                gpui::queue::MAIN_THREAD_QUEUE_DEPTH.store(queued, Ordering::Relaxed);
                // A queue this deep means the main thread is producing
                // without ever returning to the message loop to drain (a
                // nested Win32 loop only pumps sent messages, so the posted
                // wake below cannot arrive) — a field dump reached 164M
                // entries and 17 GB before dying. Runnables cannot be
                // dropped (that cancels their tasks), so the alarm logs
                // once per doubling instead of bounding the queue.
                if self.crossed_queue_alarm(queued) {
                    log::error!(
                        "main-thread dispatcher queue holds {queued} undrained runnables; \
                         the UI thread is enqueueing without returning to the message loop"
                    );
                }
                self.wake_main_thread();
            }
            Err(runnable) => {
                // NOTE: Runnable may wrap a Future that is !Send.
                //
                // This is usually safe because we only poll it on the main thread.
                // However if the send fails, we know that:
                // 1. main_receiver has been dropped (which implies the app is shutting down)
                // 2. we are on a background thread.
                // It is not safe to drop something !Send on the wrong thread, and
                // the app will exit soon anyway, so we must forget the runnable.
                std::mem::forget(runnable);
            }
        }
    }

    fn dispatch_after(&self, duration: Duration, runnable: RunnableVariant) {
        self.dispatch_on_threadpool_after(runnable, duration);
    }

    fn spawn_realtime(&self, f: Box<dyn FnOnce() + Send>) {
        std::thread::spawn(move || {
            // SAFETY: always safe to call
            let thread_handle = unsafe { GetCurrentThread() };

            // SAFETY: thread_handle is a valid handle to the current thread
            unsafe { SetThreadPriority(thread_handle, THREAD_PRIORITY_TIME_CRITICAL) }
                .context("thread priority")
                .log_err();

            f();
        });
    }

    fn increase_timer_resolution(&self) -> TimerResolutionGuard {
        unsafe {
            timeBeginPeriod(1);
        }
        gpui_util::defer(Box::new(|| unsafe {
            timeEndPeriod(1);
        }))
    }
}

unsafe extern "system" fn run_work_callback(
    _instance: PTP_CALLBACK_INSTANCE,
    context: *mut c_void,
) {
    let runnable = unsafe { RunnableVariant::from_raw(NonNull::new_unchecked(context as *mut ())) };
    WindowsDispatcher::execute_runnable(runnable);
}

unsafe extern "system" fn run_timer_callback(
    _instance: PTP_CALLBACK_INSTANCE,
    context: *mut c_void,
    timer: PTP_TIMER,
) {
    let runnable = unsafe { RunnableVariant::from_raw(NonNull::new_unchecked(context as *mut ())) };
    WindowsDispatcher::execute_runnable(runnable);
    unsafe { CloseThreadpoolTimer(timer) };
}
