use dispatch2::{DispatchQueue, DispatchQueueGlobalPriority, DispatchTime, GlobalQueueIdentifier};
use gpui::{
    ActivityGuard, FaultScope, PlatformDispatcher, Priority, ResourceSample, RunnableMeta,
    RunnableVariant,
};
use gpui_util::ResultExt;
use mach2::{
    kern_return::KERN_SUCCESS,
    mach_time::mach_timebase_info_data_t,
    thread_policy::{
        THREAD_EXTENDED_POLICY, THREAD_EXTENDED_POLICY_COUNT, THREAD_PRECEDENCE_POLICY,
        THREAD_PRECEDENCE_POLICY_COUNT, THREAD_TIME_CONSTRAINT_POLICY,
        THREAD_TIME_CONSTRAINT_POLICY_COUNT, thread_extended_policy_data_t,
        thread_precedence_policy_data_t, thread_time_constraint_policy_data_t,
    },
};

use async_task::Runnable;
use objc::{
    class, msg_send,
    runtime::{BOOL, YES},
    sel, sel_impl,
};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::{NSActivityOptions, NSObjectProtocol, NSProcessInfo, NSString};
use std::{ffi::c_void, ptr::NonNull, time::Duration};

pub(crate) struct MacDispatcher;

impl MacDispatcher {
    pub fn new() -> Self {
        Self
    }
}

impl PlatformDispatcher for MacDispatcher {
    fn is_main_thread(&self) -> bool {
        let is_main_thread: BOOL = unsafe { msg_send![class!(NSThread), isMainThread] };
        is_main_thread == YES
    }

    fn dispatch(&self, runnable: RunnableVariant, priority: Priority) {
        let context = runnable.into_raw().as_ptr() as *mut c_void;

        let queue_priority = match priority {
            Priority::RealtimeAudio => {
                panic!("RealtimeAudio priority should use spawn_realtime, not dispatch")
            }
            Priority::High => DispatchQueueGlobalPriority::High,
            Priority::Medium => DispatchQueueGlobalPriority::Default,
            Priority::Low => DispatchQueueGlobalPriority::Low,
        };

        unsafe {
            DispatchQueue::global_queue(GlobalQueueIdentifier::Priority(queue_priority))
                .exec_async_f(context, trampoline);
        }
    }

    fn dispatch_on_main_thread(&self, runnable: RunnableVariant, _priority: Priority) {
        let context = runnable.into_raw().as_ptr() as *mut c_void;
        unsafe {
            DispatchQueue::main().exec_async_f(context, main_thread_trampoline);
        }
    }

    fn dispatch_after(&self, duration: Duration, runnable: RunnableVariant) {
        let context = runnable.into_raw().as_ptr() as *mut c_void;
        let queue = DispatchQueue::global_queue(GlobalQueueIdentifier::Priority(
            DispatchQueueGlobalPriority::High,
        ));
        let when = DispatchTime::NOW.time(duration.as_nanos() as i64);
        unsafe {
            DispatchQueue::exec_after_f(when, &queue, context, trampoline);
        }
    }

    fn spawn_realtime(&self, f: Box<dyn FnOnce() + Send>) {
        std::thread::spawn(move || {
            set_audio_thread_priority().log_err();
            f();
        });
    }

    fn prevent_app_nap(&self, reason: &str) -> ActivityGuard {
        MacActivity::begin(
            reason,
            NSActivityOptions::UserInitiatedAllowingIdleSystemSleep,
        )
    }

    fn sample_draw_resources(&self, process_counters: bool) -> Option<ResourceSample> {
        sample_resources(process_counters)
    }
}

thread_local! {
    // `pthread_mach_thread_np` returns the thread's port without taking a
    // new send right, unlike `mach_thread_self`, which leaks one per call
    // unless it is deallocated.
    static CURRENT_THREAD_PORT: libc::mach_port_t =
        // SAFETY: always safe to call with the current thread.
        unsafe { libc::pthread_mach_thread_np(libc::pthread_self()) };
}

/// Reads the calling thread's CPU time (`thread_info`, about 0.5 µs) and,
/// when `process_counters` is set, the task's fault, page-in and
/// decompression counters (two `task_info` calls). Measured in a test
/// process on Apple silicon, the two calls together cost about 1.7 µs, and
/// about 2.9 µs with 150 more parked threads, since `TASK_EVENTS_INFO`
/// walks the task's threads. Their cost in a large app, whose memory map
/// `TASK_VM_INFO` may also walk, has not been measured.
///
/// The fault counters are the whole task's: macOS has no per-thread fault
/// count. Decompressing a compressed page runs in the faulting thread, so
/// it shows up in that thread's system time.
fn sample_resources(process_counters: bool) -> Option<ResourceSample> {
    let (user, system) = thread_cpu_times()?;
    let mut sample = ResourceSample {
        user,
        system,
        faults: None,
        major_faults: None,
        decompressions: None,
        fault_scope: FaultScope::Process,
    };
    if process_counters {
        if let Some(events) = task_events() {
            sample.faults = unsaturated(events.faults);
            sample.major_faults = unsaturated(events.pageins);
        }
        sample.decompressions = task_decompressions();
    }
    Some(sample)
}

/// `TASK_EVENTS_INFO`'s counters are 32-bit, and the kernel pins them at
/// `i32::MAX` once they would overflow, after which a difference between
/// two readings is zero rather than the true count.
fn unsaturated(count: i32) -> Option<u64> {
    (count < i32::MAX).then(|| count.max(0) as u64)
}

fn thread_cpu_times() -> Option<(Duration, Duration)> {
    let port = CURRENT_THREAD_PORT.with(|port| *port);
    let mut info = std::mem::MaybeUninit::<libc::thread_basic_info>::zeroed();
    let mut count = libc::THREAD_BASIC_INFO_COUNT;
    // SAFETY: `info` has room for THREAD_BASIC_INFO_COUNT integers, and
    // `port` names the calling thread, which outlives the call.
    let result = unsafe {
        libc::thread_info(
            port,
            libc::THREAD_BASIC_INFO as libc::thread_flavor_t,
            info.as_mut_ptr() as libc::thread_info_t,
            &mut count,
        )
    };
    if result != KERN_SUCCESS {
        return None;
    }
    // SAFETY: thread_info succeeded, so it filled `info`.
    let info = unsafe { info.assume_init() };
    let duration = |time: libc::time_value_t| {
        Duration::from_secs(time.seconds.max(0) as u64)
            + Duration::from_micros(time.microseconds.max(0) as u64)
    };
    Some((duration(info.user_time), duration(info.system_time)))
}

fn task_events() -> Option<mach2::task_info::task_events_info> {
    let mut info = mach2::task_info::task_events_info::default();
    let mut count = mach2::task_info::TASK_EVENTS_INFO_COUNT;
    // SAFETY: `info` has room for TASK_EVENTS_INFO_COUNT integers.
    let result = unsafe {
        mach2::task::task_info(
            mach2::traps::mach_task_self(),
            mach2::task_info::TASK_EVENTS_INFO,
            &mut info as *mut _ as mach2::task_info::task_info_t,
            &mut count,
        )
    };
    (result == KERN_SUCCESS).then_some(info)
}

fn task_decompressions() -> Option<u64> {
    use mach2::task_info::{TASK_VM_INFO, task_info_t, task_vm_info};
    // `decompressions` arrived in revision 5 of the structure: a kernel
    // that fills fewer integers than reach past it did not write it.
    const REVISION_5_COUNT: u32 = ((std::mem::offset_of!(task_vm_info, decompressions)
        + std::mem::size_of::<i32>())
        / std::mem::size_of::<u32>()) as u32;
    let mut info = task_vm_info::default();
    let mut count = (std::mem::size_of::<task_vm_info>() / std::mem::size_of::<u32>()) as u32;
    // SAFETY: `info` has room for `count` integers.
    let result = unsafe {
        mach2::task::task_info(
            mach2::traps::mach_task_self(),
            TASK_VM_INFO,
            &mut info as *mut _ as task_info_t,
            &mut count,
        )
    };
    (result == KERN_SUCCESS && count >= REVISION_5_COUNT)
        .then(|| info.decompressions.max(0) as u64)
}

pub(crate) struct MacActivity {
    activity: Retained<ProtocolObject<dyn NSObjectProtocol>>,
}

// The activity token returned by NSProcessInfo is thread-safe
unsafe impl Send for MacActivity {}

impl MacActivity {
    pub(crate) fn begin(reason: &str, options: NSActivityOptions) -> ActivityGuard {
        let activity = Self {
            activity: NSProcessInfo::processInfo()
                .beginActivityWithOptions_reason(options, &NSString::from_str(reason)),
        };
        ActivityGuard::new(move || drop(activity))
    }
}

impl Drop for MacActivity {
    fn drop(&mut self) {
        unsafe { NSProcessInfo::processInfo().endActivity(&self.activity) };
    }
}

fn set_audio_thread_priority() -> anyhow::Result<()> {
    // https://chromium.googlesource.com/chromium/chromium/+/master/base/threading/platform_thread_mac.mm#93

    // SAFETY: always safe to call
    let thread_id = unsafe { libc::pthread_self() };

    // SAFETY: thread_id is a valid thread id
    let thread_id = unsafe { libc::pthread_mach_thread_np(thread_id) };

    // Fixed priority thread
    let mut policy = thread_extended_policy_data_t { timeshare: 0 };

    // SAFETY: thread_id is a valid thread id
    // SAFETY: thread_extended_policy_data_t is passed as THREAD_EXTENDED_POLICY
    let result = unsafe {
        mach2::thread_policy::thread_policy_set(
            thread_id,
            THREAD_EXTENDED_POLICY,
            &mut policy as *mut _ as *mut _,
            THREAD_EXTENDED_POLICY_COUNT,
        )
    };

    if result != KERN_SUCCESS {
        anyhow::bail!("failed to set thread extended policy");
    }

    // relatively high priority
    let mut precedence = thread_precedence_policy_data_t { importance: 63 };

    // SAFETY: thread_id is a valid thread id
    // SAFETY: thread_precedence_policy_data_t is passed as THREAD_PRECEDENCE_POLICY
    let result = unsafe {
        mach2::thread_policy::thread_policy_set(
            thread_id,
            THREAD_PRECEDENCE_POLICY,
            &mut precedence as *mut _ as *mut _,
            THREAD_PRECEDENCE_POLICY_COUNT,
        )
    };

    if result != KERN_SUCCESS {
        anyhow::bail!("failed to set thread precedence policy");
    }

    const GUARANTEED_AUDIO_DUTY_CYCLE: f32 = 0.75;
    const MAX_AUDIO_DUTY_CYCLE: f32 = 0.85;

    // ~128 frames @ 44.1KHz
    const TIME_QUANTUM: f32 = 2.9;

    const AUDIO_TIME_NEEDED: f32 = GUARANTEED_AUDIO_DUTY_CYCLE * TIME_QUANTUM;
    const MAX_TIME_ALLOWED: f32 = MAX_AUDIO_DUTY_CYCLE * TIME_QUANTUM;

    let mut timebase_info = mach_timebase_info_data_t { numer: 0, denom: 0 };
    // SAFETY: timebase_info is a valid pointer to a mach_timebase_info_data_t struct
    unsafe { mach2::mach_time::mach_timebase_info(&mut timebase_info) };

    let ms_to_abs_time = ((timebase_info.denom as f32) / (timebase_info.numer as f32)) * 1000000f32;

    let mut time_constraints = thread_time_constraint_policy_data_t {
        period: (TIME_QUANTUM * ms_to_abs_time) as u32,
        computation: (AUDIO_TIME_NEEDED * ms_to_abs_time) as u32,
        constraint: (MAX_TIME_ALLOWED * ms_to_abs_time) as u32,
        preemptible: 0,
    };

    // SAFETY: thread_id is a valid thread id
    // SAFETY: thread_precedence_pthread_time_constraint_policy_data_t is passed as THREAD_TIME_CONSTRAINT_POLICY
    let result = unsafe {
        mach2::thread_policy::thread_policy_set(
            thread_id,
            THREAD_TIME_CONSTRAINT_POLICY,
            &mut time_constraints as *mut _ as *mut _,
            THREAD_TIME_CONSTRAINT_POLICY_COUNT,
        )
    };

    if result != KERN_SUCCESS {
        anyhow::bail!("failed to set thread time constraint policy");
    }

    Ok(())
}

fn run_runnable(context: *mut c_void) {
    let runnable =
        unsafe { Runnable::<RunnableMeta>::from_raw(NonNull::new_unchecked(context as *mut ())) };

    let location = runnable.metadata().location;
    let spawned = runnable.metadata().spawned;
    gpui::profiler::update_running_task(spawned, location);
    runnable.run();
    gpui::profiler::save_task_timing();
}

/// The trampoline for the global (background) queues.
extern "C" fn trampoline(context: *mut c_void) {
    run_runnable(context);

    // macOS dispatches onto GCD, so GPUI owns no background worker threads and
    // there is no park of ours to announce from -- `queue.rs` is unreachable on
    // this backend. A GCD worker instead sits idle between tasks, and it is
    // long-lived: the pool reuses threads rather than retiring them, so any
    // per-thread state an embedder keeps accumulates for the life of the
    // process unless it hears about these moments.
    //
    // Just-finished-a-task is that moment. It fires far more often than a park
    // would, so the hook is documented as needing to be cheap and
    // self-rate-limiting; that is a better trade than leaving every background
    // thread on this platform unannounced.
    gpui::thread_idle();
}

/// The trampoline for the main queue. Deliberately does NOT announce an idle
/// point: the main thread has a real one -- the `kCFRunLoopBeforeWaiting`
/// observer in `platform.rs`, which fires once the run loop has drained
/// everything and is about to sleep. Announcing after every dispatched task
/// instead would fire mid-frame, between two tasks of the same burst, and a
/// hook that paces itself would then spend its budget there and skip the
/// genuine idle point that follows.
extern "C" fn main_thread_trampoline(context: *mut c_void) {
    run_runnable(context);
}

#[cfg(test)]
mod tests {
    use super::sample_resources;

    #[test]
    fn touching_fresh_memory_shows_up_in_the_samples() {
        let before = sample_resources(true).expect("thread_info succeeds");
        const SIZE: usize = 64 * 1024 * 1024;
        let mut memory = vec![0u8; SIZE];
        for page in memory.chunks_mut(4096) {
            page[0] = 1;
        }
        std::hint::black_box(&memory);
        let after = sample_resources(true).expect("thread_info succeeds");

        let faults = after.faults.expect("TASK_EVENTS_INFO succeeds")
            - before.faults.expect("TASK_EVENTS_INFO succeeds");
        // SAFETY: sysconf has no preconditions.
        let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) }.max(4096) as u64;
        let pages = SIZE as u64 / page_size;
        // Half the pages leaves room for the kernel faulting several pages
        // in at once.
        assert!(
            faults >= pages / 2,
            "touching 64 MiB of fresh pages faults them in: {faults} faults for {pages} pages"
        );
        assert!(
            after.user + after.system > before.user + before.system,
            "and costs the thread CPU time: {before:?} -> {after:?}"
        );
        // Older kernels fill an earlier revision of `task_vm_info`, without
        // the counter.
        if crate::window::is_macos_version_at_least(
            cocoa::foundation::NSOperatingSystemVersion::new(12, 0, 0),
        ) {
            assert!(
                after.decompressions.is_some(),
                "macOS 12 and later report decompressions"
            );
        }
        assert!(after.major_faults.is_some());
    }

    #[test]
    fn thread_cpu_time_alone_reads_no_process_counters() {
        let sample = sample_resources(false).expect("thread_info succeeds");
        assert_eq!(
            (sample.faults, sample.major_faults, sample.decompressions),
            (None, None, None)
        );
    }
}
