//! Tells macOS's performance controller when the main thread is drawing a
//! frame and by when it has to be done.
//!
//! Apple silicon picks a thread's cores and clock from how busy its thread
//! group has been, and a window that draws a few milliseconds per refresh
//! looks idle to it: when nothing else in the app is busy, the main thread's
//! frames run on the slower cores at well under half the clock. A Core
//! Animation client work interval is how AppKit and UIKit describe their own
//! frames to the controller. Started with the frame's deadline before its work
//! and finished after, it has the controller run the frame fast enough to meet
//! the deadline: a frame due at the next refresh that needs more than a sliver
//! of it runs on the fastest cores at full clock.
//!
//! That costs energy, so only frames that answer the user are reported: those
//! drawn while the window's input, or an animation the app marks as the
//! user's, is recent (`Window::mark_frames_interactive`). Redraws for
//! anything else, such as streamed text or background work landing, keep the
//! controller's default scheduling, and an idle window reports nothing.
//!
//! Neither of the obvious levers does this. The main thread is already
//! user-interactive, which only makes it eligible for the fastest cores. An
//! `os_workgroup` interval moves the thread into a thread group of its own,
//! which the controller keeps on the slower cores whatever deadlines it
//! misses. Reporting a frame only after it finished (`work_interval_notify`)
//! moved nothing either; reporting its start with the deadline did.
//!
//! The calls are libsystem's work interval interface, which the SDK does not
//! declare, so they are looked up at run time. Their absence, or a refusal
//! (the kernel allows a task one Core Animation client interval, which AppKit
//! would otherwise take), leaves frames unreported, as before.

use std::{
    cell::{Cell, OnceCell},
    ffi::{CStr, c_void},
    io,
    marker::PhantomData,
    mem,
    ptr::{self, NonNull},
};

const WORK_INTERVAL_FLAG_JOINABLE: u32 = 0x1;
const WORK_INTERVAL_TYPE_CA_CLIENT: u32 = 0x3 << 28;
const FLAGS: u32 = WORK_INTERVAL_FLAG_JOINABLE | WORK_INTERVAL_TYPE_CA_CLIENT;

fn last_error() -> io::Error {
    io::Error::last_os_error()
}

type CreateFn = unsafe extern "C" fn(*mut *mut c_void, u32) -> i32;
type HandleFn = unsafe extern "C" fn(*mut c_void) -> i32;
type InstanceAllocFn = unsafe extern "C" fn(*mut c_void) -> *mut c_void;
type InstanceClearFn = unsafe extern "C" fn(*mut c_void);
type InstanceSetFn = unsafe extern "C" fn(*mut c_void, u64);
type InstanceReportFn = unsafe extern "C" fn(*mut c_void) -> i32;

struct Calls {
    create: CreateFn,
    join: HandleFn,
    destroy: HandleFn,
    instance_alloc: InstanceAllocFn,
    instance_clear: InstanceClearFn,
    instance_set_start: InstanceSetFn,
    instance_set_deadline: InstanceSetFn,
    instance_set_finish: InstanceSetFn,
    instance_start: InstanceReportFn,
    instance_finish: InstanceReportFn,
}

/// Looks up a libsystem function by name.
///
/// # Safety
///
/// `F` must be the function pointer type of `name`'s C signature.
unsafe fn lookup<F: Copy>(name: &CStr) -> Option<F> {
    // SAFETY: `name` is NUL-terminated; RTLD_DEFAULT searches every loaded image.
    let symbol = unsafe { libc::dlsym(libc::RTLD_DEFAULT, name.as_ptr()) };
    // SAFETY: a non-null symbol is the function the caller named `F` for, and
    // a function pointer has the size of a data pointer on Apple platforms.
    (!symbol.is_null()).then(|| unsafe { mem::transmute_copy::<*mut c_void, F>(&symbol) })
}

impl Calls {
    fn resolve() -> Option<Self> {
        // SAFETY: each type is the C signature of the function it is looked up for.
        unsafe {
            Some(Self {
                create: lookup(c"work_interval_create")?,
                join: lookup(c"work_interval_join")?,
                destroy: lookup(c"work_interval_destroy")?,
                instance_alloc: lookup(c"work_interval_instance_alloc")?,
                instance_clear: lookup(c"work_interval_instance_clear")?,
                instance_set_start: lookup(c"work_interval_instance_set_start")?,
                instance_set_deadline: lookup(c"work_interval_instance_set_deadline")?,
                instance_set_finish: lookup(c"work_interval_instance_set_finish")?,
                instance_start: lookup(c"work_interval_instance_start")?,
                instance_finish: lookup(c"work_interval_instance_finish")?,
            })
        }
    }
}

/// The main thread's Core Animation client work interval, joined for the
/// life of the process: the kernel releases it when the task exits.
struct FrameWorkInterval {
    calls: Calls,
    instance: NonNull<c_void>,
}

impl FrameWorkInterval {
    fn create() -> Option<Self> {
        let Some(calls) = Calls::resolve() else {
            log::debug!("frame work interval: libsystem has no work interval instances");
            return None;
        };
        let mut handle = ptr::null_mut();
        // SAFETY: `handle` is a valid out pointer.
        let created = unsafe { (calls.create)(&mut handle, FLAGS) };
        if created != 0 || handle.is_null() {
            log::info!("frames go unreported: no frame work interval ({})", last_error());
            return None;
        }
        // SAFETY: `handle` is the work interval just created.
        if unsafe { (calls.join)(handle) } != 0 {
            log::info!("frames go unreported: joining their interval failed ({})", last_error());
            // SAFETY: `handle` was created above and nothing else holds it.
            if unsafe { (calls.destroy)(handle) } != 0 {
                log::debug!("destroying the frame work interval failed: {}", last_error());
            }
            return None;
        }
        // SAFETY: `handle` is a live, joined work interval.
        let instance = NonNull::new(unsafe { (calls.instance_alloc)(handle) })?;
        Some(Self { calls, instance })
    }
}

thread_local! {
    static FRAME_WORK_INTERVAL: OnceCell<Option<FrameWorkInterval>> = const { OnceCell::new() };
    static FRAME_IN_PROGRESS: Cell<bool> = const { Cell::new(false) };
}

/// A frame reported as started; dropping it reports the frame finished.
pub(crate) struct FrameWork {
    // The interval belongs to the thread that started the frame.
    _not_send: PhantomData<*const ()>,
}

/// Reports that the calling (main) thread starts drawing a frame that has to
/// be done by `deadline`, a host time. `None` while another frame is being
/// reported on this thread, or when the system takes no reports.
pub(crate) fn frame_starts(deadline: u64) -> Option<FrameWork> {
    if FRAME_IN_PROGRESS.get() {
        return None;
    }
    FRAME_WORK_INTERVAL.with(|cell| {
        let interval = cell.get_or_init(FrameWorkInterval::create).as_ref()?;
        let instance = interval.instance.as_ptr();
        let start = objc2_core_video::CVGetCurrentHostTime();
        // SAFETY: `instance` belongs to the interval this thread joined.
        let started = unsafe {
            (interval.calls.instance_clear)(instance);
            (interval.calls.instance_set_start)(instance, start);
            (interval.calls.instance_set_deadline)(instance, deadline);
            (interval.calls.instance_start)(instance)
        };
        if started != 0 {
            log::debug!("frame work interval start refused: {}", last_error());
            return None;
        }
        FRAME_IN_PROGRESS.set(true);
        Some(FrameWork { _not_send: PhantomData })
    })
}

impl Drop for FrameWork {
    fn drop(&mut self) {
        FRAME_IN_PROGRESS.set(false);
        FRAME_WORK_INTERVAL.with(|cell| {
            let Some(Some(interval)) = cell.get() else {
                return;
            };
            let instance = interval.instance.as_ptr();
            let finish = objc2_core_video::CVGetCurrentHostTime();
            // SAFETY: `instance` was started by `frame_starts` on this thread.
            let finished = unsafe {
                (interval.calls.instance_set_finish)(instance, finish);
                (interval.calls.instance_finish)(instance)
            };
            if finished != 0 {
                log::debug!("frame work interval finish refused: {}", last_error());
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn libsystem_has_every_work_interval_call() {
        assert!(Calls::resolve().is_some());
    }

    #[test]
    fn frames_are_reported_one_at_a_time() {
        let deadline = objc2_core_video::CVGetCurrentHostTime() + 1_000_000;
        let Some(frame) = frame_starts(deadline) else {
            // A host that refuses the interval reports nothing, which is the fallback.
            return;
        };
        assert!(frame_starts(deadline).is_none(), "a nested frame is not reported");
        drop(frame);
        let next = frame_starts(deadline);
        assert!(next.is_some(), "the next frame is reported once the first finished");
    }
}
