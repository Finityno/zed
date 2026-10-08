use super::*;
use std::hint::black_box;

#[cfg(gpui_dependency_census)]
mod census {
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::sync::atomic::{AtomicBool, AtomicIsize, AtomicUsize, Ordering::Relaxed};

    static ENABLED: AtomicBool = AtomicBool::new(false);
    static LIVE: AtomicIsize = AtomicIsize::new(0);
    static PEAK: AtomicIsize = AtomicIsize::new(0);
    static BYTES: AtomicUsize = AtomicUsize::new(0);
    static CALLS: AtomicUsize = AtomicUsize::new(0);

    struct Counter;

    fn allocated(bytes: usize) {
        BYTES.fetch_add(bytes, Relaxed);
        CALLS.fetch_add(1, Relaxed);
        let live = LIVE.fetch_add(bytes as isize, Relaxed) + bytes as isize;
        PEAK.fetch_max(live, Relaxed);
    }

    unsafe impl GlobalAlloc for Counter {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            let pointer = unsafe { System.alloc(layout) };
            if !pointer.is_null() && ENABLED.load(Relaxed) {
                allocated(layout.size());
            }
            pointer
        }

        unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
            let pointer = unsafe { System.alloc_zeroed(layout) };
            if !pointer.is_null() && ENABLED.load(Relaxed) {
                allocated(layout.size());
            }
            pointer
        }

        unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
            if ENABLED.load(Relaxed) {
                LIVE.fetch_sub(layout.size() as isize, Relaxed);
            }
            unsafe { System.dealloc(pointer, layout) };
        }

        unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, bytes: usize) -> *mut u8 {
            let resized = unsafe { System.realloc(pointer, layout, bytes) };
            if !resized.is_null() && ENABLED.load(Relaxed) {
                LIVE.fetch_sub(layout.size() as isize, Relaxed);
                allocated(bytes);
            }
            resized
        }
    }

    #[global_allocator]
    static ALLOCATOR: Counter = Counter;

    pub(super) fn start() {
        LIVE.store(0, Relaxed);
        PEAK.store(0, Relaxed);
        BYTES.store(0, Relaxed);
        CALLS.store(0, Relaxed);
        ENABLED.store(true, Relaxed);
    }

    pub(super) fn finish() -> (isize, isize, usize, usize) {
        ENABLED.store(false, Relaxed);
        (LIVE.load(Relaxed), PEAK.load(Relaxed), BYTES.load(Relaxed), CALLS.load(Relaxed))
    }
}

#[repr(C)]
struct Timespec {
    seconds: std::os::raw::c_long,
    nanoseconds: std::os::raw::c_long,
}

unsafe extern "C" {
    fn clock_gettime(clock: std::os::raw::c_int, result: *mut Timespec) -> std::os::raw::c_int;
}

fn thread_cpu_ns() -> u64 {
    let mut time = Timespec { seconds: 0, nanoseconds: 0 };
    assert_eq!(unsafe { clock_gettime(3, &mut time) }, 0);
    time.seconds as u64 * 1_000_000_000 + time.nanoseconds as u64
}

fn record(cx: &mut App, case: &str, state: &StateVersion, deadline: Instant) -> Recorded {
    let outer = cx.begin_recording_dependencies();
    if matches!(case, "entity" | "all" | "nested") {
        note_access(&cx.entities, EntityId::from(1));
    }
    if case == "entities-16" {
        for entity in 1..=16 {
            note_access(&cx.entities, EntityId::from(entity));
        }
    }
    if matches!(case, "global" | "all" | "nested") {
        note_global_read(cx, TypeId::of::<u32>());
    }
    if matches!(case, "state" | "all" | "nested") {
        note_state_read(state);
    }
    if case == "deadline" {
        note_deadline(deadline);
    }
    if matches!(case, "nested" | "nested-empty") {
        let inner = cx.begin_recording_dependencies();
        if case == "nested" {
            note_access(&cx.entities, EntityId::from(2));
            note_global_read(cx, TypeId::of::<u64>());
        }
        black_box(cx.finish_recording_dependencies(inner));
    }
    cx.finish_recording_dependencies(outer)
}

/// Measures the complete recorder, including log writes, nested recordings,
/// output ownership and release. Census builds are separate from bare timings.
#[test]
#[ignore]
fn profile_dependency_records() {
    let case = std::env::var("GPUI_DEPENDENCY_PROFILE_CASE").expect("profile case");
    assert!(["empty", "entity", "entities-16", "global", "state", "all", "nested", "nested-empty", "deadline"].contains(&case.as_str()));
    let iterations = std::env::var("GPUI_DEPENDENCY_PROFILE_ITERATIONS")
        .expect("profile iterations").parse::<usize>().expect("iteration count");
    let cx = crate::TestAppContext::single();
    let state = StateVersion::default();
    let deadline = Instant::now() + std::time::Duration::from_secs(3600);
    cx.update(|cx| {
        cx.set_view_retention(true);
        for _ in 0..32 {
            black_box(record(cx, &case, &state, deadline));
        }
        let mut held = Vec::with_capacity(1024);
        let example = record(cx, &case, &state, deadline);
        let signature = [example.all.entities.len(), example.all.globals.len(), example.all.states.len(), example.own.entities.len(), example.own.globals.len(), example.own.states.len(), usize::from(example.all.rebuild_at.is_some())];
        let expected = match case.as_str() {
            "entity" => [1, 0, 0, 1, 0, 0, 0],
            "entities-16" => [16, 0, 0, 16, 0, 0, 0],
            "global" => [0, 1, 0, 0, 1, 0, 0],
            "state" => [0, 0, 1, 0, 0, 1, 0],
            "all" => [1, 1, 1, 1, 1, 1, 0],
            "nested" => [2, 2, 1, 1, 1, 1, 0],
            "deadline" => [0, 0, 0, 0, 0, 0, 1],
            _ => [0; 7],
        };
        assert_eq!(signature, expected);
        drop(example);
        #[cfg(gpui_dependency_census)]
        census::start();
        let cpu = thread_cpu_ns();
        let elapsed = Instant::now();
        for index in 0..iterations {
            held.push(record(cx, &case, &state, deadline));
            if index % 1024 == 1023 {
                black_box(&held);
                held.clear();
            }
        }
        held.clear();
        for _ in 0..1024 {
            held.push(record(cx, &case, &state, deadline));
        }
        let elapsed_ns = elapsed.elapsed().as_nanos();
        let cpu_ns = thread_cpu_ns() - cpu;
        #[cfg(gpui_dependency_census)]
        let memory = census::finish();
        #[cfg(not(gpui_dependency_census))]
        let memory = (0, 0, 0, 0);
        println!("DEPENDENCY_PROFILE {{\"case\":{case:?},\"iterations\":{iterations},\"cpu_ns\":{cpu_ns},\"elapsed_ns\":{elapsed_ns},\"held_bytes\":{},\"peak_bytes\":{},\"requested_bytes\":{},\"allocation_calls\":{},\"signature\":{:?}}}", memory.0, memory.1, memory.2, memory.3, signature);
        black_box(held);
    });
}
