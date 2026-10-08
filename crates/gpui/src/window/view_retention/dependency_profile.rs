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
    if matches!(case, "nested" | "nested-empty" | "nested-deadline") {
        let inner = cx.begin_recording_dependencies();
        if case == "nested" {
            note_access(&cx.entities, EntityId::from(2));
            note_global_read(cx, TypeId::of::<u64>());
        }
        if case == "nested-deadline" {
            note_deadline(deadline);
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
    assert!(["empty", "entity", "entities-16", "global", "state", "all", "nested", "nested-empty", "nested-deadline", "deadline"].contains(&case.as_str()));
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
            "deadline" | "nested-deadline" => [0, 0, 0, 0, 0, 0, 1],
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

fn change_code(change: Option<DependencyChange>) -> usize {
    match change {
        None => 0,
        Some(DependencyChange::Entity) => 1,
        Some(DependencyChange::Global) => 2,
        Some(DependencyChange::State) => 3,
        Some(DependencyChange::Deadline) => 4,
    }
}

#[test]
#[ignore]
fn profile_state_checks() {
    let case = std::env::var("GPUI_STATE_PROFILE_CASE").expect("profile case");
    let iterations = std::env::var("GPUI_STATE_PROFILE_ITERATIONS")
        .expect("profile iterations").parse::<usize>().expect("iteration count");
    let count = match case.as_str() {
        "normal-empty" | "except-empty" => 0,
        "except-only" | "normal-1" | "except-deadline" | "except-global" | "except-entity" => 1,
        "except-first-8" | "except-last-8" | "except-changed-8" | "except-absent-8"
        | "except-absent-changed-8" | "except-unchanged-8" | "normal-8" => 8,
        "except-first-64" | "except-last-64" | "except-unchanged-64"
        | "except-absent-64" | "normal-changed-64" | "normal-64" => 64,
        _ => panic!("unknown state profile case"),
    };
    let expected = match case.as_str() {
        "except-deadline" => 4,
        "except-global" => 2,
        "except-entity" => 1,
        "except-changed-8" | "except-absent-changed-8" | "normal-changed-64" => 3,
        _ => 0,
    };
    let cx = crate::TestAppContext::single();
    cx.update(|cx| {
        cx.set_view_retention(true);
        let marker = StateVersion::default();
        let normal = case.starts_with("normal-");
        let absent = case.starts_with("except-absent-");
        let states: Vec<_> = (0..count).map(|index| {
            if !normal && !absent
                && index == if case.contains("-last-") { count - 1 } else { 0 }
            {
                marker.clone()
            } else {
                StateVersion::default()
            }
        }).collect();
        let now = Instant::now();
        let entity = EntityId::from(1);
        let global = TypeId::of::<u32>();
        let recording = cx.begin_recording_dependencies();
        for state in &states {
            note_state_read(state);
        }
        if case == "except-global" {
            note_global_read(cx, global);
        }
        if case == "except-entity" {
            note_access(&cx.entities, entity);
        }
        if case == "except-deadline" {
            note_deadline(now);
        }
        let dependencies = cx.finish_recording_dependencies(recording).all;
        if !case.starts_with("except-unchanged-") {
            marker.bump();
        }
        if matches!(case.as_str(), "except-changed-8" | "except-absent-changed-8" | "normal-changed-64") {
            states[count - 1].bump();
        }
        if case == "except-global" {
            cx.dependencies.global_changed(global);
        }
        if case == "except-entity" {
            cx.entities.access_log.stamp_changed(entity);
        }
        let check = |cx: &App| {
            let dependencies = black_box(&dependencies);
            let cx = black_box(cx);
            if normal {
                cx.dependencies_changed(dependencies, false, black_box(now))
            } else {
                cx.dependencies_changed_except(dependencies, false, black_box(now), black_box(&marker))
            }
        };
        for _ in 0..32 {
            assert_eq!(change_code(black_box(check(cx))), expected);
        }
        #[cfg(gpui_dependency_census)]
        census::start();
        let cpu = thread_cpu_ns();
        let elapsed = Instant::now();
        let mut signature = 0;
        for _ in 0..iterations {
            signature += change_code(black_box(check(cx)));
        }
        let elapsed_ns = elapsed.elapsed().as_nanos();
        let cpu_ns = thread_cpu_ns() - cpu;
        #[cfg(gpui_dependency_census)]
        let memory = census::finish();
        #[cfg(not(gpui_dependency_census))]
        let memory = (0, 0, 0, 0);
        assert_eq!(signature, iterations * expected);
        println!("STATE_PROFILE {{\"case\":{case:?},\"iterations\":{iterations},\"cpu_ns\":{cpu_ns},\"elapsed_ns\":{elapsed_ns},\"held_bytes\":{},\"peak_bytes\":{},\"requested_bytes\":{},\"allocation_calls\":{},\"signature\":{signature}}}", memory.0, memory.1, memory.2, memory.3);
    });
}
