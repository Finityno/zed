use super::*;
use std::{hash::{Hash, Hasher}, hint::black_box};

#[cfg(gpui_splice_census)]
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

struct RemovalRow {
    present: bool,
    identity: u64,
    height: f32,
}

impl Render for RemovalRow {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        div().h(px(self.height)).when(self.present, |this| {
            this.child(
                div().id(("profile-state", self.identity)).size(px(10.)).bg(hsla(0.4, 0.5, 0.5, 1.)),
            )
        })
    }
}

struct RemovalPanel(Vec<Entity<RemovalRow>>);

impl Render for RemovalPanel {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        div().size_full().flex().flex_col().children(self.0.iter().cloned())
    }
}

struct RemovalShell(Entity<RemovalPanel>);

impl Render for RemovalShell {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        div().size_full().child(self.0.clone())
    }
}

#[test]
#[ignore]
fn profile_gap_state_removal() {
    let case = std::env::var("SPLICE_PROFILE_CASE").expect("SPLICE_PROFILE_CASE");
    let iterations: usize = std::env::var("SPLICE_PROFILE_ITERATIONS")
        .expect("SPLICE_PROFILE_ITERATIONS").parse().expect("iterations");
    assert!(iterations > 0 && iterations.is_multiple_of(2));
    let rows = if case.ends_with("256") { 256 } else { 32 };
    let retained = !case.starts_with("off-");
    #[cfg(gpui_splice_census)]
    census::start();
    let mut cx = TestAppContext::single();
    cx.update(|cx| cx.set_view_retention(retained));
    let leaves: Vec<_> = (0..rows)
        .map(|_| cx.new(|_| RemovalRow { present: true, identity: 0, height: 10. }))
        .collect();
    let window = cx.add_window(|_, cx| {
        RemovalShell(cx.new(|_| RemovalPanel(leaves.clone())))
    });
    let frame = |cx: &mut TestAppContext| {
        cx.update_window(window.into(), |_, window, cx| window.draw(cx).clear(cx)).unwrap();
    };
    for _ in 0..4 {
        frame(&mut cx);
    }
    let started = Instant::now();
    let cpu_started = thread_cpu_ns();
    for iteration in 0..iterations {
        cx.update(|cx| {
            if case.starts_with("keep-") {
                window.update(cx, |_, _, cx| cx.notify()).unwrap();
            } else {
                for leaf in &leaves {
                    leaf.update(cx, |leaf, cx| {
                        if case.starts_with("remove-") || case.starts_with("off-remove-") {
                            leaf.present = iteration % 2 == 0;
                        } else if case.starts_with("replace-") {
                            leaf.identity = iteration as u64 + 1;
                        } else if case.starts_with("grow-") {
                            leaf.height = 10. + (iteration % 3) as f32;
                        } else {
                            panic!("unknown fixture {case}");
                        }
                        cx.notify();
                    });
                }
            }
        });
        frame(&mut cx);
        black_box(&cx);
    }
    let cpu_ns = thread_cpu_ns() - cpu_started;
    let elapsed_ns = started.elapsed().as_nanos();
    #[cfg(gpui_splice_census)]
    let (held_bytes, peak_bytes, requested_bytes, allocation_calls) = census::finish();
    #[cfg(not(gpui_splice_census))]
    let (held_bytes, peak_bytes, requested_bytes, allocation_calls) = (-1, -1, 0, 0);
    let (state_count, state_box_bytes, frame_hash) = cx.update_window(window.into(), |_, window, _| {
        let states = window.rendered_frame.element_states.iter().filter(|((id, _), _)| {
            matches!(id.0.last(), Some(crate::ElementId::NamedInteger(name, _)) if name.as_ref() == "profile-state")
        });
        let (count, bytes) = states.fold((0, 0), |(count, bytes), (_, state)| {
            (count + 1, bytes + std::mem::size_of_val(state.inner.as_ref()))
        });
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        describe_frame(window).hash(&mut hasher);
        (count, bytes, hasher.finish())
    }).unwrap();
    println!("SPLICE_PROFILE {{\"case\":\"{case}\",\"iterations\":{iterations},\"cpu_ns\":{cpu_ns},\"elapsed_ns\":{elapsed_ns},\"signature\":{frame_hash},\"state_count\":{state_count},\"state_box_bytes\":{state_box_bytes},\"held_bytes\":{held_bytes},\"peak_bytes\":{peak_bytes},\"requested_bytes\":{requested_bytes},\"allocation_calls\":{allocation_calls}}}");
}
