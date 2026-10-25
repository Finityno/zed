//! Opt-in native regression checks for demand-driven frames (ported from
//! gpui-fast PR 84). They open real windows, so they are ignored by default;
//! run one case per process on an interactive desktop:
//!
//! ```text
//! $env:GPUI_NATIVE_CASE = 'paint_race'   # or input_restore, soft_recovery, recovery
//! $env:GPUI_NATIVE_DCOMP = '1'           # repeat with 0
//! cargo test -p gpui_windows --features test-support --lib native_tests::interactive_frame_lifecycle -- --ignored --exact --nocapture --test-threads=1
//! ```
//!
//! - `paint_race`: the worker invalidates a window while its paint is still
//!   running callbacks (after they re-armed demand). The invalidation must
//!   survive the paint, or chained next-frame callbacks stall.
//! - `input_restore`: high-rate synthetic input keeps presentation going for
//!   the sustain period and then parks; a minimized window that changed
//!   presents on restore without another notify; 120 chained callbacks across
//!   two windows; the survivor presents after the other closes.
//! - `soft_recovery`: recreates the devices the way a failed resize does
//!   (`invalidate_devices`), pausing the recovery while it messages windows
//!   so that closing one window and opening another overlaps it. This is the
//!   hang the PR left open, reproduced without resetting the GPU.
//! - `recovery`: the same with a real GPU reset (`DXCap.exe -forcetdr`); also
//!   needs `GPUI_TEST_ALLOW_GPU_RESET=1`. `GPUI_NATIVE_CHURN=0` skips the
//!   window churn.
//!
//! `GPUI_NATIVE_TEST_LOG` names a file for the raw observations. A watchdog
//! ends the test process (exit code 90) if a case stops making progress.

use gpui::{
    App, AppContext, Application, AsyncApp, Bounds, Context, InteractiveElement, IntoElement,
    ParentElement, QuitMode, Render, Styled, Window, WindowBounds, WindowHandle, WindowOptions,
    div, point, px, rgb, size,
};
use std::{
    collections::BTreeMap,
    fmt::Write as _,
    rc::Rc,
    sync::{
        Arc, OnceLock,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};
use windows::Win32::{
    Foundation::{HWND, POINT},
    Graphics::Gdi::{ClientToScreen, GetUpdateRect},
    UI::{
        Input::KeyboardAndMouse::{
            INPUT, INPUT_0, INPUT_MOUSE, MOUSEEVENTF_ABSOLUTE, MOUSEEVENTF_MOVE,
            MOUSEEVENTF_MOVE_NOCOALESCE, MOUSEINPUT, SendInput,
        },
        WindowsAndMessaging::{
            GetCursorPos, GetSystemMetrics, SM_CXSCREEN, SM_CYSCREEN, SetCursorPos,
        },
    },
};

static ACTIVE: AtomicBool = AtomicBool::new(false);
static PHASE: AtomicUsize = AtomicUsize::new(0);
static PAUSE_RECOVERY: AtomicBool = AtomicBool::new(false);
static DONE: AtomicBool = AtomicBool::new(false);
static STATE: OnceLock<parking_lot::Mutex<Trace>> = OnceLock::new();
static PAINT_TARGET: AtomicUsize = AtomicUsize::new(0);
static PAINT_PHASE: AtomicUsize = AtomicUsize::new(0);
static PAINT_RETAINED: AtomicBool = AtomicBool::new(false);

const RECOVERY_STARTED: usize = 1;
const RECOVERY_PAUSED: usize = 2;
const RECOVERY_FINISHED: usize = 3;

struct Trace {
    start: Instant,
    /// Per window: (microseconds since start, Present HRESULT, DirectComposition).
    presents: BTreeMap<usize, Vec<(u64, i32, bool)>>,
    inputs: BTreeMap<usize, Vec<u64>>,
    recovered: Vec<u64>,
    device_errors: Vec<i32>,
    checkpoints: Vec<String>,
}

fn state() -> &'static parking_lot::Mutex<Trace> {
    STATE.get_or_init(|| {
        parking_lot::Mutex::new(Trace {
            start: Instant::now(),
            presents: BTreeMap::new(),
            inputs: BTreeMap::new(),
            recovered: Vec::new(),
            device_errors: Vec::new(),
            checkpoints: Vec::new(),
        })
    })
}

fn elapsed_micros(trace: &Trace) -> u64 {
    trace.start.elapsed().as_micros() as u64
}

pub(crate) fn present(hwnd: HWND, direct_composition: bool, status: i32) {
    if !ACTIVE.load(Ordering::Relaxed) {
        return;
    }
    let mut trace = state().lock();
    let at = elapsed_micros(&trace);
    trace
        .presents
        .entry(hwnd.0 as usize)
        .or_default()
        .push((at, status, direct_composition));
}

pub(crate) fn input(hwnd: HWND) {
    if !ACTIVE.load(Ordering::Relaxed) {
        return;
    }
    let mut trace = state().lock();
    let at = elapsed_micros(&trace);
    trace.inputs.entry(hwnd.0 as usize).or_default().push(at);
}

pub(crate) fn recovered() {
    if ACTIVE.load(Ordering::Relaxed) {
        let mut trace = state().lock();
        let at = elapsed_micros(&trace);
        trace.recovered.push(at);
    }
}

pub(crate) fn recovery_started() {
    if ACTIVE.load(Ordering::Relaxed) {
        PHASE.store(RECOVERY_STARTED, Ordering::SeqCst);
        mark("device recovery started");
    }
}

pub(crate) fn device_removed(status: i32) {
    if ACTIVE.load(Ordering::Relaxed) {
        state().lock().device_errors.push(status);
        mark(&format!("GetDeviceRemovedReason=0x{:08X}", status as u32));
    }
}

/// Called before the recovery messages each window. Where the PR paused with
/// the window list read-locked, this pauses where the list used to be locked.
pub(crate) fn recovery_sending() {
    if ACTIVE.load(Ordering::Relaxed) && PAUSE_RECOVERY.swap(false, Ordering::SeqCst) {
        PHASE.store(RECOVERY_PAUSED, Ordering::SeqCst);
        mark("recovery is messaging windows; allowing UI lifecycle overlap");
        std::thread::sleep(Duration::from_millis(350));
    }
}

pub(crate) fn recovery_finished() {
    if ACTIVE.load(Ordering::Relaxed) {
        PHASE.store(RECOVERY_FINISHED, Ordering::SeqCst);
        mark("device recovery finished");
    }
}

/// The paint has validated its update region and is about to run GPUI.
pub(crate) fn paint_started(hwnd: HWND) {
    if PAINT_TARGET.load(Ordering::SeqCst) == hwnd.0 as usize {
        PAINT_PHASE
            .compare_exchange(1, 2, Ordering::SeqCst, Ordering::SeqCst)
            .ok();
    }
}

/// The paint is returning.
pub(crate) fn paint_done(hwnd: HWND) {
    if PAINT_TARGET.load(Ordering::SeqCst) != hwnd.0 as usize {
        return;
    }
    if PAINT_PHASE.load(Ordering::SeqCst) == 2 {
        // Let the real worker consume the demand the callbacks re-armed, and
        // invalidate, before this paint returns. No synthetic input or redraw.
        PAINT_PHASE.store(3, Ordering::SeqCst);
        let start = Instant::now();
        while PAINT_PHASE.load(Ordering::SeqCst) == 3 {
            assert!(
                start.elapsed() < Duration::from_secs(2),
                "worker did not invalidate during paint"
            );
            std::thread::sleep(Duration::from_millis(1));
        }
        assert!(
            unsafe { GetUpdateRect(hwnd, None, false) }.as_bool(),
            "worker invalidation has no update region"
        );
        mark("worker invalidation completed before the current paint returned");
    }
    if PAINT_PHASE.load(Ordering::SeqCst) == 4 {
        let retained = unsafe { GetUpdateRect(hwnd, None, false) }.as_bool();
        PAINT_RETAINED.store(retained, Ordering::SeqCst);
        mark(&format!("next paint's update region retained: {retained}"));
        PAINT_PHASE.store(5, Ordering::SeqCst);
    }
}

/// The worker is about to take a window's demand.
pub(crate) fn before_frame_request(hwnd: HWND) {
    if PAINT_TARGET.load(Ordering::SeqCst) != hwnd.0 as usize {
        return;
    }
    let start = Instant::now();
    while PAINT_PHASE.load(Ordering::SeqCst) == 2 {
        assert!(
            start.elapsed() < Duration::from_secs(2),
            "paint did not finish its callbacks"
        );
        std::thread::sleep(Duration::from_millis(1));
    }
}

/// The worker invalidated a window.
pub(crate) fn paint_invalidated(hwnd: HWND) {
    if PAINT_TARGET.load(Ordering::SeqCst) == hwnd.0 as usize
        && PAINT_PHASE
            .compare_exchange(3, 4, Ordering::SeqCst, Ordering::SeqCst)
            .is_ok()
    {
        let start = Instant::now();
        while PAINT_PHASE.load(Ordering::SeqCst) == 4 {
            assert!(
                start.elapsed() < Duration::from_secs(2),
                "paint did not acknowledge invalidation"
            );
            std::thread::sleep(Duration::from_millis(1));
        }
    }
}

fn dump() {
    let Some(path) = std::env::var_os("GPUI_NATIVE_TEST_LOG") else {
        return;
    };
    let trace = state().lock();
    let mut text = String::new();
    writeln!(text, "phase: {}", PHASE.load(Ordering::SeqCst)).ok();
    for checkpoint in &trace.checkpoints {
        writeln!(text, "checkpoint: {checkpoint}").ok();
    }
    writeln!(text, "device_errors: {:?}", trace.device_errors).ok();
    writeln!(text, "renderer_recovery_times_us: {:?}", trace.recovered).ok();
    for (hwnd, inputs) in &trace.inputs {
        writeln!(text, "inputs[{hwnd:#x}]: {inputs:?}").ok();
    }
    for (hwnd, presents) in &trace.presents {
        writeln!(text, "presents[{hwnd:#x}] (us, status, dcomp): {presents:?}").ok();
    }
    drop(trace);
    if let Err(error) = std::fs::write(&path, text) {
        eprintln!("writing {path:?}: {error}");
    }
}

fn mark(message: &str) {
    println!("NATIVE: {message}");
    state().lock().checkpoints.push(message.to_owned());
    dump();
}

fn successes(hwnd: usize) -> Vec<u64> {
    state()
        .lock()
        .presents
        .get(&hwnd)
        .into_iter()
        .flatten()
        .filter(|(_, status, _)| *status == 0)
        .map(|(at, _, _)| *at)
        .collect()
}

fn native(window: &Window) -> usize {
    match raw_window_handle::HasWindowHandle::window_handle(window)
        .map(|handle| handle.as_raw())
    {
        Ok(raw_window_handle::RawWindowHandle::Win32(handle)) => handle.hwnd.get() as usize,
        _ => panic!("not a Win32 window"),
    }
}

struct View {
    renders: Arc<AtomicUsize>,
    value: bool,
}

impl Render for View {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        self.renders.fetch_add(1, Ordering::SeqCst);
        div()
            .size_full()
            .on_mouse_move(cx.listener(|view, _, _, cx| {
                view.value = !view.value;
                cx.notify();
            }))
            .bg(rgb(if self.value { 0x286e42 } else { 0x243654 }))
            .child("Native frame lifecycle regression")
    }
}

fn open(cx: &mut App, index: usize) -> (WindowHandle<View>, Arc<AtomicUsize>) {
    let renders = Arc::new(AtomicUsize::new(0));
    let view_renders = renders.clone();
    let window = cx
        .open_window(
            WindowOptions {
                window_bounds: Some(WindowBounds::Windowed(Bounds::new(
                    point(px(100. + index as f32 * 450.), px(100.)),
                    size(px(420.), px(320.)),
                ))),
                ..Default::default()
            },
            |_, cx| {
                cx.new(|_| View {
                    renders: view_renders,
                    value: false,
                })
            },
        )
        .unwrap();
    (window, renders)
}

async fn pause(cx: &AsyncApp, milliseconds: u64) {
    cx.background_executor()
        .timer(Duration::from_millis(milliseconds))
        .await;
}

fn chain(window: &Window, completed: Arc<AtomicUsize>, left: usize) {
    window.on_next_frame(move |window, _| {
        completed.fetch_add(1, Ordering::SeqCst);
        if left > 1 {
            chain(window, completed, left - 1);
        }
    });
}

#[allow(
    clippy::disallowed_methods,
    reason = "The reset tool blocks only a dedicated native-test OS thread."
)]
fn force_gpu_reset() {
    std::thread::spawn(|| {
        assert!(
            std::process::Command::new("DXCap.exe")
                .arg("-forcetdr")
                .status()
                .unwrap()
                .success()
        );
    });
}

/// Moves the cursor over `hwnd` about 240 times a second for about a second,
/// then returns where it was.
fn produce_high_rate_input(hwnd: usize) -> (i32, i32) {
    unsafe {
        let mut original = POINT::default();
        GetCursorPos(&mut original).unwrap();
        let mut point = POINT { x: 100, y: 100 };
        ClientToScreen(HWND(hwnd as _), &mut point).ok().unwrap();
        let width = GetSystemMetrics(SM_CXSCREEN) - 1;
        let height = GetSystemMetrics(SM_CYSCREEN) - 1;
        for index in 0..240 {
            let event = INPUT {
                r#type: INPUT_MOUSE,
                Anonymous: INPUT_0 {
                    mi: MOUSEINPUT {
                        dx: (point.x + index % 2) * 65535 / width,
                        dy: point.y * 65535 / height,
                        dwFlags: MOUSEEVENTF_MOVE
                            | MOUSEEVENTF_ABSOLUTE
                            | MOUSEEVENTF_MOVE_NOCOALESCE,
                        ..Default::default()
                    },
                },
            };
            assert_eq!(SendInput(&[event], std::mem::size_of::<INPUT>() as i32), 1);
            std::thread::sleep(Duration::from_millis(4));
        }
        (original.x, original.y)
    }
}

async fn wait_for_phase(cx: &AsyncApp, phase: usize, attempts: usize) -> bool {
    for _ in 0..attempts {
        if PHASE.load(Ordering::SeqCst) == phase {
            return true;
        }
        pause(cx, 20).await;
    }
    PHASE.load(Ordering::SeqCst) == phase
}

/// Closes one window and opens another while the recovery is messaging
/// windows, then checks the replacement presents.
async fn recover_with_window_churn(
    cx: &mut AsyncApp,
    start_recovery: impl FnOnce(),
) {
    let (transient, _) = cx.update(|cx| open(cx, 2));
    pause(cx, 300).await;
    PAUSE_RECOVERY.store(true, Ordering::SeqCst);
    start_recovery();
    assert!(wait_for_phase(cx, RECOVERY_PAUSED, 4500).await);
    mark("UI closing one window and opening another during recovery dispatch");
    transient
        .update(cx, |_, window, _| window.remove_window())
        .unwrap();
    let (replacement, _) = cx.update(|cx| open(cx, 2));
    assert!(wait_for_phase(cx, RECOVERY_FINISHED, 250).await);
    replacement
        .update(cx, |view, _, cx| {
            view.value = true;
            cx.notify();
        })
        .unwrap();
    pause(cx, 300).await;
    let hwnd = replacement.update(cx, |_, window, _| native(window)).unwrap();
    assert!(!successes(hwnd).is_empty(), "the replacement window never presented");
    replacement
        .update(cx, |_, window, _| window.remove_window())
        .unwrap();
    mark("recovery and overlapping window churn completed");
}

#[test]
#[ignore = "opens windows on an interactive desktop; see the module docs"]
fn interactive_frame_lifecycle() {
    let case = std::env::var("GPUI_NATIVE_CASE").unwrap_or_else(|_| "input_restore".into());
    assert!(matches!(
        case.as_str(),
        "input_restore" | "paint_race" | "soft_recovery" | "recovery"
    ));
    if case == "recovery" {
        assert_eq!(
            std::env::var("GPUI_TEST_ALLOW_GPU_RESET").as_deref(),
            Ok("1"),
            "the recovery case resets the GPU"
        );
    }
    let churn = std::env::var("GPUI_NATIVE_CHURN").as_deref() != Ok("0");
    let disabled = std::env::var("GPUI_NATIVE_DCOMP").as_deref() == Ok("0");
    unsafe {
        if disabled {
            std::env::set_var("GPUI_DISABLE_DIRECT_COMPOSITION", "true");
        } else {
            std::env::remove_var("GPUI_DISABLE_DIRECT_COMPOSITION");
        }
    }
    ACTIVE.store(true, Ordering::SeqCst);
    mark(&format!(
        "pid={} case={case} DirectComposition={}",
        std::process::id(),
        !disabled
    ));
    std::thread::spawn(|| {
        let start = Instant::now();
        let mut paused_at = None;
        while !DONE.load(Ordering::SeqCst) {
            if PHASE.load(Ordering::SeqCst) == RECOVERY_PAUSED && paused_at.is_none() {
                paused_at = Some(Instant::now());
            }
            if paused_at.is_some_and(|at| at.elapsed() > Duration::from_secs(8))
                || start.elapsed() > Duration::from_secs(110)
            {
                mark("FAIL watchdog: no progress; terminating only this test process");
                std::process::exit(90);
            }
            std::thread::sleep(Duration::from_millis(50));
        }
    });
    let platform = Rc::new(crate::WindowsPlatform::new(false).unwrap());
    let test_platform = platform.clone();
    Application::with_platform(platform).run(move |cx| {
        cx.set_quit_mode(QuitMode::Explicit);
        let (first, _) = open(cx, 0);
        let (second, second_renders) = open(cx, 1);
        cx.spawn(async move |cx| {
            let a = first
                .update(cx, |_, window, _| {
                    window.activate_window();
                    native(window)
                })
                .unwrap();
            let b = second.update(cx, |_, window, _| native(window)).unwrap();
            pause(cx, 3000).await;
            assert!(!successes(a).is_empty() && !successes(b).is_empty());
            {
                let trace = state().lock();
                for hwnd in [a, b] {
                    assert!(
                        trace.presents[&hwnd]
                            .iter()
                            .any(|(_, status, dcomp)| *status == 0 && *dcomp == !disabled)
                    );
                }
            }
            mark("initial DXGI Present S_OK and actual DirectComposition state verified");

            if case == "paint_race" {
                let callbacks = Arc::new(AtomicUsize::new(0));
                PAINT_TARGET.store(a, Ordering::SeqCst);
                PAINT_PHASE.store(1, Ordering::SeqCst);
                first
                    .update(cx, |_, window, _| chain(window, callbacks.clone(), 12))
                    .unwrap();
                pause(cx, 2000).await;
                let completed = callbacks.load(Ordering::SeqCst);
                mark(&format!(
                    "paint race: phase={}, callbacks={completed}/12",
                    PAINT_PHASE.load(Ordering::SeqCst)
                ));
                if PAINT_PHASE.load(Ordering::SeqCst) == 5
                    && PAINT_RETAINED.load(Ordering::SeqCst)
                    && completed == 12
                {
                    mark("PASS");
                    DONE.store(true, Ordering::SeqCst);
                }
                cx.update(|cx| cx.quit());
                return;
            }

            if case == "input_restore" {
                let producer = std::thread::spawn(move || produce_high_rate_input(a));
                pause(cx, 2600).await;
                let original = producer.join().unwrap();
                unsafe { SetCursorPos(original.0, original.1).unwrap() };
                let (last_input, input_count) = {
                    let trace = state().lock();
                    let inputs = trace.inputs.get(&a).cloned().unwrap_or_default();
                    (inputs.last().copied().unwrap_or(0), inputs.len())
                };
                let tail: Vec<_> = successes(a)
                    .into_iter()
                    .filter(|at| *at >= last_input + 100_000 && *at <= last_input + 850_000)
                    .collect();
                let mut edges = vec![last_input + 100_000];
                edges.extend(tail.iter().copied());
                edges.push(last_input + 850_000);
                let maximum_gap = edges
                    .windows(2)
                    .map(|pair| pair[1] - pair[0])
                    .max()
                    .unwrap_or(0);
                mark(&format!(
                    "high-rate input: {input_count} events; {} successful presents in the \
                     100-850 ms tail; largest gap {maximum_gap} us",
                    tail.len()
                ));
                assert!(input_count >= 100, "synthetic input did not arrive");
                if tail.len() < 10
                    || tail.last().is_none_or(|at| *at < last_input + 750_000)
                    || maximum_gap > 150_000
                {
                    mark("FAIL: presentation sustain stopped or stalled after high-rate input");
                    cx.update(|cx| cx.quit());
                    return;
                }
                pause(cx, 600).await;
                let before = successes(a).len();
                pause(cx, 400).await;
                assert_eq!(successes(a).len(), before, "presentation did not return to idle");
                mark("presentation tail maintained, then parked");
            }

            second
                .update(cx, |_, window, _| window.minimize_window())
                .unwrap();
            pause(cx, 300).await;
            second
                .update(cx, |view, _, cx| {
                    view.value = true;
                    cx.notify();
                })
                .unwrap();

            if case == "soft_recovery" {
                recover_with_window_churn(cx, || test_platform.recreate_devices()).await;
            }
            if case == "recovery" {
                if churn {
                    recover_with_window_churn(cx, force_gpu_reset).await;
                } else {
                    force_gpu_reset();
                    assert!(wait_for_phase(cx, RECOVERY_FINISHED, 4500).await);
                    assert!(!state().lock().device_errors.is_empty());
                    mark("real device reset recovered while the second window was minimized");
                }
                assert!(state().lock().recovered.len() >= 2);
            }

            let before = successes(b).len();
            let before_renders = second_renders.load(Ordering::SeqCst);
            second
                .update(cx, |_, window, _| window.activate_window())
                .unwrap();
            pause(cx, 500).await;
            assert!(
                successes(b).len() > before,
                "restore did not present without an extra notify"
            );
            assert!(
                second_renders.load(Ordering::SeqCst) > before_renders,
                "restore did not rebuild the changed view"
            );
            let before = successes(b).len();
            second
                .update(cx, |view, _, cx| {
                    view.value = false;
                    cx.notify();
                })
                .unwrap();
            pause(cx, 200).await;
            assert!(successes(b).len() > before, "no successful present after restore");
            mark("minimize / restore resumed DXGI presentation");

            let callbacks = Arc::new(AtomicUsize::new(0));
            for handle in [first, second] {
                let callbacks = callbacks.clone();
                handle
                    .update(cx, |_, window, _| chain(window, callbacks, 60))
                    .unwrap();
            }
            pause(cx, 3000).await;
            assert_eq!(callbacks.load(Ordering::SeqCst), 120);
            second
                .update(cx, |_, window, _| window.remove_window())
                .unwrap();
            let before = successes(a).len();
            first
                .update(cx, |view, _, cx| {
                    view.value = true;
                    cx.notify();
                })
                .unwrap();
            pause(cx, 300).await;
            assert!(successes(a).len() > before);
            mark("PASS: survivor presents after the other window closes; 120 chained callbacks");
            DONE.store(true, Ordering::SeqCst);
            cx.update(|cx| cx.quit());
        })
        .detach();
    });
    assert!(DONE.load(Ordering::SeqCst), "native regression did not complete");
    ACTIVE.store(false, Ordering::SeqCst);
}
