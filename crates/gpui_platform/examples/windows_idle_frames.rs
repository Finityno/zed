//! What GPUI's Windows frame scheduling costs, per idle pattern.
//!
//! ```text
//! cargo run -p gpui_platform --release --example windows_idle_frames -- <case> <out.json> [--seconds N] [--no-dcomp]
//! ```
//!
//! Cases (one per process):
//!
//! - `static`: one focused window that never changes. Expect no renders,
//!   paints, presents or vsync waits.
//! - `blink`: a focused window whose caret toggles every 500 ms through a
//!   timer and `cx.notify`, as a text input's does: two renders a second.
//! - `animation`: continuous `request_animation_frame`, one render per vsync.
//! - `shimmer`: a static `ShimmerText`, which re-presents at the time
//!   animation rate (30 fps by default) without rendering.
//! - `two_windows`: two static windows.
//! - `unfocused_animation`: the `animation` view in a window that is not
//!   focused (another window is), held to GPUI's ~30 fps inactive-window cap.
//! - `streaming`: what an agent turn streaming in looks like: a transcript
//!   view notified 24 times a second as its text grows, a spinner view
//!   notified 30 times a second by a timer, and a shimmering label.
//! - `verify`: correctness, not cost: 25 notifies and 60 chained
//!   `on_next_frame` callbacks in each of two windows must all be delivered.
//! - `two_monitors`: a static window on each of the first two displays (side
//!   by side on the one display when there is only one).
//! - `two_monitors_animation`: the `animation` view on the second display,
//!   and a static, focused window on the first: only the second display's
//!   clock should tick, at its own refresh rate.
//!
//! Measures the process's CPU time and cycles, the cycles of the UI thread
//! and of the clock threads (`VSyncProvider` and each `MonitorVBlank`), how
//! often the clock threads were switched in (their wakes), view renders, the
//! displays' refresh rates, and the platform's frame counters, over
//! `--seconds` (default 20) after a 3 second warm-up. Opens real windows; run
//! it on an interactive desktop.

fn main() {
    #[cfg(target_os = "windows")]
    windows_idle_frames::run();
    #[cfg(not(target_os = "windows"))]
    eprintln!("This benchmark requires Windows.");
}

#[cfg(target_os = "windows")]
mod windows_idle_frames {
    use gpui::{
        App, AppContext as _, Bounds, Context, FocusHandle, InteractiveElement as _,
        IntoElement, ParentElement as _, Render, ShimmerText, Styled as _, Window,
        WindowBounds, WindowHandle, WindowOptions, div, point, px, rgb, size,
    };
    use std::{
        ffi::c_void,
        fmt::Write as _,
        path::PathBuf,
        rc::Rc,
        sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
        },
        time::{Duration, Instant},
    };

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetCurrentProcess() -> *mut c_void;
        fn GetCurrentProcessId() -> u32;
        fn GetCurrentThread() -> *mut c_void;
        fn GetProcessTimes(
            process: *mut c_void,
            created: *mut u64,
            exited: *mut u64,
            kernel: *mut u64,
            user: *mut u64,
        ) -> i32;
        fn QueryProcessCycleTime(process: *mut c_void, cycles: *mut u64) -> i32;
        fn QueryThreadCycleTime(thread: *mut c_void, cycles: *mut u64) -> i32;
        fn CreateToolhelp32Snapshot(flags: u32, process_id: u32) -> *mut c_void;
        fn Thread32First(snapshot: *mut c_void, entry: *mut ThreadEntry) -> i32;
        fn Thread32Next(snapshot: *mut c_void, entry: *mut ThreadEntry) -> i32;
        fn OpenThread(access: u32, inherit: i32, thread_id: u32) -> *mut c_void;
        fn GetThreadDescription(thread: *mut c_void, description: *mut *mut u16) -> i32;
        fn LocalFree(memory: *mut c_void) -> *mut c_void;
        fn CloseHandle(handle: *mut c_void) -> i32;
    }

    #[link(name = "ntdll")]
    unsafe extern "system" {
        fn NtQuerySystemInformation(
            class: u32,
            information: *mut c_void,
            length: u32,
            return_length: *mut u32,
        ) -> i32;
    }

    #[link(name = "user32")]
    unsafe extern "system" {
        fn EnumDisplayDevicesW(
            device: *const u16,
            index: u32,
            display_device: *mut u8,
            flags: u32,
        ) -> i32;
        fn EnumDisplaySettingsW(device: *const u16, mode: u32, dev_mode: *mut u8) -> i32;
    }

    #[repr(C)]
    struct ThreadEntry {
        size: u32,
        usage: u32,
        thread_id: u32,
        owner_process_id: u32,
        base_priority: i32,
        delta_priority: i32,
        flags: u32,
    }

    const TH32CS_SNAPTHREAD: u32 = 0x4;
    const THREAD_QUERY_LIMITED_INFORMATION: u32 = 0x0800;
    const INVALID_HANDLE_VALUE: *mut c_void = -1isize as *mut c_void;

    /// The names of the threads that tick windows: the compositor clock and
    /// (since per-monitor clocks) one per monitor.
    const CLOCK_THREAD_NAMES: [&str; 2] = ["VSyncProvider", "MonitorVBlank"];

    /// The clock threads, found by their names so this also measures builds
    /// that predate the platform's frame counters.
    struct ClockThreads(Vec<(u32, *mut c_void)>);

    impl ClockThreads {
        fn find() -> Self {
            unsafe {
                let snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
                if snapshot == INVALID_HANDLE_VALUE {
                    return Self(Vec::new());
                }
                let process_id = GetCurrentProcessId();
                let mut entry = ThreadEntry {
                    size: std::mem::size_of::<ThreadEntry>() as u32,
                    usage: 0,
                    thread_id: 0,
                    owner_process_id: 0,
                    base_priority: 0,
                    delta_priority: 0,
                    flags: 0,
                };
                let mut found = Vec::new();
                let mut more = Thread32First(snapshot, &mut entry) != 0;
                while more {
                    if entry.owner_process_id == process_id {
                        let thread =
                            OpenThread(THREAD_QUERY_LIMITED_INFORMATION, 0, entry.thread_id);
                        if !thread.is_null() {
                            if thread_name(thread)
                                .is_some_and(|name| CLOCK_THREAD_NAMES.contains(&name.as_str()))
                            {
                                found.push((entry.thread_id, thread));
                            } else {
                                CloseHandle(thread);
                            }
                        }
                    }
                    more = Thread32Next(snapshot, &mut entry) != 0;
                }
                CloseHandle(snapshot);
                Self(found)
            }
        }

        fn cycles(&self) -> u64 {
            self.0
                .iter()
                .map(|(_, thread)| {
                    let mut cycles = 0;
                    unsafe { QueryThreadCycleTime(*thread, &mut cycles) };
                    cycles
                })
                .sum()
        }

        /// How often the clock threads have been switched in, summed: each
        /// is a wake (a vblank waited for, a timer, a request).
        fn context_switches(&self) -> u64 {
            let ids: Vec<u32> = self.0.iter().map(|(id, _)| *id).collect();
            thread_context_switches(&ids)
        }
    }

    impl Drop for ClockThreads {
        fn drop(&mut self) {
            for (_, thread) in &self.0 {
                unsafe { CloseHandle(*thread) };
            }
        }
    }

    /// The summed context switches of the threads of this process with the
    /// given ids, from `SystemProcessInformation`.
    fn thread_context_switches(thread_ids: &[u32]) -> u64 {
        const SYSTEM_PROCESS_INFORMATION: u32 = 5;
        const STATUS_INFO_LENGTH_MISMATCH: i32 = 0xC000_0004_u32 as i32;
        // x64 layouts of SYSTEM_PROCESS_INFORMATION and
        // SYSTEM_THREAD_INFORMATION.
        const PROCESS_THREAD_COUNT: usize = 4;
        const PROCESS_ID: usize = 80;
        const PROCESS_SIZE: usize = 256;
        const THREAD_ID: usize = 48;
        const THREAD_CONTEXT_SWITCHES: usize = 64;
        const THREAD_SIZE: usize = 80;
        let mut buffer = vec![0u64; 1 << 17];
        loop {
            let mut needed = 0;
            let status = unsafe {
                NtQuerySystemInformation(
                    SYSTEM_PROCESS_INFORMATION,
                    buffer.as_mut_ptr().cast(),
                    (buffer.len() * 8) as u32,
                    &mut needed,
                )
            };
            if status == STATUS_INFO_LENGTH_MISMATCH {
                buffer.resize((needed as usize / 8 + 1) * 2, 0);
                continue;
            }
            if status < 0 {
                return 0;
            }
            break;
        }
        let bytes = unsafe {
            std::slice::from_raw_parts(buffer.as_ptr().cast::<u8>(), buffer.len() * 8)
        };
        let read_u32 = |at: usize| {
            u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap_or_default())
        };
        let read_u64 = |at: usize| {
            u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap_or_default())
        };
        let process_id = unsafe { GetCurrentProcessId() } as u64;
        let mut offset = 0;
        loop {
            if read_u64(offset + PROCESS_ID) == process_id {
                let threads = read_u32(offset + PROCESS_THREAD_COUNT) as usize;
                return (0..threads)
                    .map(|index| offset + PROCESS_SIZE + index * THREAD_SIZE)
                    .filter(|thread| thread_ids.contains(&(read_u64(thread + THREAD_ID) as u32)))
                    .map(|thread| read_u32(thread + THREAD_CONTEXT_SWITCHES) as u64)
                    .sum();
            }
            let next = read_u32(offset) as usize;
            if next == 0 {
                return 0;
            }
            offset += next;
        }
    }

    /// The current refresh rate of every display attached to the desktop.
    fn refresh_rates() -> Vec<u32> {
        const DISPLAY_DEVICE_SIZE: usize = 840;
        const DISPLAY_DEVICE_STATE_FLAGS: usize = 324;
        const DISPLAY_DEVICE_ATTACHED_TO_DESKTOP: u32 = 0x1;
        const DEVMODE_SIZE: usize = 220;
        const DEVMODE_SIZE_FIELD: usize = 68;
        const DEVMODE_DISPLAY_FREQUENCY: usize = 184;
        const ENUM_CURRENT_SETTINGS: u32 = u32::MAX;
        let read_u32 = |bytes: &[u8], at: usize| {
            u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap_or_default())
        };
        let mut rates = Vec::new();
        for index in 0.. {
            let mut device = [0u8; DISPLAY_DEVICE_SIZE];
            device[..4].copy_from_slice(&(DISPLAY_DEVICE_SIZE as u32).to_le_bytes());
            if unsafe { EnumDisplayDevicesW(std::ptr::null(), index, device.as_mut_ptr(), 0) }
                == 0
            {
                break;
            }
            if read_u32(&device, DISPLAY_DEVICE_STATE_FLAGS) & DISPLAY_DEVICE_ATTACHED_TO_DESKTOP
                == 0
            {
                continue;
            }
            let name: Vec<u16> = device[4..68]
                .chunks_exact(2)
                .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
                .collect();
            let mut mode = [0u8; DEVMODE_SIZE];
            mode[DEVMODE_SIZE_FIELD..DEVMODE_SIZE_FIELD + 2]
                .copy_from_slice(&(DEVMODE_SIZE as u16).to_le_bytes());
            if unsafe {
                EnumDisplaySettingsW(name.as_ptr(), ENUM_CURRENT_SETTINGS, mode.as_mut_ptr())
            } != 0
            {
                rates.push(read_u32(&mode, DEVMODE_DISPLAY_FREQUENCY));
            }
        }
        rates
    }

    unsafe fn thread_name(thread: *mut c_void) -> Option<String> {
        let mut description = std::ptr::null_mut();
        if unsafe { GetThreadDescription(thread, &mut description) } < 0 || description.is_null() {
            return None;
        }
        let mut length = 0;
        while unsafe { *description.add(length) } != 0 {
            length += 1;
        }
        let name =
            String::from_utf16_lossy(unsafe { std::slice::from_raw_parts(description, length) });
        unsafe { LocalFree(description.cast()) };
        Some(name)
    }

    #[derive(Clone, Copy)]
    struct Sample {
        process_time_100ns: u64,
        process_cycles: u64,
        main_thread_cycles: u64,
        clock_thread_cycles: u64,
        clock_thread_wakes: u64,
        renders: usize,
        frames: gpui_windows::FrameCounters,
    }

    fn sample(clock_threads: &ClockThreads, renders: &AtomicUsize) -> Sample {
        let (mut created, mut exited, mut kernel, mut user) = (0, 0, 0, 0);
        let (mut process_cycles, mut main_thread_cycles) = (0, 0);
        unsafe {
            GetProcessTimes(
                GetCurrentProcess(),
                &mut created,
                &mut exited,
                &mut kernel,
                &mut user,
            );
            QueryProcessCycleTime(GetCurrentProcess(), &mut process_cycles);
            QueryThreadCycleTime(GetCurrentThread(), &mut main_thread_cycles);
        }
        Sample {
            process_time_100ns: kernel + user,
            process_cycles,
            main_thread_cycles,
            clock_thread_cycles: clock_threads.cycles(),
            clock_thread_wakes: clock_threads.context_switches(),
            renders: renders.load(Ordering::Relaxed),
            frames: gpui_windows::frame_counters(),
        }
    }

    #[derive(Clone, Copy, PartialEq)]
    enum Case {
        Static,
        Blink,
        Animation,
        Shimmer,
        Streaming,
        TwoWindows,
        UnfocusedAnimation,
        Verify,
        TwoMonitors,
        TwoMonitorsAnimation,
    }

    impl Case {
        fn parse(name: &str) -> Option<Self> {
            Some(match name {
                "static" => Self::Static,
                "blink" => Self::Blink,
                "animation" => Self::Animation,
                "shimmer" => Self::Shimmer,
                "streaming" => Self::Streaming,
                "two_windows" => Self::TwoWindows,
                "unfocused_animation" => Self::UnfocusedAnimation,
                "verify" => Self::Verify,
                "two_monitors" => Self::TwoMonitors,
                "two_monitors_animation" => Self::TwoMonitorsAnimation,
                _ => return None,
            })
        }
    }

    #[derive(Clone, Copy, PartialEq)]
    enum Content {
        Static,
        Blink,
        Animation,
        Shimmer,
        Streaming,
    }

    const STREAMED_WORDS: [&str; 8] = [
        "frames", "are", "scheduled", "on", "demand,", "so", "nothing", "spins",
    ];

    /// Notifies itself every `interval`, running `step` first.
    fn every<T: 'static>(
        interval: Duration,
        cx: &mut Context<T>,
        step: impl Fn(&mut T) + 'static,
    ) -> gpui::Task<()> {
        cx.spawn(async move |this, cx| {
            loop {
                cx.background_executor().timer(interval).await;
                if this
                    .update(cx, |this, cx| {
                        step(this);
                        cx.notify();
                    })
                    .is_err()
                {
                    break;
                }
            }
        })
    }

    /// A transcript whose text grows by a word per streamed chunk.
    struct StreamedText {
        words: usize,
        renders: Arc<AtomicUsize>,
        _stream: gpui::Task<()>,
    }

    impl Render for StreamedText {
        fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
            self.renders.fetch_add(1, Ordering::Relaxed);
            let text = (0..self.words)
                .map(|word| STREAMED_WORDS[word % STREAMED_WORDS.len()])
                .collect::<Vec<_>>()
                .join(" ");
            div().w(px(560.)).child(text)
        }
    }

    /// A spinner ticked by a timer, as an app's paced decorative ticker is.
    struct Spinner {
        phase: usize,
        renders: Arc<AtomicUsize>,
        _tick: gpui::Task<()>,
    }

    impl Render for Spinner {
        fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
            self.renders.fetch_add(1, Ordering::Relaxed);
            div().flex().gap_1().children((0..8).map(|dot| {
                div()
                    .size(px(6.))
                    .rounded_full()
                    .bg(if dot == self.phase % 8 { rgb(0xe8e8e8) } else { rgb(0x404246) })
            }))
        }
    }

    struct BenchView {
        renders: Arc<AtomicUsize>,
        content: Content,
        focus_handle: FocusHandle,
        caret_visible: bool,
        lit: bool,
        started_at: Instant,
        streamed: Option<(gpui::Entity<Spinner>, gpui::Entity<StreamedText>)>,
        _blink: Option<gpui::Task<()>>,
    }

    impl BenchView {
        fn new(content: Content, renders: Arc<AtomicUsize>, cx: &mut Context<Self>) -> Self {
            let blink = (content == Content::Blink).then(|| {
                cx.spawn(async move |this, cx| {
                    loop {
                        cx.background_executor()
                            .timer(Duration::from_millis(500))
                            .await;
                        if this
                            .update(cx, |this, cx| {
                                this.caret_visible = !this.caret_visible;
                                cx.notify();
                            })
                            .is_err()
                        {
                            break;
                        }
                    }
                })
            });
            let streamed = (content == Content::Streaming).then(|| {
                let spinner = cx.new(|cx| Spinner {
                    phase: 0,
                    renders: renders.clone(),
                    _tick: every(Duration::from_micros(33_333), cx, |spinner: &mut Spinner| {
                        spinner.phase += 1
                    }),
                });
                let text = cx.new(|cx| StreamedText {
                    words: 0,
                    renders: renders.clone(),
                    _stream: every(Duration::from_micros(41_667), cx, |text: &mut StreamedText| {
                        // Keep the transcript to a screenful, as a list would.
                        text.words = (text.words + 1) % 400
                    }),
                });
                (spinner, text)
            });
            Self {
                renders,
                content,
                focus_handle: cx.focus_handle(),
                streamed,
                caret_visible: true,
                lit: true,
                started_at: Instant::now(),
                _blink: blink,
            }
        }
    }

    impl Render for BenchView {
        fn render(&mut self, window: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
            self.renders.fetch_add(1, Ordering::Relaxed);
            let body = match self.content {
                Content::Static => div().child("GPUI Windows idle benchmark").into_any_element(),
                Content::Blink => div()
                    .flex()
                    .items_center()
                    .child("Focused input")
                    .child(div().w(px(2.)).h(px(18.)).ml_1().when_caret(self.caret_visible))
                    .into_any_element(),
                Content::Animation => {
                    window.request_animation_frame();
                    let phase = (self.started_at.elapsed().as_secs_f32() * 0.5).fract();
                    div()
                        .w(px(600.))
                        .h(px(24.))
                        .bg(rgb(0x303236))
                        .child(div().ml(px(phase * 560.)).w(px(40.)).h_full().bg(rgb(0x65b879)))
                        .into_any_element()
                }
                Content::Shimmer => ShimmerText::new("Thinking about frame scheduling")
                    .highlight_color(gpui::white())
                    .into_any_element(),
                Content::Streaming => {
                    let (spinner, text) = self.streamed.clone().expect("streaming views");
                    div()
                        .flex()
                        .flex_col()
                        .gap_2()
                        .child(
                            div().flex().gap_2().child(spinner).child(
                                ShimmerText::new("Thinking").highlight_color(gpui::white()),
                            ),
                        )
                        .child(text)
                        .into_any_element()
                }
            };
            div()
                .track_focus(&self.focus_handle)
                .size_full()
                .flex()
                .flex_col()
                .p_8()
                .gap_4()
                .font_family("Segoe UI")
                .text_color(rgb(0xe8e8e8))
                .bg(rgb(0x202124))
                .child(body)
                .child(
                    div()
                        .w(px(320.))
                        .h(px(180.))
                        .rounded_lg()
                        .border_2()
                        .border_color(if self.lit { rgb(0x65b879) } else { rgb(0x404246) })
                        .bg(rgb(0x303236)),
                )
        }
    }

    trait WhenCaret {
        fn when_caret(self, visible: bool) -> Self;
    }

    impl WhenCaret for gpui::Div {
        fn when_caret(self, visible: bool) -> Self {
            if visible { self.bg(rgb(0xe8e8e8)) } else { self }
        }
    }

    fn open(
        cx: &mut App,
        index: usize,
        content: Content,
        renders: &Arc<AtomicUsize>,
    ) -> WindowHandle<BenchView> {
        open_on(cx, None, index, content, renders)
    }

    /// Opens a window on the `display`-th display or, when there are fewer,
    /// beside the others on the first.
    fn open_on_display(
        cx: &mut App,
        display: usize,
        content: Content,
        renders: &Arc<AtomicUsize>,
    ) -> WindowHandle<BenchView> {
        let displays = cx.displays();
        let (target, index) = match displays.get(display) {
            Some(target) => (Some(target.clone()), 0),
            None => (displays.first().cloned(), display),
        };
        open_on(cx, target, index, content, renders)
    }

    fn open_on(
        cx: &mut App,
        display: Option<Rc<dyn gpui::PlatformDisplay>>,
        index: usize,
        content: Content,
        renders: &Arc<AtomicUsize>,
    ) -> WindowHandle<BenchView> {
        let renders = renders.clone();
        let origin = display
            .as_ref()
            .map_or(point(px(0.), px(0.)), |display| display.bounds().origin);
        cx.open_window(
            WindowOptions {
                window_bounds: Some(WindowBounds::Windowed(Bounds::new(
                    point(
                        origin.x + px(100. + index as f32 * 660.),
                        origin.y + px(100.),
                    ),
                    size(px(640.), px(480.)),
                ))),
                display_id: display.as_ref().map(|display| display.id()),
                ..Default::default()
            },
            move |window, cx| {
                let view = cx.new(|cx| BenchView::new(content, renders, cx));
                view.read(cx).focus_handle.clone().focus(window, cx);
                view
            },
        )
        .expect("opening a benchmark window")
    }

    fn chain(window: &Window, completed: Rc<AtomicUsize>, remaining: usize) {
        window.on_next_frame(move |window, _| {
            completed.fetch_add(1, Ordering::Relaxed);
            if remaining > 1 {
                chain(window, completed, remaining - 1);
            }
        });
    }

    pub fn run() {
        let mut arguments = std::env::args().skip(1);
        let usage = "usage: windows_idle_frames <static|blink|animation|shimmer|streaming|two_windows|unfocused_animation|verify|two_monitors|two_monitors_animation> <out.json> [--seconds N] [--no-dcomp]";
        let case_name = arguments.next().expect(usage);
        let case = Case::parse(&case_name).expect(usage);
        let output = PathBuf::from(arguments.next().expect(usage));
        let mut seconds = 20.0;
        let mut direct_composition = true;
        while let Some(argument) = arguments.next() {
            match argument.as_str() {
                "--seconds" => {
                    seconds = arguments
                        .next()
                        .and_then(|value| value.parse().ok())
                        .expect(usage)
                }
                "--no-dcomp" => direct_composition = false,
                _ => panic!("{usage}"),
            }
        }
        if !direct_composition {
            // Read once, when the platform is created below.
            unsafe { std::env::set_var("GPUI_DISABLE_DIRECT_COMPOSITION", "true") };
        }

        gpui_platform::application().run(move |cx: &mut App| {
            let renders = Arc::new(AtomicUsize::new(0));
            let mut windows = Vec::new();
            match case {
                Case::Static => windows.push(open(cx, 0, Content::Static, &renders)),
                Case::Blink => windows.push(open(cx, 0, Content::Blink, &renders)),
                Case::Animation => windows.push(open(cx, 0, Content::Animation, &renders)),
                Case::Shimmer => windows.push(open(cx, 0, Content::Shimmer, &renders)),
                Case::Streaming => windows.push(open(cx, 0, Content::Streaming, &renders)),
                Case::TwoWindows | Case::Verify => {
                    windows.push(open(cx, 0, Content::Static, &renders));
                    windows.push(open(cx, 1, Content::Static, &renders));
                }
                Case::UnfocusedAnimation => {
                    // Rendered by the animating window only.
                    windows.push(open(cx, 0, Content::Animation, &renders));
                    let focused = open(cx, 1, Content::Static, &Arc::new(AtomicUsize::new(0)));
                    focused
                        .update(cx, |_, window, _| window.activate_window())
                        .ok();
                    windows.push(focused);
                }
                Case::TwoMonitors => {
                    windows.push(open_on_display(cx, 0, Content::Static, &renders));
                    windows.push(open_on_display(cx, 1, Content::Static, &renders));
                }
                Case::TwoMonitorsAnimation => {
                    windows.push(open_on_display(cx, 1, Content::Animation, &renders));
                    let focused =
                        open_on_display(cx, 0, Content::Static, &Arc::new(AtomicUsize::new(0)));
                    focused
                        .update(cx, |_, window, _| window.activate_window())
                        .ok();
                    windows.push(focused);
                }
            }
            let displays = cx.displays().len();
            cx.spawn(async move |cx| {
                cx.background_executor().timer(Duration::from_secs(3)).await;
                let clock_threads = ClockThreads::find();
                let before = sample(&clock_threads, &renders);
                let start = Instant::now();
                let mut verify_report = String::new();
                if case == Case::Verify {
                    let notified_before = renders.load(Ordering::Relaxed);
                    for frame in 0..25 {
                        for window in &windows {
                            window
                                .update(cx, |view, _, cx| {
                                    view.lit = frame % 2 == 0;
                                    cx.notify();
                                })
                                .ok();
                        }
                        cx.background_executor()
                            .timer(Duration::from_millis(80))
                            .await;
                    }
                    let update_renders = renders.load(Ordering::Relaxed) - notified_before;
                    let completed = Rc::new(AtomicUsize::new(0));
                    for window in &windows {
                        let completed = completed.clone();
                        window
                            .update(cx, |_, window, _| chain(window, completed, 60))
                            .ok();
                    }
                    cx.background_executor().timer(Duration::from_secs(3)).await;
                    let callbacks = completed.load(Ordering::Relaxed);
                    write!(
                        verify_report,
                        ", \"update_renders\": {update_renders}, \"chained_callbacks\": {callbacks}"
                    )
                    .ok();
                    assert!(update_renders >= 50, "{update_renders} renders for 50 notifies");
                    assert_eq!(callbacks, 120, "chained next-frame callbacks were stranded");
                } else {
                    cx.background_executor()
                        .timer(Duration::from_secs_f64(seconds))
                        .await;
                }
                let elapsed = start.elapsed().as_secs_f64();
                let after = sample(&clock_threads, &renders);
                let counters = {
                    let (a, b) = (after.frames, before.frames);
                    [
                        ("wake_signals", a.wake_signals - b.wake_signals),
                        ("request_wakes", a.request_wakes - b.request_wakes),
                        (
                            "deadline_request_wakes",
                            a.deadline_request_wakes - b.deadline_request_wakes,
                        ),
                        ("expedite_wakes", a.expedite_wakes - b.expedite_wakes),
                        ("worker_wakes", a.worker_wakes - b.worker_wakes),
                        ("deadline_wakeups", a.deadline_wakeups - b.deadline_wakeups),
                        ("vsync_waits", a.vsync_waits - b.vsync_waits),
                        (
                            "monitor_vsync_waits",
                            a.monitor_vsync_waits - b.monitor_vsync_waits,
                        ),
                        (
                            "vsyncs_without_paint",
                            a.vsyncs_without_paint - b.vsyncs_without_paint,
                        ),
                        (
                            "vsync_waits_skipped",
                            a.vsync_waits_skipped - b.vsync_waits_skipped,
                        ),
                        ("invalidations", a.invalidations - b.invalidations),
                        ("hand_offs", a.hand_offs - b.hand_offs),
                        ("hand_off_wakes", a.hand_off_wakes - b.hand_off_wakes),
                        ("paints", a.paints - b.paints),
                        ("presents", a.presents - b.presents),
                    ]
                    .iter()
                    .map(|(name, value)| format!(",\n  \"{name}\": {value}"))
                    .collect::<String>()
                };
                let refresh_rates = refresh_rates()
                    .iter()
                    .map(u32::to_string)
                    .collect::<Vec<_>>()
                    .join(", ");
                let json = format!(
                    "{{\n  \"case\": \"{case_name}\",\n  \"direct_composition\": {direct_composition},\n  \"displays\": {displays},\n  \"refresh_rates_hz\": [{refresh_rates}],\n  \"elapsed_seconds\": {elapsed:.3},\n  \"process_cpu_ms\": {:.3},\n  \"process_cycles\": {},\n  \"main_thread_cycles\": {},\n  \"clock_threads\": {},\n  \"clock_thread_cycles\": {},\n  \"clock_thread_wakes\": {},\n  \"renders\": {}{counters}{verify_report}\n}}\n",
                    (after.process_time_100ns - before.process_time_100ns) as f64 / 10_000.0,
                    after.process_cycles - before.process_cycles,
                    after.main_thread_cycles - before.main_thread_cycles,
                    clock_threads.0.len(),
                    after.clock_thread_cycles - before.clock_thread_cycles,
                    after.clock_thread_wakes - before.clock_thread_wakes,
                    after.renders - before.renders,
                );
                print!("{json}");
                std::fs::write(&output, json).expect("writing the result");
                if matches!(case, Case::Static | Case::TwoMonitors) {
                    assert_eq!(after.renders, before.renders, "a static window rendered");
                }
                cx.update(|cx| cx.quit());
            })
            .detach();
        });
    }
}
