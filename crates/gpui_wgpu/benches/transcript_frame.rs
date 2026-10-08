//! Frame cost of a chat-transcript-shaped window, drawn headlessly with the
//! real cosmic-text shaper. Prints per-frame wall time and heap allocation
//! counts for streaming, scrolling and unchanged redraws.
//!
//! cargo bench -p gpui_wgpu --features test-support --bench transcript_frame

use gpui::{
    App, AppContext as _, Context, FontWeight, HighlightStyle, InteractiveElement as _,
    IntoElement, ListAlignment, ListState, ParentElement as _, Render, SharedString,
    Styled as _, StyledText, TestAppContext, TestDispatcher,
    Window, WindowHandle, div, hsla, list, px, size,
};
use gpui_wgpu::CosmicTextSystem;
use std::{
    alloc::{GlobalAlloc, Layout, System},
    borrow::Cow,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering::Relaxed},
    },
    time::{Duration, Instant},
};

struct CountingAllocator;
static ALLOCATIONS: AtomicU64 = AtomicU64::new(0);
static ALLOCATED_BYTES: AtomicU64 = AtomicU64::new(0);

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Relaxed);
        ALLOCATED_BYTES.fetch_add(layout.size() as u64, Relaxed);
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Relaxed);
        ALLOCATED_BYTES.fetch_add(new_size as u64, Relaxed);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

const LILEX: &[u8] = include_bytes!("../../../assets/fonts/lilex/Lilex-Regular.ttf");
const IBM_PLEX: &[u8] =
    include_bytes!("../../../assets/fonts/ibm-plex-sans/IBMPlexSans-Regular.ttf");

const WORDS: &[&str] = &[
    "the", "render", "loop", "keeps", "every", "frame", "under", "budget", "while", "streaming",
    "tokens", "arrive", "from", "model", "and", "layout", "wraps", "paragraphs", "across",
    "narrow", "panes", "with", "inline", "code", "spans", "like", "`shape_text`", "or",
    "`ListState`", "that", "measure", "rows",
];

fn paragraph(seed: usize, words: usize) -> String {
    let mut text = String::new();
    for index in 0..words {
        if index > 0 {
            text.push(' ');
        }
        text.push_str(WORDS[(seed * 7 + index * 13) % WORDS.len()]);
    }
    text.push('.');
    text
}

struct Message {
    author: SharedString,
    paragraphs: Vec<SharedString>,
    code: Vec<SharedString>,
}

fn message(index: usize) -> Message {
    let paragraph_count = 1 + index % 4;
    Message {
        author: if index % 2 == 0 { "You".into() } else { "Assistant".into() },
        paragraphs: (0..paragraph_count)
            .map(|paragraph_index| paragraph(index * 5 + paragraph_index, 20 + (index * 3 + paragraph_index * 11) % 40).into())
            .collect(),
        code: if index % 3 == 1 {
            (0..12)
                .map(|line| format!("    let value_{line} = compute(input, {line}) + offset;").into())
                .collect()
        } else {
            Vec::new()
        },
    }
}

/// Bold every other backtick span, the way markdown emphasis runs land.
fn highlights(text: &str) -> Vec<(std::ops::Range<usize>, HighlightStyle)> {
    let mut highlights = Vec::new();
    let mut start = None;
    for (offset, character) in text.char_indices() {
        if character == '`' {
            match start.take() {
                Some(begin) => highlights.push((
                    begin..offset + 1,
                    HighlightStyle {
                        font_weight: Some(FontWeight::BOLD),
                        color: Some(hsla(0.6, 0.5, 0.5, 1.)),
                        ..Default::default()
                    },
                )),
                None => start = Some(offset),
            }
        }
    }
    highlights
}

struct Transcript {
    messages: Vec<Message>,
    list: ListState,
}

impl Transcript {
    fn new(count: usize) -> Self {
        let messages: Vec<Message> = (0..count).map(message).collect();
        let list = ListState::new(messages.len(), ListAlignment::Bottom, px(1000.));
        Self { messages, list }
    }

    fn stream(&mut self, step: usize, cx: &mut Context<Self>) {
        let last = self.messages.len() - 1;
        let Some(tail) = self.messages[last].paragraphs.last_mut() else {
            return;
        };
        let mut text = tail.to_string();
        text.push(' ');
        text.push_str(WORDS[step % WORDS.len()]);
        if step % 24 == 23 {
            text.push('.');
            self.messages[last].paragraphs.push(paragraph(step, 1).into());
        } else {
            *self.messages[last].paragraphs.last_mut().unwrap() = text.into();
        }
        self.list.remeasure_items(last..last + 1);
        cx.notify();
    }

    fn render_message(&self, index: usize) -> gpui::AnyElement {
        let message = &self.messages[index];
        div()
            .id(index)
            .w_full()
            .px_4()
            .py_2()
            .flex()
            .flex_col()
            .gap_2()
            .hover(|style| style.bg(hsla(0., 0., 0.5, 0.05)))
            .child(
                div()
                    .flex()
                    .gap_2()
                    .text_sm()
                    .text_color(hsla(0., 0., 0.4, 1.))
                    .child(message.author.clone())
                    .child(div().id("copy").px_1().hover(|style| style.bg(hsla(0., 0., 0.5, 0.2))).child("Copy")),
            )
            .children(message.paragraphs.iter().map(|text| {
                StyledText::new(text.clone()).with_highlights(highlights(text))
            }))
            .when(!message.code.is_empty(), |this| {
                this.child(
                    div()
                        .font_family("Lilex")
                        .text_sm()
                        .p_2()
                        .rounded_md()
                        .bg(hsla(0., 0., 0.95, 1.))
                        .children(message.code.iter().map(|line| div().child(line.clone()))),
                )
            })
            .into_any_element()
    }
}

use gpui::prelude::FluentBuilder as _;

impl Render for Transcript {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let this = cx.entity().downgrade();
        div()
            .size_full()
            .font_family("IBM Plex Sans")
            .text_color(hsla(0., 0., 0.1, 1.))
            .bg(hsla(0., 0., 1., 1.))
            .child(
                list(self.list.clone(), move |index, _window, cx: &mut App| {
                    this.upgrade()
                        .map(|this| this.read(cx).render_message(index))
                        .unwrap_or_else(|| div().into_any_element())
                })
                .size_full(),
            )
    }
}

struct Sample {
    times: Vec<Duration>,
    allocations: Vec<u64>,
    bytes: Vec<u64>,
}

impl Sample {
    fn report(mut self, name: &str) {
        self.times.sort();
        self.allocations.sort();
        self.bytes.sort();
        let count = self.times.len();
        let median = |values: &[Duration]| values[values.len() / 2];
        let p90 = |values: &[Duration]| values[values.len() * 9 / 10];
        println!(
            "{name:<22} frames={count:<4} median={:>8.1}us p90={:>8.1}us allocs/frame={:>7} bytes/frame={:>9}",
            median(&self.times).as_secs_f64() * 1e6,
            p90(&self.times).as_secs_f64() * 1e6,
            self.allocations[count / 2],
            self.bytes[count / 2],
        );
    }
}

fn draw(cx: &mut TestAppContext, window: WindowHandle<Transcript>) -> (Duration, u64, u64) {
    cx.update_window(window.into(), |_, window, cx| {
        let allocations = ALLOCATIONS.load(Relaxed);
        let bytes = ALLOCATED_BYTES.load(Relaxed);
        let start = Instant::now();
        window.draw(cx).clear(cx);
        let elapsed = start.elapsed();
        (
            elapsed,
            ALLOCATIONS.load(Relaxed) - allocations,
            ALLOCATED_BYTES.load(Relaxed) - bytes,
        )
    })
    .unwrap()
}

fn measure(
    cx: &mut TestAppContext,
    window: WindowHandle<Transcript>,
    frames: usize,
    mut change: impl FnMut(&mut TestAppContext, usize),
) -> Sample {
    let mut sample = Sample { times: Vec::new(), allocations: Vec::new(), bytes: Vec::new() };
    cx.update_window(window.into(), |_, window, _| window.reset_frame_work_stats(false)).ok();
    for frame in 0..frames {
        change(cx, frame);
        let (time, allocations, bytes) = draw(cx, window);
        sample.times.push(time);
        sample.allocations.push(allocations);
        sample.bytes.push(bytes);
    }
    if std::env::var("WORK").is_ok() {
        let work = cx.update_window(window.into(), |_, window, _| window.frame_work_stats()).unwrap();
        println!("{work:?}");
    }
    sample
}

fn main() {
    let frames: usize = std::env::var("FRAMES").ok().and_then(|value| value.parse().ok()).unwrap_or(400);
    let messages: usize = std::env::var("MESSAGES").ok().and_then(|value| value.parse().ok()).unwrap_or(400);
    let text_system = CosmicTextSystem::new_without_system_fonts("IBM Plex Sans");
    gpui::PlatformTextSystem::add_fonts(&text_system, vec![Cow::Borrowed(LILEX), Cow::Borrowed(IBM_PLEX)])
        .expect("fonts load");
    let mut cx = TestAppContext::build_with_text_system(TestDispatcher::new(0), None, Arc::new(text_system));
    if std::env::var("GPUI_RETAINED_VIEWS").as_deref() == Ok("1") {
        cx.update(|cx| cx.set_view_retention(true));
    }
    let window = cx.add_window(|_, _| Transcript::new(messages));
    cx.update_window(window.into(), |_, window, _| window.resize(size(px(800.), px(1000.)))).ok();
    for _ in 0..5 {
        draw(&mut cx, window);
    }

    let transcript = window.root(&mut cx).unwrap();
    measure(&mut cx, window, frames, |cx, step| {
        transcript.update(cx, |transcript, cx| transcript.stream(step, cx));
    })
    .report("stream_tail");

    measure(&mut cx, window, frames, |cx, _| {
        cx.update_window(window.into(), |_, window, _| window.refresh()).ok();
    })
    .report("redraw_unchanged");

    measure(&mut cx, window, frames, |cx, step| {
        let distance = if (step / 100) % 2 == 0 { px(-37.) } else { px(37.) };
        transcript.update(cx, |transcript, cx| {
            transcript.list.scroll_by(distance);
            cx.notify();
        });
    })
    .report("scroll");

    let widths = [800., 640., 520., 700.];
    measure(&mut cx, window, frames / 4, |cx, step| {
        let width = widths[step % widths.len()];
        cx.update_window(window.into(), |_, window, _| window.resize(size(px(width), px(1000.)))).ok();
    })
    .report("resize");
}
