//! Memory a window keeps after a long streaming session whose text keeps
//! changing size (zoom steps, headings, code) and that shows images as it
//! goes, drawn headlessly through the real wgpu renderer and cosmic-text
//! shaper (on lavapipe the atlas pages are host memory).
//!
//! VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json \
//!   cargo run -p gpui_wgpu --features test-support --example long_session_memory
//!
//! `FRAMES` (default 2400) frames stream words drawn from Latin, Greek and
//! Cyrillic, a new message every 40 frames; every `ZOOM_EVERY` (default 100)
//! frames all text grows by a pixel, and every `IMAGE_EVERY`-th message
//! (default 0, none) carries its own 512x512 image. Prints RSS, RSS after
//! `malloc_trim`, the atlas pages' bytes and how many glyphs' raster bounds
//! the text system remembers, every `REPORT_EVERY` (default 600) frames.

use gpui::{
    AppContext as _, Context, HeadlessAppContext, IntoElement, ListAlignment, ListState,
    ParentElement as _, Render, RenderImage, SharedString, Styled as _, Window, WindowHandle, div,
    hsla, img, list, prelude::FluentBuilder as _, px, size,
};
use gpui_wgpu::{CosmicTextSystem, WgpuHeadlessRenderer};
use std::{borrow::Cow, sync::Arc};

const LILEX: &[u8] = include_bytes!("../../../assets/fonts/lilex/Lilex-Regular.ttf");
const IBM_PLEX: &[u8] =
    include_bytes!("../../../assets/fonts/ibm-plex-sans/IBMPlexSans-Regular.ttf");

/// Character ranges the streamed words draw from: Latin, Latin-1, Latin
/// Extended-A, Greek and Cyrillic, all covered by the bundled fonts.
const RANGES: &[(u32, u32)] = &[
    (0x61, 0x7a),
    (0x41, 0x5a),
    (0x30, 0x39),
    (0xc0, 0xff),
    (0x100, 0x17f),
    (0x391, 0x3c9),
    (0x410, 0x44f),
];
const SIZES: &[f32] = &[12., 13., 14., 15., 16., 18., 20., 24.];

fn word(seed: usize) -> String {
    let (start, end) = RANGES[(seed / 3) % RANGES.len()];
    let span = (end - start + 1) as usize;
    (0..3 + seed % 6)
        .filter_map(|index| char::from_u32(start + ((seed * 31 + index * 17) % span) as u32))
        .collect()
}

struct Message {
    font_size: f32,
    code: bool,
    text: String,
    image: Option<Arc<RenderImage>>,
}

struct Stream {
    messages: Vec<Message>,
    list: ListState,
    zoom: f32,
    image_every: usize,
}

fn image(seed: usize) -> Arc<RenderImage> {
    let pixel = image::Rgba([(seed * 37) as u8, (seed * 91) as u8, 128, 255]);
    let frame = image::Frame::new(image::RgbaImage::from_pixel(512, 512, pixel));
    Arc::new(RenderImage::new(vec![frame]))
}

impl Stream {
    fn step(&mut self, step: usize, cx: &mut Context<Self>) {
        if step.is_multiple_of(40) {
            let index = step / 40;
            self.messages.push(Message {
                font_size: SIZES[index % SIZES.len()],
                code: index % 5 == 4,
                text: String::new(),
                image: (self.image_every > 0 && index.is_multiple_of(self.image_every))
                    .then(|| image(index)),
            });
            self.list
                .splice(self.messages.len() - 1..self.messages.len() - 1, 1);
        }
        let Some(message) = self.messages.last_mut() else {
            return;
        };
        if !message.text.is_empty() {
            message.text.push(' ');
        }
        message.text.push_str(&word(step));
        let last = self.messages.len() - 1;
        self.list.remeasure_items(last..last + 1);
        cx.notify();
    }
}

impl Render for Stream {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let this = cx.entity().downgrade();
        let zoom = self.zoom;
        div()
            .size_full()
            .font_family("IBM Plex Sans")
            .text_color(hsla(0., 0., 0.1, 1.))
            .bg(hsla(0., 0., 1., 1.))
            .child(
                list(self.list.clone(), move |index, _window, cx| {
                    let Some(this) = this.upgrade() else {
                        return div().into_any_element();
                    };
                    let message = &this.read(cx).messages[index];
                    let text: SharedString = message.text.clone().into();
                    div()
                        .px_4()
                        .py_2()
                        .text_size(px(message.font_size + zoom))
                        .when(message.code, |this| this.font_family("Lilex"))
                        .child(text)
                        .when_some(message.image.clone(), |this, image| {
                            this.child(img(image).w(px(256.)).h(px(256.)))
                        })
                        .into_any_element()
                })
                .size_full(),
            )
    }
}

fn rss_kib() -> u64 {
    let statm = std::fs::read_to_string("/proc/self/statm").unwrap_or_default();
    let pages: u64 = statm
        .split_whitespace()
        .nth(1)
        .and_then(|field| field.parse().ok())
        .unwrap_or(0);
    pages * 4
}

unsafe extern "C" {
    fn malloc_trim(pad: usize) -> i32;
}

fn env_or(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn draw(cx: &mut HeadlessAppContext, window: WindowHandle<Stream>) {
    cx.update_window(window.into(), |_, window, cx| {
        window.draw(cx).clear(cx);
        window.present_if_needed();
    })
    .expect("window draws");
}

fn report(cx: &mut HeadlessAppContext, frame: usize) {
    let raster_bounds = cx.update(|cx| cx.text_system().remembered_raster_bounds_count());
    let gauges = gpui::render_memory_gauges();
    let atlas_kib = (gauges.atlas_monochrome_bytes + gauges.atlas_polychrome_bytes) / 1024;
    let rss = rss_kib();
    // SAFETY: glibc's malloc_trim only returns free heap pages to the system.
    unsafe { malloc_trim(0) };
    println!(
        "frame={frame} rss_kib={rss} rss_trimmed_kib={} atlas_kib={atlas_kib} \
         raster_bounds={raster_bounds}",
        rss_kib()
    );
}

fn main() -> anyhow::Result<()> {
    let frames = env_or("FRAMES", 2400);
    let zoom_every = env_or("ZOOM_EVERY", 100).max(1);
    let report_every = env_or("REPORT_EVERY", 600).max(1);
    let image_every = env_or("IMAGE_EVERY", 0);
    let text_system = CosmicTextSystem::new_without_system_fonts("IBM Plex Sans");
    gpui::PlatformTextSystem::add_fonts(
        &text_system,
        vec![Cow::Borrowed(LILEX), Cow::Borrowed(IBM_PLEX)],
    )?;
    let mut cx = HeadlessAppContext::with_platform(Arc::new(text_system), Arc::new(()), || {
        Ok(Some(Box::new(WgpuHeadlessRenderer::new()?)))
    });
    let window = cx.open_window(size(px(900.), px(1100.)), |_, cx| {
        cx.new(|_| Stream {
            messages: Vec::new(),
            list: ListState::new(0, ListAlignment::Bottom, px(1000.)),
            zoom: 0.,
            image_every,
        })
    })?;
    draw(&mut cx, window);
    report(&mut cx, 0);

    let stream = window.root(&mut cx)?;
    for step in 1..=frames {
        cx.update(|cx| {
            stream.update(cx, |stream, cx| {
                if step.is_multiple_of(zoom_every) {
                    stream.zoom += 1.;
                }
                stream.step(step, cx);
            })
        });
        draw(&mut cx, window);
        if step.is_multiple_of(report_every) {
            report(&mut cx, step);
        }
    }
    Ok(())
}
