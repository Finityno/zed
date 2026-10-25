//! Shows whether a window's first frame is laid out at the size it is shown
//! at.
//!
//! ```text
//! cargo run -p gpui_platform --example windows_first_frame -- <windowed|maximized> [--seconds N]
//! ```
//!
//! Opens a window the way an app restoring its last placement does: hidden,
//! with client decorations, then activated once `open_window` has drawn its
//! first frame. The content marks its own right and bottom edges with 6 px
//! bars and prints the viewport it was laid out for, so a screen capture of
//! the first frames shows a frame laid out for another size at once: a bar
//! short of the window's edge, or none. Each render's viewport is also
//! printed to stderr, and the example exits with status 1 if any render was
//! laid out for another size than the window was shown at.

fn main() {
    #[cfg(target_os = "windows")]
    first_frame::run();
    #[cfg(not(target_os = "windows"))]
    eprintln!("This example requires Windows.");
}

#[cfg(target_os = "windows")]
mod first_frame {
    use gpui::{
        App, AppContext as _, Bounds, Context, IntoElement, ParentElement as _, Render,
        Styled as _, TitlebarOptions, Window, WindowBounds, WindowDecorations, WindowOptions, div,
        px, rgb, size,
    };
    use std::{
        cell::RefCell,
        rc::Rc,
        time::{Duration, Instant},
    };

    struct Edges {
        started: Instant,
        viewports: Rc<RefCell<Vec<gpui::Size<gpui::Pixels>>>>,
    }

    impl Render for Edges {
        fn render(&mut self, window: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
            let viewport = window.viewport_size();
            self.viewports.borrow_mut().push(viewport);
            eprintln!(
                "render at {:>5} ms: viewport {}x{}",
                self.started.elapsed().as_millis(),
                viewport.width.as_f32(),
                viewport.height.as_f32()
            );
            div()
                .size_full()
                .relative()
                .bg(rgb(0x202124))
                .text_color(rgb(0xe8e8e8))
                .text_size(px(28.))
                .p_8()
                .child(format!(
                    "laid out for {} x {}",
                    viewport.width.as_f32(),
                    viewport.height.as_f32()
                ))
                .child(div().absolute().top_0().bottom_0().right_0().w(px(6.)).bg(rgb(0xff3030)))
                .child(div().absolute().left_0().right_0().bottom_0().h(px(6.)).bg(rgb(0x30ff60)))
        }
    }

    pub fn run() {
        let mut arguments = std::env::args().skip(1);
        let usage = "usage: windows_first_frame <windowed|maximized> [--seconds N]";
        let maximized = match arguments.next().as_deref() {
            Some("windowed") => false,
            Some("maximized") => true,
            _ => panic!("{usage}"),
        };
        let mut seconds = 4.0;
        while let Some(argument) = arguments.next() {
            match argument.as_str() {
                "--seconds" => {
                    seconds = arguments
                        .next()
                        .and_then(|value| value.parse().ok())
                        .expect(usage)
                }
                _ => panic!("{usage}"),
            }
        }
        let started = Instant::now();
        let viewports: Rc<RefCell<Vec<gpui::Size<gpui::Pixels>>>> = Rc::default();
        let laid_out = viewports.clone();
        gpui_platform::application().run(move |cx: &mut App| {
            let viewports = viewports.clone();
            let bounds = Bounds::centered(None, size(px(1400.), px(900.)), cx);
            let window = cx
                .open_window(
                    WindowOptions {
                        window_bounds: Some(if maximized {
                            WindowBounds::Maximized(bounds)
                        } else {
                            WindowBounds::Windowed(bounds)
                        }),
                        titlebar: Some(TitlebarOptions {
                            title: Some("GPUI first frame".into()),
                            appears_transparent: true,
                            traffic_light_position: None,
                        }),
                        focus: false,
                        show: false,
                        window_decorations: Some(WindowDecorations::Client),
                        ..Default::default()
                    },
                    move |_, cx| cx.new(|_| Edges { started, viewports }),
                )
                .expect("opening the window");
            eprintln!("opened at {} ms; activating", started.elapsed().as_millis());
            window
                .update(cx, |_, window, _| window.activate_window())
                .ok();
            cx.spawn(async move |cx| {
                cx.background_executor()
                    .timer(Duration::from_secs_f64(seconds))
                    .await;
                cx.update(|cx| cx.quit());
            })
            .detach();
        });
        let viewports = laid_out.borrow();
        let shown = viewports.last().copied();
        let other = viewports.iter().filter(|viewport| Some(**viewport) != shown).count();
        eprintln!(
            "{} renders, {other} laid out for another size than the window was shown at",
            viewports.len()
        );
        if other > 0 {
            std::process::exit(1);
        }
    }
}
