//! Shows whether a window's first frame is laid out at the size it is shown
//! at.
//!
//! ```text
//! cargo run -p gpui_platform --example windows_first_frame -- <windowed|maximized> [--seconds N] [--then maximize|restore|resize]
//! ```
//!
//! Opens a window the way an app restoring its last placement does: hidden,
//! with client decorations, then activated once `open_window` has drawn its
//! first frame. The content marks its own right and bottom edges with 6 px
//! bars and prints the viewport it was laid out for, so a screen capture of
//! the first frames shows a frame laid out for another size at once: a bar
//! short of the window's edge, or none. Each render's viewport is also
//! printed to stderr, and the example exits with status 1 if any render was
//! laid out for another size than the window was shown at. With `--then`,
//! the window is maximized, restored or resized a second after it opens,
//! and each render must be at the size it was shown at or the size it ends
//! at.

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
        let usage = "usage: windows_first_frame <windowed|maximized> [--seconds N] [--then maximize|restore|resize]";
        let maximized = match arguments.next().as_deref() {
            Some("windowed") => false,
            Some("maximized") => true,
            _ => panic!("{usage}"),
        };
        let mut seconds = 4.0;
        let mut then: Option<String> = None;
        while let Some(argument) = arguments.next() {
            match argument.as_str() {
                "--then" => then = Some(arguments.next().expect(usage)),
                "--seconds" => {
                    seconds = arguments
                        .next()
                        .and_then(|value| value.parse().ok())
                        .expect(usage)
                }
                _ => panic!("{usage}"),
            }
        }
        let then_given = then.is_some();
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
            let then = then.clone();
            cx.spawn(async move |cx| {
                if let Some(then) = then {
                    cx.background_executor().timer(Duration::from_secs(1)).await;
                    window
                        .update(cx, |_, window, _| match then.as_str() {
                            "maximize" | "restore" => window.zoom_window(),
                            "resize" => window.resize(size(px(1700.), px(1000.))),
                            _ => panic!("unknown --then {then}"),
                        })
                        .ok();
                }
                cx.background_executor()
                    .timer(Duration::from_secs_f64(seconds))
                    .await;
                cx.update(|cx| cx.quit());
            })
            .detach();
        });
        let viewports = laid_out.borrow();
        // The sizes laid out for, in order: the size the window was shown at,
        // and with `--then` the size it ended at. Any other is a frame laid
        // out for a size the window never showed.
        let mut sizes = viewports.clone();
        sizes.dedup();
        let expected = if then_given { 2 } else { 1 };
        let other = sizes.len().saturating_sub(expected);
        eprintln!(
            "{} renders at {:?}; {other} at a size the window was not shown at",
            viewports.len(),
            sizes
                .iter()
                .map(|size| format!("{}x{}", size.width.as_f32(), size.height.as_f32()))
                .collect::<Vec<_>>()
        );
        if other > 0 {
            std::process::exit(1);
        }
    }
}
