#![cfg_attr(target_family = "wasm", no_main)]

//! A frosted card over a busy backdrop: coloured stripes and text, blurred
//! beneath a rounded, half-transparent tint, as frosted menus paint it.

#[path = "example_support/fonts.rs"]
mod example_support;

use gpui::{
    App, Bounds, Context, Corners, Window, WindowBounds, WindowOptions, canvas, div, hsla,
    prelude::*, px, size,
};
use gpui_platform::application;

struct BackdropBlur;

impl Render for BackdropBlur {
    fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
        let stripes = (0..24).map(|index| {
            div()
                .w(px(25.))
                .h_full()
                .bg(hsla(index as f32 / 24., 0.7, 0.5, 1.))
        });
        let lines = (0..12).map(|index| {
            div()
                .text_color(hsla(0., 0., if index % 2 == 0 { 1. } else { 0. }, 1.))
                .child("The quick brown fox jumps over the lazy dog")
        });
        div()
            .size_full()
            .relative()
            .child(div().absolute().size_full().flex().children(stripes))
            .child(div().absolute().size_full().p_4().flex().flex_col().children(lines))
            .child(
                div()
                    .absolute()
                    .top(px(120.))
                    .left(px(110.))
                    .w(px(380.))
                    .h(px(260.))
                    .child(
                        canvas(
                            |_, _, _| {},
                            |bounds, _, window, _| {
                                let corners = Corners::all(px(16.));
                                window.paint_layer(bounds, |window| {
                                    window.paint_backdrop_blur(bounds, corners, px(8.), 1.4);
                                });
                                window.paint_quad(gpui::fill(bounds, hsla(0., 0., 0.1, 0.5)).corner_radii(corners));
                            },
                        )
                        .size_full(),
                    ),
            )
    }
}

fn run_example() {
    application().run(|cx: &mut App| {
        if !example_support::load_fonts(cx) {
            return;
        }
        let bounds = Bounds::centered(None, size(px(600.), px(500.)), cx);
        cx.open_window(
            WindowOptions {
                window_bounds: Some(WindowBounds::Windowed(bounds)),
                ..Default::default()
            },
            |_, cx| cx.new(|_| BackdropBlur),
        )
        .unwrap();
        cx.activate(true);
    });
}

#[cfg(not(target_family = "wasm"))]
fn main() {
    run_example();
}

#[cfg(target_family = "wasm")]
#[wasm_bindgen::prelude::wasm_bindgen(start)]
pub fn start() {
    gpui_platform::web_init();
    run_example();
}
