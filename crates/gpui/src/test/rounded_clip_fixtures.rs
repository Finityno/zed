//! Pixel fixtures for rounded content clipping, rendered through a real GPU
//! renderer. Each renderer's tests run them with its headless renderer
//! installed in the [`TestAppContext`], so every backend is held to the same
//! pixels: a renderer that clips a rounded subtree differently from the
//! others shows up here rather than in a screenshot.

use crate::{
    AppContext as _, Context, Div, Entity, IntoElement, ParentElement as _, Path, Render,
    StyleRefinement, Styled as _, TestAppContext, Window, canvas, div, point, px, rgb, size,
};
use std::{cell::Cell, rc::Rc};

/// Every fixture is a 64 by 64 logical pixel square.
const SIDE: f32 = 64.;

const WHITE: u32 = 0xffffff;
const BLACK: u32 = 0x000000;
const RED: u32 = 0xff0000;
const GREEN: u32 = 0x00ff00;
const BLUE: u32 = 0x0000ff;

/// Channels may differ from the expected colour by this much, for the
/// renderers' colour-space round trips; antialiased edges are never sampled.
const TOLERANCE: i32 = 3;

/// Renders every fixture through `cx`'s headless renderer and describes each
/// pixel that came out wrong; empty when all of them pass.
pub fn check_rounded_clip_fixtures(cx: &mut TestAppContext) -> anyhow::Result<Vec<String>> {
    let mut failures = Vec::new();
    for fixture in FIXTURES {
        let pixels = render(cx, fixture.build)?;
        let mut check = Check {
            fixture: fixture.name,
            pixels: &pixels,
            failures: &mut failures,
        };
        (fixture.check)(&mut check);
    }
    check_identical_nested_clips(cx, &mut failures)?;
    check_cached_child_under_changing_radius(cx, &mut failures)?;
    Ok(failures)
}

struct Fixture {
    name: &'static str,
    build: fn() -> Div,
    check: fn(&mut Check),
}

const FIXTURES: &[Fixture] = &[
    Fixture {
        name: "square child under a rounded parent",
        build: || white_square().child(rounded_square(24.).child(square(SIDE, BLACK))),
        check: |check| {
            check.expect(32., 32., BLACK, "centre");
            check.expect(1., 1., WHITE, "the square child leaked out of the rounded corner");
            check.expect(62., 62., WHITE, "the square child leaked out of the rounded corner");
        },
    },
    Fixture {
        name: "visible overflow keeps a square child",
        build: || {
            white_square().child(
                div()
                    .size(px(SIDE))
                    .rounded(px(24.))
                    .child(square(SIDE, BLACK)),
            )
        },
        check: |check| check.expect(1., 1., BLACK, "visible overflow clipped the child"),
    },
    Fixture {
        name: "inset child under a rounded parent",
        build: || {
            white_square().child(
                rounded_square(24.).p(px(16.)).child(
                    div()
                        .size(px(32.))
                        .overflow_hidden()
                        .child(square(32., BLACK)),
                ),
            )
        },
        check: |check| {
            // The inset child's own corner lies well inside the parent's
            // shape: it must stay square rather than take the parent's radius.
            check.expect(17., 17., BLACK, "the inset child took its ancestor's radius");
            check.expect(46., 46., BLACK, "the inset child took its ancestor's radius");
            check.expect(15., 15., WHITE, "padding");
        },
    },
    Fixture {
        name: "rounded child cropped by a shorter square ancestor",
        build: || {
            white_square().child(
                div().w(px(SIDE)).h(px(32.)).overflow_hidden().child(
                    div()
                        .w(px(SIDE))
                        .h(px(128.))
                        .rounded(px(24.))
                        .overflow_hidden()
                        .child(div().w(px(SIDE)).h(px(128.)).bg(rgb(BLACK))),
                ),
            )
        },
        check: |check| {
            // The child's bottom corners are far below the crop; cropping
            // must not move them up to the ancestor's bottom edge.
            check.expect(1., 30., BLACK, "the crop moved the child's bottom corner up");
            check.expect(1., 34., WHITE, "the crop let the child through");
            check.expect(1., 1., WHITE, "the child's top corner");
        },
    },
    Fixture {
        name: "overlapping rounded siblings",
        build: || {
            white_square()
                .relative()
                .child(
                    div()
                        .absolute()
                        .size(px(48.))
                        .rounded(px(16.))
                        .overflow_hidden()
                        .child(square(48., RED)),
                )
                .child(
                    div()
                        .absolute()
                        .left(px(24.))
                        .top(px(16.))
                        .size(px(32.))
                        .rounded(px(12.))
                        .overflow_hidden()
                        .child(square(32., BLACK)),
                )
        },
        check: |check| {
            check.expect(40., 32., BLACK, "the later sibling");
            // Inside the later sibling's bounds but outside its rounded
            // corner, the earlier sibling shows through: one sibling's clip
            // must not mask the other's pixels.
            check.expect(25., 17., RED, "the later sibling's clip masked the earlier one");
            check.expect(1., 1., WHITE, "the earlier sibling's rounded corner");
        },
    },
    Fixture {
        name: "many small rounded siblings",
        build: || {
            white_square().flex().flex_wrap().children((0..64).map(|_| {
                div()
                    .size(px(8.))
                    .rounded(px(3.))
                    .overflow_hidden()
                    .child(square(8., BLACK))
            }))
        },
        check: |check| {
            for row in 0..8 {
                for column in 0..8 {
                    let (x, y) = (column as f32 * 8., row as f32 * 8.);
                    check.expect(x, y, WHITE, "a sibling's corner");
                    check.expect(x + 4., y + 4., BLACK, "a sibling's centre");
                }
            }
        },
    },
    Fixture {
        name: "a hundred nested rounded clips",
        build: || {
            let mut child = square(SIDE, BLACK);
            // Distinct radii, so no two scopes can be merged as identical.
            for depth in 0..100 {
                child = div()
                    .size(px(SIDE))
                    .rounded(px(20. + depth as f32 * 0.01))
                    .overflow_hidden()
                    .child(child);
            }
            white_square().child(child)
        },
        check: |check| {
            check.expect(0., 0., WHITE, "corner");
            check.expect(32., 32., BLACK, "centre");
        },
    },
    Fixture {
        name: "unequal borders clip at the padding edge",
        build: || {
            let mut parent = rounded_square(24.).border_color(rgb(BLUE));
            parent.style().border_widths.left = Some(px(8.).into());
            parent.style().border_widths.top = Some(px(4.).into());
            parent.style().border_widths.right = Some(px(2.).into());
            parent.style().border_widths.bottom = Some(px(6.).into());
            white_square().child(parent.child(square(SIDE, BLACK)))
        },
        check: |check| {
            check.expect(1., 1., WHITE, "outside the outer corner");
            check.expect(32., 2., BLUE, "the top border");
            check.expect(4., 32., BLUE, "the left border");
            check.expect(63., 32., BLUE, "the right border");
            check.expect(32., 61., BLUE, "the bottom border");
            check.expect(32., 5., BLACK, "inside the top padding edge");
            check.expect(9., 32., BLACK, "inside the left padding edge");
            check.expect(61., 32., BLACK, "inside the right padding edge");
        },
    },
    Fixture {
        name: "an empty padding box hides every descendant",
        build: || {
            let mut parent = rounded_square(24.).border_color(rgb(BLUE));
            parent.style().border_widths.left = Some(px(40.).into());
            parent.style().border_widths.right = Some(px(40.).into());
            white_square().child(parent.child(square(SIDE, BLACK)))
        },
        check: |check| check.expect_none(BLACK, "a descendant drew in an empty padding box"),
    },
    Fixture {
        name: "paths under a rounded parent",
        build: || {
            white_square().child(
                rounded_square(24.).child(
                    canvas(
                        |_, _, _| (),
                        |bounds, (), window, _| {
                            let mut path = Path::new(bounds.origin);
                            path.line_to(bounds.top_right());
                            path.line_to(bounds.bottom_right());
                            path.line_to(bounds.bottom_left());
                            path.line_to(bounds.origin);
                            window.paint_path(path, rgb(BLACK));
                        },
                    )
                    .size(px(SIDE)),
                ),
            )
        },
        check: |check| {
            check.expect(32., 32., BLACK, "centre");
            check.expect(1., 1., WHITE, "the path leaked out of the rounded corner");
        },
    },
    Fixture {
        name: "a rounded clip between two path batches",
        build: || {
            white_square()
                .relative()
                .child(corner_path(0., RED))
                .child(
                    div()
                        .absolute()
                        .size(px(SIDE))
                        .rounded(px(24.))
                        .overflow_hidden()
                        .child(square(SIDE, BLACK)),
                )
                .child(corner_path(48., GREEN))
        },
        check: |check| {
            check.expect(1., 1., RED, "the earlier path");
            check.expect(62., 1., GREEN, "the later path was clipped by the scope before it");
            check.expect(32., 32., BLACK, "centre");
        },
    },
    Fixture {
        name: "command palette rows keep square corners",
        build: || {
            let row = |height: f32, color: u32| {
                div()
                    .w_full()
                    .h(px(height))
                    .flex_shrink_0()
                    .overflow_hidden()
                    .bg(rgb(color))
            };
            // Only the palette is rounded; its input and selected row clip
            // to plain rectangles of their own.
            white_square().child(
                rounded_square(16.)
                    .flex()
                    .flex_col()
                    .child(row(16., BLACK))
                    .child(row(16., RED))
                    .child(row(8., BLUE))
                    .child(row(8., BLACK))
                    .child(row(16., GREEN)),
            )
        },
        check: |check| {
            for x in [0., 63.] {
                for y in [16., 31.] {
                    check.expect(x, y, RED, "the input took the palette's radius");
                }
                for y in [32., 39.] {
                    check.expect(x, y, BLUE, "the selected row took the palette's radius");
                }
                check.expect(x, 0., WHITE, "the palette's top corner");
                check.expect(x, 63., WHITE, "the palette's bottom corner");
            }
            check.expect(32., 0., BLACK, "the palette's top edge");
            check.expect(32., 63., GREEN, "the palette's bottom edge");
        },
    },
];

fn white_square() -> Div {
    div().size(px(SIDE)).bg(rgb(WHITE))
}

fn rounded_square(radius: f32) -> Div {
    div().size(px(SIDE)).rounded(px(radius)).overflow_hidden()
}

fn square(side: f32, color: u32) -> Div {
    div().size(px(side)).bg(rgb(color))
}

/// A 16 by 16 path in the top row, `left` pixels from the left edge, drawn
/// as a batch of its own.
fn corner_path(left: f32, color: u32) -> impl IntoElement {
    canvas(
        |_, _, _| (),
        move |bounds, (), window, _| {
            let start = bounds.origin + point(px(left), px(0.));
            let mut path = Path::new(start);
            path.line_to(start + point(px(16.), px(0.)));
            path.line_to(start + point(px(16.), px(16.)));
            path.line_to(start + point(px(0.), px(16.)));
            path.line_to(start);
            window.paint_path(path, rgb(color));
        },
    )
    .absolute()
    .size(px(SIDE))
}

struct FixturePixels {
    image: image::RgbaImage,
    scale_factor: f32,
}

impl FixturePixels {
    /// The first device pixel of the logical pixel at (`x`, `y`).
    fn at(&self, x: f32, y: f32) -> [u8; 4] {
        let device = |logical: f32| (logical * self.scale_factor) as u32;
        let (x, y) = (device(x), device(y));
        if x >= self.image.width() || y >= self.image.height() {
            return [0; 4];
        }
        self.image.get_pixel(x, y).0
    }
}

struct Check<'a> {
    fixture: &'static str,
    pixels: &'a FixturePixels,
    failures: &'a mut Vec<String>,
}

impl Check<'_> {
    fn expect(&mut self, x: f32, y: f32, color: u32, what: &str) {
        let actual = self.pixels.at(x, y);
        if !matches(actual, color) {
            self.failures.push(format!(
                "{}: ({x}, {y}) is {actual:?}, not #{color:06x}: {what}",
                self.fixture
            ));
        }
    }

    fn expect_none(&mut self, color: u32, what: &str) {
        let found = self
            .pixels
            .image
            .enumerate_pixels()
            .find(|(_, _, pixel)| matches(pixel.0, color));
        if let Some((x, y, _)) = found {
            self.failures.push(format!(
                "{}: device pixel ({x}, {y}) is #{color:06x}: {what}",
                self.fixture
            ));
        }
    }
}

fn matches(actual: [u8; 4], color: u32) -> bool {
    let expected = rgb(color);
    let channels = [expected.r, expected.g, expected.b, expected.a]
        .map(|channel| (channel * 255.).round() as i32);
    actual
        .iter()
        .zip(channels)
        .all(|(actual, expected)| (*actual as i32 - expected).abs() <= TOLERANCE)
}

struct FixtureView(fn() -> Div);

impl Render for FixtureView {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        (self.0)()
    }
}

fn render(cx: &mut TestAppContext, build: fn() -> Div) -> anyhow::Result<FixturePixels> {
    let window = cx.add_window(move |_, _| FixtureView(build));
    cx.update_window(window.into(), |_, window, cx| {
        window.resize(size(px(SIDE), px(SIDE)));
        window.draw(cx).clear(cx);
        Ok(FixturePixels {
            image: window.render_to_image()?,
            scale_factor: window.scale_factor(),
        })
    })?
}

/// A subtree clipped twice by the same shape must come out exactly as one
/// clipped once: coverage applied twice would darken the antialiased curve.
fn check_identical_nested_clips(
    cx: &mut TestAppContext,
    failures: &mut Vec<String>,
) -> anyhow::Result<()> {
    let single = render(cx, || {
        white_square().child(rounded_square(24.).child(square(SIDE, BLACK)))
    })?;
    let nested = render(cx, || {
        white_square()
            .child(rounded_square(24.).child(rounded_square(24.).child(square(SIDE, BLACK))))
    })?;
    let mismatch = single
        .image
        .enumerate_pixels()
        .zip(nested.image.pixels())
        .find(|((_, _, single), nested)| {
            single
                .0
                .iter()
                .zip(nested.0)
                .any(|(single, nested)| (*single as i32 - nested as i32).abs() > TOLERANCE)
        });
    if let Some(((x, y, single), nested)) = mismatch {
        failures.push(format!(
            "identical nested rounded clips: device pixel ({x}, {y}) is {:?} under two clips \
             and {:?} under one",
            nested.0, single.0
        ));
    }
    Ok(())
}

struct CachedChild(Rc<Cell<usize>>);

impl Render for CachedChild {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        self.0.set(self.0.get() + 1);
        square(SIDE, BLACK)
    }
}

struct RadiusParent {
    radius: f32,
    child: Entity<CachedChild>,
}

impl Render for RadiusParent {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        white_square().child(
            rounded_square(self.radius)
                .child(self.child.clone().cached(StyleRefinement::default().size_full())),
        )
    }
}

/// Only the ancestor changes, its radius or nothing at all, around a cached
/// child. The child's cache is keyed on the rounded clip it is drawn under, so
/// it is drawn again exactly when the radius changes and replayed otherwise,
/// and either way it must show the current clip, not the one before.
fn check_cached_child_under_changing_radius(
    cx: &mut TestAppContext,
    failures: &mut Vec<String>,
) -> anyhow::Result<()> {
    const FIXTURE: &str = "a cached child under an ancestor whose radius changes";
    let renders = Rc::new(Cell::new(0));
    let window = cx.add_window({
        let renders = renders.clone();
        move |_, cx| RadiusParent {
            radius: 24.,
            child: cx.new(|_| CachedChild(renders)),
        }
    });
    let radii = [24., 24., 0., 0., 24.];
    for radius in radii {
        window.update(cx, |parent, _, cx| {
            parent.radius = radius;
            cx.notify();
        })?;
        let pixels = cx.update_window(window.into(), |_, window, cx| {
            window.resize(size(px(SIDE), px(SIDE)));
            window.draw(cx).clear(cx);
            anyhow::Ok(FixturePixels {
                image: window.render_to_image()?,
                scale_factor: window.scale_factor(),
            })
        })??;
        let mut check = Check {
            fixture: FIXTURE,
            pixels: &pixels,
            failures,
        };
        let corner = if radius == 0. { BLACK } else { WHITE };
        check.expect(1., 1., corner, "the replayed child kept its previous clip");
        check.expect(32., 32., BLACK, "centre");
    }
    // Once at first, then once per change of radius.
    let expected_renders = 1 + radii.windows(2).filter(|pair| pair[0] != pair[1]).count();
    if renders.get() != expected_renders {
        failures.push(format!(
            "{FIXTURE}: the child rendered {} times, not {expected_renders}",
            renders.get()
        ));
    }
    Ok(())
}
