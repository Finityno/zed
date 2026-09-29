//! Frames laid out with layout nodes kept from earlier frames must be the
//! frames laid out from scratch.
//!
//! The oracle drives two windows holding the same view through the same
//! random history of changes. One keeps its layout nodes across frames; the
//! other has layout keys turned off, so every node is made afresh and dropped
//! at the end of each frame, as before nodes were kept. Every frame, the two
//! must paint the same primitives in the same places and leave the same
//! hitboxes.

use crate::{
    AnyElement, App, AvailableSpace, Bounds, Context, DevicePixels, Element, ElementId, Font,
    FontId, FontMetrics, FontRun, FrameWorkStats, GlobalElementId, GlyphId, Hsla,
    InspectorElementId, IntoElement, LayoutId, LineLayout, ListAlignment, ListOffset, ListState,
    NoopTextSystem, Pixels, PlatformTextSystem, Render, RenderGlyphParams, Result, SharedString,
    Size, Style, StyleRefinement, TestAppContext, TestDispatcher, TextRenderingMode,
    UniformListScrollHandle, Window, WindowHandle, div, hsla, list, point, prelude::*, px, size,
    uniform_list,
};
use rand::{Rng as _, SeedableRng as _, rngs::StdRng};
use std::{borrow::Cow, sync::Arc};

const WORDS: [&str; 12] = [
    "a",
    "grid",
    "cell",
    "ticking",
    "value",
    "with a longer label",
    "42",
    "17",
    "lorem ipsum dolor",
    "x",
    "a label long enough to be cut short",
    "sit amet",
];

const PALETTE: [Hsla; 5] = [
    hsla(0.0, 0.0, 0.1, 1.0),
    hsla(0.6, 0.7, 0.5, 1.0),
    hsla(0.3, 0.6, 0.4, 1.0),
    hsla(0.0, 0.8, 0.6, 1.0),
    hsla(0.1, 0.9, 0.5, 0.5),
];

const CELLS: usize = 16;
const ROW_HEIGHT: f32 = 20.;
const INITIAL_ROWS: u64 = 30;
const SEGMENT_HEIGHT: f32 = 18.;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct CellState {
    word: usize,
    color: usize,
    width: f32,
    background: bool,
    underline: bool,
    truncate: bool,
    hidden: bool,
}

#[derive(Clone, Copy, Debug)]
enum CellFlag {
    Background,
    Underline,
    Truncate,
    Hidden,
}

#[derive(Clone, Copy, Debug)]
enum RowIdentity {
    Position,
    Id,
    Wrapped,
}

#[derive(Clone, Debug)]
enum Change {
    Word { cell: usize, word: usize },
    Color { cell: usize, color: usize },
    Width { cell: usize, width: f32 },
    Toggle { cell: usize, flag: CellFlag },
    Paragraph { words: usize, width: f32 },
    InsertRow { at: usize },
    RemoveRow { at: usize },
    Scroll { top: f32 },
    InsertChip { at: usize },
    RemoveChip { at: usize },
    RotateChips { by: usize },
    RowIdentity(RowIdentity),
    Direction,
    Segment { at: usize, word: usize },
    SegmentCount { count: usize },
    SegmentWidth { width: f32 },
    Tint,
    Select { row: usize },
    Resize { width: f32, height: f32 },
    Redraw,
}

impl Change {
    fn random(rng: &mut StdRng) -> Self {
        let cell = rng.random_range(0..CELLS);
        match rng.random_range(0..100) {
            0..16 => Change::Word {
                cell,
                word: rng.random_range(0..WORDS.len()),
            },
            16..22 => Change::Color {
                cell,
                color: rng.random_range(0..PALETTE.len()),
            },
            22..28 => Change::Width {
                cell,
                width: rng.random_range(20.0..220.0),
            },
            28..37 => Change::Toggle {
                cell,
                flag: [
                    CellFlag::Background,
                    CellFlag::Underline,
                    CellFlag::Truncate,
                    CellFlag::Hidden,
                ][rng.random_range(0..4)],
            },
            37..43 => Change::Paragraph {
                words: rng.random_range(0..40),
                width: rng.random_range(60.0..400.0),
            },
            43..49 => Change::InsertRow {
                at: rng.random_range(0..64),
            },
            49..54 => Change::RemoveRow {
                at: rng.random_range(0..64),
            },
            54..62 => Change::Scroll {
                top: rng.random_range(0.0..400.0),
            },
            62..64 => Change::InsertChip {
                at: rng.random_range(0..16),
            },
            64..66 => Change::RemoveChip {
                at: rng.random_range(0..16),
            },
            66..67 => Change::RotateChips {
                by: rng.random_range(1..4),
            },
            67..70 => Change::RowIdentity(
                [RowIdentity::Position, RowIdentity::Id, RowIdentity::Wrapped]
                    [rng.random_range(0..3)],
            ),
            70..72 => Change::Direction,
            72..78 => Change::Segment {
                at: rng.random_range(0..8),
                word: rng.random_range(0..WORDS.len()),
            },
            78..80 => Change::SegmentCount {
                count: rng.random_range(0..8),
            },
            80..83 => Change::SegmentWidth {
                width: rng.random_range(40.0..300.0),
            },
            83..85 => Change::Tint,
            85..87 => Change::Select {
                row: rng.random_range(0..64),
            },
            87..90 => Change::Resize {
                width: rng.random_range(300.0..1000.0),
                height: rng.random_range(240.0..800.0),
            },
            _ => Change::Redraw,
        }
    }
}

/// A small application: plain cells, wrapping paragraphs, a row of chips,
/// a cached child view, an element laying segments out as roots of their
/// own in its prepaint, and the same rows in a uniform list and a list.
struct OracleView {
    cells: Vec<CellState>,
    paragraph_words: usize,
    paragraph_width: Pixels,
    rows: Vec<u64>,
    chips: Vec<u64>,
    next_row: u64,
    row_identity: RowIdentity,
    scroll_top: Pixels,
    column: bool,
    segments: Vec<usize>,
    segment_width: Pixels,
    tint: usize,
    /// The row that asks the list to scroll it into view when it is shown.
    selected: Option<u64>,
    badge: crate::Entity<Badge>,
    uniform_scroll: UniformListScrollHandle,
    list_state: ListState,
}

impl OracleView {
    fn new(cx: &mut Context<Self>) -> Self {
        Self {
            cells: (0..CELLS)
                .map(|ix| CellState {
                    word: ix % WORDS.len(),
                    color: ix % PALETTE.len(),
                    width: 40. + (ix % 5) as f32 * 30.,
                    background: ix.is_multiple_of(3),
                    truncate: ix.is_multiple_of(4),
                    ..CellState::default()
                })
                .collect(),
            paragraph_words: 12,
            paragraph_width: px(180.),
            rows: (0..INITIAL_ROWS).collect(),
            chips: (INITIAL_ROWS..INITIAL_ROWS + 6).collect(),
            next_row: INITIAL_ROWS + 6,
            row_identity: RowIdentity::Position,
            scroll_top: px(0.),
            column: false,
            segments: vec![0, 5, 8],
            segment_width: px(160.),
            tint: 0,
            selected: None,
            badge: cx.new(|_| Badge { count: 0 }),
            uniform_scroll: UniformListScrollHandle::new(),
            list_state: ListState::new(INITIAL_ROWS as usize, ListAlignment::Top, px(40.)),
        }
    }

    fn apply(&mut self, change: &Change, cx: &mut Context<Self>) {
        match *change {
            Change::Word { cell, word } => self.cells[cell].word = word,
            Change::Color { cell, color } => self.cells[cell].color = color,
            Change::Width { cell, width } => self.cells[cell].width = width,
            Change::Toggle { cell, flag } => {
                let cell = &mut self.cells[cell];
                let value = match flag {
                    CellFlag::Background => &mut cell.background,
                    CellFlag::Underline => &mut cell.underline,
                    CellFlag::Truncate => &mut cell.truncate,
                    CellFlag::Hidden => &mut cell.hidden,
                };
                *value = !*value;
            }
            Change::Paragraph { words, width } => {
                self.paragraph_words = words;
                self.paragraph_width = px(width);
            }
            Change::InsertRow { at } => {
                let at = at % (self.rows.len() + 1);
                self.rows.insert(at, self.next_row);
                self.next_row += 1;
                self.list_state.splice(at..at, 1);
            }
            Change::RemoveRow { at } => {
                if self.rows.is_empty() {
                    return;
                }
                let at = at % self.rows.len();
                self.rows.remove(at);
                self.list_state.splice(at..at + 1, 0);
            }
            Change::Scroll { top } => self.scroll_top = px(top),
            Change::InsertChip { at } => {
                let at = at % (self.chips.len() + 1);
                self.chips.insert(at, self.next_row);
                self.next_row += 1;
            }
            Change::RemoveChip { at } => {
                if !self.chips.is_empty() {
                    let at = at % self.chips.len();
                    self.chips.remove(at);
                }
            }
            Change::RotateChips { by } => {
                if !self.chips.is_empty() {
                    let by = by % self.chips.len();
                    self.chips.rotate_left(by);
                }
            }
            Change::RowIdentity(identity) => self.row_identity = identity,
            Change::Direction => self.column = !self.column,
            Change::Segment { at, word } => {
                if !self.segments.is_empty() {
                    let at = at % self.segments.len();
                    self.segments[at] = word;
                }
            }
            Change::SegmentCount { count } => {
                self.segments.resize(count, 1);
            }
            Change::SegmentWidth { width } => self.segment_width = px(width),
            Change::Tint => {
                self.tint += 1;
                self.badge.update(cx, |badge, cx| {
                    badge.count += 1;
                    cx.notify();
                });
            }
            Change::Select { row } => {
                self.selected = (!self.rows.is_empty()).then(|| self.rows[row % self.rows.len()]);
            }
            Change::Resize { .. } | Change::Redraw => return,
        }
        cx.notify();
    }
}

fn render_cell(cell: CellState, tint: usize) -> AnyElement {
    div()
        .flex()
        .flex_row()
        .w(px(cell.width))
        .h(px(18.))
        .text_color(PALETTE[(cell.color + tint + 1) % PALETTE.len()])
        .when(cell.background, |this| this.bg(PALETTE[cell.color]))
        .when(cell.underline, |this| this.underline())
        .when(cell.truncate, |this| {
            this.overflow_hidden().whitespace_nowrap().text_ellipsis()
        })
        .when(cell.hidden, |this| this.hidden())
        .child(WORDS[cell.word])
        .into_any_element()
}

fn render_row(row: u64, identity: RowIdentity, selected: bool) -> AnyElement {
    let word = WORDS[(row as usize * 7) % WORDS.len()];
    let row_element = div()
        .when(selected, |this| {
            // Asks the list to scroll the row into view, which rolls the
            // list's prepaint back and lays its items out again.
            this.child(
                crate::canvas(
                    |bounds, window, _| window.request_autoscroll(bounds),
                    |_, _, _, _| {},
                )
                .w(px(4.))
                .h(px(ROW_HEIGHT)),
            )
        })
        .flex()
        .flex_row()
        .gap_2()
        .h(px(ROW_HEIGHT))
        .child(SharedString::from(format!("row {row}")))
        .child(
            div()
                .w(px(8. + (row % 4) as f32 * 6.))
                .h(px(8.))
                .bg(PALETTE[row as usize % PALETTE.len()]),
        )
        .child(
            div()
                .border_1()
                .border_color(PALETTE[1])
                .px_1()
                .child(word),
        );
    match identity {
        RowIdentity::Position => row_element.into_any_element(),
        RowIdentity::Id => row_element.id(("row", row)).into_any_element(),
        RowIdentity::Wrapped => div().id(("row", row)).child(row_element).into_any_element(),
    }
}

fn render_chip(chip: u64, identity: RowIdentity) -> AnyElement {
    let chip_element = div()
        .flex()
        .flex_row()
        .px_1()
        .h(px(16.))
        .min_w(px(10. + (chip % 5) as f32 * 8.))
        .bg(PALETTE[chip as usize % PALETTE.len()])
        .when(chip.is_multiple_of(2), |this| {
            this.border_1().border_color(PALETTE[0])
        })
        .child(SharedString::from(chip.to_string()));
    match identity {
        RowIdentity::Position => chip_element.into_any_element(),
        RowIdentity::Id => chip_element.id(("chip", chip)).into_any_element(),
        RowIdentity::Wrapped => div()
            .id(("chip", chip))
            .child(chip_element)
            .into_any_element(),
    }
}

impl Render for OracleView {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        let paragraph: String = (0..self.paragraph_words)
            .map(|ix| WORDS[ix % WORDS.len()])
            .collect::<Vec<_>>()
            .join(" ");

        let rows = self.rows.clone();
        let identity = self.row_identity;
        self.uniform_scroll
            .0
            .borrow()
            .base_handle
            .set_offset(point(px(0.), -self.scroll_top));
        let uniform_rows = uniform_list("uniform rows", rows.len(), {
            let rows = rows.clone();
            move |range, _, _| {
                range
                    .map(|ix| render_row(rows[ix], identity, false))
                    .collect()
            }
        })
        .track_scroll(&self.uniform_scroll)
        .w(px(260.))
        .h(px(120.));

        if !rows.is_empty() {
            let item_ix =
                ((self.scroll_top / px(ROW_HEIGHT)).floor() as usize).min(rows.len() - 1);
            self.list_state.scroll_to(ListOffset {
                item_ix,
                offset_in_item: px(self.scroll_top.as_f32() % ROW_HEIGHT),
            });
        }
        let selected = self.selected;
        let list_rows = list(self.list_state.clone(), move |ix, _, _| {
            render_row(rows[ix], identity, selected == Some(rows[ix]))
        })
        .w(px(260.))
        .h(px(120.));

        let tint = self.tint;
        div()
            .size_full()
            .flex()
            .flex_wrap()
            .gap_2()
            .when(self.column, |this| this.flex_col())
            .child(
                div()
                    .flex()
                    .flex_row()
                    .flex_wrap()
                    .w(px(420.))
                    .gap_1()
                    .children(self.cells.iter().map(|&cell| render_cell(cell, tint))),
            )
            .child(
                div()
                    .w(self.paragraph_width)
                    .text_color(PALETTE[tint % PALETTE.len()])
                    .child(paragraph.clone()),
            )
            .child(
                // A flex item is measured for its content size before it is
                // shrunk to fit, so this text is shaped unconstrained and then
                // at whatever width it ends up with.
                div()
                    .flex()
                    .flex_row()
                    .w(self.paragraph_width * 0.8)
                    .child(div().text_color(PALETTE[1]).child(paragraph))
                    .child(div().w(px(24.)).h(px(12.)).bg(PALETTE[2])),
            )
            .child(
                self.badge
                    .clone()
                    .cached(StyleRefinement::default().w(px(120.)).h(px(24.))),
            )
            .child(
                div()
                    .flex()
                    .flex_row()
                    .gap_1()
                    .children(self.chips.iter().map(|&chip| render_chip(chip, identity))),
            )
            .child(Segments {
                words: self.segments.clone(),
                width: self.segment_width,
                tint,
            })
            .child(uniform_rows)
            .child(list_rows)
    }
}

/// A child view that is cached, and sometimes notified on its own.
struct Badge {
    count: usize,
}

impl Render for Badge {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        div()
            .flex()
            .flex_row()
            .gap_1()
            .size_full()
            .children((0..self.count % 4 + 1).map(|ix| {
                div()
                    .w(px(6. + ix as f32 * 3.))
                    .h(px(10.))
                    .bg(PALETTE[ix % PALETTE.len()])
            }))
            .child(SharedString::from(self.count.to_string()))
    }
}

/// Lays a segment out per word, each as a root of its own in prepaint, in a
/// box whose height a measurement reports: how virtualized content measures
/// what it shows.
struct Segments {
    words: Vec<usize>,
    width: Pixels,
    tint: usize,
}

impl IntoElement for Segments {
    type Element = Self;

    fn into_element(self) -> Self::Element {
        self
    }
}

impl Element for Segments {
    type RequestLayoutState = ();
    type PrepaintState = Vec<AnyElement>;

    fn id(&self) -> Option<ElementId> {
        None
    }

    fn source_location(&self) -> Option<&'static core::panic::Location<'static>> {
        None
    }

    fn request_layout(
        &mut self,
        _: Option<&GlobalElementId>,
        _: Option<&InspectorElementId>,
        window: &mut Window,
        _: &mut App,
    ) -> (LayoutId, ()) {
        let extent = size(self.width, px(SEGMENT_HEIGHT * self.words.len() as f32));
        (
            window.request_measured_layout(Style::default(), move |_, _, _, _| extent),
            (),
        )
    }

    fn prepaint(
        &mut self,
        _: Option<&GlobalElementId>,
        _: Option<&InspectorElementId>,
        bounds: Bounds<Pixels>,
        _: &mut (),
        window: &mut Window,
        cx: &mut App,
    ) -> Vec<AnyElement> {
        self.words
            .iter()
            .enumerate()
            .map(|(ix, &word)| {
                let mut segment = div()
                    .flex()
                    .flex_row()
                    .bg(PALETTE[(ix + self.tint) % PALETTE.len()])
                    .child(div().w(px(6.)).h(px(6.)).bg(PALETTE[ix % PALETTE.len()]))
                    .child(WORDS[word])
                    .into_any_element();
                segment.layout_as_root(
                    size(
                        AvailableSpace::Definite(bounds.size.width),
                        AvailableSpace::MinContent,
                    ),
                    window,
                    cx,
                );
                segment.prepaint_at(
                    bounds.origin + point(px(0.), px(SEGMENT_HEIGHT * ix as f32)),
                    window,
                    cx,
                );
                segment
            })
            .collect()
    }

    fn paint(
        &mut self,
        _: Option<&GlobalElementId>,
        _: Option<&InspectorElementId>,
        _: Bounds<Pixels>,
        _: &mut (),
        segments: &mut Vec<AnyElement>,
        window: &mut Window,
        cx: &mut App,
    ) {
        for segment in segments {
            segment.paint(window, cx);
        }
    }
}

/// The no-op text system, except that every glyph rasterizes to a small box,
/// so text paints a sprite per glyph and what went where is compared.
struct GlyphBoxTextSystem(NoopTextSystem);

impl PlatformTextSystem for GlyphBoxTextSystem {
    fn add_fonts(&self, fonts: Vec<Cow<'static, [u8]>>) -> Result<()> {
        self.0.add_fonts(fonts)
    }

    fn all_font_names(&self) -> Vec<String> {
        self.0.all_font_names()
    }

    fn font_id(&self, descriptor: &Font) -> Result<FontId> {
        self.0.font_id(descriptor)
    }

    fn font_metrics(&self, font_id: FontId) -> FontMetrics {
        self.0.font_metrics(font_id)
    }

    fn typographic_bounds(&self, font_id: FontId, glyph_id: GlyphId) -> Result<Bounds<f32>> {
        self.0.typographic_bounds(font_id, glyph_id)
    }

    fn advance(&self, font_id: FontId, glyph_id: GlyphId) -> Result<Size<f32>> {
        self.0.advance(font_id, glyph_id)
    }

    fn glyph_for_char(&self, font_id: FontId, ch: char) -> Option<GlyphId> {
        self.0.glyph_for_char(font_id, ch)
    }

    /// Each glyph's box is as wide as its id says, so which glyph was painted
    /// shows as well as where.
    fn glyph_raster_bounds(&self, params: &RenderGlyphParams) -> Result<Bounds<DevicePixels>> {
        Ok(Bounds {
            origin: point(DevicePixels(0), DevicePixels(-8)),
            size: size(DevicePixels(2 + (params.glyph_id.0 % 7) as i32), DevicePixels(9)),
        })
    }

    fn rasterize_glyph(
        &self,
        _params: &RenderGlyphParams,
        raster_bounds: Bounds<DevicePixels>,
    ) -> Result<(Size<DevicePixels>, Vec<u8>)> {
        let area = raster_bounds.size.width.0 * raster_bounds.size.height.0;
        Ok((raster_bounds.size, vec![u8::MAX; area.max(0) as usize]))
    }

    /// The no-op layout, with each glyph named after its character, so that
    /// glyphs of different characters paint differently.
    fn layout_line(&self, text: &str, font_size: Pixels, runs: &[FontRun]) -> LineLayout {
        let mut layout = self.0.layout_line(text, font_size, runs);
        for glyph in layout.runs.iter_mut().flat_map(|run| run.glyphs.iter_mut()) {
            if let Some(character) = text[glyph.index..].chars().next() {
                glyph.id = GlyphId(character as u32 + 16);
            }
        }
        layout
    }

    fn recommended_rendering_mode(&self, font_id: FontId, font_size: Pixels) -> TextRenderingMode {
        self.0.recommended_rendering_mode(font_id, font_size)
    }

    fn glyph_dilation_for_color(&self, color: Hsla) -> u8 {
        self.0.glyph_dilation_for_color(color)
    }
}

/// What the last drawn frame shows and where it can be hit, as text two
/// frames can be compared by. Atlas tiles are left out, since two windows
/// need not place a glyph in the same tile.
fn describe_rendered_frame(window: &Window) -> Vec<String> {
    let scene = &window.rendered_frame.scene;
    let mut lines = Vec::new();
    lines.extend(scene.shadows.iter().map(|shadow| format!("{shadow:?}")));
    lines.extend(scene.quads.iter().map(|quad| format!("{quad:?}")));
    lines.extend(
        scene
            .underlines
            .iter()
            .map(|underline| format!("{underline:?}")),
    );
    lines.extend(scene.monochrome_sprites.iter().map(|sprite| {
        format!(
            "monochrome {} {:?} {:?} {:?}",
            sprite.order, sprite.bounds, sprite.content_mask, sprite.color
        )
    }));
    lines.extend(scene.subpixel_sprites.iter().map(|sprite| {
        format!(
            "subpixel {} {:?} {:?} {:?}",
            sprite.order, sprite.bounds, sprite.content_mask, sprite.color
        )
    }));
    lines.extend(scene.polychrome_sprites.iter().map(|sprite| {
        format!(
            "polychrome {} {:?} {:?}",
            sprite.order, sprite.bounds, sprite.content_mask
        )
    }));
    lines.extend(
        scene
            .paths
            .iter()
            .map(|path| format!("path {} {:?}", path.order, path.bounds)),
    );
    lines.extend(window.rendered_frame.hitboxes.iter().map(|hitbox| {
        format!(
            "hitbox {:?} {:?} {:?}",
            hitbox.bounds, hitbox.content_mask, hitbox.behavior
        )
    }));
    lines
}

fn apply(cx: &mut TestAppContext, window: WindowHandle<OracleView>, change: &Change) {
    match *change {
        Change::Resize { width, height } => {
            cx.simulate_window_resize(window.into(), size(px(width), px(height)));
        }
        _ => window
            .update(cx, |view, _, cx| view.apply(change, cx))
            .unwrap(),
    }
}

fn draw(cx: &mut TestAppContext, window: WindowHandle<OracleView>) -> (Vec<String>, FrameWorkStats) {
    cx.update_window(window.into(), |_, window, cx| {
        window.reset_frame_work_stats(false);
        window.draw(cx).clear(cx);
        (describe_rendered_frame(window), window.frame_work_stats())
    })
    .unwrap()
}

fn text_system_context(seed: u64) -> TestAppContext {
    TestAppContext::build_with_text_system(
        TestDispatcher::new(seed),
        None,
        Arc::new(GlyphBoxTextSystem(NoopTextSystem)),
    )
}

/// The work both windows did over a run: what the retaining window did, and
/// what the window laying out from scratch did.
#[derive(Default)]
struct RunWork {
    retaining: FrameWorkStats,
    from_scratch: FrameWorkStats,
}

fn add(total: &mut FrameWorkStats, frame: &FrameWorkStats) {
    total.layout_nodes += frame.layout_nodes;
    total.layout_nodes_reused += frame.layout_nodes_reused;
    total.layout_nodes_created += frame.layout_nodes_created;
    total.measure_calls += frame.measure_calls;
    total.measured_nodes_dirtied += frame.measured_nodes_dirtied;
}

fn run(seed: u64, steps: usize) -> RunWork {
    let mut cx = text_system_context(seed);
    let retaining = cx.add_window(|_, cx| OracleView::new(cx));
    let from_scratch = cx.add_window(|_, cx| OracleView::new(cx));
    cx.update_window(from_scratch.into(), |_, window, _| {
        window.layout_keys.set_enabled(false)
    })
    .unwrap();
    let mut rng = StdRng::seed_from_u64(seed);
    let mut history: Vec<Vec<Change>> = Vec::new();
    let mut work = RunWork::default();

    for step in 0..steps {
        let changes: Vec<Change> = if step == 0 {
            Vec::new()
        } else {
            (0..rng.random_range(1..=3))
                .map(|_| Change::random(&mut rng))
                .collect()
        };
        for change in &changes {
            apply(&mut cx, retaining, change);
            apply(&mut cx, from_scratch, change);
        }
        history.push(changes);

        let (expected, scratch_work) = draw(&mut cx, from_scratch);
        let (actual, retaining_work) = draw(&mut cx, retaining);
        assert_eq!(
            scratch_work.layout_nodes_reused, 0,
            "a window without layout keys cannot reuse a node"
        );
        add(&mut work.retaining, &retaining_work);
        add(&mut work.from_scratch, &scratch_work);

        if actual != expected {
            let first = actual
                .iter()
                .zip(&expected)
                .position(|(actual, expected)| actual != expected)
                .unwrap_or(actual.len().min(expected.len()));
            let excerpt = |lines: &[String]| {
                lines
                    .iter()
                    .enumerate()
                    .skip(first.saturating_sub(2))
                    .take(5)
                    .map(|(ix, line)| format!("  {ix}: {line}"))
                    .collect::<Vec<_>>()
                    .join("\n")
            };
            let history = history
                .iter()
                .enumerate()
                .map(|(step, changes)| format!("  {step}: {changes:?}"))
                .collect::<Vec<_>>()
                .join("\n");
            panic!(
                "seed {seed}, step {step}: the retained frame differs from the frame laid out \
                 from scratch at line {first} ({} lines against {})\n\
                 retained:\n{}\nfrom scratch:\n{}\nchanges so far:\n{history}",
                actual.len(),
                expected.len(),
                excerpt(&actual),
                excerpt(&expected),
            );
        }
    }
    work
}

#[test]
fn retained_layout_frames_match_frames_laid_out_from_scratch() {
    let mut retaining = FrameWorkStats::default();
    for seed in 0..24 {
        let work = run(seed, 60);
        add(&mut retaining, &work.retaining);
    }
    assert!(
        retaining.layout_nodes_reused > retaining.layout_nodes_created,
        "the retaining window should reuse most of its nodes: {retaining:?}"
    );
}

/// Draws a view once, then again after `change`, and returns the work of
/// the frame that followed the change.
///
/// Notifying a view can draw a frame of its own before the explicit one
/// here, so the counters start before the change rather than before the
/// draw, and whichever frame took the change is the one counted.
fn work_after<V: Render>(
    build: impl FnOnce(&mut Context<V>) -> V + 'static,
    change: impl FnOnce(&mut V, &mut Context<V>),
) -> FrameWorkStats {
    let mut cx = text_system_context(0);
    let window = cx.add_window(|_, cx| build(cx));
    cx.update_window(window.into(), |_, window, cx| {
        window.draw(cx).clear(cx);
        window.reset_frame_work_stats(false);
    })
    .unwrap();
    window
        .update(&mut cx, |view, _, cx| {
            change(view, cx);
            cx.notify();
        })
        .unwrap();
    cx.update_window(window.into(), |_, window, cx| {
        if window.frame_work_stats().frames == 0 {
            window.draw(cx).clear(cx);
        }
        window.frame_work_stats()
    })
    .unwrap()
}

/// A frame asking for exactly what the last one did makes no node and
/// writes to none: every style, child list and measurement stands.
#[test]
fn an_unchanged_frame_reuses_every_layout_node() {
    let work = work_after(OracleView::new, |_, _| {});
    assert_eq!(work.layout_nodes_created, 0, "{work:?}");
    assert_eq!(work.layout_style_writes, 0, "{work:?}");
    assert_eq!(work.layout_children_writes, 0, "{work:?}");
    assert!(work.layout_nodes_reused > 100, "{work:?}");
}

/// A chip row whose chips are identified keeps every chip's node when one is
/// inserted ahead of them; identified by position, every chip after it lands
/// on its neighbour's node.
#[test]
fn identified_children_keep_their_nodes_when_one_is_inserted_ahead() {
    let insert_at_head = |identity: RowIdentity| {
        work_after(
            move |cx| {
                let mut view = OracleView::new(cx);
                view.row_identity = identity;
                view.chips = (100..140).collect();
                view
            },
            |view, _| view.chips.insert(0, 999),
        )
    };
    let by_position = insert_at_head(RowIdentity::Position);
    let by_id = insert_at_head(RowIdentity::Id);
    assert!(
        by_id.layout_style_writes + 3 < by_position.layout_style_writes,
        "identified: {by_id:?}\npositional: {by_position:?}"
    );
}

/// List items without an id are keyed by their index, so a list scrolled by
/// a row keeps the nodes of the items still in view.
#[test]
fn list_items_keep_their_nodes_when_the_list_scrolls() {
    let work = work_after(OracleView::new, |view, _| view.scroll_top = px(ROW_HEIGHT));
    let still = work_after(OracleView::new, |_, _| {});
    // The row scrolled in is new on both lists; the ones still in view are
    // not, so hardly anything more is made than on a still frame.
    assert!(
        work.layout_nodes_created <= still.layout_nodes_created + 30,
        "scrolled: {work:?}\nstill: {still:?}"
    );
}

/// Nodes no element asks for any more are released: however long the
/// history, the tree holds no more than the last frame asked for.
#[test]
fn keeping_layout_nodes_does_not_grow_the_tree() {
    let mut cx = text_system_context(7);
    let window = cx.add_window(|_, cx| OracleView::new(cx));
    let mut rng = StdRng::seed_from_u64(7);
    for _ in 0..200 {
        let change = Change::random(&mut rng);
        apply(&mut cx, window, &change);
        let (_, work) = draw(&mut cx, window);
        let held = cx
            .update_window(window.into(), |_, window, _| {
                window.layout_engine.as_ref().map_or(0, |engine| engine.node_count())
            })
            .unwrap();
        assert!(
            held as u64 <= work.layout_nodes,
            "{held} nodes held after a frame that asked for {}",
            work.layout_nodes
        );
    }
}

/// A dashboard of panels of labels, one of which changes a label per frame.
struct Dashboard {
    panels: usize,
    labels: usize,
    frame: usize,
}

impl Render for Dashboard {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        let changing = self.frame % self.panels;
        div()
            .size_full()
            .flex()
            .flex_row()
            .flex_wrap()
            .gap_1()
            .children((0..self.panels).map(|panel| {
                div()
                    .flex()
                    .flex_col()
                    .w(px(180.))
                    .p_1()
                    .border_1()
                    .border_color(PALETTE[panel % PALETTE.len()])
                    .children((0..self.labels).map(|label| {
                        let value = if panel == changing && label == 0 {
                            self.frame
                        } else {
                            panel * self.labels + label
                        };
                        div()
                            .flex()
                            .flex_row()
                            .justify_between()
                            .child(WORDS[(panel + label) % WORDS.len()])
                            .child(SharedString::from(format!("{:>5}", value % 100_000)))
                    }))
            }))
    }
}

/// A transcript of paragraphs in a list, the last of which grows by a word a
/// frame, as a streamed reply does.
struct Transcript {
    list_state: ListState,
    paragraphs: usize,
    last_words: usize,
}

impl Render for Transcript {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        let paragraphs = self.paragraphs;
        let last_words = self.last_words;
        div()
            .size_full()
            .flex()
            .flex_row()
            .child(
                // A sidebar that does not change, as most of a window does not
                // while one reply streams in.
                div().flex().flex_col().w(px(240.)).children((0..120).map(|row| {
                    div()
                        .flex()
                        .flex_row()
                        .gap_1()
                        .child(div().size(px(10.)).bg(PALETTE[row % PALETTE.len()]))
                        .child(WORDS[row % WORDS.len()])
                        .child(SharedString::from(format!("{row}")))
                })),
            )
            .child(
            list(self.list_state.clone(), move |ix, _, _| {
                let words = if ix + 1 == paragraphs {
                    last_words
                } else {
                    20 + ix % 30
                };
                let text: String = (0..words)
                    .map(|word| WORDS[(ix + word) % WORDS.len()])
                    .collect::<Vec<_>>()
                    .join(" ");
                div()
                    .flex()
                    .flex_col()
                    .p_2()
                    .child(div().text_color(PALETTE[1]).child(format!("message {ix}")))
                    .child(div().w(px(560.)).child(text))
                    .into_any_element()
            })
            .w(px(600.))
            .h_full(),
        )
    }
}

fn measure_frames<V: Render>(
    name: &str,
    view: impl Fn(&mut Context<V>) -> V + Clone + 'static,
    mut step: impl FnMut(&mut V, &mut Context<V>),
) {
    for retained in [false, true] {
        let mut cx = text_system_context(0);
        let window = cx.add_window({
            let view = view.clone();
            move |_, cx| view(cx)
        });
        cx.update_window(window.into(), |_, window, _| {
            window.layout_keys.set_enabled(retained)
        })
        .unwrap();
        cx.simulate_window_resize(window.into(), size(px(1600.), px(1200.)));
        for _ in 0..3 {
            cx.update_window(window.into(), |_, window, cx| window.draw(cx).clear(cx))
                .unwrap();
        }
        cx.update_window(window.into(), |_, window, _| {
            window.reset_frame_work_stats(true)
        })
        .unwrap();
        for _ in 0..30 {
            window
                .update(&mut cx, |view, _, cx| {
                    step(view, cx);
                    cx.notify();
                })
                .unwrap();
            cx.update_window(window.into(), |_, window, cx| {
                if window.frame_work_stats().frames == 0 {
                    window.draw(cx).clear(cx);
                }
            })
            .unwrap();
            cx.update_window(window.into(), |_, window, cx| window.draw(cx).clear(cx))
                .unwrap();
        }
        let work = cx
            .update_window(window.into(), |_, window, _| window.frame_work_stats())
            .unwrap();
        let per_frame = |count: u64| count as f64 / work.frames as f64;
        let per_frame_ms =
            |duration: std::time::Duration| duration.as_secs_f64() * 1000. / work.frames as f64;
        eprintln!(
            "{name}, layout nodes kept {retained}: per frame {:.0} nodes, {:.0} reused, \
             {:.1} created, {:.1} measured nodes dirtied, {:.0} carried, {:.1} replayed, \
             {:.0} measure calls, {:.1} replay measure calls, {:.1} lines shaped; \
             build {:.2} ms, prepaint {:.2} ms (layout {:.2} ms, measuring {:.2} ms), \
             paint {:.2} ms",
            per_frame(work.layout_nodes),
            per_frame(work.layout_nodes_reused),
            per_frame(work.layout_nodes_created),
            per_frame(work.measured_nodes_dirtied),
            per_frame(work.measurements_carried),
            per_frame(work.measurements_replayed),
            per_frame(work.measure_calls),
            per_frame(work.replay_measure_calls),
            per_frame(work.lines_shaped),
            per_frame_ms(work.build_time),
            per_frame_ms(work.prepaint_time),
            per_frame_ms(work.compute_layout_time),
            per_frame_ms(work.measure_time),
            per_frame_ms(work.paint_time),
        );
    }
}

/// Prints what frames of a few large, mostly unchanging windows cost with
/// layout nodes kept and without: a dashboard where one label ticks, a
/// transcript whose last paragraph grows, and the same transcript scrolled.
/// Every frame is drawn twice, once with the change and once as it is, as a
/// window redrawn for a hover or an animation would be. Run with
/// `cargo test -p gpui --lib --features test-support frame_work_of_large_windows -- --ignored --nocapture`.
#[test]
#[ignore]
fn frame_work_of_large_windows() {
    measure_frames(
        "dashboard, one label ticking",
        |_| Dashboard {
            panels: 40,
            labels: 40,
            frame: 0,
        },
        |view, _| view.frame += 1,
    );
    let transcript = |_: &mut Context<Transcript>| Transcript {
        list_state: {
            let state = ListState::new(200, ListAlignment::Bottom, px(200.));
            state.set_follow_mode(crate::FollowMode::Tail);
            state
        },
        paragraphs: 200,
        last_words: 1,
    };
    measure_frames("transcript, streaming", transcript, |view, _| {
        view.last_words += 1
    });
    measure_frames(
        "transcript, scrolling 7 px a frame",
        move |cx| {
            let view = transcript(cx);
            view.list_state.scroll_to(ListOffset {
                item_ix: 100,
                offset_in_item: px(0.),
            });
            view
        },
        |view, _| view.list_state.scroll_by(px(7.)),
    );
}

/// Text in a narrow box, above a probe that lands below it.
struct WrappedText {
    text: SharedString,
    color: Hsla,
}

impl Render for WrappedText {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        div()
            .flex()
            .flex_col()
            .w(px(200.))
            .child(
                div()
                    .w(px(120.))
                    .text_color(self.color)
                    .child(self.text.clone()),
            )
            .child(div().w(px(20.)).h(px(10.)).bg(PALETTE[2]))
    }
}

fn wrapped_text(text: &'static str) -> impl FnOnce(&mut Context<WrappedText>) -> WrappedText {
    move |_| WrappedText {
        text: text.into(),
        color: PALETTE[0],
    }
}

/// Text drawn again as it was takes last frame's measurement over, and
/// recolored text does too, repainted with its new colour: neither is
/// measured again nor dirties the nodes above it.
#[test]
fn text_measured_the_same_way_keeps_its_measurement() {
    let unchanged = work_after(wrapped_text("a label that wraps in the box"), |_, _| {});
    assert_eq!(unchanged.measured_nodes_dirtied, 0, "{unchanged:?}");
    assert_eq!(unchanged.measure_calls, 0, "{unchanged:?}");
    assert!(unchanged.measurements_carried >= 1, "{unchanged:?}");

    let recolored = work_after(wrapped_text("a label that wraps in the box"), |view, _| {
        view.color = PALETTE[3]
    });
    assert_eq!(recolored.measured_nodes_dirtied, 0, "{recolored:?}");
    assert_eq!(recolored.measure_calls, 0, "{recolored:?}");
    assert!(recolored.measurements_carried >= 1, "{recolored:?}");
}

/// Changed text that measures what it measured before, under every
/// constraint it was measured under, leaves its node and the nodes above it
/// clean; text that wraps differently is measured again.
#[test]
fn changed_text_is_measured_again_only_when_its_size_changes() {
    let same_size = work_after(wrapped_text("value 42"), |view, _| {
        view.text = "value 17".into()
    });
    assert_eq!(same_size.measured_nodes_dirtied, 0, "{same_size:?}");
    assert_eq!(same_size.measure_calls, 0, "{same_size:?}");
    assert_eq!(same_size.measurements_replayed, 1, "{same_size:?}");

    let longer = work_after(wrapped_text("value 42"), |view, _| {
        view.text = "a value long enough to wrap onto a second line".into()
    });
    assert_eq!(longer.measured_nodes_dirtied, 1, "{longer:?}");
    assert!(longer.measure_calls > 0, "{longer:?}");
}

/// Rows without ids, whose text lands on a neighbour's node when a row is
/// inserted ahead of them.
struct ShiftingRows {
    rows: Vec<u64>,
}

impl Render for ShiftingRows {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        div().flex().flex_col().children(
            self.rows
                .iter()
                .map(|row| div().child(SharedString::from(format!("row number {row}")))),
        )
    }
}

/// Text a node carries over from frame to frame never asks the line layout
/// cache for its lines, which the cache would then drop; the element hands
/// them back, so text shifting onto another node is not shaped again.
#[test]
fn text_moving_to_another_node_is_not_shaped_again() {
    let mut cx = text_system_context(0);
    let window = cx.add_window(|_, _| ShiftingRows {
        rows: (0..8).collect(),
    });
    for _ in 0..4 {
        cx.update_window(window.into(), |_, window, cx| {
            window.refresh();
            window.draw(cx).clear(cx);
        })
        .unwrap();
    }
    cx.update_window(window.into(), |_, window, _| {
        window.reset_frame_work_stats(false)
    })
    .unwrap();
    window
        .update(&mut cx, |view, _, cx| {
            view.rows.insert(0, 100);
            cx.notify();
        })
        .unwrap();
    let work = cx
        .update_window(window.into(), |_, window, cx| {
            window.draw(cx).clear(cx);
            window.frame_work_stats()
        })
        .unwrap();
    assert!(
        work.measurements_replayed >= 7,
        "rows landed on their neighbours' nodes: {work:?}"
    );
    assert_eq!(work.lines_shaped, 1, "only the inserted row is new: {work:?}");
}
