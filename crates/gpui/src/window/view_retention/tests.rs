//! Frames that draw views again from the last frame must be the frames drawn
//! from scratch.
//!
//! The oracle drives two windows of one app, with view retention on, through
//! the same random history of changes: one draws its views again from the
//! last frame wherever it can, the other is refreshed before every frame, so
//! every view is built. Every frame, the two must paint the same primitives
//! in the same places and leave the same hitboxes. The changes keep to what
//! retention asks of an application: a model is changed by updating it and
//! notifying it, and state outside entities is declared.

use super::{DrawDependency, ViewRebuildReason, describe_frame, first_difference};
use crate::{
    AnyElement, App, Context, Entity, Global, Hsla, IntoElement, ListAlignment,
    ListOffset, ListState, Modifiers, Pixels, Render, SharedString, StyleRefinement, TestAppContext,
    Window, WindowHandle, deferred, div, hsla, list, point, prelude::*, px, size,
};
use rand::{Rng as _, SeedableRng as _, rngs::StdRng};
use std::{
    cell::Cell,
    rc::Rc,
    time::{Duration, Instant},
};

const WORDS: [&str; 8] = [
    "a",
    "card",
    "a longer label that wraps",
    "42",
    "value",
    "x",
    "another label",
    "sit amet dolor",
];

const PALETTE: [Hsla; 4] = [
    hsla(0.0, 0.0, 0.1, 1.0),
    hsla(0.6, 0.7, 0.5, 1.0),
    hsla(0.3, 0.6, 0.4, 1.0),
    hsla(0.0, 0.8, 0.6, 1.0),
];

const CARDS: usize = 8;

/// A model the cards read without observing it.
struct Model {
    labels: Vec<usize>,
}

/// A window's summary of the model, which cards show and a view drawn after
/// them writes as it renders.
struct Summary(usize);

/// Works the model's summary out as it renders, after the cards reading it
/// were drawn, and writes it when it changed.
struct Summarizer {
    model: Entity<Model>,
    summary: Entity<Summary>,
}

impl Render for Summarizer {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let summary = self.model.read(cx).labels.iter().sum::<usize>() % 7;
        if self.summary.read(cx).0 != summary {
            self.summary.update(cx, |written, _| written.0 = summary);
        }
        div().w(px(4.)).h(px(4.))
    }
}

/// A global some cards read.
#[derive(PartialEq)]
struct Theme(usize);

impl Global for Theme {}

/// What both windows share: the model, state outside entities that cards
/// declare or opt out over, and the clock's origin.
struct Shared {
    model: Entity<Model>,
    registry: Rc<Cell<usize>>,
    dependency: DrawDependency,
    untracked: Rc<Cell<usize>>,
    started_at: Instant,
}

#[derive(Clone, Copy, Debug)]
enum CardKind {
    Model,
    Global,
    Dependency,
    Clock,
    OptedOut,
    Plain,
    Actions,
    /// Reads the model without depending on it, and is notified when what
    /// it shows from it changes, as the contract of
    /// [`crate::Context::untrack_reads_of`] asks.
    Detached,
}

impl CardKind {
    fn of(ix: usize) -> Self {
        [
            CardKind::Model,
            CardKind::Global,
            CardKind::Dependency,
            CardKind::Clock,
            CardKind::OptedOut,
            CardKind::Plain,
            CardKind::Actions,
            CardKind::Detached,
        ][ix % 8]
    }
}

struct Card {
    ix: usize,
    count: usize,
    kind: CardKind,
    popover: bool,
    summary: Entity<Summary>,
    /// Asks the list it is in to scroll it into view, which rolls the list's
    /// prepaint back and lays its items out again.
    reveal: bool,
    /// Handles the probe action, which a card reading the actions shows
    /// once it is focused.
    handles: bool,
    focus_handle: crate::FocusHandle,
    inner: Entity<Inner>,
    shared: Rc<Shared>,
}

impl Card {
    fn new(ix: usize, shared: Rc<Shared>, summary: Entity<Summary>, cx: &mut Context<Self>) -> Self {
        let kind = CardKind::of(ix);
        if matches!(kind, CardKind::OptedOut) {
            cx.set_view_retainable(false);
        }
        if matches!(kind, CardKind::Detached) {
            cx.untrack_reads_of(&shared.model);
        }
        let model = shared.model.clone();
        Self {
            ix,
            count: 0,
            kind,
            popover: false,
            summary,
            reveal: false,
            handles: false,
            focus_handle: cx.focus_handle(),
            inner: cx.new(|_| Inner { ix, count: 0, model }),
            shared,
        }
    }
}

impl Render for Card {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let shared = &self.shared;
        let detail: SharedString = match self.kind {
            CardKind::Model => {
                let labels = &shared.model.read(cx).labels;
                let word = WORDS[labels[self.ix % labels.len()]];
                format!("{word} {}", self.summary.read(cx).0).into()
            }
            CardKind::Global => format!("theme {}", cx.global::<Theme>().0).into(),
            CardKind::Dependency => {
                window.depend_on(&shared.dependency);
                format!("registry {}", shared.registry.get()).into()
            }
            CardKind::Clock => {
                let elapsed = cx
                    .background_executor()
                    .now()
                    .saturating_duration_since(shared.started_at)
                    .as_secs();
                window.rebuild_at(shared.started_at + Duration::from_secs(elapsed + 1));
                format!("{elapsed}s ago").into()
            }
            CardKind::OptedOut => format!("untracked {}", shared.untracked.get()).into(),
            CardKind::Plain => "plain".into(),
            CardKind::Detached => {
                let labels = &shared.model.read(cx).labels;
                format!("detached {}", WORDS[labels[self.ix % labels.len()]]).into()
            }
            CardKind::Actions => {
                let available = actions_asked_of_last_frame(window)
                    && window.is_action_available(&probe_actions::Probe, cx);
                format!("probe {available}").into()
            }
        };
        let popover = self.popover.then(|| {
            let label = WORDS[shared.model.read(cx).labels[0]];
            deferred(
                div()
                    .absolute()
                    .top(px(30.))
                    .left(px(10.))
                    .w(px(90.))
                    .bg(PALETTE[1])
                    .child(label)
                    // Reads the theme only as it paints, which is after the
                    // card, so the card depends on it only through what it
                    // deferred.
                    .child(
                        crate::canvas(
                            |_, _, _| {},
                            |bounds, _, window, cx| {
                                let theme = cx.global::<Theme>().0;
                                window.paint_quad(crate::fill(bounds, PALETTE[theme % PALETTE.len()]));
                            },
                        )
                        .w(px(8.))
                        .h(px(8.)),
                    ),
            )
            .with_priority(1)
        });
        div()
            .id(("card", self.ix))
            .flex()
            .flex_col()
            .w(px(140.))
            .p_1()
            .border_1()
            .border_color(PALETTE[self.ix % PALETTE.len()])
            .hover(|style| style.bg(PALETTE[2]))
            // A group container the inner view's group hover resolves, and
            // a group hover resolving the shell's: both containers are in
            // another view than the hover.
            .group("card")
            .when(self.ix.is_multiple_of(4), |this| {
                this.group_hover("shell", |style| style.border_color(PALETTE[0]))
            })
            .track_focus(&self.focus_handle)
            .when(self.handles, |this| {
                this.on_action(|_: &probe_actions::Probe, _, _| {})
            })
            .child(SharedString::from(format!("card {} {}", self.ix, self.count)))
            .child(detail)
            .child(self.inner.clone())
            .children(popover)
            .when(self.reveal, |this| {
                this.child(
                    crate::canvas(
                        |bounds, window, _| window.request_autoscroll(bounds),
                        |_, _, _, _| {},
                    )
                    .w(px(4.))
                    .h(px(4.)),
                )
            })
    }
}

/// The actions are asked of the frame before, which the first frame does
/// not have.
fn actions_asked_of_last_frame(window: &Window) -> bool {
    window.rendered_frame.dispatch_tree.len() > 0
}

/// A view nested in a card, which reads the model as well.
struct Inner {
    ix: usize,
    count: usize,
    model: Entity<Model>,
}

impl Render for Inner {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let labels = &self.model.read(cx).labels;
        let word = WORDS[labels[(self.ix + 3) % labels.len()]];
        div()
            .id(("inner", self.ix))
            .flex()
            .flex_row()
            .gap_1()
            .child(
                div()
                    .w(px(6. + self.count as f32 * 2.))
                    .h(px(6.))
                    .bg(PALETTE[1])
                    .when(self.ix.is_multiple_of(3), |this| {
                        this.group_hover("card", |style| style.bg(PALETTE[3]))
                    }),
            )
            .child(word)
            // Paints by whether its hitbox is hovered, and is not notified
            // when that changes: the hover it was painted by is a dependency.
            .child(
                crate::canvas(
                    |bounds, window, _| window.insert_hitbox(bounds, crate::HitboxBehavior::Normal),
                    |bounds, hitbox, window, _| {
                        if hitbox.is_hovered(window) {
                            window.paint_quad(crate::fill(bounds, PALETTE[0]));
                        }
                    },
                )
                .w(px(12.))
                .h(px(12.)),
            )
    }
}

struct Shell {
    cards: Vec<Entity<Card>>,
    list_cards: Vec<Entity<Card>>,
    list_state: ListState,
    /// Shows where the list is scrolled to, beside it.
    list_reader: Entity<ScrollReader>,
    summarizer: Entity<Summarizer>,
    summary: Entity<Summary>,
    column: bool,
    tint: usize,
    /// Handles the probe action at the root, which every card reading the
    /// actions shows while nothing is focused.
    handles: bool,
    /// The height of a header above everything, in steps.
    header: usize,
    /// The index the next inserted card gets: never one a card has had, so
    /// that no two cards share an element id after a removal.
    next_card_ix: usize,
}

impl Shell {
    fn new(shared: &Rc<Shared>, cx: &mut Context<Self>) -> Self {
        let summary = cx.new(|_| Summary(0));
        let cards = (0..CARDS)
            .map(|ix| cx.new(|cx| Card::new(ix, shared.clone(), summary.clone(), cx)))
            .collect();
        let list_cards: Vec<_> = (0..CARDS)
            .map(|ix| cx.new(|cx| Card::new(ix + 100, shared.clone(), summary.clone(), cx)))
            .collect();
        let list_state = ListState::new(list_cards.len(), ListAlignment::Top, px(20.));
        Self {
            cards,
            summarizer: cx.new({
                let model = shared.model.clone();
                let summary = summary.clone();
                |_| Summarizer { model, summary }
            }),
            summary,
            list_reader: cx.new(|_| ScrollReader {
                list_state: list_state.clone(),
                seen: Rc::default(),
            }),
            list_state,
            list_cards,
            column: false,
            tint: 0,
            handles: false,
            header: 0,
            next_card_ix: CARDS,
        }
    }
}

impl Render for Shell {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        let list_cards = self.list_cards.clone();
        let items = list(self.list_state.clone(), move |ix, _, _| {
            list_cards[ix].clone().into_any_element()
        })
        .w(px(160.))
        .h(px(150.));
        div()
            .size_full()
            .flex()
            .flex_wrap()
            .gap_1()
            .when(self.column, |this| this.flex_col())
            .bg(PALETTE[self.tint % PALETTE.len()])
            .group("shell")
            .when(self.handles, |this| {
                this.on_action(|_: &probe_actions::Probe, _, _| {})
            })
            .child(div().w_full().h(px(self.header as f32 * 7.5)))
            .child(items)
            .child(self.list_reader.clone())
            .children(self.cards.iter().enumerate().map(|(ix, card)| {
                if ix == 1 {
                    card.clone()
                        .cached(StyleRefinement::default().w(px(140.)).h(px(70.)))
                        .into_any_element()
                } else {
                    card.clone().into_any_element()
                }
            }))
            .child(self.summarizer.clone())
    }
}

#[derive(Clone, Debug)]
enum Change {
    Label { at: usize, word: usize },
    QuietLabelThenNotifyShell { at: usize, word: usize },
    Theme(usize),
    Registry(usize),
    Clock { millis: u64 },
    Untracked(usize),
    CountCard { ix: usize },
    NotifyCard { ix: usize },
    CountInner { ix: usize },
    Tint,
    Column,
    Popover { ix: usize },
    Reveal { ix: usize },
    InsertCard,
    RemoveCard { ix: usize },
    Scroll { top: usize },
    Wheel { delta: f32 },
    Mouse { x: f32, y: f32 },
    Resize { width: f32, height: f32 },
    /// Grows or shrinks a header above everything else, which moves what
    /// follows it, as a streaming reply grows the rows around it.
    Header { height: usize },
    ShellHandles,
    CardHandles { ix: usize },
    Focus { ix: Option<usize> },
    Redraw,
}

impl Change {
    fn random(rng: &mut StdRng) -> Self {
        let ix = rng.random_range(0..CARDS);
        match rng.random_range(0..100) {
            0..12 => Change::Label {
                at: rng.random_range(0..CARDS),
                word: rng.random_range(0..WORDS.len()),
            },
            12..16 => Change::QuietLabelThenNotifyShell {
                at: rng.random_range(0..CARDS),
                word: rng.random_range(0..WORDS.len()),
            },
            16..22 => Change::Theme(rng.random_range(0..5)),
            22..28 => Change::Registry(rng.random_range(0..5)),
            28..34 => Change::Clock {
                millis: rng.random_range(0..2500),
            },
            34..38 => Change::Untracked(rng.random_range(0..5)),
            38..48 => Change::CountCard { ix },
            48..56 => Change::NotifyCard { ix },
            56..62 => Change::CountInner { ix },
            62..65 => Change::Tint,
            65..67 => Change::Column,
            67..69 => Change::Popover { ix },
            69..71 => Change::Reveal { ix },
            71..73 => Change::InsertCard,
            73..75 => Change::RemoveCard { ix },
            75..80 => Change::Scroll {
                top: rng.random_range(0..CARDS),
            },
            // Half the time by whole device pixels (the test window's scale
            // is 2), by which the views in the list can be drawn moved.
            80..84 => Change::Wheel {
                delta: if rng.random_bool(0.5) {
                    (rng.random_range(-240.0f32..240.0)).round() / 2.
                } else {
                    rng.random_range(-120.0..120.0)
                },
            },
            84..92 => Change::Mouse {
                x: rng.random_range(0.0..700.0),
                y: rng.random_range(0.0..500.0),
            },
            92..95 => Change::Resize {
                width: rng.random_range(300.0..900.0),
                height: rng.random_range(240.0..700.0),
            },
            95..97 => match rng.random_range(0..3) {
                0 => Change::ShellHandles,
                _ => Change::Header {
                    height: rng.random_range(0..8),
                },
            },
            97..98 => Change::CardHandles { ix },
            98..99 => Change::Focus {
                ix: rng.random_bool(0.7).then_some(ix),
            },
            _ => Change::Redraw,
        }
    }
}

struct Oracle {
    shared: Rc<Shared>,
    retaining: WindowHandle<Shell>,
    from_scratch: WindowHandle<Shell>,
    /// Last, so that the handles the shared state holds are dropped first.
    cx: TestAppContext,
}

impl Oracle {
    fn new() -> Self {
        // Glyphs paint as boxes named after their characters, so what text
        // says is compared as well as where it goes.
        let mut cx = super::super::layout_retention_tests::text_system_context(0);
        cx.update(|cx| {
            cx.set_view_retention(true);
            cx.set_global(Theme(0));
        });
        let model = cx.new(|_| Model {
            labels: (0..CARDS).map(|ix| ix % WORDS.len()).collect(),
        });
        let shared = Rc::new(Shared {
            model,
            registry: Rc::new(Cell::new(0)),
            dependency: DrawDependency::new(),
            untracked: Rc::new(Cell::new(0)),
            started_at: cx.executor().now(),
        });
        let retaining = cx.add_window({
            let shared = shared.clone();
            move |_, cx| Shell::new(&shared, cx)
        });
        let from_scratch = cx.add_window({
            let shared = shared.clone();
            move |_, cx| Shell::new(&shared, cx)
        });
        Self {
            cx,
            shared,
            retaining,
            from_scratch,
        }
    }

    fn windows(&self) -> [WindowHandle<Shell>; 2] {
        [self.retaining, self.from_scratch]
    }

    fn update_cards(&mut self, ix: usize, update: impl Fn(&mut Card, &mut Context<Card>)) {
        for window in self.windows() {
            let card = window
                .read_with(&self.cx, |shell, _| shell.cards.get(ix).cloned())
                .unwrap();
            if let Some(card) = card {
                card.update(&mut self.cx, |card, cx| update(card, cx));
            }
        }
    }

    /// Notifies the detached cards, and the views nested in them, that show
    /// the label at `at`: they do not depend on the model they read it from.
    fn notify_detached_readers(&mut self, at: usize) {
        for window in self.windows() {
            let cards = window
                .read_with(&self.cx, |shell, _| {
                    shell
                        .cards
                        .iter()
                        .chain(&shell.list_cards)
                        .cloned()
                        .collect::<Vec<_>>()
                })
                .unwrap();
            for card in cards {
                card.update(&mut self.cx, |card, cx| {
                    if !matches!(card.kind, CardKind::Detached) {
                        return;
                    }
                    // The popover it defers shows the first label.
                    if card.ix % CARDS == at || (card.popover && at == 0) {
                        cx.notify();
                    }
                    if (card.ix + 3) % CARDS == at {
                        card.inner.update(cx, |_, cx| cx.notify());
                    }
                });
            }
        }
    }

    fn update_shells(&mut self, update: impl Fn(&mut Shell, &mut Context<Shell>)) {
        for window in self.windows() {
            window
                .update(&mut self.cx, |shell, _, cx| {
                    update(shell, cx);
                    cx.notify();
                })
                .unwrap();
        }
    }

    fn apply(&mut self, change: &Change) {
        let model = self.shared.model.clone();
        match *change {
            Change::Label { at, word } => {
                model.update(&mut self.cx, |model, cx| {
                    model.labels[at] = word;
                    cx.notify();
                });
                self.notify_detached_readers(at);
            }
            Change::QuietLabelThenNotifyShell { at, word } => {
                model.update(&mut self.cx, |model, _| model.labels[at] = word);
                self.update_shells(|_, _| {});
                self.notify_detached_readers(at);
            }
            Change::Theme(theme) => self.cx.update(|cx| cx.set_global(Theme(theme))),
            Change::Registry(value) => {
                self.shared.registry.set(value);
                let dependency = self.shared.dependency.clone();
                self.cx.update(|cx| dependency.changed(cx));
            }
            Change::Clock { millis } => self
                .cx
                .executor()
                .advance_clock(Duration::from_millis(millis)),
            Change::Untracked(value) => self.shared.untracked.set(value),
            Change::CountCard { ix } => self.update_cards(ix, |card, cx| {
                card.count += 1;
                cx.notify();
            }),
            Change::NotifyCard { ix } => self.update_cards(ix, |_, cx| cx.notify()),
            Change::CountInner { ix } => self.update_cards(ix, |card, cx| {
                card.inner.update(cx, |inner, cx| {
                    inner.count += 1;
                    cx.notify();
                });
            }),
            Change::Tint => self.update_shells(|shell, _| shell.tint += 1),
            Change::Column => self.update_shells(|shell, _| shell.column = !shell.column),
            Change::Popover { ix } => self.update_cards(ix, |card, cx| {
                card.popover = !card.popover;
                cx.notify();
            }),
            Change::Reveal { ix } => {
                for window in self.windows() {
                    let card = window
                        .read_with(&self.cx, |shell, _| shell.list_cards[ix].clone())
                        .unwrap();
                    card.update(&mut self.cx, |card, cx| {
                        card.reveal = !card.reveal;
                        cx.notify();
                    });
                }
            }
            Change::InsertCard => {
                let shared = self.shared.clone();
                self.update_shells(move |shell, cx| {
                    let ix = shell.next_card_ix;
                    shell.next_card_ix += 1;
                    let summary = shell.summary.clone();
                    let card = cx.new(|cx| Card::new(ix, shared.clone(), summary, cx));
                    shell.cards.insert(0, card);
                })
            }
            Change::RemoveCard { ix } => self.update_shells(move |shell, _| {
                if shell.cards.len() > 1 {
                    let ix = ix % shell.cards.len();
                    shell.cards.remove(ix);
                }
            }),
            Change::Scroll { top } => self.update_shells(move |shell, _| {
                shell.list_state.scroll_to(ListOffset {
                    item_ix: top,
                    offset_in_item: px(0.),
                })
            }),
            Change::Mouse { x, y } => {
                for window in self.windows() {
                    self.cx.update_window(window.into(), |_, window, cx| {
                        window.dispatch_event(
                            crate::PlatformInput::MouseMove(crate::MouseMoveEvent {
                                position: point(px(x), px(y)),
                                pressed_button: None,
                                modifiers: Modifiers::default(),
                            }),
                            cx,
                        );
                    })
                    .unwrap();
                }
            }
            Change::Wheel { delta } => {
                for window in self.windows() {
                    let viewport = window
                        .read_with(&self.cx, |shell, _| shell.list_state.viewport_bounds())
                        .unwrap();
                    self.cx
                        .update_window(window.into(), |_, window, cx| {
                            window.dispatch_event(
                                crate::PlatformInput::ScrollWheel(crate::ScrollWheelEvent {
                                    position: viewport.center(),
                                    delta: crate::ScrollDelta::Pixels(point(px(0.), px(delta))),
                                    modifiers: Modifiers::default(),
                                    touch_phase: crate::TouchPhase::Moved,
                                }),
                                cx,
                            );
                        })
                        .unwrap();
                }
            }
            Change::Resize { width, height } => {
                for window in self.windows() {
                    self.cx
                        .simulate_window_resize(window.into(), size(px(width), px(height)));
                }
            }
            Change::ShellHandles => self.update_shells(|shell, _| shell.handles = !shell.handles),
            Change::Header { height } => self.update_shells(move |shell, _| shell.header = height),
            Change::CardHandles { ix } => self.update_cards(ix, |card, cx| {
                card.handles = !card.handles;
                cx.notify();
            }),
            Change::Focus { ix } => {
                for window in self.windows() {
                    self.cx
                        .update_window(window.into(), |root, window, cx| {
                            let shell = root.downcast::<Shell>().unwrap().read(cx);
                            match ix.and_then(|ix| shell.cards.get(ix)) {
                                Some(card) => {
                                    let handle = card.read(cx).focus_handle.clone();
                                    window.focus(&handle, cx);
                                }
                                None => window.blur(cx),
                            }
                        })
                        .unwrap();
                }
            }
            Change::Redraw => {}
        }
    }

    fn draw(&mut self) -> (Vec<String>, Vec<String>, crate::FrameWorkStats) {
        // The window drawing views again draws first, so that what a view
        // writes as it draws reaches the other window's readers no sooner.
        let (actual, work) = self
            .cx
            .update_window(self.retaining.into(), |_, window, cx| {
                window.draw(cx).clear(cx);
                let work = window.frame_work_stats();
                // The work of the frames drawn since the last one this drew,
                // the ones changes drew on their own included.
                window.reset_frame_work_stats(false);
                (describe_frame(window), work)
            })
            .unwrap();
        let expected = self
            .cx
            .update_window(self.from_scratch.into(), |_, window, cx| {
                window.refresh();
                window.draw(cx).clear(cx);
                describe_frame(window)
            })
            .unwrap();
        (actual, expected, work)
    }
}

/// Views drawn again, and of those drawn again moved, or around views
/// nested in them that were built, over a run.
#[derive(Default)]
struct Reuse {
    reused: u64,
    moved: u64,
    spliced: u64,
}

fn run(seed: u64, steps: usize) -> Reuse {
    let mut oracle = Oracle::new();
    let mut rng = StdRng::seed_from_u64(seed);
    let mut history: Vec<Vec<Change>> = Vec::new();
    let mut reuse = Reuse::default();
    for step in 0..steps {
        let changes: Vec<Change> = if step == 0 {
            Vec::new()
        } else {
            (0..rng.random_range(1..=3))
                .map(|_| Change::random(&mut rng))
                .collect()
        };
        for change in &changes {
            oracle.apply(change);
        }
        history.push(changes);
        let (actual, expected, work) = oracle.draw();
        reuse.reused += work.views_reused;
        reuse.moved += work.views_moved;
        reuse.spliced += work.views_spliced;
        if let Some(difference) = first_difference(&actual, &expected) {
            let history = history
                .iter()
                .enumerate()
                .map(|(step, changes)| format!("  {step}: {changes:?}"))
                .collect::<Vec<_>>()
                .join("\n");
            let only_in = |these: &[String], those: &[String]| {
                these
                    .iter()
                    .filter(|line| !those.contains(line))
                    .take(8)
                    .map(|line| format!("    {line}"))
                    .collect::<Vec<_>>()
                    .join("\n")
            };
            panic!(
                "seed {seed}, step {step}: the frame drawing views again differs from the frame \
                 drawn from scratch at {difference}\nonly drawing views again:\n{}\nonly from \
                 scratch:\n{}\nchanges so far:\n{history}",
                only_in(&actual, &expected),
                only_in(&expected, &actual),
            );
        }
    }
    reuse
}

#[test]
fn frames_drawing_views_again_match_frames_drawn_from_scratch() {
    let mut reuse = Reuse::default();
    let seeds = std::env::var("GPUI_RETAINED_VIEWS_ORACLE_SEEDS").ok().and_then(|seeds| seeds.parse().ok()).unwrap_or(16);
    let first = std::env::var("GPUI_RETAINED_VIEWS_ORACLE_FIRST_SEED")
        .ok()
        .and_then(|seed| seed.parse().ok())
        .unwrap_or(0);
    // With GPUI_RETAINED_VIEWS_ORACLE_ALL=1, every seed runs and the ones
    // that failed are listed together at the end.
    let all = std::env::var("GPUI_RETAINED_VIEWS_ORACLE_ALL").is_ok_and(|value| value == "1");
    let mut failures = Vec::new();
    // GPUI_RETAINED_VIEWS_ORACLE_LIST=594,606 runs only those seeds.
    let listed: Option<Vec<u64>> = std::env::var("GPUI_RETAINED_VIEWS_ORACLE_LIST")
        .ok()
        .map(|list| list.split(',').filter_map(|seed| seed.trim().parse().ok()).collect());
    let listed_any = listed.is_some();
    for seed in listed.unwrap_or_else(|| (first..seeds).collect()) {
        let run = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run(seed, 50))) {
            Ok(run) => run,
            Err(panic) => {
                let message = panic
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| panic.downcast_ref::<&str>().map(|message| message.to_string()))
                    .unwrap_or_default();
                if !all {
                    panic!("seed {seed}: {message}");
                }
                let first_line: String = message.lines().next().unwrap_or_default().chars().take(400).collect();
                failures.push(format!("seed {seed}: {first_line}"));
                continue;
            }
        };
        reuse.reused += run.reused;
        reuse.moved += run.moved;
        reuse.spliced += run.spliced;
    }
    assert!(failures.is_empty(), "{} seeds failed:\n{}", failures.len(), failures.join("\n"));
    if listed_any {
        return;
    }
    let Reuse {
        reused,
        moved,
        spliced,
    } = reuse;
    assert!(reused > 1000, "views were drawn again {reused} times");
    assert!(moved > 100, "views were drawn again moved {moved} times of {reused}");
    assert!(spliced > 100, "views were drawn again around others {spliced} times of {reused}");
}

/// A shell holding cards, drawn once, for the focused tests below.
fn shell_window(cx: &mut TestAppContext) -> (WindowHandle<Shell>, Rc<Shared>) {
    cx.update(|cx| {
        cx.set_view_retention(true);
        cx.set_global(Theme(0));
    });
    let model = cx.new(|_| Model {
        labels: (0..CARDS).map(|ix| ix % WORDS.len()).collect(),
    });
    let shared = Rc::new(Shared {
        model,
        registry: Rc::new(Cell::new(0)),
        dependency: DrawDependency::new(),
        untracked: Rc::new(Cell::new(0)),
        started_at: cx.executor().now(),
    });
    let window = cx.add_window({
        let shared = shared.clone();
        move |_, cx| Shell::new(&shared, cx)
    });
    draw(cx, window);
    cx.run_until_parked();
    // The second frame hovers by the first's hitboxes, which the first
    // could not.
    draw(cx, window);
    (window, shared)
}

fn draw(cx: &mut TestAppContext, window: WindowHandle<Shell>) -> crate::FrameWorkStats {
    cx.update_window(window.into(), |_, window, cx| {
        window.reset_frame_work_stats(false);
        window.draw(cx).clear(cx);
        window.frame_work_stats()
    })
    .unwrap()
}

/// The work of the frames drawn after `change`: the one a notification drew
/// on its own, if it did, or else one drawn here.
fn work_after<V: 'static>(
    cx: &mut TestAppContext,
    window: WindowHandle<V>,
    change: impl FnOnce(&mut TestAppContext),
) -> crate::FrameWorkStats {
    cx.update_window(window.into(), |_, window, _| window.reset_frame_work_stats(false))
        .unwrap();
    change(cx);
    cx.run_until_parked();
    cx.update_window(window.into(), |_, window, cx| {
        if window.frame_work_stats().frames == 0 {
            window.draw(cx).clear(cx);
        }
        window.frame_work_stats()
    })
    .unwrap()
}

fn rebuilds(cx: &mut TestAppContext, window: WindowHandle<Shell>) -> Vec<ViewRebuildReason> {
    cx.update_window(window.into(), |_, window, _| {
        window
            .view_rebuild_reasons()
            .iter()
            .map(|(_, reason)| *reason)
            .collect()
    })
    .unwrap()
}

/// With retention off, which it is unless asked for, no view is drawn from a
/// record and none is rebuilt for a reason.
#[test]
fn retention_is_off_unless_asked_for() {
    let mut cx = TestAppContext::single();
    assert!(!cx.update(|cx| cx.view_retention()));
    cx.update(|cx| cx.set_global(Theme(0)));
    let model = cx.new(|_| Model {
        labels: vec![0; CARDS],
    });
    let shared = Rc::new(Shared {
        model,
        registry: Rc::new(Cell::new(0)),
        dependency: DrawDependency::new(),
        untracked: Rc::new(Cell::new(0)),
        started_at: cx.executor().now(),
    });
    let window = cx.add_window(move |_, cx| Shell::new(&shared, cx));
    draw(&mut cx, window);
    let work = work_after(&mut cx, window, |cx| {
        window.update(cx, |_, _, cx| cx.notify()).unwrap();
    });
    // Only the cached card is drawn from the last frame, as without
    // retention it always was.
    assert_eq!(work.views_reused, 1, "{work:?}");
    assert!(rebuilds(&mut cx, window).is_empty());
}

/// Notifying a view by id builds it, and draws the views around it again
/// around it (see [`super::splice`]), and the rest, and what read it, again.
#[test]
fn notifying_a_view_builds_only_it() {
    let mut cx = TestAppContext::single();
    let (window, _) = shell_window(&mut cx);
    let inner = window
        .read_with(&cx, |shell, cx| shell.cards[2].read(cx).inner.clone())
        .unwrap();
    let work = work_after(&mut cx, window, |cx| cx.update(|cx| cx.notify(inner.entity_id())));
    let reasons = rebuilds(&mut cx, window);
    // The inner view; its card and the shell are drawn again around it, and
    // the opted-out cards are built on every frame.
    assert_eq!(work.view_rebuilds.notified, 1, "{work:?} {reasons:?}");
    assert_eq!(work.views_spliced, 2, "{work:?} {reasons:?}");
    assert_eq!(work.view_rebuilds.entity_changed, 0, "{work:?}");
    assert!(work.views_rendered <= 1 + 2, "{work:?} {reasons:?}");

    // With splices off, the card and the shell are built.
    cx.update_window(window.into(), |_, window, _| {
        window.view_retention.splices_enabled = false;
    })
    .unwrap();
    let work = work_after(&mut cx, window, |cx| cx.update(|cx| cx.notify(inner.entity_id())));
    let reasons = rebuilds(&mut cx, window);
    assert_eq!(work.view_rebuilds.notified, 3, "{work:?} {reasons:?}");
    assert_eq!(work.views_spliced, 0, "{work:?}");
    assert!(work.views_reused >= CARDS as u64, "{work:?}");
}

/// A group container built builds the views nested in it that resolved it,
/// whose hover is by its hitbox, which a container built anew does not
/// keep; group containers elsewhere build nothing.
#[test]
fn group_hover_views_are_built_with_their_group_container_only() {
    let mut cx = TestAppContext::single();
    let (window, _) = shell_window(&mut cx);
    // The inner view of every third card hovers by its card's group.
    let card = window
        .read_with(&cx, |shell, _| shell.cards[3].clone())
        .unwrap();
    let inner = card.read_with(&cx, |card, _| card.inner.entity_id());
    let work = work_after(&mut cx, window, |cx| card.update(cx, |_, cx| cx.notify()));
    let rebuilt: Vec<_> = cx
        .update_window(window.into(), |_, window, _| window.view_rebuild_reasons().to_vec())
        .unwrap();
    assert!(
        rebuilt.contains(&(inner, ViewRebuildReason::ContextChanged)),
        "the inner view resolved the card it is in: {rebuilt:?}"
    );
    // The card and its inner view, and the opted-out cards, built on every
    // frame; the shell is drawn again around them, keeping its group
    // container, so the cards hovering by its group are drawn again.
    assert!(work.views_rendered <= 2 + 2, "{work:?} {rebuilt:?}");
    assert_eq!(work.views_spliced, 1, "{work:?} {rebuilt:?}");
}

/// A model updated and notified builds every view that read it (the inner
/// view of every card), and a global written builds the
/// views that read it and no other; each is counted under its reason.
#[test]
fn changed_models_and_globals_build_the_views_that_read_them() {
    let mut cx = TestAppContext::single();
    let (window, shared) = shell_window(&mut cx);
    let model = shared.model.clone();
    let work = work_after(&mut cx, window, |cx| {
        model.update(cx, |model, cx| {
            model.labels[0] = 7;
            cx.notify();
        })
    });
    // The inner view of every card is built, and the cards that did not
    // read the model themselves are drawn again around them.
    assert!(work.view_rebuilds.entity_changed >= CARDS as u64, "{work:?}");
    assert!(work.views_spliced >= 1, "{work:?}");
    // The summary the cards show was written after they were drawn: the
    // next frame builds them again with it.
    draw(&mut cx, window);
    let work = work_after(&mut cx, window, |cx| cx.update(|cx| cx.set_global(Theme(4))));
    assert!(work.view_rebuilds.global_changed >= 1, "{work:?}");
    assert_eq!(work.view_rebuilds.entity_changed, 0, "{work:?}");
    // The views read the global, opted out or moved are built; the shell is
    // drawn again around them with the rest.
    assert!(work.views_rendered < CARDS as u64, "{work:?}");
    assert!(work.views_spliced >= 1, "{work:?}");
}

/// With rebuild culprits on, a view built because a model it read changed
/// is counted under the model's type and where it was changed.
#[test]
fn rebuilds_name_the_entity_and_global_behind_them() {
    super::culprits::force_on();
    let mut cx = TestAppContext::single();
    let (window, shared) = shell_window(&mut cx);
    let model = shared.model.clone();
    work_after(&mut cx, window, |cx| {
        model.update(cx, |model, cx| {
            model.labels[0] = 7;
            cx.notify();
        })
    });
    work_after(&mut cx, window, |cx| cx.update(|cx| cx.set_global(Theme(4))));
    let counts = super::culprits::counts();
    let blamed = |what: &str| {
        counts
            .iter()
            .any(|(line, _)| line.contains(what) && line.contains("rebuilds_name_the_entity"))
    };
    assert!(blamed("EntityChanged <- entity gpui::window::view_retention::tests::Model"), "{counts:#?}");
    assert!(blamed("GlobalChanged <- global gpui::window::view_retention::tests::Theme"), "{counts:#?}");
}

/// A declared dependency builds the views that read it once it changes, and
/// a view that opted out is built on every frame.
#[test]
fn declared_dependencies_and_opted_out_views_are_built() {
    let mut cx = TestAppContext::single();
    let (window, shared) = shell_window(&mut cx);
    let dependency = shared.dependency.clone();
    let work = work_after(&mut cx, window, |cx| cx.update(|cx| dependency.changed(cx)));
    assert!(work.view_rebuilds.state_changed >= 1, "{work:?}");
    assert!(work.view_rebuilds.opted_out >= 1, "{work:?}");
}

/// A view that said when it would look different is drawn again until then
/// and built on the first frame after it, which the window asks for.
#[test]
fn a_view_is_built_again_at_the_time_it_asked_for() {
    let mut cx = TestAppContext::single();
    let (window, _) = shell_window(&mut cx);
    cx.update_window(window.into(), |_, window, cx| {
        window.refresh();
        window.draw(cx).clear(cx);
    })
    .unwrap();
    let before = draw(&mut cx, window);
    assert_eq!(before.view_rebuilds.deadline, 0, "{before:?}");
    let after = work_after(&mut cx, window, |cx| {
        cx.executor().advance_clock(Duration::from_millis(1100));
    });
    assert!(after.frames >= 1, "the window asks for a frame at the deadline");
    assert!(after.view_rebuilds.deadline >= 1, "{after:?}");
}

/// Views recorded before fonts were added shaped their text without them,
/// so the first frame after is built whole, and the one after that draws
/// views again.
#[test]
fn adding_fonts_builds_every_view_once() {
    let mut cx = TestAppContext::single();
    let (window, _) = shell_window(&mut cx);
    let settled = draw(&mut cx, window);
    assert!(settled.views_reused > 0, "{settled:?}");
    cx.update(|cx| cx.text_system().add_fonts(Vec::new())).unwrap();
    let after = draw(&mut cx, window);
    assert_eq!(after.views_reused, 0, "{after:?}");
    assert!(after.view_rebuilds.window_refresh > 0, "{after:?}");
    let again = draw(&mut cx, window);
    assert!(again.views_reused > 0, "{again:?}");
}

/// While accessibility is active every view is built, so that its nodes are.
#[test]
fn accessibility_builds_every_view() {
    let mut cx = TestAppContext::single();
    let (window, _) = shell_window(&mut cx);
    cx.update_window(window.into(), |_, window, _| {
        window.a11y = super::super::a11y::A11y::new(
            std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true)),
            false,
            None,
        );
    })
    .unwrap();
    let work = work_after(&mut cx, window, |cx| {
        window.update(cx, |_, _, cx| cx.notify()).unwrap();
    });
    assert_eq!(work.views_reused, 0, "{work:?}");
    assert!(work.view_rebuilds.accessibility >= CARDS as u64, "{work:?}");
}

/// Verification draws a frame that drew views again once more from scratch,
/// keeps that one, counts the work of the first, and finds nothing to report
/// when the two agree.
#[test]
fn verification_draws_the_frame_again_from_scratch() {
    let mut cx = TestAppContext::single();
    let (window, _) = shell_window(&mut cx);
    cx.update_window(window.into(), |_, window, _| {
        window.view_retention.verification_interval = Some(1);
    })
    .unwrap();
    let work = work_after(&mut cx, window, |cx| {
        window.update(cx, |_, _, cx| cx.notify()).unwrap();
    });
    assert_eq!(work.frames, 1, "{work:?}");
    assert!(work.views_spliced + work.views_reused > 0, "{work:?}");
    assert_eq!(work.view_rebuilds.window_refresh, 0, "{work:?}");
    let (kept_reused, reasons) = cx
        .update_window(window.into(), |_, window, _| {
            (
                window.rendered_frame.retained_views.reused_any,
                window.view_rebuild_reasons().to_vec(),
            )
        })
        .unwrap();
    assert!(!kept_reused, "the frame kept was drawn from scratch");
    assert!(
        !reasons
            .iter()
            .any(|(_, reason)| *reason == ViewRebuildReason::WindowRefresh),
        "{reasons:?}"
    );
    assert_eq!(first_difference(&["a".into()], &["a".into()]), None);
    assert!(first_difference(&["a".into()], &["b".into()]).is_some());
}

/// A view that read a list's state is built when the list scrolls, and not
/// when it is asked to scroll to where it already is.
#[test]
fn list_state_changes_build_the_views_that_read_it() {
    struct Reader {
        list_state: ListState,
        renders: Rc<Cell<usize>>,
    }
    impl Render for Reader {
        fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
            self.renders.set(self.renders.get() + 1);
            let top = self.list_state.logical_scroll_top().item_ix;
            div().child(SharedString::from(format!("top {top}")))
        }
    }
    struct Host {
        reader: Entity<Reader>,
        list_state: ListState,
    }
    impl Render for Host {
        fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
            div()
                .size_full()
                .child(self.reader.clone())
                .child(
                    list(self.list_state.clone(), |ix, _, _| {
                        div().h(px(20.)).child(SharedString::from(format!("row {ix}")))
                            .into_any_element()
                    })
                    .h(px(60.))
                    .w(px(100.)),
                )
        }
    }
    let mut cx = TestAppContext::single();
    cx.update(|cx| cx.set_view_retention(true));
    let list_state = ListState::new(20, ListAlignment::Top, px(0.));
    let renders = Rc::new(Cell::new(0));
    let window = cx.add_window({
        let list_state = list_state.clone();
        let renders = renders.clone();
        move |_, cx| Host {
            reader: cx.new(|_| Reader {
                list_state: list_state.clone(),
                renders,
            }),
            list_state,
        }
    });
    let frame = |cx: &mut TestAppContext| {
        cx.update_window(window.into(), |_, window, cx| window.draw(cx).clear(cx))
            .unwrap()
    };
    frame(&mut cx);
    frame(&mut cx);
    let settled = renders.get();
    list_state.scroll_to(ListOffset {
        item_ix: 0,
        offset_in_item: px(0.),
    });
    window.update(&mut cx, |_, _, cx| cx.notify()).unwrap();
    frame(&mut cx);
    assert_eq!(renders.get(), settled, "scrolling to where it is changes nothing");
    list_state.scroll_to(ListOffset {
        item_ix: 5,
        offset_in_item: px(0.),
    });
    window.update(&mut cx, |_, _, cx| cx.notify()).unwrap();
    frame(&mut cx);
    assert!(renders.get() > settled, "the reader is built after a scroll");
}

/// Keeps a view's content painted inside whatever transition its host puts
/// it in, and a host that wraps it in opacity cycles, shimmers or glass mode.
struct Mover {
    child: Entity<Leaf>,
    started_at: Instant,
    transition: Option<Duration>,
    cycle: bool,
    glass: bool,
}

struct Leaf;

impl Render for Leaf {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        div()
            .w(px(40.))
            .h(px(20.))
            .bg(PALETTE[1])
            .child("leaf")
    }
}

impl Render for Mover {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        let child = self.child.clone().into_any_element();
        let transition = self.transition.map(|duration| {
            crate::TimeTransition::new(self.started_at, duration)
                .offset(point(px(0.), px(10.)), point(px(0.), px(0.)))
        });
        let cycle = self.cycle;
        div().size_full().when(self.glass, |this| this.glass(true)).child(Wrapper {
            child: Some(child),
            transition,
            cycle,
        })
    }
}

/// Paints its child inside a time transition or an opacity cycle.
struct Wrapper {
    child: Option<AnyElement>,
    transition: Option<crate::TimeTransition>,
    cycle: bool,
}

impl IntoElement for Wrapper {
    type Element = Self;

    fn into_element(self) -> Self {
        self
    }
}

impl crate::Element for Wrapper {
    type RequestLayoutState = AnyElement;
    type PrepaintState = ();

    fn id(&self) -> Option<crate::ElementId> {
        None
    }

    fn source_location(&self) -> Option<&'static core::panic::Location<'static>> {
        None
    }

    fn request_layout(
        &mut self,
        _: Option<&crate::GlobalElementId>,
        _: Option<&crate::InspectorElementId>,
        window: &mut Window,
        cx: &mut App,
    ) -> (crate::LayoutId, AnyElement) {
        let mut child = self.child.take().unwrap_or_else(|| div().into_any_element());
        let child_layout = child.request_layout(window, cx);
        let layout = window.request_layout(crate::Style::default(), [child_layout], cx);
        (layout, child)
    }

    fn prepaint(
        &mut self,
        _: Option<&crate::GlobalElementId>,
        _: Option<&crate::InspectorElementId>,
        _: crate::Bounds<crate::Pixels>,
        child: &mut AnyElement,
        window: &mut Window,
        cx: &mut App,
    ) {
        child.prepaint(window, cx);
    }

    fn paint(
        &mut self,
        _: Option<&crate::GlobalElementId>,
        _: Option<&crate::InspectorElementId>,
        _: crate::Bounds<crate::Pixels>,
        child: &mut AnyElement,
        _: &mut (),
        window: &mut Window,
        cx: &mut App,
    ) {
        let cycle = self.cycle;
        let mut paint = |window: &mut Window, cx: &mut App| {
            if cycle {
                window.with_opacity_cycle(
                    crate::OpacityCycle::new(
                        Duration::from_secs(1),
                        [(0.0, 0.2), (0.5, 1.0), (0.75, 0.6), (1.0, 0.2)],
                    ),
                    |window| child.paint(window, cx),
                );
            } else {
                child.paint(window, cx);
            }
        };
        match self.transition {
            Some(transition) => window.with_time_transition(transition, |window| paint(window, cx)),
            None => paint(window, cx),
        }
    }
}

fn mover_windows(cx: &mut TestAppContext) -> [WindowHandle<Mover>; 2] {
    cx.update(|cx| cx.set_view_retention(true));
    let started_at = cx.executor().now();
    [(); 2].map(|_| {
        cx.add_window(move |_, cx| Mover {
            child: cx.new(|_| Leaf),
            started_at,
            transition: None,
            cycle: false,
            glass: false,
        })
    })
}

/// Draws the first window drawing views again and the second from scratch,
/// returning both frames with the scene's transitions.
fn draw_pair(cx: &mut TestAppContext, windows: [WindowHandle<Mover>; 2]) -> [Vec<String>; 2] {
    windows.map(|window| {
        cx.update_window(window.into(), |_, window, cx| {
            if window.handle.window_id() == windows[1].window_id() {
                window.refresh();
            }
            window.draw(cx).clear(cx);
            let mut lines = describe_frame(window);
            lines.extend(
                window
                    .rendered_frame
                    .scene
                    .transitions
                    .iter()
                    .map(|transition| format!("transition parent {}", transition.parent)),
            );
            lines
        })
        .unwrap()
    })
}

/// A view drawn again inside a transition its host started since is moved
/// with that transition, not with the one it was painted in.
#[test]
fn a_view_drawn_again_moves_with_the_transition_it_is_drawn_in() {
    let mut cx = TestAppContext::single();
    let windows = mover_windows(&mut cx);
    let mut change = |cx: &mut TestAppContext, transition: Option<Duration>| {
        for window in windows {
            window
                .update(cx, |mover, _, cx| {
                    mover.transition = transition;
                    cx.notify();
                })
                .unwrap();
        }
        let [actual, expected] = draw_pair(cx, windows);
        assert_eq!(first_difference(&actual, &expected), None);
    };
    change(&mut cx, None);
    // Long enough not to land while the test runs: a landed transition is
    // left out of content drawn again.
    change(&mut cx, Some(Duration::from_secs(600)));
    cx.executor().advance_clock(Duration::from_millis(50));
    change(&mut cx, Some(Duration::from_secs(900)));
    change(&mut cx, None);
    let reused = cx
        .update_window(windows[0].into(), |_, window, _| window.frame_work_stats().views_reused)
        .unwrap();
    assert!(reused >= 3, "the leaf was drawn again: {reused}");
}

/// A view drawn again inside an opacity cycle it was not painted in asks for
/// the next frame, on which it is built inside it.
#[test]
fn a_view_drawn_again_into_another_opacity_cycle_is_built_on_the_next_frame() {
    let mut cx = TestAppContext::single();
    let windows = mover_windows(&mut cx);
    draw_pair(&mut cx, windows);
    for window in windows {
        window
            .update(&mut cx, |mover, _, cx| {
                mover.cycle = true;
                cx.notify();
            })
            .unwrap();
    }
    draw_pair(&mut cx, windows);
    // The cycle stamps the opacity it has now, which moves with the clock,
    // so what is compared is which quads it animates.
    let cycled = |cx: &mut TestAppContext, window: WindowHandle<Mover>| {
        cx.update_window(window.into(), |_, window, _| {
            window
                .rendered_frame
                .scene
                .quads
                .iter()
                .map(|quad| quad.background.time_animation() != 0)
                .collect::<Vec<_>>()
        })
        .unwrap()
    };
    assert_ne!(cycled(&mut cx, windows[0]), cycled(&mut cx, windows[1]));
    let asked = cx
        .update_window(windows[0].into(), |_, window, cx| window.simulate_next_frame(cx))
        .unwrap();
    assert!(asked >= 1, "the reused leaf asks for the next frame");
    draw_pair(&mut cx, windows);
    assert_eq!(cycled(&mut cx, windows[0]), cycled(&mut cx, windows[1]));
}

/// A view whose host turned glass mode on around it is built inside it in
/// the same frame: glass mode is applied while the host prepaints too, where
/// the view finds it differs from the last frame's.
#[test]
fn a_view_drawn_into_glass_mode_is_built_inside_it() {
    let mut cx = TestAppContext::single();
    let windows = mover_windows(&mut cx);
    draw_pair(&mut cx, windows);
    for window in windows {
        window
            .update(&mut cx, |mover, _, cx| {
                mover.glass = true;
                cx.notify();
            })
            .unwrap();
    }
    let [retained, from_scratch] = draw_pair(&mut cx, windows);
    assert_eq!(first_difference(&retained, &from_scratch), None);
    let glass = |cx: &mut TestAppContext, window: WindowHandle<Mover>| {
        cx.update_window(window.into(), |_, window, _| {
            window
                .rendered_frame
                .scene
                .quads
                .iter()
                .map(|quad| quad.background.is_glass_content())
                .collect::<Vec<_>>()
        })
        .unwrap()
    };
    assert_eq!(glass(&mut cx, windows[0]), glass(&mut cx, windows[1]));
    assert!(glass(&mut cx, windows[0]).iter().any(|glass| *glass));
    let asked = cx
        .update_window(windows[0].into(), |_, window, cx| window.simulate_next_frame(cx))
        .unwrap();
    assert_eq!(asked, 0, "nothing was drawn in the wrong mode");
}

/// A view whose host swapped the image cache it inherits for another is
/// built again, so that its images load through the new one.
#[test]
fn a_view_under_another_image_cache_is_built_again() {
    struct Counted(Rc<Cell<usize>>);
    impl Render for Counted {
        fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
            self.0.set(self.0.get() + 1);
            div().w(px(40.)).h(px(20.)).bg(PALETTE[1])
        }
    }
    struct Host {
        caches: [Entity<crate::RetainAllImageCache>; 2],
        current: usize,
        child: Entity<Counted>,
    }
    impl Render for Host {
        fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
            div()
                .size_full()
                .image_cache(self.caches[self.current].clone())
                .child(self.child.clone())
        }
    }
    let mut cx = TestAppContext::single();
    cx.update(|cx| cx.set_view_retention(true));
    let renders = Rc::new(Cell::new(0));
    let window = cx.add_window({
        let renders = renders.clone();
        move |_, cx| Host {
            caches: [
                crate::RetainAllImageCache::new(cx),
                crate::RetainAllImageCache::new(cx),
            ],
            current: 0,
            child: cx.new(|_| Counted(renders)),
        }
    });
    let draw = |cx: &mut TestAppContext| {
        cx.update_window(window.into(), |_, window, cx| window.draw(cx).clear(cx))
            .unwrap();
    };
    draw(&mut cx);
    cx.run_until_parked();
    let settled = renders.get();
    window.update(&mut cx, |_, _, cx| cx.notify()).unwrap();
    draw(&mut cx);
    assert_eq!(renders.get(), settled, "under the same cache the child is drawn again");
    window
        .update(&mut cx, |host, _, cx| {
            host.current = 1;
            cx.notify();
        })
        .unwrap();
    draw(&mut cx);
    assert!(renders.get() > settled, "under another cache the child is built");
}

/// A view holding a draw deferred beneath native surfaces, drawn again,
/// keeps it on the base plane: the overlay stays empty.
#[test]
fn a_view_drawn_again_keeps_its_deferred_draw_beneath_native_surfaces() {
    struct Holder;
    impl Render for Holder {
        fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
            div().size_full().child(
                deferred(div().w(px(30.)).h(px(30.)).bg(PALETTE[1])).beneath_native_surfaces(),
            )
        }
    }
    struct Host {
        holder: Entity<Holder>,
    }
    impl Render for Host {
        fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
            div().size_full().child(self.holder.clone())
        }
    }
    let mut cx = TestAppContext::single();
    cx.update(|cx| cx.set_view_retention(true));
    let window = cx.add_window(|_, cx| Host {
        holder: cx.new(|_| Holder),
    });
    let overlay = |cx: &mut TestAppContext| {
        cx.update_window(window.into(), |_, window, cx| {
            window.reset_frame_work_stats(false);
            window.draw(cx).clear(cx);
            let frame = &window.rendered_frame;
            (
                frame.scene.len() - frame.overlay_scene_start,
                frame.deferred_draws.iter().all(|draw| draw.beneath_native_surfaces),
                window.frame_work_stats().views_reused,
            )
        })
        .unwrap()
    };
    assert_eq!(overlay(&mut cx).0, 0);
    window.update(&mut cx, |_, _, cx| cx.notify()).unwrap();
    let (overlay_primitives, beneath, reused) = overlay(&mut cx);
    assert_eq!(reused, 1);
    assert!(beneath);
    assert_eq!(overlay_primitives, 0);
}

/// A view drawn again keeps its window-control areas and its hitboxes' ids,
/// which hover and pointer capture are keyed by.
#[test]
fn a_view_drawn_again_keeps_its_hitboxes_and_window_controls() {
    struct Chrome;
    impl Render for Chrome {
        fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
            div()
                .id("caption")
                .w(px(200.))
                .h(px(30.))
                .window_control_area(crate::WindowControlArea::Drag)
                .hover(|style| style.bg(PALETTE[2]))
        }
    }
    struct Host {
        chrome: Entity<Chrome>,
    }
    impl Render for Host {
        fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
            div().size_full().child(self.chrome.clone())
        }
    }
    let mut cx = TestAppContext::single();
    cx.update(|cx| cx.set_view_retention(true));
    let window = cx.add_window(|_, cx| Host {
        chrome: cx.new(|_| Chrome),
    });
    let frame = |cx: &mut TestAppContext| {
        cx.update_window(window.into(), |_, window, cx| {
            window.reset_frame_work_stats(false);
            window.draw(cx).clear(cx);
            let frame = &window.rendered_frame;
            (
                frame.hitboxes.iter().map(|hitbox| hitbox.id).collect::<Vec<_>>(),
                frame.window_control_hitboxes.len(),
                window.frame_work_stats().views_reused,
            )
        })
        .unwrap()
    };
    let (hitboxes, controls, _) = frame(&mut cx);
    window.update(&mut cx, |_, _, cx| cx.notify()).unwrap();
    let (reused_hitboxes, reused_controls, reused) = frame(&mut cx);
    assert_eq!(reused, 1);
    assert_eq!(reused_hitboxes, hitboxes);
    assert_eq!(reused_controls, controls);
    // Hovering the caption, whose hover the chrome was painted by, builds it.
    cx.update_window(window.into(), |_, window, cx| {
        window.dispatch_event(
            crate::PlatformInput::MouseMove(crate::MouseMoveEvent {
                position: point(px(10.), px(10.)),
                pressed_button: None,
                modifiers: Modifiers::default(),
            }),
            cx,
        );
    })
    .unwrap();
    cx.run_until_parked();
    frame(&mut cx);
    let hovered = cx
        .update_window(window.into(), |_, window, _| {
            window
                .rendered_frame
                .scene
                .quads
                .iter()
                .any(|quad| format!("{:?}", quad.background).contains("0.3"))
        })
        .unwrap();
    assert!(hovered, "the caption is painted hovered");
}

/// Views drawn inside a prepaint that is rolled back leave no records
/// behind: a list scrolling an item into view lays its items out twice.
#[test]
fn a_rolled_back_prepaint_leaves_no_records() {
    let mut oracle = Oracle::new();
    for (top, reveal) in [(0, 2), (0, 1), (3, 5), (2, 4), (0, 3)] {
        // The card revealed is out of view, so the list's prepaint scrolls
        // it into view: the first attempt is rolled back.
        oracle.apply(&Change::Scroll { top });
        let (actual, expected, _) = oracle.draw();
        assert_eq!(first_difference(&actual, &expected), None);
        oracle.apply(&Change::Reveal { ix: reveal });
        let (actual, expected, _) = oracle.draw();
        assert_eq!(first_difference(&actual, &expected), None);
        // Every view drawn in that frame left one record, which the next
        // frame draws it again from.
        let records = oracle
            .cx
            .update_window(oracle.retaining.into(), |_, window, _| {
                let views = &window.rendered_frame.retained_views;
                views
                    .records
                    .iter()
                    .filter(|record| views.find(&record.id).is_none())
                    .count()
            })
            .unwrap();
        assert_eq!(records, 0, "records left unfindable by a rolled-back prepaint");
        oracle.apply(&Change::Tint);
        let (actual, expected, _) = oracle.draw();
        assert_eq!(first_difference(&actual, &expected), None);
        oracle.apply(&Change::Reveal { ix: reveal });
    }
}

/// A board of panels, each a view of labels, one of which changes a label
/// per frame.
struct Board {
    panels: Vec<Entity<Panel>>,
}

struct Panel {
    ix: usize,
    value: usize,
}

impl Render for Panel {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        div()
            .flex()
            .flex_col()
            .w(px(180.))
            .p_1()
            .border_1()
            .border_color(PALETTE[self.ix % PALETTE.len()])
            .children((0..10).map(|row| {
                div()
                    .flex()
                    .flex_row()
                    .gap_1()
                    .child(div().w(px(6.)).h(px(6.)).bg(PALETTE[row % PALETTE.len()]))
                    .child(SharedString::from(format!(
                        "{} {}",
                        WORDS[(self.ix + row) % WORDS.len()],
                        self.value + row
                    )))
            }))
    }
}

impl Render for Board {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        div()
            .size_full()
            .flex()
            .flex_row()
            .flex_wrap()
            .gap_1()
            .children(self.panels.iter().cloned())
    }
}

/// The work of drawing a board of 60 panels, one of which changes per frame,
/// with views drawn again and without. Run with `--ignored --nocapture`.
#[test]
#[ignore]
fn frame_work_with_views_drawn_again() {
    for retained in [false, true] {
        let mut cx = TestAppContext::single();
        cx.update(|cx| cx.set_view_retention(retained));
        let window = cx.add_window(|_, cx| Board {
            panels: (0..60).map(|ix| cx.new(|_| Panel { ix, value: 0 })).collect(),
        });
        cx.simulate_window_resize(window.into(), size(px(1600.), px(1200.)));
        for _ in 0..3 {
            draw_board(&mut cx, window);
        }
        cx.update_window(window.into(), |_, window, _| window.reset_frame_work_stats(true))
            .unwrap();
        let started = Instant::now();
        let frames = 200;
        for frame in 0..frames {
            let panel = window
                .read_with(&cx, |board, _| board.panels[frame % 60].clone())
                .unwrap();
            panel.update(&mut cx, |panel, cx| {
                panel.value += 1;
                cx.notify();
            });
            draw_board(&mut cx, window);
        }
        let elapsed = started.elapsed();
        let work = cx
            .update_window(window.into(), |_, window, _| window.frame_work_stats())
            .unwrap();
        eprintln!(
            "retained {retained}: {:?} per frame over {} frames; elements {} views rendered {} \
             reused {} layout nodes {} build {:?} prepaint {:?} paint {:?}",
            elapsed / frames as u32,
            work.frames,
            work.elements / work.frames,
            work.views_rendered / work.frames,
            work.views_reused / work.frames,
            work.layout_nodes / work.frames,
            work.build_time / work.frames as u32,
            work.prepaint_time / work.frames as u32,
            work.paint_time / work.frames as u32,
        );
    }
}

fn draw_board(cx: &mut TestAppContext, window: WindowHandle<Board>) {
    cx.update_window(window.into(), |_, window, cx| window.draw(cx).clear(cx))
        .unwrap();
}

/// A view beside a list, reading where it is scrolled to.
struct ScrollReader {
    list_state: ListState,
    seen: Rc<Cell<usize>>,
}

impl Render for ScrollReader {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        let top = self.list_state.logical_scroll_top();
        self.seen.set(top.item_ix);
        div().child(SharedString::from(format!(
            "top {} {}",
            top.item_ix,
            top.offset_in_item.as_f32().round()
        )))
    }
}

struct ScrollHost {
    reader: Entity<ScrollReader>,
    list_state: ListState,
}

impl Render for ScrollHost {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        div().size_full().child(self.reader.clone()).child(
            list(self.list_state.clone(), |ix, _, _| {
                div()
                    .h(px(20.))
                    .child(SharedString::from(format!("row {ix}")))
                    .into_any_element()
            })
            .h(px(60.))
            .w(px(100.)),
        )
    }
}

/// A view reading a list's scroll position is built again once the wheel
/// scrolls the list.
#[test]
fn a_wheel_scroll_builds_the_views_reading_the_list() {
    let mut cx = TestAppContext::single();
    cx.update(|cx| cx.set_view_retention(true));
    let list_state = ListState::new(20, ListAlignment::Top, px(0.)).measure_all();
    let seen = Rc::new(Cell::new(usize::MAX));
    let window = cx.add_window({
        let list_state = list_state.clone();
        let seen = seen.clone();
        move |_, cx| ScrollHost {
            reader: cx.new(|_| ScrollReader {
                list_state: list_state.clone(),
                seen,
            }),
            list_state,
        }
    });
    let frame = |cx: &mut TestAppContext| {
        cx.update_window(window.into(), |_, window, cx| window.draw(cx).clear(cx))
            .unwrap()
    };
    frame(&mut cx);
    frame(&mut cx);
    assert_eq!(seen.get(), 0);
    let viewport = list_state.viewport_bounds();
    cx.update_window(window.into(), |_, window, cx| {
        window.dispatch_event(
            crate::PlatformInput::ScrollWheel(crate::ScrollWheelEvent {
                position: viewport.center(),
                delta: crate::ScrollDelta::Pixels(point(px(0.), px(-100.))),
                modifiers: Modifiers::default(),
                touch_phase: crate::TouchPhase::Moved,
            }),
            cx,
        );
    })
    .unwrap();
    let top = list_state.logical_scroll_top().item_ix;
    assert!(top > 0, "the wheel scrolled the list");
    frame(&mut cx);
    frame(&mut cx);
    assert_eq!(seen.get(), top, "the reader shows where the list is scrolled to");
}

/// With retention on, the wheel scrolling a list asks for a frame rather
/// than notifying the view drawing the list: that view depends on the wheel
/// and is built anyway, and its observers are not woken for every frame of a
/// scroll.
#[test]
fn a_wheel_scroll_does_not_notify_the_view_drawing_the_list() {
    struct ListHost {
        list_state: ListState,
    }
    impl Render for ListHost {
        fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
            div().size_full().child(
                list(self.list_state.clone(), |ix, _, _| {
                    div()
                        .h(px(20.))
                        .child(SharedString::from(format!("row {ix}")))
                        .into_any_element()
                })
                .h(px(60.))
                .w(px(100.)),
            )
        }
    }
    let mut cx = TestAppContext::single();
    cx.update(|cx| cx.set_view_retention(true));
    let list_state = ListState::new(20, ListAlignment::Top, px(0.)).measure_all();
    let window = cx.add_window({
        let list_state = list_state.clone();
        move |_, _| ListHost { list_state }
    });
    let host = window.root(&mut cx).unwrap();
    let notified = Rc::new(Cell::new(0));
    let _observation = cx.update({
        let notified = notified.clone();
        |cx| cx.observe(&host, move |_, _| notified.set(notified.get() + 1))
    });
    let frame = |cx: &mut TestAppContext| {
        cx.update_window(window.into(), |_, window, cx| {
            window.draw(cx).clear(cx);
            describe_frame(window)
        })
        .unwrap()
    };
    frame(&mut cx);
    frame(&mut cx);
    let viewport = list_state.viewport_bounds();
    for _ in 0..3 {
        cx.update_window(window.into(), |_, window, cx| {
            window.dispatch_event(
                crate::PlatformInput::ScrollWheel(crate::ScrollWheelEvent {
                    position: viewport.center(),
                    delta: crate::ScrollDelta::Pixels(point(px(0.), px(-30.))),
                    modifiers: Modifiers::default(),
                    touch_phase: crate::TouchPhase::Moved,
                }),
                cx,
            );
        })
        .unwrap();
        let scrolled = frame(&mut cx);
        cx.update_window(window.into(), |_, window, _| window.refresh())
            .unwrap();
        let from_scratch = frame(&mut cx);
        assert_eq!(first_difference(&scrolled, &from_scratch), None);
    }
    assert!(list_state.logical_scroll_top().item_ix > 0, "the wheel scrolled the list");
    assert_eq!(notified.get(), 0, "the view drawing the list was not notified");
}

/// A view reading a uniform list's offset is built again once the list
/// scrolls an item into view as it prepaints.
#[test]
fn a_uniform_list_scrolling_to_an_item_builds_the_views_reading_it() {
    struct Reader {
        handle: crate::UniformListScrollHandle,
        seen: Rc<Cell<Pixels>>,
    }
    impl Render for Reader {
        fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
            let offset = self.handle.0.borrow().base_handle.offset().y;
            self.seen.set(offset);
            div().child(SharedString::from(format!("{offset:?}")))
        }
    }
    struct Host {
        reader: Entity<Reader>,
        handle: crate::UniformListScrollHandle,
    }
    impl Render for Host {
        fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
            div().size_full().child(self.reader.clone()).child(
                crate::uniform_list("rows", 50, |range, _, _| {
                    range
                        .map(|ix| div().h(px(20.)).child(SharedString::from(format!("row {ix}"))))
                        .collect()
                })
                .track_scroll(&self.handle)
                .h(px(60.))
                .w(px(100.)),
            )
        }
    }
    let mut cx = TestAppContext::single();
    cx.update(|cx| cx.set_view_retention(true));
    let handle = crate::UniformListScrollHandle::new();
    let seen = Rc::new(Cell::new(px(1.)));
    let window = cx.add_window({
        let handle = handle.clone();
        let seen = seen.clone();
        move |_, cx| Host {
            reader: cx.new(|_| Reader {
                handle: handle.clone(),
                seen,
            }),
            handle,
        }
    });
    let frame = |cx: &mut TestAppContext| {
        cx.update_window(window.into(), |_, window, cx| window.draw(cx).clear(cx))
            .unwrap()
    };
    frame(&mut cx);
    frame(&mut cx);
    assert_eq!(seen.get(), px(0.));
    handle.scroll_to_item(30, crate::ScrollStrategy::Top);
    window.update(&mut cx, |_, _, cx| cx.notify()).unwrap();
    frame(&mut cx);
    frame(&mut cx);
    let offset = handle.0.borrow().base_handle.offset().y;
    assert!(offset < px(0.), "the list scrolled");
    assert_eq!(seen.get(), offset, "the reader shows the list's offset");

    // Asked to scroll by someone who notifies no view, the list scrolls on
    // the next frame drawn: the view drawing it depends on what it is asked.
    handle.scroll_to_item(2, crate::ScrollStrategy::Top);
    frame(&mut cx);
    frame(&mut cx);
    let scrolled_back = handle.0.borrow().base_handle.offset().y;
    assert_eq!(scrolled_back, px(-40.), "the list scrolled back");
    assert_eq!(seen.get(), scrolled_back);
}

struct Setting(usize);

impl Global for Setting {}

/// A view reading a global through `update_global` is built again once the
/// global is set, and one reading it back after writing it as it renders
/// does not depend on its own write.
#[test]
fn a_global_read_through_an_update_is_a_dependency() {
    struct Reader {
        seen: Rc<Cell<usize>>,
        renders: Rc<Cell<usize>>,
    }
    impl Render for Reader {
        fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
            self.renders.set(self.renders.get() + 1);
            let value = cx.update_global::<Setting, _>(|setting, _| setting.0);
            self.seen.set(value);
            div().child(SharedString::from(format!("{value}")))
        }
    }
    struct Host {
        reader: Entity<Reader>,
    }
    impl Render for Host {
        fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
            div().size_full().child(self.reader.clone())
        }
    }
    let mut cx = TestAppContext::single();
    cx.update(|cx| {
        cx.set_view_retention(true);
        cx.set_global(Setting(0));
    });
    let seen = Rc::new(Cell::new(usize::MAX));
    let renders = Rc::new(Cell::new(0));
    let window = cx.add_window({
        let seen = seen.clone();
        let renders = renders.clone();
        move |_, cx| Host {
            reader: cx.new(|_| Reader { seen, renders }),
        }
    });
    let frame = |cx: &mut TestAppContext| {
        cx.update_window(window.into(), |_, window, cx| window.draw(cx).clear(cx))
            .unwrap()
    };
    frame(&mut cx);
    frame(&mut cx);
    let settled = renders.get();
    window.update(&mut cx, |_, _, cx| cx.notify()).unwrap();
    frame(&mut cx);
    assert_eq!(renders.get(), settled, "its own write does not build it again");
    cx.update(|cx| cx.set_global(Setting(5)));
    window.update(&mut cx, |_, _, cx| cx.notify()).unwrap();
    frame(&mut cx);
    assert_eq!(seen.get(), 5, "the reader shows the global's value");
}

/// Setting a global to what it holds, or updating it without changing it,
/// through the change-only setters builds no view that read it.
#[test]
fn unchanged_globals_set_through_the_change_only_setters_build_nothing() {
    let mut cx = TestAppContext::single();
    let (window, _) = shell_window(&mut cx);
    let work = work_after(&mut cx, window, |cx| {
        cx.update(|cx| {
            assert!(!cx.set_global_if_changed(Theme(0)));
            window.update(cx, |_, _, cx| cx.notify()).unwrap();
        })
    });
    assert_eq!(work.view_rebuilds.global_changed, 0, "{work:?}");
    let work = work_after(&mut cx, window, |cx| {
        cx.update(|cx| cx.set_global_if_changed(Theme(2)));
    });
    assert!(work.view_rebuilds.global_changed >= 1, "{work:?}");
}

/// A view that read a model before a view drawn after it wrote the model
/// during the same draw shows what was written on the next frame, though the
/// view around both is drawn again as a whole.
#[test]
fn a_write_during_a_draw_reaches_the_views_that_read_before_it() {
    struct Measured(usize);
    struct Reader {
        model: Entity<Measured>,
        seen: Rc<Cell<usize>>,
    }
    impl Render for Reader {
        fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
            let width = self.model.read(cx).0;
            self.seen.set(width);
            div().child(SharedString::from(format!("{width}")))
        }
    }
    struct Writer {
        model: Entity<Measured>,
    }
    impl Render for Writer {
        fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
            self.model.update(cx, |model, _| {
                if model.0 != 100 {
                    model.0 = 100;
                }
            });
            div()
        }
    }
    struct Parent {
        reader: Entity<Reader>,
        writer: Entity<Writer>,
    }
    impl Render for Parent {
        fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
            div().child(self.reader.clone()).child(self.writer.clone())
        }
    }
    struct Other(usize);
    impl Render for Other {
        fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
            div().child(SharedString::from(format!("{}", self.0)))
        }
    }
    struct Root {
        parent: Entity<Parent>,
        other: Entity<Other>,
    }
    impl Render for Root {
        fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
            div().child(self.parent.clone()).child(self.other.clone())
        }
    }
    let mut cx = TestAppContext::single();
    cx.update(|cx| cx.set_view_retention(true));
    let model = cx.new(|_| Measured(0));
    let seen = Rc::new(Cell::new(usize::MAX));
    let other = cx.new(|_| Other(0));
    let window = cx.add_window({
        let seen = seen.clone();
        let other = other.clone();
        move |_, cx| Root {
            parent: cx.new(|cx| Parent {
                reader: cx.new(|_| Reader {
                    model: model.clone(),
                    seen,
                }),
                writer: cx.new(|_| Writer { model }),
            }),
            other,
        }
    });
    let frame = |cx: &mut TestAppContext| {
        cx.update_window(window.into(), |_, window, cx| window.draw(cx).clear(cx))
            .unwrap()
    };
    frame(&mut cx);
    for _ in 0..3 {
        other.update(&mut cx, |other, cx| {
            other.0 += 1;
            cx.notify();
        });
        frame(&mut cx);
    }
    assert_eq!(seen.get(), 100, "the reader shows what was written after it read");
}

/// A view that opted out is built on every frame it is drawn in, though the
/// view around it has nothing to be built for.
#[test]
fn a_view_that_opted_out_is_built_inside_a_view_drawn_again() {
    struct Untracked {
        value: Rc<Cell<usize>>,
    }
    impl Render for Untracked {
        fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
            cx.set_view_retainable(false);
            div()
                .w(px(10. + self.value.get() as f32))
                .h(px(10.))
                .bg(PALETTE[1])
        }
    }
    struct Host {
        child: Entity<Untracked>,
    }
    impl Render for Host {
        fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
            div().size_full().child(div().child(self.child.clone()))
        }
    }
    let mut cx = TestAppContext::single();
    cx.update(|cx| cx.set_view_retention(true));
    let value = Rc::new(Cell::new(0));
    let window = cx.add_window({
        let value = value.clone();
        move |_, cx| Host {
            child: cx.new(|_| Untracked { value }),
        }
    });
    let frame = |cx: &mut TestAppContext| {
        cx.update_window(window.into(), |_, window, cx| {
            window.draw(cx).clear(cx);
            window.rendered_frame.scene.monochrome_sprites.len() + window.rendered_frame.scene.len()
        })
        .unwrap()
    };
    frame(&mut cx);
    frame(&mut cx);
    let text = |cx: &mut TestAppContext| {
        cx.update_window(window.into(), |_, window, _| describe_frame(window))
            .unwrap()
    };
    let before = text(&mut cx);
    value.set(30);
    frame(&mut cx);
    assert_ne!(text(&mut cx), before, "the view that opted out was built");
}

/// A view holding another, to nest the board below a chain of views.
struct Wrap {
    child: crate::AnyView,
}

impl Render for Wrap {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        div().size_full().child(self.child.clone())
    }
}

/// The work of drawing a board of 240 panels, one of which changes per
/// frame, directly in the window and below a chain of twelve views, with
/// views drawn again and without: what each view drawn again costs should
/// not grow with how many nodes and views are nested in it. Run with
/// `--ignored --nocapture`.
#[test]
#[ignore]
fn frame_work_by_view_depth() {
    for depth in [0usize, 12] {
        for retained in [false, true] {
            let mut cx = TestAppContext::single();
            cx.update(|cx| cx.set_view_retention(retained));
            let panels: Vec<_> = (0..240)
                .map(|ix| cx.new(|_| Panel { ix, value: 0 }))
                .collect();
            let board = cx.new(|_| Board {
                panels: panels.clone(),
            });
            let mut top: crate::AnyView = board.into();
            for _ in 0..depth {
                let child = top.clone();
                top = cx.new(|_| Wrap { child }).into();
            }
            let window = cx.add_window(move |_, _| Wrap { child: top });
            cx.simulate_window_resize(window.into(), size(px(3200.), px(2400.)));
            let frame = |cx: &mut TestAppContext| {
                cx.update_window(window.into(), |_, window, cx| window.draw(cx).clear(cx))
                    .unwrap();
            };
            for _ in 0..3 {
                frame(&mut cx);
            }
            cx.update_window(window.into(), |_, window, _| window.reset_frame_work_stats(true))
                .unwrap();
            let frames = 60;
            let started = Instant::now();
            for frame_ix in 0..frames {
                panels[frame_ix % panels.len()].update(&mut cx, |panel, cx| {
                    panel.value += 1;
                    cx.notify();
                });
                frame(&mut cx);
            }
            let elapsed = started.elapsed();
            let work = cx
                .update_window(window.into(), |_, window, _| window.frame_work_stats())
                .unwrap();
            eprintln!(
                "depth {depth} retained {retained}: {:?} per change; rendered {} reused {} \
                 layout nodes {} per frame",
                elapsed / frames as u32,
                work.views_rendered / work.frames,
                work.views_reused / work.frames,
                work.layout_nodes / work.frames,
            );
        }
    }
}

/// A view painting inside a transition of its own that has landed where it
/// started, drawn again: the transition is left out rather than copied frame
/// after frame, and the view paints where it would with it.
#[test]
fn a_landed_transition_is_left_out_of_a_view_drawn_again() {
    struct Settled {
        started_at: Instant,
    }
    impl Render for Settled {
        fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
            div().size_full().child(Wrapper {
                child: Some(
                    div()
                        .w(px(40.))
                        .h(px(20.))
                        .bg(PALETTE[1])
                        .into_any_element(),
                ),
                transition: Some(
                    crate::TimeTransition::new(self.started_at, Duration::from_millis(10))
                        .offset(point(px(0.), px(10.)), point(px(0.), px(0.))),
                ),
                cycle: false,
            })
        }
    }
    struct Host {
        settled: Entity<Settled>,
        tint: usize,
    }
    impl Render for Host {
        fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
            div()
                .size_full()
                .bg(PALETTE[self.tint % PALETTE.len()])
                .child(self.settled.clone())
        }
    }
    let mut cx = TestAppContext::single();
    cx.update(|cx| cx.set_view_retention(true));
    // Landed long before the test began, by the clock the scene uses.
    let started_at = Instant::now() - Duration::from_secs(60);
    let windows = [(); 2].map(|_| {
        cx.add_window(move |_, cx| Host {
            settled: cx.new(|_| Settled { started_at }),
            tint: 0,
        })
    });
    let frame = |cx: &mut TestAppContext, refresh: bool, window: WindowHandle<Host>| {
        cx.update_window(window.into(), |_, window, cx| {
            if refresh {
                window.refresh();
            }
            window.reset_frame_work_stats(false);
            window.draw(cx).clear(cx);
            let scene = &window.rendered_frame.scene;
            (
                scene
                    .quads
                    .iter()
                    .map(|quad| format!("{:?}", quad.bounds))
                    .collect::<Vec<_>>(),
                scene.transitions.len(),
                window.frame_work_stats().views_reused,
            )
        })
        .unwrap()
    };
    frame(&mut cx, false, windows[0]);
    frame(&mut cx, true, windows[1]);
    for window in windows {
        window
            .update(&mut cx, |host, _, cx| {
                host.tint += 1;
                cx.notify();
            })
            .unwrap();
    }
    let (retained_quads, retained_transitions, reused) = frame(&mut cx, false, windows[0]);
    let (quads, transitions, _) = frame(&mut cx, true, windows[1]);
    assert_eq!(reused, 1);
    assert_eq!(retained_quads, quads);
    assert_eq!(transitions, 1);
    assert_eq!(retained_transitions, 0);
}

mod probe_actions {
    use crate as gpui;
    gpui::actions!(retention_probe, [Probe]);
}

/// A view reading the window's appearance, or the bindings for an action,
/// is built again once they change.
#[test]
fn appearance_and_key_bindings_are_dependencies() {
    struct Reader {
        renders: Rc<Cell<usize>>,
    }
    impl Render for Reader {
        fn render(&mut self, window: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
            self.renders.set(self.renders.get() + 1);
            let dark = window.appearance() == crate::WindowAppearance::Dark;
            let bound = window.bindings_for_action(&probe_actions::Probe).len();
            div().child(SharedString::from(format!("{dark} {bound}")))
        }
    }
    struct Host {
        reader: Entity<Reader>,
    }
    impl Render for Host {
        fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
            div().size_full().child(self.reader.clone())
        }
    }
    let mut cx = TestAppContext::single();
    cx.update(|cx| cx.set_view_retention(true));
    let renders = Rc::new(Cell::new(0));
    let window = cx.add_window({
        let renders = renders.clone();
        move |_, cx| Host {
            reader: cx.new(|_| Reader { renders }),
        }
    });
    let frame = |cx: &mut TestAppContext| {
        cx.update_window(window.into(), |_, window, cx| {
            window.draw(cx).clear(cx);
        })
        .unwrap()
    };
    frame(&mut cx);
    frame(&mut cx);
    let settled = renders.get();
    frame(&mut cx);
    assert_eq!(renders.get(), settled, "nothing changed");
    cx.test_window(window.into())
        .simulate_appearance_change(crate::WindowAppearance::Dark);
    cx.run_until_parked();
    frame(&mut cx);
    assert!(renders.get() > settled, "the appearance changed");
    let settled = renders.get();
    cx.update(|cx| cx.bind_keys([crate::KeyBinding::new("ctrl-p", probe_actions::Probe, None)]));
    frame(&mut cx);
    assert!(renders.get() > settled, "the key bindings changed");
}

/// A view listing the available actions is built again when an action gets
/// its first global handler, which the window asks a frame for.
#[test]
fn a_new_global_action_handler_is_a_dependency() {
    struct Reader {
        seen: Rc<Cell<usize>>,
        // The actions are asked of the frame before, which the first frame
        // does not have.
        ready: bool,
    }
    impl Render for Reader {
        fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
            if self.ready {
                self.seen.set(window.available_actions(cx).len());
            }
            div().child("actions")
        }
    }
    struct Host {
        reader: Entity<Reader>,
    }
    impl Render for Host {
        fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
            div().size_full().child(self.reader.clone())
        }
    }
    let mut cx = TestAppContext::single();
    cx.update(|cx| cx.set_view_retention(true));
    let seen = Rc::new(Cell::new(usize::MAX));
    let window = cx.add_window({
        let seen = seen.clone();
        move |_, cx| Host {
            reader: cx.new(|_| Reader { seen, ready: false }),
        }
    });
    cx.update_window(window.into(), |_, window, cx| window.draw(cx).clear(cx))
        .unwrap();
    cx.run_until_parked();
    let reader = window.read_with(&cx, |host, _| host.reader.clone()).unwrap();
    reader.update(&mut cx, |reader, cx| {
        reader.ready = true;
        cx.notify();
    });
    cx.run_until_parked();
    let before = seen.get();
    assert_ne!(before, usize::MAX, "the view listed the actions");
    cx.update(|cx| {
        cx.on_action(|_: &probe_actions::Probe, _| {});
    });
    cx.run_until_parked();
    assert_eq!(seen.get(), before + 1, "the view listed the new action");
}

/// Shows whether the probe action is available, as a key binding hint does.
struct ActionReader {
    renders: Rc<Cell<usize>>,
}

impl Render for ActionReader {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        self.renders.set(self.renders.get() + 1);
        let available = actions_asked_of_last_frame(window)
            && window.is_action_available(&probe_actions::Probe, cx);
        // A size of its own, so that the follow-up frame building it draws
        // the host again around it rather than building the host too.
        div()
            .w(px(80.))
            .h(px(20.))
            .child(SharedString::from(format!("probe {available}")))
    }
}

/// A focused host that handles the probe action or not, with rows that come
/// and go without a focus handle, and maybe a view reading the actions.
struct ActionHost {
    focus_handle: crate::FocusHandle,
    handles: bool,
    rows: usize,
    reader: Option<Entity<ActionReader>>,
}

impl Render for ActionHost {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        div()
            .size_full()
            .track_focus(&self.focus_handle)
            .when(self.handles, |this| {
                this.on_action(|_: &probe_actions::Probe, _, _| {})
            })
            .children(self.reader.clone())
            .children((0..self.rows).map(|row| {
                div()
                    .key_context("Row")
                    .on_action(|_: &probe_actions::Probe, _, _| {})
                    .child(SharedString::from(format!("row {row}")))
            }))
    }
}

fn action_host_window(
    cx: &mut TestAppContext,
    reads: bool,
) -> (WindowHandle<ActionHost>, Rc<Cell<usize>>) {
    cx.update(|cx| cx.set_view_retention(true));
    let renders = Rc::new(Cell::new(0));
    let window = cx.add_window({
        let renders = renders.clone();
        move |window, cx| {
            let focus_handle = cx.focus_handle();
            window.focus(&focus_handle, cx);
            ActionHost {
            focus_handle,
            handles: false,
            rows: 0,
            reader: reads.then(|| cx.new(|_| ActionReader { renders })),
            }
        }
    });
    // The first frame has no frame before it to ask the actions of: the
    // reader is built again once there is one.
    frames_after_one(cx, window);
    let reader = window.read_with(cx, |host, _| host.reader.clone()).unwrap();
    if let Some(reader) = reader {
        reader.update(cx, |_, cx| cx.notify());
    }
    (window, renders)
}

/// The frames the window drew on its own after one drawn here.
fn frames_after_one<V: 'static>(cx: &mut TestAppContext, window: WindowHandle<V>) -> u64 {
    cx.update_window(window.into(), |_, window, cx| {
        window.draw(cx).clear(cx);
        window.reset_frame_work_stats(false);
    })
    .unwrap();
    cx.run_until_parked();
    cx.update_window(window.into(), |_, window, _| window.frame_work_stats().frames)
        .unwrap()
}

/// A frame that changes which actions are available asks for a follow-up
/// frame, in which the views that read them are built again: they were drawn
/// from the actions of the frame before.
#[test]
fn a_frame_changing_the_actions_asks_for_another() {
    let mut cx = TestAppContext::single();
    let (window, renders) = action_host_window(&mut cx, true);
    let frames_after = |cx: &mut TestAppContext| frames_after_one(cx, window);
    frames_after(&mut cx);
    assert_eq!(frames_after(&mut cx), 0, "nothing changed");
    let before = renders.get();
    window
        .update(&mut cx, |host, _, cx| {
            host.handles = true;
            cx.notify();
        })
        .unwrap();
    assert_eq!(frames_after(&mut cx), 1, "the actions changed");
    assert_eq!(renders.get(), before + 1, "the reader was built in the follow-up");
    assert_eq!(frames_after(&mut cx), 0, "the follow-up frame changed nothing");
}

/// No follow-up frame is asked for when no view drawn read the actions, or
/// when what changed is off every path a view could have asked about: nodes
/// without a focusable node below them.
#[test]
fn a_frame_changing_actions_nobody_read_asks_for_nothing() {
    let mut cx = TestAppContext::single();
    let (window, _) = action_host_window(&mut cx, false);
    let frames_after = |cx: &mut TestAppContext| frames_after_one(cx, window);
    frames_after(&mut cx);
    window
        .update(&mut cx, |host, _, cx| {
            host.handles = true;
            cx.notify();
        })
        .unwrap();
    assert_eq!(frames_after(&mut cx), 0, "no view read the actions");

    let mut cx = TestAppContext::single();
    let (window, renders) = action_host_window(&mut cx, true);
    let frames_after = |cx: &mut TestAppContext| frames_after_one(cx, window);
    frames_after(&mut cx);
    frames_after(&mut cx);
    let before = renders.get();
    for rows in [3, 1, 4] {
        window
            .update(&mut cx, |host, _, cx| {
                host.rows = rows;
                cx.notify();
            })
            .unwrap();
        assert_eq!(frames_after(&mut cx), 0, "rows without focus handles changed");
    }
    assert_eq!(renders.get(), before, "the reader was drawn again throughout");
}

/// A cached view inside a deferred draw, from a view inside one notified
/// since the last frame, counts as inside it: a model it read, updated
/// without being notified, builds it again as it would the rest.
#[test]
fn a_deferred_view_inside_a_notified_view_counts_as_inside_it() {
    struct Shown(usize);
    struct Content {
        model: Entity<Shown>,
    }
    impl Render for Content {
        fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
            let value = self.model.read(cx).0;
            div().w(px(10. + value as f32)).h(px(10.)).bg(PALETTE[2])
        }
    }
    struct Popover {
        content: Entity<Content>,
    }
    impl Render for Popover {
        fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
            div().child(deferred(
                self.content
                    .clone()
                    .cached(StyleRefinement::default().w(px(60.)).h(px(10.))),
            ))
        }
    }
    struct Host {
        popover: Entity<Popover>,
    }
    impl Render for Host {
        fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
            div().size_full().child(self.popover.clone())
        }
    }
    let mut cx = TestAppContext::single();
    cx.update(|cx| cx.set_view_retention(true));
    let model = cx.new(|_| Shown(0));
    let window = cx.add_window({
        let model = model.clone();
        move |_, cx| Host {
            popover: cx.new(|cx| Popover {
                content: cx.new(|_| Content { model }),
            }),
        }
    });
    let frame = |cx: &mut TestAppContext| {
        cx.update_window(window.into(), |_, window, cx| {
            window.draw(cx).clear(cx);
            window
                .rendered_frame
                .scene
                .quads
                .iter()
                .map(|quad| quad.bounds.size.width)
                .collect::<Vec<_>>()
        })
        .unwrap()
    };
    frame(&mut cx);
    // The popover is built, so that the view it defers is prepainted and
    // leaves a record: drawn again whole, it would copy its deferred draw
    // without one.
    let popover = window.read_with(&cx, |host, _| host.popover.clone()).unwrap();
    popover.update(&mut cx, |_, cx| cx.notify());
    cx.run_until_parked();
    let widths = |cx: &mut TestAppContext| {
        cx.update_window(window.into(), |_, window, _| {
            window
                .rendered_frame
                .scene
                .quads
                .iter()
                .map(|quad| quad.bounds.size.width)
                .collect::<Vec<_>>()
        })
        .unwrap()
    };
    let before = widths(&mut cx);
    model.update(&mut cx, |shown, _| shown.0 = 20);
    window.update(&mut cx, |_, _, cx| cx.notify()).unwrap();
    cx.run_until_parked();
    let after = widths(&mut cx);
    assert_ne!(after, before, "the deferred view shows the model as it is");
}

/// A panel whose entries render from it through a handle, as a transcript's
/// rows render from the panel that holds them.
struct HostPanel {
    values: Vec<usize>,
    ticks: usize,
    entries: Vec<Entity<PanelEntry>>,
    tracked: Entity<PanelEntry>,
}

struct PanelEntry {
    ix: usize,
    panel: crate::WeakEntity<HostPanel>,
    badge: Option<Entity<PanelBadge>>,
    renders: Rc<Cell<usize>>,
}

/// A view nested in an entry, which reads the panel as well.
struct PanelBadge {
    panel: crate::WeakEntity<HostPanel>,
}

impl Render for PanelBadge {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let ticks = self
            .panel
            .upgrade()
            .map_or(0, |panel| panel.read(cx).entries.len());
        div().child(SharedString::from(format!("of {ticks}")))
    }
}

impl Render for PanelEntry {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        self.renders.set(self.renders.get() + 1);
        let Some(panel) = self.panel.upgrade() else {
            return div();
        };
        let value = panel.read(cx).values[self.ix];
        div()
            .child(SharedString::from(format!("entry {} {value}", self.ix)))
            .children(self.badge.clone())
            // Deferred, it is drawn after the entry, and reads the panel too.
            .child(deferred(
                crate::canvas(
                    |_, _, _| {},
                    move |bounds, _, window, cx| {
                        let ticks = panel.read(cx).ticks;
                        if ticks > usize::MAX / 2 {
                            window.paint_quad(crate::fill(bounds, PALETTE[1]));
                        }
                    },
                )
                .w(px(4.))
                .h(px(4.)),
            ))
    }
}

impl Render for HostPanel {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        div()
            .size_full()
            .child(SharedString::from(format!("ticks {}", self.ticks)))
            .children(self.entries.iter().cloned())
            .child(self.tracked.clone())
    }
}

/// An entry that untracks the panel it renders from is drawn again from the
/// last frame while the panel changes on every frame, with the view nested
/// in it and what it deferred, and is built when it is notified; an entry
/// that does not is built on every frame.
#[test]
fn a_view_untracking_its_host_is_drawn_again_while_the_host_changes() {
    let mut cx = TestAppContext::single();
    cx.update(|cx| cx.set_view_retention(true));
    let renders = Rc::new(Cell::new(0));
    let tracked_renders = Rc::new(Cell::new(0));
    let window = cx.add_window({
        let renders = renders.clone();
        let tracked_renders = tracked_renders.clone();
        move |_, cx| {
            let panel = cx.weak_entity();
            let entries = (0..3)
                .map(|ix| {
                    let panel = panel.clone();
                    let renders = renders.clone();
                    cx.new(|cx| {
                        cx.untrack_reads_of(&panel.upgrade().unwrap());
                        PanelEntry {
                            ix,
                            badge: Some(cx.new(|_| PanelBadge {
                                panel: panel.clone(),
                            })),
                            panel,
                            renders,
                        }
                    })
                })
                .collect();
            HostPanel {
                values: vec![0, 1, 2],
                ticks: 0,
                entries,
                tracked: cx.new(|_| PanelEntry {
                    ix: 0,
                    panel,
                    badge: None,
                    renders: tracked_renders,
                }),
            }
        }
    });
    draw_any(&mut cx, window);
    draw_any(&mut cx, window);
    let tick = |cx: &mut TestAppContext| {
        work_after(cx, window, |cx| {
            window
                .update(cx, |panel, _, cx| {
                    panel.ticks += 1;
                    cx.notify();
                })
                .unwrap();
        })
    };
    let (entries_before, tracked_before) = (renders.get(), tracked_renders.get());
    for _ in 0..3 {
        let work = tick(&mut cx);
        // The three entries and the badge in each, drawn again whole.
        assert!(work.views_reused >= 3, "{work:?}");
    }
    assert_eq!(renders.get(), entries_before, "the entries were drawn again");
    assert_eq!(tracked_renders.get(), tracked_before + 3, "the tracked entry was built");

    // What entry 1 shows changed, and it is notified, as the contract asks.
    work_after(&mut cx, window, |cx| {
        let entry = window
            .update(cx, |panel, _, cx| {
                panel.values[1] = 41;
                cx.notify();
                panel.entries[1].clone()
            })
            .unwrap();
        entry.update(cx, |_, cx| cx.notify());
    });
    assert_eq!(renders.get(), entries_before + 1, "only entry 1 was built");
}

fn draw_any<V: 'static>(cx: &mut TestAppContext, window: WindowHandle<V>) -> crate::FrameWorkStats {
    cx.update_window(window.into(), |_, window, cx| {
        window.reset_frame_work_stats(false);
        window.draw(cx).clear(cx);
        window.frame_work_stats()
    })
    .unwrap()
}

/// A row of a strip scrolled by an offset: a scrolled-sideways bar, two
/// lines of text, and a listener answering where in the row a press landed.
struct StripRow {
    scroll: crate::ScrollHandle,
    presses: Rc<Cell<Option<crate::Point<Pixels>>>>,
    /// Writes this model as it prepaints, which keeps it in place.
    writes: Option<Entity<Summary>>,
}

impl Render for StripRow {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        let presses = self.presses.clone();
        let writes = self.writes.clone();
        div()
            .relative()
            .flex()
            .flex_col()
            .w(px(180.))
            .h(px(80.))
            .child(
                div()
                    .id("bar")
                    .overflow_x_scroll()
                    .track_scroll(&self.scroll)
                    .w(px(100.))
                    .h(px(4.))
                    .child(div().w(px(300.)).h(px(4.)).bg(PALETTE[1])),
            )
            // The test text system's glyphs hang 19.5 below their lines' tops
            // and are 13 high: the first line's are 23.5 to 36.5 into the
            // row, the second's 57.5 to 70.5.
            .child(div().h(px(14.)).child("line one"))
            .child(div().h(px(20.)))
            .child(div().h(px(14.)).child("line two"))
            .child(
                crate::canvas(
                    move |bounds, _, cx| {
                        if let Some(writes) = &writes {
                            writes.update(cx, |_, _| {});
                        }
                        bounds
                    },
                    move |_, bounds, window, _| {
                        window.on_mouse_event(
                            move |event: &crate::MouseDownEvent, phase, _, _| {
                                if phase == crate::DispatchPhase::Bubble
                                    && bounds.contains(&event.position)
                                {
                                    presses.set(Some(event.position - bounds.origin));
                                }
                            },
                        );
                    },
                )
                .absolute()
                .top_0()
                .left_0()
                .size_full(),
            )
    }
}

/// Rows in a clipped strip, moved up by `offset`.
struct Strip {
    offset: f32,
    rows: Vec<Entity<StripRow>>,
}

impl Render for Strip {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        div().size_full().child(
            div()
                .absolute()
                .top(px(20.))
                .left(px(10.))
                .w(px(200.))
                .h(px(285.))
                .overflow_hidden()
                .child(
                    div()
                        .mt(px(-self.offset))
                        .flex()
                        .flex_col()
                        .children(self.rows.iter().cloned()),
                ),
        )
    }
}

struct StripRows {
    presses: Rc<Cell<Option<crate::Point<Pixels>>>>,
    scrolls: Vec<crate::ScrollHandle>,
}

/// The same strip in a window drawing views again and one drawn from scratch:
/// the first row writes a model as it prepaints, the second opted out of
/// being drawn moved, and the fourth is half out of view, its second line
/// never drawn.
fn strip_windows(cx: &mut TestAppContext) -> ([WindowHandle<Strip>; 2], [StripRows; 2]) {
    cx.update(|cx| cx.set_view_retention(true));
    let mut rows = Vec::new();
    let windows = [(); 2].map(|_| {
        // One per window: a model both wrote would build each window's
        // readers whenever the other draws.
        let written = cx.new(|_| Summary(0));
        let presses = Rc::new(Cell::new(None));
        let scrolls: Vec<_> = (0..4).map(|_| crate::ScrollHandle::new()).collect();
        rows.push(StripRows {
            presses: presses.clone(),
            scrolls: scrolls.clone(),
        });
        cx.add_window(move |_, cx| Strip {
            offset: 0.,
            rows: scrolls
                .into_iter()
                .enumerate()
                .map(|(ix, scroll)| {
                    let presses = presses.clone();
                    let written = written.clone();
                    cx.new(move |cx| {
                        if ix == 1 {
                            cx.set_view_movable(false);
                        }
                        StripRow {
                            scroll,
                            presses,
                            writes: (ix == 0).then_some(written),
                        }
                    })
                })
                .collect(),
        })
    });
    let rows: [StripRows; 2] = rows.try_into().ok().unwrap();
    (windows, rows)
}

/// Draws both strips, the second from scratch, returning their descriptions
/// and the first's work since the last call. The first is drawn only if a
/// change did not draw it already, so that its rebuilds are the change's.
fn draw_strips(
    cx: &mut TestAppContext,
    windows: [WindowHandle<Strip>; 2],
) -> (Vec<String>, Vec<String>, crate::FrameWorkStats) {
    let (retained, work) = cx
        .update_window(windows[0].into(), |_, window, cx| {
            if window.frame_work_stats().frames == 0 {
                window.draw(cx).clear(cx);
            }
            let work = window.frame_work_stats();
            window.reset_frame_work_stats(false);
            (describe_frame(window), work)
        })
        .unwrap();
    let from_scratch = cx
        .update_window(windows[1].into(), |_, window, cx| {
            window.refresh();
            window.draw(cx).clear(cx);
            describe_frame(window)
        })
        .unwrap();
    (retained, from_scratch, work)
}

fn scroll_strips(cx: &mut TestAppContext, windows: [WindowHandle<Strip>; 2], offset: f32) {
    for window in windows {
        window
            .update(cx, |strip, _, cx| {
                strip.offset = offset;
                cx.notify();
            })
            .unwrap();
    }
}

/// Rows moved by a scroll are drawn again moved, not built, unless they
/// write as they draw, opted out, were or are partly out of view (the last
/// row's second line was never drawn) or moved by part of a device pixel.
#[test]
fn moved_views_are_drawn_again_unless_something_of_them_was_out_of_view() {
    let mut cx = super::super::layout_retention_tests::text_system_context(0);
    let (windows, _) = strip_windows(&mut cx);
    draw_strips(&mut cx, windows);
    let (retained, from_scratch, _) = draw_strips(&mut cx, windows);
    assert_eq!(first_difference(&retained, &from_scratch), None);
    let rows = windows[0].read_with(&cx, |strip, _| strip.rows.clone()).unwrap();
    for offset in [-10., 0., 50., 55.5, 60.25] {
        scroll_strips(&mut cx, windows, offset);
        let (retained, from_scratch, work) = draw_strips(&mut cx, windows);
        assert_eq!(
            first_difference(&retained, &from_scratch),
            None,
            "scrolled to {offset}"
        );
        let reasons: Vec<_> = cx
            .update_window(windows[0].into(), |_, window, _| window.view_rebuild_reasons().to_vec())
            .unwrap();
        let rebuilt: Vec<_> = reasons.iter().map(|(view, _)| *view).collect();
        for (ix, row) in rows.iter().enumerate() {
            let built = rebuilt.contains(&row.entity_id());
            // The strip is 285 high, the rows 80: unscrolled, its bottom cuts
            // between the last row's lines, whose second is never drawn.
            // Scrolled by 50, the last row comes wholly into view. A
            // quarter-pixel scroll still moves the rows by whole device
            // pixels: elements are placed snapped to them.
            let expected = match (ix, offset) {
                (0 | 1, _) => true,
                (3, -10. | 0. | 50.) => true,
                _ => false,
            };
            assert_eq!(built, expected, "row {ix} at offset {offset}: {work:?}");
        }
        assert!(work.views_moved >= 1, "{work:?}");
    }
}

/// A press on a row drawn moved lands where it does in the row: the window
/// builds the rows drawn moved before dispatching it.
#[test]
fn a_press_on_a_moved_view_settles_it_first() {
    let mut cx = super::super::layout_retention_tests::text_system_context(0);
    let (windows, rows) = strip_windows(&mut cx);
    draw_strips(&mut cx, windows);
    draw_strips(&mut cx, windows);
    scroll_strips(&mut cx, windows, 30.);
    let (_, _, work) = draw_strips(&mut cx, windows);
    assert!(work.views_moved >= 1, "{work:?}");
    // In the third row, scrolled: the strip at (10, 20), the row 160 down
    // it, moved up 30.
    let position = point(px(10. + 7.), px(20. + 160. - 30. + 15.));
    cx.update_window(windows[0].into(), |_, window, cx| {
        assert!(window.rendered_frame.retained_views.unsettled);
        window.dispatch_event(
            crate::PlatformInput::MouseDown(crate::MouseDownEvent {
                button: crate::MouseButton::Left,
                position,
                modifiers: Modifiers::default(),
                click_count: 1,
                first_mouse: false,
            }),
            cx,
        );
        assert!(!window.rendered_frame.retained_views.unsettled);
    })
    .unwrap();
    assert_eq!(rows[0].presses.get(), Some(point(px(7.), px(15.))));
}

/// What a moved view's elements wrote of where they are moves with it, and
/// the views drawn moved are built once they stop moving.
#[test]
fn a_moved_view_moves_its_scroll_handles_and_settles_once_still() {
    let mut cx = super::super::layout_retention_tests::text_system_context(0);
    let (windows, rows) = strip_windows(&mut cx);
    draw_strips(&mut cx, windows);
    draw_strips(&mut cx, windows);
    for offset in [10., 30., 15.] {
        scroll_strips(&mut cx, windows, offset);
        let (_, _, work) = draw_strips(&mut cx, windows);
        assert!(work.views_moved >= 1, "{work:?}");
        assert_eq!(
            rows[0].scrolls[2].bounds(),
            rows[1].scrolls[2].bounds(),
            "scrolled to {offset}"
        );
    }
    let unsettled = |cx: &mut TestAppContext| {
        cx.update_window(windows[0].into(), |_, window, _| {
            window.rendered_frame.retained_views.unsettled
        })
        .unwrap()
    };
    assert!(unsettled(&mut cx));
    cx.executor().advance_clock(Duration::from_millis(300));
    cx.run_until_parked();
    assert!(!unsettled(&mut cx));
    let (retained, from_scratch, _) = draw_strips(&mut cx, windows);
    assert_eq!(first_difference(&retained, &from_scratch), None);
}

/// A view keeping its window position outside the frame, as it prepaints
/// and as it paints, moved by the window when it is drawn again elsewhere.
struct Marker {
    prepainted_at: Rc<Cell<Option<crate::Point<Pixels>>>>,
    painted_at: Rc<Cell<Option<crate::Point<Pixels>>>>,
}

impl Render for Marker {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        let prepainted_at = self.prepainted_at.clone();
        let painted_at = self.painted_at.clone();
        let keep = |at: Rc<Cell<Option<crate::Point<Pixels>>>>, window: &mut Window| {
            window.on_replayed_at_offset(move |by| at.set(at.get().map(|at| at + by)));
        };
        div().w(px(100.)).h(px(20.)).child(
            crate::canvas(
                move |bounds, window, _| {
                    prepainted_at.set(Some(bounds.origin));
                    keep(prepainted_at, window);
                },
                move |bounds, _, window, _| {
                    painted_at.set(Some(bounds.origin));
                    keep(painted_at, window);
                },
            )
            .size_full(),
        )
    }
}

struct MarkerHost {
    header: f32,
    marker: Entity<Marker>,
}

impl Render for MarkerHost {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        div()
            .size_full()
            .flex()
            .flex_col()
            .child(div().w(px(10.)).h(px(self.header)))
            .child(self.marker.clone())
    }
}

/// What a view registered with `on_replayed_at_offset`, as it prepainted
/// and as it painted, is moved with it while it is drawn again elsewhere.
#[test]
fn positions_kept_outside_the_frame_move_with_a_view_drawn_moved() {
    let mut cx = TestAppContext::single();
    cx.update(|cx| cx.set_view_retention(true));
    let prepainted_at = Rc::new(Cell::new(None));
    let painted_at = Rc::new(Cell::new(None));
    let window = cx.add_window({
        let prepainted_at = prepainted_at.clone();
        let painted_at = painted_at.clone();
        move |_, cx| MarkerHost {
            header: 10.,
            marker: cx.new(|_| Marker {
                prepainted_at,
                painted_at,
            }),
        }
    });
    // Drawing on the update's flush, if it drew nothing.
    let frame = |cx: &mut TestAppContext| {
        cx.update_window(window.into(), |_, window, cx| {
            if window.frame_work_stats().frames == 0 {
                window.draw(cx).clear(cx);
            }
            window.frame_work_stats()
        })
        .unwrap()
    };
    frame(&mut cx);
    let mut moved = 0;
    for header in [30., 25., 60.] {
        cx.update_window(window.into(), |_, window, _| window.reset_frame_work_stats(false))
            .unwrap();
        window
            .update(&mut cx, |host, _, cx| {
                host.header = header;
                cx.notify();
            })
            .unwrap();
        moved += frame(&mut cx).views_moved;
        let at = Some(point(px(0.), px(header)));
        assert_eq!(prepainted_at.get(), at, "prepainted position at {header}");
        assert_eq!(painted_at.get(), at, "painted position at {header}");
    }
    assert!(moved >= 3, "the marker was drawn moved {moved} times");
}

struct Popover;

impl Render for Popover {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        div().w(px(60.)).h(px(30.)).bg(PALETTE[2]).child("popover")
    }
}

struct PopoverHost {
    popover: Entity<Popover>,
    ticks: usize,
}

impl Render for PopoverHost {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        div()
            .size_full()
            .child(SharedString::from(format!("ticked {}", self.ticks)))
            .child(crate::deferred(self.popover.clone()))
    }
}

/// A view drawn inside something deferred keeps its record while what
/// deferred it is drawn again from the last frame, so that once that is
/// built again, the view is drawn again rather than built afresh.
#[test]
fn views_drawn_inside_a_deferred_draw_keep_their_records() {
    let mut cx = super::super::layout_retention_tests::text_system_context(0);
    cx.update(|cx| cx.set_view_retention(true));
    let windows = [(); 2].map(|_| {
        cx.add_window(|_, cx| PopoverHost {
            popover: cx.new(|_| Popover),
            ticks: 0,
        })
    });
    // Draws both windows, the second from scratch, unless the first drew on
    // the update's flush since its counts were reset.
    let draw = |cx: &mut TestAppContext| {
        windows.map(|window| {
            cx.update_window(window.into(), |_, window, cx| {
                if window.handle.window_id() == windows[1].window_id() {
                    window.refresh();
                    window.draw(cx).clear(cx);
                } else if window.frame_work_stats().frames == 0 {
                    window.draw(cx).clear(cx);
                }
                let frame = (describe_frame(window), window.frame_work_stats());
                window.reset_frame_work_stats(false);
                frame
            })
            .unwrap()
        })
    };
    draw(&mut cx);
    // Nothing changed: the host is drawn again, its deferred draw with it.
    let [(retained, work), (from_scratch, _)] = draw(&mut cx);
    assert_eq!(first_difference(&retained, &from_scratch), None);
    assert_eq!(work.views_rendered, 0, "{work:?}");
    for _ in 0..2 {
        for window in windows {
            window
                .update(&mut cx, |host, _, cx| {
                    host.ticks += 1;
                    cx.notify();
                })
                .unwrap();
        }
        let [(retained, work), (from_scratch, _)] = draw(&mut cx);
        assert_eq!(first_difference(&retained, &from_scratch), None);
        assert_eq!(work.views_rendered, 1, "only the host is built: {work:?}");
        assert_eq!(work.view_rebuilds.first_draw, 0, "{work:?}");
        let [(retained, work), (from_scratch, _)] = draw(&mut cx);
        assert_eq!(first_difference(&retained, &from_scratch), None);
        assert_eq!(work.views_rendered, 0, "{work:?}");
    }
}

/// A view showing an admitted line of text, keeping the line's geometry
/// handle as a caller resolving positions against it would. The line is
/// shaped by hand: the test text system has no admitted shaping.
struct AdmittedRow {
    layout: Rc<std::cell::RefCell<Option<crate::AdmittedTextLayout>>>,
}

impl Render for AdmittedRow {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        let text = crate::text_allocation::tests::hand_shaped_admitted_text();
        *self.layout.borrow_mut() = Some(text.layout().clone());
        div().w(px(120.)).h(px(20.)).child(text)
    }
}

struct AdmittedHost {
    header: f32,
    row: Entity<AdmittedRow>,
}

impl Render for AdmittedHost {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        div()
            .size_full()
            .flex()
            .flex_col()
            .child(div().w(px(10.)).h(px(self.header)))
            .child(self.row.clone())
    }
}

/// An admitted text layout drawn again moved answers positions where it is
/// now, as a text layout does.
#[test]
fn an_admitted_text_layout_moves_with_a_view_drawn_moved() {
    let mut cx = super::super::layout_retention_tests::text_system_context(0);
    cx.update(|cx| cx.set_view_retention(true));
    let layout = Rc::new(std::cell::RefCell::new(None));
    let window = cx.add_window({
        let layout = layout.clone();
        move |_, cx| AdmittedHost {
            header: 10.,
            row: cx.new(|_| AdmittedRow { layout }),
        }
    });
    let frame = |cx: &mut TestAppContext| {
        cx.update_window(window.into(), |_, window, cx| {
            if window.frame_work_stats().frames == 0 {
                window.draw(cx).clear(cx);
            }
            window.frame_work_stats()
        })
        .unwrap()
    };
    frame(&mut cx);
    let mut moved = 0;
    for header in [30., 25., 60.] {
        cx.update_window(window.into(), |_, window, _| window.reset_frame_work_stats(false))
            .unwrap();
        window
            .update(&mut cx, |host, _, cx| {
                host.header = header;
                cx.notify();
            })
            .unwrap();
        moved += frame(&mut cx).views_moved;
        let layout = layout.borrow().clone().expect("the row was drawn");
        let origin = point(px(0.), px(header));
        assert_eq!(layout.bounds().map(|bounds| bounds.origin), Some(origin));
        assert_eq!(layout.position_for_index(1), Some(origin + point(px(10.), px(0.))));
        assert_eq!(
            layout.closest_index_for_position(origin + point(px(19.), px(5.))),
            Some(2)
        );
    }
    assert!(moved >= 3, "the row was drawn moved {moved} times");
}

/// With rebuild culprits on, a moved view that was built rather than drawn
/// again moved is counted under what kept it in place.
#[test]
fn rebuilds_of_moved_views_name_what_kept_them_from_moving() {
    super::culprits::force_on();
    let mut cx = super::super::layout_retention_tests::text_system_context(0);
    let (windows, _rows) = strip_windows(&mut cx);
    draw_strips(&mut cx, windows);
    draw_strips(&mut cx, windows);
    for offset in [10., 30.] {
        scroll_strips(&mut cx, windows, offset);
        draw_strips(&mut cx, windows);
    }
    let counts = super::culprits::counts();
    let counted = |what: &str| counts.iter().any(|(line, _)| line.contains(what));
    assert!(
        counted("StripRow ContextChanged <- not drawn moved: StaysPut: wrote entity gpui::window::view_retention::tests::Summary while prepainting"),
        "{counts:#?}"
    );
    assert!(
        counted("StripRow ContextChanged <- not drawn moved: StaysPut: set_view_movable(false)"),
        "{counts:#?}"
    );
    assert!(counted("StripRow ContextChanged <- not drawn moved: Outside"), "{counts:#?}");
}

/// A panel counting how often its rows were built, which its rows write as
/// they prepaint, and a reader showing the count.
struct BookkeepingPanel {
    header: f32,
    builds: usize,
    reader: Option<Entity<BuildsReader>>,
    row: Option<Entity<BookkeepingRow>>,
}

impl Render for BookkeepingPanel {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        div()
            .size_full()
            .flex()
            .flex_col()
            .children(self.reader.clone())
            .child(div().w(px(10.)).h(px(self.header)))
            .children(self.row.clone())
    }
}

struct BuildsReader {
    panel: crate::WeakEntity<BookkeepingPanel>,
    renders: Rc<Cell<usize>>,
}

impl Render for BuildsReader {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        self.renders.set(self.renders.get() + 1);
        let builds = self.panel.upgrade().map_or(0, |panel| panel.read(cx).builds);
        div()
            .w(px(100.))
            .h(px(20.))
            .child(SharedString::from(format!("built {builds}")))
    }
}

/// A row that untracks its panel and writes it as it prepaints, as rows
/// built through their panel's context do.
struct BookkeepingRow {
    panel: crate::WeakEntity<BookkeepingPanel>,
}

impl Render for BookkeepingRow {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        if let Some(panel) = self.panel.upgrade() {
            cx.untrack_reads_of(&panel);
        }
        let panel = self.panel.clone();
        div().w(px(100.)).h(px(20.)).bg(PALETTE[0]).child(
            crate::canvas(
                move |_, _, cx| {
                    panel
                        .update(cx, |panel, _| panel.builds += 1)
                        .expect("the panel outlives its row");
                },
                |_, _, _, _| {},
            )
            .size_full(),
        )
    }
}

/// A row writing an entity it untracks as it prepaints is still drawn
/// moved; the write still builds the views that read the entity before it.
#[test]
fn writes_to_an_untracked_entity_do_not_keep_a_view_in_place() {
    let mut cx = TestAppContext::single();
    cx.update(|cx| cx.set_view_retention(true));
    let renders = Rc::new(Cell::new(0));
    let window = cx.add_window({
        let renders = renders.clone();
        move |_, cx| {
            let panel = cx.entity().downgrade();
            BookkeepingPanel {
                header: 10.,
                builds: 0,
                reader: Some(cx.new(|_| BuildsReader {
                    panel: panel.clone(),
                    renders,
                })),
                row: Some(cx.new(|_| BookkeepingRow { panel })),
            }
        }
    });
    let frame = |cx: &mut TestAppContext| {
        cx.update_window(window.into(), |_, window, cx| {
            if window.frame_work_stats().frames == 0 {
                window.draw(cx).clear(cx);
            }
            let work = window.frame_work_stats();
            window.reset_frame_work_stats(false);
            work
        })
        .unwrap()
    };
    frame(&mut cx);
    // The reader read the count before the row wrote it, and is built again.
    frame(&mut cx);
    assert_eq!(renders.get(), 2, "the reader was built again after the write");
    let mut moved = 0;
    for header in [30., 25., 60.] {
        window
            .update(&mut cx, |panel, _, cx| {
                panel.header = header;
                cx.notify();
            })
            .unwrap();
        moved += frame(&mut cx).views_moved;
    }
    assert!(moved >= 3, "the row was drawn moved {moved} times");
}

/// A view sized by state it does not tell the window about, for a frame to
/// find it asking for another layout than the one it kept.
struct UntoldWidth {
    width: Rc<Cell<f32>>,
}

impl Render for UntoldWidth {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        // Built wherever it moves, at the layout it kept.
        cx.set_view_movable(false);
        div()
            .w(px(self.width.get()))
            .h(px(20.))
            .bg(PALETTE[0])
            .child(div().w(px(10.)).h(px(10.)).bg(PALETTE[1]))
    }
}

struct UntoldHost {
    header: f32,
    child: Entity<UntoldWidth>,
}

impl Render for UntoldHost {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        div()
            .size_full()
            .flex()
            .flex_col()
            .child(div().w(px(10.)).h(px(self.header)))
            .child(div().pl(px(15.)).child(self.child.clone()))
    }
}

/// A view built at the layout it kept that asks for another is laid out on
/// its own at the bounds it was given, and drawn there once, not offset by
/// where its kept nodes sit in the tree around it as well.
#[test]
fn a_view_laid_out_on_its_own_at_its_bounds_is_drawn_there_once() {
    let mut cx = TestAppContext::single();
    cx.update(|cx| cx.set_view_retention(true));
    let width = Rc::new(Cell::new(40.));
    let windows = [(); 2].map(|_| {
        let width = width.clone();
        cx.add_window(move |_, cx| UntoldHost {
            header: 30.,
            child: cx.new(|_| UntoldWidth { width }),
        })
    });
    // Draws both windows, the second from scratch, unless the first drew on
    // the update's flush since it last drew here.
    let draw = |cx: &mut TestAppContext| {
        windows.map(|window| {
            cx.update_window(window.into(), |_, window, cx| {
                if window.handle.window_id() == windows[1].window_id() {
                    window.refresh();
                    window.draw(cx).clear(cx);
                } else if window.frame_work_stats().frames == 0 {
                    window.draw(cx).clear(cx);
                }
                window.reset_frame_work_stats(false);
                describe_frame(window)
            })
            .unwrap()
        })
    };
    draw(&mut cx);
    draw(&mut cx);
    // The child changes size without telling anyone, and its host moves
    // it, so it is built at the layout it kept, which it no longer asks for.
    width.set(60.);
    for window in windows {
        window
            .update(&mut cx, |host, _, cx| {
                host.header = 50.;
                cx.notify();
            })
            .unwrap();
    }
    // This frame it is laid out at the bounds it had, and its kept nodes
    // laid out as a root of their own: drawn at its origin, once.
    let [retained, _] = draw(&mut cx);
    let at = |frame: &[String], x: f32, y: f32| {
        let origin = format!("origin: Point {{ x: {x}px (scaled), y: {y}px (scaled) }}");
        frame
            .iter()
            .any(|line| line.starts_with("Quad") && line.contains(&origin))
    };
    assert!(at(&retained, 30., 100.), "{retained:#?}");
    assert!(!at(&retained, 60., 200.), "{retained:#?}");
    // The next frame builds what is around it at the layout it asks for.
    let [retained, from_scratch] = draw(&mut cx);
    assert_eq!(first_difference(&retained, &from_scratch), None);
}

/// Lines read from a model, as a reply's text is.
struct StretchLines {
    lines: Entity<usize>,
}

impl Render for StretchLines {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let lines = *self.lines.read(cx);
        div()
            .flex()
            .flex_col()
            .children((0..lines).map(|line| {
                div()
                    .h(px(20.))
                    .child(SharedString::from(format!("line {line}")))
            }))
    }
}

/// A panel whose height its row stretches, holding the lines.
struct StretchedPanel {
    lines: Entity<StretchLines>,
}

impl Render for StretchedPanel {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        div()
            .flex()
            .flex_col()
            .w(px(100.))
            .bg(PALETTE[1])
            .child(self.lines.clone())
    }
}

struct StretchHost {
    header: f32,
    panel: Entity<StretchedPanel>,
}

impl Render for StretchHost {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        div()
            .size_full()
            .flex()
            .flex_col()
            .child(div().w(px(10.)).h(px(self.header)))
            .child(
                div()
                    .flex()
                    .flex_row()
                    .child(div().w(px(40.)).h(px(162.)).bg(PALETTE[2]))
                    .child(self.panel.clone()),
            )
    }
}

/// A view whose layout changes in the frame it moves is built at the
/// layout it kept, asks for another, and still fills the height its row
/// stretches it to: it is not laid out as if nothing were around it.
#[test]
fn a_view_built_at_a_changed_layout_keeps_the_size_its_parent_gives_it() {
    let mut cx = super::super::layout_retention_tests::text_system_context(0);
    cx.update(|cx| cx.set_view_retention(true));
    let windows = [(); 2].map(|_| {
        cx.add_window(|_, cx| {
            let model = cx.new(|_| 2usize);
            let lines = cx.new(|_| StretchLines { lines: model });
            StretchHost {
                header: 10.,
                panel: cx.new(|_| StretchedPanel { lines }),
            }
        })
    });
    // Draws both windows, the second from scratch, unless the first drew on
    // the update's flush since it last drew here.
    let draw = |cx: &mut TestAppContext| {
        windows.map(|window| {
            cx.update_window(window.into(), |_, window, cx| {
                if window.handle.window_id() == windows[1].window_id() {
                    window.refresh();
                    window.draw(cx).clear(cx);
                } else if window.frame_work_stats().frames == 0 {
                    window.draw(cx).clear(cx);
                }
                window.reset_frame_work_stats(false);
                describe_frame(window)
            })
            .unwrap()
        })
    };
    draw(&mut cx);
    draw(&mut cx);
    for step in 0..3 {
        for window in windows {
            let model = window
                .read_with(&cx, |host, cx| {
                    let panel = host.panel.read(cx);
                    panel.lines.read(cx).lines.clone()
                })
                .unwrap();
            // In one update, so that one frame draws both changes.
            window
                .update(&mut cx, |host, _, cx| {
                    model.update(cx, |lines, cx| {
                        *lines += 1;
                        cx.notify();
                    });
                    host.header += 10.;
                    cx.notify();
                })
                .unwrap();
        }
        let [retained, from_scratch] = draw(&mut cx);
        assert_eq!(
            first_difference(&retained, &from_scratch),
            None,
            "step {step}"
        );
    }
}

/// Seeds of the random histories that once drew a frame unlike the one drawn
/// from scratch: a view drawn moved under the pointer with the hover it had
/// where it was, a view drawn again or around whose content asked to be
/// scrolled into view (several rows of a list revealing themselves, which a
/// list answers one a frame), and a layer of text drawn moved past the mask
/// around it.
#[test]
fn histories_that_once_differed_from_scratch_match() {
    for seed in [
        594, 606, 644, 674, 1338, 1708, 1851, 1900, 2009, 2598, 2965, 3187, 3337, 3441, 3452,
        3802, 3822, 4043,
    ] {
        run(seed, 50);
    }
}

/// Two windows of the same root described together: the first as it drew
/// since it was last described (drawing now only if it did not), the second
/// drawn now from scratch.
fn draw_both<V: 'static>(
    cx: &mut TestAppContext,
    windows: [WindowHandle<V>; 2],
) -> [Vec<String>; 2] {
    windows.map(|window| {
        cx.update_window(window.into(), |_, window, cx| {
            if window.handle.window_id() == windows[1].window_id() {
                window.refresh();
                window.draw(cx).clear(cx);
            } else if window.frame_work_stats().frames == 0 {
                window.draw(cx).clear(cx);
            }
            window.reset_frame_work_stats(false);
            let mut lines = describe_frame(window);
            // Layers order what is drawn over them, and are compared too.
            let mut layers: Vec<String> = window
                .rendered_frame
                .scene
                .paint_operations
                .iter()
                .filter_map(|operation| match operation {
                    crate::scene::PaintOperation::StartLayer(layer) => Some(format!("{layer:?}")),
                    _ => None,
                })
                .collect();
            layers.sort();
            lines.extend(layers);
            lines
        })
        .unwrap()
    })
}

fn move_mouse_in<V: 'static>(cx: &mut TestAppContext, windows: [WindowHandle<V>; 2], x: f32, y: f32) {
    for window in windows {
        cx.update_window(window.into(), |_, window, cx| {
            window.dispatch_event(
                crate::PlatformInput::MouseMove(crate::MouseMoveEvent {
                    position: point(px(x), px(y)),
                    pressed_button: None,
                    modifiers: Modifiers::default(),
                }),
                cx,
            );
        })
        .unwrap();
    }
}

struct HoverBox;

impl Render for HoverBox {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        div()
            .id("hover-box")
            .w(px(100.))
            .h(px(40.))
            .bg(PALETTE[0])
            .hover(|style| style.bg(PALETTE[1]))
    }
}

struct HoverHost {
    header: f32,
    child: Entity<HoverBox>,
}

impl Render for HoverHost {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        div()
            .size_full()
            .flex()
            .flex_col()
            .child(div().w(px(10.)).h(px(self.header)))
            .child(self.child.clone())
    }
}

/// A view styled by its hover that moves under a pointer standing still is
/// drawn hovered at once, not with the hover it had where it was.
#[test]
fn a_view_moved_under_the_pointer_is_drawn_with_its_hover_there() {
    let mut cx = TestAppContext::single();
    cx.update(|cx| cx.set_view_retention(true));
    let windows = [(); 2].map(|_| {
        cx.add_window(|_, cx| HoverHost {
            header: 10.,
            child: cx.new(|_| HoverBox),
        })
    });
    draw_both(&mut cx, windows);
    move_mouse_in(&mut cx, windows, 50., 70.);
    draw_both(&mut cx, windows);
    for header in [40., 10., 45.] {
        for window in windows {
            window
                .update(&mut cx, |host, _, cx| {
                    host.header = header;
                    cx.notify();
                })
                .unwrap();
        }
        let [retained, from_scratch] = draw_both(&mut cx, windows);
        assert_eq!(first_difference(&retained, &from_scratch), None, "header {header}");
    }
}

/// A view whose text reaches past it, into what clips it from around.
struct WideText;

impl Render for WideText {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        div()
            .w(px(40.))
            .h(px(20.))
            .child(div().w(px(200.)).child("a line far wider than its view"))
    }
}

struct ClippingHost {
    header: f32,
    child: Entity<WideText>,
}

impl Render for ClippingHost {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        div().size_full().child(
            div()
                .id("clip")
                .w(px(100.))
                .h(px(200.))
                .overflow_hidden()
                .flex()
                .flex_col()
                .child(div().w(px(10.)).h(px(self.header)))
                .child(self.child.clone()),
        )
    }
}

/// A view whose line of text the mask around it clipped is drawn again
/// moved only with its layer clipped where it lands, as painting it there
/// would clip it.
#[test]
fn a_moved_view_keeps_its_text_layers_clipped_as_painted_there() {
    let mut cx = super::super::layout_retention_tests::text_system_context(0);
    cx.update(|cx| cx.set_view_retention(true));
    let windows = [(); 2].map(|_| {
        cx.add_window(|_, cx| ClippingHost {
            header: 10.,
            child: cx.new(|_| WideText),
        })
    });
    draw_both(&mut cx, windows);
    draw_both(&mut cx, windows);
    for header in [30., 190., 20.] {
        for window in windows {
            window
                .update(&mut cx, |host, _, cx| {
                    host.header = header;
                    cx.notify();
                })
                .unwrap();
        }
        let [retained, from_scratch] = draw_both(&mut cx, windows);
        assert_eq!(first_difference(&retained, &from_scratch), None, "header {header}");
    }
}

/// A row holding a scroller tracked by a handle, and a list of its own.
struct ScrollingRow {
    handle: crate::ScrollHandle,
    list_state: ListState,
}

impl Render for ScrollingRow {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        div()
            .w(px(120.))
            .flex()
            .flex_col()
            .child(
                div()
                    .id("scroller")
                    .h(px(40.))
                    .overflow_y_scroll()
                    .track_scroll(&self.handle)
                    .child(div().h(px(120.)).bg(PALETTE[0])),
            )
            .child(
                list(self.list_state.clone(), |ix, _, _| {
                    div()
                        .h(px(20.))
                        .child(SharedString::from(format!("item {ix}")))
                        .into_any_element()
                })
                .h(px(40.)),
            )
    }
}

struct ScrollingRows {
    header: f32,
    row: Entity<ScrollingRow>,
}

impl Render for ScrollingRows {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        div()
            .size_full()
            .flex()
            .flex_col()
            .child(div().w(px(10.)).h(px(self.header)))
            .child(self.row.clone())
    }
}

/// A view holding a tracked scroller and a list, drawn moved, is drawn again
/// on the frame after, not built: the positions moved with it are not a
/// change to what it read.
#[test]
fn a_view_drawn_moved_with_scrollers_in_it_is_drawn_again_after() {
    let mut cx = super::super::layout_retention_tests::text_system_context(0);
    cx.update(|cx| cx.set_view_retention(true));
    let window = cx.add_window(|_, cx| ScrollingRows {
        header: 10.,
        row: cx.new(|_| ScrollingRow {
            handle: crate::ScrollHandle::new(),
            list_state: ListState::new(6, ListAlignment::Top, px(20.)),
        }),
    });
    let frame = |cx: &mut TestAppContext| {
        cx.update_window(window.into(), |_, window, cx| {
            if window.frame_work_stats().frames == 0 {
                window.draw(cx).clear(cx);
            }
            let work = (window.frame_work_stats(), window.view_rebuild_reasons().to_vec());
            window.reset_frame_work_stats(false);
            work
        })
        .unwrap()
    };
    frame(&mut cx);
    frame(&mut cx);
    let row = window.read_with(&cx, |rows, _| rows.row.entity_id()).unwrap();
    for header in [30., 50., 20.] {
        window
            .update(&mut cx, |rows, _, cx| {
                rows.header = header;
                cx.notify();
            })
            .unwrap();
        let (work, reasons) = frame(&mut cx);
        assert_eq!(work.views_moved, 1, "{work:?} {reasons:?}");
        let (work, reasons) = frame(&mut cx);
        assert!(
            !reasons.iter().any(|(view, _)| *view == row),
            "the row was built on the frame after it moved: {reasons:?} {work:?}"
        );
    }
}

struct ScrollingRowList {
    rows: Vec<Entity<ScrollingRow>>,
    list_state: ListState,
}

impl Render for ScrollingRowList {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        let rows = self.rows.clone();
        div().size_full().child(
            list(self.list_state.clone(), move |ix, _, _| rows[ix].clone().into_any_element())
                .w(px(200.))
                .h(px(300.)),
        )
    }
}

/// Rows holding scrollers and lists of their own, scrolled by the wheel in
/// the list holding them, are drawn moved and then drawn again, not built
/// once each scroll has passed.
#[test]
fn rows_with_scrollers_scrolled_by_the_wheel_are_not_built_after() {
    let mut cx = super::super::layout_retention_tests::text_system_context(0);
    cx.update(|cx| cx.set_view_retention(true));
    let window = cx.add_window(|_, cx| ScrollingRowList {
        rows: (0..12)
            .map(|_| {
                cx.new(|_| ScrollingRow {
                    handle: crate::ScrollHandle::new(),
                    list_state: ListState::new(6, ListAlignment::Top, px(20.)),
                })
            })
            .collect(),
        list_state: ListState::new(12, ListAlignment::Top, px(80.)),
    });
    let frame = |cx: &mut TestAppContext| {
        cx.update_window(window.into(), |_, window, cx| {
            if window.frame_work_stats().frames == 0 {
                window.draw(cx).clear(cx);
            }
            let work = (window.frame_work_stats(), window.view_rebuild_reasons().to_vec());
            window.reset_frame_work_stats(false);
            work
        })
        .unwrap()
    };
    frame(&mut cx);
    frame(&mut cx);
    frame(&mut cx);
    for _ in 0..4 {
        let viewport = window
            .read_with(&cx, |rows, _| rows.list_state.viewport_bounds())
            .unwrap();
        // Over the list, beside the rows: the wheel scrolls the list, not a
        // scroller in a row.
        let position = point(viewport.origin.x + px(160.), viewport.center().y);
        cx.update_window(window.into(), |_, window, cx| {
            window.dispatch_event(
                crate::PlatformInput::ScrollWheel(crate::ScrollWheelEvent {
                    position,
                    delta: crate::ScrollDelta::Pixels(point(px(0.), px(-10.))),
                    modifiers: Modifiers::default(),
                    touch_phase: crate::TouchPhase::Moved,
                }),
                cx,
            );
        })
        .unwrap();
        let (work, reasons) = frame(&mut cx);
        assert!(work.views_moved >= 1, "{work:?} {reasons:?}");
        // Only the list's view read the wheel; the rows read nothing that
        // moving them changed.
        assert_eq!(work.view_rebuilds.state_changed, 1, "{reasons:?}");
        let (work, reasons) = frame(&mut cx);
        assert_eq!(work.views_rendered, 0, "{reasons:?}");
    }
}

/// A transcript row styled by its hover, as message rows are.
struct HoverRow {
    ix: usize,
}

impl Render for HoverRow {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        div()
            .id(("hover-row", self.ix))
            .w(px(180.))
            .h(px(40.))
            .bg(PALETTE[self.ix % PALETTE.len()])
            .hover(|style| style.bg(PALETTE[2]))
            .child(SharedString::from(format!("row {}", self.ix)))
    }
}

/// A row's body, styled by whether the row around it, another element, is
/// hovered.
struct GroupHoverBody {
    ix: usize,
}

impl Render for GroupHoverBody {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        div()
            .w(px(160.))
            .h(px(30.))
            .bg(PALETTE[self.ix % PALETTE.len()])
            .group_hover("hover-row", |style| style.bg(PALETTE[3]))
            .child(SharedString::from(format!("body {}", self.ix)))
    }
}

struct HoverRows {
    rows: Vec<Entity<HoverRow>>,
    bodies: Vec<Entity<GroupHoverBody>>,
    list_state: ListState,
}

impl Render for HoverRows {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        let rows = self.rows.clone();
        let bodies = self.bodies.clone();
        div().size_full().child(
            list(self.list_state.clone(), move |ix, _, _| {
                if ix % 2 == 0 {
                    rows[ix].clone().into_any_element()
                } else {
                    div()
                        .id(("group-row", ix))
                        .group("hover-row")
                        .w(px(180.))
                        .h(px(40.))
                        .child(bodies[ix].clone())
                        .into_any_element()
                }
            })
            .w(px(200.))
            .h(px(300.)),
        )
    }
}

/// Scrolling rows styled by their hover, or by the hover of the row around
/// them, under a pointer standing still, one wheel event a frame, draws
/// exactly one frame for each event: no view is left drawn with a hover it
/// no longer has, to be built on a follow-up frame, and no view is built as
/// notified when nothing notified it.
#[test]
fn a_wheel_scroll_under_the_pointer_draws_one_frame_a_wheel_event() {
    let mut cx = super::super::layout_retention_tests::text_system_context(0);
    cx.update(|cx| cx.set_view_retention(true));
    let window = cx.add_window(|_, cx| HoverRows {
        rows: (0..40).map(|ix| cx.new(|_| HoverRow { ix })).collect(),
        bodies: (0..40).map(|ix| cx.new(|_| GroupHoverBody { ix })).collect(),
        list_state: ListState::new(40, ListAlignment::Top, px(40.)),
    });
    let pointer = point(px(100.), px(150.));
    let frame = |cx: &mut TestAppContext| {
        cx.update_window(window.into(), |_, window, cx| {
            if window.frame_work_stats().frames == 0 {
                window.draw(cx).clear(cx);
            }
            let work = (window.frame_work_stats(), window.view_rebuild_reasons().to_vec());
            window.reset_frame_work_stats(false);
            work
        })
        .unwrap()
    };
    frame(&mut cx);
    cx.update_window(window.into(), |_, window, cx| {
        window.dispatch_event(
            crate::PlatformInput::MouseMove(crate::MouseMoveEvent {
                position: pointer,
                pressed_button: None,
                modifiers: Modifiers::default(),
            }),
            cx,
        );
    })
    .unwrap();
    frame(&mut cx);
    frame(&mut cx);
    let mut moved = 0;
    for event in 0..12 {
        cx.update_window(window.into(), |_, window, cx| {
            window.dispatch_event(
                crate::PlatformInput::ScrollWheel(crate::ScrollWheelEvent {
                    position: pointer,
                    delta: crate::ScrollDelta::Pixels(point(px(0.), px(-15.))),
                    modifiers: Modifiers::default(),
                    touch_phase: crate::TouchPhase::Moved,
                }),
                cx,
            );
        })
        .unwrap();
        let (work, reasons) = frame(&mut cx);
        moved += work.views_moved;
        assert_eq!(work.frames, 1, "event {event}: {work:?}");
        assert!(
            !reasons.iter().any(|(_, reason)| *reason == ViewRebuildReason::Notified),
            "event {event}: built as notified with nothing notified: {reasons:?}"
        );
        let follow_up = cx
            .update_window(window.into(), |_, window, cx| {
                window.simulate_next_frame(cx) > 0 || window.invalidator.is_dirty()
            })
            .unwrap();
        assert!(!follow_up, "event {event}: a follow-up frame was asked for");
    }
    assert!(moved > 0, "rows were drawn moved");
}

/// A window idle long enough rebuilds its layout tree smaller, and the
/// views it draws again from the last frame named nodes of the tree that is
/// gone. In this history a list scrolls an item into view after that,
/// laying its items out twice: the first time made nodes under the keys of a
/// card's record, and handed them back, and the second drew the card again
/// at the root its record named, which had been removed.
#[test]
fn views_drawn_again_after_the_layout_tree_was_rebuilt_are_laid_out_afresh() {
    run(418, 50);
}

/// A row of a feed, which shows how often it ticked; the live one ticks on
/// every frame, as a streaming reply does.
struct FeedRow {
    ix: usize,
    ticks: usize,
    /// Grows a line every tick, so that ticking changes its layout.
    grows: bool,
}

impl Render for FeedRow {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        let lines = if self.grows { 1 + self.ticks } else { 1 };
        div()
            .id(("row", self.ix))
            .flex()
            .flex_col()
            .w(px(160.))
            .border_1()
            .border_color(PALETTE[self.ix % PALETTE.len()])
            .hover(|style| style.bg(PALETTE[2]))
            .child(SharedString::from(format!("row {} ticked {}", self.ix, self.ticks)))
            .children((1..lines).map(|line| div().h(px(8.)).child(SharedString::from(format!("{line}")))))
    }
}

/// A panel holding many rows, which renders them and nothing that changes.
struct Feed {
    rows: Vec<Entity<FeedRow>>,
    renders: Rc<Cell<usize>>,
}

impl Render for Feed {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        self.renders.set(self.renders.get() + 1);
        div()
            .size_full()
            .flex()
            .flex_col()
            .gap_1()
            .children(self.rows.iter().cloned())
    }
}

fn feed_windows(cx: &mut TestAppContext, grows: bool) -> ([WindowHandle<Feed>; 2], Rc<Cell<usize>>) {
    cx.update(|cx| cx.set_view_retention(true));
    let renders = Rc::new(Cell::new(0));
    let windows = [(); 2].map(|_| {
        let renders = renders.clone();
        cx.add_window(move |_, cx| Feed {
            rows: (0..12)
                .map(|ix| {
                    cx.new(|_| FeedRow {
                        ix,
                        ticks: 0,
                        grows: grows && ix == 11,
                    })
                })
                .collect(),
            renders,
        })
    });
    (windows, renders)
}

/// Ticks the live row in both windows and draws them, the second from
/// scratch, returning the first's work.
fn tick_feeds(cx: &mut TestAppContext, windows: [WindowHandle<Feed>; 2]) -> crate::FrameWorkStats {
    for window in windows {
        let live = window.read_with(cx, |feed, _| feed.rows[11].clone()).unwrap();
        live.update(cx, |row, cx| {
            row.ticks += 1;
            cx.notify();
        });
    }
    let (retained, work) = cx
        .update_window(windows[0].into(), |_, window, cx| {
            if window.frame_work_stats().frames == 0 {
                window.draw(cx).clear(cx);
            }
            let work = window.frame_work_stats();
            window.reset_frame_work_stats(false);
            (describe_frame(window), work)
        })
        .unwrap();
    let from_scratch = cx
        .update_window(windows[1].into(), |_, window, cx| {
            window.refresh();
            window.draw(cx).clear(cx);
            describe_frame(window)
        })
        .unwrap();
    assert_eq!(first_difference(&retained, &from_scratch), None);
    work
}

/// A row notified on every frame builds only itself: the panel around it is
/// drawn again around it, the other rows copied along, and the panel is not
/// rendered.
#[test]
fn a_panel_is_drawn_again_around_its_live_row() {
    let mut cx = super::super::layout_retention_tests::text_system_context(0);
    let (windows, renders) = feed_windows(&mut cx, false);
    tick_feeds(&mut cx, windows);
    tick_feeds(&mut cx, windows);
    let rendered_before = renders.get();
    for _ in 0..4 {
        let work = tick_feeds(&mut cx, windows);
        assert_eq!(work.views_rendered, 1, "{work:?}");
        assert_eq!(work.views_spliced, 1, "{work:?}");
    }
    // Only the window drawing from scratch rendered its panel, once a tick
    // when refreshed: the frame the tick drew there drew it again around its
    // row as well.
    assert_eq!(renders.get(), rendered_before + 4);
}

/// A live row that grows asks for another layout: the panel is built
/// instead of drawn again around it, and both frames still match.
#[test]
fn a_panel_is_built_when_its_live_row_changes_its_layout() {
    let mut cx = super::super::layout_retention_tests::text_system_context(0);
    let (windows, renders) = feed_windows(&mut cx, true);
    tick_feeds(&mut cx, windows);
    tick_feeds(&mut cx, windows);
    let rendered_before = renders.get();
    for _ in 0..4 {
        let work = tick_feeds(&mut cx, windows);
        // Tried and rolled back (counted as a change of context), or not
        // tried: a panel built at a layout one of its views did not keep is
        // built again on the next frame.
        // The panel and the row, and the row once more when the splice was
        // tried and rolled back.
        assert_eq!(work.views_spliced, 0, "{work:?}");
        assert!(work.views_rendered <= 3, "{work:?}");
    }
    // Every frame rendered the panel: one per tick drawing views again, and
    // two drawing from scratch, the frame the tick drew and the refreshed
    // one.
    assert_eq!(renders.get(), rendered_before + 4 * 3);
}

/// A row reading a dependency and a model of its own.
struct DependentRow {
    ix: usize,
    dependency: DrawDependency,
    model: Entity<usize>,
}

impl Render for DependentRow {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        window.depend_on(&self.dependency);
        let count = *self.model.read(cx);
        div()
            .w(px(200.))
            .h(px(20.))
            .bg(PALETTE[(self.ix + count) % PALETTE.len()])
            .child(SharedString::from(format!("row {} {count}", self.ix)))
    }
}

struct DependentRows {
    rows: Vec<Entity<DependentRow>>,
    renders: Rc<Cell<usize>>,
}

impl Render for DependentRows {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        self.renders.set(self.renders.get() + 1);
        div().flex().flex_col().children(self.rows.iter().cloned())
    }
}

/// A host none of whose own reads changed, holding a row whose dependency
/// or model changed (no view was notified), is drawn again around the row,
/// not built.
#[test]
fn a_host_is_drawn_again_around_a_row_whose_dependencies_changed() {
    let mut cx = super::super::layout_retention_tests::text_system_context(0);
    cx.update(|cx| cx.set_view_retention(true));
    let renders = Rc::new(Cell::new(0));
    let window = cx.add_window({
        let renders = renders.clone();
        move |_, cx| DependentRows {
            rows: (0..6)
                .map(|ix| {
                    cx.new(|cx| DependentRow {
                        ix,
                        dependency: DrawDependency::new(),
                        model: cx.new(|_| 0),
                    })
                })
                .collect(),
            renders,
        }
    });
    // Drawing on the update's flush, if it drew nothing.
    let frame = |cx: &mut TestAppContext| {
        cx.update_window(window.into(), |_, window, cx| {
            if window.frame_work_stats().frames == 0 {
                window.draw(cx).clear(cx);
            }
            window.frame_work_stats()
        })
        .unwrap()
    };
    frame(&mut cx);
    let row = window.read_with(&cx, |host, _| host.rows[3].clone()).unwrap();
    let rendered_before = renders.get();
    for step in 0..4 {
        cx.update_window(window.into(), |_, window, _| window.reset_frame_work_stats(false))
            .unwrap();
        if step % 2 == 0 {
            row.update(&mut cx, |row, cx| row.dependency.changed(cx));
        } else {
            let model = row.read_with(&cx, |row, _| row.model.clone());
            model.update(&mut cx, |count, cx| {
                *count += 1;
                cx.notify();
            });
        }
        let work = frame(&mut cx);
        assert_eq!(work.views_rendered, 1, "{work:?}");
        assert_eq!(work.views_spliced, 1, "{work:?}");
    }
    assert_eq!(renders.get(), rendered_before, "the host was not built");
}

/// A badge in a row, painted by whether the row's group is hovered.
struct RowBadge {
    count: usize,
}

impl Render for RowBadge {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        // The same size whatever its count, so that building it asks for
        // the layout it had.
        div()
            .w(px(10.))
            .h(px(10.))
            .bg(PALETTE[self.count % 3])
            .group_hover("row", |style| style.border_1().border_color(PALETTE[3]))
    }
}

/// A row that is a group container, with a badge view in it.
struct GroupRow {
    ix: usize,
    badge: Entity<RowBadge>,
}

impl Render for GroupRow {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        div()
            .id(("group-row", self.ix))
            .group("row")
            .w(px(100.))
            .h(px(20.))
            .child(self.badge.clone())
    }
}

struct GroupRows {
    rows: Vec<Entity<GroupRow>>,
}

impl Render for GroupRows {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        div()
            .size_full()
            .flex()
            .flex_col()
            .children(self.rows.iter().cloned())
    }
}

/// A view built in a gap of a view drawn again around it resolves the group
/// containers around it as it did: a badge hovering by its row's group,
/// rebuilt while the row is hovered and drawn again around it, still paints
/// hovered.
#[test]
fn a_view_built_in_a_gap_resolves_the_groups_around_it() {
    let mut cx = super::super::layout_retention_tests::text_system_context(0);
    cx.update(|cx| cx.set_view_retention(true));
    let windows = [(); 2].map(|_| {
        cx.add_window(|_, cx| GroupRows {
            rows: (0..6)
                .map(|ix| {
                    let badge = cx.new(|_| RowBadge { count: 0 });
                    cx.new(|_| GroupRow { ix, badge })
                })
                .collect(),
        })
    });
    let draw_both = |cx: &mut TestAppContext| {
        let (retained, work) = cx
            .update_window(windows[0].into(), |_, window, cx| {
                if window.frame_work_stats().frames == 0 {
                    window.draw(cx).clear(cx);
                }
                let work = window.frame_work_stats();
                window.reset_frame_work_stats(false);
                (describe_frame(window), work)
            })
            .unwrap();
        let from_scratch = cx
            .update_window(windows[1].into(), |_, window, cx| {
                window.refresh();
                window.draw(cx).clear(cx);
                describe_frame(window)
            })
            .unwrap();
        assert_eq!(first_difference(&retained, &from_scratch), None);
        work
    };
    draw_both(&mut cx);
    // Over the fourth row.
    for window in windows {
        cx.update_window(window.into(), |_, window, cx| {
            window.dispatch_event(
                crate::PlatformInput::MouseMove(crate::MouseMoveEvent {
                    position: point(px(50.), px(3. * 20. + 10.)),
                    pressed_button: None,
                    modifiers: Modifiers::default(),
                }),
                cx,
            );
        })
        .unwrap();
    }
    draw_both(&mut cx);
    draw_both(&mut cx);
    for _ in 0..3 {
        for window in windows {
            let badge = window
                .read_with(&cx, |rows, cx| rows.rows[3].read(cx).badge.clone())
                .unwrap();
            badge.update(&mut cx, |badge, cx| {
                badge.count += 1;
                cx.notify();
            });
        }
        let work = draw_both(&mut cx);
        // The rows and the row around the badge, drawn again around it.
        assert_eq!(work.views_spliced, 2, "{work:?}");
        assert_eq!(work.views_rendered, 1, "{work:?}");
    }
}

/// A badge in a transcript row: a count, which ticks as a spinner would.
struct TranscriptBadge {
    count: usize,
}

impl Render for TranscriptBadge {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        div()
            .w(px(40.))
            .h(px(12.))
            .child(SharedString::from(format!("{:03}", self.count % 1000)))
    }
}

/// A transcript row: lines of text and a badge, sized by its content.
struct TranscriptRow {
    ix: usize,
    lines: usize,
    tail: usize,
    badge: Entity<TranscriptBadge>,
}

impl Render for TranscriptRow {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        let ix = self.ix;
        div()
            .id(("transcript-row", ix))
            .flex()
            .flex_col()
            .w_full()
            .p_1()
            .border_1()
            .border_color(PALETTE[ix % PALETTE.len()])
            .child(
                div()
                    .flex()
                    .flex_row()
                    .gap_1()
                    .child(SharedString::from(format!("message {ix}")))
                    .child(self.badge.clone()),
            )
            .children((0..self.lines).map(move |line| {
                div().child(SharedString::from(format!(
                    "{ix}:{line} the quick brown fox jumps over the lazy dog"
                )))
            }))
            .child(SharedString::from("x".repeat(self.tail % 40)))
    }
}

/// A transcript: rows in a list, bottom-aligned as a chat is.
struct Transcript {
    rows: Vec<Entity<TranscriptRow>>,
    list_state: ListState,
}

impl Render for Transcript {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        let rows = self.rows.clone();
        div().size_full().child(
            list(self.list_state.clone(), move |ix, _, _| rows[ix].clone().into_any_element())
                .size_full(),
        )
    }
}

fn transcript_windows(cx: &mut TestAppContext) -> [WindowHandle<Transcript>; 2] {
    cx.update(|cx| cx.set_view_retention(true));
    [(); 2].map(|_| {
        let window = cx.add_window(|_, cx| {
            let rows: Vec<_> = (0..20)
                .map(|ix| {
                    let badge = cx.new(|_| TranscriptBadge { count: 0 });
                    cx.new(|_| TranscriptRow {
                        ix,
                        lines: 1 + ix % 4,
                        tail: 1,
                        badge,
                    })
                })
                .collect();
            Transcript {
                list_state: ListState::new(rows.len(), ListAlignment::Bottom, px(100.)),
                rows,
            }
        });
        cx.simulate_window_resize(window.into(), size(px(600.), px(400.)));
        window
    })
}

/// Streams into the last row of both windows, a character or, with
/// `line`, a line, and draws them, the second from scratch, returning the
/// first's work.
fn stream_transcripts(
    cx: &mut TestAppContext,
    windows: [WindowHandle<Transcript>; 2],
    line: bool,
) -> crate::FrameWorkStats {
    for window in windows {
        let last = window
            .read_with(cx, |transcript, _| transcript.rows[19].clone())
            .unwrap();
        last.update(cx, |row, cx| {
            row.tail += 1;
            if line {
                row.lines += 1;
            }
            cx.notify();
        });
    }
    let (retained, work) = cx
        .update_window(windows[0].into(), |_, window, cx| {
            if window.frame_work_stats().frames == 0 {
                window.draw(cx).clear(cx);
            }
            let work = window.frame_work_stats();
            window.reset_frame_work_stats(false);
            (describe_frame(window), work)
        })
        .unwrap();
    let from_scratch = cx
        .update_window(windows[1].into(), |_, window, cx| {
            window.refresh();
            window.draw(cx).clear(cx);
            describe_frame(window)
        })
        .unwrap();
    assert_eq!(first_difference(&retained, &from_scratch), None);
    work
}

/// A list item whose content changed without changing its size is laid out
/// again on its own, in the space the list gave it, and the panel holding
/// the list is drawn again around it; one that grew has the panel built, for
/// the list to place the items around it again.
#[test]
fn a_list_item_laid_out_again_at_its_size_is_drawn_again_in_place() {
    let mut cx = super::super::layout_retention_tests::text_system_context(0);
    let windows = transcript_windows(&mut cx);
    stream_transcripts(&mut cx, windows, false);
    stream_transcripts(&mut cx, windows, false);
    for frame in 0..8 {
        let line = frame % 4 == 3;
        let work = stream_transcripts(&mut cx, windows, line);
        if line {
            assert_eq!(work.views_spliced, 0, "{work:?}");
        } else {
            assert_eq!(work.views_spliced, 1, "{work:?}");
            assert_eq!(work.views_rendered, 1, "{work:?}");
        }
    }
}

/// The work of drawing a 200-row transcript, with views drawn again and
/// without, while one row's badge ticks, the last row streams (a line every
/// eighth frame), the list scrolls, and nothing changes but the panel being
/// notified. Run with `--release --ignored --nocapture`; to profile one,
/// `GPUI_BENCH_ONLY=Stream:true` and `GPUI_BENCH_FRAMES=20000`.
#[test]
#[ignore]
fn frame_work_transcript() {
    let only = std::env::var("GPUI_BENCH_ONLY").ok();
    let iterations: u32 = std::env::var("GPUI_BENCH_FRAMES")
        .ok()
        .and_then(|frames| frames.parse().ok())
        .unwrap_or(240);
    #[derive(Clone, Copy, Debug)]
    enum Scenario {
        Tick,
        Stream,
        Scroll,
        Redraw,
    }
    for scenario in [Scenario::Tick, Scenario::Stream, Scenario::Scroll, Scenario::Redraw] {
        for retained in [false, true] {
            if only
                .as_ref()
                .is_some_and(|only| *only != format!("{scenario:?}:{retained}"))
            {
                continue;
            }
            let mut cx = super::super::layout_retention_tests::text_system_context(0);
            cx.update(|cx| cx.set_view_retention(retained));
            let window = cx.add_window(|_, cx| {
                let rows: Vec<_> = (0..200)
                    .map(|ix| {
                        let badge = cx.new(|_| TranscriptBadge { count: 0 });
                        cx.new(|_| TranscriptRow {
                            ix,
                            lines: 1 + ix % 4,
                            tail: 0,
                            badge,
                        })
                    })
                    .collect();
                Transcript {
                    list_state: ListState::new(rows.len(), ListAlignment::Bottom, px(200.)),
                    rows,
                }
            });
            cx.simulate_window_resize(window.into(), size(px(900.), px(1200.)));
            let frame = |cx: &mut TestAppContext| {
                cx.update_window(window.into(), |_, window, cx| window.draw(cx).clear(cx))
                    .unwrap();
            };
            let (rows, list_state) = window
                .read_with(&cx, |transcript, _| {
                    (transcript.rows.clone(), transcript.list_state.clone())
                })
                .unwrap();
            if matches!(scenario, Scenario::Scroll) {
                list_state.scroll_to(ListOffset {
                    item_ix: 150,
                    offset_in_item: px(0.),
                });
            }
            for _ in 0..4 {
                frame(&mut cx);
            }
            cx.update_window(window.into(), |_, window, _| window.reset_frame_work_stats(true))
                .unwrap();
            let frames = iterations;
            let started = Instant::now();
            for frame_ix in 0..frames as usize {
                match scenario {
                    Scenario::Tick => {
                        let badge = rows[190].read_with(&cx, |row, _| row.badge.clone());
                        badge.update(&mut cx, |badge, cx| {
                            badge.count += 1;
                            cx.notify();
                        });
                    }
                    Scenario::Stream => rows[199].update(&mut cx, |row, cx| {
                        row.tail += 1;
                        if frame_ix % 8 == 7 {
                            row.lines += 1;
                        }
                        cx.notify();
                    }),
                    Scenario::Scroll => {
                        let delta = if (frame_ix / 60) % 2 == 0 { 6. } else { -6. };
                        list_state.scroll_by(px(delta));
                        window.update(&mut cx, |_, _, cx| cx.notify()).unwrap();
                    }
                    Scenario::Redraw => window.update(&mut cx, |_, _, cx| cx.notify()).unwrap(),
                }
                frame(&mut cx);
            }
            let elapsed = started.elapsed();
            let work = cx
                .update_window(window.into(), |_, window, _| window.frame_work_stats())
                .unwrap();
            let draws = work.frames as f64 / frames as f64;
            eprintln!(
                "{scenario:?} retained {retained}: {:?} per change ({draws:.1} draws); per draw \
                 rendered {:.1} reused {:.1} moved {:.1} spliced {:.1} elements {:.0}, build \
                 {:?} prepaint {:?} paint {:?}",
                elapsed / frames,
                work.views_rendered as f64 / work.frames as f64,
                work.views_reused as f64 / work.frames as f64,
                work.views_moved as f64 / work.frames as f64,
                work.views_spliced as f64 / work.frames as f64,
                work.elements as f64 / work.frames as f64,
                work.build_time / work.frames as u32,
                work.prepaint_time / work.frames as u32,
                work.paint_time / work.frames as u32,
            );
        }
    }
}
