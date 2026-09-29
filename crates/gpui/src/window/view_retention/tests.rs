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
    ListOffset, ListState, Modifiers, Render, SharedString, StyleRefinement, TestAppContext,
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

/// A global some cards read.
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
        ][ix % 6]
    }
}

struct Card {
    ix: usize,
    count: usize,
    kind: CardKind,
    popover: bool,
    /// Asks the list it is in to scroll it into view, which rolls the list's
    /// prepaint back and lays its items out again.
    reveal: bool,
    inner: Entity<Inner>,
    shared: Rc<Shared>,
}

impl Card {
    fn new(ix: usize, shared: Rc<Shared>, cx: &mut Context<Self>) -> Self {
        let kind = CardKind::of(ix);
        if matches!(kind, CardKind::OptedOut) {
            cx.set_view_retainable(false);
        }
        let model = shared.model.clone();
        Self {
            ix,
            count: 0,
            kind,
            popover: false,
            reveal: false,
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
                WORDS[labels[self.ix % labels.len()]].into()
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
            .child(div().w(px(6. + self.count as f32 * 2.)).h(px(6.)).bg(PALETTE[1]))
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
    column: bool,
    tint: usize,
}

impl Shell {
    fn new(shared: &Rc<Shared>, cx: &mut Context<Self>) -> Self {
        let cards = (0..CARDS)
            .map(|ix| cx.new(|cx| Card::new(ix, shared.clone(), cx)))
            .collect();
        let list_cards: Vec<_> = (0..CARDS)
            .map(|ix| cx.new(|cx| Card::new(ix + 100, shared.clone(), cx)))
            .collect();
        Self {
            cards,
            list_state: ListState::new(list_cards.len(), ListAlignment::Top, px(20.)),
            list_cards,
            column: false,
            tint: 0,
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
            .children(self.cards.iter().enumerate().map(|(ix, card)| {
                if ix == 1 {
                    card.clone()
                        .cached(StyleRefinement::default().w(px(140.)).h(px(70.)))
                        .into_any_element()
                } else {
                    card.clone().into_any_element()
                }
            }))
            .child(items)
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
    Mouse { x: f32, y: f32 },
    Resize { width: f32, height: f32 },
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
            80..92 => Change::Mouse {
                x: rng.random_range(0.0..700.0),
                y: rng.random_range(0.0..500.0),
            },
            92..95 => Change::Resize {
                width: rng.random_range(300.0..900.0),
                height: rng.random_range(240.0..700.0),
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
        let mut cx = TestAppContext::single();
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
            Change::Label { at, word } => model.update(&mut self.cx, |model, cx| {
                model.labels[at] = word;
                cx.notify();
            }),
            Change::QuietLabelThenNotifyShell { at, word } => {
                model.update(&mut self.cx, |model, _| model.labels[at] = word);
                self.update_shells(|_, _| {});
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
                    let ix = shell.cards.len();
                    let card = cx.new(|cx| Card::new(ix, shared.clone(), cx));
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
            Change::Resize { width, height } => {
                for window in self.windows() {
                    self.cx
                        .simulate_window_resize(window.into(), size(px(width), px(height)));
                }
            }
            Change::Redraw => {}
        }
    }

    fn draw(&mut self) -> (Vec<String>, Vec<String>, usize) {
        let expected = self
            .cx
            .update_window(self.from_scratch.into(), |_, window, cx| {
                window.refresh();
                window.draw(cx).clear(cx);
                describe_frame(window)
            })
            .unwrap();
        let (actual, reused) = self
            .cx
            .update_window(self.retaining.into(), |_, window, cx| {
                window.reset_frame_work_stats(false);
                window.draw(cx).clear(cx);
                (describe_frame(window), window.frame_work_stats().views_reused as usize)
            })
            .unwrap();
        (actual, expected, reused)
    }
}

fn run(seed: u64, steps: usize) -> usize {
    let mut oracle = Oracle::new();
    let mut rng = StdRng::seed_from_u64(seed);
    let mut history: Vec<Vec<Change>> = Vec::new();
    let mut reused = 0;
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
        let (actual, expected, reused_now) = oracle.draw();
        reused += reused_now;
        if let Some(difference) = first_difference(&actual, &expected) {
            let history = history
                .iter()
                .enumerate()
                .map(|(step, changes)| format!("  {step}: {changes:?}"))
                .collect::<Vec<_>>()
                .join("\n");
            panic!(
                "seed {seed}, step {step}: the frame drawing views again differs from the frame \
                 drawn from scratch at {difference}\nchanges so far:\n{history}"
            );
        }
    }
    reused
}

#[test]
fn frames_drawing_views_again_match_frames_drawn_from_scratch() {
    let mut reused = 0;
    let seeds = std::env::var("GPUI_RETAINED_VIEWS_ORACLE_SEEDS").ok().and_then(|seeds| seeds.parse().ok()).unwrap_or(16);
    for seed in 0..seeds {
        reused += run(seed, 50);
    }
    assert!(reused > 1000, "views were drawn again {reused} times");
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
    let model = cx.new(|_| Model { labels: vec![0; CARDS] });
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

/// Notifying a view by id builds it and the views around it, and draws the
/// rest, and what read it, again.
#[test]
fn notifying_a_view_builds_only_it_and_the_views_around_it() {
    let mut cx = TestAppContext::single();
    let (window, _) = shell_window(&mut cx);
    let inner = window
        .read_with(&cx, |shell, cx| shell.cards[2].read(cx).inner.clone())
        .unwrap();
    let work = work_after(&mut cx, window, |cx| cx.update(|cx| cx.notify(inner.entity_id())));
    let reasons = rebuilds(&mut cx, window);
    // The inner view, its card and the shell; the opted-out cards are built
    // on every frame.
    assert_eq!(work.view_rebuilds.notified, 3, "{work:?} {reasons:?}");
    assert_eq!(work.view_rebuilds.entity_changed, 0, "{work:?}");
    assert!(work.views_reused >= CARDS as u64, "{work:?}");
}

/// A model updated and notified builds every view that read it (the inner
/// view of every card, and so every card), and a global written builds the
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
    assert!(work.view_rebuilds.entity_changed >= 2 * CARDS as u64, "{work:?}");
    let work = work_after(&mut cx, window, |cx| cx.update(|cx| cx.set_global(Theme(4))));
    assert!(work.view_rebuilds.global_changed >= 1, "{work:?}");
    assert_eq!(work.view_rebuilds.entity_changed, 0, "{work:?}");
    assert!(work.views_reused >= CARDS as u64, "{work:?}");
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

/// Verification draws a frame that drew views again once more from scratch
/// and finds nothing to report when the two agree.
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
    assert_eq!(work.frames, 2, "the frame was drawn again: {work:?}");
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
/// it in, and a host that wraps it in opacity cycles or shimmers.
struct Mover {
    child: Entity<Leaf>,
    started_at: Instant,
    transition: Option<Duration>,
    cycle: bool,
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
        div().size_full().child(Wrapper {
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
    change(&mut cx, Some(Duration::from_millis(200)));
    cx.executor().advance_clock(Duration::from_millis(50));
    change(&mut cx, Some(Duration::from_millis(300)));
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
