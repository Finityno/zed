use crate::{
    AnyElement, App, Bounds, Element, GlobalElementId, InspectorElementId, IntoElement, LayoutId,
    Pixels, Window,
};

/// Builds a `Deferred` element, which delays the layout and paint of its child.
pub fn deferred(child: impl IntoElement) -> Deferred {
    Deferred {
        child: Some(child.into_any_element()),
        priority: 0,
        beneath_native_surfaces: false,
    }
}

/// An element which delays the painting of its child until after all of
/// its ancestors, while keeping its layout as part of the current element tree.
pub struct Deferred {
    child: Option<AnyElement>,
    priority: usize,
    beneath_native_surfaces: bool,
}

impl Deferred {
    /// Sets the `priority` value of the `deferred` element, which
    /// determines the drawing order relative to other deferred elements,
    /// with higher values being drawn on top.
    pub fn with_priority(mut self, priority: usize) -> Self {
        self.priority = priority;
        self
    }

    /// Keeps the child on the window's base scene plane, beneath any native
    /// child surface the window hosts, instead of the overlay plane a deferred
    /// draw normally lands on once [`Window::enable_scene_overlay`] has been
    /// called.
    ///
    /// Use this when the deferral only exists to paint the child after its
    /// siblings — content that is part of the page, not a popup, menu or
    /// tooltip that must show above a native view. Such a draw is painted
    /// before every other deferred draw regardless of priority, because the
    /// overlay plane composites above the base plane anyway, and it never
    /// counts as overlay content for input capture. A draw nested inside a
    /// deferred draw that is on the overlay plane stays on the overlay plane.
    pub fn beneath_native_surfaces(mut self) -> Self {
        self.beneath_native_surfaces = true;
        self
    }
}

impl Element for Deferred {
    type RequestLayoutState = ();
    type PrepaintState = ();

    fn id(&self) -> Option<crate::ElementId> {
        None
    }

    fn source_location(&self) -> Option<&'static core::panic::Location<'static>> {
        None
    }

    fn request_layout(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        window: &mut Window,
        cx: &mut App,
    ) -> (LayoutId, ()) {
        let layout_id = self.child.as_mut().unwrap().request_layout(window, cx);
        (layout_id, ())
    }

    fn prepaint(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        _bounds: Bounds<Pixels>,
        _request_layout: &mut Self::RequestLayoutState,
        window: &mut Window,
        _cx: &mut App,
    ) {
        let child = self.child.take().unwrap();
        let element_offset = window.element_offset();
        if self.beneath_native_surfaces {
            window.defer_draw_beneath_native_surfaces(child, element_offset, self.priority, None)
        } else {
            window.defer_draw(child, element_offset, self.priority, None)
        }
    }

    fn paint(
        &mut self,
        _id: Option<&GlobalElementId>,
        _inspector_id: Option<&InspectorElementId>,
        _bounds: Bounds<Pixels>,
        _request_layout: &mut Self::RequestLayoutState,
        _prepaint: &mut Self::PrepaintState,
        _window: &mut Window,
        _cx: &mut App,
    ) {
    }
}

impl IntoElement for Deferred {
    type Element = Self;

    fn into_element(self) -> Self::Element {
        self
    }
}

impl Deferred {
    /// Sets a priority for the element. A higher priority conceptually means painting the element
    /// on top of deferred draws with a lower priority (i.e. closer to the viewer).
    pub fn priority(mut self, priority: usize) -> Self {
        self.priority = priority;
        self
    }
}

#[cfg(test)]
mod tests {
    use std::{cell::Cell, rc::Rc};

    use crate::{
        Context, Entity, StyleRefinement, TestAppContext, Window, anchored, deferred, div, point,
        prelude::*, px, size,
    };

    struct SubscriptionLeaf {
        value: usize,
        output: Rc<Cell<usize>>,
    }

    struct SubscriptionPanel {
        leaf: Entity<SubscriptionLeaf>,
        deferred_child: bool,
        renders: Rc<Cell<usize>>,
    }

    impl Render for SubscriptionPanel {
        fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
            self.renders.set(self.renders.get() + 1);
            let leaf = self.leaf.clone();
            let child = crate::canvas(
                |_, _, _| {},
                move |_, _, _, cx| {
                    let leaf = leaf.read(cx);
                    leaf.output.set(leaf.value);
                },
            )
            .size(px(50.));
            if self.deferred_child {
                deferred(child).into_any_element()
            } else {
                child.into_any_element()
            }
        }
    }

    struct SubscriptionRoot {
        panel: Entity<SubscriptionPanel>,
        deferred_panel: bool,
        renders: Rc<Cell<usize>>,
    }

    impl Render for SubscriptionRoot {
        fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
            self.renders.set(self.renders.get() + 1);
            let panel = self
                .panel
                .clone()
                .cached(StyleRefinement::default().size(px(100.)));
            if self.deferred_panel {
                deferred(panel).into_any_element()
            } else {
                panel.into_any_element()
            }
        }
    }

    #[gpui::test]
    fn test_cached_reuse_keeps_entity_subscriptions(cx: &mut TestAppContext) {
        for deferred_child in [true, false] {
            for deferred_panel in [false, true] {
                let output = Rc::new(Cell::new(0));
                let renders = Rc::new(Cell::new(0));
                let root_renders = Rc::new(Cell::new(0));
                let leaf = cx.new(|_| SubscriptionLeaf {
                    value: 1,
                    output: output.clone(),
                });
                let panel = cx.new(|_| SubscriptionPanel {
                    leaf: leaf.clone(),
                    deferred_child,
                    renders: renders.clone(),
                });
                let window = cx.open_window(size(px(800.), px(600.)), |_, _| SubscriptionRoot {
                    panel,
                    deferred_panel,
                    renders: root_renders.clone(),
                });
                cx.run_until_parked();
                assert_eq!(output.get(), 1);
                let initial_renders = renders.get();
                for _ in 0..2 {
                    window
                        .update(cx, |_, _, cx| cx.notify())
                        .expect("notify root");
                    cx.run_until_parked();
                    assert_eq!(
                        renders.get(),
                        initial_renders,
                        "panel should reuse its cache"
                    );
                }

                let renders_before_notify = root_renders.get();
                leaf.update(cx, |leaf, cx| {
                    leaf.value = 2;
                    cx.notify();
                });
                cx.run_until_parked();
                let context =
                    format!("deferred child {deferred_child}, deferred panel {deferred_panel}");
                assert!(
                    root_renders.get() > renders_before_notify,
                    "leaf notification should redraw: {context}"
                );
                // A plain entity read has no dispatch-tree path to invalidate the
                // cached panel, so refresh to run the paint callback again.
                window
                    .update(cx, |_, window, _| window.refresh())
                    .expect("refresh canvas");
                cx.run_until_parked();
                assert_eq!(output.get(), 2, "{context}");
            }
        }
    }

    /// A stand-in for a dock panel hosting a popover (deferred draw) whose
    /// content opens another popover (a deferred draw created while
    /// prepainting the first one's content).
    struct PanelView;

    impl Render for PanelView {
        fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
            div().key_context("Panel").size_full().child(
                deferred(
                    anchored().position(point(px(10.), px(10.))).child(
                        div().key_context("Popover").w(px(200.)).h(px(200.)).child(
                            deferred(
                                anchored().position(point(px(30.), px(30.))).child(
                                    div()
                                        .key_context("NestedMenu")
                                        .debug_selector(|| "NESTED_MENU".into())
                                        .w(px(50.))
                                        .h(px(50.)),
                                ),
                            )
                            .with_priority(2),
                        ),
                    ),
                )
                .with_priority(1),
            )
        }
    }

    struct RootView {
        panel: Entity<PanelView>,
    }

    impl Render for RootView {
        fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
            div().key_context("Root").size_full().child(
                self.panel
                    .clone()
                    .cached(StyleRefinement::default().size_full()),
            )
        }
    }

    /// Regression test for a crash with nested deferred draws (e.g. a popover
    /// menu inside a popover hosted by a cached dock panel). Prepaint indices
    /// recorded during the deferred draw rounds must index the same
    /// `deferred_draws` vector that `reuse_prepaint` slices on the next frame;
    /// previously they were measured against a transient per-round vector, so
    /// reusing the panel's subtree grafted the wrong deferred draws and
    /// panicked in the dispatch tree.
    #[gpui::test]
    fn test_nested_deferred_draws_with_reused_views(cx: &mut TestAppContext) {
        let window = cx.open_window(size(px(800.), px(600.)), |_, cx| {
            let panel = cx.new(|_| PanelView);
            RootView { panel }
        });
        cx.run_until_parked();

        let menu_bounds = window
            .update(cx, |_, window, _| {
                window
                    .rendered_frame
                    .debug_bounds
                    .get("NESTED_MENU")
                    .copied()
            })
            .unwrap()
            .expect("NESTED_MENU debug bounds not found");
        assert_eq!(menu_bounds.size, size(px(50.), px(50.)));

        // Re-render only the root view; the panel is cached, so its subtree -
        // including both deferred draw records - is reused from the previous
        // frame.
        window.update(cx, |_, _, cx| cx.notify()).unwrap();
        cx.run_until_parked();

        // Reuse the subtree a second time, exercising ranges that were
        // themselves recorded during a reused frame.
        window.update(cx, |_, _, cx| cx.notify()).unwrap();
        cx.run_until_parked();

        // Re-render the panel itself again to prove the popovers still draw.
        window
            .update(cx, |root, _, cx| {
                root.panel.update(cx, |_, cx| cx.notify());
            })
            .unwrap();
        cx.run_until_parked();

        window
            .update(cx, |_, window, _| {
                assert_eq!(window.rendered_frame.deferred_draws.len(), 2);
                assert!(
                    window
                        .rendered_frame
                        .debug_bounds
                        .contains_key("NESTED_MENU")
                );
            })
            .unwrap();
    }

    /// A page whose composer is deferred only to paint after its siblings,
    /// with a menu popped out of it, next to an ordinary popover.
    struct PlanesView;

    const BENEATH_WIDTH: f32 = 50.;
    const NESTED_MENU_WIDTH: f32 = 10.;
    const POPOVER_WIDTH: f32 = 20.;
    const MISPLACED_WIDTH: f32 = 30.;

    impl Render for PlanesView {
        fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
            div()
                .size_full()
                .child(
                    // Kept beneath native surfaces despite outranking the
                    // popover's priority; the menu it opens must still show
                    // above a native view.
                    deferred(
                        div()
                            .w(px(BENEATH_WIDTH))
                            .h(px(BENEATH_WIDTH))
                            .bg(crate::rgb(0xff0000))
                            .child(
                                deferred(
                                    anchored().position(point(px(5.), px(5.))).child(
                                        div()
                                            .w(px(NESTED_MENU_WIDTH))
                                            .h(px(NESTED_MENU_WIDTH))
                                            .bg(crate::rgb(0x00ff00)),
                                    ),
                                )
                                .with_priority(2),
                            ),
                    )
                    .with_priority(7)
                    .beneath_native_surfaces(),
                )
                .child(
                    deferred(
                        anchored().position(point(px(100.), px(100.))).child(
                            div()
                                .w(px(POPOVER_WIDTH))
                                .h(px(POPOVER_WIDTH))
                                .bg(crate::rgb(0x0000ff))
                                // Asking for the base plane from inside an
                                // overlay-plane draw is refused, or the popover
                                // would be painted over by its own content's
                                // backdrop.
                                .child(
                                    deferred(
                                        div()
                                            .w(px(MISPLACED_WIDTH))
                                            .h(px(MISPLACED_WIDTH))
                                            .bg(crate::rgb(0xffff00)),
                                    )
                                    .with_priority(3)
                                    .beneath_native_surfaces(),
                                ),
                        ),
                    )
                    .with_priority(1),
                )
        }
    }

    /// Quad widths in logical pixels, in draw order (scene quads are scaled).
    fn quad_widths(scene: &crate::Scene, scale_factor: f32) -> Vec<f32> {
        scene
            .quads
            .iter()
            .map(|quad| quad.bounds.size.width.0 / scale_factor)
            .collect()
    }

    #[gpui::test]
    fn test_deferred_draws_beneath_native_surfaces_stay_on_the_base_plane(
        cx: &mut TestAppContext,
    ) {
        let window = cx.open_window(size(px(800.), px(600.)), |_, _| PlanesView);
        cx.run_until_parked();

        window
            .update(cx, |_, window, _| {
                let scene = &window.rendered_frame.scene;
                let split = window.rendered_frame.overlay_scene_start;
                assert!(split < scene.len(), "the overlay plane must not be empty");

                let mut base_plane = crate::Scene::default();
                let mut overlay_plane = crate::Scene::default();
                scene.split_planes_into(split, &mut base_plane, &mut overlay_plane);

                let scale_factor = window.scale_factor();
                assert_eq!(
                    quad_widths(&base_plane, scale_factor),
                    vec![BENEATH_WIDTH]
                );
                assert_eq!(
                    quad_widths(&overlay_plane, scale_factor),
                    vec![POPOVER_WIDTH, NESTED_MENU_WIDTH, MISPLACED_WIDTH],
                    "overlay draws paint in priority order; a base-plane request nested in one stays with it"
                );
            })
            .unwrap();
    }

    struct NoOverlayContentView;

    impl Render for NoOverlayContentView {
        fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
            div().size_full().child(
                deferred(div().w(px(BENEATH_WIDTH)).h(px(BENEATH_WIDTH)).bg(crate::rgb(0xff0000)))
                    .beneath_native_surfaces(),
            )
        }
    }

    #[gpui::test]
    fn test_a_frame_with_only_base_plane_deferred_draws_has_an_empty_overlay(
        cx: &mut TestAppContext,
    ) {
        let window = cx.open_window(size(px(800.), px(600.)), |_, _| NoOverlayContentView);
        cx.run_until_parked();

        window
            .update(cx, |_, window, _| {
                let scene = &window.rendered_frame.scene;
                assert_eq!(window.rendered_frame.overlay_scene_start, scene.len());
                assert_eq!(
                    quad_widths(scene, window.scale_factor()),
                    vec![BENEATH_WIDTH]
                );
            })
            .unwrap();
    }

    /// The same planes as [`PlanesView`], with every surface taking input.
    struct HitboxPlanesView;

    impl Render for HitboxPlanesView {
        fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
            div()
                .size_full()
                .occlude()
                .child(
                    deferred(
                        div()
                            .w(px(BENEATH_WIDTH))
                            .h(px(BENEATH_WIDTH))
                            .occlude()
                            .child(deferred(
                                anchored().position(point(px(5.), px(5.))).child(
                                    div()
                                        .w(px(NESTED_MENU_WIDTH))
                                        .h(px(NESTED_MENU_WIDTH))
                                        .occlude(),
                                ),
                            )),
                    )
                    .beneath_native_surfaces(),
                )
                .child(deferred(
                    anchored().position(point(px(100.), px(100.))).child(
                        div()
                            .w(px(POPOVER_WIDTH))
                            .h(px(POPOVER_WIDTH))
                            .occlude(),
                    ),
                ))
        }
    }

    #[gpui::test]
    fn test_only_overlay_plane_hitboxes_take_input_over_native_surfaces(
        cx: &mut TestAppContext,
    ) {
        let window = cx.open_window(size(px(800.), px(600.)), |_, _| HitboxPlanesView);
        cx.run_until_parked();

        window
            .update(cx, |_, window, _| {
                let mut widths = window
                    .overlay_hit_regions
                    .iter()
                    .map(|region| region.size.width.0)
                    .collect::<Vec<_>>();
                widths.sort_by(f32::total_cmp);
                assert_eq!(
                    widths,
                    vec![NESTED_MENU_WIDTH, POPOVER_WIDTH],
                    "the root and the draw kept beneath native surfaces stay on the base plane"
                );
                let popover = window
                    .overlay_hit_regions
                    .iter()
                    .find(|region| region.size.width.0 == POPOVER_WIDTH)
                    .copied();
                assert_eq!(popover.map(|region| region.origin), Some(point(px(100.), px(100.))));
            })
            .unwrap();
    }
}
