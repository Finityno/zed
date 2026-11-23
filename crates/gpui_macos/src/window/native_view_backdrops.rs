//! Frosts a native view beneath a backdrop blur that is painted over it.
//!
//! The Metal pass blurs what the scene drew, and a native view is an opening
//! in the scene: where a frosted surface lies over one, the pass has nothing
//! of the view to sample, and the view would show through the surface sharp.
//! A view with a backdrop layer is placed beneath the GPUI view for each such
//! blur. It sits above the native views, and the compositor blurs them into
//! it from its own layer tree, where the opening lets it show.

use super::MacWindowState;
use crate::ns_string;
use gpui::{BackdropBlur, Scene};
use objc2::{
    class, msg_send,
    rc::Retained,
    runtime::{AnyClass, AnyObject, Bool, ClassBuilder, Sel},
    sel,
};
use objc2_app_kit::{NSView, NSWindowOrderingMode};
use objc2_foundation::{NSPoint, NSRect, NSSize};
use parking_lot::Mutex;
use std::{
    mem, ptr,
    sync::{Arc, OnceLock},
};

#[derive(Default)]
pub(super) struct Backdrops {
    views: Vec<Backdrop>,
}

impl Backdrops {
    /// The backdrop views, which are ordered above every native view.
    pub(super) fn views(&self) -> impl Iterator<Item = *const NSView> + '_ {
        self.views
            .iter()
            .map(|backdrop| Retained::as_ptr(&backdrop.view))
    }

    pub(super) fn is_empty(&self) -> bool {
        self.views.is_empty()
    }

    /// Takes every backdrop view out of the window.
    pub(super) fn remove_all(&mut self) {
        for backdrop in self.views.drain(..) {
            backdrop.view.removeFromSuperview();
        }
    }
}

struct Backdrop {
    view: Retained<NSView>,
    /// The filter settings last given to the layer.
    blur_radius: f64,
    saturation: f64,
}

/// Where one backdrop goes, in points in the window's content view, and what
/// it does to what is beneath it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Placement {
    frame: [f64; 4],
    opacity: f64,
    corner_radius: f64,
    blur_radius: f64,
    saturation: f64,
}

/// The backdrops `scene` needs: one for each blur painted over a native view.
pub(super) fn plan(state: &MacWindowState, scene: &Scene) -> Vec<Placement> {
    if scene.native_views.is_empty() || scene.backdrop_blurs.is_empty() {
        return Vec::new();
    }
    let scale_factor = state.scale_factor() as f64;
    // SAFETY: The GPUI view is live for as long as the window state.
    let gpui_frame: NSRect = unsafe { msg_send![state.native_view.as_ptr(), frame] };
    scene
        .backdrop_blurs
        .iter()
        .filter(|blur| is_over_a_native_view(blur, scene))
        .map(|blur| {
            let frosted = blur.bounds.intersect(&blur.content_mask.bounds);
            let to_points = |value: f32| f64::from(value) / scale_factor;
            let height = to_points(frosted.size.height.0);
            Placement {
                // AppKit measures from the bottom left of the enclosing view.
                frame: [
                    gpui_frame.origin.x + to_points(frosted.origin.x.0),
                    gpui_frame.origin.y + gpui_frame.size.height
                        - to_points(frosted.origin.y.0)
                        - height,
                    to_points(frosted.size.width.0),
                    height,
                ],
                opacity: f64::from(blur.opacity.clamp(0., 1.)),
                corner_radius: to_points(blur.corner_radii.top_left.0),
                blur_radius: to_points(blur.blur_radius.0),
                saturation: f64::from(blur.saturation),
            }
        })
        .collect()
}

fn is_over_a_native_view(blur: &BackdropBlur, scene: &Scene) -> bool {
    let frosted = blur.bounds.intersect(&blur.content_mask.bounds);
    scene.native_views.iter().any(|native_view| {
        let opening = native_view
            .bounds
            .intersect(&native_view.content_mask.bounds);
        native_view.order < blur.order && !opening.intersect(&frosted).is_empty()
    })
}

/// Places the planned backdrops. The window state must be unlocked, as it is
/// for the native views' own layout: AppKit calls back into the window as
/// its view hierarchy changes.
pub(super) fn apply(window_state: &Arc<Mutex<MacWindowState>>, placements: Vec<Placement>) {
    let (mut backdrops, gpui_view) = {
        let mut lock = window_state.lock();
        if placements.is_empty() && lock.native_view_backdrops.is_empty() {
            return;
        }
        (
            mem::take(&mut lock.native_view_backdrops.views),
            lock.native_view.as_ptr(),
        )
    };
    while backdrops.len() > placements.len() {
        if let Some(backdrop) = backdrops.pop() {
            backdrop.view.removeFromSuperview();
        }
    }
    for (index, placement) in placements.iter().enumerate() {
        if index == backdrops.len() {
            // SAFETY: The class is an `NSView` subclass, and the GPUI view is
            // live and in the window's content view. Directly beneath the
            // GPUI view is above every native view's container.
            let view = unsafe {
                let view: Retained<NSView> = msg_send![backdrop_class(), new];
                view.setWantsLayer(true);
                let content_view: *mut AnyObject = msg_send![gpui_view, superview];
                let _: () = msg_send![
                    content_view,
                    addSubview: &*view,
                    positioned: NSWindowOrderingMode::Below,
                    relativeTo: gpui_view
                ];
                view
            };
            backdrops.push(Backdrop {
                view,
                blur_radius: -1.,
                saturation: -1.,
            });
        }
        let backdrop = &mut backdrops[index];
        let [x, y, width, height] = placement.frame;
        backdrop.view.setFrame(NSRect::new(
            NSPoint::new(x, y),
            NSSize::new(width.max(0.), height.max(0.)),
        ));
        backdrop.view.setAlphaValue(placement.opacity);
        // SAFETY: The view is live, and its layer, when it has one, is the
        // backdrop layer its class makes, which takes these filters.
        unsafe {
            let layer: *mut AnyObject = msg_send![&*backdrop.view, layer];
            if layer.is_null() {
                continue;
            }
            let _: () = msg_send![layer, setCornerRadius: placement.corner_radius];
            let _: () = msg_send![layer, setMasksToBounds: Bool::YES];
            if backdrop.blur_radius != placement.blur_radius
                || backdrop.saturation != placement.saturation
            {
                backdrop.blur_radius = placement.blur_radius;
                backdrop.saturation = placement.saturation;
                // The render server copies the filters it is given, so
                // changing a value means handing it a new array.
                let gaussian: *mut AnyObject =
                    msg_send![class!(CAFilter), filterWithType: ns_string("gaussianBlur")];
                let radius: *mut AnyObject =
                    msg_send![class!(NSNumber), numberWithDouble: placement.blur_radius];
                let _: () = msg_send![gaussian, setValue: radius, forKey: ns_string("inputRadius")];
                // Clamp at the view's edges rather than fading to
                // transparent, as the Metal pass clamps at its copy's.
                let yes: *mut AnyObject = msg_send![class!(NSNumber), numberWithBool: Bool::YES];
                let _: () =
                    msg_send![gaussian, setValue: yes, forKey: ns_string("inputNormalizeEdges")];
                let saturate: *mut AnyObject =
                    msg_send![class!(CAFilter), filterWithType: ns_string("colorSaturate")];
                let amount: *mut AnyObject =
                    msg_send![class!(NSNumber), numberWithDouble: placement.saturation];
                let _: () = msg_send![saturate, setValue: amount, forKey: ns_string("inputAmount")];
                let filters: *mut AnyObject = msg_send![
                    class!(NSArray),
                    arrayWithObjects: [gaussian, saturate].as_ptr(),
                    count: 2usize
                ];
                let _: () = msg_send![layer, setFilters: filters];
            }
        }
    }
    let mut lock = window_state.lock();
    if lock.native_view_backdrops.is_empty() {
        lock.native_view_backdrops.views = backdrops;
    } else {
        // A scene drawn while these were being placed has placed its own.
        drop(lock);
        for backdrop in backdrops {
            backdrop.view.removeFromSuperview();
        }
    }
}

fn backdrop_class() -> &'static AnyClass {
    static BACKDROP_CLASS: OnceLock<&'static AnyClass> = OnceLock::new();
    BACKDROP_CLASS.get_or_init(|| {
        let name = c"GPUINativeViewBackdropView";
        let Some(mut decl) = ClassBuilder::new(name, class!(NSView)) else {
            // Another copy of GPUI in this process has registered the class.
            return AnyClass::get(name).unwrap_or(class!(NSView));
        };
        // SAFETY: The function signatures match the selectors' Objective-C signatures.
        unsafe {
            decl.add_method(
                sel!(makeBackingLayer),
                make_backing_layer as extern "C" fn(_, Sel) -> *mut AnyObject,
            );
            decl.add_method(
                sel!(hitTest:),
                hit_test as extern "C" fn(_, Sel, NSPoint) -> *mut AnyObject,
            );
        }
        decl.register()
    })
}

// A backdrop layer samples the window's composited layer tree live, so it
// sees a layer-hosted web view. A within-window `NSVisualEffectView` would
// not do: its materials may blur a cached image of the window, which never
// includes one.
extern "C" fn make_backing_layer(_: &AnyObject, _: Sel) -> *mut AnyObject {
    // SAFETY: `layer` returns an autoreleased layer that the view retains as
    // its backing layer, and both setters take the argument types given.
    unsafe {
        let layer: *mut AnyObject = msg_send![class!(CABackdropLayer), layer];
        let _: () = msg_send![layer, setAllowsInPlaceFiltering: Bool::NO];
        // Sampling at half resolution halves the blur's cost and is
        // indistinguishable under a gaussian this wide.
        let _: () = msg_send![layer, setScale: 0.5f64];
        layer
    }
}

/// A backdrop takes no input: what is beneath it does.
extern "C" fn hit_test(_: &AnyObject, _: Sel, _: NSPoint) -> *mut AnyObject {
    ptr::null_mut()
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpui::{
        Bounds, ContentMask, Corners, HitboxId, Hsla, NativeViewId, PaintNativeView, Quad,
        ScaledPixels, point, size,
    };

    fn rect(x: f32, y: f32, width: f32, height: f32) -> Bounds<ScaledPixels> {
        Bounds {
            origin: point(ScaledPixels(x), ScaledPixels(y)),
            size: size(ScaledPixels(width), ScaledPixels(height)),
        }
    }

    fn frosted_surface(scene: &mut Scene, bounds: Bounds<ScaledPixels>) {
        scene.push_layer(bounds);
        scene.insert_backdrop_blur(BackdropBlur {
            order: 0,
            blur_radius: ScaledPixels(8.),
            saturation: 1.,
            opacity: 1.,
            bounds,
            content_mask: ContentMask { bounds },
            corner_radii: Corners::default(),
        });
        scene.insert_primitive(Quad {
            bounds,
            content_mask: ContentMask { bounds },
            ..Default::default()
        });
        scene.pop_layer();
    }

    #[test]
    fn only_a_blur_painted_over_a_native_view_needs_a_backdrop() {
        let view = rect(100., 0., 100., 100.);
        let mut scene = Scene::default();
        // Beneath the view, beside it, and over it.
        frosted_surface(&mut scene, rect(120., 20., 40., 40.));
        scene.insert_native_view(PaintNativeView {
            order: 0,
            id: NativeViewId::for_test(1),
            hitbox_id: HitboxId::placeholder(),
            bounds: view,
            content_mask: ContentMask { bounds: view },
            background: Hsla::default(),
        });
        frosted_surface(&mut scene, rect(0., 0., 50., 50.));
        frosted_surface(&mut scene, rect(80., 60., 60., 30.));
        scene.finish();

        let needed: Vec<_> = scene
            .backdrop_blurs
            .iter()
            .filter(|blur| is_over_a_native_view(blur, &scene))
            .map(|blur| blur.bounds)
            .collect();
        assert_eq!(needed, [rect(80., 60., 60., 30.)]);
    }
}
