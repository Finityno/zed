use cocoa::{
    appkit::{NSAppearanceNameVibrantDark, NSAppearanceNameVibrantLight},
    base::id,
    foundation::NSString,
};
use gpui::WindowAppearance;
use objc::{class, msg_send, sel, sel_impl};
use std::ffi::CStr;

pub(crate) unsafe fn window_appearance_from_native(appearance: id) -> WindowAppearance {
    let name: id = msg_send![appearance, name];
    unsafe {
        if name == NSAppearanceNameVibrantLight {
            WindowAppearance::VibrantLight
        } else if name == NSAppearanceNameVibrantDark {
            WindowAppearance::VibrantDark
        } else if name == NSAppearanceNameAqua {
            WindowAppearance::Light
        } else if name == NSAppearanceNameDarkAqua {
            WindowAppearance::Dark
        } else {
            println!(
                "unknown appearance: {:?}",
                CStr::from_ptr(name.UTF8String())
            );
            WindowAppearance::Light
        }
    }
}

/// Returns the retained `NSAppearance` for the given appearance, for
/// `-[NSWindow setAppearance:]`.
pub(crate) unsafe fn window_appearance_to_native(appearance: WindowAppearance) -> id {
    unsafe {
        let name: id = match appearance {
            WindowAppearance::Light => NSAppearanceNameAqua,
            WindowAppearance::Dark => NSAppearanceNameDarkAqua,
            WindowAppearance::VibrantLight => NSAppearanceNameVibrantLight,
            WindowAppearance::VibrantDark => NSAppearanceNameVibrantDark,
        };
        msg_send![class!(NSAppearance), appearanceNamed: name]
    }
}

#[link(name = "AppKit", kind = "framework")]
unsafe extern "C" {
    pub static NSAppearanceNameAqua: id;
    pub static NSAppearanceNameDarkAqua: id;
}
