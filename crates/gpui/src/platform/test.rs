mod dispatcher;
mod display;
mod platform;
mod window;

pub use dispatcher::*;
pub(crate) use display::*;
pub(crate) use platform::*;

#[cfg(any(test, feature = "test-support"))]
pub use platform::{TestScreenCaptureSource, TestScreenCaptureStream};
pub use window::TestWindow;
