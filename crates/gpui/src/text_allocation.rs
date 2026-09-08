use std::{fmt, sync::Arc};

/// The allocation family charged by an opted-in text renderer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TextAllocationClass {
    /// Immutable source backing shared by the element and cache.
    Source,
    /// Rust glyph and font-run buffers retained by a line layout.
    Glyphs,
    /// Wrapped-layout and cache-key storage.
    Layout,
    /// Element-owned decoration and text-run buffers.
    Element,
    /// Platform shaper workspace, including opaque native allocations.
    NativeScratch,
}

/// The exact input covered by a platform scratch measurement.
#[derive(Clone, Copy, Debug)]
pub struct TextShapingInput<'a> {
    /// Exact borrowed input; length alone cannot identify a native measurement.
    pub text: &'a str,
    /// Backend identifier; measurements are not portable across backends.
    pub backend: &'static str,
    /// Length of the immutable physical line in UTF-8 bytes.
    pub utf8_bytes: usize,
    /// Length of the immutable physical line in UTF-16 code units.
    pub utf16_units: usize,
    /// Exact input font identities and run lengths used for the measurement.
    pub font_runs: &'a [crate::FontRun],
    /// Exact point size used for shaping.
    pub font_size: crate::Pixels,
}

/// A failed admission never permits the opted-in path to shape unaccounted text.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TextAllocationError {
    /// The caller could not reserve the requested allocation.
    Denied,
    /// An allocation size could not be represented.
    Overflow,
    /// This backend has no admitted shaping implementation.
    UnsupportedBackend,
    /// The opted-in route does not implement the requested transform.
    UnsupportedTransform,
    /// No measured scratch allowance covers this platform and input.
    MissingNativeMeasurement,
    /// A backend tried to publish a reservation for a different source.
    MismatchedSource,
}

impl fmt::Display for TextAllocationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Denied => "Text allocation admission was denied",
            Self::Overflow => "Text allocation size overflowed",
            Self::UnsupportedBackend => "The text backend does not support admitted shaping",
            Self::UnsupportedTransform => "This text transform does not support admitted shaping",
            Self::MissingNativeMeasurement => "No native shaping measurement covers this input",
            Self::MismatchedSource => "Text allocation reservation belongs to another source",
        })
    }
}
impl std::error::Error for TextAllocationError {}

/// A uniquely owned reservation token transferred into a published owner.
///
/// Resizing must reserve growth before releasing any existing charge. A failed
/// resize leaves the original reservation intact. Shrinking occurs only after
/// the corresponding temporary buffers have been destroyed.
pub trait TextAllocationToken: fmt::Debug + Send + Sync {
    /// Reconcile the reservation with the live allocation's measured capacity.
    fn resize(&mut self, bytes: usize) -> Result<(), TextAllocationError>;
}

/// A real construction boundary observed only by the isolated validation build.
#[cfg(feature = "text-allocation-validation")]
#[derive(Clone, Copy, Debug)]
pub enum TextConstructionEvent {
    /// About to allocate the immutable source backing.
    Source,
    /// About to enter CoreText after scratch admission.
    NativeEntry,
    /// About to allocate the output run vector.
    RunBuffer,
    /// About to grow the current output glyph vector.
    GlyphBuffer,
}

/// Application policy for the optional, fallible text-rendering path.
pub trait TextAllocationAdmission: fmt::Debug + Send + Sync {
    /// Observe the actual construction site; production has no observer hook.
    #[cfg(feature = "text-allocation-validation")]
    fn construction(&self, _event: TextConstructionEvent) {}

    /// Observe actual native output without retaining source or glyph buffers.
    #[cfg(feature = "text-allocation-validation")]
    fn native_output(&self, _postscript_name: &str, _glyph_count: usize, _glyph_capacity: usize, _run_capacity: usize) {}

    /// Record an unsupported native fallback without registering or retaining it.
    #[cfg(feature = "text-allocation-validation")]
    fn native_unavailable_font(&self, _postscript_name: &str) {}


    /// Reserve before allocating. The returned token owns the charge.
    fn reserve(&self, class: TextAllocationClass, bytes: usize)
    -> Result<Box<dyn TextAllocationToken>, TextAllocationError>;

    /// Reserve a measured native workspace floor before entering the shaper.
    ///
    /// A mock implementation proves lifetime transfer only. Production callers
    /// must reject inputs outside their measured backend/font/input envelope.
    fn native_scratch(&self, input: TextShapingInput<'_>)
    -> Result<Box<dyn TextAllocationToken>, TextAllocationError>;
}

/// A unique construction reservation. Publishing consumes it; published
/// allocations never expose this mutable admission capability.
///
/// ```compile_fail
/// fn cannot_alias(reservation: gpui::TextAllocationReservation) {
///     let retained = reservation.clone();
/// }
/// ```
///
/// ```compile_fail
/// fn cannot_resize_after_publish(source: gpui::AdmittedTextSource,
///     layout: gpui::LineLayout, mut reservation: gpui::TextAllocationReservation) {
///     let published = gpui::AdmittedLineLayout::from_native(source, layout, reservation);
///     reservation.reconcile(0);
/// }
/// ```
#[derive(Debug)]
pub struct TextAllocationReservation {
    token: Box<dyn TextAllocationToken>,
    admission: Arc<dyn TextAllocationAdmission>,
    source: Option<AdmittedTextSource>,
    bytes: usize,
}

impl TextAllocationReservation {
    pub(crate) fn reserve(admission: &Arc<dyn TextAllocationAdmission>, class: TextAllocationClass, bytes: usize)
    -> Result<Self, TextAllocationError> {
        Ok(Self { token: admission.reserve(class, bytes)?, admission: Arc::clone(admission), source: None, bytes })
    }

    /// Reserve a backend's glyph buffers for this exact immutable source.
    pub fn for_line(source: &AdmittedTextSource, bytes: usize) -> Result<Self, TextAllocationError> {
        let mut reservation = Self::reserve(source.admission(), TextAllocationClass::Glyphs, bytes)?;
        reservation.source = Some(source.clone());
        Ok(reservation)
    }

    /// Reconcile capacity while the unique builder still owns the allocation.
    pub fn reconcile(&mut self, bytes: usize) -> Result<(), TextAllocationError> {
        self.token.resize(bytes)?;
        self.bytes = bytes;
        Ok(())
    }

    /// Grow a backend buffer only when the reservation covers the other live
    /// output buffers and both old/new allocations during reallocation.
    /// The backend supplies the exact capacity bytes of its other live buffers.
    pub fn grow_glyph_buffer(
        &self, glyphs: &mut Vec<crate::ShapedGlyph>, additional: usize, other_live_bytes: usize,
    ) -> Result<(), TextAllocationError> {
        let required = glyphs.len().checked_add(additional).ok_or(TextAllocationError::Overflow)?;
        if required <= glyphs.capacity() { return Ok(()); }
        let peak = glyphs.capacity().checked_add(required)
            .and_then(|count| count.checked_mul(std::mem::size_of::<crate::ShapedGlyph>()))
            .and_then(|bytes| bytes.checked_add(other_live_bytes)).ok_or(TextAllocationError::Overflow)?;
        if self.source.is_none() || peak > self.bytes { return Err(TextAllocationError::Denied); }
        #[cfg(feature = "text-allocation-validation")]
        self.admission.construction(TextConstructionEvent::GlyphBuffer);
        glyphs.try_reserve_exact(additional).map_err(|_| TextAllocationError::Denied)
    }

    pub(crate) fn publish(self) -> TextAllocationLease {
        TextAllocationLease { _token: self.token, admission: self.admission }
    }

    fn certifies(&self, source: &AdmittedTextSource) -> bool {
        Arc::ptr_eq(&self.admission, source.admission())
            && self.source.as_ref().is_some_and(|bound| Arc::ptr_eq(&bound.0, &source.0))
    }
}

#[derive(Debug)]
pub(crate) struct TextAllocationLease {
    _token: Box<dyn TextAllocationToken>,
    admission: Arc<dyn TextAllocationAdmission>,
}

impl TextAllocationLease {
    fn admission(&self) -> &Arc<dyn TextAllocationAdmission> { &self.admission }
}

/// Whether platform allocations require admission or remain separately observed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TextPlatformCoverage {
    /// The backend must obtain a real native scratch reservation before entry.
    StrictAdmission,
    /// Rust allocations are admitted; platform/font allocations have no claimed bound.
    ObservedUnbounded,
}

/// A physical line whose immutable backing was reserved before allocation.
#[derive(Clone, Debug)]
pub struct AdmittedTextSource(Arc<AdmittedTextSourceInner>);

#[derive(Debug)]
struct AdmittedTextSourceInner {
    platform_coverage: TextPlatformCoverage,
    text: crate::SharedString,
    allocation: TextAllocationLease,
}

impl AdmittedTextSource {
    /// Construct one source allocation. Clones share its receipt and backing.
    pub fn new(text: &str, admission: Arc<dyn TextAllocationAdmission>) -> Result<Self, TextAllocationError> {
        Self::with_platform_coverage(text, admission, TextPlatformCoverage::StrictAdmission)
    }

    /// Admit Rust storage while keeping native and font allocations explicitly unbounded.
    pub fn new_tracked_platform(text: &str, admission: Arc<dyn TextAllocationAdmission>) -> Result<Self, TextAllocationError> {
        Self::with_platform_coverage(text, admission, TextPlatformCoverage::ObservedUnbounded)
    }

    fn with_platform_coverage(text: &str, admission: Arc<dyn TextAllocationAdmission>, platform_coverage: TextPlatformCoverage) -> Result<Self, TextAllocationError> {
        if text.contains('\n') { return Err(TextAllocationError::UnsupportedTransform); }
        u32::try_from(text.len()).map_err(|_| TextAllocationError::Overflow)?;
        let metadata = std::mem::size_of::<AdmittedTextSourceInner>().checked_add(2 * std::mem::size_of::<usize>())
            .ok_or(TextAllocationError::Overflow)?;
        let reserved = text.len().checked_add(2 * std::mem::size_of::<usize>()).and_then(|bytes| bytes.checked_add(metadata))
            .ok_or(TextAllocationError::Overflow)?;
        let mut allocation = TextAllocationReservation::reserve(&admission, TextAllocationClass::Source, reserved)?;
        #[cfg(feature = "text-allocation-validation")]
        admission.construction(TextConstructionEvent::Source);
        let text = crate::SharedString::new(text);
        allocation.reconcile(metadata.checked_add(text.heap_allocation_bytes()).ok_or(TextAllocationError::Overflow)?)?;
        Ok(Self(Arc::new(AdmittedTextSourceInner { platform_coverage, text, allocation: allocation.publish() })))
    }

    /// Borrow the exact physical line without exporting an uncharged string alias.
    pub fn as_str(&self) -> &str { &self.0.text }

    /// This classification follows the source through layout and cache ownership.
    pub fn platform_coverage(&self) -> TextPlatformCoverage { self.0.platform_coverage }

    pub(crate) fn same_allocation(&self, other: &Self) -> bool { Arc::ptr_eq(&self.0, &other.0) }

    pub(crate) fn shared(&self) -> &crate::SharedString { &self.0.text }
    /// Admission policy retained by this immutable source.
    pub fn admission(&self) -> &Arc<dyn TextAllocationAdmission> { self.0.allocation.admission() }
}

#[cfg(any(test, feature = "text-allocation-validation"))]
pub(crate) mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[derive(Debug, Default)]
    pub(crate) struct Counter(pub(crate) Arc<AtomicUsize>);
    #[derive(Debug)]
    struct Token { counter: Arc<AtomicUsize>, bytes: parking_lot::Mutex<usize> }
    impl Drop for Token {
        fn drop(&mut self) { self.counter.fetch_sub(*self.bytes.get_mut(), Ordering::Relaxed); }
    }
    impl TextAllocationToken for Token {
        fn resize(&mut self, bytes: usize) -> Result<(), TextAllocationError> {
            let mut current = self.bytes.lock();
            if bytes > *current { self.counter.fetch_add(bytes - *current, Ordering::Relaxed); }
            else { self.counter.fetch_sub(*current - bytes, Ordering::Relaxed); }
            *current = bytes;
            Ok(())
        }
    }
    impl TextAllocationAdmission for Counter {
        fn reserve(&self, _: TextAllocationClass, bytes: usize) -> Result<Box<dyn TextAllocationToken>, TextAllocationError> {
            self.0.fetch_add(bytes, Ordering::Relaxed);
            Ok(Box::new(Token { counter: Arc::clone(&self.0), bytes: parking_lot::Mutex::new(bytes) }))
        }
        fn native_scratch(&self, _: TextShapingInput<'_>) -> Result<Box<dyn TextAllocationToken>, TextAllocationError> {
            Err(TextAllocationError::MissingNativeMeasurement)
        }
    }

    #[cfg_attr(test, test)]
    pub(crate) fn admitted_single_line_alignment_preserves_utf8_geometry() {
        use crate::{point, px, size, Bounds, FontId, GlyphId, LineLayout, ShapedGlyph, ShapedRun, TextAlign};
        for width in [20.0, 80.0] {
            for align in [TextAlign::Left, TextAlign::Center, TextAlign::Right] {
                let admission = Arc::new(Counter::default());
                let source = AdmittedTextSource::new("aβc", admission.clone()).expect("source");
                let allocation = TextAllocationReservation::for_line(&source, 1024).expect("glyph reservation");
                let raw = LineLayout {
                    width: px(30.0), len: 4,
                    runs: vec![ShapedRun { font_id: FontId(0), glyphs: [0, 1, 3].into_iter().enumerate().map(|(index, byte)| ShapedGlyph {
                        id: GlyphId(index as u32), position: point(px(index as f32 * 10.0), px(0.0)), index: byte, is_emoji: false,
                    }).collect() }], ..Default::default()
                };
                let line = Arc::new(AdmittedLineLayout::from_native(source, raw, allocation).expect("line"));
                let element = TextAllocationReservation::reserve(line.source().admission(), TextAllocationClass::Element,
                    std::mem::size_of::<AdmittedTextLayoutInner>() + 2 * std::mem::size_of::<usize>()).expect("geometry reservation");
                let bounds = Bounds::new(point(px(10.0), px(30.0)), size(px(width), px(20.0)));
                let layout = AdmittedTextLayout(std::rc::Rc::new(AdmittedTextLayoutInner {
                    text_align: align, line, bounds: std::cell::Cell::new(Some(bounds)), _allocation: element.publish(),
                }));
                let offset = match align { TextAlign::Left => 0.0, TextAlign::Center => (width - 30.0) / 2.0, TextAlign::Right => width - 30.0 };
                let expected_origin = point(px(10.0 + offset), px(30.0));
                assert_eq!(admitted_line_origin(bounds, px(30.0), align), expected_origin);
                for (byte, x) in [(0, 0.0), (1, 10.0), (3, 20.0), (4, 30.0)] {
                    let point = point(expected_origin.x + px(x), expected_origin.y);
                    assert_eq!(layout.position_for_index(byte), Some(point));
                    assert_eq!(layout.closest_index_for_position(point), Some(byte));
                }
                assert_eq!(layout.position_for_index(2), None);
                drop(layout);
                assert_eq!(admission.0.load(Ordering::Relaxed), 0);
            }
        }
    }

    #[cfg_attr(test, test)]
    pub(crate) fn admitted_text_source_clones_share_backing_and_charge() {
        let admission = Arc::new(Counter::default());
        let source = AdmittedTextSource::new(&"x".repeat(4096), admission.clone()).expect("source admission");
        let charged = admission.0.load(Ordering::Relaxed);
        let alias = source.clone();
        assert_eq!(source.as_str().as_ptr(), alias.as_str().as_ptr());
        assert_eq!(admission.0.load(Ordering::Relaxed), charged);
        drop(source);
        assert_eq!(admission.0.load(Ordering::Relaxed), charged);
        drop(alias);
        assert_eq!(admission.0.load(Ordering::Relaxed), 0);
    }

    #[cfg_attr(test, test)]
    pub(crate) fn admitted_line_rejects_another_source_under_the_same_policy() {
        let admission = Arc::new(Counter::default());
        let expected = AdmittedTextSource::new("first", admission.clone()).expect("expected source");
        let other = AdmittedTextSource::new("other", admission.clone()).expect("other source");
        let raw = crate::LineLayout { len: 5, ..Default::default() };
        let reservation = TextAllocationReservation::for_line(&expected,
            AdmittedLineLayout::allocation_bytes(&raw).expect("capacity")).expect("reservation");
        assert!(matches!(AdmittedLineLayout::from_native(other, raw, reservation), Err(TextAllocationError::MismatchedSource)));
        drop(expected);
        assert_eq!(admission.0.load(Ordering::Relaxed), 0);
    }

    #[cfg_attr(test, test)]
    pub(crate) fn admitted_line_published_owner_has_no_builder_alias() {
        let admission = Arc::new(Counter::default());
        let source = AdmittedTextSource::new("owned", admission.clone()).expect("source");
        let raw = crate::LineLayout { len: 5, ..Default::default() };
        let reservation = TextAllocationReservation::for_line(&source,
            AdmittedLineLayout::allocation_bytes(&raw).expect("capacity")).expect("reservation");
        let layout = Arc::new(AdmittedLineLayout::from_native(source, raw, reservation).expect("publish"));
        let charge = admission.0.load(Ordering::Relaxed);
        let alias = Arc::clone(&layout);
        drop(layout);
        assert_eq!(admission.0.load(Ordering::Relaxed), charge);
        drop(alias);
        assert_eq!(admission.0.load(Ordering::Relaxed), 0);
    }

    #[cfg_attr(test, test)]
    pub(crate) fn admitted_text_source_rejects_multiline_before_allocation() {
        let admission = Arc::new(Counter::default());
        assert!(matches!(AdmittedTextSource::new("a\nb", admission.clone()), Err(TextAllocationError::UnsupportedTransform)));
        assert_eq!(admission.0.load(Ordering::Relaxed), 0);
    }
}

/// A shaped physical line that cannot expose an unaccounted raw-layout alias.
#[derive(Debug)]
pub struct AdmittedLineLayout {
    layout: Arc<crate::LineLayout>,
    source: AdmittedTextSource,
    _allocation: TextAllocationLease,
}

impl AdmittedLineLayout {
    /// Complete a backend's pre-admitted glyph allocation after native scratch
    /// has finished producing the retained Rust buffers.
    pub fn from_native(
        source: AdmittedTextSource, layout: crate::LineLayout, mut allocation: TextAllocationReservation,
    ) -> Result<Self, TextAllocationError> {
        if !allocation.certifies(&source) || layout.len != source.as_str().len() {
            drop(layout);
            drop(allocation);
            return Err(TextAllocationError::MismatchedSource);
        }
        let reconciled = Self::allocation_bytes(&layout).and_then(|bytes| allocation.reconcile(bytes));
        if let Err(error) = reconciled {
            drop(layout);
            drop(allocation);
            return Err(error);
        }
        Ok(Self { layout: Arc::new(layout), source, _allocation: allocation.publish() })
    }

    /// Bytes retained by the glyph vectors and the two shared layout objects.
    pub fn allocation_bytes(layout: &crate::LineLayout) -> Result<usize, TextAllocationError> {
        let mut bytes = std::mem::size_of::<Self>().checked_add(std::mem::size_of::<crate::LineLayout>())
            .and_then(|bytes| bytes.checked_add(4 * std::mem::size_of::<usize>()))
            .and_then(|bytes| bytes.checked_add(layout.runs.capacity().checked_mul(std::mem::size_of::<crate::ShapedRun>())?))
            .ok_or(TextAllocationError::Overflow)?;
        for run in &layout.runs {
            bytes = bytes.checked_add(run.glyphs.capacity().checked_mul(std::mem::size_of::<crate::ShapedGlyph>())
                .ok_or(TextAllocationError::Overflow)?).ok_or(TextAllocationError::Overflow)?;
        }
        Ok(bytes)
    }

    /// Width of the full physical line.
    pub fn width(&self) -> crate::Pixels { self.layout.width }
    /// Exact source length in UTF-8 bytes.
    pub fn len(&self) -> usize { self.layout.len }
    /// Whether the source is empty.
    pub fn is_empty(&self) -> bool { self.len() == 0 }
    /// Closest source boundary at a horizontal position.
    pub fn closest_index_for_x(&self, x: crate::Pixels) -> usize { self.layout.closest_index_for_x(x) }
    /// Horizontal position of a source boundary.
    pub fn x_for_index(&self, index: usize) -> crate::Pixels { self.layout.x_for_index(index) }

    pub(crate) fn source(&self) -> &AdmittedTextSource { &self.source }

    pub(crate) fn paint(
        &self, bounds: crate::Bounds<crate::Pixels>, line_height: crate::Pixels,
        style: &AdmittedTextStyle, window: &mut crate::Window, cx: &mut crate::App,
    ) -> crate::Result<()> {
        let len = u32::try_from(self.len()).map_err(|_| TextAllocationError::Overflow)?;
        let line = crate::ShapedLine {
            layout: Arc::clone(&self.layout), text: self.source.shared().clone(),
            decoration_runs: smallvec::smallvec![crate::DecorationRun {
                len, color: style.color, background_color: style.background_color,
                underline: style.underline, strikethrough: style.strikethrough,
            }],
        };
        let origin = admitted_line_origin(bounds, self.width(), style.text_align);
        line.paint_background(origin, line_height, crate::TextAlign::Left, None, window, cx)?;
        line.paint(origin, line_height, crate::TextAlign::Left, None, window, cx)
    }

    /// Copying a subrange requires its own admitted source and glyph buffers.
    /// The initial owned route intentionally has no infallible split operation.
    pub fn try_split_at(&self, _byte_index: usize) -> Result<(Self, Self), TextAllocationError> {
        Err(TextAllocationError::UnsupportedTransform)
    }
}


/// Explicit single-line styling keeps inherited wrapping and truncation out of
/// the admitted route rather than silently applying unowned transforms.
#[derive(Clone, Copy, Debug)]
pub struct AdmittedTextStyle {
    /// An already registered base font. Tracked platform shaping may register fallback fonts.
    pub font_id: crate::FontId,
    /// Shaping size in logical pixels.
    pub font_size: crate::Pixels,
    /// Height reserved for the physical line.
    pub line_height: crate::Pixels,
    /// Foreground color.
    pub color: crate::Hsla,
    /// Optional background color.
    pub background_color: Option<crate::Hsla>,
    /// Optional underline.
    pub underline: Option<crate::UnderlineStyle>,
    /// Optional strikethrough.
    pub strikethrough: Option<crate::StrikethroughStyle>,
    /// Alignment within the shaped line.
    pub text_align: crate::TextAlign,
}

/// Geometry access that never exports a raw glyph vector or text alias.
#[derive(Clone, Debug)]
pub struct AdmittedTextLayout(std::rc::Rc<AdmittedTextLayoutInner>);

#[derive(Debug)]
struct AdmittedTextLayoutInner {
    text_align: crate::TextAlign,
    line: Arc<AdmittedLineLayout>,
    bounds: std::cell::Cell<Option<crate::Bounds<crate::Pixels>>>,
    _allocation: TextAllocationLease,
}

fn admitted_line_origin(bounds: crate::Bounds<crate::Pixels>, line_width: crate::Pixels, align: crate::TextAlign) -> crate::Point<crate::Pixels> {
    let offset = match align {
        crate::TextAlign::Left => crate::Pixels::ZERO,
        crate::TextAlign::Center => (bounds.size.width - line_width) / 2.0,
        crate::TextAlign::Right => bounds.size.width - line_width,
    };
    crate::point(bounds.origin.x + offset, bounds.origin.y)
}

impl AdmittedTextLayout {
    /// Native coverage of the actual cached layout used by this geometry.
    pub fn platform_coverage(&self) -> TextPlatformCoverage { self.0.line.source().platform_coverage() }

    /// Bounds from the most recent prepaint.
    pub fn bounds(&self) -> Option<crate::Bounds<crate::Pixels>> { self.0.bounds.get() }

    /// Closest source byte boundary at a point in window coordinates.
    pub fn closest_index_for_position(&self, position: crate::Point<crate::Pixels>) -> Option<usize> {
        self.bounds().map(|bounds| {
            let origin = admitted_line_origin(bounds, self.0.line.width(), self.0.text_align);
            self.0.line.closest_index_for_x(position.x - origin.x)
        })
    }

    /// Window position for an existing UTF-8 boundary.
    pub fn position_for_index(&self, index: usize) -> Option<crate::Point<crate::Pixels>> {
        if !self.0.line.source().as_str().is_char_boundary(index) { return None; }
        self.bounds().map(|bounds| {
            let origin = admitted_line_origin(bounds, self.0.line.width(), self.0.text_align);
            crate::point(origin.x + self.0.line.x_for_index(index), origin.y)
        })
    }
}

/// A fallibly shaped physical line with no unowned layout conversion.
pub struct AdmittedStyledText {
    layout: AdmittedTextLayout,
    style: AdmittedTextStyle,
}

impl crate::StyledText {
    /// Admit and shape before creating the element, so failure is returned to
    /// the caller instead of becoming an empty result during measurement.
    pub fn try_new_admitted(
        text: &str, admission: Arc<dyn TextAllocationAdmission>, style: AdmittedTextStyle,
        window: &crate::Window,
    ) -> Result<AdmittedStyledText, TextAllocationError> {
        Self::from_admitted_source(AdmittedTextSource::new(text, admission)?, style, window)
    }

    /// Preserve owned Rust allocations without claiming native scratch admission.
    pub fn try_new_tracked_platform(
        text: &str, admission: Arc<dyn TextAllocationAdmission>, style: AdmittedTextStyle,
        window: &crate::Window,
    ) -> Result<AdmittedStyledText, TextAllocationError> {
        Self::from_admitted_source(AdmittedTextSource::new_tracked_platform(text, admission)?, style, window)
    }

    fn from_admitted_source(source: AdmittedTextSource, style: AdmittedTextStyle, window: &crate::Window) -> Result<AdmittedStyledText, TextAllocationError> {
        let bytes = std::mem::size_of::<AdmittedTextLayoutInner>()
            .checked_add(2 * std::mem::size_of::<usize>()).ok_or(TextAllocationError::Overflow)?;
        let allocation = TextAllocationReservation::reserve(source.admission(), TextAllocationClass::Element, bytes)?;
        let line = window.text_system().shape_line_admitted(source, style.font_size, style.font_id)?;
        let layout = AdmittedTextLayout(std::rc::Rc::new(AdmittedTextLayoutInner {
            text_align: style.text_align, line, bounds: std::cell::Cell::new(None), _allocation: allocation.publish(),
        }));
        Ok(AdmittedStyledText { layout, style })
    }
}

impl AdmittedStyledText {
    /// A shared geometry handle retaining the actual source and glyph owners.
    pub fn layout(&self) -> &AdmittedTextLayout { &self.layout }
}

impl crate::IntoElement for AdmittedStyledText {
    type Element = Self;
    fn into_element(self) -> Self { self }
}

impl crate::Element for AdmittedStyledText {
    type RequestLayoutState = ();
    type PrepaintState = ();

    fn id(&self) -> Option<crate::ElementId> { None }
    fn source_location(&self) -> Option<&'static core::panic::Location<'static>> { None }

    fn request_layout(
        &mut self, _: Option<&crate::GlobalElementId>, _: Option<&crate::InspectorElementId>,
        window: &mut crate::Window, _: &mut crate::App,
    ) -> (crate::LayoutId, ()) {
        let layout = self.layout.clone();
        let line_height = self.style.line_height;
        let id = window.request_measured_layout(Default::default(), move |known, _, _, _| {
            crate::size(known.width.unwrap_or(layout.0.line.width()), known.height.unwrap_or(line_height))
        });
        (id, ())
    }

    fn prepaint(
        &mut self, _: Option<&crate::GlobalElementId>, _: Option<&crate::InspectorElementId>,
        bounds: crate::Bounds<crate::Pixels>, _: &mut (), _: &mut crate::Window, _: &mut crate::App,
    ) {
        self.layout.0.bounds.set(Some(bounds));
    }

    fn paint(
        &mut self, _: Option<&crate::GlobalElementId>, _: Option<&crate::InspectorElementId>,
        bounds: crate::Bounds<crate::Pixels>, _: &mut (), _: &mut (),
        window: &mut crate::Window, cx: &mut crate::App,
    ) {
        use gpui_util::ResultExt;
        self.layout.0.line.paint(bounds, self.style.line_height, &self.style, window, cx).log_err();
    }
}


/// Run only admitted-source/cache ownership checks against the actual code.
/// This opt-in entry point does not calibrate native allocation peaks.
#[cfg(feature = "text-allocation-validation")]
pub fn validate_admitted_text_ownership() {
    tests::admitted_text_source_clones_share_backing_and_charge();
    tests::admitted_text_source_rejects_multiline_before_allocation();
    tests::admitted_line_rejects_another_source_under_the_same_policy();
    tests::admitted_line_published_owner_has_no_builder_alias();
    tests::admitted_single_line_alignment_preserves_utf8_geometry();
    crate::text_system::validate_admitted_cache_ownership();
}
