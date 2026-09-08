#[cfg(feature = "native-probes")]
pub mod native_probe;
use std::sync::{Arc, atomic::{AtomicUsize, Ordering}};
use gpui::{TextAllocationAdmission, TextAllocationClass, TextAllocationError, TextAllocationToken, TextConstructionEvent, TextShapingInput};

#[derive(Debug, Default)]
pub struct Policy {
    pub live: Arc<AtomicUsize>,
    pub glyph_live: Arc<AtomicUsize>,
    pub source_constructions: AtomicUsize,
    pub native_entries: AtomicUsize,
    pub glyph_constructions: AtomicUsize,
    pub deny_source: bool,
    pub calibration_only_unbounded_native: bool,
    #[cfg(feature = "native-probes")]
    pub probe: Option<Arc<native_probe::ProbeState>>,
}

#[derive(Debug)]
struct Token {
    live: Arc<AtomicUsize>, glyph_live: Option<Arc<AtomicUsize>>, bytes: usize,
    #[cfg(feature = "native-probes")]
    native_probe: Option<Arc<native_probe::ProbeState>>,
}
impl Drop for Token {
    fn drop(&mut self) {
        #[cfg(feature = "native-probes")]
        if let Some(probe) = &self.native_probe { probe.marker("native-temporaries-released"); }
        self.live.fetch_sub(self.bytes, Ordering::SeqCst);
        if let Some(glyph_live) = &self.glyph_live { glyph_live.fetch_sub(self.bytes, Ordering::SeqCst); }
    }
}
impl TextAllocationToken for Token {
    fn resize(&mut self, bytes: usize) -> Result<(), TextAllocationError> {
        if bytes >= self.bytes { self.live.fetch_add(bytes - self.bytes, Ordering::SeqCst); }
        else { self.live.fetch_sub(self.bytes - bytes, Ordering::SeqCst); }
        if let Some(glyph_live) = &self.glyph_live {
            if bytes >= self.bytes { glyph_live.fetch_add(bytes - self.bytes, Ordering::SeqCst); }
            else { glyph_live.fetch_sub(self.bytes - bytes, Ordering::SeqCst); }
        }
        self.bytes = bytes;
        Ok(())
    }
}
impl TextAllocationAdmission for Policy {
    fn reserve(&self, class: TextAllocationClass, bytes: usize) -> Result<Box<dyn TextAllocationToken>, TextAllocationError> {
        if self.deny_source && class == TextAllocationClass::Source { return Err(TextAllocationError::Denied); }
        self.live.fetch_add(bytes, Ordering::SeqCst);
        let glyph_live = if class == TextAllocationClass::Glyphs {
            self.glyph_live.fetch_add(bytes, Ordering::SeqCst);
            Some(Arc::clone(&self.glyph_live))
        } else { None };
        Ok(Box::new(Token { live: Arc::clone(&self.live), glyph_live, bytes, #[cfg(feature = "native-probes")] native_probe: None }))
    }
    fn native_scratch(&self, input: TextShapingInput<'_>) -> Result<Box<dyn TextAllocationToken>, TextAllocationError> {
        assert_eq!(input.text.len(), input.utf8_bytes);
        assert_eq!(input.text.encode_utf16().count(), input.utf16_units);
        if !self.calibration_only_unbounded_native { return Err(TextAllocationError::MissingNativeMeasurement); }
        eprintln!("phase=uncalibrated-native-entry bytes={} utf16={} backend={}", input.utf8_bytes, input.utf16_units, input.backend);
        Ok(Box::new(Token { live: Arc::clone(&self.live), glyph_live: None, bytes: 0, #[cfg(feature = "native-probes")] native_probe: self.probe.clone() }))
    }
    fn native_unavailable_font(&self, name: &str) {
        #[cfg(feature = "native-probes")]
        if let Some(probe) = &self.probe { probe.record_font(name); }
        println!("UNAVAILABLE_NATIVE_FONT {:?}", name);
    }
    fn native_output(&self, name: &str, glyph_count: usize, glyph_capacity: usize, run_capacity: usize) {
        #[cfg(feature = "native-probes")]
        if let Some(probe) = &self.probe { probe.record_font(name); }
        println!("NATIVE_OUTPUT {:?} {glyph_count} {glyph_capacity} {run_capacity}", name);
    }
    fn construction(&self, event: TextConstructionEvent) {
        match event {
            TextConstructionEvent::Source => { self.source_constructions.fetch_add(1, Ordering::SeqCst); }
            TextConstructionEvent::NativeEntry => {
                self.native_entries.fetch_add(1, Ordering::SeqCst);
                #[cfg(feature = "native-probes")]
                if let Some(probe) = &self.probe { probe.marker("native-entry"); }
            }
            TextConstructionEvent::RunBuffer | TextConstructionEvent::GlyphBuffer => {
                if matches!(event, TextConstructionEvent::GlyphBuffer) { self.glyph_constructions.fetch_add(1, Ordering::SeqCst); }
                assert!(self.glyph_live.load(Ordering::SeqCst) > 0, "actual glyph allocation preceded admission");
            }
        }
        eprintln!("construction={event:?} charged={}", self.live.load(Ordering::SeqCst));
    }
}
