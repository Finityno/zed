use std::{io::{self, Write}, sync::{Arc, atomic::Ordering}};
use admitted_text_validation::{Policy, native_probe::{self, Pool, ProbeState}};

fn checkpoint(ordinal: usize, name: &str, policy: &Policy, probe: &ProbeState) -> Result<(), Box<dyn std::error::Error>> {
    probe.marker(name);
    let zone = native_probe::zone();
    println!("PHASE {ordinal} {name} pid={} rust_live={} glyph_live={} zone_blocks={} zone_live={} zone_high={} zone_allocated={}",
        std::process::id(), policy.live.load(Ordering::SeqCst), policy.glyph_live.load(Ordering::SeqCst),
        zone.blocks, zone.live, zone.high, zone.allocated);
    io::stdout().flush()?;
    let mut line = String::new();
    io::stdin().read_line(&mut line)?;
    if line != format!("CONTINUE {ordinal}\n") { return Err("phase handshake mismatch or EOF".into()); }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    if arguments != ["--measure-native-unbounded", "--hold-for-tools", "--case", "ascii4096"] {
        return Err("only the reviewed uncalibrated ascii4096 child protocol is supported".into());
    }
    let probe = Arc::new(ProbeState::default());
    let policy = Arc::new(Policy { calibration_only_unbounded_native: true, probe: Some(probe.clone()), ..Default::default() });
    let text = "x".repeat(4096);
    println!("CASE ascii4096 utf8=4096 utf16=4096 family=Menlo size=11.5 native_allowance=UNMEASURED");
    checkpoint(0, "before-fonts", &policy, &probe)?;
    {
        let pool = Pool::new()?;
        native_probe::sentinels()?;
        drop(pool);
    }
    checkpoint(1, "sentinels-drained", &policy, &probe)?;
    let setup_pool = Pool::new()?;
    let backend = gpui_macos::admitted_text_validation_backend();
    let font_id = backend.font_id(&gpui::font("Menlo"))?;
    drop(setup_pool);
    checkpoint(2, "registered-font-baseline", &policy, &probe)?;
    let run = gpui::FontRun { font_id, len: text.len() };
    let denied = Arc::new(Policy::default());
    let source = gpui::AdmittedTextSource::new(&text, denied.clone())?;
    let denial = backend.layout_line_admitted(source, gpui::px(11.5), &[run]);
    if !matches!(denial, Err(gpui::TextAllocationError::MissingNativeMeasurement))
        || denied.native_entries.load(Ordering::SeqCst) != 0 || denied.live.load(Ordering::SeqCst) != 0 {
        return Err("scratch-denied control entered native code or retained an owner".into());
    }
    println!("CONTROL scratch-denied native_entries=0 live=0");
    checkpoint(3, "denial-control-complete", &policy, &probe)?;
    let mut failed = false;
    for (operation, published, released) in [(0, "cold-published", "cold-drained"), (1, "warm-published", "warm-drained")] {
        let pool = Pool::new()?;
        let source = gpui::AdmittedTextSource::new(&text, policy.clone())?;
        let result = backend.layout_line_admitted(source, gpui::px(11.5), &[run]);
        match &result {
            Ok(layout) => println!("RESULT {operation} published len={} width={:?}", layout.len(), layout.width()),
            Err(error) => { println!("RESULT {operation} failed {error:?}"); failed = true; }
        }
        // The returned layout owns Rust source/glyph storage. Draining this pool
        // while retaining it separates native temporary cleanup from owner release.
        drop(pool);
        checkpoint(4 + operation * 2, published, &policy, &probe)?;
        drop(result);
        if policy.live.load(Ordering::SeqCst) != 0 { return Err("source/glyph charge remained after actual owner release".into()); }
        checkpoint(5 + operation * 2, released, &policy, &probe)?;
        if failed { break; }
    }
    let cleanup_pool = Pool::new()?;
    drop(backend);
    drop(cleanup_pool);
    checkpoint(8, "backend-dropped", &policy, &probe)?;
    // Identity lookup can initialize additional native state, so it happens only
    // after the runner has captured the measured allocation interval.
    let identity_pool = Pool::new()?;
    probe.print_font_identities()?;
    drop(identity_pool);
    if !probe.is_valid() { return Err("bounded marker/font directory incomplete".into()); }
    if failed { return Err("actual native operation failed; no fallback or larger case attempted".into()); }
    println!("DONE uncalibrated-measurement-only");
    Ok(())
}
