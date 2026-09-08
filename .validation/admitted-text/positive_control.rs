fn main() -> Result<(), Box<dyn std::error::Error>> {
    let source = gpui::AdmittedTextSource::new("owned", std::sync::Arc::new(admitted_text_validation::Policy::default()))?;
    let raw = gpui::LineLayout { len: 5, ..Default::default() };
    let reservation = gpui::TextAllocationReservation::for_line(&source, gpui::AdmittedLineLayout::allocation_bytes(&raw)?)?;
    let published = gpui::AdmittedLineLayout::from_native(source, raw, reservation)?;
    assert_eq!(published.len(), 5);
    Ok(())
}
