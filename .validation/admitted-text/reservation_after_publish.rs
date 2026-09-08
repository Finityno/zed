fn misuse(source: gpui::AdmittedTextSource, raw: gpui::LineLayout, mut reservation: gpui::TextAllocationReservation) {
    let _published = gpui::AdmittedLineLayout::from_native(source, raw, reservation);
    let _result = reservation.reconcile(0); // expected-error:E0382
}
fn main() {}
