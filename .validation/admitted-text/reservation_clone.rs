fn misuse(reservation: gpui::TextAllocationReservation) {
    let _retained = reservation.clone(); // expected-error:E0599
}
fn main() {}
