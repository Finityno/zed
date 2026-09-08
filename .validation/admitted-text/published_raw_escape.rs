fn misuse(published: gpui::AdmittedLineLayout) {
    let _raw = published.layout; // expected-error:E0616
}
fn main() {}
