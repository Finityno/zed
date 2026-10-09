# Held padding-resident glyph tail state

This design remains unmerged. Its first and only 3,418 alternating native
children cover 121 Window workloads and three standalone Scene owners. All 20
frozen memory targets pass: complete App/Window held requested heap falls
10.24–28.77% and allocated paint-record capacity falls 75–97.92%, exceeding both
full lane ranges and the 5%/50% goals. Requested held/peak/bytes and allocation
calls meet every frozen control with zero fixed-overhead allowance. Native
Scene/Window/record widths stay exactly 1632/12944/184 bytes. Every standalone
Scene releases all counted owners.

19 frozen fields fail. The bare cohort fails six thread CPU, six elapsed,
two whole-child CPU and one current RSS fields. The separate census fails
three current RSS and one peak RSS fields. Panel keep/removal, retained redraw,
recoloring and decorated text have adverse CPU fields; resizing, state changes
and redraw have adverse RSS fields. Every failure and full raw lane range is
preserved. No PR, merge, adoption or unchanged resampling follows this result.

Two publication flags fit existing Scene padding. Scene.len checks the local
tail kind and publishes a glyph tail without a heap lookup. Uniform glyph
ranges extend in place without scalar promotion or post-paint scanning. Non-glyph
replay movement/remap/animation/insertion remains exactly the owned source.
The native compiler inlines Scene.len, but this is mechanism evidence, not a
CPU gain. A distinct next design will keep index capture and ordinary Scene
insertion unchanged and scope coalescing to a line painter; no benefit is yet
inferred for that design.

Final native correctness has 624 common passes, 278 children, 121 exact matched
frame histories, six standalone owner releases and one cold-line expected-red/
green child per lane. Two legacy zero-test filters remain explicit coverage gaps.
Shared positive-test definition order matches. Private observers normalize logical
overlay indices and preserve raw compact metadata. Exact owned transformations
reproduce both changed source files. Earlier failed designs stay preserved.

Source: b706c55f84ac00c4fa44176742df5cb439f847d8.
Owned baseline: 08d5555709d7afab3936c7afcd9f7f0ff9ecbd65.
Frozen policy SHA256: 46c6519e7fa13140b3e94f4c6dfa07b830e0cad5f48e7fcacb05e3f9715cbe34.
Frozen consumer source: 63f5d04838ef9a00429666a22da927c457e7fc78, with coherent
rendering pin 08d5555709d7afab3936c7afcd9f7f0ff9ecbd65 and component pin
ab07cd7695af1dfd9f1c6ad06a77660d288e568d. A later read-only refresh observed
consumer 1561838d1be2dda701a59e59811973c2abb64649 with the same pins; its
independent changes are preserved. Owned rendering main and open drafts stay unchanged.

Bare and counting cohorts run separately, sequentially, and once. All source,
compiler, 59 immutable dependency objects, runtime, four binary and policy hashes
pass before/after checks. opt-level=3, 16 codegen units and debug assertions stay
fixed. Host guards precede every child. There is no overlap with builds, hashing,
compression or other profiling. Counted CPU is not CPU gain evidence.

A managed-session reconnection lost the outer supervisor receipt for three builds.
All three inner compiler receipts independently show actual exit zero, exact log
hashes and sequential start/end timestamps. The saved recovery receipt distinguishes
these observed exits from the unavailable outer receipt. No build or sample was
repeated. The reconnect occurred before paired measurements. Both native cohorts
have observed outer exits and exact logs. Original inactive native objects remain
losslessly preserved with identity and metadata receipts.

Archive SHA256: 9ba838378ef43b697b92e089510e15c453fd69c3b29c2cbdb8a3eeff277020d2. All 4664 members roundtrip to original exact bytes.
Tar hardlinks apply only to identical bytes, sizes, modes and nanosecond mtimes.
Linux fake GPUI platform/text with System and separate counting-System fixtures
establish no application, physical GPU, macOS, Windows or shipped-allocator gain.
Owning Clippy is not claimed after failed performance gates. Continue a distinct
line-local design; never repeat this failed source unchanged or weaken gates.
