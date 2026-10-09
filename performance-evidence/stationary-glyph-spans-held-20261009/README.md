# Held stationary glyph spans

This candidate remains unmerged. All eight target complete-component CPU and
elapsed gates pass, with 24.84–42.97% lower thread CPU, but 49 frozen control
fields fail: 16 ordinary CPU, 16 ordinary elapsed, eight whole-child CPU and
nine census current/peak RSS fields. All held/peak/requested-byte and
allocation-call controls pass. Do not repeat this design unchanged or weaken
its gates to seek acceptance.

Both lanes link the same 59 immutable external objects, including original
Taffy 0.13, with the same compiler and flags. Bare timing and the System
counting allocator run separately and sequentially: 2,142 bare children in
464.911 seconds, then 1,190 census children in 118.555 seconds. Source,
compiler, dependency, allocator/runtime, binary and policy hashes match before
and after both phases. No build, compression or other profiling overlaps.

Correctness passes in 274 children: 622 native test passes and 119 paired
complete-frame audits of primitives, hitboxes, state owners and frame counts.
The independent scalar insertion oracle covers 768 capacity/length/kind/prefix
cases and 48 mixed layer, clip, move, rebase, transform and animation histories.
Two zero-test filters are explicitly recorded as providing no coverage.

The current-main instrumented diagnostic identifies general retained scene
replay at 45–49 microseconds per frame. A separate count-only opportunity
fixture finds 84.63–97.93% of retained glyphs in eligible long spans. Those
instrumented figures are attribution, not uninstrumented gain claims. Prior
failed scalar glyph and stationary-layer designs remain preserved separately;
this candidate bulk-appends contiguous spans with scalar growth capacities.

The first native compile failed for a missing explicit closure offset type;
its exact source, logs and artifacts are retained. Initial oracle tests pass,
and all four final matched binaries are rebuilt after the named-case test
helper change. No paired measurement was discarded or repeated.

Source provenance is owned baseline 08d5555709d7afab3936c7afcd9f7f0ff9ecbd65 and
candidate ee5e8b086de0b82782b2733e662e0d0c24974689. The refreshed consumer pins
remain rendering 6b18c87dd5f2d1ccbee1ee641054295f2a78390b and component
031f54b627333272a8c157cbe4ca1cf591d16172. No consumer adoption is made.

Archive SHA256: 8c97dc50cbe98d2b180119324dac2ef262c3f375200828ffbc2432913d384969. All 5519 members are verified against their raw
bytes after compression. The archive includes all raw cohorts, functional
logs, exact source snapshots, fixtures, observer diagnostics, setup failures,
phase receipts and input identities. Large immutable dependency and native
binary bytes are identified by their hashes and retained in the managed
workspace; source attribution and licenses remain intact.

Coverage is native Linux GPUI with fake platform/text rendering and System.
This is not a Fincode application, physical GPU, macOS/Windows or shipped
allocator result. No shader, GPU cache, renderer or new memory owner is added.
The next distinct investigation measures paint-record payload ownership on
current main before choosing a representation change.

Reconstruct the exact original archive by concatenating, in order,
`receipts.tar.gz.part000`, `receipts.tar.gz.part001`, `receipts.tar.gz.part002`, `receipts.tar.gz.part003`.
The original archive SHA256 above remains unchanged. Parts are a transport
size adjustment; no sample or source is changed or discarded.
