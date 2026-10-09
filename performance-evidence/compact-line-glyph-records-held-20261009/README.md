# Held compact descriptors with line-local glyph ranges

This design remains unmerged. Its first 3,418 native children cover 121 Window
workloads and three standalone Scene owners. All 20 frozen memory targets pass:
complete App/Window held requested heap falls 10.36–28.83%, and allocated record
capacity falls 77.17–98.10%, exceeding both full lane ranges and the frozen
5%/50% goals. All CPU, elapsed, whole-child CPU and requested held/peak/bytes/calls
controls pass. Four native RSS controls fail; no PR, merge or adoption follows.

Bare peak RSS after a 256-row state update rises 332 KiB against its 240 KiB
baseline full-range allowance. Current RSS on a 32-row retention-off redraw rises
300 KiB against 232 KiB. The one-empty-Scene census has current/peak increases
84/64 KiB against 60/44 KiB. Every raw sample and adverse range is retained.
The gates are unchanged, and this design must never be resampled unchanged.

The candidate combines direct non-glyph descriptors with callback-free line
segments. Native descriptor width changes 184 to 168 bytes; Scene/Window widths
remain 1632/12944 bytes, with no fixed heap/header budget. Public GPU payloads
and vectors are preserved. Arbitrary underline callbacks establish a fresh
floor even when they paint nothing, so captured retained prefixes cannot grow.
Public single-glyph insertion remains scalar; generic replay preserves source
range boundaries, moves, clipping, transition and shimmer remapping.

Native checks have 626 suite passes across both lanes, 278 correctness children,
121 matched frame histories, six released standalone owners and a cold-line
owner contract that fails before and passes after. Before the first paired
sample, source review strengthened the candidate movement test to require 22
scalar operations versus five operations for three actual glyph ranges. Only
that test body changed from the initial algorithm source; its earlier two
candidate binaries, receipts and successful checks are preserved. Baseline
binaries stayed unchanged. The original baseline scalar test and the strengthened
candidate range test have the same test name but explicitly different bodies.

Both measured lanes use the same compiler, immutable 59 external objects and
runtime libraries, with separate bare System and counting-System binaries.
All 3,418 exits/log identities and actual outer completion receipts are verified;
bare and census phases are sequential. The detached bare supervisor survived a
managed reattach. An initial launch guard self-match was rejected before spawn
and corrected by excluding only its own PID; no samples were repeated. Inactive
owned cache and completed binaries were losslessly preserved for disk reserve.

The archive contains exact source, frozen protocol, transforms, build/check
receipts, raw samples, source/runtime/allocator/binary identities and all failures.
Part concatenation reconstructs its SHA256 in evidence.json. Every member was
verified against its original bytes; hardlinks require identical bytes, size,
mode and nanosecond timestamp. Completed native binaries remain losslessly
preserved in the managed workspace with round-trip identities.

These are Linux fake-platform/Text/System component results. No whole-app,
physical GPU, macOS/Windows, shipped-allocator or Fincode adoption gain is claimed.
