# Held line-local glyph coalescing

This design remains unmerged. Its first and only 3,418 alternating native
children cover 121 Window workloads and three standalone Scene owners. All 20
frozen memory targets pass: complete App/Window held requested heap falls
10.24–28.77% and allocated paint-record capacity falls 75–97.92%, exceeding both
full lane ranges and the 5%/50% goals. Every requested held/peak/bytes and
allocation-call control passes with zero fixed-overhead allowance. Native
Scene/Window/record widths stay 1632/12944/184 bytes. All counted Scene owners release.

All frozen bare thread CPU, elapsed and whole-child CPU controls pass. Eight
RSS fields fail: one current and one peak field in the bare cohort, four current
and two peak fields in the separate census. Scroll-off-512, panel growth/keep/
removal, nested resize and an empty eight-Scene control have adverse fields.
Every failure and full raw lane range is preserved. No PR, merge, adoption or
unchanged resampling follows this result; successful CPU controls do not justify acceptance.

Coalescing is scoped to a line-local stack floor. Every arbitrary underline
callback resumes at the end of the prefix it could capture, even when it paints
nothing. Ordinary Scene len, clear, clipped layers, insertion and original enum
discriminants remain unchanged. Scalar/non-glyph replay retains its original
movement/remap/animation/insertion body. New range replay has a local floor for
each source record. There is no Scene publication flag, fixed heap/header growth
or post-paint scan. Direct descriptor writes happen only within a callback-free
line segment. The public single-glyph painter uses original insertion.

Final native correctness has 626 common passes, 278 children, 121 exact matched
frame histories, six standalone owner releases and one cold-line expected-red/
green child per lane. The shared real Window regression captures a retained
prefix in an underline callback that paints nothing; replaying it after the rest
of the line still yields only the original two glyphs. Two legacy zero-test
filters remain explicit coverage gaps. Shared positive-test definition order
matches, all four changed source files reproduce from saved owned transformations,
and compiled production is exact after stripping named observers/shared fixtures.

Source: 014f9332ca1a468520da0a0ba80f2fb53b4bd015.
Owned baseline: 08d5555709d7afab3936c7afcd9f7f0ff9ecbd65.
Frozen policy SHA256: e8835e404f1ca2a5d5e1a1980c125914aabbc546402db4a47cb954d91f63d04f.
Frozen consumer: 2bf1d44412e7c9e284b912f38a2074c6a3580d9c, with coherent
rendering pin 08d5555709d7afab3936c7afcd9f7f0ff9ecbd65 and component pin
ab07cd7695af1dfd9f1c6ad06a77660d288e568d. Independent consumer work and owned drafts are preserved.

Bare and counting cohorts run separately and once. All source/compiler/59 immutable
dependency objects/runtime/four binary/policy hashes pass before and after both
phases. opt-level=3, 16 codegen units and debug assertions remain fixed. Host guards
precede every child. No timing overlaps builds, compression, hashing or other
profiling. Counted CPU is not CPU gain evidence.

The first candidate census link failed with signal 7 under disk pressure before
any paired sample. Exact diagnostics, source/options and all 17 failed objects
are preserved. Only inactive cache owned by this loop was losslessly compressed;
all 59 native dependency objects were protected and checked before/after. Only
that failed pre-metric build was retried, with identical source/options and an
observed successful exit. The other three successful builds were preserved.

A managed-session reattachment lost access to the bare outer supervisor and its
completion receipt. No outer bare exit or completed 1600-second guard is claimed.
All 2,178 native child exit codes are observed zero, all exact child log hashes
verify, the controller end log records the complete cohort, and each child retains
its 70-second subprocess deadline. No build or sample was repeated after reconnect.
The subsequent census has an observed outer exit and exact log. This supervision
coverage limit remains explicit, including in the local recovery receipt.

Archive SHA256: 014d14a1af7cad1be3242709285246fb41202017f1103c1f1ae3b23298129fd4. All 4678 members roundtrip to exact original bytes.
Tar links require identical bytes, sizes, modes and nanosecond mtimes. Native
binaries and failed object bytes stay losslessly preserved locally with identities.
Linux fake GPUI platform/text with System/separate counting-System fixtures
establish no application, physical GPU, macOS, Windows or shipped-allocator gain.
Owning Clippy is not claimed after failed performance gates. Continue a distinct
smaller descriptor representation with line-local ranges and all frozen controls;
never repeat either held design unchanged or weaken acceptance gates.
