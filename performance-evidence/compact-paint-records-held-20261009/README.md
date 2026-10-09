# Held compact paint records

The candidate is unmerged. All 20 declared memory targets pass across 3,332
native children: paint records shrink from 184 to 168 bytes, with 8.6957% less
requested owner storage. Transcript fixtures retain 64 KiB less and use
1.017–1.191% less complete-component live requested heap. Colored fixtures
retain 384 KiB less and use 2.500–2.555% less live requested heap. All ordinary
CPU, elapsed, whole-child CPU, peak RSS, requested held/peak/total byte and
allocation-call controls pass. Six current RSS controls fail their frozen
baseline ranges, so do not merge, adopt, resample unchanged or weaken gates.

The failures are bare value-on-16-bare-nested (+564 KiB versus484 variation),
value-on-80-bare-flat (+692 versus516), color-redraw-off-32-32 (+1508 versus388),
and census resize-on-80-bare-nested (+1036 versus516), keep-256 (+268 versus260),
color-recolor-on-200-32 (+1656 versus196). Requested byte reductions do not
establish a process RSS reduction.

Twenty current-main complete-window owner attribution children precede the
prototype and frozen policy. Bare timing comprises 2,142 children in488.979
seconds, followed separately by1,190 census children in123.578 seconds.
All59 immutable original externs, original Taffy0.13, compiler/flags and
System allocator/runtime identities match across both lanes and before/after
phases. No timing overlaps builds, compression, hashing or other profiling.

Common correctness passes in274 children: 622 native test passes plus119
paired complete-frame primitive/hitbox/state-owner/count audits. The native
footprint contract fails on baseline and passes on the candidate; both common
suites explicitly skip that separately verified expected-red contract. The
new independent snapshot regression passes before/after: renderer-array
mutations do not alter stored quad/shadow/path/underline/image snapshots,
path vertices remain independently owned, and shadow blur extents retain their
clip rule. Zero-test filters are explicitly excluded from coverage claims.
The candidate's complete Scene suite also passes before the cohort.

This flattens the private record enum; public Primitive payloads, scalar
replay/move/rebase/animation behavior, buffer counts/capacities and rendering
APIs remain. No prior held glyph-span scanner, cache, new heap owner or unsafe
code is copied into this design. Source isf71a4eba34a6dc288e8180dc61945d8be4fcbb95
on owned baseline08d5555709d7afab3936c7afcd9f7f0ff9ecbd65. Consumer pins remain
rendering6b18c87dd5f2d1ccbee1ee641054295f2a78390b and
component031f54b627333272a8c157cbe4ca1cf591d16172. No consumer adoption is made.

Archive SHA256:21a17a5d9d2753f7232106293f75f2a61efa1c2534bac62037eb47484ead1602. All5020 raw/source/fixture/identity members resolve
to verified original bytes. Exact duplicate members share tar links only when
content and file metadata match. Large native/dependency artifacts are
identified by hashes and retained in the managed workspace. Original source
attribution and licenses remain intact.

Coverage is Linux GPUI fake platform/text and System requested counting.
No Fincode application, physical GPU, macOS/Windows or shipped allocator gain
is established. Next investigate the intermediate permutation buffer used
while copying sorted glyphs, with captured current-main scene evidence.
