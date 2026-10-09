# Held cached-line-only glyph records

This design remains unmerged. All 3,418 alternating native children across
121 Window and three standalone Scene workloads are preserved. All 18 primary
record-owner and complete App/Window live requested heap results are unchanged,
so all 36 primary memory fields fail their 50%/5% targets. The descriptor count
can fall while old fresh-paint vector capacity stays allocated: normal shrinking
requires 120 low-use clears per Scene. Neither histories nor gates were extended
to obtain a pass.

110 frozen fields fail in total: 23 thread CPU, 22 elapsed, 21 whole-child CPU,
36 primary owner/live targets, five RSS and three requested-byte/allocation-call
controls. Two recolor workloads incur extra allocation calls, and one requests
178,848 extra bytes. 28 secondary CPU/elapsed fields exceed the optional 5% and
both full variation ranges, but they do not qualify this design. All bare RSS
controls pass. All standalone owners are released and Scene/Window widths remain
exactly unchanged. No PR, merge, consumer adoption or unchanged repetition occurs.

Final correctness has 624 common native passes, 278 children, 121 matched frame
histories, six separate empty-owner children and the native cached-line regression
failing exactly once before and passing exactly once after. Original cold glyph
insertion, text painting and paint-index publication are preserved. Typed overlay
audit expands compact indices to their original logical glyph boundary while raw
positions are retained. Zero-test filters are explicit coverage gaps.

Source: e18d38f62a106a8ad0bae9cd849cc50a02b78413.
Owned baseline: 08d5555709d7afab3936c7afcd9f7f0ff9ecbd65.
Frozen policy SHA256: 584c8107241810c664bcf39d2b91565193553a11568507ad5136031005d87692.
Frozen consumer source: 8812bc6b2027c445b3d05307e49836dad813dbce,
rendering pin 08d5555709d7afab3936c7afcd9f7f0ff9ecbd65 and component
ab07cd7695af1dfd9f1c6ad06a77660d288e568d. Independent consumer work is preserved.
Decorated cases remain controls; their zero cached-line replays make them outside
this distinct design's declared primary scope. Earlier held policies remain intact.

All exact compiler, allocator, native runtime, 59 immutable external objects,
source and four final binaries pass before/after checks. Initial suffix and test
helper setup failures, their original bytes and native compile errors are retained.
Shared positive-test definition order was aligned before performance samples;
earlier baseline identities and diagnostic correctness logs remain separately
preserved with relocation receipts. Final matched correctness reran after that
alignment. Bare and counting phases are separate and sequential; counted CPU is
not used as CPU gain evidence. No timing overlaps compilation, hashing/compression
or another profile. Inactive objects are preserved with digest/metadata receipts.

Archive SHA256: 10683f87b0f1b86af9c2f486be90d2ad21946e5d6f8383c6e883309af481df7c. All 4984 members resolve to exact original bytes.
Only identical content and metadata share tar hardlinks. This is Linux GPUI fake
platform/text with System and separate System counting allocation. No physical GPU,
application, macOS, Windows or shipped-allocator gain is established. No owning
Clippy is claimed after failed performance gates. Continue with a distinct direct
glyph-record emitter which avoids allocating wide per-glyph records initially and
preserves original scalar replay rather than rebuilding then scanning records.
