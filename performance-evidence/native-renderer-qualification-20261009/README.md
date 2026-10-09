# Native renderer qualification and complete-frame attribution

This is a diagnostic foundation, not an optimized source or a merge qualification.
The owned rendering source is `08d5555709d7afab3936c7afcd9f7f0ff9ecbd65`.
Fincode was frozen at `3d7e2953ff58befa0859c8a2e67f0a84682735f2`, with coherent
three rendering pins at that revision and component `ab07cd7695af1dfd9f1c6ad06a77660d288e568d`.
The final refresh at `fb05508b59112510814e9f3d507677eaed6d0b59` preserves those pins.

All 54 existing native WGPU checks pass. 144 bare and 96 counting-System children
cover complete shaped transcript Windows: streaming, scrolling, unchanged redraws,
resizes, two active windows and an inactive sibling, with retention and grayscale/
subpixel modes on/off and a matched renderer-free control. Actual Vulkan CPU adapter,
dual-source blending, every active frame submission, zero inactive-sibling submissions,
retention on/off pixel equality, and complete renderer logical-owner release pass.
The GPU wait is included; readback/checksums are outside timed frames.

Retained scrolling medians on software Vulkan are 7.91–8.15 ms completed wall time,
0.77–0.79 ms foreground CPU and 15.76–16.35 ms process CPU per frame. Foreground
encoding/submit uses 0.26–0.31 ms. Full ranges are retained, including large variance
in grayscale and streaming; these numbers do not qualify an optimization.

Tracked resident renderer bytes for one 1600x2000-device-pixel window include
6,400,000 depth bytes, 2,097,152 instance-buffer bytes, 1,048,576 grayscale or
4,194,304 subpixel-atlas bytes, plus 12,800,000 bytes in the headless readback target.
No cache exists yet. A future cache must declare and enforce its own bounded budget
and measure requested Rust heap, renderer bytes, native RSS and reclamation together.
Zero renderer gauges do not establish that process RSS returned to the OS.

Limits: LLVMPipe Mesa25.0.7/LLVM19.1.7 with two software driver workers, Linux,
System allocator, headless offscreen output. This is not a physical GPU, macOS,
Windows, native presentation, shipped allocator or application percentage. Headless
windows have independent devices; production may share a context. Rust census
includes all Rust threads/setup/warmup and the declared diagnostic sample Vec;
Mesa C requests/library mappings and allocator retention remain unattributed by it.
Native RSS and adverse post-drop resident values are preserved. No malloc_trim.

The unchanged production source is rebuilt coherently. Only isolated private
observers are appended under a non-shipping cfg; the original production prefixes
and all 63 immutable external objects, compiler, runtime, tools and binaries were
verified. Initial dependency-alias, required-feature, harness-import/DPI and host-
guard setup faults are retained; they occurred before the first frozen child.
No first diagnostic child was repeated. Bare timing and census phases are sequential
and no builds, hashing/compression or other profiling overlap them.

The archive contains every first child and receipt, frozen policy and identities,
all setup/adverse results, source snapshot, raw per-frame samples, pixel contracts,
variation and coverage limits. Large native binaries are preserved locally with
round-trip identities, not copied into this source evidence archive.

Next: qualify bounded opaque tile compositing against actual directly rendered pixels,
including grayscale/subpixel, clipping and integer/fractional moves, and renderer/RSS/
requested-memory release, before defining a production scroll-cache candidate.
No PR, production merge, application adoption or performance gain is implied.
