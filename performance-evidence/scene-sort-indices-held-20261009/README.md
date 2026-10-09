# Held direct sorted glyph copying

This candidate remains unmerged. All4,060 alternating native children and
326 correctness children are preserved. Sorting copies directly from the
packed64/128-bit source indices instead of filling a second permutation
vector. Stable/equal/wide sorting and in-place reuse remain correct: the
native no-extra-owner regression fails before and passes after.

None of the20 primary complete controlled Scene.finish fixtures reaches the
frozen5% CPU/elapsed target. Thread-CPU median differences range from2.297%
more to1.208% less. No CPU gain is established. All20 scratch owner targets
reduce20%, releasing5,196–18,800 requested bytes. Complete controlledScene
held heap is0.774–0.979% lower; four colored fixtures miss the frozen0.8%
target. All145 held/peak/requested-byte/allocation-call nonincrease controls
pass. Twelve ordinaryCPU/elapsed/wholechildCPU fields and27RSS fields fail.
Together with40 primary timing and four live-target failures,83 frozenfields
fail. No unchanged rerun, weaker gate, PR, merge or consumer adoption is made.

The119 completeWindow workloads match frame-by-frame; all26 controlled
Scene signatures match.620 common native test passes exclude the separately
checked expected-red owner regression. Zero-test filters remain explicit
coverage gaps. The first fixture-only compile failed because libc was not a
GPUI dependency; its exact source, artifacts and logs are preserved. The
clock was corrected to the same existing externTimespec helper before any
paired samples. All59 immutable external objects and exact compiler,
allocator, runtime and source identities match before/after all cohorts.

Source43ea16c93782ee5f2b86cf34c37795b9cbcf4632 starts from owned main
08d5555709d7afab3936c7afcd9f7f0ff9ecbd65. RefreshedFincode main
251866f12140de81510352122bdea5ddab87b007 retains allthree rendering pins
6b18c87dd5f2d1ccbee1ee641054295f2a78390b and component
031f54b627333272a8c157cbe4ca1cf591d16172.

Archive SHA256:0096387d92b4c144d92caa4b95a6b6831882f36d07eab24c6c7a459eaa2ffee2. All5819 members resolve to verified original
bytes. Identical content+metadata members share tar hardlinks. Large native
and dependency objects remain identified and preserved in the managedcloud.

Scope:LinuxGPUI fakeplatform/text, bareSystem and separate requested-counting
System allocator. ControlledScene.finish fixtures reconstruct actual captured
glyphkeys/counts with defaultgeometry/colors; non-glyph arrays were captured
already sorted. Immutable JSON inputs are parsed before requestedcensus;
Scene construction and warmup are counted, only finish loops are timed.
Neither exact application scenes nor physicalGPU/macOS/Windows/shippedallocator
performance is established. No owningClippy/guards are claimed after failed
performance gates. The frozen policy's memory_budget_reason accidentally
mentions a prior append design; its unchanged zeroheap/churn and all stated
primary/control gates are enforced. Continue with line-sized paint records,
retained range boundaries and original immutable sprite ownership.
