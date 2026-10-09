# Bounded sorted dependency-read merging

Large retained-frame unions repeatedly sort already ordered entity/global reads.
This candidate keeps the original small-list algorithm and empty-input sharing,
and dispatches only unions with 64 or more reads to an out-of-line linear merge.
Earliest floor-lifted versions, duplicate keys and ordering remain equivalent.
No cache, pool, field, state checker or renderer behavior changes.

One 560-child alternating comparison across 20 complete native workloads qualifies.
Four 256-row changed-component cases use 9.41–11.96% less thread CPU and 9.47–11.98%
less elapsed time, above both full lane ranges and the frozen 5% target. All normal
CPU/elapsed/whole-child, requested heap/allocation and native RSS controls pass.
Every held/peak/requested-byte/allocation-call median remains identical; no memory
or RSS percentage gain is established. All raw samples and adverse source evidence
are preserved; none are dropped or retried until passing.

Native read/floor oracle coverage includes 438,048 combinations plus 63/64/65 boundary
cases. Counted-key regression fails before and passes after; retained frame oracle,
dependency updates, wheel scrolling, live-row panels, layout/atomic-claim and nested
recording contracts pass. Actual owning strict Clippy and benchmark isolation pass.
The scoped lint root has 26 members with exact relevant current-root dependencies,
lints and profiles; native snapshots retain the full original root and lock.

The earlier unbounded inlined design remains held on its own evidence branch with
nine failed controls. Its invalid overlapping cohort is also preserved and unused
for qualification. This design changes the dispatch/call architecture; its single
valid cohort uses a repaired sequential runner with unchanged gain/control gates.

These are Linux complete App/Window fixture observations with a fake renderer and
System allocator. They establish no whole-app, physical GPU, macOS/Windows or shipped
allocator gain. Coherent owned component/Fincode adoption needs separate checks.
Archive receipts retain exact source/compiler/stdlibs/common extern/runtime/binary
identities, all 20 workloads, both measurement modes, direct native/lint results and
coverage limits. Every archive member is hash verified. Original copyrights,
licenses and source/package metadata are preserved; provenance uses plain revisions.

Review follow-up adds 54 empty-side and asymmetric-tail floor cases against the
original algorithm. Native exact-helper tests and actual owning strict Clippy
pass. Production bytes and all timing samples remain unchanged; supplemental
receipts are in `review-receipts.tar.gz`.
