# Held bounded state-only reconciliation

This design is held and unmerged. Its first 2,040 fresh native children cover
42 complete-App dependency-check cases and 12 complete-Window controls. All four
marker-bearing 64/256-state CPU and elapsed targets pass: 72.07–89.08% lower bare
thread CPU, above both full lane ranges and the frozen 20% goal. The separate
census eliminates all primary transient allocation calls, bytes and peak, with
zero held bytes in every primary child. All twelve complete-Window CPU/RSS
controls pass. Twenty-one other frozen or additional strict fields fail.

Changed 64/256-state lists with an absent exception marker are 18.95/15.54%
slower in bare thread CPU, with elapsed and whole-child CPU also failing their
baseline ranges. Calling the original checker and then reconciling a state
result traverses these lists twice. One bare and three census RSS fields fail.
Nine additional counting-cohort CPU/elapsed controls fail. Two candidate
all-samples-zero retained checks fail despite zero medians: four total census
children contain two extra allocations totaling 144 bytes. The process-global
counter includes other child threads; no allocating owner/thread was traced.
These outliers are preserved and remain failures, with no guessed attribution.

The original normal checker and small-list body are byte-identical; lists of
at least 64 states enter a non-inlined reconciliation path. There are no new
App/Window fields, persistent caches or fixed heap/header allowances. The
native state oracle covers 63/64/65/128/256 boundaries, absent/first/middle/last
markers and all state/global/entity/deadline combinations with notifications
on/off. Sixty correctness children pass, including matched histories for all
12 Window scenarios, retained output, layout, atomic claims and input contracts.

Before the first paired sample, fixture review corrected missing shared module
support, implemented explicit resize/state-bump/retention-off controls, and added
current RSS reporting outside the component timing/counting interval. All failed
setup receipts and successful earlier unpaired binaries are losslessly preserved.
No repeated timing cohort or weakened gate occurred. Final four native binaries,
compiler, immutable 59 dependency objects, runtime, allocator scope, frozen
policy and all raw logs have exact identities. Actual external deadline receipts
prove successful, sequential bare and census phases; every child passed its
output signature. No owning Clippy, PR, merge or consumer adoption was attempted
for this held design.

The archive contains source, protocols, transforms, checks, raw samples, receipts
and identities. Every member and archive part was verified byte-for-byte.
Completed binaries remain losslessly preserved in the managed workspace with
round-trip hashes. Never repeat this design unchanged to seek acceptance.

These are Linux fake-platform/System component results, not whole-app, physical
GPU, macOS/Windows, shipped-allocator or Fincode adoption percentages. Continue
with a distinct marker-present routing design that preserves the original
absent-marker path; qualify its owner attribution and all controls separately.
