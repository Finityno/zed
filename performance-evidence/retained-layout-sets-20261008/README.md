# Retained layout-key gathering qualification

Candidate `26c8b7fa65224be324f8eaa5a4a47840e3041431` replaces the temporary
vector of retained layout-key slices with a cloneable iterator. The engine
validates the first root and every node before claiming any set. No field,
cache, pool or persistent owner is added. Baseline is owned revision
`c295ae14662edcb2b6e6cfd1d1861dd0b1feab11`.

Across 336 fresh native children and twelve complete-frame workloads, panel
updates with 32/256 unchanged rows use 7.54/8.57% fewer allocation calls and
1.60/4.05% fewer requested bytes over the controlled App/Window lifetime.
Each counting lane is deterministic; gains exceed both observed ranges and
the frozen5% allocation-call target. Held/peak requested heap is unchanged in
these primary cases. Every CPU, elapsed, requested heap and native RSS control
passes its frozen gate. All scene/hitbox hashes and state-owner counts/bytes
match. CPU medians are2.79–3.00% lower, below observed variation; no CPU
improvement percentage is established.

Bare timing is separate from System counting allocators. Both lanes link the
exact same immutable compiled dependency objects from the recorded native
Cargo build, with distinct sources/output directories. The atomic claim
contract passes before/after; six candidate native contracts (including the
16-seed retained-frame oracle), strict owning Clippy and scoped benchmark
feature isolation pass. All preparatory failures remain in the archive.

`evidence.json` records every metric, gate and identity; `receipts.tar.gz`
preserves the complete raw cohort and exact compiler/source inputs. This is
a Linux native fake-platform complete-frame allocation result. No whole-app,
GPU, shipped allocator, macOS or Windows performance gain is established.
The previously failed removed-gap owner fix remains separate and unmerged.
