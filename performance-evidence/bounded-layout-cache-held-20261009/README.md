# Held bounded layout-constraint cache

This distinct candidate keeps the original 0.13 layout/leaf/default-feature API
and 368-byte cache, replacing fixed-kind eviction with nine FIFO constraints.
Original source authors and MIT terms are preserved in the owned local crate.

Across 2352 native children and 84 workloads, four nested-panel updates use
35.52–43.61% less thread CPU. All bare CPU, elapsed, whole-child CPU and RSS
gates pass. The census fails 14 frozen fields: 13 current/peak RSS controls
and a 72-byte peak requested-heap increase for a small resize control.
The candidate remains held and unmerged. Never repeat it unchanged or weaken
its gates to obtain acceptance. No consumer adoption or performance PR exists.

Eighteen pure-tree comparisons have identical cache/style/layout/tree sizes,
requested held/peak/total bytes and allocation calls, with all owners released.
Three meaningful cache regressions fail before and pass after; 121 dependency
tests, native correctness and all 168 per-frame audit children pass. A constant
pure-tree footprint does not establish complete-component RSS equivalence.

The archive contains raw samples/logs, source snapshots, frozen policy,
adverse results, exact identities and sequential process-group receipts.
Compiled artifacts remain local with explicit preservation identities.
The isolated Cargo-generated owned-path record matches the seeded root record;
1828 unrelated records are unchanged. Full root compile, strict owning Clippy
and guards were not run after the performance rejection.

These are native Linux fake-platform/text component measurements with System,
not application, physical GPU, macOS/Windows or shipped-allocator gains.

Concatenate `receipts.tar.gz.part000` and `receipts.tar.gz.part001` in that
order to restore `receipts.tar.gz`. Its SHA256 remains
`407f403d36fc4d6ff77f63d7d7180700ab77793d18c27cd574d090f50e9cf582`.
The archive is split only to fit the connector request-size limit.
