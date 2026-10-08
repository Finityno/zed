# Held linear dependency-read merge

This candidate remains unmerged. Its first valid560-child cohort shows11.42–13.13%
less threadCPU/elapsed in four256-row changed-component workloads. Nine frozen
controls fail: seven normal retained CPU/elapsed/whole-child fields and two native
RSS fields. Every requested held/peak/churn byte and allocation-call median is
unchanged. Do not rerun this design unchanged or loosen acceptance gates.

The candidate merges sorted entity/global dependency reads in linear order instead
of sorting the concatenation. It preserves earliest floor-lifted versions, repeated
keys, deadlines and the existing zero-floor empty-input sharing. Source, compiler,
runtime, exact59 common externs, separate allocation/timing binaries, all20 workloads
and every raw result are recorded in the archive. The native read/floor oracle
checks219024 combinations; counted-key and retained-frame/dependency/scroll/layout
contracts pass. StrictClippy/adoption were not run because the performance gates fail.

The archive also preserves60 baseline diagnostic observations and560 invalid
children from an earlier12.038-second CPU/census overlap. No qualification uses
those invalid results. The repaired sequential supervisor retains unchanged source
and frozen gates; completed receipts prove both valid phases do not overlap.

These are Linux complete App/Window fixture results, with a fake renderer and System
allocator. They establish no whole-app, physical GPU, macOS/Windows or shipped
allocator gain. Source provenance is plain revisions in evidence.json. Original
source/package metadata and licenses remain preserved. The next design must be
distinct, with fresh ownership attribution and frozen controls.
