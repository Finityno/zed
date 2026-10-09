# Held borrowed line-color probes

The frozen 2,604-child comparison across 93 complete App/Window workloads
meets all four churn targets: 15.65-27.32% fewer allocation calls and
3.46-5.57% fewer requested bytes in highlighted fast-mode redraws. Primary
held and peak heap stay unchanged; primary CPU differences are below variation.
Six retained highlighted streaming/recolor CPU, elapsed and whole-child fields,
and two census RSS fields fail. Preserve this design unmerged and never repeat
it unchanged or weaken the gates. No PR or adoption is qualified.

The probe borrows decoration colors, retaining owned colors only for fresh
sprite recording. This adds no cache fields or GPU resources and leaves atlas
retirement/generation, retained verification and glyph-only checks intact.
App, Window and owned-key sizes are identically 1,896, 12,944 and 232 bytes.
The real paint allocation regression fails before with 960 bytes/two calls and
passes after with zero. Thirty-eight final native correctness invocations pass,
and 140 functional plus all 2,604 paired scene/state/replay outputs match.

Sixteen current-main diagnostic children attribute key spills above eight runs.
All raw samples, setup failures, rejected helper sources and exact binary
identities are preserved. The invalid idle-phase regression was corrected into
a real paint callback before any paired performance sample; its failures are
not evidence of an allocation regression. The original successful diagnostic
child rejected by a parser was reparsed without repeating that child.

Both lanes share the same 59 immutable externs, compiler, dependency/root/lock
identities and System allocator. Four builds, bare timing and counting census
run sequentially with external deadlines; every archive member is hash verified.
Recolor controls issue individual row notifications and represent a burst,
not a batched theme change. Native Linux fake-platform results are not app,
physical GPU, macOS/Windows or shipped-allocator percentages. Continue with
the retained capacity of existing replay maps in idle windows as a distinct
memory owner; retain every adverse result from this design.
