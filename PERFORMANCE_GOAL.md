# Fincode rendering performance goal

Reduce CPU time, allocation churn, live and retained heap, process RSS, frame
latency and GPU work in the owned GPUI fork. Prioritize long chat transcripts,
streaming responses, scrolling, editor and Review views, multiple panels,
background windows and repeated hide/wake cycles. Preserve every interaction,
text measurement, hitbox, overlay and notification contract.

## Baseline and source comparison

At the initial 8 October 2026 audit, Fincode and this fork's main both used
`ef46053b5794ccb57cda5dc658581ca2a399a4e2`. The application pins `gpui`,
`gpui_platform` and `sum_tree` together. Its fast-layout import is
`e89b7276d028fb7f9db3da5d290c0f56afb0de6b`, followed by retained replay changes
in `408fe59c6985ef3bd7e62dd35de1dadcb3b38454`.

The requested source comparison inspected revision
`5ea0f5b14f39f85ce3e3b34b3b8d31a9cc8a813f`. Our implementation is adapted into
its own modules; the inspected history does not identify one exact version pin
for that source. Revision dates alone do not establish feature equivalence.

| Opportunity | Current fork | Next evidence needed |
| --- | --- | --- |
| Carried text measurement ownership | Moves uniquely held layouts; copies shared layouts | Preserve the shared-owner tests |
| Leaf measurements at final width | Text prepaint already calls `fit_to_width` | Compare probe-width and independent-root regressions |
| Dependency-list allocation | Empty nested snapshots share storage after a qualified owned merge | Preserve recorder and retained-output regressions |
| GPU scroll layers | Not present in the inspected fork | Prototype CPU/frame gains, GPU cache bounds and hitbox/overlay correctness |
| Transcript scroll-layer stability | Requires the layer architecture first | Test per-row updates, at-end reads, parent notifications and overscan |
| Retained splice safety | Different implementation | Reproduce independent-root, deferred overlay and layout-claim regressions |
| Focus cleanup | Separate existing draft | Complete its application latency and CPU gates before acceptance |

Cached scrolling is the largest architectural opportunity. Published numbers
from another consumer are motivation for a prototype, not Fincode evidence.
New GPU caches must be bounded, reclaimed appropriately and measured together
with CPU and latency; lower CPU alone is insufficient.

## Completed and held iterations

Revision `33439ed70c2a38e7371fac721c26582de3f784f2` shares empty nested
dependency snapshots. Its complete recorder comparison qualified across 400
native children: 36.23% lower thread CPU, 50% lower held/peak requested heap
and 33.33% fewer requested bytes and allocation calls. Controls fit the frozen
baseline ranges. See `performance-evidence/empty-nested-20261008`.

Fincode adopted all three rendering dependencies at that revision in
`89d8d92ed576f1a87761c7a56cf4e078e7b66e33`, with the matching owned component
revision `0905a743b7f7b4f895f3ee7041bddbb23926aa20`. Separate scoped production
compile, native UI retention contracts, strict Clippy and source guards pass.
The component revision also restores caret blinking after a canceled pause
and window reactivation; its native regression fails before and passes after.
Merged source bytes were verified. No app-wide or GPU percentage is established.

Two allocation designs remain held. The broader empty-list pool failed RSS;
its original failed cohort is preserved. The in-place retained-state checker
at `caf83a60de3dc429263c81ea70be47f4838c3658` removes measured transient
allocation and lowers target checker CPU by 79–93%, but eight frozen ordinary
CPU/elapsed and RSS controls fail. Its 600-child evidence is preserved on its
owned candidate branch. Neither design may be repeated unchanged to seek a
passing cohort.

Continue with equivalent retained-splice ownership and independent-root
regressions before introducing scroll layers. Account for the independent
atlas retirement already present in main and preserve active idle-window,
focus and text-ownership work.

The removed-gap state release at `c3e601eed133e68f2950b53062cb6d4aaa19aeb7`
is also held. Its native regression fails before and passes after, and 280
complete-frame comparisons release all obsolete owners with 10/80 KiB less
held requested heap. Eleven frozen CPU, allocation-churn and RSS controls
fail. Preserve `performance-evidence/retained-splice-state-held-20261008`;
do not resample this design unchanged. Continue with transient layout-key
gathering, measured as a distinct candidate from current owned main.

## Iteration and merge policy

Use a focused branch for each distinct candidate. Start from current main and
record the exact application pin. Attribute CPU with a trace or a complete
measured component, and use existing live heap evidence to choose memory owners.
Declare a meaningful gain target and representative controls before measuring.
Use repeated alternating native comparisons with matched source, dependencies,
compiler and allocator. Keep bare timing separate from allocation instrumentation.
Retain every result, failed prototype and coverage limit.

Accept improvements only after their correctness checks pass and the important
gain exceeds measured run variation. Other CPU, memory and latency changes must
fit the declared controls. Record any fixed cache overhead explicitly. Component
measurements do not establish whole-application or native-platform gains.

Review the final diff and exact head, create a PR in the owned fork, attach it to
the working chat, resolve actionable feedback and squash-merge a qualified head.
Verify the merged bytes. Use `[skip ci]` and no GitHub-linkable issue references
in commit messages. Run scoped validation directly in the managed cloud, with
process-group deadlines, instead of validation Actions. Preserve unrelated work
and the user's running application.

Moving the application to a merged fork revision is a separate integration
change: update all three pins together, regenerate the lockfile and qualify
application behavior. Do not claim an application improvement from a fork merge
that Fincode has not adopted.

## Continuation

An hourly continuation is enabled in the working chat. Read its current handoff
and active processes before beginning; continue an existing iteration without
launching a duplicate. Never overlap native timing with compilation or other
profiling. After a qualified merge, select the next distinct important owner.
Keep failed designs held with evidence; do not rerun them unchanged to seek a
passing sample or weaken their acceptance gates.

Report verified improvements, meaningful failures and blockers requiring action.
Stay quiet when nothing actionable has changed. Public-facing repository links
must use owned forks. Preserve copyright, license and required attribution;
record comparison provenance with plain revisions and prose.
