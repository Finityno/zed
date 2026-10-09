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
fail. Its evidence is retained on branch `perf/retained-splice-state-20261008`
under `performance-evidence/retained-splice-state-held-20261008`; do not
resample this design unchanged.

Distinct layout-key gathering at `26c8b7fa65224be324f8eaa5a4a47840e3041431`
qualified across 336 native children and twelve complete-frame workloads.
Panel updates with 32/256 unchanged rows use 7.54/8.57% fewer allocation calls
and 1.60/4.05% fewer requested bytes. The gain exceeds deterministic variation
and the frozen 5% target; all CPU, requested heap and native RSS controls pass.
Held/peak heap is unchanged in the primary cases. CPU differences are below
observed variation, so no CPU percentage is established. Scene/hitbox hashes,
state-owner counts, atomic-claim and retained-frame regressions, strict owning
Clippy and benchmark isolation pass. Both lanes link identical immutable
dependency objects. See `performance-evidence/retained-layout-sets-20261008`.
Qualify coherent application adoption separately, then continue with the next
distinct important owner and bounded renderer work.

Fincode adopted layout replay at owned rendering `861e0d8570cb179c1cb357d770cc58530c2dd312`
and component `7eec25934e487e684a96326e212f690f6a4d4aa8` in
`6984a63f579c266e6c9188ff38f9eba4d0c26d30`. Separate native retention on/off,
marquee/census, scoped production compilation, strict UI/component Clippy and
guards pass. Every original registry record is preserved after lock regeneration;
merged changed bytes and the combined tree are verified. No app/GPU gain is inferred.

Distinct state-only reconciliation at `a946b415c2f5967d36f4f82b8530833c94dee4af`
also remains held. Its ordinary checker body is unchanged and both lanes use
identical native external objects. All 800 children pass correctness, CPU,
elapsed, whole-child CPU and requested-memory controls, with target checker
CPU 79.23–91.94% lower and transient allocations eliminated. Five census
native RSS fields fail their frozen ranges. Preserve its branch and evidence;
never rerun this design unchanged to seek acceptance.

Zero-read nested-stretch omission at `96daa6ebbe402f5eac4867e2d0c73e06cca20eaf`
qualifies across 600 native children and fifteen complete recorder workloads.
Parents with their own dependencies and an empty child share immutable all/own
snapshots: 50% lower held requested heap, 49.98–50.03% lower peak, 36.36–43.75%
fewer calls and 47.95–62.32% fewer requested bytes. Four entity/state/mixed
workloads also use32.44–38.83% less CPU/elapsed above both full lane ranges;
the deadline-only CPU difference is below variation. All frozen CPU, elapsed,
whole-child CPU, heap and native RSS controls pass. Owner-sharing fails before
and passes after; recorder/retained/frame/layout contracts, strict owning Clippy
and benchmark isolation pass. See `performance-evidence/empty-recording-stretches-20261008`.
Setup, warm-up, App capacity and a preallocated output Vec are excluded from
requested census; arrays for 1024 retained outputs are included. This is Linux
recorder evidence, not an app/GPU/native-platform percentage. Qualify coherent
component/Fincode adoption separately, then continue with the next important owner.

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

Fincode adopted zero-read stretch omission coherently in
`b9c8335e1b2258894e063488829f81045e47bb74`, with owned component
`1c23db7f78e41440acddb2ee7174b0bd431f6dd9`. Scoped native integration,
retention on/off, strict UI/component checks and original registry-lock records
pass; merged changed bytes and the combined tree are verified. Subsequent
independent adoption currently pins rendering `b64fd9389c58a28048fa7c52d56447e73feffd89`
and component `d8e4aa36aebaee078574ea83e9b630ca5b0e6a80`; preserve those changes.

Empty-right state sharing at `b5bd9e8fae0809bf9af4da0b96798b78e316ae40`
remains held: eight allocation targets improve 7.02–9.05%, but three native RSS
controls fail. Unbounded linear read merging at
`12fdb142f4db5be22b8c5ffbcd0e871ef836011f` also remains held: four CPU targets
improve 11.42–13.13%, but nine normal CPU/RSS fields fail. Its 560 valid children
and 560 invalid children from an earlier overlapping measurement are preserved
under `performance-evidence/linear-read-union-held-20261008` on its branch.
Never repeat these designs unchanged or weaken their frozen gates.

Bounded out-of-line read merging at `eff2bd53e09225062f725f3e84c564445b7444c0`
qualifies across 560 native children and 20 complete App/Window workloads. Four
256-row changed-component cases use 9.41–11.96% less thread CPU and 9.47–11.98%
less elapsed time, exceeding both full lane ranges and the frozen 5% target.
All normal CPU/elapsed/whole-child, requested heap/churn and native RSS
controls pass. Every held/peak/requested-byte/allocation-call median is
unchanged; no memory or RSS percentage gain is established.

The original small-list algorithm and empty-input sharing are retained;
only unions of 64 or more combined reads enter a non-inlined linear kernel.
438,048 read/floor combinations, boundary cases, native retained output,
dependency changes, wheel scrolling, live-row panels, layout/atomic-claim and
nested contracts pass. Strict actual owning Clippy and benchmark isolation
pass. Both lanes use the same 59 immutable native external objects, with
separate bare and counting-allocator measurements. Receipts prove sequential
phases; no failed cohort was repeated unchanged. See
`performance-evidence/bounded-read-union-20261009` for all raw samples,
identities, variation and coverage limits.

These are Linux complete-component results with a fake renderer and System
allocator, not application, physical GPU, macOS/Windows or shipped-allocator
percentages. Qualify coherent component/Fincode adoption separately, then
continue with representative transcript/streaming, scrolling, editor/Review,
multiple-panel and background-window bottlenecks. Bounded GPU scroll caching
still needs renderer-memory, compositing and reclamation evidence.


In-place glyph ranges at `e18490e74befe5c2bd24de8137aae5b58235b177`
remain held. All 3,418 native children and 20 memory targets pass, but 18
CPU/elapsed/whole-child/census RSS fields fail. All bare RSS controls pass.
Native output, retained histories and owner release pass. Never resample the
source unchanged or weaken its frozen gates. See
`performance-evidence/in-place-glyph-ranges-held-20261009`. Continue with a
distinct publication marker inside glyph descriptors to remove the Scene-wide
header and unnecessary non-glyph sealing. No consumer or app/GPU gain.
