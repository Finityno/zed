# Held window-owned effects reclamation

The one frozen 2,296-child comparison across 82 complete App/Window workloads
passes all eight memory targets but fails 18 other fields. Complete-App held
heap falls 58.74–75.66%, queue ownership falls 87.5–93.75%, and native current
RSS falls 21.39–33.41%. Five CPU/elapsed/whole-child controls, four census RSS
controls, eight primary allocation-call budgets and one active background
requested-byte budget fail. Preserve this design unmerged; never repeat it
unchanged or weaken its gates. No PR or adoption is qualified.

The existing window callback waits for every window to be quiet for 30 seconds
before shrinking an empty oversized queue to an 8,192-effect reserve. App and
Window sizes are unchanged, with no enqueue hooks or independent App task.
Rearming still adds 19 calls per window including the resize, against eight
budgeted calls. Active background bursts that do not draw add 18.29 MB of
requested refill bytes, above the declared 3.28 MB allowance. These costs are
included, not hidden by skipping the intermediate two-second callback.

Thirty-six native contract invocations pass, with four real lifecycle regressions
red on baseline and all eight green on candidate in bare and census lanes.
Ninety-six functional checks and all 2,296 paired callback, scene, hitbox,
state and owner outputs match. Every source/compiler/runtime/common 59 native
extern/binary identity and all frozen inputs are reverified after the sequential
build, bare and census phases. All raw samples and original setup failures
are archived and every member hash verified. Exact binaries remain durable.

The ten current-main ownership children confirm a drained 9 MiB effects queue
with no retained closures. Linux fake-platform/System measurements are not
Fincode app, physical GPU, macOS/Windows or shipped-allocator percentages.
No-window, already-quiet-before-burst and last-window-close reclamation are
outside this window-scoped design. Continue with the measured task-rearming
allocation overhead as a distinct candidate; preserve this failed cohort.
