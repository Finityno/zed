# Empty nested dependency recording

A recording with nested scopes but no entity, global, state or deadline reads
keeps the same all and own dependencies. It now shares those snapshots instead
of allocating their three empty reference-counted headers twice. Nonempty paths
and the app's field layout are unchanged; no cache is added.

Measured production baseline: `ef46053b5794ccb57cda5dc658581ca2a399a4e2`.
Measured candidate: `16295846a6b12ef6066aa4bff6c1f145c648cd29`.
Both builds contain the identical Linux-only recorder fixture, including a
nested-deadline control. The isolated workspace keeps the exact owning crate
sources and a locked dependency closure. Normalized compiler arguments match;
all direct external artifacts except the respective `gpui_platform` build match.

Rust 1.99.0, x86_64 Linux, System allocator, opt-level 3, codegen-units 16 and
debug assertions enabled. These are recorder component results, without a
window, text shaping, GPU work or installed Fincode process. The counting
allocator is compiled separately; its timing is not an acceptance metric.

| Empty nested workload | Baseline | Candidate | Change |
| --- | ---: | ---: | ---: |
| Bare median thread CPU, ns | 73,424,035 | 46,820,054 | -36.2% |
| Held requested bytes, 1,024 outputs | 98,304 | 49,152 | -50% |
| Peak requested bytes | 98,304 | 49,152 | -50% |
| Total requested bytes in census | 294,912 | 196,608 | -33.3% |
| Allocation calls in census | 18,432 | 12,288 | -33.3% |

Use decision.json for the exact timing medians; the table's rounded comparison
is descriptive. Baseline CPU ranges from 73,270,075 to 79,684,302 ns and candidate
from 46,557,442 to 47,061,504 ns; elapsed ranges are also disjoint. Fifteen
alternating fresh-process timing pairs and five separate census pairs cover ten
workloads: all 400 native children exit successfully with matching signatures.
The bare loop records 262,144 iterations with retirement each 1,024 outputs,
then retains a final 1,024 outputs. The census records 1,024 iterations and the
final held batch; app construction, warmed log capacities and final held-batch
release are outside its measurement interval.

All CPU/elapsed and native peak-RSS increases fit the predeclared full baseline
max-minus-min range. All nonprimary requested heap/bytes/calls remain exact.
Positive CPU controls include entities-16 +2.53% and deadline +0.83%, accepted
inside baseline variation; that does not establish zero effect. Native RSS
medians have positive changes up to 3.10% in bare controls, also inside their
observed baseline ranges. No whole-app CPU, RSS, macOS or Windows gain is claimed.

Two new ownership/deadline contracts, the 16-seed retained-frame oracle,
entity/global invalidation tests, benchmark-feature guard and strict owning
all-target/all-feature Clippy with pinned Rust 1.98.1 pass. Initial build setup
failures and newer-compiler Clippy warnings from the earlier pool experiment
remain preserved on that held branch. That pool design failed a nested census
RSS gate and is closed/unmerged; it was not resampled unchanged.

The committed receipt archive includes every raw child, frozen policy,
controllers, complete owning source snapshots, build/check logs and exact
compiler/dependency/binary identities. The four executable binaries are retained
in the managed workspace under /workspace/scratch/gpui-nested-loop, identified
by binary-identities.json; the public archive excludes executable bytes.
Fincode must adopt the merged owned-fork revision before this affects releases.
