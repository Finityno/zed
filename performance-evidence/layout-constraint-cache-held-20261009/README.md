# Held layout constraint cache upgrade

The complete nested-settings update uses 34.83–37.54% less native thread CPU
and 34.83–37.54% less elapsed time across four 16/80-row retention-on/off
workloads. Each gain clears the frozen 15% target and both full lane ranges.
The design remains held because 55 frozen controls fail. Do not repeat it
unchanged or weaken the controls; no PR, merge or application adoption occurs.

The failed fields include current native RSS, resize peak requested heap,
peak native RSS and one retained 80-row flat-header CPU/elapsed/whole-child
workload. The largest current RSS changes exceed the explicit node-cache
budget by more than one MiB. Controlled pure-tree ownership costs 40 bytes
per node slot with no extra allocations and releases all bytes on drop;
that preflight did not predict these complete component RSS/peak costs.

2352 alternating native children cover 84 workloads: 1512 bare children in
231.30 seconds, followed by 840 separate census children in 65.85 seconds.
620 final native test passes and 168 per-frame audit children pass; all paired
terminal scenes, state owners and node counts match. Two zero-test patterns
are recorded as no coverage. The original 80 CPU attribution children and
all earlier functional/setup/parser results are retained in the archive.
The public Window layout clock misses direct engine calls; a distinct private
all-engine diagnostic identified the nested layout bottleneck. That clocked
attribution is separate from the final bare timing.

Both lanes use compiler 1.99 and identical 58 other native externs and three
Taffy transitive objects. The intended dependency changes from 0.13 to 0.14,
including its new default balanced-flex feature and leaf-layout API adapter.
Exact source, flags, runtime, allocator and package identities are recorded.
Actual isolated Cargo regeneration validates the seeded dependency record;
1828 other lock records remain identical. Owning locked Cargo, strict Clippy,
source guards and adoption qualification were not run after failed gates.

The evidence is native Linux GPUI with a fake platform/text system and System
allocator. It establishes no whole-application, physical-GPU, macOS, Windows
or shipped-allocator gain. Preserve the independent arena, frame-demand,
atlas and glyph-replay changes. Continue with a distinct layout-owner design
that retains the existing dependency, followed by separate consumer checks
only when a candidate qualifies.
