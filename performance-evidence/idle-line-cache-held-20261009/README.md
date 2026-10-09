# Held idle line-replay table cleanup

All four large idle memory targets pass: 3.36-4.42% less requested live heap,
87.497% less empty-table ownership, and 2.84-4.39% less native current RSS.
Each gain exceeds both full lane ranges. The candidate remains unmerged:
35 CPU, elapsed, whole-child, RSS and cold-refill peak-heap controls fail.
Refill raises requested peak by 259,944-262,996 bytes. Do not resample this
source unchanged or weaken its frozen limits. No PR or adoption qualifies.

The existing idle callback shrinks both replay tables to at least 128 entries
or twice their largest live length. It preserves live line records and adds
no fields, timer, task, draw or hot-paint change. Native App and Window sizes
remain 1,896 and 12,944 bytes. Real empty-map idle regressions fail on baseline
and pass in both retention modes; live sprites and recent-draw quiet protection
pass. All 212 functional and 3,108 paired scene/state/replay outputs match.
Forty-six correctness invocations yield 134 passes plus two expected baseline
regressions. Existing atlas, retained, wheel, layout and frame-demand guards pass.

Twelve owner diagnostics attribute the current empty tables directly, including
the adverse small-owner RSS result. Current-main changes were preserved before
candidate creation. Both lanes use the same 59 immutable externs, compiler,
root and lockfile, with bare timing and counting allocator census run separately.
All raw data, exact source/runtime identities and cold allocation budgets are
retained. Each archive member is hash verified after sequential measurement.

These are complete native Linux App/Window fixtures using fake text, platform,
renderer and System. The 1,024-row exposed viewport is a high-water stress case;
results are not whole Fincode, physical GPU, macOS/Windows or shipped-allocator
percentages. Continue with the distinct retained layout-cache bottleneck.
