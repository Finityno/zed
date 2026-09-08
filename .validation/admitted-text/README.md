# Isolated admitted-text validation

This is source scaffolding, not an executed calibration. Do not run its Cargo or native commands without the coordinator's explicit slot. It depends on the actual normal GPUI libraries with an opt-in validation feature, not their full unit-test/dev-dependency graph.

`python3 check_contract.py` first checks positive-control with the exact same feature set as the negative bins. Every negative case must produce the expected diagnostic at its marked source line; unrelated errors fail validation. Full stdout JSON and stderr logs are retained.

`ownership` calls the actual source checks and the same private cache operation used in production, plus denied-source allocation-site observation. `native-calibration` defaults to scratch denial and asserts that the actual backend's native-entry probe was never called. `--measure-native-unbounded` is a deliberately UNCALIBRATED, harness-only collection mode. Its zero scratch token is not a bound and must never be copied into production policy. The initial fixture is one 4096-byte ASCII line in Menlo at 11.5 pixels. It neither opens a window nor attaches to Fincode.

`--hold-for-tools` keeps only the isolated fixture process alive after source/layout release, awaiting one line on stdin, so the parent can collect its full allocation history. The future runner must retain the exact child PID returned by spawn, use that PID for tools, and release/terminate only that child. It must never use process-name matching.

Source scaffolding does not yet include the reviewed corpus/OS/font-fingerprint manifest, autorelease-pool phase boundaries, trace event parser, independent malloc/VM completeness sentinels, or a production measured allowance. These remain required before a native calibration claim.

The next unrun ownership fixture stage additionally fills all 128 cache slots, checks that two requests for the 129th source return separately charged uncached layouts, and verifies that one of two physical lines retires while the other is reused. A wrong-source shaping result must leave the cache empty. Glyph growth uses the actual `TextAllocationReservation::grow_glyph_buffer` helper now called by CoreText: denial and overflow must preserve pointer/capacity/content/charge and skip the actual allocation-site probe; a later admitted retry reaches that probe. The fake shaping closure used for cache topology is not offered as native allocation proof.

Proposed next granted command is `CARGO_BUILD_JOBS=2 cargo build --manifest-path .validation/admitted-text/Cargo.toml --bin ownership -q --message-format=short`, followed only by `.validation/admitted-text/target/debug/ownership`. This is one normal-library executable, not GPUI's unit-test harness, and does not construct a native backend or invoke CoreText. Do not build or run native-calibration under this proposed slot.

## First native case source stage

The native binary now requires `--features native-probes` at build time. Only that feature invokes the fixture-only clang/Objective-C shim; normal ownership/compiler-control builds do not compile it. The shim supplies scoped autorelease pools, all-zone snapshots, bounded malloc/calloc/realloc/default-zone/anonymous-VM/CFString sentinels and fixed marker allocations. Validation-only backend callbacks record actual output font names and vector capacities, or an unregistered-font failure, without changing normal ownership.

The reviewed native argv is exactly `--measure-native-unbounded --hold-for-tools --case ascii4096`. Nine bounded phase handshakes bracket sentinels, font registration, denied entry, cold/warm publication and cleanup, and backend release. Identity lookups happen only after final measured-interval capture. No arbitrary input or staircase is accepted by this executable.

After a separate root grant, `python3 .validation/admitted-text/run_native_case.py --output <new-directory>` launches only its own exact child binary/PID. Child-only NoCompact logging is configured in a copied environment. Each phase's tool capture has a combined 60-second deadline; output is capped at 64 MiB per file and 192 MiB total. It snapshots heap/vmmap at each pause, captures full event/high-water history once at the final checkpoint, then releases its child. This avoids repeated copies of the same complete history. Failure or timeout stops only the runner-owned child/tool processes and does not switch profiler or increase input size.

The runner merely checks sentinel address presence as an initial failure gate. Allocation/free/realloc lifecycle matching, stack coverage, operation-interval reconstruction and native retained ownership remain explicitly PENDING in result.json even if every tool returns successfully. native_peak_bytes stays null and native_allowance stays UNMEASURED. No source/GPU/native cap or pin claim follows from a successful capture.

The runner now also requires `--source-manifest <pre-build-manifest.json>`. It verifies each frozen source hash before spawning the child and rejects a manifest newer than the built executable. The build coordinator must capture this manifest before the granted build; the source freeze manifest is recorded in the phase scratch directory.

A separate monitor remains active during every tool capture. It samples only the owned child's RSS via exact PID, and walks the entire artifact directory including StackLogging files. Either a 512 MiB sampled RSS reading or 512 MiB logical artifact size aborts the synthetic run and terminates only runner-owned processes. The monitor uses a 250 ms interval plus sampling cost, with a 2-second sampling-command timeout. These are safety guards, not production allocation bounds: between-sample peaks, open-unlinked files, sparse-file physical allocation, filesystem compression, and storage outside the configured artifact directory are not inferred from these readings.

Each operation's autorelease pool is now drained while the Rust result is retained, before the published checkpoint. Dropping that result before the next checkpoint separately measures retained source/glyph-owner release. Native/backend residual remains explicitly unresolved.
