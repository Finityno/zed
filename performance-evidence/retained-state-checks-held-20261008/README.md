# Held retained-state check experiment

Status: **failed and unmerged**. Source `caf83a60de3dc429263c81ea70be47f4838c3658` checks dependency states in place while excluding the per-frame marker, instead of cloning and filtering a snapshot. This is the complete App invalidation checker called by retained-splice selection. It adds no persistent cache or owner fields.

The fixed comparison contains 450 bare native children and 150 separate census children, fifteen cases, fifteen timing pairs and five census pairs. Alternating fresh processes run on the same CPU with no concurrent compiler/profiler. Setup and recording sit outside measurement. System bare timing is separate from counting allocation instrumentation. Native RSS uses the child's wait4 result.

| Primary | Thread CPU change | Elapsed change | Median native RSS change | Census allocation calls |
| --- | --- | --- | --- | --- |
| except-only | -79.61% | -79.61% | +244 KiB | 32768 -> 0 |
| except-first-8 | -92.41% | -92.41% | +308 KiB | 98304 -> 0 |
| except-last-8 | -92.59% | -92.60% | +244 KiB | 98304 -> 0 |
| except-first-64 | -90.72% | -90.72% | +228 KiB | 196608 -> 0 |
| except-last-64 | -91.30% | -91.28% | +292 KiB | 196608 -> 0 |

All five primary allocation gates and CPU gain gates pass, with zero final requested retention. **Eight controls fail**: normal-empty CPU, normal-one-state CPU and elapsed, and RSS in normal-eight-state, normal-64-state, exception-last-eight, exception-first-64 and exception-last-64 fixtures. Median RSS increases in the failed cases are 228–292 KiB and exceed the frozen baseline ranges. These results disqualify the design despite its large primary gains. Full medians, ranges, raw samples and exact failed comparisons are in `evidence.json` and the receipt archive.

The three dependency correctness contracts, retained frame oracle (sixteen seeds), scroller-row contract, strict pinned Rust 1.98.1 Clippy through the repository script and benchmark test-support isolation guard pass. Timing uses Rust 1.99.0 with optimization level three and the System allocator. Original crate manifests and the same narrow owning dependency closure are retained. Compiler arguments normalize identically, and transitive source inputs match; raw hashes of several separately built external artifacts differ. The cause and their RSS contribution are not isolated. This limitation cannot erase a failed gate.

No whole-app, GPU, frame-latency, shipped allocator, macOS or Windows performance result is established. Preserve this cohort, do not repeat the unchanged design, and do not weaken the gates to accept it. A future iteration needs a distinct design or bottleneck.

Receipt archive SHA256: `52b5775d44d30c1b8df0739a253bc97cb0c0886ef2305d48d1e4c8e7af9c276e`.
