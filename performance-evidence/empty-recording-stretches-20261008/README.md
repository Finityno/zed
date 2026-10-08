# Qualified empty-read nested recording omission

Do not append a nested dependency stretch when it contains no entity, global,
state or deadline reads. Such a child excludes nothing from its parent; the
existing no-nested path can share the parent’s immutable all/own arrays. No
field, cache or pool is added, and read-bearing siblings remain excluded.

All 600 native children across fifteen complete App recorder workloads pass
the frozen gates. Five parent-owned empty-child fixtures use 50% less held
requested snapshot heap, 49.98–50.03% less peak requested heap, 36.36–43.75%
fewer allocation calls and 47.95–62.32% fewer requested bytes. The gains exceed
both full lane ranges and the declared40% heap/20% churn targets. Every
CPU, elapsed, whole-child CPU, heap and native RSS control passes.

The owner-sharing regression fails before and passes after. Four recorder
contracts and six retained/frame/layout contracts, strict owning Clippy and
benchmark isolation pass. Both native lanes link59 identical immutable
external objects and use the same added fixtures. Exact source/compiler/
runtime/allocator/binary identities, raw samples, commands and expected
baseline failure are in the verified archive. Binaries remain in cloud storage.

Requested census excludes App setup, warm-up and a preallocated output Vec;
held arrays belong to1024 retained result records. Bare timing and counting
census are separate. This is Linux component evidence, not whole-app,
physical GPU, shipped allocator, macOS or Windows performance. Empty replay
stretches are unchanged. Coherent component/Fincode adoption is a separate
qualification, then continue with another distinct important owner.
