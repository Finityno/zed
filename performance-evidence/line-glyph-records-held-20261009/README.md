# Held line-sized glyph records

This design remains unmerged. All 3,418 alternating native children across
121 complete Window workloads and three standalone Scene controls are preserved.
Twenty transcript and colored-text targets reduce paint-record capacity by
75–96.875% and complete App/Window live requested heap by 10.235–28.466%.
All frozen primary targets, requested held/peak/byte/allocation controls pass.
The predeclared cost is eight bytes per Scene, sixteen per Window; standalone
owner fixtures release all requested live bytes after drop.

49 fields fail: 15 ordinary thread CPU, 15 elapsed, 14 whole-child CPU,
four current RSS and one peak RSS. 48 failures occur in bare timing and one
in counting-allocator RSS. This design is held without PR, merge, adoption,
unchanged repetition or weakened gates. Retained colored replay rebuilding
records then compacting them is a candidate follow-up mechanism to avoid,
not a measured attribution of every failed field.

624 common native test passes, 278 correctness children, 121 matched per-frame
audits and six separate empty-Scene release children pass. The cold-line owner
regression is verified as exactly one failing test before and one passing test
after. Published prefixes, independent payloads, clipping, movement, animation,
rebase, layers and native-surface/overlay contracts pass. Compact overlay indices
are projected to expanded logical glyph boundaries in both audited lanes;
original compact positions and counts remain in owner receipts. The two zero-test
filters are explicitly identified as gaps, not successes.

Frozen policy SHA256: f52e2f0170b07edc7470cf606352e65980684e832b5b165614fc1c4b5c09d442.
Production source: 74c131c3c0185ce1f1955f44e7ac2809b13745a9.
Owned baseline: 08d5555709d7afab3936c7afcd9f7f0ff9ecbd65.
Frozen Fincode source: 55943cc3fe70d861b86b34da03e34ad6562889af.
A later refresh observed 8812bc6b2027c445b3d05307e49836dad813dbce;
its three rendering pins still match the baseline and its component remains
ab07cd7695af1dfd9f1c6ad06a77660d288e568d. Independent consumer work is preserved.

The first fixture compile failure, original prototype setup failure and pre-close
fixture/binary identities are preserved separately. The final four binaries,
compiler, System allocator, runtime, exact private sources and all 59 immutable
external objects pass before/after checks. No timing overlaps a build, compression
or other profile. Counting CPU is not used as CPU gain evidence.

Archive SHA256: 3e74761bc929292bc8bde3b933b02f3e2b327c73a3361344a35a4322e365a97d. All 5148 members resolve to original exact bytes;
only identical content and metadata share tar hardlinks. Large native/dependency
objects remain identified in the managed cloud, with inactive-cache preservation
receipts where applicable. This is Linux GPUI fake-platform/text with System and
a separate System counting allocator. No physical GPU, application, macOS,
Windows or shipped-allocator gain is established. Owning Clippy is not claimed
after failed performance gates. Continue with a distinct cached-line-only encoding
that preserves cold/decorated painting and published boundaries.
