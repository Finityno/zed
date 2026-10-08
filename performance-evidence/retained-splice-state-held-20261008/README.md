# Held retained-splice state release

The removed-element native regression fails at owned revision
`c295ae14662edcb2b6e6cfd1d1861dd0b1feab11` and passes at candidate
`c3e601eed133e68f2950b53062cb6d4aaa19aeb7`. Rebuilt gaps stop claiming
obsolete element state while preserving replayed siblings.

The candidate is **unmerged**. Across 280 native children and ten complete-frame
workloads, it releases 100% of removed owners (32 and 256) and decreases held
requested heap by 10,240 and 81,920 bytes. All scene/hitbox signatures match.
Seven native contracts, strict owning Clippy and benchmark feature isolation pass.
The independent-root regression also passes before the fix; this fork already
has a safe equivalent for that specific source interaction.

Eleven frozen controls fail: two retention-disabled CPU controls, five census
RSS controls and four requested-byte/allocation-call controls. Reappearing rows
correctly recreate their removed state, increasing churn by 13,824/110,592
requested bytes and 192/1,536 calls over eight iterations. These adverse results
remain part of the evidence; this source/design must not be rerun unchanged or
accepted by weakening the gates. Dependency artifact path effects are unisolated.

`evidence.json` records every gate, raw artifact identity and coverage limit.
`receipts.tar.gz` preserves the complete raw cohort, sources, build/check receipts
and preparatory failures. This establishes a retained-owner bug and a held fix,
not a qualified application or GPU performance improvement.
