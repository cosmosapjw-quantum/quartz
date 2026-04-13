# Transposition Table Notes

## Current Design
- Striped lock TT with shared node entries
- `max_tt_size` parameter (default: 100K entries)
- When TT is full: expansion stops (crude but safe)

## Exactness Notes

The TT key used by search is now `tt_hash()`, not plain board `hash()`.

That matters for history-dependent games:

- chess includes repetition-sensitive rule state and halfmove-related draw state
- go includes superko/history-sensitive rule state

This avoids the older failure mode where two positions with the same board hash
but different legal-future semantics were merged in the TT.

For chess, exact repeated search also depends on the returned history-token
metadata path. A raw external FEN is still insufficient to recreate prior
repetition state by itself.

## Known Limitation
No eviction/aging policy. Options for future work:
- Bucket-local LRU with visit-weighted priority
- Age-based: tag entries with search generation, evict old ones
- Retain root subtree on advance_root(), evict outside subtree

## Memory Budget (Gomoku 15×15)
- MctsNode: ~120 bytes + edges (~64 bytes each)
- 100K entries ≈ 12-20 MB
- 64GB RAM: plenty of headroom

## Impact on Ablation
For fixed-budget ablations (HaltMode::Fixed), TT size is not a confound
as long as all modes use the same TT config. The TT limit only matters
for very long searches or large games (Go 19×19).

Current recommendation: keep TT enabled in normal experiments and compare
controller changes under the same exact-hash policy.
