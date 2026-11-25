# Long-game collection: safe preparation and future execution

Current state: DISABLED. Do not start games without separate explicit authorization. The two SMOKE_FIXTURE records are not a corpus.

## Read-only checks
From the exact research worktree:
    D=docs/research/quartz_fouraxis06_20261125
    python3 "$D/collector.py" --dry-run
    python3 "$D/collector.py" --verify --output "$D/C07_SMOKE"
    python3 "$D/audit.py"

These commands do not start neural search or generate production games. The default production output directory was absent at C09. Keep it absent until approval.

## Future pilot, not executed
Choose a dedicated volume with sufficient disk reserve; inspect current host capacity. The current proposed10k plan fails its conservative disk preflight. Do not lower the reserve automatically. Use a separate small plan (16--32 natural games), exact code/model/runtime identity, appropriate output, separate seed and explicitly bounded game count. The disabled plans are examples, not authorization. Use the collector's canonical digest of the new plan, not the raw JSON file hash. An enabled plan alone is insufficient: require exact authorization digest, explicit output/shard and max-new-games. Invoke the locked virtualenv interpreter with PYTHONPATH pointing at this worktree, only after approval. See collector.py --help for the executable interface. Do not use Python -O.

Start with one shard and one worker; after verifying actual throughput, memory, disk and resume, consider independent shards under a measured global resource cap. A shard lock prevents same-shard competing writers, not cross-shard hardware contention. Inspect any orphan pending files; do not blindly delete. Preserve all naturally terminated games, all censored records and their reasons. Never manufacture long games by avoiding legal wins or label cap as draw. Verify exact game checksums, legality and terminal labels before further research. Record model strength and distribution limits; F1 is a smoke checkpoint.

## Scientific validation design
Distinguish complete-game return from exact WDL certificate; not an oracle for other legal actions. Use natural game length CDF, at-risk counts per ply, game/family-clustered uncertainty and separate position-weighted vs game-weighted estimands. Check cross-split D4 state overlaps and exclude contaminated TEST probes. Independently construct arena opening families, since default arena and self-play example seed overlap; do not use same openings as a heldout arena without explicit separation. Future arena pilot calibrates candidate_visits against baseline visits on TRAIN actual per-move transport and native/CPU cost, then freeze before TEST. Swap colors, keep opening family and paired seeds, report complete pairs, censored pairs, WDL, mean/p95 time, NN counts and uncertainty. No claim of Elo or equal time from smoke.
