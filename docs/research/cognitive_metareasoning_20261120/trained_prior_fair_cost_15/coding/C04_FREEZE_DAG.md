# C4. Cohort freeze, experiment ordering and resources

Source generator freeze_cohorts.py creates 128 train, 32 validation, 48 independent seeded synthetic neighbor-growth test and 16 separately engineered immediate-win diagnostic roots on 15x15 Freestyle. All groups derive independently from a predeclared seed formula; D4 canonical board representations are checked to avoid train/validation/test equivalence, including prior C14 24 frozen boards. Data were generated before any new Rapfi outputs and include no labels.

The 48 neutral positions are **synthetic random neighbor growth**, not human/natural high-level play; no population-level strength claim follows. The 16 constructed terminal-in-one positions are positive controls, not IID outcomes. Neither source generation nor the 13-feature model may be retuned after test labels are observed.

Dependencies: C5 Python model code/Rust leaf evaluator; C6 build and numerical parity; C7 first invoke real Rapfi only for train+validation, freeze θ and commit; C8 first create test-only labels AFTER frozen model Git SHA and run actual Rust CPU arms. No Rapfi executable or weights copied to public artifacts. Preemptive stage snapshots and remote SHA checks are performed by scripts/cog05_commit_backup_publish.sh.
