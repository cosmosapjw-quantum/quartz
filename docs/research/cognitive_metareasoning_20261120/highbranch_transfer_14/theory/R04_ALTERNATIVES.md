# R4. Constructive alternatives and selected research

Option A selected: four u64 root-mask words cover indices 0..224. Full candidate edge arena allocated at root construction, first 8 MCTS visits limited to an initial top16 subset, optional rules-only reopening at checkpoint, then the SAME root reaches total32. Original priors, TT keys, child legality, accumulated N/W/M2 unchanged at activation. This measures selection intervention but saves no up-front edge memory; include full expansion cost.

Option B postponed: physically append newly allocated root edges to a fixed OnceLock arena slab, which risks lock-free pointer publication and transposition-table ownership and is not needed to test decision utility.

Option C existing progressive widening config: changes root and descendants simultaneously, so it is not a clean equivalent to root-only attention. Option D Rapfi NNUE: use the genuinely trained external alpha-beta engine solely as independent reference, not as falsely native ONNX prior and not as a perfect solver.

Precommitted source strata: neutral reproducible legal openings and separately constructed immediate-win tactical stress cases. Do not pool their effects as representative random games. Primary freestyle rule0; separate Renju rule4 legality check. Compute budgets K=16, reopened r≤4, 8 +24 additional visits; root forced-win shortcut disabled when testing the discovery mechanism.

Use evidence priority: omitted legal immediate winning moves may bypass fixation threshold (a true immediate win is a deterministic game-rule witness); if no win exists, an exposed initial leader that permits opponent immediate win may trigger safe candidate opening under a declared condition. Count activated moves, selected moves, Rapfi agreement, immediate-win action rate and end-to-end CPU separately. Stop without tuning if the full-legal baseline remains equal or stronger.
