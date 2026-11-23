# C02 — formula, data and code map

| Mathematical object | Runtime source / operation | Independent evidence |
|---|---|---|
| Legal root and player signs | Replay public history through unchanged GameState; validate before apply_move | Python board rules and reachable-state enumeration |
| u_a=(v_a+1)/2 | Uncached recursive minimax of one root child; root-player max/opponent min | Independent Python negamax and exact node count |
| Generic recognition observation | Four SplitMix64 uniform playout paths per nonterminal child | Independent Python RNG/cue replay; terminal cue counted once |
| Known incumbent b | Initial exact solve plus already known terminal children; monotone max on queries | Offline true candidate values, final move ownership, pathwise assertion |
| Categorical p_a | TRAIN-only +1-smoothed feature / remaining / global bins | Saved raw training counts and model rederivation |
| Fixed model charge | TRAIN mean solver nodes by remaining | Held-out residuals versus actual visited-node charge |
| g_a(b) | Categorical expected positive excess | Three-point closed formula and rational oracle |
| z_a | Guarded closed-form expected-excess root | Rational index identity and numerical boundaries |
| Gibbs q | Uniform-prior max-subtracted softmax over STOP and eligible FOCUS | Variational identity, independent trace probabilities and RNG draws |
| Realized net gain | Final exact utility minus actual nodes/time at frozen prices | Paired parent aggregation, archived raw repetitions |
| Original physical mechanism | theory/statphys_check.py plus actual sampled Gibbs | Exact/symbolic matrix identities and distinct runtime implementation tests |

No runtime request accepts an offline u, oracle or reference policy label. The survey endpoint deliberately performs every child solve for TRAIN measurement. Runtime run mode must not perform such surveys secretly. Model lookup and paid initial/query work stay inside each policy's decision timer.

A native constructor and independent Python oracle/harness own separate files but share the frozen JSON contract. Actual production function names and any additive output fields will be recorded at C05, then verified at C06. Existing finite_metalevel and kg_stop remain reference sources, not rewritten implementations.
