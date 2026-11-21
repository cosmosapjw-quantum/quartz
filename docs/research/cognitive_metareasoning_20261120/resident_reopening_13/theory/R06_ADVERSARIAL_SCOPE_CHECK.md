# R6. Independent mathematical counterchecks and exclusions

**Check 1: historical counters.** `OnceLock` and slab pointer immutability do not alone prove score/counter preservation if PUCT performs a hidden root refresh. The concrete API must forbid root reset/rebuild, mutate only the atomic selection mask and invoke `MctsEngine::run` with visit limit 24 on the same object after 8. Confirm unchanged root hash, unchanged edge prior values, exact per-edge `(n,w,m2)` at the instant, and engine evaluator-call count unchanged by mask activation.

**Check 2: legality.** Root gating is a selection choice, not a state move-rule change. The `Connect4::legal_moves()` must remain identical to unmasked game state for any descendant; root action index is column0..6 and forbidden column must never be chosen while inactive. If `M_0=A` (≤3 legal moves), all reopening methods must be no-op and behave identically to all-legal.

**Check 3: selection under disallowed edge first.** Existing `select_core` initial `best_idx=0` and `idx==0` best replacement logic can accidentally choose disabled edge0. Patch must use an explicit `found_any` and skip inactive edges prior to snapshot/score; test a gate omitting the first materialized candidate, not just leading top3.

**Check 4: prior mass.** For `P=(1/2,1/3,1/6)`, activated subset first two has `ΣP=5/6`, which is not one. We intentionally preserve original full prior rather than renormalize. A "new optimality" proof conditioned on normalized active prior would be inapplicable. Compare full-root baseline under the same original prior and evaluator.

**Check 5: finite zero-visit/deadline.** Root visit 8 and 24 are exact completed-iteration counts when n_threads=1 and terminal root excluded. For a terminal child, game-theory value is a deterministic +1/-1/0 with current-player negamax; root WDL outcome interpretation comes **only from external oracle action score**, not immediate tactical witness. End-to-end wall time includes root evaluation/probe and may be very small (~sub-ms); NPS rounded integer milliseconds must not be used as cost.

**Check 6: causal scope.** An artificially designed early top3 mask produces a manipulable deficit relative to unrestricted MCTS. A positive result against the restricted sham is not evidence over all-legal. Two test groups/24 opening families are correlated synthetic finite samples, not IID Monte-Carlo search-seed replications. Don't infer a no-go theorem from a null comparison or transport human Einstellung findings to machine search.

**Verdict:** The finite invariant is valid under the explicitly testable serial ownership assumptions. Algorithmic superiority is unresolved. Highest-risk code finding is "invalid gate may select index0" and "reverting root memory state corrupts TT"; these require targeted TDD before publication.
