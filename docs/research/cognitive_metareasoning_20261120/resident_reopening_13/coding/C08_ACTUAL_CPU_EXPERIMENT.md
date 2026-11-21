# C8. Actual 9216 released Rust resident-root search experiments (frozen 384)

**Recorded source:** committed real Python runner `bf5bda76ba817476f635a8d245603d8c95949211` with 5 behavior tests RED→GREEN plus prior C1–C7 immutable inputs. **Compiled real Rust release engine:** source `8c63a60973ac11b9e1fa4cdf93d387083c65ee57`, binary SHA256 `5dd61a01cc87dd715183f3d4145367b471182e0676a9a6bde7c38c9c225eede0`. C13 frozen exact Pons oracle source file SHA256 `73cf3b37ddbcc7f14298ee0634b760f5008bfb2194bd5cd5fc31dc24ceb1a817` was NOT opened until all Rust requests returned.

**Real run:** 384 new Connect4 game positions (24 mirror-disjoint five-ply opening families, A/B each 12 families) × 3 evaluators (Uniform, actual trained CPU tiny MLP, trained CPU Conv) × 8 controller arms = **9,216 real in-process Rust CPU research requests**, each executes first8 and 16 additional visits on the **SAME root MctsEngine**, **18,432 invocations of real MCTS search** (but only 24 accumulated visits per request). Rust child process batch wall ~5.7818s. Root legal 7×6, root-forced-win=false, PUCT c=2, seed17, no Dirichlet, all candidate edges preallocated. Original root priors/TT/hash and each earlier edge N/W/M2 remain exactly unchanged at activation, root visit 8→24; all no-online-solver checks pass in actual endpoint for every result.

**Eight frozen treatments:** resident_closed_sham; resident_evidence; resident_random_{1,2,3} with predeclared candidate hash seeds; resident_all_after8; resident_full24 (unrestricted root all along); resident_closed_no_probe. All regimes use one original full root prior evaluation and real original root setup; exact final WDL utility only posthoc. One-ply tactical evidence triggered on a minority; root without evidence retains original top3 and cannot be claimed improved. Historical root stats reused, no late slab edge insertion.

| Mode (pooled 384) | Closed sham mean exact regret | Evidence mean exact regret | First-unrestricted full24 mean regret | Evidence mean physical end-to-end ms | Full24 ms | Evidence triggers | Reopened strictly-better omitted roots |
|---|---:|---:|---:|---:|---:|---:|---:|
| Uniform | .32552083 | .32552083 | **.2578125** | .246073 | .242839 | 23 | **0** |
| tiny MLP | .22135417 | .10677083 | **.08854167** | .293154 | .292242 | 73 | **22** |
| ConvNet | .19010417 | .09635417 | **.07552083** | .653547 | .681922 | 61 | **20** |

Every mode and both cohorts: **evidence vs full24 mean CPU-cost mismatches <10%**, using measured wall time of original prior + full root setup + 8 visits + probe + mask activation + 16 visits; model-load excluded but reported. Note: endpoint wall timer is sampled **before JSON serialization/transport**, so it does not include final output formatting; do not claim it does. Pure search time and model calls separate. Actual study has high-frequency sub-ms OS variability, not process CPU time or hard deadline.

**Primary all-legal comparison:** evidence−full24 pooled exact WDL regret:
- Uniform **+.06770833**, family bootstrap 95% **[-.015625,+.153646]**.
- MLP **+.01822917**, family bootstrap **[-.013021,+.052083]**.
- ConvNet **+.02083333**, family bootstrap **[-.013021,+.059896]**.
Positive point differences favor **unrestricted full24** (lower regret). Every confidence interval includes0; **NO established superiority of selective resident reopening over all-legal.**

**Within engineered top3 mask only:** evidence−sham pooled exact regret:
- Uniform **0**, CI [0,0]; posthoc strictly-better omissions 43, recovered0.
- MLP **−.11458333**, bootstrap CI **[-.15104167,−.08072917]**, 34 roots with superior omitted legal action, 22 were actually reactivated, final chosen WDL **improved in 24/384**, never worsened.
- ConvNet **−.09375**, bootstrap CI **[-.13541667,−.0546875]**, 29 eligible and 20 reactivated, final WDL **improved in 19/384**, never worsened.

Three equal-count oracle-blind random root-activation arms also executed. Pooled MLP evidence−random regret deltas **−.07552,−.07552,−.08333** with 24-family conditional intervals below0; pooled Conv **−.06771,−.06771,−.046875** also below0; all pooled means actual CPU differences <10%. This supports *targeted recovery conditional on artificial top3 masking*, not a generally stronger MCTS. Randomization seed changes candidate reopening, **not independent Monte Carlo search tree trajectories**.

**Group check:** A/B separately MLP evidence regret .104167/.109375 versus full24 .072917/.104167; Conv evidence .119792/.072917 versus full24 .083333/.067708. All group full24 cost differences <10%. Uniform remains negative. **252/384** test root positions have genuinely different exact WDL classes among legal actions; the others are saturated for sign(WDL) regret.

**Status:** real same-root state and same-root continuation `implementation-verified`; 9216 released Rust root outcomes and source model bytes `numerically checked` by runner. Independent critical C9 audit must rederive tactical rules, hash seed, original/root prior ranking, edge stats and solver mirror before issuing the final claim gate.
