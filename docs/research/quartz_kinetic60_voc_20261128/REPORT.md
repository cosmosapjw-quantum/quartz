# K60: state-conditioned continuation value and adaptive STOP

## Scientific verdict
Prespecified performance/cost criterion PASS on fresh held-out NN panel; mechanism control shows no adaptive-specific contribution. K60 state-table policy is behaviorally identical to fixed H4 on all 896 matched episodes. Keep H4 as the simpler scoped reference, not the adaptive table as a promoted general controller. Parent K59 3b3c75af85fa265747e87661b8999459a772fc38, clean/remote equal and parent restore receipt verified.

## Paired continuation estimation
50 of 64 K59 families contained unresolved Q roots, 1470 candidate states in total. Sampled 4 per represented family, 200 roots, outcome-blind. Each of five root choices actually played to termination with same subsequent K59 assigned-h REPEAT policy vs same-model B32, frozen seven checkpoints. 1000 games, 9779 moves, 90916 physical calls (90936 logical), 800 paired comparisons, 781 eligible, 261 changed moves. Conditioning scope is this fixed-continuation, selected occupancy; not posterior optimal value.
|h|eligible|action changed|STOP better|STOP worse|sum gain2|root saved calls|
|---|---:|---:|---:|---:|---:|---:|
|4|200|103|11|5|-12|3069|
|8|200|84|9|5|-8|2269|
|12|191|51|4|2|-4|1478|
|16|190|23|2|3|2|714|
Exact family-weighted signed gain minus .002 times extra root calls, 16-cell table, 8-family support. Independent integer fit recheck 16/16. All possible eligible h4 cells (0/1) choose STOP. High-gap h4 cells structurally impossible since d<=h+1 for >=2 legal moves, and 2*(4+1)<16. For <2 moves normal proof/lock bypass. Unsupported cells continue; no inference from empty cells. Fixed price inherited K37, binning/support experimental choices. No post-test tuning. Added fixed H4 descriptive comparator before any holdout games after observing training degeneracy; appended order for H4 so CPU timing contrasts not confirmatory.

## Fresh held-out arena (64 families, 7 fixed checkpoints, both colors)
All arms separately play identical model B32; no Quartz-v-Quartz or cross-model matches. 3584 games, 31643 moves, 326113 physical/logical calls.
|arm|games|W/D/L|score|Q NN/move|B NN/move|NN reduction|CPU reduction|
|---|---:|---|---:|---:|---:|---:|---:|
|CONTROL|896|731/0/165|0.815848|8.175451|17.251718|0.526108|0.528386|
|REPEAT|896|740/0/156|0.825893|4.239922|17.112389|0.752231|0.745305|
|ADAPT|896|751/0/145|0.838170|1.770601|16.941413|0.895487|0.885669|
|H4|896|751/0/145|0.838170|1.770601|16.941413|0.895487|0.882369|
Primary ADAPT-REPEAT paired score +0.012276786, one-sided lower95 +0.002232143 (criterion >= -.02), extra saving +0.143255958, lower95 +0.129778371 (criterion >= 0). Absolute ADAPT score lower .803571429, savings lower .893183074, CPU positive. Independent stdlib family bootstrap lower95 +.002232143, +.130087045. Two-sided score percentile CI [0,.024553571]; no unqualified superiority claim. ADAPT-REPEAT 273 trajectory changes, 29 improved scores, 18 worsened. Conditional on fixed models/opening generators; no independent new model panel. ADAPT-H4 all 896 histories/scores/physical calls identical; 7779 normalized root receipts identical, every 1431 adaptive attempt stops at h4 (cell0=761, cell1=670); no later decision observed. Distinguish empirical success of early search cap from failure to identify benefit of adaptation.

## Rapfi separate fixed-cap diagnostic
16 compact fresh families, both colors, Q80 and Q96 each vs same Rapfi B128. 64 actual games, 2150 moves, 164283 physical calls. Q80 11/2/19 score .375, saving .450503, CPU .354021; Q96 15/0/17 score .46875, saving .380757, CPU .260428. Score difference -.09375, two-sided bootstrap [-.203125,0]; savings +.0697468, CI [.0143676,.1182087]. Small exploratory panel, no general transfer of H4/ADAPT. The original Rapfi job completed all four shards and failed only at manifest path (rapfi59.py typo); retained failure log, recovered manifest after full coordinate replay and agreement with all shard counts, repaired future script filename; executed source f38eac27 retained.

## Verification and limitations
Total main empirical games 4648, played moves 43572, physical evaluator calls 581312 (NN/Rapfi separately identified; not a speed comparison). 13 separate smoke games and 2172 parity calls excluded. Native K60 opt-in command, source f38eac27, red-green test; new release build and Rust full suite 665 pass/0 fail/89 ignored, existing compiler warnings. 48 old/new real request semantic matches, 12 all-false Q20 and 12 all-true H4 matches, 2 invalid requests rejected. 1000 continuation and 3584 holdout NN games independently replayed, 12281 total proof receipts, 1709419 proof queries, 12061 snapshots; all physical counters and cache verified, 20 train hits, no held-out hits. Six natural exact distinct roots, 30 receipts and 25 completed native decisions independently checked by terminal-only minimax, remaining five native censored receipts retained; all 6 independent solves complete. Rapfi 913 proof receipts, 930830 queries, 18835 waves. Not all NN outputs independently re-inferred. CPU-only, four training and six arena workers, no new NN training/native subagents; no main merge. Formatting preflight twice detected whitespace, repaired before publication. No scientific consequence.

## Drift watchdog and next
Original 15f3a198 README/QUARTZ_THEORY, 7241ad0c THESIS read. P1 regret-reduction per compute remains target; empirical fixed-continuation gain and actual downside support separating P_flip and harm, not calibrated posterior regret or a nondegenerate adaptive controller. P2 original A1_legacy_base comparator unchanged and untested here. P3 emergent signatures open, not optimization targets. P4 CPU benefit measured only in current host.
K61: freeze fixed H4 as simplest benchmark reference. Use actual H4 occupancy and fresh conditional continuation counterfactuals (including high-value rare harmful roots, sampled on predecision features, report weighted population estimand and stratified exploratory enrichments). Derive selective extension from observable backed-up action-value contrast, proof interval and expected positive harm, rather than merely root flip/margin or a globally estimated score mean. Distinguish downside and signed gain; require heldout policy actually selects both stop and continue, compare H4 and Q20 vs identical B32 under frozen NI and incremental compute contract. No tuning on K60 heldout; use disjoint new opening families and genuinely new existing checkpoint panel if available. Rapfi Q96 retained; separate calibrated Rapfi risk/cost study, do not apply Q80 blindly. Original A1/P3 still open.
