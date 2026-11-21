# R4. Constructive counterexample to confusing rate and net-value dispatch

Take three possible hidden states θ1,θ2,θ3 with b=(1/2,3/10,1/5). Three terminal actions a_i get utility U(a_i,θ_j)=δ_ij. Thus without a new computation the optimal terminal action is a1 with expected utility1/2 and risk R=1/2.

Two valid computations:
  c_partial: reveal only whether θ=θ3 (binary outcome), k_partial=1/20. If θ3 true, choose a3; otherwise choose a1. Expected terminal utility after observation=1/5+1/2=7/10, so **G_partial=1/5**.
  c_full: reveal θ exactly (three outcomes), k_full=1/5. Expected terminal utility=1, so **G_full=1/2**.

One-step value per resource: I_partial=(1/5)/(1/20)=4; I_full=(1/2)/(1/5)=5/2. Thus *max ratio* selects **c_partial**.

With λ=1 utility/resource, the *net* one-step improvements are N_partial=1/5−1/20=3/20 and N_full=1/2−1/5=3/10. Thus *max net* selects **c_full**, despite both computations passing the stop threshold. This exact finite example is a **derived scope-restricted counterexample to ratio-vs-net dispatch equivalence**, not a claim no rate-index method can ever be useful.

One-step stop equivalence is preserved: G_c/k_c > λ iff G_c−λk_c>0 whenever k_c>0. It would be erroneous to claim the ratio rule necessarily stops at the wrong time for H=1 merely because it orders computations differently.

Alternative governing objectives:
  (A) fixed wall-budget optimal terminal risk: min E R(b_H) subject to Σk≤B, explicit resource state;
  (B) priced metalevel Bellman as R3 (adopted for this theoretical reference);
  (C) rate index as an approximate anytime throughput scheduling rule (retain as baseline, not promoted);
  (D) heuristically tuned human-like O-metric objective (rejected by frozen THESIS P3 non-circularity constraint).
These are distinct problems, not equivalent by notation. The next discrimination computes all four on frozen finite examples without game-specific features.

Independent symbolic check: Wolfram rational arithmetic computed base1/2, expected post partial7/10, full1, gain {1/5,1/2}, rate {4,5/2} and net {3/20,3/10}, confirming reversed choices. Source and original CAS result preserved in R7 checks. A finite exact emulator and independent Rust numerical implementation must reproduce the same reversal without tuning parameters to it.