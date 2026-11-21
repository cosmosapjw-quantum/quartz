# R7. Executed Wolfram and independent exact Fraction mathematics

**Wolfram Language exact rational evaluation** (stateless kernel) of b=(1/2,3/10,1/5), identity-payoff matrix, binary partial observation "θ3 or not" and full ternary observation returned:
- prior best terminal expected utility 1/2 and Bayes risk 1/2;
- expected terminal utility after partial observation 7/10; after full 1;
- gross EVSI (1/5,1/2), cost (1/20,1/5);
- normalized gross-gain/cost index (4,5/2);
- priced λ=1 net gain (3/20,3/10);
- exact ratio selects partial while net selects full, both with positive one-step net value.
This was a true direct evaluation, not a claimed literature theorem.

**Independent executed Python Fraction evaluator**: theory/verify_R07_exact.py, exact output theory/R07_EXACT_RESULTS.json, status PASS. It independently constructs the full matrix, conditional channels and exact Bayes updates; verifies 3-state/3-action gain, costs, ratios, net gains and 0–3 horizon Bellman recursion. Exact V_h values are h=0:1/2 STOP; h=1:1/5 choose full; h=2:1/5 choose full; h=3:1/5 choose full. It checked all 6 terminal-action permutations and 6 hidden-state permutations, null-observation EVSI=0, perfect-channel EVSI≥binary-partial EVSI≥null EVSI, and cost scaling by1000 with λ divided by1000 leaves the H1 Bellman decision invariant.

The Wolfram tool call for the optional stop equivalence produced a non-evaluated conjunction due to a syntactic expression choice; it is not counted as an independent CAS PASS. The exact one-step stop predicate equivalence was instead proved in R5 algebraically and verified for both computations in Python Fraction. No neural MCTS sampling, gameplay, human cognitive or hardware-speed claim is certified by these exact tests.

Provenance: source exact file identity and finite arithmetic inputs are in the generated R07_EXACT_RESULTS.json. Test status is restricted to enumerated finite cases. Derived R5 lemmas do not become novel merely because they passed these checks.