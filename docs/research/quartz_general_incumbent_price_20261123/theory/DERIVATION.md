# C25 R04–R06: retained decisions and price-aware stopping

Author: native constructor `/root/c25_theory`. This file is on the **shared local container** at `/tmp/quartz_c25_theory.md`, not on the remote Desktop Commander device. Companion exact-check files are `/tmp/quartz_c25_theory_examples.py` and `/tmp/quartz_c25_theory_examples.json` on the same local container. The parent owns integration, independent checking and claim admission. No repository files were changed.

## Scope, sources and evidence status

Selected work: R04 candidate construction, R05 proof, and constructor-owned R06 condition/edge-case checks. Read PhysMath Research Loop 1.3.0 `SKILL.md`, `references/THEORY_LOOP.md`, `references/POLICY_INTEGRATION.md` and `references/SUBAGENT_MODE.md`, plus computation-audit `SKILL.md`. Applied the actual R04–R06 contracts; no recursive review or further delegation.

Read remote worktree `/home/cosmosapjw/quantum/research/quartz_general_meta24_20261123` at observed HEAD `4998c771c08788e4f0da8d2723347121ae16fbf2`, including:

- `docs/research/quartz_general_cheap_certificates_20261123/theory/DERIVATION.md` (the actual derivation path; the initially supplied path without `theory/` was absent).
- The same directory's `NEXT_STAGE.json` and `CODE_CONTRACT.json`.
- `docs/THESIS.md` (original decision-regret principle, Gomoku P2 target, game-specific-injection restriction).

The statements below are directly derived conditional mathematics. The three SPD examples were actually checked with exact SymPy rationals, not floating simulation. This is a constructor check; it is not the parent's independent R06/R07 check or production implementation verification. No literature novelty or new strong no-go is asserted.

## R04: selected construction and competing rules

Keep the finite root-action set fixed and preserve a retained incumbent initialized by the same ordinary score, initial states and deterministic tie break as `PLAIN_STOP`. Refine valid utility intervals and replace this incumbent only upon certified strict improvement. Add two distinct stop modes: a theorem-backed deterministic cell-stencil charge, and an empirical wall-time prediction rule. Both use the same retained action.

| Rule | Mathematical consequence | Remaining trade-off |
|---|---|---|
| Final ordinary-score argmax | No general monotonicity of selected true utility follows from more computation or improved individual scores | Can accept real gains quickly; can also lose utility |
| Retain; switch only if `L_j > U_i` | Every switch strictly increases true utility | Overlap can reject a real gain |
| Retain + charged-work stop | Stops when every admitted paid continuation is bounded above by the current priced value | A sufficient test, not a complete optimal allocation policy |
| Retain + predicted-wall-time stop | Same utility-retention guarantee, subject to valid intervals | Economic stopping is empirical unless a separate cost theorem is available |

The chosen switch candidate is the smallest-index maximizer of the lower endpoints. This is a simple safe choice with one-step closure of available certified switches. It is **not generally the minimax-regret action**: for `A=[1/2,1]`, `B=[3/5,7/10]`, the robust gaps are `G_A=1/5`, `G_B=2/5`, although B has larger lower endpoint. Minimizing robust regret among permitted safe replacements is a possible alternative, not selected implicitly here.

## R05.1: definitions and assumptions

Let `A={0,...,K-1}`, `1<=K<infinity`, be the complete set of possible final actions. Each action has a fixed true utility `u_a`; in the inherited benchmark `0<=u_a<=1`. Computation changes information and numerical state, not this target utility.

At checkpoint t, fresh endpoints enclose the same fixed utilities. Intersect all available valid evidence:

    L_a(t) = max_{s<=t} L_a^fresh(s),
    U_a(t) = min_{s<=t} U_a^fresh(s),
    L_a(t) <= u_a <= U_a(t).

Every intersection must remain nonempty and finite. Empty intervals, NaNs or a changed utility target invalidate the proposed update and require explicit error handling. A previous valid certificate is not repaired by silently deleting inconvenient historical evidence.

Initialize `i_0` from the exact same `PLAIN_STOP` ordinary readout and tie rule. At each checkpoint set `j=argmax_a(L_a,-a)` and perform:

    if L_j > U_i: i <- j
    otherwise: retain i.

Define true simple regret and its interval upper bound:

    r_i(u) = max_a u_a - u_i,
    G_i = max(0, max_{j != i} U_j - L_i).

For K=1, define `G_i=0`. The benchmark retains its previous admitted arithmetic/input domain; these decision lemmas do not widen it. Exact C24 enclosures are prerequisites.

## R05.2: retained-utility theorem

**Theorem 1.** Under the enclosure assumptions, each accepted transition `i->j` satisfies

    u_j-u_i >= L_j-U_i > 0.

Therefore at every subsequent checkpoint and final budget return,

    u_retained >= u_PLAIN_STOP,
    r_retained(u) <= r_PLAIN_STOP(u).

**Proof.** The endpoint inequalities give `u_j>=L_j>U_i>=u_i`. An unchanged incumbent has equal utility. Compose these inequalities over all accepted transitions. Neither convergence of the approximate scores nor CG orthogonality is used.

The sum of the certified gain lower bounds `L_new-U_old` across switches also lower-bounds the total true gain, if accumulated with exact or downward arithmetic. This is an optional trace diagnostic, not required for the basic rule.

**One-step closure.** After selecting a maximizing lower endpoint j and accepting it, every k obeys `L_k<=L_j<=U_j`. No other action can then satisfy `L_k>U_j` at that checkpoint. If the proposed maximizing j does not satisfy `L_j>U_i`, neither does any other action. One scan and at most one switch are enough.

**Ties.** Equality `L_j=U_i` is a certified nondecrease, but not a certified strict improvement. The selected strict rule retains i. True ties therefore need no arbitrary switches. A comparison proving `L_i>=max_{j!=i}U_j` certifies weak optimality; only strict separation certifies a unique maximizer. Preserve this distinction in result flags.

This theorem compares with the initial ordinary action. It does **not** assert utility noninferiority to `PLAIN_ALWAYS`: a later ordinary argmax can select an actually better action whose improvement has not been certified. Example 3 below makes that loss exactly `1/10`.

## R05.3: robust regret, sharpness and monotonicity

For j different from i,

    u_j-u_i <= U_j-L_i.

Maximizing over j and including the zero difference for i proves

    0 <= r_i(u) <= G_i.

For the rectangular uncertainty set `B=product_a[L_a,U_a]`, this bound is exact:

    sup_{v in B} [max_a v_a-v_i] = G_i.

If `G_i>0`, set `v_i=L_i` and one maximizing competitor to its upper endpoint; all other coordinates can take any admissible values. This attains `G_i`. If `G_i=0`, all candidates are no better than i throughout B. In a structured nonrectangular feasible set the same formula remains an upper bound and can be conservative; the SPD problem has such additional structure. No claim of information-theoretic optimality across richer interfaces follows from box sharpness.

For a fixed incumbent, intersection makes every U nonincreasing and L_i nondecreasing, so G_i cannot increase. At a switch `i->j`, evaluated with the same current intervals, let `M=max_a U_a`. Since `U_j>=L_j>U_i`, M is achieved outside i, and

    G_i=M-L_i,
    G_j <= M-L_j = G_i-(L_j-L_i) <= G_i.

The nonnegative right-hand side is well defined because `M>=L_j`. Thus the retained gap is nonincreasing across both interval intersections and accepted switches. This is a deterministic bound property; it does not say that a posterior expected regret must decrease on every realized information update.

## R05.4: arbitrary paid-continuation theorem

Fix the current checkpoint after processing any currently available certified zero-charge switch. Let pi be any admissible positive continuation that eventually returns an action `A_pi` from A. It may be adaptive, randomized, multi-step, change its internal approximate winner, and use any admissible computation order. Its final selector need not use the retention rule for this upper-bound theorem.

Let `C_pi` denote its **future incremental** nonnegative cost, excluding all sunk cost, and let `lambda>=0` be the fixed price in utility per cost unit. Assume every admitted positive continuation has the genuine pathwise lower bound

    C_pi >= c_min >= 0.

Then, on every realized continuation path,

    [u_{A_pi}-lambda C_pi] - u_i
        <= G_i-lambda C_pi
        <= G_i-lambda c_min.

Consequently

    G_i <= lambda c_min

is sufficient for the current retained output to be no worse in priced value than **every positive admitted continuation**. Equality permits an economically tied continuation; stopping saves the additional computation. Strict inequality makes all such continuations strictly worse. If lambda=0 or c_min=0 the rule reduces to the zero-gap test.

The proof uses only the current upper bound on any final utility gain. It does not assume myopic computation, a known horizon, a known best next query, convergence of scores, or monotonic utility along an unretained continuation.

The converse is not valid: `G_i>lambda c_min` does not prove that useful information or an attainable improvement exists. It only means this upper bound has not ruled out profitable computation. A perfect-information continuation at the minimal cost and a box-corner utility realization can attain the bound in a constructed model, but real continuations need not do so.

To call this a STOP optimum over an entire policy class, that class must consist of the current retained output and positive admitted continuations, with its permitted free certified replacements already resolved. Arbitrary zero-cost uncertified action reselection is not dominated by this paid-continuation theorem. For example, `i=[1/5,3/10]`, `j=[2/5,1/2]`, `G_i=3/10`, and `lambda c_min=1/2` reject all paid computation, yet the already-certified free switch to j improves utility by at least `1/10`. Perform that switch first; its new gap is zero.

**Sunk cost.** If `C_sunk` has already been paid, both options share it:

    (u_{A_pi}-lambda(C_sunk+C_pi))-(u_i-lambda C_sunk)
       = u_{A_pi}-u_i-lambda C_pi.

Do not add initial solver/certificate expense to c_min. Conversely, a correct local stop does not erase that initial expense or prove that the whole policy beats `PLAIN_STOP` in total net utility.

**Expected version.** Taking expectation preserves the pathwise bound. A separately established conditional lower bound `E[C_pi|h]>=c_bar` for every admitted pi also supports an expected-value stopping theorem `G_i<=lambda c_bar`. A fitted mean, median, or point predictor alone establishes neither this universal conditional bound nor the pathwise bound. Adaptive selection and a new task distribution must be covered by any claimed expected-cost theorem.

## R05.5: selected deterministic charge and wall-time distinction

The selected C25 accounting experiment defines `n=side^2` and

    W = n * (N_solver_matvecs + N_fast_action_evaluations
                               + N_ordinary_action_evaluations),
    lambda_work = 2^-30 utility / cell-stencil charge.

Each evaluator counter counts actual per-action evaluations, not one batch label. W is an explicit **cell-stencil charge**, not a total flop count or elapsed time. Validation, scalar bounds and controller work must still be included in total wall-time reporting; they can also have separately reported counts. Extra nonnegative deterministic charges can be added without invalidating the lower bound.

The admissible positive action class is: a block of CG continuation, followed **unconditionally** by a fresh FAST readout of all K actions before the next decision. Thus every successfully completed positive continuation includes at least K actual fresh action readouts, even if all CG advances are zero, and

    C_pi = W_future >= K*n = c_min.

This suffices for Theorem 2. Skipping readouts, reusing cached readouts, checking price between partial action batches, or changing the candidate frontier changes the action class and requires the cost lower bound to be updated. A solver-matvec-only counter would not establish positive c_min when a block can advance zero steps. Errors/abortions must not be misreported as successful completed continuations.

For `2<=side<=128`, `2<=K<=8`, `K*n` is an exact integer from 8 through 131072. Its product with `2^-30` is exactly representable in binary64 in this range. This is an accounting theorem for the declared computation class, not a hardware-time lower bound.

The primary wall-time experiment retains the prospectively fixed price

    lambda_wall = 1/10^6 utility / microsecond.

Its positive predicted cost is fitted only on TRAIN and frozen before fresh TEST outcomes. The prediction target should be the future block plus fresh readout/controller cost, not already-spent initial cost. The test `G_i<=lambda_wall*predicted_future_us` is explicitly **empirical**. An exact mathematical utility-gap bound does not turn its predicted cost into a certified one.

Counterexample: current intervals are `[0,1]` for both actions; incumbent true utility is 0 and the other true utility is `1/2`, so G=1. With price 1 and predicted cost 1 the heuristic stops. A perfect-information continuation whose actual cost is `1/4` has true net gain `1/2-1/4=1/4>0`. This violates no interval fact; it invalidates substitution of that prediction for a pathwise lower bound.

## R05.6: constructive frontier rules

Let `M_L=max_a L_a`. Any action with `U_a<M_L` cannot be an optimal action under the current evidence. A tie-preserving frontier is

    F = {a : U_a >= M_L}.

For the retained incumbent, define possible strict-gain challengers

    C_i = {j != i : U_j > L_i}.

If `U_j<=L_i`, then `u_j<=u_i`, so j cannot provide a strict utility gain over i. Nested intervals and later strict incumbent switches preserve that exclusion. Such actions can be frozen for a future optimized implementation, provided their valid stored bounds remain available. C25's selected all-action readout class does not apply this skipping optimization, because its c_min is tied to K readouts.

These are concrete action-dominance rules, not predictions about the value of evaluating an action. An action currently below the largest upper bound can later become the limiting challenger as other bounds shrink. Drop it only under a proved dominance relation, not merely because it is not currently first.

For a restricted computation mode r whose possible output actions are a known set A_r and whose future cost lower bound is c_r, define

    B_r=max(0,max_{j in A_r, j!=i} U_j-L_i).

The same proof prunes that mode if `B_r<=lambda*c_r`. This is a constructive mode-specific extension. It needs a verified output-set restriction; predicted gains or costs alone do not give rigorous mode dominance. The global C25 test uses all possible output actions and the common lower bound.

## R05.7: computational form and numerical direction

At each decision checkpoint:

1. Obtain fresh valid FAST intervals and intersect with retained history.
2. Preserve the initial ordinary `PLAIN_STOP` index; do not replace it merely by the largest lower bound.
3. Select the deterministic maximum-L candidate and accept it only if `L_j>U_i`.
4. Find `U_other=max_{j!=i}U_j`. If `U_other<=L_i`, set the rigorous gap to exactly zero. Otherwise compute `G_up=round_up(U_other-L_i)`.
5. If another continuation is admitted, compute `threshold_down=round_down(lambda*c_min)` and stop in charged mode only if `G_up<=threshold_down`. Otherwise continue within the fixed budget. Wall mode uses its separately labelled prediction rule.
6. At budget exhaustion, return the retained incumbent; do not fall back to an uncertified final ordinary argmax. Record that final ordinary choice separately for comparison.

For an exact decimal price such as `1/10^6`, the value supplied to a rigorous threshold product must itself be a lower enclosure of that exact rational; a nearest binary64 literal is not automatically a lower enclosure. Both factors are nonnegative. The selected dyadic charged-work price avoids this representation issue, but directed threshold code remains the clearest general contract.

An ordinary rounded subtraction can understate G, and an ordinary rounded product can overstate the threshold. Either can produce a false certified stop at a boundary. The endpoint comparison for a strict switch requires no additional arithmetic. Any optional displayed guaranteed switch gain must use downward subtraction.

Keep flags separate: accepted strict improvement, strict optimality, weak optimality if supported, and certified price stopping express different facts. A price stop with positive G is not an optimal-action certificate. True equal utilities inferred only from an offline oracle do not authorize runtime tie certificates.

## R06: exact SPD constructions and checks

Use the inherited side=2, rho=1 problem with

    A = [[5,-1,-1,0],[-1,5,0,-1],[-1,0,5,-1],[0,-1,-1,5]],
    S=1, alpha=1, beta=9.

The actual eigenvalues are 3,5,5,7, so the inherited conservative bounds are valid. For each rational b,x define

    u=b^T A^-1 b,
    ell=2b^T x-x^T A x,
    D=||b-Ax||^2,
    [L,U]=[max(0,ell+D/9),min(1,ell+D)].

These are exact mathematical intervals; production floating envelopes must contain their corresponding true utility rather than reproduce these rational endpoints exactly. The examples use arbitrary admissible rational iterates and demonstrate decision logic. They are not asserted to be actual uniform-block CG traces.

Write `b=(1,0,0,0)`, `c=(0,1/2,1/2,0)`, `d=(3/4,0,0,0)`. Exact oracle utilities are

    u_b=23/105,  u_c=5/42,  u_d=69/560.

**1. Certified gain.** Initial iterates for actions `(b,c)` are `(0,c/5)`. Ordinary scores `(0,1/10)` initialize incumbent c. After refining b to `b/5`, its interval is `[47/225,7/25]`; c has `[49/450,9/50]`. Since `47/225>9/50`, switch to b. Exact gain is `1/10`; final G is zero.

**2. Protection against a worse later ordinary choice.** Actions `(b,d)` start at `(b/20,0)`, with scores `(7/80,0)`, so b is incumbent. Later use `(b/20,d/5)`. Each score has not decreased, yet the final ordinary scores `(7/80,9/80)` choose d and lose `23/240` in true utility. The intervals are

    b: [271/1800,131/200],
    d: [47/400,63/400].

Retention keeps b, exactly preserving its optimal utility; `G_b=63/400-271/1800=1/144`. The remaining positive gap is conservative.

**3. Cost of retention conservatism.** Actions `(c,b)` start at `(c/10,0)`, with scores `(3/40,0)`, so c is incumbent. Later use `(c/10,b/20)` and scores `(3/40,7/80)`. Ordinary selection now gains `1/10` by choosing b. But the intervals are

    c: [41/450,11/50],
    b: [271/1800,131/200],

and `271/1800<11/50`, so retention keeps c. Its exact regret is `1/10` and its bound is `G_c=131/200-41/450=203/360`. A richer structural bound or further refinement could resolve the gain; this is conservatism of the selected interval interface, not a general impossibility.

**Actual command and finite outcome.** Executed `python3 /tmp/quartz_c25_theory_examples.py` with Python 3.12.14 and SymPy 1.14.0; exit code 0. The script independently constructs A, inverts it exactly, calculates every score/residual/bound, intersects initial and later intervals, and checks enclosure validity, nondecreasing retained utility, nonincreasing G, and `true_regret<=G`. Exact expected gains/gaps above are asserted. It also checks a strict-tie retention case, the cost-predictor counterexample, and the free-switch-before-price example. Raw exact rational outputs are in `/tmp/quartz_c25_theory_examples.json`.

R06 condition review also checked: lambda=0; c_min=0; one-action convention; equality and true ties; zero-progress CG with mandatory fresh readout; cap exhaustion; arbitrary adaptivity; sunk cost; finite-domain bounds; and distinction between arithmetic rounding, uncertainty conservatism, algorithm selection and wall-time modeling. Parent-independent checking and actual production behavior remain separate obligations.

## Bridge to the original root-action objective

For a fixed root utility vector theta, simple regret is `max_a theta_a-theta_i`. Holding the current history h fixed, a continuation's expected reduction in that regret equals

    E[theta_{A_pi}-theta_i | h].

Therefore a valid interval bound G_i also bounds the expected benefit entering a value-of-computation objective. The price-stop theorem is an upper-bound certificate for the question whether further computation can improve the retained decision enough to pay its cost. It is a useful module toward the original decision-quality principle; it does not by itself implement or prove optimal expected-gain allocation among FOCUS/EXPAND/MERGE actions.

The rule consumes only action utilities/bounds, action identities and computation costs. It requires no game-specific pattern, tactical rule or signature target. Transport to search nevertheless requires a precise target theta, valid bounds for that target, and coverage of every action the continuation may output. If EXPAND can introduce an unrepresented action, retain a valid unresolved-reservoir upper bound in G or limit the conclusion explicitly to the represented root set. Excluding unexplored actions silently invalidates a global regret claim.

For stochastic bounds, the deterministic proof holds on the simultaneous coverage event for all used actions/checkpoints. Individual fixed-time confidence levels do not automatically survive adaptive repeated checking. The SPD certificate does not supply game-search coverage or a calibrated posterior. A posterior expected regret may rise on an individual information update despite retained true utility and nested G being monotone; under a coherent supported posterior, the expected gain identity above is the appropriate bridge.

The frozen thesis P2 remains a distinct target: at least 30% reduction in `nn_evals_per_move` with the specified paired-seed noninferior Gomoku 7x7 play quality versus its original anchor. A cell-stencil-charge theorem or SPD wall-time result does not discharge its game-quality, neural-evaluation, budget-fairness or confidence-interval requirements. Report C25's exact initial-action protection, actual differences from `PLAIN_ALWAYS`, and observed total priced results at their respective evidence levels.

## Parent handoff

Ready for parent R05 integration and one independent R06/R07 pass. Exact theorem inputs are fixed utilities, complete finite output set, valid nested intervals and a valid cost lower bound for the explicitly defined action class. Current constructor work has no theorem gap within that contract. Implementation still needs actual counter binding, outward price/gap arithmetic, initial-state/score identity, retained budget fallback, the C24 adverse regression, and fresh frozen-cohort total-cost measurement. Negative economic results remain valid results. No extra approval or broader audit is needed for this handoff.
