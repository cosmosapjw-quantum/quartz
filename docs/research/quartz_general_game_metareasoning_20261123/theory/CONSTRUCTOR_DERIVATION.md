# Quartz C26: constructive R04–R06 exact-query metareasoning

Author: native constructor `/root/c26_theory`. All `/tmp/quartz_c26_*` files in this report are on the **shared LOCAL container**, not the Remote Desktop Commander device. No repository files were edited and no games were executed. Parent owns integration, production implementation and final claim admission.

## 1. Scope, actual sources and status

Applied PhysMath Research Loop **1.3.0**, cloud package `skill://plugins_6acdfbba921881919e0c17843706aef4/physmath-research-loop`, reading `SKILL.md`, `references/THEORY_LOOP.md` R04–R06 definitions, `references/POLICY_INTEGRATION.md`, `references/SUBAGENT_MODE.md`. Applied Mathbox computation-audit 4.0.0 `SKILL.md`, `references/checklist.md`, `references/runner.md`. No children or recursive review. This is a constructor derivation plus finite and edge-case checks, not an independent review of itself.

Directly read Remote Desktop Commander device `edd1b263-56a0-4391-8a2a-677dd00eabf5`, root `/home/cosmosapjw/quantum/research/quartz_general_meta25_20261123`: `src/game.rs` and `src/games/tictactoe.rs`, including player switching, legal-move order, terminal detection and outcome. Parent supplied HEAD `d6019d674ae76524b1895016f49b82d4f7b001b2`; this worker did not independently run Git or hash those remote files. Semantic conventions were directly read; commit binding is parent-supplied.

Original QUARTZ PDF Eq. 1 (KL-regularized expected utility), Eq. 2 (VOC), and cumulant motivation were supplied by the parent. This worker does not claim an independent image reading of that PDF. The construction below instantiates those stated mechanisms.

Literature: M. L. Weitzman, *Optimal Search for the Best Alternative*, Econometrica 47(3), 641–654 (1979), DOI `10.2307/1910412`; author working paper MIT-EL-78-008 (May 1978), primary PDF <https://www.osti.gov/servlets/purl/6795538>. Sibling `/root/c26_literature` downloaded/rendered the primary working paper: Eq. 7 printed p. 9 / PDF p. 13 and rule printed p. 10 / PDF p. 14. Its reported PDF SHA256 is `546b8b0344acd58b96e2ea9ab753a3b70dd87803d9ecd4d7b69b7fdb15373303`. This is sibling source verification. This worker independently retrieved the author-hosted published-paper search extract and MIT metadata; Harvard PDF open timed out and MIT PDF returned 429. Those failed opens are not full-source verification.

The general reservation-index rule is established literature. The stopping proof, capped-variable proof, categorical formula, examples and mapping below are directly derived here. No novelty or strong no-go claim is made. Finite checks concern the standalone mathematical oracle, not real-game prior accuracy.

## 2. GameState target, incumbent and baseline

Let r be a fixed root with player sign p_r in {-1,+1}. Enumerate all legal root actions in deterministic action-index order. At terminal descendant s, repository outcome(s) is from its current player's perspective. Thus root-perspective terminal payoff is

    v_r(s) = p_r * current_player(s) * outcome(s).

Root-perspective minimax maximizes where current_player(s)=p_r and minimizes otherwise. If v_a is the exact minimax value of root child r_a, define

    u_a = (v_a+1)/2 in {0,1/2,1}.

Ordinary negamax V(s)=max_m[-V(s_m)] with terminal V=outcome gives v_a=-V(r_a) in this alternating-turn game. Do not apply the root sign twice. A FOCUS query solves one whole previously unsolved root child, revealing that fixed u_a.

The parent-selected baseline takes the highest prior-mean action after common paid cues, ties by action index, and pays for its exact solution. Every policy shares that action, exact value and cue cost. Let b_0 be its known utility. At history h, known set K_h and unknown set S_h give

    b_h = max_{a in K_h} u_a.

Retain the existing incumbent on utility ties. Final deployment is restricted to solved actions. This restriction matters: the model does not allow deployment of an unqueried action solely because its posterior mean is high.

**Pathwise safety.** Each reveal updates b to max(b,u_a), so b_final>=b_0 for every policy, seed and true utility vector. Simple regret relative to the initial action cannot increase. Priced net improvement b_final-b_0-lambda*C_future may nevertheless be negative.

## 3. Categorical model and one-step VOC

Use TRAIN-only fitted probabilities p_a0, p_ah, p_a1 for exact minimax utility 0,1/2,1, summing to one. The parent-selected four random game-agnostic rollouts per candidate are input features for predicting exact minimax labels. Their frequencies are not automatically minimax posterior probabilities. Freeze fitted parameters and initial cue-derived laws within each evaluation root.

Conditional independence of different unknown actions is a **model approximation**, not a GameState fact. Under this model,

    g_a(b) = E[(U_a-b)_+]
           = p_ah max(1/2-b,0) + p_a1 max(1-b,0),
    d_a = lambda*c_a,
    Q_a = g_a(b)-d_a,        Q_STOP=0.

Therefore g_a(0)=p_ah/2+p_a1, g_a(1/2)=p_a1/2, and g_a(1)=0. Costs c_a are positive charged nodes; lambda is utility per node; priced cost d_a and tau below have utility units.

Theorems below assume fixed known d_a. Proposed production Q uses predicted node cost, while actual cost is logged separately. Its expectation error is exactly

    Q_model-Q_true = (g_model-g_true)-lambda*(c_model-c_true),

where the true terms are conditional means under the evaluation population. Shared initial cue/solve costs cancel in paired incremental comparison but remain in total-cost reporting. Controller overhead, training, adjudication and runtime work need explicit accounting. Shared caches can change subsequent actual query costs, violating fixed-cost assumptions unless modeled.

## 4. Exact adaptive objective and globally valid STOP

For independent values and fixed costs, the optimal future value obeys

    V(b,S)=max{b, max_{a in S}[-d_a+E V(max(b,U_a),S\{a})]},
    V(b,empty)=b.

**Theorem 1.** For finite S, independent fixed laws, exact full reveals, additive fixed costs and recall,

    V(b,S)=b iff g_a(b)<=d_a for every a in S.

**Proof.** If some g_a(b)>d_a, opening a then stopping earns b+g_a(b)-d_a>b. Conversely, take any adaptive randomized policy. Before query t, b_t>=b. The selected unknown a_t is measurable from revealed history and independent auxiliary randomness. Independence preserves its unrevealed conditional distribution. Hence

    E[b_{t+1}-b_t-d_{a_t}|history]
       =g_{a_t}(b_t)-d_{a_t}
       <=g_{a_t}(b)-d_{a_t}<=0.

At most |S| queries occur. Sum conditional expectations to obtain expected final utility minus future costs <=b. STOP attains b. Equality allows neutral queries; economic tie handling can select STOP. The sufficient condition also holds for any constrained subset of these policies. The converse only uses a positive-gain query if it is feasible.

This result does not make largest g-d the optimal positive-gain query order.

**Distribution-free bound.** If every nonempty continuation has actual charge >=c_min>0, its pathwise net improvement is <=1-b-lambda*c_min. Thus lambda*c_min>=1-b certifies economic STOP without any prior or independence assumption. With charge floor one, lambda>=1-b suffices. A path of n additional queries with at least one charge each has net improvement <=1-b-lambda*n. All are prospective bounds; sunk costs are excluded.

## 5. Exact reservation values and constructive optimality

For d_a>0 define the unique real reservation value z_a by

    E[(U_a-z_a)_+]=d_a.

The expected excess is continuous, strictly decreasing below the essential supremum, tends to infinity as z tends to minus infinity, and equals zero above the support. Reservation values may be negative. For zero priced cost, use z=ess sup U; free queries are economically harmless.

For three-point law, write mu=p_h/2+p_1 and d>0. Then

    z=1-d/p_1                if p_1>0 and d<=p_1/2,
     =(mu-d)/(p_h+p_1)      otherwise if p_h+p_1>0 and d<=mu,
     =mu-d                 otherwise.

Branches agree at boundaries. Degenerate U=0 gives z=-d; certain U=1/2 gives z=1/2-d. The script handles zero denominators by branch conditions.

**Reservation policy.** Open an unknown action of largest z when max z>b; otherwise stop. Use deterministic tie handling. This is the no-discount specialization of Weitzman's rule, not a new literature claim.

**Independent capped-value proof.** Put Y_a=min(U_a,z_a), E_a=(U_a-z_a)_+, so U_a=Y_a+E_a and E E_a=d_a. Let I_a indicate whether an adaptive policy opens a. Opening is decided before seeing U_a; independence implies E[I_a E_a]=E[I_a]d_a. Its expected payoff equals

    E[max(b,max_open U_a)-sum_open E_a]
      <= E[max(b,max_open Y_a)]
      <= E[max(b,max_all Y_a)].

The first inequality is pointwise: subtracting every nonnegative excess removes at least the excess of the maximizing observed item. Under the reservation policy, only the last opened box can have positive excess: U_a>z_a exceeds every remaining z and forces stopping. Before this final opening, the incumbent is below its z. If no excess occurs, the first inequality is equality; if one occurs, subtracting it reduces the terminal maximum exactly to that final z, again equality. At stopping all unopened capped values are bounded by their reservation values and thus by the capped terminal maximum, so the second inequality is also equality. The reservation policy attains the universal upper bound and is optimal.

This proof includes atoms, ties and any initial fallback b. It assumes all remaining boxes can be opened if desired, without an external binding cap. A one-query cap makes largest one-step g-d optimal instead. Correlated learning, partial refinements and changing costs require another model.

## 6. Gibbs variational identity, limits and scale

Fix eligible M={STOP} union S, values Q, and positive prescribed prior pi summing to one, independent of Q. Fixed base prior weights restricted and renormalized to the current eligible set are allowed. For tau>0 define

    J_tau(q)=sum_m q_m Q_m-tau*KL(q||pi),
    Z=sum_m pi_m exp(Q_m/tau),
    F_tau(Q)=tau log Z,
    q*_m=pi_m exp(Q_m/tau)/Z.

Substituting log(q*_m/pi_m)=Q_m/tau-log Z gives the exact identity

    J_tau(q)=F_tau(Q)-tau*KL(q||q*).

Thus q* is the unique maximizer; its objective advantage over any q is tau*KL(q||q*). Derivatives are grad F=q* and Hess F=(diag q*-q*q*^T)/tau, positive semidefinite with common-shift null direction. Stable computation uses (Q-max Q)/tau and log prior weights. The runtime must actually sample q* with a recorded generator and seed; argmax implements another policy.

Cost lambda*c is already in Q. KL penalizes deviation from the reference distribution; it is not physical elapsed work. Tau is a utility scale, not kelvin. No thermal-bath, KMS or quantum-dynamics claim follows.

Let M_Q=max Q and Pi_* be prior mass on its maximizers. Then

    M_Q+tau log Pi_* <= F_tau(Q) <= M_Q,
    0 <= M_Q-E_{q*}Q <= tau log(1/Pi_*).

The latter follows from E_q*Q=F+tau KL(q*||pi)>=F. As tau tends to zero, q* concentrates on Q-maximizers with relative weights pi. As tau tends to infinity, q* tends to pi. For bounded Q,

    F_tau(Q)=E_pi Q+Var_pi(Q)/(2tau)+kappa_3,pi(Q)/(6tau^2)+O(tau^-3).

This high-temperature cumulant expansion averages over metacomputation actions, not uncertain true utilities.

An affine utility change u'=alpha*u+beta, alpha>0, gives g'=alpha*g. Choosing lambda'=alpha*lambda and tau'=alpha*tau preserves Q/tau and policy; the common shift beta cancels. In normalized units u and tau are dimensionless utility; lambda has units inverse charged node.

**Economic distinction.** Full-support finite-tau Gibbs selects every unknown query with positive probability, even if all Q<0. It optimizes the regularized local objective, not unregularized net value; Theorem 1 does not automatically describe its behavior. The parent selected a **b=1 certificate mask**: all FOCUS become ineligible once a true optimum is known, and STOP is returned. Below b=1, all unknown queries and STOP remain eligible. This is Gibbs on a certified eligible set, different from unmasked Gibbs at b=1. Filtering every Q<=0 action would be another change and must not be inserted silently. The exact-check file reports masked and unmasked expectations. Gibbs on one-step Q has no multi-step optimality claim. An entropy-regularized Bellman policy would use continuation values and is another objective.

## 7. Quenched, annealed and common-mode uncertainty

For a utility vector X and fixed action prior rho, set F_tau(X)=tau log sum rho_a exp(X_a/tau). Let X=mu+epsilon, E epsilon=0, covariance Sigma, and w=softmax_rho(mu/tau). Taylor expansion gives the quenched average

    E F_tau(X)=F_tau(mu)
      +[sum_a w_a Sigma_aa-w^T Sigma w]/(2tau)
      +higher-order terms.

Its quadratic correction equals

    (1/(4tau))*sum_ab w_a w_b Var(epsilon_a-epsilon_b).

It therefore measures uncertain contrasts and annihilates common shifts. The annealed expression instead is

    tau log E[sum_a rho_a exp(X_a/tau)]
       =F_tau(mu)+sum_a w_a Sigma_aa/(2tau)+higher-order terms.

The leading difference is w^T Sigma w/(2tau). Jensen ensures annealed>=quenched when finite. For epsilon=delta*xi at fixed tau and bounded xi, the neglected third-order term is O(delta^3/tau^2). Centered Gaussian perturbations have zero odd terms. These expansions need not be uniform as tau tends to zero.

**Exact Gaussian common-mode diagnostic.** Let epsilon=Z*1 with Z~Normal(0,sigma^2), shifting every alternative including the retained choice. Then

    F_tau(mu+Z*1)=F_tau(mu)+Z,
    softmax_rho((mu+Z*1)/tau)=softmax_rho(mu/tau),
    E F_tau(mu+Z*1)=F_tau(mu),
    tau log E exp(F_tau(mu+Z*1)/tau)
       =F_tau(mu)+sigma^2/(2tau).

All are exact by the Gaussian moment-generating function. With sigma=1/10 and tau=1/5, the annealed offset is exactly 1/40=0.025, whereas the quenched offset and perfect-information decision gain are zero: E max X-max E X=0. Diagonal variance alone would falsely count this decision-irrelevant shared uncertainty. For two equal-weight alternatives with independent variance sigma^2, the quadratic quenched correction is sigma^2/(4tau), versus annealed sigma^2/(2tau). Perfect anti-correlation raises the quenched quadratic correction to sigma^2/(2tau).

These free-energy corrections are not C26 VOC by definition. Its terminal selector uses the hard observed maximum, so its one-query VOC is E[(U_a-b)_+]. A free-energy-difference VOC requires an explicitly regularized terminal objective. Keep this cumulant construction as a theory/diagnostic lane unless that objective is intentionally implemented.

## 8. Exact examples and correlation counterexample

All below except Gibbs expectations used exact Fraction arithmetic; initial b=0 for the independent examples. Costs d already include lambda.

| Box | Distribution (P0,P1/2,P1) | Cost d | Initial g-d | Reservation z |
|---|---|---:|---:|---:|
| A | (0,1,0) | 1/10 | 2/5 | 2/5 |
| B | (3/5,0,2/5) | 1/20 | 7/20 | 7/8 |
| C | (4/5,0,1/5) | 1/50 | 9/50 | 9/10 |

For A,B, greedy A-first gives 11/20=0.55. Pandora B-first, followed by A only after B=0, gives 59/100=0.59: 0.4*1+0.6*0.5-0.05-0.6*0.1=0.59. Bellman agrees. With a one-query cap, A gives 0.4 and B gives 0.35, so unrestricted Pandora cannot certify that constrained case.

For A,B,C, Pandora begins C then B and gives 163/250=0.652; greedy gives 299/500=0.598; always-open-all gives 59/100=0.59. At tau=0.1 with uniform prior over STOP and remaining actions, recursively recomputed Gibbs gives expected raw net 0.5775342718082925... with b=1 mask and 0.5697816346043772... unmasked. For A,B those values are 0.5402794444893549... and 0.5362637731518749.... These are explicit outcome-recursion expectations, checked at 60 and 90 Decimal digits with discrepancies below 1.4e-60; they are not random GameState trials.

**Correlation counterexample.** Start at known b=1/2. Let (A,B,C)=(0,1,0) or (1/2,0,1), each with probability 1/2; costs d=(1/100,3/10,3/10). Initial one-step nets are (-1/100,-1/20,-1/20), all negative. Query A, which never improves the incumbent but tells which of B,C is certainly one; query that winner. Final utility is one and cost is 31/100, so net value is 69/100 and improvement over baseline is 19/100. Exact joint-support Bellman enumeration confirms A-first optimality. This shows exactly why a product-prior approximation can miss valuable information-only queries in correlated games.

## 9. Actual computation evidence

Code: `/tmp/quartz_c26_exactcheck.py`; direct JSON: `/tmp/quartz_c26_exactcheck.json`; bounded-run JSON: `/tmp/quartz_c26_audit/exactcheck.json`; v2 manifest: `/tmp/quartz_c26_audit/manifest.json`; contract: `/tmp/quartz_c26_contract.json`.

Test population: all 15 probability triples on denominator-4 simplex times costs {1/20,1/10,1/5} gives 45 box types. All ordered pairs and incumbents {0,1/2,1} yield 45^2*3=6,075 cases. For three boxes, all 6 denominator-2 probability triples times the same costs give 18 types; all 3-multisets and three incumbents yield binomial(20,3)*3=3,420 cases. The latter is exhaustive up to relabeling on that grid. Total 9,495 cases; not all rational probabilities.

Every case checked exact Bellman=Pandora, the STOP iff condition, and each reservation expected-excess equation. Bellman recursion and the closed-form index supply orthogonal implementations, while sharing input conventions. The correlated case uses a separate posterior-on-joint-support recursion. No sampling or RNG was used. Production Gibbs still needs an actual seeded sampling check.

The direct run succeeded. A bounded run of the identical frozen script captured pre/post input hashes and resource limits. Actual command:

    python /home/agent/.codex/plugins/cache/openai-curated-remote/mathbox/4.0.0/skills/computation-audit/scripts/run_experiment.py --root /tmp --contract /tmp/quartz_c26_contract.json --input quartz_c26_exactcheck.py --output quartz_c26_audit --result quartz_c26_audit/exactcheck.json --timeout 60 --max-output-bytes 1048576 --max-memory-bytes 1073741824 --max-cpu-seconds 50 --max-cores 1 --max-threads 1 -- python quartz_c26_exactcheck.py quartz_c26_audit/exactcheck.json

Observed exit 0, run.status=completed, Python 3.12.14, runtime 2.572867 seconds. The installed `validate_manifest.py /tmp/quartz_c26_audit/manifest.json --root /tmp` returned exit 0 and `valid evidence record (run status: completed); mathematical interpretation requires review`. Script SHA256 before and after is `5781c8c842bf45b189c3b92e6b84e100e2f6208639f5947e88e973e9b8bc49fd`. Result SHA256 is `36c084714ff31965f6956a27a786622c10479a2d1e27c64e29722aad6f4316ba`; direct and bounded-run JSON are byte-identical. Outcome: **standalone implementation and finite assertion verified in the stated range**. Analytic claims use the uniform proofs above. No GameState production implementation, empirical prior, actual-cost calibration or empirical net gain has been verified.

## 10. Integration recommendation

Keep STOP, always-query, controlled random-order, greedy net, Pandora-model and Gibbs-VOC distinct. Gibbs must actually sample the original KL/Gibbs mechanism; retain and label the parent's b=1 mask. Pandora supplies an optimal independent fixed-cost model benchmark, with binding-budget departures stated. The stronger STOP and index results sharpen the mathematical model without asserting empirical independence.

Discriminating outputs are paired actual utility improvement, incremental and total work, net improvement at frozen lambda, held-out minimax calibration, cost-prediction errors and actual Gibbs query/STOP frequencies. Incumbent safety holds regardless of calibration; positive priced gain needs an actual cohort. The common-mode example protects the uncertainty/VOC distinction. Parent's reversible-generator/Lindblad realization is outside this worker's completed scope.
