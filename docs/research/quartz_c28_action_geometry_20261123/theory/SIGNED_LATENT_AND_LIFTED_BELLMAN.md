# Quartz C28-DG: signed latent directions and decision boundaries

## Scope, authority, and result status

The source anchor is Quartz commit `87c15dc828d6eea519c219fd8a412fa21dfc464d`. This is the constructive R3–R7 theory lane for a separately isolated C28-DG cycle. The concurrently maintained C27 checkout has not been edited. The mathematical target is a finite, exact-reveal computation-allocation problem. It preserves the original distinction between recognizing a promising action and deciding whether paying for another computation improves the final decision.

The signed family below is a proposed statistical closure. Its normalization, posterior, covariance, lifted Bellman formula, decision regions, and explicit examples are **derived**. The companion rational oracle provides **exact-finite checks**. This lane supplies a construction and its own checks; independent review is a separate parent-owned step. Real-game generalization, native implementation, and wall-time improvement are not established by this theory artifact. Gomoku/NN-evaluation P2 and human-GM P3 remain open.

All utilities are dimensionless in the displayed examples. Query costs have the same utility units. If measured time is converted to a cost, the conversion factor has units utility/time. No physical temperature, action, or natural-unit convention is introduced. There are no spacetime, gauge, or metric assumptions in this finite decision model.

## 1. Exact-reveal state and a signed, marginal-preserving family

Let `S` be a finite set of unqueried candidates. A candidate has an exact utility `U_i` in a finite support, here `{0, 1/2, 1}`. The best already verified candidate has utility `b` in `[0,1]`. Terminal deployment may select only already verified candidates. Querying `i` pays a fixed nonnegative cost `d_i`, reveals `U_i` exactly, removes `i` from `S`, and replaces `b` by `max(b,U_i)`. At most two further queries are available. STOP returns `b`. The costs, base probabilities, loadings, and remaining set are fixed when a value function is considered as a function of its belief coordinate.

For each candidate, let `p_i(u)` be a normalized base distribution. Define

\[
\mu_i=\sum_u p_i(u)u,\qquad
s_i=\max_{u:p_i(u)>0}|u-\mu_i|,
\]

\[
f_i(u)=\begin{cases}(u-\mu_i)/s_i,&s_i>0,\ p_i(u)>0,\\0,&s_i=0\text{ or }p_i(u)=0.\end{cases}
\]

Thus `E_p f_i=0` and `|f_i|<=1` on the operative support. Choose a signed loading `r_i` with `|r_i|<1`. Introduce `Z` in `{-1,+1}` and the conditional factors

\[
p_i^z(u)=p_i(u)[1+zr_i f_i(u)].
\]

Each factor is nonnegative and normalized; it is strictly positive wherever `p_i` is positive. For a latent mean `m=E[Z|h]` in `[-1,1]`, the joint law of the remaining candidates is

\[
P_m(\mathbf u_S)=\frac{1+m}{2}\prod_{i\in S}p_i^+(u_i)
                 +\frac{1-m}{2}\prod_{i\in S}p_i^-(u_i).
\tag{1}
\]

At `m=0`, every one-candidate marginal is exactly `p_i`, for every choice of signs. Therefore a common-loading and a signed-loading comparison at `m=0` cannot obtain an advantage by silently changing the initial marginal predictions.

For any current `m`, summing the other coordinates in (1) gives

\[
P_m(U_i=u)=p_i(u)[1+mr_i f_i(u)].
\tag{2}
\]

Write `t_iu=r_i f_i(u)`. After a supported observation `(i,u)`, Bayes' rule gives

\[
m' = \frac{m+t_{iu}}{1+mt_{iu}}.
\tag{3}
\]

The denominator is positive because `|t_iu|<1` and `|m|<=1`. It is the observation likelihood divided by `p_i(u)`. If `p_i(u)=0`, the observation lies outside the model support and must be reported as such; equation (3) is not a repair for a zero base probability. For a log-odds implementation at `|m|<1`,

\[
\ell'=\ell+\log(1+t_{iu})-\log(1-t_{iu}),\qquad m=\tanh(\ell/2).
\]

The closed form (3) also handles the certain-latent endpoints `m=+-1`. Likelihood factors commute, so the posterior after distinct supported exact observations is independent of their order.

### Covariance and the original contrast intuition

Define

\[
\delta_i=\sum_u p_i(u)u f_i(u)
=\operatorname{Var}_{p_i}(U_i)/s_i
\]

when `s_i>0`, and `delta_i=0` for a degenerate distribution. Since

\[
E[U_i|Z=z]=\mu_i+zr_i\delta_i,
\]

conditional independence within each latent component and total covariance imply, for `i!=j`,

\[
\operatorname{Cov}(U_i,U_j|h)
=r_i r_j\delta_i\delta_j(1-m^2).
\tag{4}
\]

The sign of a nonzero covariance is controlled by `r_i r_j`. This is an off-diagonal statement, not a claim that the complete covariance matrix has rank one. Its diagonal also contains within-component variance. A single signed latent mode remains restrictive: the product of three pairwise covariances is nonnegative, and is strictly positive when `|m|<1` and all three loadings and marginal variances are nonzero. At a certain-latent endpoint or with a point-mass marginal the affected covariances vanish. The model cannot express three mutually negative pairwise covariances.

At the balanced prior, signed loading changes preserve each marginal variance. Consequently

\[
\operatorname{Var}(U_i-U_j)
=\operatorname{Var}(U_i)+\operatorname{Var}(U_j)-2\operatorname{Cov}(U_i,U_j)
\]

is directly sensitive to the loading signs. For the symmetric Bernoulli targets used below, their pairwise correlation is `+-r^2`. Independence can underestimate contrast variance when the actual covariance is negative. This corrects the scope of any claim that an unknown correlation may always be conservatively set to zero; it does not invalidate the original covariance-based mechanism.

The transformations `r_i -> -r_i` for all `i` and `m -> -m` only relabel `Z` and preserve the complete joint law. Flipping only one loading changes the utility law. Learning the relevant loading signs from actual game cues remains an empirical modeling obligation.

## 2. One-query geometry and an exact action-retention certificate

For a fixed incumbent `b`, define the utility-valued coefficients

\[
\alpha_i(b)=\sum_u p_i(u)(u-b)_+-d_i,
\qquad
\beta_i(b)=r_i\sum_u p_i(u)f_i(u)(u-b)_+.
\tag{5}
\]

The gain from querying `i` when only one query remains is

\[
q_i^{(1)}(m;b)=\alpha_i(b)+m\beta_i(b),
\quad q_{\mathrm{STOP}}^{(1)}=0.
\tag{6}
\]

Thus

\[
V_1(m,b,S)=b+\max\{0,\max_{i\in S}q_i^{(1)}(m;b)\}.
\]

Each action is a line in the one-dimensional belief coordinate. The H1 region in which a fixed action is optimal is the intersection of affine inequalities, hence an interval, possibly empty or reduced to a point. A deterministic tie order must be part of the implementation contract. The oracle uses STOP first, then the smallest candidate index.

Suppose action `k` is chosen at `m_0`, including the possibility that `k` is STOP with `alpha_k=beta_k=0`. Put

\[
\gamma_{kj}=q_k^{(1)}(m_0;b)-q_j^{(1)}(m_0;b).
\]

At `m=m_0+Delta m`, retaining `k` is **exactly** equivalent to

\[
\gamma_{kj}+(\beta_k-\beta_j)\Delta m\ge0\qquad\text{for every }j,
\tag{7}
\]

where equality is allowed only when `k` precedes `j` in the tie order. This is an algebraic certificate of unchanged selection, not an uncertainty heuristic. If `k` was the unique winner, a convenient strict sufficient condition is

\[
|\Delta m|<\min_{j:\beta_k\ne\beta_j}
\frac{\gamma_{kj}}{|\beta_k-\beta_j|},
\tag{8}
\]

with an empty minimum interpreted as infinity and with the new belief still in `[-1,1]`. Equal-slope positive gaps remain positive for all beliefs. Equality at a boundary requires the actual tie rule.

This certificate is conditional on the **same** `p_i`, `r_i`, `d_i`, `b`, and `S`. A reveal ordinarily changes `b` and removes a candidate. One must first update that state; applying the old certificate across such a change is invalid. Its computational savings also depend on whether the coefficients and previous scores are already available. An `O(|S|)` certificate does not automatically make a single `O(|S|)` H1 evaluation cheaper.

## 3. Exact two-query Bellman with cancellation of Bayes denominators

With `b_u=max(b,u)`, the direct two-query first-action gain is

\[
K_i(m;b,S)
=\alpha_i(b)+m\beta_i(b)
+\sum_u p_i(u)(1+mt_{iu})
 \max\left\{0,\max_{j\in S\setminus\{i\}}
 [\alpha_j(b_u)+m'_{iu}\beta_j(b_u)]\right\}.
\tag{9}
\]

The first term is the expected direct incumbent improvement minus the first cost. The second term is the option to choose or decline one final query after observing `i`. It includes the moving incumbent and permits an information-only first query.

Substitute (3). Because `1+mt_iu` is positive on each supported branch, it may multiply every entry of the branch maximum without changing its order. This gives the exact lifted form

\[
\boxed{\begin{aligned}
K_i(m;b,S)
={}&\alpha_i(b)+m\beta_i(b)\\
&+\sum_u p_i(u)\max\left\{0,\max_{j\ne i}
\left[(1+mt_{iu})\alpha_j(b_u)
      +(m+t_{iu})\beta_j(b_u)\right]\right\}.
\end{aligned}}
\tag{10}
\]

No approximation or discarded denominator remains in (10). Zero-base-probability branches contribute zero and are skipped. If no second candidate remains, the inner maximum is zero. The root value and action are

\[
V_2(m,b,S)=b+\max\{0,\max_{i\in S}K_i(m;b,S)\}.
\tag{11}
\]

For three-point utility and an incumbent in the same grid, the coefficients are particularly small:

| `b` | `alpha_i(b)` | `beta_i(b)` |
|---|---|---|
| `0` | `p_i(1/2)/2+p_i(1)-d_i` | `r_i[p_i(1/2)f_i(1/2)/2+p_i(1)f_i(1)]` |
| `1/2` | `p_i(1)/2-d_i` | `r_i p_i(1)f_i(1)/2` |
| `1` | `-d_i` | `0` |

The `b=1` continuation terminates because all remaining costs are nonnegative and no utility can improve the incumbent.

### Convex piecewise-affine value, with the correct action-region statement

Each entry inside a maximum in (10) is affine in `m`. A finite maximum of affine functions is convex and piecewise affine. A nonnegative weighted sum of such functions, plus an affine function, remains convex and piecewise affine. Therefore each fixed-first-action value `K_i` is convex and piecewise affine in `m`; the full value in (11) has the same property.

A second proof enumerates deterministic contingent plans. For each possible first observation, a plan specifies STOP or one remaining query. The terminal payoff and incurred costs are functions of the complete utility world. For a fixed plan, its expectation under (1) is affine in `m`. Maximizing over the finite set of plans gives the same result. This is the finite-policy-tree/alpha-vector structure familiar from finite-horizon POMDPs; the general structure is established mathematics, not a novelty claim for Quartz.

Convexity alone does not imply connected H2 first-action regions. Grouping contingent plans with the same first query yields a union of intervals; this argument does not by itself exhibit a disconnected region within this particular signed family. No connectedness assumption is used. An individual H1 action line or complete contingent-plan alpha vector has an interval optimality region. The explicit reference-family partition in Section 6 is proved separately.

### Computational contract and cost limitations

Precomputing (5) for the at-most-`K+1` relevant incumbent levels costs `O(|S| K^2)` for `K` utility levels, and the lifted double-candidate evaluation then costs `O(|S|^2 K)`. With `K=3`, this is quadratic in the candidate count. The complete-world oracle costs exponentially in `|S|` and is deliberately restricted to small validation cases.

A direct Bayes evaluator can also precompute (5), reaching the same asymptotic order. Comparing against repeated probability reconstruction alone would measure those implementation differences as well as denominator cancellation. The lifted form removes the per-branch posterior division from the inner value calculation, but it introduces its own multiplications and memory accesses. Native performance must be measured with matched decisions, compiler settings, inputs, and total planner cost. No speedup follows merely from algebraic equivalence.

The modeled Bellman cost `d_i` prices queries. A deployed wall-cost comparison must additionally include cue generation, coefficient preparation, posterior maintenance, planning, and query execution. The observed timing of transport or process startup should be reported separately when it is not part of the deployed timing boundary.

## 4. A complete signed-versus-common phase diagram

Let `A` have equal probability on `0` and `1/2`. Let `B` and `C` each have equal probability on `0` and `1`. Take

\[
b=1/2,\quad m=0,\quad r_A=r_B=r,\quad r_C=\sigma r,
\quad d_A=c\ge0,\quad d_B=d_C=d\ge0,
\]

where `0<=r<1` and `sigma=+1` is the common-loading model while `sigma=-1` is the contrast model. Candidate `A` cannot improve the incumbent directly. Put

\[
x=\frac14-d,\qquad y=\frac{r^2}{4}.
\]

An observation of high `A` makes the conditional one-query scores of `B,C` equal to `x+y,x+sigma y`; low `A` reverses the loading contribution. An initial successful target query produces incumbent `1`, ending all useful computation. After a failed target query, the other target has score `x-sigma y`; `A` still cannot improve the incumbent. These observations derive **all** first-action scores:

| Model | `K_A` | `K_B=K_C` |
|---|---|---|
| common, `sigma=+1` | `-c+([x+y]_+ + [x-y]_+)/2` | `x+[x-y]_+/2` |
| contrast, `sigma=-1` | `-c+[x+y]_+` | `x+[x+y]_+/2` |

STOP has score zero, and the value is `1/2+max(0,K_A,K_B,K_C)`. This comparison includes the option of starting with a payoff-relevant target. A favorable score for information-only `A` by itself would not prove that `A` is the optimal first query.

For the common model, `A` is the unique winner exactly when

\[
2c+|x|<y.
\tag{12}
\]

Outside that strict region, a target wins when `x>0` and STOP wins when `x<0`, except at the explicit tie boundaries. For the contrast model, `A` is the unique winner exactly when

\[
c<\min\left\{x+y,\frac{y-x}{2}\right\}.
\tag{13}
\]

Outside that strict region, the best target has positive score exactly when `3x+y>0`; otherwise STOP is optimal, with ties at equality. Equations (12)–(13), together with the score table and a tie rule, specify the complete decision partition for this family. They are decision boundaries of a finite model, not thermodynamic phase transitions.

At `r=4/5`, `c=1/100`, and `d=3/10`, `x=-1/20` and `y=4/25`. The exact results are

| Model | `K_A` | `K_B=K_C` | `V_2` | Optimal first action |
|---|---:|---:|---:|---|
| product (`r=0`) | `-1/100` | `-1/20` | `1/2` | STOP |
| common | `9/200` | `-1/20` | `109/200` | A |
| contrast | `1/10` | `1/200` | `3/5` | A |

For `c=2/25` instead, keeping `r,d` unchanged, the common model chooses STOP while the contrast model chooses `A` with positive gain `3/100`. Thus the contrast direction can create an actual action crossing. It does not imply that common directions are always irrelevant: the first table already gives a profitable common-mode information query.

The independent product limit `r=0` has value

\[
V_2=\frac12+\frac32\left(\frac14-d\right)_+.
\]

An information-only first query is then dominated by querying the targets or stopping. With zero costs, the contrast/common values are respectively `7/8+r^2/8` and `7/8-r^2/8`, and target queries are optimal. The marginal distributions are unchanged while the probability that at least one target succeeds differs.

## 5. Equal information does not mean equal paid decision value

For the preceding symmetric family, flipping `C` to `1-C` maps the common law exactly to the contrast law:

\[
P_-(a,b,c)=P_+(a,b,1-c).
\tag{14}
\]

The map `(B,C)->(B,1-C)` is a bijection. It preserves `H(B,C)`, `H(B,C|A)`, and therefore

\[
I_-(A;(B,C))=I_+(A;(B,C)).
\tag{15}
\]

The latent experiment producing `A` is also unchanged, so `I(Z;A)` agrees. Nevertheless, the paid H2 decision values above differ. At the reference numbers, the common and contrast mutual information is approximately `0.2846565093` nats, while the values are respectively `109/200` and `3/5`.

This is a decision-problem dependence example. It does not violate Blackwell's comparison of experiments for a fixed state/payoff problem. The target-coordinate relabeling in (14) is **not** accompanied by relabeling the utility of choosing candidate `C`; its numerical payoff remains the actual `C` value. Information quantity alone consequently does not determine the opportunity to select a better paid action. Mutual information may still be useful as a bound or diagnostic under additional assumptions.

## 6. An exact belief threshold rather than a posterior-change proxy

Keep the reference contrast model and let the current latent mean `m` vary, holding the other state components fixed. Put `z=|m|`. Direct substitution into (10) gives

\[
K_A(m)=\begin{cases}
1/10,&0\le z\le11/16,\\
9/200+2z/25,&11/16\le z\le7/8,\\
-3/50+z/5,&7/8\le z\le1,
\end{cases}
\tag{16}
\]

and

\[
\max(K_B,K_C)=\begin{cases}
1/200+3z/25,&0\le z\le11/16,\\
-1/20+z/5,&11/16\le z\le1.
\end{cases}
\tag{17}
\]

The first-action crossing occurs at

\[
z=19/24,
\tag{18}
\]

which differs from the continuation-option kinks `11/16` and `7/8`. With STOP then `A,B,C` as tie order, the optimal first action is

\[
\pi_2(m)=\begin{cases}
C,&m<-19/24,\\
A,&-19/24\le m\le19/24,\\
B,&m>19/24.
\end{cases}
\]

At either crossing, the tied winning gain is `13/120`. The posterior can change throughout a sizeable interval without changing the next computation. A genuine theory-to-code discriminator therefore records both belief change and distance to a decision boundary, instead of treating posterior change as evidence of paid decision improvement.

## 7. Finite evidence and the next implementation contract

The command actually executed in the shared scratch environment was

```text
python3 /workspace/scratch/486e36161bdd/c28_theory/exact_lifted_h2.py
```

It exited `0`. The initial completed run made **19,252 exact/diagnostic assertions**, using 53 heterogeneous models with one to four candidates, five rational latent means, four incumbent levels, and 300 closed-family parameter combinations. The general cases include signed, zero, and common loadings; zero-probability and deterministic outcomes; zero and positive query costs; certain-latent endpoints; and the independent-product limit. The run compared every first-action H2 score by complete-world Bellman recursion, direct posterior updating, and the lifted formula in 1,060 contexts. It checked 1,570 complete posterior projections and 675 signed covariance identities. The exact action-cell certificate, including tie precedence, was checked in 5,300 cases.

For the reference family, full deterministic contingent-plan alpha vectors agree with the lifted values at 106 evaluations (nine explicit boundary points plus the 97-point `1/48` grid). The exact world-law bijection proves information equality; the floating-log entropy value is only a numerical diagnostic. Positive affine utility transformations were checked in 53 cases, with query costs scaled simultaneously. These finite tests support implementation of the formulas; the proofs above, rather than finite test counts, support the general claims.

The runnable oracle is `exact_lifted_h2.py`; its machine-readable result is `exact_theory_results.json`. Initial runner SHA256:

```text
58c09680fca14ac5b05a5fb8e20d40d397202ba36762f6a54406cdffc29fc35e
```

For the later native implementation, preserve the same inputs, fixed two-query budget, known-only incumbent, supported-observation failure semantics, state update order, and deterministic tie rule. Compare both scores and actions, including near-boundary cases. A native floating-point tolerance needs an explicit contract; exact rational equality does not authorize silently changing a decision at a floating tie.

The scientific continuation is to identify empirically justified action-contrast directions and test actual resource-priced decisions in a genuinely new domain. Reusing already observed C27 roots can validate semantic equivalence and measure engineering costs, but cannot create a fresh confirmatory policy-improvement experiment. The lifted evaluator can be tested independently of whether a signed model proves empirically useful.

## 8. Skill context and finite review boundary

Actually read and applied: PhysMath Research Loop 1.3.0 `SKILL.md`, `references/THEORY_LOOP.md` R3–R7, and `references/POLICY_INTEGRATION.md`; Mathbox research-program 4.0.0 `SKILL.md` and `references/program-protocol.md`. The cloud package locators were expanded because the child's short `e30`/`e23` aliases did not resolve. One constructive proof/oracle pass was performed. No recursive review, native game campaign, production implementation, C27 mutation, Git publication, or deletion occurred in this lane. The parent owns independent review, persistence, stage integration, and later coding stages.
