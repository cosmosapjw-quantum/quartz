# R06 — independent mathematical review by parent

The parent read the full218-line derivation and inspected the oracle's complete-world conditioning/Bellman and latent closed-form implementations. This is a second review of the algebra, not a blind proof assistant certificate. An independent Wolfram evaluator checked the information-only formula and threshold; its raw response is in evidence/WOLFRAM_SYMBOLIC_CHECK.json.

## Checked obligations

- Normalization: sum p f=0 gives sum p(1+z rho f)=1. Positivity holds on support for rho<1. Zero p is not repaired by positive rho; impossible observations must error rather than reset the posterior.
- Marginals: summing all other normalized components gives p_a exactly. rho=0 is product; rho sign is latent relabeling. The proposed model has nonnegative off-diagonal covariance, not arbitrary covariance or a rank-one full covariance matrix.
- Posterior: likelihood ratio multiplication yields log_odds additive in log1p(rho f)-log1p(-rho f); m=tanh(log_odds/2). Distinct observation order commutes. A repeated observation must not be counted twice.
- H2: substitute V1=b+M into finite backward induction and E max(b,U)=b+g. The cap decrements2->1->0. STOP is available at every stage. b=1 and nonnegative cost imply STOP.
- Pure information: U_a<=b fixes the incumbent. Conditional expectation of each fixed improvement function is a martingale, and convexity of max proves nonnegative option correction before the first query cost.
- General sign: when b changes with the observation, that martingale argument does not apply to the frozen-threshold counterfactual. The exact -1/5 example is a valid local correction, not negative classical free-information value.
- Threshold: with c,epsilon>=0, B-first cannot improve via A; A-first has only the high branch. Hence V2=1/2+[-c+(rho^2/4-epsilon)_+/2]_+. Strict gain iff rho^2>4epsilon+8c, including zero cost limits. At equality, choose STOP.
- TV bound: policy return lies in[b-D_H,1], including all modeled fixed query costs; common policy class and a definition on every true-supported history are required. The two-policy transfer bound carries factor2. It does not control arbitrary wall-time costs or rare-conditioning amplification from unconditional TV.
- Units: U and d are utility units; f,rho,log_odds are dimensionless. lambda_N and lambda_t convert nodes/time into utility. There is no omitted physical constant.

## Review corrections

Explicitly bridge field versus full log-odds notation. Preserve raw-rollout initialization. Label prior-matched and post-initial-matched controls separately. Counts of conditional finite checks must distinguish empty from nonempty history. These were incorporated before coding.

No new strong no-go is asserted. The closure excludes negative dependence by construction; empirical failure would reject this closure/implementation at its measured price, not the general metareasoning objective.

## Independence levels

1. Proof: constructive algebra.
2. Algorithmic crosscheck: full-world rational recursion versus latent formula from the same theory worker.
3. Separate symbolic tool: Wolfram expression/inequality calculation.
4. Parent review: assumptions, sign, units, support and scope.
5. Production semantics/performance: still pending coding loop.
