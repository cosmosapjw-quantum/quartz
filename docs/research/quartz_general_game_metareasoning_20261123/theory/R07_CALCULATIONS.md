# R07 — executed discrimination

## Independent exact-query oracle

The actual Fraction script checked 9,495 cases: 6,075 ordered two-candidate cases and 3,420 three-candidate cases up to relabeling on the specified finite grids. Bellman value equals Pandora value, the STOP equivalence holds and each index satisfies the expected-excess equation in all inspected cases. The bounded Mathbox execution exited 0 in 2.572867 s; manifest validation exited 0. Exact examples give Pandora/greedy values 0.59/0.55 and 0.652/0.598. Joint-support recursion gives correlated-example net 0.69 from baseline 0.5. The finite enumeration supports the independent direct proof, not universal game independence.

## Statistical-mechanics mechanism checks

A second worker executed 72 assertions (host exit 0, 0.623 s) using rational SymPy matrices, symbolic arbitrary-density entries, high-precision logs and fixed float tolerances. It verified detailed balance, stationary Gibbs law, column stochasticity, KL contraction, Gibbs variational gap, Lindblad population/coherence equations, finite Kraus completeness, total covariance and computation-specific fluctuation terms. Parent's independent Wolfram evaluator returned exact zero stationary/balance/conservation residuals. The Wolfram context endpoint failed internally, while the separate evaluator succeeded; these are recorded separately.

For q=(1/6,1/3,1/2), the exact rate matrix is
W=[[-1,1/4,1/6],[1/2,-3/4,1/3],[1/2,1/2,-1/2]].
P=I+W/2 is a selected discrete transition, not exp(W/2). Off-diagonal jump operators implement this population transfer; diagonal jumps leave every population constant.

For Q=(0,.2,-.1), tau=.1, uniform prior, q=(.11419519938459448,.8437947344813395,.042010066134066035). Starting from (.6,.3,.1), finite-transition KL fell from .7719002946995058 to 4.192334491911877e-15 over 80 steps under recorded tolerances.

The posterior-covariance example explicitly reveals approximation error: at observation accuracy .75, exact smoothed gain is about .178002 while the quadratic approximation is .300180. At an uninformative channel, actual gain and posterior-mean covariance vanish while inserting latent covariance spuriously gives 1.2007194719. Preserve both failures of an unjustified substitution. Positive information log(2) with zero decision gain is also checked. These are actual distinctions, not new runtime bonuses.

## Bounded source population feasibility

Parent enumerated all legal alternating TicTacToe histories, stopping at terminal states: 5,478 distinct reachable states. D4 canonical nonterminal counts at plies 4,5,6 are 108,153,183. This permits a prospective 100-parent-state sample per ply with disjoint TRAIN/CAL/TEST groups. No minimax outcomes or policy results were used to choose these sample sizes. The production generator will independently reproduce population/group checks.

## Reproduction identity

The local computation environment differs from the remote production environment. Manifests preserve original LOCAL /tmp paths and Python versions; copying evidence into this repository does not rewrite those execution identities. Source and output hashes bind the original computations. Production verification and measured game results remain for C01–C10.
