# R05 — constructive mathematical result

The full proof is in CONSTRUCTOR_DERIVATION.md, preserved as the actual constructor artifact. Its baseline sentence in section 2 records an earlier design; the operative R03 initialization uses highest ordinary four-rollout mean, accounts for known terminal children, and supersedes that sentence. Its proof needs only a common known incumbent, so this change does not alter any theorem.

## Exact-query theorem

For finitely many independent utility variables U_a, exact full reveals, fixed additive positive charges d_a, recall, and a known incumbent b, define g_a(b)=E[(U_a-b)_+]. Then V(b,S)=b iff every g_a(b)<=d_a. Independence preserves an unqueried candidate's law after any other reveal; b_t>=b makes its future expected excess no larger. The conditional net increments are nonpositive and telescope through at most |S| queries. A positive one-query net proves the converse.

For d_a>0 the unique real z_a satisfying E[(U_a-z_a)_+]=d_a is an ordering index. Query the largest z_a>b until none remains. With Y_a=min(U_a,z_a) and E_a=(U_a-z_a)_+, independence gives E[I_a E_a]=E[I_a]d_a for any adaptive opening indicator I_a. Every policy's expected net value is bounded above by E max(b,max_a Y_a); reservation ordering attains this bound. The constructive proof includes negative indices, atoms, ties and a fallback. An external binding query cap changes the feasible problem.

For U in {0,1/2,1}, mu=p_h/2+p_1:
z=1-d/p_1 if p_1>0 and d<=p_1/2;
else z=(mu-d)/(p_h+p_1) if p_h+p_1>0 and d<=mu;
else z=mu-d.
Branch equalities agree and zero denominators are guarded.

## Variational and fluctuation structure

For tau>0 and a fixed positive reference law pi over eligible computations, F=tau log sum pi exp(Q/tau), q*=pi exp(Q/tau)/Z. Direct substitution yields
F-[q.Q-tau KL(q||pi)]=tau KL(q||q*).
This proves the Gibbs optimizer, its gradient q* and covariance Hessian (diag(q*)-q*q*^T)/tau. Runtime sampling is part of this mechanism.

For centered epsilon with covariance Sigma, the second-order quenched correction is
[sum_a w_a Sigma_aa-w^T Sigma w]/(2tau)
= sum_ab w_a w_b Var(epsilon_a-epsilon_b)/(4tau).
It annihilates common-mode covariance; the annealed expansion does not. A common Gaussian shift gives exactly zero quenched/decision gain but sigma^2/(2tau) annealed offset. Information and decision value are separate.

## Constructive failure of the independent closure

At b=1/2, two equiprobable joint worlds (A,B,C)=(0,1,0) or (1/2,0,1), with costs (0.01,0.3,0.3), give negative marginal one-step nets for all candidates. Querying A identifies the profitable next candidate and yields net 0.69, a 0.19 improvement. A supplies information without immediate incumbent gain. This identifies the next joint-belief model; it is not a no-go for metareasoning.
