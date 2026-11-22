# R3. Conditional observation estimands and declared variables

Let D in {mcts, sequential_non_game}, parent-group G, initial computation transcript H_0, and user-selected additional computation budget c. The initial-only feature vector is X=φ(H_0), including a dimensionless top-two vote/count margin, number of exposed alternatives, initial entropy/concentration and measured resource cost when it exists BEFORE action selection. Under the actual joint source law P_D(G,H_0,Y,U,C), let a_0 be the proposed terminal action before the extra computation and a_1 after. Define:
  Y=1[a_1!=a_0] (observable decision revision).
  Z=U(a_1,θ)-U(a_0,θ) (signed value of the additional computation, conditional on known or observable task utility).
  C=actual marginal computation cost measured by monotone clock or NN-evaluation counter.

The conditional *revision* propensity is q_D(x)=P_D(Y=1|X=x). This is a predictive likelihood, not a Bayesian posterior over θ unless an explicit latent-state model is supplied. A model qhat(X) can be calibrated on held-out group-independent data without assuming that consecutive MCTS backups are IID. Standard proper scores: Brier B=E[(Y-qhat)^2] and negative Bernoulli log loss L=E[-Y log qhat-(1-Y)log(1-qhat)], with positive clip floor specified before data. ECE is finite-bin descriptive only, not a confidence guarantee.

Even perfect q_D does not determine E[Z|X]: E[Z|X]=q_D(X)E[Z|Y=1,X] because Z=0 whenever a_1=a_0. In Gomoku15 using source-only MCTS outputs, θ/true U is unknown; observe Y but Z is NOT_IDENTIFIABLE. In non-game sequential Bernoulli bandit, latent per-arm means are generated and recorded offline, so actual Z and regret change are calculable, but their online use for prediction is forbidden.

The original C17 Bayes simple regret R(b) is a different object requiring a valid posterior and likelihood over task utility states. A calibrated Y predictor cannot be inserted directly as its substitute. Game-specific root histories are a source adapter, not meta-policy input.

Experiments should report conditional revision frequency, Brier/log loss vs train-only constant and raw margin baseline, domain-specific holdout, group size, source-group counts, real CPU per root and calibration coverage; selectively continue vs STOP only in SHADOW read-only proposals until true decision-utility calibration is validated.