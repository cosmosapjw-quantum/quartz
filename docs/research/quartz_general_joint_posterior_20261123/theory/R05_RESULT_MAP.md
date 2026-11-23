# R05 — completed constructive derivation

The complete proof is in R05_DERIVATION.md; the handoff Python is byte-identical to the Mathbox-executed exact oracle. New mathematical products are a centered one-parameter mixture, exact scalar Bayes update, prior/posterior cross-covariance, finite H2 decomposition, a parametric pure-information threshold, a moving-incumbent sign counterexample, and fixed-policy/value/transfer total-variation bounds.

The common-factor model is a proposed closure. Its normalization and listed conditional identities are proved under their assumptions. Bayes and Bellman are reused general results. The existence example109/200 versus1/2 is a property of the proposed family, not a game performance result.

Notation bridge: R04's latent field L=sum atanh(rho f) is half the log odds. R05 uses ell=log odds=2L, m=tanh(ell/2). The production key will be log_odds to prevent a factor-of-two error.

The exact-query terminal contract is retained. Initialization follows C26's raw rollout mean (ascending move tie), not fitted posterior mean. That choice is shared by every control and the joint policy.
