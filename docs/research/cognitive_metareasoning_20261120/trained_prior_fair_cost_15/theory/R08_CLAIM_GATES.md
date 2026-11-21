# R8. Admit only supported claims

DERIVED: legal softmax log-loss convexity and gradient, finite positive support under finite theta/features, probability of uniform candidate discovery 4/209, explicit distinction of root mask and trained prior, no equal-visits=>equal-time implication.
NUMERICALLY CHECKED: R07 exact rational result, finite difference gradient agreement, local Hessian PSD, D4 center geometry.
LITERATURE-SUPPORTED: trained softmax expert policies can guide MCTS; progressive unpruning is not new; Rapfi trained Mixnet NNUE runs outside QUARTZ.
UNRESOLVED: actual trained surrogate's out-of-sample teacher agreement, learned-vs-uniform and attention-vs-full CPU matched effects, Rust implementation correctness, 15x15 strategy gains.
BLOCKED: unavailable byte-verified compatible trained Gomoku15 ONNX model; overcome by separate, explicitly described supervised surrogate path. Not a proof that such checkpoints cannot exist.

Decision: implement a small reproducible policy-only model and read causal contrasts as finite-source outcomes. Negative superiority result is permitted; neither lower tolerance nor test-tuned hyperparameters are permitted.