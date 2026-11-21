# R7. Executed independent finite calculations

**Execution**: python3 theory/check_r07.py (Python 3, NumPy, random seed 1507); actual machine output archived as R07_CHECKS.json. Wolfram Language independent exact checks: 1-Choose[208,4]/Choose[209,4]=4/209 True; derivative of three-action logsumexp-minus-target has zero discrepancy relative to softmax gradient.

Numerically checked: exact discovery fraction 4/209; 21-action 5-feature supervised softmax negative-log-likelihood gradient vs 1e-6 centered finite differences, max absolute error 3.374884816054191e-10. The 5×5 Hessian smallest eigenvalue 0.1312290525753647 (consistent with PSD), softmax normalization 1.0000000000000002, 1800 D4 tests for exact center-prior geometry invariance all passed. These checks validate the mathematical kernel, not any future trained model, Rust feature parity, or measured playing strength.

Limit: D4 checks here concern center geometry only; full learned features, Rust parity, teacher labels, independent heldout performance and CPU cost must be tested in C stages. No strong no-go theorem is claimed.