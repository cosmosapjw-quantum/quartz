# C6. Actual compiled Rust/Python and cross-language verification

Source before tests: C05 commit 58b8f4a98cf2a6043e53cdddf8cca7028b4125ab, separate from previously stable C14. Ran on the authorized local workstation, with offline locked Cargo, Rust system compiler and NumPy.

- Native debug: cargo test --offline --locked, **608 passed,0 failed,89 ignored**.
- Native release: cargo test --offline --locked --release, **608 passed,0 failed,89 ignored**.
- Native release executable: cargo build --offline --locked --release, SHA256 **1bf102650e28c9f0bbe9515f2f6e7d1d0c00a15f2646f78498ea79a9710d8527**.
- Policy-only Python: python3 -m unittest -v test_policy15.py, 4 passed,0 failed. Validated 13 feature range, immediate-win/defensive indicator, error paths, D4 equivariance, analytic convex gradient/Hessian.
- Cross-language real native Rust: 80 independent legal-action feature inspection requests on training, validation and BOTH newly frozen test strata, with differences computed independently against Python. Max absolute feature deviation **3.405979698278827e-08**, below prespecified 2e-7. Actual C06_FEATURE_PARITY.json records arm/source/side group coverage and Rust compiled hash.
- Actual server rejects teacher injected request (unit test) and duplicated history; 225 uniform legal actions and value0 checked. Full original C14 regression tests passed in both configurations.

These tests prove source path and feature parity, NOT teacher training, general Gomoku strength or equal-cost search (C7/C8). Raw logs and parsed parity saved in evidence/c06; all Rust compiler warnings are retained without suppressing them. C07 must evaluate real teacher labels and frozen weights, and C08 still needs real matched-time comparisons.