# C5. Actual game-neutral Python reference implementation, observed RED → GREEN

Source: quartz/experiments/metalevel_decision.py (pure Python standard library, no game or Torch dependencies) + tests/test_metalevel_decision.py (seven externally observable, fixture-backed tests).

**Observed RED before implementation:** FiniteProblem.from_mapping was an intentional NotImplementedError stub. Executed python3 -m unittest discover -s tests -p test_metalevel_decision.py -v; exit nonzero and **7/7 tests errored solely because this promised behavior did not exist**. No fixture parser, syntax or import error caused the red. Recorded actual log at /tmp/quartz_meta17_python_red.log on the authorized workstation; red evidence is a local runtime receipt, not an artifact uploaded to Git.

**GREEN after implementation:** Identical seven-test command returned exit0, seven tests PASS, zero fail. Original output /tmp/quartz_meta17_python_green.log. Tests cover:
- 3-state heterogeneous costs: R=1/2, G=(1/5,1/2), rates (4,5/2), net (3/20,3/10), H0..H3 Bellman values, correct divergent selection;
- no decision-value observation ⇒ VOC0 and STOP even if posterior informativeness can exist;
- equal-cost choice agreement between rate and net;
- two additional generic families with signed utilities and noisy likelihood, monotonic finite-horizon objective;
- action permutation, zero-observation probability branch omission and input validation;
- computational resource unit conversion by1000 compensated by λ/1000, with identical Bellman choice and numerical result.

Implementation contract: immutable dataclasses for posterior, terminal utility state×action matrix and experiment likelihood state×outcome. Strict positive cost, finite positive λ, 1e-10 probability row normalization without stealth renormalization, no NaN, wrong dimensions rejected. Posterior update skips q(y)=0. Maximum finite horizon is 4 (reference resource cap). The core emits terminal Bayes action, perfect-information regret, calibrated EVSI, gross ratio and net gains separately, and exact-dispatch finite-horizon Bellman. STOP tie preference is deterministic.

**Not claimed:** a trained posterior observation model, compiled Rust parity, live SearchPolicy integration, game play-strength gain, CPU performance, or universal high-dimensional validity. C6 adds independently coded Rust reference and full regression; C8 real execution on C4 five frozen task families still remains unrun at this point.