# C27 coding loop

## C01 — baseline and acceptance

Feature: new finite joint-posterior controller and scientific campaign, not a refactor or claimed speedup. The R09 contract is frozen at commit7c15f04d96a98bdf38fc771f7ba2a5bb59e00049. Theory close commit8cc7e3bb6112d9e64e29d61b4abc0cc68cb0c939. Original authority87c15dc828d6eea519c219fd8a412fa21dfc464d remains the base.

Actual C01 environment: Python3.12.3; rustc/cargo1.99.0; AMD Ryzen9 5900X;24 logical CPUs; one serial campaign and bounded-j2build;1,939,759,104 bytes free at inspection. NumPy is available; Torch is absent. C26 release binary retains SHA8735b1f23a970248e4ef2b26d08c8df2daf181f0d29163317b5f41cc6ef0be0c. New worktree was clean.

Acceptance has four separate parts: derived reference, correct executable/trace semantics, frozen new cohort fullcost empirical result (which may fail), and verified backup/publication/handoff. Tests passing will not turn a negative net result into efficacy.

C26 complete-game solver/sign evidence and unchanged whole-tree regression evidence are reused by exactsourceidentity. The C27 posterior/H2/code/data boundaries require new focused tests. MissingTorch remains an environment limitation of the project-wide Python suite; it is not a C27 scientific failure. No installation or old campaign rerun is authorized merely to turn inherited reports green.

## C02 — traceability

coding/FORMULA_MAP.md maps every equation to data and a concrete new executable/harness obligation. The public/runtime/offline-label boundaries, raw-rollout initialization, unit conversion and projected control are explicit.

## C03 — numerical design

Stable log-odds/log1p and log-sum-exp forms, O(m²) H2, finite cap, tolerance and distinct closure/cost/roundoff/measurement errors are specified. Timing and repeated-root semantics are frozen.

## C04 — execution DAG

Two disjoint Rust/Python implementation lanes are released after this stage. Parent retains experiment locks, empirical execution and publication. Target cache reuse and disk bounds are explicit; implementation/tests precede TRAIN freeze, CAL, TEST and one fresh review.
