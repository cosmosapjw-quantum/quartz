# C07 — fixed CAL diagnostics

Status: implementation-verified; diagnostic campaign completed. The settings frozen at C06 were used unchanged.

CAL24 roots (ply4/5/6 counts2/8/14),4 policy seeds,5 measured repeats and5 policies produced96 native requests and2400 validated policy runs. There was one unmeasured warmup per policy/seed. Campaign, independent trace verification and analysis exited0. Runtime requests did not include offline labels; each label was generated only after the model/source/price freeze.

## Paired JOINT_H2 minus PRODUCT_H2

|Metric|Root mean difference|95% descriptive root-bootstrap interval|
|---|---:|---|
|additional_nodes|0|[0, 0]|
|additional_queries|0|[0, 0]|
|charged_nodes|0|[0, 0]|
|elapsed_ns|229.71875|[90.7799479167, 389.86484375]|
|net_nodes|0|[0, 0]|
|net_wall|-0.00471285267405|[-0.00799837005636, -0.00186241880686]|
|regret|0|[0, 0]|
|utility|0|[0, 0]|

Every CAL root has zero paired utility, charged-node and query-count difference. The primary wall-priced result is negative because the instrumented native elapsed time differs. These short timing differences describe this machine and instrumented controller; serialization/IPC are separately reported.

The primary net estimate is -0.00471285267405. This is CAL, so empirical admission is deliberately not evaluated. Neither the negative net result nor the conditional diagnostics is used to adjust rho, marginal model, cost, source assignment, repeat count or policy.

## Diagnostic scope and next stage

Conditional predictive losses and selected direct-gain errors are recorded for each generating policy/stage on common candidate-history exposures. Initial conditioning slightly improves average categorical loss here, while later-history loss can worsen; no blanket calibration improvement is inferred. The fixed TEST120 campaign proceeds with the same frozen bundle dc1f76a16e769add95fa0dfb9b31454cc7cdbe7372787cc199573bf5673f40af.

Uncertainty uses10000 root-resamples (PCG64, seed2709), after median5 repeats then mean4 policy seeds per root. Roots can share descendants; intervals are descriptive and do not assert independent-game/population coverage.
