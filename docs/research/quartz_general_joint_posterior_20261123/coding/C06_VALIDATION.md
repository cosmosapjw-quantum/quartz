# C06 — correctness validation and TRAIN-only freeze

Status: implementation-verified and numerically checked. No C27 CAL/TEST labels or campaign outcomes were used for selection.

The actual C06 rerun passed18 native tests and21 Python tests. The independent exact finite oracle passed53851 assertions across336 joint model cases,5076 posterior histories,3024 initial H2 cases,120 threshold cases and81 policy-transfer cases. Final-native integration checked225 policy runs on3 fixed old TRAIN roots (3 rho values ×5 repeats ×5 policies per root), rejected5 invalid inputs and recovered on the following valid line. Source and logs are retained in evidence/C06_*.

The first oracle invocation omitted the required output file and exited1 at output writing. The corrected command exited0. This invocation error, the pre-fix formatting report and scoped style adapters are disclosed separately; no scientific assertion failure or hidden rerun-selected campaign is reported.

## Public assignment and honest model selection

The public allocation excludes all300 C26 roots. Remaining144 roots are assigned to CAL24 with ply counts2/8/14 and TEST120 with counts6/45/69, using the precommitted salt. These are new campaign roots; overlapping descendants and previously computed semantic oracle states remain possible.

Each of90 old TRAIN roots is held out exactly once in3folds. The C26 marginal model is refit on60 roots inside each fold. Mean heldout-root log likelihood improvements over the product model are:

| rho | mean logR |
|---:|---:|
|0|0|
|0.25|0.0160922175350882|
|0.5|0.05212105321334433|
|0.75|0.08255961204382367|
|0.9|0.0787166143314218|

The fixed rule chooses rho=0.75. Deployment uses the unchanged original full90 C26 base model, SHA256 0843447dc755c1d09e1bc69a44dd052e6db9c5c2ace41d1590b668cbfc36a126. The cross-validation score is a TRAIN selection statistic, not held-out empirical decision benefit.

## Fixed price and identities

All1480 instrumented TRAIN exact-query measurements were retained:1531550 total ns and32175 total solve nodes. Their ratio is47.6006216006216 ns/node. With lambda_nodes=2^-10, lambda_time_ns=2.0515750995723287e-05 utility/ns. This is the frozen local amortized calibration; its short query intervals and host dependence limit extrapolation.

Native executable SHA256:38126679ef622d7339fecdc7fd258ce2bb1428d82bbcb24545f26a7fafb66eef.
Harness SHA256:cfe068449f30a40350e5069513f9b0c6fa280517841f8ee95c3af7c900a5f75f.
Frozen bundle SHA256:dc1f76a16e769add95fa0dfb9b31454cc7cdbe7372787cc199573bf5673f40af.
Price raw SHA256:f7d343a91def0d1308058362cadbed7500482979217fe1aa9ec345ea77f34ce2.

The freeze includes final theory oracle and implementation-plan identities after cleanup-specific delivery amendments. Native controller, base model and experiment contract did not change. CAL and TEST must use this exact bundle; their outcomes cannot retune rho or price.

## Storage maintenance

The user's additional cleanup request was executed before timed pricing:12 inactive worktrees,11 old backup directories and19 superseded current archive copies removed. The C05 full source, all-ref history bundle and30 unique non-Git payloads were verified before deletion. See maintenance/CLEANUP_REPORT.md and its receipts. The initial retain-all-worktrees condition is superseded by recoverable retention, while a separately missing preexisting dirty reboot checkout remains external_or_unresolved.
