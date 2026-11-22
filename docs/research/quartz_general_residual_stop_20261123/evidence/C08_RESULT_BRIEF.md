# C23 held-out result, fixed before C09 review

72 independent TEST parents,9timing repeats each,3additional budgets,6actual policies. Times: parent medians followed by mean across parents,wall us; all setup/validation/solver/enclosures/destruction charged.

|Policy|Mean us|Positive-regret parents /72|Certified|Actually early|CG iterations saved|
|---|---:|---:|---:|---:|---:|
|NEVER|835.322458|10|0|72|6457|
|PLAIN_ALWAYS|1579.508542|2|0|0|0|
|ALWAYS|2752.759708|2|0|0|0|
|CERT_ONCE|2434.328931|2|17|17|1450|
|CERT_BLOCK|3061.350931|2|52|49|3236|
|RESIDUAL|4680.613222|0|0|40|-4572|

Strict certificate success and fixed-budget action accuracy are different: CERT_BLOCK52certificates are all exact-optimal,49actually exit before baseline work; two other parent problems remain uncertified and choose the same suboptimal finite-budget action as the controls. All decisions are correct in the tested cap128 cases, not a universal guarantee.

Against PLAIN_ALWAYS, CERT_ONCE mean time saved -854.820389us, paired parent-bootstrap95%[-1204.986504,-513.246428]; CERT_BLOCK -1481.842389us,95%[-2163.240360,-844.211282]. Therefore primary efficiency is negative. Against the more expensive ALWAYS, CERT_ONCE saves318.430778us, which is not evidence of beating the ordinary fixed solver.

Independent exact source/trace verification:741,888 original RHS components;216exact transport requests;1,296trace policy executions;26,898scalar interval containments;196certified parent-budget-policy outcomes,all oracle-optimal;236identical noncertified fallbacks. Timing11,664actual policy paths,104,976scalar containments; warmups excluded from measured summaries. Repeated cases are not independent parents.

TEST trace gzip SHA2569fdcc49d8b0d64832a3a0b7dd88f99b84d4caf7b6a7d84a3cf65a2cd1496d396 is preserved as three exact byte parts; coding/reassemble_c23_trace.py reconstructs it with SHA checks. This packaging changes no numeric payload.

C23_COST_QUALITY.png visually inspected: plotted values agree with the frozen summary; costs and decision errors are separately displayed. Scientific output is a valid source-conditioned STOP mechanism plus a negative total-cost result for the tested implementation.
