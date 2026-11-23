# R06 parent independent condition check

The parent read the complete constructor proof and independently derived the switch inequality and the two cases for G after a switch. Writing M=max U, strict L_j>U_i ensures the maximizing upper endpoint is outside i; G_i=M-L_i and G_j<=M-L_j. This proves the claimed decrease without requiring max-L to minimize regret. The latter would be false: intervals [1/2,1], [3/5,7/10] give gaps1/5 and2/5.

Checked finite nonempty intersection, fixed target utilities, all possible output actions, strict versus weak optimum and exact ties, budget exhaustion, lambda=0, zero floor, zero-progress block, and mandatory K fresh readouts. General theorem permits K=1 by convention; runtime domain remains K=2..8. No arithmetic-domain expansion is inferred.

Price units are utility/cell-stencil or utility/microsecond, not interchangeable. Retention is compared with the identical initial PLAIN_STOP action; it does not inherit C24's noninferiority to the final PLAIN_ALWAYS score. Probability statements for search require a simultaneous coverage event. Empty intervals and invalid inputs remain explicit errors.

The free-action comparison is deliberately limited: process certified available changes before comparing paid continuation; the theorem does not rule out arbitrary free uncertified reselection. Counterfactual remaining cost excludes sunk initial cost, whereas measured policy totals include it.

Independent means a separate parent mathematical/checker construction, not independent experimental replication. R07 supplies exact finite checks; the general proof remains in DERIVATION.md. No unresolved proof defect was found within the stated contract. Actual native work-counter enforcement is a coding obligation.
