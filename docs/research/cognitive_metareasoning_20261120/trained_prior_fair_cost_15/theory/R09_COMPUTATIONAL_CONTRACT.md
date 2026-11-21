# R9. Source-to-code and data contract frozen before model/teacher execution

Input: history h of unique flat moves 0..224, strictly alternating 1=black and 2=white, terminal-free Freestyle15. For action a=(x,y), candidate board entry empty, side t. phi(s,a) ∈ [0,1]^13 in this exact order:
0 constant 1;
1 1-ChebyshevDistance((x,y),(7,7))/7;
2 1-ManhattanDistance((x,y),(7,7))/14;
3,4 own/opponent stone count in the 8-neighborhood divided8;
5,6 own/opponent stone count in the Chebyshev distance-2 ring of 16 divided16;
7,8 longest four-direction contiguous own/opponent run through hypothetical action, capped5, divided5;
9,10 maximum same-side stones including the hypothetical move among any contiguous 5-cell line window through a, provided no opponent stone in window, divided5;
11,12 exact indicators for own/opponent hypothetical 5-in-row through a.
Stone run/window calculations are board-only and do not look ahead at teacher or opponent recommendation. For every legal a, logit theta·phi, stable legal softmax at T=1; trained policy-only value=0. L2 penalty lambda=0.01; initial theta zero, deterministic full-batch optimization. Learning targets are finite-time (not exact) Rapfi rule0 legal moves. Model JSON records feature order, training source hashes, model hash, optimizer settings, no Rapfi external weights.

Data: generate NEW independent training/validation/heldout source trajectories from fixed seeds. Split before any teacher evaluations, store source manifest SHA, reject overlap with prior C14, keep separate engineered tactical diagnostic group, no symmetry-transformed clone crossing. Only train+validation teacher labels are read when choosing theta; freeze model commit before test teacher actions. Teacher exe/model/config remain external read-only hashed dependencies, never Git artifacts.

Rust endpoint: research-only command. Load frozen JSON trained weights once or per request with timing disclosed, use trained policy for all MCTS evaluations (not just root mask) under native full-legal actions, compare learned-all-legal / learned-gated and separate uniform-all-legal. Root gating begins at 8 visits; second phase continues same root with N/W/M2 preservation and no external teacher values. Physical elapsed wall target includes model eval, prior, probe, masks, all MCTS calls and excludes JSON serialization. For each finite frozen stratum require per-stratum actual cost mismatch <=10% before admitting any "matched CPU" statement. Record cycles/iterations, timing, legal actions, model identity, root invariants, teacher agreement, an auditable raw trace and independent readback.