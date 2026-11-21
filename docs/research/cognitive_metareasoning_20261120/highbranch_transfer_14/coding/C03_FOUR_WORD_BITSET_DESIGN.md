# C3. Numerical design, conservation and failure semantics

Representation: words M[0..3] each u64, mask cell a using M[a/64] |= 1_u64 << (a % 64), validated a∈0..224. All-zero 4-word mask is the unrestricted sentinel; API with zero requested actions returns error. On a searched root, the old nonzero mask must be subset of every new mask wordwise. A searched unrestricted root cannot later be artificially restricted. Setter requires exclusive mutable engine access, serial root searches and fully materialized original root edge slab.

Selection: depth0 only, filter every edge whose move_to_idx lies outside mask; first-index tie fallback must not select an inactive edge. Depth>=1 selects ordinary full game legal edges; no synthetic Gomoku board pruning. At checkpoint compare entire root edge (N,W,M2,p) snapshot bit-for-bit and root identity, verify newly admitted high-index actions N=0. On completion root visit8 then32; sum edge visits must be32. Note root state GameState::legal_moves always all legal moves.

Finite mask checks are proved in theory/R05 and executed theory/combinatorial_reopening14.py for 225 singleton ids, all 25,200 two-move pairs and all critical 64-bit boundaries. The Rust test must still fail RED before implementation and pass GREEN; Python math alone does not prove the Rust hot path. No performance claim follows from equal visit count. Measure actual wall-cost separately.

Strata: neutral original legal openings vs preconstructed forced-win stress; avoid pooling as representative common game situations. Two distinct reference methods: exact local five-in-row detection; Rapfi finite trained search best move only, with missing/error explicit. Theoretical rare-event discovery is not a guaranteed selected-move improvement.
