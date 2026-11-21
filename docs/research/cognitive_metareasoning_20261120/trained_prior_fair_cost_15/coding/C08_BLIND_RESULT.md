# C08B. Actual 64×3=192 blinded Rust search requests, NO test teacher labels yet

Compiled native Rust binary C06 SHA256 1bf102650e28c9f0bbe9515f2f6e7d1d0c00a15f2646f78498ea79a9710d8527. Source code for all 192 requests was committed/published at C08A e9231eee3ab0d9674678001b072af23628cd85da before run. 64 newly frozen source histories (48 seeded synthetic and 16 constructed tactical) x 3 real QUARTZ 15x15 MCTS arms, policy-only actual training model SHA256 5ff8286dcc57192b79e960ba096bc5067c78d439a212eaa69d523f68835a3b83. Model weights were pretrained on separate 128+32 teacher splits, NOT on these 64 test labels; teacher file was absent at time of this native run. Source splits and arm execution orders were frozen, with cyclic arm order balancing.

All 192 native server requests returned valid, terminal-free legal search actions, with first8 and real continuation on preserved root edge statistics and root identity as checked by the runner. Recorded full actual prior, original root edges, bitset, selected move and wall time per call, no teacher labels passed online.

**Blind physical wall time per request** (model loading and JSON serialization excluded; includes policy evaluation, root setup, rules witness and MCTS):
- synthetic natural48: learned_gated 20076.958333µs vs learned_all_legal 20080.125000µs, absolute mismatch fraction **0.0001577264 (0.01577%)**, pass <=10%.
- engineered tactical16: learned_gated 20035.3125µs vs learned_all_legal 20031.9375µs, absolute mismatch fraction **0.0001684526 (0.01685%)**, pass <=10%.
- uniform_all_legal in natural 20052.833333µs, tactical 20039.875µs; this is a distinct prior factor, not the primary attention contrast.
- Raw file SHA256 09dc3b61dce9f4fdf0ce387b67f2669ace9811718fb08ab2498df4628316e5af; preserved under evidence/real_192_blind_20261121.

**This is only the blind compute acceptance gate.** At this point there is no external heldout teacher label agreement or evidence of policy superiority; C08C must independently obtain 64 actual Rapfi test recommended moves only after these raw and immutable CPU traces have been committed, pushed and archived. Do not claim Elo or exact game truth.