# C05 implementation-time interpretation notes

The native implementation was inspected at its actual posterior/H2/initialization/run path. The paired primary compares prior-product versus joint conditioning, while the projected control records its initial marginals before freezing. Known terminal observations contribute no latent evidence in the chosen degenerate component model. H2 hypothetical updates are local; observed markers prevent repeated actual evidence.

The probabilities are a bounded-computation statistical belief model over exact values. A complete legal board mathematically fixes deterministic minimax values; the finite learned cue representation does not make the game a physically random latent generator. Calibration diagnostics are evaluated across the frozen root/cue collection and must not be sold as a theorem of the true conditional law given an omniscient board description.

Native controller timings include trace construction and instrumented planning. RANDOM_TWO also computes diagnostic scores when tracing; its timing is therefore that of this instrumented comparator, not an optimized bare random policy. The primary JOINT_H2/PRODUCT_H2 both require H2 scoring. All five policies retain their declared instrumentation cost. No end-to-end application latency or P2 Gomoku gain is inferred.

C04 GitHub tests-gate run45167571238: cargo test--release and real-loop smoke succeeded; pytest job stopped at Check Ruff formatting, so pytest was skipped. This predates C27 implementation. gh --log-failed returned no log body; the job-step metadata is retained as evidence rather than inventing a file-level failure. Local Torch is absent. New scoped implementation validation is separate.
