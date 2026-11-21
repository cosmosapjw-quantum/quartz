# R4. Constructive alternative search

A. Frozen C14 UniformEval + center-first: inexpensive but no learned strength, unsuitable main inference.
B. Reuse compressed Rapfi NNUE as if native ONNX: INVALID. mix9svq codebook / architecture incompatible with ONNX feature-contract and Rust loader; keep Rapfi as an external independent teacher only.
C. Imitate finite-time Rapfi moves with a small log-linear legal-action scorer (selected). z=theta·phi; full legal masking softmax; coefficient and feature order are explicit and auditable across NumPy and Rust. Faster and lower capacity than larger neural model; can give falsifiable negative results. No trained value head.
D. Teacher-distilled two-layer CNN/MLP or official pretrained Gomoku-compatible policy-value ONNX (deferred unless byte-verified model and runtime become accessible); stronger expressive prior, extra inference expense and training confounds.
E. Policy-free tactical reopening (old C14), kept as positive-control but not silently blended with learned model.

Main design 2x2 identification: learned-vs-uniform policy, gated-vs-all-legal candidate attention. Equal model is required for a causal question about attention. At the first 8 visits a gated policy preserves same root N/W/M2, then reopens an explicitly limited set of omitted candidates using only online one-ply board evidence. For avoiding teacher contamination, candidate rankings and reopening use ONLY frozen prior + legal rules. If no useful score at fixed physical cost, report a negative result rather than expand model size after examining test.