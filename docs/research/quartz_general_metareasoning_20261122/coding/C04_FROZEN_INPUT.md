# C4. Frozen finite, game-neutral scenario bank, before first new algorithm outcome

Input configs/metalevel_reference_cases.v1.json SHA256 **46d69d13ea327c23e02b76c53a0a78aadc1f77ee9e405dbef1c203a6d0b471dd**. Frozen C4 preflight coding/freeze_c04_cases.py used exact Fraction rational arithmetic to verify that every prior and conditional probability row is normalized, utilities have correct dimensions, λ>0 and computation costs are strictly positive. Direct source receipt evidence/C04_FROZEN_CASES.json. No C5 implementation results were consulted in fixture generation.

Five distinct finite reference tasks (no game-specific features or actions):
1. heterogeneous_complete_vs_partial: R4 3-state/3-action counterexample. Exact expected gain/rate/net and H1 Bellman predeclared.
2. one_action_always_dominant: informative perfect-state channel with **zero decision value** because the same terminal action always wins. Expected STOP, not entropy-based computation.
3. equal_cost_complete_and_partial: when costs match, both net and rate prefer the higher EVSI computation.
4. nonidentity_signed_utility: 3-state, 2-action asymmetric signed matrix with distinct observation channels, rejecting identity-payoff special casing.
5. two_state_noisy_sensor: 2-state posterior with imperfect binary observations and independent perfect-state reveal; H3 checks sequential Bayesian updating and information garbling.

Data identity: exactly5 distinct task IDs, 3 with heterogeneous positive costs; source and computational method include no game or hidden label. The finite empirical distribution in C8 is generated from these known synthetic channels, not natural observation laws or actual MCTS beliefs. Outcomes from later C8 cannot alter task, costs, channel or scientific acceptance tolerance. The C4 source is a benchmark fixture, **not** validation of play strength or human signatures.

Next observable milestone: C5 Python RED/GREEN and C6 Rust RED/GREEN, independent with exact R7 Fraction mathematics; complete debug/release regression. Later C7 freezes simulation runner before C8 outcomes.