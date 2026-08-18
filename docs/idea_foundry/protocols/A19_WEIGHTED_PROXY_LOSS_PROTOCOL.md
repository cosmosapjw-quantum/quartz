# A19: RW-ResT Lite Evaluator Semantic Alignment & Surrogate Loss Protocol

## 1. Executive Summary & Semantic Clarification

- **Axis ID**: `A19` (`A19.native_rw_rest_lite_v1`)
- **Evaluator Architecture**: Random-Walk Rest-Tension Lite Policy-Value Architecture.
- **Previous Reporting Drift**: Variously referenced as policy logloss or win rate in early drafts.
- **Corrected Estimand & Unit**:
  - Estimand: `weighted_proxy_loss_delta` ($\Delta L_{\rm weighted}$)
  - Unit: `weighted_loss` ($L_{\rm weighted} = 1.0 \cdot L_{\rm policy\_ce} + 1.5 \cdot L_{\rm value\_mse}$)
  - Reference: Paired training against frozen baseline architecture under identical optimizer hyperparameters.
- **Scope & Non-Inference Invariant**: A reduction in surrogate proxy loss is **diagnostic training convergence evidence**, NOT a direct play-strength or Elo claim (`promotion.eligible = false`).

---

## 2. Mathematical Definition

### Loss Function
For evaluation batch $\mathcal{B} = \{(s_i, \pi_i^*, z_i)\}_{i=1}^B$:
$$L_{\rm policy}(s, \pi^*) = -\sum_{a} \pi^*(a) \log \pi_\theta(a \mid s)$$
$$L_{\rm value}(s, z) = (z - v_\theta(s))^2$$
$$L_{\rm weighted} = w_p \cdot L_{\rm policy} + w_v \cdot L_{\rm value} \quad (w_p = 1.0, w_v = 1.5)$$

### Paired Delta Estimand
$$\Delta L_{\rm weighted} = L_{\rm weighted}(\text{A19 RW-ResT}) - L_{\rm weighted}(\text{Baseline ResNet})$$
A negative value ($\Delta L_{\rm weighted} < 0$) indicates improved surrogate loss convergence.

---

## 3. Experimental Protocol

1. **Training Pairs**:
   - 3 paired seeds (seeds 11, 22, 33) initialized from identical weight distributions.
   - Matched dataset lineage: identical training replay batches drawn from frozen self-play corpus.
2. **Evaluations**:
   - Evaluated on held-out validation replay positions ($N \ge 10{,}000$ positions).
   - Bootstrapped paired $t$-interval across the 3 independent seed pairs.

---

## 4. Promotion Criteria

- **Surrogate Superiority**: $\Delta L_{\rm weighted} < -0.010$ with 95% CI upper bound $< 0$.
- **Inference Latency Invariant**: Forward pass latency on NVIDIA RTX 5070 Ti must remain within $1.15\times$ of baseline ResNet.
- **Phase 16 Promotion Requirement**: Once surrogate loss superiority is confirmed, a full 1,000-game arena tournament is required to establish play-strength (Elo) before production network replacement.
