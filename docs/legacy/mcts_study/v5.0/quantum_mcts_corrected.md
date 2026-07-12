# Quantum MCTS: Corrected One-Loop Formulation

## Overview

This document presents the **corrected and final** formulation of quantum-enhanced Monte Carlo Tree Search (MCTS) using path integral methods from quantum field theory. The key correction involves using the quadratic kernel action S_cl = κ*N_total*Σ(√q_k - p_k)² which leads to the exact augmented PUCT formula with quantum bonus 3*ℏ_eff/(4*N_k).

## 1. Classical Action (Corrected)

The classical action uses a **quadratic kernel** (not Hellinger distance):

```
S_cl = κ*N_total*Σ_k(√q_k - p_k)² - β*Σ_k N_k*Q_k
```

Where:
- `κ = c_puct` (exploration coefficient)
- `q_k = N_k/N_total` (visit fractions)
- `p_k` = neural network priors
- `β` = value weight coefficient
- `Q_k` = backed-up Q-values

**Critical**: The form (√q_k - p_k)² is essential - it gives the exact PUCT exploration term when differentiated.

## 2. Exact Hessian

For the quadratic kernel action, the exact Hessian is:

```
h_k = ∂²S_cl/∂N_k² = (κ*p_k/2) * sqrt(N_total/N_k³)
```

This is **not** an approximation - it's the exact second derivative.

## 3. One-Loop Quantum Correction

The one-loop correction from the Gaussian path integral:

```
ΔΓ^(1) = (ℏ_eff/2) * Σ_k log(h_k)
     = (ℏ_eff/2) * Σ_k [-3/2 * log(N_k) + const]
```

Taking the gradient with respect to N_k:

```
-∂ΔΓ^(1)/∂N_k = +3*ℏ_eff/(4*N_k)
```

## 4. RG Counter-Term (Pruning Entropy)

When B_trim children are pruned (integrated out), the Jacobian contributes:

```
ΔΓ_RG = ℏ_eff * log(1 + B_trim)
```

This term:
- Arises from the shrinking probability simplex
- Is added ONCE per parent node
- Does NOT affect child-to-child ranking (cancels in gradients)
- Manifests as -ℏ_eff*log(1+b)/N per visit for parent selection

## 5. Complete Effective Action

```
Γ_eff = S_cl + ΔΓ^(1) + ΔΓ_RG + const
```

## 6. Augmented PUCT Selection Rule

Taking the gradient for edge selection:

```
Score(k) = κ*p_k*sqrt(N_total/N_k) + 3*ℏ_eff/(4*N_k) - β*Q_k
```

Where:
- First term: Classical PUCT exploration
- Second term: Quantum bonus (coefficient 3/4 is EXACT, not tunable)
- Third term: Value exploitation

## 7. Implementation

```python
QUANTUM_BONUS_COEFF = 0.75  # Exactly 3/4 from theory

def augmented_puct_score(edge, parent):
    N_total = parent.total_visits
    N_k = edge.visits
    p_k = edge.prior
    Q_k = edge.q_value
    
    # Classical PUCT
    exploration = kappa * p_k * sqrt(N_total / max(1, N_k))
    
    # Quantum bonus - coefficient is FIXED by theory
    quantum_bonus = QUANTUM_BONUS_COEFF * hbar_eff / max(1, N_k)
    
    # Value term
    exploitation = -beta * Q_k
    
    return exploration + quantum_bonus + exploitation
```

## 8. Jarzynski Equality Connection

The RG counter-term satisfies a Jarzynski-type equality:

```
<exp(-ΔS_hard/ℏ_eff)>_hard = exp(-ℏ_eff*log(1+b))
```

This maps quantum MCTS pruning to nonequilibrium statistical mechanics:
- β ↔ 1/ℏ_eff (information temperature)
- W ↔ ΔS_hard (entropy cost of pruning)
- ΔF ↔ ℏ_eff*log(1+b) (free energy change)

## 9. Key Physics Insights

1. **Quadratic vs Hellinger**: The quadratic kernel (√q_k - p_k)² is necessary for exact PUCT recovery
2. **Fixed Quantum Coefficient**: The factor 3/4 in the quantum bonus is not a hyperparameter
3. **RG Term**: Accounts for information loss from pruning, improving diversity
4. **Jarzynski Mapping**: Connects to fundamental nonequilibrium physics

## 10. Effective Planck Constant

The effective Planck constant ℏ_eff controls quantum effects:

```
ℏ_eff = ℏ_base / sqrt(1 + N_total/N_scale)
```

Where:
- `ℏ_base` ~ 1.0 (baseline quantum strength)
- `N_scale` ~ 100-1000 (transition scale)
- Ensures quantum → classical limit as N → ∞

## Summary

The corrected quantum MCTS formulation:
1. Uses quadratic kernel action S_cl = κ*N_total*Σ(√q_k - p_k)²
2. Gives exact Hessian h_k = (κ*p_k/2)*sqrt(N_total/N_k³)
3. Produces quantum bonus +3*ℏ_eff/(4*N_k) with fixed coefficient
4. Includes RG term ℏ_eff*log(1+B_trim) for pruning
5. Satisfies Jarzynski equality for nonequilibrium consistency

This is the complete, mathematically rigorous formulation suitable for both theoretical analysis and practical implementation.