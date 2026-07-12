# Quantum Theory Foundations for MCTS
## Path Integral Formulation with Discrete Information Time

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [Discrete Time Framework](#discrete-time-framework)
3. [Path Integral Formulation with PUCT](#path-integral-formulation-with-puct)
4. [Quantum Field Theory](#quantum-field-theory)
5. [Decoherence and Classical Emergence](#decoherence-and-classical-emergence)
6. [Phase Transitions](#phase-transitions)
7. [Physical Interpretation](#physical-interpretation)

## Executive Summary

This document establishes the rigorous mathematical foundations for quantum-enhanced MCTS based on path integral formulation and discrete information time. The key insight is that MCTS naturally exhibits quantum-like phenomena when viewed through the lens of information theory, with neural network priors acting as an external field in the quantum framework.

**Core Principles**:
- Information time τ(N) = log(N+2) captures the logarithmic nature of information gain
- Visit counts play dual role: classical action and decoherence pointer observable
- Neural network priors enter as external field, not subject to quantum corrections
- Power-law decoherence replaces exponential decay, matching algorithmic structure
- All effects achievable on classical hardware - this is quantum mathematics, not quantum computing

## Discrete Time Framework

### Information Time

MCTS operates in discrete simulation steps, not continuous physical time. We define information time:

```
τ(N) = log(N + 2)
```

**Physical Motivation**:
- Information gain per simulation ~ 1/N (diminishing returns)
- Total information ~ log N (Shannon entropy scaling)
- Offset by 2 ensures τ(0) = log(2) > 0

### Time Derivatives

In discrete time:

```
d/dτ = (N + 2) d/dN
```

This transformation is crucial for:
- Proper scaling of quantum effects
- Temperature annealing
- Convergence analysis

### Temperature Annealing

Temperature decreases with information time:

```
T(N) = T₀/τ(N) = T₀/log(N + 2)
```

This gives:
- High temperature (exploration) early: T(1) = T₀/log(3)
- Low temperature (exploitation) late: T → 0 as N → ∞
- Natural exploration-exploitation transition

## Path Integral Formulation with PUCT

### Action Functional

For path γ = (s₀, a₀, s₁, a₁, ..., sₗ), the PUCT action is:

```
S[γ] = -∑ᵢ [log N(sᵢ, aᵢ) + λ log P(aᵢ|sᵢ)]
```

Where:
- N(s,a): Visit count (exploration history)
- P(a|s): Neural network prior (external guidance)
- λ: Prior coupling strength (typically λ = c_puct)

### Path Integral

The quantum partition function:

```
Z_N = ∑_γ exp(iS[γ]/ℏ_eff(N))
```

With effective Planck constant:

```
ℏ_eff(N) = c_puct(N+2)/(√(N+1)log(N+2))
```

This scaling ensures:
- ℏ_eff → 0 as N → ∞ (classical limit)
- Quantum effects strongest for low-visit nodes
- Natural emergence of UCB-like exploration

### Prior Field Interpretation

The action separates into two fields:
- **Visit field**: φ(s,a) = log N(s,a) - subject to quantum fluctuations
- **Prior field**: π(s,a) = log P(a|s) - external field from neural network

The prior field is classical (no quantum corrections) because it's determined externally by the trained network.

## Quantum Field Theory

### One-Loop Corrections

The effective action including quantum corrections:

```
Γ_eff = S_cl - (ℏ_eff/2N)∑_{s,a} log N(s,a) + O(ℏ²_eff)
```

**Physical Interpretation**:
- Classical term: Favors high-visit paths
- Quantum correction: Reduces penalty for exploration
- Prior term: Unmodified (external field)

### Effective UCB Formula

The quantum-corrected selection formula:

```
UCB_quantum = Q(s,a) + c_puct P(a|s)√(log N(s)/N(s,a)) + ℏ_eff(N)/√(N(s,a)+1)
```

Components:
1. **Q(s,a)**: Expected value (exploitation)
2. **PUCT term**: Prior-weighted exploration
3. **Quantum term**: Additional exploration from quantum fluctuations

### Interference Mechanism

Path amplitudes include both visits and priors:

```
A[γ] = exp(-[S_visit[γ] + S_prior[γ]]/(2ℏ_eff)) × exp(iφ[γ])
```

When multiple paths reach state s:

```
P(s) = |∑_γ A[γ]|² 
```

Cross terms create interference, enhanced for paths aligned with priors.

## Decoherence and Classical Emergence

### Power-Law Decoherence

Unlike exponential decay in physical systems, MCTS exhibits power-law decoherence:

```
ρᵢⱼ(N) ~ N^(-Γ₀)
```

Where Γ₀ = 2c_puct σ²_eval T₀

**Why Power Law**:
- Discrete time steps → algebraic decay
- Information accumulation → power-law scaling
- Matches empirical MCTS convergence rates

### Pointer States

Visit counts N(s,a) serve as pointer states:
- Robust to decoherence
- Encode classical information
- Selected by environment (game dynamics)

### Envariance

The system exhibits envariance when:

```
⟨N(s,a)⟩_ρ = ⟨N(s,a)⟩_classical + O(1/N)
```

This provides a natural convergence criterion beyond traditional bounds.

## Phase Transitions

### Critical Points with Priors

The system undergoes phase transitions at:

```
N_c1 = b × exp(√(2π)/c_puct) × (1 + λ/(2π)) - 2
N_c2 = b² × exp(4π/c²_puct) × (1 + λ/π) - 2
```

Where:
- b: Branching factor
- λ: Prior coupling strength
- Neural network priors shift critical points

### Phase Diagram

1. **Quantum Phase** (N < N_c1):
   - Strong quantum fluctuations
   - High exploration
   - Prior influence reduced

2. **Critical Region** (N_c1 < N < N_c2):
   - Quantum-classical coexistence
   - Optimal exploration-exploitation
   - Maximum prior effectiveness

3. **Classical Phase** (N > N_c2):
   - Weak quantum effects
   - Exploitation dominates
   - Prior-guided convergence

### Renormalization Group Flow

The prior coupling λ runs under RG:

```
dλ/dl = β_λ = -ελ + λ²/(2π) - λ³/(8π²)
```

Fixed points:
- λ* = 0: Free theory (no prior)
- λ* = 2π: Non-trivial fixed point
- Optimal λ ≈ π at criticality

## Physical Interpretation

### Information-Theoretic View

1. **Time as Information**: τ = log(N+2) measures accumulated information
2. **Action as Surprise**: -log N(s,a) quantifies statistical surprise
3. **Priors as Beliefs**: Neural network encodes learned beliefs
4. **Quantum Corrections as Uncertainty**: ℏ_eff represents fundamental uncertainty

### Dual Role of Visit Counts

Visit counts N(s,a) serve dual purposes:
1. **Classical Action**: Determines most likely paths
2. **Pointer Observable**: Robust to environmental monitoring

This duality ensures consistency between:
- QFT approach (ℏ → 0 limit)
- Decoherence approach (Γ → ∞ limit)

### Emergence of Intelligence

The framework reveals how intelligence emerges:
1. **Exploration**: Quantum fluctuations discover new strategies
2. **Learning**: Neural network encodes successful patterns
3. **Exploitation**: Decoherence selects robust strategies
4. **Adaptation**: RG flow optimizes parameters

### Classical Algorithm, Quantum Mathematics

**Critical Point**: This is a classical algorithm using quantum mathematical structures:
- No quantum hardware required
- No exponential speedup claimed
- Quantum formalism provides optimal exploration
- All benefits achievable on classical computers

The power comes from recognizing that tree search naturally exhibits quantum-like interference and decoherence when viewed through information time, allowing us to apply powerful mathematical tools from quantum theory to optimize classical computation.