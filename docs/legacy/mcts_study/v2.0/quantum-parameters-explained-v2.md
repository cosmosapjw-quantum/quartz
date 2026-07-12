# Quantum Parameters Explained - Version 2.0
## Physics-Derived Parameter Selection Guide

## Table of Contents
1. [Overview](#overview)
2. [Core Parameters](#core-parameters)
3. [Derived Parameters](#derived-parameters)
4. [Phase-Dependent Selection](#phase-dependent-selection)
5. [Game-Specific Tuning](#game-specific-tuning)
6. [Parameter Relationships](#parameter-relationships)
7. [Practical Examples](#practical-examples)
8. [Troubleshooting Guide](#troubleshooting-guide)

## Overview

This guide explains how to select quantum MCTS parameters based on rigorous physics principles rather than empirical tuning. All parameters are derived from the fundamental structure of information time and renormalization group analysis.

### Key Principles

1. **Information Time**: τ(N) = log(N+2) determines all time-dependent parameters
2. **RG Fixed Points**: Optimal values emerge from scale invariance
3. **Phase Transitions**: Parameters adapt based on simulation count
4. **Prior Integration**: Neural network strength affects all parameters

## Core Parameters

### 1. Effective Planck Constant (ℏ_eff)

**Physics Definition**:
```
ℏ_eff(N) = c_puct(N+2)/(√(N+1)log(N+2))
```

**What it controls**:
- Strength of quantum fluctuations
- Exploration bonus magnitude
- Quantum-classical transition rate

**Key Properties**:
- Automatically decreases with N (classical limit)
- Scaled by c_puct (couples to exploration)
- Never needs manual tuning

**Implementation**:
```python
def compute_hbar_eff(N, c_puct):
    """Compute effective Planck constant"""
    return c_puct * (N + 2) / (math.sqrt(N + 1) * math.log(N + 2))
```

### 2. Temperature (T)

**Physics Definition**:
```
T(N) = T₀/log(N+2)
```

**What it controls**:
- Exploration vs exploitation balance
- Softmax sharpness in selection
- Convergence rate

**Initial Temperature (T₀)**:
- **Default**: T₀ = 1.0
- **High exploration**: T₀ = 2.0
- **Fast convergence**: T₀ = 0.5

**Annealing Schedule**:
```python
def temperature_schedule(N, T0=1.0, mode='logarithmic'):
    if mode == 'logarithmic':
        return T0 / math.log(N + 2)
    elif mode == 'power_law':
        return T0 / (N + 1)**0.5
    elif mode == 'exponential':
        return T0 * math.exp(-N/1000)
```

### 3. Prior Coupling (λ)

**Physics Definition**:
```
λ = c_puct[1 - ε/(2π)]
```

Where ε ≈ 0.1 is the RG expansion parameter.

**What it controls**:
- Neural network influence strength
- Prior-visit balance
- Learning from experience vs prior knowledge

**Optimal Values**:
- **Standard**: λ = c_puct (symmetric treatment)
- **Prior-heavy**: λ = 1.5 × c_puct
- **Experience-heavy**: λ = 0.5 × c_puct

### 4. Exploration Constant (c_puct)

**Physics Definition**:
```
c_puct = √(2 log b)[1 + 1/(4 log N_c)]
```

Where:
- b = branching factor
- N_c = critical simulation count

**Derivation**:
```python
def compute_optimal_cpuct(branching_factor):
    """Compute physics-optimal c_puct"""
    # Base value from information theory
    c_base = math.sqrt(2 * math.log(branching_factor))
    
    # RG correction
    N_c = branching_factor * math.exp(math.sqrt(2 * math.pi) / c_base) - 2
    correction = 1 + 1 / (4 * math.log(N_c))
    
    return c_base * correction
```

## Derived Parameters

### 1. Number of Hash Functions (K)

**Physics Definition**:
```
K = ⌊√(b × L)⌋
```

Where:
- b = branching factor
- L = average game length

**Intuition**: Balances interference complexity with computational cost.

### 2. Phase Kick Probability

**Physics Definition**:
```
p(N) = min(0.1, 1/√(N+1))
```

**Schedule**: Decreases with simulation count to maintain stability.

### 3. Update Interval

**Physics Definition**:
```
update_interval(N) = ⌊√(N+1)⌋
```

**Purpose**: Balances freshness with computational efficiency.

### 4. Decoherence Rate

**Physics Definition**:
```
Γ₀ = 2c_puct σ²_eval T₀
```

Where σ²_eval is the evaluation noise variance.

## Phase-Dependent Selection

### Detecting Current Phase

```python
def detect_phase(N, branching_factor, c_puct, has_prior=True):
    """Determine MCTS phase from simulation count"""
    
    # Compute critical points
    if has_prior:
        lambda_factor = c_puct * 0.8
        N_c1 = branching_factor * math.exp(math.sqrt(2*math.pi)/c_puct)
        N_c1 *= (1 + lambda_factor/(2*math.pi)) - 2
        
        N_c2 = branching_factor**2 * math.exp(4*math.pi/c_puct**2)
        N_c2 *= (1 + lambda_factor/math.pi) - 2
    else:
        N_c1 = branching_factor * math.exp(math.sqrt(2*math.pi)/c_puct) - 2
        N_c2 = branching_factor**2 * math.exp(4*math.pi/c_puct**2) - 2
    
    if N < N_c1:
        return 'quantum', N/N_c1
    elif N < N_c2:
        return 'critical', (N-N_c1)/(N_c2-N_c1)
    else:
        return 'classical', min(1.0, (N-N_c2)/N_c2)
```

### Phase-Specific Parameters

| Phase | Quantum Strength | Prior Trust | Batch Size | Temperature Boost |
|-------|-----------------|-------------|------------|-------------------|
| Quantum | 1.0 | 0.5 | 32 | 2.0 |
| Critical | 0.5 | 1.0 | 16 | 1.0 |
| Classical | 0.1 | 1.5 | 8 | 0.5 |

## Game-Specific Tuning

### Chess/Shogi (High Branching Factor)

```python
chess_params = {
    'branching_factor': 35,
    'c_puct': 3.0,  # √(2 log 35) ≈ 2.67 × 1.12
    'num_hashes': 42,  # √(35 × 70)
    'T0': 1.2,  # Slightly higher for tactics
}
```

### Go (Extreme Branching)

```python
go_params = {
    'branching_factor': 250,
    'c_puct': 3.9,  # √(2 log 250) ≈ 3.33 × 1.17
    'num_hashes': 150,  # √(250 × 90)
    'T0': 0.8,  # Lower for positional play
}
```

### Gomoku (Moderate Branching)

```python
gomoku_params = {
    'branching_factor': 50,
    'c_puct': 3.1,  # √(2 log 50) ≈ 2.79 × 1.11
    'num_hashes': 50,  # √(50 × 50)
    'T0': 1.0,  # Standard
}
```

## Parameter Relationships

### Coupling Diagram

```
c_puct ←→ ℏ_eff(N)
  ↓         ↓
  λ    ←→  T(N)
  ↓         ↓
 N_c   ←→  Γ₀
```

### Scaling Relations

| Parameter | Scaling with N | Scaling with b |
|-----------|---------------|----------------|
| ℏ_eff | ~ 1/√N | ~ √log b |
| T | ~ 1/log N | Independent |
| p_kick | ~ 1/√N | Independent |
| Update | ~ √N | Independent |

### Critical Points

```python
def compute_critical_points(b, c_puct, lambda_coupling):
    """Compute phase transition points"""
    
    # First critical point (quantum→critical)
    exp_factor_1 = math.exp(math.sqrt(2 * math.pi) / c_puct)
    prior_factor_1 = 1 + lambda_coupling / (2 * math.pi)
    N_c1 = b * exp_factor_1 * prior_factor_1 - 2
    
    # Second critical point (critical→classical)
    exp_factor_2 = math.exp(4 * math.pi / c_puct**2)
    prior_factor_2 = 1 + lambda_coupling / math.pi
    N_c2 = b**2 * exp_factor_2 * prior_factor_2 - 2
    
    return N_c1, N_c2
```

## Practical Examples

### Example 1: Early Game Exploration

```python
# High exploration for opening
early_game_config = {
    'phase_override': 'quantum',
    'temperature_boost': 2.0,
    'prior_trust': 0.3,  # Don't trust opening book too much
    'quantum_strength': 1.5,
    'batch_size': 64  # More diverse paths
}
```

### Example 2: Endgame Precision

```python
# Precise calculation for endgame
endgame_config = {
    'phase_override': 'classical',
    'temperature_boost': 0.3,
    'prior_trust': 2.0,  # Trust endgame tablebase
    'quantum_strength': 0.05,
    'batch_size': 4  # Focus on best lines
}
```

### Example 3: Balanced Middle Game

```python
# Adaptive middle game
def middle_game_config(position_complexity):
    base_config = detect_optimal_phase(N_simulations)
    
    # Adjust for position
    if position_complexity > 0.7:
        base_config['quantum_strength'] *= 1.3
        base_config['batch_size'] *= 2
    
    return base_config
```

## Troubleshooting Guide

### Issue: Too Much Exploration

**Symptoms**: Random-looking moves, slow convergence

**Solutions**:
1. Check if N > N_c1 (should be in critical/classical phase)
2. Reduce T₀: `T0 = 0.5`
3. Increase prior trust: `lambda = 1.5 * c_puct`
4. Force phase: `phase_override = 'classical'`

### Issue: Premature Convergence

**Symptoms**: Missing tactics, repetitive play

**Solutions**:
1. Increase T₀: `T0 = 1.5`
2. Enable quantum features: `quantum_level = 'one_loop'`
3. Reduce prior trust: `lambda = 0.7 * c_puct`
4. Check ℏ_eff scaling is correct

### Issue: Inconsistent Performance

**Symptoms**: Strength varies between positions

**Solutions**:
1. Enable phase adaptation: `enable_phase_adaptation = True`
2. Tune phase transition smoothing: `smoothing = 0.2`
3. Adjust critical points for game specifics
4. Use position-specific overrides

### Issue: High Computational Overhead

**Symptoms**: Slow simulations, GPU memory issues

**Solutions**:
1. Use tree-level quantum: `quantum_level = 'tree_level'`
2. Reduce batch sizes in quantum phase
3. Increase update interval: `update_interval = 2 * sqrt(N)`
4. Enable fast approximations: `fast_mode = True`

## Summary

The physics-derived parameters provide:
1. **No manual tuning**: All values computed from first principles
2. **Automatic adaptation**: Parameters evolve with simulation count
3. **Game-specific optimization**: Branching factor determines key values
4. **Phase-aware strategy**: Different regimes require different approaches

Remember: These parameters emerge from the mathematical structure of MCTS viewed through information time. Trust the physics!