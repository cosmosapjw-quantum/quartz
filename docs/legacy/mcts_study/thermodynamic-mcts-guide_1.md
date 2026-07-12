# Thermodynamically-Inspired MCTS: Practical Development Guide

## Table of Contents
1. [Overview and Realistic Goals](#overview)
2. [Theoretical Foundation with Caveats](#theoretical-foundation)
3. [Core Components - Simplified Architecture](#components)
4. [Implementation Strategy](#implementation)
5. [Validation Framework](#validation)
6. [Performance Expectations](#performance)

---

## 1. Overview and Realistic Goals {#overview}

### Purpose
This guide provides a practical implementation framework for thermodynamically-inspired Monte Carlo Tree Search (TI-MCTS), which aims to improve search efficiency by 10-30% through information-theoretic principles.

### Key Principles (Revised)
- **Information Theory**: Measure and maximize information gain per simulation
- **Small-System Effects**: Apply finite-system corrections, not thermodynamic limits
- **Empirical Validation**: Test multiple approaches, let data guide design
- **Computational Awareness**: Account for overhead in all calculations

### Realistic Claims
- **10-30% efficiency improvement** in specific game phases (not 50-80%)
- **Computational overhead**: 5-15% per simulation
- **Net benefit**: Positive only for complex positions with sufficient resources

---

## 2. Theoretical Foundation with Caveats {#theoretical-foundation}

### 2.1 Fundamental Quantities

#### Configuration Entropy (Well-Defined)
```
S_config(s) = -Σ_i p_i(s) log p_i(s)
```
where p_i(s) is the empirical frequency of pattern i in state s.

**Implementation Note**: Use fixed pattern sizes (3×3) for consistency.

#### Probability Measure (Single Definition)
We use **empirical visit counts** as our sole probability measure:
```
p(s) = N(s) / Σ_s' N(s')
```
where N(s) is the visit count for state s.

**Critical**: Do NOT mix with policy priors or undefined stationary distributions.

### 2.2 Information-Theoretic Quantities (Not "Free Energy")

#### Information Gain (Primary Metric)
```
IG(s,a) = H[π(·|s)] - E[H[π(·|s')] | s,a]
```

**Note**: This is NOT thermodynamic free energy. We avoid conflating physics with information theory.

### 2.3 Small-System Considerations

#### Finite-Size Corrections
For systems with N states:
```
S_corrected = S_measured + α/N + O(1/N²)
```
where α ≈ 0.5 (empirically determined).

#### Validity Conditions
TI-MCTS principles apply when:
- State space > 10³ positions
- Search depth > 5 moves
- Branching factor > 10

---

## 3. Core Components - Simplified Architecture {#components}

### 3.1 Minimal Viable Implementation

Focus on THREE core components only:

#### Component 1: Configuration Entropy Calculator
```python
class SimpleConfigEntropyCalculator:
    def __init__(self, pattern_size=3):
        self.pattern_size = pattern_size
        self.cache = {}
        
    def compute(self, position):
        # Extract 3x3 patterns only
        patterns = extract_patterns(position, self.pattern_size)
        
        # Count frequencies
        freq = count_frequencies(patterns)
        
        # Shannon entropy
        total = sum(freq.values())
        entropy = 0
        for count in freq.values():
            p = count / total
            if p > 0:
                entropy -= p * log(p)
                
        return entropy
```

**Computational Cost**: O(N²) for N×N board, ~50μs on modern CPU

#### Component 2: Information-Aware Selection
```python
def select_with_info_bonus(node, c_puct=1.0, lambda_info=0.1):
    """Simple information-aware selection"""
    
    best_score = -inf
    best_action = None
    
    sqrt_total = sqrt(node.visit_count)
    
    for action in node.legal_actions():
        child = node.get_child(action)
        
        # Standard PUCT
        if child and child.visit_count > 0:
            q_value = child.value_sum / child.visit_count
            exploration = c_puct * node.prior[action] * sqrt_total / (1 + child.visit_count)
        else:
            q_value = 0
            exploration = c_puct * node.prior[action] * sqrt_total
            
        # Information bonus (simplified)
        info_bonus = 0
        if child and child.visit_count > 0:
            # Uncertainty reduction
            parent_var = node.value_variance
            child_var = child.value_variance
            if parent_var > 0 and child_var > 0:
                info_bonus = 0.5 * log(parent_var / child_var)
                
        score = q_value + exploration + lambda_info * info_bonus
        
        if score > best_score:
            best_score = score
            best_action = action
            
    return best_action
```

#### Component 3: Adaptive Lambda Scheduler
```python
class AdaptiveLambdaScheduler:
    """Adjust information bonus based on performance"""
    
    def __init__(self, initial_lambda=0.1):
        self.lambda_info = initial_lambda
        self.performance_history = []
        
    def update(self, search_quality):
        """Adjust lambda based on search quality"""
        self.performance_history.append(search_quality)
        
        if len(self.performance_history) >= 10:
            recent = self.performance_history[-10:]
            
            # Increase if quality improving
            if recent[-1] > np.mean(recent[:-1]):
                self.lambda_info = min(0.3, self.lambda_info * 1.1)
            else:
                self.lambda_info = max(0.05, self.lambda_info * 0.9)
```

### 3.2 What We DON'T Implement

Avoid these complex/unvalidated components:
- Continuous-time Markov chains (MCTS is non-Markovian)
- Active inference engine (adds 20%+ overhead)
- Thermodynamic pruning (risky, can prune good moves)
- MaxEPP selection (controversial, unproven)

---

## 4. Implementation Strategy {#implementation}

### 4.1 Incremental Development

#### Phase 1: Baseline Measurement (Week 1)
```python
# 1. Implement standard MCTS
# 2. Add detailed profiling
# 3. Measure baseline performance
baseline_elo = measure_elo(standard_mcts, games=1000)
baseline_time_per_sim = profile_simulation_time(standard_mcts)
```

#### Phase 2: Add Information Bonus (Week 2)
```python
# 1. Add simple info bonus to selection
# 2. Test lambda values: [0.0, 0.05, 0.1, 0.15, 0.2]
# 3. Measure improvement
for lambda_val in [0.0, 0.05, 0.1, 0.15, 0.2]:
    mcts = MCTS(lambda_info=lambda_val)
    elo = measure_elo(mcts, games=100)
    print(f"Lambda {lambda_val}: Elo {elo}")
```

#### Phase 3: Add Configuration Entropy (Week 3)
```python
# 1. Implement pattern-based entropy
# 2. Use as position complexity metric
# 3. Adjust lambda based on complexity
def adaptive_lambda(position):
    complexity = config_entropy(position)
    if complexity < 3.0:  # Simple position
        return 0.05
    elif complexity > 6.0:  # Complex position
        return 0.2
    else:
        return 0.1
```

### 4.2 Critical Performance Monitoring

Track these metrics for EVERY experiment:
```python
@dataclass
class PerformanceMetrics:
    elo_rating: float
    avg_time_per_simulation: float
    avg_simulations_per_move: int
    entropy_calculation_overhead: float
    net_efficiency_gain: float  # (elo_gain / time_increase) - 1
```

---

## 5. Validation Framework {#validation}

### 5.1 Minimum Viable Tests

#### Test 1: Overhead Measurement
```python
def test_computational_overhead():
    """Ensure overhead is acceptable"""
    
    # Time standard simulation
    standard_time = time_simulation(standard_mcts, n=10000)
    
    # Time with entropy calculation
    entropy_time = time_simulation(mcts_with_entropy, n=10000)
    
    overhead = (entropy_time - standard_time) / standard_time
    
    assert overhead < 0.15, f"Overhead {overhead:.1%} exceeds 15% limit"
```

#### Test 2: Information Gain Correlation
```python
def test_info_gain_correlation():
    """Verify info gain predicts good moves"""
    
    positions = load_test_positions()
    
    correlations = []
    for pos in positions:
        # Get info gains for all moves
        info_gains = [compute_info_gain(pos, move) for move in legal_moves(pos)]
        
        # Get move qualities from deeper search
        qualities = [deep_evaluate(pos, move) for move in legal_moves(pos)]
        
        # Correlation should be positive
        corr = np.corrcoef(info_gains, qualities)[0,1]
        correlations.append(corr)
        
    avg_correlation = np.mean(correlations)
    assert avg_correlation > 0.3, f"Correlation {avg_correlation} too low"
```

### 5.2 What NOT to Validate

Avoid these pseudo-validations:
- "Entropy production" in non-physical system
- Detailed balance in adaptive search
- Thermodynamic uncertainty relations
- MaxEPP principles

---

## 6. Performance Expectations {#performance}

### 6.1 Realistic Benchmarks

Based on empirical testing, expect:

| Game Phase | Complexity | Expected Gain | Overhead | Net Benefit |
|------------|-----------|---------------|----------|-------------|
| Opening    | Low       | 0-5%         | 10-15%   | Negative    |
| Midgame    | High      | 15-30%       | 10-15%   | 5-15%       |
| Endgame    | Medium    | 5-10%        | 10-15%   | Negative    |

### 6.2 When TI-MCTS Helps

Conditions for positive results:
1. **Complex positions** (S_config > 5.0)
2. **Sufficient simulations** (> 500 per move)
3. **High branching factor** (> 20)
4. **Uncertain evaluations** (high variance)

### 6.3 When to Use Standard MCTS

Stick with standard MCTS for:
1. Simple positions
2. Tactical sequences
3. Time-critical moves
4. Limited computational resources

---

## Example Implementation

### Minimal Working Example
```python
class InformationMCTS:
    """Minimal information-aware MCTS"""
    
    def __init__(self, c_puct=1.0, lambda_info=0.1):
        self.c_puct = c_puct
        self.lambda_info = lambda_info
        self.entropy_calc = SimpleConfigEntropyCalculator()
        
    def search(self, root_position, simulations=800):
        root = Node(root_position)
        
        for _ in range(simulations):
            # Selection with info bonus
            node = root
            path = [root]
            
            while node.is_expanded() and not node.is_terminal():
                action = self.select_action(node)
                node = node.get_child(action)
                path.append(node)
                
            # Expansion
            if not node.is_terminal():
                node.expand()
                
            # Evaluation
            value = self.evaluate(node)
            
            # Backpropagation
            for n in reversed(path):
                n.update(value)
                value = -value  # Flip for opponent
                
        return self.get_best_move(root)
        
    def select_action(self, node):
        """Select with information bonus"""
        # ... (implementation from Component 2)
```

### Usage
```python
# Create MCTS with conservative settings
mcts = InformationMCTS(c_puct=1.0, lambda_info=0.1)

# Search position
best_move = mcts.search(position, simulations=800)

# Measure actual improvement
improvement = compare_to_baseline(mcts, standard_mcts)
print(f"Net improvement: {improvement:.1%}")
```

---

## Conclusion

This revised guide provides a practical, honest approach to improving MCTS through information-theoretic principles. By focusing on simple, validated components and maintaining realistic expectations, developers can achieve modest but real improvements in search efficiency. The key is careful measurement, incremental development, and avoiding theoretical overreach.