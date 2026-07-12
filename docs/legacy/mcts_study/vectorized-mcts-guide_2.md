# Practical Development Guide: Wave-Based Vectorized MCTS

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Core Concepts](#core-concepts)
3. [Architecture Overview](#architecture-overview)
4. [Implementation Strategy](#implementation-strategy)
5. [Data Structures](#data-structures)
6. [Wave Processing Algorithm](#wave-processing-algorithm)
7. [Performance Optimization](#performance-optimization)
8. [Testing and Validation](#testing-and-validation)
9. [Common Pitfalls](#common-pitfalls)

---

## 1. Executive Summary

This guide provides a practical approach to implementing vectorized Monte Carlo Tree Search (MCTS) based on Google DeepMind's MCTX framework. We target realistic performance improvements of 2-10x over well-optimized multi-threaded baselines while maintaining code clarity and playing strength.

### Key Principles

1. **Wave-Based Processing**: Process simulations in synchronized batches
2. **Simple Diversity**: Use lightweight randomization instead of complex mechanisms
3. **Dynamic Memory**: Adapt to actual tree shapes rather than fixed allocation
4. **Honest Benchmarking**: Compare against state-of-the-art implementations

### Target Performance

- **Hardware**: Modern CPU (8+ cores) + GPU (RTX 3060 or better)
- **Expected Speedup**: 2-10x over optimized baselines
- **Throughput**: 20,000-80,000 simulations/second (game-dependent)

---

## 2. Core Concepts

### 2.1 Wave Processing (MCTX-Style)

Traditional MCTS processes simulations sequentially:
```
for i in range(num_simulations):
    path = select()         # Uses updated tree
    value = evaluate()      # One at a time
    backup(path, value)     # Immediate update
```

Wave-based processing groups simulations:
```
for wave_start in range(0, num_simulations, wave_size):
    paths = select_batch(tree_snapshot, wave_size)    # All see same tree
    values = evaluate_batch(paths)                     # GPU efficiency
    backup_batch(paths, values)                        # Bulk update
```

### 2.2 Why This Works

1. **Relaxed Consistency**: Temporary inconsistency within a wave is acceptable
2. **Batch Efficiency**: Full GPU utilization for neural network evaluation
3. **Natural Diversity**: Different starting points and noise ensure exploration
4. **Empirical Validation**: MCTX proves this maintains playing strength

### 2.3 Realistic Expectations

| Baseline Type | Expected Speedup | Rationale |
|--------------|------------------|-----------|
| Single-threaded | 10-40x | Full parallelization |
| Multi-threaded (poor) | 5-20x | Better GPU utilization |
| Well-optimized | 2-10x | Improved batching/memory |
| State-of-art | 1.5-3x | Incremental improvements |

---

## 3. Architecture Overview

### 3.1 System Components

```
┌─────────────────────────────────────────────┐
│                Main Controller               │
├─────────────┬──────────────┬────────────────┤
│  Tree Pool  │   Scheduler  │  NN Evaluator  │
│  (Dynamic)  │   (Waves)    │   (Batched)    │
└─────────────┴──────────────┴────────────────┘
       ↓              ↓               ↓
 [CPU Memory]    [Work Queue]    [GPU Memory]
```

### 3.2 Processing Pipeline

```
Wave Selection → Position Extraction → NN Evaluation → Tree Update
     (CPU)            (CPU)              (GPU)           (CPU)
```

### 3.3 Memory Management Strategy

- **Dynamic Allocation**: Grow trees as needed
- **Memory Pools**: Reuse allocations between games
- **Split Storage**: Hot data (CPU) vs. evaluation data (GPU)

---

## 4. Implementation Strategy

### 4.1 Technology Stack

```yaml
Primary Framework: PyTorch
- Unified tensor operations
- Excellent GPU support
- JIT compilation available

Language: Python + C++ extensions
- Python for high-level logic
- C++ for critical sections only

Dependencies:
- numpy: CPU array operations
- torch: GPU operations
- numba: JIT compilation (optional)
```

### 4.2 Development Phases

#### Phase 1: Basic Wave Processing
```python
class BasicWaveMCTS:
    def __init__(self, wave_size=256):
        self.wave_size = wave_size
        self.tree = DynamicTree()
        
    def search(self, position, num_simulations):
        for wave in range(0, num_simulations, self.wave_size):
            # Simple wave processing
            paths = self.select_wave()
            values = self.evaluate_wave(paths)
            self.backup_wave(paths, values)
        return self.get_best_move()
```

#### Phase 2: Optimization
- Profile and identify bottlenecks
- Optimize memory access patterns
- Tune wave sizes

#### Phase 3: Advanced Features
- Dynamic wave sizing
- Adaptive tree growth
- Multi-GPU support

---

## 5. Data Structures

### 5.1 Dynamic Tree Structure

```python
class DynamicTree:
    """
    Efficient tree structure that grows as needed.
    Uses separate arrays for better cache locality.
    """
    
    def __init__(self, initial_size=10000):
        # Node data (Structure of Arrays)
        self.capacity = initial_size
        self.size = 0
        
        # Statistics
        self.visits = np.zeros(initial_size, dtype=np.int32)
        self.total_values = np.zeros(initial_size, dtype=np.float32)
        
        # Tree structure
        self.parent_ids = np.full(initial_size, -1, dtype=np.int32)
        self.child_starts = np.full(initial_size, -1, dtype=np.int32)
        self.num_children = np.zeros(initial_size, dtype=np.int16)
        
        # Edge data (separate array)
        self.edge_capacity = initial_size * 9  # Assuming max 9 actions
        self.edges = self._create_edge_array(self.edge_capacity)
        
    def add_node(self, parent_id=-1):
        """Add a new node, growing arrays if needed."""
        if self.size >= self.capacity:
            self._grow_arrays()
            
        node_id = self.size
        self.size += 1
        
        if parent_id >= 0:
            self._link_child(parent_id, node_id)
            
        return node_id
    
    def _grow_arrays(self):
        """Double array capacity when needed."""
        new_capacity = self.capacity * 2
        
        # Resize numpy arrays efficiently
        self.visits = np.resize(self.visits, new_capacity)
        self.total_values = np.resize(self.total_values, new_capacity)
        # ... resize other arrays
        
        self.capacity = new_capacity
```

### 5.2 Batch-Friendly Node Access

```python
class BatchNodeAccessor:
    """
    Provides efficient batch access to tree nodes.
    """
    
    def __init__(self, tree):
        self.tree = tree
        
    def get_ucb_scores_batch(self, node_ids, c_puct=1.0):
        """
        Compute UCB scores for multiple nodes at once.
        Returns: [batch_size, max_actions] array
        """
        batch_size = len(node_ids)
        max_actions = 9  # Game-specific
        
        # Pre-allocate output
        ucb_scores = np.full((batch_size, max_actions), -np.inf)
        
        # Vectorized computation
        visits = self.tree.visits[node_ids]
        values = self.tree.total_values[node_ids]
        
        # Compute Q-values
        q_values = np.divide(
            values, 
            visits, 
            out=np.zeros_like(values, dtype=np.float32),
            where=(visits > 0)
        )
        
        # Get child statistics (vectorized)
        for i, node_id in enumerate(node_ids):
            child_start = self.tree.child_starts[node_id]
            if child_start < 0:
                continue
                
            num_children = self.tree.num_children[node_id]
            child_visits = self.tree.visits[child_start:child_start + num_children]
            
            # UCB formula
            exploration = c_puct * np.sqrt(visits[i]) / (1 + child_visits)
            ucb_scores[i, :num_children] = q_values[i] + exploration
            
        return ucb_scores
```

---

## 6. Wave Processing Algorithm

### 6.1 Core Wave Selection

```python
class WaveSelector:
    """
    Implements MCTX-style wave selection.
    """
    
    def __init__(self, tree, wave_size=256):
        self.tree = tree
        self.wave_size = wave_size
        self.noise_epsilon = 0.01
        
    def select_wave(self, root_id=0):
        """
        Select a wave of paths through the tree.
        All selections see the same tree snapshot.
        """
        paths = np.full((self.wave_size, self.max_depth), -1, dtype=np.int32)
        path_lengths = np.zeros(self.wave_size, dtype=np.int32)
        
        # Start from root (with different random seeds)
        current_nodes = np.full(self.wave_size, root_id, dtype=np.int32)
        
        for depth in range(self.max_depth):
            # Get UCB scores for all nodes in parallel
            ucb_scores = self.tree.get_ucb_scores_batch(current_nodes)
            
            # Add small noise for diversity (not complex interference)
            noise = np.random.normal(0, self.noise_epsilon, ucb_scores.shape)
            ucb_scores += noise
            
            # Select best actions
            best_actions = np.argmax(ucb_scores, axis=1)
            
            # Get child nodes
            next_nodes = self.tree.get_children_batch(current_nodes, best_actions)
            
            # Update paths
            paths[:, depth] = current_nodes
            
            # Check for leaves
            is_leaf = next_nodes < 0
            path_lengths += ~is_leaf
            
            # Continue with non-leaf nodes
            current_nodes = np.where(is_leaf, current_nodes, next_nodes)
            
            # Early exit if all leaves
            if np.all(is_leaf):
                break
                
        return paths, path_lengths
```

### 6.2 Simple Diversity Mechanisms

```python
class DiversityManager:
    """
    Simple, efficient diversity mechanisms.
    No O(n²) interference matrices.
    """
    
    @staticmethod
    def add_dirichlet_noise(priors, alpha=0.3, epsilon=0.25):
        """AlphaZero-style Dirichlet noise."""
        noise = np.random.dirichlet([alpha] * len(priors))
        return (1 - epsilon) * priors + epsilon * noise
    
    @staticmethod
    def temperature_sampling(scores, temperature=1.0):
        """Softmax temperature for action selection."""
        if temperature == 0:
            return np.argmax(scores)
        
        exp_scores = np.exp((scores - np.max(scores)) / temperature)
        probs = exp_scores / exp_scores.sum()
        return np.random.choice(len(scores), p=probs)
    
    @staticmethod
    def progressive_widening(node_visits, cpw=1.5, alpha=0.5):
        """Limit branching based on visit count."""
        return int(np.ceil(cpw * (node_visits ** alpha)))
```

---

## 7. Performance Optimization

### 7.1 GPU Batch Evaluation

```python
class GPUEvaluator:
    """
    Efficient GPU evaluation with proper batching.
    """
    
    def __init__(self, model_path, device='cuda', optimal_batch_size=256):
        self.device = torch.device(device)
        self.model = torch.jit.load(model_path).to(self.device)
        self.model.eval()
        
        # Pre-allocate buffers
        self.position_buffer = torch.zeros(
            (optimal_batch_size, 3, 8, 8),  # Example shape
            device=self.device,
            dtype=torch.float32
        )
        
    def evaluate_batch(self, positions):
        """
        Evaluate positions with minimal overhead.
        """
        batch_size = len(positions)
        
        # Fill buffer (minimize allocations)
        self.position_buffer[:batch_size] = torch.from_numpy(positions)
        
        # Single forward pass
        with torch.no_grad():
            with torch.cuda.amp.autocast():  # Mixed precision
                values, policies = self.model(self.position_buffer[:batch_size])
        
        # Return as numpy for CPU processing
        return values.cpu().numpy(), policies.cpu().numpy()
```

### 7.2 Memory Access Optimization

```python
def optimize_memory_layout(tree):
    """
    Reorganize tree data for better cache locality.
    """
    # Group frequently accessed data
    hot_data = np.column_stack([
        tree.visits,
        tree.total_values,
        tree.num_children
    ])
    
    # Separate cold data
    cold_data = np.column_stack([
        tree.parent_ids,
        tree.child_starts
    ])
    
    return hot_data, cold_data
```

### 7.3 Profile-Guided Optimization

```python
class PerformanceMonitor:
    """
    Profile and identify bottlenecks.
    """
    
    def __init__(self):
        self.timings = defaultdict(list)
        
    @contextmanager
    def measure(self, name):
        start = time.perf_counter()
        yield
        self.timings[name].append(time.perf_counter() - start)
    
    def profile_wave(self, wave_mcts):
        """Profile one complete wave."""
        with self.measure('total'):
            with self.measure('selection'):
                paths = wave_mcts.select_wave()
            
            with self.measure('extraction'):
                positions = wave_mcts.extract_positions(paths)
            
            with self.measure('evaluation'):
                values = wave_mcts.evaluate_batch(positions)
            
            with self.measure('backup'):
                wave_mcts.backup_wave(paths, values)
        
        return self.generate_report()
```

---

## 8. Testing and Validation

### 8.1 Correctness Tests

```python
def test_wave_vs_sequential():
    """
    Verify wave processing produces similar results to sequential.
    """
    # Same position, same random seed
    position = create_test_position()
    np.random.seed(42)
    
    # Sequential MCTS
    seq_mcts = SequentialMCTS()
    seq_result = seq_mcts.search(position, 1000)
    
    # Wave-based MCTS
    np.random.seed(42)
    wave_mcts = WaveMCTS(wave_size=100)
    wave_result = wave_mcts.search(position, 1000)
    
    # Should produce very similar visit distributions
    assert visit_distribution_similarity(seq_result, wave_result) > 0.9
```

### 8.2 Performance Benchmarks

```python
def benchmark_implementations():
    """
    Fair comparison against strong baselines.
    """
    implementations = {
        'leela_zero': LeelaCPUBackend(),        # Strong baseline
        'katago': KataGoParallel(),             # State-of-art
        'wave_mcts': WaveMCTS(),                # Our implementation
    }
    
    results = {}
    test_positions = load_benchmark_positions()
    
    for name, impl in implementations.items():
        times = []
        for pos in test_positions:
            start = time.perf_counter()
            impl.search(pos, time_limit=1.0)  # Fixed time
            times.append(time.perf_counter() - start)
        
        results[name] = {
            'mean_time': np.mean(times),
            'simulations': impl.get_simulation_count()
        }
    
    return results
```

### 8.3 Quality Validation

```python
def validate_playing_strength():
    """
    Ensure maintained or improved playing strength.
    """
    # Play matches against baseline
    results = play_tournament(
        player1=WaveMCTS(),
        player2=OptimizedBaseline(),
        num_games=1000
    )
    
    win_rate = results['wins'] / results['total']
    
    # Should maintain strength (statistical test)
    assert statistical_significance_test(win_rate, 0.5) > 0.95
```

---

## 9. Common Pitfalls

### 9.1 Over-Engineering

**Pitfall**: Adding complex features without empirical justification.

**Solution**: Start simple, add complexity only when benchmarks show benefit.

### 9.2 Unfair Comparisons

**Pitfall**: Comparing against weak or outdated baselines.

**Solution**: Use current versions of strong open-source implementations.

### 9.3 Memory Management

**Pitfall**: Fixed allocation wastes memory or limits tree size.

**Solution**: Implement dynamic growth with memory pools.

### 9.4 Synchronization Overhead

**Pitfall**: Too many small waves create overhead.

**Solution**: Use larger waves (256-512) to amortize costs.

### 9.5 GPU Underutilization

**Pitfall**: Small batches don't saturate GPU.

**Solution**: Accumulate positions until optimal batch size reached.

---

## Implementation Checklist

- [ ] Basic wave selection working
- [ ] Dynamic tree growth implemented
- [ ] GPU evaluation integrated
- [ ] Memory pools for efficiency
- [ ] Profiling shows bottlenecks addressed
- [ ] Tests pass vs. sequential implementation
- [ ] Benchmarks show expected speedup
- [ ] Playing strength maintained
- [ ] Code is clean and documented

## Conclusion

This guide provides a practical path to implementing wave-based vectorized MCTS. By focusing on proven techniques from MCTX while maintaining realistic expectations, you can achieve meaningful performance improvements over traditional implementations. Remember: optimize based on profiling, benchmark fairly, and prioritize correctness over complexity.