# Data Model: C++ MCTS Simulation Runner

**Spec ID**: 002-cpp-simulation-runner
**Date**: 2025-10-02  
**Status**: COMPREHENSIVE SPECIFICATION
**Purpose**: Complete type specifications, memory layouts, API contracts, and validation rules ensuring zero API mismatches

---

## CRITICAL CONTEXT FROM mcts_guide.md

This implementation **corrects fundamental architectural violations** in the current codebase:

1. **Python touches hot loops** (WRONG) → **C++ simulation runner** (CORRECT per mcts_guide.md:69-70)
2. **GIL held 100-200 times/sim** (WRONG) → **GIL released once, held only for inference callback** (CORRECT per mcts_guide.md:650-653)
3. **Synchronous GPU blocking** (WRONG) → **Async pattern with queues** (CORRECT per mcts_guide.md:213-293)
4. **Python dict for moves** (WRONG) → **C++ uint16_t* array** (CORRECT per mcts_guide.md:86-91)
5. **Thread pool per search** (WRONG) → **Shared tree, pre-allocated workers** (CORRECT per mcts_guide.md:47-62)

**This spec implements the ORIGINAL DESIGN**, not the broken current implementation.

---

## Core C++ Classes

### `mcts::SimulationRunner`

**Purpose**: Complete MCTS simulation cycles (selection → expansion → backup) entirely in C++ without returning to Python. Implements the architecture described in mcts_guide.md Section 3.

**Original Design Reference**: mcts_guide.md lines 213-293 (AsyncMCTSCoordinator pattern)

**Header**: `cpp_extensions/mcts/simulation_runner.hpp`

**Complete Class Definition**:
```cpp
namespace mcts {

class SimulationRunner {
public:
    // Constructor - takes references to shared thread-safe components
    explicit SimulationRunner(
        MCTSTree& tree,
        PUCTSelector& selector,
        BackupManager& backup_manager,
        VirtualLossManager& virtual_loss_manager
    );

    // Primary API: Run single simulation (called from Python with GIL released)
    // Returns: true if successful, false on error
    // Thread-safe: Yes (multiple runners can call simultaneously)
    // GIL: MUST be released by caller via py::call_guard<py::gil_scoped_release>
    bool run_simulation(
        const IGameState& root_state,
        NodeIndex root_index,
        InferenceCallback& inference_callback
    );

    // Batch API: Run multiple simulations efficiently
    // Returns: vector of success flags (length = num_simulations)
    // Use case: Called from AlphaZeroMCTS.search() to run full search
    std::vector<bool> run_simulations_batch(
        const IGameState& root_state,
        NodeIndex root_index,
        size_t num_simulations,
        InferenceCallback& inference_callback
    );

    // Configuration
    void set_max_path_length(size_t max_length);
    size_t get_max_path_length() const;

    // Statistics (thread-safe via atomics)
    struct RunnerStats {
        size_t total_simulations;       // Total attempts
        size_t successful_simulations;  // Completed successfully
        size_t terminal_nodes_hit;      // Reached terminal states
        size_t expansion_count;         // Nodes expanded
        float avg_path_length;          // Average selection path length
        float avg_simulation_time_us;   // Microseconds per simulation
    };
    RunnerStats get_statistics() const;
    void reset_statistics();

private:
    // === Shared Components (thread-safe, not owned) ===
    MCTSTree& tree_;                      // Shared MCTS tree
    PUCTSelector& selector_;              // PUCT selection logic
    BackupManager& backup_manager_;       // Value backup logic
    VirtualLossManager& virtual_loss_manager_; // Virtual loss coordination

    // === Per-Runner Working Memory (NOT shared between threads) ===
    std::vector<NodeIndex> path_buffer_;  // Reused selection path buffer
    IGameState* working_state_;           // Cloned state for simulation
    size_t max_path_length_;              // Max tree depth (default: 256)

    // === Statistics (atomic for thread safety) ===
    mutable std::atomic<size_t> total_simulations_{0};
    mutable std::atomic<size_t> successful_simulations_{0};
    mutable std::atomic<size_t> terminal_nodes_hit_{0};
    mutable std::atomic<size_t> expansion_count_{0};
    mutable std::atomic<float> cumulative_path_length_{0.0f};
    mutable std::atomic<float> cumulative_simulation_time_us_{0.0f};

    // === Internal Phase Methods (correspond to mcts_guide.md:254-292) ===
    
    // Selection: Traverse tree using PUCT until leaf/terminal
    // Corresponds to mcts_guide.md:254-265
    // Returns: true if valid leaf found, false on error
    bool select_leaf(
        IGameState& state,              // Working state (modified in-place)
        NodeIndex current_index,        // Starting node (usually root)
        std::vector<NodeIndex>& path    // Output: nodes from root to leaf
    );

    // Expansion: Request inference and create children
    // Corresponds to mcts_guide.md:268-270
    // Returns: Leaf value (from neural network or terminal)
    float expand_node(
        IGameState& state,              // State at leaf node
        NodeIndex node_index,           // Leaf node to expand
        InferenceCallback& callback     // Inference callback (holds GIL briefly)
    );

    // Terminal handling: Get exact game outcome
    // Returns: Terminal value from current player's perspective
    float get_terminal_value(
        const IGameState& state,
        NodeIndex node_index
    );

    // Backup: Propagate value up path with sign flipping
    // Corresponds to mcts_guide.md:285-292 and 1193-1198
    // Returns: true if successful
    bool backup_value(
        const std::vector<NodeIndex>& path,  // Leaf-to-root path
        float leaf_value                     // Value to propagate
    );
};

} // namespace mcts
```

**Memory Layout**:
```
SimulationRunner instance: ~1200 bytes
├─ References (4 × 8 bytes) = 32 bytes
├─ path_buffer_ (cap 256 × 4 bytes + overhead) = ~1056 bytes
├─ working_state_ pointer = 8 bytes
├─ max_path_length_ = 8 bytes
├─ Atomics (6 × 8 bytes + padding) = 64 bytes
└─ Total ≈ 1168 bytes

Memory efficiency: ~1.2KB per thread (8 threads = 9.6KB total)
```

**Thread Safety Guarantees**:

| Component | Safety Mechanism | Notes |
|-----------|-----------------|-------|
| `tree_`, `selector_`, `backup_manager_`, `virtual_loss_manager_` | Reference to shared thread-safe components | All use atomic operations internally |
| `path_buffer_` | Per-runner instance | Each thread has own SimulationRunner → no sharing |
| `working_state_` | Cloned fresh each simulation | Never shared between simulations |
| Statistics | `std::atomic<T>` | Lock-free thread-safe updates |

**Performance Characteristics** (from mcts_guide.md performance model):

```
Single-threaded baseline (after C++ migration):
├─ Selection (C++): 100µs (vs 400µs Python)
├─ State cloning: 15µs (once vs 10-15× Python)
├─ Inference callback: 100µs (batched, unchanged)
├─ Expansion (C++): 20µs (vs 80µs Python)
├─ Backup (C++): 15µs (vs 60µs Python)
└─ Total: 250µs → 4,000 sims/sec

Multi-threaded (8 threads, 80% efficiency):
└─ 4,000 × 8 × 0.80 = 25,600 sims/sec

With improved GPU batching:
└─ Inference: 100µs → 20µs (concurrent requests)
└─ Total: 170µs → 5,882 sims/sec/thread
└─ 8 threads × 5,882 × 0.85 = 40,000 sims/sec ✓ TARGET MET
```

**API Contracts**:

| Method | Preconditions | Postconditions | Error Handling |
|--------|--------------|----------------|----------------|
| `run_simulation` | root_index valid, root_state.is_terminal()=false, callback non-null, GIL RELEASED | Tree updated atomically, virtual loss removed, path_buffer cleared | Returns false, logs error (never throws) |
| `run_simulations_batch` | Same as run_simulation, num_simulations > 0 | num_simulations attempts made | Returns vector with per-sim success flags |
| `get_statistics` | None | Atomic snapshot of counters | Never fails |
| `reset_statistics` | None | All counters = 0 | Never fails |

**Critical Implementation Notes** (preventing common bugs from mcts_guide.md Section 8):

1. **Value Sign Flipping** (mcts_guide.md:1193-1198):
   ```cpp
   // CORRECT: Flip sign at each level
   for (size_t i = 0; i < path.size(); ++i) {
       float value_for_node = (i % 2 == 0) ? leaf_value : -leaf_value;
       tree_.update_node(path[i], value_for_node);
   }
   ```

2. **Virtual Loss RAII** (mcts_guide.md:258-259):
   ```cpp
   // Apply immediately after selection
   VirtualLossGuard vl_guard(virtual_loss_manager_, path);
   // Automatically removed on scope exit (even if exception thrown)
   ```

3. **State Cloning** (mcts_guide.md:1166-1177):
   ```cpp
   // Clone ONCE per simulation, not per move
   working_state_ = root_state.clone();
   for (NodeIndex node : path) {
       uint16_t move = tree_.get_move(node);
       working_state_->apply_move_inplace(move);  // No additional cloning!
   }
   ```

4. **Terminal Check** (mcts_guide.md:1224-1234):
   ```cpp
   // ALWAYS check terminal before expansion
   while (!state.is_terminal() && tree_.is_expanded(current)) {
       current = select_child(current);
   }
   ```

---

## `mcts::InferenceCallback`

**Purpose**: Abstract interface for neural network inference from C++. Allows SimulationRunner to request evaluations without coupling to Python inference infrastructure.

**Original Design Reference**: mcts_guide.md:268-270 (async inference queue pattern)

**Header**: `cpp_extensions/mcts/simulation_runner.hpp`

**Interface Definition**:
```cpp
namespace mcts {

class InferenceCallback {
public:
    virtual ~InferenceCallback() = default;

    // Primary inference method
    // @param state: Game position to evaluate
    // @return pair of (policy vector, value scalar)
    // @throws: May throw on inference failure (caught by SimulationRunner)
    // @thread-safety: Must be thread-safe (called from multiple simulation threads)
    // @gil: Implementer MUST acquire GIL if calling Python
    virtual std::pair<std::vector<float>, float>
    request_inference(const IGameState& state) = 0;

    // Batch inference (optional optimization for future)
    // Default: Sequential calls to request_inference
    virtual std::vector<std::pair<std::vector<float>, float>>
    request_inference_batch(const std::vector<const IGameState*>& states) {
        std::vector<std::pair<std::vector<float>, float>> results;
        results.reserve(states.size());
        for (const auto* state : states) {
            results.push_back(request_inference(*state));
        }
        return results;
    }
};

} // namespace mcts
```

**Concrete Python Bridge Implementation**:
```cpp
// In cpp_extensions/mcts/python_bindings.cpp

class PyInferenceCallback : public mcts::InferenceCallback {
public:
    explicit PyInferenceCallback(py::object python_callable)
        : python_callable_(std::move(python_callable)) {
        if (python_callable_.is_none()) {
            throw std::runtime_error("PyInferenceCallback: callable is None");
        }
    }

    std::pair<std::vector<float>, float>
    request_inference(const IGameState& state) override {
        // CRITICAL: GIL acquired automatically by pybind11 when touching py::object
        // Corresponds to mcts_guide.md:650-653 (GIL scoped release pattern)
        
        // Call Python function (returns Future)
        // Python signature: def inference_fn(state: IGameState) -> Future[(policy, value)]
        py::object future = python_callable_(&state);
        
        // Block on Future.result(timeout=1.0)
        // IMPORTANT: Future.result() releases GIL internally while waiting (CPython)
        // This allows other threads to submit inference requests concurrently
        py::object result_tuple;
        try {
            result_tuple = future.attr("result")(1.0);  // 1 second timeout
        } catch (const py::error_already_set& e) {
            if (e.matches(PyExc_TimeoutError)) {
                throw std::runtime_error("Inference timeout after 1.0s");
            }
            throw;
        }

        // Extract (policy, value) tuple
        if (!py::isinstance<py::tuple>(result_tuple)) {
            throw std::runtime_error("Inference must return (policy, value) tuple");
        }
        
        py::tuple result = result_tuple.cast<py::tuple>();
        if (result.size() != 2) {
            throw std::runtime_error(
                "Inference tuple must have 2 elements, got " + 
                std::to_string(result.size())
            );
        }

        // Extract policy as numpy array → std::vector<float>
        py::array_t<float> policy_array = result[0].cast<py::array_t<float>>();
        auto policy_buf = policy_array.request();
        
        if (policy_buf.ndim != 1) {
            throw std::runtime_error(
                "Policy must be 1D array, got " + 
                std::to_string(policy_buf.ndim) + "D"
            );
        }

        std::vector<float> policy(
            static_cast<float*>(policy_buf.ptr),
            static_cast<float*>(policy_buf.ptr) + policy_buf.size
        );

        // Extract value as float scalar
        float value = result[1].cast<float>();

        return {std::move(policy), value};
    }

private:
    py::object python_callable_;
};
```

**API Contracts**:

| Requirement | Specification | Validation |
|-------------|---------------|------------|
| **Return Type** | `pair<vector<float>, float>` where policy is probabilities, value is win probability | policy.size() == state.action_space_size() |
| **Policy Constraints** | Sum ≈ 1.0 (±0.01), all values ≥ 0 | assert(abs(sum(policy) - 1.0) < 0.01); assert(all(p >= 0)) |
| **Value Range** | value ∈ [-1, 1] | assert(value >= -1.0 && value <= 1.0) |
| **Timeout** | Complete within 1.0 second or throw | Configurable, prevents deadlock |
| **Thread Safety** | May be called concurrently from multiple threads | Python GIL protects callable, queues are thread-safe |
| **Error Handling** | Throw std::runtime_error on failure | SimulationRunner catches and returns false |

**Performance Breakdown** (from profiling analysis):

```
PyInferenceCallback::request_inference() timeline:
├─ 0µs: Enter (GIL already held from pybind11)
├─ +10µs: Call python_callable_(state) → submits to queue
├─ +15µs: Call future.result(1.0) → BLOCKS but releases GIL
│         (Other threads can now submit requests concurrently)
│         GPU worker collects 32-64 requests into batch
├─ +1000-3000µs: GPU inference (batched)
├─ +3010µs: future.result() returns → GIL reacquired
├─ +3015µs: Extract policy array (numpy → std::vector)
├─ +3020µs: Extract value scalar
└─ +3025µs: Return

Total time: ~3ms (dominated by GPU batch processing)
GIL held: ~35µs (10µs submit + 25µs extraction)
GIL released: ~2990µs (waiting on GPU)
```

**Critical Design Insight** (from mcts_guide.md:650-662):

The key to this design is that `Future.result()` **releases the GIL while blocking**, allowing other threads to continue selecting and submitting inference requests. This enables the GPU worker to collect 32-64 requests into efficient batches.

**Without C++ runner** (current broken implementation):
- Python loop holds GIL → one request submitted at a time
- GPU worker sees requests arrive serially → batch size stuck at 1-2
- GPU utilization <5%

**With C++ runner** (this spec):
- 8 threads release GIL simultaneously
- All threads select and submit requests concurrently
- GPU worker batches 32-64 requests
- GPU utilization 80-92% ✓

---

## `mcts::MCTSTree` Extensions

**Purpose**: Add move storage to existing SoA tree structure. This eliminates the 400MB Python dict overhead and removes hash lookups from the hot path.

**Original Design Reference**: mcts_guide.md:76-106 (SoA memory layout)

**Modification to Existing Class**:
```cpp
// In cpp_extensions/mcts/tree.hpp

class MCTSTree {
public:
    // === EXISTING FIELDS (unchanged) ===
    alignas(64) float* visit_counts_;       // N (visit count)
    alignas(64) float* total_values_;       // W (total value)
    alignas(64) float* prior_probs_;        // P (NN prior)
    alignas(64) float* virtual_losses_;     // VL (temporary penalty)
    alignas(64) int32_t* parent_indices_;   // Parent node index
    alignas(64) int32_t* first_child_indices_; // First child index
    alignas(64) uint16_t* num_children_;    // Number of children
    alignas(64) uint8_t* flags_;            // expanded|terminal|player

    // === NEW FIELD: Move Storage ===
    alignas(64) uint16_t* moves_;           // Move that created each node
    
    // === NEW METHODS: Move Access ===
    
    // Get move that led to this node
    // Returns: Move index [0, action_space_size) or UINT16_MAX for root
    // Thread-safe: Read-only after expansion
    uint16_t get_move(NodeIndex index) const {
        assert(is_valid_index(index));
        return moves_[index];
    }

    // Set move for node (called during expansion)
    // Thread-safe: Only called by thread that allocated the node
    void set_move(NodeIndex index, uint16_t move) {
        assert(is_valid_index(index));
        assert(move < 65535);  // UINT16_MAX reserved for root
        moves_[index] = move;
    }

    // === MODIFIED: Constructor ===
    explicit MCTSTree(size_t max_nodes = 50'000'000) 
        : max_nodes_(max_nodes), node_count_(0), next_free_index_(0) {
        
        // Allocate all SoA arrays (including new moves_ array)
        visit_counts_ = allocate_aligned<float>(max_nodes);
        total_values_ = allocate_aligned<float>(max_nodes);
        prior_probs_ = allocate_aligned<float>(max_nodes);
        virtual_losses_ = allocate_aligned<float>(max_nodes);
        parent_indices_ = allocate_aligned<int32_t>(max_nodes);
        first_child_indices_ = allocate_aligned<int32_t>(max_nodes);
        num_children_ = allocate_aligned<uint16_t>(max_nodes);
        flags_ = allocate_aligned<uint8_t>(max_nodes);
        moves_ = allocate_aligned<uint16_t>(max_nodes);  // NEW

        // Initialize all arrays to zero
        std::memset(visit_counts_, 0, max_nodes * sizeof(float));
        std::memset(total_values_, 0, max_nodes * sizeof(float));
        std::memset(prior_probs_, 0, max_nodes * sizeof(float));
        std::memset(virtual_losses_, 0, max_nodes * sizeof(float));
        std::memset(parent_indices_, 0xFF, max_nodes * sizeof(int32_t)); // -1
        std::memset(first_child_indices_, 0xFF, max_nodes * sizeof(int32_t)); // -1
        std::memset(num_children_, 0, max_nodes * sizeof(uint16_t));
        std::memset(flags_, 0, max_nodes * sizeof(uint8_t));
        std::memset(moves_, 0xFF, max_nodes * sizeof(uint16_t)); // UINT16_MAX
    }

    // === MODIFIED: Expansion ===
    void expand_node(
        NodeIndex parent_index,
        const std::vector<uint16_t>& legal_moves,
        const std::vector<float>& prior_probs
    ) {
        assert(legal_moves.size() == prior_probs.size());
        assert(legal_moves.size() <= 362);  // Max for Go 19×19

        // Allocate contiguous children
        NodeIndex first_child = allocate_nodes(legal_moves.size());
        if (first_child == NULL_NODE_INDEX) {
            throw std::runtime_error("Tree capacity exhausted");
        }

        // Initialize children
        for (size_t i = 0; i < legal_moves.size(); ++i) {
            NodeIndex child = first_child + i;

            // Set move (NEW)
            moves_[child] = legal_moves[i];

            // Set other fields (existing)
            prior_probs_[child] = prior_probs[i];
            parent_indices_[child] = parent_index;
            visit_counts_[child] = 0.0f;
            total_values_[child] = 0.0f;
            virtual_losses_[child] = 0.0f;
            first_child_indices_[child] = NULL_NODE_INDEX;
            num_children_[child] = 0;
            flags_[child] = 0;  // Not expanded, not terminal
        }

        // Update parent
        first_child_indices_[parent_index] = first_child;
        num_children_[parent_index] = static_cast<uint16_t>(legal_moves.size());
        flags_[parent_index] |= FLAG_EXPANDED;
    }

    // === MODIFIED: Clear ===
    void clear() {
        // Optimization: Only clear used portion for small trees
        // Corresponds to mcts_guide.md Section 7.2 performance tips
        const size_t clear_threshold = 500000;  // 500k nodes
        size_t nodes_to_clear = (next_free_index_ < clear_threshold)
                                ? next_free_index_
                                : max_nodes_;

        // Clear all arrays (including moves_)
        std::memset(visit_counts_, 0, nodes_to_clear * sizeof(float));
        std::memset(total_values_, 0, nodes_to_clear * sizeof(float));
        std::memset(virtual_losses_, 0, nodes_to_clear * sizeof(float));
        std::memset(moves_, 0xFF, nodes_to_clear * sizeof(uint16_t)); // UINT16_MAX

        // ... clear other arrays ...

        node_count_ = 0;
        next_free_index_ = 0;
        root_index_ = NULL_NODE_INDEX;
    }

private:
    template<typename T>
    T* allocate_aligned(size_t count) {
        void* ptr = nullptr;
        if (posix_memalign(&ptr, 64, count * sizeof(T)) != 0) {
            throw std::bad_alloc();
        }
        return static_cast<T*>(ptr);
    }
};
```

**Memory Impact Analysis**:

| Metric | Before (27 bytes/node) | After (29 bytes/node) | Delta |
|--------|------------------------|----------------------|-------|
| Single node | 27 bytes | 29 bytes | +2 bytes (+7.4%) |
| 1M nodes | 27 MB | 29 MB | +2 MB |
| 10M nodes | 270 MB | 290 MB | +20 MB |
| 50M nodes (max) | 1.35 GB | 1.45 GB | +100 MB |

**Comparison to Python Dict**:

```
Python dict overhead (10M entries):
├─ Hash table: ~400 MB (load factor 0.66)
├─ Entry overhead: 40 bytes × 10M = 400 MB
├─ PyObject overhead: ~200 MB
└─ Total: ~1000 MB

C++ uint16_t array (10M entries):
├─ Array: 2 bytes × 10M = 20 MB
└─ Total: 20 MB

Savings: 980 MB (98% reduction) ✓
```

**Performance Impact**:

```
Move retrieval (per simulation with 10 moves):

Python dict lookup:
├─ Hash computation: 10 × 50ns = 500ns
├─ Cache miss: 10 × 100ns = 1000ns
├─ PyObject deref: 10 × 20ns = 200ns
└─ Total: 1700ns

C++ array access:
├─ Index lookup: 10 × 5ns = 50ns
├─ Cache hit (prefetched): 10 × 0ns = 0ns
└─ Total: 50ns

Speedup: 34× faster ✓
```

**Validation Rules**:
```cpp
// Compile-time checks
static_assert(sizeof(uint16_t) == 2, "uint16_t must be 2 bytes");
static_assert(alignof(uint16_t) <= 64, "uint16_t compatible with 64-byte alignment");

// Runtime invariants
bool MCTSTree::validate_tree() const {
    for (NodeIndex i = 0; i < node_count_; ++i) {
        uint16_t move = moves_[i];
        
        // Root node special case
        if (i == root_index_) {
            assert(move == UINT16_MAX);
            assert(parent_indices_[i] == NULL_NODE_INDEX);
        }
        
        // Non-root nodes
        else {
            assert(move < max_action_space_);  // Valid action
            
            // Move must be legal in parent state (expensive check, debug only)
            #ifdef DEBUG_VALIDATION
            NodeIndex parent = parent_indices_[i];
            if (parent != NULL_NODE_INDEX) {
                IGameState* parent_state = reconstruct_state_at_node(parent);
                assert(parent_state->is_legal_move(move));
                delete parent_state;
            }
            #endif
        }
    }
    return true;
}
```

---

## Python Integration Layer

### `CppInferenceBridge`

**Purpose**: Adapts Python inference infrastructure to C++ InferenceCallback interface. Handles GIL correctly and integrates with async GPU batching.

**Original Design Reference**: mcts_guide.md:213-293 (async coordinator pattern)

**Location**: `src/core/mcts.py` (new class, ~150 lines)

**Complete Implementation**:
```python
import threading
import time
from typing import Callable, Dict, Any
from concurrent.futures import Future
from games.interface import IGameState

class CppInferenceBridge:
    """
    Bridge between C++ SimulationRunner and Python inference infrastructure.
    
    This class provides a callable interface that C++ can invoke to request
    neural network inference. It handles:
    - GIL management (acquired only during queue submission)
    - Future-based async results
    - Timeout handling
    - Statistics collection
    
    Thread Safety: All methods are thread-safe.
    Performance: Minimal GIL hold time (~10µs per request).
    """
    
    def __init__(self, inference_fn: Callable[[IGameState], Future]):
        """
        Initialize bridge with inference function.
        
        Args:
            inference_fn: Function that takes IGameState and returns 
                         Future[(policy: np.ndarray, value: float)]
                         
        The inference_fn is typically SearchCoordinator.request_inference()
        which submits to GPUInferenceWorker queue.
        """
        self.inference_fn = inference_fn
        
        # Statistics (thread-safe with lock)
        self._lock = threading.Lock()
        self._stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'timeout_errors': 0,
            'inference_errors': 0,
            'total_latency_ms': 0.0,
            'max_latency_ms': 0.0,
            'min_latency_ms': float('inf')
        }
    
    def __call__(self, game_state: IGameState) -> Future:
        """
        Request inference for game state (called from C++ with GIL).
        
        This method is called by PyInferenceCallback::request_inference().
        The calling sequence is:
        
        1. C++ SimulationRunner calls PyInferenceCallback::request_inference()
        2. pybind11 acquires GIL automatically
        3. PyInferenceCallback calls this __call__ method
        4. We submit to inference queue (10µs with GIL)
        5. Return Future immediately
        6. C++ calls Future.result() which RELEASES GIL while waiting
        
        Args:
            game_state: Position to evaluate
            
        Returns:
            Future that will contain (policy: np.ndarray, value: float) tuple
            
        Raises:
            RuntimeError: If inference system is unavailable
            
        Performance:
            - GIL hold time: ~10µs (queue submission only)
            - Does NOT block on result
        """
        start_time = time.perf_counter()
        
        try:
            # Submit to inference queue (non-blocking, ~10µs with GIL)
            future = self.inference_fn(game_state)
            
            # Update statistics
            with self._lock:
                self._stats['total_requests'] += 1
            
            # Return Future immediately (C++ will block on Future.result())
            return future
            
        except Exception as e:
            # Inference system failure (queue full, worker dead, etc.)
            with self._lock:
                self._stats['inference_errors'] += 1
            raise RuntimeError(f"Inference request failed: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get inference bridge statistics (thread-safe).
        
        Returns:
            Dictionary with:
            - total_requests: Total inference requests submitted
            - successful_requests: Requests that completed successfully
            - timeout_errors: Requests that exceeded timeout
            - inference_errors: System failures (queue full, etc.)
            - avg_latency_ms: Average end-to-end latency
            - max_latency_ms: Worst-case latency
            - min_latency_ms: Best-case latency
        """
        with self._lock:
            stats = dict(self._stats)
            
            # Compute average latency
            if stats['successful_requests'] > 0:
                stats['avg_latency_ms'] = (
                    stats['total_latency_ms'] / stats['successful_requests']
                )
            else:
                stats['avg_latency_ms'] = 0.0
            
            return stats
    
    def reset_statistics(self) -> None:
        """Reset all statistics counters (thread-safe)."""
        with self._lock:
            self._stats = {
                'total_requests': 0,
                'successful_requests': 0,
                'timeout_errors': 0,
                'inference_errors': 0,
                'total_latency_ms': 0.0,
                'max_latency_ms': 0.0,
                'min_latency_ms': float('inf')
            }
```

Continuing in next message due to length...


### `AlphaZeroMCTS` Integration

**Purpose**: Refactor existing AlphaZeroMCTS to use C++ simulation runner while maintaining backward compatibility.

**Location**: `src/core/mcts.py` (modifications to existing class)

**Complete Integration Pattern**:
```python
class AlphaZeroMCTS:
    """
    MCTS engine with optional C++ simulation runner.
    
    This class can operate in two modes:
    1. Legacy Python mode (use_cpp_runner=False): Original _run_simulation loop
    2. C++ runner mode (use_cpp_runner=True): Batch simulations in C++
    
    The C++ mode implements the architecture from mcts_guide.md Section 3.
    """
    
    def __init__(
        self,
        inference_fn: Callable[[IGameState], Future],
        num_threads: int = 8,
        use_cpp_runner: bool = True,
        cpuct: float = 1.25,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25
    ):
        """
        Initialize MCTS engine.
        
        Args:
            inference_fn: Async inference function (returns Future)
            num_threads: Number of parallel simulation threads (ignored in legacy mode)
            use_cpp_runner: Use C++ simulation runner (recommended)
            cpuct: PUCT exploration constant
            dirichlet_alpha: Dirichlet noise alpha parameter
            dirichlet_epsilon: Dirichlet noise mixing weight
        """
        self.inference_fn = inference_fn
        self.num_threads = num_threads
        self.use_cpp_runner = use_cpp_runner
        self.cpuct = cpuct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        
        # Create C++ infrastructure (shared across searches)
        self.tree = mcts_py.MCTSTree(max_nodes=50_000_000)
        self.selector = mcts_py.create_puct_selector(
            mcts_py.PUCTConfig(cpuct=cpuct)
        )
        self.backup_manager = mcts_py.create_backup_manager(self.tree)
        self.virtual_loss_manager = mcts_py.create_test_virtual_loss_manager(
            self.tree,
            mcts_py.VirtualLossConfig(magnitude=1.0)
        )
        
        if use_cpp_runner:
            # Create C++ simulation runner (one per thread)
            # Each thread gets its own runner to avoid path_buffer sharing
            self.runners = [
                mcts_py.SimulationRunner(
                    self.tree,
                    self.selector,
                    self.backup_manager,
                    self.virtual_loss_manager
                )
                for _ in range(num_threads)
            ]
            
            # Create inference bridge
            self.inference_bridge = CppInferenceBridge(inference_fn)
            
            # Create Python callback wrapper for C++
            self.cpp_callback = mcts_py.PyInferenceCallback(self.inference_bridge)
        else:
            # Legacy mode: Python simulation loop
            self.runners = None
            self.inference_bridge = None
            self.cpp_callback = None
        
        # State tracking
        self.root_state = None
        self.root_index = mcts_py.NULL_NODE_INDEX
        self._move_mapping = {}  # Only used in legacy mode
        
        # Statistics
        self._simulations_completed = 0
        self._total_search_time = 0.0
        self._lock = threading.Lock()
    
    def search(
        self,
        root_state: IGameState,
        simulations: int,
        add_noise: bool = False
    ) -> Dict[int, float]:
        """
        Run MCTS search from root state.
        
        Args:
            root_state: Game position to search from
            simulations: Number of simulations to run
            add_noise: Add Dirichlet noise to root (for exploration)
            
        Returns:
            Dictionary mapping moves to visit counts
            
        Performance:
            - C++ mode: 30,000-40,000 sims/sec (8 threads)
            - Legacy mode: ~1,000 sims/sec (baseline)
        """
        start_time = time.perf_counter()
        
        # Reset tree and expand root in Python (both modes)
        self.tree.clear()
        self.root_state = root_state
        self.root_index = self._expand_root(root_state, add_noise)
        
        if self.use_cpp_runner:
            # C++ mode: Batch simulations with thread pool
            self._search_cpp(root_state, simulations)
        else:
            # Legacy mode: Python simulation loop (backward compatible)
            self._search_python(simulations)
        
        # Record statistics
        elapsed = time.perf_counter() - start_time
        with self._lock:
            self._simulations_completed += simulations
            self._total_search_time += elapsed
        
        # Extract visit counts
        return self._get_visit_counts()
    
    def _search_cpp(self, root_state: IGameState, simulations: int) -> None:
        """
        Run simulations using C++ runner with thread pool.
        
        Implements the architecture from mcts_guide.md:254-292.
        Key insight: All threads release GIL simultaneously, allowing
        concurrent inference requests to batch efficiently.
        """
        # Divide simulations across threads
        sims_per_thread = simulations // self.num_threads
        remainder = simulations % self.num_threads
        
        def worker_thread(thread_id: int, num_sims: int):
            """Worker function for thread pool."""
            runner = self.runners[thread_id]
            
            # Run simulations (GIL released by pybind11)
            results = runner.run_simulations_batch(
                root_state,
                self.root_index,
                num_sims,
                self.cpp_callback
            )
            
            # Log failures (if any)
            failures = sum(1 for success in results if not success)
            if failures > 0:
                self.logger.warning(
                    f"Thread {thread_id}: {failures}/{num_sims} simulations failed"
                )
        
        # Launch threads
        threads = []
        for i in range(self.num_threads):
            num_sims = sims_per_thread + (1 if i < remainder else 0)
            if num_sims > 0:
                thread = threading.Thread(
                    target=worker_thread,
                    args=(i, num_sims)
                )
                thread.start()
                threads.append(thread)
        
        # Wait for completion
        for thread in threads:
            thread.join()
    
    def _search_python(self, simulations: int) -> None:
        """
        Legacy Python simulation loop (backward compatible).
        
        This is the ORIGINAL implementation that has performance issues.
        Kept for debugging and comparison purposes.
        """
        for _ in range(simulations):
            self._run_simulation()  # Existing implementation
    
    def _expand_root(
        self,
        root_state: IGameState,
        add_noise: bool
    ) -> NodeIndex:
        """
        Expand root node in Python (both C++ and legacy modes).
        
        This stays in Python because:
        1. Only happens once per search (not performance critical)
        2. Requires neural network evaluation
        3. Handles Dirichlet noise injection
        """
        # Check terminal
        if root_state.is_terminal():
            raise ValueError("Cannot search from terminal position")
        
        # Allocate root node
        root_index = self.tree.allocate_node()
        if root_index == mcts_py.NULL_NODE_INDEX:
            raise RuntimeError("Failed to allocate root node")
        
        # Request inference
        future = self.inference_fn(root_state)
        try:
            policy, value = future.result(timeout=5.0)
        except Exception as e:
            raise RuntimeError(f"Root inference failed: {e}")
        
        # Extract policy (handle batch dimension)
        if policy.ndim > 1:
            policy = policy[0]
        
        # Mask illegal moves
        legal_moves = root_state.get_legal_moves()
        legal_moves_set = set(legal_moves)
        
        for move in range(len(policy)):
            if move not in legal_moves_set:
                policy[move] = 0.0
        
        policy_sum = np.sum(policy)
        if policy_sum > 0:
            policy = policy / policy_sum
        else:
            # Fallback: uniform over legal moves
            policy = np.zeros(len(policy))
            for move in legal_moves:
                policy[move] = 1.0 / len(legal_moves)
        
        # Add Dirichlet noise if requested
        if add_noise:
            noise = np.random.dirichlet([self.dirichlet_alpha] * len(legal_moves))
            for i, move in enumerate(legal_moves):
                policy[move] = (
                    (1 - self.dirichlet_epsilon) * policy[move] +
                    self.dirichlet_epsilon * noise[i]
                )
            policy = policy / np.sum(policy)
        
        # Extract legal move priors
        legal_move_indices = []
        legal_move_priors = []
        for move in legal_moves:
            legal_move_indices.append(move)
            legal_move_priors.append(float(policy[move]))
        
        # Expand root node in C++ tree
        self.tree.expand_node(
            root_index,
            legal_move_indices,
            legal_move_priors
        )
        
        # Set root flags
        flags = mcts_py.NodeFlags()
        flags.set_expanded(True)
        flags.set_current_player(root_state.get_current_player())
        self.tree.set_flags(root_index, flags)
        
        return root_index
    
    def _get_visit_counts(self) -> Dict[int, float]:
        """
        Extract visit counts for root children.
        
        Returns:
            Dictionary mapping move indices to visit counts
        """
        if self.root_index == mcts_py.NULL_NODE_INDEX:
            return {}
        
        visit_counts = {}
        first_child = self.tree.get_first_child_index(self.root_index)
        num_children = self.tree.get_num_children(self.root_index)
        
        for i in range(num_children):
            child_index = first_child + i
            move = self.tree.get_move(child_index)  # Use C++ move storage
            visits = self.tree.get_visit_count(child_index)
            visit_counts[move] = visits
        
        return visit_counts
    
    def get_policy(
        self,
        root_state: IGameState,
        temperature: float = 1.0
    ) -> np.ndarray:
        """
        Convert visit counts to policy distribution.
        
        Args:
            root_state: Current game state
            temperature: Temperature for visit count exponentiation
                        (1.0 = proportional, 0.0 = argmax)
        
        Returns:
            Policy array of shape (action_space_size,)
        """
        visit_counts = self._get_visit_counts()
        policy = np.zeros(root_state.action_space_size, dtype=np.float32)
        
        if not visit_counts:
            return policy
        
        # Apply temperature
        if temperature > 0:
            for move, visits in visit_counts.items():
                policy[move] = visits ** (1.0 / temperature)
            policy_sum = np.sum(policy)
            if policy_sum > 0:
                policy = policy / policy_sum
        else:
            # Temperature = 0: Deterministic (pick most visited)
            best_move = max(visit_counts, key=visit_counts.get)
            policy[best_move] = 1.0
        
        return policy
    
    def get_value(self, root_state: IGameState) -> float:
        """
        Get position value estimate from C++ tree.
        
        Returns:
            Value estimate from current player's perspective [-1, 1]
        """
        if self.root_index == mcts_py.NULL_NODE_INDEX:
            # No search performed - use neural network evaluation
            future = self.inference_fn(root_state)
            try:
                _, value = future.result(timeout=1.0)
                return float(value)
            except Exception as e:
                self.logger.warning(f"Neural network evaluation failed: {e}")
                return 0.0
        
        # Get Q-value from C++ tree
        return self.backup_manager.get_q_value(self.root_index)
    
    def reset(self) -> None:
        """Reset search tree and internal state."""
        self.tree.clear()
        self.virtual_loss_manager.reset_all_virtual_loss()
        self.backup_manager.reset_statistics()
        self.root_state = None
        self.root_index = mcts_py.NULL_NODE_INDEX
        
        if not self.use_cpp_runner:
            self._move_mapping.clear()  # Legacy mode only
    
    @property
    def tree_size(self) -> int:
        """Get current number of nodes in C++ tree."""
        return self.tree.get_node_count()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive MCTS performance statistics."""
        stats = {
            'tree_size': self.tree.get_node_count(),
            'max_tree_size': self.tree.get_max_nodes(),
            'memory_usage_mb': self.tree.get_memory_usage() / (1024 * 1024),
            'bytes_per_node': self.tree.get_bytes_per_node(),
            'simulations_completed': self._simulations_completed,
            'total_search_time': self._total_search_time,
            'avg_simulations_per_second': (
                self._simulations_completed / self._total_search_time
                if self._total_search_time > 0 else 0
            ),
            'virtual_loss_stats': self.virtual_loss_manager.get_statistics(),
            'backup_stats': self.backup_manager.get_statistics(),
        }
        
        if self.use_cpp_runner:
            # Add C++ runner statistics
            runner_stats = [r.get_statistics() for r in self.runners]
            stats['runner_stats'] = runner_stats
            stats['inference_bridge_stats'] = self.inference_bridge.get_statistics()
        
        return stats
```

---

## Threading Model

**Architecture** (from mcts_guide.md:47-62):

```
                    SearchCoordinator
                    (bounded thread pool: 8 workers)
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
   Worker 1              Worker 2          Worker 8
        │                   │                   │
        ├─► AlphaZeroMCTS.search()              │
        │   └─► SimulationRunner (instance 1)   │
        │       └─► run_simulations_batch()     │
        │           └─► GIL RELEASED ✓          │
        │                                        │
   (All workers operate on SHARED tree with atomics)
                            │
                            ▼
                    Shared MCTSTree
                    (atomic operations)
                            │
                            ▼
            GPUInferenceWorker (single thread)
            (batches concurrent requests)
```

**Thread Safety by Component**:

| Component | Synchronization | Contention Risk | Notes |
|-----------|----------------|-----------------|-------|
| **MCTSTree** | std::atomic<float> for N,W,VL | Low (75-85% efficiency) | CAS loops for updates |
| **SimulationRunner** | Per-thread instances | None | No shared state |
| **VirtualLossManager** | Atomic operations | Low | Coordinated via tree atomics |
| **BackupManager** | Atomic operations | Low | Same atomics as tree |
| **PUCTSelector** | Read-only | None | Stateless algorithm |
| **InferenceCallback** | Python queue (thread-safe) | Minimal | Queue.put() ~10µs |
| **GPUInferenceWorker** | Queue + condition variable | None | Single consumer thread |

**GIL Management Timeline** (8 concurrent simulation threads):

```
Time →
Thread 1: [GIL═10µs═][Released════════════3ms════════════][GIL═25µs═]
Thread 2:      [GIL═10µs═][Released════════3ms════════════][GIL═25µs═]
Thread 3:           [GIL═10µs═][Released════3ms════════════][GIL═25µs═]
Thread 4:                [GIL═10µs═][Released════3ms════════][GIL═25µs═]
Thread 5:                     [GIL═10µs═][Released════3ms════][GIL═25µs═]
Thread 6:                          [GIL═10µs═][Released════3ms][GIL═25µs═]
Thread 7:                               [GIL═10µs═][Released═3ms][GIL═25µs═]
Thread 8:                                    [GIL═10µs═][Rel═3ms][GIL═25µs═]
                                                              │
                                                              └─► All 8 requests
                                                                  batched together
                                                                  
Total GIL hold time: 8 × (10µs + 25µs) = 280µs
Total wall time: ~3ms (dominated by GPU batch processing)
GIL efficiency: 280µs / 3000µs = 9.3% (90.7% parallel) ✓
```

**Comparison: Current (Broken) vs. Spec (Correct)**:

| Metric | Current Python Loop | This Spec (C++ Runner) | Improvement |
|--------|-------------------|----------------------|-------------|
| GIL cycles/sim | 100-200 | 2 (submit + extract) | 50-100× |
| GIL hold time/sim | ~800µs | ~35µs | 23× |
| Thread efficiency | 3% | 75-85% | 25-28× |
| Sims/sec (8 threads) | 246 | 35,000-40,000 | 142-163× |
| GPU batch size | 1-2 | 32-64 | 16-32× |
| GPU utilization | <5% | 80-92% | 16-18× |

---

## Testing Fixtures

### `DummyGameState` (C++)

**Purpose**: Lightweight deterministic game for unit testing without coupling to real games.

**Location**: `cpp_extensions/mcts/test_utils.hpp` (new file)

**Complete Implementation**:
```cpp
// test_utils.hpp
#pragma once

#include "tree.hpp"
#include "../games/interface.hpp"
#include <vector>
#include <memory>
#include <cstdint>

namespace mcts {
namespace testing {

/**
 * @brief Dummy game state for deterministic testing
 * 
 * Properties:
 * - Fixed branching factor (configurable, default 5)
 * - Deterministic outcomes (based on move count)
 * - Minimal state (40 bytes)
 * - Fast cloning (<1µs)
 * - No complex game logic
 */
class DummyGameState : public IGameState {
public:
    explicit DummyGameState(
        int branching_factor = 5,
        int max_depth = 10,
        int current_depth = 0,
        int current_player = 0,
        int move_count = 0
    )
        : branching_factor_(branching_factor)
        , max_depth_(max_depth)
        , current_depth_(current_depth)
        , current_player_(current_player)
        , move_count_(move_count)
    {
        // Pre-compute legal moves
        legal_moves_.resize(branching_factor_);
        for (int i = 0; i < branching_factor_; ++i) {
            legal_moves_[i] = i;
        }
    }
    
    // IGameState interface
    
    IGameState* clone() const override {
        return new DummyGameState(
            branching_factor_,
            max_depth_,
            current_depth_,
            current_player_,
            move_count_
        );
    }
    
    void apply_move_inplace(int move) override {
        if (move < 0 || move >= branching_factor_) {
            throw std::invalid_argument("Invalid move");
        }
        if (is_terminal()) {
            throw std::logic_error("Cannot apply move to terminal state");
        }
        
        move_count_++;
        current_depth_++;
        current_player_ = 1 - current_player_;  // Alternate players
    }
    
    std::vector<int> get_legal_moves() const override {
        if (is_terminal()) {
            return {};
        }
        return legal_moves_;
    }
    
    bool is_terminal() const override {
        return current_depth_ >= max_depth_;
    }
    
    float get_result() const override {
        if (!is_terminal()) {
            return 0.0f;  // Not terminal
        }
        
        // Deterministic result based on move count
        // Even move count → player 0 wins (+1)
        // Odd move count → player 1 wins (-1)
        return (move_count_ % 2 == 0) ? 1.0f : -1.0f;
    }
    
    int get_current_player() const override {
        return current_player_;
    }
    
    int action_space_size() const override {
        return branching_factor_;
    }
    
    void extract_features(float* output) const override {
        // Minimal feature extraction (2 planes)
        std::memset(output, 0, 2 * sizeof(float));
        output[current_player_] = 1.0f;
    }
    
    int get_num_planes() const override {
        return 2;  // One plane per player
    }
    
    // Test utilities
    
    int get_depth() const { return current_depth_; }
    int get_move_count() const { return move_count_; }
    int get_branching_factor() const { return branching_factor_; }

private:
    int branching_factor_;
    int max_depth_;
    int current_depth_;
    int current_player_;
    int move_count_;
    std::vector<int> legal_moves_;
};

/**
 * @brief Mock inference callback for deterministic testing
 * 
 * Returns:
 * - Uniform policy over legal moves
 * - Configurable fixed value
 * - Tracks call count for verification
 */
class MockInferenceCallback : public InferenceCallback {
public:
    explicit MockInferenceCallback(float fixed_value = 0.5f)
        : fixed_value_(fixed_value)
        , call_count_(0)
    {}
    
    std::pair<std::vector<float>, float>
    request_inference(const IGameState& state) override {
        call_count_.fetch_add(1, std::memory_order_relaxed);
        
        // Uniform policy over legal moves
        auto legal_moves = state.get_legal_moves();
        int action_space = state.action_space_size();
        
        std::vector<float> policy(action_space, 0.0f);
        if (!legal_moves.empty()) {
            float prob = 1.0f / legal_moves.size();
            for (int move : legal_moves) {
                if (move < action_space) {
                    policy[move] = prob;
                }
            }
        }
        
        return {policy, fixed_value_};
    }
    
    // Test utilities
    size_t get_call_count() const {
        return call_count_.load(std::memory_order_relaxed);
    }
    
    void reset_call_count() {
        call_count_.store(0, std::memory_order_relaxed);
    }

private:
    float fixed_value_;
    std::atomic<size_t> call_count_;
};

/**
 * @brief Create a simple test tree with known structure
 * 
 * Creates a tree with:
 * - Root with N children
 * - Each child with M grandchildren
 * - Deterministic visit counts and values
 * 
 * Useful for testing selection, backup, and tree traversal.
 */
std::shared_ptr<MCTSTree> create_test_tree(
    int root_children = 5,
    int grandchildren = 3
) {
    auto tree = std::make_shared<MCTSTree>(10000);
    
    // Create root
    NodeIndex root = tree->add_root_node(1.0f, 0);
    
    // Add children to root
    std::vector<uint16_t> child_moves;
    std::vector<float> child_priors;
    for (int i = 0; i < root_children; ++i) {
        child_moves.push_back(i);
        child_priors.push_back(1.0f / root_children);
    }
    tree->expand_node(root, child_moves, child_priors);
    
    // Add grandchildren
    NodeIndex first_child = tree->get_first_child_index(root);
    for (int i = 0; i < root_children; ++i) {
        NodeIndex child = first_child + i;
        
        std::vector<uint16_t> gc_moves;
        std::vector<float> gc_priors;
        for (int j = 0; j < grandchildren; ++j) {
            gc_moves.push_back(j);
            gc_priors.push_back(1.0f / grandchildren);
        }
        tree->expand_node(child, gc_moves, gc_priors);
        
        // Set some visits for testing
        tree->set_visit_count(child, 10.0f + i);
        tree->set_total_value(child, 5.0f + i * 0.5f);
    }
    
    return tree;
}

} // namespace testing
} // namespace mcts
```

---

## Memory Layout Reference

### Complete Node Structure (29 bytes)

```
Byte Offset │ Field              │ Type      │ Size │ Alignment │ Purpose
────────────┼────────────────────┼───────────┼──────┼───────────┼──────────────────
0           │ visit_count        │ float32   │ 4    │ 64-byte   │ N: Visit count
4           │ total_value        │ float32   │ 4    │ array     │ W: Total value
8           │ prior_prob         │ float32   │ 4    │ aligned   │ P: NN prior
12          │ virtual_loss       │ float32   │ 4    │           │ VL: Temp penalty
16          │ parent_index       │ int32     │ 4    │           │ Parent node (-1 for root)
20          │ first_child_index  │ int32     │ 4    │           │ First child (-1 if unexpanded)
24          │ num_children       │ uint16    │ 2    │           │ Child count
26          │ move               │ uint16    │ 2    │ **NEW**   │ Move to reach node
28          │ flags              │ uint8     │ 1    │           │ expanded|terminal|player
────────────┴────────────────────┴───────────┴──────┴───────────┴──────────────────
Total: 29 bytes per node (target <64 bytes ✓✓✓)
```

### Memory Budget Breakdown (10M nodes)

| Array | Element Type | Count | Total Size | Percentage |
|-------|-------------|-------|------------|------------|
| visit_counts_ | float (4 bytes) | 10M | 40 MB | 13.8% |
| total_values_ | float (4 bytes) | 10M | 40 MB | 13.8% |
| prior_probs_ | float (4 bytes) | 10M | 40 MB | 13.8% |
| virtual_losses_ | float (4 bytes) | 10M | 40 MB | 13.8% |
| parent_indices_ | int32 (4 bytes) | 10M | 40 MB | 13.8% |
| first_child_indices_ | int32 (4 bytes) | 10M | 40 MB | 13.8% |
| num_children_ | uint16 (2 bytes) | 10M | 20 MB | 6.9% |
| **moves_** | **uint16 (2 bytes)** | **10M** | **20 MB** | **6.9%** |
| flags_ | uint8 (1 byte) | 10M | 10 MB | 3.4% |
| **Total** | | | **290 MB** | **100%** |

**Cache Line Utilization** (Ryzen 5900X with 64-byte cache lines):

```
Selection Phase (hot path):
├─ visit_count, total_value, prior_prob, virtual_loss (16 bytes)
├─ Fits in single cache line: ✓
├─ SIMD operations: 8 floats per AVX2 instruction
└─ Expected L1 hit rate: >95%

Backup Phase:
├─ visit_count, total_value, virtual_loss (12 bytes)
├─ Fits in single cache line: ✓
└─ Atomic CAS: ~20ns on L1 cache

Move Retrieval:
├─ moves_[index] (2 bytes)
├─ Likely prefetched with parent_indices_
└─ Expected L1 hit rate: >90%
```

---

## Validation Rules

### Compile-Time Assertions

```cpp
// In cpp_extensions/mcts/tree.hpp

namespace mcts {

// Type size guarantees
static_assert(sizeof(float) == 4, "float must be 32-bit");
static_assert(sizeof(int32_t) == 4, "int32_t must be 32-bit");
static_assert(sizeof(uint16_t) == 2, "uint16_t must be 16-bit");
static_assert(sizeof(uint8_t) == 1, "uint8_t must be 8-bit");

// Alignment guarantees
static_assert(alignof(float) <= 64, "float compatible with 64-byte alignment");
static_assert(alignof(int32_t) <= 64, "int32_t compatible with 64-byte alignment");
static_assert(alignof(uint16_t) <= 64, "uint16_t compatible with 64-byte alignment");

// Node size target
static_assert(
    sizeof(float) * 4 +      // visit_count, total_value, prior_prob, virtual_loss
    sizeof(int32_t) * 2 +    // parent_index, first_child_index
    sizeof(uint16_t) * 2 +   // num_children, move
    sizeof(uint8_t) * 1      // flags
    <= 64,
    "Node data must fit in 64 bytes for cache efficiency"
);

} // namespace mcts
```

### Runtime Invariants

```cpp
bool MCTSTree::validate_tree() const {
    for (NodeIndex i = 0; i < node_count_; ++i) {
        // Visit count invariants
        float visit_count = visit_counts_[i];
        float total_value = total_values_[i];
        assert(visit_count >= 0.0f);
        assert(std::abs(total_value) <= visit_count + 1e-6);
        
        // Prior probability invariants
        float prior_prob = prior_probs_[i];
        assert(prior_prob >= 0.0f && prior_prob <= 1.0f);
        
        // Virtual loss invariants
        float virtual_loss = virtual_losses_[i];
        assert(virtual_loss >= 0.0f);
        
        // Parent-child relationships
        int32_t parent = parent_indices_[i];
        if (i == root_index_) {
            assert(parent == NULL_NODE_INDEX);
        } else {
            assert(parent >= 0 && parent < static_cast<int32_t>(i));
        }
        
        // Child indices
        int32_t first_child = first_child_indices_[i];
        uint16_t num_children = num_children_[i];
        if (first_child != NULL_NODE_INDEX) {
            assert(first_child > static_cast<int32_t>(i));  // DAG property
            assert(first_child + num_children <= static_cast<int32_t>(node_count_));
        }
        
        // Move validation
        uint16_t move = moves_[i];
        if (i == root_index_) {
            assert(move == UINT16_MAX);  // Root has no move
        } else {
            assert(move < max_action_space_);  // Valid action
            
            // Verify move is legal in parent state (expensive, debug only)
            #ifdef MCTS_EXPENSIVE_VALIDATION
            if (parent != NULL_NODE_INDEX) {
                IGameState* parent_state = reconstruct_state_at_node(parent);
                auto legal_moves = parent_state->get_legal_moves();
                bool found = std::find(
                    legal_moves.begin(),
                    legal_moves.end(),
                    move
                ) != legal_moves.end();
                assert(found);
                delete parent_state;
            }
            #endif
        }
        
        // Flags validation
        uint8_t flags = flags_[i];
        bool is_expanded = (flags & FLAG_EXPANDED) != 0;
        bool is_terminal = (flags & FLAG_TERMINAL) != 0;
        
        if (is_expanded) {
            // Expanded nodes must have children (unless terminal)
            if (!is_terminal) {
                assert(num_children > 0);
                assert(first_child != NULL_NODE_INDEX);
            }
        } else {
            // Unexpanded nodes have no children
            assert(num_children == 0);
            assert(first_child == NULL_NODE_INDEX);
        }
        
        if (is_terminal) {
            // Terminal nodes cannot have children
            assert(num_children == 0);
        }
    }
    
    return true;
}
```

### Python Binding Validation

```python
# In tests/unit/test_tree_validation.py

def test_tree_invariants():
    """Validate all tree invariants after operations."""
    tree = mcts_py.MCTSTree(max_nodes=10000)
    
    # Add root
    root = tree.add_root_node(1.0, 0)
    assert tree.validate_tree()
    
    # Expand root
    legal_moves = [0, 1, 2, 3, 4]
    priors = [0.2, 0.2, 0.2, 0.2, 0.2]
    tree.expand_node(root, legal_moves, priors)
    assert tree.validate_tree()
    
    # Update visits
    first_child = tree.get_first_child_index(root)
    for i in range(5):
        child = first_child + i
        tree.set_visit_count(child, 10.0)
        tree.set_total_value(child, 5.0)
    assert tree.validate_tree()
    
    # Verify move storage
    for i in range(5):
        child = first_child + i
        move = tree.get_move(child)
        assert move == legal_moves[i]
    
    # Clear and verify
    tree.clear()
    assert tree.get_node_count() == 0
    assert tree.validate_tree()
```

---

## Performance Validation

### Expected Performance Metrics

| Metric | Target | Measurement Method | Acceptance Criteria |
|--------|--------|-------------------|-------------------|
| **Simulations/sec** | 30,000-40,000 | Timed search with 800 sims | Mean ≥ 30k over 10 runs |
| **Thread Efficiency** | 75-85% | Speedup / num_threads | ≥ 0.75 for 8 threads |
| **GIL Contention** | <10% | Profile GIL wait time | <10% of CPU time |
| **Memory Usage** | <1GB | Tree size at 10M nodes | ≤ 1024 MB |
| **GPU Utilization** | 80-92% | nvidia-smi during search | Mean ≥ 80% |
| **Batch Size** | 32-64 | Log batch sizes | Mean ≥ 32 |
| **Memory Leaks** | 0 | 1-hour soak test | Growth < 10 MB/hour |

### Benchmark Code

```python
# tests/performance/test_simulation_runner_performance.py

import pytest
import time
import numpy as np
from src.core.mcts import AlphaZeroMCTS
from cpp_extensions.mcts import testing

@pytest.mark.benchmark
def test_throughput_exceeds_target():
    """Verify C++ runner achieves 30k+ sims/sec."""
    
    # Create lightweight test game
    root_state = testing.DummyGameState(branching_factor=5, max_depth=10)
    
    # Create MCTS with C++ runner
    def mock_inference(state):
        future = Future()
        policy = np.ones(5) / 5.0
        value = 0.5
        future.set_result((policy, value))
        return future
    
    mcts = AlphaZeroMCTS(
        inference_fn=mock_inference,
        num_threads=8,
        use_cpp_runner=True
    )
    
    # Run multiple trials
    trial_results = []
    for trial in range(10):
        start = time.perf_counter()
        mcts.search(root_state, simulations=800, add_noise=False)
        elapsed = time.perf_counter() - start
        
        sims_per_sec = 800 / elapsed
        trial_results.append(sims_per_sec)
        
        mcts.reset()
    
    # Statistical validation (mean and std)
    mean_throughput = np.mean(trial_results)
    std_throughput = np.std(trial_results)
    
    print(f"Throughput: {mean_throughput:.0f} ± {std_throughput:.0f} sims/sec")
    
    # Acceptance criteria
    assert mean_throughput >= 30000, \
        f"Throughput {mean_throughput:.0f} below target 30000 sims/sec"
    
    # Coefficient of variation should be low (<10%)
    cv = std_throughput / mean_throughput
    assert cv < 0.10, \
        f"High variance (CV={cv:.2%}), indicates unstable performance"
```

---

**Document Status**: COMPLETE (1600+ lines)
**Last Updated**: 2025-10-02
**Next Step**: Implement SimulationRunner following this specification exactly
**Reference**: mcts_guide.md for original design intent

