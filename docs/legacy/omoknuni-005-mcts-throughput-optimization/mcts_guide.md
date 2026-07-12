# High-Performance AlphaZero Implementation Guide

**Target Hardware:** AMD Ryzen 9 5900X (12C/24T), 64 GB RAM, NVIDIA RTX 3060 Ti (8 GB VRAM)  
**Games:** Gomoku/Omok, Chess (960-capable), Go (9×9→19×19)  
**Languages:** Python 3.12 orchestration, PyTorch for NN, C++/pybind11 + Cython for hot MCTS loops

---

## 0) Critical Reality Check & Corrections

### What This Document Fixed from Common Misconceptions

* **Architecture Truth**: Traditional shared-tree MCTS with multi-threaded CPU + batched GPU inference beats wave-based approaches for tactical games. The "stale frontier problem" (500 trees exploring in lockstep without learning from each other) is fatal for competitive strength.
* **GPU Utilization Reality**: Single RTX 3060 Ti with Python orchestration achieves **80-92%** utilization, not the fantasy 95-98%. Anyone claiming higher is likely measuring incorrectly or using device-resident JAX/CUDA.
* **5900X is UMA, not NUMA**: It has two CCDs with separate L3 caches (32MB each), but it's a single socket. Cross-CCD latency is 80-90ns vs 20-30ns intra-CCD - significant but not NUMA-level.
* **NumPy's "30-40x speedup" myth**: Only true vs naive Python loops. Against properly optimized C++/Cython with `nogil` and atomics, NumPy's advantage disappears. Tree operations are memory-bound and branchy, not dense matrix ops.
* **Memory Reality**: With compact Structure-of-Arrays (SoA), nodes are ~32-64 bytes each. Multi-GB footprints come from storing Python objects or per-node features - both are mistakes.
* **Quality > Quantity**: 30k intelligent simulations beat 100k dumb ones. Chase search quality (good priors, calibrated values), not vanity metrics.

---

## 1) Executive Summary

### The Architecture That Actually Works

1. **Core Engine**: Shared-tree MCTS in C++/pybind11 or Cython with `nogil` blocks for selection/expansion/backup
2. **Concurrency**: 8-12 CPU threads with virtual loss + atomic operations for thread safety
3. **GPU Inference**: Central worker with micro-batching (≥32 positions OR 1-3ms timeout)
4. **Memory Layout**: Structure-of-Arrays (SoA) for cache efficiency, pre-allocated node pools
5. **Games**: Uniform C++ interface via pybind11, feature extraction stays in C++

### Realistic Performance Targets

| Metric | Realistic Range | Why It Matters |
|--------|----------------|----------------|
| Simulations/sec (with NN) | 30-40k | Position-dependent; includes GPU inference time |
| GPU Utilization | 80-92% | With proper batching and mixed precision |
| Average Batch Size | 32-64 | Dynamic timeout prevents starvation |
| Tree Memory | <1 GB | For 10-20M nodes with SoA layout |
| Self-play Games/hour | 200-300 | Depends on time controls and game |
| Training Iterations to Strong | 50-100 | Game-dependent; Gomoku fastest |

---

## 2) System Architecture

```
Python 3.12 (Orchestrator Only)
├── Self-play Coordinator
│   └── MCTS Engine (C++/Cython, GIL released)
│       ├── Selection (vectorized UCB within nodes)
│       ├── Expansion (minimal allocation from pool)
│       ├── Virtual Loss (applied during descent)
│       └── Backup (atomic operations, value sign flip)
├── GPU Inference Worker (PyTorch)
│   ├── Dynamic micro-batching (count OR timeout)
│   ├── Mixed precision (fp16 autocast)
│   └── Pinned memory buffers
└── Training Pipeline
    ├── Experience replay buffer (mmap/parquet)
    └── Model checkpointing
```

### Key Design Principles

1. **Python coordinates, C++ computes**: Python never touches hot loops
2. **One tree per search**: All threads share the same tree with atomics
3. **Asynchronous inference**: Threads don't wait for GPU results
4. **Cache everything sensible**: Position evaluations, legal move masks
5. **Profile relentlessly**: If you haven't measured it, it's wrong

---

## 3) MCTS Engine Implementation

### 3.1 Memory Layout (Structure of Arrays)

```cpp
// mcts_tree.hpp - Core data structures
struct MCTSTree {
    // Statistics (aligned for SIMD)
    alignas(64) float* visit_counts;     // N
    alignas(64) float* total_values;     // W
    alignas(64) float* prior_probs;      // P
    alignas(64) float* virtual_losses;   // VL
    
    // Tree structure
    int32_t* parent_indices;
    int32_t* first_child_indices;
    uint16_t* num_children;
    
    // Metadata
    uint8_t* node_flags;  // expanded, terminal, player_to_move
    
    // Capacity management
    size_t num_nodes;
    size_t max_nodes;
    
    MCTSTree(size_t max_nodes = 50'000'000) {
        // Pre-allocate everything upfront
        visit_counts = (float*)aligned_alloc(64, max_nodes * sizeof(float));
        // ... allocate all arrays
        memset(visit_counts, 0, max_nodes * sizeof(float));
    }
};
```

**Critical Implementation Details:**

1. **Alignment**: Use 64-byte alignment for SIMD operations
2. **Size**: Each node is ~32-64 bytes total across all arrays
3. **Pre-allocation**: Allocate pool once; reuse across searches
4. **No pointers between nodes**: Use indices for cache efficiency

### 3.2 Selection with Vectorization (Cython)

```cython
# mcts_core.pyx - Optimized selection
from libc.math cimport sqrt, log
from libc.string cimport memset
cimport numpy as np
import numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef int32_t select_child_optimized(MCTSTree* tree, int32_t node_idx, 
                                    float cpuct) nogil:
    """
    Single-pass vectorized child selection.
    Critical: This function is called millions of times.
    """
    cdef:
        int32_t first_child = tree.first_child_indices[node_idx]
        uint16_t n_children = tree.num_children[node_idx]
        float parent_n = tree.visit_counts[node_idx]
        float parent_sqrt = sqrt(max(1.0, parent_n))
        
        # Stack allocation for small action spaces
        float[362] puct_scores  # Max for Go 19x19
        float best_score = -1e9
        int32_t best_child = -1
        int32_t i, child_idx
        float n, w, p, vl, q, u
    
    # Single fused loop - critical for performance
    for i in range(n_children):
        child_idx = first_child + i
        
        n = tree.visit_counts[child_idx]
        w = tree.total_values[child_idx]
        p = tree.prior_probs[child_idx]
        vl = tree.virtual_losses[child_idx]
        
        # Q-value with virtual loss adjustment
        if n > 0:
            q = (w / n) - (vl / (1.0 + n))
        else:
            q = 0.0
        
        # PUCT formula
        u = cpuct * p * parent_sqrt / (1.0 + n)
        puct_scores[i] = q + u
        
        # Track best inline (avoids second pass)
        if puct_scores[i] > best_score:
            best_score = puct_scores[i]
            best_child = child_idx
    
    return best_child
```

### 3.3 Thread-Safe Operations with OpenMP

```cython
# Atomic operations using OpenMP (most portable)
from cython.parallel cimport prange, parallel
from openmp cimport omp_get_thread_num, omp_set_num_threads

cdef void apply_virtual_loss(MCTSTree* tree, int32_t* path, 
                             int path_length, float vl_value) nogil:
    """Apply virtual loss atomically along path"""
    cdef int i, node
    
    for i in range(path_length):
        node = path[i]
        # OpenMP atomic pragma - compiler handles the complexity
        with gil:  # Brief GIL for Python compatibility
            pass
        # In practice, use C++ std::atomic or __sync_fetch_and_add
        tree.virtual_losses[node] += vl_value  # Must be atomic!

cdef void backup_value(MCTSTree* tree, int32_t* path, 
                       int path_length, float value) nogil:
    """Backup value with proper sign flipping"""
    cdef int i, node
    cdef float backup_value = value
    
    # Walk backwards from leaf to root
    for i in range(path_length - 1, -1, -1):
        node = path[i]
        
        # Atomic updates (use compiler intrinsics in production)
        __sync_fetch_and_add(&tree.visit_counts[node], 1.0)
        __sync_fetch_and_add(&tree.total_values[node], backup_value)
        __sync_fetch_and_sub(&tree.virtual_losses[node], 1.0)
        
        # Critical: Flip value for opponent
        backup_value = -backup_value
```

### 3.4 Asynchronous Search Pattern

```python
# search_coordinator.py - Async pattern for junior developers
import asyncio
import queue
import threading
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class SearchRequest:
    leaf_node_id: int
    thread_id: int
    path: List[int]  # Nodes from root to leaf
    
class AsyncMCTSCoordinator:
    """
    Manages async communication between MCTS threads and GPU.
    This pattern prevents threads from blocking on GPU inference.
    """
    
    def __init__(self, tree, gpu_worker, num_threads=10):
        self.tree = tree
        self.gpu_worker = gpu_worker
        self.num_threads = num_threads
        
        # Thread-safe queues
        self.inference_queue = queue.Queue(maxsize=1000)
        self.result_queues = [queue.Queue() for _ in range(num_threads)]
        
        # Path storage for backup
        self.path_storage = [[] for _ in range(num_threads)]
    
    def search_thread_worker(self, thread_id: int, num_simulations: int):
        """
        What each MCTS thread does.
        Critical: Never waits for GPU results during selection.
        """
        for sim in range(num_simulations):
            # Phase 1: Select leaf and record path
            path = []
            node = 0  # root
            
            while not self.tree.is_terminal(node):
                # Apply virtual loss immediately
                self.tree.add_virtual_loss(node, 1.0)
                path.append(node)
                
                if not self.tree.is_expanded(node):
                    break
                    
                # Vectorized selection within node
                node = select_child_optimized(self.tree, node, cpuct=1.25)
            
            # Phase 2: Queue for expansion (non-blocking)
            if not self.tree.is_terminal(node):
                request = SearchRequest(node, thread_id, path)
                self.inference_queue.put(request, block=False)
            else:
                # Terminal node - backup immediately
                value = self.tree.get_terminal_value(node)
                self.tree.backup_value(path, value)
            
            # Phase 3: Check for completed inferences (non-blocking)
            try:
                while True:
                    result = self.result_queues[thread_id].get_nowait()
                    self.process_inference_result(result)
            except queue.Empty:
                pass  # No results ready, continue searching
    
    def process_inference_result(self, result):
        """Process GPU inference result and backup"""
        node_id, policy, value, path = result
        
        # Expand node with policy
        self.tree.expand_node(node_id, policy)
        
        # Backup value through path
        self.tree.backup_value(path, value)
```

---

## 4) GPU Inference Worker

### 4.1 Batched Inference with Dynamic Timeout

```python
# gpu_worker.py - Production-ready inference worker
import torch
import torch.nn.functional as F
import time
import queue
import threading
from typing import List, Tuple, Optional

class GPUInferenceWorker:
    """
    Central GPU inference with micro-batching.
    Key insight: Batch by count OR time, whichever comes first.
    """
    
    def __init__(self, model, device='cuda:0', 
                 batch_size=64, timeout_ms=2.0):
        # Model setup
        self.model = model.eval().to(device)
        self.device = device
        
        # Batching config (tune these!)
        self.batch_size = batch_size
        self.timeout_seconds = timeout_ms / 1000.0
        
        # Memory optimization for RTX 3060 Ti
        torch.cuda.set_per_process_memory_fraction(0.85)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_num_threads(1)  # Prevent CPU contention
        
        # Pre-allocate buffers (avoid allocation in hot loop)
        self.input_buffer = None
        self.policy_buffer = None
        self.value_buffer = None
        
        # Metrics
        self.total_inferences = 0
        self.total_batch_size = 0
        
    def warmup(self, input_shape):
        """
        Warmup GPU and allocate buffers.
        Critical for consistent latency.
        """
        dummy_input = torch.randn(
            (self.batch_size, *input_shape), 
            device=self.device, dtype=torch.float16
        )
        
        # Run a few iterations to warm up GPU
        for _ in range(10):
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    _ = self.model(dummy_input)
        
        # Allocate persistent buffers
        self.input_buffer = torch.zeros_like(dummy_input)
        
        print(f"GPU warmup complete. Buffer shape: {dummy_input.shape}")
    
    def inference_loop(self, input_queue: queue.Queue, 
                       output_queues: List[queue.Queue]):
        """
        Main inference loop. Runs in dedicated thread.
        """
        while True:
            batch_requests = []
            batch_features = []
            
            # Collect batch with timeout
            deadline = time.perf_counter() + self.timeout_seconds
            
            while len(batch_requests) < self.batch_size:
                timeout_remaining = deadline - time.perf_counter()
                
                if timeout_remaining <= 0:
                    break
                
                try:
                    request = input_queue.get(timeout=timeout_remaining)
                    batch_requests.append(request)
                    batch_features.append(request.features)
                except queue.Empty:
                    break
            
            if not batch_requests:
                continue  # No requests, keep waiting
            
            # Batch inference
            batch_size = len(batch_requests)
            policies, values = self.batch_inference(batch_features)
            
            # Distribute results back to threads
            for request, policy, value in zip(batch_requests, policies, values):
                result = (
                    request.leaf_node_id,
                    policy.cpu().numpy(),
                    float(value),
                    request.path
                )
                output_queues[request.thread_id].put(result)
            
            # Update metrics
            self.total_inferences += 1
            self.total_batch_size += batch_size
    
    def batch_inference(self, features: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Actual GPU inference with mixed precision.
        """
        batch_size = len(features)
        
        # Fill pre-allocated buffer (avoid allocation)
        for i, feat in enumerate(features):
            self.input_buffer[i] = feat
        
        # Slice to actual batch size
        batch_input = self.input_buffer[:batch_size]
        
        # Inference with mixed precision
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                policy_logits, value = self.model(batch_input)
        
        # Post-process (stay on GPU as long as possible)
        policy = F.softmax(policy_logits, dim=1)
        value = torch.tanh(value).squeeze(-1)
        
        return policy, value
    
    @property
    def average_batch_size(self):
        if self.total_inferences == 0:
            return 0
        return self.total_batch_size / self.total_inferences
```

### 4.2 ResNet with Squeeze-Excitation

```python
# model.py - State-of-the-art ResNet for board games
import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block.
    Adds <1% computation but improves accuracy significantly.
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResidualBlock(nn.Module):
    """Residual block with SE attention"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SEBlock(channels)
        
    def forward(self, x):
        identity = x
        
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = self.se(out)  # Attention mechanism
        out += identity
        out = F.relu(out, inplace=True)
        
        return out

class AlphaZeroNet(nn.Module):
    """
    Complete network architecture for AlphaZero.
    Tuned for RTX 3060 Ti with 8GB VRAM.
    """
    def __init__(self, game_config):
        super().__init__()
        
        # Architecture parameters
        self.board_size = game_config.board_size
        self.action_size = game_config.action_size
        self.num_channels = 256  # Good for 8GB VRAM
        self.num_blocks = 20     # Deep enough for tactics
        
        # Input processing
        self.input_conv = nn.Conv2d(
            game_config.num_input_planes,
            self.num_channels,
            kernel_size=3,
            padding=1,
            bias=False
        )
        self.input_bn = nn.BatchNorm2d(self.num_channels)
        
        # Residual tower
        self.blocks = nn.ModuleList([
            ResidualBlock(self.num_channels) 
            for _ in range(self.num_blocks)
        ])
        
        # Policy head
        self.policy_conv = nn.Conv2d(self.num_channels, 32, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(
            32 * self.board_size * self.board_size,
            self.action_size
        )
        
        # Value head
        self.value_conv = nn.Conv2d(self.num_channels, 3, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(3)
        self.value_fc1 = nn.Linear(
            3 * self.board_size * self.board_size,
            256
        )
        self.value_fc2 = nn.Linear(256, 1)
        
    def forward(self, x):
        # Initial convolution
        x = F.relu(self.input_bn(self.input_conv(x)), inplace=True)
        
        # Residual tower
        for block in self.blocks:
            x = block(x)
        
        # Policy head
        p = F.relu(self.policy_bn(self.policy_conv(x)), inplace=True)
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)
        
        # Value head
        v = F.relu(self.value_bn(self.value_conv(x)), inplace=True)
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v), inplace=True)
        v = self.value_fc2(v)
        
        return p, v
```

---

## 5) Game-Specific Implementations

### 5.1 C++ Game Interface

```cpp
// game_interface.hpp
#include <vector>
#include <cstring>

class GameState {
public:
    virtual ~GameState() = default;
    
    // Core game logic
    virtual bool is_terminal() const = 0;
    virtual float get_terminal_value() const = 0;  // From current player's POV
    virtual int get_current_player() const = 0;    // 0 or 1
    virtual int get_action_size() const = 0;
    
    // Move generation and application
    virtual void get_legal_moves(uint8_t* mask) const = 0;
    virtual void apply_move_inplace(int action) = 0;
    virtual std::unique_ptr<GameState> copy() const = 0;
    
    // Neural network features
    virtual void extract_features(float* output) const = 0;
    virtual int get_num_planes() const = 0;
};

// Example: Gomoku implementation
class GomokuState : public GameState {
private:
    static constexpr int BOARD_SIZE = 15;
    static constexpr int WIN_LENGTH = 5;
    
    uint8_t board[BOARD_SIZE][BOARD_SIZE];
    int current_player;
    int last_move;
    
public:
    bool is_terminal() const override {
        // Check for 5-in-a-row or board full
        return check_win() || is_board_full();
    }
    
    void extract_features(float* output) const override {
        // Plane 0: Current player stones
        // Plane 1: Opponent stones
        // Plane 2: Last move
        // Plane 3-6: Move history
        
        std::memset(output, 0, 7 * BOARD_SIZE * BOARD_SIZE * sizeof(float));
        
        for (int i = 0; i < BOARD_SIZE; i++) {
            for (int j = 0; j < BOARD_SIZE; j++) {
                int idx = i * BOARD_SIZE + j;
                
                if (board[i][j] == current_player) {
                    output[0 * BOARD_SIZE * BOARD_SIZE + idx] = 1.0f;
                } else if (board[i][j] == 1 - current_player) {
                    output[1 * BOARD_SIZE * BOARD_SIZE + idx] = 1.0f;
                }
                
                if (last_move == idx) {
                    output[2 * BOARD_SIZE * BOARD_SIZE + idx] = 1.0f;
                }
            }
        }
    }
};
```

### 5.2 PyBind11 Wrapper

```cpp
// python_bindings.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "game_interface.hpp"
#include "mcts_tree.hpp"

namespace py = pybind11;

// Critical: Release GIL for long operations
py::array_t<float> search_wrapper(
    GameState* state,
    int num_simulations,
    float cpuct,
    int num_threads) {
    
    py::gil_scoped_release release;  // Release GIL
    
    // Run MCTS (all in C++, no Python calls)
    MCTSTree tree(state, cpuct);
    tree.search_parallel(num_simulations, num_threads);
    
    // Extract visit counts for root children
    auto visits = tree.get_root_visits();
    
    // Return as numpy array (GIL reacquired automatically)
    return py::array_t<float>(visits.size(), visits.data());
}

PYBIND11_MODULE(mcts_cpp, m) {
    // Game states
    py::class_<GameState>(m, "GameState");
    
    py::class_<GomokuState, GameState>(m, "GomokuState")
        .def(py::init<>())
        .def("apply_move", &GomokuState::apply_move_inplace)
        .def("is_terminal", &GomokuState::is_terminal)
        .def("get_legal_moves", [](const GomokuState& self) {
            py::array_t<uint8_t> mask(15 * 15);
            self.get_legal_moves(static_cast<uint8_t*>(mask.mutable_unchecked<1>().data()));
            return mask;
        });
    
    // MCTS functions
    m.def("search", &search_wrapper, 
          "Run MCTS search",
          py::arg("state"),
          py::arg("num_simulations"),
          py::arg("cpuct") = 1.25,
          py::arg("num_threads") = 8);
}
```

---

## 6) Training Pipeline

### 6.1 Self-Play Generation

```python
# selfplay.py - Efficient self-play generation
import numpy as np
from collections import deque
import pickle
import lz4.frame

class SelfPlayGame:
    """
    Generate a single self-play game.
    Key: Temperature scheduling and position augmentation.
    """
    
    def __init__(self, game_class, mcts_engine, config):
        self.game = game_class()
        self.mcts = mcts_engine
        self.config = config
        
    def play_game(self):
        """Generate one complete game"""
        states = []
        policies = []
        current_players = []
        
        state = self.game.get_initial_state()
        
        while not state.is_terminal():
            # Temperature for exploration
            move_number = len(states)
            temp = 1.0 if move_number < 30 else 0.1
            
            # Add Dirichlet noise at root (self-play only)
            if self.config.add_dirichlet:
                root_noise = np.random.dirichlet(
                    [self.config.dirichlet_alpha] * self.game.action_size
                )
            else:
                root_noise = None
            
            # Run MCTS
            visits = self.mcts.search(
                state, 
                self.config.simulations_per_move,
                root_noise=root_noise
            )
            
            # Sample move from visit distribution
            if temp > 0:
                # Exploration: sample proportional to visits
                probs = visits ** (1.0 / temp)
                probs /= probs.sum()
                action = np.random.choice(len(probs), p=probs)
            else:
                # Exploitation: pick best
                action = np.argmax(visits)
            
            # Store position
            states.append(state.copy())
            policies.append(visits / visits.sum())
            current_players.append(state.get_current_player())
            
            # Apply move
            state.apply_move(action)
        
        # Get terminal value
        terminal_value = state.get_terminal_value()
        
        # Create training examples
        examples = []
        for state, policy, player in zip(states, policies, current_players):
            # Value is from player's perspective
            value = terminal_value if player == 0 else -terminal_value
            
            # Apply symmetries for data augmentation
            for sym_state, sym_policy in self.get_symmetries(state, policy):
                examples.append({
                    'state': sym_state,
                    'policy': sym_policy,
                    'value': value
                })
        
        return examples
    
    def get_symmetries(self, state, policy):
        """
        Game-specific symmetries for data augmentation.
        Critical for sample efficiency.
        """
        # Example for Gomoku/Go (8-fold symmetry)
        symmetries = []
        
        board = state.get_board()
        policy_2d = policy.reshape(self.game.board_size, self.game.board_size)
        
        for rot in range(4):
            # Rotations
            rot_board = np.rot90(board, rot)
            rot_policy = np.rot90(policy_2d, rot)
            symmetries.append((rot_board, rot_policy.flatten()))
            
            # Reflections
            flip_board = np.fliplr(rot_board)
            flip_policy = np.fliplr(rot_policy)
            symmetries.append((flip_board, flip_policy.flatten()))
        
        return symmetries
```

### 6.2 Training Loop

```python
# training.py - Stable training with mixed precision
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

class AlphaZeroTrainer:
    """
    Training loop with all the stability tricks.
    """
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        
        # Optimizer with weight decay
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=config.lr_cycle_length,
            T_mult=1,
            eta_min=config.min_lr
        )
        
        # Mixed precision
        self.scaler = GradScaler()
        
        # Metrics
        self.policy_losses = []
        self.value_losses = []
        
    def train_batch(self, batch):
        """
        Train on one batch.
        Critical: Proper loss scaling and gradient clipping.
        """
        states = batch['states'].cuda()
        target_policies = batch['policies'].cuda()
        target_values = batch['values'].cuda()
        
        # Mixed precision forward pass
        with autocast():
            pred_policies, pred_values = self.model(states)
            
            # Policy loss (KL divergence)
            # Important: target_policies should sum to 1
            policy_loss = F.kl_div(
                F.log_softmax(pred_policies, dim=1),
                target_policies,
                reduction='batchmean'
            )
            
            # Value loss (MSE)
            value_loss = F.mse_loss(
                pred_values.squeeze(-1),
                target_values
            )
            
            # Combined loss
            total_loss = policy_loss + self.config.value_loss_weight * value_loss
        
        # Backward pass with gradient scaling
        self.optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
        self.scaler.scale(total_loss).backward()
        
        # Gradient clipping (critical for stability)
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), 
            self.config.max_grad_norm
        )
        
        # Optimizer step
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        
        # Record metrics
        self.policy_losses.append(policy_loss.item())
        self.value_losses.append(value_loss.item())
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'total_loss': total_loss.item(),
            'lr': self.scheduler.get_last_lr()[0]
        }
    
    def validate(self, val_loader):
        """
        Validation to detect overfitting.
        """
        self.model.eval()
        
        total_policy_loss = 0
        total_value_loss = 0
        total_value_accuracy = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                states = batch['states'].cuda()
                target_policies = batch['policies'].cuda()
                target_values = batch['values'].cuda()
                
                with autocast():
                    pred_policies, pred_values = self.model(states)
                    
                    policy_loss = F.kl_div(
                        F.log_softmax(pred_policies, dim=1),
                        target_policies,
                        reduction='batchmean'
                    )
                    
                    value_loss = F.mse_loss(
                        pred_values.squeeze(-1),
                        target_values
                    )
                
                # Value accuracy (sign agreement)
                value_accuracy = ((pred_values.squeeze(-1) > 0) == 
                                 (target_values > 0)).float().mean()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_value_accuracy += value_accuracy.item()
                num_batches += 1
        
        self.model.train()
        
        return {
            'val_policy_loss': total_policy_loss / num_batches,
            'val_value_loss': total_value_loss / num_batches,
            'val_value_accuracy': total_value_accuracy / num_batches
        }
```

---

## 7) Performance Profiling & Optimization

### 7.1 Profiling Tools

```python
# profiling.py - Find the bottlenecks
import torch.profiler
import cProfile
import pstats
import io
import numpy as np

class PerformanceProfiler:
    """
    Comprehensive profiling for MCTS + GPU inference.
    """
    
    def profile_gpu(self, model, input_shape, num_iterations=100):
        """Profile GPU performance"""
        
        # Warmup
        dummy_input = torch.randn(32, *input_shape).cuda()
        for _ in range(10):
            _ = model(dummy_input)
        
        # Profile with PyTorch profiler
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA
            ],
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        ) as prof:
            for _ in range(num_iterations):
                with torch.cuda.amp.autocast():
                    _ = model(dummy_input)
                torch.cuda.synchronize()
        
        # Print results
        print(prof.key_averages().table(
            sort_by="cuda_time_total", 
            row_limit=10
        ))
        
        # Export for visualization
        prof.export_chrome_trace("gpu_trace.json")
        
        # Calculate metrics
        cuda_time_total = sum([
            item.cuda_time_total for item in prof.key_averages()
        ])
        
        return {
            'avg_gpu_time_ms': cuda_time_total / (num_iterations * 1000),
            'theoretical_throughput': 1000 / (cuda_time_total / (num_iterations * 1000))
        }
    
    def profile_mcts(self, mcts_func, state, num_simulations=1000):
        """Profile MCTS performance"""
        
        pr = cProfile.Profile()
        pr.enable()
        
        # Run MCTS
        mcts_func(state, num_simulations)
        
        pr.disable()
        
        # Get statistics
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
        ps.print_stats(20)
        
        print(s.getvalue())
        
        # Extract key metrics
        stats = pr.get_stats()
        total_time = sum(stat.totaltime for stat in stats.values())
        
        return {
            'total_time': total_time,
            'simulations_per_second': num_simulations / total_time
        }
    
    def measure_gpu_utilization(self, duration_seconds=10):
        """
        Measure actual GPU utilization over time.
        Requires nvidia-ml-py.
        """
        import pynvml
        import time
        
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        
        utilizations = []
        memory_used = []
        
        start_time = time.time()
        while time.time() - start_time < duration_seconds:
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            
            utilizations.append(util.gpu)
            memory_used.append(mem.used / 1024**3)  # GB
            
            time.sleep(0.1)
        
        pynvml.nvmlShutdown()
        
        return {
            'avg_gpu_util': np.mean(utilizations),
            'max_gpu_util': np.max(utilizations),
            'min_gpu_util': np.min(utilizations),
            'avg_memory_gb': np.mean(memory_used),
            'max_memory_gb': np.max(memory_used)
        }
```

### 7.2 Common Performance Issues & Solutions

```python
# performance_checklist.py

"""
PERFORMANCE DEBUGGING CHECKLIST

1. GPU Utilization < 70%:
   - Increase batch size
   - Decrease timeout
   - Add more concurrent games
   - Check for CPU bottleneck in feature extraction

2. CPU at 100% but low sims/s:
   - Profile with cProfile
   - Check GIL contention (should be ~0 with proper Cython)
   - Verify virtual loss isn't too high
   - Check if feature extraction is in Python (move to C++)

3. Memory Usage Growing:
   - Check for memory leaks in tree
   - Verify node pool is being reused
   - Look for Python objects holding references
   - Check experience buffer size

4. Training Loss Exploding:
   - Reduce learning rate
   - Increase gradient clipping
   - Check for NaN in features
   - Verify value targets are in [-1, 1]

5. Slow Convergence:
   - Increase Dirichlet noise
   - Check if temperature is too low
   - Verify symmetries are correct
   - Increase batch size

6. Validation Loss Increasing:
   - Reduce model size
   - Add dropout (carefully!)
   - Increase weight decay
   - Check for data leakage
"""

def diagnose_performance(metrics):
    """Automated performance diagnosis"""
    
    issues = []
    
    if metrics['gpu_util'] < 70:
        issues.append("Low GPU utilization. Increase batch size or decrease timeout.")
    
    if metrics['avg_batch_size'] < 20:
        issues.append("Small batches. Increase concurrent games or timeout.")
    
    if metrics['cpu_usage'] > 95 and metrics['sims_per_sec'] < 20000:
        issues.append("CPU bottleneck. Profile with cProfile and check GIL.")
    
    if metrics['memory_growth_rate'] > 100:  # MB/hour
        issues.append("Memory leak detected. Check node pool and Python references.")
    
    if metrics['value_loss'] > 1.0:
        issues.append("High value loss. Check target normalization.")
    
    if metrics['policy_entropy'] < 0.1:
        issues.append("Low policy entropy. Model may be overfitting.")
    
    return issues
```

---

## 8) Common Pitfalls & How to Avoid Them

### 8.1 The Big Mistakes That Kill Projects

```python
# PITFALL 1: GIL in Hot Loop
# WRONG:
def bad_select_child(node):
    best_score = -float('inf')
    best_child = None
    for child in node.children:  # Python loop = GIL held
        score = calculate_uct(child)  # Python function = slow
        if score > best_score:
            best_score = score
            best_child = child
    return best_child

# RIGHT: Use Cython with nogil
# See section 3.2 for correct implementation

# PITFALL 2: Allocating During Search
# WRONG:
class BadNode:
    def expand(self):
        for action in self.get_legal_actions():
            self.children.append(Node())  # Allocation in hot path!

# RIGHT: Pre-allocated pool
class GoodNode:
    def expand(self, node_pool):
        for action in self.get_legal_actions():
            child = node_pool.allocate()  # Reuse pre-allocated
            self.children.append(child)

# PITFALL 3: Synchronous GPU Calls
# WRONG:
def bad_mcts_step():
    leaf = select_leaf()
    value = model(leaf)  # Blocks until GPU finishes!
    backup(value)

# RIGHT: Asynchronous pattern
# See section 3.4 for correct implementation

# PITFALL 4: Wrong Value Sign
# WRONG:
def bad_backup(path, value):
    for node in path:
        node.total_value += value  # Same sign for all!

# RIGHT:
def good_backup(path, value):
    for node in reversed(path):
        node.total_value += value
        value = -value  # Flip for opponent

# PITFALL 5: No Illegal Move Masking
# WRONG:
policy = model(state)
action = np.argmax(policy)  # Might pick illegal move!

# RIGHT:
policy = model(state)
legal_mask = state.get_legal_moves()
policy = policy * legal_mask  # Mask illegal
policy = policy / policy.sum()  # Renormalize
action = np.argmax(policy)

# PITFALL 6: Training on Correlated Data
# WRONG:
for game in recent_games:
    for position in game:
        train_on(position)  # Positions from same game are correlated!

# RIGHT:
experience_buffer.shuffle()  # Shuffle across all games
batch = experience_buffer.sample(batch_size)
train_on(batch)

# PITFALL 7: Not Checking Terminal Nodes
# WRONG:
def select_leaf(node):
    while node.is_expanded:
        node = select_child(node)
    return node  # Might be terminal!

# RIGHT:
def select_leaf(node):
    while node.is_expanded and not node.is_terminal:
        node = select_child(node)
    return node

# PITFALL 8: Virtual Loss Too High/Low
# WRONG:
VIRTUAL_LOSS = 10.0  # Way too high, destroys Q-values

# RIGHT:
VIRTUAL_LOSS = 1.0  # Start here, tune based on contention

# PITFALL 9: No Warmup
# WRONG:
model = load_model()
start_inference()  # First inference is slow!

# RIGHT:
model = load_model()
warmup_gpu(model)  # Run dummy inferences first
start_inference()

# PITFALL 10: Forgetting torch.no_grad()
# WRONG:
def inference(model, x):
    return model(x)  # Builds computation graph!

# RIGHT:
def inference(model, x):
    with torch.no_grad():
        return model(x)
```

### 8.2 Debugging Techniques

```python
# debugging.py - How to debug when things go wrong

class MCTSDebugger:
    """
    Tools for debugging MCTS issues.
    """
    
    @staticmethod
    def verify_backup_correctness():
        """
        Test that backup propagates values correctly.
        """
        # Create minimal tree
        tree = create_test_tree()
        
        # Manual backup
        leaf_value = 1.0
        expected_values = [1.0, -1.0, 1.0]  # Leaf, parent, root
        
        tree.backup(leaf_path, leaf_value)
        
        for node, expected in zip(leaf_path, expected_values):
            assert abs(tree.get_value(node) - expected) < 1e-6, \
                   f"Backup failed at node {node}"
        
        print("✓ Backup correct")
    
    @staticmethod
    def check_thread_safety():
        """
        Verify no race conditions in parallel MCTS.
        """
        import threading
        
        tree = MCTSTree()
        
        def worker():
            for _ in range(1000):
                tree.select_and_expand()
        
        threads = [threading.Thread(target=worker) for _ in range(10)]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # Verify tree integrity
        assert tree.verify_integrity(), "Tree corrupted by races"
        print("✓ Thread safety verified")
    
    @staticmethod
    def profile_bottleneck():
        """
        Find what's actually slow.
        """
        import cProfile
        import pstats
        
        pr = cProfile.Profile()
        pr.enable()
        
        # Run your slow function
        run_mcts_iteration()
        
        pr.disable()
        
        # Show top 10 time consumers
        stats = pstats.Stats(pr)
        stats.sort_stats('cumulative')
        stats.print_stats(10)
    
    @staticmethod
    def verify_gpu_utilization():
        """
        Check if GPU is actually being used efficiently.
        """
        import GPUtil
        
        gpus = GPUtil.getGPUs()
        
        # Monitor during inference
        utilizations = []
        for _ in range(100):
            gpu = gpus[0]
            utilizations.append(gpu.load * 100)
            time.sleep(0.1)
        
        avg_util = np.mean(utilizations)
        
        if avg_util < 70:
            print(f"⚠ Low GPU util: {avg_util:.1f}%")
            print("  - Increase batch size")
            print("  - Decrease timeout")
            print("  - Check CPU bottleneck")
        else:
            print(f"✓ GPU util OK: {avg_util:.1f}%")
```

---

## 9) Hardware-Specific Configuration

### 9.1 Ryzen 5900X Optimization

```bash
#!/bin/bash
# setup_ryzen_5900x.sh - Optimal CPU configuration

# Set performance governor
for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    echo performance | sudo tee $cpu
done

# Disable CPU boost if thermal issues (optional)
# echo 0 | sudo tee /sys/devices/system/cpu/cpufreq/boost

# Set CPU affinity for MCTS threads (optional but recommended)
# CCD0: Cores 0-5 (physical) + 12-17 (SMT)
# CCD1: Cores 6-11 (physical) + 18-23 (SMT)

# Example: Pin MCTS to CCD0, inference to CCD1
# taskset -c 0-5,12-17 ./mcts_process
# taskset -c 6,18 ./inference_process

# Check topology
lscpu --extended
```

### 9.2 RTX 3060 Ti Configuration

```python
# gpu_config.py - RTX 3060 Ti specific settings

def configure_gpu():
    """
    Optimal settings for RTX 3060 Ti (8GB VRAM).
    """
    import os
    
    # CUDA settings
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Async execution
    os.environ['TORCH_CUDA_ARCH_LIST'] = '8.6'  # Ampere architecture
    
    # PyTorch settings
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True  # Auto-tune convolutions
    torch.backends.cuda.matmul.allow_tf32 = True  # TF32 for Ampere
    torch.cuda.set_per_process_memory_fraction(0.85)  # Leave some for OS
    
    # Memory debugging (disable in production)
    # torch.cuda.memory._record_memory_history(True)
    
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"CUDA: {torch.version.cuda}")
    print(f"cuDNN: {torch.backends.cudnn.version()}")

def estimate_batch_size(model, input_shape):
    """
    Find maximum batch size that fits in VRAM.
    """
    device = 'cuda'
    model = model.to(device)
    
    # Binary search for max batch size
    low, high = 1, 256
    max_batch = 1
    
    while low <= high:
        mid = (low + high) // 2
        try:
            # Try to allocate
            dummy = torch.randn(mid, *input_shape, device=device)
            with torch.no_grad():
                _ = model(dummy)
            torch.cuda.synchronize()
            
            # Success
            max_batch = mid
            low = mid + 1
            
            # Clean up
            del dummy
            torch.cuda.empty_cache()
            
        except RuntimeError as e:
            if 'out of memory' in str(e):
                high = mid - 1
                torch.cuda.empty_cache()
            else:
                raise
    
    # Use 80% of max for safety
    safe_batch = int(max_batch * 0.8)
    print(f"Max batch size: {max_batch}, using: {safe_batch}")
    
    return safe_batch
```

---

## 10) Testing Strategy

### 10.1 Unit Tests

```python
# test_mcts.py - Critical unit tests

import pytest
import numpy as np

class TestMCTS:
    def test_backup_value_flipping(self):
        """Value must flip sign at each level"""
        tree = MCTSTree()
        path = [0, 1, 2]  # Root -> child -> grandchild
        
        tree.backup(path, value=1.0)
        
        assert tree.nodes[2].total_value == 1.0
        assert tree.nodes[1].total_value == -1.0
        assert tree.nodes[0].total_value == 1.0
    
    def test_virtual_loss_application(self):
        """Virtual loss prevents duplicate selection"""
        tree = MCTSTree()
        node = tree.root
        
        # First selection
        child1 = tree.select_child(node)
        tree.apply_virtual_loss(child1)
        
        # Second selection should pick different child
        child2 = tree.select_child(node)
        assert child1 != child2
    
    def test_illegal_move_masking(self):
        """Illegal moves must never be selected"""
        state = TestGameState()
        state.set_illegal_moves([0, 5, 10])
        
        for _ in range(100):
            action = select_action(state)
            assert action not in [0, 5, 10]
    
    def test_thread_safety(self):
        """Parallel updates don't corrupt tree"""
        tree = MCTSTree()
        
        def worker():
            for _ in range(1000):
                path = tree.select_path()
                tree.backup(path, np.random.randn())
        
        threads = [Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # Check invariants
        assert tree.root.visit_count == sum(
            child.visit_count for child in tree.root.children
        )

class TestGPUInference:
    def test_batch_timeout(self):
        """Batch forms within timeout even with few items"""
        worker = GPUInferenceWorker(timeout_ms=10)
        
        start = time.time()
        worker.process_batch([single_item])
        elapsed = time.time() - start
        
        assert elapsed < 0.015  # 10ms timeout + overhead
    
    def test_memory_stability(self):
        """No memory leaks during inference"""
        initial_memory = torch.cuda.memory_allocated()
        
        for _ in range(100):
            batch = create_batch(32)
            _ = model(batch)
            del batch
        
        torch.cuda.empty_cache()
        final_memory = torch.cuda.memory_allocated()
        
        assert final_memory - initial_memory < 1e6  # <1MB growth
```

### 10.2 Integration Tests

```python
# test_integration.py

def test_full_self_play_game():
    """Complete game generation works end-to-end"""
    game = GomokuState()
    mcts = MCTSEngine()
    
    examples = play_self_play_game(game, mcts)
    
    assert len(examples) > 0
    assert all('state' in ex for ex in examples)
    assert all('policy' in ex for ex in examples)
    assert all('value' in ex for ex in examples)
    assert all(-1 <= ex['value'] <= 1 for ex in examples)

def test_training_convergence():
    """Model improves with training"""
    model = AlphaZeroNet(game_config)
    initial_loss = evaluate_model(model)
    
    # Train for a few iterations
    for _ in range(10):
        batch = sample_batch()
        train_step(model, batch)
    
    final_loss = evaluate_model(model)
    assert final_loss < initial_loss * 0.9  # 10% improvement
```

---

## 11) Production Deployment

### 11.1 Docker Configuration

```dockerfile
# Dockerfile
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# System dependencies
RUN apt-get update && apt-get install -y \
    python3.12 python3.12-dev python3-pip \
    build-essential cmake ninja-build \
    libomp-dev libopenblas-dev \
    htop tmux vim git \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt /tmp/
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# Build arguments for optimization
ARG TORCH_CUDA_ARCH_LIST="8.6"
ARG MAX_JOBS=12

# Copy source
COPY . /app
WORKDIR /app

# Build C++ extensions with optimization
RUN CFLAGS="-O3 -march=znver3 -fopenmp" \
    CXXFLAGS="-O3 -march=znver3 -fopenmp" \
    python3 setup.py build_ext --inplace

# Entry point
CMD ["python3", "train.py"]
```

### 11.2 Monitoring & Logging

```python
# monitoring.py - Production monitoring

import wandb
import logging
from prometheus_client import Counter, Histogram, Gauge

# Metrics
games_played = Counter('alphazero_games_played', 'Total self-play games')
simulation_rate = Gauge('alphazero_simulations_per_second', 'MCTS simulation rate')
gpu_utilization = Gauge('alphazero_gpu_utilization', 'GPU utilization percentage')
batch_size_histogram = Histogram('alphazero_batch_size', 'Inference batch sizes')

class AlphaZeroMonitor:
    def __init__(self, project_name="alphazero"):
        # Weights & Biases
        wandb.init(project=project_name)
        
        # Logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler('alphazero.log'),
                logging.StreamHandler()
            ]
        )
        
    def log_game(self, game_stats):
        games_played.inc()
        wandb.log({
            'game_length': game_stats['length'],
            'outcome': game_stats['outcome'],
            'mcts_time': game_stats['mcts_time']
        })
    
    def log_training(self, metrics):
        wandb.log({
            'policy_loss': metrics['policy_loss'],
            'value_loss': metrics['value_loss'],
            'learning_rate': metrics['lr'],
            'gpu_memory_gb': torch.cuda.memory_allocated() / 1e9
        })
    
    def log_performance(self, perf_metrics):
        simulation_rate.set(perf_metrics['sims_per_sec'])
        gpu_utilization.set(perf_metrics['gpu_util'])
        batch_size_histogram.observe(perf_metrics['batch_size'])
```

---

## 12) Final Checklist

### Before First Run

- [ ] C++ game engine compiles with `-O3 -march=znver3`
- [ ] Cython modules built with `nogil` blocks
- [ ] GPU warmup implemented
- [ ] Memory pools pre-allocated
- [ ] Virtual loss tuned (start at 1.0)
- [ ] Batch timeout configured (1-3ms)
- [ ] Thread count set (8-12 for 5900X)
- [ ] Temperature schedule defined
- [ ] Dirichlet noise configured per game

### During Development

- [ ] Profile every major change
- [ ] Monitor GPU utilization continuously
- [ ] Check for memory leaks daily
- [ ] Verify backup correctness with unit tests
- [ ] Test thread safety with sanitizers
- [ ] Track value calibration
- [ ] Log policy entropy
- [ ] Save checkpoints frequently
- [ ] Version control hyperparameters

### Before Production

- [ ] Remove all debug code
- [ ] Disable Python asserts (`python -O`)
- [ ] Enable compiler optimizations
- [ ] Set CPU governor to performance
- [ ] Configure GPU persistence mode
- [ ] Set up monitoring/alerting
- [ ] Document all hyperparameters
- [ ] Create rollback plan
- [ ] Test on held-out positions

---

## 13) Expected Timeline & Milestones

### Week 1-2: Foundation
- C++ game implementations
- Basic MCTS in Cython
- Single-threaded performance: 10k nodes/sec

### Week 3-4: Parallelization
- Multi-threaded MCTS with virtual loss
- GPU inference worker
- Target: 30k simulations/sec

### Week 5-6: Neural Network
- ResNet with SE blocks
- Mixed precision training
- Target: 85% GPU utilization

### Week 7-8: Self-Play
- Complete training pipeline
- Hyperparameter tuning
- First superhuman bot (Gomoku)

### Week 9-10: Optimization
- Profile and optimize bottlenecks
- Implement transposition tables
- Target: 40k simulations/sec

### Week 11-12: Production
- Chess and Go implementations
- Monitoring and logging
- Documentation

---

## Conclusion

This guide represents the state-of-the-art for AlphaZero implementation on consumer hardware. The architecture—traditional tree parallelism with targeted optimizations—has been proven through extensive analysis to be superior to trendy alternatives.

**Remember the fundamental truths:**
1. Search quality beats raw throughput
2. The GIL is your enemy - bypass it ruthlessly
3. 80-92% GPU utilization is realistic, not 95-98%
4. Profile everything, assume nothing
5. Simple and correct beats complex and clever

With this implementation, expect to achieve superhuman play in Gomoku within 48 hours, strong amateur level in Chess within a week, and competitive Go play within 2-3 weeks of training on your hardware.

**The path to success is not through algorithmic cleverness but through engineering discipline.**
