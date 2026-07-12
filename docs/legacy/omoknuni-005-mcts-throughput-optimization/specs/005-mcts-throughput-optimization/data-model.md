# Data Model: MCTS Throughput Optimization

**Feature**: MCTS Throughput Optimization
**Date**: 2025-10-20
**Reference**: [plan.md](plan.md#d1-data-model-data-modelmd)

## Overview

This document defines the data structures, memory layouts, and validation rules for the zero-copy MCTS optimization. All structures are designed to minimize allocations and enable move semantics.

---

## Core Data Structures

### InferenceRequest (NEW - Move-Only)

**Purpose**: Neural network inference request with pre-extracted features (owned, not shared)

**C++ Definition**:
```cpp
struct InferenceRequest {
    std::vector<float> features;        // OWNED (moved from thread-local buffer)
    int32_t node_index;                 // Tree node requiring evaluation
    int32_t action_space_size;          // Number of legal moves
    int16_t board_size;                 // Board dimension (8, 9, 15, or 19)
    int16_t planes;                     // Feature plane count (25-36)
    std::vector<int16_t> path;          // Move path from root (optional reconstruction)
    uint64_t request_id;                // Unique request identifier

    // Move-only semantics (no copy allowed)
    InferenceRequest(InferenceRequest&&) = default;
    InferenceRequest& operator=(InferenceRequest&&) = default;
    InferenceRequest(const InferenceRequest&) = delete;
    InferenceRequest& operator=(const InferenceRequest&) = delete;
};
```

**Memory Layout**:
- `features`: Dynamic vector (size = `planes × board_size²`), moved from thread-local buffer
- `node_index`: 4 bytes
- `action_space_size`: 4 bytes
- `board_size`: 2 bytes
- `planes`: 2 bytes
- `path`: Dynamic vector (size = depth from root, typically 10-50 moves)
- `request_id`: 8 bytes

**Validation Rules**:
- `features.size() == planes × board_size × board_size`
- `action_space_size > 0 && action_space_size <= 512`
- `planes >= 17 && planes <= 36` (range for all 3 games)
- `board_size ∈ {8, 9, 15, 19}`

**State Transitions**:
1. **Created**: Thread-local buffer allocated and filled
2. **Moved to Queue**: `std::move(request)` transfers ownership to queue
3. **Batched**: Coordinator aggregates multiple requests
4. **Destroyed**: After inference results distributed

**Relationships**:
- Submitted to `AsyncInferenceQueue` (producer: simulation threads)
- Consumed by `BatchInferenceCoordinator` (consumer: coordinator thread)

---

### ThreadLocalState (MODIFIED)

**Purpose**: Per-thread simulation state with pre-allocated feature buffer

**C++ Definition** (additions only):
```cpp
struct ThreadLocalState {
    // ... EXISTING FIELDS (tree traversal state, RNG, etc.) ...

    // NEW: Pre-allocated feature buffer (Phase 1A)
    std::vector<float> feature_buffer;  // Size = max_planes × max_board²
    bool feature_buffer_initialized;    // Guard against double-initialization
};
```

**Memory Layout**:
- `feature_buffer`: 36 × 19 × 19 × 4 bytes = 51,984 bytes ≈ 52KB per thread
- Total for 8 threads: 8 × 52KB = 416KB (negligible)

**Initialization** (once per thread):
```cpp
void initialize_feature_buffer(int max_planes, int max_board_size) {
    if (!feature_buffer_initialized) {
        feature_buffer.resize(max_planes * max_board_size * max_board_size);
        feature_buffer_initialized = true;
    }
}
```

**Validation Rules**:
- `feature_buffer.size() == max_planes × max_board_size²`
- `feature_buffer_initialized == true` before use

**Reuse Pattern**:
1. Fill buffer with `game->extract_features_to_buffer(state, feature_buffer.data())`
2. Move buffer to `InferenceRequest`: `request.features = std::move(feature_buffer)`
3. Buffer is now empty (moved-from state)
4. On next simulation, buffer is resized and refilled (automatic via `std::move` semantics)

---

### FeatureBuffer (NEW - Phase 2, Pinned Memory Pool)

**Purpose**: Pre-allocated pinned CPU tensor buffer for fast GPU transfer

**Python Definition**:
```python
class FeatureBuffer:
    def __init__(self, max_batch=64, max_planes=36, max_h=19, max_w=19):
        # Pinned CPU buffer (allocated once, reused forever)
        self.pinned_buffer = torch.zeros(
            (max_batch, max_planes, max_h, max_w),
            dtype=torch.float32,
            pin_memory=True
        )

        # GPU buffer (allocated once, reused forever)
        self.gpu_buffer = torch.zeros(
            (max_batch, max_planes, max_h, max_w),
            dtype=torch.float32,
            device='cuda'
        )

        # CUDA stream for non-blocking transfers
        self.stream = torch.cuda.Stream()

        # Metadata
        self.max_batch_size = max_batch
        self.max_planes = max_planes
        self.max_height = max_h
        self.max_width = max_w
        self.total_bytes = max_batch * max_planes * max_h * max_w * 4
        self.is_pinned = self.pinned_buffer.is_pinned()
```

**Memory Layout**:
- Pinned CPU buffer: 64 × 36 × 19 × 19 × 4 bytes = 3,326,976 bytes ≈ 3.3MB
- GPU buffer: 64 × 36 × 19 × 19 × 4 bytes = 3,326,976 bytes ≈ 3.3MB
- Total overhead: 6.6MB (0.04% of 8GB VRAM, negligible)

**Validation Rules**:
- `is_pinned == True` (enforced at initialization)
- `total_bytes <= MCTS_PINNED_BYTES_CAP` (configurable pinned memory cap, default: 32MB on 64GB RAM systems)
- **Warning**: Over-allocating pinned memory can degrade system performance (see NVIDIA best practices). Use `--pinned-mb` CLI override or `MCTS_PINNED_BYTES_CAP` environment variable to adjust. Typical safe range: 16-64MB on systems with ≥32GB RAM.

**Reuse Pattern** (event-based handoff, no synchronize in hot path):
1. Fill slice: `pinned_buffer[0:batch_size, 0:planes, 0:H, 0:W].copy_(features)`
2. Transfer async: `gpu_buffer[:batch_size, ...].copy_(pinned_buffer[:batch_size, ...], non_blocking=True)` (within stream context)
3. Record event: `event = torch.cuda.Event(); event.record(stream)`
4. Return tuple: `return (gpu_buffer[:batch_size, :planes, :H, :W], event, stream)`

**Coordinator Usage**:
- Execute model forward on returned stream: `with torch.cuda.stream(stream): model(batch_tensor)`
- OR wait on event if using different stream: `coordinator_stream.wait_event(event)`
- NO `stream.synchronize()` in hot path (maintains true async overlap)

**Performance Impact**:
- Tensor creation: 37ms → ≤2.0ms (p95, batch_size=64, 18× improvement)
- GIL holding time: 37ms → <0.5ms (p95, 74× reduction)
- True async overlap: H2D transfer can overlap with previous GPU inference or CPU prep

---

### ProfilingMetrics (EXTENDED)

**Purpose**: Performance measurement data for profiling campaigns and phase validation

**C++ Definition** (new fields only):
```cpp
struct ProfilingMetrics {
    // ... EXISTING FIELDS (simulation timing, GPU utilization, etc.) ...

    // PHASE 1 METRICS
    double state_cloning_us;            // Time in state cloning (target: ~0 after Phase 1)
    double feature_extraction_us;       // Time in in-place extraction
    uint64_t state_clone_count;         // Count of clone() calls (target: 0)
    uint64_t feature_move_count;        // Count of std::move(features) to queue

    // PHASE 2 METRICS
    double tensor_creation_ms;          // Time to create batch tensor (target: ≤2.0ms)
    double h2d_transfer_ms;             // Host-to-device transfer time (target: ≤1.0ms)
    int32_t openmp_thread_count;        // Actual OpenMP threads used (target: >1)
    bool openmp_enabled;                // True if OpenMP linked
    double pinned_buffer_reuse_pct;     // % of batches using pre-allocated buffer (target: 100%)

    // PHASE 3A METRICS (optional)
    int32_t active_coordinators;        // Number of active coordinator threads
    double coordinator_blocking_pct;    // % of time coordinators block (target: <10%)
};
```

**Validation Rules**:
- **Phase 1 Acceptance**:
  - `state_cloning_us / total_time_us < 0.01` (<1%)
  - `state_clone_count == 0`
  - `feature_move_count == simulation_count`

- **Phase 2 Acceptance**:
  - `tensor_creation_ms <= 2.0`
  - `h2d_transfer_ms <= 1.0`
  - `openmp_thread_count > 1`
  - `openmp_enabled == true`
  - `pinned_buffer_reuse_pct == 100.0`

- **Phase 3A Acceptance** (optional):
  - `coordinator_blocking_pct < 10.0`
  - `active_coordinators ∈ {2, 3, 4}`

**Collection Pattern**:
```cpp
// Start timer
auto start = std::chrono::high_resolution_clock::now();

// Operation
game->extract_features_to_buffer(state, buffer);

// Record
auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
    std::chrono::high_resolution_clock::now() - start
);
profiling.record_feature_extraction(duration.count());
```

---

## Memory Budget Analysis

### Per-Thread Memory (8 simulation threads)

| Component | Size per Thread | Total (8 threads) |
|-----------|-----------------|-------------------|
| Feature buffer | 52KB | 416KB |
| Node arena block | 256KB (4096 nodes × 64 bytes) | 2MB |
| Other thread state | ~16KB | 128KB |
| **Total** | **324KB** | **2.6MB** |

**Conclusion**: Thread-local buffers add negligible overhead (<3MB for 8 threads).

---

### Global Memory (Shared Structures)

| Component | Size | Usage |
|-----------|------|-------|
| MCTS tree (10M nodes) | 270MB | 27 bytes/node SoA layout |
| Inference queue ring | 1MB | 4096-entry ring buffer metadata |
| Inference queue payload (worst-case) | ~212MB | In-flight features @ 4096 entries × ~52KB (Go 19×19); recommend reducing ring depth or using fixed pool |
| Pinned CPU buffer | 3.3MB (default) | Pre-allocated, reused; configurable via `MCTS_PINNED_BYTES_CAP` (default: 32MB cap on 64GB RAM) |
| GPU buffer | 3.3MB (default) | Pre-allocated, reused |
| **Total (typical)** | **~280MB** | Under 1GB target ✅ |
| **Total (worst-case spike)** | **~490MB** | During full queue saturation (rare); still under budget |

**Conclusion**: All optimizations stay within memory budget. Typical usage <300MB. Worst-case spikes during queue saturation ~490MB (still << 1GB target).

**Recommendation**: For memory-constrained environments, either (a) reduce queue depth from 4096 to 1024-2048, or (b) refactor queue to hold handles into a fixed-size feature pool (e.g., 256 pre-allocated 52KB buffers = 13MB fixed) with recycling.

---

### GPU Memory (VRAM Budget: 8GB)

| Component | Size | Percentage |
|-----------|------|------------|
| Feature buffers | 3.3MB | 0.04% |
| Neural network model | ~200MB | 2.4% |
| PyTorch overhead | ~500MB | 6.0% |
| **Total** | **703.3MB** | **8.4%** |

**Remaining VRAM**: 8GB - 703MB = 7.3GB (91.6% available for other uses)

**Conclusion**: Pinned memory optimization uses <0.1% of VRAM, negligible impact.

---

## Validation & Testing

### Unit Tests

**Feature Extraction Correctness**:
```python
def test_inplace_extraction_identical_to_copy():
    """Verify in-place extraction produces identical features to copy-based"""
    state = create_test_state()

    # Copy-based (legacy)
    state_copy = state.clone()
    features_copy = extract_features(state_copy)

    # In-place (new)
    features_inplace = extract_features_to_buffer(state, buffer)

    np.testing.assert_array_equal(features_copy, features_inplace)
```

**Move Semantics**:
```cpp
TEST(InferenceRequest, MoveSemantics) {
    InferenceRequest req;
    req.features.resize(1000);

    size_t orig_size = req.features.size();
    InferenceRequest moved = std::move(req);

    // Original is moved-from (empty)
    EXPECT_EQ(req.features.size(), 0);

    // Moved-to has ownership
    EXPECT_EQ(moved.features.size(), orig_size);
}
```

**Pinned Buffer Reuse**:
```python
def test_pinned_buffer_never_reallocates():
    """Verify pinned buffer is reused (same memory address)"""
    bridge = DLPackInferenceBridge()

    addr_before = bridge.pinned_buffer.data_ptr()

    for _ in range(100):  # 100 batches
        bridge.create_batch_tensor(requests)
        addr_after = bridge.pinned_buffer.data_ptr()
        assert addr_after == addr_before, "Buffer reallocated!"
```

---

### Integration Tests

**End-to-End Zero-Copy**:
```python
def test_zero_copy_simulation():
    """Verify state cloning eliminated in full simulation"""
    allocations_before = get_allocation_count()

    run_mcts_search(simulations=800)

    allocations_after = get_allocation_count()

    # Allow only thread-local buffer allocations (one-time)
    max_allowed = 8  # 8 threads × 1 buffer each
    assert allocations_after - allocations_before <= max_allowed
```

---

## References

- [plan.md](plan.md): Complete implementation plan with code examples
- [research.md](research.md): Research findings and architectural decisions
- [contracts/](contracts/): API interface specifications
- [CLAUDE.md](../../CLAUDE.md): Constitution principles (zero-copy, coordinator efficiency)

---

## Summary

All data structures optimized for zero-copy operations and minimal allocation. Thread-local buffers (416KB total) and pinned memory pools (6.6MB total) add negligible overhead while enabling 58-75× performance improvement. Move semantics enforce ownership transfer, eliminating state cloning bottleneck (86.6% of baseline execution time).
