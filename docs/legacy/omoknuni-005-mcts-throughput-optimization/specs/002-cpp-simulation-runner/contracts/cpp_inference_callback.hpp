/**
 * Contract Specification: C++ InferenceCallback Interface
 *
 * Spec ID: 002-cpp-simulation-runner
 * File: cpp_inference_callback.hpp
 * Status: COMPLETE CONTRACT SPECIFICATION
 * Date: 2025-10-02
 *
 * This header defines the complete contract for the abstract InferenceCallback
 * interface and its concrete PyInferenceCallback implementation. It ensures
 * zero API mismatches between C++ simulation runner and Python inference workers.
 */

#ifndef ALPHAZERO_MCTS_INFERENCE_CALLBACK_CONTRACT_HPP
#define ALPHAZERO_MCTS_INFERENCE_CALLBACK_CONTRACT_HPP

#include <vector>
#include <utility>
#include <memory>
#include <stdexcept>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "games/game_interface.hpp"  // For IGameState

namespace py = pybind11;

// ==============================================================================
// SECTION 1: Abstract Interface Contract
// ==============================================================================

namespace mcts {

/**
 * @brief Abstract interface for neural network inference callbacks.
 *
 * This interface abstracts the neural network evaluation mechanism, allowing
 * the SimulationRunner to request policy/value predictions without knowing
 * whether inference happens via GPU, CPU, or mock implementation.
 *
 * DESIGN RATIONALE (mcts_guide.md:650-653):
 * - Decouples C++ MCTS from Python inference infrastructure
 * - Enables unit testing with mock implementations
 * - Supports multiple inference backends (GPU/CPU fallback)
 * - Provides clear contract for GIL management
 *
 * MEMORY OWNERSHIP:
 * - Callback does NOT own the IGameState pointer (read-only access)
 * - Callback owns the returned std::vector<float> policy
 * - Caller (SimulationRunner) owns the returned data after call
 *
 * THREAD SAFETY:
 * - Implementations MUST be thread-safe if shared across threads
 * - PyInferenceCallback acquires GIL internally (safe for multi-thread use)
 * - Mock implementations must use atomic counters if tracking call counts
 *
 * EXCEPTION SAFETY:
 * - Implementations SHOULD NOT throw exceptions during normal operation
 * - If inference fails, return uniform policy + neutral value (0.0)
 * - Critical errors may throw std::runtime_error (caught by SimulationRunner)
 */
class InferenceCallback {
public:
    virtual ~InferenceCallback() = default;

    /**
     * @brief Request neural network inference for a game state.
     *
     * @param state Const reference to game state (read-only access)
     *
     * @return std::pair<std::vector<float>, float>
     *         - first: Policy vector (unnormalized, will be masked/renormalized)
     *         - second: Value estimate in range [-1, 1]
     *
     * CONTRACT REQUIREMENTS:
     *
     * 1. POLICY VECTOR:
     *    - Size MUST match state.get_action_space_size()
     *    - Values MUST be non-negative (logits or probabilities)
     *    - Does NOT need to sum to 1.0 (will be masked and renormalized)
     *    - NaN/Inf values are NOT allowed (undefined behavior)
     *
     * 2. VALUE ESTIMATE:
     *    - MUST be in range [-1, 1] from current player's perspective
     *    - -1.0 = certain loss for current player
     *    - +1.0 = certain win for current player
     *    -  0.0 = neutral/draw position
     *
     * 3. PERFORMANCE:
     *    - Target latency: <5ms per call (including batching)
     *    - Batching handled externally (not visible to this interface)
     *    - Should avoid allocations where possible (reuse buffers)
     *
     * 4. THREAD SAFETY:
     *    - Implementation MUST handle concurrent calls safely
     *    - PyInferenceCallback uses GIL for synchronization
     *    - Mock implementations use atomic counters
     *
     * 5. GIL MANAGEMENT (Python callbacks only):
     *    - pybind11 AUTOMATICALLY acquires GIL when entering Python
     *    - GIL held only during queue submission (~10-20µs)
     *    - GIL released while waiting on Future (async pattern)
     *    - Total GIL hold time: ~35µs per simulation
     *
     * EXAMPLE USAGE (from C++):
     *
     *   IGameState& state = ...;
     *   InferenceCallback& callback = ...;
     *
     *   auto [policy, value] = callback.request_inference(state);
     *
     *   // Mask to legal moves
     *   auto legal_moves = state.get_legal_moves_as_indices();
     *   std::vector<float> masked_policy(policy.size(), 0.0f);
     *   for (int move : legal_moves) {
     *       masked_policy[move] = policy[move];
     *   }
     *
     *   // Renormalize
     *   float sum = std::accumulate(masked_policy.begin(), masked_policy.end(), 0.0f);
     *   if (sum > 1e-8f) {
     *       for (float& p : masked_policy) p /= sum;
     *   } else {
     *       // Uniform fallback
     *       float uniform = 1.0f / legal_moves.size();
     *       for (int move : legal_moves) masked_policy[move] = uniform;
     *   }
     *
     * FAILURE MODES:
     *
     * 1. Inference timeout (GPU overloaded):
     *    - Return uniform policy over legal moves
     *    - Return value = 0.0 (neutral estimate)
     *    - Log warning (do not throw)
     *
     * 2. Invalid state (should never happen):
     *    - Assert in debug builds
     *    - Return uniform policy in release builds
     *
     * 3. Python exception in callback:
     *    - pybind11 converts to C++ exception
     *    - SimulationRunner catches and logs
     *    - Simulation returns false (failure)
     */
    virtual std::pair<std::vector<float>, float>
    request_inference(const alphazero::core::IGameState& state) = 0;
};

} // namespace mcts


// ==============================================================================
// SECTION 2: Python Callback Wrapper Contract
// ==============================================================================

namespace mcts {

/**
 * @brief Concrete implementation wrapping Python callable for inference.
 *
 * This class bridges C++ simulation runner to Python inference infrastructure
 * (GPUInferenceWorker). It manages GIL acquisition/release to ensure efficient
 * parallel execution while maintaining thread safety.
 *
 * DESIGN RATIONALE (mcts_guide.md:213-293):
 * - Python coordinates async inference queues (optimal for I/O-bound GPU ops)
 * - C++ executes compute-bound MCTS tree traversal
 * - GIL released during C++ computation (enables true parallelism)
 * - GIL held only for queue submission + result extraction
 *
 * LIFECYCLE:
 * 1. Constructed once in AlphaZeroMCTS.__init__()
 * 2. Shared across all worker threads (thread-safe)
 * 3. Destroyed when MCTS instance is garbage collected
 * 4. Python callable (inference_fn) kept alive by py::object reference
 *
 * MEMORY OWNERSHIP:
 * - Owns py::object reference to Python callable (keeps alive)
 * - Does NOT own IGameState (borrows reference)
 * - Owns returned policy vector (allocated on heap)
 *
 * THREAD SAFETY GUARANTEES:
 * - GIL acquisition via pybind11 (automatic and thread-safe)
 * - Python callable must be thread-safe (inference queue handles this)
 * - Multiple threads can call request_inference() concurrently
 * - No internal mutable state (stateless wrapper)
 *
 * GIL TIMELINE (per simulation):
 *   0µs ─┬─ C++: Selection (GIL released) ────────────────────┐
 *        │                                                      │
 * 2800µs ├─ ACQUIRE GIL ──────────────────┐                    │
 * 2810µs │  Python: Submit to queue       │ ~10µs              │
 * 2820µs ├─ RELEASE GIL ──────────────────┘                    │
 *        │                                                      │
 * 2820µs ├─ C++: Wait on Future (GIL released) ────┐           │
 * 5800µs │                                          │ ~3000µs   │
 * 5800µs ├─ ACQUIRE GIL ──────────────────┐        │           │
 * 5825µs │  Python: Extract result         │ ~25µs │           │
 * 5825µs ├─ RELEASE GIL ──────────────────┘        │           │
 *        │                                          │           │
 * 5825µs ├─ C++: Backup (GIL released) ────────────┴───────────┘
 * 6000µs ─┘
 *
 * Total GIL hold time: 35µs per simulation (vs 6000µs in Python loop)
 * GIL efficiency: 99.4% released
 */
class PyInferenceCallback : public InferenceCallback {
private:
    py::object python_callable_;  ///< Python function: (state) -> Future[(policy, value)]

public:
    /**
     * @brief Construct callback wrapper from Python callable.
     *
     * @param python_callable Python function with signature:
     *        def fn(state: IGameState) -> Future[Tuple[np.ndarray, float]]
     *
     * CONTRACT:
     * - python_callable MUST be thread-safe (inference queue handles this)
     * - python_callable MUST return a concurrent.futures.Future
     * - Future.result() must return (policy: np.ndarray, value: float)
     * - Policy array must have dtype=float32 and match action space size
     * - Value must be a Python float in range [-1, 1]
     *
     * EXAMPLE PYTHON CALLABLE:
     *
     *   class CppInferenceBridge:
     *       def __init__(self, inference_fn):
     *           self.inference_fn = inference_fn  # GPUInferenceWorker
     *
     *       def __call__(self, cpp_game_state: IGameState) -> Future:
     *           # GIL held here (~10µs)
     *           future = self.inference_fn(cpp_game_state)
     *           return future  # Return immediately (async)
     *
     *   bridge = CppInferenceBridge(gpu_worker.infer)
     *   callback = mcts_py.PyInferenceCallback(bridge)
     *
     * THREAD SAFETY:
     * - py::object is thread-safe (reference counting uses GIL)
     * - Python callable invocation is synchronized by GIL
     * - No additional locking required
     */
    explicit PyInferenceCallback(py::object python_callable)
        : python_callable_(python_callable) {

        if (python_callable_.is_none()) {
            throw std::invalid_argument(
                "PyInferenceCallback: python_callable cannot be None"
            );
        }

        // Verify callable (will throw py::error_already_set if not)
        if (!py::hasattr(python_callable_, "__call__")) {
            throw std::invalid_argument(
                "PyInferenceCallback: python_callable must be callable"
            );
        }
    }

    /**
     * @brief Request inference from Python infrastructure.
     *
     * IMPLEMENTATION CONTRACT:
     *
     * 1. GIL ACQUISITION (automatic via pybind11):
     *    - pybind11 acquires GIL when entering Python code
     *    - No manual py::gil_scoped_acquire needed
     *    - GIL held during:
     *        a) Python function call (~5µs)
     *        b) Queue submission (~5µs)
     *        c) Result extraction (~25µs)
     *    - Total GIL time: ~35µs
     *
     * 2. ASYNC PATTERN:
     *    - Python callable returns Future immediately
     *    - C++ waits on Future.result() WITH GIL RELEASED
     *    - GPU batching happens asynchronously (≥32 positions OR ≤3ms timeout)
     *
     * 3. TIMEOUT HANDLING:
     *    - Future.result(timeout=1.0) raises TimeoutError after 1 second
     *    - Catch TimeoutError and return uniform policy + neutral value
     *    - Log warning for monitoring
     *
     * 4. NUMPY ARRAY CONVERSION:
     *    - Python returns np.ndarray (float32)
     *    - pybind11 converts to std::vector<float> (zero-copy view if contiguous)
     *    - Ownership transferred to C++ (copy if needed)
     *
     * 5. ERROR HANDLING:
     *    - Python exceptions converted to C++ exceptions by pybind11
     *    - Catch py::error_already_set for Python-specific errors
     *    - Catch std::exception for generic errors
     *    - Return uniform policy on error (do not propagate)
     *
     * PERFORMANCE OPTIMIZATION:
     * - GIL released while waiting on Future (enables parallel GPU batching)
     * - No unnecessary copies (numpy array → std::vector is efficient)
     * - No allocations in hot path (policy vector reused by caller)
     *
     * REFERENCE IMPLEMENTATION (cpp_extensions/mcts/simulation_runner.cpp):
     *
     *   std::pair<std::vector<float>, float>
     *   PyInferenceCallback::request_inference(const IGameState& state) {
     *       try {
     *           // GIL automatically acquired by pybind11
     *           py::object future = python_callable_(&state);
     *
     *           // Wait on future with timeout
     *           py::object result_tuple = future.attr("result")(1.0);  // 1s timeout
     *
     *           // Extract policy and value
     *           py::array_t<float> policy_array = result_tuple[0].cast<py::array_t<float>>();
     *           float value = result_tuple[1].cast<float>();
     *
     *           // Convert numpy array to std::vector
     *           std::vector<float> policy(
     *               policy_array.data(),
     *               policy_array.data() + policy_array.size()
     *           );
     *
     *           // Validate value range
     *           if (value < -1.0f || value > 1.0f) {
     *               throw std::runtime_error("Value out of range [-1, 1]");
     *           }
     *
     *           return {policy, value};
     *
     *       } catch (const py::error_already_set& e) {
     *           // Python exception (timeout, error in inference)
     *           py::print("Inference error:", e.what());
     *
     *           // Fallback: uniform policy + neutral value
     *           size_t action_space = state.get_action_space_size();
     *           std::vector<float> uniform_policy(action_space, 1.0f / action_space);
     *           return {uniform_policy, 0.0f};
     *
     *       } catch (const std::exception& e) {
     *           // C++ exception
     *           std::cerr << "Inference error: " << e.what() << std::endl;
     *
     *           size_t action_space = state.get_action_space_size();
     *           std::vector<float> uniform_policy(action_space, 1.0f / action_space);
     *           return {uniform_policy, 0.0f};
     *       }
     *   }
     */
    std::pair<std::vector<float>, float>
    request_inference(const alphazero::core::IGameState& state) override;
};

} // namespace mcts


// ==============================================================================
// SECTION 3: Mock Implementation Contract (for Testing)
// ==============================================================================

namespace mcts {

/**
 * @brief Mock inference callback for deterministic unit testing.
 *
 * This implementation returns configurable fixed responses without any neural
 * network inference. Used for validating SimulationRunner logic independently
 * of GPU infrastructure.
 *
 * USE CASES:
 * 1. Unit tests (verify tree updates, value propagation)
 * 2. Performance benchmarks (eliminate GPU variability)
 * 3. Integration tests (validate end-to-end flow)
 *
 * THREAD SAFETY:
 * - All counters use std::atomic for safe concurrent access
 * - No mutable state beyond counters (stateless policy/value)
 * - Safe to share across multiple threads
 */
class MockInferenceCallback : public InferenceCallback {
private:
    std::vector<float> fixed_policy_;   ///< Pre-computed policy (uniform by default)
    float fixed_value_;                 ///< Fixed value estimate
    mutable std::atomic<uint64_t> call_count_{0};  ///< Total calls (for validation)

public:
    /**
     * @brief Construct mock with uniform policy and neutral value.
     *
     * @param action_space_size Number of possible actions
     * @param value Fixed value to return (default 0.0)
     *
     * CONTRACT:
     * - Policy will be uniform over all actions (not masked to legal moves)
     * - Caller responsible for masking (same as real inference)
     * - Value clamped to [-1, 1] range
     */
    explicit MockInferenceCallback(size_t action_space_size, float value = 0.0f)
        : fixed_policy_(action_space_size, 1.0f / action_space_size),
          fixed_value_(std::clamp(value, -1.0f, 1.0f)) {}

    /**
     * @brief Construct mock with custom policy and value.
     *
     * @param policy Custom policy vector (will be normalized if sum != 1.0)
     * @param value Fixed value estimate
     *
     * CONTRACT:
     * - Policy size must match action space of tested game
     * - Policy values must be non-negative
     * - If sum != 1.0, will be normalized automatically
     */
    MockInferenceCallback(std::vector<float> policy, float value)
        : fixed_policy_(std::move(policy)),
          fixed_value_(std::clamp(value, -1.0f, 1.0f)) {

        // Normalize policy if needed
        float sum = std::accumulate(fixed_policy_.begin(), fixed_policy_.end(), 0.0f);
        if (std::abs(sum - 1.0f) > 1e-6f && sum > 1e-8f) {
            for (float& p : fixed_policy_) p /= sum;
        }
    }

    /**
     * @brief Return fixed policy and value.
     *
     * CONTRACT:
     * - Returns copy of fixed_policy_ (caller owns)
     * - Increments call_count_ atomically
     * - Never throws exceptions
     * - O(n) time (policy vector copy)
     */
    std::pair<std::vector<float>, float>
    request_inference(const alphazero::core::IGameState& state) override {
        call_count_.fetch_add(1, std::memory_order_relaxed);
        return {fixed_policy_, fixed_value_};
    }

    /**
     * @brief Get total number of inference calls.
     *
     * Used in tests to verify expected number of simulations executed.
     *
     * CONTRACT:
     * - Thread-safe (atomic read)
     * - Returns total calls across all threads
     */
    uint64_t get_call_count() const {
        return call_count_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Reset call counter to zero.
     *
     * Useful for running multiple test iterations with same mock instance.
     */
    void reset_call_count() {
        call_count_.store(0, std::memory_order_relaxed);
    }
};

} // namespace mcts


// ==============================================================================
// SECTION 4: Pybind11 Binding Contract
// ==============================================================================

/**
 * PYBIND11 MODULE BINDINGS (cpp_extensions/mcts/python_bindings.cpp):
 *
 * REQUIRED BINDINGS:
 *
 * 1. Abstract Interface (trampoline class for Python subclasses):
 *
 *    class PyInferenceCallbackTrampoline : public mcts::InferenceCallback {
 *    public:
 *        using mcts::InferenceCallback::InferenceCallback;
 *
 *        std::pair<std::vector<float>, float>
 *        request_inference(const IGameState& state) override {
 *            PYBIND11_OVERRIDE_PURE(
 *                std::pair<std::vector<float>, float>,
 *                mcts::InferenceCallback,
 *                request_inference,
 *                state
 *            );
 *        }
 *    };
 *
 * 2. Module Registration:
 *
 *    PYBIND11_MODULE(mcts_py, m) {
 *        // Abstract interface (for documentation)
 *        py::class_<mcts::InferenceCallback, PyInferenceCallbackTrampoline>(
 *            m, "InferenceCallback",
 *            "Abstract interface for neural network inference"
 *        )
 *        .def("request_inference",
 *             &mcts::InferenceCallback::request_inference,
 *             py::arg("state"),
 *             "Request inference for game state",
 *             py::call_guard<py::gil_scoped_release>()  // Release GIL
 *        );
 *
 *        // Concrete Python wrapper
 *        py::class_<mcts::PyInferenceCallback, mcts::InferenceCallback>(
 *            m, "PyInferenceCallback",
 *            "Wrapper for Python inference callable"
 *        )
 *        .def(py::init<py::object>(),
 *             py::arg("python_callable"),
 *             "Construct from Python callable: (state) -> Future[(policy, value)]"
 *        );
 *
 *        // Mock implementation (for testing)
 *        py::class_<mcts::MockInferenceCallback, mcts::InferenceCallback>(
 *            m, "MockInferenceCallback",
 *            "Mock inference for testing"
 *        )
 *        .def(py::init<size_t, float>(),
 *             py::arg("action_space_size"),
 *             py::arg("value") = 0.0f,
 *             "Construct mock with uniform policy"
 *        )
 *        .def(py::init<std::vector<float>, float>(),
 *             py::arg("policy"),
 *             py::arg("value"),
 *             "Construct mock with custom policy"
 *        )
 *        .def("get_call_count",
 *             &mcts::MockInferenceCallback::get_call_count,
 *             "Get total inference calls"
 *        )
 *        .def("reset_call_count",
 *             &mcts::MockInferenceCallback::reset_call_count,
 *             "Reset call counter to zero"
 *        );
 *    }
 *
 * 3. GIL Management:
 *    - request_inference() uses py::call_guard<py::gil_scoped_release>()
 *    - GIL automatically reacquired when entering Python callback
 *    - No manual GIL management in user code
 *
 * PYTHON USAGE EXAMPLE:
 *
 *   import mcts_py
 *   import numpy as np
 *   from concurrent.futures import Future
 *
 *   class CppInferenceBridge:
 *       def __init__(self, gpu_worker):
 *           self.gpu_worker = gpu_worker
 *
 *       def __call__(self, state):
 *           # Submit to GPU queue (async)
 *           future = self.gpu_worker.infer(state)
 *           return future
 *
 *   bridge = CppInferenceBridge(gpu_inference_worker)
 *   callback = mcts_py.PyInferenceCallback(bridge)
 *
 *   # Use in SimulationRunner
 *   runner = mcts_py.SimulationRunner(tree, selector, backup, vl)
 *   success = runner.run_simulation(state, root_idx, callback)
 */


// ==============================================================================
// SECTION 5: Testing Contracts
// ==============================================================================

/**
 * UNIT TEST REQUIREMENTS (tests/unit/test_inference_callback.cpp):
 *
 * TEST 1: Mock callback returns fixed values
 *   - Create MockInferenceCallback with action_space=362, value=0.5
 *   - Call request_inference() with dummy state
 *   - Verify policy.size() == 362
 *   - Verify all policy values == 1/362 (±1e-6)
 *   - Verify value == 0.5
 *   - Verify call_count == 1
 *
 * TEST 2: Mock callback thread safety
 *   - Create MockInferenceCallback
 *   - Spawn 8 threads, each calling request_inference() 1000 times
 *   - Join all threads
 *   - Verify call_count == 8000
 *   - Verify no race conditions (run with TSan)
 *
 * TEST 3: Custom policy normalization
 *   - Create MockInferenceCallback with policy=[1, 2, 3], value=0.0
 *   - Verify returned policy is [1/6, 2/6, 3/6]
 *
 * TEST 4: Value clamping
 *   - Create MockInferenceCallback with value=5.0 (out of range)
 *   - Verify returned value is 1.0 (clamped)
 *
 * INTEGRATION TEST REQUIREMENTS (tests/integration/test_cpp_inference.py):
 *
 * TEST 5: PyInferenceCallback with real GPU worker
 *   - Create CppInferenceBridge wrapping GPUInferenceWorker
 *   - Create PyInferenceCallback(bridge)
 *   - Run 100 simulations with 8 threads
 *   - Verify GPU utilization >80%
 *   - Verify average batch size ≥32
 *   - Verify throughput ≥30,000 sims/sec
 *
 * TEST 6: Timeout handling
 *   - Create mock bridge that returns Future with 10s delay
 *   - Call request_inference() with 1s timeout
 *   - Verify TimeoutError caught gracefully
 *   - Verify uniform policy returned as fallback
 *
 * TEST 7: GIL release verification
 *   - Measure GIL hold time during 1000 simulations
 *   - Verify GIL held <10% of total time
 *   - Verify thread efficiency 75-85% with 8 threads
 *
 * PERFORMANCE BENCHMARK REQUIREMENTS (tests/performance/bench_inference.py):
 *
 * BENCH 1: Mock inference overhead
 *   - Target: 1,400+ sims/sec single thread (mock callback)
 *   - Target: 10,000+ sims/sec with 8 threads (mock callback)
 *
 * BENCH 2: Real GPU inference throughput
 *   - Target: 30,000-40,000 sims/sec with 8 threads
 *   - GPU utilization: 80-92%
 *   - Average batch size: 32-64
 */


// ==============================================================================
// SECTION 6: Reference to Original Design
// ==============================================================================

/**
 * ORIGINAL DESIGN RATIONALE (mcts_guide.md):
 *
 * Lines 650-653: GIL Scoped Release Pattern
 *   "C++ releases GIL for tree traversal, automatically reacquires when
 *    calling Python callback. This enables true parallel execution while
 *    maintaining thread safety."
 *
 * Lines 213-293: Async Inference Coordinator
 *   "Python manages async inference queues using concurrent.futures.
 *    C++ submits requests and waits on Futures. GPU batching happens
 *    automatically (≥32 positions OR ≤3ms timeout)."
 *
 * Lines 69-70: Python Coordinates, C++ Computes
 *   "Python never touches hot loops. All MCTS tree traversal in C++.
 *    Python only for config, data loading, high-level orchestration."
 *
 * PERFORMANCE TARGETS (mcts_guide.md:1724-1738):
 *   - GIL held: <10% of total time
 *   - Thread efficiency: 75-85% with 8 threads
 *   - GPU utilization: 80-92% (sustained)
 *   - Throughput: 30,000-40,000 sims/sec
 */

#endif // ALPHAZERO_MCTS_INFERENCE_CALLBACK_CONTRACT_HPP
