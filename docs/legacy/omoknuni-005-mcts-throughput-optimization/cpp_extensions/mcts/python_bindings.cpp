/**
 * @file python_bindings.cpp
 * @brief Python bindings for MCTS components including virtual loss
 *
 * This module exposes the high-performance MCTS tree and virtual loss
 * mechanisms to Python for testing and integration with the AlphaZero
 * training pipeline.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <memory>
#include <vector>
#include <chrono>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "tree.hpp"
#include "tiny_node.hpp"
#include "tiny_node_tree.hpp"
#include "tree_adapter.hpp"
#include "virtual_loss.hpp"
#include "selection.hpp"
#include "backup.hpp"
#include "simulation_runner.hpp"
#include "inference_callback.hpp"
#include "async_inference_queue.hpp"
#include "continuous_simulation_runner.hpp"
#include "batch_inference_coordinator.hpp"
#include "instrumentation.hpp"
#include "dlpack_bridge.hpp"
#include "profiling/enhanced_profiler.hpp"
#include "profiling/validation.hpp"
#include "state_pool.hpp"

namespace py = pybind11;

namespace mcts {

// Forward declare wrap_dlpack_capsule (implemented in dlpack_python.cpp)
PyObject* wrap_dlpack_capsule(DLManagedTensor* tensor);

namespace python {

using NoGil = py::call_guard<py::gil_scoped_release>;

/**
 * @brief Create a simple MCTS tree for testing purposes
 *
 * This creates a minimal tree with a few nodes for virtual loss testing.
 * In production, trees would be created by the search algorithm.
 */
std::shared_ptr<MCTSTree> create_test_tree(int max_nodes = 1000) {
    auto tree = std::make_shared<MCTSTree>(max_nodes);

    // Add root node for testing
    NodeIndex root = tree->add_root_node(0.5f, 0);

    return tree;
}

/**
 * @brief Factory function to create virtual loss manager with test tree
 */
std::shared_ptr<VirtualLossManager> create_test_virtual_loss_manager(
    std::shared_ptr<MCTSTree> tree,
    const VirtualLossConfig& config = VirtualLossConfig()) {
    return std::make_shared<VirtualLossManager>(*tree, config);
}

/**
 * @brief Factory function to create PUCT selector
 */
std::shared_ptr<PUCTSelector> create_puct_selector(
    const PUCTConfig& config = PUCTConfig()) {
    return std::make_shared<PUCTSelector>(config);
}

/**
 * @brief Factory function to create backup manager
 */
std::shared_ptr<BackupManager> create_backup_manager(
    std::shared_ptr<MCTSTree> tree,
    const BackupConfig& config = BackupConfig()) {
    return std::make_shared<BackupManager>(*tree, config);
}

PYBIND11_MODULE(mcts_py, m) {
    m.doc() = "MCTS Tree and Virtual Loss Python bindings";

    // T039: OpenMP thread count reporting for Phase 2 validation
    m.def("get_openmp_threads", []() -> int {
        #ifdef _OPENMP
            return omp_get_max_threads();
        #else
            return 1;  // No OpenMP support
        #endif
    }, "Get maximum number of OpenMP threads available");

    m.def("get_openmp_enabled", []() -> bool {
        #ifdef _OPENMP
            return true;
        #else
            return false;
        #endif
    }, "Check if OpenMP support is compiled in");

    // Node index type
    m.attr("NULL_NODE_INDEX") = NULL_NODE_INDEX;

    // Node Flags
    py::class_<NodeFlags>(m, "NodeFlags")
        .def(py::init<>())
        .def("is_expanded", &NodeFlags::is_expanded)
        .def("is_terminal", &NodeFlags::is_terminal)
        .def("current_player", &NodeFlags::current_player)
        .def("is_expanding", &NodeFlags::is_expanding)
        .def("set_expanded", &NodeFlags::set_expanded)
        .def("set_terminal", &NodeFlags::set_terminal)
        .def("set_current_player", &NodeFlags::set_current_player)
        .def("set_expanding", &NodeFlags::set_expanding)
        .def_readwrite("flags", &NodeFlags::flags);

    // Node Info
    py::class_<NodeInfo>(m, "NodeInfo")
        .def(py::init<>())
        .def_readwrite("index", &NodeInfo::index)
        .def_readwrite("visit_count", &NodeInfo::visit_count)
        .def_readwrite("total_value", &NodeInfo::total_value)
        .def_readwrite("prior_prob", &NodeInfo::prior_prob)
        .def_readwrite("virtual_loss", &NodeInfo::virtual_loss)
        .def_readwrite("parent_index", &NodeInfo::parent_index)
        .def_readwrite("first_child_index", &NodeInfo::first_child_index)
        .def_readwrite("num_children", &NodeInfo::num_children)
        .def_readwrite("flags", &NodeInfo::flags)
        .def("q_value", &NodeInfo::q_value)
        .def("is_root", &NodeInfo::is_root);

    // Virtual Loss Configuration
    py::class_<VirtualLossConfig>(m, "VirtualLossConfig")
        .def(py::init<>())
        .def(py::init<float, bool>(), py::arg("magnitude"), py::arg("enable") = true)
        .def_readwrite("magnitude", &VirtualLossConfig::magnitude)
        .def_readwrite("enable_virtual_loss", &VirtualLossConfig::enable_virtual_loss);

    // MCTS Tree (complete interface for production use)
    py::class_<MCTSTree, std::shared_ptr<MCTSTree>>(m, "MCTSTree")
        .def(py::init<std::size_t>(), py::arg("max_nodes") = 50000000, NoGil())
        .def("allocate_node", &MCTSTree::allocate_node, NoGil())
        .def("allocate_nodes", &MCTSTree::allocate_nodes, NoGil())
        .def("deallocate_node", &MCTSTree::deallocate_node, NoGil())
        .def("deallocate_nodes", &MCTSTree::deallocate_nodes, NoGil())
        .def("get_node_count", &MCTSTree::get_node_count)
        .def("get_max_nodes", &MCTSTree::get_max_nodes)
        .def("add_root_node", &MCTSTree::add_root_node, NoGil())
        .def("get_root_index", &MCTSTree::get_root_index)
        .def("is_valid_index", &MCTSTree::is_valid_index)
        .def("clear", &MCTSTree::clear, NoGil())
        // Node data access (with GIL release for hot path performance)
        .def("get_visit_count", &MCTSTree::get_visit_count, NoGil())
        .def("get_total_value", &MCTSTree::get_total_value, NoGil())
        .def("get_prior_prob", &MCTSTree::get_prior_prob, NoGil())
        .def("get_virtual_loss", &MCTSTree::get_virtual_loss, NoGil())
        .def("get_parent_index", &MCTSTree::get_parent_index, NoGil())
        .def("get_first_child_index", &MCTSTree::get_first_child_index, NoGil())
        .def("get_num_children", &MCTSTree::get_num_children, NoGil())
        .def("get_flags", &MCTSTree::get_flags, NoGil())
        .def("get_node_info", &MCTSTree::get_node_info, NoGil())
        // Node data modification (with GIL release)
        .def("set_visit_count", &MCTSTree::set_visit_count, NoGil())
        .def("set_total_value", &MCTSTree::set_total_value, NoGil())
        .def("set_prior_prob", &MCTSTree::set_prior_prob, NoGil())
        .def("set_virtual_loss", &MCTSTree::set_virtual_loss, NoGil())
        .def("set_parent_index", &MCTSTree::set_parent_index, NoGil())
        .def("set_first_child_index", &MCTSTree::set_first_child_index, NoGil())
        .def("set_num_children", &MCTSTree::set_num_children, NoGil())
        .def("set_flags", &MCTSTree::set_flags, NoGil())
        .def("atomic_try_set_expanded", &MCTSTree::atomic_try_set_expanded, NoGil(),
            "Atomically try to set expanded flag. Returns true if successful (caller owns expansion), "
            "false if already expanded by another thread.")
        .def("atomic_try_mark_expanding", &MCTSTree::atomic_try_mark_expanding, NoGil(),
            "Atomically mark a node as in-flight for expansion. Returns true if caller owns the request.")
        .def("clear_expanding_flag", &MCTSTree::clear_expanding_flag, NoGil(),
            "Clear the expanding flag after inference completes (success or failure).")
        // Memory and performance
        .def("get_memory_usage", &MCTSTree::get_memory_usage)
        .def("get_bytes_per_node", &MCTSTree::get_bytes_per_node)
        .def("get_available_nodes", &MCTSTree::get_available_nodes)
        .def("has_space_for", &MCTSTree::has_space_for)
        .def("validate_tree", &MCTSTree::validate_tree)
        // Move storage methods
        .def("get_move", &MCTSTree::get_move, NoGil())
        .def("set_move", &MCTSTree::set_move, NoGil());

    // Virtual Loss Manager
    py::class_<VirtualLossManager, std::shared_ptr<VirtualLossManager>>(m, "VirtualLossManager")
        .def("get_config", &VirtualLossManager::get_config,
             py::return_value_policy::reference_internal, NoGil())
        .def("set_config", &VirtualLossManager::set_config, NoGil())
        .def("get_virtual_loss", &VirtualLossManager::get_virtual_loss, NoGil())
        .def("reset_all_virtual_loss", &VirtualLossManager::reset_all_virtual_loss, NoGil())
        .def("apply_virtual_loss", &VirtualLossManager::apply_virtual_loss,
             py::arg("node_index"), py::arg("magnitude") = -1.0f, NoGil())
        .def("remove_virtual_loss", &VirtualLossManager::remove_virtual_loss,
             py::arg("node_index"), py::arg("magnitude") = -1.0f, NoGil())
        .def("apply_virtual_loss_to_path", &VirtualLossManager::apply_virtual_loss_to_path, NoGil())
        .def("remove_virtual_loss_from_path", &VirtualLossManager::remove_virtual_loss_from_path, NoGil())
        .def("get_statistics", &VirtualLossManager::get_statistics, NoGil());

    // Virtual Loss Statistics
    py::class_<VirtualLossManager::VirtualLossStats>(m, "VirtualLossStats")
        .def_readonly("total_applications", &VirtualLossManager::VirtualLossStats::total_applications)
        .def_readonly("total_removals", &VirtualLossManager::VirtualLossStats::total_removals)
        .def_readonly("current_active_paths", &VirtualLossManager::VirtualLossStats::current_active_paths)
        .def_readonly("max_virtual_loss", &VirtualLossManager::VirtualLossStats::max_virtual_loss)
        .def_readonly("avg_virtual_loss", &VirtualLossManager::VirtualLossStats::avg_virtual_loss);

    // Virtual Loss Guard (RAII wrapper)
    py::class_<VirtualLossGuard>(m, "VirtualLossGuard")
        .def(py::init<VirtualLossManager&, const std::vector<NodeIndex>&>(), NoGil())
        .def("is_valid", &VirtualLossGuard::is_valid, NoGil())
        .def("release", &VirtualLossGuard::release, NoGil());

    // PUCT Configuration
    py::class_<PUCTConfig>(m, "PUCTConfig")
        .def(py::init<>())
        .def_readwrite("cpuct", &PUCTConfig::cpuct)
        .def_readwrite("fpu_value", &PUCTConfig::fpu_value)
        .def_readwrite("use_fpu", &PUCTConfig::use_fpu)
        .def_readwrite("enable_simd", &PUCTConfig::enable_simd);

    // Selection Result
    py::class_<SelectionResult>(m, "SelectionResult")
        .def(py::init<>())
        .def_readwrite("selected_child", &SelectionResult::selected_child)
        .def_readwrite("best_puct_value", &SelectionResult::best_puct_value)
        .def_readwrite("child_position", &SelectionResult::child_position)
        .def_readwrite("valid", &SelectionResult::valid);

    // PUCT Selector
    py::class_<PUCTSelector, std::shared_ptr<PUCTSelector>>(m, "PUCTSelector")
        .def("select_child", &PUCTSelector::select_child, NoGil())
        .def("set_config", &PUCTSelector::set_config, NoGil())
        .def("get_config", &PUCTSelector::get_config,
             py::return_value_policy::reference_internal, NoGil())
        .def_static("is_avx2_supported", &PUCTSelector::is_avx2_supported);

    // Backup Configuration
    py::class_<BackupConfig>(m, "BackupConfig")
        .def(py::init<>())
        .def(py::init<bool, bool, float, float>(),
             py::arg("enable_value_clipping"), py::arg("enable_statistics") = true,
             py::arg("value_clip_min") = -1.0f, py::arg("value_clip_max") = 1.0f)
        .def_readwrite("enable_value_clipping", &BackupConfig::enable_value_clipping)
        .def_readwrite("enable_statistics", &BackupConfig::enable_statistics)
        .def_readwrite("value_clip_min", &BackupConfig::value_clip_min)
        .def_readwrite("value_clip_max", &BackupConfig::value_clip_max);

    // Backup Result
    py::class_<BackupResult>(m, "BackupResult")
        .def(py::init<>())
        .def_readwrite("success", &BackupResult::success)
        .def_readwrite("nodes_updated", &BackupResult::nodes_updated)
        .def_readwrite("final_root_value", &BackupResult::final_root_value)
        .def_readwrite("original_leaf_value", &BackupResult::original_leaf_value);

    // Backup Manager
    py::class_<BackupManager, std::shared_ptr<BackupManager>>(m, "BackupManager")
        .def("backup_value_along_path", &BackupManager::backup_value_along_path,
             py::arg("path"), py::arg("leaf_value"), py::arg("virtual_loss_manager") = nullptr, NoGil())
        .def("backup_terminal_value", &BackupManager::backup_terminal_value,
             py::arg("path"), py::arg("terminal_value"), py::arg("virtual_loss_manager") = nullptr, NoGil())
        .def("update_node_atomic", &BackupManager::update_node_atomic,
             py::arg("node_index"), py::arg("value_increment"), py::arg("visit_increment") = 1.0f, NoGil())
        .def("get_q_value", &BackupManager::get_q_value, NoGil())
        .def("validate_backup_path", &BackupManager::validate_backup_path, NoGil())
        .def("get_config", &BackupManager::get_config,
             py::return_value_policy::reference_internal, NoGil())
        .def("set_config", &BackupManager::set_config, NoGil())
        .def("get_statistics", &BackupManager::get_statistics, NoGil())
        .def("reset_statistics", &BackupManager::reset_statistics, NoGil());

    // Backup Statistics
    py::class_<BackupManager::BackupStats>(m, "BackupStats")
        .def_readonly("total_backups", &BackupManager::BackupStats::total_backups)
        .def_readonly("successful_backups", &BackupManager::BackupStats::successful_backups)
        .def_readonly("total_nodes_updated", &BackupManager::BackupStats::total_nodes_updated)
        .def_readonly("path_validation_failures", &BackupManager::BackupStats::path_validation_failures)
        .def_readonly("avg_path_length", &BackupManager::BackupStats::avg_path_length)
        .def_readonly("avg_absolute_leaf_value", &BackupManager::BackupStats::avg_absolute_leaf_value);

    // Backup Guard (RAII wrapper)
    py::class_<BackupGuard>(m, "BackupGuard")
        .def(py::init<BackupManager&, VirtualLossManager&, const std::vector<NodeIndex>&, float>())
        .def("was_successful", &BackupGuard::was_successful)
        .def("get_result", &BackupGuard::get_result, py::return_value_policy::reference_internal)
        .def("cleanup", &BackupGuard::cleanup);

    // Factory functions
    m.def("create_test_tree", &create_test_tree, py::arg("max_nodes") = 1000,
          "Create a test MCTS tree with basic nodes");

    m.def("create_test_virtual_loss_manager", &create_test_virtual_loss_manager,
          py::arg("tree"), py::arg("config") = VirtualLossConfig(),
          "Create a virtual loss manager for the given tree");

    m.def("create_puct_selector", &create_puct_selector,
          py::arg("config") = PUCTConfig(),
          "Create a PUCT selector with given configuration");

    m.def("create_backup_manager", &create_backup_manager,
          py::arg("tree"), py::arg("config") = BackupConfig(),
          "Create a backup manager for the given tree");

    // InferenceCallback - Abstract base class
    py::class_<InferenceCallback>(m, "InferenceCallback",
        "Abstract base class for neural network inference callbacks")
        .def("request_inference", &InferenceCallback::request_inference,
             py::arg("state"),
             "Request neural network inference for a game state");

    // PyInferenceCallback - Python callable wrapper
    py::class_<PyInferenceCallback, InferenceCallback>(m, "PyInferenceCallback",
        "Python inference callback wrapper for MCTS simulation runner.\n\n"
        "Wraps a Python callable to make it usable as an inference callback in C++.\n"
        "The callable should have signature: (state: IGameState) -> tuple[list[float], float]\n\n"
        "Example:\n"
        "    def my_inference(state):\n"
        "        policy = [0.1, 0.2, ...]  # Probability distribution\n"
        "        value = 0.5                # Position evaluation\n"
        "        return (policy, value)\n\n"
        "    callback = mcts_py.PyInferenceCallback(my_inference)")
        .def(py::init<py::object>(),
             py::arg("python_fn"),
             "Construct callback with a Python callable");

    // SimulationRunner - Phase 2 implementation complete
    py::class_<SimulationRunner>(m, "SimulationRunner",
        "High-performance MCTS simulation runner (C++ implementation).\n\n"
        "Executes complete MCTS simulations with GIL released, enabling true parallel search.\n"
        "Performance: 30k-40k simulations/second with 8 threads.")
        .def(py::init<MCTSTree&, PUCTSelector&, BackupManager&, VirtualLossManager&>(),
             py::arg("tree"), py::arg("selector"), py::arg("backup"), py::arg("virtual_loss"),
             "Construct simulation runner with required MCTS components")
        .def("run_simulation",
             &SimulationRunner::run_simulation,
             py::arg("root_state"), py::arg("root_index"), py::arg("inference_fn"),
             "Run a single MCTS simulation (select → expand → backup) with GIL released.\n\n"
             "Args:\n"
             "    root_state: Game state at root position\n"
             "    root_index: Root node index in tree\n"
             "    inference_fn: InferenceCallback for neural network evaluation\n\n"
             "Returns:\n"
             "    bool: True if simulation completed successfully");

    // BatchInferenceCallback - for batched async inference
    py::class_<BatchInferenceCallback>(m, "BatchInferenceCallback",
        "Abstract base class for batched neural network inference")
        .def("batch_inference", &BatchInferenceCallback::batch_inference,
             py::arg("states"),
             "Evaluate multiple game states in a single batch.\n\n"
             "Args:\n"
             "    states: List of IGameState pointers\n\n"
             "Returns:\n"
             "    List of (policy, value) tuples");

    // PyBatchInferenceCallback - Python wrapper
    py::class_<PyBatchInferenceCallback, BatchInferenceCallback>(m, "PyBatchInferenceCallback",
        "Python wrapper for batch inference callbacks")
        .def(py::init<py::object>(),
             py::arg("python_fn"),
             "Construct batch callback with Python callable");

    // AsyncInferenceQueue - Non-blocking inference queue for async MCTS
    // T019: InferenceRequest not exposed to Python (contains unique_ptr state)
    // Python code doesn't need to access InferenceRequest directly

    py::class_<InferenceResult>(m, "InferenceResult")
        .def(py::init<>())
        .def_readwrite("request_id", &InferenceResult::request_id)
        .def_readwrite("policy", &InferenceResult::policy)
        .def_readwrite("value", &InferenceResult::value)
        .def("__repr__", [](const InferenceResult& res) {
            return "<InferenceResult id=" + std::to_string(res.request_id) +
                   " value=" + std::to_string(res.value) + ">";
        });

    py::class_<AsyncInferenceQueue>(m, "AsyncInferenceQueue",
             "Thread-safe async inference queue for decoupling simulation from GPU\n\n"
             "Allows simulation threads to submit inference requests without blocking,\n"
             "while a coordinator thread batches requests for efficient GPU inference.")
        .def(py::init<>(),
             "Create empty async inference queue")
        .def("submit_request",
             [](AsyncInferenceQueue& queue, IGameState* state,
                NodeIndex node_index, std::vector<int16_t> path) -> uint64_t {
                 // T013-T014: Extract features in-place (ZERO COPY optimization!)
                 // Features extracted here instead of in coordinator
                 int board_size = state->getBoardSize();
                 int num_feature_planes = state->get_num_feature_planes();
                 int action_space_size = state->getActionSpaceSize();

                 // Extract features into temporary buffer
                 std::vector<float> features;
                 features.resize(num_feature_planes * board_size * board_size);
                 state->extract_features_to_buffer(features.data());

                 // Build request with move semantics
                 InferenceRequest request;
                 request.features = std::move(features);  // MOVE, not copy!
                 request.node_index = node_index;
                 request.action_space_size = action_space_size;
                 request.board_size = static_cast<int16_t>(board_size);
                 request.planes = static_cast<int16_t>(num_feature_planes);
                 request.path = std::move(path);

                 // Submit request (moves features into queue)
                 return queue.submit_request(std::move(request));
             },
             py::arg("state"), py::arg("node_index"), py::arg("path"),
             "Submit inference request with pre-extracted features (Phase 1 optimization)\n\n"
             "Args:\n"
             "    state: Game state to evaluate (features extracted in-place)\n"
             "    node_index: Tree node to expand\n"
             "    path: Path from root to node\n\n"
             "Returns:\n"
             "    int: Unique request ID for retrieving result")
        .def("collect_batch",
             &AsyncInferenceQueue::collect_batch,
             py::arg("min_batch_size"), py::arg("timeout_ms"),
             NoGil(),
             "Collect batch of pending requests\n\n"
             "Returns when either min_batch_size requests available OR timeout elapsed.\n\n"
             "Args:\n"
             "    min_batch_size: Minimum batch size to wait for\n"
             "    timeout_ms: Maximum wait time in milliseconds\n\n"
             "Returns:\n"
             "    list[InferenceRequest]: Batch of requests to process")
        .def("submit_results",
             &AsyncInferenceQueue::submit_results,
             py::arg("results"),
             "Submit batch of inference results\n\n"
             "Args:\n"
             "    results: List of InferenceResult matching collected requests")
        .def("try_get_result",
             &AsyncInferenceQueue::try_get_result,
             py::arg("request_id"),
             "Try to retrieve result for a request (non-blocking)\n\n"
             "Args:\n"
             "    request_id: Request ID from submit_request()\n\n"
             "Returns:\n"
             "    InferenceResult or None if not available")
        .def("has_results",
             &AsyncInferenceQueue::has_results,
             "Check if any results are available\n\n"
             "Returns:\n"
             "    bool: True if results available for retrieval")
        .def("pending_count",
             &AsyncInferenceQueue::pending_count,
             "Get number of pending requests\n\n"
             "Returns:\n"
             "    int: Number of requests waiting for inference")
        .def("results_count",
             &AsyncInferenceQueue::results_count,
             "Get number of completed results\n\n"
             "Returns:\n"
             "    int: Number of results available for retrieval")
        .def("get_memory_usage",
             &AsyncInferenceQueue::get_memory_usage,
             "Get memory usage estimate in bytes\n\n"
             "Returns:\n"
             "    int: Estimated memory usage")
        .def("submit_request_with_backpressure",
             [](AsyncInferenceQueue& queue, IGameState* state,
                NodeIndex node_index, std::vector<int16_t> path, double timeout_ms) -> uint64_t {
                 // T060: Phase 5 backpressure-enabled submission
                 int board_size = state->getBoardSize();
                 int num_feature_planes = state->get_num_feature_planes();
                 int action_space_size = state->getActionSpaceSize();

                 // Extract features into temporary buffer
                 std::vector<float> features;
                 features.resize(num_feature_planes * board_size * board_size);
                 state->extract_features_to_buffer(features.data());

                 // Build request with move semantics
                 InferenceRequest request;
                 request.features = std::move(features);
                 request.node_index = node_index;
                 request.action_space_size = action_space_size;
                 request.board_size = static_cast<int16_t>(board_size);
                 request.planes = static_cast<int16_t>(num_feature_planes);
                 request.path = std::move(path);

                 // Submit request with backpressure (blocks if queue full)
                 return queue.submit_request_with_backpressure(std::move(request), timeout_ms);
             },
             py::arg("state"), py::arg("node_index"), py::arg("path"), py::arg("timeout_ms") = 0.0,
             "Submit inference request with backpressure (Phase 5 multi-coordinator optimization)\n\n"
             "Blocks when queue is full (4096 entries) until space becomes available.\n"
             "Essential for multi-coordinator scenarios to prevent queue overflow.\n\n"
             "Args:\n"
             "    state: Game state to evaluate (features extracted in-place)\n"
             "    node_index: Tree node to expand\n"
             "    path: Path from root to node\n"
             "    timeout_ms: Maximum wait time in milliseconds (0 = infinite)\n\n"
             "Returns:\n"
             "    int: Unique request ID for retrieving result\n\n"
             "Raises:\n"
             "    RuntimeError: If timeout expires without space becoming available")
        .def("notify_dequeued",
             &AsyncInferenceQueue::notify_dequeued,
             "Notify waiting threads that queue space is available (Phase 5)\n\n"
             "Called after collect_batch() to wake simulation threads blocked in\n"
             "submit_request_with_backpressure(). Enables backpressure mechanism.")
        .def("shutdown",
             &AsyncInferenceQueue::shutdown,
             "Wake up threads waiting in collect_batch() for clean shutdown");

    // Instrumentation controls
    m.def("set_instrumentation_enabled",
          [](bool enabled) {
              Instrumentation::instance().set_enabled(enabled);
          },
          py::arg("enabled"),
          "Enable or disable instrumentation metrics collection");

    m.def("reset_instrumentation_metrics",
          []() {
              Instrumentation::instance().reset();
          },
          "Reset instrumentation counters and timers");

    m.def("get_instrumentation_snapshot",
          []() {
              py::dict result;
              const auto snapshot = Instrumentation::instance().snapshot();
              for (const auto& entry : snapshot) {
                  const auto metric = entry.first;
                  const auto& data = entry.second;
                  py::dict payload;
                  payload["calls"] = data.call_count;
                  payload["total_ns"] = py::int_(data.total_elapsed_ns);
                  const double avg_ns = (data.call_count > 0)
                      ? static_cast<double>(data.total_elapsed_ns) / static_cast<double>(data.call_count)
                      : 0.0;
                  payload["avg_ns"] = avg_ns;
                  result[py::str(metric_to_string(metric))] = payload;
              }
              return result;
          },
          "Get snapshot of instrumentation metrics as a dictionary");

    // ContinuousSimulationRunner - Async MCTS runner for 30k+ sims/sec
    py::class_<ContinuousSimulationRunner, SimulationRunner>(m, "ContinuousSimulationRunner",
             "Continuous MCTS simulation runner with async inference\n\n"
             "Runs simulations continuously without blocking on neural network inference.\n"
             "Achieves 30,000+ sims/sec by decoupling simulation threads from GPU latency.")
        .def(py::init<MCTSTree&, PUCTSelector&, BackupManager&, VirtualLossManager&>(),
             py::arg("tree"), py::arg("selector"), py::arg("backup"), py::arg("virtual_loss"),
             "Create continuous simulation runner\n\n"
             "Args:\n"
             "    tree: Shared MCTS tree\n"
             "    selector: PUCT selector\n"
             "    backup: Backup manager\n"
             "    virtual_loss: Virtual loss manager")
        .def("run_continuous",
             &ContinuousSimulationRunner::run_continuous,
             py::arg("root_state"), py::arg("root_index"), py::arg("queue"), py::arg("num_simulations"),
             NoGil(),
             "Run continuous MCTS simulations with async inference\n\n"
             "Simulations run in a continuous loop without blocking on inference:\n"
             "1. Select to leaf (fast C++ tree traversal)\n"
             "2. Submit inference request to queue (non-blocking)\n"
             "3. Immediately start next simulation\n"
             "4. Process completed results asynchronously\n"
             "5. Continue until quota reached\n\n"
             "Performance: 30,000+ sims/sec with 8-12 threads\n\n"
             "Args:\n"
             "    root_state: Game state at root\n"
             "    root_index: Root node index\n"
             "    queue: AsyncInferenceQueue for request/result exchange\n"
             "    num_simulations: Number of simulations to complete\n\n"
             "Returns:\n"
             "    int: Number of successfully completed simulations");

    // BatchInferenceCoordinator - Background batching coordinator
    py::class_<BatchInferenceCoordinator>(m, "BatchInferenceCoordinator",
             "Background coordinator for batched GPU inference\n\n"
             "Spawns a background thread that continuously collects inference requests\n"
             "from AsyncInferenceQueue, batches them, calls Python for GPU inference\n"
             "(single GIL crossing per batch), and distributes results back.\n\n"
             "This reduces GIL time from >50% to <30% by batching all GPU calls.")
        .def(py::init<>(),
             "Create batch inference coordinator (thread not started)")
        .def("start",
             &BatchInferenceCoordinator::start,
             py::arg("queue"), py::arg("callback"), py::arg("batch_size"), py::arg("timeout_ms"),
             "Start background coordinator thread\n\n"
             "Args:\n"
             "    queue: AsyncInferenceQueue for request/result exchange\n"
             "    callback: BatchInferenceCallback for GPU inference\n"
             "    batch_size: Minimum batch size (e.g., 32)\n"
             "    timeout_ms: Maximum wait time for batch (e.g., 2.0ms)")
        .def("stop",
             &BatchInferenceCoordinator::stop,
             "Stop background thread and wait for completion")
        .def("is_running",
             &BatchInferenceCoordinator::is_running,
             "Check if coordinator is running\n\n"
             "Returns:\n"
             "    bool: True if coordinator thread is active")
        .def("set_timeout",
             &BatchInferenceCoordinator::set_timeout,
             py::arg("timeout_ms"),
             "Update batch timeout dynamically (for adaptive batching)\n\n"
             "Args:\n"
             "    timeout_ms: New timeout in milliseconds (e.g., 2.0-10.0)\n\n"
             "Thread-safe: Can be called while coordinator is running.")
        .def("get_timeout",
             &BatchInferenceCoordinator::get_timeout,
             "Get current batch timeout\n\n"
             "Returns:\n"
             "    float: Current timeout in milliseconds")
        .def("set_batch_size",
             &BatchInferenceCoordinator::set_batch_size,
             py::arg("batch_size"),
             "Update minimum batch size dynamically\n\n"
             "Args:\n"
             "    batch_size: New minimum batch size")
        .def("get_batch_size",
             &BatchInferenceCoordinator::get_batch_size,
             "Get current minimum batch size\n\n"
             "Returns:\n"
             "    int: Current batch size");

    // PinnedBuffer - CUDA pinned memory buffer with reference counting (T007b)
    py::class_<PinnedBuffer, std::shared_ptr<PinnedBuffer>>(m, "PinnedBuffer",
             "CUDA pinned memory buffer with thread-safe reference counting\n\n"
             "Provides memory allocation optimized for GPU transfers:\n"
             "- CUDA pinned memory (cudaMallocHost) for 2-3× faster GPU transfers\n"
             "- Thread-safe atomic reference counting\n"
             "- Automatic cleanup when reference count reaches 0\n"
             "- Fallback to regular malloc if CUDA unavailable")
        .def(py::init<size_t, bool>(),
             py::arg("size_bytes"), py::arg("use_cuda") = true,
             "Allocate pinned memory buffer\n\n"
             "Args:\n"
             "    size_bytes: Size of buffer in bytes (must be > 0)\n"
             "    use_cuda: Try CUDA pinned memory (true) or force malloc (false)\n\n"
             "Raises:\n"
             "    ValueError: If size_bytes is 0\n"
             "    MemoryError: If allocation fails")
        .def("data",
             [](PinnedBuffer& self) {
                 return reinterpret_cast<uintptr_t>(self.data());
             },
             "Get pointer to buffer data as integer address\n\n"
             "Returns:\n"
             "    int: Memory address of buffer")
        .def("size",
             &PinnedBuffer::size,
             "Get buffer size in bytes\n\n"
             "Returns:\n"
             "    int: Size of allocated buffer")
        .def("is_cuda_pinned",
             &PinnedBuffer::is_cuda_pinned,
             "Check if buffer is CUDA pinned memory\n\n"
             "Returns:\n"
             "    bool: True if cudaMallocHost, False if malloc")
        .def("ref_count",
             [](const std::shared_ptr<PinnedBuffer>& self) {
                 return self.use_count();
             },
             "Get current reference count (shared_ptr use count)\n\n"
             "Returns:\n"
             "    int: Current reference count");

    // BufferPool - Buffer pool with size classes (T007b)
    // Note: BufferPool is a singleton, so we don't allow Python to create instances
    py::class_<BufferPool, std::unique_ptr<BufferPool, py::nodelete>>(m, "BufferPool",
             "Buffer pool with size classes for efficient reuse\n\n"
             "Thread-safe singleton that manages pools of pre-allocated buffers\n"
             "in common sizes (4KB, 64KB, 1MB, 4MB) to amortize allocation overhead.\n\n"
             "Performance:\n"
             "- 90%+ cache hit rate during steady state\n"
             "- O(1) lookup for pooled sizes\n"
             "- Thread-safe operations")
        .def_static("instance",
                    &BufferPool::instance,
                    py::return_value_policy::reference,
                    "Get singleton BufferPool instance\n\n"
                    "Returns:\n"
                    "    BufferPool: Global buffer pool instance")
        .def("acquire",
             &BufferPool::acquire,
             py::arg("min_size"), py::arg("use_cuda") = true,
             "Acquire buffer from pool or allocate new one\n\n"
             "Searches pool for cached buffer of appropriate size. If found,\n"
             "returns immediately. Otherwise allocates new buffer.\n\n"
             "Args:\n"
             "    min_size: Minimum required size in bytes\n"
             "    use_cuda: Prefer CUDA pinned memory (true) or malloc (false)\n\n"
             "Returns:\n"
             "    PinnedBuffer: Buffer with reference count 1\n\n"
             "Raises:\n"
             "    MemoryError: If allocation fails")
        .def("release",
             &BufferPool::release,
             py::arg("buffer"),
             "Return buffer to pool for reuse\n\n"
             "If buffer has ref_count == 1 and pool has space, caches buffer\n"
             "for reuse. Otherwise lets buffer be freed.\n\n"
             "Args:\n"
             "    buffer: Buffer to return")
        .def("clear",
             &BufferPool::clear,
             "Clear all cached buffers (for testing/cleanup)")
        .def("get_stats",
             [](const BufferPool& self) {
                 const auto stats = self.get_stats();
                 py::dict result;
                 result["total_allocated"] = stats.total_allocated;
                 result["total_reused"] = stats.total_reused;
                 result["current_pooled"] = stats.current_pooled;
                 result["current_bytes"] = stats.current_bytes;
                 return result;
             },
             "Get pool statistics (for monitoring/debugging)\n\n"
             "Returns:\n"
             "    dict: Statistics with keys:\n"
             "        - total_allocated: Lifetime allocations\n"
             "        - total_reused: Cache hits\n"
             "        - current_pooled: Buffers in pool now\n"
             "        - current_bytes: Total bytes in pool")
        .def("set_max_buffers_per_class",
             &BufferPool::set_max_buffers_per_class,
             py::arg("max_buffers"),
             "Configure maximum buffers per size class\n\n"
             "Args:\n"
             "    max_buffers: Maximum buffers to cache per size (default: 16)");

    // CUDA availability check (T007b)
    m.def("is_cuda_available",
          &is_cuda_available,
          "Check if CUDA is available for pinned memory allocation\n\n"
          "Returns:\n"
          "    bool: True if CUDA runtime is available, False otherwise");

    // DLPack Tensor Capsule API (T007c)
    py::class_<TensorShape>(m, "TensorShape",
             "Tensor shape metadata for DLPack capsules\n\n"
             "Represents 4D tensor shape: (batch, planes, height, width)")
        .def(py::init<int64_t, int64_t, int64_t, int64_t>(),
             py::arg("batch_size"), py::arg("num_planes"), py::arg("height"), py::arg("width"),
             "Create tensor shape\n\n"
             "Args:\n"
             "    batch_size: Number of states in batch\n"
             "    num_planes: Number of feature planes\n"
             "    height: Board height\n"
             "    width: Board width")
        .def_readwrite("batch_size", &TensorShape::batch_size)
        .def_readwrite("num_planes", &TensorShape::num_planes)
        .def_readwrite("height", &TensorShape::height)
        .def_readwrite("width", &TensorShape::width);

    m.def("create_dlpack_capsule",
          [](std::shared_ptr<PinnedBuffer> buffer, const TensorShape& shape, bool use_cuda) -> py::object {
              // Create DLManagedTensor
              DLManagedTensor* managed_tensor = create_dlpack_tensor(buffer, shape, use_cuda);

              // Wrap in PyCapsule
              PyObject* capsule = wrap_dlpack_capsule(managed_tensor);
              if (!capsule) {
                  throw std::runtime_error("Failed to create DLPack capsule");
              }

              // Return as py::object (takes ownership)
              return py::reinterpret_steal<py::object>(capsule);
          },
          py::arg("buffer"), py::arg("shape"), py::arg("use_cuda") = false,
          "Create DLPack capsule from pinned buffer\n\n"
          "Creates zero-copy tensor capsule compatible with torch.from_dlpack().\n"
          "The capsule shares ownership of the buffer via reference counting.\n\n"
          "Args:\n"
          "    buffer: PinnedBuffer containing tensor data\n"
          "    shape: TensorShape (batch, planes, height, width)\n"
          "    use_cuda: Whether buffer is CUDA pinned memory\n\n"
          "Returns:\n"
          "    PyCapsule: DLPack capsule for torch.from_dlpack()\n\n"
          "Example:\n"
          "    >>> buffer = mcts_py.PinnedBuffer(192, use_cuda=False)\n"
          "    >>> shape = mcts_py.TensorShape(1, 3, 4, 4)\n"
          "    >>> capsule = mcts_py.create_dlpack_capsule(buffer, shape)\n"
          "    >>> tensor = torch.from_dlpack(capsule)\n"
          "    >>> tensor.shape\n"
          "    torch.Size([1, 3, 4, 4])");

    // GameType enum (T007d)
    py::enum_<GameType>(m, "GameType",
             "Game type enumeration\n\n"
             "Defines supported game types with their feature dimensions.")
        .value("GOMOKU", GameType::GOMOKU, "Gomoku (36 planes, 15×15 board)")
        .value("CHESS", GameType::CHESS, "Chess (30 planes, 8×8 board)")
        .value("GO", GameType::GO, "Go (25 planes, 19×19 board)")
        .export_values();

    m.def("get_num_planes",
          &get_num_planes,
          py::arg("game_type"),
          "Get number of feature planes for a game type\n\n"
          "Args:\n"
          "    game_type: GameType enum value\n\n"
          "Returns:\n"
          "    int: Number of feature planes");

    m.def("get_board_size",
          [](GameType game_type) {
              auto [height, width] = get_board_size(game_type);
              return py::make_tuple(height, width);
          },
          py::arg("game_type"),
          "Get board dimensions for a game type\n\n"
          "Args:\n"
          "    game_type: GameType enum value\n\n"
          "Returns:\n"
          "    tuple[int, int]: (height, width)");

    m.def("create_batch_tensor",
          [](int batch_size, GameType game_type, bool use_cuda) -> py::object {
              // Create DLManagedTensor
              DLManagedTensor* managed_tensor = create_batch_tensor(batch_size, game_type, use_cuda);

              // Wrap in PyCapsule
              PyObject* capsule = wrap_dlpack_capsule(managed_tensor);
              if (!capsule) {
                  throw std::runtime_error("Failed to create DLPack capsule");
              }

              // Return as py::object (takes ownership)
              return py::reinterpret_steal<py::object>(capsule);
          },
          py::arg("batch_size"), py::arg("game_type"), py::arg("use_cuda") = false,
          "Create batch tensor for game states (T007d)\n\n"
          "Allocates pinned memory buffer and creates DLPack tensor for a batch.\n"
          "Features are initialized to zeros (stub implementation - real feature\n"
          "extraction will be added in T007e).\n\n"
          "Args:\n"
          "    batch_size: Number of states in batch\n"
          "    game_type: GameType (GOMOKU/CHESS/GO)\n"
          "    use_cuda: Use CUDA pinned memory for faster GPU transfers\n\n"
          "Returns:\n"
          "    PyCapsule: DLPack capsule for torch.from_dlpack()\n\n"
          "Example:\n"
          "    >>> capsule = mcts_py.create_batch_tensor(64, mcts_py.GameType.GOMOKU)\n"
          "    >>> tensor = torch.from_dlpack(capsule)\n"
          "    >>> tensor.shape\n"
          "    torch.Size([64, 36, 15, 15])");

    m.def("create_batch_tensor_from_states",
          [](const py::list& states, bool use_cuda) -> py::object {
              // Validate input
              if (states.empty()) {
                  throw std::invalid_argument("states list cannot be empty");
              }

              // Convert Python list to C++ vector
              std::vector<const alphazero::core::IGameState*> state_ptrs;
              state_ptrs.reserve(states.size());

              for (size_t i = 0; i < states.size(); ++i) {
                  // Extract IGameState pointer directly (all game states inherit from IGameState)
                  py::object state_obj = py::reinterpret_borrow<py::object>(states[i]);

                  // Try to cast to IGameState pointer
                  const alphazero::core::IGameState* state_ptr = nullptr;
                  try {
                      state_ptr = state_obj.cast<const alphazero::core::IGameState*>();
                  } catch (const py::cast_error& e) {
                      throw std::invalid_argument(
                          "State at index " + std::to_string(i) +
                          " is not a valid game state (must inherit from IGameState)"
                      );
                  }

                  if (state_ptr == nullptr) {
                      throw std::invalid_argument("State at index " + std::to_string(i) + " is null");
                  }

                  state_ptrs.push_back(state_ptr);
              }

              // Validate all states are the same type
              alphazero::core::GameType first_type = state_ptrs[0]->getGameType();
              int first_board_size = state_ptrs[0]->getBoardSize();
              for (size_t i = 1; i < state_ptrs.size(); ++i) {
                  if (state_ptrs[i]->getGameType() != first_type) {
                      throw std::invalid_argument(
                          "All states must be the same game type. State 0 has type " +
                          std::to_string(static_cast<int>(first_type)) + " but state " +
                          std::to_string(i) + " has type " +
                          std::to_string(static_cast<int>(state_ptrs[i]->getGameType()))
                      );
                  }
                  if (state_ptrs[i]->getBoardSize() != first_board_size) {
                      throw std::invalid_argument(
                          "All states must have the same board size. State 0 has size " +
                          std::to_string(first_board_size) + " but state " +
                          std::to_string(i) + " has size " +
                          std::to_string(state_ptrs[i]->getBoardSize())
                      );
                  }
              }

              // Create DLManagedTensor with real feature extraction
              DLManagedTensor* managed_tensor = nullptr;
              try {
                  managed_tensor = create_batch_tensor_from_states(state_ptrs, use_cuda);
              } catch (const std::exception& e) {
                  throw std::runtime_error(
                      "Failed to create batch tensor from states: " + std::string(e.what())
                  );
              }

              if (managed_tensor == nullptr) {
                  throw std::runtime_error("create_batch_tensor_from_states returned null");
              }

              // Wrap in PyCapsule
              PyObject* capsule = wrap_dlpack_capsule(managed_tensor);
              if (!capsule) {
                  // Note: wrap_dlpack_capsule handles cleanup on failure
                  throw std::runtime_error("Failed to create DLPack capsule");
              }

              // Return as py::object (takes ownership)
              return py::reinterpret_steal<py::object>(capsule);
          },
          py::arg("states"), py::arg("use_cuda") = false,
          "Create batch tensor from actual game states with feature extraction (T007e/T007f)\n\n"
          "Extracts features from a list of game states and creates a DLPack tensor.\n"
          "This function performs real feature extraction (unlike create_batch_tensor\n"
          "which creates zero-initialized tensors).\n\n"
          "Args:\n"
          "    states: List of game state objects (GomokuState, ChessState, or GoState)\n"
          "            All states must be the same game type and board size.\n"
          "    use_cuda: Use CUDA pinned memory for faster GPU transfers\n\n"
          "Returns:\n"
          "    PyCapsule: DLPack capsule for torch.from_dlpack()\n\n"
          "Raises:\n"
          "    ValueError: If states list is empty\n"
          "    ValueError: If states have different game types or board sizes\n"
          "    TypeError: If states are not valid game state objects\n"
          "    RuntimeError: If tensor creation or feature extraction fails\n\n"
          "Example:\n"
          "    >>> import alphazero_py as mcts_py\n"
          "    >>> import torch\n"
          "    >>> # Create game states\n"
          "    >>> states = [mcts_py.GomokuState() for _ in range(32)]\n"
          "    >>> # Make some moves\n"
          "    >>> states[0].make_move(112)  # Center move\n"
          "    >>> # Create batch tensor with real features\n"
          "    >>> capsule = mcts_py.create_batch_tensor_from_states(states)\n"
          "    >>> tensor = torch.from_dlpack(capsule)\n"
          "    >>> tensor.shape\n"
          "    torch.Size([32, 36, 15, 15])\n"
          "    >>> # Features contain actual game state (not zeros)\n"
          "    >>> assert tensor.sum() > 0  # Non-zero features\n");

    // OpenMP verification functions (T002: OpenMP Verification Script)
    m.def("get_omp_max_threads", []() {
        #ifdef _OPENMP
            return omp_get_max_threads();
        #else
            return 1;  // OpenMP not available
        #endif
    }, "Get maximum number of OpenMP threads (T002 verification)");

    m.def("benchmark_feature_extraction", [](int batch_size, int iterations) {
        #ifdef _OPENMP
            // Run benchmark of parallel loop performance
            // This simulates the feature extraction workload without needing game states
            std::vector<double> times_ms;
            times_ms.reserve(iterations);

            const int feature_size = 36 * 15 * 15;  // Gomoku: 36 planes, 15x15 board
            std::vector<float> dummy_buffer(batch_size * feature_size);

            for (int iter = 0; iter < iterations; ++iter) {
                auto start = std::chrono::high_resolution_clock::now();

                // Parallel loop that mimics feature extraction workload
                #pragma omp parallel for schedule(static) if(batch_size > 8)
                for (int b = 0; b < batch_size; ++b) {
                    // Simulate feature extraction work (memory writes + computation)
                    for (int i = 0; i < feature_size; ++i) {
                        dummy_buffer[b * feature_size + i] = static_cast<float>(b + i);
                    }
                }

                auto end = std::chrono::high_resolution_clock::now();

                // Record time in milliseconds
                double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
                times_ms.push_back(elapsed_ms);
            }

            return times_ms;
        #else
            throw std::runtime_error("OpenMP not available");
        #endif
    }, py::arg("batch_size") = 64, py::arg("iterations") = 10,
    "Benchmark feature extraction to verify OpenMP is active (T002 verification)");

    // ========================================================================
    // Enhanced Profiling System Bindings (Phase 3)
    // ========================================================================

    // ProfileLevel enum
    py::enum_<mcts::profiling::ProfileLevel>(m, "ProfileLevel",
             "Profiling level for compile-time configuration")
        .value("NONE", mcts::profiling::ProfileLevel::None, "No profiling (0% overhead)")
        .value("BASIC", mcts::profiling::ProfileLevel::Basic, "Timers only (<0.1% overhead)")
        .value("DETAILED", mcts::profiling::ProfileLevel::Detailed, "Timers + hardware counters (<0.5% overhead)")
        .value("FULL", mcts::profiling::ProfileLevel::Full, "Everything including memory tracking (<1% overhead)")
        .export_values();

    // EnhancedProfiler singleton
    py::class_<mcts::profiling::EnhancedProfiler>(m, "EnhancedProfiler",
             "Enhanced profiling system for C++ MCTS implementation\n\n"
             "Provides comprehensive instrumentation of all MCTS operations including:\n"
             "- State cloning waste (2-3× per simulation)\n"
             "- OpenMP parallelization failures\n"
             "- Thread idle time (60% waste)\n"
             "- CAS retry counts (atomic contention)\n"
             "- Mutex wait times (allocation contention)\n"
             "- Python bridge overhead\n\n"
             "All metrics are collected with <1% overhead and zero-copy when disabled.")
        .def_static("instance", &mcts::profiling::EnhancedProfiler::instance,
                    py::return_value_policy::reference,
                    "Get singleton instance of the profiler\n\n"
                    "Returns:\n"
                    "    EnhancedProfiler: Global profiler instance")
        .def("set_enabled", &mcts::profiling::EnhancedProfiler::set_enabled,
             py::arg("enabled"),
             "Enable or disable profiling\n\n"
             "When disabled, all profiling macros become no-ops with zero overhead.\n\n"
             "Args:\n"
             "    enabled: True to enable profiling, False to disable")
        .def("is_enabled", &mcts::profiling::EnhancedProfiler::is_enabled,
             "Check if profiling is currently enabled\n\n"
             "Returns:\n"
             "    bool: True if profiling is active")
        .def("set_level", &mcts::profiling::EnhancedProfiler::set_level,
             py::arg("level"),
             "Set profiling level (NONE/BASIC/DETAILED/FULL)\n\n"
             "Args:\n"
             "    level: ProfileLevel enum value")
        .def("get_level", &mcts::profiling::EnhancedProfiler::get_level,
             "Get current profiling level\n\n"
             "Returns:\n"
             "    ProfileLevel: Current level")
        .def("start_session", &mcts::profiling::EnhancedProfiler::start_session,
             py::arg("name"),
             "Start a new profiling session\n\n"
             "Resets all metrics and begins collecting data.\n\n"
             "Args:\n"
             "    name: Session name for identification")
        .def("stop_session", &mcts::profiling::EnhancedProfiler::stop_session,
             "Stop the current profiling session\n\n"
             "Finalizes metric collection and prepares for export.")
        .def("reset_metrics", &mcts::profiling::EnhancedProfiler::reset_metrics,
             "Reset all collected metrics to zero\n\n"
             "Useful for starting fresh without stopping the session.")
        .def("export_json", &mcts::profiling::EnhancedProfiler::export_json,
             py::arg("filename"),
             "Export profiling results to JSON file\n\n"
             "Creates a comprehensive JSON report with:\n"
             "- All 295 metrics (240 original + 55 bottleneck-specific)\n"
             "- Per-thread statistics\n"
             "- Percentiles (P50/P75/P90/P95/P99)\n"
             "- Bottleneck severity scores\n\n"
             "Args:\n"
             "    filename: Output file path (e.g., 'cpp_profiling.json')")
        .def("export_chrome_trace", &mcts::profiling::EnhancedProfiler::export_chrome_trace,
             py::arg("filename"),
             "Export profiling timeline to Chrome Trace format\n\n"
             "Creates a JSON file that can be opened in chrome://tracing\n"
             "for visual timeline analysis.\n\n"
             "Args:\n"
             "    filename: Output file path (e.g., 'trace.json')")
        .def("export_markdown", &mcts::profiling::EnhancedProfiler::export_markdown,
             py::arg("filename"),
             "Export profiling report to Markdown\n\n"
             "Creates a human-readable report with tables and analysis.\n\n"
             "Args:\n"
             "    filename: Output file path (e.g., 'profiling_report.md')")
        .def("print_summary", &mcts::profiling::EnhancedProfiler::print_summary,
             "Print profiling summary to console\n\n"
             "Displays key metrics and bottleneck detection results.");

    // Convenience function for profiling context manager (Python-side)
    m.def("create_profiling_session",
          [](const std::string& name, bool enable) -> mcts::profiling::EnhancedProfiler& {
              auto& profiler = mcts::profiling::EnhancedProfiler::instance();
              profiler.set_enabled(enable);
              profiler.start_session(name);
              return profiler;
          },
          py::arg("name"), py::arg("enable") = true,
          py::return_value_policy::reference,
          "Create and start a profiling session\n\n"
          "Convenience function that enables profiler and starts a session.\n\n"
          "Args:\n"
          "    name: Session name\n"
          "    enable: Whether to enable profiling (default: True)\n\n"
          "Returns:\n"
          "    EnhancedProfiler: Profiler instance\n\n"
          "Example:\n"
          "    >>> profiler = mcts_py.create_profiling_session('my_analysis')\n"
          "    >>> # Run MCTS searches...\n"
          "    >>> profiler.stop_session()\n"
          "    >>> profiler.export_json('results.json')");

    // Validation framework bindings (Phase 4)
    py::class_<mcts::profiling::ValidationResult>(m, "ValidationResult",
             "Profiling validation test result")
        .def_readonly("test_name", &mcts::profiling::ValidationResult::test_name,
                     "Name of the validation test")
        .def_readonly("passed", &mcts::profiling::ValidationResult::passed,
                     "Whether the test passed")
        .def_readonly("message", &mcts::profiling::ValidationResult::message,
                     "Test result message")
        .def_readonly("duration_ms", &mcts::profiling::ValidationResult::duration_ms,
                     "Test execution time in milliseconds");

    m.def("validate_profiling",
          &mcts::profiling::validate_profiling_infrastructure,
          "Run comprehensive profiling validation suite\n\n"
          "Tests all aspects of the profiling system:\n"
          "- Enable/disable functionality\n"
          "- Session management\n"
          "- Metric recording (timers, counters, gauges)\n"
          "- Export formats (JSON)\n"
          "- Zero overhead when disabled (<5%)\n"
          "- Thread safety\n\n"
          "Returns:\n"
          "    list[ValidationResult]: Results for each test\n\n"
          "Example:\n"
          "    >>> results = mcts_py.validate_profiling()\n"
          "    >>> for result in results:\n"
          "    ...     print(f'{result.test_name}: {'PASS' if result.passed else 'FAIL'}')\n"
          "    ...     print(f'  {result.message}')");

    m.def("run_profiling_validation",
          &mcts::profiling::run_validation,
          "Run validation and print results to console\n\n"
          "Convenience function that runs validation and prints a formatted report.\n\n"
          "Returns:\n"
          "    bool: True if all tests passed, False otherwise\n\n"
          "Example:\n"
          "    >>> if mcts_py.run_profiling_validation():\n"
          "    ...     print('Profiling system is ready!')\n"
          "    ... else:\n"
          "    ...     print('Fix validation failures before use')");

    // ========================================================================
    // State Pool Bindings (T018b)
    // ========================================================================

    // ThreadLocalStatePool::Stats (v2: lock-free ring buffer + lazy allocation)
    py::class_<ThreadLocalStatePool::Stats>(m, "StatePoolStats")
        .def(py::init<>())
        .def_readonly("total_acquires", &ThreadLocalStatePool::Stats::total_acquires)
        .def_readonly("total_releases", &ThreadLocalStatePool::Stats::total_releases)
        .def_readonly("slots_allocated", &ThreadLocalStatePool::Stats::slots_allocated)
        .def_readonly("ring_size", &ThreadLocalStatePool::Stats::ring_size)
        .def_readonly("peak_usage", &ThreadLocalStatePool::Stats::peak_usage)
        .def("__repr__", [](const ThreadLocalStatePool::Stats& s) {
            return "StatePoolStats(acquires=" + std::to_string(s.total_acquires) +
                   ", releases=" + std::to_string(s.total_releases) +
                   ", allocated=" + std::to_string(s.slots_allocated) + "/" +
                   std::to_string(s.ring_size) +
                   ", peak=" + std::to_string(s.peak_usage) + ")";
        });

    // ThreadLocalStatePool (v2: lock-free ring buffer + lazy allocation)
    py::class_<ThreadLocalStatePool>(m, "ThreadLocalStatePool")
        .def(py::init<GameType, size_t>(),
             py::arg("game_type"),
             py::arg("ring_size") = 512,
             "Create a thread-local state pool (v2: lock-free + lazy allocation)\n\n"
             "Args:\n"
             "    game_type: Type of game (GameType.GOMOKU, GameType.CHESS, GameType.GO)\n"
             "    ring_size: Ring buffer capacity (default: 512)\n\n"
             "DESIGN v2 (Performance + Memory Fix):\n"
             "- Lock-free ring buffer (no mutex contention!)\n"
             "- Lazy allocation (allocate on first access to slot)\n"
             "- No pre-allocation (starts with 0 memory)\n"
             "- Memory usage = peak concurrent simulations\n\n"
             "Example: 100 peak concurrent → 100 slots × 120KB = 12MB")
        .def("acquire", &ThreadLocalStatePool::acquire,
             py::return_value_policy::reference,
             py::call_guard<py::gil_scoped_release>(),
             "Acquire a state from the pool (lock-free, lazy allocation)\n\n"
             "Gets next slot from ring buffer. If slot is null (first access),\n"
             "allocates state lazily. Otherwise returns existing state.\n\n"
             "Performance: ~5ns if slot exists, ~1μs if allocating\n\n"
             "Returns:\n"
             "    IGameState*: Pointer to state (owned by pool)")
        .def("release", &ThreadLocalStatePool::release,
             py::arg("state"),
             py::call_guard<py::gil_scoped_release>(),
             "Release a state back to the pool (no-op)\n\n"
             "Ring buffer automatically reuses states via wraparound.\n"
             "This is a no-op for performance, included for API compat.\n\n"
             "Performance: ~2ns (updates stats only)\n\n"
             "Args:\n"
             "    state: State to release (ignored)")
        .def("get_stats", &ThreadLocalStatePool::get_stats,
             "Get pool statistics\n\n"
             "Returns:\n"
             "    StatePoolStats: Acquire/release counts, allocated slots, peak usage")
        .def("reset_stats", &ThreadLocalStatePool::reset_stats,
             "Reset statistics counters\n\n"
             "Useful for benchmarking specific code sections.");

    // Thread-local pool accessor function
    m.def("get_thread_state_pool",
          &get_thread_state_pool,
          py::arg("game_type"),
          py::arg("ring_size") = 512,
          py::return_value_policy::reference,
          "Get or create the thread-local state pool (lock-free + lazy)\n\n"
          "This is the recommended way to access state pools.\n"
          "Each thread gets its own pool.\n\n"
          "Args:\n"
          "    game_type: Type of game\n"
          "    ring_size: Ring buffer capacity (default: 512, only used on first call)\n\n"
          "Returns:\n"
          "    ThreadLocalStatePool*: Thread-local pool instance\n\n"
          "Example:\n"
          "    >>> import mcts_py\n"
          "    >>> pool = mcts_py.get_thread_state_pool(GameType.GOMOKU, 512)\n"
          "    >>> state = pool.acquire()  # Lazy alloc on first access\n"
          "    >>> # ... use state ...\n"
          "    >>> pool.release(state)  # No-op, ring buffer reuses\n"
          "    >>> stats = pool.get_stats()\n"
          "    >>> print(f'Allocated: {stats.slots_allocated}/{stats.ring_size}')");

    // ====== TinyNode and TinyNodeTree (T024f) ======

    // TinyNode structure (read-only access from Python)
    py::class_<TinyNode>(m, "TinyNode")
        .def_readonly("move", &TinyNode::move,
                     "Move that led to this node (uint16_t)")
        .def_readonly("parent_idx", &TinyNode::parent_idx,
                     "Parent node index (uint32_t)")
        .def_readonly("first_child_idx", &TinyNode::first_child_idx,
                     "First child index (uint32_t)")
        .def_readonly("next_sibling_idx", &TinyNode::next_sibling_idx,
                     "Next sibling index (uint32_t)")
        .def_readonly("prior_scaled", &TinyNode::prior_scaled,
                     "Prior probability (scaled to 0-65535)")
        .def_readonly("flags", &TinyNode::flags,
                     "Node flags (uint8_t)")
        .def_readonly("zobrist_hash", &TinyNode::zobrist_hash,
                     "Zobrist hash (uint64_t)")
        .def("get_visit_count", [](const TinyNode& node) {
            return node.visit_count.load(std::memory_order_relaxed);
        }, "Get visit count (atomic read)")
        .def("get_total_value", [](const TinyNode& node) {
            return node.total_value_scaled.load(std::memory_order_relaxed);
        }, "Get total value scaled (atomic read)")
        .def("get_virtual_loss", [](const TinyNode& node) {
            return node.virtual_loss.load(std::memory_order_relaxed);
        }, "Get virtual loss (atomic read)")
        .def("is_terminal", &TinyNode::is_terminal,
             "Check if node is terminal")
        .def("is_expanded", &TinyNode::is_expanded,
             "Check if node is expanded")
        .def("is_root", &TinyNode::is_root,
             "Check if node is root")
        .def("get_value", &TinyNode::get_value,
             "Get floating-point value")
        .def("get_prior", &TinyNode::get_prior,
             "Get floating-point prior probability")
        .def("get_q_value", &TinyNode::get_q_value,
             "Get Q-value (W/N)");

    // TinyNodeTree - Zero-copy MCTS tree
    py::class_<TinyNodeTree, std::shared_ptr<TinyNodeTree>>(m, "TinyNodeTree")
        .def(py::init<std::size_t>(),
             py::arg("max_nodes") = 50000000,
             NoGil(),
             "Initialize TinyNodeTree with specified capacity\n\n"
             "Args:\n"
             "    max_nodes: Maximum number of nodes (default: 50M)")
        .def("allocate_node", &TinyNodeTree::allocate_node, NoGil(),
             "Allocate a single node (O(1) bump allocation)\n\n"
             "Returns:\n"
             "    int: Node index, or -1 if pool is full")
        .def("deallocate_node", &TinyNodeTree::deallocate_node, NoGil(),
             py::arg("index"),
             "Deallocate a node back to the free list\n\n"
             "Args:\n"
             "    index: Node index to deallocate")
        .def("clear", &TinyNodeTree::clear, NoGil(),
             "Clear all nodes and reset tree (O(1) operation)")
        .def("get_node",
             [](TinyNodeTree& tree, int32_t index) -> TinyNode* {
                 return tree.get_node(index);
             },
             py::arg("index"),
             py::return_value_policy::reference_internal,
             "Get pointer to node by index\n\n"
             "Args:\n"
             "    index: Node index\n\n"
             "Returns:\n"
             "    TinyNode*: Pointer to node, or None if invalid")
        .def("is_valid_index", &TinyNodeTree::is_valid_index,
             py::arg("index"),
             "Check if node index is valid\n\n"
             "Args:\n"
             "    index: Node index\n\n"
             "Returns:\n"
             "    bool: True if valid")
        .def("get_node_count", &TinyNodeTree::get_node_count,
             "Get current number of allocated nodes\n\n"
             "Returns:\n"
             "    int: Node count")
        .def("get_max_nodes", &TinyNodeTree::get_max_nodes,
             "Get maximum capacity\n\n"
             "Returns:\n"
             "    int: Maximum nodes")
        .def("get_memory_usage", &TinyNodeTree::get_memory_usage,
             "Get memory usage in bytes\n\n"
             "Returns:\n"
             "    int: Memory usage (bytes)")
        .def("get_bytes_per_node", &TinyNodeTree::get_bytes_per_node,
             "Get bytes per node (always 64 due to alignment)\n\n"
             "Returns:\n"
             "    float: Bytes per node")
        .def("get_root_index", &TinyNodeTree::get_root_index,
             "Get root node index (0 if tree has nodes)\n\n"
             "Returns:\n"
             "    int: Root index, or -1 if empty")
        .def("has_space_for", &TinyNodeTree::has_space_for,
             py::arg("count"),
             "Check if tree has space for additional nodes\n\n"
             "Args:\n"
             "    count: Number of nodes to check\n\n"
             "Returns:\n"
             "    bool: True if space available")
        .def("init_root", &TinyNodeTree::init_root,
             py::arg("zobrist_hash"),
             NoGil(),
             "Initialize root node\n\n"
             "Args:\n"
             "    zobrist_hash: Initial zobrist hash for root\n\n"
             "Returns:\n"
             "    int: Root index (always 0)")
        .def("validate", &TinyNodeTree::validate,
             "Validate tree structure and constraints\n\n"
             "Returns:\n"
             "    bool: True if tree is valid")
        // Child management (T024f-2)
        .def("add_child", &TinyNodeTree::add_child,
             py::arg("parent_idx"),
             py::arg("move"),
             py::arg("prior_prob"),
             py::arg("zobrist_hash"),
             NoGil(),
             "Add a child node to a parent\n\n"
             "Args:\n"
             "    parent_idx: Parent node index\n"
             "    move: Move that leads to this child (uint16_t)\n"
             "    prior_prob: Prior probability (0.0-1.0)\n"
             "    zobrist_hash: Zobrist hash for child position\n\n"
             "Returns:\n"
             "    int: Child index, or -1 if allocation fails")
        .def("expand_node",
             [](TinyNodeTree& tree, int32_t parent_idx, py::array_t<uint16_t> moves,
                py::array_t<float> priors, py::array_t<uint64_t> zobrist_hashes) -> bool {
                 // Get numpy array buffers (WITH GIL - required for numpy access)
                 auto moves_buf = moves.request();
                 auto priors_buf = priors.request();
                 auto zobrist_buf = zobrist_hashes.request();

                 if (moves_buf.ndim != 1 || priors_buf.ndim != 1 || zobrist_buf.ndim != 1) {
                     throw std::runtime_error("All arrays must be 1-dimensional");
                 }

                 size_t num_children = static_cast<size_t>(moves_buf.shape[0]);
                 if (static_cast<size_t>(priors_buf.shape[0]) != num_children ||
                     static_cast<size_t>(zobrist_buf.shape[0]) != num_children) {
                     throw std::runtime_error("All arrays must have the same length");
                 }

                 // Now release GIL for the actual tree expansion
                 bool result;
                 {
                     py::gil_scoped_release release;
                     result = tree.expand_node(
                         parent_idx,
                         static_cast<uint16_t*>(moves_buf.ptr),
                         static_cast<float*>(priors_buf.ptr),
                         static_cast<uint64_t*>(zobrist_buf.ptr),
                         num_children
                     );
                 }
                 return result;
             },
             py::arg("parent_idx"),
             py::arg("moves"),
             py::arg("priors"),
             py::arg("zobrist_hashes"),
             "Expand a node by adding all legal children\n\n"
             "Args:\n"
             "    parent_idx: Parent node index\n"
             "    moves: Numpy array of moves (uint16_t)\n"
             "    priors: Numpy array of prior probabilities (float)\n"
             "    zobrist_hashes: Numpy array of zobrist hashes (uint64_t)\n\n"
             "Returns:\n"
             "    bool: True if all children added successfully")
        .def("get_child_count", &TinyNodeTree::get_child_count,
             py::arg("parent_idx"),
             "Get number of children for a node\n\n"
             "Args:\n"
             "    parent_idx: Parent node index\n\n"
             "Returns:\n"
             "    int: Number of children")
        .def("get_children", &TinyNodeTree::get_children,
             py::arg("parent_idx"),
             "Get all child indices as a list\n\n"
             "Args:\n"
             "    parent_idx: Parent node index\n\n"
             "Returns:\n"
             "    list[int]: List of child indices")
        // Path traversal (T024f-3)
        .def("get_path_to_node", &TinyNodeTree::get_path_to_node,
             py::arg("node_idx"),
             "Collect path from root to a node\n\n"
             "Args:\n"
             "    node_idx: Target node index\n\n"
             "Returns:\n"
             "    list[int]: Path from root to node [root, ..., node]")
        .def("select_best_child", &TinyNodeTree::select_best_child,
             py::arg("parent_idx"),
             py::arg("c_puct") = 1.0f,
             "Select best child using PUCT formula\n\n"
             "PUCT = Q + c_puct * P * sqrt(N_parent) / (1 + N_child + VL)\n\n"
             "Args:\n"
             "    parent_idx: Parent node index\n"
             "    c_puct: Exploration constant (default 1.0)\n\n"
             "Returns:\n"
             "    int: Index of best child, or -1 if no children")
        .def("apply_virtual_loss", &TinyNodeTree::apply_virtual_loss,
             py::arg("node_idx"),
             py::arg("magnitude") = 1,
             NoGil(),
             "Apply virtual loss to a node (thread-safe)\n\n"
             "Args:\n"
             "    node_idx: Node index\n"
             "    magnitude: Virtual loss magnitude (default 1)")
        .def("remove_virtual_loss", &TinyNodeTree::remove_virtual_loss,
             py::arg("node_idx"),
             py::arg("magnitude") = 1,
             NoGil(),
             "Remove virtual loss from a node (thread-safe)\n\n"
             "Args:\n"
             "    node_idx: Node index\n"
             "    magnitude: Virtual loss magnitude (default 1)")
        .def("backup_value",
             [](TinyNodeTree& tree, py::list path, float leaf_value) {
                 // Convert Python list to C++ vector
                 std::vector<int32_t> cpp_path;
                 for (auto item : path) {
                     cpp_path.push_back(item.cast<int32_t>());
                 }

                 // Release GIL for backup
                 {
                     py::gil_scoped_release release;
                     tree.backup_value(cpp_path, leaf_value);
                 }
             },
             py::arg("path"),
             py::arg("leaf_value"),
             "Backup value from leaf to root\n\n"
             "Propagates value up the tree with sign flipping (negamax).\n"
             "Thread-safe using atomic operations.\n\n"
             "Args:\n"
             "    path: Path from root to leaf (list of node indices)\n"
             "    leaf_value: Value at the leaf node");

    // ====== TreeAdapter Bindings (T024f-5) ======
    py::class_<TreeAdapter>(m, "TreeAdapter", "Adapter for TinyNodeTree with MCTSTree-compatible API")
        .def(py::init<std::size_t>(),
             py::arg("max_nodes") = 50'000'000,
             "Initialize TreeAdapter with specified capacity")

        // Tree management
        .def("get_root_index", &TreeAdapter::get_root_index,
             "Get root node index (always 0 when tree has nodes)")
        .def("get_node_count", &TreeAdapter::get_node_count,
             "Get current number of nodes in tree")
        .def("get_max_nodes", &TreeAdapter::get_max_nodes,
             "Get maximum capacity of tree")
        .def("get_memory_usage", &TreeAdapter::get_memory_usage,
             "Get memory usage in bytes")
        .def("get_bytes_per_node", &TreeAdapter::get_bytes_per_node,
             "Get bytes per node (actual memory efficiency)")
        .def("clear", &TreeAdapter::clear,
             "Clear all nodes and reset tree")

        .def("add_root_node", &TreeAdapter::add_root_node,
             py::arg("prior_prob"),
             py::arg("current_player"),
             py::arg("zobrist_hash") = 0,
             "Add root node to empty tree")

        .def("allocate_node", &TreeAdapter::allocate_node,
             "Allocate a single node from the pool")
        .def("allocate_nodes", &TreeAdapter::allocate_nodes,
             py::arg("count"),
             "Allocate multiple nodes from the pool")
        .def("deallocate_node", &TreeAdapter::deallocate_node,
             py::arg("index"),
             "Deallocate a single node back to the pool")
        .def("deallocate_nodes", &TreeAdapter::deallocate_nodes,
             py::arg("first_index"),
             py::arg("count"),
             "Deallocate multiple nodes back to the pool")

        .def("get_available_nodes", &TreeAdapter::get_available_nodes,
             "Get number of available nodes in the pool")
        .def("has_space_for", &TreeAdapter::has_space_for,
             py::arg("count"),
             "Check if tree has space for additional nodes")
        .def("validate_tree", &TreeAdapter::validate_tree,
             "Validate tree structure and constraints")

        // Node data accessors
        .def("get_visit_count", &TreeAdapter::get_visit_count,
             py::arg("index"),
             "Get visit count for node")
        .def("get_total_value", &TreeAdapter::get_total_value,
             py::arg("index"),
             "Get total value for node")
        .def("get_prior_prob", &TreeAdapter::get_prior_prob,
             py::arg("index"),
             "Get prior probability for node")
        .def("get_virtual_loss", &TreeAdapter::get_virtual_loss,
             py::arg("index"),
             "Get virtual loss for node")
        .def("get_parent_index", &TreeAdapter::get_parent_index,
             py::arg("index"),
             "Get parent index for node")
        .def("get_first_child_index", &TreeAdapter::get_first_child_index,
             py::arg("index"),
             "Get first child index for node")
        .def("get_num_children", &TreeAdapter::get_num_children,
             py::arg("index"),
             "Get number of children for node")
        .def("get_flags", &TreeAdapter::get_flags,
             py::arg("index"),
             "Get flags for node")
        .def("get_node_info", &TreeAdapter::get_node_info,
             py::arg("index"),
             "Get complete node information for debugging")

        // Node data mutators
        .def("set_visit_count", &TreeAdapter::set_visit_count,
             py::arg("index"),
             py::arg("value"),
             "Set visit count for node")
        .def("set_total_value", &TreeAdapter::set_total_value,
             py::arg("index"),
             py::arg("value"),
             "Set total value for node")
        .def("set_prior_prob", &TreeAdapter::set_prior_prob,
             py::arg("index"),
             py::arg("value"),
             "Set prior probability for node")
        .def("set_virtual_loss", &TreeAdapter::set_virtual_loss,
             py::arg("index"),
             py::arg("value"),
             "Set virtual loss for node")
        .def("set_parent_index", &TreeAdapter::set_parent_index,
             py::arg("index"),
             py::arg("parent"),
             "Set parent index for node")
        .def("set_first_child_index", &TreeAdapter::set_first_child_index,
             py::arg("index"),
             py::arg("first_child"),
             "Set first child index for node")
        .def("set_num_children", &TreeAdapter::set_num_children,
             py::arg("index"),
             py::arg("count"),
             "Set number of children for node (NO-OP in TinyNodeTree)")
        .def("set_flags", &TreeAdapter::set_flags,
             py::arg("index"),
             py::arg("flags"),
             "Set flags for node")

        // Atomic operations
        .def("atomic_try_set_expanded", &TreeAdapter::atomic_try_set_expanded,
             py::arg("index"),
             "Atomically try to set expanded flag")
        .def("atomic_try_mark_expanding", &TreeAdapter::atomic_try_mark_expanding,
             py::arg("index"),
             "Atomically mark node as being expanded")
        .def("clear_expanding_flag", &TreeAdapter::clear_expanding_flag,
             py::arg("index"),
             "Clear the expanding flag on a node")

        // TinyNodeTree-specific extensions
        .def("get_zobrist_hash", &TreeAdapter::get_zobrist_hash,
             py::arg("index"),
             "Get zobrist hash for node")
        .def("set_zobrist_hash", &TreeAdapter::set_zobrist_hash,
             py::arg("index"),
             py::arg("hash"),
             "Set zobrist hash for node")
        .def("get_move", &TreeAdapter::get_move,
             py::arg("index"),
             "Get move that led to this node")
        .def("set_move", &TreeAdapter::set_move,
             py::arg("index"),
             py::arg("move"),
             "Set move that led to this node")

        .def("get_tiny_tree",
             static_cast<TinyNodeTree* (TreeAdapter::*)()>(&TreeAdapter::get_tiny_tree),
             py::return_value_policy::reference_internal,
             "Get underlying TinyNodeTree");
}

} // namespace python
} // namespace mcts
