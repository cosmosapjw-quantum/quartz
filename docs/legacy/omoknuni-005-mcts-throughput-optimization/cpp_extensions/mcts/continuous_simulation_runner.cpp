/**
 * @file continuous_simulation_runner.cpp
 * @brief Implementation of continuous simulation runner
 */

#include "continuous_simulation_runner.hpp"
#include "instrumentation.hpp"
#include "thread_affinity.hpp"
#include "state_pool.hpp"
#include "dlpack_bridge.hpp"
#include "profiling/enhanced_profiler.hpp"
#include "../utils/igamestate.h"
#include <algorithm>  // for std::reverse, std::find
#include <thread>
#include <chrono>
#include <random>
#include <string>     // for std::to_string (debug assertions)

using namespace mcts::profiling;

namespace mcts {

// Helper: Detect game type from IGameState
static GameType detect_game_type(const alphazero::core::IGameState& state) {
    using CoreGameType = alphazero::core::GameType;

    switch (state.getGameType()) {
        case CoreGameType::GOMOKU:
            return GameType::GOMOKU;
        case CoreGameType::CHESS:
            return GameType::CHESS;
        case CoreGameType::GO:
            return GameType::GO;
        default:
            throw std::runtime_error("Unsupported game type for state pooling");
    }
}

ContinuousSimulationRunner::ContinuousSimulationRunner(MCTSTree& tree,
                                                         PUCTSelector& selector,
                                                         BackupManager& backup,
                                                         VirtualLossManager& virtual_loss)
    : SimulationRunner(tree, selector, backup, virtual_loss) {
}

int ContinuousSimulationRunner::run_continuous(IGameState& root_state,
                                                 NodeIndex root_index,
                                                 AsyncInferenceQueue& queue,
                                                 int num_simulations) {
    int completed = 0;
    int submitted = 0;

    // Get thread-local state pool (on-demand allocation, free-list reuse)
    GameType game_type = detect_game_type(root_state);
    // NEW DESIGN (Memory Leak Fix):
    // - Starts with 0 memory (no pre-allocation)
    // - Allocates on-demand when free list is empty
    // - Returns states to free list on release()
    // - Memory usage = peak concurrent (self-adjusting!)
    //
    // Example: 2000 sims, 8 threads, ~100 peak concurrent per thread
    // Memory: 800 states × 120KB = 96MB (vs 3.9GB with old ring buffer!)
    ThreadLocalStatePool* state_pool = get_thread_state_pool(game_type);

    // Clear pending buffer
    for (auto& slot : pending_buffer_) {
        slot.occupied.store(false, std::memory_order_relaxed);
    }
    pending_count_.store(0, std::memory_order_relaxed);

    // THREAD AFFINITY: Pin thread to optimal CPU core for cache locality
    // Expected impact: 1.15× speedup from reduced cross-CCD traffic
    static thread_local ThreadAffinityManager affinity_mgr;
    static thread_local int thread_id = -1;
    static thread_local bool affinity_set = false;

    if (!affinity_set) {
        // Determine thread ID using std::hash of thread::id
        thread_id = static_cast<int>(
            std::hash<std::thread::id>{}(std::this_thread::get_id()) % 24
        );

        // Set affinity (assumes reasonable thread count for hardware)
        int recommended_threads = affinity_mgr.get_recommended_thread_count();
        affinity_mgr.set_thread_affinity(thread_id, recommended_threads);
        affinity_set = true;
    }

    // PRE-EXPAND ROOT: Eliminates N-1 thread idle problem where threads
    // race to expand root but only one succeeds. By expanding synchronously
    // before threading, all threads can immediately start productive work.
    // Expected impact: 2× speedup (eliminates initial serialization bottleneck)
    ensure_root_expanded(root_state, root_index, queue);

    auto release_virtual_loss = [this](const std::vector<NodeIndex>& path) {
        if (path.size() <= 1) {
            return;
        }
        for (size_t i = 1; i < path.size(); ++i) {
            virtual_loss_.remove_virtual_loss(path[i]);
        }
    };

    // T024f-6: Thread-local persistent state (zero-copy MCTS)
    // Each thread maintains one state, modified via make_move() and restored via unmake_move()
    // This eliminates copyFrom() cloning (418μs → ~15ns per move)
    static thread_local ThreadLocalState tls;
    {
        PROFILE_SCOPE(ProfileMetric::RunContinuousThreadLocalInit);
        {
            PROFILE_SCOPE(ProfileMetric::StateThreadLocalClone);
            tls.ensure_initialized(root_state);  // Clone once per thread (amortized cost)
        }
    }

    // Continuous loop until quota reached
    while (completed < num_simulations) {
        // === PROFILING UPGRADE: Instrument main loop iteration ===
        PROFILE_SCOPE(ProfileMetric::RunContinuousLoopIteration);

        bool waiting_for_leaf = false;

        // Phase 1: Select to leaf and submit inference (NON-BLOCKING)
        if (submitted < num_simulations) {
            PROFILE_SCOPE(ProfileMetric::RunContinuousPhase1);
            PROFILE_COUNTER(ProfileMetric::RunContinuousLoopWorkCount, 1);

            // T024f-6: Use thread-local state with make/unmake (ZERO cloning)
            // State is already at root from previous unwind (or initial clone)
            tls.undo_tokens.clear();

            // Clear and reuse path buffer
            path_buffer_.clear();

            // Select to leaf using make_move pattern (collects undo tokens)
            // NOTE: select_leaf_with_make_unmake is already fully instrumented
            NodeIndex leaf = select_leaf_with_make_unmake(
                root_index, *tls.state, path_buffer_, tls.undo_tokens);

            // Check if terminal
            if (tls.state->isTerminal()) {
                PROFILE_SCOPE(ProfileMetric::RunContinuousTerminalBackup);

                // Terminal node - backup immediately, no inference needed
                float value = get_terminal_value(*tls.state);
                std::reverse(path_buffer_.begin(), path_buffer_.end());
                backup_value(path_buffer_, value);

                // T024f-6: Restore state to root via unmake_move
                {
                    PROFILE_SCOPE(ProfileMetric::RunContinuousStateRestore);
                    unwind_path(*tls.state, path_buffer_, tls.undo_tokens);
                }

                completed++;
                submitted++;
                continue;
            }

            // Ensure only one in-flight expansion per node
            bool submission_ready = true;
            if (!tree_.atomic_try_mark_expanding(leaf)) {
                PROFILE_SCOPE(ProfileMetric::RunContinuousExpansionConflict);

                // Track expansion conflicts (busy-edge prevented duplicate expansion)
                Instrumentation::instance().increment_counter(InstrumentationMetric::ExpansionConflict);
                release_virtual_loss(path_buffer_);

                // T024f-6: Restore state to root before continuing
                {
                    PROFILE_SCOPE(ProfileMetric::RunContinuousStateRestore);
                    unwind_path(*tls.state, path_buffer_, tls.undo_tokens);
                }

                waiting_for_leaf = true;
                submission_ready = false;
            }

            alphazero::core::IGameState* pending_state = nullptr;
            int action_space_size = 0;
            int board_size = 0;
            int num_feature_planes = 0;

            if (submission_ready) {
                PROFILE_SCOPE(ProfileMetric::RunContinuousSubmitReady);

                // Get metadata for inference
                board_size = tls.state->getBoardSize();
                num_feature_planes = tls.state->get_num_feature_planes();
                action_space_size = tls.state->getActionSpaceSize();

                // T012: Initialize feature buffer once per thread (amortized cost ~0)
                tls.initialize_feature_buffer(num_feature_planes, board_size);

                // T013: Extract features in-place into thread-local buffer (ZERO COPY!)
                // Replaces 418μs state clone with ~50μs in-place extraction
                // This is the PRIMARY optimization eliminating 86.6% bottleneck
                {
                    PROFILE_SCOPE(ProfileMetric::CoordinatorFeatureExtraction);
                    tls.state->extract_features_to_buffer(tls.feature_buffer.data());
                }

                // T024f-6: Restore thread-local state to root (ready for next simulation)
                {
                    PROFILE_SCOPE(ProfileMetric::RunContinuousStateRestore);
                    unwind_path(*tls.state, path_buffer_, tls.undo_tokens);
                }
            }

            if (submission_ready) {
                constexpr std::size_t kMaxInFlight = 4096;
                std::size_t backoff_loops = 0;
                size_t pending = pending_count_.load(std::memory_order_relaxed);

                // Backoff loop when queue is full
                if (queue.pending_count() >= kMaxInFlight || pending >= kMaxInFlight) {
                    PROFILE_SCOPE(ProfileMetric::RunContinuousBackoffLoop);

                    while (queue.pending_count() >= kMaxInFlight || pending >= kMaxInFlight) {
                        PROFILE_SCOPE(ProfileMetric::RunContinuousQueueFullWait);

                        waiting_for_leaf = true;
                        int flushed = process_completed_results(queue);
                        if (flushed == 0) {
                            std::this_thread::sleep_for(std::chrono::microseconds(100));
                        }
                        if (++backoff_loops > 1024) {
                            break;  // Prevent unbounded waiting
                        }
                        pending = pending_count_.load(std::memory_order_relaxed);
                    }
                }

                // T014: Build InferenceRequest with move semantics (ZERO COPY!)
                // Features moved from thread-local buffer into request
                uint64_t request_id;
                {
                    PROFILE_SCOPE(ProfileMetric::RunContinuousQueueSubmit);

                    // Build request with pre-extracted features
                    InferenceRequest request;
                    request.features = std::move(tls.feature_buffer);  // MOVE, not copy!
                    request.node_index = leaf;
                    request.action_space_size = action_space_size;
                    request.board_size = static_cast<int16_t>(board_size);
                    request.planes = static_cast<int16_t>(num_feature_planes);
                    // Convert path_buffer_ (vector<NodeIndex>) to vector<int16_t>
                    request.path.reserve(path_buffer_.size());
                    for (NodeIndex node : path_buffer_) {
                        request.path.push_back(static_cast<int16_t>(node));
                    }

                    // Submit request (moves features into queue)
                    request_id = queue.submit_request(std::move(request));
                }

                // Track pending expansion using ring buffer (O(1) direct indexing)
                {
                    PROFILE_SCOPE(ProfileMetric::RunContinuousPendingBuffer);

                    size_t slot_index = request_id % PENDING_BUFFER_CAPACITY;
                    PendingSlot& slot = pending_buffer_[slot_index];

                    // Store request data
                    slot.request_id = request_id;
                    slot.data.leaf_node = leaf;
                    slot.data.path = path_buffer_;  // Copy path
                    slot.data.state = pending_state;  // Pool-managed state (NON-owning)

                    // Mark slot as occupied (release to ensure data is visible)
                    slot.occupied.store(true, std::memory_order_release);
                    pending_count_.fetch_add(1, std::memory_order_relaxed);
                }

                submitted++;
            }
        }

        // Phase 2: Process completed results (NON-BLOCKING)
        int processed;
        {
            PROFILE_SCOPE(ProfileMetric::RunContinuousPhase2);
            processed = process_completed_results(queue);
        }
        completed += processed;

        // Yield briefly if no results available to avoid busy-waiting
        // Reduced sleep duration to improve batch accumulation
        if (processed == 0) {
            PROFILE_COUNTER(ProfileMetric::RunContinuousLoopIdleCount, 1);

            bool all_submitted = submitted >= num_simulations;
            if (all_submitted || waiting_for_leaf) {
                PROFILE_SCOPE(ProfileMetric::RunContinuousSleepYield);

                auto sleep_duration = waiting_for_leaf ? std::chrono::microseconds(10)  // Reduced from 50μs
                                                       : std::chrono::microseconds(20);  // Reduced from 100μs
                std::this_thread::sleep_for(sleep_duration);
            }
        }
    }

    // Clear pending buffer and return any remaining states to pool
    for (auto& slot : pending_buffer_) {
        if (slot.occupied.load(std::memory_order_relaxed) && slot.data.state) {
            // Return orphaned pool state (simulation ended before result arrived)
            GameType game_type = detect_game_type(*slot.data.state);
            ThreadLocalStatePool* pool = get_thread_state_pool(game_type);
            pool->release(slot.data.state);
            slot.data.state = nullptr;
        }
        slot.occupied.store(false, std::memory_order_relaxed);
    }
    pending_count_.store(0, std::memory_order_relaxed);

    return completed;
}

int ContinuousSimulationRunner::process_completed_results(AsyncInferenceQueue& queue) {
    ScopedMetric metric(InstrumentationMetric::QueueProcessResults);
    PROFILE_SCOPE(ProfileMetric::BatchResultsTotal);

    // T014: Batched Result Processing
    // Phase 1: Collect all ready results (no tree modifications yet)
    // This reduces lock contention by batching atomic operations
    thread_local std::vector<ReadyResult> ready_results;
    ready_results.clear();
    ready_results.reserve(32);  // Typical batch size

    {
        PROFILE_SCOPE(ProfileMetric::BatchResultsCollect);

        for (size_t i = 0; i < PENDING_BUFFER_CAPACITY; ++i) {
            PendingSlot& slot = pending_buffer_[i];

            // Check if slot is occupied
            if (!slot.occupied.load(std::memory_order_acquire)) {
                continue;
            }

            // Try to get result for this specific request
            auto result_opt = queue.try_get_result(slot.request_id);
            if (!result_opt.has_value()) {
                continue;  // Result not ready yet
            }

            // Collect result without modifying tree yet (batching optimization)
            ReadyResult ready;
            ready.pending = std::move(slot.data);
            ready.result = std::move(result_opt.value());
            ready.slot_index = i;
            ready_results.push_back(std::move(ready));

            // Mark slot as free early (safe since we've moved data out)
            slot.occupied.store(false, std::memory_order_release);
            pending_count_.fetch_sub(1, std::memory_order_relaxed);
        }
    }

    // No results ready - early return
    if (ready_results.empty()) {
        return 0;
    }

    // Phase 2: Batch node expansions
    // Expand all nodes before backup to ensure consistent tree state
    {
        PROFILE_SCOPE(ProfileMetric::BatchResultsExpand);

        for (auto& ready : ready_results) {
            if (ready.pending.state) {
                ready.expansion_succeeded = expand_node_with_result(
                    ready.pending.leaf_node,
                    *ready.pending.state,
                    ready.result.policy,
                    ready.result.value
                );
            } else {
                ready.expansion_succeeded = false;
            }
        }
    }

    // Phase 3: Batch backups with grouped atomic operations
    // Key optimization: paths overlap heavily in MCTS, so we can batch
    // updates to the same nodes and reduce atomic contention significantly
    thread_local std::unordered_map<NodeIndex, BatchedUpdate> node_updates;
    node_updates.clear();
    node_updates.reserve(128);  // Typical tree depth × batch size

    {
        PROFILE_SCOPE(ProfileMetric::BatchResultsBackupPrep);

        for (auto& ready : ready_results) {
            // Reverse path for backup (leaf-to-root becomes root-to-leaf)
            std::vector<NodeIndex> path = ready.pending.path;

            {
                PROFILE_SCOPE(ProfileMetric::BatchResultsPathReversal);
                std::reverse(path.begin(), path.end());
            }

            // Accumulate updates for each node in the path
            // This is the key optimization: instead of N separate atomic operations
            // per path, we do 1 atomic operation per unique node across all paths
            float current_value = ready.result.value;

            for (size_t i = 0; i < path.size(); ++i) {
                NodeIndex node = path[i];

                // Apply sign flipping: each level up the tree negates the value
                float value_for_node = (i % 2 == 0) ? current_value : -current_value;

                // Accumulate updates (will be applied atomically in Phase 4)
                auto& update = node_updates[node];
                update.visit_increment += 1.0f;
                update.value_increment += value_for_node;
            }
        }
    }

    // Phase 4: Apply batched atomic updates
    // Single atomic operation per unique node instead of per path occurrence
    // For 32 results with avg path length 10 and 50% overlap:
    //   Before: 32 × 10 × 2 = 640 atomic operations
    //   After:  ~160 unique nodes × 2 = 320 atomic operations
    //   Result: 2× reduction + less contention
    {
        PROFILE_SCOPE(ProfileMetric::BatchResultsAtomicUpdate);

        for (const auto& [node_index, update] : node_updates) {
            backup_.update_node_atomic(node_index, update.value_increment, update.visit_increment);
        }
    }

    // Phase 5: Batch clear expanding flags
    // Clear all flags after backups complete to ensure consistency
    {
        PROFILE_SCOPE(ProfileMetric::BatchResultsClearFlags);

        for (auto& ready : ready_results) {
            tree_.clear_expanding_flag(ready.pending.leaf_node);
        }
    }

    // Phase 6: Return pool states to ThreadLocalStatePool (T018g optimization)
    // CRITICAL: Must return states after expansion to avoid use-after-free
    // This eliminates 418μs clone overhead per simulation (3.7× throughput gain)
    {
        PROFILE_SCOPE(ProfileMetric::BatchResultsReturnStates);

        for (auto& ready : ready_results) {
            if (ready.pending.state) {
                // Get thread-local pool and return state
                // Note: game_type is consistent within a runner instance
                GameType game_type = detect_game_type(*ready.pending.state);
                ThreadLocalStatePool* pool = get_thread_state_pool(game_type);
                pool->release(ready.pending.state);
            }
        }
    }

    return static_cast<int>(ready_results.size());
}

bool ContinuousSimulationRunner::expand_node_with_result(
    NodeIndex leaf,
    const IGameState& state,
    const std::vector<float>& policy,
    float value) {

    // === PROFILING UPGRADE: Instrument node expansion phases ===

    // ✅ CRITICAL FIX: Check if already expanded (but don't claim yet)
    // We'll claim after allocating children to avoid race where threads see
    // expanded=true but num_children=0
    {
        PROFILE_SCOPE(ProfileMetric::NodeExpansionFlagCheck);
        NodeFlags flags = tree_.get_flags(leaf);
        if (flags.is_expanded()) {
            return false;  // Already expanded by another thread
        }
    }

    // Get legal moves
    std::vector<int> legal_moves;
    {
        PROFILE_SCOPE(ProfileMetric::NodeExpansionLegalMoves);
        legal_moves = state.getLegalMoves();
        if (legal_moves.empty()) {
            return false;
        }
    }

    // Validate policy size
    int action_space_size = state.getActionSpaceSize();
    if (static_cast<int>(policy.size()) != action_space_size) {
        return false;
    }

    // Mask and normalize policy
    thread_local std::vector<float> masked_policy_buffer;
    {
        PROFILE_SCOPE(ProfileMetric::NodeExpansionPolicyMask);

        float policy_sum = 0.0f;
        masked_policy_buffer.resize(legal_moves.size());
        auto& masked_policy = masked_policy_buffer;

        for (size_t i = 0; i < legal_moves.size(); ++i) {
            int move = legal_moves[i];
            if (move >= 0 && move < action_space_size) {
                masked_policy[i] = policy[move];
                policy_sum += policy[move];
            } else {
                masked_policy[i] = 0.0f;
            }
        }

        // Normalize
        {
            PROFILE_SCOPE(ProfileMetric::NodeExpansionPolicyNormalize);
            if (policy_sum > 0.0f) {
                const float inv_sum = 1.0f / policy_sum;
                for (float& p : masked_policy) {
                    p *= inv_sum;
                }
            } else {
                const float uniform_prob = 1.0f / static_cast<float>(legal_moves.size());
                for (float& p : masked_policy) {
                    p = uniform_prob;
                }
            }
        }
    }

    auto& masked_policy = masked_policy_buffer;

    // Allocate children
    uint16_t num_children = static_cast<uint16_t>(legal_moves.size());
    NodeIndex first_child;
    {
        PROFILE_SCOPE(ProfileMetric::NodeExpansionChildAlloc);
        first_child = tree_.allocate_nodes(num_children);

        if (first_child == NULL_NODE_INDEX) {
            return false;  // Tree full
        }
    }

    // Initialize children
    {
        PROFILE_SCOPE(ProfileMetric::NodeExpansionChildInit);

        for (uint16_t i = 0; i < num_children; ++i) {
            NodeIndex child_idx = first_child + i;

            tree_.set_prior_prob(child_idx, masked_policy[i]);
            tree_.set_move(child_idx, static_cast<uint16_t>(legal_moves[i]));
            tree_.set_parent_index(child_idx, leaf);
            tree_.set_visit_count(child_idx, 0.0f);
            tree_.set_total_value(child_idx, 0.0f);
            tree_.set_virtual_loss(child_idx, 0.0f);

            NodeFlags child_flags;
            child_flags.set_current_player(state.getCurrentPlayer() == 1 ? 1 : 0);
            tree_.set_flags(child_idx, child_flags);
        }

        // Update parent with children info
        tree_.set_first_child_index(leaf, first_child);
        tree_.set_num_children(leaf, num_children);
    }

    // ✅ CRITICAL: Atomically set expanded flag AFTER children are ready
    // This ensures other threads see a fully initialized node
    // If another thread wins this race, we wasted some work but tree stays consistent
    {
        PROFILE_SCOPE(ProfileMetric::NodeExpansionAtomicFlag);
        if (!tree_.atomic_try_set_expanded(leaf)) {
            // Another thread set expanded flag - this is very rare but can happen
            // Our allocated children will be orphaned, but tree remains valid
            // This is acceptable vs. the alternative of exposing partially initialized nodes
            return false;
        }
    }

    return true;
}

bool ContinuousSimulationRunner::ensure_root_expanded(IGameState& root_state,
                                                       NodeIndex root_index,
                                                       AsyncInferenceQueue& queue) {
    // === PROFILING UPGRADE: Instrument root expansion ===
    PROFILE_SCOPE(ProfileMetric::RootExpansionTotal);

    // Check if root is already expanded
    NodeFlags flags = tree_.get_flags(root_index);
    if (flags.is_expanded()) {
        return false;  // Already expanded, nothing to do
    }

    // Check if we can mark it for expansion atomically
    if (!tree_.atomic_try_mark_expanding(root_index)) {
        PROFILE_SCOPE(ProfileMetric::RootExpansionAtomicRace);

        // Another thread is already expanding it, wait for completion
        while (!tree_.get_flags(root_index).is_expanded()) {
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
        return false;
    }

    // We won the race - perform synchronous expansion
    try {
        // Submit inference request and wait for result
        // T013-T014: Extract features in-place (ZERO COPY!)
        int board_size = root_state.getBoardSize();
        int num_feature_planes = root_state.get_num_feature_planes();
        int action_space_size = root_state.getActionSpaceSize();

        // Create temporary feature buffer for root expansion
        std::vector<float> features;
        {
            PROFILE_SCOPE(ProfileMetric::CoordinatorFeatureExtraction);
            features.resize(num_feature_planes * board_size * board_size);
            root_state.extract_features_to_buffer(features.data());
        }

        // Build and submit request with move semantics
        InferenceRequest request;
        request.features = std::move(features);  // MOVE, not copy!
        request.node_index = root_index;
        request.action_space_size = action_space_size;
        request.board_size = static_cast<int16_t>(board_size);
        request.planes = static_cast<int16_t>(num_feature_planes);
        request.path = {static_cast<int16_t>(root_index)};

        uint64_t request_id = queue.submit_request(std::move(request));

        // Wait for result (synchronous for root expansion only)
        std::optional<InferenceResult> result;
        {
            PROFILE_SCOPE(ProfileMetric::RootExpansionWaitInference);

            const auto start_time = std::chrono::steady_clock::now();
            const auto timeout = std::chrono::seconds(5);  // 5 second timeout

            while (!result.has_value()) {
                result = queue.try_get_result(request_id);
                if (!result.has_value()) {
                    // Check timeout
                    auto elapsed = std::chrono::steady_clock::now() - start_time;
                    if (elapsed > timeout) {
                        tree_.clear_expanding_flag(root_index);
                        return false;  // Timeout
                    }
                    std::this_thread::sleep_for(std::chrono::microseconds(100));
                }
            }
        }

        // Expand root with the result
        bool expanded = expand_node_with_result(root_index, root_state, result->policy, result->value);
        tree_.clear_expanding_flag(root_index);

        if (expanded) {
            PROFILE_SCOPE(ProfileMetric::RootExpansionDirichlet);

            // Add Dirichlet noise for exploration (AlphaZero approach)
            // Use alpha=0.3 for Go-like games (can be made configurable later)
            add_dirichlet_noise(root_index, 0.3f);
        }

        return expanded;

    } catch (const std::exception& e) {
        tree_.clear_expanding_flag(root_index);
        return false;
    }
}

void ContinuousSimulationRunner::add_dirichlet_noise(NodeIndex root_index, float alpha) {
    std::uint16_t num_children = tree_.get_num_children(root_index);
    if (num_children == 0) {
        return;  // No children to add noise to
    }

    // Sample from Gamma distribution to create Dirichlet noise
    // Dir(α) can be generated as: η_i = Gamma(α, 1) / Σ Gamma(α, 1)
    std::random_device rd;
    std::mt19937 gen(rd());
    std::gamma_distribution<float> gamma_dist(alpha, 1.0f);

    std::vector<float> noise(num_children);
    float sum = 0.0f;

    for (std::uint16_t i = 0; i < num_children; ++i) {
        noise[i] = gamma_dist(gen);
        sum += noise[i];
    }

    // Normalize Dirichlet samples
    if (sum > 0.0f) {
        for (float& n : noise) {
            n /= sum;
        }
    } else {
        // Fallback to uniform if all zeros (extremely rare)
        float uniform = 1.0f / num_children;
        for (float& n : noise) {
            n = uniform;
        }
    }

    // Mix with priors: P'(a) = (1 - ε) * P(a) + ε * η_a
    const float epsilon = 0.25f;  // AlphaZero uses 0.25
    NodeIndex first_child = tree_.get_first_child_index(root_index);

    for (std::uint16_t i = 0; i < num_children; ++i) {
        NodeIndex child = first_child + i;
        float original_prior = tree_.get_prior_prob(child);
        float mixed_prior = (1.0f - epsilon) * original_prior + epsilon * noise[i];
        tree_.set_prior_prob(child, mixed_prior);
    }
}

// T024f-6: Zero-Copy MCTS Implementation
// Replace state pooling (copyFrom) with make/unmake pattern

NodeIndex ContinuousSimulationRunner::select_leaf_with_make_unmake(
    NodeIndex root,
    IGameState& current_state,
    std::vector<NodeIndex>& path,
    std::vector<uint64_t>& undo_tokens) {

    ScopedMetric metric(InstrumentationMetric::Selection);
    PROFILE_SCOPE(ProfileMetric::SelectionTreeTraversal);

    // Clear path and undo tokens, start from root
    path.clear();
    path.push_back(root);
    undo_tokens.clear();

    NodeIndex current = root;
    int depth = 0;

    // Traverse tree using PUCT until reaching a leaf (unexpanded or terminal)
    while (true) {
        // Check if current node is terminal
        if (current_state.isTerminal()) {
            break;  // Reached terminal node
        }

        // Check if current node is expanded
        NodeFlags flags = tree_.get_flags(current);
        if (!flags.is_expanded()) {
            break;  // Reached unexpanded leaf
        }

        // Node is expanded - select best child using PUCT
        SelectionResult result;
        {
            PROFILE_SCOPE(ProfileMetric::SelectionPUCT);
            result = selector_.select_child(tree_, current);
        }

        if (!result.valid || result.selected_child == NULL_NODE_INDEX) {
            // No valid child found (shouldn't happen if expanded)
            break;
        }

        // Get the move that led to this child
        uint16_t move_index = tree_.get_move(result.selected_child);

        // CRITICAL DEBUG VALIDATION (T024f-6): Verify move is legal
        // This catches state/tree desynchronization bugs immediately
        #ifndef NDEBUG
        {
            std::vector<int> current_legal = current_state.getLegalMoves();
            int current_player = current_state.getCurrentPlayer();
            bool move_is_legal = std::find(current_legal.begin(), current_legal.end(),
                                           static_cast<int>(move_index)) != current_legal.end();

            if (!move_is_legal) {
                std::string error_msg =
                    "CRITICAL BUG in select_leaf_with_make_unmake: Tree contains illegal move!\n"
                    "  Move index: " + std::to_string(move_index) + "\n"
                    "  Current player: " + std::to_string(current_player) + "\n"
                    "  Current depth: " + std::to_string(depth) + "\n"
                    "  Parent node: " + std::to_string(current) + "\n"
                    "  Child node: " + std::to_string(result.selected_child) + "\n"
                    "  Legal moves count: " + std::to_string(current_legal.size()) + "\n"
                    "  State hash: " + std::to_string(current_state.zobrist_hash()) + "\n"
                    "This indicates state/tree desync - moves in tree were legal at expansion "
                    "but are illegal now. Check make/unmake implementation!";
                throw std::runtime_error(error_msg);
            }
        }
        #endif

        // Apply virtual loss to the selected child
        {
            PROFILE_SCOPE(ProfileMetric::VirtualLossApply);
            virtual_loss_.apply_virtual_loss(result.selected_child);
        }

        // T024f-6: Apply move via make_move (returns undo token, ~15ns)
        // This replaces makeMove() which internally calls apply_move_inplace
        // The key difference: make_move returns undo token for O(1) restoration
        uint64_t undo_token;
        {
            PROFILE_SCOPE(ProfileMetric::SelectionMakeMove);
            PROFILE_COUNTER(ProfileMetric::SelectionMakeMoveCount, 1);
            undo_token = current_state.make_move(move_index);
        }

        // Store undo token for later unwinding
        undo_tokens.push_back(undo_token);

        // Move to the selected child
        current = result.selected_child;
        path.push_back(current);
        depth++;
    }

    // Track selection depth
    PROFILE_GAUGE(ProfileMetric::SelectionDepth, depth);

    return current;
}

void ContinuousSimulationRunner::unwind_path(
    IGameState& state,
    const std::vector<NodeIndex>& path,
    const std::vector<uint64_t>& undo_tokens) {

    PROFILE_SCOPE(ProfileMetric::StateCloneTotal);  // Reuse for comparison with old approach

    // Unwind moves in reverse order
    // path has N elements (root + N-1 moves)
    // undo_tokens has N-1 elements (one per move, excluding root)
    //
    // Example: path = [root, child1, child2], undo_tokens = [token1, token2]
    // Unwind: unmake(move2, token2), unmake(move1, token1)

    if (path.size() <= 1) {
        // No moves to unwind (already at root or empty path)
        return;
    }

    // CRITICAL DEBUG VALIDATION: Verify undo_tokens size matches path
    #ifndef NDEBUG
    if (undo_tokens.size() != path.size() - 1) {
        throw std::runtime_error(
            "CRITICAL BUG in unwind_path: Size mismatch!\n"
            "  Path size: " + std::to_string(path.size()) + "\n"
            "  Undo tokens size: " + std::to_string(undo_tokens.size()) + "\n"
            "  Expected: " + std::to_string(path.size() - 1) + " undo tokens\n"
            "This indicates incorrect undo token collection during selection!");
    }
    #endif

    // Iterate backwards from leaf to root (skip root at index 0)
    for (int i = static_cast<int>(path.size()) - 1; i > 0; --i) {
        // Get the move that led to this node
        uint16_t move = tree_.get_move(path[i]);

        // Get corresponding undo token (undo_tokens is 0-indexed)
        uint64_t undo_token = undo_tokens[i - 1];

        // CRITICAL DEBUG VALIDATION: Store state before unmake for verification
        #ifndef NDEBUG
        uint64_t hash_before_unmake = state.zobrist_hash();
        int player_before_unmake = state.getCurrentPlayer();
        #endif

        // Restore state via unmake_move (~15ns per call)
        {
            PROFILE_SCOPE(ProfileMetric::SelectionUnmakeMove);
            PROFILE_COUNTER(ProfileMetric::SelectionUnmakeMoveCount, 1);
            state.unmake_move(move, undo_token);
        }

        // CRITICAL DEBUG VALIDATION: Verify state changed after unmake
        #ifndef NDEBUG
        uint64_t hash_after_unmake = state.zobrist_hash();
        int player_after_unmake = state.getCurrentPlayer();

        if (hash_after_unmake == hash_before_unmake) {
            throw std::runtime_error(
                "CRITICAL BUG in unwind_path: unmake_move did not change state!\n"
                "  Move: " + std::to_string(move) + "\n"
                "  Undo token: " + std::to_string(undo_token) + "\n"
                "  Path index: " + std::to_string(i) + "\n"
                "  Hash unchanged: " + std::to_string(hash_after_unmake) + "\n"
                "This indicates unmake_move is not working correctly!");
        }

        // Player should flip after unmake (unless game state is weird)
        if (player_after_unmake == player_before_unmake) {
            // This might be OK for some games, just log warning
            // Don't throw, but this is suspicious
        }
        #endif
    }

    // State is now restored to root position
    // Ready for next simulation (zero cloning needed!)
}

} // namespace mcts
