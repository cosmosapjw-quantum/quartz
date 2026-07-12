/**
 * @file hardware_counters.hpp
 * @brief Hardware performance counter integration (perf_event_open, VTune ITT)
 *
 * Provides access to CPU performance monitoring unit (PMU) counters:
 * - Linux: perf_event_open syscall
 * - Intel VTune: ITT API integration
 * - AMD: uProf integration (future)
 */

#pragma once

#include <cstdint>
#include <array>
#include <string>
#include <vector>

namespace mcts {
namespace profiling {

/**
 * @brief Hardware counter types
 */
enum class HWCounterType : std::uint8_t {
    CPUCycles,              // Total CPU cycles
    Instructions,           // Instructions executed
    CacheReferences,        // Cache references
    CacheMisses,            // Cache misses
    BranchInstructions,     // Branch instructions
    BranchMisses,           // Branch mispredictions
    L1DCacheMisses,         // L1 data cache misses
    L1ICacheMisses,         // L1 instruction cache misses
    L2CacheMisses,          // L2 cache misses
    LLCMisses,              // Last-level cache misses
    TLBMisses,              // TLB misses
    PageFaults,             // Page faults
    ContextSwitches,        // Context switches
    StalledCyclesFrontend,  // Frontend stalled cycles
    StalledCyclesBackend,   // Backend stalled cycles
    RefCycles,              // Reference cycles (unaffected by frequency scaling)

    Count
};

/**
 * @brief Hardware counter reader (Linux perf_event_open)
 */
class HardwareCounterReader {
public:
    HardwareCounterReader();
    ~HardwareCounterReader();

    /**
     * @brief Initialize hardware counters
     *
     * @return true if successful, false if not supported
     */
    bool initialize();

    /**
     * @brief Check if counters are available
     */
    bool is_available() const { return initialized_; }

    /**
     * @brief Start counting
     */
    void start();

    /**
     * @brief Stop counting
     */
    void stop();

    /**
     * @brief Reset all counters
     */
    void reset();

    /**
     * @brief Read counter value
     *
     * @param counter Counter type to read
     * @return Counter value (cumulative since start)
     */
    std::uint64_t read(HWCounterType counter);

    /**
     * @brief Read all counters
     *
     * @return Array of counter values
     */
    std::array<std::uint64_t, static_cast<std::size_t>(HWCounterType::Count)> read_all();

    /**
     * @brief Get counter name
     */
    static const char* counter_name(HWCounterType counter);

    /**
     * @brief Get error message if initialization failed
     */
    const std::string& get_error() const { return error_message_; }

private:
    bool initialized_;
    std::string error_message_;

#ifdef __linux__
    // perf_event_open file descriptors
    static constexpr std::size_t MAX_COUNTERS = static_cast<std::size_t>(HWCounterType::Count);
    std::array<int, MAX_COUNTERS> perf_fds_;

    /**
     * @brief Open perf_event_open for a counter
     */
    int open_perf_event(std::uint32_t type, std::uint64_t config);
#endif
};

#ifdef USE_VTUNE
/**
 * @brief Intel VTune ITT API integration
 *
 * Provides task markers for VTune profiler visualization.
 */
class VTuneProfiler {
public:
    /**
     * @brief Initialize VTune profiler
     */
    static bool initialize();

    /**
     * @brief Create domain for tasks
     */
    static void create_domain(const char* name);

    /**
     * @brief Begin task (appears in VTune timeline)
     */
    static void begin_task(const char* name);

    /**
     * @brief End current task
     */
    static void end_task();

    /**
     * @brief Create string handle (for repeated task names)
     */
    static void* create_string_handle(const char* name);

    /**
     * @brief Begin task with pre-created handle (faster)
     */
    static void begin_task_handle(void* handle);

    /**
     * @brief Add metadata to task
     */
    static void add_metadata(const char* key, const char* value);

    /**
     * @brief Check if VTune is attached
     */
    static bool is_attached();

private:
    static void* domain_;
    static bool initialized_;
};

// Convenience macros for VTune
#define VTUNE_TASK(name) \
    ::mcts::profiling::VTuneProfiler::begin_task(name); \
    struct __vtune_task_raii { \
        ~__vtune_task_raii() { ::mcts::profiling::VTuneProfiler::end_task(); } \
    } __vtune_task_instance;

#else
#define VTUNE_TASK(name) do {} while(0)
#endif

/**
 * @brief Detect CPU frequency (GHz)
 *
 * Reads from /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq
 * or uses RDTSC calibration.
 */
double detect_cpu_frequency();

/**
 * @brief Convert CPU cycles to nanoseconds
 */
inline std::uint64_t cycles_to_ns(std::uint64_t cycles, double cpu_ghz) {
    return static_cast<std::uint64_t>(cycles / cpu_ghz);
}

/**
 * @brief Convert nanoseconds to CPU cycles
 */
inline std::uint64_t ns_to_cycles(std::uint64_t ns, double cpu_ghz) {
    return static_cast<std::uint64_t>(ns * cpu_ghz);
}

/**
 * @brief Hardware counter group (for reading multiple counters atomically)
 *
 * Uses perf_event_open group leader feature to read multiple
 * counters with a single syscall.
 */
class HardwareCounterGroup {
public:
    HardwareCounterGroup();
    ~HardwareCounterGroup();

    /**
     * @brief Add counter to group
     */
    bool add_counter(HWCounterType counter);

    /**
     * @brief Initialize group
     */
    bool initialize();

    /**
     * @brief Start counting
     */
    void start();

    /**
     * @brief Stop counting
     */
    void stop();

    /**
     * @brief Read all counters in group (single syscall)
     */
    std::vector<std::uint64_t> read_all();

    /**
     * @brief Get counter count in group
     */
    std::size_t size() const { return counter_types_.size(); }

private:
    std::vector<HWCounterType> counter_types_;
    int group_leader_fd_;
    std::vector<int> counter_fds_;
    bool initialized_;
};

/**
 * @brief PEBS (Precise Event-Based Sampling) support
 *
 * Provides precise instruction-level profiling on Intel CPUs.
 */
class PEBSProfiler {
public:
    /**
     * @brief Check if PEBS is supported
     */
    static bool is_supported();

    /**
     * @brief Enable PEBS for cache misses
     */
    bool enable_cache_miss_sampling(std::uint64_t sample_period);

    /**
     * @brief Enable PEBS for branch mispredictions
     */
    bool enable_branch_miss_sampling(std::uint64_t sample_period);

    /**
     * @brief Read PEBS samples
     */
    std::vector<std::uint64_t> read_samples();
};

/**
 * @brief CPU topology information
 */
struct CPUTopology {
    std::uint32_t num_cores;
    std::uint32_t num_threads;
    std::uint32_t numa_nodes;
    std::uint32_t l1d_cache_size;
    std::uint32_t l1i_cache_size;
    std::uint32_t l2_cache_size;
    std::uint32_t l3_cache_size;
    std::uint32_t cache_line_size;
};

/**
 * @brief Detect CPU topology
 */
CPUTopology detect_cpu_topology();

/**
 * @brief Memory bandwidth measurement
 */
class MemoryBandwidthMonitor {
public:
    /**
     * @brief Start monitoring memory bandwidth
     */
    bool start();

    /**
     * @brief Stop monitoring
     */
    void stop();

    /**
     * @brief Get memory read bandwidth (GB/s)
     */
    double get_read_bandwidth();

    /**
     * @brief Get memory write bandwidth (GB/s)
     */
    double get_write_bandwidth();

    /**
     * @brief Get total memory bandwidth (GB/s)
     */
    double get_total_bandwidth();

private:
    std::uint64_t start_time_ns_;
    std::uint64_t bytes_read_;
    std::uint64_t bytes_written_;
    bool running_;
};

} // namespace profiling
} // namespace mcts
