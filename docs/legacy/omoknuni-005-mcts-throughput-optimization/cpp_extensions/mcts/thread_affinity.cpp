/**
 * @file thread_affinity.cpp
 * @brief Implementation of thread affinity management for multi-CCD CPUs
 */

#include "thread_affinity.hpp"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <thread>

#ifdef __linux__
#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#endif

namespace mcts {

ThreadAffinityManager::ThreadAffinityManager() {
    detect_topology();
}

void ThreadAffinityManager::detect_topology() {
#ifdef __linux__
    std::ifstream cpuinfo("/proc/cpuinfo");
    if (!cpuinfo.is_open()) {
        detect_generic_topology();
        return;
    }

    std::string line;
    bool found_ryzen_5900x = false;

    // Detect CPU model
    while (std::getline(cpuinfo, line)) {
        if (line.find("model name") != std::string::npos) {
            if (line.find("5900X") != std::string::npos ||
                line.find("5900") != std::string::npos) {
                found_ryzen_5900x = true;
                topology_.is_ryzen_5900x = true;
                topology_.is_multi_ccd = true;
                break;
            }
        }
    }

    if (found_ryzen_5900x) {
        setup_ryzen_5900x_topology();
    } else {
        detect_generic_topology();
    }
#else
    detect_generic_topology();
#endif
}

void ThreadAffinityManager::setup_ryzen_5900x_topology() {
    // Ryzen 5900X specific topology
    // 12 physical cores (2 CCDs × 6 cores)
    // 24 logical cores with SMT

    topology_.total_physical_cores = 12;
    topology_.total_logical_cores = 24;

    // CCD0: Physical cores 0-5
    topology_.ccd0_cores = {0, 1, 2, 3, 4, 5};

    // CCD1: Physical cores 6-11
    topology_.ccd1_cores = {6, 7, 8, 9, 10, 11};

    // SMT siblings: Logical cores 12-23
    // 12-17 are siblings of CCD0 (0-5)
    // 18-23 are siblings of CCD1 (6-11)
    topology_.smt_siblings = {12, 13, 14, 15, 16, 17,
                              18, 19, 20, 21, 22, 23};
}

void ThreadAffinityManager::detect_generic_topology() {
    // Generic topology detection using hardware_concurrency
    int hw_threads = std::thread::hardware_concurrency();

    if (hw_threads == 0) {
        // Unable to detect, use conservative defaults
        topology_.total_physical_cores = 4;
        topology_.total_logical_cores = 8;
    } else {
        topology_.total_logical_cores = hw_threads;

        // Assume SMT enabled (common for modern CPUs)
        topology_.total_physical_cores = hw_threads / 2;

        // Generic mapping: first half physical, second half SMT
        for (int i = 0; i < topology_.total_physical_cores; ++i) {
            if (i < topology_.total_physical_cores / 2) {
                topology_.ccd0_cores.push_back(i);
            } else {
                topology_.ccd1_cores.push_back(i);
            }
        }

        for (int i = topology_.total_physical_cores;
             i < topology_.total_logical_cores; ++i) {
            topology_.smt_siblings.push_back(i);
        }
    }

    topology_.is_ryzen_5900x = false;
    topology_.is_multi_ccd = (topology_.total_physical_cores >= 8);
}

bool ThreadAffinityManager::set_thread_affinity(int thread_id, int total_threads) {
#ifdef __linux__
    if (!topology_.is_ryzen_5900x && !topology_.is_multi_ccd) {
        // No optimization for unknown/single-CCD CPUs
        return false;
    }

    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);

    if (topology_.is_ryzen_5900x) {
        // Ryzen 5900X optimized placement
        if (total_threads <= 6) {
            // Use only CCD0 for best cache locality
            int core_id = topology_.ccd0_cores[thread_id % 6];
            CPU_SET(core_id, &cpuset);
        }
        else if (total_threads <= 12) {
            // Use both CCDs, physical cores only
            if (thread_id < 6) {
                int core_id = topology_.ccd0_cores[thread_id];
                CPU_SET(core_id, &cpuset);
            } else {
                int core_id = topology_.ccd1_cores[thread_id - 6];
                CPU_SET(core_id, &cpuset);
            }
        }
        else {
            // Use SMT siblings for >12 threads
            if (thread_id < 12) {
                // Physical cores
                int core_id = (thread_id < 6) ?
                    topology_.ccd0_cores[thread_id] :
                    topology_.ccd1_cores[thread_id - 6];
                CPU_SET(core_id, &cpuset);
            } else {
                // SMT siblings
                int sibling_idx = thread_id - 12;
                if (sibling_idx < static_cast<int>(topology_.smt_siblings.size())) {
                    int core_id = topology_.smt_siblings[sibling_idx];
                    CPU_SET(core_id, &cpuset);
                }
            }
        }
    }
    else {
        // Generic multi-core placement
        int core_id = thread_id % topology_.total_physical_cores;
        if (core_id < static_cast<int>(topology_.ccd0_cores.size())) {
            CPU_SET(topology_.ccd0_cores[core_id], &cpuset);
        } else {
            int ccd1_idx = core_id - topology_.ccd0_cores.size();
            if (ccd1_idx < static_cast<int>(topology_.ccd1_cores.size())) {
                CPU_SET(topology_.ccd1_cores[ccd1_idx], &cpuset);
            }
        }
    }

    // Set affinity for calling thread
    int result = pthread_setaffinity_np(pthread_self(),
                                        sizeof(cpu_set_t),
                                        &cpuset);
    return (result == 0);
#else
    // Thread affinity not supported on non-Linux platforms
    (void)thread_id;
    (void)total_threads;
    return false;
#endif
}

bool ThreadAffinityManager::is_supported() const {
#ifdef __linux__
    return true;
#else
    return false;
#endif
}

int ThreadAffinityManager::get_recommended_thread_count() const {
    if (topology_.is_ryzen_5900x) {
        // Ryzen 5900X: Optimal is 12 physical cores
        return 12;
    }

    if (topology_.is_multi_ccd) {
        // Generic multi-CCD: Use all physical cores
        return topology_.total_physical_cores;
    }

    // Conservative default: Use half of logical cores
    return std::max(4, topology_.total_logical_cores / 2);
}

} // namespace mcts
