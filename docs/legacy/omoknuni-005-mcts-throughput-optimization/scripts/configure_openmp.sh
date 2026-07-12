#!/bin/bash
#
# OpenMP Runtime Configuration for MCTS Throughput Recovery
#
# This script configures OpenMP environment variables for optimal performance
# on the Ryzen 9 5900X (12 cores, 24 threads, dual-CCD architecture).
#
# Usage:
#     source scripts/configure_openmp.sh
#

echo "========================================================================"
echo "Configuring OpenMP for MCTS Throughput Recovery"
echo "========================================================================"

# OMP_NUM_THREADS: Use all 12 physical cores
# Ryzen 5900X has 12 physical cores (6 per CCD)
# Using physical cores only (no hyperthreads) for consistent performance
export OMP_NUM_THREADS=12
echo "✓ OMP_NUM_THREADS=12 (all physical cores)"

# OMP_PROC_BIND: Pin threads to cores
# 'close' = pin to nearby cores (better for cache locality)
# 'spread' = distribute across cores (better for memory bandwidth)
# We use 'close' as feature extraction is cache-sensitive
export OMP_PROC_BIND=close
echo "✓ OMP_PROC_BIND=close (pin to nearby cores for cache locality)"

# OMP_PLACES: Define what constitutes a "place"
# 'cores' = one thread per physical core (not hyperthread)
export OMP_PLACES=cores
echo "✓ OMP_PLACES=cores (use physical cores, not hyperthreads)"

# OMP_NESTED: Disable nested parallelism
# CRITICAL: Must be FALSE to prevent MCTS threads + OpenMP threads conflict
export OMP_NESTED=FALSE
echo "✓ OMP_NESTED=FALSE (prevent nested parallelism conflicts)"

# OMP_WAIT_POLICY: Active wait for low latency
# 'ACTIVE' = busy-wait (low latency, high CPU usage)
# 'PASSIVE' = yield CPU (higher latency, lower CPU usage)
# We use 'ACTIVE' for MCTS performance (latency-sensitive)
export OMP_WAIT_POLICY=ACTIVE
echo "✓ OMP_WAIT_POLICY=ACTIVE (low latency, busy-wait)"

# OMP_SCHEDULE: Dynamic scheduling for load balancing
# Default is 'static' which is fine for uniform workloads
# We keep default (no need to set explicitly)
# export OMP_SCHEDULE=static
# echo "✓ OMP_SCHEDULE=static (default, uniform workload)"

echo ""
echo "OpenMP configuration complete. Environment variables set:"
echo "  OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "  OMP_PROC_BIND=$OMP_PROC_BIND"
echo "  OMP_PLACES=$OMP_PLACES"
echo "  OMP_NESTED=$OMP_NESTED"
echo "  OMP_WAIT_POLICY=$OMP_WAIT_POLICY"
echo ""
echo "Verify with:"
echo "  python -c 'import os; print({k:v for k,v in os.environ.items() if k.startswith(\"OMP_\")})'"
echo "========================================================================"
