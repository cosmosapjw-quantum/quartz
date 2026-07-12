#!/bin/bash
################################################################################
# Comprehensive MCTS Profiling Script
################################################################################
# This script performs complete profiling of the MCTS system including:
# - C++ backend profiling
# - Python coordination profiling
# - GPU inference profiling
# - Thread contention analysis
# - Memory profiling
# - Combined analysis and reporting
################################################################################

set -e  # Exit on error

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PROFILING_DIR="${PROJECT_ROOT}/profiling_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SESSION_DIR="${PROFILING_DIR}/session_${TIMESTAMP}"
BUILD_DIR="${PROJECT_ROOT}/build_profiling"

# Default parameters
GAME_TYPE="gomoku"
NUM_THREADS=4
NUM_SIMULATIONS=800
BATCH_SIZE=32
TIMEOUT_MS=2.0
RUN_CPP_PROFILING=1
RUN_PYTHON_PROFILING=1
RUN_GPU_PROFILING=1
RUN_MEMORY_PROFILING=1
COMPILE_ONLY=0
CLEAN_BUILD=0

# Function to print colored messages
print_header() {
    echo -e "\n${BLUE}=================================================================================${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}=================================================================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ $1${NC}"
}

# Parse command line arguments
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Comprehensive MCTS profiling script that analyzes all system components.

OPTIONS:
    -g, --game GAME          Game type: gomoku, chess, go (default: gomoku)
    -t, --threads NUM        Number of threads (default: 4)
    -s, --simulations NUM    Number of simulations (default: 800)
    -b, --batch-size NUM     GPU batch size (default: 32)
    --timeout MS             Batch timeout in ms (default: 2.0)

    --skip-cpp               Skip C++ profiling
    --skip-python            Skip Python profiling
    --skip-gpu               Skip GPU profiling
    --skip-memory            Skip memory profiling

    --compile-only           Only compile C++ code, don't run profiling
    --clean                  Clean build before compiling

    -h, --help               Show this help message

EXAMPLES:
    # Run complete profiling with defaults
    $0

    # Profile chess with 8 threads
    $0 --game chess --threads 8

    # Only compile C++ profiling code
    $0 --compile-only

    # Skip GPU profiling (for CPU-only systems)
    $0 --skip-gpu

EOF
    exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -g|--game)
            GAME_TYPE="$2"
            shift 2
            ;;
        -t|--threads)
            NUM_THREADS="$2"
            shift 2
            ;;
        -s|--simulations)
            NUM_SIMULATIONS="$2"
            shift 2
            ;;
        -b|--batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --timeout)
            TIMEOUT_MS="$2"
            shift 2
            ;;
        --skip-cpp)
            RUN_CPP_PROFILING=0
            shift
            ;;
        --skip-python)
            RUN_PYTHON_PROFILING=0
            shift
            ;;
        --skip-gpu)
            RUN_GPU_PROFILING=0
            shift
            ;;
        --skip-memory)
            RUN_MEMORY_PROFILING=0
            shift
            ;;
        --compile-only)
            COMPILE_ONLY=1
            shift
            ;;
        --clean)
            CLEAN_BUILD=1
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Create output directory
mkdir -p "${SESSION_DIR}"

# Log file for this session
LOG_FILE="${SESSION_DIR}/profiling.log"
exec 2>&1 | tee "${LOG_FILE}"

print_header "MCTS COMPREHENSIVE PROFILING"
echo "Session ID: ${TIMESTAMP}"
echo "Output Directory: ${SESSION_DIR}"
echo ""
echo "Configuration:"
echo "  Game Type: ${GAME_TYPE}"
echo "  Threads: ${NUM_THREADS}"
echo "  Simulations: ${NUM_SIMULATIONS}"
echo "  Batch Size: ${BATCH_SIZE}"
echo "  Timeout: ${TIMEOUT_MS}ms"
echo ""

################################################################################
# STEP 1: Compile C++ Profiling Code
################################################################################
print_header "STEP 1: Building C++ Profiling System"

# Clean build if requested
if [[ ${CLEAN_BUILD} -eq 1 ]]; then
    print_info "Cleaning previous build..."
    rm -rf "${BUILD_DIR}"
fi

# Create build directory
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

# Configure with CMake
print_info "Configuring CMake..."
cmake "${PROJECT_ROOT}/cpp_extensions/mcts/profiling" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_FLAGS="-O3 -march=native -fopenmp" \
    2>&1 | tail -5

# Build
print_info "Building C++ profiling code..."
make -j$(nproc) 2>&1 | tail -10

if [[ -f "profiling_test" ]]; then
    print_success "C++ profiling system built successfully"
else
    print_error "Failed to build C++ profiling system"
    exit 1
fi

# If compile-only mode, exit here
if [[ ${COMPILE_ONLY} -eq 1 ]]; then
    print_success "Compilation complete (--compile-only mode)"
    exit 0
fi

################################################################################
# STEP 2: Run C++ Profiling
################################################################################
if [[ ${RUN_CPP_PROFILING} -eq 1 ]]; then
    print_header "STEP 2: Running C++ Backend Profiling"

    cd "${BUILD_DIR}"

    print_info "Running C++ profiling test..."
    ./profiling_test > "${SESSION_DIR}/cpp_profiling_output.txt" 2>&1

    # Move generated reports to session directory
    if [[ -f "mcts_profile.json" ]]; then
        mv mcts_profile.json "${SESSION_DIR}/cpp_profile.json"
        print_success "C++ profile saved: cpp_profile.json"
    fi

    if [[ -f "mcts_trace.json" ]]; then
        mv mcts_trace.json "${SESSION_DIR}/cpp_trace.json"
        print_success "Chrome trace saved: cpp_trace.json"
    fi

    if [[ -f "mcts_profile.md" ]]; then
        mv mcts_profile.md "${SESSION_DIR}/cpp_profile.md"
        print_success "Markdown report saved: cpp_profile.md"
    fi
else
    print_info "Skipping C++ profiling (--skip-cpp)"
fi

################################################################################
# STEP 3: Build Python Extensions with Profiling
################################################################################
print_header "STEP 3: Building Python Extensions with Profiling"

cd "${PROJECT_ROOT}"

# Set environment for profiling build
export CFLAGS="-O3 -march=native -fopenmp -DPROFILE_LEVEL=2"
export CXXFLAGS="-O3 -march=native -fopenmp -DPROFILE_LEVEL=2"

print_info "Installing Python package with profiling enabled..."
pip install -e . --force-reinstall --no-deps 2>&1 | tail -5

print_success "Python extensions built with profiling"

################################################################################
# STEP 4: Run Python Profiling
################################################################################
if [[ ${RUN_PYTHON_PROFILING} -eq 1 ]]; then
    print_header "STEP 4: Running Python Coordination Profiling"

    cd "${PROJECT_ROOT}"

    print_info "Running comprehensive Python profiler..."
    python scripts/comprehensive_mcts_profiler.py \
        --game "${GAME_TYPE}" \
        --threads "${NUM_THREADS}" \
        --simulations "${NUM_SIMULATIONS}" \
        --batch-size "${BATCH_SIZE}" \
        --timeout-ms "${TIMEOUT_MS}" \
        --output-dir "${SESSION_DIR}/python_profiling" \
        2>&1 | tee "${SESSION_DIR}/python_profiling.log"

    if [[ -d "${SESSION_DIR}/python_profiling" ]]; then
        print_success "Python profiling complete"
    else
        print_error "Python profiling failed"
    fi
else
    print_info "Skipping Python profiling (--skip-python)"
fi

################################################################################
# STEP 5: GPU Profiling
################################################################################
if [[ ${RUN_GPU_PROFILING} -eq 1 ]]; then
    print_header "STEP 5: Running GPU Inference Profiling"

    # Check if CUDA is available
    if command -v nvidia-smi &> /dev/null; then
        print_info "GPU detected, running GPU profiling..."

        cd "${PROJECT_ROOT}"

        # Run GPU profiling
        python -c "
import sys
sys.path.insert(0, '.')
from src.telemetry.gpu_profiler import GPUProfiler
from src.neural.inference_worker import GPUInferenceWorker
import numpy as np
import json

# Create GPU worker
worker = GPUInferenceWorker(
    model_path=None,
    device='cuda:0',
    batch_size=${BATCH_SIZE},
    timeout_ms=${TIMEOUT_MS}
)

# Warmup
if '${GAME_TYPE}' == 'gomoku':
    shape = (36, 15, 15)
elif '${GAME_TYPE}' == 'chess':
    shape = (30, 8, 8)
else:
    shape = (25, 19, 19)

worker.warmup(shape)

# Profile
with GPUProfiler(device='cuda:0') as profiler:
    positions = [np.random.randn(*shape).astype(np.float32) for _ in range(${BATCH_SIZE})]

    for i in range(100):
        with profiler.profile_batch(len(positions)):
            policies, values = worker.batch_inference(positions)

# Get metrics
from dataclasses import asdict
gpu_report = profiler.generate_report()
metrics = asdict(gpu_report) if gpu_report else {}

# Save results
with open('${SESSION_DIR}/gpu_profile.json', 'w') as f:
    json.dump(metrics, f, indent=2, default=str)

print(f'GPU Profiling Results:')
print(f'  Avg GPU Utilization: {metrics.get(\"avg_gpu_utilization\", 0):.1f}%')
print(f'  Avg Memory Usage: {metrics.get(\"avg_memory_used_mb\", 0):.2f}MB')
print(f'  Avg Power: {metrics.get(\"avg_power_draw_w\", 0):.1f}W')
" 2>&1 | tee "${SESSION_DIR}/gpu_profiling.log"

        if [[ -f "${SESSION_DIR}/gpu_profile.json" ]]; then
            print_success "GPU profiling complete"
        fi
    else
        print_info "No GPU detected, skipping GPU profiling"
    fi
else
    print_info "Skipping GPU profiling (--skip-gpu)"
fi

################################################################################
# STEP 6: Memory Profiling
################################################################################
if [[ ${RUN_MEMORY_PROFILING} -eq 1 ]]; then
    print_header "STEP 6: Running Memory Profiling"

    cd "${PROJECT_ROOT}"

    print_info "Running memory profiling with tracemalloc..."

    python -c "
import tracemalloc
import gc
import sys
sys.path.insert(0, '.')
from src.core.mcts import AlphaZeroMCTS
from src.games.game_state import create_game_state
import json

tracemalloc.start()
gc.collect()

# Create game and MCTS
root_state = create_game_state('${GAME_TYPE}')

# Mock inference function
def mock_inference(state):
    import numpy as np
    from concurrent.futures import Future
    future = Future()
    if '${GAME_TYPE}' == 'gomoku':
        policy = np.ones(225) / 225
    elif '${GAME_TYPE}' == 'chess':
        policy = np.ones(4096) / 4096
    else:
        policy = np.ones(361) / 361
    future.set_result((policy, 0.0))
    return future

mcts = AlphaZeroMCTS(
    inference_fn=mock_inference,
    num_threads=${NUM_THREADS}
)

# Take initial snapshot
snapshot1 = tracemalloc.take_snapshot()

# Run searches
print('Running memory profiling searches...')
for i in range(10):
    mcts.search(root_state, ${NUM_SIMULATIONS} // 10)
    print(f'  Search {i+1}/10 complete')

# Take final snapshot
snapshot2 = tracemalloc.take_snapshot()

# Get statistics
stats = snapshot2.compare_to(snapshot1, 'lineno')
top_stats = snapshot2.statistics('lineno')[:20]

# Calculate totals
total_memory = sum(stat.size for stat in stats) / 1024**2  # MB
peak_memory = sum(stat.size for stat in top_stats) / 1024**2  # MB

# Save results
memory_report = {
    'total_memory_mb': total_memory,
    'peak_memory_mb': peak_memory,
    'tree_size': mcts.tree_size,
    'top_allocations': []
}

for stat in top_stats[:10]:
    memory_report['top_allocations'].append({
        'file': stat.traceback.format()[0] if stat.traceback else 'unknown',
        'size_mb': stat.size / 1024**2
    })

with open('${SESSION_DIR}/memory_profile.json', 'w') as f:
    json.dump(memory_report, f, indent=2)

print(f'Memory Profiling Results:')
print(f'  Total Memory: {total_memory:.1f}MB')
print(f'  Peak Memory: {peak_memory:.1f}MB')
print(f'  Tree Size: {mcts.tree_size} nodes')

tracemalloc.stop()
" 2>&1 | tee "${SESSION_DIR}/memory_profiling.log"

    if [[ -f "${SESSION_DIR}/memory_profile.json" ]]; then
        print_success "Memory profiling complete"
    fi
else
    print_info "Skipping memory profiling (--skip-memory)"
fi

################################################################################
# STEP 7: Generate Combined Report
################################################################################
print_header "STEP 7: Generating Combined Analysis Report"

cd "${PROJECT_ROOT}"

print_info "Analyzing all profiling data..."

python -c "
import json
import os
from pathlib import Path
from datetime import datetime

session_dir = Path('${SESSION_DIR}')
report = {
    'session_id': '${TIMESTAMP}',
    'timestamp': datetime.now().isoformat(),
    'configuration': {
        'game_type': '${GAME_TYPE}',
        'threads': ${NUM_THREADS},
        'simulations': ${NUM_SIMULATIONS},
        'batch_size': ${BATCH_SIZE},
        'timeout_ms': ${TIMEOUT_MS}
    },
    'results': {}
}

# Load C++ profiling results
cpp_profile = session_dir / 'cpp_profile.json'
if cpp_profile.exists():
    with open(cpp_profile) as f:
        report['results']['cpp'] = json.load(f)
        print('✓ Loaded C++ profiling data')

# Load Python profiling results
python_report = session_dir / 'python_profiling' / 'profile_report.json'
if python_report.exists() and python_report.parent.is_dir():
    latest_session = max(python_report.parent.glob('session_*/profile_report.json'))
    with open(latest_session) as f:
        report['results']['python'] = json.load(f)
        print('✓ Loaded Python profiling data')

# Load GPU profiling results
gpu_profile = session_dir / 'gpu_profile.json'
if gpu_profile.exists():
    with open(gpu_profile) as f:
        report['results']['gpu'] = json.load(f)
        print('✓ Loaded GPU profiling data')

# Load memory profiling results
memory_profile = session_dir / 'memory_profile.json'
if memory_profile.exists():
    with open(memory_profile) as f:
        report['results']['memory'] = json.load(f)
        print('✓ Loaded memory profiling data')

# Analyze bottlenecks
bottlenecks = []

# C++ bottlenecks
if 'cpp' in report['results'] and 'timing_stats' in report['results']['cpp']:
    stats = report['results']['cpp']['timing_stats']
    total_time = sum(s.get('total_ns', 0) for s in stats.values())
    for op, stat in stats.items():
        pct = 100.0 * stat.get('total_ns', 0) / total_time if total_time > 0 else 0
        if pct > 10:
            bottlenecks.append({
                'component': 'C++',
                'operation': op,
                'impact_pct': pct,
                'severity': 'high' if pct > 30 else 'medium'
            })

# Python bottlenecks
if 'python' in report['results'] and 'metrics' in report['results']['python']:
    python_metrics = report['results']['python']['metrics']
    if 'python' in python_metrics:
        gil_metrics = python_metrics['python'].get('gil_metrics', {})
        if 'summary' in gil_metrics:
            gil_eff = gil_metrics['summary'].get('gil_efficiency', 100)
            if gil_eff < 80:
                bottlenecks.append({
                    'component': 'Python',
                    'operation': 'GIL contention',
                    'impact_pct': 100 - gil_eff,
                    'severity': 'high' if gil_eff < 60 else 'medium'
                })

# GPU bottlenecks
if 'gpu' in report['results']:
    gpu_util = report['results']['gpu'].get('avg_gpu_utilization', 0)
    if gpu_util < 70:
        bottlenecks.append({
            'component': 'GPU',
            'operation': 'Low utilization',
            'impact_pct': 100 - gpu_util,
            'severity': 'high' if gpu_util < 40 else 'medium'
        })

# Sort bottlenecks by severity and impact
bottlenecks.sort(key=lambda x: (x['severity'] == 'high', x['impact_pct']), reverse=True)
report['bottlenecks'] = bottlenecks

# Save combined report
with open(session_dir / 'combined_report.json', 'w') as f:
    json.dump(report, f, indent=2)

# Generate summary
print('\\n' + '='*60)
print('PROFILING SUMMARY')
print('='*60)
print(f'Session: {report[\"session_id\"]}')
print(f'Configuration: {report[\"configuration\"][\"game_type\"]} with {report[\"configuration\"][\"threads\"]} threads')
print('')

if bottlenecks:
    print('Top Bottlenecks:')
    for b in bottlenecks[:5]:
        severity = '🔴' if b['severity'] == 'high' else '🟡'
        print(f'  {severity} {b[\"component\"]}: {b[\"operation\"]} ({b[\"impact_pct\"]:.1f}%)')
else:
    print('No significant bottlenecks detected')

print('')
print('Performance Metrics:')

# C++ metrics
if 'cpp' in report['results'] and 'summary' in report['results']['cpp']:
    cpp_summary = report['results']['cpp']['summary']
    if 'simulations_per_second' in cpp_summary:
        print(f'  C++ Throughput: {cpp_summary[\"simulations_per_second\"]:.1f} sims/sec')

# GPU metrics
if 'gpu' in report['results']:
    gpu = report['results']['gpu']
    print(f'  GPU Utilization: {gpu.get(\"avg_gpu_utilization\", 0):.1f}%')
    print(f'  GPU Memory: {gpu.get(\"avg_memory_used_mb\", 0):.2f}MB')

# Memory metrics
if 'memory' in report['results']:
    mem = report['results']['memory']
    print(f'  Peak Memory: {mem.get(\"peak_memory_mb\", 0):.1f}MB')
    print(f'  Tree Size: {mem.get(\"tree_size\", 0)} nodes')

print('='*60)
" 2>&1 | tee -a "${SESSION_DIR}/analysis.log"

if [[ -f "${SESSION_DIR}/combined_report.json" ]]; then
    print_success "Combined report generated: combined_report.json"
fi

################################################################################
# STEP 8: Generate HTML Dashboard
################################################################################
print_header "STEP 8: Generating HTML Dashboard"

cat > "${SESSION_DIR}/dashboard.html" << 'EOF'
<!DOCTYPE html>
<html>
<head>
    <title>MCTS Profiling Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .header { background: #2c3e50; color: white; padding: 20px; border-radius: 5px; }
        .section { background: white; padding: 20px; margin: 20px 0; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .metric { display: inline-block; margin: 10px; padding: 15px; background: #ecf0f1; border-radius: 5px; }
        .metric-value { font-size: 24px; font-weight: bold; color: #2c3e50; }
        .metric-label { font-size: 12px; color: #7f8c8d; }
        .bottleneck { padding: 10px; margin: 5px 0; border-left: 4px solid #e74c3c; background: #fff5f5; }
        .bottleneck.medium { border-left-color: #f39c12; background: #fffdf5; }
        table { width: 100%; border-collapse: collapse; }
        th, td { padding: 10px; text-align: left; border-bottom: 1px solid #ecf0f1; }
        th { background: #34495e; color: white; }
        .chart { margin: 20px 0; }
    </style>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head>
<body>
    <div class="header">
        <h1>MCTS Profiling Dashboard</h1>
        <p>Session: <span id="session-id"></span> | <span id="timestamp"></span></p>
    </div>

    <div class="section">
        <h2>Key Metrics</h2>
        <div id="metrics"></div>
    </div>

    <div class="section">
        <h2>Identified Bottlenecks</h2>
        <div id="bottlenecks"></div>
    </div>

    <div class="section">
        <h2>Performance Charts</h2>
        <div id="charts"></div>
    </div>

    <div class="section">
        <h2>Detailed Results</h2>
        <div id="details"></div>
    </div>

    <script>
        // Load and display profiling data
        fetch('combined_report.json')
            .then(response => response.json())
            .then(data => {
                document.getElementById('session-id').textContent = data.session_id;
                document.getElementById('timestamp').textContent = new Date(data.timestamp).toLocaleString();

                // Display metrics
                const metricsDiv = document.getElementById('metrics');
                const metrics = extractMetrics(data);
                metrics.forEach(m => {
                    metricsDiv.innerHTML += `
                        <div class="metric">
                            <div class="metric-value">${m.value}</div>
                            <div class="metric-label">${m.label}</div>
                        </div>
                    `;
                });

                // Display bottlenecks
                const bottlenecksDiv = document.getElementById('bottlenecks');
                if (data.bottlenecks && data.bottlenecks.length > 0) {
                    data.bottlenecks.forEach(b => {
                        bottlenecksDiv.innerHTML += `
                            <div class="bottleneck ${b.severity}">
                                <strong>${b.component}:</strong> ${b.operation}
                                (${b.impact_pct.toFixed(1)}% impact)
                            </div>
                        `;
                    });
                } else {
                    bottlenecksDiv.innerHTML = '<p>No significant bottlenecks detected</p>';
                }

                // Create charts
                createCharts(data);
            });

        function extractMetrics(data) {
            const metrics = [];

            if (data.results.gpu) {
                metrics.push({
                    value: data.results.gpu.avg_gpu_utilization?.toFixed(1) + '%',
                    label: 'GPU Utilization'
                });
            }

            if (data.results.memory) {
                metrics.push({
                    value: data.results.memory.peak_memory_mb?.toFixed(0) + 'MB',
                    label: 'Peak Memory'
                });
            }

            metrics.push({
                value: data.configuration.threads,
                label: 'Threads'
            });

            metrics.push({
                value: data.configuration.simulations,
                label: 'Simulations'
            });

            return metrics;
        }

        function createCharts(data) {
            // Placeholder for charts - would need actual data processing
            document.getElementById('charts').innerHTML = '<p>Charts would be generated here from profiling data</p>';
        }
    </script>
</body>
</html>
EOF

print_success "HTML dashboard generated: dashboard.html"

################################################################################
# Final Summary
################################################################################
print_header "PROFILING COMPLETE"

echo ""
echo "All profiling reports have been saved to:"
echo "  ${SESSION_DIR}"
echo ""
echo "Key files generated:"
echo "  - combined_report.json    : Complete profiling data"
echo "  - dashboard.html          : Interactive HTML dashboard"

if [[ -f "${SESSION_DIR}/cpp_profile.json" ]]; then
    echo "  - cpp_profile.json        : C++ backend metrics"
fi

if [[ -f "${SESSION_DIR}/cpp_trace.json" ]]; then
    echo "  - cpp_trace.json          : Chrome trace (open in chrome://tracing)"
fi

if [[ -f "${SESSION_DIR}/gpu_profile.json" ]]; then
    echo "  - gpu_profile.json        : GPU performance metrics"
fi

if [[ -f "${SESSION_DIR}/memory_profile.json" ]]; then
    echo "  - memory_profile.json     : Memory usage analysis"
fi

echo ""
echo "To view the dashboard:"
echo "  cd ${SESSION_DIR}"
echo "  python -m http.server 8000"
echo "  # Then open http://localhost:8000/dashboard.html"
echo ""

print_success "Profiling session ${TIMESTAMP} completed successfully!"