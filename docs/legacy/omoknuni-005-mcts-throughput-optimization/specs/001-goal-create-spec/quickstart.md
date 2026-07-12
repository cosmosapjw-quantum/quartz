# QuickStart: High-Performance AlphaZero Engine

## Prerequisites

### Hardware Requirements
- **CPU**: AMD Ryzen 5900X (or equivalent 12+ core processor)
- **GPU**: NVIDIA RTX 3060 Ti with 8GB VRAM (or equivalent)
- **RAM**: 64GB system memory
- **Storage**: 100GB+ free space (for models and training data)

### Software Requirements
- **OS**: Linux (Ubuntu 22.04+ recommended)
- **Python**: 3.12
- **CUDA**: 12.1+ with cuDNN 8.9+
- **Build tools**: GCC 11+, CMake 3.20+, Ninja

## Installation

### 1. Clone Repository
```bash
git clone <repository_url>
cd alphazero-engine
```

### 2. System Dependencies
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y build-essential cmake ninja-build
sudo apt install -y python3.12-dev python3.12-venv
sudo apt install -y libomp-dev libopenblas-dev

# Install CUDA 12.1 (if not already installed)
# Follow NVIDIA installation guide for your distribution
```

### 3. Python Environment
```bash
# Create virtual environment
python3.12 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Build Native Extensions
```bash
# Configure build with optimizations
export CFLAGS="-O3 -march=znver3 -fopenmp"
export CXXFLAGS="-O3 -march=znver3 -fopenmp"

# Build C++ extensions
python -m pip install -e . --config-settings build-dir=build
```

### 5. Verify Installation
```bash
# Run basic tests
python -m pytest tests/contract/ -v           # API contract tests (34 tests)
python -m pytest tests/unit/ -v               # Unit tests for implemented components
python -m pytest tests/integration/ -v        # Integration tests

# Check GPU availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"

# Validate neural network inference optimizations
python scripts/validate_micro_batching.py      # T015: Dynamic micro-batching
python scripts/validate_mixed_precision.py    # T016: Mixed precision inference
python scripts/validate_pinned_memory.py      # T017: Pinned memory optimization
python scripts/validate_cpu_fallback.py       # T018: CPU fallback mechanism

# Test unified game interface (T023)
python -m pytest tests/unit/test_game_adapter_interface.py -v

# Test inference integration pipeline (T019)
python -m pytest tests/integration/test_inference_integration.py -v

# Test Python bindings for games (T024)
python -m pytest tests/unit/test_python_bindings.py -v

# Run Python bindings demonstration
python examples/python_bindings_demo.py
```

## Python Bindings Test

### 1. Test Game Creation and Interface
```bash
# Test all game types and basic functionality
python -c "
import sys
sys.path.insert(0, 'build/cpp_extensions/games')
import alphazero_py
import numpy as np

# Test game creation
for game_type in [alphazero_py.GameType.GOMOKU, alphazero_py.GameType.CHESS, alphazero_py.GameType.GO]:
    game = alphazero_py.create_game(game_type)
    game_name = alphazero_py.game_type_to_string(game_type)
    print(f'{game_name}: {game.get_board_size()}x{game.get_board_size()}, actions: {game.get_action_space_size()}')

# Test numpy integration
gomoku = alphazero_py.create_game(alphazero_py.GameType.GOMOKU)
tensor = gomoku.get_tensor_representation()
print(f'Tensor: {tensor.shape}, dtype: {tensor.dtype}, contiguous: {tensor.flags.c_contiguous}')
"
```

Expected output:
```
gomoku: 15x15, actions: 225
chess: 8x8, actions: 20480
go: 19x19, actions: 362
Tensor: (3, 15, 15), dtype: float32, contiguous: True
✓ Python bindings working correctly
```

### 2. Performance Benchmark
```bash
# Run comprehensive Python bindings demonstration
python examples/python_bindings_demo.py
```

Expected performance output:
```
Created 1000 games in 0.5s (2000+ games/s) ✓
Extracted 1000 tensors in 0.004s (250k+ tensors/s) ✓
Made/undid 1000 moves in 0.007s (140k+ moves/s) ✓
```

## Quick Test: Gomoku Engine

### 1. Basic MCTS Search
```bash
# Test MCTS search with random neural network
python scripts/test_mcts.py --game gomoku --simulations 1000 --threads 8
```

Expected output:
```
✓ MCTS search completed
✓ Simulations/second: ~30,000+ (target: 30-40k)
✓ GPU utilization: 80-92%
✓ Memory usage: <1GB
✓ Thread efficiency: >85%
```

### 2. Neural Network Inference
```bash
# Test GPU inference batching
python scripts/test_inference.py --model models/gomoku_random.pth --batch-sizes 16,32,64
```

Expected output:
```
Batch size 16: 2.1ms latency, 7,600 pos/sec
Batch size 32: 3.2ms latency, 10,000 pos/sec  ✓
Batch size 64: 5.1ms latency, 12,500 pos/sec  ✓
GPU memory usage: 2.1GB / 8.0GB
```

### 3. Self-Play Generation
```bash
# Generate 10 Gomoku games
python scripts/generate_games.py --game gomoku --num-games 10 --output data/test_games/
```

Expected output:
```
Game 1/10: 34 moves, 1.2s, Winner: Player 1
Game 2/10: 28 moves, 0.9s, Winner: Player 2
...
Generation rate: 240 games/hour ✓ (target: 200-300)
Training examples: 420 positions
```

## Training Pipeline Test

### 1. Initialize Experience Buffer
```bash
# Create buffer for training data
python scripts/init_experience_buffer.py --game gomoku --capacity 100000
```

### 2. Generate Initial Training Data
```bash
# Generate 1000 self-play games
python scripts/self_play.py \
    --game gomoku \
    --model models/gomoku_random.pth \
    --num-games 1000 \
    --output data/training/gomoku/ \
    --simulations 400
```

### 3. Train Model for One Iteration
```bash
# Train neural network on self-play data
python scripts/train.py \
    --config configs/gomoku_train.yaml \
    --data data/training/gomoku/ \
    --steps 1000 \
    --output models/gomoku_iter1.pth
```

Expected training output:
```
Step 100/1000: policy_loss=1.85, value_loss=0.42, lr=0.001
Step 200/1000: policy_loss=1.76, value_loss=0.38, lr=0.001
...
Step 1000/1000: policy_loss=1.12, value_loss=0.28, lr=0.001
✓ Model saved: models/gomoku_iter1.pth
✓ Validation accuracy: 76% (target: >75%)
```

### 4. Evaluate Improvement
```bash
# Compare new model against random baseline
python scripts/evaluate.py \
    --old-model models/gomoku_random.pth \
    --new-model models/gomoku_iter1.pth \
    --game gomoku \
    --games 50
```

Expected evaluation output:
```
Evaluation: 50 games between models
New model wins: 42/50 (84%)
Draw rate: 6/50 (12%)
Old model wins: 2/50 (4%)
✓ New model shows clear improvement
```

## Performance Validation

### 1. Memory Soak Test (1 hour)
```bash
# Run memory stability test
python scripts/soak_test.py --duration 3600 --game gomoku --simulations 800
```

Expected output:
```
Starting 1-hour soak test...
Initial memory: 156 MB
30 min: 158 MB (+2 MB)
60 min: 160 MB (+4 MB)
✓ Memory growth: 4 MB/hour (target: <10 MB/hour)
✓ No memory leaks detected
```

### 2. Performance Benchmark
```bash
# Run comprehensive performance test
python scripts/benchmark.py --config configs/performance_test.yaml
```

Expected benchmarks:
```
=== Performance Benchmark Results ===
MCTS simulations/sec: 35,200 ✓ (target: 30-40k)
GPU utilization: 88% ✓ (target: 80-92%)
Average batch size: 45 ✓ (target: 32-64)
CPU utilization: 91% ✓ (target: 85-95%)
Tree memory usage: 420 MB ✓ (target: <1GB)
Thread efficiency: 87% ✓ (target: >85%)

All performance targets met! 🎉
```

### 3. Deterministic Test
```bash
# Verify reproducible results with fixed seed
python scripts/deterministic_test.py --seed 42 --iterations 3
```

Expected output:
```
Run 1: visit_counts = [0, 145, 67, 892, 0, ...]
Run 2: visit_counts = [0, 145, 67, 892, 0, ...]
Run 3: visit_counts = [0, 145, 67, 892, 0, ...]
✓ All runs produced identical results
✓ Deterministic behavior verified
```

## Configuration

### Key Configuration Files
- `configs/gomoku_train.yaml` - Gomoku training parameters
- `configs/chess_train.yaml` - Chess training parameters
- `configs/go_train.yaml` - Go training parameters
- `configs/hardware_tuning.yaml` - Hardware-specific optimizations

### Important Parameters

#### MCTS Configuration
```yaml
# configs/mcts_defaults.yaml
mcts:
  simulations_per_move: 800
  cpuct: 1.25
  virtual_loss: 1.0
  num_threads: 8  # Tune for your CPU
  timeout_ms: 3.0
```

#### GPU Inference
```yaml
# configs/inference_defaults.yaml
inference:
  batch_size: 64      # Adjust for VRAM
  timeout_ms: 3.0     # Balance latency/throughput
  mixed_precision: true
  memory_fraction: 0.85
  warmup_iterations: 10
```

#### Training Pipeline
```yaml
# configs/training_defaults.yaml
training:
  batch_size: 512
  learning_rate: 0.001
  weight_decay: 1e-4
  validation_split: 0.1
  checkpoint_every: 1000
```

## Troubleshooting

### Performance Issues

#### Low GPU Utilization (<70%)
```bash
# Increase batch size or concurrent games
python scripts/tune_batch_size.py --target-util 85

# Check CPU bottleneck
python scripts/profile_cpu.py --duration 60
```

#### High Memory Usage
```bash
# Check for memory leaks
python scripts/memory_profiler.py --duration 300

# Reduce tree size if needed
# Edit config: max_tree_nodes: 25_000_000
```

#### Thread Contention
```bash
# Monitor atomic operation efficiency
python scripts/monitor_threads.py --duration 60

# Reduce thread count if contention >10%
# Edit config: num_threads: 6
```

### Build Issues

#### CMake Configuration Fails
```bash
# Clear build cache
rm -rf build/ *.so
pip install -e . --force-reinstall --no-deps
```

#### CUDA Compilation Errors
```bash
# Verify CUDA installation
nvcc --version
nvidia-smi

# Set explicit CUDA architecture
export TORCH_CUDA_ARCH_LIST="8.6"  # For RTX 3060 Ti
pip install -e . --force-reinstall
```

### Game-Specific Issues

#### Illegal Move Errors
- Check game rule implementation in `cpp_extensions/games/`
- Verify legal move masking before policy normalization
- Run game-specific unit tests: `pytest tests/unit/test_<game>_rules.py`

#### Training Instability
- Reduce learning rate by 2-5x
- Increase gradient clipping: `max_grad_norm: 1.0`
- Check for NaN values: `python scripts/check_training_health.py`

## Next Steps

### Full Training Run
Once quick tests pass, start full training:

```bash
# Gomoku (fastest to converge)
python scripts/full_training.py --game gomoku --target superhuman

# Chess (1 week estimated)
python scripts/full_training.py --game chess --target amateur

# Go (2-3 weeks estimated)
python scripts/full_training.py --game go --board-size 19 --target competitive
```

### Monitor Training Progress
```bash
# Real-time training dashboard
python scripts/training_dashboard.py --port 8080
# Open http://localhost:8080 in browser
```

### Performance Optimization
```bash
# Hardware-specific tuning
python scripts/hardware_optimizer.py --cpu ryzen5900x --gpu rtx3060ti
```

---

**Success Criteria**: All performance targets met, memory stability confirmed, training pipeline operational. Ready for full-scale reinforcement learning training! 🚀