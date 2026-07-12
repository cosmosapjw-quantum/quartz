# Training Guide - AlphaZero Engine

**Version:** 1.0
**Last Updated:** 2025-09-25
**Contact:** Development Team

This comprehensive guide enables users to train high-performance AlphaZero models for Gomoku, Chess, and Go, achieving superhuman performance on consumer hardware.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Hyperparameter Recommendations](#hyperparameter-recommendations)
3. [Game-Specific Settings](#game-specific-settings)
4. [Training Process](#training-process)
5. [Performance Optimization](#performance-optimization)
6. [Monitoring & Evaluation](#monitoring--evaluation)
7. [Troubleshooting](#troubleshooting)
8. [Advanced Configuration](#advanced-configuration)
9. [Expected Performance Metrics](#expected-performance-metrics)

---

## Quick Start

### Prerequisites

**Hardware Requirements:**
- CPU: AMD Ryzen 5900X or equivalent (8+ cores, 16+ threads)
- GPU: NVIDIA RTX 3060 Ti or better (8GB+ VRAM)
- RAM: 32GB+ system memory
- Storage: 100GB+ free space for training data

**Software Requirements:**
- CUDA 12.x drivers
- Docker 24.0+ with nvidia-container-toolkit (recommended)
- Python 3.12+ (for bare metal setup)

### Training Your First Model

**Docker Setup (Recommended):**
```bash
# Start training environment
docker-compose up -d training

# Begin Gomoku training (easiest to achieve superhuman performance)
docker-compose exec training python -m src.training.training_loop \
    --game gomoku \
    --config config/training_gomoku.yaml \
    --output-dir results/gomoku_run_1
```

**Bare Metal Setup:**
```bash
# Setup environment
source venv/bin/activate
pip install -r requirements.txt

# Build C++ extensions with optimizations
export CFLAGS="-O3 -march=znver3 -fopenmp"
export CXXFLAGS="-O3 -march=znver3 -fopenmp"
python -m pip install -e . --config-settings build-dir=build

# Start training
python -m src.training.training_loop \
    --game gomoku \
    --config config/development.yaml \
    --target-time-hours 48
```

---

## Hyperparameter Recommendations

### Core MCTS Parameters

**Simulations per Move:**
- **Recommended:** 800 simulations
- **Range:** 400-1600 depending on hardware and time constraints
- **Notes:** Higher values improve play strength but reduce training speed

**Exploration Constant (PUCT):**
- **Default:** 1.0
- **Gomoku:** 1.0-1.4 (moderate exploration)
- **Chess:** 0.8-1.2 (tactical precision important)
- **Go:** 1.2-1.6 (higher exploration for complex strategy)

**Temperature Settings:**
- **Initial Temperature:** 1.0 (full exploration)
- **Temperature Decay:** 0.95 per move
- **Temperature Cutoff:** Move 30 for Gomoku/Chess, Move 60 for Go
- **Final Temperature:** 0.1 (near-deterministic play)

### Neural Network Architecture

**Model Size:**
- **Channels:** 256 (optimal for RTX 3060 Ti)
- **ResNet Blocks:** 20 (balance of capacity and training speed)
- **Squeeze-Excitation Ratio:** 0.25
- **Parameter Count:** ~24M parameters

**Training Hyperparameters:**
```yaml
# Learning rate schedule
learning_rate: 0.001
lr_schedule: "cosine"
lr_warmup_steps: 1000
lr_min: 0.000001

# Optimizer settings
optimizer: "AdamW"
weight_decay: 0.0001
momentum: 0.9

# Training stability
gradient_clipping: 1.0
batch_size: 512
use_mixed_precision: true
```

### Experience Buffer Configuration

**Buffer Management:**
- **Max Examples:** 1,000,000 (adjust based on available memory)
- **Min Examples Before Training:** 10,000
- **Sampling Temperature:** 1.0 (uniform sampling)
- **Games per Iteration:** 50-100

**Data Storage:**
- **Format:** Memory-mapped Parquet files
- **RAM Cache:** 512MB (16,000+ entries)
- **Persistence:** Automatic across training sessions

---

## Game-Specific Settings

### Gomoku (15x15, 5-in-a-row)

**Fastest Path to Superhuman Performance (24-48 hours):**

```yaml
game:
  game_type: "gomoku"
  board_size: 15
  win_condition: 5
  rule_variant: "standard"

mcts:
  simulations: 800
  exploration_constant: 1.2
  dirichlet_alpha: 0.3
  dirichlet_weight: 0.25

training:
  self_play_games_per_iteration: 100
  training_steps_per_iteration: 1000
  evaluation_frequency: 5
  target_training_time_hours: 48
```

**Key Success Factors:**
- High Dirichlet noise (α=0.3) for opening variety
- Aggressive self-play generation (100 games/iteration)
- Frequent evaluation to track progress

### Chess (8x8, Standard Rules)

**Strong Amateur Performance (1 week):**

```yaml
game:
  game_type: "chess"
  board_size: 8
  rule_variant: "standard"  # Includes Chess960 support

mcts:
  simulations: 1000
  exploration_constant: 1.0
  dirichlet_alpha: 0.2
  dirichlet_weight: 0.2

training:
  self_play_games_per_iteration: 50
  training_steps_per_iteration: 1500
  max_game_length: 450
  evaluation_frequency: 10
  target_training_time_hours: 168  # 1 week
```

**Key Success Factors:**
- Lower Dirichlet noise (α=0.2) for tactical precision
- Longer maximum game length for endgame training
- Extended training time for complex strategy development

### Go (9x9 to 19x19)

**Competitive Performance (Variable by board size):**

```yaml
game:
  game_type: "go"
  board_size: 19  # Start with 19x19 for full game
  rule_variant: "chinese"

mcts:
  simulations: 1200
  exploration_constant: 1.4
  dirichlet_alpha: 0.03  # Very low for strategic depth
  dirichlet_weight: 0.25

training:
  self_play_games_per_iteration: 25  # Longer games
  training_steps_per_iteration: 2000
  max_game_length: 600
  evaluation_frequency: 20
```

**Key Success Factors:**
- Minimal Dirichlet noise (α=0.03) for long-term strategy
- Higher exploration constant for complex position evaluation
- Extended simulations per move for deep calculation

---

## Training Process

### Training Pipeline Overview

The training process follows a continuous cycle:

1. **Self-Play Generation** → Generate training games with current model
2. **Experience Collection** → Store positions, policies, and outcomes
3. **Model Training** → Update neural network on collected experience
4. **Evaluation** → Test model strength against baselines
5. **Checkpointing** → Save progress and best models

### Typical Training Schedule

**Phase 1: Bootstrapping (First 10 iterations)**
- High exploration to build diverse experience
- Frequent model evaluation to track initial progress
- Conservative training parameters to ensure stability

**Phase 2: Rapid Improvement (Iterations 11-100)**
- Optimized hyperparameters for each game
- Aggressive self-play generation
- Regular evaluation against baseline models

**Phase 3: Refinement (Iterations 100+)**
- Reduced exploration for stronger positional understanding
- Focus on endgame and tactical training
- Head-to-head evaluation against previous versions

### Expected Timeline by Game

| Game   | Superhuman Strength | Training Time | Key Milestones |
|--------|-------------------|---------------|----------------|
| Gomoku | 48 hours          | 150-200 iterations | Hour 12: Basic tactics, Hour 24: Advanced patterns, Hour 48: Superhuman |
| Chess  | 1 week            | 500-800 iterations | Day 1: Legal moves, Day 3: Basic tactics, Week 1: Strong amateur |
| Go     | 2-4 weeks*        | 1000+ iterations | Week 1: Basic rules, Week 2: Territory, Week 4: Complex strategy |

*Go timing depends heavily on board size (9x9 faster than 19x19)

---

## Performance Optimization

### Hardware Optimization

**GPU Utilization:**
```bash
# Monitor GPU usage during training
nvidia-smi -l 1

# Target: 80-92% GPU utilization
# If lower: Increase batch sizes, reduce inference timeout
# If OOM: Decrease batch size, enable gradient checkpointing
```

**CPU Threading:**
```bash
# Find optimal thread count for your hardware
python scripts/tune_threads.py --game gomoku --quick-test

# Typical optimal ranges:
# - 8-core CPU: 6-8 threads
# - 12-core CPU: 8-10 threads
# - 16-core CPU: 10-12 threads
```

**Memory Management:**
```bash
# Monitor memory usage
python scripts/check_memory_usage.py --duration 3600

# Target: <1GB for MCTS tree, stable over time
# If memory grows: Check for leaks in experience buffer
```

### Training Speed Optimization

**Batch Size Tuning:**
```bash
# Optimize batch sizes for your GPU
python scripts/tune_batch_size.py --game gomoku --max-vram 85

# RTX 3060 Ti recommended ranges:
# - Training batch: 256-512
# - Inference batch: 32-64
```

**Virtual Loss Optimization:**
```bash
# Tune virtual loss for optimal thread coordination
python scripts/tune_virtual_loss.py --game gomoku --quick-test

# Typical optimal: 0.8-1.2
# Higher values: Reduced contention, less exploration
# Lower values: More exploration, potential contention
```

### System-Level Optimization

**CUDA Settings:**
```bash
export CUDA_LAUNCH_BLOCKING=0
export CUDA_CACHE_DISABLE=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID
```

**CPU Affinity:**
```bash
# Pin training process to specific cores (optional)
taskset -c 0-11 python -m src.training.training_loop
```

---

## Monitoring & Evaluation

### Key Metrics to Track

**Training Metrics:**
- **Loss Values:** Policy loss, value loss, total loss
- **Gradient Norms:** Should remain stable (1-10 range)
- **Learning Rate:** Following schedule correctly
- **Training Speed:** Steps per second, games per hour

**Performance Metrics:**
- **Simulations/Second:** Target 30,000-40,000 including NN inference
- **GPU Utilization:** Target 80-92% sustained
- **Memory Usage:** <1GB for tree, stable over time
- **Thread Efficiency:** <10% contention

**Model Strength Metrics:**
- **Glicko-2 Rating:** Track improvement over baselines
- **Win Rate:** Against previous versions and baselines
- **Average Game Length:** Indicator of tactical understanding

### TensorBoard Integration

```bash
# Launch TensorBoard to monitor training
tensorboard --logdir logs/tensorboard --port 6006

# Key dashboards to monitor:
# - Training Loss Curves
# - Performance Metrics
# - Model Evaluation Results
# - System Resource Usage
```

### Automated Evaluation

The system automatically evaluates models using:
- **Random Baseline:** 0 ELO anchor point
- **Previous Model Versions:** Track improvement over time
- **Glicko-2 Rating System:** Statistical confidence in ratings
- **Head-to-Head Matches:** Direct comparison between models

---

## Troubleshooting

### Common Training Issues

**Issue: Training Loss Not Decreasing**
```bash
# Check learning rate schedule
grep "learning_rate" logs/alphazero.log

# Verify gradient flow
python -c "
from src.training.trainer import AlphaZeroTrainer
trainer = AlphaZeroTrainer('gomoku')
trainer.check_gradients()  # Should show non-zero gradients
"

# Solutions:
# 1. Increase learning rate (0.003-0.01)
# 2. Reduce batch size to 256
# 3. Check for NaN gradients
# 4. Verify data loading correctness
```

**Issue: GPU Memory Errors (OOM)**
```bash
# Enable automatic batch size reduction
export ALPHAZERO_NEURAL_NETWORK_ENABLE_OOM_RECOVERY=true

# Manually reduce batch sizes
export ALPHAZERO_TRAINING_BATCH_SIZE=256
export ALPHAZERO_MCTS_BATCH_SIZE_MAX=32

# Enable gradient checkpointing
export ALPHAZERO_NEURAL_NETWORK_USE_GRADIENT_CHECKPOINTING=true
```

**Issue: Low GPU Utilization (<60%)**
```bash
# Increase batch sizes
export ALPHAZERO_MCTS_BATCH_SIZE_MIN=64
export ALPHAZERO_MCTS_BATCH_SIZE_MAX=128

# Reduce inference timeout
export ALPHAZERO_MCTS_INFERENCE_TIMEOUT_MS=2.0

# Check for CPU bottlenecks
python scripts/profile_training.py --component inference_worker
```

**Issue: Training Instability/Divergence**
```bash
# Enable gradient clipping
export ALPHAZERO_NEURAL_NETWORK_GRADIENT_CLIPPING=0.5

# Reduce learning rate
export ALPHAZERO_NEURAL_NETWORK_LEARNING_RATE=0.0005

# Enable mixed precision stability
export ALPHAZERO_NEURAL_NETWORK_USE_MIXED_PRECISION=true
export ALPHAZERO_NEURAL_NETWORK_MIXED_PRECISION_LOSS_SCALE=512
```

**Issue: Memory Leaks During Long Training**
```bash
# Run memory leak detection
python scripts/check_memory_leaks.py --duration 7200

# Enable periodic cleanup
export ALPHAZERO_TRAINING_ENABLE_PERIODIC_CLEANUP=true
export ALPHAZERO_TRAINING_CLEANUP_FREQUENCY=100

# Monitor with continuous testing
python tests/soak/test_memory_stability.py --duration 3600
```

### Performance Regression Detection

**Issue: Training Speed Degradation**
```bash
# Run performance benchmarks
python -m pytest tests/performance/test_benchmarks.py -v

# Compare against baseline
python scripts/compare_performance.py \
    --baseline results/baseline_metrics.json \
    --current results/current_metrics.json

# Profile specific components
python scripts/profile_training.py --component mcts_search
```

**Issue: Model Quality Regression**
```bash
# Run evaluation against known strong baselines
python -m src.training.evaluator \
    --model1 models/current.pth \
    --model2 models/baseline.pth \
    --games 100

# Check for training data contamination
python scripts/validate_training_data.py --data-dir training_data/
```

### System-Level Issues

**Issue: Thread Safety Violations**
```bash
# Run with thread sanitizer (debug build)
export TSAN_OPTIONS="halt_on_error=1 history_size=7"
python -m pytest tests/integration/ -v

# Enable debug logging
export ALPHAZERO_SYSTEM_LOG_LEVEL=DEBUG
tail -f logs/alphazero.log | grep -i "thread\|lock\|atomic"
```

**Issue: CUDA Driver/Runtime Mismatch**
```bash
# Check CUDA installation
nvidia-smi
nvcc --version

# Verify PyTorch CUDA compatibility
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"

# Rebuild with correct CUDA version
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## Advanced Configuration

### Custom Loss Functions

For specialized training requirements, you can modify loss weighting:

```python
# In src/training/trainer.py
class AlphaZeroTrainer:
    def __init__(self, game_type: str, loss_weights: dict = None):
        self.loss_weights = loss_weights or {
            'policy': 1.0,    # Standard policy loss weight
            'value': 1.0,     # Standard value loss weight
            'l2': 0.0001      # L2 regularization weight
        }
```

### Multi-Game Training

Train on multiple games simultaneously:

```yaml
# config/multi_game.yaml
training:
  multi_game_training: true
  game_ratios:
    gomoku: 0.4
    chess: 0.4
    go: 0.2
  shared_trunk: true  # Share feature extraction layers
```

### Curriculum Learning

Progressive difficulty training:

```yaml
training:
  curriculum_learning: true
  curriculum_schedule:
    - stage: "basic"
      iterations: 50
      opponent_strength: "random"
    - stage: "intermediate"
      iterations: 100
      opponent_strength: "previous_model"
    - stage: "advanced"
      iterations: -1  # Until completion
      opponent_strength: "best_model"
```

### Distributed Training (Experimental)

For multi-GPU setups:

```bash
# Launch distributed training
python -m torch.distributed.launch \
    --nproc_per_node=2 \
    src/training/distributed_training.py \
    --config config/distributed.yaml
```

---

## Expected Performance Metrics

### Target Performance by Training Phase

**Gomoku Training Progression:**

| Training Hour | ELO Rating | Win Rate vs Random | Key Abilities |
|---------------|------------|-------------------|---------------|
| 2-4 hours     | 100-300    | 70-80%           | Basic 3-in-a-row threats |
| 8-12 hours    | 500-800    | 90-95%           | 4-in-a-row tactics |
| 24-36 hours   | 1200-1500  | 99%+             | Advanced patterns, forks |
| 48+ hours     | 1800+      | Superhuman       | Complex strategy, sacrifices |

**Chess Training Progression:**

| Training Day | Approximate Rating | Key Abilities |
|--------------|-------------------|---------------|
| 1-2 days     | 800-1200         | Legal moves, basic tactics |
| 3-4 days     | 1400-1600        | Tactical combinations |
| 5-7 days     | 1700-1900        | Positional understanding |

**Hardware Performance Targets:**

| Metric | Target | Measurement Method |
|--------|--------|--------------------|
| Simulations/sec | 30,000-40,000 | Including NN inference time |
| GPU Utilization | 80-92% | nvidia-smi during active training |
| Memory Usage | <1GB | MCTS tree memory footprint |
| Training Speed | 200-300 games/hour | Self-play generation rate |
| Thread Efficiency | >85% | CPU utilization / theoretical max |

### Validation Commands

```bash
# Validate training performance
python scripts/validate_training_performance.py \
    --target-simulations 35000 \
    --target-gpu-util 85 \
    --duration 300

# Check model strength progression
python scripts/validate_model_strength.py \
    --game gomoku \
    --checkpoint-dir checkpoints/ \
    --baseline-games 50

# System-wide performance validation
python -m pytest tests/performance/test_benchmarks.py -v -k "test_full_system_performance"

# Validate training guide implementation
python -m pytest tests/unit/test_training_guide_validation.py -v

# End-to-end training validation
python scripts/validate_end_to_end_training.py \
    --game gomoku \
    --duration 1800 \
    --target-improvement 50

# Memory stability during training
python scripts/validate_memory_stability.py \
    --duration 3600 \
    --max-memory-gb 16 \
    --check-interval 60

# GPU memory optimization validation
python scripts/validate_gpu_optimization.py \
    --batch-sizes 32,64,128 \
    --memory-fraction 0.9 \
    --target-utilization 85

# Neural network inference validation
python scripts/validate_inference_pipeline.py \
    --games gomoku,chess,go \
    --batch-sizes 16,32,64 \
    --timeout-ms 3.0
```

### Complete Training Workflow Example

Here's a complete example of training a Gomoku model from scratch:

```bash
# Step 1: Environment setup
source venv/bin/activate
export CUDA_VISIBLE_DEVICES=0

# Step 2: Configuration validation
python scripts/validate_config.py --config config/training_gomoku.yaml

# Step 3: System performance baseline
python scripts/baseline_system_performance.py --output baseline_metrics.json

# Step 4: Start training with monitoring
python -m src.training.training_loop \
    --game gomoku \
    --config config/training_gomoku.yaml \
    --output-dir results/gomoku_$(date +%Y%m%d_%H%M%S) \
    --target-time-hours 48 \
    --enable-tensorboard \
    --checkpoint-frequency 5 \
    --evaluation-frequency 10

# Step 5: Monitor training progress
tensorboard --logdir results/gomoku_*/tensorboard &
python scripts/monitor_training.py --output-dir results/gomoku_*

# Step 6: Validate final performance
python scripts/validate_superhuman_performance.py \
    --model results/gomoku_*/final_model.pth \
    --baseline-games 100 \
    --target-win-rate 0.99
```

---

## Support and Resources

**Documentation:**
- [API Reference](api.md) - Complete API documentation
- [Operations Runbook](operations.md) - Deployment and maintenance
- [Architecture Guide](../specs/001-goal-create-spec/plan.md) - System design details

**Performance Tools:**
- `scripts/tune_*.py` - Hyperparameter optimization scripts
- `tests/performance/` - Benchmark and regression tests
- `tests/soak/` - Long-term stability validation

**Monitoring:**
- TensorBoard integration for training metrics
- Prometheus/Grafana for system monitoring (see operations.md)
- Built-in telemetry system for performance tracking

**Community:**
- GitHub Issues: Report bugs and request features
- Performance Database: Share optimization results
- Model Zoo: Download pre-trained baselines

---

*This training guide enables reproducible superhuman performance on consumer hardware. Follow the game-specific recommendations and performance optimization guidelines for best results.*