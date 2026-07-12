# Operations Runbook - AlphaZero Engine

**Version:** 1.0
**Last Updated:** 2025-09-25
**Contact:** Operations Team

This runbook provides comprehensive operational procedures for deploying, monitoring, and maintaining the AlphaZero engine in production environments.

---

## Table of Contents

1. [Deployment Procedures](#deployment-procedures)
2. [Configuration Management](#configuration-management)
3. [Monitoring & Observability](#monitoring--observability)
4. [Error Handling & Fault Tolerance](#error-handling--fault-tolerance)
5. [Troubleshooting Guide](#troubleshooting-guide)
6. [Maintenance Tasks](#maintenance-tasks)
7. [Performance Optimization](#performance-optimization)
8. [Security & Compliance](#security--compliance)
9. [Disaster Recovery](#disaster-recovery)

---

## Deployment Procedures

### Prerequisites

**Hardware Requirements:**
- CPU: AMD Ryzen 5900X or equivalent (8+ cores, 16+ threads)
- GPU: NVIDIA RTX 3060 Ti or better (8GB+ VRAM, CUDA 12.x)
- RAM: 32GB+ system memory
- Storage: 100GB+ SSD for models/data, 10GB+ for logs

**Software Requirements:**
- Docker 24.0+ with nvidia-container-toolkit
- Python 3.12+ (for bare metal deployment)
- CUDA 12.x drivers
- CMake 3.18+ (for development builds)

### Docker Deployment (Recommended)

#### Quick Production Deployment

```bash
# Clone repository
git clone <repository-url> && cd omoknuni

# Build production image
./scripts/docker/build.sh -t runtime

# Deploy with docker-compose
docker-compose up -d runtime

# Verify deployment
docker-compose logs runtime
curl http://localhost:8000/health
```

#### Development Deployment

```bash
# Start development environment
docker-compose up -d dev

# Access Jupyter Lab at http://localhost:8888
# Password is in logs: docker-compose logs dev
```

#### Training Deployment

```bash
# Start training environment
docker-compose up -d training

# Monitor training progress
docker-compose logs -f training

# Access TensorBoard at http://localhost:6007
```

### Bare Metal Deployment

#### Environment Setup

```bash
# Create virtual environment
python3.12 -m venv venv --prompt alphazero
source venv/bin/activate

# Install dependencies
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# Build C++ extensions (production optimizations)
export CFLAGS="-O3 -march=native -fopenmp"
export CXXFLAGS="-O3 -march=native -fopenmp"
python -m pip install -e . --config-settings build-dir=build
```

#### Service Configuration

```bash
# Create systemd service
sudo cp scripts/alphazero.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable alphazero
sudo systemctl start alphazero

# Verify service status
sudo systemctl status alphazero
sudo journalctl -u alphazero -f
```

### Cloud Deployment

#### AWS EC2 with GPU

```bash
# Launch p3.2xlarge or g4dn.xlarge instance
# Install CUDA drivers and Docker
curl -fsSL https://get.docker.com | sh
sudo systemctl start docker

# Install nvidia-container-toolkit
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/ubuntu20.04/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt update && sudo apt install -y nvidia-container-toolkit
sudo systemctl restart docker

# Deploy AlphaZero
git clone <repository-url> && cd omoknuni
docker-compose up -d runtime
```

#### Google Cloud Platform

```bash
# Create VM with GPU
gcloud compute instances create alphazero-prod \
  --zone=us-central1-a \
  --machine-type=n1-standard-4 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --image-family=ubuntu-2004-lts \
  --image-project=ubuntu-os-cloud \
  --maintenance-policy=TERMINATE

# SSH and deploy
gcloud compute ssh alphazero-prod
# Follow bare metal deployment steps
```

### Configuration Validation

```bash
# Validate configuration before deployment
python -c "from src.utils.config import ConfigManager; manager = ConfigManager('config/production.yaml'); config = manager.load_config(); print('✅ Production configuration valid')" || echo "❌ Configuration error"

# Test GPU availability
python -c "import torch; print(f'✅ GPU detected: {torch.cuda.get_device_name()}') if torch.cuda.is_available() else print('❌ No GPU available')"
```

---

## Configuration Management

### Environment-Specific Configurations

**Configuration Files:**
- `config/default.yaml` - Balanced settings for general use
- `config/development.yaml` - Development-optimized (faster iterations)
- `config/production.yaml` - Production-optimized (maximum performance)

### Environment Variable Overrides

Use the `ALPHAZERO_<SECTION>_<PARAMETER>` pattern for runtime configuration:

```bash
# Production optimizations
export ALPHAZERO_MCTS_SIMULATIONS=1600
export ALPHAZERO_MCTS_THREADS=12
export ALPHAZERO_NEURAL_NETWORK_USE_MIXED_PRECISION=true
export ALPHAZERO_SYSTEM_LOG_LEVEL=WARNING
export ALPHAZERO_SYSTEM_MAX_MEMORY_GB=64

# Training optimizations
export ALPHAZERO_TRAINING_BATCH_SIZE=1024
export ALPHAZERO_TRAINING_SELF_PLAY_GAMES_PER_ITERATION=100
export ALPHAZERO_TRAINING_PARALLEL_SELF_PLAY_GAMES=8
```

### Configuration Validation

```bash
# Validate current configuration
python scripts/validate_config.py

# Test configuration with dry-run
python scripts/dry_run_training.py --config config/production.yaml

# Compare configurations
python scripts/compare_configs.py config/default.yaml config/production.yaml
```

### Hot Configuration Reload

```bash
# Reload configuration without restart (if supported)
kill -SIGHUP $(pgrep -f alphazero)

# Or restart service
sudo systemctl restart alphazero
docker-compose restart runtime
```

---

## Monitoring & Observability

### Built-in Telemetry

The AlphaZero engine includes comprehensive Prometheus-compatible metrics:

**Key Metrics:**
- `alphazero_simulations_per_second` - MCTS performance
- `alphazero_gpu_utilization_percent` - GPU efficiency
- `alphazero_memory_usage_gb` - Memory consumption
- `alphazero_games_generated_per_hour` - Self-play rate
- `alphazero_inference_batch_size_avg` - Batching efficiency

### Monitoring Stack Setup

#### Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'alphazero'
    static_configs:
      - targets: ['localhost:8080']
    scrape_interval: 5s
    metrics_path: /metrics
```

#### Grafana Dashboards

Import the provided dashboard from `monitoring/grafana-dashboard.json`:

**Key Panels:**
- MCTS Performance (simulations/sec over time)
- GPU Utilization (real-time percentage)
- Memory Usage (system and GPU memory)
- Training Progress (loss curves, games/hour)
- System Health (CPU, disk, network)

#### Alerting Rules

```yaml
# alerts.yml
groups:
  - name: alphazero
    rules:
      - alert: HighGPUUtilization
        expr: alphazero_gpu_utilization_percent > 95
        for: 5m
        annotations:
          summary: "GPU utilization consistently above 95%"

      - alert: LowSimulationRate
        expr: alphazero_simulations_per_second < 25000
        for: 2m
        annotations:
          summary: "MCTS performance below target (25k sims/sec)"

      - alert: MemoryLeak
        expr: increase(alphazero_memory_usage_gb[1h]) > 2
        for: 1h
        annotations:
          summary: "Memory usage increased by >2GB in 1 hour"
```

### Log Management

#### Log Configuration

```bash
# Configure structured logging
export ALPHAZERO_SYSTEM_LOG_LEVEL=INFO
export ALPHAZERO_SYSTEM_LOG_FILE=/var/log/alphazero/engine.log

# Enable JSON logging for better parsing
export ALPHAZERO_LOG_FORMAT=json
```

#### Log Rotation

```bash
# Setup logrotate
sudo tee /etc/logrotate.d/alphazero << 'EOF'
/var/log/alphazero/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    sharedscripts
    postrotate
        systemctl reload alphazero || true
    endscript
}
EOF
```

#### Log Analysis

```bash
# Monitor real-time logs
tail -f /var/log/alphazero/engine.log

# Search for errors
grep -i error /var/log/alphazero/engine.log

# Performance statistics
grep "simulations_per_second" /var/log/alphazero/engine.log | tail -100

# Training progress
grep "training_loss" /var/log/alphazero/engine.log | tail -20
```

### Health Checks

```bash
# Application health endpoint
curl -f http://localhost:8000/health

# GPU health check
nvidia-smi

# Memory health check
free -h && df -h

# Process health check
pgrep -af alphazero
```

---

## Error Handling & Fault Tolerance

### Built-in Error Handling Framework

The AlphaZero engine includes a comprehensive error handling framework designed for production stability:

**Key Components:**
- **Custom Exception Hierarchy** - Specialized exceptions with context and severity levels
- **Thread Health Monitoring** - Automatic failure tracking with exponential backoff and termination
- **GPU Operation Management** - Timeout protection and CUDA error categorization
- **Model Validation** - Comprehensive integrity checks for neural network models
- **Centralized Error Reporting** - Metrics collection and trend analysis

### Error Monitoring

**Check Error Summary:**
```bash
# View error statistics from the error reporter
python << 'EOF'
from src.utils.errors import error_reporter
summary = error_reporter.get_error_summary()
print(f"Total errors: {summary['total_errors']}")
print(f"Error types: {summary['error_counts']}")
EOF
```

**Monitor Thread Health:**
```bash
# Check thread health status in logs
grep -i "thread.*failure" /var/log/alphazero/engine.log
grep -i "emergency shutdown" /var/log/alphazero/engine.log
grep -i "thread.*terminated" /var/log/alphazero/engine.log
```

**GPU Error Detection:**
```bash
# Check for GPU operation errors
grep -i "cuda.*error\|gpu.*error\|oom" /var/log/alphazero/engine.log
grep -i "critical.*inference\|gpu.*timeout" /var/log/alphazero/engine.log
```

### Error Recovery Procedures

**Thread Recovery:**
- Threads automatically recover from transient failures with exponential backoff
- Persistent failures trigger automatic thread termination after configurable thresholds
- Emergency shutdown procedures protect system integrity during critical errors

**GPU Recovery:**
- CUDA out-of-memory errors trigger automatic batch size reduction (T050)
- GPU operation timeouts have configurable limits with graceful degradation
- Critical GPU errors trigger CPU fallback mechanisms

**Model Validation Recovery:**
- Model loading failures provide detailed diagnostics for troubleshooting
- Integrity checks prevent corrupted model deployment
- Automatic model compatibility validation before inference

### Error Handling Configuration

**Thread Health Settings:**
```python
# Configure thread health monitoring thresholds
THREAD_HEALTH_CONFIG = {
    "max_consecutive_failures": 5,    # Thread termination threshold
    "failure_backoff": 0.5,           # Initial backoff time (seconds)
    "max_backoff": 10.0               # Maximum backoff time (seconds)
}
```

**GPU Operation Settings:**
```python
# Configure GPU operation management
GPU_OPERATION_CONFIG = {
    "default_timeout": 30.0,          # Default operation timeout (seconds)
    "critical_memory_threshold": 0.9  # Memory usage threshold for warnings
}
```

### Testing Error Handling

**Run Error Handling Tests:**
```bash
# Comprehensive error handling test suite
python -m pytest tests/unit/test_error_handling.py -v

# Test specific components
python -m pytest tests/unit/test_error_handling.py -v -k "ThreadHealth"
python -m pytest tests/unit/test_error_handling.py -v -k "GPUOperation"
python -m pytest tests/unit/test_error_handling.py -v -k "ModelValidator"
```

**Manual Error Simulation:**
```bash
# Test error handling under controlled conditions
python << 'EOF'
from src.utils.errors import ThreadHealthMonitor, InferenceError

# Simulate thread failures
monitor = ThreadHealthMonitor(max_consecutive_failures=3)
for i in range(5):
    error = InferenceError(f'Simulated failure {i}')
    should_continue = monitor.record_failure('test_thread', error)
    print(f'Failure {i+1}: continue={should_continue}')
    if not should_continue:
        break
EOF
```

---

## Troubleshooting Guide

### Common Issues

#### 1. CUDA Out of Memory (OOM)

**Symptoms:**
- `RuntimeError: CUDA out of memory`
- GPU utilization drops to 0%
- Training/inference stops

**Solutions:**
```bash
# Reduce batch size
export ALPHAZERO_NEURAL_NETWORK_BATCH_SIZE_PREFERRED=32
export ALPHAZERO_TRAINING_BATCH_SIZE=256

# Enable gradient checkpointing
export ALPHAZERO_NEURAL_NETWORK_USE_GRADIENT_CHECKPOINTING=true

# Restart inference workers
docker-compose restart runtime
```

**Prevention:**
- Monitor GPU memory with `nvidia-smi`
- Use batch size optimization script: `python scripts/tune_batch_size.py`
- Set conservative memory limits in production.yaml

#### 2. Low MCTS Performance

**Symptoms:**
- Simulations/sec below 25,000
- High CPU usage, low GPU utilization
- Slow game generation

**Diagnostics:**
```bash
# Check thread contention
python scripts/tune_threads.py --game gomoku --quick-test

# Check virtual loss settings
python scripts/tune_virtual_loss.py --game gomoku --quick-test

# Profile MCTS operations
python scripts/profile_mcts.py --duration 60
```

**Solutions:**
```bash
# Optimize thread count
export ALPHAZERO_MCTS_THREADS=8

# Adjust virtual loss magnitude
export ALPHAZERO_MCTS_VIRTUAL_LOSS=1.0

# Increase batch timeout
export ALPHAZERO_MCTS_INFERENCE_TIMEOUT_MS=5.0
```

#### 3. Training Instability

**Symptoms:**
- NaN losses
- Exploding gradients
- Model performance degradation

**Solutions:**
```bash
# Enable gradient clipping
export ALPHAZERO_NEURAL_NETWORK_GRADIENT_CLIPPING=1.0

# Reduce learning rate
export ALPHAZERO_NEURAL_NETWORK_LEARNING_RATE=0.0001

# Enable mixed precision
export ALPHAZERO_NEURAL_NETWORK_USE_MIXED_PRECISION=true

# Restore from last stable checkpoint
python scripts/restore_checkpoint.py --checkpoint checkpoints/best_model.pth
```

#### 4. Docker Container Issues

**Container won't start:**
```bash
# Check Docker daemon
sudo systemctl status docker

# Check GPU runtime
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi

# Check container logs
docker-compose logs runtime

# Rebuild image
docker-compose build --no-cache runtime
```

#### 5. Configuration Errors

**Invalid configuration:**
```bash
# Validate configuration
python -c "from src.utils.config import load_config; load_config('config/production.yaml')"

# Check environment variables
env | grep ALPHAZERO

# Reset to defaults
unset $(env | grep ALPHAZERO | cut -d= -f1)
```

### Debugging Procedures

#### 1. Performance Debugging

```bash
# Enable profiling
export ALPHAZERO_SYSTEM_ENABLE_PROFILING=true
export ALPHAZERO_SYSTEM_PROFILE_OUTPUT_DIR=logs/profiles

# Run performance benchmarks
python -m pytest tests/performance/ -v

# Analyze profiles
python scripts/analyze_profiles.py logs/profiles/
```

#### 2. Memory Debugging

```bash
# Run memory leak detection
python scripts/check_memory_leaks.py --duration 3600

# Monitor memory usage
watch -n 5 "free -h && nvidia-smi --query-gpu=memory.used,memory.total --format=csv"

# Analyze memory patterns
python scripts/memory_analysis.py logs/memory_profile.json
```

#### 3. Network Debugging

```bash
# Check port availability
netstat -tlnp | grep 8000

# Test inference API
curl -X POST http://localhost:8000/inference \
  -H "Content-Type: application/json" \
  -d '{"positions": ["test_position"]}'

# Monitor network traffic
sudo tcpdump -i any port 8000
```

### Emergency Procedures

#### Critical System Recovery

```bash
# Emergency stop
sudo systemctl stop alphazero
docker-compose down

# Clear GPU processes
sudo pkill -f python

# Reset GPU
sudo nvidia-smi --gpu-reset

# Restart with safe configuration
export ALPHAZERO_MCTS_THREADS=4
export ALPHAZERO_NEURAL_NETWORK_BATCH_SIZE_PREFERRED=16
sudo systemctl start alphazero
```

#### Data Recovery

```bash
# Backup current state
tar -czf backup-$(date +%Y%m%d-%H%M%S).tar.gz \
  checkpoints/ training_data/ config/ logs/

# Restore from backup
tar -xzf backup-20250925-120000.tar.gz

# Verify model integrity
python scripts/validate_checkpoint.py checkpoints/best_model.pth
```

---

## Maintenance Tasks

### Routine Operations

#### Daily Tasks

```bash
#!/bin/bash
# daily_maintenance.sh

# Check system health
python scripts/health_check.py

# Monitor disk space
df -h | awk '$5 > 80 {print "Warning: " $1 " is " $5 " full"}'

# Rotate logs if needed
sudo logrotate -f /etc/logrotate.d/alphazero

# Backup critical data
./scripts/backup_daily.sh

# Performance report
python scripts/daily_performance_report.py > reports/daily-$(date +%Y%m%d).txt
```

#### Weekly Tasks

```bash
#!/bin/bash
# weekly_maintenance.sh

# Update performance baselines
python scripts/update_benchmarks.py

# Clean old checkpoints (keep best 10)
python scripts/cleanup_checkpoints.py --keep 10

# System optimization check
python scripts/tune_all_parameters.py --quick-test

# Generate performance trends
python scripts/weekly_performance_analysis.py
```

#### Monthly Tasks

```bash
#!/bin/bash
# monthly_maintenance.sh

# Full system benchmark
python -m pytest tests/performance/ --benchmark

# Comprehensive memory leak test
python tests/soak/test_memory_stability.py --duration 3600

# Update documentation
python scripts/update_metrics_docs.py

# Security audit
python scripts/security_audit.py
```

### Scaling Operations

#### Horizontal Scaling

```bash
# Scale inference workers
export ALPHAZERO_TRAINING_PARALLEL_SELF_PLAY_GAMES=16

# Multi-GPU setup
export ALPHAZERO_NEURAL_NETWORK_GPU_ID=0,1,2,3
export ALPHAZERO_NEURAL_NETWORK_DATA_PARALLEL=true

# Distributed training (if implemented)
python -m torch.distributed.launch --nproc_per_node=4 scripts/distributed_training.py
```

#### Vertical Scaling

```bash
# Increase memory limits
export ALPHAZERO_SYSTEM_MAX_MEMORY_GB=128
export ALPHAZERO_SYSTEM_MAX_GPU_MEMORY_FRACTION=0.95

# Optimize for high-end hardware
export ALPHAZERO_MCTS_THREADS=16
export ALPHAZERO_MCTS_MAX_TREE_SIZE_MB=4096
export ALPHAZERO_NEURAL_NETWORK_BATCH_SIZE_PREFERRED=256
```

### Update Procedures

#### Model Updates

```bash
# Download new model
wget https://models.alphazero.com/latest/gomoku_v2.pth -O models/gomoku_v2.pth

# Validate model
python scripts/validate_checkpoint.py models/gomoku_v2.pth

# Deploy new model (rolling update)
cp models/gomoku_v2.pth models/gomoku_current.pth
sudo systemctl reload alphazero

# Verify deployment
python scripts/test_inference.py --model models/gomoku_current.pth
```

#### Software Updates

```bash
# Backup current installation
tar -czf alphazero-backup-$(date +%Y%m%d).tar.gz .

# Pull latest code
git fetch origin
git checkout v1.1.0

# Update dependencies
pip install -r requirements.txt --upgrade

# Rebuild extensions
pip install -e . --force-reinstall

# Run regression tests
python -m pytest tests/integration/ -v

# Deploy update
sudo systemctl restart alphazero
```

### Backup and Restore

#### Backup Procedures

```bash
#!/bin/bash
# backup_production.sh

BACKUP_DATE=$(date +%Y%m%d-%H%M%S)
BACKUP_DIR="/backups/alphazero-$BACKUP_DATE"

mkdir -p "$BACKUP_DIR"

# Backup model checkpoints
cp -r checkpoints/ "$BACKUP_DIR/"

# Backup configuration
cp -r config/ "$BACKUP_DIR/"

# Backup training data (if small enough)
if [ $(du -s training_data/ | cut -f1) -lt 10000000 ]; then
    cp -r training_data/ "$BACKUP_DIR/"
fi

# Backup logs (last 7 days)
find logs/ -name "*.log" -mtime -7 -exec cp {} "$BACKUP_DIR/logs/" \;

# Create archive
tar -czf "alphazero-backup-$BACKUP_DATE.tar.gz" -C /backups "alphazero-$BACKUP_DATE"

# Upload to cloud storage (if configured)
aws s3 cp "alphazero-backup-$BACKUP_DATE.tar.gz" s3://alphazero-backups/

# Cleanup old backups (keep 30 days)
find /backups -name "alphazero-backup-*.tar.gz" -mtime +30 -delete
```

#### Restore Procedures

```bash
#!/bin/bash
# restore_production.sh

BACKUP_FILE="$1"
if [ -z "$BACKUP_FILE" ]; then
    echo "Usage: $0 <backup_file>"
    exit 1
fi

# Stop services
sudo systemctl stop alphazero

# Create restore point
cp -r . "/tmp/alphazero-restore-point-$(date +%Y%m%d-%H%M%S)"

# Extract backup
tar -xzf "$BACKUP_FILE" -C /tmp/
RESTORE_DIR=$(tar -tzf "$BACKUP_FILE" | head -1 | cut -f1 -d"/")

# Restore files
cp -r "/tmp/$RESTORE_DIR/checkpoints/"* checkpoints/
cp -r "/tmp/$RESTORE_DIR/config/"* config/

# Validate restore
python scripts/validate_checkpoint.py checkpoints/best_model.pth

# Restart services
sudo systemctl start alphazero

# Verify functionality
python scripts/health_check.py
```

---

## Performance Optimization

### Target Performance Metrics

- **MCTS Performance:** 30,000+ simulations/second
- **GPU Utilization:** 80-92% sustained during search
- **Memory Usage:** <1GB for 10M node trees
- **Training Speed:** 200+ self-play games/hour
- **Latency:** <100ms average response time

### Optimization Scripts

```bash
# Complete parameter optimization
python scripts/optimize_all_parameters.py --games gomoku,chess,go

# Individual optimizations
python scripts/tune_threads.py --game gomoku --simulations 800 --iterations 50
python scripts/tune_virtual_loss.py --game gomoku --iterations 50
python scripts/tune_batch_size.py --game gomoku --iterations 100
python scripts/tune_timeout.py --game gomoku --iterations 100
```

### Hardware-Specific Tuning

#### AMD Ryzen 5900X Optimization

```bash
# CPU optimization
export ALPHAZERO_MCTS_THREADS=12
export ALPHAZERO_NEURAL_NETWORK_CPU_THREADS=24

# Memory optimization
export ALPHAZERO_SYSTEM_ENABLE_MEMORY_GROWTH=false
export ALPHAZERO_SYSTEM_MAX_MEMORY_GB=64

# Compiler flags for maximum performance
export CXXFLAGS="-O3 -march=znver3 -fopenmp -funroll-loops"
```

#### NVIDIA RTX 3060 Ti Optimization

```bash
# GPU optimization
export ALPHAZERO_NEURAL_NETWORK_USE_TENSORRT=true
export ALPHAZERO_NEURAL_NETWORK_USE_MIXED_PRECISION=true
export ALPHAZERO_SYSTEM_MAX_GPU_MEMORY_FRACTION=0.95

# Batch size optimization
export ALPHAZERO_NEURAL_NETWORK_BATCH_SIZE_PREFERRED=128
export ALPHAZERO_MCTS_BATCH_SIZE_MAX=128
```

---

## Security & Compliance

### Security Hardening

#### Container Security

```bash
# Run containers with minimal privileges
docker run --user 1000:1000 \
           --read-only \
           --tmpfs /tmp \
           --cap-drop ALL \
           alphazero:runtime

# Use security profiles
docker run --security-opt seccomp=seccomp-profile.json \
           --security-opt apparmor=alphazero-profile \
           alphazero:runtime
```

#### Network Security

```bash
# Configure firewall
sudo ufw allow from 10.0.0.0/8 to any port 8000
sudo ufw allow from 192.168.0.0/16 to any port 8000
sudo ufw deny 8000

# Enable TLS for API endpoints
export ALPHAZERO_API_USE_TLS=true
export ALPHAZERO_API_TLS_CERT_PATH=/etc/ssl/certs/alphazero.crt
export ALPHAZERO_API_TLS_KEY_PATH=/etc/ssl/private/alphazero.key
```

#### Data Protection

```bash
# Encrypt sensitive data
gpg --symmetric --cipher-algo AES256 checkpoints/production_model.pth

# Secure file permissions
chmod 600 config/production.yaml
chmod 700 logs/
chmod 755 checkpoints/
```

### Compliance Monitoring

```bash
# Audit log access
audit2allow -a | grep alphazero

# Monitor file changes
inotifywait -m -r -e modify,create,delete /opt/alphazero/

# Generate compliance report
python scripts/generate_compliance_report.py --format html
```

---

## Disaster Recovery

### Recovery Time Objectives (RTO)

- **Critical System Failure:** <30 minutes
- **Data Corruption:** <2 hours
- **Complete System Loss:** <4 hours

### Recovery Point Objectives (RPO)

- **Model Checkpoints:** <1 hour
- **Configuration Changes:** <15 minutes
- **Training Data:** <24 hours

### Disaster Recovery Procedures

#### Complete System Recovery

```bash
#!/bin/bash
# disaster_recovery.sh

echo "Starting disaster recovery procedure..."

# 1. Provision new hardware/instances
# (Manual step - provision equivalent hardware)

# 2. Install base system
curl -fsSL https://get.docker.com | sh
sudo systemctl start docker

# 3. Restore from backup
LATEST_BACKUP=$(aws s3 ls s3://alphazero-backups/ | sort | tail -1 | awk '{print $4}')
aws s3 cp "s3://alphazero-backups/$LATEST_BACKUP" .
tar -xzf "$LATEST_BACKUP"

# 4. Deploy application
git clone <repository-url>
cd omoknuni
cp ../alphazero-*/config/* config/
cp ../alphazero-*/checkpoints/* checkpoints/

# 5. Start services
docker-compose up -d runtime

# 6. Validate recovery
python scripts/health_check.py
python scripts/validate_checkpoint.py checkpoints/best_model.pth

echo "Disaster recovery complete. System operational."
```

#### Automated Failover

```bash
# Configure health check monitoring
# If health check fails for >5 minutes, trigger failover

#!/bin/bash
# failover_monitor.sh

while true; do
    if ! curl -f http://localhost:8000/health; then
        echo "Health check failed. Starting failover..."

        # Start standby instance
        aws ec2 start-instances --instance-ids i-standby123

        # Update DNS to point to standby
        aws route53 change-resource-record-sets \
          --hosted-zone-id Z123456 \
          --change-batch file://failover-dns.json

        break
    fi
    sleep 60
done
```

---

## Quick Reference

### Emergency Contacts

- **Operations Team:** ops@alphazero.com
- **Development Team:** dev@alphazero.com
- **Security Team:** security@alphazero.com

### Critical Commands

```bash
# Emergency stop
sudo systemctl stop alphazero
docker-compose down

# Emergency restart
sudo systemctl restart alphazero
docker-compose restart runtime

# Health check
curl http://localhost:8000/health

# Performance check
python scripts/health_check.py

# View logs
sudo journalctl -u alphazero -f
docker-compose logs -f runtime
```

### Configuration Shortcuts

```bash
# Production mode
export ALPHAZERO_MODE=production
source scripts/production_env.sh

# Development mode
export ALPHAZERO_MODE=development
source scripts/development_env.sh

# Debug mode
export ALPHAZERO_SYSTEM_LOG_LEVEL=DEBUG
export ALPHAZERO_SYSTEM_ENABLE_PROFILING=true
```

---

**Document Version:** 1.0
**Last Updated:** 2025-09-25
**Next Review:** 2025-10-25

For additional support, consult the [API Documentation](api.md) and [Training Guide](training_guide.md).