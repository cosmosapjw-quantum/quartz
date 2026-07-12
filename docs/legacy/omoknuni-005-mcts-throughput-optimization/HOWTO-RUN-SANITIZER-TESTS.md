# How to Run Sanitizer Tests

## Sanitizer Build System (T049)

The sanitizer build system provides comprehensive memory error and race condition detection using AddressSanitizer (ASan), ThreadSanitizer (TSan), and UndefinedBehaviorSanitizer (UBSan).

### Quick Commands

```bash
# Run sanitizer unit tests (without sanitizer builds)
python -m pytest tests/unit/test_sanitizer_builds.py -v

# Check system requirements for sanitizer builds
python scripts/build_with_sanitizers.py --check-only

# Build with specific sanitizer (requires clang/clang++)
python scripts/build_with_sanitizers.py --sanitizer asan
python scripts/build_with_sanitizers.py --sanitizer tsan --test
python scripts/build_with_sanitizers.py --sanitizer ubsan --clean

# Build with all sanitizers
python scripts/build_with_sanitizers.py --all --clean --test
```

### Sanitizer Types

1. **AddressSanitizer (ASan)**: Detects memory leaks, use-after-free, buffer overflows
2. **ThreadSanitizer (TSan)**: Detects race conditions, data races, thread safety issues
3. **UndefinedBehaviorSanitizer (UBSan)**: Detects undefined behavior, integer overflows

### System Requirements

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install clang llvm cmake build-essential
```

**macOS:**
```bash
brew install llvm
# or
xcode-select --install
```

### Configuration

Sanitizer configurations are defined in `pyproject.toml`:
- `[tool.scikit-build.cmake.define.asan]` - AddressSanitizer settings
- `[tool.scikit-build.cmake.define.tsan]` - ThreadSanitizer settings
- `[tool.scikit-build.cmake.define.ubsan]` - UndefinedBehaviorSanitizer settings

### CI Integration

The GitHub Actions CI automatically runs sanitizer builds in the `test-sanitizers` job:
- Matrix build testing all three sanitizers
- Clang compiler for better sanitizer support
- Timeout protection (30 minutes per sanitizer)
- Automatic artifact upload on failures

### Test Categories

**Pytest Markers:**
- `@pytest.mark.sanitizer` - General sanitizer tests
- `@pytest.mark.asan` - AddressSanitizer-specific tests
- `@pytest.mark.tsan` - ThreadSanitizer-specific tests
- `@pytest.mark.ubsan` - UndefinedBehaviorSanitizer-specific tests

**Run specific sanitizer tests:**
```bash
python -m pytest -m "asan" -v
python -m pytest -m "tsan" -v
python -m pytest -m "ubsan" -v
python -m pytest -m "sanitizer" -v
```

### Environment Variables

**AddressSanitizer:**
```bash
export ASAN_OPTIONS="detect_leaks=1:abort_on_error=1:detect_stack_use_after_return=true"
export ASAN_SYMBOLIZER_PATH=$(which llvm-symbolizer)
```

**ThreadSanitizer:**
```bash
export TSAN_OPTIONS="halt_on_error=1:history_size=7"
```

**UndefinedBehaviorSanitizer:**
```bash
export UBSAN_OPTIONS="print_stacktrace=1:halt_on_error=1"
```

### Expected Performance Impact

- **AddressSanitizer**: 2-3x slower, 2-3x memory usage
- **ThreadSanitizer**: 5-15x slower, 5-10x memory usage
- **UndefinedBehaviorSanitizer**: 20% slower, minimal memory impact

### Troubleshooting

**Build Failures:**
```bash
# Clean build artifacts
python scripts/build_with_sanitizers.py --clean

# Check compiler availability
which clang clang++
clang --version

# Install missing dependencies
sudo apt-get install libomp-dev
```

**Test Timeouts:**
- Sanitizer tests have 30-minute timeout in CI
- Local testing may run slower on older hardware
- Use `--tb=short` for concise failure output

**Memory Issues:**
- TSan requires significant memory (>4GB recommended)
- ASan may cause OOM on systems with <8GB RAM
- Run single sanitizers instead of `--all` on limited systems

### Integration with AlphaZero Components

The sanitizer system integrates with:
- C++ MCTS engine for race condition detection
- Memory management in node pools and tree structures
- Threading in search coordination and inference workers
- GPU memory management and CUDA operations

### Continuous Integration

Sanitizer builds run automatically on:
- Push to main branch or development branches
- Pull requests (manual trigger required)
- Nightly builds for comprehensive testing

Results available in GitHub Actions:
- Build logs with sanitizer output
- Artifact uploads for failed runs
- Performance comparison with baseline builds

---

*For detailed sanitizer configuration, see `pyproject.toml` sections `[tool.scikit-build.cmake.define.{asan,tsan,ubsan}]`*