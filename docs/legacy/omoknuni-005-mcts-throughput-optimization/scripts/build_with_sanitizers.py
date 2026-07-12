#!/usr/bin/env python3
"""
Sanitizer Build Script
======================

Script to build the AlphaZero engine with different sanitizers for local testing.
Supports AddressSanitizer (ASan), ThreadSanitizer (TSan), and UndefinedBehaviorSanitizer (UBSan).

Usage:
    python scripts/build_with_sanitizers.py --sanitizer asan
    python scripts/build_with_sanitizers.py --sanitizer tsan --test
    python scripts/build_with_sanitizers.py --all --clean
"""

import argparse
import os
import subprocess
import sys
import shutil
from pathlib import Path
from typing import Dict, List, Optional


class SanitizerBuilder:
    """Handles building the project with different sanitizers."""

    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path.cwd()
        self.build_dir = self.project_root / "build"

        # Sanitizer configurations
        self.sanitizer_configs = {
            'asan': {
                'name': 'AddressSanitizer',
                'cmake_defines': {
                    'ENABLE_ASAN': 'ON',
                    'CMAKE_BUILD_TYPE': 'Debug'
                },
                'env_vars': {
                    'ASAN_OPTIONS': 'detect_leaks=1:abort_on_error=1:detect_stack_use_after_return=true',
                    'ASAN_SYMBOLIZER_PATH': shutil.which('llvm-symbolizer') or 'llvm-symbolizer',
                    'SANITIZER_BUILD': 'asan'
                }
            },
            'tsan': {
                'name': 'ThreadSanitizer',
                'cmake_defines': {
                    'ENABLE_TSAN': 'ON',
                    'CMAKE_BUILD_TYPE': 'Debug'
                },
                'env_vars': {
                    'TSAN_OPTIONS': 'halt_on_error=1:history_size=7',
                    'SANITIZER_BUILD': 'tsan'
                }
            },
            'ubsan': {
                'name': 'UndefinedBehaviorSanitizer',
                'cmake_defines': {
                    'ENABLE_UBSAN': 'ON',
                    'CMAKE_BUILD_TYPE': 'Debug'
                },
                'env_vars': {
                    'UBSAN_OPTIONS': 'print_stacktrace=1:halt_on_error=1',
                    'SANITIZER_BUILD': 'ubsan'
                }
            }
        }

    def clean_build(self):
        """Clean previous build artifacts."""
        print("🧹 Cleaning build directory...")
        if self.build_dir.exists():
            shutil.rmtree(self.build_dir)

        # Also clean pip build artifacts
        subprocess.run(['pip', 'uninstall', '-y', 'omoknuni'], check=False)

        build_dirs = ['build', 'dist', '*.egg-info']
        for pattern in build_dirs:
            for path in self.project_root.glob(pattern):
                if path.is_dir():
                    print(f"  Removing {path}")
                    shutil.rmtree(path)

    def check_requirements(self) -> bool:
        """Check if required tools are available."""
        print("🔍 Checking requirements...")

        required_tools = ['clang', 'clang++']
        missing_tools = []

        for tool in required_tools:
            if not shutil.which(tool):
                missing_tools.append(tool)

        if missing_tools:
            print(f"❌ Missing required tools: {', '.join(missing_tools)}")
            print("   Install clang/clang++ for better sanitizer support:")
            print("   Ubuntu/Debian: sudo apt-get install clang")
            print("   macOS: xcode-select --install or brew install llvm")
            return False

        # Check for llvm-symbolizer (helpful for ASan)
        if not shutil.which('llvm-symbolizer'):
            print("⚠️  llvm-symbolizer not found - stack traces may not be symbolized")
            print("   Install with: sudo apt-get install llvm")

        print("✅ Requirements check passed")
        return True

    def build_with_sanitizer(self, sanitizer: str, force_rebuild: bool = False) -> bool:
        """Build the project with specified sanitizer."""
        if sanitizer not in self.sanitizer_configs:
            print(f"❌ Unknown sanitizer: {sanitizer}")
            print(f"   Available: {', '.join(self.sanitizer_configs.keys())}")
            return False

        config = self.sanitizer_configs[sanitizer]
        print(f"🔨 Building with {config['name']}...")

        # Set environment variables
        env = os.environ.copy()
        env.update({
            'CC': 'clang',
            'CXX': 'clang++',
        })
        env.update(config['env_vars'])

        # Build using pip with scikit-build-core
        try:
            build_args = ['pip', 'install', '-e', '.']

            if force_rebuild:
                build_args.extend(['--force-reinstall', '--no-deps'])

            # Add cmake configuration for sanitizer
            cmake_defines = config.get('cmake_defines', {})
            for key, value in cmake_defines.items():
                build_args.append(f'--config-settings=cmake.define.{key}={value}')

            print(f"Running: {' '.join(build_args)}")
            result = subprocess.run(build_args, cwd=self.project_root, env=env, check=True)

            print(f"✅ {config['name']} build completed successfully")
            return True

        except subprocess.CalledProcessError as e:
            print(f"❌ Build failed with {config['name']}: {e}")
            return False

    def run_tests(self, sanitizer: str, test_pattern: str = "sanitizer") -> bool:
        """Run tests with the specified sanitizer."""
        if sanitizer not in self.sanitizer_configs:
            print(f"❌ Unknown sanitizer: {sanitizer}")
            return False

        config = self.sanitizer_configs[sanitizer]
        print(f"🧪 Running tests with {config['name']}...")

        # Set environment variables for the test run
        env = os.environ.copy()
        env.update(config['env_vars'])

        try:
            test_args = [
                'pytest',
                'tests/unit/test_sanitizer_builds.py',
                '-v',
                '--tb=short',
                '-m', f'{sanitizer} or sanitizer'
            ]

            print(f"Running: {' '.join(test_args)}")
            result = subprocess.run(test_args, cwd=self.project_root, env=env, check=True)

            print(f"✅ Tests passed with {config['name']}")
            return True

        except subprocess.CalledProcessError as e:
            print(f"❌ Tests failed with {config['name']}: {e}")
            return False

    def build_all_sanitizers(self, clean: bool = False, test: bool = False) -> Dict[str, bool]:
        """Build with all sanitizers."""
        results = {}

        if clean:
            self.clean_build()

        for sanitizer in self.sanitizer_configs:
            print(f"\n{'='*60}")
            success = self.build_with_sanitizer(sanitizer, force_rebuild=True)
            results[sanitizer] = success

            if success and test:
                test_success = self.run_tests(sanitizer)
                results[f"{sanitizer}_tests"] = test_success

        return results

    def print_summary(self, results: Dict[str, bool]):
        """Print build summary."""
        print(f"\n{'='*60}")
        print("BUILD SUMMARY")
        print(f"{'='*60}")

        for sanitizer, success in results.items():
            status = "✅ PASS" if success else "❌ FAIL"
            config_name = self.sanitizer_configs.get(sanitizer.split('_')[0], {}).get('name', sanitizer)
            print(f"{config_name:25} {status}")

        total = len(results)
        passed = sum(results.values())
        print(f"\nResult: {passed}/{total} builds successful")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Build AlphaZero engine with sanitizers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --sanitizer asan                    # Build with AddressSanitizer
  %(prog)s --sanitizer tsan --test             # Build with ThreadSanitizer and run tests
  %(prog)s --all --clean                       # Build with all sanitizers, clean first
  %(prog)s --check-only                        # Just check requirements
        """
    )

    parser.add_argument(
        '--sanitizer',
        choices=['asan', 'tsan', 'ubsan'],
        help='Specific sanitizer to use'
    )

    parser.add_argument(
        '--all',
        action='store_true',
        help='Build with all sanitizers'
    )

    parser.add_argument(
        '--test',
        action='store_true',
        help='Run tests after building'
    )

    parser.add_argument(
        '--clean',
        action='store_true',
        help='Clean build artifacts before building'
    )

    parser.add_argument(
        '--check-only',
        action='store_true',
        help='Only check requirements, do not build'
    )

    parser.add_argument(
        '--project-root',
        type=Path,
        default=Path.cwd(),
        help='Project root directory'
    )

    args = parser.parse_args()

    if not args.sanitizer and not args.all and not args.check_only:
        parser.error("Must specify --sanitizer, --all, or --check-only")

    builder = SanitizerBuilder(args.project_root)

    # Check requirements
    if not builder.check_requirements():
        if not args.check_only:
            print("\n❌ Requirements check failed. Please install required tools.")
            sys.exit(1)
        else:
            sys.exit(1)

    if args.check_only:
        print("✅ All requirements satisfied")
        sys.exit(0)

    if args.clean:
        builder.clean_build()

    results = {}

    if args.all:
        results = builder.build_all_sanitizers(clean=args.clean, test=args.test)
    elif args.sanitizer:
        success = builder.build_with_sanitizer(args.sanitizer, force_rebuild=args.clean)
        results[args.sanitizer] = success

        if success and args.test:
            test_success = builder.run_tests(args.sanitizer)
            results[f"{args.sanitizer}_tests"] = test_success

    builder.print_summary(results)

    # Exit with error if any builds failed
    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
