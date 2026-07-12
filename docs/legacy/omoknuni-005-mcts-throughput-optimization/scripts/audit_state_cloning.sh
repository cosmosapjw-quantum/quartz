#!/bin/bash
# Audit script for state cloning in MCTS hot paths
# Purpose: Verify zero state cloning after Phase 1 implementation
# Expected: 0 occurrences of clone()/copy()/new State() in hot paths

set -e

echo "=== State Cloning Audit ==="
echo "Searching for state cloning patterns in hot paths..."
echo ""

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Hot path files to check
HOT_PATH_FILES=(
    "cpp_extensions/mcts/continuous_simulation_runner.cpp"
    "cpp_extensions/mcts/async_inference_queue.cpp"
    "cpp_extensions/mcts/batch_inference_coordinator.cpp"
)

TOTAL_VIOLATIONS=0

for file in "${HOT_PATH_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo "⚠️  File not found: $file"
        continue
    fi

    echo "Checking $file..."

    # Check for clone() calls
    CLONE_COUNT=$(grep -n "\.clone()" "$file" | grep -v "^[[:space:]]*\/\/" | grep -v "^[[:space:]]*\*" || echo "")
    if [ -n "$CLONE_COUNT" ]; then
        echo "❌ Found clone() calls:"
        echo "$CLONE_COUNT"
        TOTAL_VIOLATIONS=$((TOTAL_VIOLATIONS + $(echo "$CLONE_COUNT" | wc -l)))
    fi

    # Check for copy() calls
    COPY_COUNT=$(grep -n "\.copy()" "$file" | grep -v "^[[:space:]]*\/\/" | grep -v "^[[:space:]]*\*" || echo "")
    if [ -n "$COPY_COUNT" ]; then
        echo "❌ Found copy() calls:"
        echo "$COPY_COUNT"
        TOTAL_VIOLATIONS=$((TOTAL_VIOLATIONS + $(echo "$COPY_COUNT" | wc -l)))
    fi

    # Check for new State allocations
    NEW_STATE_COUNT=$(grep -n "new.*State\|make_shared<.*State\|make_unique<.*State" "$file" | grep -v "^[[:space:]]*\/\/" | grep -v "^[[:space:]]*\*" || echo "")
    if [ -n "$NEW_STATE_COUNT" ]; then
        echo "❌ Found new State allocations:"
        echo "$NEW_STATE_COUNT"
        TOTAL_VIOLATIONS=$((TOTAL_VIOLATIONS + $(echo "$NEW_STATE_COUNT" | wc -l)))
    fi

    echo ""
done

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [ $TOTAL_VIOLATIONS -eq 0 ]; then
    echo "✅ Zero state cloning detected in hot paths"
    echo "✅ Phase 1 constitution principle I satisfied"
    exit 0
else
    echo "❌ Found $TOTAL_VIOLATIONS state cloning violation(s)"
    echo "❌ Phase 1 not fully implemented"
    echo ""
    echo "Action required:"
    echo "  1. Review reported violations above"
    echo "  2. Replace clone/copy with move semantics"
    echo "  3. Use in-place feature extraction"
    echo "  4. Re-run this audit script"
    exit 1
fi
