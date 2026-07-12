# CRITICAL BUG: Race Condition in Async Expansion

## Problem

Multiple threads can expand the same node simultaneously, causing:
1. Massive node allocation waste (482 duplicate expansions in 1000 sims)
2. Orphaned children (overwrit first_child_index)
3. **Backup validation failures → visits don't propagate to children**
4. Flat tree (depth=1) regardless of simulation count
5. 1.00 inferences/sim (GPU severely underutilized)

## Root Cause

`continuous_simulation_runner.cpp::expand_node_with_result()` has NO check if node is already expanded.

Race condition:
1. Thread 1 & 2 both select Child_0 (unexpanded)
2. Both submit inference
3. Both get results
4. Both call `expand_node_with_result(Child_0, ...)`
5. Both allocate 225 children
6. Thread 2 overwrites `first_child_index`
7. Thread 1's path now invalid → backup fails validation
8. **Only root gets visits, children stay at 0**

## Evidence

```
Root visits: 1000
ALL 225 children: visits=0, expanded=False
Nodes allocated: 224,248
First child index: 115,831
Duplicate expansions: (224,248-115,831)/225 = 482
```

## Fix

Add atomic check-and-set in `expand_node_with_result()`:

```cpp
bool ContinuousSimulationRunner::expand_node_with_result(...) {
    // ✅ ADD THIS CHECK AT THE BEGINNING:
    NodeFlags flags = tree_.get_flags(leaf);
    if (flags.is_expanded()) {
        return false;  // Already expanded by another thread
    }

    // ... rest of expansion code ...

    // ⚠️ RACE STILL POSSIBLE between check and allocation
    // Better: Use atomic test-and-set on expanded flag
}
```

Better solution - atomic test-and-set:

```cpp
// Try to atomically mark as "being expanded"
NodeFlags old_flags = tree_.get_flags(leaf);
if (old_flags.is_expanded()) {
    return false;  // Already done
}

// Atomic compare-exchange on flags
NodeFlags new_flags = old_flags;
new_flags.set_expanded(true);
if (!tree_.atomic_cas_flags(leaf, old_flags, new_flags)) {
    return false;  // Another thread won the race
}

// Now WE own the expansion, proceed safely
// ... allocate children ...
```

## Impact

This bug causes:
- **10× performance loss**: 3,000 sims/sec instead of 30,000+
- Wasted GPU capacity (only 3,000 inf/sec vs 19,000 capable)
- Makes AlphaZero training impossible (can't learn from flat trees)

## Priority

🔴 **CRITICAL** - Blocks all performance targets and training
