"""Finite reference planner (red scaffold before implementation)."""
def plan(state, row, costs, horizon):
    return max(row(c)[0] for c in state), None
