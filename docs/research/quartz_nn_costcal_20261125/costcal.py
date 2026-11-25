"""Red scaffold for decoupled cost and whole-pool conformal rank."""
def point_cost(rows):
    return [1,1,1]
def envelope(scores):
    return (1,1.)
def tables(gain,cost,a):
    return [[[g,1.] for g in row] for row in gain]
