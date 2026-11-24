"""World enumeration versus latent-factor CDF with exact rational arithmetic."""
from fractions import Fraction as F
from itertools import product
from math import prod
from random import Random


def cdf_value(q, factors, b):
    p0, ph = (
        sum(q[z] * prod(sum(row[z][:k]) for row in factors) for z in range(2))
        for k in (1, 2)
    )
    return 1 - max(0, F(1, 2) - b) * p0 - (1 - max(b, F(1, 2))) * ph


def worlds(q, factors, b):
    return sum(
        max((b, *(F(u, 2) for u in path)))
        * sum(q[z] * prod(row[z][u] for row, u in zip(factors, path)) for z in range(2))
        for path in product(range(3), repeat=len(factors))
    )


if __name__ == "__main__":
    rng = Random(20261124)
    for trial in range(192):
        def law(n):
            counts = [rng.randrange(5) for _ in range(n)]
            counts[0] += not any(counts)
            return tuple(F(x, sum(counts)) for x in counts)
        q = law(2)
        factors = [(law(3), law(3)) for _ in range(trial % 7)]
        b = F(trial % 9, 8)
        assert cdf_value(q, factors, b) == worlds(q, factors, b)
    print("192 exact Fraction CDF/world equalities")
