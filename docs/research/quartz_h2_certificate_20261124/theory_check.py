"""Reproduce the exact one-candidate H2 continuation example."""
from fractions import Fraction as F


def value(p, horizon):
    stop = max(F(3, 5), p)
    if not horizon:
        return stop
    positive = (1 + 2 * p) / 4
    negative = 1 - positive
    continuation = (
        positive * value(3 * p / (4 * positive), horizon - 1)
        + negative * value(p / (4 * negative), horizon - 1)
        - F(1, 100)
    )
    return max(stop, continuation)


if __name__ == "__main__":
    observed = tuple(value(F(1, 2), h) for h in range(3))
    assert observed == (F(3, 5), F(133, 200), F(543, 800))
    assert observed[2] - observed[1] == F(11, 800)
    assert value(F(3, 4), 1) == F(311, 400)
    assert value(F(1, 4), 1) == F(3, 5)
    print(", ".join(map(str, observed)), "increment", observed[2] - observed[1])
