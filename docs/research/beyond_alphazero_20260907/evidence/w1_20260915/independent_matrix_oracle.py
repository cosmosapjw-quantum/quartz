"""Exact rational verification from the full prior-predictive covariance.

Independent of the W1 diagnostic implementation: imports only fractions and
solves the observation covariance system using Gauss-Jordan elimination.
The selected small n values keep this deliberately cubic verifier inexpensive.
"""

from fractions import Fraction as F


def solve(a, b):
    m = [row[:] + [rhs] for row, rhs in zip(a, b)]
    n = len(m)
    for j in range(n):
        p = next(i for i in range(j, n) if m[i][j])
        m[j], m[p] = m[p], m[j]
        pivot = m[j][j]
        m[j] = [x / pivot for x in m[j]]
        for i in range(n):
            if i != j:
                mult = m[i][j]
                m[i] = [x - mult * y for x, y in zip(m[i], m[j])]
    return [row[-1] for row in m]


def main():
    # theta ~ N(0, 1), shared bias variance 1/4, fresh noise variance 3/4.
    # Cov(Y) = (1 + 1/4) 11' + (3/4) I, Cov(theta, Y) = 1'.
    for n in (1, 2, 3, 8):
        covariance = [
            [F(5, 4) + (F(3, 4) if i == j else 0) for j in range(n)]
            for i in range(n)
        ]
        weights = solve(covariance, [F(1)] * n)
        matrix_variance = F(1) - sum(weights)
        closed_variance = F(n + 3, 5 * n + 3)
        naive_weights = [F(1, n + 1)] * n
        error = (
            F(1)
            - 2 * sum(naive_weights)
            + sum(
                naive_weights[i] * covariance[i][j] * naive_weights[j]
                for i in range(n)
                for j in range(n)
            )
        )
        if matrix_variance != closed_variance:
            raise AssertionError((n, matrix_variance, closed_variance))
        if error != F(n * n + 3 * n + 4, 4 * (n + 1) ** 2):
            raise AssertionError((n, error))
        print(
            "n", n,
            "conditional variance", matrix_variance,
            "naive prior predictive MSE", error,
            "exact matrix checks passed",
        )


if __name__ == "__main__":
    main()
