"""Exact finite audit of C27 binary-latent posterior and depth-two value."""

import json
import platform
import sys
from fractions import Fraction as F
from functools import cache
from itertools import combinations, product
from pathlib import Path

U = (F(0), F(1, 2), F(1))
COUNTS = {}


def check(name, condition):
    if not condition:
        raise AssertionError(name)
    COUNTS[name] = COUNTS.get(name, 0) + 1


def normal(dist):
    total = sum(dist.values(), F(0))
    if total <= 0:
        raise ValueError("zero probability observation")
    return tuple(sorted((w, p / total) for w, p in dist.items() if p))


def moments(law):
    mu = sum((u * p for u, p in zip(U, law)), F(0))
    scale = max(abs(u - mu) for u, p in zip(U, law) if p)
    score = tuple((u - mu) / scale if scale else F(0) for u in U)
    var = sum((p * (u - mu) ** 2 for u, p in zip(U, law)), F(0))
    delta = var / scale if scale else F(0)
    return mu, scale, score, delta


def components(laws, rho):
    return tuple(
        tuple(
            tuple(p * (1 + z * rho * f) for p, f in zip(law, moments(law)[2]))
            for law in laws
        )
        for z in (1, -1)
    )


def joint(laws, rho):
    comp = components(laws, rho)
    dist = {}
    for inds in product(range(3), repeat=len(laws)):
        weight = F(0)
        for mode in comp:
            term = F(1, 2)
            for a, k in enumerate(inds):
                term *= mode[a][k]
            weight += term
        if weight:
            dist[tuple(U[k] for k in inds)] = weight
    check("joint_normalization", sum(dist.values(), F(0)) == 1)
    return tuple(sorted(dist.items()))


def predictive(dist, a):
    return tuple(sum((p for w, p in dist if w[a] == u), F(0)) for u in U)


def observe(dist, a, u):
    return normal({w: p for w, p in dist if w[a] == u})


def update(m, rho, f):
    denom = 1 + rho * m * f
    if denom <= 0:
        raise ValueError("zero probability observation")
    return (m + rho * f) / denom


def latent_predictive(law, rho, m):
    return tuple(p * (1 + rho * m * f) for p, f in zip(law, moments(law)[2]))


def gain(law, b):
    return sum((p * max(F(0), u - b) for p, u in zip(law, U)), F(0))


@cache
def bellman(dist, costs, b, remaining, horizon):
    if not remaining or horizon == 0:
        return b
    best = b
    for a in remaining:
        tail = tuple(j for j in remaining if j != a)
        value = -costs[a]
        for u, p in zip(U, predictive(dist, a)):
            if p:
                value += p * bellman(
                    observe(dist, a, u), costs, max(b, u), tail, horizon - 1
                )
        best = max(best, value)
    return best


def h2_closed(laws, rho, m, costs, b, remaining):
    values = []
    for a in remaining:
        pa = latent_predictive(laws[a], rho, m)
        qa = gain(pa, b) - costs[a]
        continuation = F(0)
        for k, (u, p) in enumerate(zip(U, pa)):
            if not p:
                continue
            mp = update(m, rho, moments(laws[a])[2][k])
            bp = max(b, u)
            option = max(
                [F(0)]
                + [
                    gain(latent_predictive(laws[j], rho, mp), bp) - costs[j]
                    for j in remaining
                    if j != a
                ]
            )
            continuation += p * option
        values.append(qa + continuation)
    return b + max([F(0)] + values)


def option_correction(dist, costs, b, a):
    remaining = tuple(j for j in range(len(costs)) if j != a)
    before = {j: predictive(dist, j) for j in remaining}
    corr = F(0)
    bound = F(0)
    for u, p in zip(U, predictive(dist, a)):
        if not p:
            continue
        bp = max(b, u)
        post = observe(dist, a, u)
        after = {j: predictive(post, j) for j in remaining}
        updated = max([F(0)] + [gain(after[j], bp) - costs[j] for j in remaining])
        frozen = max([F(0)] + [gain(before[j], bp) - costs[j] for j in remaining])
        tv = max(
            [F(0)]
            + [
                sum(abs(x - y) for x, y in zip(before[j], after[j])) / 2
                for j in remaining
            ]
        )
        corr += p * (updated - frozen)
        bound += p * (1 - bp) * tv
    return corr, bound


def policy_action(belief, costs, b, remaining, horizon):
    best = b
    choice = None
    for a in remaining:
        tail = tuple(j for j in remaining if j != a)
        value = -costs[a]
        for u, p in zip(U, predictive(belief, a)):
            if p:
                value += p * bellman(
                    observe(belief, a, u), costs, max(b, u), tail, horizon - 1
                )
        if value > best:
            best, choice = value, a
    return choice


def evaluate_under(true, belief, costs, b, remaining, horizon):
    if not remaining or horizon == 0:
        return b
    a = policy_action(belief, costs, b, remaining, horizon)
    if a is None:
        return b
    value = -costs[a]
    tail = tuple(j for j in remaining if j != a)
    for u, p in zip(U, predictive(true, a)):
        if p:
            value += p * evaluate_under(
                observe(true, a, u),
                observe(belief, a, u),
                costs,
                max(b, u),
                tail,
                horizon - 1,
            )
    return value


def tv_dist(a, b):
    x, y = dict(a), dict(b)
    return sum(abs(x.get(w, F(0)) - y.get(w, F(0))) for w in x.keys() | y.keys()) / 2


def run():
    laws = [
        (F(1, 3), F(1, 3), F(1, 3)),
        (F(1, 2), F(0), F(1, 2)),
        (F(1, 2), F(1, 2), F(0)),
        (F(0), F(1, 2), F(1, 2)),
        (F(1), F(0), F(0)),
        (F(0), F(0), F(1)),
    ]
    models = [
        (ls, r)
        for ls in product(laws, repeat=2)
        for r in (F(0), F(1, 3), F(2, 3), F(9, 10))
    ]
    models += [
        (ls, r) for ls in product(laws[:4], repeat=3) for r in (F(0), F(1, 2), F(9, 10))
    ]
    h2_cases = posterior_histories = conditional_h2_cases = 0
    for ls, rho in models:
        dist = joint(ls, rho)
        n = len(ls)
        for a in range(n):
            check("initial_marginal_identity", predictive(dist, a) == ls[a])
        for a, b in combinations(range(n), 2):
            ma, _, _, da = moments(ls[a])
            mb, _, _, db = moments(ls[b])
            cov = sum((p * (w[a] - ma) * (w[b] - mb) for w, p in dist), F(0))
            check("rank_one_covariance", cov == rho * rho * da * db)
        for length in range(min(2, n - 1) + 1):
            for inds in combinations(range(n), length):
                for outs in product(U, repeat=length):
                    event = {
                        w: p
                        for w, p in dist
                        if all(w[a] == u for a, u in zip(inds, outs))
                    }
                    if not event:
                        continue
                    post = normal(event)
                    m = F(0)
                    for a, u in zip(inds, outs):
                        m = update(m, rho, moments(ls[a])[2][U.index(u)])
                    mr = F(0)
                    for a, u in reversed(tuple(zip(inds, outs))):
                        mr = update(mr, rho, moments(ls[a])[2][U.index(u)])
                    check("observation_order_invariance", m == mr)
                    for j in range(n):
                        if j not in inds:
                            check(
                                "posterior_equals_joint_enumeration",
                                predictive(post, j) == latent_predictive(ls[j], rho, m),
                            )
                    rem = tuple(j for j in range(n) if j not in inds)
                    costs = tuple(F(j + 1, 10) for j in range(n))
                    incumbent = max((F(1, 2),) + outs)
                    conditional_value = bellman(post, costs, incumbent, rem, 2)
                    check(
                        "conditional_h2_after_initial_reveals",
                        h2_closed(ls, rho, m, costs, incumbent, rem)
                        == conditional_value,
                    )
                    conditional_laws = tuple(predictive(post, j) for j in range(n))
                    productized = joint(conditional_laws, F(0))
                    for j in rem:
                        check(
                            "post_initial_product_control_same_marginals",
                            predictive(productized, j) == predictive(post, j),
                        )
                    for a, j in combinations(rem, 2):
                        ma = sum((u * p for u, p in zip(U, predictive(post, a))), F(0))
                        mj = sum((u * p for u, p in zip(U, predictive(post, j))), F(0))
                        conditional_cov = sum(
                            (p * (w[a] - ma) * (w[j] - mj) for w, p in post), F(0)
                        )
                        check(
                            "posterior_rank_one_covariance",
                            conditional_cov
                            == rho
                            * rho
                            * (1 - m * m)
                            * moments(ls[a])[3]
                            * moments(ls[j])[3],
                        )
                    conditional_h2_cases += 1
                    posterior_histories += 1
        costsets = [
            tuple(F(a + 1, 100) for a in range(n)),
            tuple(F(a + 1, 10) for a in range(n)),
            (F(1, 2),) * n,
        ]
        for costs, b in product(costsets, U):
            rem = tuple(range(n))
            v1 = bellman(dist, costs, b, rem, 1)
            v2 = bellman(dist, costs, b, rem, 2)
            check(
                "h2_latent_equals_joint_bellman",
                h2_closed(ls, rho, F(0), costs, b, rem) == v2,
            )
            check("horizon_and_stop_monotonicity", v2 >= v1 >= b)
            check("known_win_stops", b != 1 or v2 == 1)
            for a in range(n):
                corr, bound = option_correction(dist, costs, b, a)
                check("posterior_option_tv_bound", abs(corr) <= bound)
                if all(w[a] <= b for w, p in dist):
                    check("information_only_option_nonnegative", corr >= 0)
            h2_cases += 1
        bellman.cache_clear()
    # New parameter family: same exact marginals, pure information A, correlated B.
    info_laws = (laws[2], laws[1])
    rows = []
    for rho, eps, c in product(
        (F(0), F(1, 4), F(1, 2), F(3, 4), F(9, 10), F(1)),
        (F(0), F(1, 100), F(1, 20), F(1, 10), F(1, 5)),
        (F(0), F(1, 1000), F(1, 100), F(1, 20)),
    ):
        costs = (c, F(1, 4) + eps)
        dist = joint(info_laws, rho)
        exact = bellman(dist, costs, F(1, 2), (0, 1), 2)
        formula = F(1, 2) + max(F(0), -c + max(F(0), rho * rho / F(4) - eps) / 2)
        check("new_information_family_formula", exact == formula)
        check(
            "new_information_family_threshold",
            (exact > F(1, 2)) == (rho * rho > 4 * eps + 8 * c),
        )
        rows.append(
            {
                "rho": str(rho),
                "epsilon": str(eps),
                "c": str(c),
                "value": str(exact),
                "profitable": exact > F(1, 2),
            }
        )
    example_rho = F(4, 5)
    example_costs = (F(1, 100), F(3, 10))
    example = joint(info_laws, example_rho)
    check(
        "new_example_109_over_200",
        bellman(example, example_costs, F(1, 2), (0, 1), 2) == F(109, 200),
    )
    check(
        "new_example_product_stop",
        bellman(joint(info_laws, F(0)), example_costs, F(1, 2), (0, 1), 2) == F(1, 2),
    )
    # Negative correction with moving incumbent: this is not a causal EVSI comparison.
    negdist = joint((laws[1], laws[1]), F(9, 10))
    corr, bound = option_correction(negdist, (F(1, 10), F(1, 10)), F(0), 0)
    check("moving_incumbent_correction_can_be_negative", corr == F(-1, 5))
    # Existing exact correlated benchmark requires a general joint law, beyond positive-common-strength family.
    old = (((F(0), F(1), F(0)), F(1, 2)), ((F(1, 2), F(0), F(1)), F(1, 2)))
    check(
        "existing_correlated_069_benchmark",
        bellman(old, (F(1, 100), F(3, 10), F(3, 10)), F(1, 2), (0, 1, 2), 2)
        == F(69, 100),
    )
    # Prior misspecification: optimize under Q, score same policy under P.
    transfer_cases = 0
    for ls in [(laws[0], laws[1]), (laws[2], laws[1]), (laws[0], laws[1], laws[3])]:
        for rp, rq in product((F(0), F(1, 2), F(9, 10)), repeat=2):
            p, q = joint(ls, rp), joint(ls, rq)
            n = len(ls)
            rem = tuple(range(n))
            costs = tuple(F(j + 1, 10) for j in range(n))
            for b in U:
                vp = bellman(p, costs, b, rem, 2)
                vq = bellman(q, costs, b, rem, 2)
                deployed = evaluate_under(p, q, costs, b, rem, 2)
                width = 1 - b + sum(sorted(costs, reverse=True)[:2])
                delta = tv_dist(p, q)
                check(
                    "optimal_value_total_variation_bound", abs(vp - vq) <= width * delta
                )
                check(
                    "transferred_policy_regret_bound",
                    F(0) <= vp - deployed <= 2 * width * delta,
                )
                transfer_cases += 1
    result = {
        "schema": "c27-theory-finite-v1",
        "assertion_tested": "Exact finite joint normalization, unchanged initial marginals, latent Bayes update and observation order, rank-one covariance, closed H2 versus independently enumerated joint Bellman, conditional-option and TV bounds, parameter-family threshold, and old correlated fixture.",
        "coefficient_domain": "fractions.Fraction exact rational arithmetic; no floating tolerance or RNG",
        "counts": {
            "joint_model_cases": len(models),
            "posterior_history_cases": posterior_histories,
            "h2_cases": h2_cases,
            "conditional_h2_cases": conditional_h2_cases,
            "new_family_cases": len(rows),
            "policy_transfer_cases": transfer_cases,
            "assertions": sum(COUNTS.values()),
        },
        "check_counts": COUNTS,
        "new_information_family": {
            "b": "1/2",
            "laws": [[str(x) for x in law] for law in info_laws],
            "d_A": "c",
            "d_B": "1/4+epsilon",
            "condition": "rho^2 > 4 epsilon + 8 c for c>=0,epsilon>=0",
            "formula": "V2=1/2+max(0,-c+max(0,rho^2/4-epsilon)/2)",
            "example": {
                "rho": "4/5",
                "epsilon": "1/20",
                "c": "1/100",
                "joint_value": "109/200",
                "product_value": "1/2",
                "joint_worlds": [
                    {"u": [str(x) for x in w], "p": str(p)} for w, p in example
                ],
            },
            "grid": rows,
        },
        "moving_incumbent_negative_correction": {
            "rho": "9/10",
            "b": "0",
            "costs": ["1/10", "1/10"],
            "correction": str(corr),
            "bound": str(bound),
        },
        "software": {"python": platform.python_version()},
        "non_claims": [
            "No theorem follows from the finite count alone; direct proofs are in the companion report.",
            "No game training, CAL, TEST or runtime controller was executed.",
            "Centered common-strength family only models nonnegative offdiagonal utility covariances.",
            "The query-cost Bellman excludes its own planner CPU overhead.",
            "Terminal deployment is restricted to exact-known values; this is not posterior-mean selection of unqueried candidates.",
        ],
    }
    output = Path(sys.argv[1])
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "status": "all finite assertions passed",
                "counts": result["counts"],
                "new_example": "109/200 versus product1/2",
                "existing_example": "69/100",
                "moving_incumbent_correction": str(corr),
            }
        )
    )


if __name__ == "__main__":
    run()
