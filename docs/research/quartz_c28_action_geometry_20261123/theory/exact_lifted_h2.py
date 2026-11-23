#!/usr/bin/env python3
"""Exact finite oracle for Quartz C28-DG theory; no game-data or production claims.

Three independent views are compared: complete-world Bellman recursion,
Bayes-updated latent recursion, and the denominator-cancelled formula.
All load-bearing arithmetic is fractions.Fraction. No external dependencies.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from fractions import Fraction as F
from itertools import product
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import sys
import time

ZERO, ONE, HALF = F(0), F(1), F(1, 2)
STOP = -1
CHECKS: Counter[str] = Counter()


def require(condition, category, detail=""):
    CHECKS[category] += 1
    if not condition:
        raise AssertionError(f"{category}: {detail}")


@dataclass(frozen=True)
class Model:
    p: tuple[tuple[F, ...], ...]
    r: tuple[F, ...]
    cost: tuple[F, ...]
    values: tuple[F, ...] = (ZERO, HALF, ONE)

    def __post_init__(self):
        n = len(self.p)
        if len(self.r) != n or len(self.cost) != n:
            raise ValueError("candidate lengths differ")
        if len(set(self.values)) != len(self.values):
            raise ValueError("utility outcomes are not distinct")
        for row in self.p:
            if len(row) != len(self.values) or sum(row) != ONE or min(row) < ZERO:
                raise ValueError("invalid base probability row")
        if any(abs(r) >= ONE for r in self.r):
            raise ValueError("loadings must have absolute value below one")
        if any(d < ZERO for d in self.cost):
            raise ValueError("negative query cost")

    @property
    def n(self):
        return len(self.p)

    def features(self):
        rows = []
        for p in self.p:
            mu = sum(q * u for q, u in zip(p, self.values))
            scale = max(abs(u - mu) for q, u in zip(p, self.values) if q > 0)
            rows.append(tuple((u - mu) / scale if q > 0 and scale else ZERO
                              for q, u in zip(p, self.values)))
        return tuple(rows)


def check_m(m):
    if not -ONE <= m <= ONE:
        raise ValueError("latent mean outside [-1,1]")


def worlds(model, m):
    """Construct the entire joint law from its two latent product components."""
    check_m(m)
    feat = model.features()
    result = {}
    for indices in product(range(len(model.values)), repeat=model.n):
        plus, minus = ONE, ONE
        for i, k in enumerate(indices):
            p, t = model.p[i][k], model.r[i] * feat[i][k]
            plus *= p * (ONE + t)
            minus *= p * (ONE - t)
        probability = ((ONE + m) * plus + (ONE - m) * minus) / 2
        if probability:
            result[tuple(model.values[k] for k in indices)] = probability
    return result


def observe(distribution, i, outcome):
    mass = sum(p for w, p in distribution.items() if w[i] == outcome)
    if not mass:
        raise ValueError("unsupported exact observation")
    return mass, {w: p / mass for w, p in distribution.items() if w[i] == outcome}


def project(distribution, remaining):
    result = {}
    for w, p in distribution.items():
        key = tuple(w[i] for i in remaining)
        result[key] = result.get(key, ZERO) + p
    return result


def world_scores(model, distribution, b, remaining, horizon=2):
    """Brute-force Bellman, conditioning full worlds; no latent formula used."""
    result = {STOP: ZERO}
    if horizon == 0:
        return result
    for i in remaining:
        value = -model.cost[i]
        rest = tuple(j for j in remaining if j != i)
        for u in model.values:
            mass = sum(p for w, p in distribution.items() if w[i] == u)
            if not mass:
                continue
            _, posterior = observe(distribution, i, u)
            new_b = max(b, u)
            after = world_scores(model, posterior, new_b, rest, horizon - 1)
            value += mass * (new_b + max(after.values()))
        result[i] = value - b
    return result


def coefficients(model, i, b):
    feat = model.features()[i]
    base = sum(p * max(u - b, ZERO) for p, u in zip(model.p[i], model.values))
    slope = model.r[i] * sum(p * f * max(u - b, ZERO)
                            for p, f, u in zip(model.p[i], feat, model.values))
    return base - model.cost[i], slope


def h1_scores(model, m, b, remaining):
    result = {STOP: ZERO}
    for i in remaining:
        a, beta = coefficients(model, i, b)
        result[i] = a + m * beta
    return result


def bayes_scores(model, m, b, remaining):
    """Latent Bayes update followed by independently evaluated H1 scores."""
    result = {STOP: ZERO}
    feat = model.features()
    for i in remaining:
        a, beta = coefficients(model, i, b)
        score = a + m * beta
        rest = tuple(j for j in remaining if j != i)
        for k, u in enumerate(model.values):
            if not model.p[i][k]:
                continue
            t = model.r[i] * feat[i][k]
            denom = ONE + m * t
            if denom <= ZERO:
                raise AssertionError("nonpositive supported likelihood")
            updated = (m + t) / denom
            score += model.p[i][k] * denom * max(
                h1_scores(model, updated, max(b, u), rest).values())
        result[i] = score
    return result


def lifted_scores(model, m, b, remaining):
    """H2 formula with Bayes likelihood denominators cancelled before max."""
    result = {STOP: ZERO}
    feat = model.features()
    for i in remaining:
        a, beta = coefficients(model, i, b)
        score = a + m * beta
        for k, u in enumerate(model.values):
            if not model.p[i][k]:
                continue
            t = model.r[i] * feat[i][k]
            branches = [ZERO]
            for j in remaining:
                if i == j:
                    continue
                aj, bj = coefficients(model, j, max(b, u))
                branches.append((ONE + m * t) * aj + (m + t) * bj)
            score += model.p[i][k] * max(branches)
        result[i] = score
    return result


def choose(scores):
    """STOP wins ties, then the smallest candidate index."""
    return min(scores, key=lambda i: (-scores[i], i))


def family(r=F(4, 5), c=F(1, 100), d=F(3, 10), sign=-1):
    return Model(((HALF, HALF, ZERO), (HALF, ZERO, HALF), (HALF, ZERO, HALF)),
                 (r, r, sign * r), (c, d, d))


def family_scores(r, c, d, sign):
    x, y = F(1, 4) - d, r * r / 4
    if sign == -1:
        a = -c + max(x + y, ZERO)
        target = x + max(x + y, ZERO) / 2
    else:
        a = -c + (max(x + y, ZERO) + max(x - y, ZERO)) / 2
        target = x + max(x - y, ZERO) / 2
    return {STOP: ZERO, 0: a, 1: target, 2: target}


def mutual_information_a_targets(distribution):
    pa, py = {}, {}
    for w, p in distribution.items():
        pa[w[0]] = pa.get(w[0], ZERO) + p
        py[w[1:]] = py.get(w[1:], ZERO) + p
    return sum(float(p) * math.log(float(p / (pa[w[0]] * py[w[1:]])))
               for w, p in distribution.items())


def plan_lines(model, b, remaining):
    """Enumerate deterministic two-query contingent plans at m=+/-1."""
    wm, wp = worlds(model, -ONE), worlds(model, ONE)
    lines = [(ZERO, ZERO, STOP, ())]
    for i in remaining:
        observations = tuple(u for k, u in enumerate(model.values) if model.p[i][k])
        options = (STOP,) + tuple(j for j in remaining if j != i)
        for choices in product(options, repeat=len(observations)):
            continuation = dict(zip(observations, choices))
            def value(distribution):
                out = -model.cost[i] - b
                for w, p in distribution.items():
                    new_b = max(b, w[i])
                    j = continuation[w[i]]
                    terminal = new_b if j == STOP else max(new_b, w[j]) - model.cost[j]
                    out += p * terminal
                return out
            minus, plus = value(wm), value(wp)
            lines.append(((plus + minus) / 2, (plus - minus) / 2, i, choices))
    return lines


def check_model(model, ms, bs):
    feat = model.features()
    all_indices = tuple(range(model.n))
    for i in all_indices:
        require(sum(p * f for p, f in zip(model.p[i], feat[i])) == 0, "centering")
        require(all(abs(f) <= 1 for f in feat[i]), "bounded_features")
    for m in ms:
        dist = worlds(model, m)
        require(sum(dist.values()) == 1 and min(dist.values()) > 0, "joint_probability")
        flipped = replace(model, r=tuple(-r for r in model.r))
        require(worlds(flipped, -m) == dist, "global_latent_label_invariance")
        for i in all_indices:
            mean_i = sum(p * w[i] for w, p in dist.items())
            delta_i = sum(p * u * f for p, u, f in zip(model.p[i], model.values, feat[i]))
            for j in all_indices:
                if j <= i:
                    continue
                mean_j = sum(p * w[j] for w, p in dist.items())
                delta_j = sum(p * u * f for p, u, f in zip(model.p[j], model.values, feat[j]))
                cov = sum(p * w[i] * w[j] for w, p in dist.items()) - mean_i * mean_j
                require(cov == model.r[i] * model.r[j] * delta_i * delta_j * (1 - m*m),
                        "signed_covariance")
            for k, u in enumerate(model.values):
                if not model.p[i][k]:
                    continue
                mass, observed = observe(dist, i, u)
                t = model.r[i] * feat[i][k]
                updated = (m + t) / (1 + m*t)
                require(mass == model.p[i][k] * (1 + m*t), "predictive_probability")
                rest = tuple(j for j in all_indices if j != i)
                require(project(observed, rest) == project(worlds(model, updated), rest),
                        "posterior_complete_law")
        for b in bs:
            brute = world_scores(model, dist, b, all_indices)
            direct = bayes_scores(model, m, b, all_indices)
            lifted = lifted_scores(model, m, b, all_indices)
            require(brute == direct == lifted, "h2_all_action_scores",
                    f"model={model},m={m},b={b},world={brute},lifted={lifted}")
            require(choose(brute) == choose(lifted), "h2_tie_preserving_policy")
            require(max(lifted.values()) >= 0, "known_incumbent_retention")
            last = world_scores(model, dist, b, all_indices, horizon=1)
            h1 = h1_scores(model, m, b, all_indices)
            require(last == h1, "h1_affine")
            selected = choose(h1)
            slope = {STOP: ZERO, **{i: coefficients(model, i, b)[1] for i in all_indices}}
            for new_m in ms:
                delta = new_m - m
                gaps = {j: h1[selected] - h1[j] + (slope[selected]-slope[j])*delta
                        for j in h1 if j != selected}
                certificate = all(gap > 0 or (gap == 0 and selected < j)
                                  for j, gap in gaps.items())
                actual = choose(h1_scores(model, new_m, b, all_indices)) == selected
                require(certificate == actual, "h1_exact_action_cell_with_ties")
                old_gaps = [h1[selected] - h1[j] for j in h1 if j != selected]
                if all(g > 0 for g in old_gaps):
                    ratios = [(h1[selected]-h1[j])/abs(slope[selected]-slope[j])
                              for j in h1 if j != selected and slope[selected] != slope[j]]
                    if not ratios or abs(delta) < min(ratios):
                        require(actual, "strict_margin_radius")
    # Endpoint-midpoint convexity for every fixed first action, including STOP.
    for b in bs:
        left = lifted_scores(model, -ONE, b, all_indices)
        middle = lifted_scores(model, ZERO, b, all_indices)
        right = lifted_scores(model, ONE, b, all_indices)
        for i in middle:
            require(2 * middle[i] <= left[i] + right[i], "h2_action_convexity")
        require(2*max(middle.values()) <= max(left.values())+max(right.values()),
                "h2_optimum_convexity")
    # Units: a positive affine utility change scales every gain and cost.
    scale, shift = F(7, 3), F(-2, 5)
    transformed = replace(model, values=tuple(scale*u+shift for u in model.values),
                          cost=tuple(scale*d for d in model.cost))
    original = lifted_scores(model, F(2, 5), HALF, all_indices)
    scaled = lifted_scores(transformed, F(2, 5), scale*HALF+shift, all_indices)
    require(scaled == {i: scale*q for i, q in original.items()}, "positive_affine_units")


def main():
    started = time.perf_counter()
    seed = 2807
    rng = random.Random(seed)
    ms = (F(-1), F(-3, 4), F(0), F(2, 5), F(1))
    bs = (F(0), F(1, 3), F(1, 2), F(1))
    loadings = (F(-4, 5), F(-1, 2), F(0), F(1, 2), F(4, 5))
    costs = (F(0), F(1, 100), F(1, 20), F(1, 4), F(3, 10), F(1, 2))
    models = []
    for n in range(1, 5):
        for _ in range(12):
            rows = []
            for _i in range(n):
                weights = [rng.randrange(5) for _ in range(3)]
                if not sum(weights):
                    weights[rng.randrange(3)] = 1
                rows.append(tuple(F(w, sum(weights)) for w in weights))
            models.append(Model(tuple(rows), tuple(rng.choice(loadings) for _ in range(n)),
                                tuple(rng.choice(costs) for _ in range(n))))
    # Explicit degeneracy, certain latent mode, zero costs, and independent model controls.
    models.extend([
        Model(((ONE,ZERO,ZERO),(ZERO,ONE,ZERO),(ZERO,ZERO,ONE)),
              (F(-4,5),F(4,5),F(1,2)),(ZERO,ZERO,ZERO)),
        Model(((F(1,3),)*3,)*3,(ZERO,ZERO,ZERO),(F(1,10),)*3),
        family(sign=-1),family(sign=1),family(r=ZERO),
    ])
    for model in models:
        check_model(model, ms, bs)
    family_count = 0
    for r,c,d,sign in product((ZERO,F(1,4),HALF,F(3,4),F(4,5),F(9,10)),
                              (ZERO,F(1,100),F(1,20),F(1,10),F(1,4)),
                              (ZERO,F(1,8),F(1,4),F(3,10),HALF),(-1,1)):
        model = family(r,c,d,sign)
        direct = world_scores(model,worlds(model,ZERO),HALF,(0,1,2))
        closed = family_scores(r,c,d,sign)
        require(direct == closed, "signed_common_complete_phase_formula")
        family_count += 1
    signed, common = family(sign=-1), family(sign=1)
    ws,wc = worlds(signed,ZERO),worlds(common,ZERO)
    transformed = {(a,b,ONE-c):p for (a,b,c),p in wc.items()}
    require(ws == transformed, "information_preserving_target_bijection")
    require(mutual_information_a_targets(ws) == mutual_information_a_targets(wc),
            "mutual_information_numeric_corollary")
    example = {}
    for name,model in (("signed",signed),("common",common),("product",family(r=ZERO))):
        q = world_scores(model,worlds(model,ZERO),HALF,(0,1,2))
        example[name] = {"q": {str(i):str(v) for i,v in q.items()},
                         "value":str(HALF+max(q.values())),"first_action":choose(q)}
    require(example["signed"]["value"] == "3/5", "reference_signed_value")
    require(example["common"]["value"] == "109/200", "reference_common_value")
    require(example["product"]["value"] == "1/2", "reference_product_value")
    # Full contingent-plan alpha lines are an independent PW-affine representation.
    lines = plan_lines(signed,HALF,(0,1,2))
    special_m = (F(-1),F(-7,8),F(-19,24),F(-11,16),ZERO,
                 F(11,16),F(19,24),F(7,8),ONE)
    boundary_records = []
    for m in special_m + tuple(F(k,48) for k in range(-48,49)):
        q = lifted_scores(signed,m,HALF,(0,1,2))
        plan_scores = {i:max(a+m*beta for a,beta,j,_ in lines if j == i)
                       for i in (STOP,0,1,2)}
        require(q == plan_scores,"contingent_plan_alpha_representation")
        expected = 2 if m < -F(19,24) else 1 if m > F(19,24) else 0
        require(choose(q) == expected,"signed_action_boundary_19_over_24")
        if m in special_m:
            boundary_records.append({"m":str(m),"q":{str(i):str(v) for i,v in q.items()},
                                     "first_action":choose(q)})
    for model in models[:12]:
        for invalid in (F(-1),F(1)):
            try:
                replace(model,r=(invalid,)+model.r[1:])
            except ValueError:
                require(True,"reject_closed_loading_endpoint")
            else:
                require(False,"reject_closed_loading_endpoint")
    report = {
        "schema":"quartz-c28-dg-exact-theory-v1",
        "ssot":"87c15dc828d6eea519c219fd8a412fa21dfc464d",
        "status":"EXACT_FINITE_CHECKS_PASSED",
        "python":sys.version,"platform":platform.platform(),"random_seed":seed,
        "general_models":len(models),"latent_means":[str(x) for x in ms],
        "incumbents":[str(x) for x in bs],"closed_family_cases":family_count,
        "alpha_plan_lines":len(lines),"assertions":sum(CHECKS.values()),
        "assertions_by_category":dict(sorted(CHECKS.items())),
        "example":example,"example_information_nats":mutual_information_a_targets(ws),
        "signed_action_boundary_records":boundary_records,
        "seconds":time.perf_counter()-started,
        "runner_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "limits":["This finite run does not prove a universal theorem.",
                  "No native production controller or game-performance campaign was executed.",
                  "Floating mutual information is only a diagnostic; exact bijection proves equality.",
                  "Deterministic tie order is STOP then smallest candidate index."]}
    output = Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).with_name("exact_theory_results.json")
    output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"status":report["status"],"assertions":report["assertions"],
                      "general_models":len(models),"closed_family_cases":family_count,
                      "seconds":report["seconds"],"output":str(output)},indent=2))


if __name__ == "__main__":
    main()
