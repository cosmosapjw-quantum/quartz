"""Constructor-owned exact checks for C25 R04--R06 (not production code)."""
import json
import platform
from pathlib import Path
import sympy as s

Q = s.Rational
A = s.Matrix([[5, -1, -1, 0], [-1, 5, 0, -1], [-1, 0, 5, -1], [0, -1, -1, 5]])
Ainv = A.inv()
S = Q(1)
beta = Q(9)

def bounds(b, x):
    ell = (2 * b.dot(x) - x.dot(A * x)) / S
    d = (b - A * x).dot(b - A * x)
    return {"lo": max(Q(0), ell + d / (beta * S)),
            "hi": min(Q(1), ell + d / S), "ell": ell, "D": d}

def truth(b):
    return b.dot(Ainv * b) / S

def intersect(old, fresh):
    return [{**z, "lo": max(w["lo"], z["lo"]), "hi": min(w["hi"], z["hi"])}
            for w, z in zip(old, fresh)]

def pick_score(bs):
    return max(range(len(bs)), key=lambda a: (bs[a]["ell"], -a))

def retained(i, bs):
    j = max(range(len(bs)), key=lambda a: (bs[a]["lo"], -a))
    return j if bs[j]["lo"] > bs[i]["hi"] else i

def gap(i, bs):
    return max([Q(0)] + [bs[a]["hi"] - bs[i]["lo"] for a in range(len(bs)) if a != i])

def run(name, rhs, initial_x, next_x):
    old = [bounds(b, x) for b, x in zip(rhs, initial_x)]
    current = intersect(old, [bounds(b, x) for b, x in zip(rhs, next_x)])
    u = [truth(b) for b in rhs]
    for a in range(len(u)):
        assert old[a]["lo"] <= u[a] <= old[a]["hi"]
        assert current[a]["lo"] <= u[a] <= current[a]["hi"]
    i0 = pick_score(old)
    i1 = retained(i0, current)
    ordinary = pick_score(current)
    g0, g1 = gap(i0, old), gap(i1, current)
    assert u[i1] >= u[i0]
    assert g1 <= g0
    assert max(u) - u[i1] <= g1
    return {"name": name, "rhs": rhs, "initial_x": initial_x,
            "next_x": next_x, "true_utilities": u,
            "initial_bounds": old, "next_nested_bounds": current,
            "plain_stop_selected": i0, "next_ordinary_selected": ordinary,
            "retained_selected": i1, "retained_gain": u[i1] - u[i0],
            "ordinary_gain": u[ordinary] - u[i0],
            "retained_regret": max(u) - u[i1],
            "initial_G": g0, "retained_G": g1}

zero = s.zeros(4, 1)
b = s.Matrix([1, 0, 0, 0])
c = s.Matrix([0, Q(1, 2), Q(1, 2), 0])
d = Q(3, 4) * b
cases = [
    run("certified_switch_gain", [b, c], [zero, c / 5], [b / 5, c / 5]),
    run("retain_prevents_true_utility_drop", [b, d], [b / 20, zero], [b / 20, d / 5]),
    run("retain_rejects_uncertified_true_gain", [c, b], [c / 10, zero], [c / 10, b / 20]),
]
assert cases[0]["retained_gain"] == Q(1, 10)
assert cases[0]["retained_G"] == 0
assert cases[1]["ordinary_gain"] == -Q(23, 240)
assert cases[1]["retained_gain"] == 0
assert cases[1]["retained_G"] == Q(1, 144)
assert cases[2]["ordinary_gain"] == Q(1, 10)
assert cases[2]["retained_gain"] == 0
assert cases[2]["retained_G"] == Q(203, 360)

# Strict ties do not switch; G=0 can certify weak optimality without uniqueness.
tie_bounds = [{"lo": Q(1, 2), "hi": Q(1, 2)}, {"lo": Q(1, 2), "hi": Q(1, 2)}]
assert retained(1, tie_bounds) == 1
assert gap(1, tie_bounds) == 0

# Price predictor failure: a predictor is not a pathwise cost lower bound.
predictor = {"current_intervals": [[Q(0), Q(1)], [Q(0), Q(1)]],
             "incumbent": 0, "true_utilities": [Q(0), Q(1, 2)],
             "lambda": Q(1), "G": Q(1), "predicted_cost": Q(1),
             "true_continuation_cost": Q(1, 4), "true_continuation_gain": Q(1, 2)}
predictor["predicted_stop"] = predictor["G"] <= predictor["lambda"] * predictor["predicted_cost"]
predictor["true_net_gain"] = predictor["true_continuation_gain"] - predictor["lambda"] * predictor["true_continuation_cost"]
assert predictor["predicted_stop"] and predictor["true_net_gain"] == Q(1, 4)

# Resolve currently certified zero-charge switches before price stop.
free_bounds = [{"lo": Q(1, 5), "hi": Q(3, 10)}, {"lo": Q(2, 5), "hi": Q(1, 2)}]
assert gap(0, free_bounds) == Q(3, 10) <= Q(1, 2)
assert retained(0, free_bounds) == 1 and gap(1, free_bounds) == 0

def serial(x):
    if isinstance(x, s.MatrixBase):
        return [serial(v) for v in list(x)]
    if isinstance(x, s.Basic):
        return str(x)
    if isinstance(x, dict):
        return {k: serial(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [serial(v) for v in x]
    return x

result = {"schema": "quartz.c25.theory-constructor-check.v1",
          "status": "completed", "independence": "constructor check; independent parent verification pending",
          "assertion_tested": "Exact rational SPD examples, nested interval validity, switch gain, retention loss prevention and conservatism, price-predictor counterexample, free-switch and tie cases",
          "domain": "side=2, rho=1, S=1, alpha=1, beta=9; explicit listed rational states",
          "python": platform.python_version(), "sympy": s.__version__,
          "A": [list(A.row(k)) for k in range(4)],
          "A_eigenvalues": sorted(A.eigenvals().keys()),
          "cases": cases, "predictor_counterexample": predictor,
          "free_switch_example": {"bounds": free_bounds, "old_G": gap(0, free_bounds),
                                   "threshold": Q(1, 2), "new_incumbent": retained(0, free_bounds),
                                   "new_G": gap(1, free_bounds)},
          "tie_check": {"retained_incumbent": retained(1, tie_bounds), "G": gap(1, tie_bounds)}}
target = Path('/tmp/quartz_c25_theory_examples.json')
target.write_text(json.dumps(serial(result), indent=2) + '\n')
print(json.dumps({"status": "completed", "output": str(target),
                  "cases": [{"name": z["name"], "retained_gain": str(z["retained_gain"]),
                             "ordinary_gain": str(z["ordinary_gain"]), "G": str(z["retained_G"])}
                            for z in cases]}, indent=2))
