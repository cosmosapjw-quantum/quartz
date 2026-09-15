"""Exact prior-predictive Gaussian diagnostic; not a game or timing benchmark.

theta ~ N(0,1). The observation mean has noise variance R_true. An iid
likelihood wrongly assumes R_iid=1/n for shared error and exact replays.
No random simulation, fitted model, or adaptive stopping is used.
"""
import argparse
import json
import math
from pathlib import Path
from statistics import NormalDist

BUDGETS = (1, 8, 64, 512, 4096)
CASES = ("exact_replay", "shared_error", "independent")


def evaluate_case(n: int, kind: str) -> dict:
    if type(n) is not int or n < 1:
        raise ValueError("n must be a positive integer")
    if kind not in CASES:
        raise ValueError("unknown observation design")
    true_noise = {"exact_replay": 1.0, "shared_error": .25 + .75/n,
                  "independent": 1.0/n}[kind]
    assumed_noise = 1.0/n
    correct_var = true_noise/(1.0 + true_noise)
    iid_var = assumed_noise/(1.0 + assumed_noise)
    iid_weight = 1.0/(1.0 + assumed_noise)
    actual_mse = (1.0-iid_weight)**2 + iid_weight**2 * true_noise
    z = NormalDist().inv_cdf(.975)
    coverage = math.erf(z * math.sqrt(iid_var/(2.0*actual_mse)))
    return {
        "case": kind, "n_deliveries_or_draws": n,
        "distinct_observations": 1 if kind == "exact_replay" else n,
        "true_mean_noise_variance": true_noise,
        "correct_posterior_variance": correct_var,
        "iid_posterior_variance": iid_var,
        "iid_actual_mse": actual_mse,
        "iid_prior_predictive_coverage": coverage,
        "correct_prior_predictive_coverage": .95,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {
        "evidence_kind": "exact_model_diagnostic",
        "prior_variance": 1.0, "nominal_interval_mass": .95,
        "shared_error_variance": .25, "fresh_noise_variance": .75,
        "budgets": list(BUDGETS),
        "rows": [evaluate_case(n, kind) for kind in CASES for n in BUDGETS],
        "limitations": ["fixed_n", "known_gaussian_model", "prior_predictive_coverage",
                        "not_fixed_theta_frequentist_coverage", "not_engine_performance",
                        "distinct_ids_do_not_certify_independence"],
    }
    # Exclusive creation prevents accidental overwrite of a completed diagnostic.
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write("\n")
    print(f"wrote {len(result['rows'])} exact diagnostic rows to {args.output}")


if __name__ == "__main__":
    main()
