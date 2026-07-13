#!/usr/bin/env python3
"""Low-budget Bernoulli root-selection laboratory.

The laboratory compares three allocation rules under a fixed pull budget:

* ``uniform``: randomized round-robin allocation;
* ``raw_sequential_halving``: equal allocation within successive survivor sets,
  with elimination by raw sample mean;
* ``kg_rank_risk``: exact one-step Beta-Bernoulli knowledge gradient when it is
  positive, with a fixed incumbent/challenger risk fallback when all
  one-step knowledge gradients are zero.

The code intentionally makes no claim that the third rule is uniformly better
or has a new regret guarantee. Its purpose is to test a decision-relevant
ranking-risk computation rule with no fitted scalar exploration coefficient in
the 8--64 pull regime. Candidate-omission risk is outside this laboratory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableSequence, Optional, Sequence, Tuple


ALGORITHMS: Tuple[str, ...] = (
    "uniform",
    "raw_sequential_halving",
    "kg_rank_risk",
)
MAX_SUPPORTED_BUDGET = 256
EXPERIMENT_ID = "bernoulli_root_ranking_risk_v1"
EXECUTION_MODE = "synthetic_screening"
ALGORITHM_CONTRACTS = {
    "uniform": {
        "family": "randomized_round_robin",
        "uses_true_means": False,
        "fitted_scalar_exploration_coefficient": False,
    },
    "raw_sequential_halving": {
        "family": "raw_mean_elimination",
        "gumbel_alphazero_equivalent": False,
        "uses_true_means": False,
        "fitted_scalar_exploration_coefficient": False,
    },
    "kg_rank_risk": {
        "family": "one_step_beta_bernoulli_kg_with_rank_fallback",
        "candidate_omission_modelled": False,
        "uses_true_means": False,
        "fitted_scalar_exploration_coefficient": False,
        "fixed_prior": [1, 1],
    },
}


def stable_seed(*parts: object) -> int:
    """Return a deterministic seed independent of Python's hash randomization."""

    payload = ":".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def beta_mean(alpha: int, beta: int) -> float:
    return alpha / (alpha + beta)


def beta_variance(alpha: int, beta: int) -> float:
    total = alpha + beta
    return (alpha * beta) / (total * total * (total + 1))


def beta_cdf_integer(x: float, alpha: int, beta: int) -> float:
    """CDF of Beta(alpha, beta) for positive integer shape parameters.

    For integer shapes the regularized incomplete beta function is a finite
    binomial tail, so no external numerical library is required.
    """

    if alpha < 1 or beta < 1:
        raise ValueError("alpha and beta must be positive integers")
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0

    n = alpha + beta - 1
    one_minus_x = 1.0 - x
    value = 0.0
    for j in range(alpha, n + 1):
        value += math.comb(n, j) * (x**j) * (one_minus_x ** (n - j))
    return min(1.0, max(0.0, value))


def one_step_kg_exact(posteriors: Sequence[Tuple[int, int]], arm: int) -> Fraction:
    """Exact one-step Beta-Bernoulli knowledge gradient for one arm.

    The terminal Bayes action is the arm with the largest posterior mean.  The
    returned value is the expected increase in that terminal posterior mean
    after one observation of ``arm``.  Equivalently, under this model it is the
    expected one-step reduction in Bayes simple-decision risk.
    """

    means = [Fraction(a, a + b) for a, b in posteriors]
    current_value = max(means)
    alpha, beta = posteriors[arm]
    predictive_success = Fraction(alpha, alpha + beta)
    next_total = alpha + beta + 1
    mean_success = Fraction(alpha + 1, next_total)
    mean_failure = Fraction(alpha, next_total)
    other_best = max(
        (m for index, m in enumerate(means) if index != arm),
        default=Fraction(-1, 1),
    )
    expected_value = (
        predictive_success * max(mean_success, other_best)
        + (Fraction(1, 1) - predictive_success) * max(mean_failure, other_best)
    )
    return max(Fraction(0, 1), expected_value - current_value)


def one_step_kg(posteriors: Sequence[Tuple[int, int]], arm: int) -> float:
    """Floating presentation of :func:`one_step_kg_exact`."""

    return float(one_step_kg_exact(posteriors, arm))


def _random_argmax(values: Sequence[float | Fraction], rng: random.Random) -> int:
    maximum = max(values)
    winners = [index for index, value in enumerate(values) if value == maximum]
    return rng.choice(winners)


def _posterior_decision(posteriors: Sequence[Tuple[int, int]], rng: random.Random) -> int:
    return _random_argmax([Fraction(a, a + b) for a, b in posteriors], rng)


def _risk_fallback_arm(posteriors: Sequence[Tuple[int, int]], rng: random.Random) -> int:
    """Choose between the posterior-mean incumbent and its riskiest challenger.

    The fallback is used only when every exact one-step KG is zero. Candidates
    are ranked lexicographically by posterior probability of
    exceeding the incumbent mean, posterior mean, and posterior standard
    deviation. The final incumbent-versus-challenger comparison multiplies its
    tail probability by posterior standard deviation. This fixed structural
    convention has no fitted scalar exploration coefficient, but is not
    hyperparameter-free.
    """

    if len(posteriors) == 1:
        return 0

    means = [beta_mean(a, b) for a, b in posteriors]
    incumbent = _random_argmax(means, rng)
    incumbent_mean = means[incumbent]

    challenger_candidates: List[Tuple[Tuple[float, float, float], int]] = []
    for arm, ((alpha, beta), mean) in enumerate(zip(posteriors, means)):
        if arm == incumbent:
            continue
        probability_above = 1.0 - beta_cdf_integer(incumbent_mean, alpha, beta)
        standard_deviation = math.sqrt(beta_variance(alpha, beta))
        challenger_candidates.append(((probability_above, mean, standard_deviation), arm))
    best_key = max(key for key, _ in challenger_candidates)
    # Do not let the numerical arm index resolve an otherwise exact tie.
    tied_challengers = [arm for key, arm in challenger_candidates if key == best_key]
    challenger = rng.choice(tied_challengers)

    inc_alpha, inc_beta = posteriors[incumbent]
    ch_alpha, ch_beta = posteriors[challenger]
    incumbent_fragility = beta_cdf_integer(means[challenger], inc_alpha, inc_beta)
    challenger_danger = 1.0 - beta_cdf_integer(incumbent_mean, ch_alpha, ch_beta)
    incumbent_score = incumbent_fragility * math.sqrt(beta_variance(inc_alpha, inc_beta))
    challenger_score = challenger_danger * math.sqrt(beta_variance(ch_alpha, ch_beta))

    if incumbent_score == challenger_score:
        return rng.choice([incumbent, challenger])
    return incumbent if incumbent_score > challenger_score else challenger


class RewardTape:
    """Common-random-number Bernoulli outcomes indexed by arm and pull count."""

    def __init__(
        self,
        means: Sequence[float],
        budget: int,
        seed: int,
        trial: int,
        arm_keys: Sequence[object] | None = None,
    ):
        if arm_keys is None:
            arm_keys = list(range(len(means)))
        if len(arm_keys) != len(means):
            raise ValueError("arm_keys must match the number of means")
        self._outcomes: List[List[int]] = []
        for arm_key, mean in zip(arm_keys, means):
            rng = random.Random(stable_seed(seed, "reward", trial, arm_key))
            self._outcomes.append([1 if rng.random() < mean else 0 for _ in range(budget)])

    def pull(self, arm: int, pull_index: int) -> int:
        return self._outcomes[arm][pull_index]


@dataclass
class RunResult:
    algorithm: str
    selected_arm: int
    pulls: List[int]
    successes: List[int]
    kg_steps: int = 0
    fallback_steps: int = 0


def _update(
    arm: int,
    tape: RewardTape,
    pulls: MutableSequence[int],
    successes: MutableSequence[int],
) -> int:
    outcome = tape.pull(arm, pulls[arm])
    pulls[arm] += 1
    successes[arm] += outcome
    return outcome


def run_uniform(num_arms: int, budget: int, tape: RewardTape, rng: random.Random) -> RunResult:
    k = int(num_arms)
    pulls = [0] * k
    successes = [0] * k
    order = list(range(k))
    while sum(pulls) < budget:
        rng.shuffle(order)
        for arm in order:
            if sum(pulls) >= budget:
                break
            _update(arm, tape, pulls, successes)
    posteriors = [(1 + successes[i], 1 + pulls[i] - successes[i]) for i in range(k)]
    return RunResult("uniform", _posterior_decision(posteriors, rng), pulls, successes)


def run_raw_sequential_halving(
    num_arms: int, budget: int, tape: RewardTape, rng: random.Random
) -> RunResult:
    """Run a transparent, raw-mean sequential-halving baseline.

    This is intentionally not labelled exact Gumbel AlphaZero.  It uses no
    neural prior or completed-Q transform.  A budget of at least one pull per
    arm is required.
    """

    k = int(num_arms)
    if budget < k:
        raise ValueError("raw sequential halving requires budget >= number of arms")

    pulls = [0] * k
    successes = [0] * k
    active = list(range(k))

    while len(active) > 1 and sum(pulls) < budget:
        remaining = budget - sum(pulls)
        rounds_left = max(1, math.ceil(math.log2(len(active))))
        per_arm = max(1, remaining // (len(active) * rounds_left))
        per_arm = min(per_arm, remaining // len(active))
        if per_arm < 1:
            break

        round_order = active[:]
        rng.shuffle(round_order)
        for arm in round_order:
            for _ in range(per_arm):
                _update(arm, tape, pulls, successes)

        # Random jitter is used only for exact raw-mean ties.
        tie_noise = {arm: rng.random() for arm in active}
        ranked = sorted(
            active,
            key=lambda arm: (successes[arm] / pulls[arm], tie_noise[arm]),
            reverse=True,
        )
        active = ranked[: max(1, math.ceil(len(ranked) / 2))]

    while sum(pulls) < budget:
        order = active[:]
        rng.shuffle(order)
        for arm in order:
            if sum(pulls) >= budget:
                break
            _update(arm, tape, pulls, successes)

    raw_means = [successes[arm] / pulls[arm] for arm in active]
    local_winner = _random_argmax(raw_means, rng)
    return RunResult("raw_sequential_halving", active[local_winner], pulls, successes)


def run_kg_rank_risk(
    num_arms: int, budget: int, tape: RewardTape, rng: random.Random
) -> RunResult:
    k = int(num_arms)
    pulls = [0] * k
    successes = [0] * k
    posteriors: List[Tuple[int, int]] = [(1, 1) for _ in range(k)]
    kg_steps = 0
    fallback_steps = 0

    for _ in range(budget):
        kg_values = [one_step_kg_exact(posteriors, arm) for arm in range(k)]
        if max(kg_values) > 0:
            arm = _random_argmax(kg_values, rng)
            kg_steps += 1
        else:
            arm = _risk_fallback_arm(posteriors, rng)
            fallback_steps += 1

        outcome = _update(arm, tape, pulls, successes)
        alpha, beta = posteriors[arm]
        if outcome:
            posteriors[arm] = (alpha + 1, beta)
        else:
            posteriors[arm] = (alpha, beta + 1)

    selected = _posterior_decision(posteriors, rng)
    return RunResult("kg_rank_risk", selected, pulls, successes, kg_steps, fallback_steps)


RUNNERS = {
    "uniform": run_uniform,
    "raw_sequential_halving": run_raw_sequential_halving,
    "kg_rank_risk": run_kg_rank_risk,
}


@dataclass
class TrialRecord:
    trial: int
    algorithm: str
    budget: int
    selected_arm: int
    selected_mean: float
    simple_regret: float
    correct_selection: int
    pulls: List[int]
    successes: List[int]
    allocation_entropy: float
    kg_steps: int
    fallback_steps: int


def normalized_allocation_entropy(pulls: Sequence[int]) -> float:
    total = sum(pulls)
    if total == 0 or len(pulls) <= 1:
        return 0.0
    entropy = 0.0
    for count in pulls:
        if count:
            probability = count / total
            entropy -= probability * math.log(probability)
    return entropy / math.log(len(pulls))


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarize(records: Sequence[TrialRecord], num_arms: int) -> List[Dict[str, object]]:
    groups: Dict[Tuple[str, int], List[TrialRecord]] = {}
    for record in records:
        groups.setdefault((record.algorithm, record.budget), []).append(record)

    # Pair on the experimental unit, not row order.  A selected-algorithm run
    # may omit uniform; in that case paired fields remain null rather than
    # silently using an unpaired comparison.
    uniform_by_trial: Dict[Tuple[int, int], TrialRecord] = {
        (record.budget, record.trial): record
        for record in records
        if record.algorithm == "uniform"
    }

    summaries: List[Dict[str, object]] = []
    for (algorithm, budget), rows in sorted(groups.items(), key=lambda item: (item[0][1], item[0][0])):
        regrets = [row.simple_regret for row in rows]
        correct = [row.correct_selection for row in rows]
        selected_means = [row.selected_mean for row in rows]
        entropies = [row.allocation_entropy for row in rows]
        mean_regret = statistics.fmean(regrets)
        regret_sd = statistics.stdev(regrets) if len(regrets) > 1 else 0.0
        regret_se = regret_sd / math.sqrt(len(regrets))
        pcs = statistics.fmean(correct)
        pcs_se = math.sqrt(max(0.0, pcs * (1.0 - pcs) / len(rows)))
        mean_pulls_by_arm = [statistics.fmean(row.pulls[arm] for row in rows) for arm in range(num_arms)]

        paired_rows = [
            (row, uniform_by_trial[(row.budget, row.trial)])
            for row in rows
            if (row.budget, row.trial) in uniform_by_trial
        ]
        if len(paired_rows) == len(rows):
            paired_regret_deltas = [
                row.simple_regret - uniform.simple_regret for row, uniform in paired_rows
            ]
            paired_pcs_deltas = [
                row.correct_selection - uniform.correct_selection for row, uniform in paired_rows
            ]
            paired_mean_regret_delta: Optional[float] = statistics.fmean(paired_regret_deltas)
            paired_delta_sd = (
                statistics.stdev(paired_regret_deltas) if len(paired_regret_deltas) > 1 else 0.0
            )
            paired_delta_se = paired_delta_sd / math.sqrt(len(paired_regret_deltas))
            paired_regret_low: Optional[float] = paired_mean_regret_delta - 1.96 * paired_delta_se
            paired_regret_high: Optional[float] = paired_mean_regret_delta + 1.96 * paired_delta_se
            paired_pcs_delta: Optional[float] = statistics.fmean(paired_pcs_deltas)
            paired_trial_count = len(paired_rows)
        else:
            paired_mean_regret_delta = None
            paired_regret_low = None
            paired_regret_high = None
            paired_pcs_delta = None
            paired_trial_count = 0

        summaries.append(
            {
                "algorithm": algorithm,
                "budget": budget,
                "trials": len(rows),
                "mean_simple_regret": mean_regret,
                "regret_mc95_low": max(0.0, mean_regret - 1.96 * regret_se),
                "regret_mc95_high": mean_regret + 1.96 * regret_se,
                "median_simple_regret": _quantile(regrets, 0.5),
                "p90_simple_regret": _quantile(regrets, 0.9),
                "probability_correct_selection": pcs,
                "pcs_mc95_low": max(0.0, pcs - 1.96 * pcs_se),
                "pcs_mc95_high": min(1.0, pcs + 1.96 * pcs_se),
                "mean_selected_true_mean": statistics.fmean(selected_means),
                "mean_allocation_entropy": statistics.fmean(entropies),
                "mean_pulls_by_arm": mean_pulls_by_arm,
                "mean_kg_steps": statistics.fmean(row.kg_steps for row in rows),
                "mean_fallback_steps": statistics.fmean(row.fallback_steps for row in rows),
                # Paired signs are algorithm - uniform: negative regret is
                # better, while positive PCS is better.
                "paired_trials_vs_uniform": paired_trial_count,
                "paired_mean_regret_delta_vs_uniform": paired_mean_regret_delta,
                "paired_regret_delta_mc95_low": paired_regret_low,
                "paired_regret_delta_mc95_high": paired_regret_high,
                "paired_pcs_delta_vs_uniform": paired_pcs_delta,
            }
        )
    return summaries


def run_experiment(
    means: Sequence[float],
    budgets: Sequence[int],
    trials: int,
    seed: int,
    algorithms: Sequence[str] = ALGORITHMS,
    arm_keys: Sequence[object] | None = None,
) -> Tuple[List[TrialRecord], List[Dict[str, object]]]:
    if len(means) < 2:
        raise ValueError("at least two arms are required")
    if any(not 0.0 <= mean <= 1.0 for mean in means):
        raise ValueError("all Bernoulli means must lie in [0, 1]")
    if trials < 1:
        raise ValueError("trials must be positive")
    if not budgets or any(budget < 1 for budget in budgets):
        raise ValueError("budgets must be positive")
    if len(set(int(budget) for budget in budgets)) != len(budgets):
        raise ValueError("budgets must be unique")
    if max(budgets) > MAX_SUPPORTED_BUDGET:
        raise ValueError(
            f"budgets above {MAX_SUPPORTED_BUDGET} are outside the numerically audited scope"
        )
    if not algorithms:
        raise ValueError("at least one algorithm is required")
    if len(set(algorithms)) != len(algorithms):
        raise ValueError("algorithms must be unique")
    unknown = set(algorithms) - set(ALGORITHMS)
    if unknown:
        raise ValueError(f"unknown algorithms: {sorted(unknown)}")
    if "raw_sequential_halving" in algorithms and min(budgets) < len(means):
        raise ValueError("all budgets must be >= number of arms for raw sequential halving")

    best_mean = max(means)
    best_arms = {arm for arm, mean in enumerate(means) if mean == best_mean}
    max_budget = max(budgets)
    records: List[TrialRecord] = []

    for trial in range(trials):
        tape = RewardTape(means, max_budget, seed, trial, arm_keys=arm_keys)
        for budget in budgets:
            for algorithm in algorithms:
                rng = random.Random(stable_seed(seed, "algorithm", trial, budget, algorithm))
                result = RUNNERS[algorithm](len(means), budget, tape, rng)
                selected_mean = means[result.selected_arm]
                records.append(
                    TrialRecord(
                        trial=trial,
                        algorithm=algorithm,
                        budget=budget,
                        selected_arm=result.selected_arm,
                        selected_mean=selected_mean,
                        simple_regret=best_mean - selected_mean,
                        correct_selection=int(result.selected_arm in best_arms),
                        pulls=result.pulls,
                        successes=result.successes,
                        allocation_entropy=normalized_allocation_entropy(result.pulls),
                        kg_steps=result.kg_steps,
                        fallback_steps=result.fallback_steps,
                    )
                )

    return records, summarize(records, len(means))


def write_outputs(
    output_dir: Path,
    means: Sequence[float],
    budgets: Sequence[int],
    trials: int,
    seed: int,
    algorithms: Sequence[str],
    records: Sequence[TrialRecord],
    summaries: Sequence[Mapping[str, object]],
    include_trials: bool,
) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "summary.csv"
    json_path = output_dir / "summary.json"

    csv_fields = [
        "algorithm",
        "budget",
        "trials",
        "mean_simple_regret",
        "regret_mc95_low",
        "regret_mc95_high",
        "median_simple_regret",
        "p90_simple_regret",
        "probability_correct_selection",
        "pcs_mc95_low",
        "pcs_mc95_high",
        "mean_selected_true_mean",
        "mean_allocation_entropy",
        "mean_pulls_by_arm",
        "mean_kg_steps",
        "mean_fallback_steps",
        "paired_trials_vs_uniform",
        "paired_mean_regret_delta_vs_uniform",
        "paired_regret_delta_mc95_low",
        "paired_regret_delta_mc95_high",
        "paired_pcs_delta_vs_uniform",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for summary in summaries:
            row = dict(summary)
            row["mean_pulls_by_arm"] = json.dumps(row["mean_pulls_by_arm"], separators=(",", ":"))
            writer.writerow(row)

    payload: Dict[str, object] = {
        "schema_version": 1,
        "experiment": EXPERIMENT_ID,
        "execution_mode": EXECUTION_MODE,
        "means": list(means),
        "budgets": list(budgets),
        "trials": trials,
        "seed": seed,
        "algorithms": list(algorithms),
        "algorithm_contracts": {name: ALGORITHM_CONTRACTS[name] for name in algorithms},
        "beta_prior": [1, 1],
        "common_random_numbers": True,
        "interval_note": (
            "mc95 fields are descriptive normal Monte Carlo intervals, not algorithmic guarantees; "
            "paired deltas join rows by (trial, budget) and use algorithm minus uniform"
        ),
        "claim_status": "example_not_evidence",
        "prohibited_inferences": [
            "uniform_dominance",
            "regret_bound",
            "gumbel_alphazero_equivalence",
            "transfer_to_neural_mcts",
            "candidate_omission_control",
            "cpu_efficiency",
            "human_or_grandmaster_mechanism",
            "fully_hyperparameter_free",
        ],
        "summaries": list(summaries),
    }
    if include_trials:
        payload["trial_records"] = [asdict(record) for record in records]
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")

    return {"csv": str(csv_path), "json": str(json_path)}


def _parse_float_list(text: str) -> List[float]:
    try:
        return [float(piece.strip()) for piece in text.split(",") if piece.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _parse_int_list(text: str) -> List[int]:
    try:
        return [int(piece.strip()) for piece in text.split(",") if piece.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--means",
        type=_parse_float_list,
        default=_parse_float_list("0.58,0.55,0.52,0.50,0.48,0.45,0.42,0.40"),
        help="comma-separated Bernoulli arm means",
    )
    parser.add_argument(
        "--budgets",
        type=_parse_int_list,
        default=_parse_int_list("8,16,32,64"),
        help="comma-separated fixed pull budgets; each must be >= number of arms",
    )
    parser.add_argument("--trials", type=int, default=1000, help="Monte Carlo trials per condition")
    parser.add_argument("--seed", type=int, default=20260712, help="base random seed")
    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=ALGORITHMS,
        default=list(ALGORITHMS),
        help="allocation rules to compare",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/metacognitive_root/custom"),
    )
    parser.add_argument(
        "--include-trials",
        action="store_true",
        help="include all per-trial records in summary.json (can be large)",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    records, summaries = run_experiment(
        means=args.means,
        budgets=args.budgets,
        trials=args.trials,
        seed=args.seed,
        algorithms=args.algorithms,
    )
    paths = write_outputs(
        output_dir=args.output_dir,
        means=args.means,
        budgets=args.budgets,
        trials=args.trials,
        seed=args.seed,
        algorithms=args.algorithms,
        records=records,
        summaries=summaries,
        include_trials=args.include_trials,
    )
    print(json.dumps({"status": "ok", "outputs": paths, "conditions": len(summaries)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
