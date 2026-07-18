#!/usr/bin/env python3
"""candidate_morphology_lab — priced widening, and omission vs ranking regret.

Part of the metacognitive experiment family (see
``docs/METACOGNITIVE_EXPERIMENTS.md``). This is a *separate model family* from
the Bernoulli root lab: there the arm set is fixed and fully visible, and the
only decision is how to allocate pulls (ranking risk). Here the agent starts
with a **visible pool** of candidates and a **hidden pool** it can only reach by
paying a priced ``WIDEN`` action, so two distinct regrets exist and must be
separated:

* **ranking regret** — among the candidates it has revealed, did it commit to
  the best one? (this is the Bernoulli lab's regret)
* **omission regret** — is the true best candidate still hidden, never revealed?
  (the gap between the global best and the best it ever made visible)

``total_regret = omission_regret + ranking_regret`` exactly, so the two lanes
add up. A `WIDEN` reveals the next-highest-prior hidden arm (progressive
widening in prior order, as an AlphaZero child-expansion analogue) at a fixed
integer price charged against the same budget that funds pulls; ``STOP`` commits
early when the incumbent is confidently ahead of its best visible challenger.

## Allocators (morphology policies)

* ``no_widen`` — never reveal; spend the whole budget pulling the initial
  visible pool. Zero widen price, but omission regret is uncontrolled.
* ``eager_widen`` — reveal every hidden arm it can afford up front, then pull.
  Drives omission regret toward zero but spends price that could have bought
  ranking certainty (and wastes price when the best was already visible).
* ``priced_widen`` — reveal the next hidden arm only when its prior score looks
  better than the current incumbent's posterior mean *and* the budget can
  afford the price with pulls held in reserve; otherwise pull, and ``STOP``
  early on a normal-approximation commit gate.

## Why it matters

The `CLAIM_LEDGER` "hidden-candidate morphology lab before any dual-risk claim"
row requires exactly this omission/ranking separation before any widening or
narrowing lane can be claimed. Its kill-criterion: if **no** widen price in
**any** scenario yields a CI-separated reduction in paired omission regret vs
``no_widen``, the widening/narrowing lane is demoted — priced widening buys
nothing and should not be wired into the engine.

Prohibited (claim firewall): reading these synthetic regrets as evidence that
widening improves QUARTZ play strength, as a candidate-omission guarantee, or as
a CPU/energy claim. It is a mechanism screen on a synthetic candidate world.
"""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from quartz.experiments.bernoulli_root import beta_mean, beta_variance, stable_seed

EXPERIMENT_ID = "candidate_morphology_lab_v1"
EXECUTION_MODE = "synthetic_screening"
CANDIDATE_MORPHOLOGY_SCHEMA_VERSION = 1

ALLOCATORS: Tuple[str, ...] = ("no_widen", "eager_widen", "priced_widen")
BASELINE_ALLOCATOR = "no_widen"
MAX_SUPPORTED_BUDGET = 256

# Fixed structural constants. Like the Bernoulli lab's fallback these carry no
# fitted scalar exploration coefficient, but the lab is NOT hyperparameter-free.
PULL_RESERVE = 2  # pulls held back so a freshly revealed arm can be sampled
WIDEN_MARGIN = 0.0  # widen when next prior merely exceeds the incumbent mean
COMMIT_MIN_VISITS = 8  # never STOP on a near-empty pool
COMMIT_THRESHOLD = 0.9  # normal-approx P(incumbent beats challenger) to STOP

ALLOCATOR_CONTRACTS = {
    "no_widen": {
        "family": "fixed_visible_pool",
        "reveals_hidden": False,
        "uses_true_means": False,
        "fitted_scalar_exploration_coefficient": False,
    },
    "eager_widen": {
        "family": "reveal_all_affordable_then_pull",
        "reveals_hidden": True,
        "uses_true_means": False,
        "fitted_scalar_exploration_coefficient": False,
    },
    "priced_widen": {
        "family": "prior_optimism_priced_widen_with_commit_stop",
        "reveals_hidden": True,
        "uses_true_means": False,
        "fitted_scalar_exploration_coefficient": False,
        "structural_constants": {
            "pull_reserve": PULL_RESERVE,
            "widen_margin": WIDEN_MARGIN,
            "commit_min_visits": COMMIT_MIN_VISITS,
            "commit_threshold": COMMIT_THRESHOLD,
        },
    },
}


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


class RewardTape:
    """Common-random-number Bernoulli outcomes keyed by canonical arm identity.

    A ``WIDEN`` never consumes the tape; only ``PULL`` does. Because the tape is
    keyed by (seed, trial, canonical arm), every allocator faces byte-identical
    outcomes in the same trial world — the paired comparison is exact.
    """

    def __init__(self, means: Sequence[float], budget: int, seed: int, trial: int):
        self._outcomes: List[List[int]] = []
        for arm, mean in enumerate(means):
            rng = random.Random(stable_seed(seed, "reward", trial, arm))
            self._outcomes.append(
                [1 if rng.random() < mean else 0 for _ in range(budget)]
            )

    def pull(self, arm: int, pull_index: int) -> int:
        return self._outcomes[arm][pull_index]


@dataclass
class CandidateWorld:
    """One trial's realized candidate landscape (identical across allocators)."""

    means: List[float]
    prior_scores: List[float]
    visible: List[int]  # initial visible pool, canonical indices
    hidden_queue: List[int]  # remaining arms, highest-prior first
    global_best_mean: float


def build_world(
    means: Sequence[float],
    n_visible: int,
    prior_noise: float,
    seed: int,
    trial: int,
) -> CandidateWorld:
    """Draw prior scores (true mean + gaussian noise), order arms by prior, and
    split into the initial visible pool and a prior-ordered hidden queue.

    The per-trial prior-noise draw is what makes the true best sometimes hidden
    and sometimes visible, so omission opportunity varies across trials rather
    than being fixed by the scenario."""
    k = len(means)
    if not 1 <= n_visible <= k:
        raise ValueError("n_visible must be in [1, num_arms]")
    rng = random.Random(stable_seed(seed, "prior", trial))
    prior_scores = [means[arm] + rng.gauss(0.0, prior_noise) for arm in range(k)]
    # Present in descending prior order; ties broken by a per-trial jitter so the
    # canonical index never silently resolves an exact prior tie.
    jitter = [rng.random() for _ in range(k)]
    order = sorted(
        range(k), key=lambda arm: (prior_scores[arm], jitter[arm]), reverse=True
    )
    return CandidateWorld(
        means=[float(m) for m in means],
        prior_scores=prior_scores,
        visible=order[:n_visible],
        hidden_queue=order[n_visible:],
        global_best_mean=max(means),
    )


def _posteriors(
    visible: Sequence[int], pulls: Mapping[int, int], succ: Mapping[int, int]
):
    return {
        arm: (1 + succ.get(arm, 0), 1 + pulls.get(arm, 0) - succ.get(arm, 0))
        for arm in visible
    }


def _incumbent(visible: Sequence[int], post, rng: random.Random) -> int:
    means = {arm: beta_mean(*post[arm]) for arm in visible}
    best = max(means.values())
    winners = [arm for arm in visible if means[arm] == best]
    return rng.choice(winners)


def _pull_target(visible: Sequence[int], post, rng: random.Random) -> int:
    """Pull to resolve the incumbent-vs-challenger ranking: among the top two
    visible arms by posterior mean, pull the more uncertain one."""
    if len(visible) == 1:
        return visible[0]
    means = {arm: beta_mean(*post[arm]) for arm in visible}
    ranked = sorted(visible, key=lambda arm: (means[arm], rng.random()), reverse=True)
    top2 = ranked[:2]
    return max(top2, key=lambda arm: (beta_variance(*post[arm]), rng.random()))


def _commit_confidence(visible: Sequence[int], post) -> float:
    """Normal-approximation P(incumbent's mean exceeds its best challenger).

    A cheap STOP gate, explicitly an approximation — not a claim-bearing
    statistic. Returns 1.0 when only one arm is visible (no ranking risk)."""
    if len(visible) < 2:
        return 1.0
    means = {arm: beta_mean(*post[arm]) for arm in visible}
    ranked = sorted(visible, key=lambda arm: means[arm], reverse=True)
    inc, ch = ranked[0], ranked[1]
    m_inc, m_ch = means[inc], means[ch]
    var = beta_variance(*post[inc]) + beta_variance(*post[ch])
    if var <= 0.0:
        return 1.0 if m_inc > m_ch else 0.5
    return _normal_cdf((m_inc - m_ch) / math.sqrt(var))


@dataclass
class RunResult:
    allocator: str
    selected_arm: int
    visible_at_commit: List[int]
    pulls: Dict[int, int]
    n_widens: int
    widen_spend: int
    pull_spend: int
    budget_used: int
    stopped_early: bool


def _final_result(
    allocator: str,
    visible: Sequence[int],
    pulls: Mapping[int, int],
    succ: Mapping[int, int],
    n_widens: int,
    widen_spend: int,
    pull_spend: int,
    stopped_early: bool,
    rng: random.Random,
) -> RunResult:
    post = _posteriors(visible, pulls, succ)
    selected = _incumbent(visible, post, rng)
    return RunResult(
        allocator=allocator,
        selected_arm=selected,
        visible_at_commit=list(visible),
        pulls=dict(pulls),
        n_widens=n_widens,
        widen_spend=widen_spend,
        pull_spend=pull_spend,
        budget_used=widen_spend + pull_spend,
        stopped_early=stopped_early,
    )


def run_no_widen(
    world: CandidateWorld,
    budget: int,
    widen_cost: int,
    tape: RewardTape,
    rng: random.Random,
) -> RunResult:
    visible = list(world.visible)
    pulls: Dict[int, int] = {}
    succ: Dict[int, int] = {}
    spent = 0
    while spent < budget:
        post = _posteriors(visible, pulls, succ)
        target = _pull_target(visible, post, rng)
        outcome = tape.pull(target, pulls.get(target, 0))
        pulls[target] = pulls.get(target, 0) + 1
        succ[target] = succ.get(target, 0) + outcome
        spent += 1
    return _final_result("no_widen", visible, pulls, succ, 0, 0, spent, False, rng)


def run_eager_widen(
    world: CandidateWorld,
    budget: int,
    widen_cost: int,
    tape: RewardTape,
    rng: random.Random,
) -> RunResult:
    visible = list(world.visible)
    hidden = list(world.hidden_queue)
    pulls: Dict[int, int] = {}
    succ: Dict[int, int] = {}
    widen_spend = 0
    n_widens = 0
    # Reveal every hidden arm affordable while leaving PULL_RESERVE for pulling.
    while hidden and (budget - widen_spend) >= widen_cost + PULL_RESERVE:
        visible.append(hidden.pop(0))
        widen_spend += widen_cost
        n_widens += 1
    pull_spend = 0
    while widen_spend + pull_spend < budget:
        post = _posteriors(visible, pulls, succ)
        target = _pull_target(visible, post, rng)
        outcome = tape.pull(target, pulls.get(target, 0))
        pulls[target] = pulls.get(target, 0) + 1
        succ[target] = succ.get(target, 0) + outcome
        pull_spend += 1
    return _final_result(
        "eager_widen",
        visible,
        pulls,
        succ,
        n_widens,
        widen_spend,
        pull_spend,
        False,
        rng,
    )


def run_priced_widen(
    world: CandidateWorld,
    budget: int,
    widen_cost: int,
    tape: RewardTape,
    rng: random.Random,
) -> RunResult:
    visible = list(world.visible)
    hidden = list(world.hidden_queue)
    pulls: Dict[int, int] = {}
    succ: Dict[int, int] = {}
    widen_spend = 0
    pull_spend = 0
    n_widens = 0
    while widen_spend + pull_spend < budget:
        remaining = budget - widen_spend - pull_spend
        post = _posteriors(visible, pulls, succ)
        inc = _incumbent(visible, post, rng)
        inc_mean = beta_mean(*post[inc])
        total_visits = sum(pulls.values())

        # Priced WIDEN: reveal the next hidden arm only when its prior optimism
        # beats the incumbent's posterior mean and the price is affordable with
        # pulls held in reserve.
        if hidden and remaining >= widen_cost + PULL_RESERVE:
            next_prior = min(1.0, max(0.0, world.prior_scores[hidden[0]]))
            if next_prior > inc_mean + WIDEN_MARGIN:
                visible.append(hidden.pop(0))
                widen_spend += widen_cost
                n_widens += 1
                continue

        # STOP: commit early once the incumbent is confidently ahead.
        if (
            total_visits >= COMMIT_MIN_VISITS
            and _commit_confidence(visible, post) >= COMMIT_THRESHOLD
        ):
            return _final_result(
                "priced_widen",
                visible,
                pulls,
                succ,
                n_widens,
                widen_spend,
                pull_spend,
                True,
                rng,
            )

        target = _pull_target(visible, post, rng)
        outcome = tape.pull(target, pulls.get(target, 0))
        pulls[target] = pulls.get(target, 0) + 1
        succ[target] = succ.get(target, 0) + outcome
        pull_spend += 1
    return _final_result(
        "priced_widen",
        visible,
        pulls,
        succ,
        n_widens,
        widen_spend,
        pull_spend,
        False,
        rng,
    )


RUNNERS = {
    "no_widen": run_no_widen,
    "eager_widen": run_eager_widen,
    "priced_widen": run_priced_widen,
}


@dataclass
class TrialRecord:
    trial: int
    allocator: str
    budget: int
    widen_cost: int
    selected_arm: int
    selected_mean: float
    global_best_mean: float
    best_visible_mean: float
    ranking_regret: float
    omission_regret: float
    total_regret: float
    correct_selection: int
    best_revealed: int
    n_widens: int
    widen_spend: int
    pull_spend: int
    budget_used: int
    n_visible_at_commit: int
    stopped_early: int


def _best_visible_mean(world: CandidateWorld, visible: Sequence[int]) -> float:
    return max(world.means[arm] for arm in visible)


def run_experiment(
    means: Sequence[float],
    n_visible: int,
    prior_noise: float,
    budgets: Sequence[int],
    widen_costs: Sequence[int],
    trials: int,
    seed: int,
    allocators: Sequence[str] = ALLOCATORS,
) -> Tuple[List[TrialRecord], List[Dict[str, Any]]]:
    if len(means) < 2:
        raise ValueError("at least two arms are required")
    if any(not 0.0 <= m <= 1.0 for m in means):
        raise ValueError("all Bernoulli means must lie in [0, 1]")
    if trials < 1:
        raise ValueError("trials must be positive")
    if not budgets or any(b < 1 for b in budgets):
        raise ValueError("budgets must be positive")
    if len(set(int(b) for b in budgets)) != len(budgets):
        raise ValueError("budgets must be unique")
    if max(budgets) > MAX_SUPPORTED_BUDGET:
        raise ValueError(
            f"budgets above {MAX_SUPPORTED_BUDGET} are outside the audited scope"
        )
    if not widen_costs or any(c < 1 for c in widen_costs):
        raise ValueError("widen_costs must be positive integers")
    unknown = set(allocators) - set(ALLOCATORS)
    if unknown:
        raise ValueError(f"unknown allocators: {sorted(unknown)}")
    if len(set(allocators)) != len(allocators):
        raise ValueError("allocators must be unique")
    # Every allocator must be pullable on the smallest budget with the initial
    # pool; the visible pool alone should be samplable.
    if min(budgets) < n_visible:
        raise ValueError(
            "smallest budget must cover at least one pull per initial visible arm"
        )

    max_budget = max(budgets)
    records: List[TrialRecord] = []
    for trial in range(trials):
        world = build_world(means, n_visible, prior_noise, seed, trial)
        tape = RewardTape(means, max_budget, seed, trial)
        for budget in budgets:
            for widen_cost in widen_costs:
                for allocator in allocators:
                    rng = random.Random(
                        stable_seed(seed, "alloc", trial, budget, widen_cost, allocator)
                    )
                    result = RUNNERS[allocator](world, budget, widen_cost, tape, rng)
                    selected_mean = world.means[result.selected_arm]
                    best_visible = _best_visible_mean(world, result.visible_at_commit)
                    ranking = best_visible - selected_mean
                    omission = world.global_best_mean - best_visible
                    records.append(
                        TrialRecord(
                            trial=trial,
                            allocator=allocator,
                            budget=budget,
                            widen_cost=widen_cost,
                            selected_arm=result.selected_arm,
                            selected_mean=selected_mean,
                            global_best_mean=world.global_best_mean,
                            best_visible_mean=best_visible,
                            ranking_regret=ranking,
                            omission_regret=omission,
                            total_regret=ranking + omission,
                            correct_selection=int(
                                selected_mean >= world.global_best_mean - 1e-12
                            ),
                            best_revealed=int(
                                best_visible >= world.global_best_mean - 1e-12
                            ),
                            n_widens=result.n_widens,
                            widen_spend=result.widen_spend,
                            pull_spend=result.pull_spend,
                            budget_used=result.budget_used,
                            n_visible_at_commit=len(result.visible_at_commit),
                            stopped_early=int(result.stopped_early),
                        )
                    )
    return records, summarize(records)


def _paired_delta(values: Sequence[float]) -> Dict[str, float]:
    mean = statistics.fmean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    se = sd / math.sqrt(len(values)) if values else 0.0
    return {
        "mean": mean,
        "mc95_low": mean - 1.96 * se,
        "mc95_high": mean + 1.96 * se,
    }


def summarize(records: Sequence[TrialRecord]) -> List[Dict[str, Any]]:
    """Per-(allocator, budget, widen_cost) means plus paired deltas of each
    regret lane vs the ``no_widen`` baseline, joined on the trial (the trial
    world is shared across allocators, so the pairing is exact)."""
    groups: Dict[Tuple[str, int, int], List[TrialRecord]] = {}
    for row in records:
        groups.setdefault((row.allocator, row.budget, row.widen_cost), []).append(row)

    baseline_by_trial: Dict[Tuple[int, int, int], TrialRecord] = {
        (r.budget, r.widen_cost, r.trial): r
        for r in records
        if r.allocator == BASELINE_ALLOCATOR
    }

    summaries: List[Dict[str, Any]] = []
    for (allocator, budget, widen_cost), rows in sorted(groups.items()):
        n = len(rows)
        summary: Dict[str, Any] = {
            "allocator": allocator,
            "budget": budget,
            "widen_cost": widen_cost,
            "trials": n,
            "mean_total_regret": statistics.fmean(r.total_regret for r in rows),
            "mean_omission_regret": statistics.fmean(r.omission_regret for r in rows),
            "mean_ranking_regret": statistics.fmean(r.ranking_regret for r in rows),
            "probability_correct_selection": statistics.fmean(
                r.correct_selection for r in rows
            ),
            "best_reveal_rate": statistics.fmean(r.best_revealed for r in rows),
            "mean_n_widens": statistics.fmean(r.n_widens for r in rows),
            "mean_budget_used": statistics.fmean(r.budget_used for r in rows),
            "mean_n_visible_at_commit": statistics.fmean(
                r.n_visible_at_commit for r in rows
            ),
            "early_stop_rate": statistics.fmean(r.stopped_early for r in rows),
        }
        paired = [
            (r, baseline_by_trial[(r.budget, r.widen_cost, r.trial)])
            for r in rows
            if (r.budget, r.widen_cost, r.trial) in baseline_by_trial
        ]
        if allocator != BASELINE_ALLOCATOR and len(paired) == n and n > 0:
            omission_d = _paired_delta(
                [a.omission_regret - b.omission_regret for a, b in paired]
            )
            ranking_d = _paired_delta(
                [a.ranking_regret - b.ranking_regret for a, b in paired]
            )
            total_d = _paired_delta(
                [a.total_regret - b.total_regret for a, b in paired]
            )
            summary.update(
                {
                    "paired_trials_vs_baseline": len(paired),
                    "paired_omission_delta_vs_baseline": omission_d["mean"],
                    "paired_omission_delta_mc95_low": omission_d["mc95_low"],
                    "paired_omission_delta_mc95_high": omission_d["mc95_high"],
                    "paired_ranking_delta_vs_baseline": ranking_d["mean"],
                    "paired_ranking_delta_mc95_low": ranking_d["mc95_low"],
                    "paired_ranking_delta_mc95_high": ranking_d["mc95_high"],
                    "paired_total_delta_vs_baseline": total_d["mean"],
                    "paired_total_delta_mc95_low": total_d["mc95_low"],
                    "paired_total_delta_mc95_high": total_d["mc95_high"],
                }
            )
        else:
            summary.update(
                {
                    "paired_trials_vs_baseline": 0
                    if allocator != BASELINE_ALLOCATOR
                    else n,
                    "paired_omission_delta_vs_baseline": None,
                    "paired_omission_delta_mc95_low": None,
                    "paired_omission_delta_mc95_high": None,
                    "paired_ranking_delta_vs_baseline": None,
                    "paired_ranking_delta_mc95_low": None,
                    "paired_ranking_delta_mc95_high": None,
                    "paired_total_delta_vs_baseline": None,
                    "paired_total_delta_mc95_low": None,
                    "paired_total_delta_mc95_high": None,
                }
            )
        summaries.append(summary)
    return summaries


def widening_kill_verdict(summaries: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Kill-criterion: is there ANY (allocator, budget, widen_cost) where a
    widening allocator delivers a CI-separated reduction in paired omission
    regret vs ``no_widen`` (upper 95% bound < 0)? If none, demote the lane.

    The stronger, honest bar is also reported: does any config give a
    CI-separated reduction in *total* regret? Omission relief that is fully
    paid back in ranking regret is a net wash — the lane survives the kill
    (omission genuinely moves) but only total-regret improvement warrants
    promotion to a widening claim."""
    omission_hits: List[Dict[str, Any]] = []
    total_hits: List[Dict[str, Any]] = []
    for row in summaries:
        if row["allocator"] == BASELINE_ALLOCATOR:
            continue
        entry = {
            "scenario_id": row.get("scenario_id"),
            "allocator": row["allocator"],
            "budget": row["budget"],
            "widen_cost": row["widen_cost"],
            "paired_omission_delta_vs_baseline": row.get(
                "paired_omission_delta_vs_baseline"
            ),
            "paired_omission_delta_mc95_high": row.get(
                "paired_omission_delta_mc95_high"
            ),
            "paired_ranking_delta_vs_baseline": row.get(
                "paired_ranking_delta_vs_baseline"
            ),
            "paired_total_delta_vs_baseline": row.get("paired_total_delta_vs_baseline"),
            "paired_total_delta_mc95_high": row.get("paired_total_delta_mc95_high"),
        }
        om_hi = row.get("paired_omission_delta_mc95_high")
        if om_hi is not None and om_hi < 0.0:
            omission_hits.append(entry)
        tot_hi = row.get("paired_total_delta_mc95_high")
        if tot_hi is not None and tot_hi < 0.0:
            total_hits.append(entry)
    return {
        "candidate_morphology_schema_version": CANDIDATE_MORPHOLOGY_SCHEMA_VERSION,
        "n_ci_separated_omission_improvements": len(omission_hits),
        "widen_prices_with_omission_improvement": sorted(
            {r["widen_cost"] for r in omission_hits}
        ),
        "n_ci_separated_total_improvements": len(total_hits),
        "widen_prices_with_total_improvement": sorted(
            {r["widen_cost"] for r in total_hits}
        ),
        "widening_lane_demoted": len(omission_hits) == 0,
        "net_total_improvement_found": len(total_hits) > 0,
        "ci_separated_omission_improvements": omission_hits,
        "ci_separated_total_improvements": total_hits,
    }


def records_as_dicts(records: Sequence[TrialRecord]) -> List[Dict[str, Any]]:
    return [asdict(r) for r in records]
