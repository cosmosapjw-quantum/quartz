#!/usr/bin/env python3
"""Post-hoc description only: consumes frozen C26 results; never fits a model."""

from pathlib import Path
import gzip
import hashlib
import json
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DOC = Path(__file__).resolve().parents[1]
POLICIES = [
    "INITIAL_STOP",
    "ALL_ACTIONS",
    "RANDOM_ORDER",
    "GREEDY_VOC",
    "PANDORA",
    "GIBBS_VOC",
]


def rows(path):
    with gzip.open(path, "rt") as f:
        for line in f:
            yield json.loads(line)


out = {
    "schema": "c26-posthoc-diagnostics-v1",
    "scope": "Descriptive post-hoc diagnosis on frozen results; no refitting, retuning, or new confirmatory endpoint.",
    "inputs": {},
    "splits": {},
}
analyses = {}
for split in ["CAL", "TEST"]:
    directory = DOC / "coding/results" / split
    analysis = json.loads((directory / "analysis.json").read_text())
    analyses[split] = analysis
    parent = {
        (r["id"], r["policy"]): r["metrics"]
        for r in rows(directory / "parent_metrics.jsonl.gz")
        if r["cap"] == 9
    }
    ids = sorted(k[0] for k in parent if k[1] == "INITIAL_STOP")
    groups = {}
    for name, predicate in [
        ("initial_already_optimal", lambda b: b["regret"] == 0),
        ("initial_improvable", lambda b: b["regret"] > 0),
    ]:
        chosen = [i for i in ids if predicate(parent[i, "INITIAL_STOP"])]
        additional = [parent[i, "PANDORA"]["additional_nodes"] for i in chosen]
        gains = [
            parent[i, "PANDORA"]["utility"] - parent[i, "INITIAL_STOP"]["utility"]
            for i in chosen
        ]
        groups[name] = {
            "parents": len(chosen),
            "parents_with_queries": sum(
                parent[i, "PANDORA"]["additional_queries"] > 0 for i in chosen
            ),
            "sum_additional_nodes_over_parents": sum(additional),
            "mean_additional_nodes": statistics.mean(additional)
            if additional
            else None,
            "sum_utility_gain_over_parents": sum(gains),
        }
    trace_sums = {
        "source_seed_cells": 0,
        "queries": 0,
        "predicted_gain_sum": 0.0,
        "realized_gain_sum": 0.0,
        "predicted_nodes_sum": 0.0,
        "actual_nodes_sum": 0,
    }
    for record in rows(directory / "trace_raw.jsonl.gz"):
        if record["cap"] != 9:
            continue
        run = next(r for r in record["response"]["runs"] if r["policy"] == "PANDORA")
        trace_sums["source_seed_cells"] += 1
        for decision in run["decisions"]:
            if decision["chosen_move"] is None:
                continue
            candidate = next(
                c
                for c in decision["candidates"]
                if c["move"] == decision["chosen_move"]
            )
            trace_sums["queries"] += 1
            trace_sums["predicted_gain_sum"] += candidate["g"]
            trace_sums["realized_gain_sum"] += (
                decision["b_after"] - decision["b_before"]
            )
            trace_sums["predicted_nodes_sum"] += candidate["predicted_nodes"]
            trace_sums["actual_nodes_sum"] += decision["queried_nodes"]
    comparisons = []
    for comparison in analysis["paired_comparisons"]:
        if comparison["cap"] == 9 and (
            comparison["policy"] == "PANDORA"
            or (
                comparison["policy"] == "GIBBS_VOC"
                and comparison["baseline"] == "PANDORA"
            )
        ):
            comparisons.append(
                {
                    "policy": comparison["policy"],
                    "baseline": comparison["baseline"],
                    "metrics": {
                        k: {"mean": v["mean"], "ci95": v["ci95"]}
                        for k, v in comparison["metrics"].items()
                        if k
                        in [
                            "utility",
                            "net_nodes",
                            "net_wall",
                            "charged_nodes",
                            "elapsed_us",
                        ]
                    },
                }
            )
    cal = json.loads((directory / "calibration.json").read_text())
    out["splits"][split] = {
        "parents": len(ids),
        "primary": analysis["primary"],
        "available_mean_gain": statistics.mean(
            parent[i, "INITIAL_STOP"]["regret"] for i in ids
        ),
        "pandora_groups": groups,
        "pandora_selected_query_diagnostics": trace_sums,
        "selected_comparisons": comparisons,
        "calibration": {
            k: cal[k]
            for k in [
                "pooled_candidates",
                "equal_parent_brier",
                "within_root_residual_dependence",
            ]
        },
        "full_cap_means": {
            p: analysis["policy_means"]["9:" + p]["mean"] for p in POLICIES
        },
    }
    out["inputs"][split] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(directory.iterdir())
        if p.is_file()
    }
(DOC / "evidence/C08_DIAGNOSTICS.json").write_text(
    json.dumps(out, indent=2, sort_keys=True) + "\n"
)

plt.rcParams.update(
    {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
)
fig, (ax, cost) = plt.subplots(
    1,
    2,
    figsize=(13.4, 5.0),
    gridspec_kw={"width_ratios": [1, 1.6]},
    layout="constrained",
)
for i, split in enumerate(["CAL", "TEST"]):
    x = analyses[split]["primary"]
    color = "#3178a8" if split == "CAL" else "#b3483d"
    ax.errorbar(
        x["mean"],
        i,
        xerr=[[x["mean"] - x["ci95"][0]], [x["ci95"][1] - x["mean"]]],
        fmt="o",
        capsize=5,
        markersize=7,
        color=color,
    )
    ax.text(
        x["mean"],
        i + 0.14,
        f"{x['mean']:+.5f}",
        ha="center",
        color=color,
        fontweight="bold",
    )
ax.axvline(0, color="0.45", lw=1, ls="--")
ax.set(
    yticks=[0, 1],
    yticklabels=["CAL (diagnostic; 60 roots)", "TEST (primary; 150 roots)"],
    ylim=(-0.45, 1.55),
    xlabel="Pandora − initial: utility − node cost",
    title="A. Primary gate not met on held-out TEST",
)
ax.invert_yaxis()
ax.grid(axis="x", alpha=0.18)
labels = [
    "Initial stop",
    "All actions",
    "Random order",
    "Greedy VOC",
    "Pandora",
    "Gibbs VOC",
]
means = out["splits"]["TEST"]["full_cap_means"]
left = [0.0] * len(POLICIES)
for metric, label, color in [
    ("cue_nodes", "Cues", "#a6b8cb"),
    ("initial_nodes", "Initial exact solve", "#5a7898"),
    ("additional_nodes", "Additional exact queries", "#db9b51"),
]:
    values = [means[p][metric] for p in POLICIES]
    cost.barh(range(len(POLICIES)), values, left=left, label=label, color=color)
    left = [x + y for x, y in zip(left, values)]
for i, p in enumerate(POLICIES):
    cost.text(
        left[i] + 1.3,
        i,
        f"{left[i]:.1f}  |  U={means[p]['utility']:.4f}",
        va="center",
        fontsize=9,
    )
cost.set(
    yticks=list(range(len(POLICIES))),
    yticklabels=labels,
    xlim=(0, 174),
    xlabel="Mean charged nodes per root (full cap)",
    title="B. Paid work and exact decision utility",
)
cost.invert_yaxis()
cost.legend(loc="lower right", fontsize=8, frameon=False)
cost.grid(axis="x", alpha=0.15)
fig.suptitle(
    "Quartz C26 · Frozen TRAIN model · 95% paired stratified root bootstrap",
    fontsize=13,
)
fig.savefig(DOC / "figures/C26_primary_and_costs.png", dpi=180)
svg_path = DOC / "figures/C26_primary_and_costs.svg"
fig.savefig(svg_path)
svg_path.write_text(
    "\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n"
)
plt.close(fig)
print(
    json.dumps(
        {
            s: {
                "groups": x["pandora_groups"],
                "query_diagnostics": x["pandora_selected_query_diagnostics"],
            }
            for s, x in out["splits"].items()
        },
        sort_keys=True,
    )
)
