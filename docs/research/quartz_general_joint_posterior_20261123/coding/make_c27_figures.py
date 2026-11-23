"""Render C27 frozen-result figures without fitting or campaign execution."""

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DOC = Path(__file__).resolve().parents[1]
OUT = DOC / "figures"
OUT.mkdir(exist_ok=True)
analyses = {
    split: json.loads((DOC / "coding/results" / split / "analysis.json").read_text())
    for split in ("CAL", "TEST")
}
plt.rcParams.update(
    {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
)
fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.5), layout="constrained")
ax = axes[0]
for y, split in ((1, "CAL"), (0, "TEST")):
    metric = analyses[split]["primary"]["metrics"]["net_wall"]
    mean, (lo, hi) = metric["mean"], metric["ci95"]
    ax.errorbar(
        mean,
        y,
        xerr=[[mean - lo], [hi - mean]],
        fmt="o",
        capsize=5,
        markersize=7,
        color="#AA4B27" if split == "CAL" else "#206A8E",
    )
    ax.text(-0.0094, y + 0.19, f"{mean:.6f}  [{lo:.6f}, {hi:.6f}]", fontsize=9)
ax.axvline(0, color="#666666", lw=1, ls="--")
ax.set(
    yticks=[0, 1],
    yticklabels=["TEST (120 roots)", "CAL (24 roots)"],
    xlim=(-0.0098, 0.001),
    ylim=(-0.5, 1.6),
    xlabel="JOINT − PRODUCT: utility − frozen price × native time",
    title="A. Paid decision value",
)
ax.grid(axis="x", alpha=0.18)
ax.text(
    0.02,
    0.02,
    "Paired utility difference = 0 on every root",
    transform=ax.transAxes,
    fontsize=9,
)
rows = {
    row["predictor"]: row
    for row in analyses["TEST"]["conditional_diagnostics"][
        "by_generating_policy_stage_predictor"
    ]
    if row["generating_policy"] == "JOINT_H2" and row["stage"] == "later"
}
names = ["product", "initial_projected", "joint"]
values = [rows[name]["mean_log_loss"] for name in names]
ax = axes[1]
x = np.arange(3)
bars = ax.bar(x, values, width=0.62, color=["#858B92", "#C79444", "#206A8E"])
for bar, value in zip(bars, values, strict=True):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.008,
        f"{value:.6f}",
        ha="center",
        fontsize=9,
    )
ax.set(
    xticks=x,
    xticklabels=["Product", "Initial projection", "Joint posterior"],
    ylabel="Mean categorical log loss (lower is better)",
    ylim=(0, 0.43),
    title="B. TEST prediction on the same later histories",
)
ax.grid(axis="y", alpha=0.18)
ax.set_axisbelow(True)
n = rows["joint"]
ax.text(
    0.02,
    0.96,
    f"{n['roots']} roots · {n['candidate_history_exposures']} candidate-history exposures",
    transform=ax.transAxes,
    va="top",
    fontsize=9,
)
fig.suptitle("Quartz C27: predictive improvement without a policy change", fontsize=13)
fig.supxlabel(
    "A: 10,000 descriptive root-bootstrap draws; shared descendants limit inference.  "
    "B: exposure-weighted diagnostic, not the primary gate.",
    fontsize=8,
)
for extension in ("png", "svg"):
    fig.savefig(
        OUT / f"C27_RESULTS.{extension}",
        dpi=180,
        metadata={"Date": None} if extension == "svg" else None,
    )
svg_path = OUT / "C27_RESULTS.svg"
svg_path.write_text(
    "\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n"
)
plt.close(fig)
receipt = {
    "source": str(Path(__file__).relative_to(DOC)),
    "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "inputs": {
        split: hashlib.sha256(
            (DOC / "coding/results" / split / "analysis.json").read_bytes()
        ).hexdigest()
        for split in analyses
    },
    "plot_values": {
        "primary": {s: analyses[s]["primary"]["metrics"]["net_wall"] for s in analyses},
        "test_later_log_loss": dict(zip(names, values, strict=True)),
    },
    "new_fitting_or_campaign": False,
}
(OUT / "FIGURE_PROVENANCE.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(
    json.dumps(
        {"outputs": ["C27_RESULTS.png", "C27_RESULTS.svg"], "new_campaign": False}
    )
)
