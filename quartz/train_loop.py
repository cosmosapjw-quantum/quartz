"""Training-loop helpers: early stopping, epoch execution, and plot generation."""

from __future__ import annotations

import json
import math
import os

import numpy as np
from tqdm import tqdm


def early_stopping_enabled(patience, concurrent=False):
    """Enable early stopping whenever patience is positive."""
    return patience > 0


def round_or_none(value, digits=4):
    return None if value is None else round(value, digits)


def load_epoch_history(log_path):
    """Load per-epoch records from the JSONL training log."""
    history = []
    if not os.path.exists(log_path):
        return history
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("_type") == "eval":
                continue
            history.append(entry)
    return history


def build_elo_plot_series(history):
    """Build absolute-Elo and error-bar series for plotting."""
    eval_points = []
    for row in history:
        candidate_elo = row.get("published_elo")
        champion_elo = row.get("champion_elo")
        elo_gap = row.get("elo_gap")
        match_delta = row.get("delta_elo")
        if candidate_elo is None:
            continue
        if champion_elo is None and elo_gap is not None:
            champion_elo = candidate_elo - elo_gap
        elif champion_elo is None and match_delta is not None:
            champion_elo = candidate_elo - match_delta
        if elo_gap is None and champion_elo is not None:
            elo_gap = candidate_elo - champion_elo
        midpoint = None
        half_gap = None
        if champion_elo is not None and elo_gap is not None:
            midpoint = 0.5 * (candidate_elo + champion_elo)
            half_gap = abs(elo_gap) * 0.5
        eval_points.append(
            {
                "iter": row.get("iter"),
                "candidate_elo": candidate_elo,
                "champion_elo": champion_elo,
                "elo_gap": elo_gap,
                "error_mid": midpoint,
                "error_half": half_gap,
                "score_rate": row.get("score_rate"),
                "match_delta_elo": match_delta,
                "eval_verdict": row.get("eval_verdict"),
            }
        )
    return eval_points


def build_metric_plot_series(history, field):
    """Return sparse (iteration, value) pairs for a single plotted metric."""
    series = []
    for row in history:
        iteration = row.get("iter")
        value = row.get(field)
        if iteration is None or value is None:
            continue
        series.append((iteration, value))
    return series


def build_best_elo_series(elo_points):
    """Track champion Elo, promoting only on explicit promotion verdicts."""
    best_elo = []
    running_best = None
    for point in elo_points:
        champion_elo = point.get("champion_elo")
        candidate_elo = point.get("candidate_elo")
        verdict = point.get("eval_verdict")

        if champion_elo is not None:
            running_best = champion_elo if running_best is None else max(running_best, champion_elo)

        promoted = verdict == "promote"
        if verdict is None and champion_elo is not None and candidate_elo is not None:
            promoted = candidate_elo >= champion_elo and (point.get("match_delta_elo") or 0) > 0
        if promoted and candidate_elo is not None:
            running_best = candidate_elo if running_best is None else max(running_best, candidate_elo)

        best_elo.append(running_best)
    return best_elo


def generate_training_plots(log_path, output_dir):
    """Write training metric plots after training completes."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except Exception as exc:
        print(f"  [WARN] Plot generation skipped (matplotlib unavailable: {exc})")
        return False

    history = load_epoch_history(log_path)
    if not history:
        return False

    iters = [h.get("iter") for h in history]
    loss = build_metric_plot_series(history, "loss")
    p_loss = build_metric_plot_series(history, "p_loss")
    v_loss = build_metric_plot_series(history, "v_loss")
    loss_ema = build_metric_plot_series(history, "loss_ema")
    elo_points = build_elo_plot_series(history)

    fig, ax = plt.subplots(figsize=(9, 5))

    def plot_metric(series, label, **kwargs):
        if not series:
            return
        xs, ys = zip(*series)
        ax.plot(xs, ys, label=label, marker="o", markersize=3.5, **kwargs)

    plot_metric(loss, "loss", linewidth=2.0)
    plot_metric(p_loss, "p_loss", linewidth=1.5, alpha=0.9)
    plot_metric(v_loss, "v_loss", linewidth=1.5, alpha=0.9)
    if loss_ema:
        xs, ys = zip(*loss_ema)
        ax.plot(xs, ys, label="loss_ema", linewidth=2.0, linestyle="--", marker="o", markersize=3.0)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.grid(True, alpha=0.25)
    if iters and min(iters) != max(iters):
        ax.set_xlim(min(iters), max(iters))
    handles, labels = ax.get_legend_handles_labels()
    if any(label and not label.startswith("_") for label in labels):
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "training_loss.png"), dpi=140)
    plt.close(fig)

    if elo_points:
        elo_iters = [p["iter"] for p in elo_points]
        candidate_elo = [p["candidate_elo"] for p in elo_points]
        champion_elo = [p["champion_elo"] for p in elo_points]

        fig, (ax_elo, ax_sr) = plt.subplots(2, 1, figsize=(10, 7), height_ratios=[3, 1], sharex=True)
        best_elo = build_best_elo_series(elo_points)

        ax_elo.plot(elo_iters, best_elo, color="#2563EB", linewidth=2.5, marker="o", markersize=5, label="Best Elo", zorder=4)
        ax_elo.plot(elo_iters, candidate_elo, color="#93C5FD", linewidth=1.0, marker=".", markersize=3, label="Candidate", alpha=0.6, zorder=2)
        if any(v is not None for v in champion_elo):
            ax_elo.plot(
                elo_iters,
                champion_elo,
                color="#D1D5DB",
                linewidth=1.0,
                linestyle="--",
                marker=".",
                markersize=3,
                label="Champion",
                alpha=0.5,
                zorder=1,
            )

        delta_data = [
            (it, be, p.get("match_delta_elo"))
            for it, be, p in zip(elo_iters, best_elo, elo_points)
            if be is not None and p.get("match_delta_elo") is not None
        ]
        if delta_data:
            d_it, d_best, d_delta = zip(*delta_data)
            d_lo = [b - abs(d) for b, d in zip(d_best, d_delta)]
            d_hi = [b + abs(d) for b, d in zip(d_best, d_delta)]
            ax_elo.fill_between(d_it, d_lo, d_hi, alpha=0.12, color="#2563EB", label="\u00b1 match \u0394Elo")

        ax_elo.set_ylabel("Elo Rating", fontsize=11)
        ax_elo.set_title("Elo Progression", fontsize=13, fontweight="bold")
        ax_elo.grid(True, alpha=0.2)
        ax_elo.legend(loc="upper left", fontsize=9, framealpha=0.9)
        all_elos = [v for v in candidate_elo + champion_elo if v is not None]
        if all_elos:
            q1 = sorted(all_elos)[len(all_elos) // 10]
            q9 = sorted(all_elos)[len(all_elos) * 9 // 10]
            iqr = max(q9 - q1, 100)
            ax_elo.set_ylim(min(all_elos[0], q1 - iqr * 0.3), q9 + iqr * 0.5)

        score_rate = [p.get("score_rate") for p in elo_points]
        if any(v is not None for v in score_rate):
            colors = []
            for sr in score_rate:
                if sr is None:
                    colors.append("#9CA3AF")
                elif sr > 0.55:
                    colors.append("#16A34A")
                elif sr < 0.45:
                    colors.append("#DC2626")
                else:
                    colors.append("#F59E0B")
            ax_sr.bar(
                elo_iters,
                [s if s is not None else 0 for s in score_rate],
                color=colors,
                width=max(1, (max(elo_iters) - min(elo_iters)) / len(elo_iters) * 0.6),
                edgecolor="none",
                alpha=0.85,
            )
            ax_sr.axhline(y=0.5, color="#9CA3AF", linewidth=0.8, linestyle="--", alpha=0.6)
            ax_sr.axhline(y=0.55, color="#16A34A", linewidth=0.6, linestyle=":", alpha=0.4)
            ax_sr.set_ylabel("Score Rate", fontsize=10)
            ax_sr.set_ylim(0, 1)
            ax_sr.legend(
                handles=[
                    Line2D([0], [0], color="#16A34A", marker="s", linestyle="", markersize=7, label="Promoted (>55%)"),
                    Line2D([0], [0], color="#F59E0B", marker="s", linestyle="", markersize=7, label="Marginal"),
                    Line2D([0], [0], color="#DC2626", marker="s", linestyle="", markersize=7, label="Rejected (<45%)"),
                ],
                loc="upper right",
                fontsize=8,
                framealpha=0.9,
            )

        ax_sr.set_xlabel("Iteration", fontsize=11)
        ax_sr.grid(True, alpha=0.2)

        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "training_elo.png"), dpi=150)
        plt.close(fig)

    return True


class EarlyStopping:
    """Stop training when smoothed loss stops improving."""

    def __init__(self, patience=10, min_delta=0.001, warmup=10, ema_alpha=0.3):
        self.patience = patience
        self.min_delta = min_delta
        self.warmup = warmup
        self.ema_alpha = ema_alpha
        self.best_loss = float("inf")
        self.counter = 0
        self.should_stop = False
        self.num_updates = 0
        self.loss_ema = None

    def step(self, loss):
        self.num_updates += 1
        if self.loss_ema is None:
            self.loss_ema = loss
        else:
            a = self.ema_alpha
            self.loss_ema = a * loss + (1 - a) * self.loss_ema

        if self.num_updates <= self.warmup:
            if self.loss_ema < self.best_loss:
                self.best_loss = self.loss_ema
            return False

        if self.loss_ema < self.best_loss - self.min_delta:
            self.best_loss = self.loss_ema
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


class StepEarlyStopping:
    """Loose within-iteration plateau stopper."""

    def __init__(self, patience=8, min_delta=5e-4, min_fraction=0.7, ema_alpha=0.2, planned_steps=1):
        self.patience = max(1, int(patience))
        self.min_delta = float(min_delta)
        self.min_fraction = float(max(0.0, min(1.0, min_fraction)))
        self.ema_alpha = float(max(0.01, min(1.0, ema_alpha)))
        self.planned_steps = max(1, int(planned_steps))
        self.min_steps = max(1, int(math.ceil(self.planned_steps * self.min_fraction)))
        self.best_loss = float("inf")
        self.loss_ema = None
        self.counter = 0
        self.triggered = False

    def step(self, loss, steps_done):
        loss = float(loss)
        if self.loss_ema is None:
            self.loss_ema = loss
        else:
            a = self.ema_alpha
            self.loss_ema = a * loss + (1.0 - a) * self.loss_ema

        if self.loss_ema < self.best_loss - self.min_delta:
            self.best_loss = self.loss_ema
            self.counter = 0
        else:
            self.counter += 1

        if steps_done < self.min_steps:
            return False
        if self.counter >= self.patience:
            self.triggered = True
        return self.triggered

    def summary(self, steps_done):
        return {
            "triggered": bool(self.triggered),
            "steps_done": int(steps_done),
            "min_steps": int(self.min_steps),
            "planned_steps": int(self.planned_steps),
            "counter": int(self.counter),
            "loss_ema": round_or_none(self.loss_ema),
        }


def train_epoch(model, optimizer, replay, cfg, device, n_steps, backend=None, inner_stop_cfg=None):
    """Train for n_steps. Uses backend.train_step if available (JAX JIT)."""
    torch = None
    F = None
    total_loss, total_pl, total_vl = 0.0, 0.0, 0.0
    loader = replay.build_dataloader(
        cfg["batch"],
        n_steps,
        pin_memory=(backend is None and getattr(device, "type", "cpu") != "cpu"),
    )
    if loader is None:
        return 0.0, 0.0, 0.0, 0, None
    if backend is None:
        import torch as _torch
        import torch.nn.functional as _F

        torch = _torch
        F = _F
        model.train()
    executed_steps = 0
    step_stopper = None
    if inner_stop_cfg and int(inner_stop_cfg.get("patience", 0) or 0) > 0:
        step_stopper = StepEarlyStopping(
            patience=inner_stop_cfg.get("patience", 8),
            min_delta=inner_stop_cfg.get("min_delta", 5e-4),
            min_fraction=inner_stop_cfg.get("min_fraction", 0.7),
            ema_alpha=inner_stop_cfg.get("ema_alpha", 0.2),
            planned_steps=n_steps,
        )

    with tqdm(loader, total=n_steps, desc="  Training", leave=False) as pbar:
        for states_t, policies_t, values_t in pbar:
            if backend is not None:
                loss, pl, vl = backend.train_step(states_t.numpy(), policies_t.numpy(), values_t.numpy())
            else:
                states_t = states_t.to(device, non_blocking=True)
                policies_t = policies_t.to(device, non_blocking=True)
                values_t = values_t.to(device, non_blocking=True)
                logits, pred_v = model(states_t)
                log_probs = F.log_softmax(logits, dim=-1)
                pl = -(policies_t * log_probs).sum(dim=-1).mean()
                vl = F.mse_loss(pred_v, values_t)
                loss_t = pl + vl
                optimizer.zero_grad(set_to_none=True)
                loss_t.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                loss, pl, vl = loss_t.item(), pl.item(), vl.item()

            total_loss += loss
            total_pl += pl
            total_vl += vl
            executed_steps += 1
            pbar.set_postfix(loss=f"{loss:.3f}", p=f"{pl:.3f}", v=f"{vl:.3f}")
            if step_stopper and step_stopper.step(loss, executed_steps):
                break

    n = max(executed_steps, 1)
    return (
        total_loss / n,
        total_pl / n,
        total_vl / n,
        executed_steps,
        step_stopper.summary(executed_steps) if step_stopper is not None else None,
    )


__all__ = [
    "EarlyStopping",
    "StepEarlyStopping",
    "build_best_elo_series",
    "build_elo_plot_series",
    "build_metric_plot_series",
    "early_stopping_enabled",
    "generate_training_plots",
    "load_epoch_history",
    "round_or_none",
    "train_epoch",
]
