"""Retrospective C29 game replay and exact uniform-rollout channel diagnostic."""
import argparse
import gzip
import hashlib
import json
import math
from collections import defaultdict
from fractions import Fraction as F
from functools import cache
from random import Random
from statistics import mean, median

from verify_native import native

LINES = ((0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7),
         (2, 5, 8), (0, 4, 8), (2, 4, 6))


def place(board, i, sign):
    return board[:i] + (sign,) + board[i + 1:]


@cache
def game_truth(board, turn, root):
    for a, b, c in LINES:
        if board[a] and board[a] == board[b] == board[c]:
            u = 1 + board[a] * root
            return u, tuple(F(i == u) for i in range(3))
    moves = [i for i, x in enumerate(board) if not x]
    if not moves:
        return 1, (F(0), F(1), F(0))
    branches = [game_truth(place(board, i, turn), -turn, root) for i in moves]
    utility = (max if turn == root else min)(u for u, _ in branches)
    law = tuple(sum(p[y] for _, p in branches) / len(moves) for y in range(3))
    return utility, law


def game_histories(requests):
    for req in requests:
        board = (0,) * 9
        turn = 1
        for move in req["history"]:
            board = place(board, move, turn)
            turn = -turn
        children = {i: game_truth(place(board, i, turn), -turn, turn)
                    for i, v in enumerate(board) if not v}
        features = []
        for i, (u, p) in children.items():
            child = place(board, i, turn)
            terminal = all(child) or any(child[a] and child[a] == child[b] == child[c]
                                         for a, b, c in LINES)
            if not terminal:
                sibling = max((v[0] for j, v in children.items() if i != j), default=-1)
                features.append(((len(req["history"]), u), sibling, p))
        yield features


def channel_diagnostic(requests):
    cases = list(game_histories(requests))
    base = defaultdict(lambda: [1., 1., 1.])
    detail = defaultdict(lambda: [0., 0., 0.])
    support = defaultdict(set)
    for root in cases[:180]:
        for stratum, sibling, p in root:
            for y in range(3):
                base[stratum][y] += float(p[y])
                detail[stratum, sibling][y] += float(p[y])
    scores = []
    for root in cases[180:]:
        losses = []
        for stratum, sibling, p in root:
            support[stratum].add(p)
            row = base[stratum]
            shared = [x / sum(row) for x in row]
            cell = detail[stratum, sibling]
            rich = [(x + 12 * prior) / (sum(cell) + 12) for x, prior in zip(cell, shared)]
            losses.append(sum(float(w) * math.log2(a / b)
                              for w, a, b in zip(p, rich, shared)))
        if losses:
            scores.append(mean(losses))
    rng = Random(291)
    bootstrap = sorted(mean(rng.choices(scores, k=len(scores))) for _ in range(2000))
    return {
        "training_roots": 180, "retrospective_evaluation_roots": len(scores),
        "strata_with_different_exact_laws": sum(len(p) > 1 for p in support.values()),
        "strata": len(support), "sibling_predictive_gain_bits_root_mean": mean(scores),
        "descriptive_root_bootstrap_95_bits": [bootstrap[50], bootstrap[1949]],
    }


def replay(binary, source, out, baseline):
    raw = source.read_bytes()
    data = [json.loads(x) for x in gzip.decompress(raw).splitlines()]
    assert len(data) == 300 and len({x["id"] for x in data}) == 300
    summary = {"request_gzip_sha256": hashlib.sha256(raw).hexdigest(),
               "residual_diagnostic": channel_diagnostic(data)}
    if binary:
        requests = [dict(x, guard=True, repeats=23) for x in data]
        rows = native(binary, requests)
        assert all(row["equivalent_full_h2"] for row in rows)
        if baseline:
            old = [json.loads(x) for x in gzip.decompress(baseline.read_bytes()).splitlines()]
            assert len(old) == len(rows)
            keys = ("id", "initial_known_move", "baseline_move", "noisy_move", "observations",
                    "additional_rollout_nodes", "offline_realized_baseline", "offline_realized_noisy",
                    "planned_value")
            for previous, current in zip(old, rows):
                for key in keys:
                    assert previous[key] == current[key], (current["id"], key)
            summary["matched_frozen_c29_responses"] = len(old)
        for label, part in (("all", rows), ("cal", rows[180:240]),
                            ("guard_active", [x for x in rows if x["guard_stops"]]),
                            ("guard_inactive", [x for x in rows if not x["guard_stops"]])):
            if not part:
                continue
            full = [median(x["timing_full_h2_ns"]) for x in part]
            gated = [median(x["timing_noisy_ns"]) for x in part]
            base = [median(x["timing_baseline_ns"]) for x in part]
            summary[label] = {
                "roots": len(part), "guard_stops": sum(x["guard_stops"] for x in part),
                "full_h2_mean_of_root_median_ns": mean(full),
                "guarded_h2_mean_of_root_median_ns": mean(gated),
                "initial_only_mean_of_root_median_ns": mean(base),
                "median_paired_ns_delta": median(b - a for a, b in zip(full, gated)),
                "rollout_nodes": sum(x["additional_rollout_nodes"] for x in part),
                "realized_utility_gain_vs_initial_only": mean(
                    x["offline_realized_noisy"] - x["offline_realized_baseline"] for x in part
                ),
            }
        summary["per_root"] = [{k: x[k] for k in (
            "id", "equivalent_full_h2", "guard_stops", "noisy_move", "query_count",
            "additional_rollout_nodes", "offline_realized_noisy", "timing_baseline_ns",
            "timing_full_h2_ns", "timing_noisy_ns",
        )} for x in rows]
    out.write_text(json.dumps(summary, indent=2 if not binary else None) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_root"}, indent=2))


if __name__ == "__main__":
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("requests", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--binary")
    parser.add_argument("--baseline-responses", type=Path)
    args = parser.parse_args()
    replay(args.binary, args.requests, args.output, args.baseline_responses)
