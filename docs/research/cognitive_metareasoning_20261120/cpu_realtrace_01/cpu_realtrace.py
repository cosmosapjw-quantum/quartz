"""CPU-only retrospective QUARTZ budget-ladder analysis (NOT incremental VOC).

All rows combine real trained-network initial priors and pre-existing
independent-restart MCTS search traces. The reference action is a stronger
search proxy, not an exact minimax label. No oracle labels enter features.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from math import exp, isfinite, log
import random

BUDGETS = (8, 16, 32, 64)
RIDGE = 10.0
BASIC = ("search_top", "search_gap", "search_entropy")
EXTRA = ("prior_top", "prior_gap", "prior_entropy", "prior_value", "prior_agree", "search_tied")


def _policy(values):
    if not isinstance(values, (list, tuple)) or len(values) < 2:
        raise ValueError("policy invalid length")
    x = [float(z) for z in values]
    if any(not isfinite(z) or z < 0 for z in x):
        raise ValueError("policy nonfinite or negative")
    total = sum(x)
    if not (total > 0 and abs(total - 1.0) <= 1e-4):
        raise ValueError("policy not normalized")
    return [z / total for z in x]


def _score(values):
    p = _policy(values)
    ranked = sorted(p, reverse=True)
    return {
        "argmax": p.index(max(p)),
        "top": ranked[0],
        "gap": ranked[0] - ranked[1],
        "entropy": -sum(x * log(x) for x in p if x > 0) / log(len(p)),
        "tie_count": sum(abs(x - ranked[0]) < 1e-7 for x in p),
        "support": sum(x > 1e-12 for x in p),
    }


def build_records(traces, assays, priors, positions):
    """Strict match 4 budgets and A4 assay, output 3 retrospective steps/root.

    Does not infer the effect of additional computation within a single MCTS run.
    """
    pos_ids = [p["id"] for p in positions]
    if len(set(pos_ids)) != len(pos_ids):
        raise ValueError("duplicate position ID")
    pos_ids = set(pos_ids)
    assay_index = {}
    for a in assays:
        if a.get("system") != "A4":
            continue
        key = (a["checkpoint_id"], a["position_id"], a["budget"])
        if key in assay_index:
            raise ValueError(f"duplicate A4 assay row: {key}")
        assay_index[key] = a
    out = []
    trace_keys = set()
    used_assay = set()
    for t in traces:
        ck, pid = t["checkpoint_id"], t["position_id"]
        key = (ck, pid)
        if key in trace_keys:
            raise ValueError(f"duplicate trace key: {key}")
        trace_keys.add(key)
        if pid not in pos_ids:
            raise ValueError("missing position")
        if key not in priors:
            raise ValueError(f"missing prior for {key}")
        if tuple(t["trace_budgets"]) != BUDGETS:
            raise ValueError("unexpected budgets")
        if t.get("trace_source") != "fresh":
            raise ValueError("non-fresh trace cannot be silently pooled")
        if len(t["trace_policies"]) != len(BUDGETS):
            raise ValueError("trace budget/policy length mismatch")
        nn = _score(priors[key]["policy"])
        value = float(priors[key]["value"])
        if not isfinite(value) or abs(value) > 1.001:
            raise ValueError("nonfinite or invalid NN value")
        scores = [_score(policy) for policy in t["trace_policies"]]
        refs = set()
        for b, sc in zip(BUDGETS, scores):
            akey = (ck, pid, b)
            if akey not in assay_index:
                raise ValueError(f"missing A4 assay {akey}")
            a = assay_index[akey]
            used_assay.add(akey)
            if a.get("search_continuation") != "independent_restarts":
                raise ValueError("trace continuation not an independent restart")
            if sc["argmax"] != a["argmax_effective"]:
                raise ValueError(f"trace/assay argmax mismatch {akey}")
            if nn["argmax"] != a["argmax_prior"]:
                raise ValueError(f"NN prior argmax mismatch {akey}")
            refs.add(int(a["reference_best"]))
        if len(refs) != 1:
            raise ValueError("reference target changed within a budget ladder")
        reference = refs.pop()
        for i, (b, next_b) in enumerate(zip(BUDGETS[:-1], BUDGETS[1:])):
            cur, later = scores[i], scores[i + 1]
            feats = {
                "search_top": cur["top"],
                "search_gap": cur["gap"],
                "search_entropy": cur["entropy"],
                "search_tied": float(cur["tie_count"] >= 2),
                "prior_top": nn["top"],
                "prior_gap": nn["gap"],
                "prior_entropy": nn["entropy"],
                "prior_value": value,
                "prior_agree": float(nn["argmax"] == cur["argmax"]),
            }
            out.append({
                "checkpoint_id": ck, "position_id": pid,
                "budget": b, "next_budget": next_b,
                "search_continuation": "independent_restarts",
                "reference_best": reference,
                "current_argmax": cur["argmax"],
                "next_argmax": later["argmax"],
                "nn_argmax": nn["argmax"],
                "current_correct": int(cur["argmax"] == reference),
                "next_correct": int(later["argmax"] == reference),
                "delta_reference": int(later["argmax"] == reference) - int(cur["argmax"] == reference),
                "current_tied": cur["tie_count"] >= 2,
                "next_tied": later["tie_count"] >= 2,
                "features": feats,
            })
    if set(assay_index) != used_assay:
        raise ValueError("A4 assay has unbound rows")
    return sorted(out, key=lambda row: (row["position_id"], row["checkpoint_id"], row["budget"]))


def _fold_map(rows, folds):
    ids = sorted({r["position_id"] for r in rows})
    if folds < 2 or folds > len(ids):
        raise ValueError("invalid fold count")
    return {pid: i % folds for i, pid in enumerate(ids)}


def _feature_vector(row, extra):
    f = row["features"]
    return [float(row["budget"] == b) for b in BUDGETS[:-1]] + [
        float(f[k]) for k in BASIC + (EXTRA if extra else ())
    ]


def _train_ridge(train_x, train_y, test_x):
    """Train-fold-only mean/scales, unpenalized intercept, fixed L2 penalty."""
    import numpy as np
    x = np.asarray(train_x, dtype=np.float64)
    y = np.asarray(train_y, dtype=np.float64)
    z = np.asarray(test_x, dtype=np.float64)
    if not (x.ndim == 2 and len(x) and len(z)):
        raise ValueError("missing train/test features")
    mu = x.mean(axis=0)
    sd = x.std(axis=0)
    sd[sd < 1e-12] = 1.0
    x = (x - mu) / sd
    z = (z - mu) / sd
    train = np.column_stack([np.ones(len(x)), x])
    test = np.column_stack([np.ones(len(z)), z])
    regularizer = np.eye(train.shape[1], dtype=np.float64) * RIDGE
    regularizer[0, 0] = 0
    coef = np.linalg.solve(train.T @ train + regularizer, train.T @ y)
    return [float(max(-1.0, min(1.0, v))) for v in test @ coef]


def crossfit(rows, *, folds=6):
    """Cross-fitted signed-budget-proxy predictors, same position held out."""
    fmap = _fold_map(rows, folds)
    predictions = {key: [None] * len(rows) for key in ("M0", "M1", "M2")}
    for fold in range(folds):
        train_idx = [i for i, r in enumerate(rows) if fmap[r["position_id"]] != fold]
        test_idx = [i for i, r in enumerate(rows) if fmap[r["position_id"]] == fold]
        if not train_idx or not test_idx:
            raise ValueError("empty group fold")
        for b in BUDGETS[:-1]:
            values = [rows[i]["delta_reference"] for i in train_idx if rows[i]["budget"] == b]
            if not values:
                raise ValueError("budget missing in train fold")
            mean = sum(values) / len(values)
            for i in test_idx:
                if rows[i]["budget"] == b:
                    predictions["M0"][i] = mean
        y = [rows[i]["delta_reference"] for i in train_idx]
        for key, enriched in (("M1", False), ("M2", True)):
            result = _train_ridge(
                [_feature_vector(rows[i], enriched) for i in train_idx],
                y,
                [_feature_vector(rows[i], enriched) for i in test_idx],
            )
            for i, pred in zip(test_idx, result):
                predictions[key][i] = pred
    for p in predictions.values():
        if any(x is None or not isfinite(x) for x in p):
            raise AssertionError("missing/nonfinite out-of-fold prediction")
    return predictions


def _mse(rows, p, subset=None):
    index = range(len(rows)) if subset is None else subset
    err = [(rows[i]["delta_reference"] - p[i]) ** 2 for i in index]
    return sum(err) / len(err) if err else None


def _interval(values):
    vals = sorted(values)
    return [vals[int((len(vals) - 1) * q)] for q in (0.025, 0.975)]


def summarize(rows, predictions, *, folds=6, bootstraps=2000, seed=20261120):
    """Deterministic descriptive CV summary; bootstrap resamples positions, not rows."""
    if len({len(v) for v in predictions.values()}) != 1 or any(
        len(v) != len(rows) for v in predictions.values()
    ):
        raise ValueError("row/prediction size mismatch")
    fmap = _fold_map(rows, folds)
    count = Counter(str(r["delta_reference"]) for r in rows)
    budgets = {}
    for b in BUDGETS[:-1]:
        batch = [r for r in rows if r["budget"] == b]
        budgets[str(b)] = {
            "total": len(batch),
            "signed_counts": dict(Counter(str(r["delta_reference"]) for r in batch)),
            "mean_signed": sum(r["delta_reference"] for r in batch) / len(batch),
            "current_tied": sum(r["current_tied"] for r in batch),
            "next_tied": sum(r["next_tied"] for r in batch),
        }
    result = {
        "schema": "quartz_cog01b_retrospective_cpu_v1",
        "interpretation": "independent_restart_reference_agreement_proxy_NOT_marginal_VOC",
        "n_rows": len(rows),
        "n_positions": len({r["position_id"] for r in rows}),
        "n_checkpoints": len({r["checkpoint_id"] for r in rows}),
        "budgets": budgets,
        "signed_counts": dict(count),
        "model_mse": {key: _mse(rows, p) for key, p in predictions.items()},
        "sensitivity": {},
    }
    for label, selector in {
        "exclude_self_reference": lambda r: r["checkpoint_id"] != "C06_latest",
        "no_ties": lambda r: not r["current_tied"] and not r["next_tied"],
        "exclude_self_reference_and_ties": lambda r: r["checkpoint_id"] != "C06_latest" and not r["current_tied"] and not r["next_tied"],
    }.items():
        idx = [i for i, r in enumerate(rows) if selector(r)]
        result["sensitivity"][label] = {
            "n_rows": len(idx),
            "mse_on_subset_without_refitting": {
                key: _mse(rows, p, idx) for key, p in predictions.items()
            },
        }

    # Equal 25% *nominal* budget transition (8->16 only) per held-out fold.
    selected = {key: set() for key in predictions}
    primary_idx = [i for i, r in enumerate(rows) if r["budget"] == 8]
    for f in range(folds):
        idx = [i for i in primary_idx if fmap[rows[i]["position_id"]] == f]
        k = max(1, len(idx) // 4)
        for method, p in predictions.items():
            chosen = sorted(idx, key=lambda i: (-p[i], rows[i]["checkpoint_id"], rows[i]["position_id"]))[:k]
            selected[method].update(chosen)
    sel = {}
    for model, choice in selected.items():
        sel[model] = {
            "n_selected": len(choice),
            "coverage": len(choice) / len(primary_idx),
            "mean_delta_among_selected": sum(rows[i]["delta_reference"] for i in choice) / len(choice),
            "mean_delta_per_all_positions": sum(rows[i]["delta_reference"] for i in choice) / len(primary_idx),
            "selected_improvements": sum(rows[i]["delta_reference"] == 1 for i in choice),
            "selected_harms": sum(rows[i]["delta_reference"] == -1 for i in choice),
        }
    q = len(selected["M1"]) / len(primary_idx)
    sel["random_matched_expected"] = {
        "coverage": q,
        "mean_delta_per_all_positions": q * sum(rows[i]["delta_reference"] for i in primary_idx) / len(primary_idx),
        "semantics": "offline expectation; NOT prospectively deployable",
    }
    result["selection_8_to_16_quartile"] = sel

    # Cluster bootstrap only conditional on frozen OOF predictions; no retraining.
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        groups[r["position_id"]].append(i)
    ids = sorted(groups)
    rng = random.Random(seed)
    mse_diff = []
    selection_diff = []
    if bootstraps < 1:
        raise ValueError("nonpositive bootstrap count")
    for _ in range(bootstraps):
        sampled = [rng.choice(ids) for _ in ids]
        ix = [i for pid in sampled for i in groups[pid]]
        mse_diff.append(
            sum((rows[i]["delta_reference"] - predictions["M2"][i]) ** 2 -
                (rows[i]["delta_reference"] - predictions["M1"][i]) ** 2 for i in ix) / len(ix)
        )
        # The selected sets are fixed by held-out-fold rank, not re-ranked per bootstrap.
        selection_diff.append(sum(
            (int(i in selected["M2"]) - int(i in selected["M1"])) * rows[i]["delta_reference"]
            for pid in sampled for i in groups[pid] if rows[i]["budget"] == 8
        ) / sum(sum(rows[i]["budget"] == 8 for i in groups[pid]) for pid in sampled))
    result["conditional_position_bootstrap"] = {
        "seed": seed, "resamples": bootstraps,
        "mse_M2_minus_M1_CI95": _interval(mse_diff),
        "quartile_gain_M2_minus_M1_CI95": _interval(selection_diff),
        "meaning": "position bootstrap of fixed out-of-fold predictions; model refitting uncertainty omitted",
    }
    return result
