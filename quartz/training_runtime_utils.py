"""Shared runtime helpers that should not live in the legacy facade."""

from __future__ import annotations

import time

import numpy as np


def make_json_safe(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): make_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(v) for v in value]
    return value


def benchmark_eval_parallel_workers(
    hw,
    cfg,
    eval_games,
    candidate_factory,
    champion_factory,
    game_factory,
    profile_path,
    *,
    has_eval_system,
    eval_worker_candidates_fn,
    eval_config_cls,
    training_evaluator_cls,
    save_eval_autotune_profile_fn,
):
    if not has_eval_system or eval_games <= 1:
        return 1, []

    candidates = eval_worker_candidates_fn(hw, cfg, eval_games)
    if len(candidates) == 1:
        return candidates[0], []

    pilot_games = max(4, min(8, int(eval_games)))
    benchmarks = []
    best_workers = candidates[0]
    best_score = -1.0

    for workers in candidates:
        bench_cfg = eval_config_cls(
            num_games=pilot_games,
            promotion_threshold=0.55,
            confidence=0.95,
            sanity_check_interval=10**9,
            sanity_games=0,
            color_swap=True,
            max_moves=cfg.get("max_moves", 500),
            seed=cfg.get("seed", None),
            parallel_workers=workers,
        )
        evaluator = training_evaluator_cls(config=bench_cfg)
        candidate = candidate_factory()
        champion = champion_factory()
        try:
            t0 = time.time()
            result = evaluator.evaluate_checkpoint(
                candidate=candidate,
                champion=champion,
                game_factory=game_factory,
                candidate_id="eval_autotune",
                generation=0,
                candidate_factory=candidate_factory,
                champion_factory=champion_factory,
            )
            elapsed = max(time.time() - t0, 1e-6)
            scored_games = int(result.tally.get("scored", 0) if result.tally else 0)
            total_games = int(result.conditions.get("num_games", pilot_games))
            games_per_s = float(total_games) / elapsed
            score = games_per_s * (1.0 if scored_games > 0 else 0.5)
            row = {
                "workers": workers,
                "games": total_games,
                "elapsed_s": round(elapsed, 3),
                "games_per_s": round(games_per_s, 4),
                "scored_games": scored_games,
                "score": round(score, 4),
            }
            benchmarks.append(row)
            if score > best_score:
                best_score = score
                best_workers = workers
        except Exception as exc:
            benchmarks.append({"workers": workers, "error": str(exc)})
        finally:
            try:
                candidate.reset()
            except Exception:
                pass
            try:
                champion.reset()
            except Exception:
                pass

    save_eval_autotune_profile_fn(profile_path, hw, cfg, eval_games, best_workers, benchmarks)
    return best_workers, benchmarks


__all__ = ["benchmark_eval_parallel_workers", "make_json_safe"]
