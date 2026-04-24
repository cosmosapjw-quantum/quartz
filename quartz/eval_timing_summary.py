"""Helpers for summarizing evaluation timing payloads."""

from __future__ import annotations


def _safe_float(value, default=0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _sum_match_elapsed(rows: list[dict]) -> float:
    total = 0.0
    seen_batches = set()
    for row in rows:
        timing = row.get("timing_s") or {}
        batch_id = timing.get("batch_id")
        if batch_id:
            if batch_id in seen_batches:
                continue
            seen_batches.add(batch_id)
            total += _safe_float(timing.get("batch_elapsed_s"))
        else:
            total += _safe_float(timing.get("match_elapsed_s"))
    return total


def summarize_ablation_eval_timings(eval_payload: dict | None) -> dict:
    payload = eval_payload or {}
    matches = list(payload.get("matches") or [])
    condition_timings = dict(payload.get("eval_condition_timings") or {})

    total_games = 0
    by_condition_rows: dict[str, list[dict]] = {}
    for row in matches:
        games = int(row.get("games", 0) or 0)
        total_games += games
        by_condition_rows.setdefault(str(row.get("eval_condition")), []).append(row)
    total_match_elapsed_s = _sum_match_elapsed(matches)

    total_startup_s = 0.0
    by_condition = []
    for eval_name, timing in sorted(condition_timings.items()):
        rows = by_condition_rows.get(eval_name, [])
        games = sum(int(row.get("games", 0) or 0) for row in rows)
        match_elapsed_s = _sum_match_elapsed(rows)
        startup_s = (
            _safe_float(timing.get("cfg_build_s"))
            + _safe_float(timing.get("engine_load_s"))
            + _safe_float(timing.get("campaign_bootstrap_s"))
        )
        total_startup_s += startup_s
        total_s = startup_s + match_elapsed_s
        by_condition.append(
            {
                "eval_condition": eval_name,
                "games": games,
                "pairs": int(timing.get("pairs", 0) or 0),
                "engine_count": int(timing.get("engine_count", 0) or 0),
                "startup_s": startup_s,
                "match_elapsed_s": match_elapsed_s,
                "total_elapsed_s": total_s,
                "games_per_s_search_only": (games / match_elapsed_s) if match_elapsed_s > 0 else None,
                "games_per_s_end_to_end": (games / total_s) if total_s > 0 else None,
                "startup_share": (startup_s / total_s) if total_s > 0 else None,
            }
        )

    total_elapsed_s = total_startup_s + total_match_elapsed_s
    return {
        "conditions": by_condition,
        "total_games": total_games,
        "total_startup_s": total_startup_s,
        "total_match_elapsed_s": total_match_elapsed_s,
        "total_elapsed_s": total_elapsed_s,
        "games_per_s_search_only": (total_games / total_match_elapsed_s) if total_match_elapsed_s > 0 else None,
        "games_per_s_end_to_end": (total_games / total_elapsed_s) if total_elapsed_s > 0 else None,
        "startup_share": (total_startup_s / total_elapsed_s) if total_elapsed_s > 0 else None,
    }


def summarize_controller_stage2_timings(stage2_payload: dict | None) -> dict:
    payload = stage2_payload or {}
    matches = list(payload.get("matches") or [])
    checkpoint_timings = dict(payload.get("checkpoint_timings") or {})

    total_games = 0
    total_match_elapsed_s = 0.0
    by_checkpoint_rows: dict[str, list[dict]] = {}
    for row in matches:
        games = int(row.get("games", 0) or 0)
        total_games += games
        total_match_elapsed_s += _safe_float((row.get("timing_s") or {}).get("match_elapsed_s"))
        by_checkpoint_rows.setdefault(str(row.get("checkpoint_path")), []).append(row)

    total_startup_s = 0.0
    by_checkpoint = []
    for checkpoint_path, timing in sorted(checkpoint_timings.items()):
        rows = by_checkpoint_rows.get(checkpoint_path, [])
        games = sum(int(row.get("games", 0) or 0) for row in rows)
        match_elapsed_s = sum(
            _safe_float((row.get("timing_s") or {}).get("match_elapsed_s"))
            for row in rows
        )
        startup_s = _safe_float(timing.get("client_bootstrap_s"))
        total_startup_s += startup_s
        total_s = startup_s + match_elapsed_s
        by_checkpoint.append(
            {
                "checkpoint_path": checkpoint_path,
                "pairs": int(timing.get("pairs", 0) or 0),
                "client_count": int(timing.get("client_count", 0) or 0),
                "games": games,
                "startup_s": startup_s,
                "match_elapsed_s": match_elapsed_s,
                "total_elapsed_s": total_s,
                "games_per_s_search_only": (games / match_elapsed_s) if match_elapsed_s > 0 else None,
                "games_per_s_end_to_end": (games / total_s) if total_s > 0 else None,
                "startup_share": (startup_s / total_s) if total_s > 0 else None,
            }
        )

    total_elapsed_s = total_startup_s + total_match_elapsed_s
    return {
        "checkpoints": by_checkpoint,
        "total_games": total_games,
        "total_startup_s": total_startup_s,
        "total_match_elapsed_s": total_match_elapsed_s,
        "total_elapsed_s": total_elapsed_s,
        "games_per_s_search_only": (total_games / total_match_elapsed_s) if total_match_elapsed_s > 0 else None,
        "games_per_s_end_to_end": (total_games / total_elapsed_s) if total_elapsed_s > 0 else None,
        "startup_share": (total_startup_s / total_elapsed_s) if total_elapsed_s > 0 else None,
    }
