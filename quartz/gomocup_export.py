"""Helpers for exporting a QUARTZ champion as a Gomocup-ready bundle."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from .runtime_support import AlphaZeroNet, GAME_CONFIGS
from .backend import load_torch_state_dict_checked
from .encoders import get_encoder
from .onnx_support import export_onnx

EXPORT_SEARCH_KEYS = (
    "search_profile",
    "vl_mode",
    "tt_enabled",
    "c_puct",
    "sigma_0",
    "min_visits",
    "check_interval",
    "budget_ms",
    "max_visits",
)


def gomocup_rule_for_game(game_name: str) -> str:
    mapping = {
        "gomoku15": "freestyle",
        "gomoku15_free": "freestyle",
        "gomoku15_std": "standard",
        "gomoku15_renju": "renju",
        "gomoku15_caro": "caro",
        "gomoku15_omok": "omok",
    }
    return mapping.get(str(game_name or "").strip().lower(), "freestyle")


def normalize_search_cfg(search_cfg: dict | None) -> dict:
    cfg = {}
    for key in EXPORT_SEARCH_KEYS:
        if search_cfg and search_cfg.get(key) is not None:
            cfg[key] = search_cfg[key]
    return cfg


def build_gomocup_manifest(
    game_name: str,
    model_path: str | Path,
    onnx_name: str,
    *,
    include_checkpoint: bool = True,
    metadata: dict | None = None,
    search_cfg: dict | None = None,
) -> dict:
    metadata = dict(metadata or {})
    search_cfg = normalize_search_cfg(search_cfg)
    about = {
        "name": metadata.pop("about_name", "QUARTZ-Gomocup"),
        "version": metadata.pop("about_version", "0.2"),
        "author": metadata.pop("author", "cosmosapjw+Codex"),
        "country": metadata.pop("country", "KR"),
    }
    source = {
        "model_path": str(model_path),
        "condition": metadata.pop("condition", None),
        "seed": metadata.pop("seed", None),
        "training_metrics": metadata.pop("training_metrics", {}),
        "selection_metrics": metadata.pop("selection_metrics", {}),
        "extra": metadata,
    }
    return {
        "format_version": 1,
        "game": game_name,
        "gomocup_rule": gomocup_rule_for_game(game_name),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "onnx_model": onnx_name,
        "checkpoint_copy": "champion.pt" if include_checkpoint else None,
        "about": about,
        "search": search_cfg,
        "source": source,
    }


def export_gomocup_bundle(
    model_path: str | Path,
    game_name: str,
    output_dir: str | Path,
    *,
    metadata: dict | None = None,
    search_cfg: dict | None = None,
    include_checkpoint: bool = True,
    onnx_name: str = "gomocup_model.onnx",
    manifest_name: str = "gomocup_manifest.json",
    verbose: bool = False,
) -> dict:
    import torch

    if game_name not in GAME_CONFIGS:
        raise KeyError(f"unsupported game for export: {game_name}")

    model_path = Path(model_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = dict(GAME_CONFIGS[game_name])
    cfg["_name"] = game_name
    try:
        cfg["_encoder"] = get_encoder(game_name)
    except KeyError:
        cfg["_encoder"] = None

    model = AlphaZeroNet(cfg).to(torch.device("cpu"))
    load_torch_state_dict_checked(model, str(model_path), torch, map_location="cpu")
    model.eval()

    onnx_path = output_dir / onnx_name
    export_onnx(model, cfg, str(onnx_path), verbose=verbose)

    copied_checkpoint = None
    if include_checkpoint:
        copied_checkpoint = output_dir / "champion.pt"
        shutil.copy2(model_path, copied_checkpoint)

    manifest = build_gomocup_manifest(
        game_name,
        model_path,
        onnx_name,
        include_checkpoint=include_checkpoint,
        metadata=metadata,
        search_cfg=search_cfg,
    )
    manifest_path = output_dir / manifest_name
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return {
        "bundle_dir": str(output_dir),
        "onnx_path": str(onnx_path),
        "manifest_path": str(manifest_path),
        "checkpoint_copy": str(copied_checkpoint)
        if copied_checkpoint is not None
        else None,
        "manifest": manifest,
    }
