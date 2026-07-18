#!/usr/bin/env python3
"""CLI-adjacent runtime helpers for QUARTZ."""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CliRuntimeHooks:
    alphazero_net_cls: type
    load_torch_state_dict: object
    run_model_batch: object | None = None


def load_actor_source_from_checkpoint(
    checkpoint_path,
    cfg,
    device,
    backend_preference="torch",
    backend_template=None,
    runtime_hooks: CliRuntimeHooks | None = None,
):
    """Load an inference actor from checkpoint using the matching backend."""
    backend_preference = str(backend_preference or "torch").lower()
    if backend_preference == "jax":
        if (
            backend_template is not None
            and getattr(backend_template, "name", "") == "jax"
        ):
            return backend_template.load_actor(checkpoint_path)
        from quartz.backend import create_backend

        backend = create_backend(cfg, device="jax", preference="jax")
        if hasattr(backend, "load_actor"):
            return backend.load_actor(checkpoint_path)
        if not backend.load(checkpoint_path):
            raise FileNotFoundError(checkpoint_path)
        return backend.create_actor()

    import torch

    if runtime_hooks is None:
        from quartz.models_torch import AlphaZeroNet
        from quartz.backend import load_torch_state_dict

        runtime_hooks = CliRuntimeHooks(
            alphazero_net_cls=AlphaZeroNet,
            load_torch_state_dict=load_torch_state_dict,
        )

    actor = runtime_hooks.alphazero_net_cls(cfg).to(device)
    actor.load_state_dict(
        runtime_hooks.load_torch_state_dict(checkpoint_path, torch, map_location=device)
    )
    actor.eval()
    return actor


def serve(model, cfg, device, runtime_hooks: CliRuntimeHooks | None = None):
    import sys

    model.eval()
    print(f"alphazero_server ready ({cfg['_name']})", file=sys.stderr, flush=True)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            if req.get("cmd") == "quit":
                break
            features = req.get("features", [])
            action_mask = req.get("action_mask", [])
            n_act = req.get("num_actions", cfg["actions"])
            expected = cfg["ch"] * cfg["board"] * cfg["board"]
            if runtime_hooks is not None and runtime_hooks.run_model_batch is not None:
                if len(features) == expected:
                    batch = [
                        np.array(features, dtype=np.float32).reshape(
                            cfg["ch"], cfg["board"], cfg["board"]
                        )
                    ]
                else:
                    batch = [
                        np.zeros(
                            (cfg["ch"], cfg["board"], cfg["board"]), dtype=np.float32
                        )
                    ]
                probs_batch, vals_np = runtime_hooks.run_model_batch(
                    model, device, batch
                )
                probs = probs_batch[0]
                value = float(vals_np[0])
            else:
                import torch
                import torch.nn.functional as F

                if len(features) == expected:
                    x = (
                        torch.tensor(features, dtype=torch.float32)
                        .reshape(1, cfg["ch"], cfg["board"], cfg["board"])
                        .to(device)
                    )
                else:
                    x = torch.zeros(
                        1, cfg["ch"], cfg["board"], cfg["board"], device=device
                    )
                with torch.no_grad():
                    logits, val = model(x)
                    probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
                value = float(val.item())
            masked = np.zeros(n_act, dtype=np.float32)
            for i in range(min(len(action_mask), n_act)):
                if action_mask[i]:
                    masked[i] = probs[i] if i < len(probs) else 0.0
            s = masked.sum()
            if s > 1e-8:
                masked /= s
            print(
                json.dumps({"status": "ok", "policy": masked.tolist(), "value": value}),
                flush=True,
            )
        except Exception:
            print(
                json.dumps({"status": "error", "policy": [], "value": 0.0}), flush=True
            )
