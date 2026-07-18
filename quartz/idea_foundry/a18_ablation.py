"""A18 deterministic evaluator ablation substrate.

This module deliberately separates three things that are easy to conflate:

* an existing AlphaZero checkpoint used only to initialise both arms;
* a parameter/inference-FLOP matched direct evaluator comparison;
* evidence about playing strength (which this module never produces).

The two arms instantiate the exact same model.  Both execute the same noisy
auxiliary branch during training, but the matched baseline assigns that branch
zero loss weight.  ``forward`` never calls the auxiliary branch, so leaf
evaluation is deterministic and contains no denoising loop or random sample.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import random
import shutil
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from quartz.backend import load_checkpoint_with_metadata, validate_torch_state_dict
from quartz.models_torch import AlphaZeroNet
from quartz.replay import ReplayBuffer
from quartz.train_loop import train_epoch

A18_SCHEMA_VERSION = 1
A18_AXIS_ID = "A18"
A18_MODEL_FAMILY = "a18_shared_direct_residual_aux_v1"
BASELINE_VARIANT = "matched_direct_baseline"
DIFFUSION_VARIANT = "latent_gaussian_denoise"
VARIANTS = (BASELINE_VARIANT, DIFFUSION_VARIANT)
CHECKPOINT_ROLE = "trained_a18_evaluator"
DIRECT_INFERENCE_CONTRACT = "clean_direct_policy_value_only_v1"
DIAGNOSTIC_PLOT_NAME = "DIAGNOSTIC_a18_evaluator_ablation.png"
A18_HOLDOUT_DERIVATION_RULE = (
    "preserve source order and exclude every position whose exact float32 state "
    "identity occurs in the paired training replay"
)

_MODEL_CFG_KEYS = ("board", "ch", "actions", "filters", "blocks", "vh")
_CHECKPOINT_CONTROLLER_KEYS = (
    "search_profile",
    "vl_mode",
    "penalty_mode",
    "prior_refresh_rate",
    "prior_refresh_temp",
    "c_puct",
    "iters",
    "n_threads",
    "batch_size",
)
_CONTROLLER_KEYS = _CHECKPOINT_CONTROLLER_KEYS + ("root_only_shaping",)


class A18ContractError(RuntimeError):
    """Raised when an A18 experiment would violate its preregistered contract."""


def _json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def stable_hash(payload: Any) -> str:
    return hashlib.sha256(_json_bytes(payload)).hexdigest()


def sha256_file(path: str | os.PathLike[str]) -> str:
    target = Path(path)
    if not target.is_file():
        raise A18ContractError(f"required file is absent: {target}")
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        tmp_path = Path(handle.name)
    os.replace(tmp_path, path)


def _atomic_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        for row in rows:
            handle.write(_json_bytes(row).decode("utf-8"))
            handle.write("\n")
        tmp_path = Path(handle.name)
    os.replace(tmp_path, path)


def write_deterministic_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    """Write a byte-reproducible tensor+JSON checkpoint container."""

    state_dict = payload.get("model_state_dict")
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise A18ContractError(
            "deterministic checkpoint requires a non-empty state_dict"
        )
    tensor_rows = []
    tensor_blobs = []
    for index, key in enumerate(sorted(state_dict)):
        tensor = state_dict[key]
        array = np.asarray(tensor.detach().cpu().numpy()).copy(order="C")
        buffer = io.BytesIO()
        np.lib.format.write_array(buffer, array, version=(2, 0), allow_pickle=False)
        name = f"tensors/{index:04d}.npy"
        tensor_rows.append({"key": key, "path": name})
        tensor_blobs.append((name, buffer.getvalue()))
    metadata = {
        key: value for key, value in payload.items() if key != "model_state_dict"
    }
    metadata["checkpoint_format"] = "quartz_a18_deterministic_zip_v1"
    metadata["tensor_index"] = tensor_rows
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = path.parents[2] / ".a18_checkpoint_tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w+b",
        dir=temp_dir,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        tmp_path = Path(handle.name)
    try:
        with zipfile.ZipFile(
            tmp_path, mode="w", compression=zipfile.ZIP_STORED
        ) as archive:
            for name, data in [
                ("checkpoint.json", _json_bytes(metadata)),
                *tensor_blobs,
            ]:
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_STORED
                info.create_system = 3
                info.external_attr = 0o100644 << 16
                archive.writestr(info, data)
        os.replace(tmp_path, path)
        try:
            temp_dir.rmdir()
        except OSError:
            pass
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def _clean_checkpoint_temps(output: Path) -> None:
    temp_dir = output / ".a18_checkpoint_tmp"
    if not temp_dir.exists():
        return
    for path in temp_dir.glob(".*.a18ckpt.*.tmp"):
        if path.is_file() and not path.is_symlink():
            path.unlink()
    try:
        temp_dir.rmdir()
    except OSError as exc:
        raise A18ContractError(
            f"unexpected file in A18 checkpoint temp directory: {exc}"
        ) from exc


def read_deterministic_checkpoint(path: Path) -> dict[str, Any]:
    import torch

    try:
        with zipfile.ZipFile(path, mode="r") as archive:
            metadata = json.loads(archive.read("checkpoint.json").decode("utf-8"))
            if metadata.get("checkpoint_format") != "quartz_a18_deterministic_zip_v1":
                raise A18ContractError(f"unsupported A18 checkpoint format: {path}")
            tensor_index = metadata.pop("tensor_index")
            state_dict = {}
            for row in tensor_index:
                array = np.load(
                    io.BytesIO(archive.read(row["path"])), allow_pickle=False
                )
                state_dict[str(row["key"])] = torch.from_numpy(array.copy())
    except (
        OSError,
        KeyError,
        ValueError,
        zipfile.BadZipFile,
        json.JSONDecodeError,
    ) as exc:
        raise A18ContractError(
            f"cannot read deterministic A18 checkpoint {path}"
        ) from exc
    metadata["model_state_dict"] = state_dict
    return metadata


def canonical_model_cfg(cfg: Mapping[str, Any]) -> dict[str, int]:
    result: dict[str, int] = {}
    for key in _MODEL_CFG_KEYS:
        value = cfg.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise A18ContractError(f"model cfg {key!r} must be a positive integer")
        result[key] = int(value)
    return result


def canonical_controller_contract(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Return controller fields that are explicitly stored in a checkpoint."""

    missing = [key for key in _CHECKPOINT_CONTROLLER_KEYS if key not in cfg]
    if missing:
        raise A18ContractError(
            "bootstrap checkpoint lacks fixed controller fields: " + ", ".join(missing)
        )
    return {key: cfg[key] for key in _CHECKPOINT_CONTROLLER_KEYS}


def architecture_contract(cfg: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": A18_SCHEMA_VERSION,
        "model_family": A18_MODEL_FAMILY,
        "model_cfg": canonical_model_cfg(cfg),
        "direct_inference": DIRECT_INFERENCE_CONTRACT,
        "auxiliary_head": "three_by_three_noise_predictor",
        "spatial_alignment": "same_resolution_padding_one",
        "policy_head": "existing_spatial_head",
        "value_head": "existing_global_mean_pool",
    }


def set_reproducible_seed(seed: int, *, deterministic_algorithms: bool = True) -> None:
    import torch

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_algorithms:
        torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def determinism_runtime_contract() -> dict[str, Any]:
    import torch

    contract = {
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "float32_matmul_precision": str(torch.get_float32_matmul_precision()),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }
    expected = {
        "deterministic_algorithms": True,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
        "float32_matmul_precision": "highest",
        "cublas_workspace_config": ":4096:8",
    }
    if contract != expected:
        raise A18ContractError(f"determinism runtime contract mismatch: {contract!r}")
    return contract


class A18MatchedEvaluator(__import__("torch").nn.Module):
    """Direct evaluator with a training-only, parameter-matched aux branch."""

    def __init__(self, cfg: Mapping[str, Any]):
        import torch.nn as nn

        super().__init__()
        self.cfg = canonical_model_cfg(cfg)
        self.direct = AlphaZeroNet(self.cfg)
        channels = self.cfg["filters"]
        # No normalisation layer is used here: the direct path owns all running
        # statistics, and auxiliary training must not mutate inference state.
        self.aux_noise_head = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 1, bias=True),
        )

    def encode_clean(self, states):
        return self.direct.tower(self.direct.input_conv(states))

    def heads_from_latent(self, latent):
        policy = self.direct.p_head(latent).reshape(latent.size(0), -1)
        policy = self.direct.p_fc(policy)
        value = self.direct.v_head(latent).mean(dim=(2, 3))
        value = self.direct.v_fc(value).squeeze(-1)
        return policy, value

    def forward(self, states):
        # Production inference is intentionally this clean, direct path only.
        return self.heads_from_latent(self.encode_clean(states))

    def training_losses(
        self,
        states,
        policies,
        values,
        *,
        variant: str,
        denoise_weight: float,
        corruption_sigma: float,
        generator,
    ):
        import torch
        import torch.nn.functional as F

        if variant not in VARIANTS:
            raise A18ContractError(f"unknown A18 variant: {variant}")
        latent = self.encode_clean(states)
        logits, predicted_values = self.heads_from_latent(latent)
        policy_loss = -(policies * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
        value_loss = F.mse_loss(predicted_values, values)

        noise = torch.randn(
            latent.shape,
            dtype=latent.dtype,
            device=latent.device,
            generator=generator,
        )
        noisy_latent = latent + float(corruption_sigma) * noise
        predicted_noise = self.aux_noise_head(noisy_latent)
        auxiliary_loss = F.mse_loss(predicted_noise, noise)
        effective_weight = 0.0 if variant == BASELINE_VARIANT else float(denoise_weight)
        total = policy_loss + value_loss + effective_weight * auxiliary_loss
        return total, policy_loss, value_loss, auxiliary_loss


class A18TrainingBackend:
    """``train_epoch``-compatible A18 PyTorch backend.

    Keeping the adapter at the backend boundary lets the established replay
    sampling, batching, gradient clipping, and epoch accounting remain the
    common training substrate.
    """

    def __init__(
        self,
        model: A18MatchedEvaluator,
        *,
        variant: str,
        seed: int,
        device: str,
        learning_rate: float,
        weight_decay: float,
        momentum: float,
        grad_clip_norm: float,
        denoise_weight: float,
        corruption_sigma: float,
    ):
        import torch

        if variant not in VARIANTS:
            raise A18ContractError(f"unknown A18 variant: {variant}")
        self.name = "torch_a18"
        self.model = model
        self.variant = variant
        self.device = torch.device(device)
        self.model.to(self.device)
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=float(learning_rate),
            momentum=float(momentum),
            weight_decay=float(weight_decay),
        )
        self.grad_clip_norm = float(grad_clip_norm)
        generator_device = self.device.type if self.device.type == "cuda" else "cpu"
        self.noise_generator = torch.Generator(device=generator_device)
        self.noise_generator.manual_seed(int(seed))
        self.denoise_weight = float(denoise_weight)
        self.corruption_sigma = float(corruption_sigma)
        self.learner_updates_completed = 0
        self.last_auxiliary_loss: float | None = None

    def train_step(self, states_np, policies_np, values_np):
        import torch

        self.model.train()
        states = torch.as_tensor(states_np, dtype=torch.float32, device=self.device)
        policies = torch.as_tensor(policies_np, dtype=torch.float32, device=self.device)
        values = torch.as_tensor(values_np, dtype=torch.float32, device=self.device)
        total, policy_loss, value_loss, auxiliary_loss = self.model.training_losses(
            states,
            policies,
            values,
            variant=self.variant,
            denoise_weight=self.denoise_weight,
            corruption_sigma=self.corruption_sigma,
            generator=self.noise_generator,
        )
        self.optimizer.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
        self.optimizer.step()
        self.learner_updates_completed += 1
        self.last_auxiliary_loss = float(auxiliary_loss.detach().cpu())
        return (
            float(total.detach().cpu()),
            float(policy_loss.detach().cpu()),
            float(value_loss.detach().cpu()),
        )

    def eval(self):
        self.model.eval()
        return self

    def predict(self, states_np):
        import torch

        self.model.eval()
        with torch.inference_mode():
            states = torch.as_tensor(states_np, dtype=torch.float32, device=self.device)
            if states.ndim == 3:
                states = states.unsqueeze(0)
            logits, values = self.model(states)
            return (
                torch.softmax(logits, dim=-1).cpu().numpy(),
                values.cpu().numpy(),
            )


def load_bootstrap_model(
    checkpoint_path: str | os.PathLike[str],
    *,
    seed: int,
    device: str,
) -> tuple[A18MatchedEvaluator, dict[str, Any]]:
    import torch

    path = Path(checkpoint_path)
    checkpoint_sha256 = sha256_file(path)
    state_dict, cfg = load_checkpoint_with_metadata(path, torch, map_location="cpu")
    if not isinstance(cfg, Mapping):
        raise A18ContractError(f"bootstrap checkpoint has no wrapped cfg: {path}")
    model_cfg = canonical_model_cfg(cfg)
    set_reproducible_seed(seed)
    model = A18MatchedEvaluator(model_cfg)
    reason = validate_torch_state_dict(model.direct, state_dict)
    if reason is not None:
        raise A18ContractError(f"bootstrap checkpoint is incompatible: {reason}")
    model.direct.load_state_dict(state_dict, strict=True)
    model.to(device)
    return model, {
        "path": str(path.resolve()),
        "sha256": checkpoint_sha256,
        "model_cfg": model_cfg,
        "controller_contract": canonical_controller_contract(cfg),
        "parameter_count": int(sum(p.numel() for p in model.direct.parameters())),
    }


def parameter_count(model: A18MatchedEvaluator) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def parameter_breakdown(model: A18MatchedEvaluator) -> dict[str, int]:
    deployed = int(sum(parameter.numel() for parameter in model.direct.parameters()))
    train_only = int(
        sum(parameter.numel() for parameter in model.aux_noise_head.parameters())
    )
    return {
        "deployed_direct": deployed,
        "train_only_auxiliary": train_only,
        "total_training": deployed + train_only,
    }


def _count_module_flops(model, invoke) -> int:
    import torch.nn as nn

    total = 0
    hooks = []

    def count_conv(module, inputs, output):
        nonlocal total
        output_elements = int(output.numel())
        kernel = int(module.kernel_size[0] * module.kernel_size[1])
        operations = kernel * int(module.in_channels // module.groups) * 2
        total += output_elements * operations

    def count_linear(module, inputs, output):
        nonlocal total
        total += int(output.numel()) * int(module.in_features) * 2

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(count_conv))
        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(count_linear))
    try:
        invoke()
    finally:
        for hook in hooks:
            hook.remove()
    return int(total)


def estimate_direct_flops(model: A18MatchedEvaluator, *, device: str = "cpu") -> int:
    """Count Conv/Linear multiply-adds for a batch-one direct inference.

    The convention is versioned and intentionally conservative: one multiply
    and one add count as two operations; normalisation/activation costs are not
    included.  The purpose is arm matching, not hardware throughput prediction.
    """

    import torch

    previous_training = model.training
    model.eval()
    try:
        cfg = model.cfg
        dummy = torch.zeros(
            (1, cfg["ch"], cfg["board"], cfg["board"]),
            dtype=torch.float32,
            device=device,
        )

        def invoke():
            with torch.inference_mode():
                model(dummy)

        return _count_module_flops(model, invoke)
    finally:
        model.train(previous_training)


def estimate_training_forward_flops(
    model: A18MatchedEvaluator,
    *,
    device: str = "cpu",
) -> int:
    """Count the common direct+aux training forward for one position."""

    import torch

    cfg = model.cfg
    states = torch.zeros(
        (1, cfg["ch"], cfg["board"], cfg["board"]),
        dtype=torch.float32,
        device=device,
    )
    policies = torch.full(
        (1, cfg["actions"]),
        1.0 / cfg["actions"],
        dtype=torch.float32,
        device=device,
    )
    values = torch.zeros((1,), dtype=torch.float32, device=device)
    generator_device = "cuda" if str(device).startswith("cuda") else "cpu"
    generator = torch.Generator(device=generator_device).manual_seed(0)
    previous_training = model.training
    model.train()
    try:

        def invoke():
            model.training_losses(
                states,
                policies,
                values,
                variant=BASELINE_VARIANT,
                denoise_weight=1.0,
                corruption_sigma=0.1,
                generator=generator,
            )

        return _count_module_flops(model, invoke)
    finally:
        model.train(previous_training)


def assert_direct_inference_deterministic(
    model: A18MatchedEvaluator,
    *,
    device: str,
) -> None:
    import torch

    cfg = model.cfg
    probe = torch.linspace(
        -1.0,
        1.0,
        steps=cfg["ch"] * cfg["board"] * cfg["board"],
        device=device,
    ).reshape(1, cfg["ch"], cfg["board"], cfg["board"])
    model.eval()
    with torch.inference_mode():
        first = model(probe)
        second = model(probe)
    if not all(torch.equal(left, right) for left, right in zip(first, second)):
        raise A18ContractError("direct inference is not bitwise deterministic")


def _validate_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    if spec.get("schema_version") != A18_SCHEMA_VERSION:
        raise A18ContractError("A18 spec schema_version must be 1")
    if spec.get("axis_id") != A18_AXIS_ID:
        raise A18ContractError("A18 spec axis_id must be 'A18'")
    evidence_tier = spec.get("evidence_tier")
    if evidence_tier not in {"smoke_readiness", "study_candidate"}:
        raise A18ContractError(
            "evidence_tier must be smoke_readiness or study_candidate"
        )
    seeds = spec.get("seeds")
    if not isinstance(seeds, list) or len(seeds) < 1:
        raise A18ContractError("seeds must be a non-empty list")
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds):
        raise A18ContractError("all seeds must be integers")
    if len(set(seeds)) != len(seeds):
        raise A18ContractError("seeds must be unique")
    if evidence_tier == "study_candidate" and len(seeds) < 3:
        raise A18ContractError("study_candidate requires at least three paired seeds")
    inputs = spec.get("inputs")
    if not isinstance(inputs, list) or {row.get("seed") for row in inputs} != set(
        seeds
    ):
        raise A18ContractError("inputs must cover each preregistered seed exactly")
    if len(inputs) != len(seeds):
        raise A18ContractError("inputs contain duplicate seed rows")
    if evidence_tier == "study_candidate":
        for row in inputs:
            if not isinstance(row.get("evaluation_replay"), str) or not isinstance(
                row.get("evaluation_replay_sha256"), str
            ):
                raise A18ContractError(
                    "study_candidate requires evaluation_replay and evaluation_replay_sha256 "
                    "for every seed"
                )
            source_seed = row.get("evaluation_source_seed")
            if isinstance(source_seed, bool) or not isinstance(source_seed, int):
                raise A18ContractError(
                    "study_candidate requires integer evaluation_source_seed"
                )
            if source_seed == row.get("seed") or source_seed not in seeds:
                raise A18ContractError(
                    "evaluation_source_seed must name a different registered seed"
                )
            if not isinstance(
                row.get("evaluation_source_replay"), str
            ) or not isinstance(row.get("evaluation_source_replay_sha256"), str):
                raise A18ContractError(
                    "study_candidate requires evaluation source replay lineage"
                )
            if not isinstance(
                row.get("evaluation_derivation_receipt"), str
            ) or not isinstance(row.get("evaluation_derivation_receipt_sha256"), str):
                raise A18ContractError(
                    "study_candidate requires a hashed evaluation derivation receipt"
                )
    compute = spec.get("compute_contract")
    if not isinstance(compute, Mapping):
        raise A18ContractError("compute_contract must be an object")
    for key in ("learner_updates", "batch_size", "evaluation_positions"):
        value = compute.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise A18ContractError(f"compute_contract.{key} must be a positive integer")
    for key in (
        "learning_rate",
        "weight_decay",
        "momentum",
        "grad_clip_norm",
        "denoise_weight",
        "corruption_sigma",
    ):
        value = compute.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise A18ContractError(f"compute_contract.{key} must be numeric")
        if not math.isfinite(float(value)) or float(value) < 0.0:
            raise A18ContractError(
                f"compute_contract.{key} must be finite and non-negative"
            )
    if float(compute["denoise_weight"]) <= 0.0:
        raise A18ContractError("diffusion denoise_weight must be positive")
    if compute.get("optimizer") != "sgd":
        raise A18ContractError("compute_contract.optimizer must be sgd")
    if compute.get("scheduler") != "none":
        raise A18ContractError("compute_contract.scheduler must be none")
    if not 0.0 <= float(compute["momentum"]) < 1.0:
        raise A18ContractError("compute_contract.momentum must be in [0, 1)")
    if float(compute["grad_clip_norm"]) <= 0.0:
        raise A18ContractError("compute_contract.grad_clip_norm must be positive")
    controller = spec.get("controller_contract")
    if not isinstance(controller, Mapping) or not controller:
        raise A18ContractError("controller_contract must be a non-empty object")
    if set(controller) != set(_CONTROLLER_KEYS):
        raise A18ContractError(
            "controller_contract must contain exactly: " + ", ".join(_CONTROLLER_KEYS)
        )
    if not isinstance(controller.get("root_only_shaping"), bool):
        raise A18ContractError("controller_contract.root_only_shaping must be boolean")
    if spec.get("controller_frozen") is not True:
        raise A18ContractError("controller_frozen must be true")
    if spec.get("automatic_promotion") is not False:
        raise A18ContractError("automatic_promotion must be false")
    return dict(spec)


def load_spec(path: str | os.PathLike[str]) -> dict[str, Any]:
    target = Path(path)
    try:
        spec = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise A18ContractError(f"cannot read A18 spec {target}: {exc}") from exc
    if not isinstance(spec, Mapping):
        raise A18ContractError("A18 spec root must be an object")
    checked = _validate_spec(spec)
    checked["__spec_path"] = str(target.resolve())
    return checked


def _source_hashes() -> list[dict[str, str]]:
    repo_root = Path(__file__).resolve().parents[2]
    sources = [
        Path(__file__).resolve(),
        repo_root / "scripts" / "a18_evaluator_ablation.py",
        repo_root / "scripts" / "a18_prepare_holdouts.py",
        repo_root / "quartz" / "models_torch.py",
        repo_root / "quartz" / "backend.py",
        repo_root / "quartz" / "replay.py",
        repo_root / "quartz" / "train_loop.py",
    ]
    return [
        {
            "path": str(path.relative_to(repo_root)),
            "sha256": sha256_file(path),
        }
        for path in sources
    ]


def _input_hashes(
    spec: Mapping[str, Any],
    inputs: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    spec_path = spec.get("__spec_path")
    public_spec = {
        key: value for key, value in spec.items() if not key.startswith("__")
    }
    hashes = {
        "experiment_spec": (
            sha256_file(spec_path) if spec_path else stable_hash(public_spec)
        ),
    }
    for row in inputs:
        seed = int(row["seed"])
        hashes[f"seed_{seed}.bootstrap_checkpoint"] = str(
            row["bootstrap_checkpoint_sha256"]
        )
        hashes[f"seed_{seed}.checkpoint_status"] = str(row["checkpoint_status_sha256"])
        hashes[f"seed_{seed}.training_replay"] = str(row["training_replay_sha256"])
        if row.get("evaluation_replay_sha256"):
            hashes[f"seed_{seed}.evaluation_replay"] = str(
                row["evaluation_replay_sha256"]
            )
            hashes[f"seed_{seed}.evaluation_source_replay"] = str(
                row["evaluation_source_replay_sha256"]
            )
            hashes[f"seed_{seed}.evaluation_derivation_receipt"] = str(
                row["evaluation_derivation_receipt_sha256"]
            )
    return hashes


def _state_identity(state: np.ndarray) -> str:
    array = np.ascontiguousarray(state, dtype=np.float32)
    digest = hashlib.sha256()
    digest.update(_json_bytes({"shape": list(array.shape), "dtype": "float32"}))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _replay_examples_and_hashes(
    path: str | os.PathLike[str],
) -> tuple[list[Any], list[str]]:
    replay = ReplayBuffer(1_000_000)
    count = replay.load(str(path))
    if count <= 0:
        raise A18ContractError(f"replay contains no states: {path}")
    examples = replay.examples_at_indices(range(count))
    return examples, [_state_identity(example.state) for example in examples]


def replay_state_hash_contract(
    training_replay_path: str | os.PathLike[str],
    evaluation_replay_path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Verify exact serialized-state identity disjointness between replay shards."""

    training_examples, training_hash_list = _replay_examples_and_hashes(
        training_replay_path
    )
    evaluation_examples, evaluation_hash_list = _replay_examples_and_hashes(
        evaluation_replay_path
    )
    training_hashes = set(training_hash_list)
    evaluation_hashes = set(evaluation_hash_list)
    overlap = training_hashes & evaluation_hashes
    if overlap:
        raise A18ContractError(
            "training/evaluation replay state-hash groups overlap: "
            f"{len(overlap)} shared groups"
        )
    return {
        "schema_version": 1,
        "group_key": "sha256(float32_shape_and_c_order_state_bytes)",
        "training_positions": len(training_examples),
        "training_state_groups": len(training_hashes),
        "evaluation_positions": len(evaluation_examples),
        "evaluation_state_groups": len(evaluation_hashes),
        "overlap_state_groups": 0,
        "verified_disjoint": True,
    }


def _write_replay_deterministic(replay: ReplayBuffer, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=output.parent, prefix=".a18-replay-"
    ) as tmp_dir:
        raw_path = Path(tmp_dir) / "raw.npz"
        normalized_path = Path(tmp_dir) / "normalized.npz"
        replay.save(str(raw_path))
        with (
            zipfile.ZipFile(raw_path, "r") as source,
            zipfile.ZipFile(
                normalized_path,
                "w",
                compression=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            ) as target,
        ):
            for name in sorted(source.namelist()):
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.create_system = 3
                info.external_attr = 0o100644 << 16
                target.writestr(info, source.read(name), compresslevel=9)
        os.replace(normalized_path, output)


def derive_state_disjoint_evaluation_replay(
    training_replay_path: str | os.PathLike[str],
    evaluation_source_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    *,
    training_seed: int,
    evaluation_source_seed: int,
) -> dict[str, Any]:
    """Deterministically exclude source positions seen in the training replay."""

    if int(training_seed) == int(evaluation_source_seed):
        raise A18ContractError("evaluation_source_seed must differ from training_seed")
    training_examples, training_hashes_list = _replay_examples_and_hashes(
        training_replay_path
    )
    source_examples, source_hashes = _replay_examples_and_hashes(evaluation_source_path)
    training_hashes = set(training_hashes_list)
    retained = [
        (example, state_hash)
        for example, state_hash in zip(source_examples, source_hashes)
        if state_hash not in training_hashes
    ]
    if not retained:
        raise A18ContractError(
            "state-disjoint evaluation derivation retained no positions"
        )
    output_replay = ReplayBuffer(len(retained))
    for example, _state_hash in retained:
        entries = list(zip(example.policy.idx.tolist(), example.policy.val.tolist()))
        output_replay.add_sparse(
            example.state,
            entries,
            example.value,
            example.policy.n_actions,
            metadata=example.metadata,
        )
    output = Path(output_path)
    _write_replay_deterministic(output_replay, output)
    return _expected_state_disjoint_derivation_receipt(
        training_replay_path,
        evaluation_source_path,
        output,
        training_seed=training_seed,
        evaluation_source_seed=evaluation_source_seed,
    )


def _expected_state_disjoint_derivation_receipt(
    training_replay_path: str | os.PathLike[str],
    evaluation_source_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    *,
    training_seed: int,
    evaluation_source_seed: int,
) -> dict[str, Any]:
    """Reconstruct the canonical receipt from current inputs and output bytes."""

    if int(training_seed) == int(evaluation_source_seed):
        raise A18ContractError("evaluation_source_seed must differ from training_seed")
    training_examples, training_hashes_list = _replay_examples_and_hashes(
        training_replay_path
    )
    source_examples, source_hashes = _replay_examples_and_hashes(evaluation_source_path)
    derived_examples, derived_hashes = _replay_examples_and_hashes(output_path)
    training_hashes = set(training_hashes_list)
    expected_hashes = [
        state_hash for state_hash in source_hashes if state_hash not in training_hashes
    ]
    if not expected_hashes:
        raise A18ContractError(
            "state-disjoint evaluation derivation retained no positions"
        )
    if derived_hashes != expected_hashes:
        raise A18ContractError(
            "derived evaluation replay is not the ordered exclusion subset"
        )
    contract = replay_state_hash_contract(training_replay_path, output_path)
    excluded_hashes = set(source_hashes) & training_hashes
    output = Path(output_path)
    return {
        "schema_version": 1,
        "axis_id": A18_AXIS_ID,
        "artifact_kind": "evaluation_replay_derivation_receipt",
        "generation_rule": A18_HOLDOUT_DERIVATION_RULE,
        "training_seed": int(training_seed),
        "evaluation_source_seed": int(evaluation_source_seed),
        "training_replay": str(Path(training_replay_path).resolve()),
        "training_replay_sha256": sha256_file(training_replay_path),
        "evaluation_source_replay": str(Path(evaluation_source_path).resolve()),
        "evaluation_source_replay_sha256": sha256_file(evaluation_source_path),
        "output_replay": str(output.resolve()),
        "output_replay_sha256": sha256_file(output),
        "source_positions": len(source_examples),
        "source_state_groups": len(set(source_hashes)),
        "excluded_positions": len(source_examples) - len(derived_examples),
        "excluded_state_groups": len(excluded_hashes),
        "retained_positions": len(derived_examples),
        "retained_state_groups": len(set(derived_hashes)),
        "state_disjoint_contract": contract,
        "scientific_status": "DERIVED_INPUT_NOT_EFFICACY",
    }


def verify_state_disjoint_evaluation_replay_receipt(
    training_replay_path: str | os.PathLike[str],
    evaluation_source_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    receipt: Mapping[str, Any],
    *,
    training_seed: int,
    evaluation_source_seed: int,
) -> dict[str, Any]:
    """Fail closed unless an existing shard and receipt match the current rule."""

    if not isinstance(receipt, Mapping):
        raise A18ContractError("holdout derivation receipt must be a JSON object")
    expected = _expected_state_disjoint_derivation_receipt(
        training_replay_path,
        evaluation_source_path,
        output_path,
        training_seed=training_seed,
        evaluation_source_seed=evaluation_source_seed,
    )
    if _json_bytes(dict(receipt)) != _json_bytes(expected):
        raise A18ContractError(
            "existing holdout derivation receipt does not match current inputs, rule, "
            "or output hash"
        )
    return expected


def inspect_inputs(spec: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Validate and hash every real bootstrap checkpoint/replay input."""

    import torch

    checked = []
    expected_controller = dict(spec["controller_contract"])
    expected_model_cfg = None
    registered_inputs = {int(row["seed"]): row for row in spec["inputs"]}
    for row in sorted(spec["inputs"], key=lambda item: int(item["seed"])):
        seed = int(row["seed"])
        checkpoint = Path(row["bootstrap_checkpoint"])
        checkpoint_status_path = Path(row["checkpoint_status"])
        replay_path = Path(row["training_replay"])
        checkpoint_hash = sha256_file(checkpoint)
        checkpoint_status_hash = sha256_file(checkpoint_status_path)
        replay_hash = sha256_file(replay_path)
        declared_checkpoint_hash = row.get("bootstrap_checkpoint_sha256")
        declared_checkpoint_status_hash = row.get("checkpoint_status_sha256")
        declared_replay_hash = row.get("training_replay_sha256")
        if declared_checkpoint_hash != checkpoint_hash:
            raise A18ContractError(f"seed {seed} bootstrap checkpoint hash drift")
        if declared_checkpoint_status_hash != checkpoint_status_hash:
            raise A18ContractError(f"seed {seed} checkpoint status hash drift")
        if declared_replay_hash != replay_hash:
            raise A18ContractError(f"seed {seed} training replay hash drift")
        try:
            checkpoint_status = json.loads(
                checkpoint_status_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise A18ContractError(
                f"seed {seed} invalid checkpoint status receipt"
            ) from exc
        if checkpoint_status.get("preferred_posttrain_checkpoint") != checkpoint.name:
            raise A18ContractError(
                f"seed {seed} bootstrap is not the preferred post-train checkpoint"
            )
        if (
            checkpoint.name != "latest.pt"
            or checkpoint_status.get("latest_exists") is not True
        ):
            raise A18ContractError(
                f"seed {seed} requires an existing preferred latest.pt"
            )
        state_dict, cfg = load_checkpoint_with_metadata(
            checkpoint, torch, map_location="cpu"
        )
        if not isinstance(cfg, Mapping):
            raise A18ContractError(f"seed {seed} bootstrap checkpoint lacks cfg")
        model_cfg = canonical_model_cfg(cfg)
        controller = canonical_controller_contract(cfg)
        expected_checkpoint_controller = {
            key: expected_controller[key] for key in _CHECKPOINT_CONTROLLER_KEYS
        }
        if controller != expected_checkpoint_controller:
            raise A18ContractError(f"seed {seed} controller contract drift")
        if expected_model_cfg is None:
            expected_model_cfg = model_cfg
        elif model_cfg != expected_model_cfg:
            raise A18ContractError(f"seed {seed} model architecture drift")
        direct = AlphaZeroNet(model_cfg)
        reason = validate_torch_state_dict(direct, state_dict)
        if reason is not None:
            raise A18ContractError(
                f"seed {seed} bootstrap state_dict mismatch: {reason}"
            )
        replay = ReplayBuffer(1_000_000)
        replay_positions = replay.load(str(replay_path))
        if replay_positions < int(spec["compute_contract"]["evaluation_positions"]):
            raise A18ContractError(
                f"seed {seed} replay has {replay_positions} positions, below fixed evaluation count"
            )
        first = replay.examples_at_indices([0])[0]
        if tuple(first.state.shape) != (
            model_cfg["ch"],
            model_cfg["board"],
            model_cfg["board"],
        ):
            raise A18ContractError(f"seed {seed} replay state shape mismatch")
        if int(first.policy.n_actions) != model_cfg["actions"]:
            raise A18ContractError(f"seed {seed} replay action count mismatch")
        observed_root_only = []
        for example in replay.examples_at_indices(range(replay_positions)):
            summary = (example.metadata or {}).get("controller_summary")
            if not isinstance(summary, Mapping) or not isinstance(
                summary.get("root_only_shaping"), bool
            ):
                raise A18ContractError(
                    f"seed {seed} replay lacks complete root_only_shaping telemetry"
                )
            observed_root_only.append(summary["root_only_shaping"])
        expected_root_only = expected_controller["root_only_shaping"]
        if set(observed_root_only) != {expected_root_only}:
            raise A18ContractError(f"seed {seed} replay root_only_shaping drift")
        evaluation_receipt = None
        if spec["evidence_tier"] == "study_candidate":
            source_seed = int(row["evaluation_source_seed"])
            source_row = registered_inputs[source_seed]
            evaluation_source_path = Path(row["evaluation_source_replay"])
            if (
                evaluation_source_path.resolve()
                != Path(source_row["training_replay"]).resolve()
            ):
                raise A18ContractError(
                    f"seed {seed} evaluation source path is not registered seed {source_seed} replay"
                )
            if (
                row["evaluation_source_replay_sha256"]
                != source_row["training_replay_sha256"]
            ):
                raise A18ContractError(
                    f"seed {seed} evaluation source hash lineage drift"
                )
            if (
                sha256_file(evaluation_source_path)
                != row["evaluation_source_replay_sha256"]
            ):
                raise A18ContractError(f"seed {seed} evaluation source file hash drift")
            evaluation_path = Path(row["evaluation_replay"])
            evaluation_hash = sha256_file(evaluation_path)
            if evaluation_hash != row["evaluation_replay_sha256"]:
                raise A18ContractError(f"seed {seed} evaluation replay hash drift")
            receipt_path = Path(row["evaluation_derivation_receipt"])
            receipt_hash = sha256_file(receipt_path)
            if receipt_hash != row["evaluation_derivation_receipt_sha256"]:
                raise A18ContractError(
                    f"seed {seed} evaluation derivation receipt drift"
                )
            try:
                derivation = json.loads(receipt_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise A18ContractError(
                    f"seed {seed} invalid evaluation derivation receipt"
                ) from exc
            canonical_derivation = verify_state_disjoint_evaluation_replay_receipt(
                replay_path,
                evaluation_source_path,
                evaluation_path,
                derivation,
                training_seed=seed,
                evaluation_source_seed=source_seed,
            )
            state_groups = canonical_derivation["state_disjoint_contract"]
            if state_groups["evaluation_positions"] < int(
                spec["compute_contract"]["evaluation_positions"]
            ):
                raise A18ContractError(
                    f"seed {seed} evaluation replay is below fixed evaluation count"
                )
            evaluation_receipt = {
                "evaluation_replay": str(evaluation_path.resolve()),
                "evaluation_replay_sha256": evaluation_hash,
                "evaluation_source_seed": source_seed,
                "evaluation_source_replay": str(evaluation_source_path.resolve()),
                "evaluation_source_replay_sha256": row[
                    "evaluation_source_replay_sha256"
                ],
                "evaluation_derivation_receipt": str(receipt_path.resolve()),
                "evaluation_derivation_receipt_sha256": receipt_hash,
                "state_group_contract": state_groups,
            }
        checked.append(
            {
                "seed": seed,
                "bootstrap_checkpoint": str(checkpoint.resolve()),
                "bootstrap_checkpoint_sha256": checkpoint_hash,
                "checkpoint_status": str(checkpoint_status_path.resolve()),
                "checkpoint_status_sha256": checkpoint_status_hash,
                "checkpoint_selection_reason": "preferred_posttrain_checkpoint=latest.pt",
                "best_checkpoint_excluded_from_evidence": bool(
                    checkpoint_status.get("best_checkpoint_bootstrap_seeded")
                ),
                "bootstrap_parameter_count": int(
                    sum(tensor.numel() for tensor in state_dict.values())
                ),
                "training_replay": str(replay_path.resolve()),
                "training_replay_sha256": replay_hash,
                "replay_positions": int(replay_positions),
                "model_cfg": model_cfg,
                "controller_provenance": {
                    "checkpoint_cfg_fields": list(_CHECKPOINT_CONTROLLER_KEYS),
                    "replay_telemetry_fields": ["root_only_shaping"],
                    "root_only_shaping_coverage": 1.0,
                },
                **(evaluation_receipt or {}),
            }
        )
    return checked


def candidate_checkpoint_path(output_dir: Path, seed: int, variant: str) -> Path:
    return output_dir / "models" / f"seed_{int(seed)}" / f"{variant}.a18ckpt"


def _checkpoint_payload(
    backend: A18TrainingBackend,
    *,
    spec: Mapping[str, Any],
    input_row: Mapping[str, Any],
    training_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    model = backend.model
    model_contract = architecture_contract(model.cfg)
    params = parameter_breakdown(model)
    flops = estimate_direct_flops(model, device=str(backend.device))
    training_flops = estimate_training_forward_flops(model, device=str(backend.device))
    return {
        "schema_version": A18_SCHEMA_VERSION,
        "model_state_dict": model.state_dict(),
        "cfg": dict(model.cfg),
        "a18_metadata": {
            "schema_version": A18_SCHEMA_VERSION,
            "axis_id": A18_AXIS_ID,
            "checkpoint_role": CHECKPOINT_ROLE,
            "synthetic": False,
            "variant": backend.variant,
            "seed": int(input_row["seed"]),
            "evidence_tier": spec["evidence_tier"],
            "claim_scope": "diagnostic_ablation_readiness_only",
            "automatic_promotion": False,
            "model_family": A18_MODEL_FAMILY,
            "architecture_contract": model_contract,
            "architecture_hash": stable_hash(model_contract),
            "parameter_count": params["total_training"],
            "parameter_breakdown": params,
            "direct_inference_flops": flops,
            "training_forward_flops": training_flops,
            "flop_convention": "conv_linear_multiply_add_as_two_v1",
            "direct_inference_contract": DIRECT_INFERENCE_CONTRACT,
            "controller_contract": dict(spec["controller_contract"]),
            "controller_contract_hash": stable_hash(spec["controller_contract"]),
            "compute_contract": dict(spec["compute_contract"]),
            "compute_contract_hash": stable_hash(spec["compute_contract"]),
            "determinism_runtime_contract": determinism_runtime_contract(),
            "source_checkpoint_sha256": input_row["bootstrap_checkpoint_sha256"],
            "training_replay_sha256": input_row["training_replay_sha256"],
            "sample_schedule_hash": training_metrics["sample_schedule_hash"],
            "learner_updates_completed": int(backend.learner_updates_completed),
            "training_metrics": dict(training_metrics),
            "prohibited_inferences": [
                "play_strength",
                "efficacy",
                "production_readiness",
            ],
        },
    }


def _training_evidence_payload(
    spec: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    public_spec = {
        key: value for key, value in spec.items() if not key.startswith("__")
    }
    return {
        "schema_version": A18_SCHEMA_VERSION,
        "axis_id": A18_AXIS_ID,
        "role": "ablation_readiness",
        "scientific_status": "DIAGNOSTIC_ONLY",
        "spec_hash": stable_hash(public_spec),
        "controller_contract_hash": stable_hash(spec["controller_contract"]),
        "compute_contract_hash": stable_hash(spec["compute_contract"]),
        "seed_contract": {
            "mode": "paired_fixed",
            "seeds": list(spec["seeds"]),
            "variants": list(VARIANTS),
        },
        "rows": list(rows),
    }


def _load_existing_training_attempts(
    path: Path,
    spec: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise A18ContractError(f"invalid retained training evidence: {path}") from exc
    expected = _training_evidence_payload(spec, [])
    for key in (
        "schema_version",
        "axis_id",
        "role",
        "spec_hash",
        "controller_contract_hash",
        "compute_contract_hash",
        "seed_contract",
    ):
        if payload.get(key) != expected[key]:
            raise A18ContractError(f"retained training evidence {key} drift: {path}")
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise A18ContractError(f"retained training evidence rows are invalid: {path}")
    return [dict(row) for row in rows]


def train_candidates(
    spec: Mapping[str, Any],
    output_dir: str | os.PathLike[str],
    *,
    device: str,
) -> list[dict[str, Any]]:
    """Train both paired arms from each real bootstrap checkpoint."""
    output = Path(output_dir)
    set_reproducible_seed(int(spec["seeds"][0]))
    _clean_checkpoint_temps(output)
    training_rows_path = output / "training_rows.v1.json"
    checked_inputs = inspect_inputs(spec)
    rows = _load_existing_training_attempts(training_rows_path, spec)
    completed_rows: list[dict[str, Any]] = []
    for input_row in checked_inputs:
        seed = int(input_row["seed"])
        for variant in VARIANTS:
            pair_attempts = [
                row
                for row in rows
                if row.get("seed") == seed and row.get("variant") == variant
            ]
            successful = [
                row
                for row in pair_attempts
                if row.get("execution_status") == "succeeded"
            ]
            if len(successful) > 1:
                raise A18ContractError(
                    f"seed {seed} {variant} has duplicate success rows"
                )
            if successful:
                retained = successful[0]
                expected_path = candidate_checkpoint_path(
                    output, seed, variant
                ).resolve()
                if Path(str(retained.get("checkpoint", ""))).resolve() != expected_path:
                    raise A18ContractError(f"seed {seed} {variant} retained path drift")
                if sha256_file(expected_path) != retained.get("checkpoint_sha256"):
                    raise A18ContractError(f"seed {seed} {variant} retained hash drift")
                load_candidate_checkpoint(
                    expected_path,
                    expected_seed=seed,
                    expected_variant=variant,
                    spec=spec,
                    device=device,
                )
                completed_rows.append(retained)
                continue
            attempt = len(pair_attempts) + 1
            try:
                row = _train_one_candidate(
                    spec,
                    input_row,
                    output,
                    variant=variant,
                    device=device,
                )
                row["attempt"] = attempt
            except Exception as exc:
                rows.append(
                    {
                        "schema_version": A18_SCHEMA_VERSION,
                        "axis_id": A18_AXIS_ID,
                        "role": "ablation_readiness",
                        "claim_scope": "ablation_readiness_only",
                        "seed": seed,
                        "variant": variant,
                        "attempt": attempt,
                        "execution_status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                        "promotion": {"auto": False, "eligible": False},
                    }
                )
                _atomic_json(training_rows_path, _training_evidence_payload(spec, rows))
                raise
            rows.append(row)
            completed_rows.append(row)
            _atomic_json(training_rows_path, _training_evidence_payload(spec, rows))
    return completed_rows


def _train_one_candidate(
    spec: Mapping[str, Any],
    input_row: Mapping[str, Any],
    output: Path,
    *,
    variant: str,
    device: str,
) -> dict[str, Any]:
    compute = spec["compute_contract"]
    seed = int(input_row["seed"])
    # Reset all sampling/model RNGs before each arm so both receive the same
    # replay batches and identical initial auxiliary parameters.
    set_reproducible_seed(seed)
    model, bootstrap = load_bootstrap_model(
        input_row["bootstrap_checkpoint"],
        seed=seed,
        device=device,
    )
    if bootstrap["sha256"] != input_row["bootstrap_checkpoint_sha256"]:
        raise A18ContractError(f"seed {seed} bootstrap changed during training")
    backend = A18TrainingBackend(
        model,
        variant=variant,
        seed=seed,
        device=device,
        learning_rate=float(compute["learning_rate"]),
        weight_decay=float(compute["weight_decay"]),
        momentum=float(compute["momentum"]),
        grad_clip_norm=float(compute["grad_clip_norm"]),
        denoise_weight=float(compute["denoise_weight"]),
        corruption_sigma=float(compute["corruption_sigma"]),
    )
    replay = ReplayBuffer(1_000_000)
    loaded = replay.load(input_row["training_replay"])
    if loaded != input_row["replay_positions"]:
        raise A18ContractError(f"seed {seed} replay changed during training")
    schedule_size = int(compute["learner_updates"]) * int(compute["batch_size"])
    np.random.seed(seed)
    sampled_indices = np.random.randint(0, loaded, size=schedule_size)
    sample_schedule_hash = stable_hash(sampled_indices.tolist())
    # Rewind before handing control to ReplayBuffer.build_dataloader so this
    # recorded schedule is the schedule actually consumed by train_epoch.
    np.random.seed(seed)
    loss, policy_loss, value_loss, steps_done, stop_summary = train_epoch(
        backend.model,
        backend.optimizer,
        replay,
        {"batch": int(compute["batch_size"])},
        backend.device,
        int(compute["learner_updates"]),
        backend=backend,
    )
    if steps_done != int(compute["learner_updates"]):
        raise A18ContractError(
            f"seed {seed} {variant} executed {steps_done} learner updates, expected "
            f"{compute['learner_updates']}"
        )
    assert_direct_inference_deterministic(backend.model, device=device)
    metrics = {
        "loss": float(loss),
        "policy_loss": float(policy_loss),
        "value_loss": float(value_loss),
        "auxiliary_loss": backend.last_auxiliary_loss,
        "step_early_stopping": stop_summary,
        "sample_schedule_hash": sample_schedule_hash,
        "sample_schedule_count": schedule_size,
    }
    payload = _checkpoint_payload(
        backend,
        spec=spec,
        input_row=input_row,
        training_metrics=metrics,
    )
    checkpoint_path = candidate_checkpoint_path(output, seed, variant)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    write_deterministic_checkpoint(checkpoint_path, payload)
    return {
        "schema_version": A18_SCHEMA_VERSION,
        "axis_id": A18_AXIS_ID,
        "role": "ablation_readiness",
        "claim_scope": "ablation_readiness_only",
        "execution_status": "succeeded",
        "seed": seed,
        "variant": variant,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "parameter_count": payload["a18_metadata"]["parameter_count"],
        "parameter_breakdown": payload["a18_metadata"]["parameter_breakdown"],
        "direct_inference_flops": payload["a18_metadata"]["direct_inference_flops"],
        "training_forward_flops": payload["a18_metadata"]["training_forward_flops"],
        "promotion": {"auto": False, "eligible": False},
        **metrics,
    }


def load_candidate_checkpoint(
    path: str | os.PathLike[str],
    *,
    expected_seed: int,
    expected_variant: str,
    spec: Mapping[str, Any],
    device: str,
) -> tuple[A18MatchedEvaluator, dict[str, Any]]:
    """Load a real, completed A18 checkpoint and validate every binding."""

    target = Path(path)
    checkpoint_sha256 = sha256_file(target)
    raw = read_deterministic_checkpoint(target)
    if not isinstance(raw, Mapping):
        raise A18ContractError(f"candidate checkpoint is not wrapped: {target}")
    state_dict = raw.get("model_state_dict")
    metadata = raw.get("a18_metadata")
    cfg = raw.get("cfg")
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise A18ContractError(f"candidate checkpoint has no state_dict: {target}")
    if not isinstance(metadata, Mapping) or not isinstance(cfg, Mapping):
        raise A18ContractError(f"candidate checkpoint lacks A18 metadata: {target}")
    required_equal = {
        "schema_version": A18_SCHEMA_VERSION,
        "axis_id": A18_AXIS_ID,
        "checkpoint_role": CHECKPOINT_ROLE,
        "synthetic": False,
        "variant": expected_variant,
        "seed": int(expected_seed),
        "direct_inference_contract": DIRECT_INFERENCE_CONTRACT,
        "controller_contract_hash": stable_hash(spec["controller_contract"]),
        "compute_contract_hash": stable_hash(spec["compute_contract"]),
    }
    for key, expected in required_equal.items():
        if metadata.get(key) != expected:
            raise A18ContractError(
                f"candidate {target} metadata {key!r} mismatch: "
                f"{metadata.get(key)!r} != {expected!r}"
            )
    if metadata.get("controller_contract") != spec["controller_contract"]:
        raise A18ContractError(f"candidate {target} embeds controller contract drift")
    if metadata.get("compute_contract") != spec["compute_contract"]:
        raise A18ContractError(f"candidate {target} embeds compute contract drift")
    expected_updates = int(spec["compute_contract"]["learner_updates"])
    if (
        metadata.get("learner_updates_completed") != expected_updates
        or expected_updates <= 0
    ):
        raise A18ContractError(
            f"candidate {target} is not a completed trained checkpoint"
        )
    model = A18MatchedEvaluator(cfg)
    reason = validate_torch_state_dict(model, state_dict)
    if reason is not None:
        raise A18ContractError(f"candidate {target} state_dict mismatch: {reason}")
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    current_architecture = architecture_contract(model.cfg)
    if metadata.get("architecture_hash") != stable_hash(current_architecture):
        raise A18ContractError(f"candidate {target} architecture hash drift")
    current_params = parameter_breakdown(model)
    current_flops = estimate_direct_flops(model, device=device)
    current_training_flops = estimate_training_forward_flops(model, device=device)
    if metadata.get("parameter_count") != current_params["total_training"]:
        raise A18ContractError(f"candidate {target} parameter count drift")
    if metadata.get("parameter_breakdown") != current_params:
        raise A18ContractError(f"candidate {target} parameter breakdown drift")
    if metadata.get("direct_inference_flops") != current_flops:
        raise A18ContractError(f"candidate {target} FLOP count drift")
    if metadata.get("training_forward_flops") != current_training_flops:
        raise A18ContractError(f"candidate {target} training FLOP count drift")
    assert_direct_inference_deterministic(model, device=device)
    return model, {
        **dict(metadata),
        "path": str(target.resolve()),
        "sha256": checkpoint_sha256,
    }


def _evaluation_examples(
    replay_path: str,
    *,
    n_positions: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    replay = ReplayBuffer(1_000_000)
    count = replay.load(replay_path)
    if count < n_positions:
        raise A18ContractError(
            f"evaluation replay has {count} positions, expected at least {n_positions}"
        )
    rng = np.random.default_rng(int(seed))
    indices = np.sort(rng.choice(count, size=n_positions, replace=False))
    examples = replay.examples_at_indices(indices)
    states = np.asarray([item.state for item in examples], dtype=np.float32)
    policies = np.asarray([item.policy.dense() for item in examples], dtype=np.float32)
    values = np.asarray([item.value for item in examples], dtype=np.float32)
    return states, policies, values, stable_hash(indices.tolist())


def _diagnostic_metrics(
    model: A18MatchedEvaluator,
    states: np.ndarray,
    policies: np.ndarray,
    values: np.ndarray,
    *,
    device: str,
    batch_size: int,
) -> dict[str, float]:
    import torch

    eps = 1e-8
    policy_nll = 0.0
    value_sqerr = 0.0
    brier = 0.0
    top1_hits = 0
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(states), batch_size):
            end = min(len(states), start + batch_size)
            batch = torch.as_tensor(
                states[start:end], dtype=torch.float32, device=device
            )
            logits, predicted_values = model(batch)
            probs = torch.softmax(logits, dim=-1).cpu().numpy().astype(np.float64)
            predicted_values_np = predicted_values.cpu().numpy().astype(np.float64)
            target_policy = policies[start:end].astype(np.float64)
            policy_nll += float(-(target_policy * np.log(np.maximum(probs, eps))).sum())
            brier += float(np.square(probs - target_policy).sum())
            top1_hits += int(
                (np.argmax(probs, axis=1) == np.argmax(target_policy, axis=1)).sum()
            )
            value_sqerr += float(
                np.square(
                    predicted_values_np - values[start:end].astype(np.float64)
                ).sum()
            )
    n = float(len(states))
    return {
        "policy_target_nll": policy_nll / n,
        "policy_brier": brier / n,
        "policy_top1_accuracy": top1_hits / n,
        "value_mse": value_sqerr / n,
    }


def _latency_diagnostic(
    model: A18MatchedEvaluator,
    states: np.ndarray,
    *,
    device: str,
    batch_sizes: Sequence[int],
    warmups: int,
    repetitions: int,
) -> dict[str, Any]:
    import torch

    if warmups < 1 or repetitions < 1:
        raise A18ContractError("latency warmups and repetitions must be positive")
    result = {}
    model.eval()
    for batch_size in batch_sizes:
        if batch_size < 1:
            raise A18ContractError("latency batch sizes must be positive")
        repeats = int(math.ceil(batch_size / len(states)))
        batch_np = np.concatenate([states] * repeats, axis=0)[:batch_size]
        batch = torch.as_tensor(batch_np, dtype=torch.float32, device=device)
        with torch.inference_mode():
            for _ in range(warmups):
                model(batch)
            if str(device).startswith("cuda"):
                torch.cuda.synchronize()
            samples = []
            for _ in range(repetitions):
                start = time.perf_counter_ns()
                model(batch)
                if str(device).startswith("cuda"):
                    torch.cuda.synchronize()
                samples.append((time.perf_counter_ns() - start) / 1e6)
        result[str(batch_size)] = {
            "median_ms": float(np.median(samples)),
            "min_ms": float(np.min(samples)),
            "repetitions": repetitions,
        }
    return result


def _write_diagnostic_plot(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    seeds = sorted({int(row["seed"]) for row in rows})
    by_key = {(int(row["seed"]), str(row["variant"])): row for row in rows}
    metrics = (
        ("policy_target_nll", "Policy target NLL delta"),
        ("value_mse", "Value MSE delta"),
        ("latency_batch1_ms", "Batch-1 latency delta (ms)"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for axis, (metric, title) in zip(axes, metrics):
        deltas = [
            float(by_key[(seed, DIFFUSION_VARIANT)][metric])
            - float(by_key[(seed, BASELINE_VARIANT)][metric])
            for seed in seeds
        ]
        axis.axhline(0.0, color="#111827", linewidth=0.8)
        axis.bar([str(seed) for seed in seeds], deltas, color="#64748B")
        axis.set_title(title)
        axis.set_xlabel("Paired seed")
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle("A18 DIAGNOSTIC ONLY — descriptive paired deltas, no efficacy claim")
    fig.text(
        0.5,
        0.01,
        "Controller/updates/batches are fixed. Training-replay smoke metrics are not held-out evidence.",
        ha="center",
        fontsize=9,
        color="#991B1B",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _load_training_rows(
    output: Path,
    spec: Mapping[str, Any],
) -> tuple[Path, list[dict[str, Any]], list[dict[str, Any]]]:
    path = output / "training_rows.v1.json"
    if not path.is_file():
        raise A18ContractError(f"required training evidence is absent: {path}")
    rows = _load_existing_training_attempts(path, spec)
    expected = {(int(seed), variant) for seed in spec["seeds"] for variant in VARIANTS}
    successes = [row for row in rows if row.get("execution_status") == "succeeded"]
    observed = {(row.get("seed"), row.get("variant")) for row in successes}
    if observed != expected or len(successes) != len(expected):
        raise A18ContractError("training evidence paired-arm identity mismatch")
    for row in successes:
        checkpoint = Path(str(row.get("checkpoint", "")))
        if sha256_file(checkpoint) != row.get("checkpoint_sha256"):
            raise A18ContractError(f"training candidate hash drift: {checkpoint}")
    return path, successes, rows


def analyze_candidates(
    spec: Mapping[str, Any],
    output_dir: str | os.PathLike[str],
    *,
    device: str,
) -> dict[str, Any]:
    """Validate paired checkpoints and emit versioned diagnostic artifacts."""

    output = Path(output_dir)
    set_reproducible_seed(int(spec["seeds"][0]))
    inputs = inspect_inputs(spec)
    training_rows_path, training_rows, training_attempt_rows = _load_training_rows(
        output, spec
    )
    compute = spec["compute_contract"]
    rows: list[dict[str, Any]] = []
    pair_contracts = []
    for input_row in inputs:
        seed = int(input_row["seed"])
        evaluation_path = input_row["training_replay"]
        evaluation_kind = "training_replay_smoke_only"
        if spec["evidence_tier"] == "study_candidate":
            evaluation_path = str(input_row["evaluation_replay"])
            if input_row["state_group_contract"]["overlap_state_groups"] != 0:
                raise A18ContractError(f"seed {seed} evaluation state-group overlap")
            evaluation_kind = "verified_state_hash_disjoint_replay"
        states, policies, values, position_index_hash = _evaluation_examples(
            evaluation_path,
            n_positions=int(compute["evaluation_positions"]),
            seed=seed,
        )
        paired_meta = {}
        for variant in VARIANTS:
            checkpoint = candidate_checkpoint_path(output, seed, variant)
            model, metadata = load_candidate_checkpoint(
                checkpoint,
                expected_seed=seed,
                expected_variant=variant,
                spec=spec,
                device=device,
            )
            if (
                metadata["source_checkpoint_sha256"]
                != input_row["bootstrap_checkpoint_sha256"]
            ):
                raise A18ContractError(
                    f"seed {seed} {variant} bootstrap identity drift"
                )
            if (
                metadata["training_replay_sha256"]
                != input_row["training_replay_sha256"]
            ):
                raise A18ContractError(f"seed {seed} {variant} replay identity drift")
            metrics = _diagnostic_metrics(
                model,
                states,
                policies,
                values,
                device=device,
                batch_size=int(compute["batch_size"]),
            )
            latency = _latency_diagnostic(
                model,
                states,
                device=device,
                batch_sizes=[
                    int(value) for value in compute.get("latency_batch_sizes", [1])
                ],
                warmups=int(compute.get("latency_warmups", 1)),
                repetitions=int(compute.get("latency_repetitions", 3)),
            )
            row = {
                "seed": seed,
                "variant": variant,
                "checkpoint": metadata["path"],
                "checkpoint_sha256": metadata["sha256"],
                "parameter_count": metadata["parameter_count"],
                "parameter_breakdown": metadata["parameter_breakdown"],
                "direct_inference_flops": metadata["direct_inference_flops"],
                "training_forward_flops": metadata["training_forward_flops"],
                "evaluation_source": str(Path(evaluation_path).resolve()),
                "evaluation_source_sha256": sha256_file(evaluation_path),
                "evaluation_kind": evaluation_kind,
                "evaluation_positions": len(states),
                "position_index_hash": position_index_hash,
                **metrics,
                "latency": latency,
                "latency_batch1_ms": latency["1"]["median_ms"],
            }
            rows.append(row)
            paired_meta[variant] = metadata
        baseline = paired_meta[BASELINE_VARIANT]
        diffusion = paired_meta[DIFFUSION_VARIANT]
        for key in (
            "architecture_hash",
            "parameter_count",
            "direct_inference_flops",
            "training_forward_flops",
            "controller_contract_hash",
            "compute_contract_hash",
            "source_checkpoint_sha256",
            "training_replay_sha256",
            "sample_schedule_hash",
            "learner_updates_completed",
        ):
            if baseline[key] != diffusion[key]:
                raise A18ContractError(f"seed {seed} paired-arm mismatch in {key}")
        pair_contracts.append(
            {
                "seed": seed,
                "parameter_match": True,
                "direct_inference_flop_match": True,
                "training_forward_flop_match": True,
                "controller_match": True,
                "compute_match": True,
                "bootstrap_match": True,
                "replay_match": True,
            }
        )

    data_payload = {
        "schema_version": A18_SCHEMA_VERSION,
        "axis_id": A18_AXIS_ID,
        "role": "ablation_readiness",
        "evidence_status": "skeleton_only",
        "artifact_kind": "diagnostic_data",
        "scientific_status": "DIAGNOSTIC_ONLY",
        "claim_scope": "ablation_readiness_only",
        "evidence_tier": spec["evidence_tier"],
        "automatic_promotion": False,
        "rows": rows,
        "paired_contracts": pair_contracts,
        "prohibited_inferences": ["play_strength", "efficacy", "production_readiness"],
    }
    data_path = output / "data.v1.json"
    _atomic_json(data_path, data_payload)
    plot_path = output / DIAGNOSTIC_PLOT_NAME
    _write_diagnostic_plot(plot_path, rows)
    manifest_payload = {
        "schema_version": A18_SCHEMA_VERSION,
        "axis_id": A18_AXIS_ID,
        "role": "ablation_readiness",
        "evidence_status": "skeleton_only",
        "study_id": spec.get("study_id"),
        "artifact_kind": "ablation_manifest",
        "scientific_status": "DIAGNOSTIC_ONLY",
        "evidence_tier": spec["evidence_tier"],
        "claim_scope": "ablation_readiness_only",
        "controller_frozen": True,
        "source_hashes": _source_hashes(),
        "input_hashes": _input_hashes(spec, inputs),
        "seed_contract": {
            "mode": "paired_fixed",
            "seeds": list(spec["seeds"]),
            "variants": list(VARIANTS),
            "same_bootstrap_within_seed": True,
            "same_replay_batches_within_seed": True,
            "deterministic_algorithms_required": True,
        },
        "controller_contract": dict(spec["controller_contract"]),
        "controller_contract_hash": stable_hash(spec["controller_contract"]),
        "compute_contract": dict(spec["compute_contract"]),
        "compute_contract_hash": stable_hash(spec["compute_contract"]),
        "determinism_runtime_contract": determinism_runtime_contract(),
        "seeds": list(spec["seeds"]),
        "input_inventory": inputs,
        "evaluation_split_contract": (
            {
                "estimand": "training_replay_smoke_only",
                "exact_state_disjoint_verified": False,
                "game_or_trajectory_group_disjoint_verified": False,
            }
            if spec["evidence_tier"] == "smoke_readiness"
            else {
                "estimand": "exact_serialized_state_disjoint_only",
                "exact_state_disjoint_verified": True,
                "game_or_trajectory_group_disjoint_verified": False,
            }
        ),
        "variant_contract": {
            BASELINE_VARIANT: {
                "auxiliary_branch_executed": True,
                "denoise_loss_weight": 0.0,
            },
            DIFFUSION_VARIANT: {
                "auxiliary_branch_executed": True,
                "denoise_loss_weight": float(compute["denoise_weight"]),
            },
            "direct_inference": DIRECT_INFERENCE_CONTRACT,
        },
        "deployment_contract": {
            "candidate_format": "quartz_a18_deterministic_zip_v1",
            "candidate_extension": ".a18ckpt",
            "torch_load_compatible": False,
            "production_compatible": False,
            "full_study_blocker": (
                "implement and validate a production direct-path export/loader before "
                "deployment or live-search evidence"
            ),
        },
        "artifacts": {
            "data": {"path": data_path.name, "sha256": sha256_file(data_path)},
            "plot": {"path": plot_path.name, "sha256": sha256_file(plot_path)},
            "training_rows": {
                "path": training_rows_path.name,
                "sha256": sha256_file(training_rows_path),
                "attempt_count": len(training_attempt_rows),
                "failed_attempt_count": sum(
                    row.get("execution_status") == "failed"
                    for row in training_attempt_rows
                ),
            },
            "candidate_checkpoints": [
                {
                    "seed": row["seed"],
                    "variant": row["variant"],
                    "path": row["checkpoint"],
                    "sha256": row["checkpoint_sha256"],
                }
                for row in training_rows
            ],
        },
        "automatic_promotion": False,
        "promotion": {"auto": False, "eligible": False},
        "prohibited_inferences": ["play_strength", "efficacy", "production_readiness"],
    }
    manifest_path = output / "manifest.v1.json"
    _atomic_json(manifest_path, manifest_payload)
    # Idea-lab integration artifacts intentionally mirror the versioned
    # scientific payload instead of inventing a second interpretation.
    run_manifest = {
        **manifest_payload,
        "artifact_kind": "run_manifest",
        "versioned_manifest": manifest_path.name,
        "versioned_manifest_sha256": sha256_file(manifest_path),
    }
    _atomic_json(output / "run_manifest.json", run_manifest)
    bound_rows = [
        {
            "schema_version": A18_SCHEMA_VERSION,
            "axis_id": A18_AXIS_ID,
            "role": "ablation_readiness",
            "evidence_status": "skeleton_only",
            "claim_scope": "ablation_readiness_only",
            "promotion": {"auto": False, "eligible": False},
            **row,
        }
        for row in rows
    ]
    _atomic_jsonl(output / "rows.jsonl", bound_rows)
    shutil.copyfile(plot_path, output / "diagnostic.png")
    summary = {
        "schema_version": A18_SCHEMA_VERSION,
        "axis_id": A18_AXIS_ID,
        "role": "ablation_readiness",
        "execution_status": "completed_no_promotion",
        "evidence_status": "skeleton_only",
        "claim_scope": "ablation_readiness_only",
        "scientific_status": "DIAGNOSTIC_ONLY",
        "evidence_tier": spec["evidence_tier"],
        "paired_seed_count": len(spec["seeds"]),
        "training_attempt_count": len(training_attempt_rows),
        "failed_training_attempt_count": sum(
            row.get("execution_status") == "failed" for row in training_attempt_rows
        ),
        "evaluation_estimand": manifest_payload["evaluation_split_contract"][
            "estimand"
        ],
        "parameter_match": all(row["parameter_match"] for row in pair_contracts),
        "direct_inference_flop_match": all(
            row["direct_inference_flop_match"] for row in pair_contracts
        ),
        "training_forward_flop_match": True,
        "production_compatible_export": False,
        "full_study_blockers": [
            "production direct-path export/loader is not implemented or validated",
            "game/trajectory-group-disjoint held-out replay is not verified",
        ],
        "controller_match": all(row["controller_match"] for row in pair_contracts),
        "compute_match": all(row["compute_match"] for row in pair_contracts),
        "promotion": {"auto": False, "eligible": False},
        "prohibited_inferences": ["play_strength", "efficacy", "production_readiness"],
        "artifacts": {
            "run_manifest": "run_manifest.json",
            "rows": "rows.jsonl",
            "summary": "summary.json",
            "diagnostic_plot": "diagnostic.png",
            "versioned_manifest": manifest_path.name,
            "versioned_data": data_path.name,
            "versioned_diagnostic_plot": plot_path.name,
            "training_rows": training_rows_path.name,
        },
    }
    _atomic_json(output / "summary.json", summary)
    return manifest_payload


def run_smoke_or_study(
    spec: Mapping[str, Any],
    output_dir: str | os.PathLike[str],
    *,
    device: str,
) -> dict[str, Any]:
    train_candidates(spec, output_dir, device=device)
    return analyze_candidates(spec, output_dir, device=device)


__all__ = [
    "A18ContractError",
    "A18MatchedEvaluator",
    "A18TrainingBackend",
    "BASELINE_VARIANT",
    "DIAGNOSTIC_PLOT_NAME",
    "DIFFUSION_VARIANT",
    "analyze_candidates",
    "architecture_contract",
    "assert_direct_inference_deterministic",
    "candidate_checkpoint_path",
    "estimate_direct_flops",
    "estimate_training_forward_flops",
    "inspect_inputs",
    "load_bootstrap_model",
    "load_candidate_checkpoint",
    "load_spec",
    "parameter_count",
    "parameter_breakdown",
    "replay_state_hash_contract",
    "run_smoke_or_study",
    "sha256_file",
    "stable_hash",
    "train_candidates",
]
