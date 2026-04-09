#!/usr/bin/env python3
"""Unified training backends for QUARTZ."""

from __future__ import annotations

import copy
import os
import time

import numpy as np


def load_torch_state_dict(path, torch_module, map_location="cpu"):
    """Load a trusted local checkpoint state dict with weights_only fallback.

    Newer PyTorch defaults can reject some older QUARTZ checkpoints when
    `weights_only=True`. For project-local checkpoints we first try the safer
    path, then fall back to full unpickling only for the known compatibility
    failure mode.
    """
    try:
        return torch_module.load(path, map_location=map_location, weights_only=True)
    except Exception as exc:
        msg = str(exc)
        compat_error = (
            "Weights only load failed" in msg
            or "WeightsUnpickler" in msg
            or "Unsupported operand" in msg
        )
        if not compat_error:
            raise
        return torch_module.load(path, map_location=map_location, weights_only=False)


def validate_torch_state_dict(model, state_dict):
    """Return None when a state dict fits the model, else a short reason."""
    if not isinstance(state_dict, dict):
        return f"checkpoint payload is {type(state_dict).__name__}, expected state_dict"
    model_state = model.state_dict()
    missing = [key for key in model_state.keys() if key not in state_dict]
    unexpected = [key for key in state_dict.keys() if key not in model_state]
    mismatched = []
    for key, tensor in state_dict.items():
        if key not in model_state:
            continue
        if tuple(getattr(tensor, "shape", ())) != tuple(model_state[key].shape):
            mismatched.append(
                f"{key}: ckpt={tuple(getattr(tensor, 'shape', ()))} model={tuple(model_state[key].shape)}"
            )
    if missing or unexpected or mismatched:
        parts = []
        if missing:
            parts.append(f"missing={len(missing)}")
        if unexpected:
            parts.append(f"unexpected={len(unexpected)}")
        if mismatched:
            parts.append(f"mismatched={len(mismatched)}")
        return ", ".join(parts)
    return None


def load_torch_state_dict_checked(model, path, torch_module, map_location="cpu"):
    """Load a state dict only when it matches the target model exactly."""
    state_dict = load_torch_state_dict(path, torch_module, map_location=map_location)
    reason = validate_torch_state_dict(model, state_dict)
    if reason is not None:
        raise RuntimeError(f"incompatible checkpoint: {reason}")
    model.load_state_dict(state_dict)
    return state_dict


def detect_backends(preference="auto"):
    out = {
        "jax": False,
        "jax_gpu": False,
        "jax_checked": False,
        "torch": False,
        "torch_gpu": False,
    }
    preference = str(preference or "auto").lower()

    try:
        import torch

        out["torch"] = True
        out["torch_gpu"] = bool(torch.cuda.is_available())
    except Exception:
        pass

    should_probe_jax = preference == "jax" or not out["torch"]
    if should_probe_jax:
        out["jax_checked"] = True
        try:
            import jax

            out["jax"] = True
            try:
                out["jax_gpu"] = any(d.platform in ("gpu", "rocm", "cuda") for d in jax.devices())
            except Exception:
                out["jax_gpu"] = False
        except Exception:
            pass

    return out


def select_backend(detection, preference="auto"):
    preference = str(preference or "auto").lower()
    if preference == "jax":
        if detection["jax"]:
            return "jax"
        raise RuntimeError("JAX requested but not available.")
    if preference == "torch":
        if detection["torch"]:
            return "torch"
        raise RuntimeError("PyTorch requested but not available.")

    if detection["torch_gpu"]:
        return "torch"
    if detection["torch"]:
        return "torch"
    if detection["jax_gpu"]:
        return "jax"
    if detection["jax"]:
        return "jax"
    raise RuntimeError("No ML backend available. Install PyTorch or JAX.")


class JAXBackend:
    """JIT-compiled training and inference via JAX/Flax."""

    def __init__(self, cfg):
        import jax
        import jax.numpy as jnp
        import optax

        self.jax = jax
        self.jnp = jnp
        self.optax = optax
        self.cfg = cfg
        self.name = "jax"

        from quartz.jax_models import AlphaZeroJAX

        bs = cfg["board"]
        self.model = AlphaZeroJAX(
            board_size=bs,
            in_ch=cfg["ch"],
            n_actions=cfg["actions"],
            n_filters=cfg["filters"],
            n_blocks=cfg["blocks"],
            value_hidden=cfg["vh"],
            se_blocks=min(2, cfg["blocks"]),
        )

        rng = jax.random.PRNGKey(42)
        dummy = jnp.ones((1, cfg["ch"], bs, bs), dtype=jnp.float32)
        variables = self.model.init(rng, dummy, train=False)
        self.params = variables["params"]
        self.batch_stats = variables.get("batch_stats", {})

        self.tx = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.sgd(learning_rate=0.02, momentum=0.9),
            optax.add_decayed_weights(1e-4),
        )
        self.opt_state = self.tx.init(self.params)
        self._jit_train_step = jax.jit(self._train_step_impl)
        self._jit_predict = jax.jit(self._predict_impl)

        n_params = sum(p.size for p in jax.tree.leaves(self.params))
        print(f"  JAX backend: {n_params:,} params on {jax.devices()}")

    def _train_step_impl(self, params, batch_stats, opt_state, states, policies, values):
        def loss_fn(p):
            variables = {"params": p, "batch_stats": batch_stats}
            (logits, vals), updates = self.model.apply(
                variables, states, train=True, mutable=["batch_stats"]
            )
            log_probs = logits - self.jax.scipy.special.logsumexp(
                logits, axis=-1, keepdims=True
            )
            p_loss = -(policies * log_probs).sum(axis=-1).mean()
            v_loss = ((vals - values) ** 2).mean()
            return p_loss + v_loss, (p_loss, v_loss, updates.get("batch_stats", batch_stats))

        (loss, (p_loss, v_loss, new_bs)), grads = self.jax.value_and_grad(
            loss_fn, has_aux=True
        )(params)
        updates, new_opt = self.tx.update(grads, opt_state, params)
        new_params = self.optax.apply_updates(params, updates)
        return new_params, new_bs, new_opt, loss, p_loss, v_loss

    def _predict_impl(self, params, batch_stats, x):
        variables = {"params": params, "batch_stats": batch_stats}
        logits, values = self.model.apply(variables, x, train=False)
        probs = self.jax.nn.softmax(logits, axis=-1)
        return probs, values

    def train_step(self, states_np, policies_np, values_np):
        states = self.jnp.array(states_np)
        policies = self.jnp.array(policies_np)
        values = self.jnp.array(values_np)
        self.params, self.batch_stats, self.opt_state, loss, p_loss, v_loss = self._jit_train_step(
            self.params, self.batch_stats, self.opt_state, states, policies, values
        )
        return float(loss), float(p_loss), float(v_loss)

    def eval(self):
        return self

    def predict(self, state_np):
        x = self.jnp.array(state_np)
        if x.ndim == 3:
            x = x[None]
        probs, vals = self._jit_predict(self.params, self.batch_stats, x)
        return np.array(probs), np.array(vals)

    def set_lr(self, lr):
        self.tx = self.optax.chain(
            self.optax.clip_by_global_norm(1.0),
            self.optax.sgd(learning_rate=lr, momentum=0.9),
            self.optax.add_decayed_weights(1e-4),
        )
        self.opt_state = self.tx.init(self.params)

    def save(self, path):
        import pickle

        with open(path, "wb") as f:
            pickle.dump({"params": self.params, "batch_stats": self.batch_stats}, f)

    def load(self, path):
        import pickle

        if os.path.exists(path):
            with open(path, "rb") as f:
                data = pickle.load(f)
            self.params = data["params"]
            self.batch_stats = data.get("batch_stats", {})
            return True
        return False

    def get_torch_model(self):
        return None

    def create_actor(self):
        return JAXActor(self)

    def load_actor(self, path):
        import pickle

        with open(path, "rb") as f:
            data = pickle.load(f)
        return JAXActor(
            self,
            params=data["params"],
            batch_stats=data.get("batch_stats", {}),
        )


class JAXActor:
    """Immutable inference snapshot for self-play/eval."""

    def __init__(self, backend, params=None, batch_stats=None):
        self.name = "jax_actor"
        self._backend = backend
        self._params = backend.params if params is None else params
        self._batch_stats = backend.batch_stats if batch_stats is None else batch_stats

    def eval(self):
        return self

    def predict(self, state_np):
        x = self._backend.jnp.array(state_np)
        if x.ndim == 3:
            x = x[None]
        probs, vals = self._backend._jit_predict(self._params, self._batch_stats, x)
        return np.array(probs), np.array(vals)


class PyTorchBackend:
    """Standard PyTorch training and inference."""

    def __init__(self, cfg, device="auto"):
        import torch
        import torch.nn.functional as F

        self.torch = torch
        self.F = F
        self.cfg = cfg
        self.name = "torch"

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        from quartz.alphazero_train import AlphaZeroNet

        self.model = AlphaZeroNet(cfg).to(self.device)
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=0.02, momentum=0.9, weight_decay=1e-4
        )

        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"  PyTorch backend: {n_params:,} params on {self.device}")

    def train_step(self, states_np, policies_np, values_np):
        torch = self.torch
        self.model.train()
        states = torch.tensor(states_np, dtype=torch.float32).to(self.device)
        policies = torch.tensor(policies_np, dtype=torch.float32).to(self.device)
        values = torch.tensor(values_np, dtype=torch.float32).to(self.device)

        logits, pred_v = self.model(states)
        log_probs = self.F.log_softmax(logits, dim=-1)
        p_loss = -(policies * log_probs).sum(dim=-1).mean()
        v_loss = self.F.mse_loss(pred_v, values)
        loss = p_loss + v_loss

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        return loss.item(), p_loss.item(), v_loss.item()

    def eval(self):
        self.model.eval()
        return self

    def predict(self, state_np):
        torch = self.torch
        self.model.eval()
        with torch.inference_mode():
            x = torch.tensor(state_np, dtype=torch.float32).to(self.device)
            if x.dim() == 3:
                x = x.unsqueeze(0)
            logits, vals = self.model(x)
            probs = self.F.softmax(logits, dim=-1).cpu().numpy()
            return probs, vals.cpu().numpy()

    def set_lr(self, lr):
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    def save(self, path):
        self.torch.save(self.model.state_dict(), path)

    def load(self, path):
        if os.path.exists(path):
            load_torch_state_dict_checked(
                self.model, path, self.torch, map_location=self.device
            )
            return True
        return False

    def get_torch_model(self):
        return self.model

    def create_actor(self):
        actor = copy.deepcopy(self.model)
        actor.eval()
        return actor


def create_backend(cfg, device="auto", preference="auto"):
    """Create the best available backend."""
    det = detect_backends(preference=preference)
    jax_status = "skipped" if not det.get("jax_checked", False) else ("✅" if det["jax"] else "❌")
    print(
        f"  Backend detection: JAX={jax_status}"
        f"{'(GPU)' if det['jax_gpu'] else ''} "
        f"PyTorch={'✅' if det['torch'] else '❌'}"
        f"{'(GPU)' if det['torch_gpu'] else ''}"
    )

    if device in ("jax",):
        preference = "jax"
    elif device in ("torch", "cuda"):
        preference = "torch"

    backend_name = select_backend(det, preference)

    if backend_name == "jax":
        try:
            be = JAXBackend(cfg)
            bs = cfg["board"]
            dummy = np.zeros((2, cfg["ch"], bs, bs), dtype=np.float32)
            dummy_pol = np.zeros((2, cfg["actions"]), dtype=np.float32)
            dummy_val = np.zeros(2, dtype=np.float32)
            t0 = time.time()
            be.train_step(dummy, dummy_pol, dummy_val)
            be.predict(dummy)
            jit_time = time.time() - t0
            print(f"  JAX JIT warmup: {jit_time:.1f}s (subsequent calls ~10-100× faster)")
            return be
        except Exception as e:
            print(f"  JAX init failed ({e}), falling back to PyTorch")

    return PyTorchBackend(cfg, device=device)
