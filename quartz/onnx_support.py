"""
QUARTZ ONNX Support — Export & Inference
=========================================

Export trained AlphaZeroNet to ONNX format for deployment without PyTorch/JAX.
Run inference via onnxruntime (CPU, CUDA, ROCm, TensorRT, DirectML).

Usage:
  # Export
  from quartz.onnx_support import export_onnx
  export_onnx(model, cfg, "model.onnx")

  # Inference
  from quartz.onnx_support import OnnxPredictor
  pred = OnnxPredictor("model.onnx")
  policy, value = pred.predict(board_tensor)

  # As MCTS evaluator
  mcts = TreeMCTS(cfg, model=None, device='cpu')
  mcts_with_onnx = TreeMCTS(cfg, model=OnnxModelWrapper("model.onnx", cfg), device='cpu')
"""

import os
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def export_onnx(model, cfg, output_path, opset_version=17, verbose=False):
    """Export PyTorch AlphaZeroNet to ONNX format.

    Args:
        model: AlphaZeroNet instance (PyTorch)
        cfg: game config dict (needs 'ch', 'board', 'actions')
        output_path: path to save .onnx file
        opset_version: ONNX opset (default 17)
        verbose: print export details

    Returns:
        Path to exported .onnx file
    """
    try:
        import torch
    except ImportError:
        raise ImportError("PyTorch required for ONNX export: pip install torch")

    model.eval()
    ch = cfg['ch']
    bs = cfg['board']
    dummy_input = torch.randn(1, ch, bs, bs)

    output_path = str(output_path)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        opset_version=opset_version,
        input_names=["board"],
        output_names=["policy_logits", "value"],
        dynamic_axes={
            "board": {0: "batch"},
            "policy_logits": {0: "batch"},
            "value": {0: "batch"},
        },
    )

    file_size = os.path.getsize(output_path)
    if verbose:
        print(f"Exported: {output_path} ({file_size / 1024:.1f} KB)")
        print(f"  Input:  board [{ch}, {bs}, {bs}]")
        print(f"  Output: policy_logits [{cfg['actions']}], value [1]")

    # Verify with onnx checker if available
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        if verbose:
            print("  ✅ ONNX validation passed")
    except ImportError:
        if verbose:
            print("  (onnx package not installed — skipping validation)")
    except Exception as e:
        logger.warning(f"ONNX validation warning: {e}")

    return output_path


def _select_providers():
    """Auto-detect best ONNX Runtime execution provider.

    Priority: TensorRT > CUDA > ROCm > DirectML > CPU
    """
    try:
        import onnxruntime as ort
    except ImportError:
        return None, []

    available = ort.get_available_providers()
    preference = [
        "TensorrtExecutionProvider",
        "CUDAExecutionProvider",
        "ROCMExecutionProvider",       # AMD ROCm
        "MIGraphXExecutionProvider",   # AMD MIGraphX
        "DmlExecutionProvider",        # Windows DirectML
        "CPUExecutionProvider",
    ]

    selected = []
    for p in preference:
        if p in available:
            selected.append(p)
            break
    if not selected:
        selected = ["CPUExecutionProvider"]

    return ort, selected


class OnnxPredictor:
    """ONNX Runtime inference session with auto-detected GPU provider.

    Supports: CUDA, ROCm, TensorRT, DirectML, CPU.
    Thread-safe for concurrent inference.
    """

    def __init__(self, model_path, provider=None):
        """
        Args:
            model_path: path to .onnx file
            provider: override provider (e.g. "CUDAExecutionProvider")
                      None = auto-detect best available
        """
        ort, auto_providers = _select_providers()
        if ort is None:
            raise ImportError("onnxruntime required: pip install onnxruntime-gpu")

        providers = [provider] if provider else auto_providers

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_opts.intra_op_num_threads = os.cpu_count() or 4

        self.session = ort.InferenceSession(
            str(model_path), sess_options=sess_opts, providers=providers
        )
        self.input_name = self.session.get_inputs()[0].name
        self.provider = self.session.get_providers()[0]
        self._path = str(model_path)

        logger.info(f"ONNX session: {self.provider} ({model_path})")

    def predict(self, board_tensor):
        """Run inference on a single board tensor.

        Args:
            board_tensor: numpy array (C, H, W) or (B, C, H, W)

        Returns:
            (policy_probs, value) — numpy arrays
        """
        if board_tensor.ndim == 3:
            board_tensor = board_tensor[np.newaxis]

        board_tensor = board_tensor.astype(np.float32)
        outputs = self.session.run(None, {self.input_name: board_tensor})
        logits = outputs[0]   # (B, n_actions)
        values = outputs[1]   # (B, 1) or (B,)

        # Softmax
        logits_max = logits - logits.max(axis=-1, keepdims=True)
        exp_logits = np.exp(logits_max)
        policy = exp_logits / exp_logits.sum(axis=-1, keepdims=True)

        value = values.flatten()
        return policy.squeeze(0), float(value[0])

    def predict_batch(self, boards):
        """Batch inference. boards: (B, C, H, W)."""
        boards = boards.astype(np.float32)
        outputs = self.session.run(None, {self.input_name: boards})
        logits = outputs[0]
        values = outputs[1].flatten()

        logits_max = logits - logits.max(axis=-1, keepdims=True)
        exp_logits = np.exp(logits_max)
        policies = exp_logits / exp_logits.sum(axis=-1, keepdims=True)

        return policies, values

    def __repr__(self):
        return f"OnnxPredictor({self._path}, provider={self.provider})"


class OnnxModelWrapper:
    """Wraps OnnxPredictor to match the interface expected by TreeMCTS.

    TreeMCTS calls model(x) → (logits, value). This wrapper provides that
    interface using ONNX Runtime, enabling TreeMCTS to use ONNX models
    without PyTorch.

    Usage:
        model = OnnxModelWrapper("model.onnx", cfg)
        mcts = TreeMCTS(cfg, model=model, device='cpu')
    """

    def __init__(self, model_path, cfg, provider=None):
        self.predictor = OnnxPredictor(model_path, provider)
        self.cfg = cfg
        self._eval_mode = True

    def __call__(self, x):
        """Mimic PyTorch model(x) → (logits_tensor, value_tensor).

        Returns numpy arrays shaped like PyTorch tensors.
        """
        if hasattr(x, 'numpy'):
            x = x.detach().cpu().numpy()
        elif not isinstance(x, np.ndarray):
            x = np.array(x, dtype=np.float32)

        if x.ndim == 3:
            x = x[np.newaxis]

        outputs = self.predictor.session.run(
            None, {self.predictor.input_name: x.astype(np.float32)}
        )
        logits = outputs[0]
        values = outputs[1]

        # Return as numpy-backed fake tensors (TreeMCTS uses .cpu().numpy())
        return _FakeTensor(logits), _FakeTensor(values)

    def eval(self):
        return self

    def parameters(self):
        return iter([])

    def to(self, device):
        return self


class _FakeTensor:
    """Minimal tensor-like wrapper for numpy arrays.

    Supports the .cpu().numpy(), .squeeze(), .item() chain
    that TreeMCTS._evaluate_leaf uses.
    """

    def __init__(self, data):
        self._data = np.array(data, dtype=np.float32) if not isinstance(data, np.ndarray) else data

    def cpu(self):
        return self

    def detach(self):
        return self

    def numpy(self):
        return self._data

    def squeeze(self, dim=0):
        return _FakeTensor(np.squeeze(self._data, axis=dim))

    def item(self):
        return float(self._data.flat[0])

    def size(self, dim=None):
        if dim is not None:
            return self._data.shape[dim]
        return self._data.shape

    def __len__(self):
        return len(self._data)


# ════════════════════════════════════════════
# § Self-tests
# ════════════════════════════════════════════

def _run_tests():
    print("Testing ONNX support...")

    # Test FakeTensor
    ft = _FakeTensor(np.array([[1.0, 2.0, 3.0]]))
    assert ft.cpu().numpy().shape == (1, 3)
    assert ft.squeeze(0).item() == 1.0  # first element
    print("  [PASS] FakeTensor")

    # Test provider detection
    ort, providers = _select_providers()
    if ort is not None:
        print(f"  [PASS] onnxruntime detected: {providers}")
    else:
        print("  [SKIP] onnxruntime not installed")

    # Test export (requires torch)
    try:
        import torch
        import torch.nn as nn

        class TinyNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 8, 3, padding=1)
                self.p = nn.Linear(8 * 7 * 7, 49)
                self.v = nn.Linear(8 * 7 * 7, 1)

            def forward(self, x):
                h = self.conv(x).relu()
                h = h.reshape(h.size(0), -1)
                return self.p(h), self.v(h).tanh()

        model = TinyNet()
        cfg = {"ch": 3, "board": 7, "actions": 49}
        path = export_onnx(model, cfg, "/tmp/test_quartz.onnx", verbose=True)
        assert os.path.exists(path)
        print("  [PASS] ONNX export")

        # Test inference if onnxruntime available
        if ort is not None:
            pred = OnnxPredictor(path)
            board = np.random.randn(3, 7, 7).astype(np.float32)
            policy, value = pred.predict(board)
            assert policy.shape == (49,), f"Wrong policy shape: {policy.shape}"
            assert abs(policy.sum() - 1.0) < 0.01, f"Policy not normalized: {policy.sum()}"
            assert -1.0 <= value <= 1.0, f"Value out of range: {value}"
            print(f"  [PASS] ONNX inference (policy sum={policy.sum():.4f}, value={value:.4f})")

            # Test OnnxModelWrapper
            wrapper = OnnxModelWrapper(path, cfg)
            x = np.random.randn(1, 3, 7, 7).astype(np.float32)
            logits, val = wrapper(x)
            assert logits.numpy().shape == (1, 49)
            print("  [PASS] OnnxModelWrapper")

        os.unlink(path)

    except ImportError:
        print("  [SKIP] torch not installed — export test skipped")

    print("\n[ALL PASS] ONNX support tests.")


if __name__ == "__main__":
    _run_tests()
