"""
QUARTZ — Calibrated Metacontroller for Neural MCTS
===================================================

A research-grade AlphaZero training framework with:
- QUARTZ adaptive search controller (6 modes)
- Game-agnostic board encoders (Gomoku, Go, Chess)
- Glicko-2 evaluation system with PromotionGate
- ONNX export/inference for deployment
- Auto GPU detection (CUDA, ROCm, Metal)
- 3-tier self-play architecture (Rust MCTS / Rust+NN / Python TreeMCTS)
"""

__version__ = "0.10.0"

from .encoders import (
    GameEncoder,
    GomokuEncoder,
    GoEncoder,
    ChessEncoder,
    get_encoder,
    ENCODERS,
)

from .onnx_support import (
    export_onnx,
    OnnxPredictor,
    OnnxModelWrapper,
)

from .gpu_detect import (
    detect_gpu,
    recommend_install,
    GpuInfo,
)
