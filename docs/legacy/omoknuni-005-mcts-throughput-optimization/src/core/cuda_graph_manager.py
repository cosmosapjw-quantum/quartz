"""
CUDA Graph Manager for Neural Network Inference
================================================

Implements pre-captured CUDA graphs for eliminating kernel launch overhead.
Critical optimization for small kernels (15×15 Gomoku) that are launch-bound.

Expected improvement: 2-3× throughput increase
Reference: comments.md Section 3, Issue #3D

Architecture:
- Pre-capture graphs for common batch sizes: {8, 16, 32, 64, 128, 256}
- Static input/output buffers to avoid dynamic allocation
- Automatic fallback for non-standard batch sizes
- Thread-safe graph replay

Performance Impact:
- Without graphs: ~500μs Python/kernel launch overhead per batch
- With graphs:    ~5μs graph replay overhead
- Speedup:        ~100× reduction in launch overhead
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, List, Optional
import logging
from threading import Lock


class CUDAGraphManager:
    """Manages pre-captured CUDA graphs for efficient inference.

    Pre-captures the entire forward pass for fixed batch sizes, eliminating
    Python and CUDA kernel launch overhead. Critical for small kernels.

    Args:
        model: PyTorch model to capture
        input_shape: Input tensor shape (C, H, W) without batch dimension
        batch_sizes: List of batch sizes to pre-capture (default: common sizes)
        device: CUDA device (default: cuda)
        dtype: Input data type (default: float16 for faster transfers)
        use_amp: Whether to use automatic mixed precision (default: True)
        warmup_iterations: Number of warmup runs before capture (default: 3)

    Example:
        >>> model = create_resnet_eca_model('gomoku', size='128x12').cuda()
        >>> graph_mgr = CUDAGraphManager(model, input_shape=(36, 15, 15))
        >>> graph_mgr.warmup_and_capture()
        >>> policy, value = graph_mgr.infer(batch_tensor)  # Fast graph replay
    """

    def __init__(
        self,
        model: nn.Module,
        input_shape: Tuple[int, int, int],
        batch_sizes: Optional[List[int]] = None,
        device: str = 'cuda',
        dtype: torch.dtype = torch.float16,
        use_amp: bool = True,
        warmup_iterations: int = 3
    ):
        self.model = model.to(device).eval()
        self.input_shape = input_shape  # (C, H, W)
        self.batch_sizes = batch_sizes or [8, 16, 32, 64, 128, 256]
        self.device = torch.device(device)
        self.dtype = dtype
        self.use_amp = use_amp
        self.warmup_iterations = warmup_iterations

        # Storage for captured graphs
        self.graphs: Dict[int, torch.cuda.CUDAGraph] = {}
        self.static_inputs: Dict[int, torch.Tensor] = {}
        self.static_policy_outputs: Dict[int, torch.Tensor] = {}
        self.static_value_outputs: Dict[int, torch.Tensor] = {}

        # Thread safety
        self.lock = Lock()
        self.is_captured = False

        # Metrics
        self.graph_hits = 0
        self.graph_misses = 0
        self.fallback_count = 0

        self.logger = logging.getLogger(__name__)

    def warmup_and_capture(self) -> None:
        """Warmup model and capture CUDA graphs for all batch sizes.

        This should be called once after model initialization, before inference.
        """
        if self.device.type != 'cuda':
            self.logger.warning("CUDA graphs only supported on CUDA devices. Skipping capture.")
            return

        self.logger.info(f"Warming up and capturing CUDA graphs for batch sizes: {self.batch_sizes}")

        for batch_size in self.batch_sizes:
            self._capture_batch_size(batch_size)

        self.is_captured = True
        self.logger.info(
            f"✅ CUDA graph capture complete for {len(self.graphs)} batch sizes. "
            f"Expected 2-3× inference speedup."
        )

    def _capture_batch_size(self, batch_size: int) -> None:
        """Capture CUDA graph for a specific batch size.

        Args:
            batch_size: Batch size to capture
        """
        C, H, W = self.input_shape

        # Allocate static input buffer
        static_input = torch.zeros(
            (batch_size, C, H, W),
            dtype=self.dtype,
            device=self.device
        )
        self.static_inputs[batch_size] = static_input

        # Warmup: Run forward pass multiple times to initialize cudnn, etc.
        self.logger.info(f"  Warming up batch size {batch_size}...")
        for _ in range(self.warmup_iterations):
            with torch.no_grad():
                if self.use_amp:
                    with torch.amp.autocast('cuda'):
                        policy, value = self.model(static_input)
                else:
                    policy, value = self.model(static_input)

        # Synchronize before capture
        torch.cuda.synchronize()

        # Capture the graph
        self.logger.info(f"  Capturing graph for batch size {batch_size}...")
        graph = torch.cuda.CUDAGraph()

        with torch.cuda.graph(graph):
            with torch.no_grad():
                if self.use_amp:
                    with torch.amp.autocast('cuda'):
                        policy_out, value_out = self.model(static_input)
                else:
                    policy_out, value_out = self.model(static_input)

        # Store captured graph and output references
        self.graphs[batch_size] = graph
        self.static_policy_outputs[batch_size] = policy_out
        self.static_value_outputs[batch_size] = value_out

        self.logger.info(f"  ✅ Captured graph for batch size {batch_size}")

    def infer(
        self,
        batch_tensor: torch.Tensor,
        return_logits: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run inference using pre-captured CUDA graph if available.

        Args:
            batch_tensor: Input tensor (B, C, H, W)
            return_logits: If True, return raw logits; if False, apply softmax

        Returns:
            tuple: (policy, value)
                policy: Action probabilities or logits (B, num_actions)
                value: Position evaluation (B, 1)

        Note:
            If batch size doesn't have a pre-captured graph, falls back to
            regular inference (slower but still functional).
        """
        batch_size = batch_tensor.size(0)

        # Check if we have a pre-captured graph for this batch size
        if batch_size in self.graphs:
            return self._infer_with_graph(batch_tensor, batch_size, return_logits)
        else:
            return self._infer_fallback(batch_tensor, return_logits)

    def _infer_with_graph(
        self,
        batch_tensor: torch.Tensor,
        batch_size: int,
        return_logits: bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run inference using pre-captured graph (fast path).

        Args:
            batch_tensor: Input tensor (B, C, H, W)
            batch_size: Batch size (must be in self.graphs)
            return_logits: Whether to return raw logits or probabilities

        Returns:
            tuple: (policy, value)
        """
        with self.lock:
            # Copy input data to static buffer
            self.static_inputs[batch_size].copy_(batch_tensor, non_blocking=True)

            # Replay the captured graph (very fast: ~5μs vs ~500μs)
            self.graphs[batch_size].replay()

            # Get outputs from static buffers
            policy_out = self.static_policy_outputs[batch_size].clone()
            value_out = self.static_value_outputs[batch_size].clone()

            self.graph_hits += 1

        # Apply softmax if needed (outside lock for better concurrency)
        if not return_logits:
            policy_out = torch.softmax(policy_out.float(), dim=1)

        return policy_out, value_out

    def _infer_fallback(
        self,
        batch_tensor: torch.Tensor,
        return_logits: bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fallback inference for non-standard batch sizes (slow path).

        Args:
            batch_tensor: Input tensor (B, C, H, W)
            return_logits: Whether to return raw logits or probabilities

        Returns:
            tuple: (policy, value)
        """
        self.fallback_count += 1

        if self.fallback_count <= 5:
            self.logger.warning(
                f"Batch size {batch_tensor.size(0)} not in pre-captured graphs {list(self.graphs.keys())}. "
                f"Using slower fallback inference. Consider adding this batch size to capture list."
            )

        with torch.no_grad():
            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    policy_out, value_out = self.model(batch_tensor)
            else:
                policy_out, value_out = self.model(batch_tensor)

        if not return_logits:
            policy_out = torch.softmax(policy_out.float(), dim=1)

        return policy_out, value_out

    def get_stats(self) -> Dict[str, int]:
        """Get usage statistics for monitoring.

        Returns:
            Dictionary with hit/miss/fallback counts
        """
        total_calls = self.graph_hits + self.fallback_count
        hit_rate = (self.graph_hits / total_calls * 100) if total_calls > 0 else 0

        return {
            'graph_hits': self.graph_hits,
            'fallback_count': self.fallback_count,
            'total_calls': total_calls,
            'hit_rate_percent': hit_rate,
            'captured_batch_sizes': list(self.graphs.keys()),
        }

    def clear_stats(self) -> None:
        """Reset usage statistics."""
        self.graph_hits = 0
        self.fallback_count = 0

    def __repr__(self) -> str:
        stats = self.get_stats()
        return (
            f"CUDAGraphManager("
            f"captured_sizes={stats['captured_batch_sizes']}, "
            f"hits={stats['graph_hits']}, "
            f"fallbacks={stats['fallback_count']}, "
            f"hit_rate={stats['hit_rate_percent']:.1f}%)"
        )


def create_graph_manager_for_model(
    model: nn.Module,
    game: str,
    batch_sizes: Optional[List[int]] = None,
    **kwargs
) -> CUDAGraphManager:
    """Factory function to create CUDAGraphManager for game-specific models.

    Args:
        model: PyTorch model (already on CUDA)
        game: Game type ('gomoku', 'chess', 'go', 'go9', 'go19')
        batch_sizes: Batch sizes to capture (default: [8,16,32,64,128,256])
        **kwargs: Additional arguments for CUDAGraphManager

    Returns:
        CUDAGraphManager ready for warmup_and_capture()

    Example:
        >>> from src.neural.model import create_resnet_eca_model
        >>> model = create_resnet_eca_model('gomoku', size='128x12').cuda()
        >>> graph_mgr = create_graph_manager_for_model(model, 'gomoku')
        >>> graph_mgr.warmup_and_capture()
    """
    # Determine input shape based on game
    game = game.lower()

    if game == 'gomoku' or game.startswith('gomoku_'):
        input_shape = (36, 15, 15)
    elif game == 'chess':
        input_shape = (30, 8, 8)
    elif game == 'go' or game == 'go9':
        input_shape = (25, 9, 9)
    elif game == 'go19':
        input_shape = (25, 19, 19)
    else:
        raise ValueError(f"Unsupported game: {game}")

    return CUDAGraphManager(
        model=model,
        input_shape=input_shape,
        batch_sizes=batch_sizes,
        **kwargs
    )


if __name__ == '__main__':
    """Test and benchmark CUDA graph capture."""
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

    from src.neural.model import create_resnet_eca_model
    import time

    print("="*80)
    print("CUDA Graph Manager Test")
    print("="*80)

    # Create model
    print("\nCreating ResNet-ECA 128×12 model...")
    model = create_resnet_eca_model('gomoku', size='128x12')
    model = model.cuda()
    print(f"Model created: {model.get_num_parameters():,} parameters")

    # Create graph manager
    print("\nCreating CUDA graph manager...")
    graph_mgr = create_graph_manager_for_model(model, 'gomoku')

    # Warmup and capture
    print("\nWarming up and capturing graphs...")
    graph_mgr.warmup_and_capture()

    # Test inference
    print("\n" + "="*80)
    print("Testing Inference Speed (batch size 64)")
    print("="*80)

    batch_size = 64
    test_input = torch.randn(batch_size, 36, 15, 15, device='cuda', dtype=torch.float16)

    # Without graphs (baseline)
    print("\nBaseline (no CUDA graphs):")
    model.eval()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(100):
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                policy, value = model(test_input)
    torch.cuda.synchronize()
    elapsed_no_graph = time.perf_counter() - start
    pps_no_graph = (batch_size * 100) / elapsed_no_graph
    print(f"  Time: {elapsed_no_graph*1000:.2f} ms for 100 iterations")
    print(f"  Throughput: {pps_no_graph:,.1f} positions/sec")

    # With graphs
    print("\nWith CUDA graphs:")
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(100):
        policy, value = graph_mgr.infer(test_input)
    torch.cuda.synchronize()
    elapsed_with_graph = time.perf_counter() - start
    pps_with_graph = (batch_size * 100) / elapsed_with_graph
    print(f"  Time: {elapsed_with_graph*1000:.2f} ms for 100 iterations")
    print(f"  Throughput: {pps_with_graph:,.1f} positions/sec")

    # Speedup
    speedup = pps_with_graph / pps_no_graph
    print(f"\n✅ CUDA Graph Speedup: {speedup:.2f}×")
    print(f"   Expected: 2-3× (comments.md)")
    print(f"   Status: {'✅ PASS' if speedup >= 2.0 else '⚠️  BELOW TARGET'}")

    # Stats
    print(f"\nGraph Manager Stats:")
    stats = graph_mgr.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\n" + "="*80)
