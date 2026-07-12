"""
AlphaZero Neural Network Architecture
====================================

ResNet-based architecture with Squeeze-Excitation blocks for board game position evaluation.
Optimized for RTX 3060 Ti (8GB VRAM) with mixed precision support.

Architecture:
- Initial 3x3 conv layer (input_channels -> 192)
- 15 Residual blocks with SE attention (192 channels each)
- Policy head: 1x1 conv + linear layer
- Value head: 1x1 conv + global average pool + linear layers

Model size: ~10M parameters (fits with batch size 64 in 8GB VRAM)
Inference target: <10ms per batch of 64 with FP16 mixed precision
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List, Literal, Dict
import math


# ============================================================================
# EFFICIENT ATTENTION MODULES (Review.txt optimization)
# ============================================================================

class ECA(nn.Module):
    """Efficient Channel Attention (ECA) - Lightweight alternative to SE.

    Uses 1D convolution on channel statistics instead of FC layers.
    Provides similar performance to SE with near-zero overhead.

    Reference: ECA-Net (CVPR 2020)
    https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_ECA-Net_Efficient_Channel_Attention_for_Deep_Convolutional_Neural_Networks_CVPR_2020_paper.pdf

    Args:
        channels: Number of input channels
        k: Kernel size for 1D conv (default: 3 for comments.md recommendation)
    """
    def __init__(self, channels: int, k: int = 3):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,C,H,W)
        y = self.pool(x)                       # (B,C,1,1)
        y = y.squeeze(-1).transpose(1, 2)      # (B,1,C)
        y = self.conv(y)                       # (B,1,C)
        y = self.sig(y).transpose(1, 2).unsqueeze(-1)  # (B,C,1,1)
        return x * y


class ResidualBlockECA(nn.Module):
    """Residual block with ECA attention (comments.md recommendation).

    Clean ResNet block with ECA instead of SE for minimal overhead.
    This is the building block for ResNet-ECA 128×12 architecture.

    Architecture: Conv-BN-ReLU-Conv-BN-ECA + residual connection

    Args:
        channels: Number of input/output channels
        k: ECA kernel size (default: 3)
    """

    def __init__(self, channels: int, k: int = 3):
        super().__init__()
        self.channels = channels

        # First convolution
        self.conv1 = nn.Conv2d(
            channels, channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(channels)

        # Second convolution
        self.conv2 = nn.Conv2d(
            channels, channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(channels)

        # ECA attention (param-free, negligible overhead)
        self.eca = ECA(channels, k=k)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize convolution weights using He initialization."""
        for m in [self.conv1, self.conv2]:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

        # Initialize BatchNorm
        for m in [self.bn1, self.bn2]:
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through residual block with ECA.

        Args:
            x: Input tensor (batch_size, channels, height, width)

        Returns:
            Output tensor with same shape as input
        """
        identity = x

        # First conv-bn-relu
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        # Second conv-bn
        out = self.conv2(out)
        out = self.bn2(out)

        # ECA attention (replaces SE, much lighter)
        out = self.eca(out)

        # Residual connection
        out += identity
        out = F.relu(out)

        return out


# ============================================================================
# RE-PARAMETERIZATION UTILITIES (RepVGG-style train→deploy fusion)
# ============================================================================

def _fuse_conv_bn_weights(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fuse Conv2d and BatchNorm2d into single Conv2d (inference optimization)."""
    w = conv.weight
    b = conv.bias if conv.bias is not None else torch.zeros(w.size(0), device=w.device)
    g, beta, mean, var, eps = bn.weight, bn.bias, bn.running_mean, bn.running_var, bn.eps
    std = torch.sqrt(var + eps)
    w_fused = w * (g / std).reshape(-1, 1, 1, 1)
    b_fused = beta + (b - mean) * (g / std)
    return w_fused, b_fused


def _pad_1x1_to_3x3(w: torch.Tensor) -> torch.Tensor:
    """Pad 1x1 conv kernel to 3x3 for fusion."""
    if w is None:
        return None
    out_c, in_c, _, _ = w.shape
    w3 = torch.zeros((out_c, in_c, 3, 3), dtype=w.dtype, device=w.device)
    w3[:, :, 1:2, 1:2] = w
    return w3


def _identity_kernel_3x3(out_c: int, in_c: int, device: torch.device) -> torch.Tensor:
    """Create 3x3 identity kernel for residual connection fusion."""
    k = torch.zeros((out_c, in_c, 3, 3), device=device)
    if out_c == in_c:
        idx = torch.arange(out_c, device=device)
        k[idx, idx, 1, 1] = 1.0
    return k


# ============================================================================
# REPVGG-STYLE BLOCKS (Multi-branch train → Single conv deploy)
# ============================================================================

class RepECABlock(nn.Module):
    """Re-parameterizable block with ECA attention.

    Training: 3×3 conv + 1×1 conv + identity (if applicable)
    Inference: Single fused 3×3 conv (via switch_to_deploy())

    Provides rich training representation with fast inference.

    Args:
        in_ch: Input channels
        out_ch: Output channels
        stride: Stride for convolutions
        use_eca: Whether to use ECA attention
    """
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, use_eca: bool = True):
        super().__init__()
        self.in_ch, self.out_ch, self.stride = in_ch, out_ch, stride
        self.deploy = False

        # Training branches
        self.conv3 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch)

        self.conv1 = nn.Conv2d(in_ch, out_ch, 1, stride=stride, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)

        self.has_identity = (in_ch == out_ch and stride == 1)
        self.id_bn = nn.BatchNorm2d(out_ch) if self.has_identity else None

        self.eca = ECA(out_ch, k=5) if use_eca else nn.Identity()
        self.act = nn.ReLU(inplace=True)
        self.rbr_reparam = None  # Deployed fused conv

    @torch.no_grad()
    def get_equivalent_kernel_bias(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute equivalent kernel and bias by fusing all branches."""
        k3, b3 = _fuse_conv_bn_weights(self.conv3, self.bn3)
        k1, b1 = _fuse_conv_bn_weights(self.conv1, self.bn1)
        k1 = _pad_1x1_to_3x3(k1)

        if self.has_identity:
            kid = _identity_kernel_3x3(self.out_ch, self.in_ch, self.conv3.weight.device)
            g, beta = self.id_bn.weight, self.id_bn.bias
            mean, var, eps = self.id_bn.running_mean, self.id_bn.running_var, self.id_bn.eps
            std = torch.sqrt(var + eps)
            kid = kid * (g / std).reshape(-1, 1, 1, 1)
            bid = beta - mean * (g / std)
        else:
            kid = torch.zeros_like(k3)
            bid = torch.zeros_like(b3)

        return k3 + k1 + kid, b3 + b1 + bid

    @torch.no_grad()
    def switch_to_deploy(self):
        """Fuse multi-branch training structure into single conv for inference."""
        if self.deploy:
            return
        k, b = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(self.in_ch, self.out_ch, 3, stride=self.stride, padding=1, bias=True)
        # Move to same device as the original conv weights
        device = self.conv3.weight.device
        self.rbr_reparam = self.rbr_reparam.to(device)
        self.rbr_reparam.weight.data.copy_(k)
        self.rbr_reparam.bias.data.copy_(b)

        # Drop training branches
        for m in [self.conv3, self.bn3, self.conv1, self.bn1, self.id_bn]:
            if m is not None:
                m.requires_grad_(False)
        self.conv3 = self.bn3 = self.conv1 = self.bn1 = self.id_bn = None
        self.deploy = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.deploy:
            y = self.rbr_reparam(x)
        else:
            y = self.bn3(self.conv3(x)) + self.bn1(self.conv1(x))
            if self.has_identity:
                y = y + self.id_bn(x)
        y = self.eca(y)
        return self.act(y)


# ============================================================================
# GHOST BOTTLENECK (Efficient feature generation)
# ============================================================================

class GhostModule(nn.Module):
    """Generate more features from cheap operations.

    Splits features into intrinsic (primary conv) and ghost (cheap ops).
    Reduces FLOPs significantly with minimal quality loss.

    Args:
        in_ch: Input channels
        out_ch: Output channels
        ratio: Ratio of intrinsic to ghost features
        kernel_size: Kernel size for primary conv
        dw_size: Kernel size for cheap operation
        stride: Stride
        relu: Whether to use ReLU activation
    """
    def __init__(self, in_ch: int, out_ch: int, ratio: int = 2, kernel_size: int = 1,
                 dw_size: int = 3, stride: int = 1, relu: bool = True):
        super().__init__()
        intrinsic_ch = int(round(out_ch / ratio))
        ghost_ch = out_ch - intrinsic_ch

        self.primary = nn.Sequential(
            nn.Conv2d(in_ch, intrinsic_ch, kernel_size, stride, kernel_size // 2 if kernel_size > 1 else 0, bias=False),
            nn.BatchNorm2d(intrinsic_ch),
            nn.ReLU(inplace=True) if relu else nn.Identity(),
        )
        self.cheap = nn.Sequential(
            nn.Conv2d(intrinsic_ch, ghost_ch, dw_size, 1, dw_size // 2, groups=intrinsic_ch, bias=False),
            nn.BatchNorm2d(ghost_ch),
            nn.ReLU(inplace=True) if relu else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.primary(x)
        z = self.cheap(y)
        return torch.cat([y, z], dim=1)


class GhostBottleneck(nn.Module):
    """Ghost bottleneck residual block.

    Uses Ghost modules for efficient feature transformation.

    Args:
        in_ch: Input channels
        hidden_ch: Hidden layer channels
        out_ch: Output channels
        stride: Stride
        use_eca: Whether to use ECA attention
    """
    def __init__(self, in_ch: int, hidden_ch: int, out_ch: int, stride: int = 1, use_eca: bool = True):
        super().__init__()
        self.conv1 = GhostModule(in_ch, hidden_ch, relu=True)
        self.dw = nn.Conv2d(hidden_ch, hidden_ch, 3, stride=stride, padding=1, groups=hidden_ch, bias=False)
        self.dw_bn = nn.BatchNorm2d(hidden_ch)
        self.conv2 = GhostModule(hidden_ch, out_ch, relu=False)
        self.eca = ECA(out_ch, k=3) if use_eca else nn.Identity()

        self.shortcut = nn.Identity() if (stride == 1 and in_ch == out_ch) else nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.dw_bn(self.dw(out))
        out = self.conv2(out)
        out = self.eca(out)
        out = out + residual
        return self.act(out)


# ============================================================================
# SHUFFLENETV2-STYLE UNITS (Efficient channel mixing)
# ============================================================================

def channel_shuffle(x: torch.Tensor, groups: int = 2) -> torch.Tensor:
    """Channel shuffle operation for efficient information mixing."""
    b, c, h, w = x.size()
    assert c % groups == 0
    x = x.reshape(b, groups, c // groups, h, w).transpose(1, 2).contiguous()
    return x.reshape(b, c, h, w)


class ShuffleV2Unit(nn.Module):
    """ShuffleNetV2 basic unit.

    Uses channel splitting and shuffling for efficient mixing.

    Args:
        channels: Number of channels (must be even)
        use_eca: Whether to use ECA attention
    """
    def __init__(self, channels: int, use_eca: bool = True):
        super().__init__()
        assert channels % 2 == 0, "ShuffleV2 requires even number of channels"
        half = channels // 2

        self.branch = nn.Sequential(
            nn.Conv2d(half, half, 1, 1, 0, bias=False),
            nn.BatchNorm2d(half),
            nn.ReLU(inplace=True),
            nn.Conv2d(half, half, 3, 1, 1, groups=half, bias=False),
            nn.BatchNorm2d(half),
            nn.Conv2d(half, half, 1, 1, 0, bias=False),
            nn.BatchNorm2d(half),
            nn.ReLU(inplace=True),
        )
        self.eca = ECA(channels, k=3) if use_eca else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = torch.chunk(x, 2, dim=1)
        y2 = self.branch(x2)
        y = torch.cat([x1, y2], dim=1)
        y = channel_shuffle(y, 2)
        return self.eca(y)


# ============================================================================
# EARLY EXIT HEADS (Conditional computation)
# ============================================================================

class EarlyExitPolicyHead(nn.Module):
    """Early exit policy head for grid-based games."""
    def __init__(self, in_ch: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, 2, 1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(2, 1, 1, bias=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.head(x)  # (B,1,H,W)
        B, _, H, W = y.shape
        return y.view(B, H * W)


class EarlyExitValueHead(nn.Module):
    """Early exit value head."""
    def __init__(self, in_ch: int, hidden: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True)
        )
        self.fc = nn.Sequential(
            nn.Linear(1, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        y = F.adaptive_avg_pool2d(y, 1).view(y.size(0), -1)
        return self.fc(y)


class EarlyExitHead(nn.Module):
    """Combined early exit head (policy + value)."""
    def __init__(self, in_ch: int):
        super().__init__()
        self.policy = EarlyExitPolicyHead(in_ch)
        self.value = EarlyExitValueHead(in_ch)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.policy(x), self.value(x)


# ============================================================================
# ORIGINAL SQUEEZE-EXCITATION (Preserved for compatibility)
# ============================================================================

class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation block for channel attention.

    Computes channel-wise attention weights to recalibrate feature maps.

    Args:
        channels: Number of input channels
        reduction: Reduction ratio for bottleneck (default: 16)
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.channels = channels
        self.reduction = reduction

        # Squeeze: Global average pooling
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)

        # Excitation: Two fully connected layers with bottleneck
        reduced_channels = max(1, channels // reduction)
        self.fc1 = nn.Linear(channels, reduced_channels, bias=False)
        self.fc2 = nn.Linear(reduced_channels, channels, bias=False)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using proper initialization."""
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, mode='fan_out', nonlinearity='sigmoid')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through SE block.

        Args:
            x: Input tensor (batch_size, channels, height, width)

        Returns:
            Recalibrated feature maps with same shape as input
        """
        batch_size, channels, _, _ = x.size()

        # Squeeze: Global context embedding
        y = self.global_avgpool(x)  # (B, C, 1, 1)
        y = y.view(batch_size, channels)  # (B, C)

        # Excitation: Channel-wise scaling
        y = F.relu(self.fc1(y))  # (B, C//r)
        y = torch.sigmoid(self.fc2(y))  # (B, C)
        y = y.view(batch_size, channels, 1, 1)  # (B, C, 1, 1)

        # Scale original features
        return x * y


class ResidualBlock(nn.Module):
    """Residual block with Squeeze-Excitation attention.

    Architecture: Conv-BN-ReLU-Conv-BN-SE + residual connection

    Args:
        channels: Number of input/output channels
        use_se: Whether to include SE attention (default: True)
    """

    def __init__(self, channels: int, use_se: bool = True):
        super().__init__()
        self.channels = channels
        self.use_se = use_se

        # First convolution
        self.conv1 = nn.Conv2d(
            channels, channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(channels)

        # Second convolution
        self.conv2 = nn.Conv2d(
            channels, channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(channels)

        # Squeeze-Excitation block
        if use_se:
            self.se = SqueezeExcitation(channels)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize convolution weights using He initialization."""
        for m in [self.conv1, self.conv2]:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

        # Initialize BatchNorm
        for m in [self.bn1, self.bn2]:
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through residual block.

        Args:
            x: Input tensor (batch_size, channels, height, width)

        Returns:
            Output tensor with same shape as input
        """
        identity = x

        # First conv-bn-relu
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        # Second conv-bn
        out = self.conv2(out)
        out = self.bn2(out)

        # Squeeze-excitation attention
        if self.use_se:
            out = self.se(out)

        # Residual connection
        out += identity
        out = F.relu(out)

        return out


class PolicyHead(nn.Module):
    """Policy head for action probability prediction.

    Architecture: 1x1 conv -> flatten -> linear layer

    Args:
        input_channels: Number of input channels from backbone
        num_actions: Number of possible actions (board size squared)
        board_size: Optional board size tuple (height, width). If None, inferred from num_actions
    """

    def __init__(self, input_channels: int, num_actions: int, board_size: Optional[Tuple[int, int]] = None):
        super().__init__()
        self.input_channels = input_channels
        self.num_actions = num_actions

        # Infer board size from num_actions if not provided
        if board_size is None:
            board_size = self._infer_board_size(num_actions)
        self.board_height, self.board_width = board_size

        # 1x1 convolution to reduce channels
        self.conv = nn.Conv2d(input_channels, 2, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(2)

        # Linear layer for final policy logits - computed lazily on first forward pass
        self.fc = None

        # Initialize weights for conv and bn layers
        self._init_conv_weights()

    def _infer_board_size(self, num_actions: int) -> Tuple[int, int]:
        """Infer board size from number of actions.

        Args:
            num_actions: Number of possible actions

        Returns:
            Tuple of (height, width)
        """
        # Handle common game types
        if num_actions == 225:  # Gomoku 15x15
            return (15, 15)
        elif num_actions == 361:  # Go 19x19
            return (19, 19)
        elif num_actions == 64:  # Chess board positions
            return (8, 8)
        elif num_actions in (4096, 20480):  # Chess with move encodings
            return (8, 8)  # Still use 8x8 board for spatial features
        else:
            # Default: assume square board
            board_size = int(math.sqrt(num_actions))
            if board_size * board_size == num_actions:
                return (board_size, board_size)
            else:
                # Fallback for non-square boards
                return (15, 15)  # Default to Gomoku size

    def _init_conv_weights(self):
        """Initialize weights for conv and bn layers."""
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def _init_fc_weights(self):
        """Initialize weights for the linear layer."""
        if self.fc is not None:
            nn.init.kaiming_normal_(self.fc.weight, mode='fan_out', nonlinearity='linear')
            nn.init.constant_(self.fc.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through policy head.

        Args:
            x: Input features (batch_size, channels, height, width)

        Returns:
            Policy logits (batch_size, num_actions)
        """
        batch_size = x.size(0)

        # 1x1 conv and activation
        out = self.conv(x)
        out = self.bn(out)
        out = F.relu(out)

        # Flatten spatial dimensions
        out = out.view(batch_size, -1)

        # Lazy initialization of linear layer
        if self.fc is None:
            flattened_size = out.size(1)
            self.fc = nn.Linear(flattened_size, self.num_actions)
            # Move to same device as input
            self.fc = self.fc.to(out.device)
            self._init_fc_weights()

        # Final linear layer
        logits = self.fc(out)

        return logits


class ValueHead(nn.Module):
    """Value head for position evaluation.

    Architecture: 1x1 conv -> global avg pool -> linear layers -> tanh

    Args:
        input_channels: Number of input channels from backbone
    """

    def __init__(self, input_channels: int):
        super().__init__()
        self.input_channels = input_channels

        # 1x1 convolution to reduce channels
        self.conv = nn.Conv2d(input_channels, 1, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(1)

        # Global average pooling
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)

        # Fully connected layers
        self.fc1 = nn.Linear(1, 256)
        self.fc2 = nn.Linear(256, 1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.kaiming_normal_(self.fc2.weight, mode='fan_out', nonlinearity='tanh')
        nn.init.constant_(self.fc2.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through value head.

        Args:
            x: Input features (batch_size, channels, height, width)

        Returns:
            Value estimates (batch_size, 1) in range [-1, 1]
        """
        batch_size = x.size(0)

        # 1x1 conv and activation
        out = self.conv(x)
        out = self.bn(out)
        out = F.relu(out)

        # Global average pooling
        out = self.global_avgpool(out)  # (batch_size, 1, 1, 1)
        out = out.view(batch_size, -1)   # (batch_size, 1)

        # Fully connected layers
        out = F.relu(self.fc1(out))
        value = torch.tanh(self.fc2(out))

        return value


class AlphaZeroNet(nn.Module):
    """AlphaZero neural network with ResNet backbone and dual heads.

    Architecture optimized for board games with configurable input shapes.
    Designed to fit in 8GB VRAM with batch size 64.

    Args:
        input_channels: Number of input feature planes
        num_actions: Number of possible actions (typically board_size^2)
        num_blocks: Number of residual blocks (default: 20)
        hidden_channels: Number of channels in residual blocks (default: 256)
        use_se: Whether to use Squeeze-Excitation (default: True)
    """

    def __init__(
        self,
        input_channels: int,
        num_actions: int,
        num_blocks: int = 20,
        hidden_channels: int = 256,
        use_se: bool = True
    ):
        super().__init__()
        self.input_channels = input_channels
        self.num_actions = num_actions
        self.num_blocks = num_blocks
        self.hidden_channels = hidden_channels
        self.use_se = use_se

        # Initial convolution layer
        self.initial_conv = nn.Conv2d(
            input_channels, hidden_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.initial_bn = nn.BatchNorm2d(hidden_channels)

        # Residual tower
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_channels, use_se=use_se)
            for _ in range(num_blocks)
        ])

        # Output heads
        self.policy_head = PolicyHead(hidden_channels, num_actions)
        self.value_head = ValueHead(hidden_channels)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize initial convolution weights."""
        nn.init.kaiming_normal_(self.initial_conv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.initial_bn.weight, 1)
        nn.init.constant_(self.initial_bn.bias, 0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the network.

        Args:
            x: Input features (batch_size, channels, height, width)

        Returns:
            tuple: (policy_logits, values)
                policy_logits: Action probabilities (batch_size, num_actions)
                values: Position values (batch_size, 1) in range [-1, 1]
        """
        # Initial convolution
        out = self.initial_conv(x)
        out = self.initial_bn(out)
        out = F.relu(out)

        # Residual tower
        for block in self.residual_blocks:
            out = block(out)

        # Dual heads
        policy_logits = self.policy_head(out)
        values = self.value_head(out)

        return policy_logits, values

    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_memory_usage(self, batch_size: int, input_shape: Tuple[int, int, int]) -> dict:
        """Estimate GPU memory usage for inference.

        Args:
            batch_size: Batch size for inference
            input_shape: Input shape (channels, height, width)

        Returns:
            Dictionary with memory usage estimates in MB
        """
        # Parameter memory
        param_memory = sum(p.numel() * 4 for p in self.parameters()) / (1024 * 1024)  # 4 bytes per float32

        # Activation memory (rough estimate)
        c, h, w = input_shape
        activation_memory = batch_size * self.hidden_channels * h * w * 4 * (self.num_blocks + 2) / (1024 * 1024)

        # Output memory
        output_memory = batch_size * (self.num_actions + 1) * 4 / (1024 * 1024)

        total_memory = param_memory + activation_memory + output_memory

        return {
            'parameters_mb': param_memory,
            'activations_mb': activation_memory,
            'outputs_mb': output_memory,
            'total_mb': total_memory,
            'fits_8gb': total_memory < 7000,  # Conservative 7GB limit (leave 1GB safety)
            'optimal_batch_size': self._estimate_optimal_batch_size(input_shape)
        }

    def _estimate_optimal_batch_size(self, input_shape: Tuple[int, int, int]) -> int:
        """Estimate optimal batch size for RTX 3060 Ti.

        Args:
            input_shape: Input shape (channels, height, width)

        Returns:
            Recommended batch size for maximum GPU utilization
        """
        # Parameter memory (constant)
        param_memory_gb = sum(p.numel() * 4 for p in self.parameters()) / (1024**3)

        # Available memory for activations (7GB - parameters - 500MB safety)
        available_memory_gb = 7.0 - param_memory_gb - 0.5

        # Estimate activation memory per sample
        c, h, w = input_shape
        activation_per_sample_gb = (self.hidden_channels * h * w * 4 * (self.num_blocks + 2)) / (1024**3)

        # Calculate optimal batch size
        optimal_batch = int(available_memory_gb / activation_per_sample_gb)

        # Round down to nearest power of 2 and clamp to reasonable range
        optimal_batch = min(512, max(32, 2 ** int(optimal_batch.bit_length() - 1)))

        return optimal_batch


class AlphaZeroECA(nn.Module):
    """AlphaZero-style network with ECA attention (comments.md top recommendation).

    Clean ResNet trunk with ECA attention instead of SE for minimal overhead.
    This is the recommended architecture for Gomoku on RTX 3060 Ti.

    Expected performance (RTX 3060 Ti, FP16):
    - 128×12: ~28-40k positions/sec (3.7M params)
    - 96×12:  ~49-70k positions/sec (2.2M params, Ghost variant)

    Reference: comments.md Section 2A

    Args:
        in_ch: Number of input feature planes
        C: Channel count (128 for balanced, 96 for ultra-light)
        B: Number of residual blocks (12 recommended for Gomoku)
        board: Board size (15 for Gomoku, 19 for Go)
        num_actions: Number of possible actions
    """
    def __init__(self, in_ch: int, C: int, B: int, board: int, num_actions: int):
        super().__init__()
        self.in_ch = in_ch
        self.C = C
        self.B = B
        self.board = board
        self.num_actions = num_actions

        # Stem: Initial 3x3 conv
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, C, 3, padding=1, bias=False),
            nn.BatchNorm2d(C),
            nn.ReLU(inplace=True)
        )

        # Body: Residual blocks with ECA
        self.body = nn.ModuleList([
            ResidualBlockECA(C, k=3) for _ in range(B)
        ])

        # Policy head (2 planes → FC to board²)
        self.p_head = nn.Conv2d(C, 2, 1, bias=False)
        self.p_bn = nn.BatchNorm2d(2)
        self.p_fc = nn.Linear(2 * board * board, num_actions)

        # Value head
        self.v_head = nn.Conv2d(C, 1, 1, bias=False)
        self.v_bn = nn.BatchNorm2d(1)
        self.v_fc1 = nn.Linear(board * board, 256)
        self.v_fc2 = nn.Linear(256, 1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor (batch_size, in_ch, board, board)

        Returns:
            tuple: (policy_logits, values)
                policy_logits: (batch_size, num_actions)
                values: (batch_size, 1) in range [-1, 1]
        """
        # Stem
        z = self.stem(x)

        # Body (residual tower)
        for block in self.body:
            z = block(z)

        # Policy head
        p = F.relu(self.p_bn(self.p_head(z)))
        p = p.flatten(1)  # (B, 2*board*board)
        p = self.p_fc(p)  # (B, num_actions)

        # Value head
        v = F.relu(self.v_bn(self.v_head(z)))
        v = v.flatten(1)  # (B, board*board)
        v = F.relu(self.v_fc1(v))
        v = torch.tanh(self.v_fc2(v))  # (B, 1)

        return p, v

    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class GhostAlphaZeroECA(nn.Module):
    """Ultra-light AlphaZero with Ghost bottlenecks + ECA (comments.md alternative).

    Uses Ghost modules for efficient feature generation with ECA attention.
    Expected ~2× faster than standard ResNet-ECA with minimal quality loss.

    Expected performance (RTX 3060 Ti, FP16):
    - 96×12: ~49-70k positions/sec (2.2M params)

    Reference: comments.md Section 2B

    Args:
        in_ch: Number of input feature planes
        C: Channel count (96 recommended for ultra-light)
        B: Number of blocks (12 recommended)
        board: Board size
        num_actions: Number of possible actions
    """
    def __init__(self, in_ch: int, C: int, B: int, board: int, num_actions: int):
        super().__init__()
        self.in_ch = in_ch
        self.C = C
        self.B = B
        self.board = board
        self.num_actions = num_actions

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, C, 3, padding=1, bias=False),
            nn.BatchNorm2d(C),
            nn.ReLU(inplace=True)
        )

        # Body: Ghost bottlenecks with ECA
        self.body = nn.ModuleList([
            GhostBottleneck(C, hidden_ch=C, out_ch=C, stride=1, use_eca=True)
            for _ in range(B)
        ])

        # Policy head
        self.p_head = nn.Conv2d(C, 2, 1, bias=False)
        self.p_bn = nn.BatchNorm2d(2)
        self.p_fc = nn.Linear(2 * board * board, num_actions)

        # Value head
        self.v_head = nn.Conv2d(C, 1, 1, bias=False)
        self.v_bn = nn.BatchNorm2d(1)
        self.v_fc1 = nn.Linear(board * board, 256)
        self.v_fc2 = nn.Linear(256, 1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor (batch_size, in_ch, board, board)

        Returns:
            tuple: (policy_logits, values)
        """
        # Stem
        z = self.stem(x)

        # Body (ghost bottlenecks)
        for block in self.body:
            z = block(z)

        # Policy head
        p = F.relu(self.p_bn(self.p_head(z)))
        p = p.flatten(1)
        p = self.p_fc(p)

        # Value head
        v = F.relu(self.v_bn(self.v_head(z)))
        v = v.flatten(1)
        v = F.relu(self.v_fc1(v))
        v = torch.tanh(self.v_fc2(v))

        return p, v

    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_resnet_eca_model(game: str, size: str = '128x12', **kwargs) -> AlphaZeroECA:
    """Factory for ResNet-ECA models (comments.md top recommendation).

    Creates clean ResNet architecture with ECA attention for optimal
    speed/quality trade-off on RTX 3060 Ti.

    Args:
        game: Game type ('gomoku', 'chess', 'go', 'go9', 'go19')
        size: Model size - '128x12' (balanced, RECOMMENDED) or '96x12' (fast)
        **kwargs: Override parameters (in_ch, C, B, board, num_actions)

    Returns:
        AlphaZeroECA model configured for the game

    Expected Performance (RTX 3060 Ti, FP16):
        128×12: ~28-40k pps, 3.7M params (comments.md Section 1, Table)
        96×12:  ~49-70k pps, 2.2M params (ultra-light variant)

    Reference: comments.md Final Recommendations #1
    """
    game = game.lower()
    size = size.lower()

    # Size configurations
    size_configs = {
        '128x12': {'C': 128, 'B': 12},  # Balanced (RECOMMENDED)
        '96x12':  {'C': 96,  'B': 12},  # Fast
    }

    if size not in size_configs:
        raise ValueError(f"Invalid size: {size}. Must be '128x12' or '96x12'")

    config = size_configs[size].copy()

    # Game-specific configurations
    if game == 'gomoku' or game == 'gomoku_freestyle':
        config.update({
            'in_ch': 36,        # Enhanced Gomoku planes
            'board': 15,
            'num_actions': 225,
        })
    elif game == 'gomoku_renju' or game == 'gomoku_omok':
        config.update({
            'in_ch': 36,
            'board': 15,
            'num_actions': 225,
        })
    elif game == 'chess':
        config.update({
            'in_ch': 30,        # Enhanced Chess planes
            'board': 8,
            'num_actions': 4096,
        })
    elif game == 'go' or game == 'go9':
        config.update({
            'in_ch': 25,        # Enhanced Go planes
            'board': 9,
            'num_actions': 81,
        })
    elif game == 'go19':
        config.update({
            'in_ch': 25,
            'board': 19,
            'num_actions': 361,
        })
    else:
        raise ValueError(f"Unsupported game: {game}")

    # Apply user overrides
    config.update(kwargs)

    return AlphaZeroECA(**config)


def create_ghost_resnet_eca_model(game: str, **kwargs) -> GhostAlphaZeroECA:
    """Factory for Ghost-ResNet-ECA models (comments.md ultra-light variant).

    Creates ultra-light architecture with Ghost bottlenecks + ECA.
    ~2× faster than standard ResNet-ECA with minimal quality loss.

    Args:
        game: Game type
        **kwargs: Override parameters

    Returns:
        GhostAlphaZeroECA model (96×12 by default)

    Expected Performance (RTX 3060 Ti, FP16):
        96×12: ~49-70k pps, 2.2M params

    Reference: comments.md Section 2B
    """
    game = game.lower()

    # Default: 96×12 configuration
    config = {'C': 96, 'B': 12}

    # Game-specific configurations
    if game == 'gomoku' or game.startswith('gomoku_'):
        config.update({
            'in_ch': 36,
            'board': 15,
            'num_actions': 225,
        })
    elif game == 'chess':
        config.update({
            'in_ch': 30,
            'board': 8,
            'num_actions': 4096,
        })
    elif game == 'go' or game == 'go9':
        config.update({
            'in_ch': 25,
            'board': 9,
            'num_actions': 81,
        })
    elif game == 'go19':
        config.update({
            'in_ch': 25,
            'board': 19,
            'num_actions': 361,
        })
    else:
        raise ValueError(f"Unsupported game: {game}")

    # Apply user overrides
    config.update(kwargs)

    return GhostAlphaZeroECA(**config)


def create_model_for_game(game: str, use_fast_model: bool = False, model_size: str = 'small', **kwargs) -> nn.Module:
    """Factory function to create game-specific models.

    Args:
        game: Game type ('gomoku', 'chess', 'go')
        use_fast_model: If True, use FastMCTSNet (optimized), else AlphaZeroNet (default)
        model_size: Size for FastMCTSNet - 'nano', 'small' (default), 'medium', 'large'
        **kwargs: Additional model parameters

    Returns:
        Configured neural network model (AlphaZeroNet or FastMCTSNet)

    Raises:
        ValueError: If game type is not supported
    """
    # Use optimized FastMCTSNet if requested
    if use_fast_model:
        return create_fast_model_for_game(game, size=model_size, **kwargs)

    # Otherwise, use traditional AlphaZeroNet
    game = game.lower()

    # Set optimized defaults for RTX 3060 Ti (8GB VRAM)
    # Reduced from 20 blocks × 256 channels (23.8M params) to achieve target 10M params
    # and improve inference speed by 2.36× (180ms → 76ms estimated)
    default_kwargs = {
        'num_blocks': 15,      # Reduced from 20 for faster inference (GPU bottleneck fix)
        'hidden_channels': 192, # Reduced from 256 to achieve ~10M parameter target
        'use_se': True
    }
    default_kwargs.update(kwargs)

    # Game-specific configurations with ENHANCED feature planes
    if game == 'gomoku':
        return AlphaZeroNet(
            input_channels=36,  # Enhanced Gomoku: 36 planes with threat detection, run-length analysis
            num_actions=225,    # 15x15 board
            **default_kwargs
        )
    elif game == 'chess':
        return AlphaZeroNet(
            input_channels=30,  # Enhanced Chess: 30 planes with proper move history, castling, en passant
            num_actions=4096,   # 64 squares * 64 possible moves (simplified)
            **default_kwargs
        )
    elif game == 'go':
        return AlphaZeroNet(
            input_channels=25,  # Enhanced Go: 25 planes with proper move history separation
            num_actions=361,    # 19x19 board
            **default_kwargs
        )
    else:
        raise ValueError(f"Unsupported game type: {game}. Supported: 'gomoku', 'chess', 'go'")


def create_random_model(game: str, seed: Optional[int] = None) -> AlphaZeroNet:
    """Create a randomly initialized model for testing/baseline.

    Args:
        game: Game type
        seed: Random seed for reproducible initialization

    Returns:
        Randomly initialized model
    """
    if seed is not None:
        torch.manual_seed(seed)

    model = create_model_for_game(game)

    # Apply custom initialization if needed
    for module in model.modules():
        if isinstance(module, nn.Linear):
            # Xavier initialization for linear layers
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    return model


# Mixed precision compatibility
def enable_mixed_precision(model: nn.Module) -> nn.Module:
    """Enable mixed precision training compatibility.

    Args:
        model: Neural network model (AlphaZeroNet or FastMCTSNet)

    Returns:
        Model with mixed precision optimizations
    """
    # Convert BatchNorm to FP32 for numerical stability
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.float()

    return model


# Model validation utilities
def validate_model_output(model: AlphaZeroNet, input_tensor: torch.Tensor) -> bool:
    """Validate model outputs have correct shapes and ranges.

    Args:
        model: AlphaZeroNet model
        input_tensor: Sample input tensor

    Returns:
        True if outputs are valid
    """
    model.eval()
    with torch.no_grad():
        policy_logits, values = model(input_tensor)

        # Check shapes
        batch_size = input_tensor.size(0)
        if policy_logits.shape != (batch_size, model.num_actions):
            return False
        if values.shape != (batch_size, 1):
            return False

        # Check value range
        if not (-1 <= values.min() <= values.max() <= 1):
            return False

        # Check for NaN/inf
        if torch.isnan(policy_logits).any() or torch.isnan(values).any():
            return False
        if torch.isinf(policy_logits).any() or torch.isinf(values).any():
            return False

    return True


# ============================================================================
# FAST MCTS NET (Lightweight optimized architecture)
# ============================================================================

class FastMCTSNet(nn.Module):
    """Lightweight AlphaZero-style network with modern efficient architectures.

    Implements the optimization strategies from review.txt:
    - RepVGG-style blocks (train multi-branch, deploy single conv)
    - ECA attention (efficient alternative to SE)
    - Ghost/Shuffle bottlenecks (reduced FLOPs)
    - Early-exit heads (conditional computation)

    Expected performance gains:
    - RepVGG+ECA: +25-50% model speed
    - Ghost+Shuffle: +40-80% model speed
    - Early exits: +20-60% throughput (position-dependent)

    Args:
        input_channels: Number of input feature planes
        num_actions: Number of possible actions
        trunk_channels: Base channel count (default: 64)
        entry_blocks: Number of RepECA blocks at entry (default: 2)
        middle_blocks: Number of middle blocks (default: 8)
        exit_blocks: Number of RepECA blocks at exit (default: 2)
        middle_type: Type of middle blocks ('ghost' or 'shuffle', default: 'ghost')
        use_eca: Whether to use ECA attention (default: True)
        early_exit_points: Block indices for early exits (e.g., [4, 8])
        exit_entropy_threshold: Entropy threshold for early exit (None = disabled)
        exit_value_threshold: |Value| threshold for early exit (None = disabled)
    """
    def __init__(self,
                 input_channels: int,
                 num_actions: int,
                 trunk_channels: int = 64,
                 entry_blocks: int = 2,
                 middle_blocks: int = 8,
                 exit_blocks: int = 2,
                 middle_type: Literal['ghost', 'shuffle'] = 'ghost',
                 use_eca: bool = True,
                 early_exit_points: Optional[List[int]] = None,
                 exit_entropy_threshold: Optional[float] = None,
                 exit_value_threshold: Optional[float] = None):
        super().__init__()
        self.input_channels = input_channels
        self.num_actions = num_actions
        self.trunk_channels = trunk_channels
        self.entry_blocks = entry_blocks
        self.middle_blocks = middle_blocks
        self.exit_blocks = exit_blocks
        self.middle_type = middle_type
        self.use_eca = use_eca
        self.early_exit_points = early_exit_points or []
        self.exit_entropy_threshold = exit_entropy_threshold
        self.exit_value_threshold = exit_value_threshold

        C = trunk_channels

        # Stem convolution
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, C, 3, 1, 1, bias=False),
            nn.BatchNorm2d(C),
            nn.ReLU(inplace=True)
        )

        # Build backbone blocks
        blocks = []
        block_idx = 0

        # Entry: RepECA blocks
        for _ in range(entry_blocks):
            blocks.append(RepECABlock(C, C, use_eca=use_eca))
            block_idx += 1

        # Middle: Ghost or Shuffle blocks
        for _ in range(middle_blocks):
            if middle_type == 'ghost':
                blocks.append(GhostBottleneck(C, hidden_ch=C, out_ch=C, stride=1, use_eca=use_eca))
            elif middle_type == 'shuffle':
                if C % 2 != 0:
                    raise ValueError(f"ShuffleV2 requires even channels, got {C}")
                blocks.append(ShuffleV2Unit(C, use_eca=use_eca))
            else:
                raise ValueError(f"Invalid middle_type: {middle_type}, must be 'ghost' or 'shuffle'")
            block_idx += 1

        # Exit: RepECA blocks
        for _ in range(exit_blocks):
            blocks.append(RepECABlock(C, C, use_eca=use_eca))
            block_idx += 1

        self.blocks = nn.ModuleList(blocks)

        # Main output heads
        self.policy_head = PolicyHead(C, num_actions)
        self.value_head = ValueHead(C)

        # Early exit heads (optional)
        self.early_exits = nn.ModuleDict()
        if self.early_exit_points:
            for idx in self.early_exit_points:
                self.early_exits[str(idx)] = EarlyExitHead(C)

    def switch_to_deploy(self):
        """Fuse RepVGG blocks for inference (call after training)."""
        for block in self.blocks:
            if isinstance(block, RepECABlock):
                block.switch_to_deploy()

    @staticmethod
    def _compute_entropy(logits: torch.Tensor) -> torch.Tensor:
        """Compute policy entropy for early exit gating."""
        logp = F.log_softmax(logits, dim=-1)
        p = logp.exp()
        return -(p * logp).sum(dim=-1)  # (B,)

    def forward(self, x: torch.Tensor,
                inference_mode: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with optional early exits.

        Args:
            x: Input tensor (batch_size, channels, height, width)
            inference_mode: If True, enable early exit gating

        Returns:
            tuple: (policy_logits, values)
                policy_logits: (batch_size, num_actions)
                values: (batch_size, 1)
        """
        # Stem
        out = self.stem(x)

        # Backbone with optional early exits
        if inference_mode and self.early_exits and (self.exit_entropy_threshold or self.exit_value_threshold):
            for idx, block in enumerate(self.blocks, start=1):
                out = block(out)

                # Check for early exit
                if str(idx) in self.early_exits:
                    policy_logits, values = self.early_exits[str(idx)](out)

                    # Compute exit criteria
                    should_exit = torch.zeros(policy_logits.size(0), dtype=torch.bool, device=x.device)

                    if self.exit_entropy_threshold is not None:
                        entropy = self._compute_entropy(policy_logits)
                        should_exit |= (entropy <= self.exit_entropy_threshold)

                    if self.exit_value_threshold is not None:
                        value_confidence = values.abs().squeeze(-1)
                        should_exit |= (value_confidence >= self.exit_value_threshold)

                    # Exit if all samples meet criteria
                    if should_exit.all():
                        return policy_logits, values
        else:
            # Standard forward (no early exits)
            for block in self.blocks:
                out = block(out)

        # Main heads
        policy_logits = self.policy_head(out)
        values = self.value_head(out)

        return policy_logits, values

    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# FAST MODEL FACTORY FUNCTIONS (Game-specific optimized configs)
# ============================================================================

def create_fast_model_for_game(game: str, size: str = 'small', **kwargs) -> FastMCTSNet:
    """Factory function to create optimized FastMCTSNet models.

    Provides balanced configurations with speed vs capacity trade-offs based on
    comprehensive capacity analysis. See docs/network_capacity_analysis.md.

    Args:
        game: Game type ('gomoku', 'gomoku_renju', 'chess', 'go')
        size: Model size - 'nano', 'small', 'medium', 'large' (default: 'small')
              - nano: ~1.2M params, 6× speedup, amateur+ strength
              - small: ~2.5M params, 4× speedup, expert strength (RECOMMENDED)
              - medium: ~5M params, 3× speedup, master strength
              - large: ~10M params, 2× speedup, master+ strength
        **kwargs: Override default parameters

    Returns:
        Configured FastMCTSNet model

    Raises:
        ValueError: If game type or size is not supported
    """
    game = game.lower()
    size = size.lower()

    # Size configurations (balanced for RTX 3060 Ti)
    # Based on capacity analysis: need 2-5M params for superhuman Gomoku
    size_configs = {
        'nano': {
            'trunk_channels': 96,
            'entry_blocks': 2,
            'middle_blocks': 10,
            'exit_blocks': 2,
            # Expected: ~1.2M params, 6× speedup, amateur+ strength
        },
        'small': {
            'trunk_channels': 128,
            'entry_blocks': 3,
            'middle_blocks': 12,
            'exit_blocks': 3,
            # Expected: ~2.5M params, 4× speedup, expert strength (superhuman possible)
        },
        'medium': {
            'trunk_channels': 160,
            'entry_blocks': 3,
            'middle_blocks': 16,
            'exit_blocks': 3,
            # Expected: ~5M params, 3× speedup, master strength (superhuman likely)
        },
        'large': {
            'trunk_channels': 192,
            'entry_blocks': 4,
            'middle_blocks': 18,
            'exit_blocks': 4,
            # Expected: ~10M params, 2× speedup, master+ strength (superhuman guaranteed)
        },
    }

    if size not in size_configs:
        raise ValueError(f"Invalid size: {size}. Must be one of: nano, small, medium, large")

    # Base defaults from size configuration
    base_defaults = {
        **size_configs[size],
        'middle_type': 'ghost',
        'use_eca': True,
    }

    # Early exit configurations by size (disable for larger models)
    # Note: Early exits showed lower performance in benchmarks - disabled by default
    early_exit_configs = {
        'nano': {'early_exit_points': [], 'exit_entropy_threshold': None, 'exit_value_threshold': None},
        'small': {'early_exit_points': [], 'exit_entropy_threshold': None, 'exit_value_threshold': None},
        'medium': {'early_exit_points': [], 'exit_entropy_threshold': None, 'exit_value_threshold': None},
        'large': {'early_exit_points': [], 'exit_entropy_threshold': None, 'exit_value_threshold': None},
    }

    # Game-specific configurations
    if game == 'gomoku' or game == 'gomoku_freestyle':
        config = {
            **base_defaults,
            **early_exit_configs[size],
            'input_channels': 36,  # Enhanced Gomoku planes
            'num_actions': 225,    # 15×15 board
        }
    elif game == 'gomoku_renju' or game == 'gomoku_omok':
        config = {
            **base_defaults,
            **early_exit_configs[size],
            'input_channels': 36,
            'num_actions': 225,
        }
    elif game == 'chess':
        config = {
            **base_defaults,
            **early_exit_configs[size],
            'input_channels': 30,  # Enhanced Chess planes
            'num_actions': 4096,   # Simplified move encoding
        }
    elif game == 'go' or game == 'go9':
        config = {
            **base_defaults,
            **early_exit_configs[size],
            'input_channels': 25,  # Enhanced Go planes
            'num_actions': 81,     # 9×9 board
        }
    elif game == 'go19':
        config = {
            **base_defaults,
            **early_exit_configs[size],
            'input_channels': 25,
            'num_actions': 361,    # 19×19 board
        }
    else:
        raise ValueError(f"Unsupported game type: {game}. "
                        f"Supported: 'gomoku', 'gomoku_renju', 'chess', 'go', 'go9', 'go19'")

    # Apply user overrides
    config.update(kwargs)

    return FastMCTSNet(**config)


if __name__ == "__main__":
    """Basic testing when run directly."""
    print("AlphaZero Model Architecture Test")
    print("=" * 40)

    # Test different game configurations
    games = ['gomoku', 'chess', 'go']

    for game in games:
        print(f"\nTesting {game.capitalize()} model:")
        model = create_model_for_game(game)

        # Get model info
        num_params = model.get_num_parameters()
        print(f"  Parameters: {num_params:,} (~{num_params/1e6:.1f}M)")

        # Test forward pass
        if game == 'gomoku':
            test_input = torch.randn(4, 36, 15, 15)
        elif game == 'chess':
            test_input = torch.randn(4, 30, 8, 8)  # Enhanced Chess: 30 planes
        else:  # go
            test_input = torch.randn(4, 25, 19, 19)  # Enhanced Go: 25 planes

        policy_logits, values = model(test_input)
        print(f"  Policy shape: {policy_logits.shape}")
        print(f"  Value shape: {values.shape}")
        print(f"  Value range: [{values.min():.3f}, {values.max():.3f}]")

        # Memory estimation
        memory_info = model.get_memory_usage(64, test_input.shape[1:])
        optimal_batch = memory_info['optimal_batch_size']
        optimal_memory = model.get_memory_usage(optimal_batch, test_input.shape[1:])

        print(f"  Memory (batch=64): {memory_info['total_mb']:.1f}MB")
        print(f"  Optimal batch size: {optimal_batch}")
        print(f"  Memory (optimal): {optimal_memory['total_mb']:.1f}MB")
        print(f"  GPU utilization: {optimal_memory['total_mb']/8000*100:.1f}% of 8GB")

        # Validation
        is_valid = validate_model_output(model, test_input)
        print(f"  Output validation: {'✅' if is_valid else '❌'}")

    # Test FastMCTSNet (lightweight optimized architecture)
    print("\n" + "=" * 40)
    print("FastMCTSNet (Optimized Architecture) Test")
    print("=" * 40)

    for game in games:
        print(f"\nTesting {game.capitalize()} FastMCTSNet:")
        fast_model = create_fast_model_for_game(game)

        # Get model info
        num_params = fast_model.get_num_parameters()
        print(f"  Parameters: {num_params:,} (~{num_params/1e6:.1f}M)")

        # Test forward pass
        if game == 'gomoku':
            test_input = torch.randn(4, 36, 15, 15)
        elif game == 'chess':
            test_input = torch.randn(4, 30, 8, 8)
        else:  # go
            test_input = torch.randn(4, 25, 19, 19)

        # Test without early exits
        policy_logits, values = fast_model(test_input, inference_mode=False)
        print(f"  Policy shape: {policy_logits.shape}")
        print(f"  Value shape: {values.shape}")
        print(f"  Value range: [{values.min():.3f}, {values.max():.3f}]")

        # Test with early exits (inference mode)
        policy_logits2, values2 = fast_model(test_input, inference_mode=True)
        print(f"  Early exit enabled: policy={policy_logits2.shape}, value={values2.shape}")

        # Test deploy mode (fused convolutions)
        fast_model.switch_to_deploy()
        policy_logits3, values3 = fast_model(test_input, inference_mode=False)
        print(f"  Deploy mode (fused): policy={policy_logits3.shape}, value={values3.shape}")

        # Validation
        # Create a simple validation (similar to AlphaZeroNet validation)
        fast_model.eval()
        with torch.no_grad():
            p, v = fast_model(test_input)
            is_valid = (p.shape == (4, fast_model.num_actions) and
                       v.shape == (4, 1) and
                       -1 <= v.min() <= v.max() <= 1)
        print(f"  Output validation: {'✅' if is_valid else '❌'}")

    print("\n" + "=" * 40)
    print("Comparison: AlphaZeroNet vs FastMCTSNet")
    print("=" * 40)

    # Compare parameter counts
    az_model = create_model_for_game('gomoku')
    fast_model = create_fast_model_for_game('gomoku')
    az_params = az_model.get_num_parameters()
    fast_params = fast_model.get_num_parameters()

    print(f"\nGomoku models:")
    print(f"  AlphaZeroNet: {az_params:,} params (~{az_params/1e6:.1f}M)")
    print(f"  FastMCTSNet:  {fast_params:,} params (~{fast_params/1e6:.1f}M)")
    print(f"  Reduction:    {(1 - fast_params/az_params)*100:.1f}%")
    print(f"\nExpected speedups (FastMCTSNet):")
    print(f"  Model inference: 1.25-1.8× (RepVGG+ECA+Ghost)")
    print(f"  With early exits: 1.5-2.5× (position-dependent)")
    print(f"  Combined: 1.9-4.5× total throughput gain")
