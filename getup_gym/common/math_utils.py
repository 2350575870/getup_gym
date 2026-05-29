"""Math utilities for getup_gym."""

import torch
import numpy as np
from isaacgym.torch_utils import *


def torch_rand_float(lower: float, upper: float, shape: tuple, device: str) -> torch.Tensor:
    """Random float sampling (non-TorchScript to avoid nvrtc arch issues on RTX 5090)."""
    return torch.rand(*shape, device=device) * (upper - lower) + lower


def quat_apply_yaw(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Apply only the yaw component of a quaternion to a vector."""
    yaw_quat = quat.clone().view(-1, 4)
    yaw_quat[:, :2] = 0.0
    yaw_quat = yaw_quat / torch.norm(yaw_quat, dim=1, keepdim=True)
    return quat_apply(yaw_quat, vec)


def wrap_to_pi(angles: torch.Tensor) -> torch.Tensor:
    """Wrap angles to [-pi, pi]."""
    angles %= 2 * np.pi
    angles -= 2 * np.pi * (angles > np.pi)
    return angles


def torch_rand_sqrt_float(lower: float, upper: float, shape: tuple, device: str) -> torch.Tensor:
    """Random float sampling with sqrt distribution."""
    r = 2 * torch.rand(*shape, device=device) - 1
    r = torch.where(r < 0.0, -torch.sqrt(-r), torch.sqrt(r))
    r = (r + 1.0) / 2.0
    return (upper - lower) * r + lower
