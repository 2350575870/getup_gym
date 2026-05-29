import torch
from typing import Tuple
import numpy as np
from isaacgym import gymtorch, gymapi, gymutil
from legged_gym.utils import torch_jit_utils, math


def CreateWireframeSphereLines(
    radius: float, color_list: list, pose: torch.Tensor, device: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    """create wireframe sphere lines

    Args:
        radius (float): radius of the sphere
        color (list): color of the sphere [r,g,b] (0~1)
        pose (torch.Tensor): pose of the sphere [x,y,z,quat_x,quat_y,quat_z,quat_w]
        device (str): device to use

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: wireframe sphere lines, color of the sphere
    """
    pose = pose.to(device)
    radius45 = radius * 0.7071067811865475
    wire = torch.zeros(4, 6, device=device, requires_grad=False, dtype=torch.float32)
    wire[0, :] = torch.tensor([-radius, 0, 0, radius, 0, 0], device=device, requires_grad=False, dtype=torch.float32)
    wire[1, :] = torch.tensor([0, -radius, 0, 0, radius, 0], device=device, requires_grad=False, dtype=torch.float32)
    wire[2, :] = torch.tensor(
        [-radius45, -radius45, 0, radius45, radius45, 0], device=device, requires_grad=False, dtype=torch.float32
    )
    wire[3, :] = torch.tensor(
        [-radius45, radius45, 0, radius45, -radius45, 0], device=device, requires_grad=False, dtype=torch.float32
    )

    color = torch.zeros(4, 3, device=device, requires_grad=False, dtype=torch.float32)
    color[0, :] = torch.tensor(color_list, device=device, requires_grad=False, dtype=torch.float32)
    color[1, :] = torch.tensor(color_list, device=device, requires_grad=False, dtype=torch.float32)
    color[2, :] = torch.tensor(color_list, device=device, requires_grad=False, dtype=torch.float32)
    color[3, :] = torch.tensor(color_list, device=device, requires_grad=False, dtype=torch.float32)

    wire_local = wire.view(-1, 3)
    wire_pose = pose.repeat(wire_local.shape[0], 1)
    wire_global = torch_jit_utils.quat_apply(wire_pose[:, 3:7], wire_local)
    wire_global = wire_global + wire_pose[:, :3]
    wire_global = wire_global.view(-1, 6)

    return wire_global, color


def drw_dbg_viz(gym, viewer, pos: torch.Tensor, lines: torch.Tensor, color: torch.Tensor):
    """draw debug visualizer with given position and pattern configuration

    Args:
        gym (_type_): isaacgym handler
        viewer (_type_): isaacgym viewer
        pos (torch.Tensor): drawing position 3dim(only translation) or 7dim(translation+rotation)
        lines (torch.Tensor): lines to be drawn
        color (torch.Tensor): line color
    """
    pos = pos.view(-1, pos.shape[-1])
    pos_dim = pos.shape[1]
    pos2 = pos.repeat(1, 2 * lines.shape[0]).view(-1, pos_dim)
    local_vertices = lines.view(-1, 3)
    local_vertices = local_vertices.repeat(pos.shape[0], 1)
    if pos2.shape[1] == 3:  # no rotation
        global_vertices = local_vertices + pos2
    else:
        global_vertices = torch_jit_utils.quat_apply(pos2[:, 3:7], local_vertices) + pos2[:, :3]
    global_vertices = global_vertices.view(-1, 6)
    color = color.repeat(global_vertices.shape[0], 1)
    gym.add_lines(viewer, None, global_vertices.shape[0], global_vertices.cpu().numpy(), color.cpu().numpy())
