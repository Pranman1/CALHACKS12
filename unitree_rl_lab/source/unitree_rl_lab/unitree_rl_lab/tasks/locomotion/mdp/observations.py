from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.managers import SceneEntityCfg

__all__ = [
    "gait_phase",
    "robot_skateboard_relative_position",
    "robot_skateboard_xy_distance",
    "skateboard_orientation_body_frame",
    "feet_skateboard_relative_height",
]


def gait_phase(env: ManagerBasedRLEnv, period: float) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf"):
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    global_phase = (env.episode_length_buf * env.step_dt) % period / period

    phase = torch.zeros(env.num_envs, 2, device=env.device)
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
    return phase


"""
Skateboard-specific observations.
"""


def robot_skateboard_relative_position(
    env: ManagerBasedRLEnv,
    robot_cfg: "SceneEntityCfg" = None,
    skateboard_cfg: "SceneEntityCfg" = None,
) -> torch.Tensor:
    """Relative position of robot base to skateboard center (in robot's frame).
    
    Returns 3D position vector (x, y, z) of skateboard relative to robot base.
    This tells the robot where the skateboard is beneath it.
    """
    # Import locally to avoid import issues during module loading
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.managers import SceneEntityCfg
    try:
        from isaaclab.utils.math import quat_apply_inverse
    except ImportError:
        from isaaclab.utils.math import quat_rotate_inverse as quat_apply_inverse
    
    if robot_cfg is None:
        robot_cfg = SceneEntityCfg("robot")
    if skateboard_cfg is None:
        skateboard_cfg = SceneEntityCfg("skateboard")
    
    robot: Articulation = env.scene[robot_cfg.name]
    skateboard: RigidObject = env.scene[skateboard_cfg.name]
    
    # Calculate position difference in world frame
    pos_diff = skateboard.data.root_pos_w - robot.data.root_pos_w
    
    # Transform to robot's body frame
    pos_diff_body = quat_apply_inverse(robot.data.root_quat_w, pos_diff)
    
    return pos_diff_body


def robot_skateboard_xy_distance(
    env: ManagerBasedRLEnv,
    robot_cfg: "SceneEntityCfg" = None,
    skateboard_cfg: "SceneEntityCfg" = None,
) -> torch.Tensor:
    """Horizontal (XY) distance between robot and skateboard center.
    
    Returns scalar distance. Robot should keep this near zero to stay centered.
    """
    # Import locally to avoid import issues during module loading
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.managers import SceneEntityCfg
    
    if robot_cfg is None:
        robot_cfg = SceneEntityCfg("robot")
    if skateboard_cfg is None:
        skateboard_cfg = SceneEntityCfg("skateboard")
    
    robot: Articulation = env.scene[robot_cfg.name]
    skateboard: RigidObject = env.scene[skateboard_cfg.name]
    
    # Calculate XY distance
    pos_diff = robot.data.root_pos_w[:, :2] - skateboard.data.root_pos_w[:, :2]
    distance = torch.norm(pos_diff, dim=-1, keepdim=True)
    
    return distance


def skateboard_orientation_body_frame(
    env: ManagerBasedRLEnv,
    skateboard_cfg: "SceneEntityCfg" = None,
) -> torch.Tensor:
    """Skateboard's projected gravity in its own body frame.
    
    Tells robot if skateboard is tilting. Should be (0, 0, -1) when level.
    """
    # Import locally to avoid import issues during module loading
    from isaaclab.assets import RigidObject
    from isaaclab.managers import SceneEntityCfg
    
    if skateboard_cfg is None:
        skateboard_cfg = SceneEntityCfg("skateboard")
    
    skateboard: RigidObject = env.scene[skateboard_cfg.name]
    return skateboard.data.projected_gravity_b


def feet_skateboard_relative_height(
    env: ManagerBasedRLEnv,
    robot_cfg: "SceneEntityCfg" = None,
    skateboard_cfg: "SceneEntityCfg" = None,
) -> torch.Tensor:
    """Height of robot feet relative to skateboard top surface.
    
    Returns height difference for each foot. Should be near zero when feet on skateboard.
    """
    # Import locally to avoid import issues during module loading
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.managers import SceneEntityCfg
    
    if robot_cfg is None:
        robot_cfg = SceneEntityCfg("robot", body_names=".*ankle_roll.*")
    if skateboard_cfg is None:
        skateboard_cfg = SceneEntityCfg("skateboard")
    
    robot: Articulation = env.scene[robot_cfg.name]
    skateboard: RigidObject = env.scene[skateboard_cfg.name]
    
    # Get feet positions
    feet_pos = robot.data.body_pos_w[:, robot_cfg.body_ids, :]
    
    # Calculate height difference
    skateboard_z = skateboard.data.root_pos_w[:, 2].unsqueeze(1)
    feet_z = feet_pos[:, :, 2]
    height_diff = feet_z - skateboard_z
    
    # Flatten to return all feet heights
    return height_diff.view(env.num_envs, -1)
