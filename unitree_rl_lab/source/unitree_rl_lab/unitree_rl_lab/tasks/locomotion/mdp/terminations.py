from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


"""
Skateboard-specific termination functions.
"""


def robot_off_skateboard(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    skateboard_cfg: SceneEntityCfg = SceneEntityCfg("skateboard"),
    max_distance: float = 0.3,
) -> torch.Tensor:
    """Terminate if robot moves too far from skateboard center horizontally.
    
    Args:
        env: The environment.
        robot_cfg: Robot scene entity configuration.
        skateboard_cfg: Skateboard scene entity configuration.
        max_distance: Maximum allowed XY distance in meters (default 0.3m = 30cm).
    
    Returns:
        Boolean tensor indicating which environments should terminate.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    skateboard: RigidObject = env.scene[skateboard_cfg.name]
    
    # Calculate XY distance between robot and skateboard
    pos_diff = robot.data.root_pos_w[:, :2] - skateboard.data.root_pos_w[:, :2]
    distance = torch.norm(pos_diff, dim=-1)
    
    # Terminate if distance exceeds threshold
    return distance > max_distance


def feet_off_skateboard(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
    skateboard_cfg: SceneEntityCfg = SceneEntityCfg("skateboard"),
    max_height_diff: float = 0.3,
    max_xy_distance: float = 0.5,
) -> torch.Tensor:
    """Terminate if robot feet are too far from skateboard (fell off).
    
    Args:
        env: The environment.
        robot_cfg: Robot feet configuration.
        skateboard_cfg: Skateboard scene entity configuration.
        max_height_diff: Maximum vertical distance from skateboard (meters).
        max_xy_distance: Maximum horizontal distance from skateboard center (meters).
    
    Returns:
        Boolean tensor indicating which environments should terminate.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    skateboard: RigidObject = env.scene[skateboard_cfg.name]
    
    # Get feet positions
    feet_pos = robot.data.body_pos_w[:, robot_cfg.body_ids, :]
    
    # Check vertical distance (Z)
    skateboard_z = skateboard.data.root_pos_w[:, 2].unsqueeze(1)
    feet_z = feet_pos[:, :, 2]
    z_violation = torch.abs(feet_z - skateboard_z) > max_height_diff
    
    # Check horizontal distance (XY)
    feet_xy = feet_pos[:, :, :2]
    skateboard_xy = skateboard.data.root_pos_w[:, :2].unsqueeze(1)
    xy_distance = torch.norm(feet_xy - skateboard_xy, dim=-1)
    xy_violation = xy_distance > max_xy_distance
    
    # Terminate if ANY foot violates either constraint
    any_foot_off = torch.any(z_violation | xy_violation, dim=-1)
    
    return any_foot_off


def skateboard_tilted(
    env: ManagerBasedRLEnv,
    skateboard_cfg: SceneEntityCfg = SceneEntityCfg("skateboard"),
    max_tilt_angle: float = 0.5,
) -> torch.Tensor:
    """Terminate if skateboard tilts too much (for moving skateboard scenarios).
    
    Args:
        env: The environment.
        skateboard_cfg: Skateboard scene entity configuration.
        max_tilt_angle: Maximum tilt angle in radians (default 0.5 rad ≈ 28°).
    
    Returns:
        Boolean tensor indicating which environments should terminate.
    """
    skateboard: RigidObject = env.scene[skateboard_cfg.name]
    
    # Check if projected gravity Z component deviates too much from -1
    # When level: projected_gravity_b = (0, 0, -1), so z = -1
    # When tilted: z component moves away from -1
    gravity_z = skateboard.data.projected_gravity_b[:, 2]
    
    # Convert to angle: cos(theta) = gravity_z (when normalized)
    # Terminate if tilt exceeds threshold
    tilt_violation = gravity_z > -torch.cos(torch.tensor(max_tilt_angle, device=env.device))
    
    return tilt_violation

