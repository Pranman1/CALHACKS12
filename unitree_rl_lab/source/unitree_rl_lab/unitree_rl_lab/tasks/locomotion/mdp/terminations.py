from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.managers import SceneEntityCfg

# Try to import Isaac Lab modules, but don't fail if they're not available
try:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.managers import SceneEntityCfg
except ImportError:
    # Define dummy classes if imports fail
    class Articulation:
        pass
    class RigidObject:
        pass
    class SceneEntityCfg:
        def __init__(self, *args, **kwargs):
            pass

__all__ = [
    "robot_off_skateboard",
    "feet_off_skateboard",
    "skateboard_tilted",
    "body_below_skateboard",
    "body_ground_collision",
    "feet_touch_ground",
    "robot_ground_contact",
]


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


def body_below_skateboard(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*"),
    skateboard_cfg: SceneEntityCfg = SceneEntityCfg("skateboard"),
    margin: float = 0.05,
) -> torch.Tensor:
    """Terminate if ANY robot body part goes below the skateboard surface (touching ground).
    
    This distinguishes between:
    - Feet on skateboard (GOOD - at skateboard height or slightly above)
    - Feet/body on ground (BAD - below skateboard height)
    
    Args:
        env: The environment.
        robot_cfg: Robot body configuration (all bodies).
        skateboard_cfg: Skateboard scene entity configuration.
        margin: Safety margin below skateboard (meters) - parts below this = ground contact.
    
    Returns:
        Boolean tensor indicating which environments should terminate.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    skateboard: Articulation = env.scene[skateboard_cfg.name]
    
    # Get all robot body positions (Z height)
    body_positions_z = robot.data.body_pos_w[:, robot_cfg.body_ids, 2]  # (num_envs, num_bodies)
    
    # Get skateboard top surface height
    # From URDF: deck is ~2.5-3cm above root (trucks at 0.025m), add small margin for deck thickness
    skateboard_top_z = skateboard.data.root_pos_w[:, 2] + 0.03  # (num_envs,) - 3cm above root
    
    # Check if ANY body is below skateboard surface - margin
    min_allowed_height = skateboard_top_z - margin  # (num_envs,)
    body_below = body_positions_z < min_allowed_height.unsqueeze(1)  # (num_envs, num_bodies)
    
    # Terminate if ANY body part is below the threshold
    any_body_below = torch.any(body_below, dim=-1)  # (num_envs,)
    
    return any_body_below


def body_ground_collision(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*"),
    skateboard_cfg: SceneEntityCfg = SceneEntityCfg("skateboard"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*"),
    threshold: float = 1.0,
    height_margin: float = 0.1,
) -> torch.Tensor:
    """Terminate if robot body collides with GROUND (not skateboard).
    
    Uses HYBRID approach: Contact forces + Height check
    - Contact sensor detects if body is touching SOMETHING
    - Height check determines if that something is ground (low) vs skateboard (high)
    
    Logic: IF (body has contact) AND (body is near ground level) → Ground collision!
    
    Args:
        env: The environment.
        robot_cfg: Robot body configuration.
        skateboard_cfg: Skateboard configuration.
        sensor_cfg: Contact sensor configuration.
        threshold: Minimum contact force (N) to count as collision.
        height_margin: Height margin below skateboard to consider as "ground level" (m).
    
    Returns:
        Boolean tensor indicating which environments should terminate.
    """
    # Get assets and sensor
    robot: Articulation = env.scene[robot_cfg.name]
    skateboard: Articulation = env.scene[skateboard_cfg.name]
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    
    # Get contact forces for all bodies
    contact_forces = contact_sensor.data.net_forces_w  # (num_envs, num_sensor_bodies, 3)
    contact_force_magnitude = torch.norm(contact_forces, dim=-1)  # (num_envs, num_sensor_bodies)
    
    # Get body heights
    body_positions_z = robot.data.body_pos_w[:, robot_cfg.body_ids, 2]  # (num_envs, num_bodies)
    
    # Skateboard surface height
    skateboard_top_z = skateboard.data.root_pos_w[:, 2] + 0.03  # (num_envs,)
    ground_level_threshold = skateboard_top_z - height_margin  # (num_envs,)
    
    # Check if body has contact AND is near ground level
    has_contact = contact_force_magnitude > threshold  # (num_envs, num_sensor_bodies)
    is_low = body_positions_z < ground_level_threshold.unsqueeze(1)  # (num_envs, num_bodies)
    
    # Combine: body must have BOTH contact AND low height (assumes sensor bodies match robot bodies)
    ground_collision = has_contact & is_low  # (num_envs, num_bodies)
    
    # Terminate if ANY body has ground collision
    any_ground_collision = torch.any(ground_collision, dim=-1)  # (num_envs,)
    
    return any_ground_collision


def feet_touch_ground(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
    ground_clearance: float = 0.06,
) -> torch.Tensor:
    """Terminate if feet/legs touch or get too close to ground.
    
    This is a SIMPLE height-based check. Since the robot is on a skateboard (~8cm high),
    any foot that goes below the threshold MUST be touching/near the ground.
    
    Key measurements from G1 URDF:
    - Ankle_roll origin is measured by body_pos_w
    - Actual foot bottom is 3cm BELOW the ankle_roll origin
    - Skateboard deck is at ~8cm above ground
    - When foot is on ground (0cm), ankle origin is at ~3cm
    
    Args:
        env: The environment.
        robot_cfg: Robot body configuration (default: ankle_roll links = feet).
        ground_clearance: Height threshold (m) - below this = ground contact.
                         Default 0.06m = 6cm clearance (accounts for 3cm foot + 3cm margin)
    
    Returns:
        Boolean tensor indicating which environments should terminate.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    
    # Get ankle_roll link heights (the lowest tracking point on legs)
    body_positions_z = robot.data.body_pos_w[:, robot_cfg.body_ids, 2]  # (num_envs, num_bodies)
    
    # Check if ANY tracked body (foot/leg) is below the ground clearance threshold
    below_threshold = body_positions_z < ground_clearance  # (num_envs, num_bodies)
    
    # Terminate if ANY foot/leg is too low (touching ground)
    any_foot_on_ground = torch.any(below_threshold, dim=-1)  # (num_envs,)
    
    return any_foot_on_ground


def robot_ground_contact(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
    threshold: float = 1.0,
    ground_height: float = 0.08,
) -> torch.Tensor:
    """Terminate if robot feet contact something below the skateboard (i.e., the ground).
    
    Combines contact force detection with height check to differentiate between:
    - Skateboard contacts: feet at ~0.08-0.12m with contact forces (OK)
    - Ground contacts: feet below 0.08m with contact forces (TERMINATE)
    
    Args:
        env: The environment.
        robot_cfg: Robot body configuration (default: ankle_roll links = feet).
        sensor_cfg: Contact sensor configuration (default: ankle_roll contacts).
        threshold: Minimum contact force (N) to count as contact (default 1.0N).
        ground_height: Height threshold (m) - below this = ground contact (default 0.08m).
    
    Returns:
        Boolean tensor indicating which environments should terminate.
    """
    # Get robot articulation and feet positions
    robot: Articulation = env.scene[robot_cfg.name]
    feet_positions_z = robot.data.body_pos_w[:, robot_cfg.body_ids, 2]
    
    # Get contact forces on feet
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    contact_force_magnitude = torch.norm(contact_forces, dim=-1)
    
    # Feet are contacting something AND are below skateboard height (so must be ground)
    has_contact = contact_force_magnitude > threshold
    below_skateboard = feet_positions_z < ground_height
    
    # If ANY foot is both in contact and below skateboard height, terminate
    ground_contact = has_contact & below_skateboard
    return torch.any(ground_contact, dim=-1)

