from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.managers import SceneEntityCfg
    from isaaclab.sensors import ContactSensor

# Try to import Isaac Lab modules, but don't fail if they're not available
try:
    from isaaclab.utils.math import quat_apply_inverse, quat_rotate_vector
except ImportError:
    try:
        from isaaclab.utils.math import quat_rotate_inverse as quat_apply_inverse
        from isaaclab.utils.math import quat_rotate_vector
    except ImportError:
        quat_apply_inverse = None
        quat_rotate_vector = None

try:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.managers import SceneEntityCfg
    from isaaclab.sensors import ContactSensor
except ImportError:
    # Define dummy classes if imports fail
    class Articulation:
        pass
    class RigidObject:
        pass
    class SceneEntityCfg:
        def __init__(self, *args, **kwargs):
            pass
    class ContactSensor:
        pass

__all__ = [
    "energy",
    "stand_still",
    "orientation_l2",
    "upward",
    "joint_position_penalty",
    "feet_stumble",
    "feet_height_body",
    "foot_clearance_reward",
    "feet_too_near",
    "feet_contact_without_cmd",
    "air_time_variance_penalty",
    "feet_gait",
    "joint_mirror",
    "robot_skateboard_alignment",
    "skateboard_orientation",
    "feet_on_skateboard",
    "robot_skateboard_contact",
    "feet_near_skateboard_centerline",
    "feet_off_centerline_penalty",
    "com_within_support_polygon",
    "com_over_skateboard",
    "time_survived",
    "feet_flat_on_skateboard",
]

"""
Joint penalties.
"""


def energy(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize the energy used by the robot's joints."""
    asset: Articulation = env.scene[asset_cfg.name]

    qvel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    qfrc = asset.data.applied_torque[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(qvel) * torch.abs(qfrc), dim=-1)


def stand_still(
    env: ManagerBasedRLEnv, command_name: str = "base_velocity", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]

    reward = torch.sum(torch.abs(asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    return reward * (cmd_norm < 0.1)


"""
Robot.
"""


def orientation_l2(
    env: ManagerBasedRLEnv, desired_gravity: list[float], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward the agent for aligning its gravity with the desired gravity vector using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    desired_gravity = torch.tensor(desired_gravity, device=env.device)
    cos_dist = torch.sum(asset.data.projected_gravity_b * desired_gravity, dim=-1)  # cosine distance
    normalized = 0.5 * cos_dist + 0.5  # map from [-1, 1] to [0, 1]
    return torch.square(normalized)


def upward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.square(1 - asset.data.projected_gravity_b[:, 2])
    return reward


def joint_position_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, stand_still_scale: float, velocity_threshold: float
) -> torch.Tensor:
    """Penalize joint position error from default on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command("base_velocity"), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    reward = torch.linalg.norm((asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    return torch.where(torch.logical_or(cmd > 0.0, body_vel > velocity_threshold), reward, stand_still_scale * reward)


"""
Feet rewards.
"""


def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    forces_xy = torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
    # Penalize feet hitting vertical surfaces
    reward = torch.any(forces_xy > 4 * forces_z, dim=1).float()
    return reward


def feet_height_body(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    cur_footpos_translated = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_pos_w[:, :].unsqueeze(1)
    footpos_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[
        :, :
    ].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footpos_in_body_frame[:, i, :] = quat_apply_inverse(asset.data.root_quat_w, cur_footpos_translated[:, i, :])
        footvel_in_body_frame[:, i, :] = quat_apply_inverse(asset.data.root_quat_w, cur_footvel_translated[:, i, :])
    foot_z_target_error = torch.square(footpos_in_body_frame[:, :, 2] - target_height).view(env.num_envs, -1)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(footvel_in_body_frame[:, :, :2], dim=2))
    reward = torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def foot_clearance_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_height: float, std: float, tanh_mult: float
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
    reward = foot_z_target_error * foot_velocity_tanh
    return torch.exp(-torch.sum(reward, dim=1) / std)


def feet_too_near(
    env: ManagerBasedRLEnv, threshold: float = 0.2, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    feet_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    distance = torch.norm(feet_pos[:, 0] - feet_pos[:, 1], dim=-1)
    return (threshold - distance).clamp(min=0)


def feet_contact_without_cmd(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, command_name: str = "base_velocity"
) -> torch.Tensor:
    """
    Reward for feet contact when the command is zero.
    """
    # asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    # Check if command exists, if not assume robot is stationary (always reward contact)
    try:
        command_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
        reward = torch.sum(is_contact, dim=-1).float()
        return reward * (command_norm < 0.1)
    except KeyError:
        # No command manager configured - assume robot is stationary, always reward contact
        reward = torch.sum(is_contact, dim=-1).float()
        return reward


def air_time_variance_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize variance in the amount of time each foot spends in the air/on the ground relative to each other"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")
    # compute the reward
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    return torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(
        torch.clip(last_contact_time, max=0.5), dim=1
    )


"""
Feet Gait rewards.
"""


def feet_gait(
    env: ManagerBasedRLEnv,
    period: float,
    offset: list[float],
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.5,
    command_name=None,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    global_phase = ((env.episode_length_buf * env.step_dt) % period / period).unsqueeze(1)
    phases = []
    for offset_ in offset:
        phase = (global_phase + offset_) % 1.0
        phases.append(phase)
    leg_phase = torch.cat(phases, dim=-1)

    reward = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    for i in range(len(sensor_cfg.body_ids)):
        is_stance = leg_phase[:, i] < threshold
        reward += ~(is_stance ^ is_contact[:, i])

    if command_name is not None:
        cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
        reward *= cmd_norm > 0.1
    return reward


"""
Other rewards.
"""


def joint_mirror(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "joint_mirror_joints_cache") or env.joint_mirror_joints_cache is None:
        # Cache joint positions for all pairs
        env.joint_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints
        ]
    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over all joint pairs
    for joint_pair in env.joint_mirror_joints_cache:
        # Calculate the difference for each pair and add to the total reward
        reward += torch.sum(
            torch.square(asset.data.joint_pos[:, joint_pair[0][0]] - asset.data.joint_pos[:, joint_pair[1][0]]),
            dim=-1,
        )
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
    return reward


"""
Skateboard-specific rewards.
"""


def robot_skateboard_alignment(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    skateboard_cfg: SceneEntityCfg = SceneEntityCfg("skateboard"),
) -> torch.Tensor:
    """Reward robot for staying centered on skateboard (horizontal alignment).
    
    Uses exponential reward based on XY distance between robot base and skateboard center.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    skateboard: RigidObject = env.scene[skateboard_cfg.name]
    
    # Calculate XY distance between robot and skateboard
    pos_diff = robot.data.root_pos_w[:, :2] - skateboard.data.root_pos_w[:, :2]
    distance = torch.norm(pos_diff, dim=-1)
    
    # Exponential reward - decreases as robot moves away from skateboard center
    return torch.exp(-10.0 * distance)


def skateboard_orientation(
    env: ManagerBasedRLEnv,
    skateboard_cfg: SceneEntityCfg = SceneEntityCfg("skateboard"),
) -> torch.Tensor:
    """Reward skateboard for staying level (flat orientation).
    
    Checks that the skateboard's projected gravity points downward.
    Useful for moving skateboard scenarios.
    """
    skateboard: RigidObject = env.scene[skateboard_cfg.name]
    
    # Projected gravity should point down (0, 0, -1) when skateboard is level
    # We want projected_gravity_b[:, 2] to be close to -1
    gravity_error = torch.square(1.0 + skateboard.data.projected_gravity_b[:, 2])
    return torch.exp(-5.0 * gravity_error)


def feet_on_skateboard(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
    skateboard_cfg: SceneEntityCfg = SceneEntityCfg("skateboard"),
    threshold: float = 0.05,
) -> torch.Tensor:
    """Reward robot feet for being on/near the skateboard surface.
    
    Checks vertical distance between feet and skateboard top surface.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    skateboard: RigidObject = env.scene[skateboard_cfg.name]
    
    # Get feet positions
    feet_pos = robot.data.body_pos_w[:, robot_cfg.body_ids, :]
    
    # Calculate vertical distance from skateboard top surface
    # Assuming skateboard top is at skateboard.z + small offset
    skateboard_top_z = skateboard.data.root_pos_w[:, 2].unsqueeze(1)
    feet_z = feet_pos[:, :, 2]
    
    # Distance of feet from skateboard top
    z_distance = torch.abs(feet_z - skateboard_top_z)
    
    # Reward feet close to skateboard surface
    reward = torch.sum(torch.exp(-10.0 * z_distance), dim=-1)
    
    # Also check XY alignment - feet should be within skateboard bounds
    feet_xy = feet_pos[:, :, :2]
    skateboard_xy = skateboard.data.root_pos_w[:, :2].unsqueeze(1)
    xy_distance = torch.norm(feet_xy - skateboard_xy, dim=-1)
    
    # Penalize if feet are far from skateboard horizontally
    xy_reward = torch.sum(torch.exp(-5.0 * xy_distance), dim=-1)
    
    return (reward + xy_reward) / (2.0 * len(robot_cfg.body_ids))


def robot_skateboard_contact(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward for robot feet maintaining contact with skateboard.
    
    Uses contact sensor to detect when feet are touching the skateboard.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    
    # Check if feet bodies are in contact
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0
    
    # Reward having both feet in contact
    num_feet_in_contact = torch.sum(is_contact.float(), dim=-1)
    
    return num_feet_in_contact / len(sensor_cfg.body_ids)


def feet_near_skateboard_centerline(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
    skateboard_cfg: SceneEntityCfg = SceneEntityCfg("skateboard"),
    max_distance: float = 0.15,
) -> torch.Tensor:
    """Reward feet being near the centerline of the skateboard (along long axis).
    
    Computes perpendicular distance from each foot center to the skateboard's lengthwise 
    centerline (the line running along the skateboard's length). This encourages proper 
    stance with feet centered on the board width, preventing tipping left/right.
    
    Returns value between 0 (feet far from centerline) and 1 (feet on centerline).
    Uses exponential decay: reward = exp(-average_deviation / max_distance)
    
    Args:
        env: The environment.
        robot_cfg: Robot body configuration (feet).
        skateboard_cfg: Skateboard configuration.
        max_distance: Distance scale for exponential decay (meters). Default 0.15m.
        
    Returns:
        Reward tensor (num_envs,) with values between 0 and 1.
    """
    # Get assets
    robot: Articulation = env.scene[robot_cfg.name]
    skateboard: Articulation = env.scene[skateboard_cfg.name]
    
    # Foot positions (world frame)
    foot_pos_w = robot.data.body_pos_w[:, robot_cfg.body_ids, :]  # (num_envs, num_feet, 3)
    
    # Skateboard center (world frame)
    skateboard_pos_w = skateboard.data.root_pos_w  # (num_envs, 3)
    
    # Get skateboard's forward direction (long axis) in world frame
    # Simple approximation: use the vector between the two feet as the long axis
    if len(robot_cfg.body_ids) >= 2:
        foot_vec = foot_pos_w[:, 1, :2] - foot_pos_w[:, 0, :2]  # (num_envs, 2) - vector between feet
        foot_vec_norm = torch.norm(foot_vec, dim=-1, keepdim=True) + 1e-6
        forward_xy = foot_vec / foot_vec_norm  # Normalized direction
    else:
        # Fallback: assume skateboard aligned with X-axis
        forward_xy = torch.tensor([[1.0, 0.0]], device=env.device).repeat(env.num_envs, 1)
    
    # For each foot, compute perpendicular distance to centerline
    # Vector from skateboard center to foot (XY plane only)
    foot_to_skateboard = foot_pos_w[..., :2] - skateboard_pos_w[:, None, :2]  # (num_envs, num_feet, 2)
    
    # Project foot_to_skateboard onto forward direction (parallel component)
    parallel_distance = torch.sum(
        foot_to_skateboard * forward_xy[:, None, :], dim=-1
    )  # (num_envs, num_feet)
    
    # Parallel vector component
    parallel_vec = parallel_distance.unsqueeze(-1) * forward_xy[:, None, :]  # (num_envs, num_feet, 2)
    
    # Perpendicular vector component (distance from centerline - minimize this!)
    perpendicular_vec = foot_to_skateboard - parallel_vec  # (num_envs, num_feet, 2)
    perpendicular_distance = torch.norm(perpendicular_vec, dim=-1)  # (num_envs, num_feet)
    
    # Average perpendicular distance across both feet
    avg_distance = torch.mean(perpendicular_distance, dim=-1)  # (num_envs,)
    
    # Exponential decay reward: 1.0 when on centerline, decays as feet move away
    # Returns values between 0 and 1
    reward = torch.exp(-avg_distance / max_distance)  # (num_envs,)
    
    return reward


def feet_off_centerline_penalty(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
    skateboard_cfg: SceneEntityCfg = SceneEntityCfg("skateboard"),
    max_distance: float = 0.15,
) -> torch.Tensor:
    """Penalty for feet being far from the skateboard centerline.

    Returns penalty value between 0 (on centerline, no penalty) and 1 (far from centerline, max penalty).
    This is the inverse of feet_near_skateboard_centerline - designed for use with negative weight.
    """
    # Get assets
    robot: Articulation = env.scene[robot_cfg.name]
    skateboard: Articulation = env.scene[skateboard_cfg.name]

    # Get ankle/heel positions (world frame) - This is our reference point
    ankle_pos_w = robot.data.body_pos_w[:, robot_cfg.body_ids, :]  # (num_envs, num_feet, 3)
    
    # For simplicity, use ankle position directly (no need to offset to true foot center)
    # The ankle position should be close enough for centerline calculation
    foot_pos_w = ankle_pos_w  # (num_envs, num_feet, 3)

    # Skateboard center (world frame)
    skateboard_pos_w = skateboard.data.root_pos_w  # (num_envs, 3)

    # Get skateboard's forward direction (long axis) in world frame
    if len(robot_cfg.body_ids) >= 2:
        # Use vector between feet
        foot_vec = foot_pos_w[:, 1, :2] - foot_pos_w[:, 0, :2]
        foot_vec_norm = torch.norm(foot_vec, dim=-1, keepdim=True) + 1e-6
        forward_xy = foot_vec / foot_vec_norm
    else:
        forward_xy = torch.tensor([[1.0, 0.0]], device=env.device).repeat(env.num_envs, 1)

    # Compute perpendicular distance to centerline
    foot_to_skateboard = foot_pos_w[..., :2] - skateboard_pos_w[:, None, :2]
    parallel_distance = torch.sum(foot_to_skateboard * forward_xy[:, None, :], dim=-1)
    parallel_vec = parallel_distance.unsqueeze(-1) * forward_xy[:, None, :]
    perpendicular_vec = foot_to_skateboard - parallel_vec
    perpendicular_distance = torch.norm(perpendicular_vec, dim=-1)

    # Average distance
    avg_distance = torch.mean(perpendicular_distance, dim=-1)

    # Return penalty: 0 when on centerline, increases as feet move away
    penalty = 1.0 - torch.exp(-avg_distance / max_distance)

    return penalty


def com_within_support_polygon(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    feet_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
    margin: float = 0.05,
) -> torch.Tensor:
    """Reward for keeping the robot's center of mass (CoM) projection within the support polygon.
    
    This is a fundamental principle of bipedal balance:
    - CoM projection inside support polygon (between feet) = stable
    - CoM projection outside support polygon = falling/unstable
    
    For skateboarding, this ensures the robot maintains proper weight distribution
    and doesn't lean too far in any direction, which would cause tipping.
    
    The reward is highest when CoM is well within the support polygon (with margin),
    and decreases as it approaches or exceeds the polygon boundaries.
    
    Args:
        env: The environment.
        robot_cfg: Robot configuration (for CoM).
        feet_cfg: Feet body configuration (defines support polygon).
        margin: Safety margin inside support polygon for max reward (meters).
        
    Returns:
        Reward tensor (num_envs,) between 0 and 1.
    """
    # Get assets
    robot: Articulation = env.scene[robot_cfg.name]
    
    # Center of mass position (world frame, XY only)
    com_pos_w = robot.data.root_pos_w[:, :2]  # (num_envs, 2) - using base as proxy for CoM
    
    # Foot positions (world frame, XY only)
    feet_pos_w = robot.data.body_pos_w[:, feet_cfg.body_ids, :2]  # (num_envs, num_feet, 2)
    
    # Compute support polygon boundaries
    # For 2 feet, support polygon is a line segment between the feet
    # Distance from CoM to this line segment determines stability
    
    # Midpoint between feet (center of support)
    feet_center = torch.mean(feet_pos_w, dim=1)  # (num_envs, 2)
    
    # Vector between feet
    feet_vec = feet_pos_w[:, 1, :] - feet_pos_w[:, 0, :]  # (num_envs, 2)
    feet_dist = torch.norm(feet_vec, dim=-1, keepdim=True) + 1e-6  # (num_envs, 1)
    feet_dir = feet_vec / feet_dist  # Normalized direction between feet
    
    # Vector from feet center to CoM
    com_to_center = com_pos_w - feet_center  # (num_envs, 2)
    
    # Project CoM onto line between feet (parallel component)
    parallel_dist = torch.sum(com_to_center * feet_dir, dim=-1, keepdim=True)  # (num_envs, 1)
    
    # Perpendicular distance from CoM to line between feet
    parallel_vec = parallel_dist * feet_dir  # (num_envs, 2)
    perpendicular_vec = com_to_center - parallel_vec  # (num_envs, 2)
    perpendicular_dist = torch.norm(perpendicular_vec, dim=-1)  # (num_envs,)
    
    # Check if CoM is within the line segment (not beyond either foot)
    half_feet_dist = feet_dist.squeeze(-1) / 2.0  # (num_envs,)
    parallel_dist = parallel_dist.squeeze(-1).abs()  # (num_envs,)
    
    # Distance from support polygon boundary
    # If within line segment: distance is perpendicular_dist
    # If beyond line segment: distance is sqrt(perpendicular^2 + (parallel - half_feet)^2)
    beyond_feet = parallel_dist - half_feet_dist
    beyond_feet = torch.clamp(beyond_feet, min=0.0)  # (num_envs,) - 0 if within, positive if beyond
    
    total_dist = torch.sqrt(perpendicular_dist ** 2 + beyond_feet ** 2)  # (num_envs,)
    
    # Reward function:
    # - Inside margin: reward = 1.0
    # - At margin: reward = 0.37 (exp(-1))
    # - Outside: exponentially decreasing
    reward = torch.exp(-total_dist / margin)  # (num_envs,)
    
    return reward


def com_over_skateboard(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    skateboard_cfg: SceneEntityCfg = SceneEntityCfg("skateboard"),
    margin: float = 0.1,
) -> torch.Tensor:
    """Reward for keeping the robot's center of mass (CoM) projection over the skateboard.
    
    This ensures the robot's weight is distributed over the skateboard surface,
    not leaning off to the sides. Critical for skateboarding balance!
    
    Args:
        env: The environment.
        robot_cfg: Robot configuration (for CoM).
        skateboard_cfg: Skateboard configuration.
        margin: Tolerance distance from skateboard center (meters).
        
    Returns:
        Reward tensor (num_envs,) - high when CoM is over skateboard.
    """
    # Get assets
    robot: Articulation = env.scene[robot_cfg.name]
    skateboard: Articulation = env.scene[skateboard_cfg.name]
    
    # Center of mass position (world frame, XY only)
    com_pos_xy = robot.data.root_pos_w[:, :2]  # (num_envs, 2) - using base as proxy for CoM
    
    # Skateboard center position (world frame, XY only)
    skateboard_pos_xy = skateboard.data.root_pos_w[:, :2]  # (num_envs, 2)
    
    # Distance from CoM to skateboard center (XY plane)
    distance = torch.norm(com_pos_xy - skateboard_pos_xy, dim=-1)  # (num_envs,)
    
    # Exponential decay reward - high when close to skateboard center
    reward = torch.exp(-distance / margin)  # (num_envs,)
    
    return reward


def time_survived(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward for each timestep the robot survives/stays balanced.
    
    This is a simple constant reward per step that encourages the robot to stay
    balanced on the skateboard as long as possible.
    
    Args:
        env: The environment.
    
    Returns:
        Constant reward tensor of ones with shape (num_envs,).
    """
    return torch.ones(env.num_envs, device=env.device)


def feet_flat_on_skateboard(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
    skateboard_cfg: SceneEntityCfg = SceneEntityCfg("skateboard"),
    threshold_angle: float = 0.2,
) -> torch.Tensor:
    """Reward for keeping feet flat on the skateboard surface.
    
    Measures the angle between the foot's orientation and the skateboard's orientation.
    Feet should be flat (parallel) to the skateboard for stable balance.
    
    For skateboarding, flat feet are crucial because:
    - They maximize contact area with the board
    - They provide better stability and balance
    - They prevent sliding off due to poor surface contact
    
    Args:
        env: The environment.
        robot_cfg: Robot foot body configuration (ankles).
        skateboard_cfg: Skateboard configuration.
        threshold_angle: Maximum deviation angle (radians) before penalty. Default 0.2 (~11°).
        
    Returns:
        Reward tensor (num_envs,) with values between 0 (feet not flat) and 1 (feet perfectly flat).
    """
    # Get assets
    robot: Articulation = env.scene[robot_cfg.name]
    skateboard: Articulation = env.scene[skateboard_cfg.name]
    
    # Get foot body orientations (quaternions) in world frame
    foot_quats = robot.data.body_quat_w[:, robot_cfg.body_ids, :]  # (num_envs, num_feet, 4)
    
    # Get skateboard orientation in world frame
    skateboard_quat = skateboard.data.root_quat_w  # (num_envs, 4)
    
    # Calculate angle between quaternions using relative orientation
    # We'll use the angle between the up vectors (Z-axis) in each frame
    # Simplified approach: measure how different the orientations are
    
    # Expand skateboard quat to match foot quats shape
    skateboard_quat_expanded = skateboard_quat.unsqueeze(1).expand(-1, foot_quats.shape[1], -1)  # (num_envs, num_feet, 4)
    
    # Calculate relative orientation (foot relative to skateboard)
    # For quaternions: q_relative = q_skateboard^(-1) * q_foot
    # Simplified: we measure the angle between the orientations
    # Using the angle extracted from the relative quaternion
    q_foot = foot_quats  # (num_envs, num_feet, 4)
    q_board = skateboard_quat_expanded  # (num_envs, num_feet, 4)
    
    # Extract quaternion components
    # For q1 and q2, the angle between them is: 2 * arccos(|q1 · q2|)
    # where · is dot product
    dot_products = torch.sum(q_foot * q_board, dim=-1)  # (num_envs, num_feet)
    dot_products = torch.clamp(dot_products, -1.0, 1.0)
    
    # Calculate orientation difference angle
    # Use absolute value to get shortest rotation path
    angles = 2.0 * torch.acos(torch.abs(dot_products))  # (num_envs, num_feet)
    
    # Average angle across all feet
    avg_angle = torch.mean(angles, dim=-1)  # (num_envs,)
    
    # Reward: exponential decay based on how close to 0° (perfectly aligned)
    # Returns 1.0 when angle=0, decays toward 0 as angle increases
    reward = torch.exp(-avg_angle / threshold_angle)  # (num_envs,)
    
    return reward


    