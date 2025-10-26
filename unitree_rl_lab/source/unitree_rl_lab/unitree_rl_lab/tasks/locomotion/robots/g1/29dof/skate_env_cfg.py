import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_29DOF_CFG as ROBOT_CFG
from unitree_rl_lab.tasks.locomotion import mdp

# Import skateboard-specific functions - direct imports
from unitree_rl_lab.tasks.locomotion.mdp.observations import (
    robot_skateboard_relative_position,
    robot_skateboard_xy_distance,
    skateboard_orientation_body_frame,
    feet_skateboard_relative_height,
)
from unitree_rl_lab.tasks.locomotion.mdp.rewards import (
    robot_skateboard_alignment,
    skateboard_orientation,
    feet_on_skateboard,
    robot_skateboard_contact,
    feet_near_skateboard_centerline,
    feet_off_centerline_penalty,
    com_within_support_polygon,
    com_over_skateboard,
    time_survived,
    feet_contact_without_cmd,
    feet_stumble,
    upward,  # We are replacing this, but the import is harmless
)
from unitree_rl_lab.tasks.locomotion.mdp.terminations import (
    robot_off_skateboard,
    feet_off_skateboard,
    skateboard_tilted,
    body_ground_collision,
    feet_touch_ground,
    robot_ground_contact,
)

# Define skateboard configuration using URDF
import os
# Use absolute path to avoid path calculation issues across different installations
SKATEBOARD_URDF_PATH = r"C:\Users\Jisoo\calhackstest\CALHACKS12\unitree_rl_lab\assets\robots\skateboard_description\urdf\skateboard.urdf"

SKATEBOARD_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Skateboard",
    spawn=sim_utils.UrdfFileCfg(
        asset_path=SKATEBOARD_URDF_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=1.0,
            retain_accelerations=False,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,  # Increased from 4 for more stable contacts
            solver_velocity_iteration_count=2,  # Increased from 0 for smoother dynamics
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            drive_type="force",
            target_type="position",
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=50.0,  # Increased from 0.0 - makes trucks realistic/stiff for sim-to-real
                damping=5.0,  # Increased from 0.0 - adds realistic resistance
            ),
        ),
        fix_base=False,  # Allow skateboard to move (but with realistic stiffness)
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.025),  # Wheels touching ground (deck 2.5cm above ground)
        rot=(1.0, 0.0, 0.0, 0.0),  # Flat orientation (w, x, y, z quaternion)
        lin_vel=(0.0, 0.0, 0.0),  # Stationary
        ang_vel=(0.0, 0.0, 0.0),  # No rotation
    ),
    actuators={},  # No actuators for skateboard
)


@configclass
class SkateboardSceneCfg(InteractiveSceneCfg):
    """Configuration for the skateboard balancing scene."""

    # ground terrain - flat plane
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )

    # skateboard - stationary
    skateboard: ArticulationCfg = SKATEBOARD_CFG

    # robot - starts on skateboard, rotated 90 degrees (parallel to skateboard length)
    robot: ArticulationCfg = ROBOT_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.90),  # Higher up to be on skateboard (adjust as needed)
            rot=(0.7071, 0.0, 0.0, 0.7071),  # 90 degree rotation around Z-axis (w, x, y, z quaternion)
            joint_pos={
                # CHANGED: Start in a neutral, straight-legged pose
                "left_hip_pitch_joint": -0.2,
                "right_hip_pitch_joint": -0.2,
                ".*_knee_joint": 0.4,
                ".*_ankle_pitch_joint": -0.4,
                # Arms can stay in their default pose
                ".*_shoulder_pitch_joint": 0.3,
                "left_shoulder_roll_joint": 0.25,
                "right_shoulder_roll_joint": -0.25,
                ".*_elbow_joint": 0.97,
                "left_wrist_roll_joint": 0.15,
                "right_wrist_roll_joint": -0.15,
            },
            joint_vel={".*": 0.0},
        ),
    )

    # sensors
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )
    
    # No ground_contact sensor needed - using height-based termination instead

    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.5, 1.0),
            "dynamic_friction_range": (0.5, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "mass_distribution_params": (-0.5, 1.0),
            "operation": "add",
        },
    )
    
    # DOMAIN RANDOMIZATION: Randomize ONLY skateboard truck stiffness for sim-to-real
    # This helps the policy adapt to variations in truck tightness (loose vs tight trucks)
    skateboard_truck_stiffness = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("skateboard", joint_names=".*"),
            "stiffness_distribution_params": (0.5, 2.0),  # 50% to 200% of base stiffness (25 to 100)
            "damping_distribution_params": (0.5, 2.0),  # 50% to 200% of base damping (2.5 to 10)
            "operation": "scale",
        },
    )

    # reset
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "yaw": (-0.1, 0.1)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.9, 1.1),
            "velocity_range": (0.0, 0.0),
        },
    )
    
    reset_skateboard = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
            "asset_cfg": SceneEntityCfg("skateboard"),
        },
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, noise=Unoise(n_min=-1.5, n_max=1.5))
        last_action = ObsTerm(func=mdp.last_action)
        
        # skateboard-specific observations
        skateboard_relative_pos = ObsTerm(
            func=robot_skateboard_relative_position,
            noise=Unoise(n_min=-0.02, n_max=0.02),
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "skateboard_cfg": SceneEntityCfg("skateboard"),
            },
        )
        skateboard_xy_distance = ObsTerm(
            func=robot_skateboard_xy_distance,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "skateboard_cfg": SceneEntityCfg("skateboard"),
            },
        )

        def __post_init__(self):
            self.history_length = 5
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
        last_action = ObsTerm(func=mdp.last_action)
        
        # skateboard-specific observations (privileged)
        skateboard_relative_pos = ObsTerm(
            func=robot_skateboard_relative_position,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "skateboard_cfg": SceneEntityCfg("skateboard"),
            },
        )
        skateboard_xy_distance = ObsTerm(
            func=robot_skateboard_xy_distance,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "skateboard_cfg": SceneEntityCfg("skateboard"),
            },
        )
        feet_skateboard_height = ObsTerm(
            func=feet_skateboard_relative_height,
            params={
                "robot_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
                "skateboard_cfg": SceneEntityCfg("skateboard"),
            },
        )

        def __post_init__(self):
            self.history_length = 5

    # privileged observations
    critic: CriticCfg = CriticCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP - focused on stationary skateboard balancing."""

    # -- primary task: stay balanced on skateboard
    stay_upright = RewTerm(
        func=skateboard_orientation,
        weight=5.0,  # Reward for keeping skateboard level/flat
        params={"skateboard_cfg": SceneEntityCfg("skateboard")},
    )
    # CHANGED: Fixed bug. Now correctly REWARDS robot for being upright.
    robot_upright = RewTerm(
        func=mdp.orientation_l2,  # CHANGED: Was the flawed 'upward' function
        weight=5.0,  # CHANGED: Was -5.0. Now a positive reward.
        params={
            "desired_gravity": [0.0, 0.0, -1.0], # Robot's 'down' should be world 'down'
            "asset_cfg": SceneEntityCfg("robot")
        },
    )
    base_height = RewTerm(
        func=mdp.base_height_l2, 
        weight=-5.0,  # Penalize crouching.
        params={"target_height": 0.90}
    )
    time_survived = RewTerm(func=time_survived, weight=20.0)  # CRITICAL: 15,000 pts for full 20s - rewards long balance time!

    # -- penalties for movement (we want robot to stay still on stationary skateboard)
    base_linear_velocity_z = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    # Use joint_vel_l2 as a proxy for base XY velocity penalty (standard function doesn't exist)
    base_angular_velocity = RewTerm(
        func=mdp.ang_vel_xy_l2, 
        weight=-2.0 # Penalize "tipping" more.
    )

    # -- smooth motion (anti-jitter penalties)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.05)  # Increased 50x: discourage fast joint movements
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-5)  # Increased 100x: heavily penalize jerky motions
    action_rate = RewTerm(
        func=mdp.action_rate_l2, 
        weight=-0.05 # Relax penalty to allow faster corrections.
    )

    # -- joint limits and energy
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-5.0)
    energy = RewTerm(func=mdp.energy, weight=-2e-5)

    # -- keep arms near default (but allow waist movement for balance!)
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,  # "Unlock" arms for balancing.
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_.*_joint",
                    ".*_elbow_joint",
                    ".*_wrist_.*",
                ],
            )
        },
    )
    # REMOVED: joint_deviation_waists - Allow free waist movement for balance!
    # REMOVED: arm_velocity - Allow dynamic arm movements for balance!

    # -- feet contact (should be on skateboard)
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "threshold": 1,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["(?!.*ankle.*).*"]),
        },
    )
    
    # -- feet contact rewards
    feet_contact_still = RewTerm(
        func=feet_contact_without_cmd,
        weight=5.0,  # Reward feet contact when robot is stationary
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
            "command_name": "base_velocity",
        },
    )
    feet_stumble_penalty = RewTerm(
        func=feet_stumble,
        weight=-5.0,  # Penalize feet hitting vertical surfaces (stumbling)
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
        },
    )

    # -- skateboard-specific rewards
    robot_on_skateboard = RewTerm(
        func=robot_skateboard_alignment,
        weight=10.0,  # Increased from 5.0
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "skateboard_cfg": SceneEntityCfg("skateboard"),
        },
    )
    feet_on_skateboard = RewTerm(
        func=feet_on_skateboard,
        weight=5.0,  # Increased from 2.0
        params={
            "robot_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
            "skateboard_cfg": SceneEntityCfg("skateboard"),
        },
    )
    feet_centerline = RewTerm(
        func=feet_off_centerline_penalty,
        weight=-15.0,  # CHANGED: Was 5.0. This is now a strong PENALTY.
        params={
            "robot_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
            "skateboard_cfg": SceneEntityCfg("skateboard"),
            "max_distance": 0.04,  # 4cm tolerance - exponential decay scale
        },
    )
    com_projection = RewTerm(
        func=com_within_support_polygon,
        weight=15.0,  # Critical for balance stability (CoM between feet)
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "feet_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
            "margin": 0.05,  # 5cm margin inside support polygon
        },
    )
    com_over_board = RewTerm(
        func=com_over_skateboard,
        weight=10.0,  # CRITICAL: CoM must be projected onto skateboard surface!
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "skateboard_cfg": SceneEntityCfg("skateboard"),
            "margin": 0.15,  # 15cm tolerance from skateboard center
        },
    )
    
    # REMOVED: feet_off_skateboard_penalty - ground contact termination is strict enough


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # base_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.4})
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 1.5})
    
    # TERMINATE IMMEDIATELY if robot feet touch the ground!
    # Combines contact detection + height check to differentiate skateboard vs ground
    # Feet on skateboard (~0.08-0.12m) = OK, feet below 0.08m with contact = TERMINATE
    body_ground_collision = DoneTerm(
        func=robot_ground_contact,
        params={
            "robot_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
            "threshold": 1.0,  # 1N force threshold for contact detection
            "ground_height": 0.04,  # Below 0.08m = ground (skateboard top is ~0.08m)
        },
    )


@configclass
class SkateboardEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the stationary skateboard balancing environment."""

    # Scene settings
    scene: SkateboardSceneCfg = SkateboardSceneCfg(num_envs=4096, env_spacing=3.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2  # Reduced from 4 for faster simulation
        self.episode_length_s = 5.0  # Reduced from 10.0 for quicker episodes
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        # update sensor update periods
        self.scene.contact_forces.update_period = self.sim.dt


@configclass
class SkateboardPlayEnvCfg(SkateboardEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.episode_length_s = 30.0

