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
    com_within_support_polygon,
)
from unitree_rl_lab.tasks.locomotion.mdp.terminations import (
    robot_off_skateboard,
    feet_off_skateboard,
    skateboard_tilted,
)

# Define skateboard configuration using URDF
import os
# This file is at: unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/29dof/skate_env_cfg.py
# Need to go up to unitree_rl_lab root, then into assets
SKATEBOARD_URDF_PATH = os.path.join(
    os.path.dirname(__file__), 
    "..", "..", "..", "..", "..", "..", "..", "..", "assets", "robots", "skateboard_description", "urdf", "skateboard.urdf"
)
SKATEBOARD_URDF_PATH = os.path.abspath(SKATEBOARD_URDF_PATH)

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
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            drive_type="force",
            target_type="position",
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=0.0,
                damping=0.0,
            ),
        ),
        fix_base=False,  # Allow skateboard to move
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.05),  # Slightly above ground
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
                "left_hip_pitch_joint": -0.1,
                "right_hip_pitch_joint": -0.1,
                ".*_knee_joint": 0.3,
                ".*_ankle_pitch_joint": -0.2,
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
    stay_upright = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)  # Reduced from -10.0
    base_height = RewTerm(func=mdp.base_height_l2, weight=-2.5, params={"target_height": 0.90})  # Reduced from -5.0
    alive = RewTerm(func=mdp.is_alive, weight=10.0)  # Increased from 5.0

    # -- penalties for movement (we want robot to stay still on stationary skateboard)
    base_linear_velocity_z = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    # Use joint_vel_l2 as a proxy for base XY velocity penalty (standard function doesn't exist)
    base_angular_velocity = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.5)

    # -- smooth motion
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.001)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)

    # -- joint limits and energy
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-5.0)
    energy = RewTerm(func=mdp.energy, weight=-2e-5)

    # -- keep arms and waist near default
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
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
    joint_deviation_waists = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "waist.*",
                ],
            )
        },
    )

    # -- feet contact (should be on skateboard)
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "threshold": 1,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["(?!.*ankle.*).*"]),
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
        func=feet_near_skateboard_centerline,
        weight=20.0,  # Strong reward for proper foot placement
        params={
            "robot_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
            "skateboard_cfg": SceneEntityCfg("skateboard"),
            "max_distance": 0.15,  # 15cm tolerance from centerline
        },
    )
    com_projection = RewTerm(
        func=com_within_support_polygon,
        weight=15.0,  # Critical for balance stability
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "feet_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
            "margin": 0.05,  # 5cm margin inside support polygon
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.4})
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})
    
    # skateboard-specific constraints
    robot_off_skateboard = DoneTerm(
        func=robot_off_skateboard,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "skateboard_cfg": SceneEntityCfg("skateboard"),
            "max_distance": 0.4,  # 40cm horizontal distance limit
        },
    )
    feet_off_skateboard = DoneTerm(
        func=feet_off_skateboard,
        params={
            "robot_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
            "skateboard_cfg": SceneEntityCfg("skateboard"),
            "max_height_diff": 0.3,  # 30cm vertical limit
            "max_xy_distance": 0.6,  # 60cm horizontal limit for feet
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
        self.decimation = 4
        self.episode_length_s = 20.0
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

