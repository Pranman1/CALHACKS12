# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for skateboard object."""

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.utils import configclass

# Path to skateboard URDF
SKATEBOARD_URDF_PATH = "unitree_rl_lab/assets/urdf/skateboard/skateboard.urdf"


@configclass
class SkateboardCfg(RigidObjectCfg):
    """Configuration for a skateboard as a rigid object."""

    spawn = sim_utils.UrdfFileCfg(
        asset_path=SKATEBOARD_URDF_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=1.0,
            enable_gyroscopic_forces=True,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        # Make it a fixed base for now (stationary)
        fix_base=True,
    )

    init_state = RigidObjectCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.12),  # Slightly above ground (adjust based on your skateboard height)
        rot=(1.0, 0.0, 0.0, 0.0),  # Flat orientation (w, x, y, z quaternion)
        lin_vel=(0.0, 0.0, 0.0),  # Stationary
        ang_vel=(0.0, 0.0, 0.0),  # No rotation
    )


# Instantiate the configuration
SKATEBOARD_CFG = SkateboardCfg(prim_path="{ENV_REGEX_NS}/Skateboard")

