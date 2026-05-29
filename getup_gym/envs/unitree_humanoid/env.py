import isaacgym
import torch
import numpy as np
from collections import deque

from isaacgym import gymtorch, gymapi
from isaacgym.torch_utils import (
    quat_rotate_inverse,
    to_torch,
    get_axis_params,
    quat_apply,
)
from getup_gym.common.math_utils import torch_rand_float

from getup_gym import GETUP_GYM_ROOT_DIR
from getup_gym.envs.legged_robot import LeggedRobot, get_euler_xyz_tensor
from getup_gym.envs.unitree_humanoid.config import TianGongGetUpCfg
from getup_gym.common.rewards_init import reward_register
from getup_gym.common.helpers import class_to_dict
from getup_gym.common.math_utils import wrap_to_pi, quat_apply_yaw


class UnitreeGetUp(LeggedRobot):
    def __init__(self, cfg: TianGongGetUpCfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.unactuated_time = self.cfg.env.unactuated_timesteps * 0.02 / self.dt

    # ------------------------------------------------------------------
    #  Config parsing
    # ------------------------------------------------------------------
    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        self.myself_common_step_counter = 0

        if hasattr(self.cfg.rewards, "curriculum") and hasattr(self.cfg.rewards.curriculum, "reward_group"):
            self.reward_group_list = {}
            for group_name in self.cfg.rewards.curriculum.reward_group:
                group_cfg = getattr(self.cfg.rewards, group_name, None)
                if group_cfg is not None:
                    self.reward_group_list[group_name] = class_to_dict(group_cfg)

        if self.cfg.terrain.mesh_type not in ["heightfield", "trimesh"]:
            self.cfg.terrain.curriculum = False
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)
        # push_interval is managed dynamically in _post_physics_step_callback

    # ------------------------------------------------------------------
    #  Buffers
    # ------------------------------------------------------------------
    def _init_buffers(self):
        # gait settings
        self.gait_phase = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False)
        self.gait_cycle = torch.full(
            (self.num_envs,), self.cfg.GaitCfg.gait_cycle, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.phase_ratio = torch.tensor(
            [self.cfg.GaitCfg.gait_air_ratio_l, self.cfg.GaitCfg.gait_air_ratio_r],
            dtype=torch.float,
            device=self.device,
        ).repeat(self.num_envs, 1)
        self.phase_offset = torch.tensor(
            [self.cfg.GaitCfg.gait_phase_offset_l, self.cfg.GaitCfg.gait_phase_offset_r],
            dtype=torch.float,
            device=self.device,
        ).repeat(self.num_envs, 1)

        self.velocity_max_now = 0
        self.real_episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]
        self.dip_angle_y = torch.arcsin(
            2 * (self.base_quat[:, 1] * self.base_quat[:, 3] - self.base_quat[:, 0] * self.base_quat[:, 2])
        )

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3)

        self.common_step_counter = 0
        self.has_counter = True
        self.extras = {}

        if self.cfg.rewards.curriculum.using_pull_up:
            self.force_scale = torch.ones(self.num_envs, device=self.device) * 100
            self.force_torque = torch.zeros(self.num_envs, device=self.device)
        else:
            self.force_scale = torch.zeros(self.num_envs, device=self.device)
            self.force_torque = torch.zeros(self.num_envs, device=self.device)

        self.gravity_vec = to_torch(get_axis_params(-1.0, self.up_axis_idx), device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.forward_vec = to_torch([1.0, 0.0, 0.0], device=self.device).repeat((self.num_envs, 1))

        self.torques = torch.zeros(
            self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.actions = torch.zeros(
            self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.last_actions = torch.zeros(
            self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_dof_pos = torch.zeros_like(self.dof_pos)
        self.last_last_dof_pos = torch.zeros_like(self.dof_pos)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.commands = torch.zeros(
            self.num_envs,
            self.cfg.commands.num_commands,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.commands_scale = torch.tensor(
            [
                self.obs_scales.lin_vel,
                self.obs_scales.lin_vel,
                self.obs_scales.ang_vel,
                self.obs_scales.heading,
                self.obs_scales.base_height,
            ],
            device=self.device,
            requires_grad=False,
        )
        self.feet_air_time = torch.zeros(
            self.num_envs,
            self.feet_indices.shape[0],
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.last_contacts = torch.zeros(
            self.num_envs,
            len(self.feet_indices),
            dtype=torch.bool,
            device=self.device,
            requires_grad=False,
        )

        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
        self.measured_heights = 0

        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle

        self.kp = torch.zeros(self.num_envs, self.num_dof, device=self.device, requires_grad=False, dtype=torch.float32)
        self.kd = torch.zeros(self.num_envs, self.num_dof, device=self.device, requires_grad=False, dtype=torch.float32)
        self.action_scale = torch.zeros(self.num_dof, device=self.device, requires_grad=False, dtype=torch.float32)

        for dof_name, value in self.cfg.control.stiffness.items():
            joint_idx = torch.tensor(self.find_joint_id(dof_name), device=self.device)
            self.kp[:, joint_idx] = value
            self.kd[:, joint_idx] = self.cfg.control.damping[dof_name]
            self.action_scale[joint_idx] = self.cfg.control.action_scale[dof_name]

        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)
        self.kp_origin = self.kp.clone()
        self.kd_origin = self.kd.clone()

        self.compute_observations_without_noise()
        num_obs = self.obs_buf.shape[1]
        self.last_observations = torch.zeros(self.num_envs, num_obs, device=self.device)

        self.observations_stack = deque(maxlen=self.cfg.env.myself_setting.add_time_number)
        for i in range(self.cfg.env.myself_setting.add_time_number):
            self.observations_stack.append(torch.zeros(self.num_envs, num_obs, device=self.device))

        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.last_last_action = torch.zeros_like(self.actions)
        self.last_base_velocity = torch.zeros_like(self.base_lin_vel)
        self.last_last_base_velocity = torch.zeros_like(self.base_lin_vel)
        self.last_base_ang_velocity = torch.zeros_like(self.base_ang_vel)
        self.last_last_base_ang_velocity = torch.zeros_like(self.base_ang_vel)
        self.feet_id = torch.randn(self.num_envs, device=self.device, requires_grad=False)

        # re-acquire tensors (as in original TianGong)
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)

        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]
        self.base_euler_xyz = get_euler_xyz_tensor(self.base_quat)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state).view(self.num_envs, self.num_bodies, -1)
        self.feet_state = self.rigid_body_states[:, self.feet_indices, :]

        self.counter_l = None
        self.counter_r = None
        self.body_states_reshape = None
        self.feet_distance_z = None
        self.feet_distance_x = None
        self.feet_distance_y = None

        self.contacts_x = None
        self.contacts_z = None
        self.single_contact_x = None
        self.two_contact_x = None
        self.single_contact_z = None
        self.two_contact_z = None
        self.dof_pos_save = torch.zeros_like(self.dof_pos)
        self.dof_vel_save = torch.zeros_like(self.dof_vel)
        self.target_dof_pos = torch.zeros_like(self.dof_pos)

        # reward group init
        self.reward_group_list = {}
        for group_name in self.cfg.rewards.curriculum.reward_group:
            self.reward_group_list[group_name] = class_to_dict(getattr(self.cfg.rewards, group_name))

        # push rigid indices
        push_rigid_name = self.cfg.domain_rand.push_rigid_name
        push_direction = self.cfg.domain_rand.push_direction
        if len(push_rigid_name) != len(push_direction):
            raise AttributeError("rigid number is not equal to numbers in direction, please check it!")
        direction_map = {"x": 0, "y": 1, "z": 2}
        self.push_direction_to_id = [
            [direction_map[direction] for direction in direction_list] for direction_list in push_direction
        ]
        self.push_rigid_indices = {}
        for name in push_rigid_name:
            self.push_rigid_indices[name] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], name
            )

        # motor strength randomization
        if self.cfg.domain_rand.randomize_motor_strength:
            motor_rigid_name = self.cfg.domain_rand.joint_name_need_strength_rand
            self.motor_rigid_idx = self.find_joint_id(motor_rigid_name)
            self.motor_strength = torch_rand_float(
                self.cfg.domain_rand.motor_strength_range[0],
                self.cfg.domain_rand.motor_strength_range[1],
                (self.num_envs, len(self.motor_rigid_idx)),
                device=self.device,
            )

        self.height_targets = torch.zeros(self.num_envs, 1, device=self.device)
        self.time_counter = torch.zeros(self.num_envs, device=self.device)
        self.up_time = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)

        self.avg_feet_force_per_step = torch.zeros(
            self.num_envs, len(self.feet_indices), device=self.device, requires_grad=False
        )
        self.avg_feet_speed_per_step = torch.zeros(
            self.num_envs, len(self.feet_indices), device=self.device, requires_grad=False
        )

        if self.cfg.domain_rand.push_robots:
            min_steps = int(self.cfg.domain_rand.push_interval_min_s / self.dt)
            max_steps = int(self.cfg.domain_rand.push_interval_max_s / self.dt)
            self.next_push_step = torch.randint(
                min_steps,
                max_steps + 1,
                (self.num_envs,),
                device=self.device,
            )

    # ------------------------------------------------------------------
    #  Env creation (extra body / joint indices)
    # ------------------------------------------------------------------
    def _create_envs(self):
        super()._create_envs()
        body_names = self.body_names

        left_foot_names = [
            s for s in body_names
            if self.cfg.asset.left_foot_name in s and "keyframe" not in s and "auxiliary" not in s
        ]
        right_foot_names = [
            s for s in body_names
            if self.cfg.asset.right_foot_name in s and "keyframe" not in s and "auxiliary" not in s
        ]
        self.left_foot_indices = torch.zeros(len(left_foot_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(left_foot_names)):
            self.left_foot_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], left_foot_names[i]
            )
        self.right_foot_indices = torch.zeros(len(right_foot_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(right_foot_names)):
            self.right_foot_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], right_foot_names[i]
            )

        self.left_foot_body_indices = torch.zeros(
            len(self.cfg.asset.left_foot_body_name), dtype=torch.long, device=self.device, requires_grad=False
        )
        for i in range(len(self.cfg.asset.left_foot_body_name)):
            self.left_foot_body_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], self.cfg.asset.left_foot_body_name[i]
            )
        self.right_foot_body_indices = torch.zeros(
            len(self.cfg.asset.right_foot_body_name), dtype=torch.long, device=self.device, requires_grad=False
        )
        for i in range(len(self.cfg.asset.right_foot_body_name)):
            self.right_foot_body_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], self.cfg.asset.right_foot_body_name[i]
            )

        base_name = [s for s in body_names if self.cfg.asset.base_name in s and "keyframe" not in s]
        self.base_indices = torch.zeros(len(base_name), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(base_name)):
            self.base_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], base_name[i]
            )

        pelvis_name = ["pelvis"]
        self.pelvis_indices = torch.zeros(len(pelvis_name), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(pelvis_name)):
            self.pelvis_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], pelvis_name[i]
            )

        left_knee_names = [s for s in body_names if self.cfg.asset.left_knee_name in s and "keyframe" not in s]
        right_knee_names = [s for s in body_names if self.cfg.asset.right_knee_name in s and "keyframe" not in s]
        self.left_knee_indices = torch.zeros(len(left_knee_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(left_knee_names)):
            self.left_knee_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], left_knee_names[i]
            )
        self.right_knee_indices = torch.zeros(len(right_knee_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(right_knee_names)):
            self.right_knee_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], right_knee_names[i]
            )

        self.knee_joint_indices = torch.zeros(len(self.cfg.asset.knee_joints), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.cfg.asset.knee_joints)):
            self.knee_joint_indices[i] = self.dof_names.index(self.cfg.asset.knee_joints[i])

        self.ankle_joint_indices = torch.zeros(len(self.cfg.asset.ankle_joints), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.cfg.asset.ankle_joints)):
            self.ankle_joint_indices[i] = self.dof_names.index(self.cfg.asset.ankle_joints[i])

        self.waist_joint_indices = torch.zeros(len(self.cfg.asset.waist_joints), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.cfg.asset.waist_joints)):
            self.waist_joint_indices[i] = self.dof_names.index(self.cfg.asset.waist_joints[i])

        self.hands_indices = torch.zeros(len(self.cfg.asset.hands_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.cfg.asset.hands_names)):
            self.hands_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], self.cfg.asset.hands_names[i]
            )

        self.left_hip_joint_indices = torch.zeros(
            len(self.cfg.asset.left_hip_joints), dtype=torch.long, device=self.device, requires_grad=False
        )
        for i in range(len(self.cfg.asset.left_hip_joints)):
            self.left_hip_joint_indices[i] = self.dof_names.index(self.cfg.asset.left_hip_joints[i])
        self.right_hip_joint_indices = torch.zeros(
            len(self.cfg.asset.right_hip_joints), dtype=torch.long, device=self.device, requires_grad=False
        )
        for i in range(len(self.cfg.asset.right_hip_joints)):
            self.right_hip_joint_indices[i] = self.dof_names.index(self.cfg.asset.right_hip_joints[i])
        self.hip_joint_indices = torch.cat((self.left_hip_joint_indices, self.right_hip_joint_indices))

        self.left_hip_roll_joint_indices = torch.zeros(
            len(self.cfg.asset.left_hip_roll_joints), dtype=torch.long, device=self.device, requires_grad=False
        )
        for i in range(len(self.cfg.asset.left_hip_roll_joints)):
            self.left_hip_roll_joint_indices[i] = self.dof_names.index(self.cfg.asset.left_hip_roll_joints[i])
        self.right_hip_roll_joint_indices = torch.zeros(
            len(self.cfg.asset.right_hip_roll_joints), dtype=torch.long, device=self.device, requires_grad=False
        )
        for i in range(len(self.cfg.asset.right_hip_roll_joints)):
            self.right_hip_roll_joint_indices[i] = self.dof_names.index(self.cfg.asset.right_hip_roll_joints[i])
        self.hip_roll_joint_indices = torch.cat((self.left_hip_roll_joint_indices, self.right_hip_roll_joint_indices))

        self.left_knee_joint_indices = torch.zeros(
            len(self.cfg.asset.left_knee_joints), dtype=torch.long, device=self.device, requires_grad=False
        )
        for i in range(len(self.cfg.asset.left_knee_joints)):
            self.left_knee_joint_indices[i] = self.dof_names.index(self.cfg.asset.left_knee_joints[i])
        self.right_knee_joint_indices = torch.zeros(
            len(self.cfg.asset.right_knee_joints), dtype=torch.long, device=self.device, requires_grad=False
        )
        for i in range(len(self.cfg.asset.right_knee_joints)):
            self.right_knee_joint_indices[i] = self.dof_names.index(self.cfg.asset.right_knee_joints[i])

        self.left_hip_pitch_joint_indices = torch.zeros(
            len(self.cfg.asset.left_hip_pitch_joints), dtype=torch.long, device=self.device, requires_grad=False
        )
        for i in range(len(self.cfg.asset.left_hip_pitch_joints)):
            self.left_hip_pitch_joint_indices[i] = self.dof_names.index(self.cfg.asset.left_hip_pitch_joints[i])
        self.right_hip_pitch_joint_indices = torch.zeros(
            len(self.cfg.asset.right_hip_pitch_joints), dtype=torch.long, device=self.device, requires_grad=False
        )
        for i in range(len(self.cfg.asset.right_hip_pitch_joints)):
            self.right_hip_pitch_joint_indices[i] = self.dof_names.index(self.cfg.asset.right_hip_pitch_joints[i])
        self.hip_pitch_joint_indices = torch.cat(
            (self.left_hip_pitch_joint_indices, self.right_hip_pitch_joint_indices)
        )

        self.all_hip_joint_indices = torch.cat(
            [self.hip_pitch_joint_indices, self.hip_roll_joint_indices, self.hip_joint_indices]
        )

        self.left_shoulder_roll_joint_indices = torch.zeros(
            len(self.cfg.asset.left_shoulder_roll_joints), dtype=torch.long, device=self.device, requires_grad=False
        )
        for i in range(len(self.cfg.asset.left_shoulder_roll_joints)):
            self.left_shoulder_roll_joint_indices[i] = self.dof_names.index(self.cfg.asset.left_shoulder_roll_joints[i])
        self.right_shoulder_roll_joint_indices = torch.zeros(
            len(self.cfg.asset.right_shoulder_roll_joints), dtype=torch.long, device=self.device, requires_grad=False
        )
        for i in range(len(self.cfg.asset.right_shoulder_roll_joints)):
            self.right_shoulder_roll_joint_indices[i] = self.dof_names.index(self.cfg.asset.right_shoulder_roll_joints[i])
        self.shoulder_roll_joint_indices = torch.cat(
            (self.left_shoulder_roll_joint_indices, self.right_shoulder_roll_joint_indices)
        )

        self.left_arm_joint_indices = torch.zeros(
            len(self.cfg.asset.left_arm_joints), dtype=torch.long, device=self.device, requires_grad=False
        )
        for i in range(len(self.cfg.asset.left_arm_joints)):
            self.left_arm_joint_indices[i] = self.dof_names.index(self.cfg.asset.left_arm_joints[i])
        self.right_arm_joint_indices = torch.zeros(
            len(self.cfg.asset.right_arm_joints), dtype=torch.long, device=self.device, requires_grad=False
        )
        for i in range(len(self.cfg.asset.right_arm_joints)):
            self.right_arm_joint_indices[i] = self.dof_names.index(self.cfg.asset.right_arm_joints[i])

        self.left_leg_indices = torch.cat([self.left_hip_joint_indices, self.left_knee_joint_indices])
        self.right_leg_indices = torch.cat([self.right_hip_joint_indices, self.right_knee_joint_indices])
        self.upper_body_joint_indices = torch.cat(
            [self.right_arm_joint_indices, self.left_arm_joint_indices, self.waist_joint_indices]
        )
        self.lower_body_joint_indices = torch.cat(
            [self.all_hip_joint_indices, self.knee_joint_indices, self.ankle_joint_indices]
        )

        left_upper_body_names = []
        for target_name in self.cfg.asset.left_upper_body_names:
            for source_name in body_names:
                if target_name in source_name and "keyframe" not in source_name and "aux" not in source_name:
                    left_upper_body_names.append(source_name)
        self.left_upper_body_indices = torch.zeros(len(left_upper_body_names), dtype=torch.long, device=self.device)
        self.left_upper_body_names = left_upper_body_names
        for i, name in enumerate(left_upper_body_names):
            self.left_upper_body_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], name
            )

        right_upper_body_names = []
        for target_name in self.cfg.asset.right_upper_body_names:
            for source_name in body_names:
                if target_name in source_name and "keyframe" not in source_name and "aux" not in source_name:
                    right_upper_body_names.append(source_name)
        self.right_upper_body_indices = torch.zeros(len(right_upper_body_names), dtype=torch.long, device=self.device)
        self.right_upper_body_names = right_upper_body_names
        for i, name in enumerate(right_upper_body_names):
            self.right_upper_body_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], name
            )

        left_lower_body_names = []
        for target_name in self.cfg.asset.left_lower_body_names:
            for source_name in body_names:
                if target_name in source_name and "keyframe" not in source_name and "aux" not in source_name:
                    left_lower_body_names.append(source_name)
        self.left_lower_body_indices = torch.zeros(len(left_lower_body_names), dtype=torch.long, device=self.device)
        self.left_lower_body_names = left_lower_body_names
        for i, name in enumerate(left_lower_body_names):
            self.left_lower_body_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], name
            )

        right_lower_body_names = []
        for target_name in self.cfg.asset.right_lower_body_names:
            for source_name in body_names:
                if target_name in source_name and "keyframe" not in source_name and "aux" not in source_name:
                    right_lower_body_names.append(source_name)
        self.right_lower_body_indices = torch.zeros(len(right_lower_body_names), dtype=torch.long, device=self.device)
        self.right_lower_body_names = right_lower_body_names
        for i, name in enumerate(right_lower_body_names):
            self.right_lower_body_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], name
            )

        left_ankle_names = []
        for target_name in self.cfg.asset.left_ankle_names:
            for source_name in body_names:
                if target_name in source_name and "keyframe" not in source_name:
                    left_ankle_names.append(source_name)
        self.left_ankle_indices = torch.zeros(len(left_ankle_names), dtype=torch.long, device=self.device)
        self.left_ankle_names = left_ankle_names
        for i, name in enumerate(left_ankle_names):
            self.left_ankle_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], name
            )

        right_ankle_names = []
        for target_name in self.cfg.asset.right_ankle_names:
            for source_name in body_names:
                if target_name in source_name and "keyframe" not in source_name:
                    right_ankle_names.append(source_name)
        self.right_ankle_indices = torch.zeros(len(right_ankle_names), dtype=torch.long, device=self.device)
        self.right_ankle_names = right_ankle_names
        for i, name in enumerate(right_ankle_names):
            self.right_ankle_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], name
            )

    # ------------------------------------------------------------------
    #  Step
    # ------------------------------------------------------------------
    def step(self, actions):
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        self.avg_feet_force_per_step = torch.zeros_like(self.avg_feet_force_per_step)
        self.avg_feet_speed_per_step = torch.zeros_like(self.avg_feet_speed_per_step)
        self.render()
        for _ in range(self.cfg.control.decimation):
            if (
                self.cfg.rewards.curriculum.using_pull_up
                and ((self.common_step_counter - 1) / 24 - 1 > 1)
                and (
                    (self.common_step_counter - 1) / 24 - 1
                    < self.cfg.rewards.curriculum.pull_up_end_epochs
                )
            ):
                self._base_force_pull_up()
            self.actions *= self.real_episode_length_buf.unsqueeze(1) > self.unactuated_time
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.device == "cpu":
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)

            self.avg_feet_force_per_step += torch.norm(
                self.contact_forces[:, self.feet_indices, :], dim=-1
            )
            self.avg_feet_speed_per_step += torch.norm(
                self.rigid_body_states[:, self.feet_indices, 7:10], dim=-1
            )
        self.avg_feet_force_per_step = self.avg_feet_force_per_step / self.cfg.control.decimation
        self.avg_feet_speed_per_step = self.avg_feet_speed_per_step / self.cfg.control.decimation
        self.post_physics_step()

        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        self.teacher_encoder_obs_buf = torch.clip(self.teacher_encoder_obs_buf, -clip_obs, clip_obs)
        self.student_encoder_obs_buf = torch.clip(self.student_encoder_obs_buf, -clip_obs, clip_obs)

        return (
            self.obs_buf,
            self.privileged_obs_buf,
            self.teacher_encoder_obs_buf,
            self.student_encoder_obs_buf,
            self.rew_buf,
            self.reset_buf,
            self.extras,
        )

    # ------------------------------------------------------------------
    #  Observations
    # ------------------------------------------------------------------
    def compute_observations_without_noise(self):
        commands = self.commands[:, :3] * self.commands_scale[:3]
        if self.cfg.commands.heading_command:
            commands = self.commands[:, :5] * self.commands_scale[:5]
        if self.cfg.commands.base_height_command:
            base_height_dim = self.cfg.commands.num_commands - 1
            commands = self.commands[:, :base_height_dim] * self.commands_scale[:base_height_dim]

        self.obs_buf = torch.cat(
            (
                self.base_lin_vel * self.obs_scales.lin_vel * 0,
                self.base_ang_vel * self.obs_scales.ang_vel,
                self.projected_gravity,
                torch.zeros_like(commands),
                self.dof_pos * self.obs_scales.dof_pos,
                self.dof_vel * self.obs_scales.dof_vel,
                self.actions,
                torch.zeros_like(torch.sin(2 * torch.pi * self.gait_phase)),  # 2
                torch.zeros_like(torch.cos(2 * torch.pi * self.gait_phase)),  # 2
                torch.zeros_like(self.phase_ratio),  # 2
            ),
            dim=-1,
        )

    def compute_observations(self):
        self.compute_observations_without_noise()
        self.compute_privileged_observations()
        self.compute_teacher_encoder_observations()
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec
        self.obs_buf *= self.real_episode_length_buf.unsqueeze(1) > self.unactuated_time
        self.last_observations_save()
        self.compute_student_encoder_observations()
        return self.obs_buf

    def compute_privileged_observations(self):
        base_height = torch.mean(
            self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1
        ).unsqueeze(1)
        if self.cfg.terrain.measure_heights:
            heights = (
                torch.clip(
                    self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights,
                    -1.0,
                    1.0,
                )
                * self.obs_scales.height_measurements
            )
            self.privileged_obs_buf = torch.cat(
                (
                    self.obs_buf,
                    heights,
                    # base_height,
                ),
                dim=1,
            )
            self.privileged_obs_buf[:, 0:3] = self.base_lin_vel * self.obs_scales.lin_vel

    def compute_teacher_encoder_observations(self):
        if self.cfg.terrain.measure_heights:
            heights = (
                torch.clip(
                    self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights,
                    -1.0,
                    1.0,
                )
                * self.obs_scales.height_measurements
            )

        ContactForce_lwheel = self.contact_forces[:, self.feet_indices[0], :3].squeeze(1)  # 3
        ContactForce_rwheel = self.contact_forces[:, self.feet_indices[1], :3].squeeze(1)  # 3

        base_height = torch.mean(
            self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1
        ).unsqueeze(1)

        teacher_encoder_observations = torch.cat(
            [
                ContactForce_lwheel,
                ContactForce_rwheel,
                base_height * 0.4,
                self.root_states,
            ],
            dim=-1,
        )
        self.teacher_encoder_obs_buf = teacher_encoder_observations
        self.teacher_encoder_obs_buf *= self.real_episode_length_buf.unsqueeze(1) > self.unactuated_time

    def compute_student_encoder_observations(self):
        self.lotstime_obs = self.observations_stack[0]
        for i in range(self.cfg.env.myself_setting.add_time_number - 1):
            self.lotstime_obs = torch.cat(
                (
                    self.lotstime_obs,
                    self.observations_stack[i + 1],
                ),
                dim=1,
            )
        student_encoder_observations = self.lotstime_obs
        self.student_encoder_obs_buf = student_encoder_observations
        self.student_encoder_obs_buf *= self.real_episode_length_buf.unsqueeze(1) > self.unactuated_time

    # ------------------------------------------------------------------
    #  Reward callbacks
    # ------------------------------------------------------------------
    def _update_base_height_target(self):
        target_height_list = self.cfg.rewards.curriculum.target_height
        base_height = torch.mean(
            torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1), dim=0
        )
        self.target_height = target_height_list[0] * torch.ones(self.num_envs, device=self.device)
        idx = (base_height >= target_height_list[0] * 0.8) & (base_height < target_height_list[1])
        self.target_height[idx] = target_height_list[1]
        idx = (base_height >= target_height_list[1] * 0.8) & (base_height < target_height_list[2])
        self.target_height[idx] = target_height_list[2]
        idx = (base_height >= target_height_list[2] * 0.8) & (base_height < target_height_list[3])
        self.target_height[idx] = target_height_list[3]
        idx = (base_height >= target_height_list[3] * 0.8)
        self.target_height[idx] = self.cfg.rewards.base_height_target

    def _pre_reward_callback(self):
        self._calculate_gait_para()
        self.contacts_z = torch.abs(self.contact_forces[:, self.feet_indices, 2]) > 1.0
        self.contacts_x = torch.abs(self.contact_forces[:, self.feet_indices, 0]) > 1.0
        self.single_contact_z = torch.sum(1.0 * self.contacts_z, dim=1) == 1
        self.two_contact_z = torch.sum(1.0 * self.contacts_z, dim=1) == 2
        self.single_contact_x = torch.sum(1.0 * self.contacts_x, dim=1) == 1
        self.two_contact_x = torch.sum(1.0 * self.contacts_x, dim=1) == 2

    def _calculate_gait_para(self) -> None:
        t = self.episode_length_buf * self.dt / self.gait_cycle
        self.gait_phase[:, 0] = (t + self.phase_offset[:, 0]) % 1.0
        self.gait_phase[:, 1] = (t + self.phase_offset[:, 1]) % 1.0

    def _gait_clock(self, phase, air_ratio, delta_t):
        swing_flag = (phase >= delta_t) & (phase <= (air_ratio - delta_t))
        stand_flag = (phase >= (air_ratio + delta_t)) & (phase <= (1 - delta_t))

        trans_flag1 = phase < delta_t
        trans_flag2 = (phase > (air_ratio - delta_t)) & (phase < (air_ratio + delta_t))
        trans_flag3 = phase > (1 - delta_t)

        I_frc = (
            1.0 * swing_flag
            + (0.5 + phase / (2 * delta_t)) * trans_flag1
            - (phase - air_ratio - delta_t) / (2.0 * delta_t) * trans_flag2
            + 0.0 * stand_flag
            + (phase - 1 + delta_t) / (2 * delta_t) * trans_flag3
        )
        I_spd = 1.0 - I_frc
        return I_frc, I_spd

    def _reward_update(self):
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        stand_idx = base_height < 0.35
        walk_idx = (base_height >= 0.4)
        moving_idx = (self.force_scale < 60.0)

        self.reward_group = self.reward_group_list["reward_group_2"]
        if torch.sum(1.0 * stand_idx) > self.num_envs * 2 / 3:
            self.reward_group = self.reward_group_list["reward_group_2"]
        elif torch.sum(1.0 * walk_idx) > self.num_envs * 2 / 3:
            self.reward_group = self.reward_group_list["reward_group_3"]
            if torch.sum(1.0 * moving_idx) > self.num_envs * 2 / 3:
                self.reward_group = self.reward_group_list["reward_group_4"]

    def _reward_register(self):
        reward_register(self.RewardManager)

    # ------------------------------------------------------------------
    #  Torques
    # ------------------------------------------------------------------
    def _compute_torques(self, actions):
        torques = torch.zeros_like(actions)

        for dof_name, ctype in self.cfg.control.control_type.items():
            joint_idx = torch.tensor(self.find_joint_id(dof_name), device=self.device)
            if ctype == "P":
                torques[:, joint_idx] = self.kp[:, joint_idx] * (
                    actions[:, joint_idx] * self.action_scale[joint_idx]
                    + self.default_dof_pos[:, joint_idx]
                    - self.dof_pos[:, joint_idx]
                ) - self.kd[:, joint_idx] * self.dof_vel[:, joint_idx]
            elif ctype == "V":
                torques[:, joint_idx] = self.kp[:, joint_idx] * (
                    actions[:, joint_idx] * self.action_scale[joint_idx] - self.dof_vel[:, joint_idx]
                ) - self.kd[:, joint_idx] * (
                    self.dof_vel[:, joint_idx] - self.last_dof_vel[:, joint_idx]
                ) / self.sim_params.dt
            elif ctype == "T":
                torques[:, joint_idx] = actions[:, joint_idx] * self.action_scale[joint_idx]
            else:
                raise KeyError(f"control type don't have {ctype}")

        if self.cfg.domain_rand.randomize_motor_strength:
            torques[:, self.motor_rigid_idx] = self.motor_strength[:, :] * torques[:, self.motor_rigid_idx]

        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    # ------------------------------------------------------------------
    #  Reset
    # ------------------------------------------------------------------
    def _reset_dofs(self, env_ids):
        self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(
            0.1, 1.5, (len(env_ids), self.num_dof), device=self.device
        )
        self.dof_vel[env_ids] = 0.0

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

    def _reset_root_states(self, env_ids):
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch_rand_float(
                -1.0, 1.0, (len(env_ids), 2), device=self.device
            )
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]

        self.root_states[env_ids, 7:13] = torch_rand_float(
            -0.5, 0.5, (len(env_ids), 6), device=self.device
        )
        back_idx = torch.where(env_ids < 2000)[0]
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        back_idx = back_idx.to(dtype=torch.int32)

        self.root_states[env_ids[back_idx], 4:5] = 1.0
        self.root_states[env_ids[back_idx], 2:3] += 0.2

        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

    def _reset_pd_params(self, env_ids):
        kp_scale = torch.rand_like(self.kp) * (
            self.cfg.domain_rand.kp_range[1] - self.cfg.domain_rand.kp_range[0]
        ) + self.cfg.domain_rand.kp_range[0]
        kd_scale = torch.rand_like(self.kd) * (
            self.cfg.domain_rand.kd_range[1] - self.cfg.domain_rand.kd_range[0]
        ) + self.cfg.domain_rand.kd_range[0]

        self.kp[env_ids] = self.kp_origin[env_ids] * kp_scale[env_ids]
        self.kd[env_ids] = self.kd_origin[env_ids] * kd_scale[env_ids]

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        if len(env_ids) == 0:
            return
        if self.cfg.domain_rand.randomize_pd:
            self._reset_pd_params(env_ids)
        self.last_dof_pos[env_ids] = 0.0
        self.last_last_dof_pos[env_ids] = 0.0
        self.avg_feet_force_per_step[env_ids] = 0.0
        self.avg_feet_speed_per_step[env_ids] = 0.0

    # ------------------------------------------------------------------
    #  Push robots
    # ------------------------------------------------------------------
    def _push_robots(self, push_ids):
        push_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)
        push_vel[push_ids, 0:1] = torch_rand_float(
            -self.cfg.domain_rand.max_push_vel_x,
            self.cfg.domain_rand.max_push_vel_x,
            (len(push_ids), 1),
            device=self.device,
        )
        push_vel[push_ids, 1:2] = torch_rand_float(
            -self.cfg.domain_rand.max_push_vel_y,
            self.cfg.domain_rand.max_push_vel_y,
            (len(push_ids), 1),
            device=self.device,
        )
        push_vel[push_ids, 2:3] = torch_rand_float(
            -self.cfg.domain_rand.max_push_vel_z,
            self.cfg.domain_rand.max_push_vel_z,
            (len(push_ids), 1),
            device=self.device,
        )

        push_vel_base = quat_apply_yaw(self.base_quat, push_vel)

        self.root_states[push_ids, 7:8] = push_vel_base[push_ids, 0:1]
        self.root_states[push_ids, 8:9] = push_vel_base[push_ids, 1:2]
        self.root_states[push_ids, 9:10] = push_vel_base[push_ids, 2:3]
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    # ------------------------------------------------------------------
    #  Post-physics callback
    # ------------------------------------------------------------------
    def _post_physics_step_callback(self):
        env_ids = (
            (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt) == 0)
            .nonzero(as_tuple=False)
            .flatten()
        )
        self._resample_commands(env_ids)
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(
                0.5 * wrap_to_pi(self.commands[:, 3] - heading), -1.0, 1.0
            )

        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
        self._update_base_height_target()

        if (self.common_step_counter - 1) / 24 - 1 > self.cfg.domain_rand.push_start_epoch:
            if self.cfg.domain_rand.push_robots:
                push_mask = self.common_step_counter >= self.next_push_step
                if push_mask.any():
                    push_ids = push_mask.nonzero(as_tuple=False).flatten()
                    self._push_robots(push_ids)
                    min_steps = int(self.cfg.domain_rand.push_interval_min_s / self.dt)
                    max_steps = int(self.cfg.domain_rand.push_interval_max_s / self.dt)
                    self.next_push_step[push_ids] = self.common_step_counter + torch.randint(
                        min_steps,
                        max_steps + 1,
                        (len(push_ids),),
                        device=self.device,
                    )

        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)

    # ------------------------------------------------------------------
    #  Noise scale
    # ------------------------------------------------------------------
    def _get_noise_scale_vec(self, cfg):
        self.compute_observations_without_noise()
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = 0.0
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[6:9] = noise_scales.gravity * noise_level
        noise_vec[9:12] = 0.0  # commands
        noise_vec[12 : 12 + self.num_actions] = (
            noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        )
        noise_vec[12 + self.num_actions : 12 + 2 * self.num_actions] = (
            noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        )
        noise_vec[12 + 2 * self.num_actions : 12 + 3 * self.num_actions] = 0.0  # previous actions
        noise_vec[12 + 3 * self.num_actions : 12 + 3 * self.num_actions + 6] = 0.0
        if self.cfg.terrain.measure_heights:
            noise_vec[12 + 3 * self.num_actions :] = (
                noise_scales.height_measurements
                * noise_level
                * self.obs_scales.height_measurements
            )
        return noise_vec

    # ------------------------------------------------------------------
    #  Base height target & pull-up
    # ------------------------------------------------------------------
    def _update_base_height_target(self) -> None:
        target_height_list = self.cfg.rewards.curriculum.target_height
        base_height = torch.mean(
            torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1),
            dim=0,
        )

        self.target_height = target_height_list[0] * torch.ones(self.num_envs, device=self.device)
        idx = (base_height >= target_height_list[0] * 0.8) & (base_height < target_height_list[1])
        self.target_height[idx] = target_height_list[1]
        idx = (base_height >= target_height_list[1] * 0.8) & (base_height < target_height_list[2])
        self.target_height[idx] = target_height_list[2]
        idx = (base_height >= target_height_list[2] * 0.8) & (base_height < target_height_list[3])
        self.target_height[idx] = target_height_list[3]
        idx = base_height >= target_height_list[3] * 0.8
        self.target_height[idx] = self.cfg.rewards.base_height_target

    def _base_force_pull_up(self):
        force_max = self.cfg.rewards.curriculum.base_pull_up_max
        target_height_list = self.cfg.rewards.curriculum.target_height
        base_height = torch.mean(
            self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1
        )
        time_coeff = torch.tensor(
            (
                self.cfg.rewards.curriculum.pull_up_end_epochs
                - ((self.common_step_counter - 1) / 24)
            )
            / self.cfg.rewards.curriculum.pull_up_end_epochs,
            device=self.device,
        )
        time_coeff = torch.clamp(time_coeff, 0.0, 1.0)
        if self.cfg.rewards.curriculum.using_pull_up_end is not True:
            time_coeff = torch.ones_like(time_coeff)

        force_scale = (
            (target_height_list[3] - base_height) / (target_height_list[3])
        ) * force_max * time_coeff

        self.force_scale = torch.clamp(force_scale, 0.0, force_max)

        forces = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)
        forces[:, self.base_indices[0], 2] = self.force_scale
        forces = forces * (
            self.real_episode_length_buf.unsqueeze(1) > self.unactuated_time
        ).unsqueeze(1)

        torques = torch.zeros_like(forces)
        rotate_scale = torch.abs(self.root_states[:2000, 4] + 1)
        height_scale = (
            (target_height_list[3] - base_height) / (target_height_list[3])
        )[:2000]
        torques[:2000, self.base_indices[0], 0] = -15.0 * time_coeff * rotate_scale * height_scale
        torques = torques * (
            self.real_episode_length_buf.unsqueeze(1) > self.unactuated_time
        ).unsqueeze(1)

        self.gym.apply_rigid_body_force_tensors(
            self.sim,
            gymtorch.unwrap_tensor(forces),
            gymtorch.unwrap_tensor(torques),
            gymapi.ENV_SPACE,
        )

    # ------------------------------------------------------------------
    #  Overstep termination
    # ------------------------------------------------------------------
    def _overstep_termination(self):
        time_max = self.cfg.rewards.terminations.time_overstep_max * torch.ones(self.num_envs, device=self.device)
        base_height = torch.mean(
            self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1
        )

        indices = torch.cat([self.base_indices, self.left_upper_body_indices, self.right_upper_body_indices])
        body_contact_idx = (
            torch.sum(
                1.0 * (torch.norm(self.contact_forces[:, indices, :], dim=-1) > 4.0),
                dim=1,
            )
            > 0
        )
        across_torque_idx = torch.sum(torch.abs(self.torques), dim=1) > 200
        body_height_idx = (base_height < 0.6) & (self.target_height > 0.7)

        self.time_counter[torch.where(body_contact_idx)] += self.dt * torch.ones_like(
            self.time_counter[torch.where(body_contact_idx)]
        )

        reset_idx_1 = self.time_counter > time_max
        reset_idx_2 = body_height_idx
        reset_mask = reset_idx_1 & reset_idx_2

        reset_idx = torch.where(reset_mask)
        if len(reset_idx[0]) > 0:
            self.reset_buf[reset_idx] = 1
            self.time_counter[reset_idx] = 0.0
