from getup_gym import GETUP_GYM_ROOT_DIR, envs
from time import time
from warnings import WarningMessage
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil, gymutil

import torch
from torch import Tensor
from typing import Tuple, Dict

from getup_gym import GETUP_GYM_ROOT_DIR
from getup_gym.common.terrain import Terrain
from getup_gym.common.math_utils import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float, torch_rand_float
from getup_gym.common.helpers import class_to_dict
from getup_gym.common.base_task import BaseTask
from getup_gym.common.reward_manager import RewardManager
from isaacgym.terrain_utils import *
from collections import deque
from abc import abstractmethod


def get_euler_xyz_tensor(quat):
    r, p, w = get_euler_xyz(quat)
    euler_xyz = torch.stack((r, p, w), dim=1)
    euler_xyz[euler_xyz > np.pi] -= 2 * np.pi
    return euler_xyz


class LeggedRobot(BaseTask):

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = False
        self.init_done = False
        self._parse_cfg(self.cfg)
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)

        if self.cfg.terrain.mesh_type == "hhd_trimesh":
            self.hhd_terrain_init()
        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        if hasattr(self, 'reward_group_list') and self.reward_group_list:
            self._init_reward_manager()
        else:
            self._prepare_reward_function()
        self.init_done = True

    def step(self, actions):
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
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

            if hasattr(self, "unactuated_time"):
                self.actions *= (
                    self.real_episode_length_buf.unsqueeze(1) > self.unactuated_time
                )

            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(
                self.sim, gymtorch.unwrap_tensor(self.torques)
            )
            self.gym.simulate(self.sim)
            if self.device == "cpu":
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)

            if (
                (self.common_step_counter - 1) / 24 - 1
                > self.cfg.domain_rand.force_push_start_epoch
            ):
                if self.cfg.domain_rand.randomize_force_push and (
                    self.common_step_counter % self.cfg.domain_rand.push_interval == 0
                ):
                    self.force_push()
        self.post_physics_step()

        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(
                self.privileged_obs_buf, -clip_obs, clip_obs
            )
        self.teacher_encoder_obs_buf = torch.clip(
            self.teacher_encoder_obs_buf, -clip_obs, clip_obs
        )
        self.student_encoder_obs_buf = torch.clip(
            self.student_encoder_obs_buf, -clip_obs, clip_obs
        )

        return (
            self.obs_buf,
            self.privileged_obs_buf,
            self.teacher_encoder_obs_buf,
            self.student_encoder_obs_buf,
            self.rew_buf,
            self.reset_buf,
            self.extras,
        )

    def post_physics_step(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1
        if hasattr(self, "real_episode_length_buf"):
            self.real_episode_length_buf += 1

        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(
            self.base_quat, self.root_states[:, 7:10]
        )
        self.base_ang_vel[:] = quat_rotate_inverse(
            self.base_quat, self.root_states[:, 10:13]
        )
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.feet_state = self.rigid_body_states[:, self.feet_indices, :]

        # Optional pre-reward callbacks for subclasses (e.g. gait calculation, rigid body params)
        if hasattr(self, '_pre_reward_callback'):
            self._pre_reward_callback()

        self._post_physics_step_callback()

        # Optional base height target update (used by fall-recovery envs)
        if hasattr(self, '_update_base_height_target'):
            self._update_base_height_target()

        self.check_termination()

        # Optional reward group update for curriculum-based reward managers
        if hasattr(self, '_reward_update'):
            self._reward_update()

        # Compute rewards: either via RewardManager or flat reward functions
        if hasattr(self, 'RewardManager'):
            self.rew_buf, self.episode_sums = self.RewardManager.compute_reward(self, self.reward_group)
        else:
            self.compute_reward()

        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self.compute_observations()

        self.last_last_action[:] = self.last_actions[:]
        self.last_last_base_velocity[:] = self.last_base_velocity[:]
        self.last_last_base_ang_velocity[:] = self.last_base_ang_velocity[:]
        self.last_actions[:] = self.actions[:]
        self.last_base_velocity[:] = self.base_lin_vel[:]
        self.last_base_ang_velocity[:] = self.base_ang_vel[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]
        if hasattr(self, "last_dof_pos"):
            self.last_last_dof_pos[:] = self.last_dof_pos[:]
            self.last_dof_pos[:] = self.dof_pos[:]

        self.last_observations_save()

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

    def check_termination(self):
        self.reset_buf = torch.any(
            torch.norm(
                self.contact_forces[:, self.termination_contact_indices, :], dim=-1
            )
            > 1.0,
            dim=1,
        )
        if self.cfg.rewards.terminations.using_time_overstep_terminate:
            self._overstep_termination()
        self.time_out_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= self.time_out_buf

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        if self.cfg.commands.curriculum and (
            self.common_step_counter % self.max_episode_length == 0
        ):
            self.update_command_curriculum(env_ids)

        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)
        self._resample_commands(env_ids)

        self.last_actions[env_ids] = 0.0
        self.last_last_action[env_ids] = 0.0
        self.last_base_velocity[env_ids] = 0.0
        self.last_last_base_velocity[env_ids] = 0.0
        self.last_base_ang_velocity[env_ids] = 0.0
        self.last_last_base_ang_velocity[env_ids] = 0.0
        self.last_dof_vel[env_ids] = 0.0
        self.feet_air_time[env_ids] = 0.0
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        if hasattr(self, "real_episode_length_buf"):
            self.real_episode_length_buf[env_ids] = 0

        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"][key] = (
                torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            )
            self.episode_sums[key][env_ids] = 0.0
            if hasattr(self, 'RewardManager'):
                self.RewardManager.reset_episode_sums(key, env_ids)
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(
                self.terrain_levels.float()
            )
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = 2.3
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

        if self.cfg.domain_rand.randomize_motor_strength:
            self.motor_strength[env_ids] = torch_rand_float(
                self.cfg.domain_rand.motor_strength_range[0],
                self.cfg.domain_rand.motor_strength_range[1],
                (len(env_ids), len(self.motor_rigid_idx)),
                device=self.device,
            )

    def compute_reward(self):
        self.rew_buf[:] = 0.0
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.0)
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew

    @abstractmethod
    def compute_observations_without_noise(self):
        raise NotImplementedError

    @abstractmethod
    def compute_observations(self):
        raise NotImplementedError

    @abstractmethod
    def compute_privileged_observations(self):
        raise NotImplementedError

    @abstractmethod
    def compute_teacher_encoder_observations(self):
        raise NotImplementedError

    def compute_student_encoder_observations(self):
        self.lotstime_obs = self.observations_stack[0]
        for i in range(self.cfg.env.observation_stack.add_time_number - 1):
            self.lotstime_obs = torch.cat(
                (self.lotstime_obs, self.observations_stack[i + 1]), dim=1
            )
        student_encoder_observations = self.lotstime_obs
        self.student_encoder_obs_buf = student_encoder_observations
        if hasattr(self, "unactuated_time"):
            self.student_encoder_obs_buf *= (
                self.real_episode_length_buf.unsqueeze(1) > self.unactuated_time
            )

    def create_sim(self):
        self.up_axis_idx = 2
        self.sim = self.gym.create_sim(
            self.sim_device_id,
            self.graphics_device_id,
            self.physics_engine,
            self.sim_params,
        )
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ["heightfield", "trimesh"]:
            self.terrain = Terrain(self.cfg.terrain, self.num_envs)
        if mesh_type == "plane":
            self._create_ground_plane()
        elif mesh_type == "heightfield":
            self._create_heightfield()
        elif mesh_type == "trimesh":
            self._create_trimesh()
        elif mesh_type == "hhd_trimesh":
            self._hhd_trimesh()
        elif mesh_type is not None:
            raise ValueError(
                "Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]"
            )
        self._create_envs()

    def set_camera(self, position, lookat):
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def _process_rigid_shape_props(self, props, env_id):
        if self.cfg.domain_rand.randomize_friction:
            if env_id == 0:
                friction_range = self.cfg.domain_rand.friction_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(
                    friction_range[0], friction_range[1], (num_buckets, 1), device="cpu"
                )
                self.friction_coeffs = friction_buckets[bucket_ids]
            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]
        if self.cfg.domain_rand.randomize_restitution:
            restitution_range = self.cfg.domain_rand.restitution_range
            self.restitution_coeffs = torch_rand_float(
                restitution_range[0],
                restitution_range[1],
                (self.num_envs, 1),
                device=self.device,
            )
            for s in range(len(props)):
                props[s].restitution = self.restitution_coeffs[env_id]
        return props

    def _process_dof_props(self, props, env_id):
        if env_id == 0:
            self.dof_pos_limits = torch.zeros(
                self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False
            )
            self.dof_vel_limits = torch.zeros(
                self.num_dof, dtype=torch.float, device=self.device, requires_grad=False
            )
            self.torque_limits = torch.zeros(
                self.num_dof, dtype=torch.float, device=self.device, requires_grad=False
            )
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
        return props

    def _process_rigid_body_props(self, props, env_id):
        if env_id == 0:
            self.total_mass = sum(p.mass for p in props)
        if self.cfg.domain_rand.randomize_base_mass:
            rng = self.cfg.domain_rand.added_mass_range
            props[0].mass += np.random.uniform(rng[0], rng[1])
        if self.cfg.domain_rand.add_mass_ranges:
            props[0].com.x += np.random.uniform(
                self.cfg.domain_rand.add_mass_range_x["lower"],
                self.cfg.domain_rand.add_mass_range_x["upper"],
            )
            props[0].com.y += np.random.uniform(
                self.cfg.domain_rand.add_mass_range_y["lower"],
                self.cfg.domain_rand.add_mass_range_y["upper"],
            )
            props[0].com.z += np.random.uniform(
                self.cfg.domain_rand.add_mass_range_z["lower"],
                self.cfg.domain_rand.add_mass_range_z["upper"],
            )
        return props

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
        if (self.common_step_counter - 1) / 24 - 1 > self.cfg.domain_rand.push_start_epoch:
            if self.cfg.domain_rand.push_robots and (
                self.common_step_counter % self.cfg.domain_rand.push_interval == 0
            ):
                self._push_robots()
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)

    def _resample_commands(self, env_ids):
        self.commands[env_ids, 0] = torch_rand_float(
            self.command_ranges["lin_vel_x"][0],
            self.command_ranges["lin_vel_x"][1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(
            self.command_ranges["lin_vel_y"][0],
            self.command_ranges["lin_vel_y"][1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)

        x_only_idx = torch.where(env_ids < self.cfg.commands.ranges.lin_vel_x_only)
        without_lin_vel_y = torch.where(
            env_ids
            < (
                self.cfg.commands.ranges.lin_vel_x_only
                + self.cfg.commands.ranges.without_lin_vel_y
            )
        )
        self.commands[env_ids[x_only_idx], 1:3] = 0.0
        self.commands[env_ids[without_lin_vel_y], 1] = 0.0

        self.commands[env_ids, 2] = torch_rand_float(
            self.command_ranges["ang_vel_yaw"][0],
            self.command_ranges["ang_vel_yaw"][1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)

        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(
                self.command_ranges["angle_pitch"][0],
                self.command_ranges["angle_pitch"][1],
                (len(env_ids), 1),
                device=self.device,
            ).squeeze(1)
            self.commands[env_ids, 4] = torch_rand_float(
                self.command_ranges["angle_roll"][0],
                self.command_ranges["angle_roll"][1],
                (len(env_ids), 1),
                device=self.device,
            ).squeeze(1)

        if self.cfg.commands.base_height_command:
            lower_height_idx = torch.where(env_ids < 1500)
            normal_height_idx = torch.where((env_ids >= 1500) & (env_ids < 3000))
            high_height_idx = torch.where(env_ids >= 3000)
            height_dims = self.cfg.commands.num_commands - 1
            if len(env_ids[lower_height_idx]) > 0:
                self.commands[env_ids[lower_height_idx], height_dims] = self.command_ranges[
                    "rew_base_height"
                ][0]
            if len(env_ids[normal_height_idx]) > 0:
                self.commands[env_ids[normal_height_idx], height_dims] = (
                    self.command_ranges["rew_base_height"][0]
                    + self.command_ranges["rew_base_height"][1]
                ) / 2
            if len(env_ids[high_height_idx]) > 0:
                self.commands[env_ids[high_height_idx], height_dims] = self.command_ranges[
                    "rew_base_height"
                ][1]

        self.commands[env_ids, :2] *= (
            torch.norm(self.commands[env_ids, :2], dim=1) > 0.05
        ).unsqueeze(1)

    def _compute_desired_projected_gravity(self) -> torch.Tensor:
        yaw = self.base_euler_xyz[:, 2]
        desired_quat = quat_from_euler_xyz(self.commands[:, 4], self.commands[:, 3], yaw)
        gravity = quat_rotate_inverse(desired_quat, self.gravity_vec)
        return gravity

    @abstractmethod
    def _compute_torques(self, actions):
        raise NotImplementedError

    def _reset_dofs(self, env_ids):
        self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(
            0.5, 1.5, (len(env_ids), self.num_dof), device=self.device
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
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

    def _push_robots(self):
        push_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)
        push_vel[:, 0:1] = torch_rand_float(
            -self.cfg.domain_rand.max_push_vel_x,
            self.cfg.domain_rand.max_push_vel_x,
            (self.num_envs, 1),
            device=self.device,
        )
        push_vel[:, 1:2] = torch_rand_float(
            -self.cfg.domain_rand.max_push_vel_y,
            self.cfg.domain_rand.max_push_vel_y,
            (self.num_envs, 1),
            device=self.device,
        )
        push_vel_base = quat_apply_yaw(self.base_quat, push_vel)
        self.root_states[:, 7:8] = push_vel_base[:, 0:1]
        self.root_states[:, 8:9] = push_vel_base[:, 1:2]
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def force_push(self):
        push_rigid_indices = self.push_rigid_indices
        push_force_x_range = self.cfg.domain_rand.push_force_x_range
        push_force_y_range = self.cfg.domain_rand.push_force_y_range
        push_force_z_range = self.cfg.domain_rand.push_force_z_range
        push_force = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)
        force_tensor = torch.zeros([self.num_envs, self.num_bodies, 3], device=self.device)
        for i, (_, idx) in enumerate(push_rigid_indices.items()):
            push_force[:, 0:1] = torch_rand_float(
                push_force_x_range[0],
                push_force_x_range[1],
                (self.num_envs, 1),
                device=self.device,
            )
            push_force[:, 1:2] = torch_rand_float(
                push_force_y_range[0],
                push_force_y_range[1],
                (self.num_envs, 1),
                device=self.device,
            )
            push_force[:, 2:3] = torch_rand_float(
                push_force_z_range[0],
                push_force_z_range[1],
                (self.num_envs, 1),
                device=self.device,
            )
            force_tensor[:, idx, self.push_direction_to_id[i]] = push_force[
                :, self.push_direction_to_id[i]
            ]
        force_tensor = gymtorch.unwrap_tensor(force_tensor)
        self.gym.apply_rigid_body_force_tensors(self.sim, force_tensor)
        self.gym.refresh_dof_state_tensor(self.sim)

    def _shank_change(self):
        max_change_angel = self.cfg.observation_stack.max_change_angel
        add_range_matrix = (
            torch.rand_like(self.dof_pos[:, 1], device=self.device) * 2 * max_change_angel
            - max_change_angel
        )
        self.dof_pos[:, 1] += add_range_matrix
        self.gym.set_dof_state_tensor(self.sim, self.dof_pos)

    def _update_terrain_curriculum(self, env_ids):
        if not self.init_done:
            return
        distance = torch.norm(
            self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1
        )
        move_up = distance > self.terrain.env_length / 2
        move_down = (
            distance
            < torch.norm(self.commands[env_ids, :2], dim=1) * self.max_episode_length_s * 0.5
        ) * ~move_up
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        self.terrain_levels[env_ids] = torch.where(
            self.terrain_levels[env_ids] >= self.max_terrain_level,
            torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
            torch.clip(self.terrain_levels[env_ids], 0),
        )
        self.env_origins[env_ids] = self.terrain_origins[
            self.terrain_levels[env_ids], self.terrain_types[env_ids]
        ]

    def update_command_curriculum(self, env_ids):
        if (
            torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length
            > 0.8 * self.reward_scales["tracking_lin_vel"]
        ):
            self.command_ranges["lin_vel_x"][0] = np.clip(
                self.command_ranges["lin_vel_x"][0] - 0.5,
                -self.cfg.commands.max_curriculum,
                0.0,
            )
            self.command_ranges["lin_vel_x"][1] = np.clip(
                self.command_ranges["lin_vel_x"][1] + 0.5,
                0.0,
                self.cfg.commands.max_curriculum,
            )

    def _get_noise_scale_vec(self, cfg):
        self.compute_observations_without_noise()
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = 0.0
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[6:9] = noise_scales.gravity * noise_level
        add_dims = self.cfg.commands.num_commands - 3
        noise_vec[9 : 12 + add_dims] = 0.0
        noise_vec[add_dims + 12 : 18 + add_dims] = (
            noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        )
        noise_vec[add_dims + 18 : 26 + add_dims] = (
            noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        )
        noise_vec[add_dims + 26 : 34 + add_dims] = 0.0
        if self.cfg.terrain.measure_heights:
            noise_vec[34 + add_dims : 235 + add_dims] = (
                noise_scales.height_measurements
                * noise_level
                * self.obs_scales.height_measurements
            )
        return noise_vec

    def _init_buffers(self):
        self.velocity_max_now = 0
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
            2
            * (
                self.base_quat[:, 1] * self.base_quat[:, 3]
                - self.base_quat[:, 0] * self.base_quat[:, 2]
            )
        )

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(
            self.num_envs, -1, 3
        )

        self.common_step_counter = 0
        self.has_counter = True
        self.extras = {}

        if self.cfg.rewards.curriculum.using_pull_up:
            self.force_scale = torch.ones(self.num_envs, device=self.device) * 100
            self.force_torque = torch.zeros(self.num_envs, device=self.device)
        else:
            self.force_scale = torch.zeros(self.num_envs, device=self.device)
            self.force_torque = torch.zeros(self.num_envs, device=self.device)

        self.gravity_vec = to_torch(
            get_axis_params(-1.0, self.up_axis_idx), device=self.device
        ).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1.0, 0.0, 0.0], device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.torques = torch.zeros(
            self.num_envs,
            self.num_actions,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.p_gains = torch.zeros(
            self.num_actions, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.d_gains = torch.zeros(
            self.num_actions, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.actions = torch.zeros(
            self.num_envs,
            self.num_actions,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.last_actions = torch.zeros(
            self.num_envs,
            self.num_actions,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
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

        self.default_dof_pos = torch.zeros(
            self.num_dof, dtype=torch.float, device=self.device, requires_grad=False
        )
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle

        p_gains_i = 0
        for dof_name in self.cfg.control.stiffness.keys():
            self.p_gains[p_gains_i + int(self.num_dofs / 2)] = self.cfg.control.stiffness[
                dof_name
            ]
            self.d_gains[p_gains_i + int(self.num_dofs / 2)] = self.cfg.control.damping[
                dof_name
            ]
            self.p_gains[p_gains_i] = self.cfg.control.stiffness[dof_name]
            self.d_gains[p_gains_i] = self.cfg.control.damping[dof_name]
            p_gains_i += 1
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

        self.kp = torch.zeros(
            self.num_dof, device=self.device, requires_grad=False, dtype=torch.float32
        )
        self.kd = torch.zeros(
            self.num_dof, device=self.device, requires_grad=False, dtype=torch.float32
        )
        self.action_scale = torch.zeros(
            self.num_dof, device=self.device, requires_grad=False, dtype=torch.float32
        )
        if (
            hasattr(self.cfg.control, "action_scale")
            and isinstance(self.cfg.control.action_scale, dict)
        ):
            for dof_name, value in self.cfg.control.stiffness.items():
                joint_idx = self.find_joint_id(dof_name)
                if joint_idx:
                    joint_idx_t = torch.tensor(joint_idx, device=self.device)
                    self.kp[joint_idx_t] = value
                    self.kd[joint_idx_t] = self.cfg.control.damping[dof_name]
                    self.action_scale[joint_idx_t] = self.cfg.control.action_scale[dof_name]

        self.compute_observations_without_noise()
        num_obs = self.obs_buf.shape[1]
        self.last_observations = torch.zeros(self.num_envs, num_obs, device=self.device)

        self.observations_stack = deque(maxlen=self.cfg.env.observation_stack.add_time_number)
        for i in range(self.cfg.env.observation_stack.add_time_number):
            self.observations_stack.append(
                torch.zeros(self.num_envs, num_obs, device=self.device)
            )

        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.last_last_action = torch.zeros_like(self.actions)
        self.last_base_velocity = torch.zeros_like(self.base_lin_vel)
        self.last_last_base_velocity = torch.zeros_like(self.base_lin_vel)
        self.last_base_ang_velocity = torch.zeros_like(self.base_ang_vel)
        self.last_last_base_ang_velocity = torch.zeros_like(self.base_ang_vel)
        self.feet_id = torch.randn(self.num_envs, device=self.device, requires_grad=False)

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
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state).view(
            self.num_envs, self.num_bodies, -1
        )
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

        base_name = "torso"
        self.base_indice = self.gym.find_actor_rigid_body_handle(
            self.envs[0], self.actor_handles[0], base_name
        )
        self.base_indices = torch.tensor(
            [self.base_indice], dtype=torch.long, device=self.device, requires_grad=False
        )

        push_rigid_name = self.cfg.domain_rand.push_rigid_name
        push_direction = self.cfg.domain_rand.push_direction
        if len(push_rigid_name) != len(push_direction):
            raise AttributeError(
                "rigid number is not epual to numbers in direction, please check it!"
            )
        self.push_direction_to_id = []

        direction_map = {"x": 0, "y": 1, "z": 2}
        self.push_direction_to_id = [
            [direction_map[direction] for direction in direction_list]
            for direction_list in push_direction
        ]

        self.push_rigid_indices = {}
        for name in push_rigid_name:
            self.push_rigid_indices[name] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], name
            )

        if self.cfg.domain_rand.randomize_motor_strength:
            motor_rigid_name = self.cfg.domain_rand.joint_name_need_strength_rand
            self.motor_rigid_idx = self.find_joint_id(motor_rigid_name)
            self.motor_strength = torch_rand_float(
                self.cfg.domain_rand.motor_strength_range[0],
                self.cfg.domain_rand.motor_strength_range[1],
                (self.num_envs, len(self.motor_rigid_idx)),
                device=self.device,
            )

        self.time_counter = torch.zeros(self.num_envs, device=self.device)

        body_name = ["torso", "lhip", "lfem", "ltib", "rhip", "rfem", "rtib"]
        self.body_indices = self.find_rigid_body_id(body_name)

        self.real_episode_length_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )
        self.height_targets = torch.zeros(self.num_envs, 1, device=self.device)
        self.up_time = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self.last_dof_pos = torch.zeros_like(self.dof_pos)
        self.last_last_dof_pos = torch.zeros_like(self.dof_pos)

    def find_rigid_body_id(self, rigid_body_name) -> list:
        if isinstance(rigid_body_name, str):
            rigid_body_name = [rigid_body_name]
        body_names = []
        for name in rigid_body_name:
            body_names.append(
                self.gym.find_actor_rigid_body_handle(
                    self.envs[0], self.actor_handles[0], name
                )
            )
        return body_names

    def find_joint_id(self, joint_name) -> list:
        if isinstance(joint_name, str):
            joint_name = [joint_name]
        joint_id = []
        for name in joint_name:
            matches = [idx for idx, joint in enumerate(self.dof_names) if name in joint]
            joint_id.extend(matches)
        return joint_id

    def _prepare_reward_function(self):
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale == 0:
                self.reward_scales.pop(key)
            else:
                self.reward_scales[key] *= self.dt
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name == "termination":
                continue
            self.reward_names.append(name)
            name = "_reward_" + name
            self.reward_functions.append(getattr(self, name))

        self.episode_sums = {
            name: torch.zeros(
                self.num_envs,
                dtype=torch.float,
                device=self.device,
                requires_grad=False,
            )
            for name in self.reward_scales.keys()
        }

    def _init_reward_manager(self):
        """Initialize RewardManager for environments using reward groups."""
        reward_list = list(set(key for reward_value in self.reward_group_list.values() for key in reward_value))
        self.RewardManager = RewardManager(self.num_envs, reward_list, self.dt, self.device)
        if hasattr(self, '_reward_register'):
            self._reward_register()
        self.episode_sums = {
            name: torch.zeros(
                self.num_envs,
                dtype=torch.float,
                device=self.device,
                requires_grad=False,
            )
            for name in reward_list
        }
        # Default to first reward group
        self.reward_group = self.reward_group_list[list(self.reward_group_list.keys())[0]]

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)

    def _create_heightfield(self):
        hf_params = gymapi.HeightFieldParams()
        hf_params.column_scale = self.terrain.cfg.horizontal_scale
        hf_params.row_scale = self.terrain.cfg.horizontal_scale
        hf_params.vertical_scale = self.terrain.cfg.vertical_scale
        hf_params.nbRows = self.terrain.tot_cols
        hf_params.nbColumns = self.terrain.tot_rows
        hf_params.transform.p.x = -self.terrain.cfg.border_size
        hf_params.transform.p.y = -self.terrain.cfg.border_size
        hf_params.transform.p.z = 0.0
        hf_params.static_friction = self.cfg.terrain.static_friction
        hf_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        hf_params.restitution = self.cfg.terrain.restitution
        self.gym.add_heightfield(self.sim, self.terrain.heightsamples, hf_params)
        self.height_samples = (
            torch.tensor(self.terrain.heightsamples)
            .view(self.terrain.tot_rows, self.terrain.tot_cols)
            .to(self.device)
        )

    def _create_trimesh(self):
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]
        tm_params.transform.p.x = -self.terrain.cfg.border_size
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        self.gym.add_triangle_mesh(
            self.sim,
            self.terrain.vertices.flatten(order="C"),
            self.terrain.triangles.flatten(order="C"),
            tm_params,
        )
        self.height_samples = (
            torch.tensor(self.terrain.heightsamples)
            .view(self.terrain.tot_rows, self.terrain.tot_cols)
            .to(self.device)
        )

    def _create_envs(self):
        asset_path = self.cfg.asset.file.format(GETUP_GYM_ROOT_DIR=GETUP_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity
        asset_options.enable_gyroscopic_forces = self.cfg.asset.enable_gyroscopic_forces

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        self.dof_dict = self.gym.get_asset_dof_dict(robot_asset)

        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = self.cfg.asset.foot_name
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        base_init_state_list = (
            self.cfg.init_state.pos
            + self.cfg.init_state.rot
            + self.cfg.init_state.lin_vel
            + self.cfg.init_state.ang_vel
        )
        self.base_init_state = to_torch(
            base_init_state_list, device=self.device, requires_grad=False
        )
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0.0, 0.0, 0.0)
        env_upper = gymapi.Vec3(0.0, 0.0, 0.0)
        self.actor_handles = []
        self.envs = []
        for i in range(self.num_envs):
            env_handle = self.gym.create_env(
                self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs))
            )
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1.0, 1.0, (2, 1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)

            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)

            actor_handle = self.gym.create_actor(
                env_handle,
                robot_asset,
                start_pose,
                self.cfg.asset.name,
                i,
                self.cfg.asset.self_collisions,
                0,
            )
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(
                env_handle, actor_handle, body_props, recomputeInertia=True
            )
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

        self.feet_indices = torch.zeros(
            len(feet_names), dtype=torch.long, device=self.device, requires_grad=False
        )
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], feet_names[i]
            )

        self.penalised_contact_indices = torch.zeros(
            len(penalized_contact_names),
            dtype=torch.long,
            device=self.device,
            requires_grad=False,
        )
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], penalized_contact_names[i]
            )

        self.termination_contact_indices = torch.zeros(
            len(termination_contact_names),
            dtype=torch.long,
            device=self.device,
            requires_grad=False,
        )
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], termination_contact_names[i]
            )
        self.body_names = body_names

    def _get_env_origins(self):
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            max_init_level = self.cfg.terrain.max_init_terrain_level
            if not self.cfg.terrain.curriculum:
                max_init_level = self.cfg.terrain.num_rows - 1
            self.terrain_levels = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
            self.terrain_types = torch.div(
                torch.arange(self.num_envs, device=self.device),
                (self.num_envs / self.cfg.terrain.num_cols),
                rounding_mode="floor",
            ).to(torch.long)
            self.max_terrain_level = self.cfg.terrain.num_rows
            self.terrain_origins = (
                torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            )
            self.env_origins[:] = self.terrain_origins[
                self.terrain_levels, self.terrain_types
            ]
        elif self.cfg.terrain.mesh_type == "hhd_trimesh":
            self.custom_origins = False
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            hhd_num_cols = 100
            hhd_num_rows = int(self.num_envs / hhd_num_cols) + 1
            xx, yy = torch.meshgrid(torch.arange(hhd_num_rows), torch.arange(hhd_num_cols))
            spacing = self.cfg.env.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[: self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[: self.num_envs]
            self.env_origins[:, 2] = 0
        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            spacing = self.cfg.env.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[: self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[: self.num_envs]
            self.env_origins[:, 2] = 0.0

    def _parse_cfg(self, cfg):
        """Parse config and set runtime parameters."""
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        if hasattr(self.cfg.rewards, 'scales'):
            self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        else:
            self.reward_scales = {}
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)

        # Support reward groups (used by some envs for curriculum)
        if hasattr(self.cfg.rewards, 'curriculum') and hasattr(self.cfg.rewards.curriculum, 'reward_group'):
            self.reward_group_list = {}
            for group_name in self.cfg.rewards.curriculum.reward_group:
                group_cfg = getattr(self.cfg.rewards, group_name, None)
                if group_cfg is not None:
                    self.reward_group_list[group_name] = class_to_dict(group_cfg)

        if self.cfg.terrain.mesh_type not in ["heightfield", "trimesh"]:
            self.cfg.terrain.curriculum = False
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        if hasattr(self.cfg.domain_rand, 'push_interval_s'):
            self.cfg.domain_rand.push_interval = np.ceil(
                self.cfg.domain_rand.push_interval_s / self.dt
            )

    def _draw_debug_vis(self):
        if not self.terrain.cfg.measure_heights:
            return
        self.gym.clear_lines(self.viewer)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 0))
        for i in range(self.num_envs):
            base_pos = (self.root_states[i, :3]).cpu().numpy()
            heights = self.measured_heights[i].cpu().numpy()
            height_points = (
                quat_apply_yaw(self.base_quat[i].repeat(heights.shape[0]), self.height_points[i])
                .cpu()
                .numpy()
            )
            for j in range(heights.shape[0]):
                x = height_points[j, 0] + base_pos[0]
                y = height_points[j, 1] + base_pos[1]
                z = heights[j]
                sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)

    def _init_height_points(self):
        y = torch.tensor(self.cfg.terrain.measured_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(self.cfg.terrain.measured_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(
            self.num_envs,
            self.num_height_points,
            3,
            device=self.device,
            requires_grad=False,
        )
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def _get_heights(self, env_ids=None):
        if self.cfg.terrain.mesh_type == "plane":
            return torch.zeros(
                self.num_envs,
                self.num_height_points,
                device=self.device,
                requires_grad=False,
            )
        elif self.cfg.terrain.mesh_type == "none":
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids is not None:
            points = quat_apply_yaw(
                self.base_quat[env_ids].repeat(1, self.num_height_points),
                self.height_points[env_ids],
            ) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(
                self.base_quat.repeat(1, self.num_height_points), self.height_points
            ) + (self.root_states[:, :3]).unsqueeze(1)

        points += self.terrain.cfg.border_size
        points = (points / self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0] - 2)
        py = torch.clip(py, 0, self.height_samples.shape[1] - 2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px + 1, py]
        heights3 = self.height_samples[px, py + 1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

    def pitch_theta_bias_calculation(self):
        self.pitch_theta = 2 * torch.arctan(self.base_quat[:, 2] / self.base_quat[:, 0])
        self.pitch_theta_bias = self.pitch_theta - self.cfg.observation_stack.pitch_theta_setting

    def last_observations_save(self):
        self.observations_stack.append(self.obs_buf)

    def _base_force_pull_up(self):
        force_max = self.cfg.rewards.curriculum.base_pull_up_max
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
        if not getattr(self.cfg.rewards, "using_pull_up_end", True):
            time_coeff = torch.ones_like(time_coeff)

        base_height_target = getattr(self.cfg.rewards, "base_height_target", 0.5)
        force_scale = (
            (base_height_target - base_height) / base_height_target
        ) * force_max * time_coeff

        self.force_scale = torch.clamp(force_scale, 0.0, force_max)

        forces = torch.zeros(
            (self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float
        )

        body_idx = getattr(self.cfg.rewards, "pull_up_body_index", None)
        if body_idx is None:
            if hasattr(self, "base_indices") and len(self.base_indices) > 1:
                body_idx = int(self.base_indices[1].item())
            else:
                body_idx = getattr(self, "base_indice", 0)
        if isinstance(body_idx, torch.Tensor):
            body_idx = int(body_idx.item())

        forces[:, body_idx, 2] = self.force_scale

        torques = torch.zeros_like(forces)

        env_start = getattr(self.cfg.rewards, "pull_up_env_start", 0)
        env_end = getattr(self.cfg.rewards, "pull_up_env_end", self.num_envs)
        torque_scale = getattr(self.cfg.rewards, "pull_up_torque_scale", -10.0)

        if env_end > env_start:
            rotate_scale = torch.abs(self.root_states[env_start:env_end, 4] + 1)
            height_scale = (
                (base_height_target - base_height) / base_height_target
            )[env_start:env_end]
            torques[env_start:env_end, body_idx, 0] = (
                torque_scale * time_coeff * rotate_scale * height_scale
            )

        if hasattr(self, "real_episode_length_buf") and hasattr(self, "unactuated_time"):
            mask = (
                self.real_episode_length_buf.unsqueeze(1) > self.unactuated_time
            ).unsqueeze(1)
            forces = forces * mask
            torques = torques * mask

        self.gym.apply_rigid_body_force_tensors(
            self.sim,
            gymtorch.unwrap_tensor(forces),
            gymtorch.unwrap_tensor(torques),
            gymapi.ENV_SPACE,
        )

    def _overstep_termination(self):
        time_max = self.cfg.rewards.terminations.time_overstep_max * torch.ones(
            self.num_envs, device=self.device
        )
        base_height = torch.mean(
            self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1
        )

        body_contact_idx = (
            torch.sum(
                1.0
                * (
                    torch.norm(self.contact_forces[:, self.body_indices, :], dim=-1) > 4.0
                ),
                dim=1,
            )
            > 0
        )
        across_torque_idx = torch.sum(torch.abs(self.torques), dim=1) > 200
        body_height_idx = base_height < 0.6

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

    # ------------ reward functions ----------------
    def _reward_orientation(self):
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_rew_base_height(self):
        base_height = torch.mean(
            self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1
        )
        return torch.square(base_height - self.cfg.rewards.base_height_target)

    def _reward_torques(self):
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        return torch.sum(torch.square(self.dof_vel), dim=1)

    def _reward_dof_acc(self):
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)

    def _reward_action_rate(self):
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_collision(self):
        return torch.sum(
            1.0
            * (
                torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1)
                > 0.1
            ),
            dim=1,
        )

    def _reward_termination(self):
        return self.reset_buf * ~self.time_out_buf

    def _reward_dof_pos_limits(self):
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.0)
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.0)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        return torch.sum(
            (torch.abs(self.dof_vel) - self.dof_vel_limits * self.cfg.rewards.soft_dof_vel_limit).clip(
                min=0.0, max=1.0
            ),
            dim=1,
        )

    def _reward_torque_limits(self):
        return torch.sum(
            (torch.abs(self.torques) - self.torque_limits * self.cfg.rewards.soft_torque_limit).clip(
                min=0.0
            ),
            dim=1,
        )

    def _reward_tracking_lin_vel(self):
        lin_vel_sigma = self.cfg.rewards.tracking_sigma
        lin_vel_error = torch.sum(
            torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1
        )
        return torch.exp(-lin_vel_error / lin_vel_sigma)

    def _reward_tracking_lin_vel_y(self):
        lin_vel_sigma = self.cfg.rewards.tracking_sigma
        lin_vel_error = torch.square(self.commands[:, 1] - self.base_lin_vel[:, 1])
        return torch.exp(-lin_vel_error / lin_vel_sigma)

    def _reward_tracking_lin_vel_x(self):
        lin_vel_sigma = self.cfg.rewards.tracking_sigma
        lin_vel_error = torch.square(self.commands[:, 0] - self.base_lin_vel[:, 0])
        return torch.exp(-lin_vel_error / lin_vel_sigma)

    def _reward_tracking_ang_vel(self):
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.cfg.rewards.tracking_sigma)

    def _reward_feet_air_time(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 1
        contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.0) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1)
        self.feet_air_time *= ~contact_filt
        return rew_airTime

    def _reward_stumble(self):
        return torch.any(
            torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2)
            > 5 * torch.abs(self.contact_forces[:, self.feet_indices, 2]),
            dim=1,
        )

    def _reward_feet_contact_forces(self):
        return torch.sum(
            (
                torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1)
                - self.cfg.rewards.max_contact_force
            ).clip(min=0.0),
            dim=1,
        )
