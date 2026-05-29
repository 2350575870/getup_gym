"""Base task class for Isaac Gym environments."""

import sys
import numpy as np
import torch
from isaacgym import gymapi, gymutil
from isaacgym.torch_utils import quat_apply


class BaseTask:
    """Base class for RL tasks in Isaac Gym."""

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        self.gym = gymapi.acquire_gym()
        self.sim_params = sim_params
        self.physics_engine = physics_engine
        self.sim_device = sim_device
        sim_device_type, self.sim_device_id = gymutil.parse_device_str(self.sim_device)
        self.headless = headless

        if sim_device_type == "cuda" and sim_params.use_gpu_pipeline:
            self.device = self.sim_device
        else:
            self.device = "cpu"

        self.graphics_device_id = self.sim_device_id
        if self.headless:
            self.graphics_device_id = -1

        self.num_envs = cfg.env.num_envs
        self.num_envs_teacher = getattr(cfg.env, "num_envs_teacher", 0)
        self.num_actions = cfg.env.num_actions
        self.layer_size = getattr(cfg.env, "layer_size", 0)

        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        self.obs_buf = None
        self.teacher_encoder_obs_buf = None
        self.student_encoder_obs_buf = None
        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.privileged_obs_buf = None
        self.extras = {}

        self.create_sim()
        self.gym.prepare_sim(self.sim)
        self.set_viewer()

    def set_viewer(self):
        self.enable_viewer_sync = True
        self.viewer = None
        self.viewer_move = gymapi.Vec3(0, 0, 0)
        if not self.headless:
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_ESCAPE, "QUIT")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_V, "toggle_viewer_sync")

    def get_observations(self):
        return self.obs_buf

    def get_privileged_observations(self):
        return self.privileged_obs_buf

    def get_teacher_encoder_observations(self):
        return self.teacher_encoder_obs_buf

    def get_student_encoder_observations(self):
        return self.student_encoder_obs_buf

    def reset_idx(self, env_ids):
        raise NotImplementedError

    def reset(self):
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        obs, privileged_obs, *_ = self.step(
            torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False)
        )
        return obs, privileged_obs

    def step(self, actions):
        raise NotImplementedError

    def render(self, sync_frame_time=True):
        if self.viewer:
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    sys.exit()
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self.enable_viewer_sync = not self.enable_viewer_sync
            if self.device != "cpu":
                self.gym.fetch_results(self.sim, True)
            if self.enable_viewer_sync:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)
                if sync_frame_time:
                    self.gym.sync_frame_time(self.sim)
            else:
                self.gym.poll_viewer_events(self.viewer)

    def _prepare_reward_function(self):
        """Prepares a list of reward functions which will be called to compute total reward.
        Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non-zero reward scales in cfg.
        """
        from getup_gym.common.helpers import class_to_dict

        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
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
            self.reward_functions.append(getattr(self, "_reward_" + name))
        self.episode_sums = {
            name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            for name in self.reward_scales.keys()
        }

    def create_sim(self):
        raise NotImplementedError
