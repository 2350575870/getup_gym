"""Play / evaluation script for getup_gym."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from getup_gym.utils.registry import task_registry
from getup_gym.common.helpers import get_args
import torch
import numpy as np


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    # Override for evaluation
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 50)
    env_cfg.env.num_envs_teacher = 0
    env_cfg.terrain.num_rows = 1
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.rewards.curriculum.using_pull_up = False
    env_cfg.terrain.mesh_type = "plane"
    env_cfg.commands.ranges.lin_vel_x = [0.0, 0.0]
    env_cfg.commands.ranges.ang_vel_yaw = [0.0, 0.0]
    env_cfg.commands.ranges.lin_vel_y = [0.0, 0.0]

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    teacher_obs = env.get_teacher_encoder_observations()
    student_obs = env.get_student_encoder_observations()
    actions = torch.zeros((env.num_envs, env.num_actions), device=env.device)
    obs, _, teacher_obs, student_obs, rews, dones, infos = env.step(actions.detach())

    train_cfg.runner.resume = True
    ppo_runner, _ = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    encoder_policy, inference_mode = ppo_runner.get_encoder_inference_policy(device=env.device)

    print("Running policy...")
    for _ in range(10 * int(env.max_episode_length)):
        if inference_mode == "CTS" and encoder_policy is not None:
            zt = encoder_policy(student_obs)["zt"]
            obs_input = torch.cat((zt, obs), dim=-1)
        else:
            obs_input = obs
        actions = policy(obs_input.detach())
        obs, _, teacher_obs, student_obs, rews, dones, infos = env.step(actions.detach())


if __name__ == "__main__":
    args = get_args()
    play(args)
