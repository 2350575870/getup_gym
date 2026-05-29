# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import isaacgym
from getup_gym import WHEEL_GYM_ROOT_DIR
import os

from getup_gym.envs import *

from getup_gym.HhdRslRl.tools.utils_gym.logger import Logger
from getup_gym.HhdRslRl.tools.Teacher_task_register import teacher_task_registry
from getup_gym.HhdRslRl.tools.utils_gym.helpers import get_args, export_policy_as_jit, print_welcome_message, encoder_export_policy_as_jit
from getup_gym.HhdRslRl.tools.utils_gym.Zlog import zzs_basic_graph_logger
from getup_gym.HhdRslRl.tools.hhdplot import HhdPlot
# 导入t-SNE可视化工具
from getup_gym.HhdRslRl.tools.tSNE import RepresentationVisualizer
import pickle
import matplotlib.pyplot as plt
from getup_gym.HhdRslRl.tools.eval_class import StairClimbingAnalyzer

if os.path.exists("./legged_gym/envs/CustomEnvironments"):
    from legged_gym.envs.CustomEnvironments import *

import numpy as np
import torch
import copy

def TSplay(args):

    #训练环境以及训练参数的初始化
    env_cfg, train_cfg = teacher_task_registry.get_cfgs(name=args.task)#导入环境配置参数，训练参数
     
    #演示环境参数初始化（override some parameters for testing）
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 50)
    env_cfg.env.num_envs_teacher = 0
    env_cfg.terrain.num_rows = 1
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = True
    # env_cfg.domain_rand.randomize_friction = True
    # env_cfg.domain_rand.push_robots = True

    env_cfg.domain_rand.randomize_friction = False
    # env_cfg.domain_rand.friction_range = friction_range = [0.2, 0.5]
    env_cfg.domain_rand.push_robots = True
    env_cfg.rewards.using_pull_up = False

    env_cfg.commands.ranges.lin_vel_x = [0.0, 0.0]
    env_cfg.commands.ranges.ang_vel_yaw = [-0.0, -0.0]
    env_cfg.commands.ranges.lin_vel_y = [0.0, 0.0]
    env_cfg.terrain.mesh_type = "plane"
    env_cfg.terrain.terrain_proportions = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]

    # train_cfg.seed = 4098

    export_to_onnx = False

    #根据__init__()函数中设置的训练名，训练对应的三个初始化类进行play环境的初始化（prepare environment）
    env, _ = teacher_task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    
    #获取环境的观测值
    obs = env.get_observations()
    teacher_obs = env.get_teacher_encoder_observations()
    student_obs = env.get_student_encoder_observations()
    actions = torch.zeros((env.num_envs, env.num_actions), device=env.device)
    obs, _, teacher_obs, student_obs, rews, dones, infos = env.step(actions.detach())
    
    # 导入policy（load policy）
    train_cfg.runner.resume = True #结束一段policy之后是否需要重新开始？
    ppo_runner, train_cfg = teacher_task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    encoder_policy, inference_mode = ppo_runner.get_encoder_inference_policy(device=env.device)
    teacher_encoder_policy = ppo_runner.get_teacher_encoder_policy(device=env.device)
    

    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(
            WHEEL_GYM_ROOT_DIR,
            "logs",
            args.task,
            "exported",
            "policies",
        )
        # export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        # print("Exported policy as jit script to: ", path)
        # if inference_mode in ["RMA-TeacherStudent", "CTS"]:
        #     encoder_export_policy_as_jit(ppo_runner.student_encoder_alg.encoder, path)
        #     print("Exported encoder policy as jit script to: ", path)

    stop_state_log = 1000  # number of steps before plotting states
    robot_index = [0]  # which robot is used for logging
    joint_index = 0  # which joint is used for logging

    stop_rew_log = env.max_episode_length + 1  # number of steps before print average episode rewards
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1.0, 1.0, 0.0])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0

    all_number = env_cfg.env.num_envs
    unfinished = 0

    #plot class init
    save_path = "data_plot"
    hhdplot = HhdPlot(env, 500, save_path=save_path)
    completed_trials = 0

    # run policy
    print("Running policy...")
    for i in range(10 * int(env.max_episode_length)):
        # 你的原始逻辑完全保留
        if inference_mode in ["RMA-Teacher"]:
            zt = encoder_policy(teacher_obs)["zt"]
        elif inference_mode in ["RMA-TeacherStudent", "CTS"]:
            zt = encoder_policy(student_obs)["zt"]
        
        if inference_mode in ["RMA-Teacher", "RMA-TeacherStudent", "CTS"]:
            zt_and_obs = torch.cat((zt, obs), dim=-1)  # Concatenate teacher latent and actions
        else:
            zt_and_obs = obs
            
        actions = policy(zt_and_obs.detach())              #用于具体保存训练得到的policy参数。这些参数可以输出到机器人进行实机演示
        
        # 获取 base 高度（从 obs 中提取，假设最后几维包含 base 高度或从 env 获取）
        base_height = env.root_states[:, 2] if hasattr(env, 'root_states') else torch.zeros(env.num_envs, device=env.device)
        
        # 你的原始step逻辑完全保留
        obs, _, teacher_obs, student_obs, rews, dones, infos = env.step(actions.detach())

        if dones is not None:
            all_number += torch.sum(dones)
            unfinished += torch.sum(dones)
            
        # if i%100 == 0:
        #     print(f"heights is: {teacher_obs[0, :10].cpu().numpy()}")

        if i == env.max_episode_length // 3:
            success_rate = (all_number - unfinished)/all_number *100
            print(f"total episode is: {all_number}, unfinished number is: {unfinished}  success_rate is: {success_rate}")

        # hhdplot.save_plot_data()

        # if i == 400:
        #     hhdplot.plot_all_params()
        #     hhdplot.save_to_csv()



if __name__ == "__main__":
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    USE_ZZS_LOGGER = False
    args = get_args()
    print_welcome_message()
    TSplay(args)
