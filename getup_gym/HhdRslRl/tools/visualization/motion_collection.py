import sys
from getup_gym import WHEEL_GYM_ROOT_DIR
import os

import isaacgym
from getup_gym.envs import *
from getup_gym.HhdRslRl.tools.utils_gym.helpers import  get_args, export_policy_as_jit
from getup_gym.HhdRslRl.tools.utils_gym.logger import Logger
from getup_gym.HhdRslRl.tools.Teacher_task_register import teacher_task_registry

import numpy as np
import torch
import time

import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Value


def play(args):
    env_cfg, train_cfg = teacher_task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 100)
    env_cfg.terrain.num_rows = 4
    env_cfg.terrain.num_cols = 4
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False


    env, _ = teacher_task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    teacher_obs = env.get_teacher_encoder_observations()
    student_obs = env.get_student_encoder_observations()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = teacher_task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    encoder_policy, inference_mode = ppo_runner.get_encoder_inference_policy(device=env.device)
   
    os.makedirs('./visualization/data', exist_ok=True)
    root_state = []
    for i in range(10*int(env.max_episode_length)):
        # result = env.gym.fetch_results(env.sim, True)
        if inference_mode in ["RMA-Teacher"]:
            zt = encoder_policy(teacher_obs)
        elif inference_mode in ["RMA-TeacherStudent", "CTS"]:
            zt = encoder_policy(student_obs) 
        
        if inference_mode in ["RMA-Teacher", "RMA-TeacherStudent", "CTS"]:
            zt_and_obs = torch.cat((zt["zt"], obs), dim=-1)  # Concatenate teacher latent and actions
        else:
            zt_and_obs = obs
            
        actions = policy(zt_and_obs.detach())              #用于具体保存训练得到的policy参数。这些参数可以输出到机器人进行实机演示
        obs, _, teacher_obs, student_obs, rews, dones, infos = env.step(actions.detach())
        # print(f"teacher obs is: {teacher_obs[0,0:10]}")

        if env.real_episode_length_buf[0] >= env.unactuated_time:
            root_state.append(env.rigid_body_states[0].detach().cpu().numpy())

        if len(root_state) == 200:
            np.save(f'./visualization/data/{train_cfg.runner.experiment_name}_root_state.npy', np.array(root_state), allow_pickle=True)
            break
    

if __name__ == '__main__':
    args = get_args()
    play(args)
