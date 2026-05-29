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

import time
import os
from collections import deque
import statistics

from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
#models import
from getup_gym.HhdRslRl.Modules.actor_critic import ActorCritic
from getup_gym.HhdRslRl.basement.base_modules.ModulesBase import ModulesBase
from getup_gym.HhdRslRl.basement.base_runner.RunnerBase import RunnerBase
from getup_gym.HhdRslRl.basement.base_algorithm.AlgorithmBase import AlgorithmBase
from getup_gym.HhdRslRl.Algorithms.RMAppo import RMAPPO
from getup_gym.HhdRslRl.Env.TSvec import TSVecEnv
from getup_gym.HhdRslRl.tools.algo_registry import get_algo_class, get_net_class, get_encoder_class
from getup_gym.HhdRslRl.Modules.identity import IdentityEncoder
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np


class TSOnPolicyRunner(RunnerBase):
    def __init__(self, env: TSVecEnv, train_cfg, log_dir=None, device="cpu"):
        self.cfg = train_cfg["runner"]
        self.encoder_config = self.cfg["encoder_training_setting"]
        self.alg_cfg = self.cfg["algorithm_config"]
        self.policy_cfg = self.cfg["policy_config"]
        self.symmetrical_cfg = self.cfg["symmetrical_loss_config"]
        self.teacher_encoder_cfg = self.encoder_config["teacher_encoder_config"]
        self.student_encoder_cfg = self.encoder_config["student_encoder_config"]
        self.encoder_alg_cfg = self.encoder_config["student_encoder_alg_config"]
        self.device = device
        self.env: TSVecEnv = env
        self.using_tsne_loss = False

        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        
        #using force guidance or not
        self.using_force_guidance = (
            self.cfg["force_guidance_config"]["using_force_guidance"] 
            if self.cfg["force_guidance_config"]["using_force_guidance"] is not None
            else False
        )
        
        self.using_time_mode = (
            True if self.encoder_config["teacher_encoder_name"] in ["UNet", "MLP-UNet"]
            else False
        )
        
        self.using_world_model = self.cfg["world_model_config"]["using_world_model"]

        #using encoder or not
        self.using_encoder = self.encoder_config["algorithms"] is not None
        
        #get all observations
        actions_zeros = torch.zeros(self.env.num_envs, self.env.num_actions, device=self.device, requires_grad=False)
        obs, privileged_obs, teacher_encoder_obs, student_encoder_obs, _, _, self.extras = self.env.step(actions_zeros)

        #actor-critic obs number init
        num_obs = obs.size(1)
        if privileged_obs is not None:
            num_critic_obs = privileged_obs.size(1)
        else:
            num_critic_obs = obs.size(1)

        if self.using_encoder:
            #get encoder input and output
            #get encoder input size
            teacher_encoder_input_size = teacher_encoder_obs.size(1)
            student_encoder_input_size = student_encoder_obs.size(1)
            #num obs and num privileged obs changed
            num_obs, num_critic_obs = (num_obs + self.env.layer_size), (num_critic_obs + self.env.layer_size)

        actor_critic_class = get_net_class(self.cfg["policy_class_name"])  # ActorCritic
        actor_critic = actor_critic_class(
            num_actor_obs = num_obs, num_critic_obs = num_critic_obs, num_actions = self.env.num_actions, 
            **self.policy_cfg
        ).to(self.device)
        
        #encoder init and set
        if self.using_encoder:
            #get encoder class
            #if the teacher encoder use identity, the algo is changed to state observer
            teacher_encoder_class = (
                IdentityEncoder if self.encoder_config["teacher_encoder_name"] == "Identity"
                else get_encoder_class(self.encoder_config["teacher_encoder_name"])
            )
            student_encoder_class = get_encoder_class(self.encoder_config["student_encoder_name"])
            #teacher encoder alg class get
            #if the alg class is None, the teacher encoder will be updated use ppo
            #get student encoder alg class
            student_encoder_alg_class = get_algo_class(self.encoder_config["student_encoder_alg_class_name"])

            #judge the inference mode:
            #create encoder
            self.teacher_encoder: ModulesBase = (
                teacher_encoder_class(
                    teacher_encoder_input_size, self.env.layer_size
                ).to(self.device) if self.encoder_config["teacher_encoder_name"] == "Identity"
                else teacher_encoder_class(
                    **self.teacher_encoder_cfg, layer_size=self.env.layer_size, encoder_input_size=teacher_encoder_input_size, 
                ).to(self.device)
            )
            if self.encoder_config["algorithms"] == "RMA-Teacher":
                #set inference mode
                self.inference_mode: str = "RMA-Teacher"    
                self.encoder_save_type = ["teacher"]
            elif self.encoder_config["algorithms"] == "RMA-TeacherStudent":
                self.inference_mode: str = "RMA-TeacherStudent"
                self.encoder_save_type = ["teacher", "student"]
                #create teacher and student encoder
                #student encoder
                self.student_encoder: ModulesBase = student_encoder_class(
                    encoder_input_size = student_encoder_input_size, layer_size = self.env.layer_size, 
                    **self.student_encoder_cfg
                ).to(self.device) 
                #encoder alg init
                self.student_encoder_alg: AlgorithmBase = student_encoder_alg_class(
                    self.num_steps_per_env, self.env.num_envs, self.student_encoder, self.teacher_encoder,
                    **self.encoder_alg_cfg, device=self.device
                )
            elif self.encoder_config["algorithms"] == "CTS":
                #虽然现在是复制和粘贴，方便了添加新算法，但是为了后续代码的简洁必须修改
                #inference mode init
                self.inference_mode: str = "CTS"
                #encoder save type init
                self.encoder_save_type = ["teacher", "student"]
                #student encoder
                self.student_encoder: ModulesBase = student_encoder_class(
                    encoder_input_size = student_encoder_input_size, layer_size = self.env.layer_size, 
                    **self.student_encoder_cfg
                ).to(self.device)
                #encoder alg init
                self.student_encoder_alg: AlgorithmBase = student_encoder_alg_class(
                    self.num_steps_per_env, self.env.num_envs, self.student_encoder, self.teacher_encoder,
                    **self.encoder_alg_cfg, device=self.device
                )
            else:
                raise ValueError("can't find algorithm in settings, please check the algo's name!", self.cfg["algorithms"])          
        else:
            #delete the teacher encoder and use only the basement reinforcement learning
            self.inference_mode = "Basement-Reinforcement-Learning"
            self.encoder_save_type = [None]
            self.teacher_encoder = None
            self.using_time_mode = False
            self.using_force_guidance = False

        #get alg class 
        alg_class = get_algo_class(self.cfg["algorithm_class_name"])
        self.alg: AlgorithmBase = alg_class(
            actor_critic, self.teacher_encoder, **self.symmetrical_cfg, device=self.device, 
            inference_mode = self.inference_mode, num_teacher = self.env.num_envs_teacher, using_force_guidance = self.using_force_guidance,
            **self.alg_cfg
        )

        # alg init
        print(f"the algorithms used is: {self.inference_mode}")

        # init storage and model
        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            [num_obs],
            [num_critic_obs],
            [self.env.num_actions],
            self.using_force_guidance,
        )
        
        if self.using_world_model:
            #world model class get
            world_model_class = get_encoder_class(self.cfg["world_model_config"]["world_model_type"])
            #world model init
            self.world_model: ModulesBase = world_model_class(
                encoder_input_size = num_critic_obs,
                layer_size = num_critic_obs - self.env.layer_size + 1,
                **self.cfg["world_model_config"]["model_config"],
            ).to(self.device)
            #future state buffer init
            self.alg.storage.world_model_params_init(
                self.cfg["world_model_config"]["buffer_max_lenth"], num_critic_obs - self.env.layer_size + 1
            )
            self.alg.world_model_init(self.world_model)

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        _, _ = self.env.reset()

    def learn(self, num_learning_iterations, init_at_random_ep_len=False, experiment_log=None):
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)

        if experiment_log is not None:
            # write experiment log to tensorboard recursively
            for key, value in experiment_log.items():
                self.writer.add_text(key, str(value), 0)

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )
        #actor-critic obs get
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs, = obs.to(self.device), critic_obs.to(self.device)
        self.alg.actor_critic.train()  # switch to train mode (for dropout for example)
        teacher_encoder_obs = None
        
        time_buf = None
        force_buf = None

        if self.using_encoder:
            #encoder obs get and cat
            teacher_encoder_obs = self.env.get_teacher_encoder_observations()
            student_encoder_obs = self.env.get_student_encoder_observations()
            teacher_encoder_obs, student_encoder_obs = teacher_encoder_obs.to(self.device), student_encoder_obs.to(self.device)
            
            #cat layer and obs 
            st = (self.teacher_encoder.forward(teacher_encoder_obs, self.extras["time"])
                  if self.encoder_config["teacher_encoder_name"] in ["UNet", "MLP-UNet"] 
                  else self.teacher_encoder.forward(teacher_encoder_obs)["zt"]
                  ) 
            
            #get zt
            if self.inference_mode == "RMA-Teacher":
                zt = (self.teacher_encoder.forward(teacher_encoder_obs, self.extras["time"])
                  if self.encoder_config["teacher_encoder_name"] in ["UNet", "MLP-UNet"] 
                  else self.teacher_encoder.forward(teacher_encoder_obs)["zt"]
                  ) 
                self.teacher_encoder.train()
            elif self.inference_mode in ["RMA-TeacherStudent", "CTS"]:
                zt = self.student_encoder.forward(student_encoder_obs)["zt"]
                #encoder training mode
                self.teacher_encoder.train()
                self.student_encoder.train()

            if self.inference_mode in ["RMA-Teacher", "RMA-TeacherStudent"]:
                #cat the encoder layer output and actor observation
                obs = torch.cat((zt, obs), dim=-1)
            elif self.inference_mode == "CTS":
                self.num_teacher = self.env.num_envs_teacher
                teacher_obs = torch.cat((st[:self.num_teacher, :], obs[:self.num_teacher, :]), dim=-1)
                student_obs = torch.cat((zt[self.num_teacher: self.env.num_envs, :], obs[self.num_teacher: self.env.num_envs, :]), dim=-1)
                obs = torch.cat((teacher_obs, student_obs), dim=0)

            critic_obs = torch.cat((st,critic_obs), dim=-1)

            if self.using_force_guidance:
                force_buf = self.extras["force"]

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                #开始采集数据
                collect_data_num_steps = self.cfg["world_model_config"]["buffer_max_lenth"] if self.using_world_model else self.num_steps_per_env
                for i in range(collect_data_num_steps):
                    
                    #using obs to get actions
                    actions = (self.alg.act(obs, critic_obs, teacher_encoder_obs, time_buf, force_buf)
                               if i < self.num_steps_per_env
                               else self.alg.actor_critic.act(obs))

                    #encoder inference
                    if self.using_encoder:
                        teacher_encoder_obs, student_encoder_obs = teacher_encoder_obs.to(self.device), student_encoder_obs.to(self.device)
                        #get new zt, obs and critic obs
                        st = (self.teacher_encoder.forward(teacher_encoder_obs, self.extras["time"])
                            if self.encoder_config["teacher_encoder_name"] in ["UNet", "MLP-UNet"] 
                            else self.teacher_encoder.forward(teacher_encoder_obs)["zt"]
                            ) 
                        #get zt
                        if self.inference_mode == "RMA-Teacher":
                            zt = (self.teacher_encoder.forward(teacher_encoder_obs, self.extras["time"])
                                if self.encoder_config["teacher_encoder_name"] in ["UNet", "MLP-UNet"] 
                                else self.teacher_encoder.forward(teacher_encoder_obs)["zt"]
                                ) 
                        elif self.inference_mode in ["RMA-TeacherStudent", "CTS"]:
                            zt = self.student_encoder.forward(student_encoder_obs)["zt"]
                            if i < self.num_steps_per_env:
                                self.student_encoder_alg.act(st, student_encoder_obs, teacher_encoder_obs)
                                
                    
                    #get next obs, critic obs and teacher obs using last actions
                    obs, privileged_obs, teacher_encoder_obs, student_encoder_obs, rewards, dones, infos = self.env.step(actions)
                    force_buf = infos["force"] if self.using_force_guidance else None
                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs, critic_obs, rewards, dones = (
                        obs.to(self.device),
                        critic_obs.to(self.device),
                        rewards.to(self.device),
                        dones.to(self.device),
                    )
                    #save reward dones and infos                    
                    self.alg.process_env_step(rewards, dones, infos)                       
                    
                    #save next state for world model
                    if self.using_world_model:
                        self.alg.storage.add_next_state_for_world_model(
                            torch.cat((critic_obs, rewards.unsqueeze(1)), dim=-1)
                        )
                        
                    if self.using_encoder:
                        #cat layer and obs
                        if self.inference_mode in ["RMA-Teacher", "RMA-TeacherStudent"]:
                            #cat the encoder layer output and actor observation
                            obs = torch.cat((zt, obs), dim=-1)
                        elif self.inference_mode == "CTS":
                            self.num_teacher = self.env.num_envs_teacher
                            teacher_obs = torch.cat((st[:self.num_teacher, :], obs[:self.num_teacher, :]), dim=-1)
                            student_obs = torch.cat((zt[self.num_teacher: self.env.num_envs, :], obs[self.num_teacher: self.env.num_envs, :]), dim=-1)
                            obs = torch.cat((teacher_obs, student_obs), dim=0)
                        #get critic obs
                        critic_obs = torch.cat((st,critic_obs), dim=-1)

                    if self.log_dir is not None:
                        # Book keeping
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0
                    

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs, force_buf)
            #利用上面采集的数据进行更新
            loss_dict = self.alg.update()
            mean_value_loss = loss_dict["mean_value_loss"]
            mean_surrogate_loss = loss_dict["mean_surrogate_loss"]
            mean_symmetrical_loss = loss_dict["mean_symmetrical_loss"]
            mean_predict_loss = loss_dict.get("mean_predict_loss", 0)
            mean_kl_loss = loss_dict.get("mean_kl_loss", 0)
            mean_entropy_loss = loss_dict.get("mean_entropy_loss", 0)
            mean_tsne_loss = loss_dict.get("mean_tsne_loss", 0)

            encoder_loss_dict = {}
            if self.inference_mode in ["RMA-TeacherStudent", "CTS"]:
                encoder_loss_dict = self.student_encoder_alg.update()
                mean_mse_loss = encoder_loss_dict.get("mean_mse_loss", 0)

            stop = time.time()
            learn_time = stop - start
            if self.log_dir is not None:
                self.log(locals())
                #encoder save path init
                self.tsne_save_dir = os.path.join(self.log_dir, "tsne")
                if not os.path.exists(self.tsne_save_dir) and self.using_tsne_loss:
                    os.makedirs(self.tsne_save_dir)
                if self.using_tsne_loss and it % 100 == 0 and self.using_encoder:
                    with torch.no_grad():
                        z = self.teacher_encoder(teacher_encoder_obs)["zt"].cpu().numpy()
                        z_student = self.student_encoder(student_encoder_obs)["zt"].cpu().numpy()
                        heights = teacher_encoder_obs[:, 93].cpu().numpy()
                        
                        # 如果维度 > 2，使用 PCA 降到 2 维展示
                        if z.shape[1] > 2:
                            pca = PCA(n_components=2)
                            z_plot = pca.fit_transform(z)
                            z_student_plot = pca.fit_transform(z_student)
                            print(f"PCA explained variance: {pca.explained_variance_ratio_}")  # 看前两个维度保留了多少信息
                        else:
                            z_plot = z
                        
                        def filter_pca_outliers_robust(z_2d, heights, threshold=3.5):
                            """基于中位数和MAD的鲁棒离群点滤除"""
                            # 计算中位数（不受离群点影响）
                            median = np.median(z_2d, axis=0)
                            
                            # 计算 MAD (Median Absolute Deviation)
                            diff = np.abs(z_2d - median)
                            mad = np.median(diff, axis=0)
                            
                            # 避免除零：如果MAD为0（数据完全重合），设为1e-8
                            mad = np.where(mad < 1e-8, 1e-8, mad)
                            
                            # 计算修正z-score：0.6745 * (x - median) / MAD
                            # 0.6745 是正态分布中 MAD ≈ 0.6745*σ 的转换系数
                            modified_z_scores = 0.6745 * (z_2d - median) / mad
                            
                            # 取两个维度的最大偏差
                            max_score = np.max(np.abs(modified_z_scores), axis=1)
                            mask = max_score < threshold
                            
                            removed = np.sum(~mask)
                            print(f"Robust filtering: Removed {removed}/{len(mask)} outliers ({removed/len(mask)*100:.1f}%)")
                            return z_2d[mask], heights[mask], mask

                        # 使用：threshold 建议 3.5（对应正态分布的 3.5σ，约 0.05% 误杀率）
                        z_plot_clean, heights_clean, mask = filter_pca_outliers_robust(
                            z_plot, heights, threshold=5.5
                        )
                        
                        # # 对Teacher滤波，获取掩码
                        # z_plot_clean, heights_clean, mask = filter_pca_outliers(z_plot, heights, threshold=2.5)
                        
                        # Student使用相同的掩码（保持对应关系）
                        z_student_clean, heights_s_clean, _ = filter_pca_outliers_robust(
                            z_student_plot, heights, threshold=5.5
                        )
                        
                        # Teacher图（清洗后）
                        plt.figure(figsize=(8, 6))
                        plt.scatter(z_plot_clean[:, 0], z_plot_clean[:, 1], c=heights_clean, cmap='terrain', s=10)
                        plt.colorbar(label='Terrain Height')
                        plt.title(f'Iter {it} - PCA projection (filtered)')
                        save_path = os.path.join(self.tsne_save_dir, f'iter_{it}_filtered')
                        plt.savefig(save_path, dpi=300, bbox_inches='tight')
                        plt.close()
                        
                        # Student图（清洗后，使用相同掩码）
                        plt.figure(figsize=(8, 6))
                        plt.scatter(z_student_plot[:, 0], z_student_plot[:, 1], 
                                c=heights, cmap='terrain', s=10, marker='x')
                        plt.colorbar(label='Terrain Height')
                        plt.title(f'Student PCA projection (filtered)')
                        student_save_path = os.path.join(self.tsne_save_dir, f'student_iter_{it}_filtered')
                        plt.savefig(student_save_path, dpi=300, bbox_inches='tight')
                        plt.close()
                self.teacher_encoder_log_dir = os.path.join(self.log_dir, "teacher_encoder_model")
                self.student_encoder_log_dir = os.path.join(self.log_dir, "student_encoder_model")
                self.world_model_log_dir = os.path.join(self.log_dir, "world_model")
                if not os.path.exists(self.teacher_encoder_log_dir) and "teacher" in self.encoder_save_type:
                    os.makedirs(self.teacher_encoder_log_dir)
                if not os.path.exists(self.student_encoder_log_dir) and "student" in self.encoder_save_type:
                    os.makedirs(self.student_encoder_log_dir)
                if not os.path.exists(self.world_model_log_dir) and self.using_world_model:
                    os.makedirs(self.world_model_log_dir)

            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, "model_{}.pt".format(it)))
                for encoder_type in self.encoder_save_type:
                    if encoder_type == "teacher":
                        self.encoder_save(os.path.join(self.teacher_encoder_log_dir, "teacher_model_{}.pt".format(it)), encoder_type)
                    elif encoder_type == "student":
                        self.encoder_save(os.path.join(self.student_encoder_log_dir, "student_model_{}.pt".format(it)), encoder_type)
                if self.using_world_model:
                    self.all_model_save(os.path.join(self.world_model_log_dir, "world_model_{}.pt".format(it)), self.world_model, self.alg.world_model_optimizer)
            ep_infos.clear()

        self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir, "model_{}.pt".format(self.current_learning_iteration)))
        for encoder_type in self.encoder_save_type:
            if encoder_type == "teacher":
                self.encoder_save(os.path.join(self.teacher_encoder_log_dir, "teacher_model_{}.pt".format(self.current_learning_iteration)), encoder_type)
            elif encoder_type == "student":
                self.encoder_save(os.path.join(self.student_encoder_log_dir, "student_model_{}.pt".format(self.current_learning_iteration)), encoder_type)
        if self.using_world_model:
            self.all_model_save(os.path.join(self.world_model_log_dir, "world_model_{}.pt".format(it)), self.world_model, self.alg.world_model_optimizer)
        
    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        ep_string = f""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    # handle scalar and zero dimensional tensor infos
                    
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar("Episode/" + key, value, locs["it"])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs["collection_time"] + locs["learn_time"]))

        # self.writer.add_scalar("Loss/value_function", locs["mean_value_loss"], locs["it"])
        # self.writer.add_scalar("Loss/surrogate", locs["mean_surrogate_loss"], locs["it"])
        for key, value in locs["loss_dict"].items():
            self.writer.add_scalar(f"loss/{key}", value, locs["it"])
        self.writer.add_scalar(f"loss/predict_loss", locs["mean_predict_loss"], locs["it"])
        self.writer.add_scalar(f"loss/kl_loss", locs["mean_kl_loss"], locs["it"])
        self.writer.add_scalar(f"loss/entropy_loss", locs["mean_entropy_loss"], locs["it"])
        self.writer.add_scalar(f"loss/tsne_loss", locs["mean_tsne_loss"], locs["it"])  #mean_mse_loss
        # self.writer.add_scalar(f"loss/mean_mse_loss", locs["mean_mse_loss"], locs["it"])
        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])
        self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar("Perf/collection time", locs["collection_time"], locs["it"])
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])
        if len(locs["rewbuffer"]) > 0:
            self.writer.add_scalar("Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"])
            if self.using_force_guidance:
                self.writer.add_scalar("Train/mean_force_magnitude", locs["force_buf"].mean().item(), locs["it"])
            self.writer.add_scalar("Train/mean_episode_length", statistics.mean(locs["lenbuffer"]), locs["it"])
            self.writer.add_scalar("Train/mean_reward/time", statistics.mean(locs["rewbuffer"]), self.tot_time)
            self.writer.add_scalar("Train/mean_episode_length/time", statistics.mean(locs["lenbuffer"]), self.tot_time)

        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs["rewbuffer"]) > 0:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                f"""{'symmetrical loss:':>{pad}} {locs['mean_symmetrical_loss']:.4f}\n"""
                f"""{'predict loss:':>{pad}} {locs['mean_predict_loss']:.4f}\n"""
                # f"""{'mse loss:':>{pad}} {locs['mean_mse_loss']:.4f}\n"""
                f"""{'tsne loss:':>{pad}} {locs['mean_tsne_loss']:.4f}\n"""
                f"""{'kl loss:':>{pad}} {locs['mean_kl_loss']:.4f}\n"""
                f"""{'entropy loss:':>{pad}} {locs['mean_entropy_loss']:.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
            )
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                # f"""{'mirror loss:':>{pad}} {locs['mean_mirror_loss']:.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
            )
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
            f"""{'ETA:':>{pad}} {self.tot_time / (locs['it']-self.current_learning_iteration + 1) * (
                               locs['num_learning_iterations']+self.current_learning_iteration - locs['it']):.1f}s ("""
            f"""{self.tot_time / (locs['it']-self.current_learning_iteration + 1) * (
                               locs['num_learning_iterations']+self.current_learning_iteration - locs['it']) / 3600:.1f}h)\n"""
        )
        log_string += f"""{str.center(width, ' ')}\n""" f"""{'#' * width}\n"""

        print(log_string)

    def save(self, path, infos=None):
        if hasattr(self.alg, "save_extra"):
            infos = self.alg.save_extra()
        torch.save(
            {
                "model_state_dict": self.alg.actor_critic.state_dict(),
                "optimizer_state_dict": self.alg.optimizer.state_dict(),
                "iter": self.current_learning_iteration,
                "infos": infos,
            },
            path,
        )

    def encoder_save(self, path, encoder_type: str = "teacher", infos=None):
        if encoder_type == "teacher":
            torch.save(
                {
                    "model_state_dict": self.teacher_encoder.state_dict(),
                    "optimizer_state_dict": self.alg.optimizer.state_dict(),
                    "iter": self.current_learning_iteration,
                    "infos": infos,
                },
                path,
            )
        elif encoder_type == "student":
            torch.save(
                {
                    "model_state_dict": self.student_encoder.state_dict(),
                    "optimizer_state_dict": self.student_encoder_alg.student_optimizer.state_dict(),
                    "iter": self.current_learning_iteration,
                    "infos": infos,
                },
                path,
            )
        elif encoder_type is None:
            pass
        
    def all_model_save(self, path, model: nn.Module, optimizer: torch.optim.Optimizer, infos=None):
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "iter": self.current_learning_iteration,
                "infos": infos,
            },
            path,
        )

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path, weights_only=True)
        
        # for name, param in self.alg.actor_critic.named_parameters():
        #     print(f"Parameter: {name}, mean: {param.data.mean():.4f}, std: {param.data.std():.4f}")
        
        self.alg.actor_critic.load_state_dict(loaded_dict["model_state_dict"])
        # self.alg.actor_critic.double_std()
        # self.alg.actor_critic.double_std()
        # self.alg.actor_critic.double_std()
        # self.alg.actor_critic.double_std()
        # for name, param in self.alg.actor_critic.named_parameters():
        #     print(f"Parameter: {name}, mean: {param.data.mean():.4f}, std: {param.data.std():.4f}")

        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        self.current_learning_iteration = loaded_dict["iter"]
        if hasattr(self.alg, "load_extra"):
            self.alg.load_extra(loaded_dict["infos"])
        return loaded_dict["infos"]
    
    def encoder_load(self, path, load_type: str) -> None:
        loaded_dict = torch.load(path, map_location="cuda:0", weights_only=True)
        if load_type == "teacher":
            self.teacher_encoder.load_state_dict(loaded_dict["model_state_dict"])
            print("teacher encoder loaded successfully!")
        elif load_type == "student":
            self.student_encoder.load_state_dict(loaded_dict["model_state_dict"])
            # self.student_encoder_alg.student_optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
            print("student encoder loaded successfully!")
        return loaded_dict["infos"]

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference
    
    def get_encoder_inference_policy(self, device=None):
        if self.inference_mode in ["RMA-Teacher"]:
            self.teacher_encoder.eval()
            if device is not None:
                self.teacher_encoder.to(device)
            return (self.teacher_encoder.backward, self.inference_mode)
        elif self.inference_mode in ["RMA-TeacherStudent", "CTS"]:
            self.student_encoder.eval()
            if device is not None:
                self.student_encoder.to(device)
            return (self.student_encoder.backward, self.inference_mode)
        else:
            return (None, self.inference_mode)

    def get_teacher_encoder_policy(self, device=None):
        if self.teacher_encoder is not None:
            self.teacher_encoder.eval()
            if device is not None:
                self.teacher_encoder.to(device)
            return self.teacher_encoder.forward
        else:
            return None 