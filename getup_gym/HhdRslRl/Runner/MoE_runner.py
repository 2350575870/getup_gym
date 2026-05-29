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
from getup_gym.HhdRslRl.Modules.moe import MoeActorCritic

class MoERunner(RunnerBase):
    def __init__(self, env: TSVecEnv, train_cfg, log_dir=None, device="cpu"):
        
        self.cfg = train_cfg["runner"]
        
        #cfg for experts init
        self.policy_cfg = self.cfg["policy_config"]
        self.encoder_config = self.cfg["encoder_training_setting"]
        self.teacher_encoder_cfg = self.encoder_config["teacher_encoder_config"]
        self.student_encoder_cfg = self.encoder_config["student_encoder_config"]
        
        #cfg for moe and moe-alg
        self.moe_cfg = self.cfg["moe_config"]
        self.moe_alg_cfg = self.cfg["moe_alg_cfg"]
        
        self.device = device
        
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        
        #obs num init
        #actor-critic obs number init
        #get all observations
        actions_zeros = torch.zeros(self.env.num_envs, self.env.num_actions, device=self.device, requires_grad=False)
        obs, privileged_obs, teacher_encoder_obs, student_encoder_obs, _, _, self.extras = self.env.step(actions_zeros)
        num_obs = obs.size(1)
        if privileged_obs is not None:
            num_critic_obs = privileged_obs.size(1)
        else:
            num_critic_obs = obs.size(1)
            
        #get encoder input and output
        #get encoder input size
        teacher_encoder_input_size = teacher_encoder_obs.size(1)
        student_encoder_input_size = student_encoder_obs.size(1)
        #num obs and num privileged obs changed
        num_obs, num_critic_obs = (num_obs + self.env.layer_size), (num_critic_obs + self.env.layer_size)

        
        #expert model init
        actor_critic_class = get_net_class("ActorCritic")
        
        
        teacher_encoder_class = (
                IdentityEncoder if self.encoder_config["teacher_encoder_name"] == "Identity"
                else get_encoder_class(self.encoder_config["teacher_encoder_name"])
            )
        student_encoder_class = get_encoder_class(self.encoder_config["student_encoder_name"])
        
        #create encoder
        self.teacher_encoder: ModulesBase = (
            teacher_encoder_class(
                teacher_encoder_input_size, self.env.layer_size, self.device
            ).to(self.device) if self.encoder_config["teacher_encoder_name"] == "Identity"
            else teacher_encoder_class(
                **self.teacher_encoder_cfg, layer_size=self.env.layer_size, encoder_input_size=teacher_encoder_input_size, 
                device=self.device,
            ).to(self.device)
        )
        
        self.student_encoder: ModulesBase = student_encoder_class(
                encdoer_input_size = student_encoder_input_size, layer_size = self.env.layer_size, device = self.device, 
                **self.student_encoder_cfg
            ).to(self.device) 
        
        self.actor_critic_list = []
        self.teacher_encoder_list = []
        self.student_encoder_list = []
        self.policy_dict = {}
        
        for i in range(len(self.moe_cfg["policy_position_list"])):
            actor_critic: ActorCritic = actor_critic_class(
                num_obs, num_critic_obs, self.env.num_actions, **self.policy_cfg
            ).to(self.device)
            
            #create encoder
            teacher_encoder: ModulesBase = (
                teacher_encoder_class(
                    teacher_encoder_input_size, self.env.layer_size, self.device
                ).to(self.device) if self.encoder_config["teacher_encoder_name"] == "Identity"
                else teacher_encoder_class(
                    **self.teacher_encoder_cfg, layer_size=self.env.layer_size, encoder_input_size=teacher_encoder_input_size, 
                    device=self.device,
                ).to(self.device)
            )
            
            student_encoder: ModulesBase = student_encoder_class(
                encdoer_input_size = student_encoder_input_size, layer_size = self.env.layer_size, device = self.device, 
                **self.student_encoder_cfg
            ).to(self.device) 
            
            path = self.moe_cfg["policy_position_list"][i]
            teacher_encoder_path = self.moe_cfg["teacher_encoder_position_list"][i]
            student_encoder_path = self.moe_cfg["student_encoder_position_list"][i]
            
            self.actor_critic_list.append(self.load(path, actor_critic))
            self.teacher_encoder_list.append(self.encoder_load(teacher_encoder_path, teacher_encoder))
            self.student_encoder_list.append(self.encoder_load(student_encoder_path, student_encoder))
            
        self.policy_dict["actor_critic"] = self.actor_critic_list
        self.policy_dict["teacher_encoder"] = self.teacher_encoder_list
        self.policy_dict["student_encoder"] = self.student_encoder_list
            
        self.moe = MoeActorCritic(self.policy_dict, device=self.device)

        

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
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs, teacher_encoder_obs, time_buf, force_buf)
                    obs, privileged_obs, teacher_encoder_obs, student_encoder_obs, rewards, dones, infos = self.env.step(actions)
                    force_buf = infos["force"] if self.using_force_guidance else None
                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs, critic_obs, rewards, dones = (
                        obs.to(self.device),
                        critic_obs.to(self.device),
                        rewards.to(self.device),
                        dones.to(self.device),
                    )
                    
                    if self.using_encoder:
                        teacher_encoder_obs, student_encoder_obs = teacher_encoder_obs.to(self.device), student_encoder_obs.to(self.device)
                        #get new zt, obs and critic obs
                        st = (self.teacher_encoder.forward(teacher_encoder_obs, self.extras["time"])
                            if self.encoder_config["teacher_encoder_name"] in ["UNet", "MLP-UNet"] 
                            else self.teacher_encoder.forward(teacher_encoder_obs)["zt"]* 0.3
                            ) 
                        #get zt
                        if self.inference_mode == "RMA-Teacher":
                            zt = (self.teacher_encoder.forward(teacher_encoder_obs, self.extras["time"])
                                if self.encoder_config["teacher_encoder_name"] in ["UNet", "MLP-UNet"] 
                                else self.teacher_encoder.forward(teacher_encoder_obs)["zt"]* 0.3
                                ) 
                        elif self.inference_mode in ["RMA-TeacherStudent", "CTS"]:
                            zt = self.student_encoder.forward(student_encoder_obs)["zt"]* 0.3
                            self.student_encoder_alg.act(st, zt, student_encoder_obs)

                        #cat layer and obs
                        if self.inference_mode in ["RMA-Teacher", "RMA-TeacherStudent"]:
                            #cat the encoder layer output and actor observation
                            obs = torch.cat((zt, obs), dim=-1)
                        elif self.inference_mode == "CTS":
                            self.num_teacher = self.env.num_envs_teacher
                            teacher_obs = torch.cat((st[:self.num_teacher, :], obs[:self.num_teacher, :]), dim=-1)
                            student_obs = torch.cat((zt[self.num_teacher: self.env.num_envs, :], obs[self.num_teacher: self.env.num_envs, :]), dim=-1)
                            obs = torch.cat((teacher_obs, student_obs), dim=0)

                        critic_obs = torch.cat((st,critic_obs), dim=-1)

                    self.alg.process_env_step(rewards, dones, infos)

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

            encoder_loss_dict = {}
            if self.inference_mode in ["RMA-TeacherStudent", "CTS"]:
                encoder_loss_dict = self.student_encoder_alg.update()
                mean_mse_loss = encoder_loss_dict["mean_mse_loss"]

            stop = time.time()
            learn_time = stop - start
            if self.log_dir is not None:
                self.log(locals())
                #encoder save path init
                self.teacher_encoder_log_dir = os.path.join(self.log_dir, "teacher_encoder_model")
                self.student_encoder_log_dir = os.path.join(self.log_dir, "student_encoder_model")
                if not os.path.exists(self.teacher_encoder_log_dir) and "teacher" in self.encoder_save_type:
                    os.makedirs(self.teacher_encoder_log_dir)
                if not os.path.exists(self.student_encoder_log_dir) and "student" in self.encoder_save_type:
                    os.makedirs(self.student_encoder_log_dir)

            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, "model_{}.pt".format(it)))
                for encoder_type in self.encoder_save_type:
                    if encoder_type == "teacher":
                        self.encoder_save(os.path.join(self.teacher_encoder_log_dir, "teacher_model_{}.pt".format(it)), encoder_type)
                    elif encoder_type == "student":
                        self.encoder_save(os.path.join(self.student_encoder_log_dir, "student_model_{}.pt".format(it)), encoder_type)
            ep_infos.clear()

        self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir, "model_{}.pt".format(self.current_learning_iteration)))
        for encoder_type in self.encoder_save_type:
            if encoder_type == "teacher":
                self.encoder_save(os.path.join(self.teacher_encoder_log_dir, "teacher_model_{}.pt".format(self.current_learning_iteration)), encoder_type)
            elif encoder_type == "student":
                self.encoder_save(os.path.join(self.student_encoder_log_dir, "student_model_{}.pt".format(self.current_learning_iteration)), encoder_type)

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
        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])
        self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar("Perf/collection time", locs["collection_time"], locs["it"])
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])
        if len(locs["rewbuffer"]) > 0:
            self.writer.add_scalar("Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"])
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

    def load(self, path, model: ActorCritic):
        loaded_dict = torch.load(path, weights_only=True)
        model.load_state_dict(loaded_dict["model_state_dict"])
        return model
    
    def encoder_load(self, path, encoder_model: ModulesBase, load_type: str) -> None:
        loaded_dict = torch.load(path, weights_only=True)
        encoder_model.load_state_dict(loaded_dict["model_state_dict"])
        return encoder_model

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
