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

import torch
import torch.nn as nn
import torch.optim as optim
from torchviz import make_dot
from torch.autograd import gradcheck
from getup_gym.HhdRslRl.Modules.actor_critic import ActorCritic
from getup_gym.HhdRslRl.basement.base_modules.ModulesBase import ModulesBase
from getup_gym.HhdRslRl.basement.base_algorithm.AlgorithmBase import AlgorithmBase
from getup_gym.HhdRslRl.Storage.TSrollout_storage import TSRolloutStorage
from getup_gym.HhdRslRl.tools.symmetrical_function import symmetrical_function
from collections import deque
import torch.nn.functional as F

class TSPPO:
    actor_critic: ActorCritic
    teacher_encoder: ModulesBase

    def __init__(
        self,
        actor_critic,
        teacher_encoder: ModulesBase,
        teacher_encoder_alg_type: str,
        using_symmetrical_loss = False,
        symmetrical_loss_coef = 0.1,
        normalize_advantage_per_mini_batch=False,
        num_learning_epochs=1,
        num_mini_batches=1,
        clip_param=0.2,
        gamma=0.998,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.0,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,
        device="cpu",
        inference_mode = "Basement-Reinforcement-Learning",
        using_force_guidance: bool = False,
        num_teacher: int = 0,
    ):
        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        # PPO components
        self.actor_critic = actor_critic
        self.teacher_encoder: ModulesBase | None = teacher_encoder
        
        self.actor_critic.to(self.device)
        self.storage = None  # initialized later

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch

        #symmetrical loss parameters
        self.using_symmetrical_loss = using_symmetrical_loss
        self.symmetrical_loss_coef = symmetrical_loss_coef

        #inference mode
        self.inference_mode = inference_mode
        self.num_teacher = num_teacher
        
        #encoder type
        self.teacher_encoder_type = "MLP"
            
        #using force guidance or not 
        self.using_force_guidance = using_force_guidance

        #optim.Adam need the parameter of the model used and learning_rate
        self.transition = TSRolloutStorage.Transition()
        if self.teacher_encoder is not None:
            self.teacher_encoder.to(self.device)
            self.optimizer = (
                optim.Adam((list(self.actor_critic.parameters()) + list(self.teacher_encoder.parameters())), learning_rate)
            )
            # + list(self.teacher_encoder.parameters())
        else:
            self.optimizer = optim.Adam(self.actor_critic.parameters(), learning_rate)
            
        #world model
        self.world_model = None
        
        #entropy init
        self.teacher_encoder_alg_type = teacher_encoder_alg_type
        self.alpha_optimizer = None
        if self.teacher_encoder_alg_type in ["KL-Entropy"]:
            self.alpha_optimizer = optim.Adam([self.teacher_encoder.log_alpha], lr=learning_rate)
            
        self.use_tsne_loss: bool = True
        self.tsne_loss_coef: float = 0.5
        self.tsne_perplexity: float = 30.0
        self.tsne_temperature: float = 1.0
        self.tsne_supervised: bool = False

    def init_storage(
        self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, using_force_guidance
    ) -> None:
        self.num_envs = num_envs
        self.storage = TSRolloutStorage(
            num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, 
            action_shape, self.teacher_encoder.encoder_input_size, using_force_guidance, self.device,
        ) if self.teacher_encoder is not None else TSRolloutStorage(
            num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, None, using_force_guidance, self.device,
        )
    
    def world_model_init(self, world_model: ModulesBase):
        self.world_model:ModulesBase = world_model
        self.world_model_optimizer = optim.Adam(
            list(self.world_model.parameters()) + list(self.teacher_encoder.parameters()), self.learning_rate
        )
        self.optimizer = optim.Adam(
            list(self.actor_critic.parameters()) + list(self.teacher_encoder.parameters()) + list(self.world_model.parameters()), self.learning_rate
        )

    def test_mode(self):
        self.actor_critic.test()

    def train_mode(self):
        self.actor_critic.train()

    def act(
            self, obs: torch.tensor, critic_obs: torch.tensor, teacher_encoder_obs: torch.tensor = None, 
            time: torch.tensor = None, force: torch.tensor = None,
    ) -> None:
        """teacher act and student act"""
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        # Compute the actions and values
        self.transition.actions = self.actor_critic.act(obs).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        if self.teacher_encoder is not None:
            self.transition.teacher_encoder_obs = teacher_encoder_obs
        if self.using_force_guidance:
            self.transition.force = force
        return self.transition.actions

    def process_env_step(
            self, 
            rewards, dones,infos,
    ) -> None:
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        # Bootstrapping on time outs
        if "time_outs" in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * infos["time_outs"].unsqueeze(1).to(self.device),
                1,
            )

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

    def compute_returns(self, last_critic_obs, last_force = None):
        last_values = self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam, last_force=last_force)

    def update(self) -> dict:
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_symmetrical_loss = 0
        mean_predict_loss = 0
        mean_kl_loss = 0
        mean_entropy_loss = 0
        mean_tsne_loss = 0
        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator_together(self.num_mini_batches, self.num_learning_epochs)
        for (
            obs_batch,
            critic_obs_batch,
            teacher_encoder_obs_batch,
            next_state_for_world_model,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hid_states_batch,
            masks_batch
        ) in generator:
            # #将actor中teacher layer 的变量替换成前向传播之后
            if self.inference_mode in ["RMA-Teacher", "RMA-TeacherStudent"]:
                zt_batch = self.teacher_encoder.backward(teacher_encoder_obs_batch)
                obs_batch[:, :self.teacher_encoder.layer_size] = zt_batch["zt"]
            elif self.inference_mode == "CTS":
                if hasattr(self.teacher_encoder, "time_embed"):
                    zt_batch = self.teacher_encoder.backward(teacher_encoder_obs_batch)
                    obs_batch[:self.num_teacher, :self.teacher_encoder.layer_size] = zt_batch[:self.num_teacher, :]
                else:
                    zt_batch = self.teacher_encoder.backward(teacher_encoder_obs_batch)
                    obs_batch[:self.num_teacher, :self.teacher_encoder.layer_size] = zt_batch["zt"][:self.num_teacher, :]
                
            if next_state_for_world_model is not None:
                next_state_predict = self.world_model.backward(critic_obs_batch)
                
            # ========== 基于地形高度的 t-SNE Loss ==========
            tsne_loss = 0
            if self.use_tsne_loss and self.teacher_encoder is not None:
                # 确保 zt 和 teacher_encoder_obs_batch 维度匹配（处理 num_teacher 切片情况）
                current_teacher_obs = teacher_encoder_obs_batch
                current_zt = zt_batch["zt"]
                
                # 计算基于高度的 t-SNE loss
                tsne_loss = self.compute_tsne_loss(
                    current_teacher_obs, 
                    current_zt, 
                    height_center_idx=93,  # 17*5 + 8 = 93，即 17列*5行 + 8列 = 中心点
                    subsample_size=512  # True=硬分桶，False=软高斯核
                )
            

            self.actor_critic.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            value_batch = self.actor_critic.evaluate(
                critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1]
            )
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            if self.normalize_advantage_per_mini_batch:
                advantages_batch = (advantages_batch - advantages_batch.mean()) / (
                    advantages_batch.std() + 1e-8
                )

            # KL
            if self.desired_kl != None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)

                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )#ppo截断
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            if self.using_symmetrical_loss:
                original_batch_size = obs_batch.shape[0]
                #get symmetrical_obs to calculate the symmetrical actions
                obs_cat_batch, _ = symmetrical_function(obs=obs_batch, actions=None)
                mean_actions_batch = self.actor_critic.act_inference(obs_cat_batch.clone())
                actions_mean_orig = mean_actions_batch[:original_batch_size]
                #action's symmetrical function
                _, actions_mean_symm_batch = symmetrical_function(obs=None, actions=mean_actions_batch[original_batch_size:])
                mse_loss = nn.MSELoss()
                if actions_mean_orig.size(1) != actions_mean_symm_batch.size(1):
                    raise ValueError(f"actions mean orig size {actions_mean_orig.size(1)} is not equal to actions mean symm batch size {actions_mean_symm_batch.size(1)}, please check your symmetrical function!")
                symmetrical_loss = mse_loss(
                    actions_mean_orig, actions_mean_symm_batch.detach()
                )
            else:
                symmetrical_loss = 0
                
            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean() + self.symmetrical_loss_coef * symmetrical_loss + self.tsne_loss_coef * tsne_loss

            #teacher loss
            if self.teacher_encoder_alg_type in ["KL", "KL-Entropy"]:
                mu_t = zt_batch["mu"]
                logvar_t = zt_batch["logvar"]
                kl_loss = -0.5 * torch.sum(
                    1 + logvar_t - mu_t.pow(2) - logvar_t.exp(), 
                    dim=-1
                ).mean()
                mean_kl_loss += kl_loss.item()
                if self.teacher_encoder_alg_type == "KL-Entropy":
                    alpha = self.teacher_encoder.alpha.squeeze()
                    entropy = self.teacher_encoder.entropy.squeeze()
                    kl_loss = kl_loss + torch.clamp(alpha, 1e-6, 1)* entropy  # 不用 -=，用普通赋值
                    mean_entropy_loss += entropy.item()
                
                loss += kl_loss
                
                if self.teacher_encoder_alg_type == "KL-Entropy":
                    # Update alpha
                    self.alpha_optimizer.zero_grad()
                    alpha_loss = -(self.teacher_encoder.log_alpha * (
                        self.teacher_encoder.entropy - self.teacher_encoder.target_entropy).detach()).mean()
                    alpha_loss.backward()
                    self.alpha_optimizer.step()
            
            if next_state_for_world_model is not None:
                # 确保所有梯度清零
                self.world_model_optimizer.zero_grad()
                self.optimizer.zero_grad()
                
                # 第一步：计算 world model 损失
                self.actor_critic.eval()
                predict_loss = F.mse_loss(next_state_predict["zt"], next_state_for_world_model)
                
                # 第二步：计算 PPO 损失（保留计算图）
                total_loss = loss + 0.1* predict_loss
                
                # # 第三步：同时进行两个反向传播
                # # 首先计算 world model 部分的梯度
                # predict_loss.backward(retain_graph=True)
                # # 然后计算 PPO 部分的梯度
                total_loss.backward()
                
                # 第四步：分别更新参数
                # 更新 world_model 和 teacher_encoder（通过 world_model_optimizer）
                nn.utils.clip_grad_norm_(list(self.world_model.parameters()) + list(self.teacher_encoder.parameters()), self.max_grad_norm)
                self.world_model_optimizer.step()
                
                # 更新 actor_critic 和 teacher_encoder（通过 optimizer）
                if self.teacher_encoder is not None:
                    # 注意：teacher_encoder 的梯度已经在 loss.backward() 中计算了
                    nn.utils.clip_grad_norm_(list(self.actor_critic.parameters()) + list(self.teacher_encoder.parameters()) + list(self.world_model.parameters()), self.max_grad_norm)
                
                else:
                    nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                self.actor_critic.train()
                
                #loss value save
                mean_predict_loss += predict_loss.item()
            else:
                # 只有 PPO 更新
                self.optimizer.zero_grad()
                loss.backward()
                if self.teacher_encoder is not None:
                    # + list(self.teacher_encoder.parameters())
                    nn.utils.clip_grad_norm_(list(self.actor_critic.parameters()) + list(self.teacher_encoder.parameters()), self.max_grad_norm)
                else:
                    nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            # mean_tsne_loss += tsne_loss.item()
            if self.using_symmetrical_loss:
                mean_symmetrical_loss += symmetrical_loss.item()
            else:
                mean_symmetrical_loss += 0
                

            # print("teacher encoder backward gradient is: ")
            # for name, param in self.teacher_encoder.named_parameters():
            #     if param.grad is not None:
            #         print(f"{name}: {param.grad.norm().item()}, shape: {param.shape}")
            #     else:
            #         print(f"{name}: None, shape is {param.shape}")

            # print("student encoder backward gradient is: ")
            # for name, param in self.encoder.student_encoder.named_parameters():
            #     if param.grad is not None:
            #         print(f"{name}: {param.grad.norm().item()}, shape: {param.shape}")
            #     else:
            #         print(f"{name}: None, shape is {param.shape}")

            # print("totol encoder gradient is: ")
            # for name, param in self.encoder.named_parameters():
            #     if param.grad is not None:
            #         print(f"{name}: {param.grad.norm().item()}, shape: {param.shape}")
            #     else:
            #         print(f"{name}: None, shape is {param.shape}")

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_symmetrical_loss /= num_updates
        mean_predict_loss /= num_updates
        mean_kl_loss /= num_updates
        mean_entropy_loss /= num_updates
        mean_tsne_loss /= num_updates

         # Print losses for debugging
        self.storage.clear()

        return {
            "mean_value_loss": mean_value_loss, 
            "mean_surrogate_loss": mean_surrogate_loss, 
            "mean_symmetrical_loss": mean_symmetrical_loss, 
            "mean_predict_loss": mean_predict_loss,
            "mean_kl_loss": mean_kl_loss,
            "mean_entropy_loss": mean_entropy_loss,
            "mean_tsne_loss": mean_tsne_loss,
        }
        
    def compute_tsne_loss(self, teacher_obs, z, height_center_idx=93, subsample_size=256):
        """
        内存优化的基于地形高度的 t-SNE Loss
        通过子采样避免 O(N^2) 显存占用
        
        Args:
            teacher_obs: [B, D] teacher encoder 输入（B可能很大，如4096）
            z: [B, latent_dim] encoder 输出
            subsample_size: 实际用于计算 t-SNE 的样本数（推荐 128-512）
        """
        batch_size = z.shape[0]
        device = z.device
        
        # 如果 batch 很小，直接计算；否则子采样
        if batch_size > subsample_size:
            # 随机采样（保证每次更新的样本不同，增加覆盖度）
            indices = torch.randperm(batch_size, device=device)[:subsample_size]
            teacher_obs = teacher_obs[indices]
            z = z[indices]
            batch_size = subsample_size
        
        # 提取中心点高度（detach，不需要梯度）
        height_map = teacher_obs[:, :187]
        center_height = height_map[:, height_center_idx].detach()
        
        # ========== 构建 P: 基于高度分桶的相似度 ==========
        num_buckets = 5
        h_min, h_max = center_height.min(), center_height.max()
        # 防止除零
        scale = h_max - h_min + 1e-8
        buckets = ((center_height - h_min) / scale * num_buckets).long().clamp(0, num_buckets-1)
        
        # 硬监督：同高度桶内样本相似度为1，否则为0，然后用 softmax 平滑
        same_bucket = (buckets.unsqueeze(1) == buckets.unsqueeze(0)).float()
        temperature = 0.5
        P = F.softmax(same_bucket / temperature, dim=1)
        
        # 对称化并归一化（标准 t-SNE 流程）
        P = (P + P.t()) / (2.0 * batch_size)
        P = P.clamp(min=1e-8).detach()  # 重要：P 是 target，不需要梯度
        
        # ========== 构建 Q: Latent 空间的 t-分布相似度 ==========
        # 使用更高效的方式计算 pairwise distance
        # z: [B, D]，计算 z @ z.t() 利用矩阵乘法优化
        z_norm = F.normalize(z, p=2, dim=1)
        # 利用 (a-b)^2 = a^2 + b^2 - 2ab，避免显式分配大矩阵
        z_sq = (z ** 2).sum(dim=1, keepdim=True)  # [B, 1]
        dist_z = z_sq + z_sq.t() - 2.0 * (z @ z.t())  # [B, B]
        dist_z = dist_z.clamp(min=0.0)  # 防止数值误差
        
        # Student-t 核（Cauchy 分布）
        Q = 1.0 / (1.0 + dist_z / self.tsne_temperature)
        
        # 排除自身相似度（置零对角线）
        mask_exclude_self = torch.eye(batch_size, device=device).bool()
        Q = Q.masked_fill(mask_exclude_self, 0.0)
        
        # 归一化
        Q = Q / Q.sum(dim=1, keepdim=True).clamp(min=1e-8)
        # 对称化
        Q = (Q + Q.t()) / (2.0 * batch_size)
        Q = Q.clamp(min=1e-8)
        
        # ========== KL 散度 ==========
        loss = (P * torch.log(P / Q)).sum()
        
        return loss
