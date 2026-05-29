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
from getup_gym.HhdRslRl.Storage.TSrollout_storage import TSRolloutStorage
from getup_gym.HhdRslRl.tools.symmetrical_function import symmetrical_function

class RMAPPO:
    actor_critic: ActorCritic
    teacher_encoder: ModulesBase

    def __init__(
        self,
        actor_critic,
        teacher_encoder: ModulesBase,
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
    ):
        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        # PPO components
        self.actor_critic = actor_critic
        self.teacher_encoder: ModulesBase = teacher_encoder
        self.teacher_encoder.to(self.device)
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


        #optim.Adam need the parameter of the model used and learning_rate
        self.transition = TSRolloutStorage.Transition()
        self.optimizer = optim.Adam((list(self.actor_critic.parameters()) + list(self.teacher_encoder.parameters())), learning_rate)

    def init_storage(
        self,
        num_envs,
        num_transitions_per_env,
        actor_obs_shape,
        critic_obs_shape,
        action_shape,
    ):
        self.num_envs = num_envs
        self.storage = TSRolloutStorage(
            num_envs,
            num_transitions_per_env,
            actor_obs_shape,
            critic_obs_shape,
            action_shape,
            self.teacher_encoder.encoder_input_size,
            self.device,
        )

    def test_mode(self):
        self.actor_critic.test()

    def train_mode(self):
        self.actor_critic.train()

    def act(
            self, obs: torch.tensor, critic_obs: torch.tensor, teacher_encoder_obs: list,
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
        self.transition.teacher_encoder_obs = teacher_encoder_obs
        self.transition.critic_observations = critic_obs
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

    def compute_returns(self, last_critic_obs):
        last_values = self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self) -> dict:
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_symmetrical_loss = 0
        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator_together(self.num_mini_batches, self.num_learning_epochs)
        for (
            obs_batch,
            teacher_encoder_obs_batch,
            critic_obs_batch,
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
            #将actor中teacher layer 的变量替换成前向传播之后
            zt_batch = self.teacher_encoder.backward(teacher_encoder_obs_batch)
            obs_batch[:, :self.teacher_encoder.layer_size] = zt_batch["zt"]

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

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean() + self.symmetrical_loss_coef* symmetrical_loss

            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(list(self.actor_critic.parameters()) + list(self.teacher_encoder.parameters()), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            if self.using_symmetrical_loss:
                mean_symmetrical_loss += symmetrical_loss.item()
            else:
                mean_symmetrical_loss += 0

            # print("teacher encoder backward gradient is: ")
            # for name, param in self.encoder.teacher_encoder_list.named_parameters():
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
        self.storage.clear()

        return {"mean_value_loss": mean_value_loss, "mean_surrogate_loss": mean_surrogate_loss, "mean_symmetrical_loss": mean_symmetrical_loss}
