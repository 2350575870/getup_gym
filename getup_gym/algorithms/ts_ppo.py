"""Teacher-Student PPO with optional force-guided (CPO) constraints."""

import torch
import torch.nn as nn
import torch.optim as optim


class TSPPO:
    """Teacher-Student Proximal Policy Optimization.

    Supports concurrent teacher-student training (CTS) mode where
    teacher-group agents use privileged teacher encoder observations
    while student-group agents use proprioceptive history observations.
    """

    def __init__(
        self,
        actor_critic,
        teacher_encoder,
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
        inference_mode="CTS",
        using_force_guidance=False,
        num_teacher=0,
        **kwargs,
    ):
        if kwargs:
            print(f"TSPPO: ignoring unexpected kwargs {list(kwargs.keys())}")
        self.device = device
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        self.actor_critic = actor_critic
        self.teacher_encoder = teacher_encoder
        self.actor_critic.to(self.device)
        if self.teacher_encoder is not None:
            self.teacher_encoder.to(self.device)
            self.optimizer = optim.Adam(
                list(self.actor_critic.parameters()) + list(self.teacher_encoder.parameters()),
                learning_rate,
            )
        else:
            self.optimizer = optim.Adam(self.actor_critic.parameters(), learning_rate)

        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.normalize_advantage_per_mini_batch = False

        self.inference_mode = inference_mode
        self.num_teacher = num_teacher
        self.using_force_guidance = using_force_guidance
        self.storage = None
        self.transition = None

    def init_storage(self, num_envs, num_transitions_per_env, obs_shape, critic_obs_shape, actions_shape, using_force_guidance):
        from getup_gym.storage.rollout_storage import RolloutStorage

        self.num_envs = num_envs
        self.transition = RolloutStorage.Transition()
        teacher_input_size = self.teacher_encoder.encoder_input_size if self.teacher_encoder is not None else None
        self.storage = RolloutStorage(
            num_envs,
            num_transitions_per_env,
            obs_shape,
            critic_obs_shape,
            actions_shape,
            teacher_input_size,
            using_force_guidance,
            self.device,
        )

    def test_mode(self):
        self.actor_critic.test()

    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, critic_obs, teacher_encoder_obs=None, time=None, force=None):
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        self.transition.actions = self.actor_critic.act(obs).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        if self.teacher_encoder is not None:
            self.transition.teacher_encoder_obs = teacher_encoder_obs
        if self.using_force_guidance:
            self.transition.force = force
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        if "time_outs" in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * infos["time_outs"].unsqueeze(1).to(self.device), 1
            )
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

    def compute_returns(self, last_critic_obs, last_force=None):
        last_values = self.actor_critic.evaluate(last_critic_obs).detach()
        if self.using_force_guidance and last_force is None:
            last_force = torch.zeros(self.num_envs, 1, device=self.device)
        self.storage.compute_returns(last_values, self.gamma, self.lam, last_force=last_force)

    def update(self) -> dict:
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0

        generator = self.storage.mini_batch_generator_together(self.num_mini_batches, self.num_learning_epochs)
        for (
            obs_batch,
            critic_obs_batch,
            teacher_encoder_obs_batch,
            _,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            _,
            _,
        ) in generator:
            # Inject teacher encoder latent for teacher-group agents
            if self.inference_mode == "CTS" and self.teacher_encoder is not None:
                zt_batch = self.teacher_encoder.backward(teacher_encoder_obs_batch)
                obs_batch[: self.num_teacher, : self.teacher_encoder.layer_size] = zt_batch["zt"][: self.num_teacher, :]

            self.actor_critic.act(obs_batch)
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            value_batch = self.actor_critic.evaluate(critic_obs_batch)
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            # KL adaptive learning rate
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1e-5)
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

            # Value loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

            self.optimizer.zero_grad()
            loss.backward()
            if self.teacher_encoder is not None:
                nn.utils.clip_grad_norm_(
                    list(self.actor_critic.parameters()) + list(self.teacher_encoder.parameters()),
                    self.max_grad_norm,
                )
            else:
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        self.storage.clear()

        return {
            "mean_value_loss": mean_value_loss / num_updates,
            "mean_surrogate_loss": mean_surrogate_loss / num_updates,
        }
