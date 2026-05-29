"""Rollout storage for on-policy algorithms with teacher-student support."""

import torch


class RolloutStorage:
    """Storage for PPO rollouts supporting teacher-student and force guidance."""

    class Transition:
        def __init__(self):
            self.observations = None
            self.critic_observations = None
            self.actions = None
            self.rewards = None
            self.dones = None
            self.values = None
            self.actions_log_prob = None
            self.action_mean = None
            self.action_sigma = None
            self.teacher_encoder_obs = None
            self.hidden_states = None
            self.force = None

        def clear(self):
            self.__init__()

    def __init__(
        self,
        num_envs,
        num_transitions_per_env,
        obs_shape,
        privileged_obs_shape,
        actions_shape,
        teacher_encoder_input_size=None,
        using_force_guidance=False,
        device="cpu",
    ):
        self.device = device
        self.obs_shape = obs_shape
        self.privileged_obs_shape = privileged_obs_shape
        self.actions_shape = actions_shape
        self.num_envs = num_envs
        self.teacher_encoder_input_size = teacher_encoder_input_size
        self.has_encoder = teacher_encoder_input_size is not None
        self.using_force_guidance = using_force_guidance
        self.num_transitions_per_env = num_transitions_per_env

        self.observations = torch.zeros(num_transitions_per_env, num_envs, *obs_shape, device=device)
        if privileged_obs_shape[0] is not None:
            self.privileged_observations = torch.zeros(
                num_transitions_per_env, num_envs, *privileged_obs_shape, device=device
            )
        else:
            self.privileged_observations = None
        self.rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=device)
        self.actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=device)
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=device, dtype=torch.bool)
        self.actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1, device=device)
        self.values = torch.zeros(num_transitions_per_env, num_envs, 1, device=device)
        self.returns = torch.zeros(num_transitions_per_env, num_envs, 1, device=device)
        self.advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=device)
        self.mu = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=device)
        self.sigma = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=device)

        if self.has_encoder:
            self.teacher_encoder_obs_buf = torch.zeros(
                num_transitions_per_env, num_envs, teacher_encoder_input_size, device=device
            )
        if self.using_force_guidance:
            self.force = torch.zeros(num_transitions_per_env, num_envs, 1, device=device)

        self.step = 0

    def add_transitions(self, transition: Transition) -> None:
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")
        self.observations[self.step].copy_(transition.observations)
        if self.privileged_observations is not None:
            self.privileged_observations[self.step].copy_(transition.critic_observations)
        self.actions[self.step].copy_(transition.actions)
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.dones[self.step].copy_(transition.dones.view(-1, 1))
        self.values[self.step].copy_(transition.values)
        self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
        self.mu[self.step].copy_(transition.action_mean)
        self.sigma[self.step].copy_(transition.action_sigma)
        if self.has_encoder:
            self.teacher_encoder_obs_buf[self.step].copy_(transition.teacher_encoder_obs)
        if self.using_force_guidance and transition.force is not None:
            self.force[self.step].copy_(transition.force)
        self.step += 1

    def clear(self):
        self.step = 0

    def compute_returns(self, last_values, gamma, lam, last_force=None):
        advantage = 0
        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:
                next_values = last_values
            else:
                next_values = self.values[step + 1]
            next_is_not_terminal = 1.0 - self.dones[step].float()
            delta = self.rewards[step] + next_is_not_terminal * gamma * next_values - self.values[step]
            if self.using_force_guidance:
                if step == self.num_transitions_per_env - 1:
                    next_force = last_force
                else:
                    next_force = self.force[step + 1]
                delta += self.force[step] - next_is_not_terminal * gamma * next_force
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            self.returns[step] = advantage + self.values[step]

        self.advantages = self.returns - self.values
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def mini_batch_generator_together(self, num_mini_batches, num_epochs=8):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(mini_batch_size * num_mini_batches, requires_grad=False, device=self.device)

        obs = self.observations.flatten(0, 1)
        critic_obs = self.privileged_observations.flatten(0, 1) if self.privileged_observations is not None else obs
        actions = self.actions.flatten(0, 1)
        target_values = self.values.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        mu_flat = self.mu.flatten(0, 1)
        sigma_flat = self.sigma.flatten(0, 1)
        teacher_obs = self.teacher_encoder_obs_buf.flatten(0, 1) if self.has_encoder else None

        for _ in range(num_epochs):
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_idx = indices[start:end]

                yield (
                    obs[batch_idx],
                    critic_obs[batch_idx],
                    teacher_obs[batch_idx] if self.has_encoder else None,
                    None,
                    actions[batch_idx],
                    target_values[batch_idx],
                    advantages[batch_idx],
                    returns[batch_idx],
                    old_actions_log_prob[batch_idx],
                    mu_flat[batch_idx],
                    sigma_flat[batch_idx],
                    (None, None),
                    None,
                )
