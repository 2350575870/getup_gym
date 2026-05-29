"""On-policy runner for teacher-student training with force guidance."""

import time
import os
from collections import deque
import statistics

from torch.utils.tensorboard import SummaryWriter
import torch

from getup_gym.modules.actor_critic import ActorCritic
from getup_gym.modules.encoder import TeacherEncoder, StudentEncoder
from getup_gym.algorithms.ts_ppo import TSPPO
from getup_gym.algorithms.encoder_mse import EncoderMSE


class OnPolicyRunner:
    """Runner for on-policy teacher-student RL training."""

    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        self.cfg = train_cfg["runner"]
        self.encoder_cfg = self.cfg["encoder_training_setting"]
        self.alg_cfg = self.cfg["algorithm_config"]
        self.policy_cfg = self.cfg["policy_config"]
        self.device = device
        self.env = env

        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        force_cfg = self.cfg.get("force_guidance_config", {})
        self.using_force_guidance = force_cfg.get("using_force_guidance", False) if force_cfg else False

        # Initialize environment to get observation shapes
        actions_zeros = torch.zeros(self.env.num_envs, self.env.num_actions, device=self.device, requires_grad=False)
        obs, privileged_obs, teacher_encoder_obs, student_encoder_obs, _, _, self.extras = self.env.step(actions_zeros)

        num_obs = obs.size(1)
        num_critic_obs = privileged_obs.size(1) if privileged_obs is not None else num_obs

        # Encoder setup
        self.using_encoder = self.encoder_cfg["algorithms"] is not None
        if self.using_encoder:
            teacher_input_size = teacher_encoder_obs.size(1)
            student_input_size = student_encoder_obs.size(1)
            num_obs += self.env.layer_size
            num_critic_obs += self.env.layer_size

            self.teacher_encoder = TeacherEncoder(
                encoder_input_size=teacher_input_size,
                layer_size=self.env.layer_size,
                **self.encoder_cfg["teacher_encoder_config"],
            ).to(self.device)
            self.student_encoder = StudentEncoder(
                encoder_input_size=student_input_size,
                layer_size=self.env.layer_size,
                **self.encoder_cfg["student_encoder_config"],
            ).to(self.device)
            self.student_encoder_alg = EncoderMSE(
                self.num_steps_per_env,
                self.env.num_envs,
                self.student_encoder,
                self.teacher_encoder,
                device=self.device,
                **self.encoder_cfg["student_encoder_alg_config"],
            )
            self.inference_mode = "CTS"
            self.encoder_save_type = ["teacher", "student"]
        else:
            self.inference_mode = "Basement-Reinforcement-Learning"
            self.encoder_save_type = [None]
            self.teacher_encoder = None
            self.student_encoder = None

        # Actor-Critic
        self.actor_critic = ActorCritic(
            num_actor_obs=num_obs,
            num_critic_obs=num_critic_obs,
            num_actions=self.env.num_actions,
            **self.policy_cfg,
        ).to(self.device)

        # Algorithm
        self.alg = TSPPO(
            self.actor_critic,
            self.teacher_encoder,
            device=self.device,
            inference_mode=self.inference_mode,
            num_teacher=getattr(self.env, "num_envs_teacher", 0),
            using_force_guidance=self.using_force_guidance,
            **self.alg_cfg,
        )
        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            [num_obs],
            [num_critic_obs],
            [self.env.num_actions],
            self.using_force_guidance,
        )

        # Logging
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        _ = self.env.reset()

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        self.alg.actor_critic.train()

        teacher_encoder_obs = None
        student_encoder_obs = None
        force_buf = self.extras.get("force") if self.using_force_guidance else None

        if self.using_encoder:
            teacher_encoder_obs = self.env.get_teacher_encoder_observations()
            student_encoder_obs = self.env.get_student_encoder_observations()
            teacher_encoder_obs = teacher_encoder_obs.to(self.device)
            student_encoder_obs = student_encoder_obs.to(self.device)

            st = self.teacher_encoder.forward(teacher_encoder_obs)["zt"]
            zt = self.student_encoder.forward(student_encoder_obs)["zt"]

            num_teacher = self.env.num_envs_teacher
            teacher_obs = torch.cat((st[:num_teacher, :], obs[:num_teacher, :]), dim=-1)
            student_obs = torch.cat((zt[num_teacher:, :], obs[num_teacher:, :]), dim=-1)
            obs = torch.cat((teacher_obs, student_obs), dim=0)
            critic_obs = torch.cat((st, critic_obs), dim=-1)

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
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs, teacher_encoder_obs, force=force_buf)

                    if self.using_encoder:
                        teacher_encoder_obs = teacher_encoder_obs.to(self.device)
                        student_encoder_obs = student_encoder_obs.to(self.device)
                        st = self.teacher_encoder.forward(teacher_encoder_obs)["zt"]
                        zt = self.student_encoder.forward(student_encoder_obs)["zt"]
                        if i < self.num_steps_per_env:
                            self.student_encoder_alg.act(st, student_encoder_obs, teacher_encoder_obs)

                    obs, privileged_obs, teacher_encoder_obs, student_encoder_obs, rewards, dones, infos = self.env.step(actions)
                    force_buf = infos.get("force") if self.using_force_guidance else None
                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs, critic_obs, rewards, dones = (
                        obs.to(self.device),
                        critic_obs.to(self.device),
                        rewards.to(self.device),
                        dones.to(self.device),
                    )
                    self.alg.process_env_step(rewards, dones, infos)

                    if self.using_encoder:
                        num_teacher = self.env.num_envs_teacher
                        teacher_obs = torch.cat((st[:num_teacher, :], obs[:num_teacher, :]), dim=-1)
                        student_obs = torch.cat((zt[num_teacher:, :], obs[num_teacher:, :]), dim=-1)
                        obs = torch.cat((teacher_obs, student_obs), dim=0)
                        critic_obs = torch.cat((st, critic_obs), dim=-1)

                    if self.log_dir is not None:
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

            loss_dict = self.alg.update()
            encoder_loss_dict = {}
            if self.using_encoder:
                encoder_loss_dict = self.student_encoder_alg.update()

            stop = time.time()
            learn_time = stop - start

            if self.log_dir is not None:
                self.log(it, num_learning_iterations, collection_time, learn_time, loss_dict, encoder_loss_dict,
                         ep_infos, rewbuffer, lenbuffer, force_buf)
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))
                    if "teacher" in self.encoder_save_type:
                        self.encoder_save(os.path.join(self.log_dir, "teacher_encoder_model", f"model_{it}.pt"), "teacher")
                    if "student" in self.encoder_save_type:
                        self.encoder_save(os.path.join(self.log_dir, "student_encoder_model", f"model_{it}.pt"), "student")
                ep_infos.clear()

        self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def log(self, it, num_learning_iterations, collection_time, learn_time, loss_dict, encoder_loss_dict,
            ep_infos, rewbuffer, lenbuffer, force_buf, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += collection_time + learn_time
        iteration_time = collection_time + learn_time

        ep_string = ""
        if ep_infos:
            for key in ep_infos[0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in ep_infos:
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar(f"Episode/{key}", value, it)
                ep_string += f"{f'Mean episode {key}:':>{pad}} {value:.4f}\n"

        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (collection_time + learn_time))

        for key, value in loss_dict.items():
            self.writer.add_scalar(f"Loss/{key}", value, it)
        for key, value in encoder_loss_dict.items():
            self.writer.add_scalar(f"Loss/{key}", value, it)
        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, it)
        self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), it)
        self.writer.add_scalar("Perf/total_fps", fps, it)
        self.writer.add_scalar("Perf/collection_time", collection_time, it)
        self.writer.add_scalar("Perf/learning_time", learn_time, it)

        if len(rewbuffer) > 0:
            self.writer.add_scalar("Train/mean_reward", statistics.mean(rewbuffer), it)
            if self.using_force_guidance and force_buf is not None:
                self.writer.add_scalar("Train/mean_force_magnitude", force_buf.mean().item(), it)
            self.writer.add_scalar("Train/mean_episode_length", statistics.mean(lenbuffer), it)

        log_string = (
            f"{'#' * width}\n"
            f"{'Learning iteration':>{pad}} {it}/{self.current_learning_iteration + num_learning_iterations}\n"
            f"{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {collection_time:.3f}s, learning: {learn_time:.3f}s)\n"
        )
        for key, value in loss_dict.items():
            log_string += f"{key:>{pad}} {value:.4f}\n"
        for key, value in encoder_loss_dict.items():
            log_string += f"{key:>{pad}} {value:.4f}\n"
        log_string += (
            f"{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"
            f"{ep_string}"
            f"{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"
            f"{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"
            f"{'Total time:':>{pad}} {self.tot_time:.2f}s\n"
            f"{'#' * width}\n"
        )
        print(log_string)

    def save(self, path, infos=None):
        torch.save(
            {
                "model_state_dict": self.alg.actor_critic.state_dict(),
                "optimizer_state_dict": self.alg.optimizer.state_dict(),
                "iter": self.current_learning_iteration,
                "infos": infos,
            },
            path,
        )

    def encoder_save(self, path, encoder_type="teacher", infos=None):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if encoder_type == "teacher":
            torch.save(
                {
                    "model_state_dict": self.teacher_encoder.state_dict(),
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

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path, map_location=self.device, weights_only=True)
        self.alg.actor_critic.load_state_dict(loaded_dict["model_state_dict"])
        if load_optimizer and "optimizer_state_dict" in loaded_dict:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        self.current_learning_iteration = loaded_dict.get("iter", 0)
        return loaded_dict.get("infos")

    def encoder_load(self, path, encoder_type="teacher"):
        loaded_dict = torch.load(path, map_location=self.device, weights_only=True)
        if encoder_type == "teacher":
            self.teacher_encoder.load_state_dict(loaded_dict["model_state_dict"])
        elif encoder_type == "student":
            self.student_encoder.load_state_dict(loaded_dict["model_state_dict"])
            if "optimizer_state_dict" in loaded_dict:
                self.student_encoder_alg.student_optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        self.current_learning_iteration = loaded_dict.get("iter", 0)

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval()
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference

    def get_encoder_inference_policy(self, device=None):
        if self.student_encoder is not None:
            self.student_encoder.eval()
            if device is not None:
                self.student_encoder.to(device)
            return self.student_encoder.forward, self.inference_mode
        return None, self.inference_mode
