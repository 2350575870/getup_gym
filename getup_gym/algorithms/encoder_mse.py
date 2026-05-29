"""MSE distillation for student encoder training."""

import torch
import torch.nn as nn
import torch.optim as optim
from getup_gym.storage.encoder_storage import EncoderStorage


class EncoderMSE:
    """Distill teacher encoder knowledge into student encoder via MSE."""

    def __init__(
        self,
        num_transition_per_env,
        num_envs,
        encoder,
        teacher_encoder,
        device,
        learning_rate=8e-4,
        num_mini_batches=3,
        num_learning_epochs=3,
        max_grad_norm=0.8,
        mse_loss_coef=1.0,
        **kwargs,
    ):
        self.num_transition_per_env = num_transition_per_env
        self.num_envs = num_envs
        self.student_encoder_input_size = encoder.encoder_input_size
        self.layer_size = encoder.layer_size
        self.device = device
        self.learning_rate = learning_rate
        self.num_mini_batch = num_mini_batches
        self.num_learning_epochs = num_learning_epochs
        self.max_grad_norm = max_grad_norm
        self.mse_loss_coef = mse_loss_coef

        self.encoder = encoder.to(device)
        self.teacher_encoder = teacher_encoder.to(device)
        self.student_optimizer = optim.Adam(self.encoder.parameters(), self.learning_rate)
        self.mse_function = nn.MSELoss()

        self.encoder_storage = EncoderStorage(
            num_transition_per_env,
            num_envs,
            self.layer_size,
            self.student_encoder_input_size,
            self.teacher_encoder.encoder_input_size,
            device,
        )
        self.encoder_transition = EncoderStorage.EncoderTransition()

    def act(self, zt_t, student_encoder_obs, teacher_encoder_obs=None):
        self.encoder_transition.zt_t = zt_t
        self.encoder_transition.student_obs = student_encoder_obs
        self.encoder_transition.teacher_obs = teacher_encoder_obs
        self.encoder_storage.add_transitions(self.encoder_transition)
        self.encoder_transition.clear()

    def update(self):
        mean_mse_loss = 0.0
        generator = self.encoder_storage.mini_batch_generator(
            self.num_mini_batch, self.num_learning_epochs
        )
        for zt_t_batch, student_obs_batch, teacher_obs_batch in generator:
            zt_s = self.encoder.backward(student_obs_batch)["zt"]
            # Use teacher_obs_batch to compute teacher latent, or directly use pre-computed zt_t_batch
            if teacher_obs_batch is not None:
                zt_t = self.teacher_encoder.backward(teacher_obs_batch)["zt"].detach()
            else:
                zt_t = zt_t_batch.detach()
            mse_loss = self.mse_function(zt_s, zt_t)
            loss = self.mse_loss_coef * mse_loss
            mean_mse_loss += mse_loss.item()

            self.student_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.encoder.parameters(), self.max_grad_norm)
            self.student_optimizer.step()

        num_updates = self.num_learning_epochs * self.num_mini_batch
        self.encoder_storage.clear()
        return {"mean_mse_loss": mean_mse_loss / num_updates}
