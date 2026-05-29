"""Storage for encoder distillation (teacher-student)."""

import torch


class EncoderStorage:
    """Storage for teacher-student encoder distillation."""

    class EncoderTransition:
        def __init__(self):
            self.zt_t = None
            self.student_obs = None
            self.teacher_obs = None

        def clear(self):
            self.__init__()

    def __init__(
        self,
        num_transition_per_env,
        num_envs,
        output_shape,
        student_encoder_input_shape,
        teacher_encoder_input_shape,
        device,
    ):
        self.device = device
        self.num_envs = num_envs
        self.num_transition_per_env = num_transition_per_env
        self.zt_t = torch.zeros(num_transition_per_env, num_envs, output_shape, device=device)
        self.student_obs = torch.zeros(num_transition_per_env, num_envs, student_encoder_input_shape, device=device)
        self.teacher_obs = (
            torch.zeros(num_transition_per_env, num_envs, teacher_encoder_input_shape, device=device)
            if teacher_encoder_input_shape is not None
            else None
        )
        self.step = 0

    def add_transitions(self, transition: EncoderTransition):
        if self.step >= self.num_transition_per_env:
            raise AssertionError("Encoder storage overflow")
        self.zt_t[self.step] = transition.zt_t
        self.student_obs[self.step] = transition.student_obs
        if self.teacher_obs is not None and transition.teacher_obs is not None:
            self.teacher_obs[self.step] = transition.teacher_obs
        self.step += 1

    def clear(self):
        self.step = 0

    def mini_batch_generator(self, num_mini_batches, num_epochs):
        batch_size = self.num_envs * self.num_transition_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(mini_batch_size * num_mini_batches, requires_grad=False, device=self.device)

        zt_t = self.zt_t.flatten(0, 1)
        student_obs = self.student_obs.flatten(0, 1)
        teacher_obs = self.teacher_obs.flatten(0, 1) if self.teacher_obs is not None else None

        for _ in range(num_epochs):
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_id = indices[start:end]
                yield (
                    zt_t[batch_id],
                    student_obs[batch_id],
                    teacher_obs[batch_id] if teacher_obs is not None else None,
                )
