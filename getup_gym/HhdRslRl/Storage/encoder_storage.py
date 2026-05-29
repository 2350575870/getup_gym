import torch
from getup_gym.HhdRslRl.basement.base_storage.StorageBase import StorageBase

class EncoderStorage(StorageBase):
    class EncoderTransition:
        def __init__(self):
            self.zt_t = None
            self.student_obs = None
            self.teacher_obs = None

        def clear(self):
            self.__init__()

    def __init__(self, num_trasition_per_env, num_envs, output_shape, student_encoder_input_shape, teacher_encoder_input_shape, device):
        # encoder所有数据的中转站和处理站
        # 主要完成一下几个任务
        #1、记录zt^(t)和zt^(s)的参数
        #2、将zt^(t)和zt^(s)的参数从单一batch转成多个mini_batch
        #3、zt^(t)需要与CTS_rollout_storage 与 CTSppo进行交互，将zt^(t)的mini_batch输出，并在PPO中参与更新
        self.device = device
        self.zt_t = torch.zeros(num_trasition_per_env, num_envs, output_shape, device=self.device)
        self.student_obs = torch.zeros(num_trasition_per_env, num_envs, student_encoder_input_shape, device=self.device)
        self.teacher_obs = torch.zeros(num_trasition_per_env, num_envs, teacher_encoder_input_shape, device=self.device)
        self.num_envs = num_envs
        self.num_trasition_per_env = num_trasition_per_env

        self.step = 0


    def add_transitions(self, transition: EncoderTransition):
        if (self.step > self.num_trasition_per_env):
            raise AssertionError("encoder storage overflow!")
        self.zt_t[self.step] = transition.zt_t
        self.student_obs[self.step] = transition.student_obs
        self.teacher_obs[self.step] = transition.teacher_obs if transition.teacher_obs is not None else None
        self.step += 1

    def clear(self):
        self.step = 0

    def mini_batch_generator(self, num_mini_batchs, num_epochs):
        """create encoder's generator"""
        batch_size = self.num_envs* self.num_trasition_per_env
        mini_batch_size = batch_size // num_mini_batchs
        indices = torch.randperm(mini_batch_size* num_mini_batchs, requires_grad=False, device=self.device)

        zt_t = self.zt_t.flatten(0,1)
        student_obs = self.student_obs.flatten(0,1)
        teacher_obs = self.teacher_obs.flatten(0,1) if self.teacher_obs is not None else None

        for epoch in range(num_epochs):
            for i in range(num_mini_batchs):
                start = i* mini_batch_size
                end = (i + 1)* mini_batch_size
                batch_id = indices[start:end]

                zt_t_batch = zt_t[batch_id]
                student_obs_batch = student_obs[batch_id]
                teacher_obs_batch = teacher_obs[batch_id] if teacher_obs is not None else None

                yield zt_t_batch, student_obs_batch, teacher_obs_batch

    def VAE_mini_batch_generator(self, num_mini_batchs, num_epoch):
        """create encoder's generator"""
        batch_size = self.num_envs* self.num_trasition_per_env
        mini_batch_size = batch_size // num_mini_batchs
        indices = torch.randperm(mini_batch_size* num_mini_batchs, requires_grad=False, device=self.device)

        zt_t = self.zt_t.flatten(0,1)
        student_obs = self.student_obs.flatten(0,1)
        teacher_obs = self.teacher_obs.flatten(0,1) if self.teacher_obs is not None else None

        for epoch in range(num_epoch):
            for i in range(num_mini_batchs):
                start = i* mini_batch_size
                end = (i + 1)* mini_batch_size
                batch_id = indices[start:end]

                zt_t_batch = zt_t[batch_id]
                student_obs_batch = student_obs[batch_id]
                teacher_obs_batch = teacher_obs[batch_id] if teacher_obs is not None else None

                yield zt_t_batch, student_obs_batch, teacher_obs_batch



        