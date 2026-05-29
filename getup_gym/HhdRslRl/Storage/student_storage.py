from legged_gym.HhdRslRl.basement.base_storage import StorageBase
import torch

class StudentEncoderStorage(StorageBase):
    def __init__(
            self,
            num_envs,
            encoder_input_size,
            layer_size,
            num_transitions_per_env,
            device,
    ):
        super().__init__()
        self.num_envs = num_envs
        self.encoder_input_size = encoder_input_size
        self.layer_size = layer_size
        self.device = device
        self.num_transitions_per_env = num_transitions_per_env

        self.encoder_input_buf: torch.tensor = torch.zeros(self.num_transitions_per_env, self.num_envs, self.encoder_input_size, device=self.device)
        self.teacher_layer_buf = torch.zeros(self.num_transitions_per_env, self.num_envs, self.layer_size, device=self.device)

    def add_transition(self, transition):
        return super().add_transitions(transition)
    
    def mini_batch_generator(self, num_mini_batchs, num_epochs):
        """create mini batch generator"""
        batch_size = self.num_envs* self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batchs
        indices = torch.randperm(mini_batch_size* num_mini_batchs, requires_grad=False, device=self.device)

        encoder_input_buf = self.encoder_input_buf.flatten(0,1)
        teacher_layer_buf = self.teacher_layer_buf.flatten(0,1)

        for epoch in range(num_epochs):
            for batch_id in range(num_mini_batchs):
                start = batch_id* batch_size
                end = (batch_id + 1)* batch_size
                batch_idx = indices[start: end]

                encoder_input_buf_batch = encoder_input_buf[batch_idx]
                teacher_layer_buf_batch = teacher_layer_buf[batch_idx]

                yield encoder_input_buf_batch, teacher_layer_buf_batch


        

    
