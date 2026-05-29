from getup_gym.HhdRslRl.basement.base_modules.ModulesBase import ModulesBase
import torch.nn as nn
from torch.distributions import Normal
import torch

class StudentEncoder(ModulesBase):
    def __init__(
            self,
            encoder_input_size,
            layer_size,
            device,
            encoder_hidden_dims,
            activation,
    ):
        super().__init__()
        self.device = device
        self.encoder_input_size = encoder_input_size
        self.layer_size = layer_size
        self.activation = get_activation(activation)
        self.encoder_hidden_dims = encoder_hidden_dims

        encoder = []
        encoder.append(nn.Linear(encoder_input_size, self.encoder_hidden_dims[0]))
        encoder.append(self.activation)
        for dim in range(len(self.encoder_hidden_dims) - 1):
            encoder.append(nn.Linear(self.encoder_hidden_dims[dim], self.encoder_hidden_dims[dim + 1]))
            encoder.append(self.activation)
        encoder.append(nn.Linear(self.encoder_hidden_dims[-1], self.layer_size))
        self.encoder = nn.Sequential(*encoder)

        Normal.set_default_validate_args = False

        # self.init_weight(self.encoder, 0.01)

        print(f"encoder MLP is: {self.encoder}")

    def init_weight(sequential,scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales)
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]

    def forward(self, obs: torch.tensor) -> dict:
        """encoder forward method"""
        if obs.size(1) != self.encoder_input_size:
            raise ValueError(f"input obs size error! hopes input obs size is: {self.encoder_input_size}, but got {obs.size(1)}")
        zt = self.encoder(obs)
        return {"zt": zt}
    
    def backward(self, obs:torch.tensor) -> dict:
        """encoder forward method used in backward"""
        if obs.size(1) != self.encoder_input_size:
            raise ValueError(f"input obs size error! hopes input obs size is: {self.encoder_input_size}, but got {obs.size(1)}")
        zt = self.encoder(obs)
        return {"zt": zt}

def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None