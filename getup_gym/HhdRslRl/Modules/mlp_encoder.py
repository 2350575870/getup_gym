from getup_gym.HhdRslRl.basement.base_modules.ModulesBase import ModulesBase
import torch.nn as nn
from torch.distributions import Normal
import torch
import math

class MlpEncoder(ModulesBase):
    def __init__(
            self,
            encoder_input_size,
            layer_size,
            encoder_hidden_dims,
            activation,
            
    ):
        super().__init__()
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

    def forward(self, obs):
        """encoder forward method"""
        if obs.size(1) != self.encoder_input_size:
            raise ValueError(f"input obs size error! hopes input obs size is: {self.encoder_input_size}, but got {obs.size(1)}")
        zt = self.encoder(obs)
        return {"zt": zt* 0.4}
    
    def backward(self, obs):
        """encoder forward method used in backward"""
        if obs.size(1) != self.encoder_input_size:
            raise ValueError(f"input obs size error! hopes input obs size is: {self.encoder_input_size}, but got {obs.size(1)}")
        zt = self.encoder(obs)
        return {"zt": zt* 0.4}

class MlpEncoderVae(ModulesBase, nn.Module):
    def __init__(
            self,
            encoder_input_size,
            layer_size,
            encoder_hidden_dims,
            activation,     
    ):
        super().__init__()
        self.encoder_input_size = encoder_input_size
        self.layer_size = layer_size
        self.activation = get_activation(activation)
        self.encoder_hidden_dims = encoder_hidden_dims

        layer = []
        layer.append(nn.Linear(encoder_input_size, self.encoder_hidden_dims[0]))
        layer.append(self.activation)
        for dim in range(len(self.encoder_hidden_dims) - 1):
            layer.append(nn.Linear(self.encoder_hidden_dims[dim], self.encoder_hidden_dims[dim + 1]))
            layer.append(self.activation)
        
        self.layer = nn.Sequential(*layer)
        self.mu_layer = nn.Linear(self.encoder_hidden_dims[-1], self.layer_size)
        self.logvar_layer = nn.Linear(self.encoder_hidden_dims[-1], self.layer_size)
        
        #get the encoder's output distribution, in order to get the entropy of the output
        self.distribution = None
        
        #target entropy and log alpha init
        self._target_entropy = -float(self.layer_size)
        self.log_alpha = nn.Parameter(torch.zeros(1))

        Normal.set_default_validate_args = False

        # self.init_weight(self.encoder, 0.01)

        print(f"encoder MLP is: {self.layer}")
        
    def forward(self, obs: torch.tensor) -> tuple:
        """forward

        Args:
            obs (torch.tensor): observation input

        Returns:
            tuple: _description_
        """
        if obs.size(1) != self.encoder_input_size:
            raise ValueError(f"input obs size error! hopes input obs size is: {self.encoder_input_size}, but got {obs.size(1)}")
        h = self.layer(obs)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        
        logvar = torch.clamp(logvar, -20, 2)
        
        std_t = torch.exp(0.5* logvar)
        self.distribution = Normal(mu, std_t)
        zt = self.distribution.rsample()
        
        return {
            "zt": zt,
            "mu": mu,
            "logvar": logvar,
        }
        
    def backward(self, obs: torch.tensor) -> tuple:
        """forward

        Args:
            obs (torch.tensor): observation input

        Returns:
            tuple: _description_
        """
        if obs.size(1) != self.encoder_input_size:
            raise ValueError(f"input obs size error! hopes input obs size is: {self.encoder_input_size}, but got {obs.size(1)}")
        h = self.layer(obs)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        
        logvar = torch.clamp(logvar, -20, 2)
        
        std_t = torch.exp(0.5* logvar)
        self.distribution = Normal(mu, std_t)
        zt = self.distribution.rsample()
        
        return {
            "zt": zt* 0.4,
            "mu": mu,
            "logvar": logvar,
        }
        
    @property
    def alpha(self):
        return self.log_alpha.exp()
    
    @property
    def entropy(self):
        """calculate the encoder's output entropy"""
        if self.distribution is None:
            raise ValueError("distribution has not been created yet!")
        #entropy calculation
        entropy = self.distribution.entropy().sum(-1).mean()
        return entropy
    
    @property
    def target_entropy(self):
        return self._target_entropy
    
        
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