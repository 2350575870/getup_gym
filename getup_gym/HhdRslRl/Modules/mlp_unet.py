import torch
import torch.nn as nn
from getup_gym.HhdRslRl.basement.base_modules.ModulesBase import ModulesBase

class MlpUNet(ModulesBase, nn.Module):
    
    def __init__(
        self, 
        encoder_input_size: int,
        layer_size: int,
        botteneck_dim: int,
        activation: str,
        time_embed_dim: list,
        device,
    ):
        super(MlpUNet, self).__init__()
        self.encoder_input_size = encoder_input_size
        self.layer_size = layer_size
        self.botteneck_dim = botteneck_dim
        self.activation = get_activation(activation)
        self.device = device
        
        self.encoder = self._block(self.encoder_input_size, botteneck_dim, activation)
        
        time_embed = []
        for dim in range(len(time_embed_dim) - 2):
            time_embed.append(nn.Linear(time_embed_dim[dim], time_embed_dim[dim + 1]))
            time_embed.append(get_activation(activation))
        time_embed.append(nn.Linear(time_embed_dim[-2], time_embed_dim[-1]))
        self.time_embed = nn.Sequential(*time_embed)
        
        self.decoder = self._block(
            botteneck_dim, self.layer_size, activation 
        )
        
    def forward(self, obs: torch.tensor, t: torch.tensor) -> list:
        """the encoder forward function, which is needed to be define

        Args:
            obs (torch.tensor): encoder's input observation
            t (torch.tensor): time observation

        Raises:
            ValueError: the encoder's botteneck dim is not equal to the time embed dim,

        Returns:
            list: zt's value list
        """
        encoder_output = self.encoder(obs)
        time_embed_output = self.time_embed(t)
        if encoder_output.size(1) != time_embed_output.size(1):
            raise ValueError("the encoder output size is not equal to time embed output! please check your encoder")
        decoder_input_obs = encoder_output + time_embed_output
        output = self.decoder(decoder_input_obs)
        
        return output
    
    def backward(self, obs: torch.tensor, t: torch.tensor) -> list:
        """the encoder forward function, which is needed to be define

        Args:
            obs (torch.tensor): encoder's input observation
            t (torch.tensor): time observation

        Raises:
            ValueError: the encoder's botteneck dim is not equal to the time embed dim,

        Returns:
            list: zt's value list
        """
        encoder_output = self.encoder(obs)
        time_embed_output = self.time_embed(t)
        if encoder_output.size(1) != time_embed_output.size(1):
            raise ValueError("the encoder output size is not equal to time embed output! please check your encoder")
        decoder_input_obs = encoder_output + time_embed_output
        output = self.decoder(decoder_input_obs)
        
        return output
    
    def _block(self, input_size: int, output_size: int, activation: str) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(input_size, output_size),
            get_activation(activation)
        )
        
        
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