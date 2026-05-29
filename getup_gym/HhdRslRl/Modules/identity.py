import torch.nn as nn
import torch

class IdentityEncoder(nn.Identity):
    """the IdentityEncoder is inherits from nn.Identity, with encoder_input size add in it.

    Args:
        nn (_type_): nn.Indentity
    """
    def __init__(self, encoder_input_size: int, layer_size: int, device: torch.device):
        """init

        Args:
            encoder_input_size (int): the encoder's input size
        """
        super(IdentityEncoder, self).__init__()
        
        self.encoder_input_size = encoder_input_size
        
        self.layer_size = layer_size
        
        self.device = device
        
    def forward(self, obs: torch.tensor):
        """the IdentityEncoder forward once

        Args:
            input (torch.tensor): input tensor

        Returns:
            dict: use the input tensor as zt
        """
        return {"zt": obs}
    
    def backward(self, obs: torch.tensor):
        """the IdentityEncoder forward once

        Args:
            input (torch.tensor): input tensor

        Returns:
            dict: use the input tensor as zt
        """
        return {"zt": obs}