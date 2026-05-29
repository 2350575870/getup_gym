import torch.nn as nn
import torch
from abc import ABC, abstractmethod

class ModulesBase(ABC, nn.Module):

    "Base class for all modules"

    def __init__(self):
        super(ModulesBase, self).__init__()


    @abstractmethod
    def forward(self, obs: torch.Tensor) -> torch.Tensor:

        "Forward pass through the module"
        #input: 
        # observation tensor

        #output: 
        # inference tensor

        pass

    @abstractmethod
    def backward(self, obs: torch.Tensor) -> torch.Tensor:

        """Backward pass through the module, it need to be implemented if the module is untrainable"""
        
        #input: 
        # observation tensor

        #output: 
        # inference tensor

        pass

    def test(self) -> None:

        "test if the module is working properly(if the module not have gradient, it need to be implemented)"
        
        pass
