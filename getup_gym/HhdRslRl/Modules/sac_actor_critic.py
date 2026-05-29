import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from getup_gym.HhdRslRl.basement.base_modules.ModulesBase import ModulesBase

class SACActorCritic(ModulesBase, nn.Module):
    """ActorCritic model class for SAC

    Args:
        ModulesBase (Module's base): Base class for all modules
    """
    
    def __init__(
        self, 
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        actor_hidden_dims: list = [256, 256, 256],
        critic_hidden_dims: list = [256, 256, 256],
        activation: str = "elu",
        action_bounds: int = 50,
    ) -> None:
        """init func. 
        in this SACActorCritic, the actor and critic are both MLP structure.
        To simplify the params setting, all of the critic use the same structure.

        Args:
            num_actor_obs (int): actor's observation dimension
            num_critic_obs (int): critic's observation dimension
            num_actions (int): action dimension
            actor_hidden_dims (list, optional): actor model hidden dims. Defaults to [256, 256, 256].
            critic_hidden_dims (list, optional): critic model hidden dims. Defaults to [256, 256, 256].
            activation (str, optional): activation func. Defaults to "elu".
            init_noise_std (float, optional): noise init param. Defaults to 1.0.
        """
        super(SACActorCritic, self).__init__()
        
        self.activation = get_activation(activation)
        
        # actor model
        actor_layers = []
        actor_layers.append(nn.Linear(num_actor_obs, actor_hidden_dims[0]))
        actor_layers.append(self.activation)
        for l in range(len(actor_hidden_dims - 1)):
            actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
            actor_layers.append(self.activation)
        self.actor_layer = nn.Sequential(*actor_layers)
        self.actor_mean = nn.Linear(actor_hidden_dims[-1], num_actions)
        self.actor_std = nn.Linear(actor_hidden_dims[-1], num_actions)
        
        #critic model
        #In SAC training, there are two critics and two target critics, We only use the critic network with the smallest 
        #target critic is used to alleviate high estimation problems
        self.critic_1 = self._model_block(input_dim=num_critic_obs, hidden_dims=critic_hidden_dims, output_dim=1)       
        self.critic_2 = self._model_block(input_dim=num_critic_obs, hidden_dims=critic_hidden_dims, output_dim=1)
        self.target_critic_1 = self._model_block(input_dim=num_critic_obs, hidden_dims=critic_hidden_dims, output_dim=1)       
        self.target_critic_2 = self._model_block(input_dim=num_critic_obs, hidden_dims=critic_hidden_dims, output_dim=1)
        
        #target critic param init
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        
        #alpha param init, we use log_alpha to optimize alpha
        self.log_alpha = torch.tensor(np.log(0.01), requires_grad=True, dtype=torch.float32)
        
        self.action_distribution = None  # to be set in the main code if needed
        self.action_bounds = action_bounds
        
        
    def _model_block(self, input_dim: int, hidden_dims: list, output_dim: int) -> nn.Module:
        """build the model block for critic

        Args:
            input_dim (int): _description_
            hidden_dims (list): _description_
            output_dim (int): _description_

        Returns:
            nn.Module: _description_
        """
        model_layers = []
        model_layers.append(nn.Linear(input_dim, hidden_dims[0]))
        model_layers.append(self.activation)
        for l in range(len(hidden_dims)):
            if l == len(hidden_dims) - 1:
                model_layers.append(nn.Linear(hidden_dims[l], output_dim))
            else:
                model_layers.append(nn.Linear(hidden_dims[l], hidden_dims[l + 1]))
                model_layers.append(self.activation)
        return nn.Sequential(*model_layers)
    
    def forward(self, obs: torch.tensor) -> tuple:
        """forward func.

        Args:
            obs (torch.tensor): observation input

        Returns:
            torch.tensor: action output
        """
        layer_midden_output = self.actor_layer(obs)
        action_mean = self.actor_mean(layer_midden_output)
        action_std = F.softplus(self.actor_std(layer_midden_output)) + 1e-7  # to avoid zero std
        self.action_distribution = Normal(action_mean, action_std)
        
        action_resample = self.action_distribution.rsample()  # reparameterization trick
        
        return (
            torch.tanh(action_resample) * self.action_bounds, self.get_action_log_prob(action_resample)
        )  # action output with bounds
        
    def get_action_log_prob(self, action_reample: torch.tensor) -> tuple:
        """get action log prob

        Args:
            obs (torch.tensor): observation input

        Returns:
            tuple: action and log prob
        """
        log_prob = self.action_distribution.log_prob(action_reample)
        log_prob -= torch.log(1 - torch.tanh(action_reample).pow(2) + 1e-7)  # correction for Tanh squashing
        # sum the log prob for each action dimension
        return log_prob.sum(-1, keepdim=True)
    
    def evaluate_critic(self, privileged_obs: torch.tensor) -> tuple:
        """evaluate the two critics

        Args:
            obs (torch.tensor): observation input

        Returns:
            tuple: the critic network with the smallest 
        """
        return self.critic_1(privileged_obs), self.critic_2(privileged_obs)
    
    def evaluate_target_critic(self, privileged_obs: torch.tensor) -> torch.tensor:
        """evaluate the two target critics

        Returns:
            tuple: the target critic network with the smallest 
        """
        return torch.min(self.target_critic_1(privileged_obs), self.target_critic_2(privileged_obs))       


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