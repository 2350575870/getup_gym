import torch
import torch.nn as nn
from getup_gym.HhdRslRl.basement.base_modules.ModulesBase import ModulesBase
from torch.distributions import Normal

class ResBlock(nn.Module):
    """res net block

    Args:
        nn (_type_): base class
    """
    def __init__(self, hidden_dims: int, dropout_prop: float, activation: str):
        """res net min block

        Args:
            hidden_dims (int): model's width
            dropout_prop (float): dropout's proportion
        """
        super(ResBlock, self).__init__()
        self.hidden_dims = hidden_dims
        self.res_block = nn.Linear(hidden_dims, hidden_dims)
        self.norm = nn.LayerNorm(hidden_dims)
        self.dropout = nn.Dropout(dropout_prop)
        self.activation = get_activation(activation)
    
    def forward(self, layer_obs):
        return self.activation(self.norm(self.res_block(layer_obs)))
    
class ResBlockGroup(nn.Module):
    """min block 

    Args:
        nn (_type_): _description_
    """
    
    def __init__(self, hidden_dims: int, dropout_prop: float, activation: str, block_num: int):
        """block's group

        Args:
            hidden_dims (int): _description_
            dropout_prop (float): _description_
            activation (str): _description_
            block_num (int): there are block_num block's in one group
        """
        super(ResBlockGroup, self).__init__()
        self.hidden_dims = hidden_dims
        self.dropout_prop = dropout_prop
        self.activation = activation
        self.block_num = block_num
        
        self.block_group = nn.Sequential(
            *[ResBlock(hidden_dims, dropout_prop, activation) for _ in range(self.block_num)]
        )

    def forward(self, group_input_obs):
        """group forward

        Args:
            group_input_obs (torch.tensor): input obs

        Returns:
            torch.tensor: block group's output
        """
        return group_input_obs + self.block_group(group_input_obs)
        
class ResNetEncoder(ModulesBase, nn.Module):
    """model for deep model, using resnet, layer norm and swish activation function
    we all use dropout to make the training more stable

    Args:
        ModulesBase (_type_): model class basement
    """
    
    def __init__(
        self, 
        encoder_input_size: int, layer_size: int, num_hidden_dims: int, 
        hidden_dims: int, dropout_prop: float, activation: str, block_num: int,
    ):
        """resnet model init

        Args:
            encoder_input_size (int): encoder input size
            layer_size (int): encoder output size
            num_hidden_dims (int): encoder's hidden dims number
            hidden_dims (int): encoder hidden layer's dims
            device (torch.device): using device
        """
        super(ResNetEncoder, self).__init__()
        self.encoder_input_size = encoder_input_size
        self.layer_size = layer_size
        self.num_hidden_dims = num_hidden_dims
        self.hidden_dims = hidden_dims
        self.dropout_prop = dropout_prop
        self.activation = activation
        self.block_num = block_num
        
        if self.num_hidden_dims % block_num != 0:
            raise ValueError(
                f"num_hidden_dims must be divid by block num! num_hidden_dims is {self.num_hidden_dims}, block num is {self.block_num}."
            )
        
        self.num_group = int(self.num_hidden_dims / self.block_num)
        self.resnet_model = nn.Sequential(
            *[ResBlockGroup(hidden_dims, dropout_prop, activation, block_num) for _ in range(self.num_group)]
        )
        
        self.input_layer = nn.Linear(self.encoder_input_size, hidden_dims)
        self.output_layer = nn.Linear(hidden_dims, self.layer_size)
        
    def forward(self, obs):
        """_summary_

        Args:
            obs (torch.tensor): model input obs
        """
        return {"zt": self.output_layer(
            self.resnet_model(self.input_layer(obs))
        )}
    
    def backward(self, obs):
        """resnet backward function. because the model all can backward, the backward func is similar to forward func

        Args:
            obs (torch.tensor): input obs
        """
        return {"zt": self.output_layer(
            self.resnet_model(self.input_layer(obs))
        )}
        
class ResNetActorCritic(nn.Module):
    is_recurrent = False
    
    def __init__(
        self, num_actor_obs: int, actor_hidden_dims: int, dropout_prop: float, actor_activation: str, actor_num_hidden_dims: int, actor_block_num: int,
        num_critic_obs: int, critic_hidden_dims: list, critic_activation: str, 
        init_noise_std: float, num_actions: int
    ):
        """this class is for actor critic, different from the resnet encoder type class

        Args:
            num_actor_obs (int): _description_
            actor_hidden_dims (int): _description_
            dropout_prop (float): _description_
            actor_activation (str): _description_
            num_critic_obs (int): _description_
            critic_hidden_dims (list): _description_
            critic_activation (str): _description_
            init_noise_std (float): _description_
            num_actions (int): _description_
        """
        super(ResNetActorCritic, self).__init__()
        self.init_noise_std = init_noise_std
        
        #actor model init
        #we only make actor to have multi layers, because we think it is no need for critic to have a multi layer
        if actor_num_hidden_dims % actor_block_num != 0:
            raise ValueError(
                f"num_hidden_dims must be divid by block num! num_hidden_dims is {actor_num_hidden_dims}, block num is {actor_block_num}."
            )
        self.group_num = int(actor_num_hidden_dims / actor_block_num)
        self.actor_midden_resnet = nn.Sequential(
            *[ResBlockGroup(actor_hidden_dims, dropout_prop, actor_activation, actor_block_num) for _ in range(self.group_num)]
        )
        self.actor_input_layer = nn.Linear(num_actor_obs, actor_hidden_dims)
        self.actor_output_layer = nn.Linear(actor_hidden_dims, num_actions)
        
        self.actor = nn.Sequential(self.actor_input_layer, self.actor_midden_resnet, self.actor_output_layer)
        self.mlp_input_dim_a = num_actor_obs
        
        #critic model init
        critic_layers = []
        self.critic_activation = get_activation(critic_activation)
        critic_layers.append(nn.Linear(num_critic_obs, critic_hidden_dims[0]))
        critic_layers.append(self.critic_activation)
        for dim in range(len(critic_hidden_dims) - 1):
            critic_layers.append(nn.Linear(critic_hidden_dims[dim], critic_hidden_dims[dim + 1]))
            critic_layers.append(self.critic_activation)
        critic_layers.append(nn.Linear(critic_hidden_dims[-1], 1))
        self.critic = nn.Sequential(*critic_layers)
        
        print(f"critic MLP: {self.critic}")
        
        self.std = nn.Parameter(init_noise_std* torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False
        
    def actor_forward(self, actor_obs):
        """actor model forward

        Args:
            actor_obs (torch.tensor): actor model input

        Raises:
            NotImplementedError: _description_

        Returns:
            torch.tensor: actor model output
        """
        return self.actor_output_layer(
            self.actor_midden_resnet(self.actor_input_layer(actor_obs))
        )
        
    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales)
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        mean = self.actor_forward(observations)
        mean = mean.clamp(-10000, 10000)  # clamp to avoid numerical issues
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        actions_mean = self.actor(observations)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value

    def enforce_minimum_std(self, min_std):
        current_std = self.std.detach()
        new_std = torch.max(current_std, min_std.detach()).detach()
        self.std.data = new_std

    def double_std(self):
        current_std = self.std.detach()
        current_std *= 2
        current_std.detach()
        self.std.data = current_std
        self.std.data.requires_grad_(True)
        
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
    elif act_name == "swish":
        return nn.SiLU()
    else:
        print("invalid activation function!")
        return None