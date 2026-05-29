import torch
import torch.nn as nn
from getup_gym.HhdRslRl.basement.base_modules.ModulesBase import ModulesBase

class MoeActorCritic(ModulesBase, nn.Module):
    """moe type encoder, which use different policy as input

    Args:
        nn (_type_): model's basic class
        ModulesBase (_type_): encoder's basic class
    """
    def __init__(
        self, encoder_list: dict, device, policy_position_list: list = None
    ) -> None:
        """the moe encoder init function

        Args:
            policy_position_list (list[str]): is the list contain all of the policy's position.
            encoder_list (list[nn.Module]): it's the encoders which have't been initialize
            device (_type_): training device
        """
        self.policy_position_list = policy_position_list
        self.encoder_list = encoder_list
        self.device = device 
        self.encoder_list: list = encoder_list
        
        #encoder initialization
        if len(policy_position_list) != len(encoder_list):
            raise ValueError(
                f"the policy position given is not same to the encoder list. got {len(policy_position_list)} policy, but have {len(encoder_list)} encoders"
            )
        
        if policy_position_list is not None:
            for encoder_idx in range(len(policy_position_list)):
                loaded_dict = torch.load(policy_position_list[encoder_idx])
                self.encoder_list[encoder_idx].load_state_dict(loaded_dict["model_state_dict"])
                #changed to eval mode
                self.encoder_list[encoder_idx].eval()
                
        #scales value policy init
        self.scales_policy = nn.Parameter(torch.ones(len(self.encoder_list), device=self.device))
                
    def student_forward(self, obs_list: list, encoder_obs_list: list) -> torch.tensor:
        """the moe encoder forward function

        Args:
            obs_list (list[torch.tensor]): all of the obs will be set in the obs list
        """
        output = 0
        for encoder_dim in range(len(obs_list)):
            student_encoder_output = self.encoder_list["student_encoder"][encoder_dim].forward(encoder_obs_list[encoder_dim])
            expert_output = self.encoder_list["actor_critic"][encoder_dim].act(
                torch.cat([student_encoder_output, obs_list[encoder_dim]], dim=-1)
            )
            output += self.scales_policy[encoder_dim]* expert_output

        return output
    
    def teacher_forward(self, obs_list: list, encoder_obs_list: list) -> list:
        """the moe encoder forward function

        Args:
            obs_list (list[torch.tensor]): all of the obs will be set in the obs list
        """
        output = []
        for encoder_dim in range(len(obs_list)):
            teacher_encoder_output = self.encoder_list["teacher_encoder"][encoder_dim].forward(encoder_obs_list[encoder_dim])
            expert_output = self.encoder_list["actor_critic"][encoder_dim].act(
                torch.cat([teacher_encoder_output, obs_list[encoder_dim]], dim=-1)
            )
            output.append(expert_output)    

        return output
    
    def forward(self, obs_list: list, teacher_encoder_obs_list: list, student_encoder_obs_list: list):
        return (
            self.teacher_forward(obs_list, teacher_encoder_obs_list), 
            self.student_forward(obs_list, student_encoder_obs_list)
        )
    
    def backward(self, obs_list: list) -> tuple:
        """the moe encoder back function

        Args:
            obs_list (list[torch.tensor]): all of the obs will be set in the obs list
        """
        output = 0
        for encoder_dim in range(len(obs_list)):
            output += self.scales_policy[encoder_dim]* self.encoder_list[encoder_dim](obs_list[encoder_dim])
        return output
        
            
            
            