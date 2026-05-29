#这是一个reward的总管理函数，所有的reward都需要在这里注册和计算
#为什么需要这个功能：主要是方便后续实现奖励分层，同时简化主环境类的代码
#主要功能：注册奖励函数，计算奖励，奖励分层
#类：RewardManager

from typing import Callable, Any, Dict
import torch
from dataclasses import dataclass
import torch
from dataclasses import MISSING, field
from getup_gym.common.base_task import BaseTask

@dataclass
class RewardManagerCfg:
    """config of reward"""

    func: Callable[..., torch.Tensor] = MISSING  # reward function

    weight: float = MISSING  # weight of the reward

    params: Dict[str, Any] = field(default_factory=dict) #params needed in the reward function

class RewardManager:
    def __init__(self, num_envs, reward_list: list, sim_dt: float, device):
        self.device = device
        # store reward functions and weights
        self.reward_dict = {}
        self.episode_sums = {
            name: torch.zeros(
                num_envs,
                dtype=torch.float,
                device=self.device,
                requires_grad=False,
            )
            for name in reward_list
        }
        self.sim_dt = sim_dt
        self.total_reward = torch.zeros(num_envs, device=self.device)

    def register_reward(self, cfg: RewardManagerCfg):
        """register a reward function
        Args:
            cfg (RewardManagerCfg): config of the reward
        """
        name = cfg.func.__name__
        self.reward_dict[name] = {"func": cfg.func, "weight": cfg.weight, "params": cfg.params}

    def compute_reward(self, env: BaseTask, reward_update_dict: dict) -> torch.Tensor:
        """compute the total reward
        Args:
            env (LeggedRobot): the environment
            reward_list (list[str]): list of reward names to compute
        Returns:
            torch.Tensor: total reward
            dict: episode_sums
        """

        #reward update use reward update dict
        # self.episode_sums = {}
        for name, update_weight in reward_update_dict.items():
            if name not in self.reward_dict:
                raise ValueError(f"Reward {name} not registered.")
            self.reward_dict[name]["weight"] = update_weight
            # self.episode_sums[name] = 0.0
            
        self.total_reward[:] = 0.0
        for name in reward_update_dict:
            if name not in self.reward_dict:
                raise ValueError(f"Reward {name} not registered.")
            reward_info = self.reward_dict[name]
            reward = reward_info["func"](env, **reward_info["params"])
            self.episode_sums[name] += reward_info["weight"] * reward * self.sim_dt
            self.total_reward += reward_info["weight"] * reward * self.sim_dt
        return self.total_reward, self.episode_sums
    
    def reset_episode_sums(self, key: str, envs_ids: torch.Tensor):
        """reset episode sums
        Args:
            done_buf (torch.Tensor): done buffer
        """
        self.episode_sums[key][envs_ids] = 0.0