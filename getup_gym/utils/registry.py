"""Task registration for getup_gym environments."""

from getup_gym.envs.bipedal_wheeled.env import JROwheel
from getup_gym.envs.bipedal_wheeled.config import GetupCfg, GetUpPPO
from getup_gym.envs.unitree_humanoid.env import HumanoidGetUp
from getup_gym.envs.unitree_humanoid.config import HumanoidGetUpCfg, HumanoidGetUpPPO
from getup_gym.common.task_registry import task_registry

task_registry.register("BipedalWheeled", JROwheel, GetupCfg(), GetUpPPO())
task_registry.register("UnitreeHumanoid", HumanoidGetUp, HumanoidGetUpCfg(), HumanoidGetUpPPO())
