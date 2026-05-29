"""Training script for getup_gym."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import isaacgym

from getup_gym.utils.registry import task_registry
from getup_gym.common.helpers import get_args


def train(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env, env_cfg = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg
    )
    runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    args = get_args()
    train(args)
