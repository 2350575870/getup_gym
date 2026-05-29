"""Task registry for environment registration."""

import os
from datetime import datetime
from typing import Type, Tuple

from getup_gym import GETUP_GYM_ROOT_DIR
from getup_gym.common.helpers import (
    get_args,
    update_cfg_from_args,
    class_to_dict,
    set_seed,
    parse_sim_params,
    get_load_path,
    get_teacher_encoder_load_path,
    get_student_encoder_load_path,
)


class TaskRegistry:
    """Registry for RL tasks and their configurations."""

    def __init__(self):
        self._tasks = {}

    def register(self, name: str, task_class: Type, task_config, train_config):
        self._tasks[name] = {
            "task_class": task_class,
            "task_config": task_config,
            "train_config": train_config,
        }

    def get_task_class(self, name: str) -> Type:
        return self._tasks[name]["task_class"]

    def get_task_config(self, name: str):
        return self._tasks[name]["task_config"]

    def get_train_config(self, name: str):
        return self._tasks[name]["train_config"]

    def get_cfgs(self, name: str):
        env_cfg = self.get_task_config(name)
        train_cfg = self.get_train_config(name)
        env_cfg.seed = train_cfg.seed
        return env_cfg, train_cfg

    def make_env(self, name: str, args=None, env_cfg=None):
        from isaacgym import gymapi, gymutil

        if args is None:
            args = get_args()
        task_class = self.get_task_class(name)
        if env_cfg is None:
            env_cfg, _ = self.get_cfgs(name)
        env_cfg, _ = update_cfg_from_args(env_cfg, None, args)
        set_seed(env_cfg.seed)
        sim_params = {"sim": class_to_dict(env_cfg.sim)}
        sim_params = parse_sim_params(args, sim_params)
        env = task_class(
            cfg=env_cfg,
            sim_params=sim_params,
            physics_engine=gymapi.SIM_PHYSX,
            sim_device=args.sim_device,
            headless=args.headless,
        )
        return env, env_cfg

    def make_alg_runner(self, env, name=None, args=None, train_cfg=None, log_root="default"):
        from getup_gym.runners.on_policy_runner import OnPolicyRunner

        if args is None:
            args = get_args()
        if train_cfg is None:
            if name is None:
                raise ValueError("Either 'name' or 'train_cfg' must be not None")
            _, train_cfg = self.get_cfgs(name)
        else:
            if name is not None:
                print(f"'train_cfg' provided -> Ignoring 'name={name}'")
        _, train_cfg = update_cfg_from_args(None, train_cfg, args)

        if log_root == "default":
            log_root = os.path.join(GETUP_GYM_ROOT_DIR, "logs", name)
            log_dir = os.path.join(
                log_root,
                datetime.now().strftime("%b%d_%H-%M-%S") + "_" + train_cfg.runner.run_name,
            )
        elif log_root is None:
            log_dir = None
        else:
            log_dir = os.path.join(
                log_root,
                datetime.now().strftime("%b%d_%H-%M-%S") + "_" + train_cfg.runner.run_name,
            )

        train_cfg_dict = class_to_dict(train_cfg)
        runner = OnPolicyRunner(env, train_cfg_dict, log_dir, device=args.rl_device)

        if train_cfg.runner.resume:
            resume_path = get_load_path(
                log_root,
                load_run=train_cfg.runner.load_run,
                checkpoint=train_cfg.runner.checkpoint,
            )
            print(f"Loading model from: {resume_path}")
            runner.load(resume_path)
            algorithms = train_cfg.runner.encoder_training_setting.algorithms
            if algorithms in ["RMA-TeacherStudent", "CTS"]:
                teacher_path = get_teacher_encoder_load_path(
                    log_root,
                    load_run=train_cfg.runner.load_run,
                    checkpoint=train_cfg.runner.checkpoint,
                )
                student_path = get_student_encoder_load_path(
                    log_root,
                    load_run=train_cfg.runner.load_run,
                    checkpoint=train_cfg.runner.checkpoint,
                )
                runner.encoder_load(teacher_path, "teacher")
                runner.encoder_load(student_path, "student")
            elif algorithms in ["RMA-Teacher"]:
                teacher_path = get_teacher_encoder_load_path(
                    log_root,
                    load_run=train_cfg.runner.load_run,
                    checkpoint=train_cfg.runner.checkpoint,
                )
                runner.encoder_load(teacher_path, "teacher")
        return runner, train_cfg


task_registry = TaskRegistry()
