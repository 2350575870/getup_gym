"""Helper utilities for getup_gym."""

import os
import copy
import torch
import numpy as np
import random
from isaacgym import gymapi
from isaacgym import gymutil
from getup_gym import GETUP_GYM_ROOT_DIR


def class_to_dict(obj) -> dict:
    """Convert a class instance to a nested dictionary recursively."""
    if not hasattr(obj, "__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        val = getattr(obj, key)
        if callable(val) and not isinstance(val, type):
            continue
        if hasattr(val, "__dict__") or (hasattr(val, "__iter__") and not isinstance(val, (str, bytes))):
            result[key] = class_to_dict(val)
        else:
            result[key] = val
    return result


def update_class_from_dict(obj, d):
    for key, val in d.items():
        attr = getattr(obj, key, None)
        if isinstance(attr, type):
            update_class_from_dict(attr, val)
        else:
            setattr(obj, key, val)
    return obj


def set_seed(seed):
    if seed == -1:
        seed = np.random.randint(0, 10000)
    print("Setting seed: {}".format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_sim_params(args, cfg):
    sim_params = gymapi.SimParams()
    if args.physics_engine == gymapi.SIM_FLEX:
        if args.device != "cpu":
            print("WARNING: Using Flex with GPU instead of PHYSX!")
    elif args.physics_engine == gymapi.SIM_PHYSX:
        sim_params.physx.use_gpu = args.use_gpu
        sim_params.physx.num_subscenes = args.subscenes
    sim_params.use_gpu_pipeline = args.use_gpu_pipeline

    if "sim" in cfg:
        if cfg["sim"].get("up_axis") is not None:
            if cfg["sim"]["up_axis"] == "z":
                cfg["sim"]["up_axis"] = 1
            elif cfg["sim"]["up_axis"] == "y":
                cfg["sim"]["up_axis"] = 0
        if cfg["sim"]["physx"]["num_threads"] is None:
            cfg["sim"]["physx"]["num_threads"] = 10
        if cfg["sim"]["physx"]["solver_type"] is None:
            cfg["sim"]["physx"]["solver_type"] = 1
        if "num_subscenes" in cfg["sim"]["physx"]:
            if cfg["sim"]["physx"]["num_subscenes"] is None:
                cfg["sim"]["physx"]["num_subscenes"] = 4
        gymutil.parse_sim_config(cfg["sim"], sim_params)

    if args.physics_engine == gymapi.SIM_PHYSX and args.num_threads > 0:
        sim_params.physx.num_threads = args.num_threads
    return sim_params


def get_load_path(root, load_run=-1, checkpoint=-1):
    try:
        runs = os.listdir(root)
        runs.sort()
        if "exported" in runs:
            runs.remove("exported")
        last_run = os.path.join(root, runs[-1])
    except Exception:
        raise ValueError("No runs in this directory: " + root)
    if load_run == -1:
        load_run = last_run
    else:
        load_run = os.path.join(root, load_run)
    if checkpoint == -1:
        models = [file for file in os.listdir(load_run) if "model" in file]
        models.sort(key=lambda m: "{:0>15}".format(m))
        if "teacher_encoder_model" in models:
            models.remove("teacher_encoder_model")
        if "student_encoder_model" in models:
            models.remove("student_encoder_model")
        model = models[-1]
    else:
        model = "model_{}.pt".format(checkpoint)
    return os.path.join(load_run, model)


def get_teacher_encoder_load_path(root, load_run=-1, checkpoint=-1):
    try:
        runs = os.listdir(root)
        runs.sort()
        if "exported" in runs:
            runs.remove("exported")
        last_run = os.path.join(root, runs[-1])
    except Exception:
        raise ValueError("No runs in this directory: " + root)
    if load_run == -1:
        load_run = last_run
    else:
        load_run = os.path.join(root, load_run)
    encoder_dir = os.path.join(load_run, "teacher_encoder_model")
    if checkpoint == -1:
        models = [file for file in os.listdir(encoder_dir) if "model" in file]
        models.sort(key=lambda m: "{:0>15}".format(m))
        model = models[-1]
    else:
        model = "model_{}.pt".format(checkpoint)
    return os.path.join(encoder_dir, model)


def get_student_encoder_load_path(root, load_run=-1, checkpoint=-1):
    try:
        runs = os.listdir(root)
        runs.sort()
        if "exported" in runs:
            runs.remove("exported")
        last_run = os.path.join(root, runs[-1])
    except Exception:
        raise ValueError("No runs in this directory: " + root)
    if load_run == -1:
        load_run = last_run
    else:
        load_run = os.path.join(root, load_run)
    encoder_dir = os.path.join(load_run, "student_encoder_model")
    if checkpoint == -1:
        models = [file for file in os.listdir(encoder_dir) if "model" in file]
        models.sort(key=lambda m: "{:0>15}".format(m))
        model = models[-1]
    else:
        model = "model_{}.pt".format(checkpoint)
    return os.path.join(encoder_dir, model)


def update_cfg_from_args(env_cfg, cfg_train, args):
    if env_cfg is not None:
        if args.num_envs is not None:
            if hasattr(env_cfg.env, "num_envs"):
                env_cfg.env.num_envs = args.num_envs
            elif hasattr(env_cfg.env, "numEnvs"):
                env_cfg.env.numEnvs = args.num_envs
            else:
                raise NotImplementedError
    if cfg_train is not None:
        if args.seed is not None:
            cfg_train.seed = args.seed
        if args.max_iterations is not None:
            cfg_train.runner.max_iterations = args.max_iterations
        if args.resume:
            cfg_train.runner.resume = args.resume
        if args.experiment_name is not None:
            cfg_train.runner.experiment_name = args.experiment_name
        if args.run_name is not None:
            cfg_train.runner.run_name = args.run_name
        if args.load_run is not None:
            cfg_train.runner.load_run = args.load_run
        if args.checkpoint is not None:
            cfg_train.runner.checkpoint = args.checkpoint
    return env_cfg, cfg_train


def get_args():
    custom_parameters = [
        {"name": "--task", "type": str, "default": "BipedalWheeled", "help": "Task name"},
        {"name": "--resume", "action": "store_true", "default": False, "help": "Resume from checkpoint"},
        {"name": "--experiment_name", "type": str, "help": "Experiment name"},
        {"name": "--run_name", "type": str, "help": "Run name"},
        {"name": "--load_run", "type": str, "help": "Load specific run"},
        {"name": "--checkpoint", "type": int, "help": "Checkpoint to load"},
        {"name": "--num_envs", "type": int, "help": "Number of environments"},
        {"name": "--seed", "type": int, "help": "Random seed"},
        {"name": "--max_iterations", "type": int, "help": "Maximum training iterations"},
        {"name": "--rl_device", "type": str, "default": "cuda:0", "help": "RL device"},
    ]
    args = gymutil.parse_arguments(description="getup_gym training", headless=True, custom_parameters=custom_parameters)
    args.physics_engine = gymapi.SIM_PHYSX
    return args
