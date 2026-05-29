# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin
import os
# 关键：在 Python 启动前强制设置
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.0;8.6;8.9;9.0'
os.environ['CUDA_ARCH_LIST'] = '8.0;8.6;8.9;9.0'

# 欺骗 NVRTC，让它认为当前是 sm_90
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_MODULE_LOADING'] = 'LAZY'

# 清空缓存
import shutil
shutil.rmtree(os.path.expanduser('~/.cache/torch_extensions'), ignore_errors=True)

import isaacgym
import numpy as np

from datetime import datetime
from getup_gym.envs import *
from getup_gym import WHEEL_GYM_ENVS_DIR, WHEEL_GYM_ROOT_DIR
from getup_gym.HhdRslRl.tools.utils_gym.helpers import get_args
from getup_gym.HhdRslRl.tools.Teacher_task_register import teacher_task_registry
from getup_gym.HhdRslRl.tools.utils_gym import ExperimentLogger
from getup_gym.HhdRslRl.tools.utils_gym import train_batch
from getup_gym.HhdRslRl.tools.utils_gym.helpers import print_welcome_message
from getup_gym.HhdRslRl.tools.utils_gym.helpers import launch_tensorboard
from getup_gym.HhdRslRl.tools.utils_gym.helpers import class_to_dict
import torch


def TStrain(args):
    logdir = ExperimentLogger.generate_logdir(args.task)
    exp_msg = {}
    # NOTE: the train batch function is not implemented yet!
    if args.train_batch <= 1:  # require experiment commit message for non batched run or first time of batched run
        exp_msg = ExperimentLogger.commit_experiment(logdir, args)  # force to commit expriment message

    env_cfg, train_cfg = teacher_task_registry.get_cfgs(name=args.task)
    if args.train_batch != 0:
        args.headless = True  # force headless mode for batched runs
        # prepare environment
        train_batch_len = train_batch.check_env_batch_config(env_cfg)

        # check if batch size is too large
        if args.train_batch > train_batch_len:
            print(f"Batch size {args.train_batch} is too large, maximum batch size is {train_batch_len}")
            os._exit(0)

        env_cfg = train_batch.parse_env_batch_config(env_cfg, args.train_batch - 1)

    if args.launch_tensorboard:
        log_root = os.path.join(WHEEL_GYM_ROOT_DIR, "logs", args.task)
        launch_tensorboard(log_root)

    env, env_cfg = teacher_task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    ppo_runner, train_cfg = teacher_task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg, log_root=logdir
    )
    exp_msg["env_cfg"] = class_to_dict(env_cfg)
    exp_msg["train_cfg"] = class_to_dict(train_cfg)
    ExperimentLogger.save_hyper_params(logdir, env_cfg, train_cfg)
    ppo_runner.learn(
        num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True, experiment_log=exp_msg
    )


if __name__ == "__main__":
    args = get_args()
    print_welcome_message()
    TStrain(args)
