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

from .base_config import BaseConfig


class LeggedRobotCfg(BaseConfig):
    class env:
        num_envs = 500               #环境数量，即：同一时刻计算的智能体数量
        num_observations = 66       #观测值数量（在这里应该是观测机器人的关节速度，机器人的IMU位置）
                                    
                                    #观测值的计算方法如下：
                                    #observations = base_lin_ver + base_ang_ver + projected_gravity + commands
                                                   #+dof_pos(defalut_dof_pos)+dof_vel+actions(last)+heights
                                    #本文中的observations=3 + 3 + 3 + 3 + 18 + 18 + 18 + 0
                                    #一般来说前四项中每一项的值都是3；后面三项由机器人的自由度决定。本代码段使用带机械臂的轮组机器人，自由度是18=6*2+3*2

        num_privileged_obs = None   # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise
        num_actions = 18            #动作数量，必须严格与后续urdf相关设置类asset里面的default_joint_angle数量保持一致
                                    #与前面的defalut_dof_pos相关，表示机器人的自由度

        env_spacing = 3.0           # not used with heightfields/trimeshes
        send_timeouts = True        # 向算法发送时钟戳（send time out information to the algorithm）
        episode_length_s = 20       # 每一秒钟运行的轨迹长度（episode length in seconds）

    class terrain:
        """设置地面摩擦等摩擦相关参数"""
        mesh_type = "plane"     #设置地面相关配置，可以选择平面（plane），崎岖地面（）等地面模型。这里设置成平面
                                # "heightfield" # none, plane, heightfield or trimesh）
        #其余参数保持原状即可
        horizontal_scale = 0.1  # [m]
        vertical_scale = 0.005  # [m]
        border_size = 25  # [m]
        curriculum = True
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.0
        # rough terrain only:
        measure_heights = False #本项用于设置observation中计算的heights项，本例中是关闭，则heights = 0
                                #此选项主要是用于设置环境空间中机器人周围观测点的高度信息（？可能还要确认一下是不是这样）。
        #后续的相关配置参数保持不变即可
        measured_points_x = [
            -0.8,
            -0.7,
            -0.6,
            -0.5,
            -0.4,
            -0.3,
            -0.2,
            -0.1,
            0.0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
        ]  # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = False  # select a unique terrain type and pass all arguments
        terrain_kwargs = ""  # Dict of arguments for selected terrain
        max_init_terrain_level = 5  # starting curriculum state
        terrain_length = 8.0
        terrain_width = 8.0
        num_rows = 10  # number of terrain rows (levels)
        num_cols = 20  # number of terrain cols (types)
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]
        # trimesh only:
        slope_treshold = (
            0.75  # slopes above this threshold will be corrected to vertical surfaces
        )

        slope_max = 0.3
        stairs_height_range = [0.01, 0.15]
        stairs_width_range = [0.3, 0.4]
        discrete_obstacles_height_range = [0.01, 0.13]
        random_uniform_terrain_max_height = 0.03

    class commands:
        curriculum = False
        max_curriculum = 1.0
        num_commands = 4  # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10.0  # time before command are changed[s]
        heading_command = True  # if true: compute ang vel command from heading error

        class ranges:
            """机器人参数限制。限制最大和最小速度"""
            lin_vel_x = [-1.0, 1.0]  # min max [m/s]
            lin_vel_y = [-1.0, 1.0]  # min max [m/s]
            ang_vel_yaw = [-1, 1]  # min max [rad/s]
            heading = [-3.14, 3.14]

    class init_state:
        """设置机器人的初始状态"""
        pos = [0.0, 0.0, 1.0]  # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]    与前面observations中的前两个参数相对应
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        default_joint_angles = {  # target angles when action = 0.0
            "leftthigh":0.0,
            "leftshank": 0.0,
            "leftwheel": 0.0,
            "rightthigh":0.0,
            "rightshank":0.0,
            "rightwheel":0.0,
            "leftbase":0.0,
            "leftshoulder":0.0,
            "leftelbow":0.0,
            "lestwristo":0.0,
            "leftwristt":0.0,
            "leftwristtt":0.0,
            "rightbase":0.0,
            "rightshoulder":0.0,
            "rightelbow":0.0,
            "rightwristo":0.0,
            "rightwristt":0.0,
            "rightwristtt":0.0,
        }   
        #与observation变量和机器人的urdf文件相关，
        #键值对中的键是urdf文件中每个关节的名字，值是每个关节变量的初始角度，这里都给0
        #内部参数的总数就是机器人的总自由度，也是observation中defalue_dof_pos的值

    class control:
        """设置控制模式等控制相关参数"""
        control_type = "P"  # P: position, V: velocity, T: torques
        # PD Drive parameters:
        #PD控制的参数设置
        #damping用于设置每个关节对应的控制模式：位控
        #stiffness用于设置每个关节对应的最大位置/速度/力矩值
        stiffness = {"leftthigh": 10.0, "leftshank": 15.0,"leftwheel":10,
                     "rightthigh":10,"rightshank":10,"rightwheel":10,
                     "leftbase":10,"leftshoulder":10,"leftelbow":10,"lestwristo":10,"leftwristtt":10,
                     "rightbase":10,"rightshoulder":10,"rightelbow":10,"rightwristo":10,"rightwristt":10,"rightwristtt":10,}  # [N*m/rad]
        damping = {"leftthigh": 1.0, "leftshank": 1.5,"leftwheel":1.0,
                   "rightthigh":1.0,"rightshank":1.0,"rightwheel":1.0,
                   "leftbase":1.0,"leftshoulder":1.0,"leftelbow":1.0,"lestwristo":1.0,"leftwristtt":1.0,
                   "rightbase":1.0,"rightshoulder":1.0,"rightelbow":1.0,"rightwristo":1.0,"rightwristt":1.0,"rightwristtt":1.0,}  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset:
        """机器人urdf模型导入以及相关配置设置"""
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/Owheel_Robot_ARM/urdf/Owheel_Robot_ARM.urdf"
                                                #机器人URDF文件的路径。{LEGGED_GYM_ROOT_DIR}是根目录替代代码
        name = "Owheel_Robot_ARM"               # 智能体的名称（actor name）
        foot_name = "wheel"                     # 足部的名称，用于与力传感器向配合,同时还用于和环境交互
                                                # 足的名称可以自己定义，但是必须要有，这很关键
                                                #（name of the feet bodies, used to index body state and contact force tensors）
        penalize_contacts_on = []
        terminate_after_contacts_on = []
        disable_gravity = False
        collapse_fixed_joints = True            # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        fix_base_link = False                   # fixe the base of the robot
        default_dof_drive_mode = 3              # 设置控制模式。0是不做控制，1是位控，2是速度控制，3是力控
                                                #see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        self_collisions = 0                     # 1 to disable, 0 to enable...bitwise filter
        replace_cylinder_with_capsule = True    # replace collision cylinders with capsules, leads to faster/more stable simulation
        flip_visual_attachments = (
            True                                # Some .obj meshes must be flipped from y-up to z-up
        )
        #其它参数一般保持不变
        density = 0.001
        angular_damping = 0.0
        linear_damping = 0.0
        max_angular_velocity = 1000.0
        max_linear_velocity = 1000.0
        armature = 0.0
        thickness = 0.01

    class domain_rand:
        randomize_friction = True
        friction_range = [0.5, 1.25]
        randomize_base_mass = False
        added_mass_range = [-1.0, 1.0]
        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 1.0

    class rewards:
        """基础奖励设置"""
        class scales:
            termination = -0.0
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -0.0
            torques = -0.00001
            dof_vel = -0.0
            dof_acc = -2.5e-7
            base_height = -0.0
            feet_air_time = 1.0
            collision = -1.0
            feet_stumble = -0.0
            action_rate = -0.01
            stand_still = -0.0

        only_positive_rewards = True  # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25  # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = (
            1.0  # percentage of urdf limits, values above this limit are penalized
        )
        soft_dof_vel_limit = 1.0
        soft_torque_limit = 1.0
        base_height_target = 1.0
        max_contact_force = 100.0  # forces above this value are penalized

    class normalization:
        """对观测值进行缩放，这是为了让神经网络的输入值在一个合理的范围内"""
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0

        clip_observations = 100.0   #观测值的限制范围
        clip_actions = 100.0        #动作的限制范围

    class noise:
        """噪声设置"""
        add_noise = False   #因为之前一直因为噪声设置问题报错，这里关闭噪声，即全部都是无偏的
        noise_level = 1.0  # scales other values

        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1

    # viewer camera:
    class viewer:
        ref_env = 0
        pos = [10, 0, 6]  # [m]
        lookat = [11.0, 5, 3.0]  # [m]

    class sim:
        """设置物理引擎的参数，一般不做改变"""
        dt = 0.005
        substeps = 1
        gravity = [0.0, 0.0, -9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx:
            """设置物理引擎的参数，一般按照原文设置就行，不需要修改"""
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0  # [m]
            bounce_threshold_velocity = 0.5  # 0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23  # 2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = (
                2  # 0: never, 1: last sub-step, 2: all sub-steps (default=2)
            )


class LeggedRobotCfgPPO(BaseConfig):
    """PPO算法初始化"""
    seed = 1                                    #设置训练随机种子
    runner_class_name = "OnPolicyRunner"        #设置算法名称

    class policy:
        init_noise_std = 1.0                    #设置初始化噪声标准差
        actor_hidden_dims = [512, 256, 128]     #设置actor神经网络隐藏层参数
        critic_hidden_dims = [512, 256, 128]    #设置critic神经网络隐藏层参数
        activation = "elu"  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        # rnn_type = 'lstm'
        # rnn_hidden_size = 512
        # rnn_num_layers = 1

    class algorithm:
        """PPO算法相关参数"""
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4  # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1.0e-3  # 5.e-4
        schedule = "adaptive"  # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.0

    class runner:
        policy_class_name = "ActorCritic"
        algorithm_class_name = "PPO"
        num_steps_per_env = 24      # 每次迭代的个数（per iteration）
        max_iterations = 200        # 设置迭代的次数为200

        # logging
        save_interval = 50          # check for potential saves every this many iterations
        experiment_name = "a2"      #实验模型名称，需要与launch中进行train的模型名称保持一致。注意修改！！！
        run_name = ""
        # load and resume
        resume = False
        load_run = -1               # -1 = last run
        checkpoint = -1             # -1 = last saved model
        resume_path = None          # updated from load_run and chkpt
