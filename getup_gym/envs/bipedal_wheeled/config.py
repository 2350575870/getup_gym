# Get-up training config for 8-DOF wheeled bipedal robot (JROwheel)
from getup_gym.HhdRslRl.basement.base_env.base_config import BaseConfig
from getup_gym.HhdRslRl.basement.base_config.encoder_config import *
class GetupCfg(BaseConfig):
    
    class terrain:
        """Terrain friction and restitution parameters"""
        border_size = 20  # [m]25
        curriculum = True
        static_friction = 1.0   
        dynamic_friction = 0.7  
        restitution = 0.4           
        # rough terrain only:
        selected = False  # (select a unique terrain type and pass all arguments)
        terrain_kwargs = ""  # Dict of arguments for selected terrain
        max_init_terrain_level = 6  # starting curriculum state5
        
        # terrain types: 
        #1. stairs(up) with step width = 0.32, height = stairs_height_range
        #2. slopes whth max slope = slope_max
        #3. stairs(up) with step width = stairs_width_range, height = stairs_height_range
        #4. stairs(down) with step width = stairs_width_range, height = stairs_height_range
        #5. discrete obstacles with height = discrete_obstacles_height_range
        #6. random uniform terrain with max height = random_uniform_terrain_max_height
        #7. stepping stones with stone size = stepping_stones_size, stone distance = stone_distance
        #8. gaps with gap size = gap_size
        #9. pits with depth = pit_depth
        #else: plane
        terrain_proportions = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        # trimesh only:
        slope_treshold = (
            0.8  # slopes above this threshold will be corrected to vertical surfaces
        )
        measure_heights = True

        #myself settings:
        mesh_type = "plane"   # "heightfield" # none, plane, heightfield or trimesh）
        horizontal_scale = 0.1  # [m]
        vertical_scale = 0.005  # [m]
        terrain_length = 6.0
        terrain_width = 6.0
        num_rows = 10  # number of terrain rows (levels)
        num_cols = 20  # number of terrain cols (types)

        measured_points_x = [
            -0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0.0,
             0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,
        ]#25
        measured_points_y = [
            -0.5,-0.4,-0.3,-0.2,-0.1,0.0,
             0.1,0.2,0.3,0.4,0.5
        ]

        #terrain params setting
        slope_max = 0.25
        stairs_height_range = [0.01, 0.1]
        stairs_width_range = [0.3, 0.42]
        discrete_obstacles_height_range = [0.01, 0.06]
        random_uniform_terrain_max_height = 0.03
    
    class env():
        num_envs = 4000
        num_envs_teacher = 3000
        num_actions = 8        
        layer_size = 12             # encoder's layer size    
        env_spacing = 3.0           # not used with heightfields/trimeshes
        send_timeouts = True        # (send time out information to the algorithm)
        episode_length_s = 20       # (episode length in seconds)
        unactuated_timesteps = 30

        class myself_setting:
            add_lots_last_time = True
            add_time_number = 5
        
    class commands():
        curriculum = False
        max_curriculum = 1.0
        num_commands = 6  # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10.0  # time before command are changed[s]
        heading_command = False  # if true: compute ang vel command from heading error
        base_height_command = False
        class ranges():
            lin_vel_x_only = 1500
            without_lin_vel_y = 500 # not add lin_vel_x_only
            lin_vel_x = [-0.5, 0.8]  # min max [m/s] 0.4
            lin_vel_y = [-0.0, 0.0]  # min max [m/s]
            ang_vel_yaw = [-0.3, 0.3]  # min max [rad/s]0.5
            angle_pitch = [-0.2, 0.2]
            angle_roll = [-0.2, 0.2]
            rew_base_height = [0.35, 0.75]
    
    class init_state:
        pos = [0.0, 0.0, 0.1]       # (x, y, z [m])
        rot = [0.0, -1.0, 0.0, 1.0]  # (x, y, z, w [quat])
        lin_vel = [0.0, 0.0, 0.0]   # x,y,z [m/s]    
        ang_vel = [0.0, 0.0, 0.0]   # x,y,z [rad/s]
        default_joint_angles = {  # target angles when action = 0.0
            "lhiproll": 0.0,
            "lfempitch": 0.4,
            "ltibpitch": -0.8,
            "lwheelrot": 0.0,
            "rhiproll": 0.0,
            "rfempitch": 0.4,
            "rtibpitch": -0.8,
            "rwheelrot": 0.0,
        }  
    
    class control:
        control_type ={"hiproll":"P","fempitch":"P","tibpitch":"P","wheelrot":"V"}   # P: position, V: velocity, T: torques
        stiffness = {"hiproll": 80.4,"fempitch": 101.4,"tibpitch": 101.7,"wheelrot":1.0}
        damping = {"hiproll": 3.5,"fempitch": 2.5,"tibpitch":2.5,"wheelrot":0.000}  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = [0.2, 0.6, 0.6, 1.0, 0.2, 0.6, 0.6, 1.0]  # action scaling factors
        #if need change the action scale value in one training, use this mode
        using_actions_scale_changing = False
        action_scale_final = [0.2, 0.2, 0.2, 1.0, 0.2, 0.2, 0.2, 1.0]
        action_scale_changing_start_epochs = 3000
        action_scale_changing_end_epochs = 2000
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4   

    class asset:
        file = "{GETUP_GYM_ROOT_DIR}/getup_gym/robots/Owheel/urdf/robot_save.urdf"
        name = "JROwheel"
        foot_name = ["lwheel",'rwheel']
        collapse_fixed_joints = True
        flip_visual_attachments = (
            False
        )
        
        terminate_after_contacts_on = []
        
        # -- rarely need to change below --
        disable_gravity = False
        fix_base_link = False
        default_dof_drive_mode = 3
        self_collisions = 0
        replace_cylinder_with_capsule = False
        enable_gyroscopic_forces = False
        penalize_contacts_on = ["torso"]

        density = 0.0                 # density
        angular_damping = 0.01           # angular damping coefficient
        linear_damping = 0.01           # linear damping coefficient
        max_angular_velocity = 100.0
        max_linear_velocity = 100.0
        armature = 0.001                # armature          
        thickness = 0.01               # thickness
        
        left_foot_indices = "lwheel"
        right_foot_indices = "rwheel"
        
    
    class domain_rand:

        randomize_friction = True  
        friction_range = [0.1, 1.2]
        randomize_base_mass = True
        added_mass_range = [-0.1, 1.2]  

        #PD domain rand
        randomize_pd = False
        kp_range = {"hiprange": [58.0, 62.0], "femrange": [39.0, 42.0], "tibrange": [39.0, 42.0], "wheelrange": [1.01, 0.99]}
        kd_range = {"hiprange": [3.4, 3.6], "femrange": [1.9, 2.1], "tibrange": [1.9, 2.1]}

        push_robots = True
        push_start_epoch = 2000
        push_interval_s = 5.0 
        max_push_vel_x = 1.0
        max_push_vel_y = 1.0

        #force push(not velocity push)
        randomize_force_push = False
        push_rigid_name = ["torso"]
        push_direction = [
            ["x", "y"],
        ]
        force_push_start_epoch = 3000
        #push the robot every force_push_interval_s seconds
        force_push_interval_s = 10.0 
        push_force_x_range = [-100.0, 100.0]
        push_force_y_range = [-100.0, 100.0]  
        push_force_z_range = [-0.0, 0.0]

        #rand restitution
        randomize_restitution = True
        restitution_range = [0.0, 0.7]

        #rand motor strength
        randomize_motor_strength = False
        joint_name_need_strength_rand = ["lfempitch", "rfempitch"]
        motor_strength_range = [0.5, 0.8]

        add_mass_ranges = True
        add_mass_range_x = {"upper": 0.02, "lower": -0.02}
        add_mass_range_y = {"upper": 0.01, "lower": -0.01}
        add_mass_range_z = {"upper": 0.01, "lower":  -0.01}

        #pose randm
        randomize_joint_pos = False
        rand_coeff = 0.4 #range = [-joint_limit* rand_coeff, joint_limit* rand_coeff]

        #torso size randm
        randomize_torso_size = False
        torso_size_coeff = 0.05 # torso size = size + [-size* torso_size_coeff, size* torso_size_coeff]
        #this is only used when the torso is box

    class rewards:
        
        class reward_group_1:
            pen_action_rate_l2 = -0.06
            pen_base_orientation_l2 = -0.8
            pen_dof_acc_l2 =-2.5e-08
            pen_femur_pos_bias_l2 = -1.0
            pen_dof_pos_bias_l2 =-0.01
            pen_no_fly_l2 =-2.0
            track_base_height_exp =2.0
            pen_termination =-100
            pen_torques_l2 =-1.0e-06
            pen_two_leg_bias_l2 =-0.01
            rew_wheel_contact_force = 4.0
        
        class reward_group_2:
            pen_termination = -200
            pen_base_orientation_l2 = -2.6
            pen_base_orientation_z_l2 = -0.6
            track_base_height_exp = 4.0
            pen_dof_acc_l2 = -2.5e-8
            pen_action_rate_l2 = -0.06
            pen_dof_pos_bias_l2 = -0.06
            pen_two_leg_bias_l2 = -0.02
            pen_no_fly_l2 = -2.0
            rew_wheel_contact_force = 4.0

        class reward_group_3:
            pen_termination = -100
            pen_base_orientation_l2 = -2.0
            pen_base_orientation_z_l2 = -1.0
            pen_torques_l2 = -1.0e-6
            pen_dof_acc_l2 = -2.5e-6
            pen_action_rate_l2 = -0.02
            pen_two_leg_bias_l2 = -0.02
            pen_dof_pos_bias_l2 = -0.14
            track_base_height_exp = 6.0

        class reward_group_4:
            pen_termination = -500.0           #
            track_lin_vel_xy_exp = 7.0
            track_ang_vel_yaw_exp = 4.0
            pen_base_orientation_l2 = -15.0              #
            pen_torques_l2 = -1.0e-5
            pen_dof_acc_l2 = -2.5e-6
            pen_dof_vel_l2 = -4.0e-3
            pen_action_rate_l2 = -0.1
            pen_dof_pos_bias_l2 = -0.12
            pen_no_fly_l2 = -0.2
            track_base_height_exp = 5.0
            pen_feet_distance_l2 = -2.0
            pen_max_velocity_l2 = 1.0
            pen_dof_pos_limits = -5.0

            pen_torque_limits = -0.05
            pen_action_smoothness_l2 = -0.02
            pen_joint_power_l2 = -1.0e-4
            
            

        class curriculum:
            """Stage-wise curriculum: reward groups, target heights, pull-up assistance."""
            reward_group = ["reward_group_1", "reward_group_2", "reward_group_3", "reward_group_4"]
            target_height = [0.3, 0.45, 0.6, 0.75]
            using_pull_up = True
            base_pull_up_max = 400.0
            using_pull_up_end = True
            pull_up_end_epochs = 3000

        class terminations:
            """Extra termination conditions beyond the default time-out."""
            using_time_overstep_terminate = False
            time_overstep_max = 10.0

        class delays:
            """Action delay schedule."""
            using_delay_action = True
            delay_start_epochs = 500

        only_positive_rewards = False
        tracking_sigma = 0.12  # tracking reward = exp(-error^2/sigma)0.0085
        soft_dof_pos_limit = (
                0.9  # percentage of URDF limits, values above this limit are penalized
            )
        soft_dof_vel_limit = 0.5  # penalizes joint velocity near limits
        soft_torque_limit = 0.8
        base_height_target = 0.75
        max_contact_force = 100.0  # forces above this value are penalized  100
        
    class normalization:
        
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            heading = 1.0
            base_height = 2.0
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
        
        clip_observations = 100.0   # observation clipping range
        clip_actions = 50.0         # action clipping range
    
    class noise:
        add_noise = True
        noise_level = 1.0  # scales other values

        class noise_scales:
            dof_pos = 0.04
            dof_vel = 0.06
            lin_vel = 0.1
            ang_vel = 0.05
            gravity = 0.05
            height_measurements = 0.1


    class viewer:
        ref_env = 0
        pos = [10, 0, 6]  # [m]
        lookat = [11.0, 5, 3.0]  # [m]

    class sim:
        """Physics engine parameters (typically left unchanged)"""
        dt = 0.005
        substeps = 1
        gravity = [0.0, 0.0, -9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx:
            """Physics engine parameters (keep defaults)"""
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 8
            num_velocity_iterations = 4
            contact_offset = 0.01  # [m]
            rest_offset = 0.0  # [m]
            bounce_threshold_velocity = 0.5  # 0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23  # 2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = (
                2  # 0: never, 1: last sub-step, 2: all sub-steps (default=2)
            )
    
class GetUpPPO(BaseConfig):
    """PPO training configuration"""
    seed = 900                                   # training random seed
    runner_class_name = "TSOnPolicyRunner"        # algorithm class name

    class runner:
        
        policy_class_name = "ActorCritic" #"ActorCritic" or "Actor_MutiCritic" "MoeActorCritic" "ResNetActorCritic"
        policy_config = PolicyConfig(
            init_noise_std = 1.0,                    
            actor_hidden_dims = [512, 256, 128],     
            critic_hidden_dims = [512, 256, 128],    
            activation = "elu",  
        )
        algorithm_class_name = "TSPPO"
        algorithm_config = AlgorithmsPPOConfig(
            value_loss_coef = 1.0,
            use_clipped_value_loss = True,
            clip_param = 0.2,
            entropy_coef = 0.01,
            num_learning_epochs = 5,
            num_mini_batches = 4,  # mini batch size = num_envs*nsteps / nminibatches
            learning_rate = 1.0e-3,  # 5.e-4
            schedule = "adaptive",  # could be adaptive, fixed
            gamma = 0.99,
            lam = 0.95,
            desired_kl = 0.01,
            max_grad_norm = 1.0,
        )
        
        #encoder setting
        #TODO: the teacher encoder will be updated use encoder-decoder alg when the teacher encoder is set to MLP-Encoder-Decoder
        #if algorithms = None, the policy will delete all the encoder and only use the basement reinforcement learning
        encoder_training_setting = TrainingAlgConfig(
            algorithms = "CTS", #"RMA-Teacher", "RMA-TeacherStudent", "CTS", None
            teacher_encoder_name = "MLP-Encoder", #"MLP-Encoder", "MLP-Encoder-Decoder" "Identity" "UNet" "MLP-UNet"            
            student_encoder_name = "MLP-Encoder", #"MLP-Encoder", "MLP-UNet"
            student_resume_check_point = "",
            teacher_encoder_alg_type = "KL-Entropy", #None
            student_encoder_alg_class_name = "MSE", #MSE
            student_encoder_inference_decimation = 10,
            teacher_encoder_config = MlpEncoderConfig(
                encoder_hidden_dims = [256, 128],
                activation = "elu",
            ),
            student_encoder_config = MlpEncoderConfig(
                encoder_hidden_dims = [256, 128],
                activation = "elu",
            ),
            student_encoder_alg_config = EncoderAlgConfig(
                learning_rate = 1.0e-3,
                num_mini_batches = 4,
                num_learning_epochs = 5, #test good is 3,maybe 4 is better
                max_grad_norm = 0.8,
                mse_loss_coef= 1.0,
                student_kl_coef= 0.1,
                alg_type = "MSE",
            )
        )
        
        force_guidance_config = ForceGuidanceConfig(
            using_force_guidance = True,
        )
    
        symmetrical_loss_config = SymmetricalConfig(
            using_symmetrical_loss = False,
            symmetrical_loss_coef = 0.02,
        )
        
        num_steps_per_env = 24      
        max_iterations = 1500        

        # logging
        save_interval = 50          # check for potential saves every this many iterations
        experiment_name = "Getup"     
        run_name = ""
        # load and resume
        resume = False
        load_run = -1               # -1 = last run
        checkpoint = -1             # -1 = last saved model
        resume_path = None          # updated from load_run and chkpt