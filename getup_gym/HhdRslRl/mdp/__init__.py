from .reward import *
from getup_gym.HhdRslRl.tools.reward_manager import RewardManager, RewardManagerCfg

pen_termination = RewardManagerCfg(
    func=termination,
    weight=-350.0
)

rew_tracking_lin_vel = RewardManagerCfg(
    func=tracking_lin_vel,
    weight=0.0,
)

rew_tracking_ang_vel = RewardManagerCfg(
    func=tracking_ang_vel,
    weight=0.0,
)

pen_orientation = RewardManagerCfg(
    func=orientation,
    weight=0.0,
)

pen_orientation_x_l2 = RewardManagerCfg(
    func=orientation_x_l2,
    weight=0.0
)

pen_orientation_y_l2 = RewardManagerCfg(
    func=orientation_y_l2,
    weight=0.0
)

pen_orientation_z_l2 = RewardManagerCfg(
    func=orientation_z_l2,
    weight=0.0
)

pen_torques = RewardManagerCfg(
    func=torques,
    weight=0.0,
)

pen_dof_acc = RewardManagerCfg(
    func=dof_acc,
    weight=0.0,
)

pen_collision = RewardManagerCfg(
    func=collision,
    weight=0.0,
)

pen_action_rate = RewardManagerCfg(
    func=action_rate,
    weight=0.0,
)

pen_dof_pos_bias = RewardManagerCfg(
    func=dof_pos_bias,
    weight=0.0,
)

pen_two_leg_bias = RewardManagerCfg(
    func=two_leg_bias,
    weight=-0.0,
)

pen_dof_pos_limits = RewardManagerCfg(
    func=dof_pos_limits,
    weight=-0.0,
)

pen_no_fly_3 = RewardManagerCfg(
    func=no_fly_3,
    weight=-0.0,
)

pen_no_fly = RewardManagerCfg(
    func=no_fly,
    weight=0.0,
)

rew_max_velocity = RewardManagerCfg(
    func=max_velocity,
    weight=0.0,
)

pen_wheel_slide_y = RewardManagerCfg(
    func=wheel_slide_y,
    weight= -0.0,
)

pen_wheel_fem_distance_x = RewardManagerCfg(
    func=wheel_fem_distance_x,
    weight= -0.0,
)

pen_feet_distance = RewardManagerCfg(
    func=feet_distance,
    weight= 0.0,
)

pen_base_height = RewardManagerCfg(
    func=base_height,
    weight=0.0
)

rew_rew_base_height = RewardManagerCfg(
    func=rew_base_height,
    weight=0.0
)

pen_hip_limits = RewardManagerCfg(
    func=hip_limits,
    weight= 0.0,
)

pen_wheel_acc = RewardManagerCfg(
    func=wheel_acc,
    weight=0.0,
)

pen_wheel_velocity_not_equal_to_command = RewardManagerCfg(
    func=wheel_velocity_not_equal_to_command,
    weight=0.0,
)

pen_torso_contact_force = RewardManagerCfg(
    func=torso_contact_force,
    weight=0.00,
)

rew_small_pos = RewardManagerCfg(
    func=small_pos,
    weight=0.0
)

rew_wheel_contact_force = RewardManagerCfg(
    func=wheel_contact_force,
    weight=0.0
)

pen_action_smoothness = RewardManagerCfg(
    func=action_smoothness,
    weight=0.0
)

pen_dof_vel = RewardManagerCfg(
    func=dof_vel,
    weight=0.0
)
pen_joint_power = RewardManagerCfg(
    func=joint_power,
    weight=0.0
)

pen_hip_limit = RewardManagerCfg(
    func=hip_limit,
    weight=0.0
)

pen_dof_vel_limits = RewardManagerCfg(
    func=dof_vel_limits,
    weight=0.0
)

pen_torque_limits = RewardManagerCfg(
    func=torque_limits,
    weight=0.0
)

rew_slow_get_up = RewardManagerCfg(
    func=slow_get_up,
    weight=0.0
)

pen_torso_force = RewardManagerCfg(
    func=torso_force,
    weight=0.0
)

pen_fly_in_sky = RewardManagerCfg(
    func=fly_in_sky,
    weight=0.0
)

pen_torso_oritation = RewardManagerCfg(
    func=torso_oritation,
    weight=0.0
)

pen_fem_bias = RewardManagerCfg(
    func=fem_bias,
    weight=0.0
)

pen_tib_pos_limit = RewardManagerCfg(
    func=tib_pos_limit,
    weight=-0.0
)

pen_fem_bias_2 = RewardManagerCfg(
    func=fem_bias_2,
    weight=0.0
)

pen_Unitree_feet_distance = RewardManagerCfg(
    func=Unitree_feet_distance,
    weight=0.0
)

pen_Unitree_left_foot_displacement = RewardManagerCfg(
    func=Unitree_left_foot_displacement,
    weight=0.0
)

pen_Unitree_right_foot_displacement = RewardManagerCfg(
    func=Unitree_right_foot_displacement,
    weight=0.0
)

pen_Unitree_ground_parallel = RewardManagerCfg(
    func=Unitree_ground_parallel,
    weight=0.0
)

pen_Unitree_feet_height_var = RewardManagerCfg(
    func=Unitree_feet_height_var,
    weight=0.0
)

pen_Unitree_dof_vel = RewardManagerCfg(
    func=Unitree_dof_vel,
    weight=0.0
)

pen_Unitree_shank_orientation = RewardManagerCfg(
    func=Unitree_shank_orientation,
    weight=0.0
)

pen_Unitree_dof_ang_asymmetry_l1 = RewardManagerCfg(
    func=Unitree_dof_ang_asymmetry_l1,
    weight=0.0
)

pen_Unitree_dof_ang_asymmetry_l2 = RewardManagerCfg(
    func=Unitree_dof_ang_asymmetry_l2,
    weight=0.0
)

pen_Unitree_dof_ang_offset_l1 = RewardManagerCfg(
    func=Unitree_dof_ang_offset_l1,
    weight=0.0
)

pen_Unitree_dof_ang_offset_l2 = RewardManagerCfg(
    func=Unitree_dof_ang_offset_l2,
    weight=0.0
)

pen_Unitree_no_fly = RewardManagerCfg(
    func=Unitree_no_fly,
    weight=0.0
)

pen_orientation_z = RewardManagerCfg(
    func=orientation_z,
    weight=0.0
)

pen_Unitree_wheel_contact_force = RewardManagerCfg(
    func=Unitree_wheel_contact_force,
    weight=0.0
)


rew_gait_feet_frc_perio = RewardManagerCfg(
    func=Unitree_gait_feet_frc_perio,
    weight=0.0
)

rew_Unitree_gait_feet_spd_perio = RewardManagerCfg(
    func=Unitree_gait_feet_spd_perio,
    weight=0.0
)

rew_Unitree_gait_feet_frc_support_perio = RewardManagerCfg(
    func=Unitree_gait_feet_frc_support_perio,
    weight=0.0
)

rew_stay_alive = RewardManagerCfg(
    func=Unitree_stay_alive,
    weight=0.0
)

rew_hand_contact = RewardManagerCfg(
    func=Unitree_hand_contact,
    weight=0.0
)

rew_hand_support = RewardManagerCfg(
    func=Unitree_hand_support,
    weight=0.0
)

pen_hand_discontact = RewardManagerCfg(
    func=Unitree_hand_discontact,
    weight=0.0
)

pen_ankle_torque = RewardManagerCfg(
    func=Unitree_ankle_torque,
    weight=0.0
)

pen_feet_force_unequal = RewardManagerCfg(
    func=Unitree_feet_force_unequal, 
    weight=0.0
)

pen_torso_orientation = RewardManagerCfg(
    func=Unitree_torso_orientation,
    weight=0.0
)

pen_base_vel = RewardManagerCfg(
    func=Unitree_base_vel,
    weight=0.0
)

pen_knee_down = RewardManagerCfg(
    func=Unitree_knee_down,
    weight=1.0
)

rew_feet_xy_force = RewardManagerCfg(
    func=Unitree_feet_xy_force,
    weight=0.0
)

rew_plevis_lower = RewardManagerCfg(
    func=Unitree_pelvis_lower_than_base,
    weight=0.0
)

rew_hip_zero = RewardManagerCfg(
    func=Unitree_hip_zero,
    weight=0.0
)

rew_feet_orientation = RewardManagerCfg(
    func=Unitree_foot_orientation,
    weight=0.0
)

pen_hip_roll_actions = RewardManagerCfg(
    func=Unitree_hip_roll_actions,
    weight=0.0
)

pen_Unitree_feet_x_alignment = RewardManagerCfg(
    func=Unitree_feet_x_alignment,
    weight=0.0
)

pen_Unitree_hand_unmove = RewardManagerCfg(
    func=Unitree_hand_unmove,
    weight=0.0
)

def reward_register(reward_manager: RewardManager):
    reward_manager.register_reward(pen_termination)
    reward_manager.register_reward(rew_tracking_lin_vel)
    reward_manager.register_reward(rew_tracking_ang_vel)
    reward_manager.register_reward(pen_orientation)
    # reward_manager.register_reward(pen_orientation_x_l2)
    # reward_manager.register_reward(pen_orientation_y_l2)
    # reward_manager.register_reward(pen_orientation_z_l2)
    reward_manager.register_reward(pen_torques)
    reward_manager.register_reward(pen_dof_acc)
    reward_manager.register_reward(pen_dof_vel)
    reward_manager.register_reward(pen_collision)
    reward_manager.register_reward(pen_base_height)
    reward_manager.register_reward(rew_rew_base_height)

    reward_manager.register_reward(pen_action_rate)
    reward_manager.register_reward(pen_dof_pos_bias)
    reward_manager.register_reward(pen_two_leg_bias)
    reward_manager.register_reward(pen_dof_pos_limits)
    reward_manager.register_reward(pen_dof_vel_limits)
    reward_manager.register_reward(pen_torque_limits)

    reward_manager.register_reward(pen_no_fly_3)
    reward_manager.register_reward(pen_no_fly)

    reward_manager.register_reward(rew_max_velocity)

    reward_manager.register_reward(pen_wheel_slide_y)
    reward_manager.register_reward(pen_wheel_fem_distance_x)
    reward_manager.register_reward(pen_feet_distance)
    reward_manager.register_reward(pen_hip_limits)
    reward_manager.register_reward(pen_wheel_acc)
    reward_manager.register_reward(pen_wheel_velocity_not_equal_to_command)
    reward_manager.register_reward(pen_torso_contact_force)
    reward_manager.register_reward(rew_small_pos)
    reward_manager.register_reward(rew_wheel_contact_force)
    reward_manager.register_reward(pen_action_smoothness)
    reward_manager.register_reward(pen_joint_power)
    reward_manager.register_reward(pen_hip_limit)
    reward_manager.register_reward(rew_slow_get_up)
    reward_manager.register_reward(pen_torso_force)
    reward_manager.register_reward(pen_fly_in_sky)
    reward_manager.register_reward(pen_torso_oritation)
    reward_manager.register_reward(pen_fem_bias)
    reward_manager.register_reward(pen_tib_pos_limit)
    reward_manager.register_reward(pen_fem_bias_2)
    reward_manager.register_reward(pen_orientation_z)

    # --- Unitree reward regist ---
    reward_manager.register_reward(pen_Unitree_dof_ang_asymmetry_l1)
    reward_manager.register_reward(pen_Unitree_dof_ang_asymmetry_l2)
    reward_manager.register_reward(pen_Unitree_dof_ang_offset_l1)
    reward_manager.register_reward(pen_Unitree_dof_ang_offset_l2)
    reward_manager.register_reward(pen_Unitree_dof_vel)
    reward_manager.register_reward(pen_Unitree_feet_distance)
    reward_manager.register_reward(pen_Unitree_feet_height_var)
    reward_manager.register_reward(pen_Unitree_ground_parallel)
    reward_manager.register_reward(pen_Unitree_left_foot_displacement)
    reward_manager.register_reward(pen_Unitree_right_foot_displacement)
    reward_manager.register_reward(pen_Unitree_no_fly)
    reward_manager.register_reward(pen_Unitree_shank_orientation)
    reward_manager.register_reward(pen_Unitree_wheel_contact_force)
    reward_manager.register_reward(rew_gait_feet_frc_perio)
    reward_manager.register_reward(rew_Unitree_gait_feet_spd_perio)
    reward_manager.register_reward(rew_Unitree_gait_feet_frc_support_perio)
    reward_manager.register_reward(rew_stay_alive)
    reward_manager.register_reward(rew_hand_contact)
    reward_manager.register_reward(rew_hand_support)
    reward_manager.register_reward(pen_hand_discontact)
    reward_manager.register_reward(pen_ankle_torque)
    reward_manager.register_reward(pen_feet_force_unequal)
    reward_manager.register_reward(pen_torso_orientation)
    reward_manager.register_reward(pen_base_vel)
    reward_manager.register_reward(pen_knee_down)
    reward_manager.register_reward(rew_feet_xy_force)
    reward_manager.register_reward(rew_plevis_lower)
    reward_manager.register_reward(rew_hip_zero)
    reward_manager.register_reward(rew_feet_orientation)
    reward_manager.register_reward(pen_hip_roll_actions)
    reward_manager.register_reward(pen_Unitree_feet_x_alignment)
    reward_manager.register_reward(pen_Unitree_hand_unmove)