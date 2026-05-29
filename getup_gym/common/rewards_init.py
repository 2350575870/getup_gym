from getup_gym.common.reward_functions import *
from getup_gym.common.reward_manager import RewardManager, RewardManagerCfg

pen_termination = RewardManagerCfg(
    func=pen_termination,
    weight=-350.0
)

track_lin_vel_xy_exp = RewardManagerCfg(
    func=track_lin_vel_xy_exp,
    weight=0.0,
)

track_ang_vel_yaw_exp = RewardManagerCfg(
    func=track_ang_vel_yaw_exp,
    weight=0.0,
)

pen_base_orientation_l2 = RewardManagerCfg(
    func=pen_base_orientation_l2,
    weight=0.0,
)

orientation_x_l2 = RewardManagerCfg(
    func=orientation_x_l2,
    weight=0.0
)

orientation_y_l2 = RewardManagerCfg(
    func=orientation_y_l2,
    weight=0.0
)

orientation_z_l2 = RewardManagerCfg(
    func=orientation_z_l2,
    weight=0.0
)

pen_torques_l2 = RewardManagerCfg(
    func=pen_torques_l2,
    weight=0.0,
)

pen_dof_acc_l2 = RewardManagerCfg(
    func=pen_dof_acc_l2,
    weight=0.0,
)

collision = RewardManagerCfg(
    func=collision,
    weight=0.0,
)

pen_action_rate_l2 = RewardManagerCfg(
    func=pen_action_rate_l2,
    weight=0.0,
)

pen_dof_pos_bias_l2 = RewardManagerCfg(
    func=pen_dof_pos_bias_l2,
    weight=0.0,
)

pen_two_leg_bias_l2 = RewardManagerCfg(
    func=pen_two_leg_bias_l2,
    weight=-0.0,
)

pen_dof_pos_limits = RewardManagerCfg(
    func=pen_dof_pos_limits,
    weight=-0.0,
)

no_fly_3 = RewardManagerCfg(
    func=no_fly_3,
    weight=-0.0,
)

pen_no_fly_l2 = RewardManagerCfg(
    func=pen_no_fly_l2,
    weight=0.0,
)

pen_max_velocity_l2 = RewardManagerCfg(
    func=pen_max_velocity_l2,
    weight=0.0,
)

pen_wheel_slide_y_l2 = RewardManagerCfg(
    func=pen_wheel_slide_y_l2,
    weight= -0.0,
)

wheel_fem_distance_x = RewardManagerCfg(
    func=wheel_fem_distance_x,
    weight= -0.0,
)

pen_feet_distance_l2 = RewardManagerCfg(
    func=pen_feet_distance_l2,
    weight= 0.0,
)

track_base_height_exp = RewardManagerCfg(
    func=track_base_height_exp,
    weight=0.0
)

hip_limits = RewardManagerCfg(
    func=hip_limits,
    weight= 0.0,
)

wheel_acc = RewardManagerCfg(
    func=wheel_acc,
    weight=0.0,
)

wheel_velocity_not_equal_to_command = RewardManagerCfg(
    func=wheel_velocity_not_equal_to_command,
    weight=0.0,
)

torso_contact_force = RewardManagerCfg(
    func=torso_contact_force,
    weight=0.00,
)

small_pos = RewardManagerCfg(
    func=small_pos,
    weight=0.0
)

rew_wheel_contact_force = RewardManagerCfg(
    func=rew_wheel_contact_force,
    weight=0.0
)

pen_action_smoothness_l2 = RewardManagerCfg(
    func=pen_action_smoothness_l2,
    weight=0.0
)

pen_dof_vel_l2 = RewardManagerCfg(
    func=pen_dof_vel_l2,
    weight=0.0
)
pen_joint_power_l2 = RewardManagerCfg(
    func=pen_joint_power_l2,
    weight=0.0
)

dof_vel_limits = RewardManagerCfg(
    func=dof_vel_limits,
    weight=0.0
)

pen_torque_limits = RewardManagerCfg(
    func=pen_torque_limits,
    weight=0.0
)

slow_get_up = RewardManagerCfg(
    func=slow_get_up,
    weight=0.0
)

torso_force = RewardManagerCfg(
    func=torso_force,
    weight=0.0
)

fly_in_sky = RewardManagerCfg(
    func=fly_in_sky,
    weight=0.0
)

torso_oritation = RewardManagerCfg(
    func=torso_oritation,
    weight=0.0
)

pen_femur_pos_bias_l2 = RewardManagerCfg(
    func=pen_femur_pos_bias_l2,
    weight=0.0
)

tib_pos_limit = RewardManagerCfg(
    func=tib_pos_limit,
    weight=-0.0
)

fem_bias_2 = RewardManagerCfg(
    func=fem_bias_2,
    weight=0.0
)

rew_left_foot_displacement = RewardManagerCfg(
    func=rew_left_foot_displacement,
    weight=0.0
)

rew_right_foot_displacement = RewardManagerCfg(
    func=rew_right_foot_displacement,
    weight=0.0
)

Unitree_ground_parallel = RewardManagerCfg(
    func=Unitree_ground_parallel,
    weight=0.0
)

Unitree_feet_height_var = RewardManagerCfg(
    func=Unitree_feet_height_var,
    weight=0.0
)

pen_shank_orientation_l2 = RewardManagerCfg(
    func=pen_shank_orientation_l2,
    weight=0.0
)

Unitree_dof_ang_asymmetry_l1 = RewardManagerCfg(
    func=Unitree_dof_ang_asymmetry_l1,
    weight=0.0
)

pen_dof_ang_asymmetry_l2 = RewardManagerCfg(
    func=pen_dof_ang_asymmetry_l2,
    weight=0.0
)

pen_dof_ang_offset_l1 = RewardManagerCfg(
    func=pen_dof_ang_offset_l1,
    weight=0.0
)

pen_dof_ang_offset_l2 = RewardManagerCfg(
    func=pen_dof_ang_offset_l2,
    weight=0.0
)

pen_base_orientation_z_l2 = RewardManagerCfg(
    func=pen_base_orientation_z_l2,
    weight=0.0
)

Unitree_gait_feet_frc_perio = RewardManagerCfg(
    func=Unitree_gait_feet_frc_perio,
    weight=0.0
)

Unitree_gait_feet_spd_perio = RewardManagerCfg(
    func=Unitree_gait_feet_spd_perio,
    weight=0.0
)

Unitree_gait_feet_frc_support_perio = RewardManagerCfg(
    func=Unitree_gait_feet_frc_support_perio,
    weight=0.0
)

rew_stay_alive = RewardManagerCfg(
    func=rew_stay_alive,
    weight=0.0
)

Unitree_hand_contact = RewardManagerCfg(
    func=Unitree_hand_contact,
    weight=0.0
)

Unitree_hand_support = RewardManagerCfg(
    func=Unitree_hand_support,
    weight=0.0
)

pen_hand_discontact_l2 = RewardManagerCfg(
    func=pen_hand_discontact_l2,
    weight=0.0
)

Unitree_ankle_torque = RewardManagerCfg(
    func=Unitree_ankle_torque,
    weight=0.0
)

pen_feet_force_unequal_l2 = RewardManagerCfg(
    func=pen_feet_force_unequal_l2, 
    weight=0.0
)

pen_torso_orientation_l2 = RewardManagerCfg(
    func=pen_torso_orientation_l2,
    weight=0.0
)

pen_base_vel_l2 = RewardManagerCfg(
    func=pen_base_vel_l2,
    weight=0.0
)

rew_knee_down = RewardManagerCfg(
    func=rew_knee_down,
    weight=0.0
)

Unitree_feet_xy_force = RewardManagerCfg(
    func=Unitree_feet_xy_force,
    weight=0.0
)

Unitree_pelvis_lower_than_base = RewardManagerCfg(
    func=Unitree_pelvis_lower_than_base,
    weight=0.0
)

pen_hip_zero_l2 = RewardManagerCfg(
    func=pen_hip_zero_l2,
    weight=0.0
)

pen_foot_orientation_l2 = RewardManagerCfg(
    func=pen_foot_orientation_l2,
    weight=0.0
)

pen_hip_roll_actions_l2 = RewardManagerCfg(
    func=pen_hip_roll_actions_l2,
    weight=0.0
)

Unitree_feet_x_alignment = RewardManagerCfg(
    func=Unitree_feet_x_alignment,
    weight=0.0
)

Unitree_hand_unmove = RewardManagerCfg(
    func=Unitree_hand_unmove,
    weight=0.0
)

def reward_register(reward_manager: RewardManager):
    reward_manager.register_reward(pen_termination)
    reward_manager.register_reward(track_lin_vel_xy_exp)
    reward_manager.register_reward(track_ang_vel_yaw_exp)
    reward_manager.register_reward(pen_base_orientation_l2)
    # reward_manager.register_reward(orientation_x_l2)
    # reward_manager.register_reward(orientation_y_l2)
    # reward_manager.register_reward(orientation_z_l2)
    reward_manager.register_reward(pen_torques_l2)
    reward_manager.register_reward(pen_dof_acc_l2)
    reward_manager.register_reward(pen_dof_vel_l2)
    reward_manager.register_reward(collision)
    reward_manager.register_reward(track_base_height_exp)
    reward_manager.register_reward(track_base_height_exp)

    reward_manager.register_reward(pen_action_rate_l2)
    reward_manager.register_reward(pen_dof_pos_bias_l2)
    reward_manager.register_reward(pen_two_leg_bias_l2)
    reward_manager.register_reward(pen_dof_pos_limits)
    reward_manager.register_reward(dof_vel_limits)
    reward_manager.register_reward(pen_torque_limits)

    reward_manager.register_reward(no_fly_3)
    reward_manager.register_reward(pen_no_fly_l2)

    reward_manager.register_reward(pen_max_velocity_l2)

    reward_manager.register_reward(pen_wheel_slide_y_l2)
    reward_manager.register_reward(wheel_fem_distance_x)
    reward_manager.register_reward(pen_feet_distance_l2)
    reward_manager.register_reward(hip_limits)
    reward_manager.register_reward(wheel_acc)
    reward_manager.register_reward(wheel_velocity_not_equal_to_command)
    reward_manager.register_reward(torso_contact_force)
    reward_manager.register_reward(small_pos)
    reward_manager.register_reward(rew_wheel_contact_force)
    reward_manager.register_reward(pen_action_smoothness_l2)
    reward_manager.register_reward(pen_joint_power_l2)
    reward_manager.register_reward(hip_limits)
    reward_manager.register_reward(slow_get_up)
    reward_manager.register_reward(torso_force)
    reward_manager.register_reward(fly_in_sky)
    reward_manager.register_reward(torso_oritation)
    reward_manager.register_reward(pen_femur_pos_bias_l2)
    reward_manager.register_reward(tib_pos_limit)
    reward_manager.register_reward(fem_bias_2)
    reward_manager.register_reward(pen_base_orientation_z_l2)

    # --- Unitree reward regist ---
    reward_manager.register_reward(Unitree_dof_ang_asymmetry_l1)
    reward_manager.register_reward(pen_dof_ang_asymmetry_l2)
    reward_manager.register_reward(pen_dof_ang_offset_l1)
    reward_manager.register_reward(pen_dof_ang_offset_l2)
    reward_manager.register_reward(pen_dof_vel_l2)
    reward_manager.register_reward(pen_feet_distance_l2)
    reward_manager.register_reward(Unitree_feet_height_var)
    reward_manager.register_reward(Unitree_ground_parallel)
    reward_manager.register_reward(rew_left_foot_displacement)
    reward_manager.register_reward(rew_right_foot_displacement)
    reward_manager.register_reward(pen_no_fly_l2)
    reward_manager.register_reward(pen_shank_orientation_l2)
    reward_manager.register_reward(rew_wheel_contact_force)
    reward_manager.register_reward(Unitree_gait_feet_frc_perio)
    reward_manager.register_reward(Unitree_gait_feet_spd_perio)
    reward_manager.register_reward(Unitree_gait_feet_frc_support_perio)
    reward_manager.register_reward(rew_stay_alive)
    reward_manager.register_reward(Unitree_hand_contact)
    reward_manager.register_reward(Unitree_hand_support)
    reward_manager.register_reward(pen_hand_discontact_l2)
    reward_manager.register_reward(Unitree_ankle_torque)
    reward_manager.register_reward(pen_feet_force_unequal_l2)
    reward_manager.register_reward(pen_torso_orientation_l2)
    reward_manager.register_reward(pen_base_vel_l2)
    reward_manager.register_reward(rew_knee_down)
    reward_manager.register_reward(Unitree_feet_xy_force)
    reward_manager.register_reward(Unitree_pelvis_lower_than_base)
    reward_manager.register_reward(pen_hip_zero_l2)
    reward_manager.register_reward(pen_foot_orientation_l2)
    reward_manager.register_reward(pen_hip_roll_actions_l2)
    reward_manager.register_reward(Unitree_feet_x_alignment)
    reward_manager.register_reward(Unitree_hand_unmove)