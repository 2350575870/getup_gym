import torch
from getup_gym.common.base_task import BaseTask
import numpy as np
from isaacgym.torch_utils import quat_rotate

def sigmoid(x, value_at_1):
    scale = np.sqrt(-2 * np.log(value_at_1))
    return torch.exp(-0.5 * (x*scale)**2)

def tolerance(x, bounds=(0.0, 0.0), margin=0.0, value_at_margin=0.1):
    lower, upper = bounds 
    assert lower < upper
    assert margin >= 0

    in_bounds = torch.logical_and(lower <= x, x <= upper)
    if margin == 0:
        value = torch.where(in_bounds, 1.0, 0)
    else:
        d = torch.where(x < lower, lower - x, x - upper) / margin
        value = torch.where(in_bounds, 1.0, sigmoid(d.double(), value_at_margin))
    
    return value

def ang_vel_xy(env: BaseTask):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(env.base_ang_vel[:, :2]), dim=1)

# def heading_cmd(env: BaseTask):
def rew_orientation(env: BaseTask):
        projected_gravity_error = torch.norm(env.projected_gravity[:,:2]- env.desired_projected_gravity[:,:2],dim=1)
        return torch.exp(-projected_gravity_error/env.cfg.rewards.tracking_sigma)
    
def track_base_height_exp(env: BaseTask):
    base_height = torch.mean(env.root_states[:, 2].unsqueeze(1) - env.measured_heights, dim=1)
    if env.cfg.commands.base_height_command:
        height_dim = env.cfg.commands.num_commands - 1
        target_base_height = env.commands[:, height_dim]
    else:
        target_base_height = env.target_height
    return torch.exp(-torch.abs(base_height - target_base_height)/env.cfg.rewards.tracking_sigma)

def small_pos(env:BaseTask):
    target_joint_pos = [0.0, 0.5, -1.0, 0.0, 0.0, 0.5, -1.0, 0.0]
    target_pos_tensor = torch.zeros_like(env.dof_pos)
    target_pos_tensor[:, :] = torch.tensor(target_joint_pos, device=env.device)
    pos_error = torch.sum(torch.square(env.dof_pos - target_pos_tensor), dim=1)
    return pos_error

def pen_base_orientation_l2(env: BaseTask):
    # 主要：z分量必须接近 -1
    z_error = torch.square(env.projected_gravity[:, 2] + 1.0)
    
    # 次要：xy分量必须接近 0（身体水平）
    xy_error = torch.sum(torch.square(env.projected_gravity[:, :2]), dim=1)
    
    # 加权组合
    return 0.02* xy_error + 2* z_error - torch.exp(-xy_error/0.25).clamp(0, 0.1)#- torch.exp(-torch.abs(env.projected_gravity[:, 0] - 0.1) / 0.1)

def orientation_x_l2(env: BaseTask):
    # Penalize non flat base orientation in yz plane
    return torch.square(env.projected_gravity[:, 0])

def pen_base_orientation_z_l2(env: BaseTask):
    up_dot = -env.projected_gravity[:, 2]
    upsize_down = up_dot < -0.2
    return upsize_down* 1.0

def orientation_y_l2(env: BaseTask):
    # Penalize non flat base orientation in xz plane
    return torch.square(env.projected_gravity[:, 1])

def orientation_z_l2(env: BaseTask):
    # Penalize upside-down orientation
    return torch.square(env.projected_gravity[:, 2])

def base_height(env: BaseTask):
    # Penalize base height away from target
    base_height = torch.mean(env.root_states[:, 2].unsqueeze(1) - env.measured_heights, dim=1)
    return torch.square(base_height - env.target_height)


def pen_torques_l2(env: BaseTask):
    # Penalize torques
    return torch.sum(torch.square(env.torques), dim=1)

def pen_dof_vel_l2(env: BaseTask):
    # Penalize dof velocities
    return torch.sum(torch.square(env.dof_vel[:, [0,1,2,4,5,6]]), dim=1)

def pen_dof_acc_l2(env: BaseTask):
    # Penalize dof accelerations
    return torch.sum(torch.square((env.last_dof_vel - env.dof_vel) / env.dt), dim=1)

def pen_action_rate_l2(env: BaseTask):
    # Penalize changes in actions
    return torch.sum(torch.square(
        env.last_actions - env.actions
    ), dim=1)
    
def action_rate_wheel(env: BaseTask):
    wheel_rate = torch.square(
                env.last_actions[:, 3] - env.actions[:, 3] + env.last_actions[:, 7] - env.actions[:, 7]
            )
    return torch.exp( - wheel_rate/env.cfg.rewards.tracking_sigma)

def collision(env: BaseTask):
    # Penalize collisions on selected bodies
    return torch.sum(
        1.0 * (torch.norm(env.contact_forces[:, env.penalised_contact_indices, :], dim=-1) > 0.1),
        dim=1,
    )

def pen_termination(env: BaseTask):
    # Terminal reward / penalty
    return env.reset_buf * ~env.time_out_buf

def pen_dof_pos_limits(env: BaseTask):
    # Penalize dof positions too close to the limit
    out_of_limits = -(env.dof_pos - env.dof_pos_limits[:, 0]).clip(max=0.0)  # lower limit
    out_of_limits += (env.dof_pos - env.dof_pos_limits[:, 1]).clip(min=0.0)
    return torch.sum(out_of_limits, dim=1)

def hip_limits(env: BaseTask):
    out_of_limit = (torch.abs(env.dof_pos[:,0]) - 0.06* torch.zeros_like(env.dof_pos[:, 0]) + torch.abs(env.dof_pos[:, 4]))
    contacts = env.contact_forces[:, env.feet_indices, 2] > 1.0
    # single_contact = torch.sum(1.0 * contacts, dim=1) == 1
    dual_contact = torch.sum(1.0 * contacts, dim=1) != 2
    return out_of_limit* (1.0* dual_contact)

def dof_vel_limits(env: BaseTask):
    # Penalize dof velocities too close to the limit
    # clip to max error = 1 rad/s per joint to avoid huge penalties
    return torch.sum(
        (torch.abs(env.dof_vel) - env.dof_vel_limits * env.cfg.rewards.soft_dof_vel_limit).clip(
            min=0.0, max=1.0
        ),
        dim=1,
    )

def pen_torque_limits(env: BaseTask):
    # penalize torques too close to the limit
    return torch.sum(
        (torch.abs(env.torques) - env.torque_limits * env.cfg.rewards.soft_torque_limit).clip(min=0.0),
        dim=1,
    )

def track_lin_vel_xy_exp(env: BaseTask):
    base_height = torch.mean(env.rigid_body_states[:, env.base_indices, 2] - env.measured_heights, dim=1)
    # Tracking of linear velocity commands (xy axes)
    lin_vel_sigma = env.cfg.rewards.tracking_sigma
    lin_vel_error = torch.sum(torch.square(env.commands[:, :2] - env.base_lin_vel[:, :2]),dim=1)    
    return torch.exp(-lin_vel_error / lin_vel_sigma)* (base_height > 0.6)

def tracking_lin_vel_y(env: BaseTask):
    # Tracking of linear velocity commands (y axis)
    lin_vel_sigma = env.cfg.rewards.tracking_sigma
    lin_vel_error = torch.square(env.commands[:, 1] - env.base_lin_vel[:, 1])
    return torch.exp(-lin_vel_error / lin_vel_sigma)

def tracking_lin_vel_x(env: BaseTask):
    # Tracking of linear velocity commands (x axis)
    lin_vel_sigma = env.cfg.rewards.tracking_sigma
    lin_vel_error = torch.square(env.commands[:, 0] - env.base_lin_vel[:, 0])
    return torch.exp(-lin_vel_error / lin_vel_sigma)

def track_ang_vel_yaw_exp(env: BaseTask):
    # Tracking of angular velocity commands (yaw)
    ang_vel_error = torch.square(env.commands[:, 2] - env.base_ang_vel[:, 2])
    lin_vel_sigma = env.cfg.rewards.tracking_sigma
    return torch.exp(-ang_vel_error / 0.12)

def feet_air_time(env: BaseTask):
    # Reward long steps
    # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
    contact = env.contact_forces[:, env.feet_indices, 2] > 1    #如果脚上的力大于1，则认为脚在地上，返回值为Ture
    contact_filt = torch.logical_or(contact, env.last_contacts)    #与上一次的contact进行逻辑或。
    env.last_contacts = contact
    first_contact = (env.feet_air_time > 0.0) * contact_filt       #py中ture = 1,false = 0
    env.feet_air_time += env.dt
    rew_airTime = torch.sum(
        (env.feet_air_time - 0.5) * first_contact, dim=1
    )  # reward only on first contact with the ground
    #rew_airTime *= torch.norm(env.commands[:, :2], dim=1) > 0.1  # no reward for zero command
    env.feet_air_time *= ~contact_filt #脚步接触地面时,脚的置空时间重置
    return rew_airTime

def stumble(env: BaseTask):
    """惩罚脚部撞击垂直表面"""
    # Penalize feet hitting vertical surfaces
    return torch.any(
        torch.norm(env.contact_forces[:, env.feet_indices, :2], dim=2)
        > 5 * torch.abs(env.contact_forces[:, env.feet_indices, 2]),
        dim=1,
    )

def pen_dof_pos_bias_l2(env: BaseTask):
    """在零指令下惩罚运动"""
    # Penalize motion at zero commands
    contact = (torch.sum(env.contacts_z, dim=-1) > 1) * 1.0
    hip_init_angle = torch.ones(env.num_envs,device=env.device) * env.default_dof_pos[:,0]
    thigh_init_angle = torch.ones(env.num_envs,device=env.device)* env.default_dof_pos[:,1]
    shank_init_angle = torch.ones(env.num_envs,device=env.device)* env.default_dof_pos[:,2]
    
    hip_bias = torch.square(env.dof_pos[:,0] - hip_init_angle) + torch.abs(env.dof_pos[:,4] - hip_init_angle)
    thigh_bias = torch.square(env.dof_pos[:,1] - thigh_init_angle) + torch.abs(
        env.dof_pos[:,5] - thigh_init_angle
    )
    shank_bias = torch.square(env.dof_pos[:,2] - shank_init_angle) + torch.abs(
        env.dof_pos[:,6] - shank_init_angle
    )
    bias_sum = torch.square(thigh_bias + shank_bias + hip_bias)* contact
    
    return bias_sum

def feet_contact_forces(env: BaseTask):
    # penalize high contact forces
    return torch.sum(
        (
            torch.norm(env.contact_forces[:, env.feet_indices, :], dim=-1) - env.cfg.rewards.max_contact_force
        ).clip(min=0.0),
        dim=1,
    )

def pen_two_leg_bias_l2(env: BaseTask):
    """惩罚：如果两只脚的关节角度差别太大，说明此时在使用交叉平衡，需要惩罚"""
    contact = (torch.sum(env.contacts_z, dim=-1) > 1) * 1.0
    hip_thick = env.dof_pos[:,0] - env.dof_pos[:,4]
    dof_thick = env.dof_pos[:,1] - env.dof_pos[:,5]
    dof_shank = env.dof_pos[:,2] - env.dof_pos[:,6]
    # if the robot is standing on two legs, then penalize the difference in joint angles
    leg_bias = (torch.square(dof_thick) + torch.square(dof_shank) + torch.square(hip_thick))
    return  leg_bias

def wheel_acc(env: BaseTask):
# Penalize dof accelerations
    return torch.sum(torch.square((env.last_dof_vel[:,[3,7]] - env.dof_vel[:,[3,7]]) / env.dt), dim=1)

def pen_no_fly_l2(env: BaseTask):
    contacts = env.contact_forces[:, env.feet_indices, 2] > 20.0
    # single_contact = torch.sum(1.0 * contacts, dim=1) == 1
    dual_contact = torch.sum(1.0 * contacts, dim=1) != 2
    return 1.0 * dual_contact

def no_fly_3(env: BaseTask):
    contacts = env.contact_forces[:, env.feet_indices, 2] > 0.1
    single_contact = torch.sum(1.0* contacts, dim=1) == 1
    no_contact = torch.sum(1.0* contacts, dim=1) == 0
    return 0.01* (1.0* single_contact) + 1.0* (1.0* no_contact)

def stand_still(env: BaseTask):
    return torch.sum(torch.abs(env.dof_pos - env.default_dof_pos), dim=1) * (
        torch.norm(env.commands[:, :2], dim=1) < 0.1
    )

def hip_rate_l1(env: BaseTask):
    """hip's reward"""
    lhip_rate = torch.abs(torch.tensor(env.dof_pos[:,0] - env.default_dof_pos[:,0], device=env.device))
    rhip_rate = torch.abs(torch.tensor(env.dof_pos[:,4] - env.default_dof_pos[:,4], device=env.device))

    hip_binary = (lhip_rate < 0.05) & (rhip_rate < 0.05)

    hip_rate = torch.sum(
        lhip_rate + rhip_rate
    )* (1.0* hip_binary) - torch.sum(lhip_rate + rhip_rate)* (1.0* ~hip_binary) 
    return hip_rate

def hip_rate_l2(env: BaseTask):
    """hip's reward"""
    lhip_rate = torch.square(env.dof_pos[:,0] - env.default_dof_pos[:,0])
    rhip_rate = torch.square(env.dof_pos[:,4] - env.default_dof_pos[:,4])

    hip_binary = (lhip_rate < 2.5e-3) & (rhip_rate < 2.5e-3)

    hip_rate = 0.1* (1.0* hip_binary) - torch.sum(lhip_rate + rhip_rate)*(1.0* ~hip_binary)
    return hip_rate

def pen_max_velocity_l2(env: BaseTask):
    abs_vel = env.base_lin_vel[:,0]
    abs_command = env.commands[:,0]


    vel_low_1 = abs_vel > abs_command* 0.8
    vel_high_1 = abs_vel < abs_command* 1.1
    
    vel_low = abs_vel < abs_command* 0.8
    vel_high = abs_vel > abs_command* 1.1

    command_low = abs_command < torch.zeros_like(abs_command)
    command_high = abs_command >= torch.zeros_like(abs_command)

    # vel_low = abs_vel < abs_command* 0.8
    # vel_high = abs_vel > abs_command* 1.2

    vel_disired = (~(vel_low | vel_high)) & command_high
    vel_disired_1 = ~(vel_low_1 | vel_high_1) & command_low
    
    reward = torch.zeros_like(env.base_lin_vel[:,0])
    reward[vel_low | vel_low_1] = -1.0
    reward[vel_high | vel_high_1] = -1.0
    reward[vel_disired | vel_disired_1] = 1.0

    return reward

def pen_feet_distance_l2(env: BaseTask):
    feet_distance = torch.abs(torch.norm(env.feet_state[:, 0, :2] - env.feet_state[:, 1, :2], dim=-1))
    # reward = torch.abs(feet_distance - env.cfg.rewards.min_feet_distance)
    reward = torch.clip(0.32 - feet_distance, 0, 1) + torch.clip(
        feet_distance - 0.38, 0, 1
    )
    return reward


def pen_action_smoothness_l2(env: BaseTask):
    """make action smooth"""
    return torch.sum(torch.square(env.actions - 2* env.last_actions + env.last_last_action), dim=-1) 


def wheel_distance_x(env: BaseTask):
    # contacts = torch.abs(env.contact_forces[:, env.feet_indices, 2]) > 0.1
    return (1.0* (env.feet_distance_x < 0.02))* 0.2 - (1.0* (env.feet_distance_x > 0.02))* env.feet_distance_x



def pen_wheel_slide_y_l2(env: BaseTask):
    """惩罚轮子在y轴方向的滑动"""
    lcontact_z = env.contact_forces[:, env.feet_indices[0], 2] > 1.0
    rcontact_z = env.contact_forces[:, env.feet_indices[1], 2] > 1.0
    lfeet_vel_y = env.rigid_body_states[:, env.feet_indices[0], 7:9]
    rfeet_vel_y = env.rigid_body_states[:, env.feet_indices[1], 7:9]
    return torch.sum(
        (1.0* lcontact_z)* lfeet_vel_y.norm(dim=-1) + (1.0* rcontact_z)* rfeet_vel_y.norm(dim=-1)
    )

def wheel_fem_distance_x(env: BaseTask):
    """惩罚轮子与FEM之间的距离"""
    lfeet_fem_distance_x = torch.abs(env.lfeet_fem_distance_x)
    rfeet_fem_distance_x = torch.abs(env.rfeet_fem_distance_x)

    return (
        lfeet_fem_distance_x* (1.0* (lfeet_fem_distance_x > 0.01)) + rfeet_fem_distance_x* (1.0* (rfeet_fem_distance_x > 0.01)) 
    )

# def standing(env: BaseTask):
#     """奖励站立"""

def wheel_velocity_not_equal_to_command(env: BaseTask):
    """惩罚轮子速度与指令不一致"""
    wheel_vel_error = torch.square(env.commands[:, 3] - env.dof_vel[:, 3]* 0.15) + torch.square(env.commands[:, 3] - env.dof_vel[:, 7]* 0.15)
    return wheel_vel_error

def torso_contact_force(env: BaseTask):
    """惩罚躯干接触地面"""
    contacts_force = torch.norm(env.contact_forces[:, env.base_indice, :], dim=-1)
    return torch.abs(contacts_force)

def rew_wheel_contact_force(env: BaseTask):
    """鼓励轮子接触力的大小"""
    wheel_force = torch.sum(torch.abs(env.contact_forces[:, env.feet_indices, 2]), dim=1)
    total_mass = torch.ones_like(wheel_force)* env.total_mass* 10
    return torch.exp(-torch.abs(wheel_force - total_mass)* 0.05)

def pen_joint_power_l2(env: BaseTask):
    """power pen"""
    base_height = torch.mean(env.root_states[:, 2].unsqueeze(1) - env.measured_heights, dim=1)
    reward = torch.sum(torch.abs(env.dof_vel)* torch.abs(env.torques), dim=1) * (base_height < 0.7)
    return reward

def slow_get_up(env: BaseTask):
    up = (
        (env.contact_forces[:, env.feet_indices[0], 2] > 80.0) & (env.contact_forces[:, env.feet_indices[1], 2] < 80)
    )
    down = ~up
    env.up_time[down] += env.dt* torch.ones_like(env.up_time[down])
    up_using_time = torch.zeros_like(env.up_time)
    up_using_time[up] = env.up_time[up]
    env.up_time[up] = 0
    reward = torch.exp(-torch.abs(up_using_time - 3*torch.ones_like(up_using_time))*10)
    return reward

def torso_force(env: BaseTask):
    torso_have_force = torch.norm(env.contact_forces[:, 0, :], dim=-1) > 10
    return torso_have_force* 1.0

def fly_in_sky(env: BaseTask):
    fly = torch.sum(torch.norm(env.contact_forces, dim=-1) < 5.0, dim=1) > 0
    return 1.0* fly
    
def torso_oritation(env: BaseTask):
    down = (env.root_states[:, 4] > 0.7) & (env.root_states[:, 4] <= 1.0)  
    return (1.0* down)

def pen_femur_pos_bias_l2(env: BaseTask):
    fem_bias = torch.sum(torch.square(env.dof_pos[:, [1, 5]] - env.default_dof_pos[:, [1, 5]]), dim=-1)
    return fem_bias

def tib_pos_limit(env: BaseTask):
    tib_pos_limit = (env.dof_pos[:, 2] > 0) | (env.dof_pos[:, 6] > 0) 
    return tib_pos_limit* 1.0

def fem_bias_2(env: BaseTask):
    fem_bias = torch.sum(torch.square(env.dof_pos[:, [1, 5]] - env.default_dof_pos[:, [1, 5]]), dim=-1)
    base_height = torch.mean(env.root_states[:, 2].unsqueeze(1) - env.measured_heights, dim=1)
    idx = base_height < 0.4
    reward = torch.zeros_like(fem_bias)
    reward[idx] = fem_bias[idx]
    return reward

# ----- Unitree reward -----

def rew_left_foot_displacement(env: BaseTask):
    base_xy = env.root_states[:, :2].clone()
    left_foot_xy = env.rigid_body_states[:, env.left_foot_indices, :2].squeeze(1)
    mse_error = torch.sum(torch.square(base_xy - left_foot_xy), dim=-1).clamp(0.3, np.inf)
    reward = torch.exp(mse_error * env.cfg.rewards.left_foot_displacement_sigma) *  (env.rigid_body_states[:, env.left_foot_indices, 2] < 0.2).squeeze(1)

    return reward

def rew_right_foot_displacement(env: BaseTask):
    base_xy = env.root_states[:, :2].clone()
    right_foot_xy = env.rigid_body_states[:, env.right_foot_indices, :2].squeeze(1)
    mse_error = torch.sum(torch.square(base_xy - right_foot_xy), dim=-1).clamp(0.3, np.inf)
    reward = torch.exp(mse_error * env.cfg.rewards.right_foot_displacement_sigma) * (env.rigid_body_states[:, env.right_foot_indices, 2] < 0.2).squeeze(1)
    return reward

def Unitree_ground_parallel(env: BaseTask):
    left_ankle_pos = env.rigid_body_states[:, env.left_ankle_indices, 2].clone() * 10
    right_ankle_pos = env.rigid_body_states[:, env.right_ankle_indices, 2].clone() * 10
    var = left_ankle_pos.var(1) + right_ankle_pos.var(1)
    var = torch.mean(torch.concat([left_ankle_pos.var(1).view(-1, 1), right_ankle_pos.var(1).view(-1, 1)], dim=-1), dim=-1)
    reward = var < 0.05
    return reward

def Unitree_feet_height_var(env: BaseTask):
    left_foot_height = env.rigid_body_states[:, env.left_foot_indices, 2].clone() * 10
    right_foot_height = env.rigid_body_states[:, env.right_foot_indices, 2].clone() * 10
    feet_distance = torch.abs(left_foot_height - right_foot_height).squeeze(1).clamp(0.2, np.inf)
    return torch.exp(feet_distance * -2)

def pen_shank_orientation_l2(env: BaseTask):
    left_knee_pos = env.rigid_body_states[:, env.left_knee_indices, :3].clone()
    right_knee_pos = env.rigid_body_states[:, env.right_knee_indices, :3].clone()
    left_foot_pos = env.rigid_body_states[:, env.left_foot_indices, :3].clone()
    right_foot_pos = env.rigid_body_states[:, env.right_foot_indices, :3].clone()

    left_feet_orientation = (left_knee_pos - left_foot_pos)[:, :, 2] / torch.norm(left_knee_pos - left_foot_pos, dim=-1)
    right_feet_orientation = (right_knee_pos - right_foot_pos)[:, :, 2] / torch.norm(right_knee_pos - right_foot_pos, dim=-1)

    feet_orientation = torch.mean(torch.concat([left_feet_orientation, right_feet_orientation], dim=-1), dim=-1)

    reward = tolerance(feet_orientation, [0.8, np.inf], 1, 0.1)#.unsqueeze(1) 

    return reward 

def Unitree_dof_ang_asymmetry_l1(env: BaseTask):
        """penalize dof position error betwreen the same joint with l1 function"""
        reward = torch.sum(torch.abs(
            env.dof_pos[:, env.left_leg_indices] - env.dof_pos[:, env.right_leg_indices]
        ), dim=-1)
        return torch.exp(-reward)
    
def pen_dof_ang_asymmetry_l2(env: BaseTask):
    """penalize dof position error betwreen the same joint with l2 function
    """
    reward = torch.sum(torch.abs(
        env.dof_pos[:, env.left_leg_indices] - env.dof_pos[:, env.right_leg_indices]
    ), dim=-1)
    return reward

def pen_dof_ang_offset_l1(env: BaseTask):
        """penalize dof position error from default with l1 function"""
        # Penalize motion at zero commands
        # print(env.default_dof_pos)
        
        reward = torch.sum(torch.square(
            env.dof_pos - env.default_dof_pos
        ), dim=-1)
        return torch.exp(-reward / 0.6)

def pen_dof_ang_offset_l2(env: BaseTask):
    """penalize dof position error from default with l2 function"""
    # Penalize motion at zero commands
    # print(env.default_dof_pos)
    base_height = torch.mean(env.root_states[:, 2].unsqueeze(1) - env.measured_heights, dim=1)
    reward = torch.sum(torch.square(
        env.dof_pos - env.default_dof_pos
    ), dim=-1)
    return reward * (base_height > 0.35)

def Unitree_gait_feet_frc_perio(env: BaseTask) -> torch.Tensor:
    """Penalize foot force during the swing phase of the gait."""
    delta_t = env.cfg.rewards.delta_t
    left_frc_swing_mask = env._gait_clock(env.gait_phase[:, 0], env.phase_ratio[:, 0], delta_t)[0]
    right_frc_swing_mask = env._gait_clock(env.gait_phase[:, 1], env.phase_ratio[:, 1], delta_t)[0]
    left_frc_score = left_frc_swing_mask * (torch.exp(-200 * torch.square(env.avg_feet_force_per_step[:, 0])))
    right_frc_score = right_frc_swing_mask * (torch.exp(-200 * torch.square(env.avg_feet_force_per_step[:, 1])))
    return left_frc_score + right_frc_score
    
def Unitree_gait_feet_spd_perio(env: BaseTask) -> torch.Tensor:
    """Penalize foot speed during the support phase of the gait."""
    delta_t = env.cfg.rewards.delta_t
    left_spd_support_mask = env._gait_clock(env.gait_phase[:, 0], env.phase_ratio[:, 0], delta_t)[1]
    right_spd_support_mask = env._gait_clock(env.gait_phase[:, 1], env.phase_ratio[:, 1], delta_t)[1]
    left_spd_score = left_spd_support_mask * (torch.exp(-100 * torch.square(env.avg_feet_speed_per_step[:, 0])))
    right_spd_score = right_spd_support_mask * (torch.exp(-100 * torch.square(env.avg_feet_speed_per_step[:, 1])))
    return left_spd_score + right_spd_score

def Unitree_gait_feet_frc_support_perio(env: BaseTask) -> torch.Tensor:
    """Reward that promotes proper support force during stance (support) phase."""
    delta_t = env.cfg.rewards.delta_t
    left_frc_support_mask = env._gait_clock(env.gait_phase[:, 0], env.phase_ratio[:, 0], delta_t)[1]
    right_frc_support_mask = env._gait_clock(env.gait_phase[:, 1], env.phase_ratio[:, 1], delta_t)[1]
    left_frc_score = left_frc_support_mask * (1 - torch.exp(-10 * torch.square(env.avg_feet_force_per_step[:, 0])))
    right_frc_score = right_frc_support_mask * (1 - torch.exp(-10 * torch.square(env.avg_feet_force_per_step[:, 1])))
    return left_frc_score + right_frc_score

def pen_hip_roll_actions_l2(env) -> torch.Tensor:
    """Penalize hip roll joint actions."""
    HipRollIdx = env.left_hip_joint_indices + env.right_hip_joint_indices
    return torch.sum(torch.abs(env.actions[:, HipRollIdx]), dim=1)

def rew_stay_alive(env: BaseTask) -> torch.Tensor:
    "reward stay alive"
    return torch.ones(env.num_envs, device=env.device)

def Unitree_hand_support(env: BaseTask):
    """奖励手部向下施加力（支撑身体）"""
    # 接触力在z方向的分量（向下为正）
    base_height = torch.mean(env.root_states[:, 2].unsqueeze(1) - env.measured_heights, dim=1)
    left_hand_fz = env.contact_forces[:, env.hands_indices[0], 2]
    right_hand_fz = env.contact_forces[:, env.hands_indices[1], 2]
    
    # 只奖励向下的力（支撑力）
    left_support = torch.clamp(left_hand_fz, 0, 100)  # 裁剪负数（向上拉力不计）
    right_support = torch.clamp(right_hand_fz, 0, 100)
    
    base_height = torch.mean(env.root_states[:, 2].unsqueeze(1) - env.measured_heights, dim=1)
    total_support = (left_support + right_support) * (base_height < 0.3)
    
    # 归一化
    return torch.clamp(total_support / 10.0, 0, 4.0)* (base_height < 0.4)  # 双手共100N约等于1.0奖励


def Unitree_hand_contact(env):
    """奖励手部与地面接触"""
    # 获取手部接触力
    base_height = torch.mean(env.root_states[:, 2].unsqueeze(1) - env.measured_heights, dim=1)
    left_hand_force = torch.norm(env.contact_forces[:, env.hands_indices[0], :], dim=-1)
    right_hand_force = torch.norm(env.contact_forces[:, env.hands_indices[1], :], dim=-1)
    
    # 奖励有接触力（接触地面）
    hand_contact = (left_hand_force > 1.0) | (right_hand_force > 1.0)
    
    # 或者奖励接触力的大小（鼓励用力撑）
    hand_force_sum = left_hand_force + right_hand_force
    
    return hand_contact.float()* (base_height < 0.45)  # 或 torch.clamp(hand_force_sum / 50.0, 0, 1)

def pen_hand_discontact_l2(env: BaseTask):
    base_height = torch.mean(env.root_states[:, 2].unsqueeze(1) - env.measured_heights, dim=1)
    left_hand_force = torch.norm(env.contact_forces[:, env.hands_indices[0], :], dim=-1)
    right_hand_force = torch.norm(env.contact_forces[:, env.hands_indices[1], :], dim=-1)
    hand_contact = ((left_hand_force > 10.0) | (right_hand_force > 10.0))
    reward = ~hand_contact & (base_height < 0.3)
    return reward

def Unitree_ankle_torque(env: BaseTask) -> torch.Tensor:
    return torch.sum(torch.square(env.torques[:, env.ankle_joint_indices]), dim=-1)

def pen_feet_force_unequal_l2(env: BaseTask) -> torch.Tensor:
    base_height = torch.mean(env.root_states[:, 2].unsqueeze(1) - env.measured_heights, dim=1)
    left_contact_force = torch.norm(env.contact_forces[:, env.right_foot_indices, :], dim=-1)
    right_contact_force = torch.norm(env.contact_forces[:, env.left_foot_indices, :], dim=-1)
    reward = torch.clamp(torch.abs((left_contact_force.squeeze(-1) - right_contact_force.squeeze(-1))) / 2, 0, 1)
    return reward


def pen_torso_orientation_l2(env: BaseTask) -> torch.Tensor:
    base_height = torch.mean(env.root_states[:, 2].unsqueeze(1) - env.measured_heights, dim=1)
    z_error = torch.square(env.projected_gravity[:, 2] + 1.0)
    orientation = torch.sum(torch.square(env.projected_gravity[:, :2]), dim=-1)
    return z_error* (base_height < 0.6) - torch.exp(-torch.abs(env.projected_gravity[:, 0] - 0.1) / 0.1)

def pen_base_vel_l2(env: BaseTask) -> torch.Tensor:
    error = torch.square(torch.norm(env.base_lin_vel[:, :], dim=-1))
    base_height = torch.mean(env.root_states[:, 2].unsqueeze(1) - env.measured_heights, dim=1)
    return error* (base_height < 0.6)* base_height 

def rew_knee_down(env: BaseTask) -> torch.Tensor:
    base_height = torch.mean(env.root_states[:, 2].unsqueeze(1) - env.measured_heights, dim=1)
    error = torch.abs(env.dof_pos[:, env.knee_joint_indices[0]] - 1.0) + torch.abs(
        env.dof_pos[:, env.knee_joint_indices[1]] - 1.0
    ) + 0.1* (
        torch.abs(env.dof_pos[:, env.hip_pitch_joint_indices[0]] + 1) + torch.abs(
            env.dof_pos[:, env.hip_pitch_joint_indices[1]] + 1
        )
    )
    
    error_2 = torch.abs(env.dof_pos[:, env.knee_joint_indices[0]] - 0.5) + torch.abs(
        env.dof_pos[:, env.knee_joint_indices[1]] - 0.5
    ) + 0.2* (
        torch.abs(env.dof_pos[:, env.hip_pitch_joint_indices[0]] + 0.25) + torch.abs(
            env.dof_pos[:, env.hip_pitch_joint_indices[1]] + 0.25
        )
    )
    return (torch.exp(-error/0.1) - error.clamp(0, 0.1))* ((base_height > 0.1) & (base_height < 0.4)) + (torch.exp(-error_2/0.1) - error_2.clamp(0, 0.1))* ((base_height > 0.4) & (base_height < 0.7))

def pen_hip_zero_l2(env: BaseTask) -> torch.Tensor:
    base_height = torch.mean(env.root_states[:, 2].unsqueeze(1) - env.measured_heights, dim=1)
    error =(
        torch.abs(env.dof_pos[:, env.hip_pitch_joint_indices[0]]) + torch.abs(
            env.dof_pos[:, env.hip_pitch_joint_indices[1]]
        )
    )
    return (torch.exp(-error/0.1) - error.clamp(0, 0.05))* ((base_height > 0.2) & (base_height < 0.4))

def Unitree_feet_xy_force(env: BaseTask) -> torch.Tensor:
    base_height = torch.mean(env.root_states[:, 2].unsqueeze(1) - env.measured_heights, dim=1)
    lcontact_force = torch.norm(env.contact_forces[:, env.feet_indices[0], :2], dim=-1)
    rcontact_force = torch.norm(env.contact_forces[:, env.feet_indices[1], :2], dim=-1)
    error = torch.abs(lcontact_force) + torch.abs(rcontact_force)
    
    return -error.clamp(0.0, 0.05) + torch.exp(-error/0.1)

def Unitree_pelvis_lower_than_base(env: BaseTask) -> torch.Tensor:
    base_height = torch.mean(env.rigid_body_states[:, env.base_indices, 2] - env.measured_heights, dim=1)
    pelvis_height = torch.mean(env.rigid_body_states[:, env.pelvis_indices, 2] - env.measured_heights, dim=1)
    error = base_height - pelvis_height
    return error


def pen_foot_orientation_l2(env: BaseTask) -> torch.Tensor:
    # Penalize feet not parallel to ground
    # Get foot quaternions
    base_height = torch.mean(env.rigid_body_states[:, env.base_indices, 2] - env.measured_heights, dim=1)
    left_foot_quat = env.rigid_body_states[:, env.left_foot_indices, 3:7].squeeze(1)
    right_foot_quat = env.rigid_body_states[:, env.right_foot_indices, 3:7].squeeze(1)
    
    # Local z-axis in world coordinates
    world_z = torch.tensor([0.0, 0.0, 1.0], device=env.device).repeat(env.num_envs, 1)
    
    # Rotate local z by foot quaternion
    left_foot_z_world = quat_rotate(left_foot_quat, world_z)
    right_foot_z_world = quat_rotate(right_foot_quat, world_z)
    
    # Dot product with world z (ground normal)
    left_alignment = torch.sum(left_foot_z_world * world_z, dim=1)
    right_alignment = torch.sum(right_foot_z_world * world_z, dim=1)
    
    # Penalize deviation from parallel (alignment should be 1 for perfect parallel)
    # Use 1 - cos^2(theta) as penalty, where cos(theta) = alignment
    left_penalty = 1.0 - left_alignment ** 2
    right_penalty = 1.0 - right_alignment ** 2
    sigma = 0.1  # 温度系数，越小惩罚越严厉
    left_reward = torch.exp(-left_penalty / sigma)
    right_reward = torch.exp(-right_penalty / sigma)
    
    return (left_reward + right_reward).clamp(0, 2.0)

def Unitree_feet_x_alignment(env: BaseTask):
    """
    奖励双脚前后位置对齐（x轴）
    目标：双脚并排，x坐标差异越小奖励越高
    """
    left_idx = env.left_foot_indices
    right_idx = env.right_foot_indices
    
    left_x = env.rigid_body_states[:, left_idx, 0]   # x坐标
    right_x = env.rigid_body_states[:, right_idx, 0] # x坐标
    
    # 前后差异
    x_diff = torch.abs(left_x - right_x).squeeze(-1)
    
    # 指数奖励：差异越小奖励越高，范围 [0, 1]
    reward = torch.exp(-x_diff / 0.2) 
    
    return reward

def Unitree_hand_unmove(env: BaseTask) -> torch.Tensor:
    """惩罚手部移动"""
    base_height = torch.mean(env.rigid_body_states[:, env.base_indices, 2] - env.measured_heights, dim=1)
    hand_actions = env.actions[:, env.hands_indices]
    hand_torques = env.torques[:, env.hands_indices]
    return (torch.sum(torch.square(hand_actions), dim=-1) + torch.sum(torch.square(hand_torques), dim=-1))* (base_height > 0.7)

