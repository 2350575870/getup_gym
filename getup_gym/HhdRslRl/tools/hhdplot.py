import matplotlib.pyplot as plt
import torch
import pandas
import numpy as np
import os

class HhdPlot:
    def __init__(self, env, save_length_max: int, figsize=(10, 6), save_path=None):
        self.env = env
        self.figsize = figsize
        self.save_path = save_path
        self.save_length_max = save_length_max
        self.last_dof_vel = torch.zeros_like(self.env.dof_vel)
        self.data_dict = {}
        self.step = 0

        #plt setting
        self.figsize = figsize

        #registe
        self.joint_name_list = ["lhiproll", "lfempitch", "ltibpitch", "rhiproll", "rfempitch", "rtibpitch", "lwheelrot", "rwheelrot"]
        self.registe_data_name()
        
        #calculate
        self.start_idx = 233
        self.total_smooth = torch.zeros_like(self.env.dof_pos)
        self.total_power = torch.zeros_like(self.env.dof_pos)
        self.total_acc = torch.zeros_like(self.env.dof_pos)

    def registe_data_name(self) -> None:
        for name in self.joint_name_list:
            actural_pos = name + "_actural_pos"
            target_pos = name + "_target_pos"
            target_torque = name + "_target_torque"
            actural_vel = name + "_actural_velocity"
            
            actural_smoothness = name + "_actural_smoothness"
            actural_acc = name + "_actural_acc"
            actural_power = name + "_actural_power"
            
            self.data_dict[actural_pos] = []
            self.data_dict[target_pos] = []
            self.data_dict[target_torque] = []
            self.data_dict[actural_vel] = []
            
            self.data_dict[actural_smoothness] = []
            self.data_dict[actural_acc] = []
            self.data_dict[actural_power] = []

    def save_plot_data(self, **kwargs):
        """save plot data"""
        #find the step
        # if self.step > self.save_length_max:
        #     raise AttributeError("step too more!")
        #joints actural pos and target pos
        
        for name in self.joint_name_list:
            joint_idx = self.env.dof_dict[name]
            actural_pos = name + "_actural_pos"
            target_pos = name + "_target_pos"
            target_torque = name + "_target_torque"
            actural_vel = name + "_actural_velocity"
            self.data_dict[actural_pos].append(self.env.dof_pos[0, joint_idx].cpu().numpy()* 180/ np.pi)
            self.data_dict[target_pos].append(self.env.target_dof_pos[0, joint_idx].cpu().numpy()* 180/ np.pi)
            self.data_dict[target_torque].append(self.env.torques[0, joint_idx].cpu().numpy())
            self.data_dict[actural_vel].append(self.env.dof_vel[0, joint_idx].cpu().numpy())
            
            smoothness = (
                self.env.actions[0, joint_idx] - 2* self.env.last_actions[0, joint_idx] + self.env.last_last_action[0, joint_idx]).cpu().numpy()
            joint_power = self.env.torques[0, joint_idx].cpu().numpy() * self.env.dof_vel[0, joint_idx].cpu().numpy()
            joint_acc = (self.env.dof_vel[0, joint_idx].cpu().numpy() - self.last_dof_vel[0, joint_idx].cpu().numpy())/self.env.dt
            
            if self.step > self.start_idx:
                self.total_smooth[0, joint_idx] = self.total_smooth[0, joint_idx] + torch.abs(
                    self.env.actions[0, joint_idx] - 2* self.env.last_actions[0, joint_idx] + self.env.last_last_action[0, joint_idx])
                self.total_power[0, joint_idx] = self.total_power[0, joint_idx] + torch.abs(
                    self.env.torques[0, joint_idx] * self.env.dof_vel[0, joint_idx]
                )
                self.total_acc[0, joint_idx] = self.total_acc[0, joint_idx] + torch.abs(
                    (self.env.dof_vel[0, joint_idx] - self.last_dof_vel[0, joint_idx])/self.env.dt
                )
            self.last_dof_vel[0, joint_idx] = self.env.dof_vel[0, joint_idx]
            self.data_dict[name + "_actural_smoothness"].append(smoothness)
            self.data_dict[name + "_actural_acc"].append(joint_acc)
            self.data_dict[name + "_actural_power"].append(joint_power)
            
        self.step += 1

    def clear(self):
        self.step = 0

    def save_to_csv(self) -> None:
        """save the data in data dict to csv"""
        df = pandas.DataFrame(self.data_dict)
        df.to_csv(self.save_path + "/" + "isaacgym_data.csv", index=False)

    def plot_all_params(self):
        """plot all params in data dict list"""
        colors = plt.cm.tab10(np.linspace(0, 1.0, num=5))
        i = 0
        for name in self.joint_name_list:
            plt.figure(i)
            actural_pos = name + "_actural_pos"
            target_pos = name + "_target_pos"
            plt.plot(self.data_dict[actural_pos], label=actural_pos, color=colors[0])
            plt.plot(self.data_dict[target_pos], label=target_pos, color=colors[1])
            plt.xlabel("times")
            plt.ylabel("dof_pos(°)")
            plt.legend(fontsize = 10)
            save_name = name + ".png"
            #save path init
            save_path = self.save_path + "/" + name
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            plt.savefig(save_path)
            # plt.show()
            i+=1
            
        for name in self.joint_name_list:
            target_torque = name + "_target_torque"
            actural_velocity = name + "_actural_velocity"
            plt.figure(i)
            plt.plot(self.data_dict[target_torque], label=target_torque, color=colors[0])
            plt.plot(self.data_dict[actural_velocity], label=actural_velocity, color=colors[1])
            plt.xlabel("times")
            plt.ylabel("torques(Nm) or velocity(rad/s)")
            plt.legend(fontsize=10)
            save_name = self.save_path + "/" + target_torque + ".png"
            os.makedirs(os.path.dirname(save_name), exist_ok=True)
            plt.savefig(save_name)
            i+=1   
        for name in self.joint_name_list:
            actural_smoothness = name + "_actural_smoothness"
            plt.figure(i)
            plt.plot(self.data_dict[actural_smoothness], label=actural_smoothness, color=colors[0])
            plt.xlabel("times")
            plt.ylabel("smoothness")
            plt.legend(fontsize=10)
            save_name = self.save_path + "/" + name + "_smoothness.png"
            os.makedirs(os.path.dirname(save_name), exist_ok=True)
            plt.savefig(save_name)
            i+=1
            
        for name in self.joint_name_list:
            actural_power = name + "_actural_power"
            plt.figure(i)
            plt.plot(self.data_dict[actural_power], label=actural_power, color=colors[0])
            plt.xlabel("times")
            plt.ylabel("power")
            plt.legend(fontsize=10)
            save_name = self.save_path + "/" + name + "_power.png"
            os.makedirs(os.path.dirname(save_name), exist_ok=True)
            plt.savefig(save_name)
            i+=1
        
        for name in self.joint_name_list:
            actural_acc = name + "_actural_acc"
            plt.figure(i)
            plt.plot(self.data_dict[actural_acc], label=actural_acc, color=colors[0])
            plt.xlabel("times")
            plt.ylabel("acc")
            plt.legend(fontsize=10)
            save_name = self.save_path + "/" + name + "_acc.png"
            os.makedirs(os.path.dirname(save_name), exist_ok=True)
            plt.savefig(save_name)
            i+=1
        #params calculate
        mean_smooth = self.total_smooth.mean().item() / (self.step - self.start_idx) if self.step > self.start_idx else 0.0
        mean_power = self.total_power.mean().item() / (self.step - self.start_idx) if self.step > self.start_idx else 0.0
        mean_acc = self.total_acc.mean().item() / (self.step - self.start_idx) if self.step > self.start_idx else 0.0
        print(f"mean_smooth is: {mean_smooth}, mean_power is: {mean_power}, mean_acc is: {mean_acc}")
            
        self.clear()
        
            





        
        



        
        