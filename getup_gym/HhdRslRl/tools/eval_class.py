import numpy as np
import torch
import copy
import matplotlib.pyplot as plt
from collections import deque
import time
import os

# ... [保留 HeightTSNECollector 类定义] ...

class StairClimbingAnalyzer:
    """
    上台阶过程自动检测与数据分析器
    自动检测：开始上台阶 -> 记录数据 -> 结束上台阶 -> 生成报告 -> 重置环境
    """
    def __init__(self, env, save_dir="logs/stair_climbing_analysis", max_trials=5):
        self.env = env
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # 状态机: 'IDLE'(等待) -> 'CLIMBING'(上台阶中) -> 'RECOVERING'(结束恢复)
        self.state = 'IDLE'
        self.current_trial = 0
        self.max_trials = max_trials
        
        # 检测参数（可根据实际地形调整）
        self.stair_detection_dist = 3  # 前方第3个网格处检测台阶
        self.height_threshold = 0.08    # 高度差阈值（米），大于此值认为检测到台阶
        self.climbing_end_threshold = 0.02  # 高度变化率阈值（m/step），小于此值认为结束
        
        # 数据记录缓冲区
        self.reset_buffers()
        
        # 时间记录
        self.start_time = None
        self.end_time = None
        
        # 平滑度计算窗口
        self.action_history = deque(maxlen=5)
        
    def reset_buffers(self):
        """重置数据缓冲区"""
        self.time_history = []
        self.dof_vel_history = []
        self.dof_acc_history = []
        self.smoothness_history = []
        self.base_height_history = []
        self.front_terrain_height_history = []
        
        self.start_step = 0
        self.end_step = 0
        
    def detect_stair_start(self, teacher_obs, base_height):
        """
        检测是否即将上台阶
        策略：检查高度图前方区域是否有明显突起
        teacher_obs: [num_envs, 187+] 前187维是17x11高程图
        base_height: [num_envs] 当前机器人高度
        """
        # 提取高度图 (假设 17x11，行优先，中心在93)
        height_map = teacher_obs[:, :187].reshape(-1, 11, 17)  # [n_envs, 11, 17]
        
        # 取前方区域（假设索引6:11是前方，8:9是中间列）
        # 根据实际高程图坐标系调整，这里假设行0是后方，行10是前方
        front_patch = height_map[:, 8:11, 7:10]  # 前方3x3区域
        front_height = front_patch.mean(dim=[1, 2])
        
        # 计算高度差
        height_diff = front_height - base_height
        
        # 如果前方显著高于当前位置，认为检测到台阶
        detecting = height_diff > self.height_threshold
        return detecting, height_diff, front_height
    
    def detect_stair_end(self, base_height, step_count):
        """
        检测是否已结束上台阶
        策略：base高度趋于稳定（变化率小）且持续一段时间
        """
        if len(self.base_height_history) < 10:
            return False
            
        # 计算最近10步的高度变化率
        recent_heights = np.array(self.base_height_history[-10:])
        height_var = np.std(recent_heights)
        
        # 如果高度变化很小，认为已站上台阶顶部
        if height_var < self.climbing_end_threshold and step_count > 30:
            return True
        return False
    
    def calculate_smoothness(self, actions):
        """
        计算动作平滑度：基于动作变化率的L2范数
        越小越平滑
        """
        self.action_history.append(actions.cpu().detach().numpy())
        if len(self.action_history) < 2:
            return 0.0
        # 计算最近两步动作的差分
        diff = np.diff(np.array(list(self.action_history)), axis=0)
        smoothness = np.mean(np.linalg.norm(diff, axis=-1))
        return smoothness
    
    def update(self, step_idx, teacher_obs, obs, actions, infos, base_height):
        """
        主更新函数，每步调用
        返回: done_trials (int) 已完成的上台阶次数，用于外部重置
        """
        done_trials = 0
        
        # 提取 DOF 数据（根据你的 obs 结构修改索引）
        # 假设 obs 包含 dof_pos 和 dof_vel，需要从中提取或从 env 直接获取
        dof_vel = self.env.dof_vel.cpu().numpy() if hasattr(self.env, 'dof_vel') else np.zeros((self.env.num_envs, 12))
        dof_acc = self.env.dof_acc.cpu().numpy() if hasattr(self.env, 'dof_acc') else np.zeros((self.env.num_envs, 12))
        
        # 检测逻辑
        detecting_stair, height_diff, front_h = self.detect_stair_start(teacher_obs, base_height)
        
        # 状态机转换
        if self.state == 'IDLE':
            if detecting_stair.any():  # 任一环境检测到台阶
                self.state = 'CLIMBING'
                self.start_step = step_idx
                self.start_time = time.time()
                print(f"[Trial {self.current_trial}] 开始上台阶检测！高度差: {height_diff[0]:.3f}m")
                self.reset_buffers()  # 清空之前的数据
                
        elif self.state == 'CLIMBING':
            # 记录数据
            current_time = time.time() - self.start_time if self.start_time else 0
            self.time_history.append(current_time)
            self.dof_vel_history.append(np.abs(dof_vel).mean(axis=1).mean())  # 平均绝对速度
            self.dof_acc_history.append(np.abs(dof_acc).mean(axis=1).mean())  # 平均绝对加速度
            self.smoothness_history.append(self.calculate_smoothness(actions))
            self.base_height_history.append(base_height.mean().item())
            self.front_terrain_height_history.append(front_h.mean().item())
            
            # 检测结束
            if self.detect_stair_end(base_height, step_idx - self.start_step):
                self.state = 'FINISHED'
                self.end_step = step_idx
                self.end_time = time.time()
                duration = self.end_time - self.start_time
                print(f"[Trial {self.current_trial}] 结束上台阶！持续时间: {duration:.2f}s, 步数: {self.end_step - self.start_step}")
                
                # 生成报告和图像
                self.generate_report()
                self.plot_analysis()
                
                done_trials = 1
                self.current_trial += 1
                
                # 重置状态机等待下一次
                self.state = 'IDLE'
                self.action_history.clear()
                
        return done_trials
    
    def generate_report(self):
        """生成统计报告"""
        if len(self.time_history) == 0:
            return
            
        # 计算平均变化值
        avg_vel = np.mean(self.dof_vel_history)
        avg_acc = np.mean(self.dof_acc_history)
        avg_smooth = np.mean(self.smoothness_history)
        
        # 计算变化趋势（线性拟合斜率）
        x = np.arange(len(self.dof_vel_history))
        vel_trend = np.polyfit(x, self.dof_vel_history, 1)[0] if len(x) > 1 else 0
        acc_trend = np.polyfit(x, self.dof_acc_history, 1)[0] if len(x) > 1 else 0
        
        report = f"""
========== 上台阶过程分析报告 (Trial {self.current_trial}) ==========
持续时间: {self.time_history[-1]:.2f}s
总步数: {len(self.time_history)}
平均 DOF 速度: {avg_vel:.4f} rad/s
平均 DOF 加速度: {avg_acc:.4f} rad/s²
平均动作平滑度: {avg_smooth:.4f}
速度变化趋势: {vel_trend:.6f} (rad/s)/step {'(上升)' if vel_trend > 0 else '(下降)'}
加速度变化趋势: {acc_trend:.6f} (rad/s²)/step {'(上升)' if acc_trend > 0 else '(下降)'}
起始高度: {self.base_height_history[0]:.3f}m
终止高度: {self.base_height_history[-1]:.3f}m
高度增益: {self.base_height_history[-1] - self.base_height_history[0]:.3f}m
================================================================
        """
        print(report)
        
        # # 保存文本报告
        # with open(os.path.join(self.save_dir, f"trial_{self.current_trial}_report.txt"), 'w') as f:
        #     f.write(report)
    
    def plot_analysis(self):
        """绘制分析图表"""
        if len(self.time_history) < 2:
            return
            
        fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
        
        # 数据准备
        t = np.array(self.time_history)
        
        # 1. 高度变化（验证上台阶过程）
        axes[0].plot(t, self.base_height_history, 'b-', linewidth=2, label='Base Height')
        axes[0].plot(t, self.front_terrain_height_history, 'r--', linewidth=2, label='Front Terrain Height')
        axes[0].set_ylabel('Height (m)')
        axes[0].set_title(f'Stair Climbing Analysis - Trial {self.current_trial}')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. DOF Velocity
        axes[1].plot(t, self.dof_vel_history, 'g-', linewidth=2)
        axes[1].fill_between(t, self.dof_vel_history, alpha=0.3)
        axes[1].set_ylabel('DOF Velocity (rad/s)')
        axes[1].grid(True, alpha=0.3)
        
        # 3. DOF Acceleration
        axes[2].plot(t, self.dof_acc_history, 'r-', linewidth=2)
        axes[2].fill_between(t, self.dof_acc_history, alpha=0.3, color='red')
        axes[2].set_ylabel('DOF Acceleration (rad/s²)')
        axes[2].grid(True, alpha=0.3)
        
        # 4. Smoothness
        axes[3].plot(t, self.smoothness_history, 'purple', linewidth=2)
        axes[3].set_ylabel('Action Smoothness')
        axes[3].set_xlabel('Time (s)')
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, f"trial_{self.current_trial}_analysis.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"图表已保存: {save_path}")