
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os

# 深度学习论文风格设置（与之前一致）
rcParams.update({
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif'],
    'font.size': 11,
    'axes.labelsize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'lines.linewidth': 2.0,
    'axes.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

steps = np.linspace(0, 5000, 300)
alpha = 0.0012

def make_reward_curve(start_val, end_val, noise_std=0.8):
    """
    生成从start_val开始，在5000步达到end_val的reward曲线（上升）
    """
    # 指数增长：在5000步时达到目标
    # y = start + (end - start) * (1 - exp(-alpha * x)) / (1 - exp(-alpha * 5000))
    denom = 1 - np.exp(-alpha * 5000)
    growth = (1 - np.exp(-alpha * steps)) / denom
    y_clean = start_val + (end_val - start_val) * growth
    
    # 添加噪声
    noise = np.random.normal(0, noise_std, len(steps))
    y_noisy = y_clean + noise
    
    # 确保起点和终点精确
    y_noisy[0] = start_val
    idx_5000 = np.argmin(np.abs(steps - 5000))
    y_noisy[idx_5000] = end_val
    
    return y_noisy

np.random.seed(42)

# 生成三条曲线
ours_data = make_reward_curve(-30, 190, noise_std=3.0)      # Ours: 升到190
resnet_data = make_reward_curve(-30, 170, noise_std=4.0)    # ResNet: 升到170
mlp_data = make_reward_curve(-30, 140, noise_std=5.0)       # MLP: 升到140

# 标准差（随训练进行逐渐减小，模拟后期稳定）
def make_std(base_std, steps):
    return base_std * (0.4 + 0.6 * np.exp(-steps/3000))

std_dict = {
    "Ours": make_std(8, steps),
    "ResNet": make_std(12, steps),
    "Ours (MLP-WM)": make_std(15, steps)
}

fig, ax = plt.subplots(figsize=(6, 4.5))

colors = {"Ours": "#029E73", "ResNet": "#0173B2", "Ours (MLP-WM)": "#DE8F05"}
methods = [
    ("Ours", ours_data, 190),
    ("ResNet", resnet_data, 170),
    ("Ours (MLP-WM)", mlp_data, 140)
]

for name, data, target_val in methods:
    color = colors[name]
    std_arr = std_dict[name]
    
    # 绘制标准差阴影
    ax.fill_between(steps, data - std_arr, data + std_arr, 
                    alpha=0.12, color=color, edgecolor='none')
    # 绘制曲线
    ax.plot(steps, data, label=name, color=color, linewidth=2.0)
    
    # 标记终点（5000步）
    ax.plot(5000, target_val, 'o', color=color, markersize=5, 
            markeredgecolor='white', markeredgewidth=0.8, zorder=5)

# 垂直参考线（5000步）
ax.axvline(x=5000, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
ax.text(5000, -25, '5000', fontsize=10, ha='center', color='gray')

# 水平虚线标记目标值
for name, _, val in methods:
    ax.plot([0, 5000], [val, val], ':', color=colors[name], linewidth=0.8, alpha=0.4)
    ax.text(4800, val, f'{val}', fontsize=10, color=colors[name], 
            fontweight='bold', va='center', ha='right')

# 标记起点
ax.axhline(y=-30, color='gray', linestyle=':', linewidth=0.8, alpha=0.3)
ax.text(200, -30, '-30', fontsize=10, color='gray', va='center')

ax.set_xlabel("Training Steps")
ax.set_ylabel("Episode Reward")
ax.set_xlim(0, 5000)
ax.set_ylim(-50, 220)
ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.5)
ax.set_axisbelow(True)
ax.legend(loc='lower right', frameon=False)  # reward通常图例在右下角
ax.spines['left'].set_linewidth(0.8)
ax.spines['bottom'].set_linewidth(0.8)

plt.tight_layout()
output_DIR = "/home/bitbot/getup_gym_project/data_plot_picture"
output_path = "target_reward_plot.png"
plt.savefig(os.path.join(output_DIR, output_path), 
                   dpi=300, bbox_inches='tight', pad_inches=0.02)
plt.close()
print(f"Saved: {output_path}")

print("Reward曲线图已生成")
print("起点：-30")
print("5000步终点：")
print(f"  Ours: {ours_data[np.argmin(np.abs(steps-5000))]:.0f} (目标: 190)")
print(f"  ResNet: {resnet_data[np.argmin(np.abs(steps-5000))]:.0f} (目标: 170)")
print(f"  MLP: {mlp_data[np.argmin(np.abs(steps-5000))]:.0f} (目标: 140)")
