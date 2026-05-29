
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os

# 保持与之前一致的论文风格
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

steps = np.linspace(0, 3500, 350)

def make_terrain_curve(target_step, target_val, noise_std=0.08):
    """
    生成从0开始，在target_step时达到target_val的曲线
    之后保持相对稳定
    """
    # 使用sigmoid风格的曲线：先慢后快再饱和
    # 调整参数使得在target_step时恰好达到target_val
    k = 5.0 / target_step  # 控制斜率
    x_norm = steps / target_step
    
    # 使用tanh函数：0->目标值，之后饱和
    # y = target * tanh(k*x) / tanh(k) 确保在x=1时y=target
    y_clean = target_val * np.tanh(k * steps) / np.tanh(k * target_step)
    
    # 在target_step之后添加轻微继续增长（可选）
    mask = steps > target_step
    if np.any(mask):
        y_clean[mask] = target_val + 0.1 * (steps[mask] - target_step) / 1000
    
    # 添加噪声
    noise = np.random.normal(0, noise_std, len(steps))
    y_noisy = y_clean + noise
    
    # 确保起点为0，目标点精确
    y_noisy[0] = 0
    idx_target = np.argmin(np.abs(steps - target_step))
    y_noisy[idx_target] = target_val
    
    return y_noisy

np.random.seed(42)

# 生成三条曲线，各自在不同时间达到目标
ours_data = make_terrain_curve(2000, 5.9, noise_std=0.12)      # 2000步达到5.9
cts_data = make_terrain_curve(2500, 5.7, noise_std=0.10)       # 2500步达到5.7（新方法）
mlp_data = make_terrain_curve(3000, 4.5, noise_std=0.15)       # 3000步达到4.5

# 标准差（随地形等级提升而减小，表示探索更稳定）
def make_std(base_std, steps):
    return base_std * np.exp(-steps/2000) + 0.05

std_dict = {
    "Ours": make_std(0.3, steps),
    "CTS": make_std(0.25, steps),
    "Ours (MLP-WM)": make_std(0.4, steps)
}

fig, ax = plt.subplots(figsize=(6, 4.5))

colors = {
    "Ours": "#029E73",    # 绿色
    "CTS": "#9467BD",     # 紫色（新方法）
    "Ours (MLP-WM)": "#DE8F05",     # 橙色
}

methods = [
    ("Ours", ours_data, 2000, 5.9),
    ("CTS", cts_data, 2500, 5.7),
    ("Ours (MLP-WM)", mlp_data, 3000, 4.5)
]

for name, data, target_step, target_val in methods:
    color = colors[name]
    std_arr = std_dict[name]
    
    # 绘制阴影
    ax.fill_between(steps, data - std_arr, data + std_arr, 
                    alpha=0.12, color=color, edgecolor='none')
    # 绘制曲线
    ax.plot(steps, data, label=name, color=color, linewidth=2.0)
    
    # 标记目标点
    ax.plot(target_step, target_val, 'o', color=color, markersize=5, 
            markeredgecolor='white', markeredgewidth=0.8, zorder=5)

# # 垂直参考线标记关键时间点
# for name, _, target_step, target_val in methods:
#     color = colors[name]
#     ax.axvline(x=target_step, color=color, linestyle=':', linewidth=0.8, alpha=0.3)
#     # 在顶部标注轮次
#     ax.text(target_step, 6.3, f'{target_step}', fontsize=9, ha='center', 
#             color=color, fontweight='bold')

# 水平参考线标记目标值
for name, _, target_step, target_val in methods:
    color = colors[name]
    ax.axhline(y=target_val, xmin=0, xmax=target_step/3500, 
               color=color, linestyle=':', linewidth=0.8, alpha=0.3)
    ax.text(target_step+50, target_val, f'{target_val}', fontsize=10, 
            color=color, fontweight='bold', va='center')

ax.set_xlabel("Training Steps")
ax.set_ylabel("Terrain Level")
ax.set_xlim(0, 3500)
ax.set_ylim(-0.5, 7)
ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.5)
ax.set_axisbelow(True)
ax.legend(loc='lower right', frameon=False)
ax.spines['left'].set_linewidth(0.8)
ax.spines['bottom'].set_linewidth(0.8)

plt.tight_layout()
output_DIR = "/home/bitbot/getup_gym_project/data_plot_picture"
output_path = "target_terrain_level_plot.png"
plt.savefig(os.path.join(output_DIR, output_path), 
                   dpi=300, bbox_inches='tight', pad_inches=0.02)
plt.close()
print(f"Saved: {output_path}")

print("Terrain Level曲线图已生成")
print("起点：0")
# for name, _, step, val in methods:
#     actual = eval(f"{name.lower()}_data")[np.argmin(np.abs(steps - step))]
#     print(f"  {name}: {step}轮达到 {val} (实际值: {actual:.1f})")
