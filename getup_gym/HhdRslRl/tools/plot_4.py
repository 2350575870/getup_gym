
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os

# 深度学习论文风格设置
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

# 生成数据
steps = np.linspace(0, 5000, 300)
x_norm = steps / 3000  # 归一化到3000步

# 构造精确满足约束的指数衰减曲线
# y = target + (0.5 - target) * exp(-alpha * x)
# 在x=3000时，y应该接近target

def make_curve(target_val, alpha=5.0, noise_std=0.01):
    """生成从0.5开始，在3000步时降到target_val的曲线"""
    y = target_val + (0.5 - target_val) * np.exp(-alpha * (steps / 3000))
    # 添加噪声
    noise = np.random.normal(0, noise_std, len(steps))
    return y + noise

np.random.seed(42)

# 三条曲线
ours_data = make_curve(0.01, alpha=6.0, noise_std=0.008)    # 降到0.01
resnet_data = make_curve(0.06, alpha=4.5, noise_std=0.012)  # 降到0.06
mlp_data = make_curve(0.1, alpha=3.5, noise_std=0.015)      # 降到0.1

# 确保起点精确为0.5（去除噪声影响的第一步）
ours_data[0] = 0.5
resnet_data[0] = 0.5
mlp_data[0] = 0.5

# 标准差（用于阴影）
ours_std = 0.02
resnet_std = 0.03
mlp_std = 0.04

fig, ax = plt.subplots(figsize=(6, 4.5))

# 配色
colors = {
    "Ours": "#029E73",    # 绿色
    "ResNet": "#0173B2",  # 蓝色
    "Ours (MLP-WM)": "#DE8F05",     # 橙色
}

# 绘制Ours（最好）
ax.plot(steps, ours_data, label="Ours", color=colors["Ours"], linewidth=2.0)
ax.fill_between(steps, ours_data - ours_std, ours_data + ours_std, 
                alpha=0.15, color=colors["Ours"], edgecolor='none')

# 绘制ResNet（中等）
ax.plot(steps, resnet_data, label="ResNet", color=colors["ResNet"], linewidth=2.0)
ax.fill_between(steps, resnet_data - resnet_std, resnet_data + resnet_std, 
                alpha=0.15, color=colors["ResNet"], edgecolor='none')

# 绘制MLP（较差）
ax.plot(steps, mlp_data, label="Ours (MLP-WM)", color=colors["Ours (MLP-WM)"], linewidth=2.0)
ax.fill_between(steps, mlp_data - mlp_std, mlp_data + mlp_std, 
                alpha=0.15, color=colors["Ours (MLP-WM)"], edgecolor='none')

# 添加垂直参考线标记3000步
ax.axvline(x=3000, color='gray', linestyle=':', linewidth=1.0, alpha=0.5)
ax.text(3000, 0.45, '3000', fontsize=9, ha='center', color='gray')

# 添加水平参考线标记目标值
ax.axhline(y=0.01, color=colors["Ours"], linestyle=':', linewidth=0.8, alpha=0.3)
ax.axhline(y=0.06, color=colors["ResNet"], linestyle=':', linewidth=0.8, alpha=0.3)
ax.axhline(y=0.1, color=colors["Ours (MLP-WM)"], linestyle=':', linewidth=0.8, alpha=0.3)

# 格式设置
ax.set_xlabel("Training Steps")
ax.set_ylabel("Prediction Loss")
ax.set_xlim(0, 5000)
ax.set_ylim(0, 0.6)

# 网格和图例
ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.5)
ax.set_axisbelow(True)
ax.legend(loc='upper right', frameon=False)

# 边框
ax.spines['left'].set_linewidth(0.8)
ax.spines['bottom'].set_linewidth(0.8)

plt.tight_layout()
output_DIR = "/home/bitbot/getup_gym_project/data_plot_picture"
output_path = "target_predict_loss_plot.png"
plt.savefig(os.path.join(output_DIR, output_path), 
                   dpi=300, bbox_inches='tight', pad_inches=0.02)
plt.close()
print(f"Saved: {output_path}")

print(f"参考图已保存: {output_path}")
print(f"\n数据验证（在3000步时的值）:")
idx_3000 = np.argmin(np.abs(steps - 3000))
print(f"Ours: {ours_data[idx_3000]:.3f} (目标: 0.01)")
print(f"ResNet: {resnet_data[idx_3000]:.3f} (目标: 0.06)")
print(f"MLP: {mlp_data[idx_3000]:.3f} (目标: 0.1)")
