import matplotlib.pyplot as plt
import numpy as np

# 模拟数据（假设你有类似的数据结构）
steps = np.arange(1000, 5001, 1000)
methods = {
    "Ours": [0.5, 0.8, 0.9, 0.95, 0.97],
    "Ours w/ Force0.02": [0.4, 0.7, 0.85, 0.88, 0.90],
    "Ours w/o SWR": [0.3, 0.6, 0.75, 0.80, 0.83],
    "Ours w/RMA-Teacher": [0.2, 0.5, 0.7, 0.78, 0.81]
}

# 设置图表样式
plt.style.use('seaborn-v0_8-whitegrid')  # 使用 seaborn 样式
fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

# 颜色和线型
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
linestyles = ['-', '--', '-.', ':']

# 绘制每条线
for (label, data), color, ls in zip(methods.items(), colors, linestyles):
    ax.plot(steps, data, label=label, color=color, linestyle=ls, linewidth=2.5, marker='o', markersize=5)

# 设置坐标轴标签
ax.set_xlabel('Training Steps', fontsize=14)
ax.set_ylabel('Mean Force Magnitude', fontsize=14)
ax.set_title('Comparison of Mean Force Magnitude Across Methods', fontsize=16, pad=15)

# 设置坐标轴刻度
ax.set_xticks(steps)
ax.set_xticklabels([f'{s}' for s in steps], fontsize=12)
ax.tick_params(axis='y', labelsize=12)

# 设置图例
ax.legend(title='Methods', title_fontsize=13, fontsize=12, loc='upper left', frameon=True, framealpha=0.9, edgecolor='black')

# 添加网格
ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

# 调整布局
plt.tight_layout()

# 保存为高清图片
plt.savefig('Train_mean_force_magnitude_improved.png', dpi=300, bbox_inches='tight')
plt.show()