import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# ==================== 配置区域（保留原有注释） ====================

DataSpace = {
    "ResNet": "/home/bitbot/getup_gym_project/logs/JROwheel/Feb11_18-04-33-tsne-picture",
    "MLP": "/home/bitbot/getup_gym_project/logs/JROwheel/Feb08_18-55-48-with-world-model",
    # "Ours": "/home/bitbot/getup_gym_project/logs/Getup/Jan08_15-15-35-ours",
    # "Ours w/ Force0.02": "/home/bitbot/getup_gym_project/logs/Getup/Jan09_15-17-15-force-0.02",
    # "Ours w/o SWR": "/home/bitbot/getup_gym_project/logs/Getup/Jan09_15-22-48-without-SWR",
    # "Ours w/ RMA-Teacher": "/home/bitbot/getup_gym_project/logs/Getup/Jan09_17-11-53-with-RMA",
}

# 标准差接口 - 填入你的计算值
# 支持：标量（常数std）或与steps等长的数组
std_dict = {
    "ResNet": 0.05,  
    "MLP": 0.08,
}

target = [
    # "Episode/rew_rew_base_height",
    # "Train/mean_force_magnitude",
    "loss/predict_loss",
]
target_name = [
    # "base heig. reward",
    # "mean force magnitude",
    "Prediction Loss",  # 论文风格标签
]
output_DIR = "/home/bitbot/getup_gym_project/data_plot_picture"

os.makedirs(output_DIR, exist_ok=True)

# ==================== 深度学习论文风格设置 ====================

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif'],
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'lines.linewidth': 2.0,
    'axes.linewidth': 0.8,
    'axes.spines.top': False,      # 去掉上边框
    'axes.spines.right': False,    # 去掉右边框
})

# 论文标准配色（色盲友好）
colors = {
    "ResNet": "#0173B2",  # 蓝
    "MLP": "#DE8F05",     # 橙
    "Ours": "#029E73",    # 绿
    "Ours w/ Force0.02": "#D55E00",  # 红
    "Ours w/o SWR": "#CC78BC",       # 紫
    "Ours w/ RMA-Teacher": "#E69F00", # 黄
}

def data_load(logdir, tags):
    # import data from tensorboard 
    event_acc = EventAccumulator(logdir, size_guidance={'scalars': 10000})
    event_acc.Reload()

    data = {}
    for tag in tags:
        if tag not in event_acc.Tags()['scalars']:
            print(f"Tag {tag} not found in {logdir}")
            continue
        events = event_acc.Scalars(tag)
        
        steps = []
        values = []
        for e in events:
            if (e.step - events[0].step) < 5000:
                steps.append(e.step - events[0].step)
                values.append(e.value)
        
        data[tag] = pd.DataFrame({"step": steps, "values": values})
    return data

def plot_paper_style():
    all_data = {}
    for name, path in DataSpace.items():
        data = data_load(path, target)
        if data:
            all_data[name] = data
    
    if not all_data:
        print("No data loaded!")
        return

    for target_i, tag in enumerate(target):
        fig, ax = plt.subplots(figsize=(6, 4.5))  # 单栏论文标准尺寸
        
        for name, data_dict in all_data.items():
            if tag not in data_dict:
                continue
            
            df = data_dict[tag].copy()
            steps = df['step'].values
            
            # 平滑处理（保留原有window=80）
            df['smooth'] = df['values'].rolling(window=80, min_periods=1).mean()
            smooth_vals = df['smooth'].values
            
            color = colors.get(name, "#333333")
            
            # 绘制标准差阴影（alpha=0.15为标准论文值）
            if name in std_dict:
                std_val = std_dict[name]
                if np.isscalar(std_val):
                    std_array = np.full_like(steps, float(std_val))
                else:
                    std_array = np.array(std_val)
                    if len(std_array) != len(steps):
                        # 插值对齐
                        std_array = np.interp(
                            steps,
                            np.linspace(steps.min(), steps.max(), len(std_array)),
                            std_array
                        )
                
                ax.fill_between(steps, 
                               smooth_vals - std_array, 
                               smooth_vals + std_array,
                               alpha=0.15, 
                               color=color, 
                               edgecolor='none')
            
            # 绘制平滑曲线
            ax.plot(steps, smooth_vals, label=name, color=color, linewidth=2.0)
        
        # 论文标准格式
        ax.set_xlabel("Training Steps")
        ax.set_ylabel(target_name[target_i])
        ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.5)
        ax.set_axisbelow(True)
        
        # 图例：右上角，无框
        ax.legend(loc='upper right', frameon=False)
        
        # 确保边框线宽一致
        ax.spines['left'].set_linewidth(0.8)
        ax.spines['bottom'].set_linewidth(0.8)
        
        plt.tight_layout()
        
        filename = f"{tag.replace('/', '_')}_paper_style.png"
        plt.savefig(os.path.join(output_DIR, filename), 
                   dpi=300, bbox_inches='tight', pad_inches=0.02)
        plt.close()
        print(f"Saved: {filename}")

if __name__ == "__main__":
    plot_paper_style()