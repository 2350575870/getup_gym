import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.font_manager as fm
import matplotlib
from matplotlib.font_manager import FontProperties

DataSpace = {
    "scale up tracking reward": "/home/eai/wheel_lab_project/logs/TianGong/2026-04-01_23-33-10",
    "scale down tracking reward": "/home/eai/wheel_lab_project/logs/TianGong/2026-04-01_22-40-56",
    # "adapt res": "/home/eai/wheel_lab_project/logs/TianGong/2026-03-20_09-22-04",
    # "curriculum": "/home/eai/wheel_lab_project/logs/TianGong/2026-03-15_14-27-42",
    # "Ours w/ Force0.02": "/home/bitbot/getup_gym_project/logs/Getup/Jan09_15-17-15-force-0.02",
    # "Ours w/o SWR": "/home/bitbot/getup_gym_project/logs/Getup/Jan09_15-22-48-without-SWR",
    # "Ours w/ RMA-Teacher": "/home/bitbot/getup_gym_project/logs/Getup/Jan09_17-11-53-with-RMA",
}
target = [
    # "Episode/rew_rew_base_height",
    # "Train/mean_force_magnitude",
    "Episode/base_height_mean",
    "Policy/mean_noise_std",
    
]
target_name = [
    # "base heig. reward",
    # "mean force magnitude",
    "Episode/base_height_mean",
    "Policy/mean_noise_std",
]
output_DIR = "/home/eai/getup_gym_project/data_plot_picture"

all_data = {}

if os.path.exists(output_DIR):
    pass
else:
    os.makedirs(output_DIR)

def data_load(logdir, tags):
    #import data from tensorboard 
    event_acc = EventAccumulator(logdir, size_guidance={'scalars': 10000})
    event_acc.Reload()

    data = {}
    for tag in tags:
        if tag not in event_acc.Tags()['scalars']:
            print(f"Tag {tag} not found in {logdir}")
            continue
        events = event_acc.Scalars(tag)
        event_acc.Reload()

        steps = []
        values = []
        for e in events:
            if (e.step - events[0].step) < 5000:
                steps.append(e.step - events[0].step)
                values.append(e.value)


        # steps = [e.step - events[0].step for e in events]
        # values = [e.value for e in events]
        data[tag] = pd.DataFrame({"step": steps, "values": values})

    return data



def hhdplot():
    # 先加载所有数据，不进行缩放
    for name, path in DataSpace.items():
        all_data[name] = data_load(path, target)
    
    # 定义缩放因子映射
    scale_factors = {
        "Ours": 1000,
        "Ours w/ RMA-Teacher": 1000,
        "Ours w/ Force0.02": 50,
        "Ours w/o SWR": 50
    }
    
    # 对每个方法的数据应用缩放因子
    # for name in all_data:
    #     if name in scale_factors:
    #         scale = scale_factors[name]
    #         for tag in all_data[name]:
    #             if tag in all_data[name]:
    #                 all_data[name][tag]['values'] = all_data[name][tag]['values'] * scale
    
    # 设置字体和颜色
    plt.rcParams['font.family'] = 'Times New Roman'
    
    # 定义颜色和线型 - 使用我之前推荐的设置
    color_linestyle_mapping = {
        "ResNet": {'color': '#2E86AB', 'linestyle': '-', 'marker': 'o'},
        "MLP": {'color': '#A23B72', 'linestyle': '--', 'marker': 's'},
        # "Ours w/o SWR": {'color': '#F18F01', 'linestyle': '-.', 'marker': '^'},
        # "Ours w/ RMA-Teacher": {'color': '#C73E1D', 'linestyle': ':', 'marker': 'D'}
    }
    
    # 如果DataSpace中还有其他方法，使用默认颜色
    default_colors = plt.cm.tab10(np.linspace(0, 1.0, num=len(DataSpace)))

    for target_i, tag in enumerate(target):
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # 绘制窗口大小为80的平滑线（主要线条）
        for i, (name, data_dict) in enumerate(all_data.items()):
            if tag in data_dict:
                df = data_dict[tag].copy()
                df['smooth_2'] = df['values'].rolling(window=80, min_periods=1).mean()
                
                # 获取颜色和线型
                if name in color_linestyle_mapping:
                    style = color_linestyle_mapping[name]
                    ax.plot(df['step'], df['smooth_2'], label=name, 
                            linewidth=4.0, alpha=1, 
                            color=style['color'], 
                            linestyle=style['linestyle'],
                            marker=style['marker'], markersize=6, markevery=100)
                else:
                    # 对于不在映射中的方法，使用默认样式
                    ax.plot(df['step'], df['smooth_2'], label=name, 
                            linewidth=4.0, alpha=1, 
                            color=default_colors[i], 
                            linestyle='-')
        
        # 设置图表样式
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)
        plt.xlabel("steps", fontsize=22)
        plt.ylabel(target_name[target_i], fontsize=22)
        
        # 改进的网格设置
        ax.grid(True, which='both', linestyle='--', linewidth=0.7, alpha=0.4)
        
        # 改进的图例设置
        ax.legend(fontsize=20, frameon=True, framealpha=0.9, 
                 edgecolor='gray', loc='best', ncol=1)
        
        plt.tight_layout()
        
        # 绘制窗口大小为10的平滑线（半透明背景线）
        for i, (name, data_dict) in enumerate(all_data.items()):
            if tag in data_dict:
                df = data_dict[tag].copy()
                df['smooth_1'] = df['values'].rolling(window=10, min_periods=10).mean()
                
                # 使用相同的颜色，但更细的线宽和半透明
                if name in color_linestyle_mapping:
                    style = color_linestyle_mapping[name]
                    ax.plot(df['step'], df['smooth_1'], label=None, 
                            linewidth=1.5, alpha=0.3, 
                            color=style['color'], 
                            linestyle=style['linestyle'])
                else:
                    ax.plot(df['step'], df['smooth_1'], label=None, 
                            linewidth=1.5, alpha=0.3, 
                            color=default_colors[i])
        
        # 保存图片
        filename = f"{tag.replace('/', '_')}.png"
        plt.savefig(os.path.join(output_DIR, filename), format='png',dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved plot: {filename}")

if __name__ == "__main__":
    hhdplot()

