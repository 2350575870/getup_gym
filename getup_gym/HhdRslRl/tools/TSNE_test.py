import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from getup_gym.HhdRslRl.Modules.mlp_encoder import MlpEncoder


class MinimalTSNECollector:
    def __init__(self, max_samples=1000, save_dir="tsne_debug"):
        self.max_samples = max_samples
        self.save_dir = save_dir
        self.data = {'zt': [], 'height': []}
        self.collected = 0
        os.makedirs(save_dir, exist_ok=True)
    
    def add(self, encoder, obs, heights_1d):
        if self.collected >= self.max_samples:
            return
        
        with torch.no_grad():
            zt = encoder(obs)["zt"]
        
        self.data['zt'].append(zt.cpu().numpy())
        self.data['height'].append(heights_1d.cpu().numpy())
        self.collected += zt.shape[0]
    
    def save(self, step):
        if self.collected < 100:
            return
        
        z = np.concatenate(self.data['zt'], axis=0)[:self.max_samples]
        h = np.concatenate(self.data['height'], axis=0)[:self.max_samples]
        
        z_2d = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(z)
        
        plt.figure(figsize=(6, 5))
        plt.scatter(z_2d[:, 0], z_2d[:, 1], c=h, cmap='viridis', s=20, alpha=0.6)
        plt.colorbar(label='Height (m)')
        plt.title(f't-SNE at step {step} (n={len(z)})')
        plt.xlabel('Dim 1')
        plt.ylabel('Dim 2')
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/tsne_{step}.png", dpi=150)
        plt.close()
        print(f"Saved: {self.save_dir}/tsne_{step}.png")


if __name__ == "__main__":
    # 配置
    collector = MinimalTSNECollector(max_samples=500, save_dir="tsne_test")
    path = "/home/bitbot/getup_gym_project/logs/JROwheel/Feb08_21-23-06/teacher_encoder_model/teacher_model_10000.pt"  # 替换为你的模型路径
    
    encoder = MlpEncoder(
        encoder_input_size=202,
        layer_size=24,
        encoder_hidden_dims=[256, 128],
        activation='elu',
    )
    
    loaded_dict = torch.load(path, weights_only=True)
    
    encoder.load_state_dict(loaded_dict["model_state_dict"])
    
    # 模拟收集数据：不同高度的地形观测
    print("Collecting synthetic data...")
    for step in range(1000):  # 50个批次
        batch_size = 4000
        
        # 生成高度变化的数据（-2m 到 +3m）
        heights = torch.linspace(-2, 3, batch_size) + torch.randn(batch_size) * 0.2
        heights = torch.clamp(heights, -2, 3)
        
        # 模拟观测：前187维包含高度信息（这里用高度重复填充模拟）
        obs = torch.randn(batch_size, 187)
        obs[:, 0] = heights  # 第一维设为高度
        obs_other = torch.zeros(batch_size, 15)  # 其他15维随机
        obs = torch.cat([obs, obs_other], dim=1)  # 最终obs维度为202
        
        # 收集
        collector.add(encoder, obs, heights)
    
    # 保存t-SNE图
    collector.save(step="final")
    print(f"Total collected: {collector.collected}")