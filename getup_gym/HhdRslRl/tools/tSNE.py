import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

class RepresentationVisualizer:
    def __init__(self, save_path='tsne_comparison.png'):
        self.save_path = save_path
        
    def collect_representations(self, encoder, dataloader, device='cuda', num_samples=1000):
        """
        收集编码器输出的表征
        Args:
            encoder: 你的 teacher 或 student encoder
            dataloader: 包含 (obs, terrain_height, label) 的数据加载器
                - terrain_height: 地形高度（用于着色）
                - label: 0=训练集, 1=测试集/OOD（用于区分分布内/外）
        Returns:
            z_list: [N, layer_size] 的表征数组
            heights: [N] 地形高度（用于颜色）
            is_ood: [N] 是否分布外
        """
        encoder.eval()
        z_list = []
        heights = []
        is_ood = []
        
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if len(z_list) * batch[0].size(0) >= num_samples:
                    break
                
                obs = batch[0].to(device)
                height = batch[1].cpu().numpy()  # 地形高度
                ood_flag = batch[2].cpu().numpy() if len(batch) > 2 else np.zeros(len(height))
                
                # 获取表征
                out = encoder(obs)
                z = out['zt'] if isinstance(out, dict) else out
                z = z.cpu().numpy()
                
                z_list.append(z)
                heights.append(height)
                is_ood.append(ood_flag)
        
        return {
            'z': np.concatenate(z_list, axis=0)[:num_samples],
            'height': np.concatenate(heights)[:num_samples],
            'is_ood': np.concatenate(is_ood)[:num_samples]
        }
    
    def compute_tsne(self, z_array, perplexity=30, n_iter=1000, random_state=42):
        """
        计算 t-SNE 降维
        perplexity: 困惑度（通常 5-50，数据点多就用大的）
        """
        print(f"Computing t-SNE for {z_array.shape[0]} samples...")
        tsne = TSNE(
            n_components=2,
            perplexity=30,           # 减小 perplexity 会减少远距离推力
            early_exaggeration=12,   # 减小（默认 12），避免早期过度推开
            learning_rate='auto',    # 自适应学习率
            n_iter=1000,             # 增加迭代，让点更稳定
            method='barnes_hut'      # 快速近似，但可能产生极端值
        )
        z_2d = tsne.fit_transform(z_array)
        return z_2d
    
    def plot_comparison(self, data_dict, title="Representation Visualization"):
        """
        绘制对比图
        data_dict: {
            'MSE': {'z': ..., 'height': ..., 'is_ood': ...},
            'Contrastive': {'z': ..., 'height': ..., 'is_ood': ...},
            'Teacher': {'z': ..., 'height': ..., 'is_ood': ...}  # 可选
        }
        """
        methods = list(data_dict.keys())
        n_methods = len(methods)
        
        fig, axes = plt.subplots(1, n_methods, figsize=(6*n_methods, 5))
        if n_methods == 1:
            axes = [axes]
        
        # 统一颜色映射（基于地形高度）
        all_heights = np.concatenate([data['height'] for data in data_dict.values()])
        vmin, vmax = all_heights.min(), all_heights.max()
        cmap = plt.cm.viridis  # 或 'coolwarm', 'plasma'
        
        for idx, (method, data) in enumerate(data_dict.items()):
            ax = axes[idx]
            
            # 计算 t-SNE（如果还没算）
            if 'z_2d' not in data:
                data['z_2d'] = self.compute_tsne(data['z'])
            
            z_2d = data['z_2d']
            heights = data['height']
            is_ood = data['is_ood']
            
            # 绘制分布内数据（圆点）
            in_dist_mask = is_ood == 0
            scatter1 = ax.scatter(
                z_2d[in_dist_mask, 0], 
                z_2d[in_dist_mask, 1],
                c=heights[in_dist_mask],
                cmap=cmap,
                vmin=vmin, vmax=vmax,
                s=20, alpha=0.6,
                label='In-distribution'
            )
            
            # 绘制分布外数据（星号，突出显示泛化能力）
            if is_ood.sum() > 0:
                scatter2 = ax.scatter(
                    z_2d[~in_dist_mask, 0],
                    z_2d[~in_dist_mask, 1],
                    c=heights[~in_dist_mask],
                    cmap=cmap,
                    vmin=vmin, vmax=vmax,
                    s=100, marker='*', alpha=0.8,
                    edgecolors='red', linewidths=1,
                    label='OOD (Unseen)'
                )
            
            ax.set_title(f'{method}\n(Structure: {"Continuous" if method=="Contrastive" else "Discrete"})', 
                        fontsize=12, fontweight='bold')
            ax.set_xlabel('t-SNE Dimension 1')
            ax.set_ylabel('t-SNE Dimension 2')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 添加颜色条
            cbar = plt.colorbar(scatter1, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Terrain Height (m)', rotation=270, labelpad=15)
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {self.save_path}")
        plt.show()

# ============ 使用示例 ============

def generate_demo_data():
    """生成模拟数据用于测试"""
    np.random.seed(42)
    n_samples = 500
    
    # 模拟：Contrastive 学习有结构，MSE 是乱的
    # Teacher/Contrastive: 螺旋结构（连续）
    theta = np.linspace(0, 4*np.pi, n_samples)
    r = np.linspace(0, 1, n_samples)
    z_contrastive = np.c_[r*np.cos(theta), r*np.sin(theta), np.random.randn(n_samples, 30)*0.1]
    height_contrastive = r  # 高度随半径增加
    
    # MSE: 坍缩成几团
    z_mse = np.random.randn(n_samples, 32) * 0.3
    cluster_centers = np.array([[1,0], [-1,0], [0,1], [0,-1]])
    for i in range(n_samples):
        z_mse[i, :2] += cluster_centers[i % 4]  # 聚到4个中心
    height_mse = np.random.rand(n_samples)  # 高度与位置无关（乱）
    
    # OOD 样本（最后50个）
    is_ood = np.zeros(n_samples)
    is_ood[-50:] = 1
    
    return {
        'Contrastive': {
            'z': z_contrastive.astype(np.float32),
            'height': height_contrastive,
            'is_ood': is_ood
        },
        'MSE': {
            'z': z_mse.astype(np.float32),
            'height': height_mse,
            'is_ood': is_ood
        }
    }

if __name__ == "__main__":
    # 测试
    viz = RepresentationVisualizer('tsne_robot_representation.png')
    data = generate_demo_data()
    viz.plot_comparison(data, title="Student Encoder: MSE vs Contrastive Distillation")