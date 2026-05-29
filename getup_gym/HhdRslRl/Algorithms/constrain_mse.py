import torch
import torch.nn as nn
import torch.nn.functional as F
from getup_gym.HhdRslRl.Storage.encoder_storage import EncoderStorage
from getup_gym.HhdRslRl.basement.base_modules.ModulesBase import ModulesBase
import torch.optim as optim
from getup_gym.HhdRslRl.basement.base_algorithm.AlgorithmBase import AlgorithmBase
from torch.distributions import Normal, kl_divergence
from torch.amp import autocast, GradScaler
import copy

class ProjectionHead(nn.Module):
    """
    对比学习投影头：将表示映射到对比空间
    通常比原始 latent 维度小，且使用非线性
    """
    def __init__(self, input_dim, hidden_dim=128, output_dim=64):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.ReLU()
        
    def forward(self, x):
        x = self.activation(self.layer1(x))
        # 对比学习通常使用归一化后的输出
        x = F.normalize(self.layer2(x), dim=1)
        return x

class EncoderContrastiveMSE(AlgorithmBase):
    encoder: ModulesBase
    
    def __init__(
            self,
            num_trasition_per_env, 
            num_envs, 
            encoder: ModulesBase,
            teacher_encoder: ModulesBase,  # 修复：应该是 ModulesBase 而非 list
            device,
            learning_rate=8e-4,
            num_mini_batches=3,
            num_learning_epochs=3,
            max_grad_norm=0.8,
            mse_loss_coef=1.0,
            student_kl_coef=0.1,
            # 对比学习新增参数
            contrastive_coef=1.0,
            temperature=0.07,
            projection_hidden_dim=128,
            projection_output_dim=64,
            use_projection_head=True,
            alg_type="MSE"  # 可选：MSE, MSE-KL, Contrastive, MSE-Contrastive
    ):
        # base params init
        self.num_transition_per_env = num_trasition_per_env
        self.num_envs = num_envs
        self.student_encoder_input_size = encoder.encoder_input_size
        self.layer_size = encoder.layer_size
        self.device = device
        self.encoder_learning_rate = learning_rate
        self.num_mini_batch = num_mini_batches
        self.num_learning_epochs = num_learning_epochs
        self.max_grad_norm = max_grad_norm
        self.mse_loss_coef = mse_loss_coef
        self.student_kl_coef = student_kl_coef
        
        # 对比学习参数
        self.contrastive_coef = contrastive_coef
        self.temperature = temperature
        self.alg_type = alg_type
        self.use_projection_head = use_projection_head
        
        # encoder params init
        self.encoder: ModulesBase = encoder
        self.teacher_encoder: ModulesBase = teacher_encoder
        
        #teacher ema decay
        # 创建 Teacher 的 EMA 副本
        self.teacher_encoder_ema = copy.deepcopy(self.teacher_encoder)
        self.teacher_encoder_ema.eval()
        for param in self.teacher_encoder_ema.parameters():
            param.requires_grad = False
        
        self.ema_decay = 0.99  # EMA 衰减率
        
        # 投影头（仅用于对比学习）
        if self.use_projection_head and "Contrastive" in self.alg_type:
            self.student_projection = ProjectionHead(
                self.layer_size, projection_hidden_dim, projection_output_dim
            ).to(device)
            self.teacher_projection = ProjectionHead(
                self.layer_size, projection_hidden_dim, projection_output_dim
            ).to(device)
            # 冻结 teacher 投影头（通常不更新，或使用 EMA）
            for param in self.teacher_projection.parameters():
                param.requires_grad = False
            
            # 优化器包含 student encoder 和 projection head
            self.student_optimizer = optim.Adam(
                list(self.encoder.parameters()) + list(self.student_projection.parameters()), 
                self.encoder_learning_rate
            )
        else:
            self.student_optimizer = optim.Adam(self.encoder.parameters(), self.encoder_learning_rate)
            self.student_projection = None
            self.teacher_projection = None
        
        self.mse_function = nn.MSELoss()
        
        # encoder storage and transition init
        self.encoder_storage = EncoderStorage(
            num_trasition_per_env, 
            num_envs, 
            self.layer_size, 
            self.student_encoder_input_size, 
            self.teacher_encoder.encoder_input_size,
            device,
        )
        self.encoder_transition = EncoderStorage.EncoderTransition()
        
        self.scaler = GradScaler()  # 新增梯度缩放器
        
        # 初始化 teacher projection 的 EMA（如果使用）
        if "Contrastive" in self.alg_type and self.use_projection_head:
            self._update_teacher_projection()
            
    def update_ema(self):
        """每轮训练后更新 EMA 影子"""
        with torch.no_grad():  # 不计算梯度
            for ema_param, real_param in zip(
                self.teacher_encoder_ema.parameters(),
                self.teacher_encoder.parameters()
            ):
                # EMA 公式实现
                ema_param.data.mul_(self.ema_decay)  # 乘以 0.999
                ema_param.data.add_(real_param.data * (1 - self.ema_decay))  # 加 0.001*新值
    
    def _update_teacher_projection(self):
        """使用 EMA 或硬拷贝更新 teacher projection"""
        # 硬拷贝方式：直接复制 student 的参数
        with torch.no_grad():
            for param_t, param_s in zip(self.teacher_projection.parameters(), 
                                       self.student_projection.parameters()):
                param_t.data.copy_(param_s.data)
    
    def act(self, _zt_t: torch.tensor, _student_encoder_obs: torch.tensor, 
            _teacher_encoder_obs: torch.tensor = None) -> None:
        self.encoder_transition.zt_t = _zt_t
        self.encoder_transition.student_obs = _student_encoder_obs
        self.encoder_transition.teacher_obs = _teacher_encoder_obs
        self.encoder_storage.add_transitions(self.encoder_transition)
        self.encoder_transition.clear()

    def info_nce_loss_fast(self, z_teacher, z_student, temperature=0.07):
        """
        优化版本：避免显式分配大矩阵，使用 in-place 操作
        """
        batch_size = z_teacher.shape[0]
        device = z_teacher.device
        
        # 计算相似度矩阵（这是必须的，但我们可以优化内存）
        # 使用 float16 如果使用了 autocast，否则 float32
        with torch.autocast(device_type="cuda", enabled=False):  # 禁用内部 autocast 避免重复转换
            sim_matrix = torch.mm(z_student, z_teacher.t())
            sim_matrix.div_(temperature)  # in-place 除法，节省内存
        
        # 标签
        labels = torch.arange(batch_size, device=device, dtype=torch.long)
        
        # CrossEntropy 计算
        loss = F.cross_entropy(sim_matrix, labels)
        
        # 准确率（no_grad 已经在外部）
        acc = (sim_matrix.argmax(dim=1) == labels).float().mean()
        
        return loss, acc

    def update(self):
        mean_mse_loss = 0
        mean_contrastive_loss = 0
        mean_accuracy = 0

        # 更新 teacher projection（如果使用 EMA，每隔几步更新一次）
        if "Contrastive" in self.alg_type and self.use_projection_head:
            self._update_teacher_projection()
        
        # 选择合适的生成器
        if self.alg_type in ["Contrastive"]:
            # 对比学习需要更大的 batch size 以获得足够的负样本
            generater = self.encoder_storage.mini_batch_generator(
                self.num_mini_batch, self.num_learning_epochs
            )
        else:
            generater = (
                self.encoder_storage.mini_batch_generator(self.num_mini_batch, self.num_learning_epochs)
                if self.alg_type == "MSE" 
                else self.encoder_storage.VAE_mini_batch_generator(self.num_mini_batch, self.num_learning_epochs)
            )
            
        self.update_ema()  # 每轮训练后更新 EMA
            
        for (zt_t_batch, student_obs_batch, teacher_obs_batch) in generater:
            
            # ========== MSE 分支 ==========
            if self.alg_type == "MSE":
                zt_s = self.encoder.backward(student_obs_batch)["zt"]
                zt_t = self.teacher_encoder_ema.backward(teacher_obs_batch)["zt"].detach()  # 使用 EMA teacher，且不计算梯度
                mse_loss = self.mse_function(zt_s, zt_t)
                loss = self.mse_loss_coef * mse_loss
                mean_mse_loss += mse_loss.item()
                
            # ========== MSE-KL 分支（原有）==========
            elif self.alg_type in ["MSE-KL"]:
                student_encoder_dict = self.encoder.backward(student_obs_batch)
                teacher_encoder_dict = self.teacher_encoder.backward(teacher_obs_batch)
                mu_t = teacher_encoder_dict["mu"]

                mse_loss = self.mse_function(student_encoder_dict["mu"], mu_t.detach())
                
                dist_s = Normal(student_encoder_dict["mu"], torch.exp(0.5 * student_encoder_dict["logvar"]))
                dist_t = Normal(teacher_encoder_dict["mu"].detach(), torch.exp(0.5 * teacher_encoder_dict["logvar"].detach()))
                kl_loss = kl_divergence(dist_s, dist_t).mean()

                loss = self.mse_loss_coef * mse_loss + self.student_kl_coef * kl_loss
                mean_mse_loss += mse_loss.item()
                
            # ========== 对比学习分支（新增）==========
            elif self.alg_type in ["Contrastive", "MSE-Contrastive"]:
                # 1. 获取 Student 编码
                student_dict = self.encoder.backward(student_obs_batch)
                zt_s = student_dict["zt"]
                
                # 2. 获取 Teacher 编码（不计算梯度）
                with torch.no_grad():
                    if isinstance(zt_t_batch, dict):
                        zt_t = self.teacher_encoder_ema.backward(teacher_obs_batch)["zt"].detach()
                    else:
                        zt_t = self.teacher_encoder_ema.backward(teacher_obs_batch)["zt"].detach()
                
                # 3. 投影到对比空间
                if self.use_projection_head:
                    z_s_proj = self.student_projection(zt_s)
                    with torch.no_grad():
                        z_t_proj = self.teacher_projection(zt_t)
                else:
                    # 直接使用 latent 空间（需归一化）
                    z_s_proj = F.normalize(zt_s, dim=1)
                    z_t_proj = F.normalize(zt_t, dim=1)
                
                # 4. 计算对比损失
                con_loss, acc = self.info_nce_loss_fast(z_t_proj, z_s_proj, self.temperature)
                mean_contrastive_loss += con_loss.item()
                mean_accuracy += acc.item()
                
                # 5. 可选：结合 MSE（MSE-Contrastive 模式）
                if self.alg_type == "MSE-Contrastive":
                    mse_loss = self.mse_function(zt_s, zt_t)
                    loss = self.contrastive_coef * con_loss + self.mse_loss_coef * mse_loss
                    mean_mse_loss += mse_loss.item()
                else:
                    loss = self.contrastive_coef * con_loss
            
            else:
                raise ValueError(f"Unknown alg_type: {self.alg_type}")

            # 反向传播
            self.student_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.encoder.parameters(), self.max_grad_norm)
            
            # 如果使用了 projection head，也要裁剪其梯度
            if self.use_projection_head and self.student_projection is not None:
                nn.utils.clip_grad_norm_(self.student_projection.parameters(), self.max_grad_norm)
                
            self.student_optimizer.step()

        num_updates = self.num_learning_epochs * self.num_mini_batch
        
        # 计算平均值
        stats = {}
        if self.alg_type in ["MSE", "MSE-KL", "MSE-Contrastive"]:
            mean_mse_loss /= num_updates
            stats["mean_mse_loss"] = mean_mse_loss
            
        if self.alg_type in ["Contrastive", "MSE-Contrastive"]:
            mean_contrastive_loss /= num_updates
            mean_accuracy /= num_updates
            stats["mean_contrastive_loss"] = mean_contrastive_loss
            stats["contrastive_accuracy"] = mean_accuracy  # 对比学习准确率（正样本排在第几）
            
        self.encoder_storage.clear()
        return stats