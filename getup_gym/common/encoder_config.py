from getup_gym.common.base_config import BaseConfig

from dataclasses import dataclass
from typing import List
import torch
from dataclasses import MISSING

@dataclass
class UNetConfig:
    """UNet model config class"""
    
    # params must need to init
    input_channel_size: int = MISSING
    """the input obs channel size"""
    
    bottleneck_channel_size: int = MISSING
    """the bottleneck channel size"""
    
    output_channel_size: int = MISSING
    """output channel size"""
    
    hidden_channel_dim: List[int] = MISSING
    """the encoder and decoder's hidden channel size"""
    
    time_embed_dim: List[int] = MISSING
    """time embed model dim"""
    
    # params have defalut value
    kernel_size: int = 3
    """cov's kernel size"""
    
    padding: int = 1
    """cov's padding size"""
    
    activation: str = "relu"
    """activation name"""
    
    def validate(self):
        """验证配置参数"""
        if len(self.hidden_channel_dim) < 1:
            raise ValueError("hidden_channel_dim must have at least one element")
        if len(self.time_embed_dim) < 2:
            raise ValueError("time_embed_dim must have at least two elements")
        if self.kernel_size % 2 == 0:
            raise ValueError("kernel_size should be odd for symmetric padding")
        if not isinstance(self.device, torch.device):
            raise ValueError("device must be a torch.device instance")
        
        
@dataclass
class EstimaterConfig:
    """config for time estimater
    """
    encoder_input_size: int = MISSING
    """the encoder input observations size"""
    
    layer_size: int = MISSING
    """the encoder's output observations size"""
    
    encoder_hidden_dims: List[int] = MISSING
    """encoder's hidden dims"""
    
    activation: str = MISSING
    """the activation encoder used"""
    
@dataclass
class MlpEncoderConfig:
    """config for all of mlp encoder
    """
    
    encoder_hidden_dims: list = MISSING
    """encoder hidden dims
    """
    
    activation: str = MISSING
    """encoder's activation type"""
    
@dataclass
class UNetConfigAlg:
    """unet config in alg setting
    """
    
    # params must need to init
    input_channel_size: int = MISSING
    """the input obs channel size"""
    
    bottleneck_channel_size: int = MISSING
    """the bottleneck channel size"""
    
    output_channel_size: int = MISSING
    """output channel size"""
    
    hidden_channel_dim: List[int] = MISSING
    """the encoder and decoder's hidden channel size"""
    
    time_embed_dim: List[int] = MISSING
    """time embed model dim"""
    
    # params have defalut value
    kernel_size: int = 3
    """cov's kernel size"""
    
    padding: int = 1
    """cov's padding size"""
    
    activation: str = "relu"
    """activation name"""

@dataclass   
class MlpUNetConfig:
    """MLP-type UNet config in alg setting
    """
    
    botteneck_dim: int = MISSING
    """the botteneck size"""
    
    activation: str = MISSING
    """activation's type"""
    
    time_embed_dim: list = MISSING
    """time embed dims"""
    
@dataclass
class EncoderAlgConfig:
    """encoder train alg config
    """
    learning_rate: float = MISSING
    """encoder learning rate"""
    
    num_mini_batches: int = MISSING
    """mini batches num"""
    
    num_learning_epochs: int = MISSING #test good is 3,maybe 4 is better
    """learning epochs each mini batch"""
    
    max_grad_norm: float = MISSING
    """max grad norm"""
    
    mse_loss_coef: float = MISSING
    """mse loss coef"""
    
    student_kl_coef: float = MISSING
    """student kl loss coef"""
    
    alg_type: str = MISSING
    """alg for training encoder
    """
    
@dataclass
class EncoderContrastiveAlgConfig:
    """encoder train alg config
    """
    learning_rate: float = MISSING
    """encoder learning rate"""
    
    num_mini_batches: int = MISSING
    """mini batches num"""
    
    num_learning_epochs: int = MISSING #test good is 3,maybe 4 is better
    """learning epochs each mini batch"""
    
    max_grad_norm: float = MISSING
    """max grad norm"""
    
    mse_loss_coef: float = MISSING
    """mse loss coef"""
    
    student_kl_coef: float = MISSING
    """student kl loss coef"""
    
    alg_type: str = MISSING
    """alg for training encoder
    """
    
    temperature: float = 0.07
                
    contrastive_coef: float = 1.0 
            
    use_projection_head: bool = True   
       
    projection_output_dim: int = 12       

@dataclass
class TrainingAlgConfig:
    """ training's alg config
    """
    algorithms: str = MISSING
    """alg using in training""" 
    
    teacher_encoder_name: str = MISSING
    """teacher encoder type name"""
    
    teacher_encoder_config: MlpEncoderConfig  = MISSING 
    """teacher's encoder config"""
    
    student_encoder_name: str = MISSING
    """student encoder type name"""
    
    student_encoder_config: MlpEncoderConfig = MISSING
    """student encoder config"""
    
    student_resume_check_point: str = MISSING
    """student resume place"""
    
    teacher_encoder_alg_type: str = MISSING
    """teacher encoder's alg using in training"""
    
    student_encoder_alg_class_name: str = MISSING
    """student encoder's alg using in training"""
    
    student_encoder_alg_config: EncoderAlgConfig = MISSING
    """student encoder training alg config"""
    
    student_encoder_inference_decimation: str = MISSING
    """student encoder inference decimation"""
    
    
@dataclass
class ForceGuidanceConfig:
    """for guidance alg config"""
    using_force_guidance: bool = MISSING
    
@dataclass
class SymmetricalConfig:
    """symmetrical config"""
    
    using_symmetrical_loss: bool = MISSING
    """if using symmetrical loss or not"""
    
    symmetrical_loss_coef: float = MISSING
    """loss's coef"""
    
@dataclass
class PolicyConfig:
    """policy config"""
    
    init_noise_std: float = MISSING
    """init std"""
                       
    actor_hidden_dims: list = MISSING
    """actor's hidden dims"""
        
    critic_hidden_dims: list = MISSING
    """critic's hidden dims"""
        
    activation: str = MISSING  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
    """policy's activation type"""
    
@dataclass
class MutiActorCriticConfig(PolicyConfig):
    """config for muti-actor-critic"""
    
    num_critic: int = MISSING
    """critic's num"""
    
@dataclass
class AlgorithmsPPOConfig:
    """alg-PPO config"""
    value_loss_coef: float = MISSING
    
    use_clipped_value_loss: bool = MISSING
    
    clip_param: float = MISSING
    
    entropy_coef: float = MISSING
    
    num_learning_epochs: int = MISSING
    
    num_mini_batches: int = MISSING  # mini batch size = num_envs*nsteps / nminibatches
    
    learning_rate: float = MISSING  # 5.e-4
    
    schedule: str = MISSING  # could be adaptive, fixed
    
    gamma: float = MISSING
    
    lam: float = MISSING
    
    desired_kl: float = MISSING
    
    max_grad_norm: float = MISSING
    
    teacher_encoder_alg_type: str = None
    """teacher encoder's alg using in training"""
    
@dataclass
class MoEConfig:
    """config for MoE
    """
    
    policy_position_list: list = MISSING
    """policy's save position
    """
    
    teacher_encoder_position_list: list = MISSING
    
    student_encoder_position_list: list = MISSING
    
@dataclass
class ResNetActorCriticConfig:
    """config for ResNet type actorcritic
    """
    actor_hidden_dims: int = MISSING
    """the actor ResNet model's width
    """
    
    dropout_prop: float = MISSING
    """the dropout proportion. if not use dropout, set this value to zero
    """
    
    actor_activation: str = MISSING
    """the actor's activation func type. defalut is swish
    """
    
    actor_num_hidden_dims: int = MISSING
    """the number of actor ResNet model's layer, also the model's length
    """
    
    actor_block_num: int = MISSING
    """actor model's block number. in the final of the block group, we will use the res link once
    """
    
    critic_hidden_dims: list = MISSING
    """critic hidden dims
    """
    
    critic_activation: str = MISSING
    """critic activation
    """
    
    init_noise_std: float = MISSING
    """model output noise init value
    """

@dataclass
class ResNetEncoderConfig:
    """config for ResNet type encoder
    """
    num_hidden_dims: int = MISSING
    """model's width
    """
    
    hidden_dims: int = MISSING
    """model's length
    """
    
    dropout_prop: float = MISSING
    """dropout proportion
    """
    
    activation: str = MISSING
    """defalute value is swish
    """
    
    block_num: int = MISSING
    """block num
    """
    
@dataclass
class WorldModelConfig:
    """config for world model
    """
    using_world_model: bool = MISSING
    """if using world model or not
    """
    
    world_model_type: str = MISSING
    """world model type name
    """
    
    model_config: MlpEncoderConfig = MISSING
    """world model config
    """
    
    buffer_max_lenth: int = MISSING
    """how much the future state the buffer must have
    """
    
    
