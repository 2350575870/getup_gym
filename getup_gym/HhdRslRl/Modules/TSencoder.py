import torch
import torch.nn as nn
from torch.distributions import Normal
import torch.optim as optim
from getup_gym.HhdRslRl.Storage.encoder_storage import EncoderStorage
from getup_gym.HhdRslRl.basement.base_modules.ModulesBase import ModulesBase

class TSEnocder(ModulesBase):
    """encoder used in teacher-student"""
    def __init__(
            self, 
            Teacher_Encoder_Input_Size: dict,
            Teacher_Encoder_Layer_Size: dict,
            Student_Encoder_Input_Size: int,
            Layer_Size: int,
            device,
            teacher_ativation="tanh",
            student_ativation="tanh",
            teacher_hidden_dims=[256, 128],
            student_hidden_dims=[256, 128],
            teacher_encoder_std_init = 1.0,
            student_encoder_std_init = 1.0,
    ):
        #params initializaion
        self.device = device
        self.teacher_encoder_input_size: dict = Teacher_Encoder_Input_Size
        self.teacher_encoder_layer_size: dict = Teacher_Encoder_Layer_Size
        self.student_encoder_input_size: int = Student_Encoder_Input_Size
        self.layer_size: int = Layer_Size 
        self.teacher_encoder_std_init = teacher_encoder_std_init
        self.student_encoder_std_init = student_encoder_std_init

        teacher_ativation = get_activation(teacher_ativation) #获取激活函数信息
        student_ativation = get_activation(student_ativation)

        teacher_encoder = []
        teacher_encoder.append(nn.Linear(Teacher_Encoder_Input_Size, teacher_hidden_dims[0]))
        teacher_encoder.append(teacher_ativation)
        for i in range(len(student_hidden_dims) - 1):
            teacher_encoder.append(nn.Linear(teacher_hidden_dims[i], teacher_hidden_dims[i + 1]))
            teacher_encoder.append(teacher_ativation)
        teacher_encoder(nn.Linear(teacher_hidden_dims[-1], Teacher_Encoder_Layer_Size))
        self.teacher_encoder = nn.Sequential(*teacher_encoder)

        student_encoder = []
        student_encoder.append(nn.Linear(self.student_encoder_input_size, student_hidden_dims[0]))
        student_encoder.append(student_ativation)
        for i in range(len(student_hidden_dims) - 1):
            student_encoder.append(nn.Linear(student_hidden_dims[i], student_hidden_dims[i + 1]))
            student_encoder.append(student_ativation)
        student_encoder.append(nn.Linear(student_hidden_dims[-1], Layer_Size))
        self.student_encoder = nn.Sequential(*student_encoder)

        Normal.set_default_validate_args = False

    def teacher_forward(self, obs: torch.tensor) -> torch.tensor:
        return self.teacher_encoder(obs)
    
    def student_forward(self, obs: torch.tensor) -> torch.tensor:
        return self.student_encoder(obs)
    
    def forward(self, teacher_obs: torch.tensor, student_obs: torch.tensor) -> list:
        """inference use different mode"""
        return {
            "zt_t": self.teacher_forward(teacher_obs),
            "zt_s": self.student_forward(student_obs)
        }



def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None