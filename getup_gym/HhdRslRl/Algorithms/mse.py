import torch
import torch.nn as nn
from getup_gym.HhdRslRl.Storage.encoder_storage import EncoderStorage
from getup_gym.HhdRslRl.basement.base_modules.ModulesBase import ModulesBase
import torch.optim as optim
from getup_gym.HhdRslRl.basement.base_algorithm.AlgorithmBase import AlgorithmBase
from torch.distributions import Normal, kl_divergence

class EncoderMSE(AlgorithmBase):
    encoder: ModulesBase
    def __init__(
            self,
            num_trasition_per_env, 
            num_envs, 
            encoder: ModulesBase,
            teacher_encoder: list,
            device,
            learning_rate = 8e-4,
            num_mini_batches = 3,
            num_learning_epochs = 3,
            max_grad_norm = 0.8,
            mse_loss_coef = 1.0,
            student_kl_coef = 0.1,
            alg_type = "MSE"
    ):
        #base params init
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
        #choose using which alg to train encoder
        self.alg_type = alg_type

        #encoder params init
        self.encoder: ModulesBase = encoder
        self.teacher_encoder: ModulesBase = teacher_encoder
        self.student_optimizer = optim.Adam(self.encoder.parameters(), self.encoder_learning_rate)
        self.mse_function = nn.MSELoss()

        #encoder storage and transition init
        self.encoder_storage = EncoderStorage(
            num_trasition_per_env, 
            num_envs, 
            self.layer_size, 
            self.student_encoder_input_size, 
            self.teacher_encoder.encoder_input_size,
            device,
        )
        self.encoder_transition = EncoderStorage.EncoderTransition()
        
    
    def act(
            self, 
            _zt_t: torch.tensor,
            _student_encoder_obs: torch.tensor, 
            _teacher_encoder_obs: torch.tensor = None
    ) -> None:
        self.encoder_transition.zt_t = _zt_t
        self.encoder_transition.student_obs = _student_encoder_obs
        self.encoder_transition.teacher_obs = _teacher_encoder_obs
        self.encoder_storage.add_transitions(self.encoder_transition)
        self.encoder_transition.clear()

    def update(self):
        mean_mse_loss = 0

        generater = (
            self.encoder_storage.mini_batch_generator(self.num_mini_batch, self.num_learning_epochs)
            if self.alg_type == "MSE" 
            else self.encoder_storage.VAE_mini_batch_generator(self.num_mini_batch, self.num_learning_epochs)
        )
        for (
            zt_t_batch, student_obs_batch, teacher_obs_batch
        ) in generater:
            if self.alg_type == "MSE":
                #output and gradient params get
                zt_s = self.encoder.backward(student_obs_batch)["zt"]
                zt_t = zt_t_batch
                #mse loss calculation
                mse_loss = self.mse_function(zt_s, zt_t.detach())

                loss = self.mse_loss_coef* mse_loss
            elif self.alg_type in ["MSE-KL"]:
                student_encoder_dict = self.encoder.backward(student_obs_batch)
                teacher_encoder_dict = self.teacher_encoder.backward(teacher_obs_batch)
                mu_t = teacher_encoder_dict["mu"]

                #mse loss calculation
                mse_loss = self.mse_function(student_encoder_dict["mu"], mu_t.detach())

                #kl loss calcalation
                # kl_loss = torch.mean((0.5* logvar_t.detach() - 0.5* (student_encoder_dict["logvar_s"])).pow()) #0.5* logvar_t.detach() - 0.5 - 
                dist_s = Normal(student_encoder_dict["mu"], torch.exp(0.5* student_encoder_dict["logvar"]))
                dist_t = Normal(teacher_encoder_dict["mu"].detach(), torch.exp(0.5* teacher_encoder_dict["logvar"].detach()))
                kl_loss = kl_divergence(dist_s, dist_t).mean()

                loss = self.mse_loss_coef* mse_loss + self.student_kl_coef* kl_loss
                
                
            self.student_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.encoder.parameters(), self.max_grad_norm)
            self.student_optimizer.step()
            mean_mse_loss += mse_loss
                
            # print("student_backward start")
            # for name, param in self.encoder.teacher_encoder_list.named_parameters():
            #     if param.grad is not None:
            #         print(f"{name}: {param.grad.norm().item()}, shape: {param.shape}")
            #     else:
            #         print(f"{name}: None, shape is {param.shape}")

            # print("student encoder backward gradient is: ")
            # for name, param in self.encoder.named_parameters():
            #     if param.grad is not None:
            #         print(f"{name}: {param.grad.norm().item()}, shape: {param.shape}")
            #     else:
            #         print(f"{name}: None, shape is {param.shape}")

        #     # print("totol encoder gradient is: ")
        #     # for name, param in self.encoder.named_parameters():
        #     #     if param.grad is not None:
        #     #         print(f"{name}: {param.grad.norm().item()}, shape: {param.shape}")
        #     #     else:
        #     #         print(f"{name}: None, shape is {param.shape}")


        num_updates = self.num_learning_epochs* self.num_mini_batch
        mean_mse_loss /= num_updates
        self.encoder_storage.clear()
        return {"mean_mse_loss": mean_mse_loss}

