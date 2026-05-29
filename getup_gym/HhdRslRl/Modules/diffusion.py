import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt

def get_beta_schedule(beta_start, beta_end, num_diffusion_timesteps):
    """get the beta schedule from params given"""
    beta_schedule = torch.linspace(beta_start, beta_end, num_diffusion_timesteps)
    return beta_schedule

class GuassianDiffusion:
    """base diffusion model use guassian
    """
    
    def __init__(
        self, beta_schedule: torch.tensor, model: nn.Module, estimater: nn.Module, device
    ) -> None:
        #params
        self.beta_schedule = beta_schedule
        self.model = model.to(device)
        self.estimater = estimater.to(device)
        self.device = device
        self.num_diffusion_timesteps = self.beta_schedule.size(0)
        
        #diffusion model params init
        self.alpha = 1. - self.beta_schedule
        self.alpha_cumprod = torch.cumprod(self.alpha, axis=0)
        self.sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - self.alpha_cumprod)
        
        #transfrom the diffusion model params to device
        self.alpha, self.alpha_cumprod, self.sqrt_alpha_cumprod, self.sqrt_one_minus_alpha_cumprod = (
            self.alpha.to(self.device),
            self.alpha_cumprod.to(self.device),
            self.sqrt_alpha_cumprod.to(self.device),
            self.sqrt_one_minus_alpha_cumprod.to(self.device)
        )
        
    def forward_diffusion_sample(self, x0: torch.tensor, t: torch.tensor) -> tuple:
        """diffusion forward by adding noise to x0

        Args:
            x0 (torch.tensor): initial clean dataset
            t (int): noise time

        Returns:
            tuple: dataset with noise, noise
        """
        
        noise = torch.randn_like(x0)
        sqrt_alpha_cumprod_t = self.sqrt_alpha_cumprod[t].reshape(-1, 1, 1)
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alpha_cumprod[t].reshape(-1, 1, 1)
        
        return (sqrt_alpha_cumprod_t* x0 + sqrt_one_minus_alpha_cumprod_t* noise, noise)
    
    @torch.no_grad()
    def ddim_sample(
        self, noise_state_action: torch.tensor, num_sampling_steps: int = 10
    ) -> torch.tensor:
        """changed the noise state-action to clean state-action

        Args:
            noise_state_action (torch.tensor): observations group(o_{t-1} and o_{t})
            num_sampling_steps: sampling steps per step

        Returns:
            torch.tensor: state's correction value
        """
        
        self.model.eval()
        #sample
        batch_size = noise_state_action.size(0)
        #time estimater inference to get the noise timestep t
        t = self.estimater(noise_state_action)["zt"].squeeze(1) #remove the channel dim
        t_processed = (t* 0.0001).to(torch.int32) 
        current_state = noise_state_action.clone().unsqueeze(1)
        
        #params calculation
        unique_timesteps = torch.unique(t_processed)
        max_timestep = unique_timesteps.max().item()
        
        #create global timesteps
        global_timesteps = self._get_timesteps(max_timestep, num_sampling_steps)
        
        #create process state for all sample
        processed_mask = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        
        #sample start
        for t in global_timesteps[:-1]:
            #get t next
            t_next = global_timesteps[global_timesteps < t][0] if len(global_timesteps[global_timesteps < t]) > 0 else 0
            #find the sample need to be processed
            batch_mask = (t_processed >= t) & (~processed_mask)
            if not batch_mask.any():
                continue
            #obstain the current batch
            batch_states = current_state[batch_mask]
            batch_t = t* torch.ones(batch_states.shape[0], device=self.device, dtype=torch.long)
            
            #diffusion modelinference
            predicted_noise = self.model.forward(batch_states, batch_t)
            #sampled states
            sample_states = self._ddim_step(
                batch_states, predicted_noise, t, t_next, eta=0.0
            )
        
            #update state
            
            current_state[batch_mask] = sample_states
            processed_mask[batch_mask] = (t_processed[batch_mask] <= t_next)
            
        return current_state.squeeze(1)
    
    def ddpm_sample(self, noise_state_action: torch.tensor) -> torch.tensor:
        """sample using ddpm 

        Args:
            noise_state_action (torch.tensor): state action with noise

        Returns:
            torch.tensor: clean state action
        """
        self.model.eval()
        #sample
        batch_size = noise_state_action.size(0)
        #time estimater inference to get the noise timestep t
        t = self.estimater(noise_state_action).squeeze(1) #remove the channel dim
        t_processed = (t* 1).to(torch.int32) 
        current_state = noise_state_action.clone()
        
        #get the max timesteps in t_processed
        unique_timesteps = torch.unique(t_processed)
        max_timestep = unique_timesteps.max().item()
        
        #create time mask(from max_timestep to zero)
        time_mask = torch.arange(max_timestep, -1, -1, device=self.device)
        
        #sample
        for time_idx in time_mask:
            batch_mask = t_processed >= time_idx
            #get the batch state-action need to be 
            batch_state = current_state[batch_mask]
            
            #params get
            alpha = self.alpha[time_idx].reshape(-1, 1, 1)
            alpha_cumprod = self.alpha_cumprod[time_idx].reshape(-1, 1, 1)
            beta = self.beta_schedule[time_idx].reshape(-1, 1, 1)
            
            #obstain predicted noise
            predict_noise = self.model.forward(batch_state, t)
            
            #obstain noise
            noise = torch.randn_like(batch_state) if time_idx > 0 else torch.zeros_like(batch_state)
            
            #remove the noise in state-action
            current_state[batch_mask] = (1/ torch.sqrt(alpha)) * (
                batch_state - ((1 - alpha) / (torch.sqrt(1 - alpha_cumprod)))* predict_noise
            ) + torch.sqrt(beta)* noise
            
        return current_state
            
        
    def _get_timesteps(self, max_timestep: int, num_steps: int) -> torch.tensor:
        """get DDIM timesteps mask use max timestep and num step

        Args:
            max_timestep (int): the max timestep in mask
            num_steps (int): the middle step number

        Returns:
            torch.tensor: timestep mask
        """
        
        #if num steps big than max timestep, use arrange to create mask
        if num_steps >= max_timestep:
            return torch.arange(max_timestep, -1, -1, device=self.device)
        else:
            #get the timestep num need to be skip
            step_size = max_timestep // num_steps
            timesteps = torch.arange(max_timestep, -1, -step_size, device=self.device)
            #add zero to mask
            return torch.cat([timesteps, torch.tensor([0], device=self.device)])
        
    def _ddim_step(
        self, x_t: torch.tensor, predicted_noise: torch.tensor, t: int, t_next: int, eta: float
    ) -> torch.tensor:
        """sample step use ddim

        Args:
            x_t (torch.tensor): noise state-action 
            predicted_noise (torch.tensor): predicted noise from Unet
            t (int): the t noise step
            t_next (int): next time noise step
            eta (float): defalut is 0.0

        Returns:
            torch.tensor: next time state-action
        """
        alpha_cumprod_t = self.alpha_cumprod[t]
        alpha_cumprod_t_next = self.alpha_cumprod[t_next] if t_next >= 0 else torch.tensor(1.0)
        
        predicted_x0 = (x_t - self.sqrt_one_minus_alpha_cumprod[t]* predicted_noise) / self.sqrt_alpha_cumprod[t]
        direction = torch.sqrt(1 - alpha_cumprod_t_next - eta**2)* predicted_noise
        
        #add noise if eta > 0
        random_noise = torch.randn_like(x_t) if eta > 0 else torch.zeros_like(x_t)
        noise = torch.sqrt((1 - alpha_cumprod_t_next) / (1 - alpha_cumprod_t))* torch.sqrt(
            1 - alpha_cumprod_t / alpha_cumprod_t_next
        )* random_noise
        
        x_next = torch.sqrt(alpha_cumprod_t_next)* predicted_x0 + direction + noise
        return x_next
        
        
        
        
        