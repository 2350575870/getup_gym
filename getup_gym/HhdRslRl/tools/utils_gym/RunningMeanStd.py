import torch
import numpy as np
import torch.nn as nn


class RunningMeanStd(nn.Module):
    # Dynamically calculate mean and std
    def __init__(self, shape, device):  # shape:the dimension of input data
        super(RunningMeanStd, self).__init__()

        self.register_buffer("mean", torch.zeros(shape).to(device))
        self.register_buffer("S", torch.zeros(shape).to(device))
        self.register_buffer("n", torch.zeros(1).to(device))

    def update(self, x):
        self.n += 1
        if self.n[0] == 1:
            self.mean = x
            self.std = x
        else:
            old_mean = self.mean.clone()
            self.mean = old_mean + (x - old_mean) / self.n[0]
            self.S = self.S + (x - old_mean) * (x - self.mean)
            self.std = torch.sqrt(self.S / self.n[0])


class RewardScaling(nn.Module):
    def __init__(self, shape, gamma, device):
        super(RewardScaling, self).__init__()
        self.device = device
        self.shape = shape  # reward shape=1
        self.gamma = gamma  # discount factor
        self.running_ms = RunningMeanStd(shape=self.shape, device=self.device)
        self.R = torch.zeros(self.shape).to(self.device)

    def __call__(self, x, enable=True):
        if not enable:
            return x

        self.R = self.gamma * self.R + x
        self.running_ms.update(self.R)
        x = x / (self.running_ms.std + 1e-8)  # Only divided std
        return x

    def reset(self, idx):  # When an episode is done,we should reset 'self.R'
        self.R[idx] = 0


class RLRunningMeanStd(nn.Module):
    def __init__(self, insize, epsilon=1e-05, per_channel=False, norm_only=False):
        super(RLRunningMeanStd, self).__init__()
        self.insize = insize
        self.epsilon = epsilon

        self.norm_only = norm_only
        self.per_channel = per_channel
        if per_channel:
            if len(self.insize) == 3:
                self.axis = [0, 2, 3]
            if len(self.insize) == 2:
                self.axis = [0, 2]
            if len(self.insize) == 1:
                self.axis = [0]
            in_size = self.insize[0]
        else:
            self.axis = [0]
            in_size = insize

        self.register_buffer("running_mean", torch.zeros(in_size, dtype=torch.float64))
        self.register_buffer("running_var", torch.ones(in_size, dtype=torch.float64))
        self.register_buffer("count", torch.ones((), dtype=torch.float64))

    def _update_mean_var_count_from_moments(self, mean, var, count, batch_mean, batch_var, batch_count):
        with torch.no_grad():
            delta = batch_mean - mean
            tot_count = count + batch_count

            new_mean = mean + delta * batch_count / tot_count
            m_a = var * count
            m_b = batch_var * batch_count
            M2 = m_a + m_b + delta**2 * count * batch_count / tot_count
            new_var = M2 / tot_count
            new_count = tot_count
        return new_mean, new_var, new_count

    def forward(self, input: torch.Tensor, denorm=False):
        tensor_size = input.shape
        input = input.view(-1, self.insize)
        if self.training:
            mean = input.mean(self.axis)  # along channel axis
            var = input.var(self.axis)
            self.running_mean, self.running_var, self.count = self._update_mean_var_count_from_moments(
                self.running_mean, self.running_var, self.count, mean, var, input.size()[0]
            )

        # change shape
        if self.per_channel:
            if len(self.insize) == 3:
                current_mean = self.running_mean.view([1, self.insize[0], 1, 1]).expand_as(input)
                current_var = self.running_var.view([1, self.insize[0], 1, 1]).expand_as(input)
            if len(self.insize) == 2:
                current_mean = self.running_mean.view([1, self.insize[0], 1]).expand_as(input)
                current_var = self.running_var.view([1, self.insize[0], 1]).expand_as(input)
            if len(self.insize) == 1:
                current_mean = self.running_mean.view([1, self.insize[0]]).expand_as(input)
                current_var = self.running_var.view([1, self.insize[0]]).expand_as(input)
        else:
            current_mean = self.running_mean
            current_var = self.running_var
        # get output

        if denorm:
            y = torch.clamp(input, min=-5.0, max=5.0)
            y = torch.sqrt(current_var.float() + self.epsilon) * y + current_mean.float()
        else:
            if self.norm_only:
                y = input / torch.sqrt(current_var.float() + self.epsilon)
            else:
                y = (input - current_mean.float()) / torch.sqrt(current_var.float() + self.epsilon)
                y = torch.clamp(y, min=-5.0, max=5.0)
        return y.reshape(tensor_size)
