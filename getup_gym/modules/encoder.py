"""Encoder modules for teacher-student distillation."""

import torch
import torch.nn as nn
from torch.distributions import Normal


class TeacherEncoder(nn.Module):
    """Privileged teacher encoder processing full state observations."""

    def __init__(
        self,
        encoder_input_size,
        layer_size,
        encoder_hidden_dims=[256, 128],
        activation="elu",
    ):
        super().__init__()
        self.encoder_input_size = encoder_input_size
        self.layer_size = layer_size
        act = get_activation(activation)

        layers = []
        layers.append(nn.Linear(encoder_input_size, encoder_hidden_dims[0]))
        layers.append(act)
        for i in range(len(encoder_hidden_dims) - 1):
            layers.append(nn.Linear(encoder_hidden_dims[i], encoder_hidden_dims[i + 1]))
            layers.append(act)
        layers.append(nn.Linear(encoder_hidden_dims[-1], layer_size))
        self.encoder = nn.Sequential(*layers)
        Normal.set_default_validate_args = False
        print(f"Teacher Encoder MLP: {self.encoder}")

    def forward(self, obs: torch.Tensor) -> dict:
        zt = self.encoder(obs)
        return {"zt": zt}

    def backward(self, obs: torch.Tensor) -> dict:
        zt = self.encoder(obs)
        return {"zt": zt}


class StudentEncoder(nn.Module):
    """Student encoder processing proprioceptive history observations."""

    def __init__(
        self,
        encoder_input_size,
        layer_size,
        encoder_hidden_dims=[256, 128],
        activation="elu",
    ):
        super().__init__()
        self.encoder_input_size = encoder_input_size
        self.layer_size = layer_size
        act = get_activation(activation)

        layers = []
        layers.append(nn.Linear(encoder_input_size, encoder_hidden_dims[0]))
        layers.append(act)
        for i in range(len(encoder_hidden_dims) - 1):
            layers.append(nn.Linear(encoder_hidden_dims[i], encoder_hidden_dims[i + 1]))
            layers.append(act)
        layers.append(nn.Linear(encoder_hidden_dims[-1], layer_size))
        self.encoder = nn.Sequential(*layers)
        Normal.set_default_validate_args = False
        print(f"Student Encoder MLP: {self.encoder}")

    def forward(self, obs: torch.Tensor) -> dict:
        if obs.size(1) != self.encoder_input_size:
            raise ValueError(
                f"Input obs size mismatch: expected {self.encoder_input_size}, got {obs.size(1)}"
            )
        zt = self.encoder(obs)
        return {"zt": zt}

    def backward(self, obs: torch.Tensor) -> dict:
        if obs.size(1) != self.encoder_input_size:
            raise ValueError(
                f"Input obs size mismatch: expected {self.encoder_input_size}, got {obs.size(1)}"
            )
        zt = self.encoder(obs)
        return {"zt": zt}


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        raise ValueError(f"Invalid activation function: {act_name}")
