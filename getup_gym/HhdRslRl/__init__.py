# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin


from getup_gym.HhdRslRl.tools.algo_registry import regist_algo, regist_net, regist_encoder, regist_encoder_alg
from getup_gym.HhdRslRl.tools.runner_register import regist_runner

from getup_gym.HhdRslRl.Modules.mlp_encoder import MlpEncoder
from getup_gym.HhdRslRl.Modules.mlp_encoder import MlpEncoderVae
from getup_gym.HhdRslRl.Modules.mlp_encoder_decoder import MlpEncoderDecoder
from getup_gym.HhdRslRl.Modules.simple_unet import SimpleUNet
from getup_gym.HhdRslRl.Modules.actor_critic import ActorCritic
from getup_gym.HhdRslRl.Modules.mlp_unet import MlpUNet
from getup_gym.HhdRslRl.Modules.moe import MoeActorCritic
from getup_gym.HhdRslRl.Modules.muti_actor_critic import Actor_MutiCritic
from getup_gym.HhdRslRl.Modules.Resnet_model import ResNetEncoder, ResNetActorCritic
from getup_gym.HhdRslRl.Algorithms.TSppo import TSPPO
from getup_gym.HhdRslRl.Algorithms.ppo import PPO
from getup_gym.HhdRslRl.Algorithms.mse import EncoderMSE
from getup_gym.HhdRslRl.Algorithms.constrain_mse import EncoderContrastiveMSE

from getup_gym.HhdRslRl.Runner.TSon_policy_runner import TSOnPolicyRunner
from getup_gym.HhdRslRl.Runner.on_policy_runner import OnPolicyRunner

regist_algo("TSPPO", TSPPO)
regist_algo("PPO", PPO)
regist_algo("MSE", EncoderMSE)
regist_algo("Constrain-MSE", EncoderContrastiveMSE)

regist_net("ActorCritic", ActorCritic)
regist_net("Moe-ActorCritic", MoeActorCritic)
regist_net("Actor_MutiCritic", Actor_MutiCritic)
regist_net("ResNetActorCritic", ResNetActorCritic)

regist_encoder("MLP-Encoder", MlpEncoder)
regist_encoder("MLP-Encoder-Vae", MlpEncoderVae)
regist_encoder("MLP-Encoder-Decoder", MlpEncoderDecoder)
regist_encoder("UNet", SimpleUNet)
regist_encoder("MLP-UNet", MlpUNet)
regist_encoder("ResNetEncoder", ResNetEncoder)

regist_runner("TSOnPolicyRunner", TSOnPolicyRunner)
regist_runner("OnPolicyRunner", OnPolicyRunner)

