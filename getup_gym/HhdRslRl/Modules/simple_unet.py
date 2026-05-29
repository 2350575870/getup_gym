import torch
import torch.nn as nn

class SimpleUNet(nn.Module):
    """simple UNet, using to training the diffusion model"""
    
    def __init__(
        self, 
        input_channel_size: int, 
        bottleneck_channel_size: int, 
        output_channel_size: int,
        hidden_channel_dim: list,
        kernel_size: int,
        padding: int,
        activation: str,
        time_embed_dim: list,
        layer_size: int,
        encoder_input_size: int,
        device,
    ) -> None:
        super(SimpleUNet, self).__init__()
        #params init
        self.input_channel_size = input_channel_size
        self.bottleneck_channel_size = bottleneck_channel_size
        self.hidden_channel_dim = hidden_channel_dim
        self.kernel_size = kernel_size
        self.padding = padding
        self.device = device
        self.encoder_input_size = encoder_input_size
        self.layer_size = layer_size
        
        #encoder model init
        self.simpleUNet_encoder = nn.ModuleList()
        self.simpleUNet_encoder.append(
            self._block(input_channel_size, hidden_channel_dim[0], kernel_size, activation, padding).to(self.device)
        )
        for dim in range(len(hidden_channel_dim) - 1):
            self.simpleUNet_encoder.append(
                self._block(hidden_channel_dim[dim],  hidden_channel_dim[dim+1], kernel_size, activation, padding).to(self.device)
            )
        
        #bottleneck init
        self.botteneck = self._block(hidden_channel_dim[-1], bottleneck_channel_size, kernel_size, activation, padding).to(self.device)
        
        #decoder init
        self.simpleUNet_decoder = nn.ModuleList()
        self.simpleUNet_decoder.append(
            self._block(
                bottleneck_channel_size + hidden_channel_dim[-1], hidden_channel_dim[-1], kernel_size, activation, padding
            ).to(self.device)
        )
        for dim in range(len(hidden_channel_dim) - 1):
            inverse_dim = len(hidden_channel_dim) - 1 - dim
            self.simpleUNet_decoder.append(
                self._block(
                    hidden_channel_dim[inverse_dim] + hidden_channel_dim[inverse_dim - 1], hidden_channel_dim[inverse_dim - 1],
                    kernel=kernel_size, activation=activation, padding=padding
                ).to(self.device)
            )
        
        #output layer init
        self.output_layer = nn.Conv1d(hidden_channel_dim[0], output_channel_size, kernel_size=1)
        
        #up sample and down sample
        self.pool = nn.MaxPool1d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        
        #time embedding
        time_embed = []
        for dim in range(len(time_embed_dim) - 2):
            time_embed.append(
                nn.Linear(time_embed_dim[dim], time_embed_dim[dim + 1])
            )
            time_embed.append(get_activation(activation))
        time_embed.append(nn.Linear(time_embed_dim[-2], time_embed_dim[-1]))
        self.time_embed = nn.Sequential(*time_embed).to(self.device)
        
        #transform layer
        self.transfrom_layer = None
        if layer_size is not None:
            self.transfrom_layer = nn.Linear(encoder_input_size, layer_size)
                       
        print("using UNet!")
        print(f"UNet encoder is: {self.simpleUNet_encoder}")
        print(f"UNet decoder is: {self.simpleUNet_decoder}")
        print(f"UNet time embed network is: {self.time_embed}")
     
    def _block(
            self, input_channel: int, output_channel: int, kernel: int, activation: str, padding: int
        ) -> None:
        """for every blocks in UNet.
        
        Args: 
            input_channel: channel size of the cov's input
            output_channel: channel size of the cov's output
            kernel: kernel size of the cov's kernel, defalut is 3
            activation: defalut is nn.Relu
            padding: 1
        Returns:
            the block of the UNet
        """
        activation = get_activation(activation)
        return nn.Sequential(
            nn.Conv1d(input_channel, output_channel, kernel, padding=padding),
            activation,
            nn.Conv1d(output_channel, output_channel, kernel, padding=padding),
            activation
        )
        
    def forward(self, x_input: torch.tensor, t: torch.tensor) -> torch.tensor:
        """the UNet inference function

        Args:
            x_input (torch.tensor): the data observation
            t (int): time of noise

        Returns:
            torch.tensor: inference tensor
        """
        t_embed = self.time_embed(t.unsqueeze(-1).float()).unsqueeze(-1)
        encoder_layer_output = []
        
        #encoder inference
        middle_layer_output = self.simpleUNet_encoder[0](x_input)
        encoder_layer_output.append(middle_layer_output)
        
        for enc_dim in range(len(self.hidden_channel_dim) - 1):
            middle_layer_output = self.simpleUNet_encoder[enc_dim + 1](self.pool(middle_layer_output))
            encoder_layer_output.append(middle_layer_output)
        
        #botteneck inference
        bottleneck_layer_output = self.botteneck(self.pool(middle_layer_output))
        if bottleneck_layer_output.size(1) != t_embed.size(1):
            raise ValueError("the bottleneck model size is not equal to t_embedding size!")
        bottleneck_layer_output = bottleneck_layer_output + t_embed
        
        #decoder inference
        middle_layer_output = self.simpleUNet_decoder[0](
            torch.cat([self.upsample(bottleneck_layer_output), encoder_layer_output[-1]], dim=1)
        )
        for dec_dim in range(len(self.hidden_channel_dim) - 1):
            inverse_dec_dim = len(self.hidden_channel_dim) - 2 - dec_dim
            middle_layer_output = self.simpleUNet_decoder[dec_dim + 1](
                torch.cat([
                    self.upsample(middle_layer_output), encoder_layer_output[inverse_dec_dim]
                ], dim=1)
            )
            
        #output layer
        output = self.output_layer(middle_layer_output)
        
        if self.transfrom_layer is not None:
            #delete the second dim and inference through transform layer
            output = self.transfrom_layer(output.squeeze(1))
        
        return output
    
    def backward(self, x_input: torch.tensor, t: torch.tensor) -> torch.tensor:
        """the UNet inference function

        Args:
            x_input (torch.tensor): the data observation
            t (int): time of noise

        Returns:
            torch.tensor: inference tensor
        """
        t_embed = self.time_embed(t.unsqueeze(-1).float()).unsqueeze(-1)
        encoder_layer_output = []
        
        #encoder inference
        middle_layer_output = self.simpleUNet_encoder[0](x_input)
        encoder_layer_output.append(middle_layer_output)
        
        for enc_dim in range(len(self.hidden_channel_dim) - 1):
            middle_layer_output = self.simpleUNet_encoder[enc_dim + 1](self.pool(middle_layer_output))
            encoder_layer_output.append(middle_layer_output)
        
        #botteneck inference
        bottleneck_layer_output = self.botteneck(self.pool(middle_layer_output))
        if bottleneck_layer_output.size(1) != t_embed.size(1):
            raise ValueError("the bottleneck model size is not equal to t_embedding size!")
        bottleneck_layer_output = bottleneck_layer_output + t_embed
        
        #decoder inference
        middle_layer_output = self.simpleUNet_decoder[0](
            torch.cat([self.upsample(bottleneck_layer_output), encoder_layer_output[-1]], dim=1)
        )
        for dec_dim in range(len(self.hidden_channel_dim) - 1):
            inverse_dec_dim = len(self.hidden_channel_dim) - 2 - dec_dim
            middle_layer_output = self.simpleUNet_decoder[dec_dim + 1](
                torch.cat([
                    self.upsample(middle_layer_output), encoder_layer_output[inverse_dec_dim]
                ], dim=1)
            )
            
        #output layer
        output = self.output_layer(middle_layer_output)
        
        if self.transfrom_layer is not None:
            #delete the second dim and inference through transform layer
            output = self.transfrom_layer(output.squeeze(1))
        
        return output

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


# 简单的UNet模型
class SimpleUNet2(nn.Module):
    def __init__(self, CHANNELS: int):
        super().__init__()
        
        # 编码器
        self.enc1 = self._block(CHANNELS, 64)
        self.enc2 = self._block(64, 128)
        self.enc3 = self._block(128, 256)
        
        # 瓶颈层
        self.bottleneck = self._block(256, 512)
        
        # 解码器
        self.dec1 = self._block(512 + 256, 256)
        self.dec2 = self._block(256 + 128, 128)
        self.dec3 = self._block(128 + 64, 64)
        
        # 输出层
        self.output = nn.Conv1d(64, CHANNELS, kernel_size=1)
        
        # 下采样和上采样
        self.pool = nn.MaxPool1d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        
        # 时间嵌入
        self.time_embed = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
        )
        
    def _block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(),
        )
    
    def forward(self, x, t):
        # 时间嵌入
        t_embed = self.time_embed(t.unsqueeze(-1).float()).unsqueeze(-1)
        
        # 编码器
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        
        # 瓶颈层（添加时间信息）
        bottleneck = self.bottleneck(self.pool(enc3))
        bottleneck = bottleneck + t_embed
        
        # 解码器
        dec1 = self.dec1(torch.cat([self.upsample(bottleneck), enc3], dim=1))
        dec2 = self.dec2(torch.cat([self.upsample(dec1), enc2], dim=1))
        dec3 = self.dec3(torch.cat([self.upsample(dec2), enc1], dim=1))
        
        return self.output(dec3)