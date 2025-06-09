# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import os
from typing import Callable, List, Optional, Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import logging
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.unets.unet_2d_blocks import DownBlock2D, UpBlock2D, DownEncoderBlock2D, UpDecoderBlock2D
from diffusers.models.resnet import ResnetBlock2D
from diffusers.models.adapter import LightAdapterBlock, AdapterBlock
from diffusers.models.transformers.transformer_2d import Transformer2DModel


logger = logging.get_logger(__name__)


def get_down_block_2d(
    in_channels: int, out_channels: int, num_layers: int=2, add_downsample: bool=True
) -> nn.Module:
    return DownEncoderBlock2D(
        in_channels=in_channels,
        out_channels=out_channels,
        # temb_channels=None,
        num_layers=num_layers,
        add_downsample=add_downsample
    )

def get_up_block_2d(
    in_channels: int, out_channels: int, num_layers: int=2, add_upsample: bool=True
) -> nn.Module:
    return UpDecoderBlock2D(
        in_channels=in_channels,
        out_channels=out_channels,
        temb_channels=None,
        num_layers=num_layers,
        add_upsample=add_upsample
    )
        

def get_resnet_block(in_channels: int, out_channels: Optional[int] = None) -> nn.Module:
    """ 
    Same as the original implementation, but no time emb
    """
    return ResnetBlock2D(
        in_channels=in_channels,
        out_channels=out_channels,
        temb_channels=None,
        non_linearity='swish',
    )

class ConditioningEmbedding(nn.Module):
    """
    Quoting from https://arxiv.org/abs/2302.05543: "Stable Diffusion uses a pre-processing method similar to VQ-GAN
    [11] to convert the entire dataset of 512 × 512 images into smaller 64 × 64 “latent images” for stabilized
    training. This requires ControlNets to convert image-based conditions to 64 × 64 feature space to match the
    convolution size. We use a tiny network E(·) of four convolution layers with 4 × 4 kernels and 2 × 2 strides
    (activated by ReLU, channels are 16, 32, 64, 128, initialized with Gaussian weights, trained jointly with the full
    model) to encode image-space conditions ... into feature maps ..."
    """

    def __init__(
        self,
        conditioning_embedding_channels: int,
        conditioning_channels: int = 3,
        block_out_channels: Tuple[int, ...] = (16, 32, 96, 256),
    ):
        super().__init__()

        self.conv_in = nn.Conv2d(conditioning_channels, block_out_channels[0], kernel_size=3, padding=1)

        self.blocks = nn.ModuleList([])

        for i in range(len(block_out_channels) - 1):
            channel_in = block_out_channels[i]
            channel_out = block_out_channels[i + 1]
            self.blocks.append(nn.Conv2d(channel_in, channel_in, kernel_size=3, padding=1))
            self.blocks.append(nn.Conv2d(channel_in, channel_out, kernel_size=3, padding=1, stride=2))

        self.conv_out = nn.Conv2d(block_out_channels[-1], conditioning_embedding_channels, kernel_size=3, padding=1)

    def forward(self, conditioning):
        embedding = self.conv_in(conditioning)
        embedding = F.silu(embedding)

        for block in self.blocks:
            embedding = block(embedding)
            embedding = F.silu(embedding)

        embedding = self.conv_out(embedding)

        return embedding


class EnvEncoder(ModelMixin, ConfigMixin):
    r"""
    A simple ResNet-like model that accepts images containing control signals such as keyposes and depth. The model
    generates multiple feature maps that are used as additional conditioning in [`UNet2DConditionModel`]. 

    This model inherits from [`ModelMixin`]. Check the superclass documentation for the generic methods the library
    implements for all the model (such as downloading or saving, etc.)

    Parameters:
        in_channels (`int`, *optional*, defaults to 3):
            Number of channels of Aapter's input(*control image*). Set this parameter to 1 if you're using gray scale
            image as *control image*.
        channels (`List[int]`, *optional*, defaults to `(320, 640, 1280, 1280)`):
            The number of channel of each downsample block's output hidden state. The `len(block_out_channels)` will
            also determine the number of downsample blocks in the Adapter.
        num_res_blocks (`int`, *optional*, defaults to 2):
            Number of ResNet blocks in each downsample block.
        downscale_factor (`int`, *optional*, defaults to 8):
            A factor that determines the total downscale factor of the Adapter.
    """

    @register_to_config
    def __init__(
        self,
        in_channels: int = 3,
        channels: Tuple[int] = (320, 640, 1280, 1280),
        num_res_blocks: int = 2,
        latent_encoder: bool = False,
        in_labels: Tuple[str] = ('env_ldr', 'env_log', 'env_nrm')
    ):
        super().__init__()

        self.in_labels = in_labels

        # Let's bring in the controlnet's controlnet_cond_embedding
        # self.conv_in = 
        if latent_encoder:
            self.conv_in = nn.Conv2d(in_channels, channels[0], kernel_size=3, padding=1)
        else:
            self.conv_in = ConditioningEmbedding(
                conditioning_embedding_channels=channels[0],
                block_out_channels=[16, 32, 96, 256],
                conditioning_channels=in_channels,
            )

        self.down_blocks = []
        output_channels = channels[0]
        for i, channel in enumerate(channels):
            input_channels = output_channels
            output_channels = channel
            is_first_block = i == 0

            down_block = get_down_block_2d(
                in_channels=input_channels,
                out_channels=output_channels,
                num_layers=num_res_blocks,
                add_downsample=not is_first_block,
            )

            self.down_blocks.append(down_block)

        self.down_blocks = nn.ModuleList(self.down_blocks)

        

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        r"""
        This function processes the input tensor `x` through the adapter model and returns a list of feature tensors,
        each representing information extracted at a different scale from the input. The length of the list is
        determined by the number of downsample blocks in the Adapter, as specified by the `channels` and
        `num_res_blocks` parameters during initialization.
        """
        
        x = self.conv_in(x)

        features = ()

        for down_block in self.down_blocks:
            x = down_block(x)
            features += (x,)

        return features



class EnvNormalEncoder(ModelMixin, ConfigMixin):

    @register_to_config
    def __init__(
        self,
        in_channels: int = 3,
        channels: Tuple[int] = (320, 640, 1280, 1280),
        num_res_blocks: int = 2,
        light_block: bool = True,
        latent_encoder: bool = False,
    ):
        super().__init__()

        if light_block:
            block_out_channels=[16, 32, 80, 80]
            if latent_encoder:
                self.conv_in = nn.Conv2d(in_channels, block_out_channels[-1], kernel_size=3, padding=1)
            else:
                self.conv_in = ConditioningEmbedding(
                    conditioning_embedding_channels=block_out_channels[-1],
                    block_out_channels=block_out_channels,
                    conditioning_channels=in_channels,
                )
            self.down_blocks = nn.ModuleList(
                [
                    LightAdapterBlock(block_out_channels[-1], channels[0], num_res_blocks, down=False),
                    *[
                        LightAdapterBlock(channels[i], channels[i + 1], num_res_blocks, down=True)
                        for i in range(len(channels) - 2)
                    ],
                    LightAdapterBlock(channels[-2], channels[-1], num_res_blocks, down=True),
                ]
            )
        else:
            block_out_channels=[16, 64, 96, 256]
            if latent_encoder:
                self.conv_in = nn.Conv2d(in_channels, channels[0], kernel_size=3, padding=1)
            else:
                self.conv_in = ConditioningEmbedding(
                    conditioning_embedding_channels=channels[0],
                    block_out_channels=block_out_channels,
                    conditioning_channels=in_channels,
                )
            self.down_blocks = []
            output_channels = channels[0]
            for i, channel in enumerate(channels):
                input_channels = output_channels
                output_channels = channel
                is_first_block = i == 0

                down_block = get_down_block_2d(
                    in_channels=input_channels,
                    out_channels=output_channels,
                    num_layers=num_res_blocks,
                    add_downsample=not is_first_block,
                )

                self.down_blocks.append(down_block)

            self.down_blocks = nn.ModuleList(self.down_blocks)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        
        x = self.conv_in(x)
        x = F.silu(x)
        features = ()
    
        for down_block in self.down_blocks:
            x = down_block(x)
            features += (x,)

        return features
    

class EnvDecoder(ModelMixin, ConfigMixin):

    @register_to_config
    def __init__(
        self,
        out_channels: int = 3,
        channels: Tuple[int] = (1280, 640, 320, 320, 256, 128, 64),
        num_res_blocks: int = 2,
        num_attention_heads: Tuple[int] = (20, 20, 10, 5),
        act_fn: str = 'exp',
        feat_input: bool = False,
        latent_encoder: bool = False,
        only_cross_attention: bool = True,
    ):
        super().__init__()

        self.up_blocks = []
        output_channels = channels[0]
        for i, channel in enumerate(channels):
            input_channels = output_channels
            output_channels = channel
            is_last_block = (i == len(channels) - 1)

            up_block = get_up_block_2d(
                in_channels=input_channels,
                out_channels=output_channels,
                num_layers=num_res_blocks,
                add_upsample=not is_last_block,
            )

            self.up_blocks.append(up_block)

        self.up_blocks = nn.ModuleList(self.up_blocks)

        # TODO
        self.conv_out = nn.Sequential(
            nn.Conv2d(channels[-1], channels[-1], kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(channels[-1], out_channels, kernel_size=3, padding=1)
        )

        self.acf_fn = act_fn
        self.feat_input = feat_input

        self.attentions = []
        # output_channels = channels[0]
        attn_in_channels = channels[0]
        for i, attention_head in enumerate(num_attention_heads):
            attention = Transformer2DModel(
                num_attention_heads=attention_head,
                attention_head_dim=channels[i] // attention_head,
                in_channels=attn_in_channels,
                num_layers=1,
                cross_attention_dim=attn_in_channels,
                use_linear_projection=True,
                only_cross_attention=only_cross_attention,
                upcast_attention=True,
                attention_type='default',
            )
            self.attentions.append(attention)
            attn_in_channels = channels[i]

        self.attentions = nn.ModuleList(self.attentions)
        
        
    def forward(self, feat: List[torch.Tensor], query: List[torch.Tensor]) -> List[torch.Tensor]:
        '''
        feat: List of features from the encoder, BCHW
        query: List of queries, BCHW
        '''
        # features = ()
        num_atten = len(self.attentions)
        x = feat[0] if self.feat_input else None
        for i, up_block in enumerate(self.up_blocks):
            if i < num_atten:
                emb = self.attentions[i](
                    query[i], 
                    encoder_hidden_states=rearrange(feat[i], 'b c h w -> b (h w) c'),
                    return_dict=False,
                )[0]
                if x is None:
                    x = emb
                else:
                    x = x + emb
            x = up_block(x)
            # features += (x,)

        # return features
        x = self.conv_out(x)
        # x = self.out_activation(x)
        if self.acf_fn == 'silu':
            x = F.silu(x)
        elif self.acf_fn == 'relu':
            x = F.relu(x)
        elif self.acf_fn == 'sigmoid':
            x = torch.sigmoid(x)
        elif self.acf_fn == 'exp':
            x = torch.exp(x.clamp(-10, 10) - 1)
        elif self.acf_fn == 'none':
            x = x
        return x