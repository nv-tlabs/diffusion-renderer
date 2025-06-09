# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Not a contribution
# Changes made by NVIDIA CORPORATION & AFFILIATES enabling Inverse Rendering, Forward Rendering, Relighting, or
# otherwise documented as NVIDIA-proprietary are not a contribution and subject to the following terms and conditions:
# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import os
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import copy
import re
import math

import torch
import torch.nn as nn

import diffusers
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.unets.unet_spatio_temporal_condition import UNetSpatioTemporalConditionModel, UNetSpatioTemporalConditionOutput
from diffusers.models.controlnet import ControlNetConditioningEmbedding
from diffusers.utils import (
    USE_PEFT_BACKEND,
    deprecate,
    is_accelerate_available,
    is_torch_version,
    scale_lora_layers,
    unscale_lora_layers,
    logging,
)
from diffusers.loaders import PeftAdapterMixin
from diffusers.models.modeling_utils import ModelMixin, load_state_dict, _load_state_dict_into_model
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.modeling_utils import ModelMixin

from .custom_unet_st_blocks import UNetMidBlockSpatioTemporal, get_down_block, get_up_block

logger = logging.get_logger(__name__)

def _init_extended_conv_layer(new_layer, old_layer, init_method, channel_scales=None):
    old_in_channels = old_layer.shape[1]
    new_in_channels = new_layer.shape[1]
    
    new_layer[:, :old_in_channels] = old_layer

    source_in_channels = old_in_channels
    source_layer = old_layer
    source_start, source_end = 0, old_in_channels

    if init_method.startswith("select"):
        # select the channels for the duplicated layers, must be in the format of select_4_8_duplicate
        channel_search = re.search(r"select_(\d+)_(\d+)", init_method).groups()
        source_start, source_end = int(channel_search[0]), int(channel_search[1])
        source_in_channels = source_end - source_start
        source_layer = old_layer[:, source_start:source_end]
        # remove select_1_2 from the init_method
        init_method = init_method.replace(f"select_{source_start}_{source_end}_", "")


    if old_in_channels < new_in_channels:
        if "zero" in init_method:
            new_layer[:, old_in_channels:] = 0
        elif init_method.startswith("duplicate"):
            # e.g., duplicate_4_8, duplicate, duplicate_4_zero, duplicate_4_zero_rescale
            duplicate_channels = [int(x) for x in init_method.split("_") if x.isdigit()]
            if len(duplicate_channels) == 0:
                duplicate_channels = list(range(old_in_channels, new_in_channels, source_in_channels))
            number_copies = 1
            start, end = old_in_channels, new_in_channels
            for start in duplicate_channels:
                end = start + source_in_channels
                if end <= new_in_channels:
                    new_layer[:, start:end] = source_layer
                    number_copies += 1
            if end > new_in_channels and 'zero' in init_method:
                new_layer[:, start:] = 0
            if 'rescale' in init_method:
                if channel_scales is None:
                    new_layer[:, old_in_channels:] = new_layer[:, old_in_channels:]/ number_copies                    
                    new_layer[:, source_start:source_end] = new_layer[:, source_start:source_end]/ number_copies
                else:
                    for i in range(number_copies):
                        new_layer[:, source_in_channels * i:source_in_channels * (i + 1)] *= channel_scales[i]

    return new_layer

class UNetCustomSpatioTemporalConditionModel(
    UNetSpatioTemporalConditionModel, ModelMixin, PeftAdapterMixin
):
    
    @register_to_config
    def __init__(
        self,
        sample_size: Optional[int] = None,
        in_channels: int = 8,
        out_channels: int = 4,
        down_block_types: Tuple[str] = (
            "CrossAttnDownBlockSpatioTemporal",
            "CrossAttnDownBlockSpatioTemporal",
            "CrossAttnDownBlockSpatioTemporal",
            "DownBlockSpatioTemporal",
        ),
        up_block_types: Tuple[str] = (
            "UpBlockSpatioTemporal",
            "CrossAttnUpBlockSpatioTemporal",
            "CrossAttnUpBlockSpatioTemporal",
            "CrossAttnUpBlockSpatioTemporal",
        ),
        block_out_channels: Tuple[int] = (320, 640, 1280, 1280),
        addition_time_embed_dim: int = 256,
        projection_class_embeddings_input_dim: int = 768,
        layers_per_block: Union[int, Tuple[int]] = 2,
        cross_attention_dim: Union[int, Tuple[int]] = 1024,
        temporal_cross_attention_dim: Optional[Union[int, Tuple[int]]] = 1024,
        transformer_layers_per_block: Union[int, Tuple[int], Tuple[Tuple]] = 1,
        num_attention_heads: Union[int, Tuple[int]] = (5, 10, 20, 20),
        num_frames: int = 25,
        ####### ------------------ Custom ------------------ #######
        multi_res_encoder_hidden_states: bool = False,
        context_embedding_type: Optional[str] = None,   # choices: [None, "clip", "time"]
        context_vocab_size: Optional[int] = 16,
    ):
        '''
        multi_res_encoder_hidden_states: bool
            If True, the encoder_hidden_states can be a list of tensors, each tensor will be used for the corresponding down_block.
        temporal_cross_attention_dim: Optional[Union[int, Tuple[int]]]
            The dimension of the temporal cross-attention layer. If None, no temporal cross-attention is used.
        '''
        super(ModelMixin, self).__init__() # NOTE: we bypass the UNetSpatioTemporalConditionModel's __init__ method

        self.multi_res_encoder_hidden_states = multi_res_encoder_hidden_states

        # copy from UNetSpatioTemporalConditionModel.__init__
        # version: v0.30.2
        self.sample_size = sample_size

        # Check inputs
        if len(down_block_types) != len(up_block_types):
            raise ValueError(
                f"Must provide the same number of `down_block_types` as `up_block_types`. `down_block_types`: {down_block_types}. `up_block_types`: {up_block_types}."
            )

        if len(block_out_channels) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `block_out_channels` as `down_block_types`. `block_out_channels`: {block_out_channels}. `down_block_types`: {down_block_types}."
            )

        if not isinstance(num_attention_heads, int) and len(num_attention_heads) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `num_attention_heads` as `down_block_types`. `num_attention_heads`: {num_attention_heads}. `down_block_types`: {down_block_types}."
            )

        if isinstance(cross_attention_dim, list) and len(cross_attention_dim) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `cross_attention_dim` as `down_block_types`. `cross_attention_dim`: {cross_attention_dim}. `down_block_types`: {down_block_types}."
            )

        if not isinstance(layers_per_block, int) and len(layers_per_block) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `layers_per_block` as `down_block_types`. `layers_per_block`: {layers_per_block}. `down_block_types`: {down_block_types}."
            )

        # input
        self.conv_in = nn.Conv2d(
            in_channels,
            block_out_channels[0],
            kernel_size=3,
            padding=1,
        )

        # time
        time_embed_dim = block_out_channels[0] * 4

        self.time_proj = Timesteps(block_out_channels[0], True, downscale_freq_shift=0)
        timestep_input_dim = block_out_channels[0]

        self.time_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)

        self.add_time_proj = Timesteps(addition_time_embed_dim, True, downscale_freq_shift=0)
        self.add_embedding = TimestepEmbedding(projection_class_embeddings_input_dim, time_embed_dim)

        self.down_blocks = nn.ModuleList([])
        self.up_blocks = nn.ModuleList([])

        if isinstance(num_attention_heads, int):
            num_attention_heads = (num_attention_heads,) * len(down_block_types)

        if isinstance(cross_attention_dim, int):
            cross_attention_dim = (cross_attention_dim,) * len(down_block_types)

        if isinstance(temporal_cross_attention_dim, int) or temporal_cross_attention_dim is None:
            temporal_cross_attention_dim = (temporal_cross_attention_dim,) * len(down_block_types)


        if isinstance(layers_per_block, int):
            layers_per_block = [layers_per_block] * len(down_block_types)

        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = [transformer_layers_per_block] * len(down_block_types)

        blocks_time_embed_dim = time_embed_dim

        # down
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block[i],
                transformer_layers_per_block=transformer_layers_per_block[i],
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=blocks_time_embed_dim,
                add_downsample=not is_final_block,
                resnet_eps=1e-5,
                cross_attention_dim=cross_attention_dim[i],
                temporal_cross_attention_dim=temporal_cross_attention_dim[i],
                num_attention_heads=num_attention_heads[i],
                resnet_act_fn="silu"
            )
            self.down_blocks.append(down_block)

        # mid
        self.mid_block = UNetMidBlockSpatioTemporal(
            block_out_channels[-1],
            temb_channels=blocks_time_embed_dim,
            transformer_layers_per_block=transformer_layers_per_block[-1],
            cross_attention_dim=cross_attention_dim[-1],
            temporal_cross_attention_dim=temporal_cross_attention_dim[-1],
            num_attention_heads=num_attention_heads[-1]
        )

        # count how many layers upsample the images
        self.num_upsamplers = 0

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        reversed_num_attention_heads = list(reversed(num_attention_heads))
        reversed_layers_per_block = list(reversed(layers_per_block))
        reversed_cross_attention_dim = list(reversed(cross_attention_dim))
        reversed_temporal_cross_attention_dim = list(reversed(temporal_cross_attention_dim))
        reversed_transformer_layers_per_block = list(reversed(transformer_layers_per_block))

        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            is_final_block = i == len(block_out_channels) - 1

            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]

            # add upsample block for all BUT final layer
            if not is_final_block:
                add_upsample = True
                self.num_upsamplers += 1
            else:
                add_upsample = False

            up_block = get_up_block(
                up_block_type,
                num_layers=reversed_layers_per_block[i] + 1,
                transformer_layers_per_block=reversed_transformer_layers_per_block[i],
                in_channels=input_channel,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                temb_channels=blocks_time_embed_dim,
                add_upsample=add_upsample,
                resnet_eps=1e-5,
                resolution_idx=i,
                cross_attention_dim=reversed_cross_attention_dim[i],
                temporal_cross_attention_dim=reversed_temporal_cross_attention_dim[i],
                num_attention_heads=reversed_num_attention_heads[i],
                resnet_act_fn="silu"
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=32, eps=1e-5)
        self.conv_act = nn.SiLU()

        self.conv_out = nn.Conv2d(
            block_out_channels[0],
            out_channels,
            kernel_size=3,
            padding=1,
        )

        # Custom: context_embedding for multipass
        self.context_embedding_type = context_embedding_type
        self.context_embedding = None
        if self.context_embedding_type is not None:
            if self.context_embedding_type.lower() == "clip":
                context_embedding_dim = cross_attention_dim if isinstance(cross_attention_dim, int) else cross_attention_dim[0]
            elif self.context_embedding_type.lower() == "time":
                context_embedding_dim = block_out_channels[0] * 4
            else:
                raise NotImplementedError
            self.context_embedding = torch.nn.Embedding(num_embeddings=context_vocab_size,
                                                        embedding_dim=context_embedding_dim, max_norm=10)
            torch.nn.init.uniform_(self.context_embedding.weight, -1e-3, 1e-3)

    @classmethod
    def from_pretrained_custom(
        cls, 
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], 
        conv_in_init: str = "select_4_8_duplicate_zero_rescale",
        conv_out_init: str = "duplicate_zero",
        reset_cross_attention: bool = False,
        **kwargs
    ):
        output_loading_info = kwargs.pop("output_loading_info", False)
        _kwargs = copy.deepcopy(kwargs)
        model, loading_info = cls.from_pretrained(
            pretrained_model_name_or_path, 
            output_loading_info=True, 
            low_cpu_mem_usage=False,
            ignore_mismatched_sizes=True,
            **kwargs
        )
        mismatched_keys = loading_info['mismatched_keys']
        conv_in_modified = any([key == 'conv_in.weight' for key, _, _ in mismatched_keys])
        conv_out_modified = any([key == 'conv_out.weight' for key, _, _ in mismatched_keys])

        _model = None
        if conv_in_modified or conv_out_modified:
            logger.warning(
                f"conv_in.weight or conv_out.weight modified in the pretrained model."
                f" try to load first z channels of modified layers with original weights."
                f" then initialize the rest of the channels with {conv_in_init}"
            )

            # pop potential `in_channels` and `out_channels` from kwargs
            in_channels = _kwargs.pop("in_channels", None)
            out_channels = _kwargs.pop("out_channels", None)

            logger.info("Loading again the model with original weights")
            _model, _loading_info = cls.from_pretrained(
                pretrained_model_name_or_path, 
                output_loading_info=True, 
                low_cpu_mem_usage=False,
                ignore_mismatched_sizes=True,
                **_kwargs
            )
            
            assert all(['conv_in' not in key for key, _, _ in _loading_info['mismatched_keys']])
            assert all(['conv_out' not in key for key, _, _ in _loading_info['mismatched_keys']])

            # load the original weights

            # conv_in
            model.conv_in.weight.data[:] = _init_extended_conv_layer(
                model.conv_in.weight.data, _model.conv_in.weight.data, conv_in_init
            )
            
            # conv_out
            model.conv_out.weight.data[:] = _init_extended_conv_layer(
                model.conv_out.weight.data, _model.conv_out.weight.data, conv_out_init
            )

        del _model

        if reset_cross_attention:
            """
            Re-initialize the cross-attention layers in the pre-loaded SD checkpoints. 
            NOTE: Don't use this if you are not sure about this.
            """
            logger.info("Resetting cross-attention weights: QKV uses default nn.linear init, out is zero init.")
            for k, v in model.named_parameters():
                if '.attn2.' in k and 'temporal' not in k:
                    if 'to_q.w' in k or 'to_k.w' in k or 'to_v.w' in k:
                        nn.init.kaiming_uniform_(v, a=math.sqrt(5))
                    if 'to_out.w' in k:
                        nn.init.zeros_(v)

        if output_loading_info:
            return model, loading_info

        return model
    
    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: Union[torch.Tensor, List[torch.Tensor]],
        added_time_ids: Optional[torch.Tensor] = None,
        skip_temporal: bool = False,
        time_context: Optional[torch.Tensor] = None, # temporal cross-attention
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        input_context: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[UNetSpatioTemporalConditionOutput, Tuple]:
        r"""
        The [`UNetSpatioTemporalConditionModel`] forward method.

        Args:
            sample (`torch.Tensor`):
                The noisy input tensor with the following shape `(batch, num_frames, channel, height, width)`.
            timestep (`torch.Tensor` or `float` or `int`): The number of timesteps to denoise an input.
            encoder_hidden_states (`torch.Tensor` or `List[torch.Tensor]`):
                The encoder hidden states with shape `(batch, sequence_length, cross_attention_dim)`.
                Note that encoder_hidden_states can be a list of tensors if `multi_res_encoder_hidden_states` is True.
                If it is a list, the length of the list should be the same as the number of down_blocks,
                each Diffusion block will use the corresponding encoder_hidden_states.
            added_time_ids: (`torch.Tensor`):
                The additional time ids with shape `(batch, num_additional_ids)`. These are encoded with sinusoidal
                embeddings and added to the time embeddings.
            skip_temporal (`bool`, *optional*, defaults to `False`):
                skip the temporal layers in the SVD by scale the feature to 0
            time_context (`torch.Tensor`, *optional*):
                You can explicitly specify a temporal context embedding tensor that will be used in the temporal cross-attention.
                if it is not provided, the temporal context will be same as the `encoder_hidden_states`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unet_slatio_temporal.UNetSpatioTemporalConditionOutput`] instead
                of a plain tuple.
        Returns:
            [`~models.unet_slatio_temporal.UNetSpatioTemporalConditionOutput`] or `tuple`:
                If `return_dict` is True, an [`~models.unet_slatio_temporal.UNetSpatioTemporalConditionOutput`] is
                returned, otherwise a `tuple` is returned where the first element is the sample tensor.
        """
        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # we're popping the `scale` instead of getting it because otherwise `scale` will be propagated
        # to the internal blocks and will raise deprecation warnings. this will be confusing for our users.
        if cross_attention_kwargs is not None:
            cross_attention_kwargs = cross_attention_kwargs.copy()
            lora_scale = cross_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        batch_size, num_frames = sample.shape[:2]
        timesteps = timesteps.expand(batch_size)

        t_emb = self.time_proj(timesteps)

        # `Timesteps` does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=sample.dtype)

        emb = self.time_embedding(t_emb)

        if added_time_ids is not None:
            time_embeds = self.add_time_proj(added_time_ids.flatten())
            time_embeds = time_embeds.reshape((batch_size, -1))
            time_embeds = time_embeds.to(emb.dtype)
            aug_emb = self.add_embedding(time_embeds)
            emb = emb + aug_emb

        # Flatten the batch and frames dimensions
        # sample: [batch, frames, channels, height, width] -> [batch * frames, channels, height, width]
        sample = sample.flatten(0, 1)
        # Repeat the embeddings num_video_frames times
        # emb: [batch, channels] -> [batch * frames, channels]
        emb = emb.repeat_interleave(num_frames, dim=0)

        _encoder_hidden_states_list = encoder_hidden_states if self.multi_res_encoder_hidden_states else [encoder_hidden_states]
        encoder_hidden_states_list = []
        for _encoder_hidden_states in _encoder_hidden_states_list:
            if _encoder_hidden_states.dim() == 3:
                # _encoder_hidden_states: [batch, 1, channels] -> [batch * frames, 1, channels]
                _encoder_hidden_states = _encoder_hidden_states.repeat_interleave(num_frames, dim=0)
            elif _encoder_hidden_states.dim() == 4:
                # _encoder_hidden_states: [batch, frames, 1, channels] -> [batch * frames, 1, channels]
                if _encoder_hidden_states.shape[1] == num_frames:
                    _encoder_hidden_states = _encoder_hidden_states.flatten(0, 1)
                elif _encoder_hidden_states.shape[1] == 1:
                    _encoder_hidden_states = _encoder_hidden_states.flatten(0, 1).repeat_interleave(num_frames, dim=0)
            encoder_hidden_states_list.append(_encoder_hidden_states)

        if not self.multi_res_encoder_hidden_states:
            encoder_hidden_states = encoder_hidden_states_list[0]

        # Custom context
        if input_context is not None and self.context_embedding is not None:
            input_context_emb = self.context_embedding(input_context)   # (B, C) or (B, 1, C)
            if input_context_emb.ndim == 2:
                input_context_emb = input_context_emb.unsqueeze(1)
            if self.context_embedding_type.lower() == "clip":
                encoder_hidden_states = encoder_hidden_states + input_context_emb.repeat_interleave(num_frames, dim=0)
            if self.context_embedding_type.lower() == "time":
                emb = emb + input_context_emb.repeat_interleave(num_frames, dim=0)

        # 2. pre-process
        sample = self.conv_in(sample)

        image_only_indicator = torch.zeros(batch_size, num_frames, dtype=sample.dtype, device=sample.device)
        if skip_temporal:
            image_only_indicator += 1

        down_block_res_samples = (sample,)
        for i, downsample_block in enumerate(self.down_blocks):
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:

                if self.multi_res_encoder_hidden_states:
                    encoder_hidden_states = encoder_hidden_states_list[i]

                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    image_only_indicator=image_only_indicator,
                    time_context=time_context,
                )
            else:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    image_only_indicator=image_only_indicator,
                )

            down_block_res_samples += res_samples

        # 4. mid
        if hasattr(self.mid_block, "has_cross_attention") and self.mid_block.has_cross_attention:
            if self.multi_res_encoder_hidden_states:
                encoder_hidden_states = encoder_hidden_states_list[-1]

        sample = self.mid_block(
            hidden_states=sample,
            temb=emb,
            encoder_hidden_states=encoder_hidden_states,
            image_only_indicator=image_only_indicator,
            time_context=time_context,
        )

        # 5. up
        for i, upsample_block in enumerate(self.up_blocks):
            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                if self.multi_res_encoder_hidden_states:
                    encoder_hidden_states = encoder_hidden_states_list[-i - 1]

                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    image_only_indicator=image_only_indicator,
                    time_context=time_context,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    image_only_indicator=image_only_indicator,
                )

        # 6. post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        # 7. Reshape back to original shape
        sample = sample.reshape(batch_size, num_frames, *sample.shape[1:])

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (sample,)

        return UNetSpatioTemporalConditionOutput(sample=sample)
