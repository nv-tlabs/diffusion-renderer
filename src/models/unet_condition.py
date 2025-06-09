# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import torch

import diffusers
from diffusers import UNet2DConditionModel


def expand_and_copy_weights(pretrained_conv, new_in_channels=8, rescale=0.5):
    """
    Expand the input channels of a pretrained convolutional layer and copy the weights.

    Parameters:
    pretrained_conv (nn.Conv2d): The pretrained convolutional layer with 4 input channels.
    new_in_channels (int): The number of input channels for the new convolutional layer.
    rescale (float): The factor to rescale the weights.

    Returns:
    nn.Conv2d: The new convolutional layer with expanded input channels and copied weights.
    """
    if pretrained_conv.in_channels != 4:
        print("The pretrained convolutional layer must have 4 input channels. Skipped.")
        return pretrained_conv

    # Define the new convolutional layer with the specified number of input channels
    new_conv = torch.nn.Conv2d(new_in_channels, pretrained_conv.out_channels,
                               kernel_size=pretrained_conv.kernel_size,
                               stride=pretrained_conv.stride,
                               padding=pretrained_conv.padding,
                               dilation=pretrained_conv.dilation,
                               groups=pretrained_conv.groups,
                               bias=(pretrained_conv.bias is not None),
                               padding_mode=pretrained_conv.padding_mode)

    # Copy the pretrained weights to the new layer
    with torch.no_grad():
        new_conv.weight[:, :4, :, :] = pretrained_conv.weight.clone()
        if new_in_channels > 4:
            num_repeats = (new_in_channels + 3) // 4  # Calculate the number of times to repeat
            for i in range(1, num_repeats):
                end_idx = min(new_in_channels, (i + 1) * 4)
                new_conv.weight[:, i * 4:end_idx, :, :].copy_(pretrained_conv.weight[:, :end_idx - i * 4, :, :].clone())

        if pretrained_conv.bias is not None:
            new_conv.bias = torch.nn.Parameter(pretrained_conv.bias.clone())

        new_conv.weight = torch.nn.Parameter(rescale * new_conv.weight)

    return new_conv


def copy_pretrained_weights(unet_pretrained, unet, exclude_key='conv_in.weight', rescale_weight=0.5):
    # Copy all layers except the first layer
    pretrained_dict = unet_pretrained.state_dict()
    new_dict = unet.state_dict()
    # Remove the first layer's weights from the pretrained dict
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k != exclude_key}

    # Update the new model's state dict with the pretrained weights (excluding the first layer)
    new_dict.update(pretrained_dict)

    # Handle the first layer separately
    with torch.no_grad():
        # Example: Expand and copy weights for the first convolutional layer
        pretrained_conv_in_weights = unet_pretrained.state_dict()[exclude_key]
        new_conv_in_weights = unet.state_dict()[exclude_key]

        # Copy the weights of the first layer as per your logic
        new_conv_in_weights[:, :4, :, :] = pretrained_conv_in_weights.clone() * rescale_weight
        if new_conv_in_weights.shape[1] > 4:
            num_repeats = (new_conv_in_weights.shape[1] + 3) // 4
            for i in range(1, num_repeats):
                end_idx = min(new_conv_in_weights.shape[1], (i + 1) * 4)
                new_conv_in_weights[:, i * 4:end_idx, :, :].copy_(pretrained_conv_in_weights[:, :end_idx - i * 4, :, :].clone())
                new_conv_in_weights[:, i * 4:end_idx, :, :] = new_conv_in_weights[:, i * 4:end_idx, :, :] * rescale_weight

        new_dict[exclude_key] = new_conv_in_weights

    # Load the updated state dict into the new model
    unet.load_state_dict(new_dict)


def load_unet_from_pretrained_with_condition(pretrained_model_name_or_path, subfolder="unet", revision=None, **kwargs):
    unet_pretrained = UNet2DConditionModel.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision, **kwargs
    )
    from collections import OrderedDict
    unet_model_config = OrderedDict(unet_pretrained.config)
    unet_model_config["in_channels"] = 8
    unet = UNet2DConditionModel.from_config(unet_model_config)
    # Copy the weights
    copy_pretrained_weights(unet_pretrained, unet)
    return unet
