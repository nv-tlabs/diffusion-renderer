# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import os
import numpy as np
from PIL import Image
import imageio
import torch


GBUFFER_PROMPT_MAPPING = {
    'basecolor':        "g-buffer surface base color",
    'metallic':         "g-buffer surface metalness",
    'roughness':        "g-buffer surface roughness",
    'normal':           "g-buffer normal",
    'depth':            "g-buffer depth",
    'diffuse_albedo':   "g-buffer surface diffuse albedo",
}

GBUFFER_INDEX_MAPPING = {
    'basecolor':        0,
    'metallic':         1,
    'roughness':        2,
    'normal':           3,
    'depth':            4,
    'diffuse_albedo':   5,
}


# SVD Utils from: https://github.com/crowsonkb/k-diffusion.git
def rand_log_normal(shape, loc=0., scale=1., device='cpu', dtype=torch.float32):
    """Draws samples from an lognormal distribution."""
    u = torch.rand(shape, dtype=dtype, device=device) * (1 - 2e-7) + 1e-7
    return torch.distributions.Normal(loc, scale).icdf(u).exp()
    # u = torch.randn(shape, dtype=dtype, device=device)
    # return (u * scale + loc).exp()


def convert_rgba_to_rgb_pil(image, background_color=(255, 255, 255)):
    """
    Converts an RGBA image to RGB with the specified background color.
    If the image is already in RGB mode, it is returned as is.

    Parameters:
        image (PIL.Image.Image): Input image (RGBA or RGB).
        background_color (tuple): Background color as an RGB tuple. Default is white (255, 255, 255).

    Returns:
        PIL.Image.Image: RGB image.
    """
    if image.mode == 'RGBA':
        background = Image.new("RGB", image.size, background_color)
        background.paste(image, mask=image.split()[3])  # 3 is the alpha channel
        return background

    return image

