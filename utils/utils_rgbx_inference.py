# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import io
import os
import pathlib
import re
from collections import defaultdict

import torch
from PIL import Image

from torch import distributed as dist


def touch(path):
    """Emulates the 'touch' command by creating the file at *path* if it does not exist.
    If the file exists, its modification time will be updated."""
    with io.open(path, 'ab'):
        os.utime(path, None)


def find_images_recursive(directory, image_extensions=['.png', '.jpg', '.jpeg', '.gif', '.tiff', '.bmp']):
    # Define common image file extensions
    images = []
    root_path = pathlib.Path(directory)

    # Walk through the directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                full_path = pathlib.Path(root) / file
                images.append(str(full_path.relative_to(root_path)))

    return images


def base_plus_ext(path, mode="folder"):
    """Split off all file extensions.

    Returns base, allext.

    Args:
        path: path with extensions

    Returns:
        path with all extensions removed
    """
    if mode == "folder":
        return os.path.dirname(path), os.path.basename(path)
    if mode == "webdataset":
        match = re.match(r"^((?:.*/|)[^.]+)[.]([^/]*)$", path)
        if not match:
            return None, None
        return match.group(1), match.group(2)
    raise NotImplementedError


def group_images_into_videos(validation_image_paths, image_group_mode="folder", subsample_every_n_frames=1):
    validation_videos = defaultdict(list)
    for image_path in validation_image_paths:
        key, allext = base_plus_ext(image_path, mode=image_group_mode)
        if key is not None:
            validation_videos[key].append(image_path)

    validation_video_list = []
    for key in validation_videos:
        validation_video_list.append(sorted(validation_videos[key])[::subsample_every_n_frames])

    return validation_video_list


def split_list_with_overlap(lst, chunk_size, overlap_size, chunk_mode="all"):
    """Splits a list into chunks with overlapping elements."""
    if overlap_size >= chunk_size:
        raise ValueError("Overlap size must be less than the chunk size.")

    chunks = []
    step = chunk_size - overlap_size

    for i in range(0, len(lst) - overlap_size, step):
        chunk = lst[i:i + chunk_size]
        chunks.append(chunk)
        if chunk_mode == "first":
            break

    if len(chunks) > 0 and chunk_mode == "drop_last" and len(chunks[-1]) < chunk_size:
        chunks = chunks[:-1]

    return chunks


def resize_upscale_without_padding(image, target_height, target_width, mode='bilinear', divisible_by=int(64)):
    """
    Resizes and upscales an image or tensor without padding, ensuring dimensions are divisible by 16.

    Parameters:
        image (PIL.Image.Image or torch.Tensor): Input image to be resized. For torch.Tensor, shape should be (C, H, W).
        target_height (int): Desired height of the output image.
        target_width (int): Desired width of the output image.
        mode (str): Resampling mode. Options for PIL: 'nearest', 'bilinear', 'bicubic', 'lanczos'.
                    For torch.Tensor, options are 'nearest', 'bilinear', 'bicubic', 'trilinear', etc.

    Returns:
        PIL.Image.Image or torch.Tensor: Resized image with dimensions divisible by 16, in the same format as the input.
    """

    if isinstance(image, Image.Image):
        # PIL Image case
        original_width, original_height = image.size

    elif isinstance(image, torch.Tensor):
        if image.dim() != 3:
            raise ValueError("Tensor image should have 3 dimensions (C, H, W)")

        original_height, original_width = image.shape[1:3]
    else:
        raise TypeError("Input image must be a PIL.Image.Image or torch.Tensor")

    # Calculate scale and new dimensions
    scale = max(target_width / original_width, target_height / original_height)
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)

    # Ensure dimensions are divisible by 8 (SD) or 64 (SVD)
    new_width = max(divisible_by, (new_width + (divisible_by - 1)) // divisible_by * divisible_by)
    new_height = max(divisible_by, (new_height + (divisible_by - 1)) // divisible_by * divisible_by)

    # Resize the image
    if isinstance(image, Image.Image):
        resized_image = image.resize((new_width, new_height), resample=getattr(Image, mode.upper(), Image.BILINEAR))
        return resized_image

    elif isinstance(image, torch.Tensor):
        # Resize the image
        resized_image = torch.nn.functional.interpolate(image.unsqueeze(0), size=(new_height, new_width),
                                                        mode=mode, align_corners=False).squeeze(0)
        return resized_image
    else:
        raise TypeError("Input image must be a PIL.Image.Image or torch.Tensor")
