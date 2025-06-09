# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import os
from contextlib import nullcontext
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse
import omegaconf
from omegaconf import OmegaConf
import imageio

import torch

from accelerate import Accelerator
from accelerate import PartialState
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from src.pipelines.pipeline_rgbx import RGBXVideoDiffusionPipeline
from utils.utils_rgbx import convert_rgba_to_rgb_pil
from utils.utils_rgbx_inference import touch, find_images_recursive, base_plus_ext, \
    group_images_into_videos, split_list_with_overlap, resize_upscale_without_padding

# MODEL_WEIGHTS_CACHE_HOME = os.path.expanduser("~/.cache/diffusion_prior/")      #FIXME:
# AVAILABLE_CHECKPOINT_LIST = [  # FIXME:
#     "dvp_multidata_multipass_v1_0",
# ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args, unknown = parser.parse_known_args()
    cfg = OmegaConf.load(args.config)
    cli = OmegaConf.from_dotlist(unknown)
    cfg = OmegaConf.merge(cfg, cli)

    assert cfg.get('save_video', False) or cfg.get('save_image', False)

    try:
        inference_height, inference_width = cfg.inference_res
    except:
        inference_height, inference_width = 512, 512
        print("[inference_height, inference_width] not provided or in a wrong format, using default values 512x512")

    accelerator = Accelerator()

    cond_mode = "image"
    if cfg.get("cond_mode", None) is not None:
        cond_mode = cfg.get("cond_mode", None)

    use_deterministic_mode = False
    if cfg.get("use_deterministic_mode", None) is not None:
        use_deterministic_mode = cfg.get("use_deterministic_mode", None)

    missing_kwargs = {}
    missing_kwargs["cond_mode"] = cond_mode
    missing_kwargs["use_deterministic_mode"] = use_deterministic_mode
    model_weights_subfolders = os.listdir(cfg.inference_model_weights)
    if "image_encoder" not in model_weights_subfolders:
        missing_kwargs["image_encoder"] = CLIPVisionModelWithProjection.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid", subfolder="image_encoder",
        )
        assert cond_mode != "image"
    if "feature_extractor" not in model_weights_subfolders:
        missing_kwargs["feature_extractor"] = CLIPImageProcessor.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid", subfolder="feature_extractor",
        )
        assert cond_mode != "image"

    pipeline = RGBXVideoDiffusionPipeline.from_pretrained(cfg.inference_model_weights, **missing_kwargs)
    distributed_state = PartialState()
    pipeline = pipeline.to(distributed_state.device)
    pipeline.set_progress_bar_config(disable=True)

    default_n_frames = pipeline.unet.config.num_frames if pipeline.unet.config.num_frames else 14
    inference_n_frames = cfg.inference_n_frames if cfg.inference_n_frames else default_n_frames
    image_group_mode = cfg.image_group_mode if cfg.get("image_group_mode", None) else "folder"
    chunk_mode = cfg.chunk_mode if cfg.get("chunk_mode", None) else "first"
    overlap_n_frames = cfg.overlap_n_frames if cfg.get("overlap_n_frames", None) else 0
    subsample_every_n_frames = int(cfg.subsample_every_n_frames) if cfg.get("image_group_mode", None) else 1

    image_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.tiff', '.bmp']
    if cfg.get("image_extensions", None):
        if not isinstance(cfg.image_extensions, (list, omegaconf.listconfig.ListConfig)):
            image_extensions = [str(cfg.image_extensions)]
        else:
            image_extensions = list(cfg.image_extensions)

    validation_image_paths = find_images_recursive(cfg.inference_input_dir, image_extensions=image_extensions)
    validation_video_list = group_images_into_videos(validation_image_paths, image_group_mode=image_group_mode,
                                                     subsample_every_n_frames=subsample_every_n_frames)

    inference_save_dir = cfg.inference_save_dir
    os.makedirs(inference_save_dir, exist_ok=True)

    success_signal_dir = os.path.join(inference_save_dir, "TMP_SUCCESS_SIGNAL")
    if os.path.exists(success_signal_dir):  # check and filter finished videos
        filtered_validation_video_list = []
        for input_image_relative_path_list in validation_video_list:
            video_relative_base_name = base_plus_ext(input_image_relative_path_list[0], mode=image_group_mode)[0]
            input_image_relative_path_chunks = split_list_with_overlap(
                input_image_relative_path_list, inference_n_frames, overlap_n_frames, chunk_mode=chunk_mode
            )
            max_chunk_ind = len(input_image_relative_path_chunks) - 1
            success_signal_str = video_relative_base_name.replace("/", "--") + f".{max_chunk_ind:04d}"
            success_signal_path = os.path.join(success_signal_dir, success_signal_str)
            if os.path.exists(success_signal_path):
                print(f"Skipping video: {success_signal_str}")
            else:
                filtered_validation_video_list.append(input_image_relative_path_list)

        validation_video_list = filtered_validation_video_list
    else:
        os.makedirs(success_signal_dir, exist_ok=True)

    with distributed_state.split_between_processes(validation_video_list) as validation_video_list:
        for i, input_image_relative_path_list in enumerate(
                tqdm(validation_video_list, desc="processing videos (image sequences): ")
        ):
            # check processed signal
            video_relative_base_name = base_plus_ext(input_image_relative_path_list[0], mode=image_group_mode)[0]

            # split into chunks
            input_image_relative_path_chunks = split_list_with_overlap(
                input_image_relative_path_list, inference_n_frames, overlap_n_frames, chunk_mode=chunk_mode
            )
            if len(input_image_relative_path_chunks) == 0:
                continue

            if cfg.get('save_image', False):
                os.makedirs(os.path.join(inference_save_dir, f"{video_relative_base_name}"), exist_ok=True)

            for chunk_ind in range(len(input_image_relative_path_chunks)):
                success_signal_str = video_relative_base_name.replace("/", "--") + f".{chunk_ind:04d}"
                success_signal_path = os.path.join(success_signal_dir, success_signal_str)
                if os.path.exists(success_signal_path):
                    print(f"Skipping chunk: {success_signal_str}")
                    continue

                current_image_relative_path_list = input_image_relative_path_chunks[chunk_ind]
                len_current_image_relative_path_list = len(current_image_relative_path_list)

                # Fill frames to inference_n_frames
                while len(current_image_relative_path_list) < inference_n_frames:
                    current_image_relative_path_list.append(current_image_relative_path_list[-1])

                # Process input image
                input_images_uint8 = []
                for ind in range(inference_n_frames):
                    input_path = os.path.join(cfg.inference_input_dir, current_image_relative_path_list[ind])
                    input_image_pil = Image.open(input_path)
                    input_image_pil = convert_rgba_to_rgb_pil(input_image_pil, background_color=(0, 0, 0))

                    if ind == 0:
                        width, height = input_image_pil.size
                        if width != inference_width or height != inference_height:
                            input_image_pil = resize_upscale_without_padding(input_image_pil, inference_height, inference_width)
                            width, height = input_image_pil.size
                    else:
                        if width != input_image_pil.size[0] or height != input_image_pil.size[1]:
                            input_image_pil = input_image_pil.resize((width, height), resample=Image.BILINEAR)

                    if cfg.get('save_image', False):
                        save_path = os.path.join(inference_save_dir, f"{video_relative_base_name}/{chunk_ind:04d}.{ind:04d}.rgb.png")
                        input_image_pil.save(save_path)

                    input_images_uint8.append(np.asarray(input_image_pil))

                # Formatting input
                # - cond_images:
                #       A dictionary of images to condition on. The keys are the names of the conditions and the values are
                #       the images. NOTE: The images should be in the range [0, 1] and BFHWC format.
                input_images = np.stack(input_images_uint8, axis=0)[None, ...].astype(np.float32) / 255.    # (1, F, H, W, C)
                cond_images = {"rgb": input_images}
                cond_labels = {"rgb": "vae"}
                if cond_mode == "image":
                    cond_images["clip_img"] = input_images[:, 0:1, ...] # NOTE: clip uses first frame only
                    cond_labels["clip_img"] = "clip"

                viz_images_uint8 = input_images_uint8
                for inference_pass in cfg.model_passes:
                    cond_images["input_context"] = inference_pass

                    if cfg.seed is None:
                        generator = None
                    else:
                        generator = torch.Generator(device=accelerator.device).manual_seed(cfg.seed)

                    if torch.backends.mps.is_available():
                        autocast_ctx = nullcontext()
                    else:
                        autocast_ctx = torch.autocast(accelerator.device.type, enabled=cfg.autocast)

                    with autocast_ctx:
                        inference_image_list = pipeline(
                            cond_images, cond_labels,
                            height=height, width=width,
                            num_frames=inference_n_frames,
                            num_inference_steps=cfg.inference_n_steps,
                            min_guidance_scale=cfg.inference_min_guidance_scale,
                            max_guidance_scale=cfg.inference_max_guidance_scale,
                            fps=cfg.get('fps', 7),
                            motion_bucket_id=cfg.get('motion_bucket_id', 127),
                            noise_aug_strength=cfg.get('cond_aug', 0),
                            generator=generator,
                        ).frames[0]  # list of pil images

                    # Save images locally
                    if cfg.get('save_image', False):
                        for ind in range(len(inference_image_list)):
                            save_path = os.path.join(inference_save_dir, f"{video_relative_base_name}/{chunk_ind:04d}.{ind:04d}.{inference_pass}.png")
                            inference_image_list[ind].save(save_path)

                    if cfg.get('save_video', False):
                        for ind in range(len(viz_images_uint8)):
                            viz_images_uint8[ind] = np.concatenate([
                                viz_images_uint8[ind],
                                np.asarray(inference_image_list[ind]),
                            ], axis=1)

                # Visualize results as video
                if cfg.get('save_video', False):
                    save_path = os.path.join(inference_save_dir, f"{video_relative_base_name}.{chunk_ind:04d}.viz.mp4")
                    imageio.mimsave(save_path, viz_images_uint8, fps=cfg.get('save_video_fps', 7), codec='h264')

                # Mark the inference is finished
                touch(success_signal_path)


if __name__ == "__main__":
    main()
