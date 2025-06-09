# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import glob
import os
import json
from contextlib import nullcontext
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse
import omegaconf
from omegaconf import OmegaConf
import imageio
import torch.utils.checkpoint
from tqdm.auto import tqdm

import torch

from accelerate import Accelerator
from accelerate import PartialState
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from peft import LoraConfig
from peft.utils import set_peft_model_state_dict

from diffusers.loaders import LoraLoaderMixin
from diffusers import (
    AutoencoderKLTemporalDecoder,
    EulerDiscreteScheduler,
)
from diffusers.utils import (
    convert_unet_state_dict_to_peft
)

from src.models.custom_unet_st import UNetCustomSpatioTemporalConditionModel
from src.models.env_encoder import EnvEncoder
from src.pipelines.pipeline_rgbx import RGBXVideoDiffusionPipeline
from utils.utils_rgbx import convert_rgba_to_rgb_pil
from utils.utils_rgbx_inference import touch, find_images_recursive, base_plus_ext, \
    group_images_into_videos, split_list_with_overlap, resize_upscale_without_padding
from utils.utils_env_proj import process_environment_map
from src.data.rendering_utils import envmap_vec

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

    # model construction
    missing_kwargs = {}
    missing_kwargs["cond_mode"] = cond_mode
    missing_kwargs["use_deterministic_mode"] = use_deterministic_mode
    model_weights_subfolders = os.listdir(cfg.inference_model_weights)

    distributed_state = PartialState()
    text_encoder, image_encoder, vae, env_encoder = None, None, None, None
    tokenizer, feature_extractor = None, None

    if cond_mode == 'env':
        env_encoder = EnvEncoder.from_pretrained(cfg.inference_model_weights, subfolder="env_encoder")
    elif cond_mode == 'image':
        feature_extractor = CLIPImageProcessor.from_pretrained(
            'stabilityai/stable-video-diffusion-img2vid', subfolder="feature_extractor",
        )
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            'stabilityai/stable-video-diffusion-img2vid', subfolder="image_encoder",
        )

    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        cfg.inference_model_weights, subfolder="vae"
    )
    unet = UNetCustomSpatioTemporalConditionModel.from_pretrained(
        cfg.inference_model_weights, subfolder="unet"
    )
    noise_scheduler = EulerDiscreteScheduler.from_pretrained(cfg.inference_model_weights, subfolder="scheduler")

    weight_dtype = torch.float16 if cfg.autocast else torch.float32
    for module in [text_encoder, image_encoder, vae, unet, env_encoder]:
        if module is not None:
            module.to(distributed_state.device, dtype=weight_dtype)

    pipeline = RGBXVideoDiffusionPipeline(
        vae=vae,
        image_encoder=image_encoder,
        feature_extractor=feature_extractor,
        unet=unet,
        scheduler=noise_scheduler,
        env_encoder=env_encoder,
        scale_cond_latents=cfg.model_pipeline.get('scale_cond_latents', False),
        cond_mode=cond_mode
    )
    pipeline.scheduler.register_to_config(timestep_spacing="trailing")
    try:
        pipeline.load_lora_weights(cfg.inference_model_weights, subfolder="lora", adapter_name="real-lora")
    except Exception as e:
        print("Failed to load LoRA weights, using default weights")

    pipeline = pipeline.to(distributed_state.device)
    pipeline.set_progress_bar_config(disable=True)

    # data preparation
    default_n_frames = pipeline.unet.config.num_frames if pipeline.unet.config.num_frames else 14
    inference_n_frames = cfg.inference_n_frames if cfg.inference_n_frames else default_n_frames
    image_group_mode = cfg.image_group_mode if cfg.get("image_group_mode", None) else "webdataset"
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
            success_signal_str = video_relative_base_name.replace("/", "--")
            success_signal_path = os.path.join(success_signal_dir, success_signal_str)
            if os.path.exists(success_signal_path):
                print(f"Skipping video: {success_signal_str}")
            else:
                filtered_validation_video_list.append(input_image_relative_path_list)

        validation_video_list = filtered_validation_video_list
    else:
        os.makedirs(success_signal_dir, exist_ok=True)

    if not isinstance(cfg.envlight, (list, omegaconf.listconfig.ListConfig)):
        envlight_path_list = [str(cfg.envlight)]
    else:
        envlight_path_list = list(cfg.envlight)

    env_resolution = (512, 512)
    if cfg.model_pipeline.get("env_resolution", None):
        env_resolution = cfg.model_pipeline.env_resolution

    cond_labels = cfg.model_pipeline.cond_images
    cond_gbuf_labels = [x for x in list(cond_labels.keys()) if "env_" not in x]

    with distributed_state.split_between_processes(validation_video_list) as validation_video_list:
        for i, input_image_relative_path_list in enumerate(
                tqdm(validation_video_list, desc="processing videos (image sequences): ")
        ):
            # check processed signal
            video_relative_base_name = base_plus_ext(input_image_relative_path_list[0], mode=image_group_mode)[0]
            success_signal_str = video_relative_base_name.replace("/", "--")
            success_signal_path = os.path.join(success_signal_dir, success_signal_str)
            if os.path.exists(success_signal_path):
                print(f"Skipping chunk: {success_signal_str}")
                continue

            if cfg.get('save_image', False) or cfg.get('save_video', False):
                save_dir = os.path.join(inference_save_dir, os.path.dirname(video_relative_base_name))
                os.makedirs(save_dir, exist_ok=True)

            # Formatting input
            # - cond_images:
            #       A dictionary of images to condition on. The keys are the names of the conditions and the values are
            #       the images. NOTE: The images should be in the range [0, 1]
            #       If np.array: BFHWC
            #       If torch.tensor: BFCHW
            # For example:
            #   basecolor: (1, 24, 512, 768, 3)
            #   normal: (1, 24, 512, 768, 3)
            #   depth: (1, 24, 512, 768, 3)
            #   roughness: (1, 24, 512, 768, 3)
            #   metallic: (1, 24, 512, 768, 3)
            #   env_ldr: (1, 24, 512, 512, 3)
            #   env_log: (1, 24, 512, 512, 3)
            #   env_nrm: (1, 24, 512, 512, 3)
            cond_images = {}
            for gbuf_label in cond_gbuf_labels:
                gbuf_path_list = []
                for x in input_image_relative_path_list:
                    if f".{gbuf_label}." in x:
                        gbuf_path_list.append(x)

                gbuf_list = []
                for ind in range(inference_n_frames):
                    input_path = os.path.join(cfg.inference_input_dir, gbuf_path_list[ind])
                    input_image_pil = Image.open(input_path)
                    input_image_pil = convert_rgba_to_rgb_pil(input_image_pil, background_color=(0, 0, 0))

                    if ind == 0:
                        width, height = input_image_pil.size
                        if width != inference_width or height != inference_height:
                            input_image_pil = resize_upscale_without_padding(input_image_pil,
                                                                             inference_height,
                                                                             inference_width)
                            width, height = input_image_pil.size
                    else:
                        if width != input_image_pil.size[0] or height != input_image_pil.size[1]:
                            input_image_pil = input_image_pil.resize((width, height), resample=Image.BILINEAR)

                    gbuf_list.append(np.asarray(input_image_pil))

                input_gbuf_array = np.stack(gbuf_list, axis=0)[None, ...].astype(np.float32) / 255.  # (1, F, H, W, C)
                if cfg.use_fixed_frame_ind:
                    input_gbuf_array = np.concatenate([ input_gbuf_array[:, cfg.fixed_frame_ind:cfg.fixed_frame_ind + 1, ...] ] * inference_n_frames, axis=1)
                cond_images[gbuf_label] = torch.from_numpy(input_gbuf_array).permute(0, 1, 4, 2, 3).to(distributed_state.device)

            # Execute inference for each env map
            viz_images_uint8 = []
            for envlight_ind in range(len(envlight_path_list)):
                envlight_path = envlight_path_list[envlight_ind]
                envlight_dict = process_environment_map(
                    envlight_path,
                    resolution=env_resolution,
                    num_frames=inference_n_frames,
                    fixed_pose=True,
                    rotate_envlight=cfg.rotate_light,
                    elevation=cfg.get('cam_elevation', 0),
                    env_format=['proj', 'fixed', 'ball'],
                    device=distributed_state.device,
                )
                cond_images['env_ldr'] = envlight_dict['env_ldr'].unsqueeze(0).permute(0, 1, 4, 2, 3)
                cond_images['env_log'] = envlight_dict['env_log'].unsqueeze(0).permute(0, 1, 4, 2, 3)
                env_nrm = envmap_vec(env_resolution, device=distributed_state.device) * 0.5 + 0.5
                cond_images['env_nrm'] = env_nrm.unsqueeze(0).unsqueeze(0).permute(0, 1, 4, 2, 3).expand_as(cond_images['env_ldr'])

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
                        cross_attention_kwargs={'scale': cfg.get('lora_scale', 0.5)},
                        dynamic_guidance=False,
                        # num_videos_per_prompt=num_infer_per_image,    # can infer more than one results
                        # drop_conds=drop_conds,        # drop_conds should be None, or a list of attribute labels that will be discarded when running diffusion.
                    ).frames[0]  # list of pil images, assume to run inference only once

                # Save images locally
                if cfg.get('save_image', False):
                    for ind in range(len(inference_image_list)):
                        save_path = os.path.join(inference_save_dir, f"{video_relative_base_name}.{ind:04d}.env{envlight_ind:04d}.png")
                        inference_image_list[ind].save(save_path)

                if cfg.get('save_video', False):
                    if len(viz_images_uint8) == 0:
                        viz_images_uint8 = [np.asarray(inference_image_list[ind]) for ind in range(len(inference_image_list))]
                    else:
                        for ind in range(len(viz_images_uint8)):
                            viz_images_uint8[ind] = np.concatenate([
                                viz_images_uint8[ind],
                                np.asarray(inference_image_list[ind]),
                            ], axis=1)

            # Visualize results as video
            if cfg.get('save_video', False):
                save_path = os.path.join(inference_save_dir, f"{video_relative_base_name}.viz.mp4")
                imageio.mimsave(save_path, viz_images_uint8, fps=cfg.get('save_video_fps', 7), codec='h264')

            # Mark the inference is finished
            touch(success_signal_path)


if __name__ == "__main__":
    main()
