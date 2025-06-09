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

import inspect
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Union, Any

import numpy as np
import PIL.Image
import torch
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from einops import rearrange

from diffusers.image_processor import PipelineImageInput
from diffusers.models import (
    AutoencoderKLTemporalDecoder,
    UNetSpatioTemporalConditionModel,
    UNet2DConditionModel
)
from diffusers.schedulers import EulerDiscreteScheduler
from diffusers.utils import BaseOutput, logging, replace_example_docstring
from diffusers.utils.torch_utils import is_compiled_module, randn_tensor
from diffusers.video_processor import VideoProcessor
from diffusers.loaders import StableDiffusionLoraLoaderMixin
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, StableDiffusionMixin
from diffusers.pipelines.stable_video_diffusion.pipeline_stable_video_diffusion import (
    _append_dims, retrieve_timesteps, _resize_with_antialiasing,
    StableVideoDiffusionPipeline, StableVideoDiffusionPipelineOutput
)
from src.models.custom_unet_st import UNetCustomSpatioTemporalConditionModel

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class RGBXVideoDiffusionPipeline(StableVideoDiffusionPipeline, DiffusionPipeline, StableDiffusionMixin, StableDiffusionLoraLoaderMixin):
    r"""
    Pipeline to generate video from an input image using Stable Video Diffusion.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Args:
        vae ([`AutoencoderKLTemporalDecoder`]):
            Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.
        image_encoder ([`~transformers.CLIPVisionModelWithProjection`]):
            Frozen CLIP image-encoder
            ([laion/CLIP-ViT-H-14-laion2B-s32B-b79K](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K)).
        unet ([`UNetSpatioTemporalConditionModel`]):
            A `UNetSpatioTemporalConditionModel` to denoise the encoded image latents.
        scheduler ([`EulerDiscreteScheduler`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents.
        feature_extractor ([`~transformers.CLIPImageProcessor`]):
            A `CLIPImageProcessor` to extract features from generated images.
    """

    model_cpu_offload_seq = "image_encoder->unet->vae"
    _callback_tensor_inputs = ["latents"]

    def __init__(
        self,
        vae: AutoencoderKLTemporalDecoder,
        image_encoder: CLIPVisionModelWithProjection,
        unet: Union[UNetSpatioTemporalConditionModel, UNet2DConditionModel],
        scheduler: EulerDiscreteScheduler,
        feature_extractor: CLIPImageProcessor,
        env_encoder: Callable = None,
        scale_cond_latents: bool = False,
        cond_mode: str = 'image',
        use_deterministic_mode=False,
    ):
        super(DiffusionPipeline, self).__init__()

        self.register_modules(
            vae=vae,
            image_encoder=image_encoder,
            unet=unet,
            scheduler=scheduler,
            feature_extractor=feature_extractor,
            env_encoder=env_encoder,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.video_processor = VideoProcessor(do_resize=True, vae_scale_factor=self.vae_scale_factor)

        self.text_encoder = None
        self.scale_cond_latents = scale_cond_latents
        self.cond_mode = cond_mode
        self.use_deterministic_mode = use_deterministic_mode
        self.is_img_model = isinstance(unet, UNet2DConditionModel)

    def example2input(self, example, target_label, cond_labels, clip_label=None):
        if target_label in example:
            target_image = example[target_label][None, ...] # BFHWC
        else:
            target_image = None
        # cond_images = [example[cond_label][None, ...] for cond_label in cond_labels]
        cond_images = {}
        for cond_label, op in cond_labels.items():
            if cond_label == 'clip_img':
                # NOTE: only take the first frame so far
                cond_images[cond_label] = example[clip_label][None, :1, ...] # B1HWC
            else:
                conds = cond_label.split('+') if '+' in cond_label else [cond_label]
                cond_images_group = []
                for cond in conds:
                    if '@' in cond:
                        cond, ch = cond.split('@')
                        ch = int(ch)
                        cond_images_group.append(example[cond][None, ..., ch:ch+1])
                    else:
                        cond_images_group.append(example[cond][None, ...])
                ret = np.concatenate(cond_images_group, axis=-1) # BFHWC
                if op in ['vae', 'clip'] and ret.shape[-1] == 1:
                    ret = ret.repeat(3, axis=-1)
                cond_images[cond_label] = ret
        return target_image, cond_images

    def _encode_image(
        self,
        image: np.ndarray,
        device: Union[str, torch.device],
        num_videos_per_prompt: int,
        do_classifier_free_guidance: bool,
    ) -> torch.Tensor:
        # either 0-255 np image, or 0-1 tensor, BFHWC
        # NOTE: this might be changed for per-frame encoding        
        dtype = next(self.image_encoder.parameters()).dtype

        bsz, fsz = image.shape[:2]
        image = rearrange(image, "b f h w c -> (b f) h w c")

        if not isinstance(image, torch.Tensor):
            # if isinstance(image, PIL.Image.Image):
            #     image = self.video_processor.pil_to_numpy(image)
            image = self.video_processor.numpy_to_pt(image)

            # We normalize the image before resizing to match with the original implementation.
            # Then we unnormalize it after resizing.
            image = image * 2.0 - 1.0
            image = _resize_with_antialiasing(image, (224, 224))
            image = (image + 1.0) / 2.0

        # Normalize the image with for CLIP input
        image = self.feature_extractor(
            images=image,
            do_normalize=True,
            do_center_crop=False,
            do_resize=False,
            do_rescale=False,
            return_tensors="pt",
        ).pixel_values

        image = image.to(device=device, dtype=dtype)
        image_embeddings = self.image_encoder(image).image_embeds # [(NF)C]
        image_embeddings = image_embeddings.unsqueeze(1)

        # duplicate image embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = image_embeddings.shape
        image_embeddings = rearrange(image_embeddings, "(b f) s c -> b f s c", b=bsz)
        image_embeddings = image_embeddings[:, None, ...].repeat(1, num_videos_per_prompt, 1, 1, 1)
        image_embeddings = image_embeddings.view(bsz * num_videos_per_prompt, fsz,  seq_len, -1) # [NFSC]


        if do_classifier_free_guidance:
            negative_image_embeddings = torch.zeros_like(image_embeddings)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            image_embeddings = torch.cat([negative_image_embeddings, image_embeddings])

        return image_embeddings

    def _encode_vae_image(
        self,
        image: torch.Tensor,
        device: Union[str, torch.device],
        num_videos_per_prompt: int,
        do_classifier_free_guidance: bool,
    ):
        image = image.to(device=device) # [-1, 1] image, N, C, H, W
        multi_frame = (image.dim() != 4)
        bsz = image.size(0)
        if multi_frame:
            image = rearrange(image, "b f c h w -> (b f) c h w")
        image_latents = self.vae.encode(image).latent_dist.mode()
        if multi_frame:
            image_latents = rearrange(image_latents, "(b f) c h w -> b f c h w", b=bsz)
            # duplicate image_latents for each generation per prompt, using mps friendly method
            image_latents = image_latents.repeat(num_videos_per_prompt, 1, 1, 1, 1)
        else:
            # duplicate image_latents for each generation per prompt, using mps friendly method
            image_latents = image_latents.repeat(num_videos_per_prompt, 1, 1, 1)

        if do_classifier_free_guidance:
            negative_image_latents = torch.zeros_like(image_latents)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            image_latents = torch.cat([negative_image_latents, image_latents])

        return image_latents

    def _encode_env(self, env_input, device, num_videos_per_prompt, do_classifier_free_guidance):
        dtype = next(self.env_encoder.parameters()).dtype
        # env_input = {k: v.to(device=device, dtype=dtype) for k, v in env_input.items()}
        multi_frame = (env_input[0].dim() != 4)
        bsz = env_input[0].size(0)

        _env_input = []
        for v in env_input:
            if multi_frame:
                v = rearrange(v, "b f c h w -> (b f) c h w")

            if self.env_encoder.config.latent_encoder:
                v = self.vae.encode(v).latent_dist.mode() * self.vae.config.scaling_factor

            _env_input.append(v)

        _env_input = torch.cat(_env_input, dim=1) # [BF, C, H, W]
        _env_embeddings = self.env_encoder(_env_input)

        env_embeddings = []
        for v in _env_embeddings:
            v = v.flatten(2).transpose(1, 2) # [B, N, C]
            if multi_frame:
                v = rearrange(v, "(b f) n c -> b f n c", b=bsz)
                v = v.repeat(num_videos_per_prompt, 1, 1, 1)
            else:
                v = v.repeat(num_videos_per_prompt, 1, 1)

            env_embeddings.append(v)

        if do_classifier_free_guidance:
            env_embeddings = [
                torch.cat([torch.zeros_like(v), v]) for v in env_embeddings
            ]

        return env_embeddings

    def _get_add_time_ids(self, *args, **kwargs):
        return super()._get_add_time_ids(*args, **kwargs)

    def decode_latents(self, latents: torch.Tensor, num_frames: int, decode_chunk_size: int = 14):
        # The same as the original, but leave it here for the sake of completeness
        # [batch, frames, channels, height, width] -> [batch*frames, channels, height, width]
        latents = latents.flatten(0, 1)

        latents = 1 / self.vae.config.scaling_factor * latents

        forward_vae_fn = self.vae._orig_mod.forward if is_compiled_module(self.vae) else self.vae.forward
        accepts_num_frames = "num_frames" in set(inspect.signature(forward_vae_fn).parameters.keys())

        # decode decode_chunk_size frames at a time to avoid OOM
        frames = []
        for i in range(0, latents.shape[0], decode_chunk_size):
            num_frames_in = latents[i : i + decode_chunk_size].shape[0]
            decode_kwargs = {}
            if accepts_num_frames:
                # we only pass num_frames_in if it's expected
                decode_kwargs["num_frames"] = num_frames_in

            frame = self.vae.decode(latents[i : i + decode_chunk_size], **decode_kwargs).sample
            frames.append(frame)
        frames = torch.cat(frames, dim=0)

        # [batch*frames, channels, height, width] -> [batch, channels, frames, height, width]
        frames = frames.reshape(-1, num_frames, *frames.shape[1:]).permute(0, 2, 1, 3, 4)

        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        frames = frames.float()
        return frames

    def prepare_latents(
        self,
        batch_size: int,
        num_frames: int,
        num_channels_latents: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: Union[str, torch.device],
        generator: torch.Generator,
        latents: Optional[torch.Tensor] = None,
    ):
        shape = (
            batch_size,
            num_frames,
            num_channels_latents, # NOTE: differ from original
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def prepare_cond_latents(
        self, cond_images, cond_mapping, num_images_per_prompt, dtype, device,
        do_classifier_free_guidance=False, drop_conds=None, cond_latents=None
    ):
        if cond_latents is not None:
            cond_latents = cond_latents.to(device=device, dtype=dtype)
        else:
            cond_latents = []
            for key, encoding in cond_mapping.items():
                if encoding not in ['vae', 'downsample']:
                    continue
                img = cond_images[key].to(device=device, dtype=dtype)
                if encoding == 'vae':
                    cond_latent = self._encode_vae_image(img, device, 1, False)
                    if self.scale_cond_latents:
                        cond_latent = cond_latent * self.vae.config.scaling_factor
                elif encoding == 'downsample':
                    height, width = img.shape[-2:]
                    cond_latent = torch.nn.functional.interpolate(
                        img,
                        size=(height // self.vae_scale_factor, width // self.vae_scale_factor),
                        mode="bilinear",
                        align_corners=False,
                        antialias=True
                    )
                if drop_conds is not None and key in drop_conds:
                    cond_latent = torch.zeros_like(cond_latent)
                # logger.info(f"cond_latent shape: {cond_latent.shape}")
                cond_latents.append(cond_latent)
            if len(cond_latents) > 0:
                cond_latents = torch.cat(cond_latents, dim=-3) # BFCHW

        if num_images_per_prompt > 1:
            cond_latents = cond_latents[:, None, ...].repeat(1, num_images_per_prompt, 1, 1, 1, 1)
            cond_latents = rearrange(cond_latents, "b x f c h w -> (b x) f c h w")

        if do_classifier_free_guidance:
            cond_latents = torch.cat([torch.zeros_like(cond_latents), cond_latents])

        return cond_latents

    @torch.no_grad()
    def __call__(
        self,
        cond_images: Dict,
        cond_mapping: Dict,
        height: int = 576,
        width: int = 1024,
        num_frames: Optional[int] = None,
        num_inference_steps: int = 25,
        sigmas: Optional[List[float]] = None,
        min_guidance_scale: float = 1.0,
        max_guidance_scale: float = 3.0,
        fps: int = 7,
        motion_bucket_id: int = 127,
        noise_aug_strength: float = 0.02,
        decode_chunk_size: Optional[int] = None,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        cond_latents: Optional[torch.Tensor] = None,
        drop_conds: Optional[list] = None,
        output_type: Optional[str] = "pil",
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        dynamic_guidance: bool = False,
        return_dict: bool = True,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            cond_images (`Dict`):
                A dictionary of images to condition on. The keys are the names of the conditions and the values are
                the images. NOTE: The images should be in the range [0, 1] and BFHWC format.
            cond_mapping (`Dict`):
                A dictionary mapping the condition names to the corresponding latent tensors
            height (`int`, *optional*, defaults to 576):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to 1024):
                The width in pixels of the generated image.
            num_frames (`int`, *optional*):
                The number of video frames to generate. Defaults to `self.unet.config.num_frames`.
            num_inference_steps (`int`, *optional*, defaults to 25):
                The number of denoising steps. More denoising steps usually lead to a higher quality video at the
                expense of slower inference.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                will be used.
            min_guidance_scale (`float`, *optional*, defaults to 1.0):
                The minimum guidance scale. Used for the classifier free guidance with first frame.
            max_guidance_scale (`float`, *optional*, defaults to 3.0):
                The maximum guidance scale. Used for the classifier free guidance with last frame.
            fps (`int`, *optional*, defaults to 7):
                Frames per second. The rate at which the generated images shall be exported to a video after
                generation. Note that Stable Diffusion Video's UNet was micro-conditioned on fps-1 during training.
            motion_bucket_id (`int`, *optional*, defaults to 127):
                Used for conditioning the amount of motion for the generation. The higher the number the more motion
                will be in the video.
            noise_aug_strength (`float`, *optional*, defaults to 0.02):
                The amount of noise added to the init image, the higher it is the less the video will look like the
                init image. Increase it for more motion.
            decode_chunk_size (`int`, *optional*):
                The number of frames to decode at a time. Higher chunk size leads to better temporal consistency at the
                expense of more memory usage.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of videos to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for video
                generation. If not provided, a latents tensor is generated by sampling using the supplied random `generator`.
            cond_latents (`torch.Tensor`, *optional*):
                Precomputed conditional latents to use instead of computing from cond_images.
            drop_conds (`list`, *optional*):
                List of condition keys to drop (zero out) during conditioning.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `pil`, `np` or `pt`.
            callback_on_step_end (`Callable`, *optional*):
                A function that is called at the end of each denoising step during inference.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function.
            cross_attention_kwargs (`Dict[str, Any]`, *optional*):
                Additional kwargs for cross attention.
            dynamic_guidance (`bool`, *optional*, defaults to False):
                Whether to use dynamic guidance scaling during denoising.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableVideoDiffusionPipelineOutput`] instead of a
                plain tuple.

        Returns:
            [`~pipelines.stable_diffusion.StableVideoDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableVideoDiffusionPipelineOutput`] is
                returned, otherwise a `tuple` of (`List[List[PIL.Image.Image]]` or `np.ndarray` or `torch.Tensor`) is
                returned.
        """
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        num_frames = num_frames if num_frames is not None else self.unet.config.num_frames
        decode_chunk_size = decode_chunk_size if decode_chunk_size is not None else num_frames

        if self.use_deterministic_mode:
            num_inference_steps = 1
            sigmas = None

        # 1. Check inputs. Raise error if not correct
        assert isinstance(cond_images, dict), "image should be a dictionary"
        image = cond_images[list(cond_images.keys())[0]]
        self.check_inputs([image], height, width) # [*] is a bypass workaround

        # 2. Define call parameters
        if isinstance(image, PIL.Image.Image):
            batch_size = 1
        elif isinstance(image, list):
            batch_size = len(image)
        else:
            batch_size = image.shape[0]
        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        self._guidance_scale = max_guidance_scale


        # NOTE: Stable Video Diffusion was conditioned on fps - 1, which is why it is reduced here.
        # See: https://github.com/Stability-AI/generative-models/blob/ed0997173f98eaf8f4edf7ba5fe8f15c6b877fd3/scripts/sampling/simple_video_sample.py#L188
        fps = fps - 1

        # 4. Encode input image using VAE
        # Prepare context embedding index if any
        input_context = None
        if "input_context" in cond_images:
            input_context = cond_images["input_context"]
            if isinstance(input_context, str):
                from utils.utils_rgbx import GBUFFER_INDEX_MAPPING
                input_context = GBUFFER_INDEX_MAPPING[input_context]
            if isinstance(input_context, int):
                input_context = torch.LongTensor([input_context]).to(device)
            if input_context.shape[0] != batch_size:
                input_context = input_context.repeat_interleave(batch_size, dim=0)
            if self.do_classifier_free_guidance:
                input_context = input_context.repeat_interleave(2, dim=0)

        # Preprocess image
        cond_images_proc = {}
        for key in cond_mapping:
            if key == "clip_img" or key == "input_context":
                continue
            # NOTE: preprocess does not support 5D input
            cond_image = rearrange(cond_images[key], "b f h w c -> (b f) h w c")
            image = self.video_processor.preprocess(cond_image).to(device) # Resize + Normalize (-1,1) for all. (bf)chw
            if noise_aug_strength != 0:
                noise = randn_tensor(image.shape, generator=generator, device=device, dtype=image.dtype)
                image = image + noise_aug_strength * noise
            cond_images_proc[key] = rearrange(image, "(b f) h w c -> b f h w c", b=batch_size)

        needs_upcasting = False # self.vae.dtype == torch.float16 and self.vae.config.force_upcast
        if needs_upcasting:
            self.vae.to(dtype=torch.float32)

        # 4.1 Encode input image
        time_context = None
        if self.cond_mode == 'image':
            # NOTE: image has BFHWC format
            hidden_embeddings = self._encode_image(cond_images['clip_img'], device, num_videos_per_prompt, self.do_classifier_free_guidance) # [NFSC]
        elif self.cond_mode == 'env':
            env_input = [cond_images_proc[key] for key in self.env_encoder.config.in_labels]
            hidden_embeddings = self._encode_env(env_input, device, num_videos_per_prompt, self.do_classifier_free_guidance)
            time_context = 0
        elif self.cond_mode == 'skip':
            unet_model = self.unet.module if isinstance(self.unet, torch.nn.parallel.DistributedDataParallel) else self.unet
            hidden_embeddings = torch.zeros(batch_size * num_videos_per_prompt, 1, 1, unet_model.config.cross_attention_dim).to(device)
            if self.do_classifier_free_guidance:
                hidden_embeddings = hidden_embeddings.repeat(2, 1, 1, 1)
        else:
            raise ValueError(f"Invalid cond_mode: {self.cond_mode}")

        input_dtype = hidden_embeddings[0].dtype
        # input_dtype = torch.float16

        if not isinstance(self.unet, UNetCustomSpatioTemporalConditionModel):
            assert hidden_embeddings.shape[1] == 1, "The image embeddings should have a single frame."
            hidden_embeddings = hidden_embeddings.squeeze(1) # [NSC]

        cond_latents = self.prepare_cond_latents(
            cond_images_proc,
            cond_mapping,
            num_videos_per_prompt,
            input_dtype,
            device,
            self.do_classifier_free_guidance,
            drop_conds,
            cond_latents
        ) # [batch, frames, channels, height, width]

        if drop_conds is not None and ('clip_img' in drop_conds or 'encoder_hidden_states' in drop_conds):
            if isinstance(hidden_embeddings, torch.Tensor):
                hidden_embeddings = torch.zeros_like(hidden_embeddings)
            else:
                hidden_embeddings = [torch.zeros_like(v) for v in hidden_embeddings]

        # cast back to fp16 if needed
        if needs_upcasting:
            self.vae.to(dtype=torch.float16)

        # Repeat the image latents for each frame so we can concatenate them with the noise
        # image_latents [batch, channels, height, width] ->[batch, num_frames, channels, height, width]
        if cond_latents.size(1) == 1:
            cond_latents = cond_latents.repeat(1, num_frames, 1, 1, 1)

        # 5. Get Added Time IDs
        added_time_ids = None
        if not self.is_img_model:
            added_time_ids = self._get_add_time_ids(
                fps,
                motion_bucket_id,
                noise_aug_strength,
                input_dtype,
                batch_size,
                num_videos_per_prompt,
                self.do_classifier_free_guidance,
            )
            added_time_ids = added_time_ids.to(device)

        # 6. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, None, sigmas)

        # 7. Prepare latent variables
        num_channels_latents = self.vae.config.latent_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_frames,
            num_channels_latents,
            height,
            width,
            input_dtype,
            device,
            generator,
            latents,
        )

        if self.is_img_model:
            latents = latents.squeeze(1)
            cond_latents = cond_latents.squeeze(1)
            # hidden_embeddings = hidden_embeddings.squeeze(1)

        # 8. Prepare guidance scale
        guidance_scale = torch.linspace(min_guidance_scale, max_guidance_scale, num_frames).unsqueeze(0)
        guidance_scale = guidance_scale.to(device, latents.dtype)
        guidance_scale = guidance_scale.repeat(batch_size * num_videos_per_prompt, 1)
        guidance_scale = _append_dims(guidance_scale, latents.ndim)

        self._guidance_scale = guidance_scale

        # 9. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                if self.use_deterministic_mode:
                    latent_model_input = torch.zeros_like(latent_model_input)

                # Concatenate image_latents over channels dimension
                latent_model_input = torch.cat([latent_model_input, cond_latents], dim=-3)

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=hidden_embeddings,
                    added_time_ids=added_time_ids,
                    cross_attention_kwargs=cross_attention_kwargs,
                    time_context=time_context,
                    return_dict=False,
                    input_context=input_context,
                )[0]

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)

                    if dynamic_guidance:
                        # dynamic_guidance_scale = (2*(i+1)/num_inference_steps * (self.guidance_scale - 1)) + 1
                        dynamic_guidance_scale = ((1-np.cos(np.pi*(i)/num_inference_steps)) * (self.guidance_scale - 1)) + 1

                        noise_pred = noise_pred_uncond + dynamic_guidance_scale * (noise_pred_cond - noise_pred_uncond)
                    else:
                        noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)

                if self.use_deterministic_mode:
                    latents = -noise_pred
                else:
                    # compute the previous noisy sample x_t -> x_t-1
                    latents = self.scheduler.step(noise_pred, t, latents).prev_sample

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        if self.is_img_model:
            latents = latents.unsqueeze(1)

        if not output_type == "latent":
            # cast back to fp16 if needed
            if needs_upcasting:
                self.vae.to(dtype=torch.float16)
            frames = self.decode_latents(latents, num_frames, decode_chunk_size)
            frames = self.video_processor.postprocess_video(video=frames, output_type=output_type)
        else:
            frames = latents

        self.maybe_free_model_hooks()

        if not return_dict:
            return frames

        return StableVideoDiffusionPipelineOutput(frames=frames)