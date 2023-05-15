# Copyright 2023 The HuggingFace Team. All rights reserved.
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

'''
NOTE This is with prior
When combined with an unCLIP prior, it can also be used for full text to image generation.

NOTE this pipeline introduce controlnet into it
'''

import inspect
from typing import Any, Callable, Dict, List, Optional, Union, Tuple

import PIL
import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass

from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, CLIPTextModelWithProjection
from transformers.models.clip.modeling_clip import CLIPTextModelOutput
from einops import rearrange

from diffusers.utils.import_utils import is_accelerate_available

from diffusers.loaders import TextualInversionLoaderMixin
from diffusers.models import AutoencoderKL, ControlNetModel, PriorTransformer
from diffusers.models.controlnet import ControlNetOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.embeddings import get_timestep_embedding
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import is_accelerate_version, logging, randn_tensor, replace_example_docstring, PIL_INTERPOLATION
from diffusers.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion.stable_unclip_image_normalizer import StableUnCLIPImageNormalizer

from ..models.unet import UNet3DConditionModel
from diffusers.utils import deprecate, logging, BaseOutput
import ipdb
import warnings

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import requests
        >>> import torch
        >>> from PIL import Image
        >>> from io import BytesIO

        >>> from diffusers import StableUnCLIPImg2ImgPipeline

        >>> pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(
        ...     "fusing/stable-unclip-2-1-l-img2img", torch_dtype=torch.float16
        ... )  # TODO update model path
        >>> pipe = pipe.to("cuda")

        >>> url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"

        >>> response = requests.get(url)
        >>> init_image = Image.open(BytesIO(response.content)).convert("RGB")
        >>> init_image = init_image.resize((768, 512))

        >>> prompt = "A fantasy landscape, trending on artstation"

        >>> images = pipe(prompt, init_image).images
        >>> images[0].save("fantasy_landscape.png")
        ```
"""


@dataclass
class TuneAVideoPipelineOutput(BaseOutput):
    videos: Union[torch.Tensor, np.ndarray]




class MultiControlNetModel(ModelMixin):
    r"""
    Multiple `ControlNetModel` wrapper class for Multi-ControlNet

    This module is a wrapper for multiple instances of the `ControlNetModel`. The `forward()` API is designed to be
    compatible with `ControlNetModel`.

    Args:
        controlnets (`List[ControlNetModel]`):
            Provides additional conditioning to the unet during the denoising process. You must set multiple
            `ControlNetModel` as a list.
    """

    def __init__(self, controlnets: Union[List[ControlNetModel], Tuple[ControlNetModel]]):
        super().__init__()
        self.nets = nn.ModuleList(controlnets)

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        controlnet_cond: List[torch.tensor],
        conditioning_scale: List[float],
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> Union[ControlNetOutput, Tuple]:
        for i, (image, scale, controlnet) in enumerate(zip(controlnet_cond, conditioning_scale, self.nets)):
            down_samples, mid_sample = controlnet(
                sample,
                timestep,
                encoder_hidden_states,
                image,
                scale,
                class_labels,
                timestep_cond,
                attention_mask,
                cross_attention_kwargs,
                return_dict,
            )

            # merge samples
            if i == 0:
                down_block_res_samples, mid_block_res_sample = down_samples, mid_sample
            else:
                down_block_res_samples = [
                    samples_prev + samples_curr
                    for samples_prev, samples_curr in zip(down_block_res_samples, down_samples)
                ]
                mid_block_res_sample += mid_sample

        return down_block_res_samples, mid_block_res_sample



class MakeAProtagonistStableUnCLIPPipeline(DiffusionPipeline, TextualInversionLoaderMixin):
    """
    Pipeline for text-guided image to image generation using stable unCLIP.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        feature_extractor ([`CLIPImageProcessor`]):
            Feature extractor for image pre-processing before being encoded.
        image_encoder ([`CLIPVisionModelWithProjection`]):
            CLIP vision model for encoding images.
        image_normalizer ([`StableUnCLIPImageNormalizer`]):
            Used to normalize the predicted image embeddings before the noise is applied and un-normalize the image
            embeddings after the noise has been applied.
        image_noising_scheduler ([`KarrasDiffusionSchedulers`]):
            Noise schedule for adding noise to the predicted image embeddings. The amount of noise to add is determined
            by `noise_level` in `StableUnCLIPPipeline.__call__`.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder.
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`KarrasDiffusionSchedulers`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents.
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
    """

    def __init__(
        self,
        # prior components
        prior_tokenizer: CLIPTokenizer,
        prior_text_encoder: CLIPTextModelWithProjection,
        prior: PriorTransformer,
        prior_scheduler: KarrasDiffusionSchedulers,
        # image encoding components
        feature_extractor: CLIPImageProcessor,
        image_encoder: CLIPVisionModelWithProjection,
        # image noising components
        image_normalizer: StableUnCLIPImageNormalizer,
        image_noising_scheduler: KarrasDiffusionSchedulers,
        # regular denoising components
        tokenizer: CLIPTokenizer,
        text_encoder: CLIPTextModel,
        unet: UNet3DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        controlnet: Union[ControlNetModel, List[ControlNetModel], Tuple[ControlNetModel], MultiControlNetModel],
        # vae
        vae: AutoencoderKL,
    ):
        super().__init__()

        if isinstance(controlnet, (list, tuple)):
            controlnet = MultiControlNetModel(controlnet)

        self.register_modules(
            prior_tokenizer=prior_tokenizer,
            prior_text_encoder=prior_text_encoder,
            prior=prior,
            prior_scheduler=prior_scheduler,
            feature_extractor=feature_extractor,
            image_encoder=image_encoder,
            image_normalizer=image_normalizer,
            image_noising_scheduler=image_noising_scheduler,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            unet=unet,
            controlnet=controlnet,
            scheduler=scheduler,
            vae=vae,
        )

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.enable_vae_slicing
    def enable_vae_slicing(self):
        r"""
        Enable sliced VAE decoding.

        When this option is enabled, the VAE will split the input tensor in slices to compute decoding in several
        steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.vae.enable_slicing()

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.disable_vae_slicing
    def disable_vae_slicing(self):
        r"""
        Disable sliced VAE decoding. If `enable_vae_slicing` was previously invoked, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_slicing()

    def enable_sequential_cpu_offload(self, gpu_id=0):
        r"""
        Offloads all models to CPU using accelerate, significantly reducing memory usage. When called, the pipeline's
        models have their state dicts saved to CPU and then are moved to a `torch.device('meta') and loaded to GPU only
        when their specific submodule has its `forward` method called.
        """
        if is_accelerate_available():
            from accelerate import cpu_offload
        else:
            raise ImportError("Please install accelerate via `pip install accelerate`")

        device = torch.device(f"cuda:{gpu_id}")

        # TODO: self.image_normalizer.{scale,unscale} are not covered by the offload hooks, so they fails if added to the list
        models = [
            self.image_encoder,
            self.prior_text_encoder,
            self.text_encoder,
            self.unet,
            self.vae,
            self.controlnet,
        ]
        for cpu_offloaded_model in models:
            if cpu_offloaded_model is not None:
                cpu_offload(cpu_offloaded_model, device)

    def enable_model_cpu_offload(self, gpu_id=0):
        r"""
        Offloads all models to CPU using accelerate, reducing memory usage with a low impact on performance. Compared
        to `enable_sequential_cpu_offload`, this method moves one whole model at a time to the GPU when its `forward`
        method is called, and the model remains in GPU until the next model runs. Memory savings are lower than with
        `enable_sequential_cpu_offload`, but performance is much better due to the iterative execution of the `unet`.
        """
        if is_accelerate_available() and is_accelerate_version(">=", "0.17.0.dev0"):
            from accelerate import cpu_offload_with_hook
        else:
            raise ImportError("`enable_model_cpu_offload` requires `accelerate v0.17.0` or higher.")

        device = torch.device(f"cuda:{gpu_id}")

        if self.device.type != "cpu":
            self.to("cpu", silence_dtype_warnings=True)
            torch.cuda.empty_cache()  # otherwise we don't see the memory savings (but they probably exist)

        hook = None
        for cpu_offloaded_model in [self.text_encoder, self.prior_text_encoder, self.image_encoder, self.unet, self.vae, self.controlnet]:
            _, hook = cpu_offload_with_hook(cpu_offloaded_model, device, prev_module_hook=hook)

        # We'll offload the last model manually.
        self.final_offload_hook = hook

    @property
    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline._execution_device
    def _execution_device(self):
        r"""
        Returns the device on which the pipeline's models will be executed. After calling
        `pipeline.enable_sequential_cpu_offload()` the execution device can only be inferred from Accelerate's module
        hooks.
        """
        if not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device


    # Copied from diffusers.pipelines.unclip.pipeline_unclip.UnCLIPPipeline._encode_prompt with _encode_prompt->_encode_prior_prompt, tokenizer->prior_tokenizer, text_encoder->prior_text_encoder
    def _encode_prior_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        text_model_output: Optional[Union[CLIPTextModelOutput, Tuple]] = None,
        text_attention_mask: Optional[torch.Tensor] = None,
    ):
        if text_model_output is None:
            batch_size = len(prompt) if isinstance(prompt, list) else 1
            # get prompt text embeddings
            text_inputs = self.prior_tokenizer(
                prompt,
                padding="max_length",
                max_length=self.prior_tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            text_mask = text_inputs.attention_mask.bool().to(device)

            untruncated_ids = self.prior_tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = self.prior_tokenizer.batch_decode(
                    untruncated_ids[:, self.prior_tokenizer.model_max_length - 1 : -1]
                )
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.prior_tokenizer.model_max_length} tokens: {removed_text}"
                )
                text_input_ids = text_input_ids[:, : self.prior_tokenizer.model_max_length]

            prior_text_encoder_output = self.prior_text_encoder(text_input_ids.to(device))

            prompt_embeds = prior_text_encoder_output.text_embeds
            prior_text_encoder_hidden_states = prior_text_encoder_output.last_hidden_state

        else:
            batch_size = text_model_output[0].shape[0]
            prompt_embeds, prior_text_encoder_hidden_states = text_model_output[0], text_model_output[1]
            text_mask = text_attention_mask

        prompt_embeds = prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)
        prior_text_encoder_hidden_states = prior_text_encoder_hidden_states.repeat_interleave(
            num_images_per_prompt, dim=0
        )
        text_mask = text_mask.repeat_interleave(num_images_per_prompt, dim=0)

        if do_classifier_free_guidance:
            uncond_tokens = [""] * batch_size

            uncond_input = self.prior_tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=self.prior_tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            uncond_text_mask = uncond_input.attention_mask.bool().to(device)
            negative_prompt_embeds_prior_text_encoder_output = self.prior_text_encoder(
                uncond_input.input_ids.to(device)
            )

            negative_prompt_embeds = negative_prompt_embeds_prior_text_encoder_output.text_embeds
            uncond_prior_text_encoder_hidden_states = (
                negative_prompt_embeds_prior_text_encoder_output.last_hidden_state
            )

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method

            seq_len = negative_prompt_embeds.shape[1]
            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len)

            seq_len = uncond_prior_text_encoder_hidden_states.shape[1]
            uncond_prior_text_encoder_hidden_states = uncond_prior_text_encoder_hidden_states.repeat(
                1, num_images_per_prompt, 1
            )
            uncond_prior_text_encoder_hidden_states = uncond_prior_text_encoder_hidden_states.view(
                batch_size * num_images_per_prompt, seq_len, -1
            )
            uncond_text_mask = uncond_text_mask.repeat_interleave(num_images_per_prompt, dim=0)

            # done duplicates

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
            prior_text_encoder_hidden_states = torch.cat(
                [uncond_prior_text_encoder_hidden_states, prior_text_encoder_hidden_states]
            )

            text_mask = torch.cat([uncond_text_mask, text_mask])

        return prompt_embeds, prior_text_encoder_hidden_states, text_mask


    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline._encode_prompt
    def _encode_prompt(
        self,
        prompt,
        device,
        num_videos_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
             prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_videos_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
        """
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            # textual inversion: procecss multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                prompt = self.maybe_convert_prompt(prompt, self.tokenizer)

            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                )
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None

            prompt_embeds = self.text_encoder(
                text_input_ids.to(device),
                attention_mask=attention_mask,
            )
            prompt_embeds = prompt_embeds[0]

        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_videos_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            # textual inversion: procecss multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                uncond_tokens = self.maybe_convert_prompt(uncond_tokens, self.tokenizer)

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_videos_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        return prompt_embeds

    def _encode_image(
        self,
        image,
        device,
        batch_size,
        num_videos_per_prompt,
        do_classifier_free_guidance,
        noise_level,
        generator,
        image_embeds,
        return_image_embeds=False
    ):
        dtype = next(self.image_encoder.parameters()).dtype

        if isinstance(image, PIL.Image.Image):
            # the image embedding should repeated so it matches the total batch size of the prompt
            repeat_by = batch_size
        else:
            # assume the image input is already properly batched and just needs to be repeated so
            # it matches the num_videos_per_prompt.
            #
            # NOTE(will) this is probably missing a few number of side cases. I.e. batched/non-batched
            # `image_embeds`. If those happen to be common use cases, let's think harder about
            # what the expected dimensions of inputs should be and how we handle the encoding.
            repeat_by = num_videos_per_prompt

        if image_embeds is None:
            if not isinstance(image, torch.Tensor):
                image = self.feature_extractor(images=image, return_tensors="pt").pixel_values

            image = image.to(device=device, dtype=dtype)
            image_embeds = self.image_encoder(image).image_embeds

        if return_image_embeds:
            return image_embeds

        image_embeds = self.noise_image_embeddings(
            image_embeds=image_embeds,
            noise_level=noise_level,
            generator=generator,
        )

        # duplicate image embeddings for each generation per prompt, using mps friendly method
        image_embeds = image_embeds.unsqueeze(1)
        bs_embed, seq_len, _ = image_embeds.shape
        image_embeds = image_embeds.repeat(1, repeat_by, 1)
        image_embeds = image_embeds.view(bs_embed * repeat_by, seq_len, -1)
        image_embeds = image_embeds.squeeze(1)

        if do_classifier_free_guidance:
            negative_prompt_embeds = torch.zeros_like(image_embeds)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            image_embeds = torch.cat([negative_prompt_embeds, image_embeds])

        return image_embeds

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.decode_latents
    # def decode_latents(self, latents):
    #     latents = 1 / self.vae.config.scaling_factor * latents
    #     image = self.vae.decode(latents).sample
    #     image = (image / 2 + 0.5).clamp(0, 1)
    #     # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
    #     image = image.cpu().permute(0, 2, 3, 1).float().numpy()
    #     return image

    def decode_latents(self, latents):
        latents = latents.to(self.vae.dtype)
        video_length = latents.shape[2]
        latents = 1 / self.vae.config.scaling_factor * latents
        latents = rearrange(latents, "b c f h w -> (b f) c h w")
        video = self.vae.decode(latents).sample
        video = rearrange(video, "(b f) c h w -> b c f h w", f=video_length)
        video = (video / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        video = video.cpu().float().numpy()
        return video


    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs with prepare_extra_step_kwargs->prepare_prior_extra_step_kwargs, scheduler->prior_scheduler
    def prepare_prior_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the prior_scheduler step, since not all prior_schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other prior_schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.prior_scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the prior_scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.prior_scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(
        self,
        prompt,
        image, ## this is for image embedding
        control_image, ## this is for controlnet # the shape should be B,F,C,H,W
        height,
        width,
        callback_steps,
        noise_level,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        image_embeds=None,
        controlnet_conditioning_scale=1.0,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Please make sure to define only one of the two."
            )

        if prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )

        if prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                "Provide either `negative_prompt` or `negative_prompt_embeds`. Cannot leave both `negative_prompt` and `negative_prompt_embeds` undefined."
            )

        if prompt is not None and negative_prompt is not None:
            if type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

        if noise_level < 0 or noise_level >= self.image_noising_scheduler.config.num_train_timesteps:
            raise ValueError(
                f"`noise_level` must be between 0 and {self.image_noising_scheduler.config.num_train_timesteps - 1}, inclusive."
            )

        if image is not None and image_embeds is not None:
            raise ValueError(
                "Provide either `image` or `image_embeds`. Please make sure to define only one of the two."
            )

        if image is None and image_embeds is None:
            raise ValueError(
                "Provide either `image` or `image_embeds`. Cannot leave both `image` and `image_embeds` undefined."
            )

        if image is not None:
            if (
                not isinstance(image, torch.Tensor)
                and not isinstance(image, PIL.Image.Image)
                and not isinstance(image, list)
            ):
                raise ValueError(
                    "`image` has to be of type `torch.FloatTensor` or `PIL.Image.Image` or `List[PIL.Image.Image]` but is"
                    f" {type(image)}"
                )


        # `prompt` needs more sophisticated handling when there are multiple
        # conditionings.
        if isinstance(self.controlnet, MultiControlNetModel):
            if isinstance(prompt, list):
                logger.warning(
                    f"You have {len(self.controlnet.nets)} ControlNets and you have passed {len(prompt)}"
                    " prompts. The conditionings will be fixed across the prompts."
                )

        # Check `image`
        if isinstance(self.controlnet, ControlNetModel):
            self.check_image(control_image, prompt, prompt_embeds)
        elif isinstance(self.controlnet, MultiControlNetModel):
            if not isinstance(control_image, list):
                raise TypeError("For multiple controlnets: `image` must be type `list`")

            # When `image` is a nested list:
            # (e.g. [[canny_image_1, pose_image_1], [canny_image_2, pose_image_2]])
            elif any(isinstance(i, list) for i in control_image):
                raise ValueError("A single batch of multiple conditionings are supported at the moment.")
            elif len(control_image) != len(self.controlnet.nets):
                raise ValueError(
                    "For multiple controlnets: `image` must have the same length as the number of controlnets."
                )

            for image_ in control_image:
                self.check_image(image_, prompt, prompt_embeds)
        else:
            assert False

        # Check `controlnet_conditioning_scale`
        if isinstance(self.controlnet, ControlNetModel):
            if not isinstance(controlnet_conditioning_scale, float):
                raise TypeError("For single controlnet: `controlnet_conditioning_scale` must be type `float`.")
        elif isinstance(self.controlnet, MultiControlNetModel):
            if isinstance(controlnet_conditioning_scale, list):
                if any(isinstance(i, list) for i in controlnet_conditioning_scale):
                    raise ValueError("A single batch of multiple conditionings are supported at the moment.")
            elif isinstance(controlnet_conditioning_scale, list) and len(controlnet_conditioning_scale) != len(
                self.controlnet.nets
            ):
                raise ValueError(
                    "For multiple controlnets: When `controlnet_conditioning_scale` is specified as `list`, it must have"
                    " the same length as the number of controlnets"
                )
        else:
            assert False


    def check_image(self, image, prompt, prompt_embeds):
        image_is_pil = isinstance(image, PIL.Image.Image)
        image_is_tensor = isinstance(image, torch.Tensor)
        image_is_pil_list = isinstance(image, list) and isinstance(image[0], PIL.Image.Image)
        image_is_tensor_list = isinstance(image, list) and isinstance(image[0], torch.Tensor)

        if not image_is_pil and not image_is_tensor and not image_is_pil_list and not image_is_tensor_list:
            raise TypeError(
                "image must be passed and be one of PIL image, torch tensor, list of PIL images, or list of torch tensors"
            )

        if image_is_pil:
            image_batch_size = 1
        elif image_is_tensor:
            image_batch_size = image.shape[0]
        elif image_is_pil_list:
            image_batch_size = len(image)
        elif image_is_tensor_list:
            image_batch_size = len(image)

        if prompt is not None and isinstance(prompt, str):
            prompt_batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            prompt_batch_size = len(prompt)
        elif prompt_embeds is not None:
            prompt_batch_size = prompt_embeds.shape[0]

        if image_batch_size != 1 and image_batch_size != prompt_batch_size:
            raise ValueError(
                f"If image batch size is not 1, image batch size must be same as prompt batch size. image batch size: {image_batch_size}, prompt batch size: {prompt_batch_size}"
            )


    def prepare_image(
        self, image, width, height, batch_size, num_images_per_prompt, device, dtype, do_classifier_free_guidance
    ):
        '''
        image here should be batch wise video, B,F,C,H,W
        '''
        if not isinstance(image, torch.Tensor):
            if isinstance(image, PIL.Image.Image):
                image = [image]

            if isinstance(image[0], PIL.Image.Image):
                images = []

                for image_ in image:
                    image_ = image_.convert("RGB")
                    image_ = image_.resize((width, height), resample=PIL_INTERPOLATION["lanczos"])
                    image_ = np.array(image_)
                    image_ = image_[None, :]
                    images.append(image_)

                image = images

                image = np.concatenate(image, axis=0)
                image = np.array(image).astype(np.float32) / 255.0
                image = image.transpose(0, 3, 1, 2)
                image = torch.from_numpy(image)
            elif isinstance(image[0], torch.Tensor):
                image = torch.cat(image, dim=0)

        image_batch_size = image.shape[0]

        if image_batch_size == 1:
            repeat_by = batch_size
        else:
            # image batch size is the same as prompt batch size
            repeat_by = num_images_per_prompt

        image = image.repeat_interleave(repeat_by, dim=0)

        image = image.to(device=device, dtype=dtype)

        if do_classifier_free_guidance:
            image = torch.cat([image] * 2)

        return image



    # Copied from diffusers.pipelines.unclip.pipeline_unclip.UnCLIPPipeline.prepare_latents
    def prepare_latents_shape(self, shape, dtype, device, generator, latents, scheduler):
        # ipdb.set_trace()
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            latents = latents.to(device)

        latents = latents * scheduler.init_noise_sigma
        return latents

    def prepare_latents(self, batch_size, num_channels_latents, video_length, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels_latents, video_length, height // self.vae_scale_factor, width // self.vae_scale_factor)
        ## B,4,F,H,W
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")

            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_unclip.StableUnCLIPPipeline.noise_image_embeddings
    def noise_image_embeddings(
        self,
        image_embeds: torch.Tensor,
        noise_level: int,
        noise: Optional[torch.FloatTensor] = None,
        generator: Optional[torch.Generator] = None,
    ):
        """
        Add noise to the image embeddings. The amount of noise is controlled by a `noise_level` input. A higher
        `noise_level` increases the variance in the final un-noised images.

        The noise is applied in two ways
        1. A noise schedule is applied directly to the embeddings
        2. A vector of sinusoidal time embeddings are appended to the output.

        In both cases, the amount of noise is controlled by the same `noise_level`.

        The embeddings are normalized before the noise is applied and un-normalized after the noise is applied.
        """
        # ipdb.set_trace()
        
        if noise is None:
            noise = randn_tensor(
                image_embeds.shape, generator=generator, device=image_embeds.device, dtype=image_embeds.dtype
            )

        noise_level = torch.tensor([noise_level] * image_embeds.shape[0], device=image_embeds.device)

        self.image_normalizer.to(image_embeds.device)
        image_embeds = self.image_normalizer.scale(image_embeds)

        image_embeds = self.image_noising_scheduler.add_noise(image_embeds, timesteps=noise_level, noise=noise)

        image_embeds = self.image_normalizer.unscale(image_embeds)

        noise_level = get_timestep_embedding(
            timesteps=noise_level, embedding_dim=image_embeds.shape[-1], flip_sin_to_cos=True, downscale_freq_shift=0
        )

        # `get_timestep_embeddings` does not contain any weights and will always return f32 tensors,
        # but we might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        noise_level = noise_level.to(image_embeds.dtype)

        image_embeds = torch.cat((image_embeds, noise_level), 1)

        return image_embeds

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        image: Union[torch.FloatTensor, PIL.Image.Image] = None,
        prompt: Union[str, List[str]] = None,
        control_image: Union[torch.FloatTensor, PIL.Image.Image, List[torch.FloatTensor], List[PIL.Image.Image]] = None,
        video_length: Optional[int] = 1,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50, # 20 in image unclip pipline
        guidance_scale: float = 7.5, # 10 in image unclip pipline
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_videos_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "tensor",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        noise_level: int = 0,
        image_embeds: Optional[torch.FloatTensor] = None,
        adapter_features: Optional[torch.Tensor] = None,
        controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
        controlnet_image_embeds_type: str = "empty",  ## "image" for using image embedding for control net
        ## text guidance
        text_guidance_scale: float = 0, # NOTE CFG on no image embedding sample
        # prior args
        prior_num_inference_steps: int = 25,
        prior_guidance_scale: float = 4.0,
        prior_latents: Optional[torch.FloatTensor] = None,
        interpolate_embed_weight: float = 0,
        return_prior_embed: bool = False, ## return prior embedding for DDIM inv
        prior_denoised_embeds: Optional[torch.FloatTensor] = None, ## the embedding used for the background, after denoising

        ## mask args
        masks: Optional[torch.FloatTensor] = None,
        inverse_mask: bool = False, ## inverse mask of image embedding and text
        start_step: int = -1, ## start to use mask
        end_step: int = 1000, ## end to use mask
        **kwargs,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, either `prompt_embeds` will be
                used or prompt is initialized to `""`.
            image (`torch.FloatTensor` or `PIL.Image.Image`):
                `Image`, or tensor representing an image batch. The image will be encoded to its CLIP embedding which
                the unet will be conditioned on. Note that the image is _not_ encoded by the vae and then used as the
                latents in the denoising process such as in the standard stable diffusion text guided image variation
                process.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 20):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 10.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.cross_attention](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py).
            noise_level (`int`, *optional*, defaults to `0`):
                The amount of noise to add to the image embeddings. A higher `noise_level` increases the variance in
                the final un-noised images. See `StableUnCLIPPipeline.noise_image_embeddings` for details.
            image_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated CLIP embeddings to condition the unet on. Note that these are not latents to be used in
                the denoising process. If you want to provide pre-generated latents, pass them to `__call__` as
                `latents`.

            prior_num_inference_steps (`int`, *optional*, defaults to 25):
                The number of denoising steps in the prior denoising process. More denoising steps usually lead to a
                higher quality image at the expense of slower inference.
            prior_guidance_scale (`float`, *optional*, defaults to 4.0):
                Guidance scale for the prior denoising process as defined in [Classifier-Free Diffusion
                Guidance](https://arxiv.org/abs/2207.12598). `prior_guidance_scale` is defined as `w` of equation 2. of
                [Imagen Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting
                `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to
                the text `prompt`, usually at the expense of lower image quality.
            prior_latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                embedding generation in the prior denoising process. Can be used to tweak the same generation with
                different prompts. If not provided, a latents tensor will ge generated by sampling using the supplied
                random `generator`.

        Examples:

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`: [`~ pipeline_utils.ImagePipelineOutput`] if `return_dict` is
            True, otherwise a `tuple`. When returning a tuple, the first element is a list with the generated images.
        """
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        if prompt is None and prompt_embeds is None:
            prompt = len(image) * [""] if isinstance(image, list) else ""

        if isinstance(self.controlnet, MultiControlNetModel):
            assert not isinstance(prompt, list)
            ## NOTE only support one prompt here

        else:
            if control_image.dim() == 5:
                prompt_len = len(prompt) if isinstance(prompt, list) else 1
                control_image = control_image.repeat(prompt_len,1,1,1,1) # B,F,3,H,W

        if len(masks.unique()) == 1 and masks.unique()[0] == 1: ## if all ones, just ignore
            masks = None

        if masks is not None:
            if not interpolate_embed_weight: ## is 0
                warnings.warn( "Using mask should use image embedding combined with prior embedding. Now only prior embedding is used, the results should be the same with no mask")
            # assert interpolate_embed_weight, "Using mask should use image embedding combined with prior embedding"

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt=prompt,
            image=image,
            control_image=control_image,
            height=height,
            width=width,
            callback_steps=callback_steps,
            noise_level=noise_level,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            image_embeds=image_embeds,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
        )

        # ipdb.set_trace()
        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        batch_size = batch_size * num_videos_per_prompt

        device = self._execution_device

        ## NOTE using prior denoised latents from the source embedding for partially editing

        if prior_denoised_embeds is None:
            # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
            # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
            # corresponds to doing no classifier free guidance.
            prior_do_classifier_free_guidance = prior_guidance_scale > 1.0

            # 3. Encode input prompt
            prior_prompt_embeds, prior_text_encoder_hidden_states, prior_text_mask = self._encode_prior_prompt(
                prompt=prompt,
                device=device,
                num_images_per_prompt=num_videos_per_prompt,
                do_classifier_free_guidance=prior_do_classifier_free_guidance,
            )
            ## prior_prompt_embeds: text embeds
            ## prior_text_encoder_hidden_states: last hidden state

            # 4. Prepare prior timesteps
            self.prior_scheduler.set_timesteps(prior_num_inference_steps, device=device)
            prior_timesteps_tensor = self.prior_scheduler.timesteps

            # 5. Prepare prior latent variables
            embedding_dim = self.prior.config.embedding_dim

            prior_latents = self.prepare_latents_shape(
                (batch_size, embedding_dim),
                prior_prompt_embeds.dtype,
                device,
                generator,
                prior_latents,
                self.prior_scheduler,
            )
            # ipdb.set_trace()


            # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
            prior_extra_step_kwargs = self.prepare_prior_extra_step_kwargs(generator, eta)


            # 7. Prior denoising loop
            for i, t in enumerate(self.progress_bar(prior_timesteps_tensor)):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([prior_latents] * 2) if prior_do_classifier_free_guidance else prior_latents
                latent_model_input = self.prior_scheduler.scale_model_input(latent_model_input, t)

                predicted_image_embedding = self.prior(
                    latent_model_input,
                    timestep=t,
                    proj_embedding=prior_prompt_embeds,
                    encoder_hidden_states=prior_text_encoder_hidden_states,
                    attention_mask=prior_text_mask,
                ).predicted_image_embedding

                if prior_do_classifier_free_guidance:
                    predicted_image_embedding_uncond, predicted_image_embedding_text = predicted_image_embedding.chunk(2)
                    predicted_image_embedding = predicted_image_embedding_uncond + prior_guidance_scale * (
                        predicted_image_embedding_text - predicted_image_embedding_uncond
                    )

                prior_latents = self.prior_scheduler.step(
                    predicted_image_embedding,
                    timestep=t,
                    sample=prior_latents,
                    **prior_extra_step_kwargs,
                ).prev_sample

                if callback is not None and i % callback_steps == 0:
                    callback(i, t, prior_latents)

            prior_latents = self.prior.post_process_latents(prior_latents)

            if return_prior_embed:
                return prior_latents
            # done prior
        else:
            prior_latents = prior_denoised_embeds

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        if isinstance(self.controlnet, MultiControlNetModel) and isinstance(controlnet_conditioning_scale, float):
            controlnet_conditioning_scale = [controlnet_conditioning_scale] * len(self.controlnet.nets)

        # ipdb.set_trace()
        # 3. Encode input prompt
        prompt_embeds = self._encode_prompt(
            prompt=prompt,
            device=device,
            num_videos_per_prompt=num_videos_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )

        prompt_embeds_text, image_embeds_text = None, None
        if text_guidance_scale:
            prompt_embeds_text = prompt_embeds[prompt_embeds.size(0)//2:] ## with text

        # 4. Encoder input image
        noise_level = torch.tensor([noise_level], device=device)

        if interpolate_embed_weight:
            # assert image is not None, "interpolate image embedding with prior embedding requires the image"
            image_embeds_given = self._encode_image(
                image=image,
                device=device,
                batch_size=batch_size,
                num_videos_per_prompt=num_videos_per_prompt,
                do_classifier_free_guidance=do_classifier_free_guidance,
                noise_level=noise_level,
                generator=generator,
                image_embeds=image_embeds,
                return_image_embeds=True,
            )
            image_embeds = interpolate_embed_weight * image_embeds_given + (1-interpolate_embed_weight) * prior_latents
        else:
            image_embeds = prior_latents
            
        image_embeds = self._encode_image(
            image=image,
            device=device,
            batch_size=batch_size,
            num_videos_per_prompt=num_videos_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            noise_level=noise_level,
            generator=generator,
            image_embeds=image_embeds,
        ) # 2B,C

        aux_latents = None
        if masks is not None:
            ## NOTE encode prior latent
            aux_latents = self._encode_image(
                image=None,
                device=device,
                batch_size=batch_size,
                num_videos_per_prompt=num_videos_per_prompt,
                do_classifier_free_guidance=do_classifier_free_guidance,
                noise_level=noise_level,
                generator=generator,
                image_embeds=prior_latents,
            ) # 2B,C

        # 
        if text_guidance_scale:
            image_embeds_text = image_embeds[:image_embeds.size(0)//2] # no image

        # 5. Prepare control image
        ## control image shape B,F,C,H,W

        # assert control_image.dim() == 5
        if isinstance(self.controlnet, ControlNetModel):
            control_image = self.prepare_image(
                image=control_image,
                width=width,
                height=height,
                batch_size=batch_size * num_videos_per_prompt,
                num_images_per_prompt=num_videos_per_prompt,
                device=device,
                dtype=self.controlnet.dtype,
                do_classifier_free_guidance=do_classifier_free_guidance,
            )
        elif isinstance(self.controlnet, MultiControlNetModel):
            images = []

            for image_ in control_image:
                image_ = self.prepare_image(
                    image=image_,
                    width=width,
                    height=height,
                    batch_size=batch_size * num_videos_per_prompt,
                    num_images_per_prompt=num_videos_per_prompt,
                    device=device,
                    dtype=self.controlnet.dtype,
                    do_classifier_free_guidance=do_classifier_free_guidance,
                )

                images.append(image_)

            control_image = images
        else:
            assert False

        # 6. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 7. Prepare latent variables
        num_channels_latents = self.unet.in_channels
        latents = self.prepare_latents(
            batch_size=batch_size,
            num_channels_latents=num_channels_latents,
            video_length=video_length,
            height=height,
            width=width,
            dtype=prompt_embeds.dtype,
            device=device,
            generator=generator,
            latents=latents,
        )

        # 8. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 9. Denoising loop

        ## control_image: B,F,C,H,W
        if isinstance(control_image, list):
            control_image = [rearrange(c_image, "b f c h w -> (b f) c h w").to(device=self.controlnet.device, dtype=self.controlnet.dtype) for c_image in control_image]
        else:
            control_image = rearrange(control_image, "b f c h w -> (b f) c h w").to(device=self.controlnet.device, dtype=self.controlnet.dtype) ## 
        # aux_latents = torch.zeros_like(image_embeds)
        if masks is not None:
            # ipdb.set_trace()
            masks = rearrange(masks, "b f c h w -> b c f h w").to(device=self.unet.device, dtype=self.unet.dtype)
            masks = torch.nn.functional.interpolate(masks, size=latents.size()[-3:], mode="nearest")
            # ipdb.set_trace()
            if inverse_mask:
                masks = 1 - masks
                image_embeds, aux_latents = aux_latents, image_embeds

        # ipdb.set_trace()
        mask_mode_cfg = kwargs.get("mask_mode", "emb") ## mask_mode: emb / latent / all
        mask_mode = mask_mode_cfg
        mask_latent_fuse_mode = kwargs.get("mask_latent_fuse_mode", "all") ## inverse or all

        for i, t in enumerate(self.progress_bar(timesteps)):
            ## t is 1000 divided in to 50 steps
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            ## 2B,C,F,H,W
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            # ipdb.set_trace()
            # controlnet(s) inference
            down_block_res_samples, mid_block_res_sample = None, None
            
            latent_model_input_control = rearrange(latent_model_input, "b c f h w -> (b f) c h w").to(dtype=self.controlnet.dtype) ##

            if controlnet_image_embeds_type == "image":
                controlnet_image_embeds = image_embeds.repeat(video_length, 1)
            else:
                # controlnet_image_embeds = torch.zeros_like(image_embeds).repeat(video_length, 1)
                # NOTE this is a support for frame wise image embedding
                controlnet_image_embeds = torch.zeros_like(image_embeds[:latent_model_input.size(0)]).repeat(video_length, 1)

            controlnet_image_embeds = controlnet_image_embeds.to(self.controlnet.dtype)
            # ipdb.set_trace()
            down_block_res_samples, mid_block_res_sample = self.controlnet(
                latent_model_input_control,
                t,
                class_labels=controlnet_image_embeds,
                encoder_hidden_states=prompt_embeds.repeat(video_length, 1, 1),
                controlnet_cond=control_image,
                conditioning_scale=controlnet_conditioning_scale,
                return_dict=False,
            )
            down_block_res_samples = [rearrange(sample, "(b f) c h w -> b c f h w", f=video_length) for sample in down_block_res_samples]
            mid_block_res_sample = rearrange(mid_block_res_sample, "(b f) c h w -> b c f h w", f=video_length)

            # ipdb.set_trace()
            if i >= start_step and i < end_step:
                _aux_latents = aux_latents
                _masks = masks
                ## NOTE this can use emb in some steps and use mask latent in other steps
                mask_mode = mask_mode_cfg
            else:
                _aux_latents, _masks = None, None
                mask_mode = "emb"
            # predict the noise residual
            
            ## NOTE mask mode list, the first is for image content, using latent mask, the second is for text, no latent mask / inverse mask
            value = torch.zeros_like(latents)
            count = torch.zeros_like(latents)

            # ipdb.set_trace()
            if mask_mode == "latent":
                cls_labels = [image_embeds, _aux_latents]
                cls_labels_aux = [None, None]
                cls_masks = [None, None]

                if mask_latent_fuse_mode == "all":
                    latent_masks = [masks, torch.ones_like(masks)] # this is used for combining latents
                else:
                    latent_masks = [masks, 1-masks] # this is used for combining latents
            
            elif mask_mode == "all":
                cls_labels = [image_embeds, image_embeds]
                cls_labels_aux = [None, _aux_latents]
                cls_masks = [None, _masks]

                if mask_latent_fuse_mode == "all":
                    latent_masks = [_masks, torch.ones_like(_masks)] # this is used for combining latents
                else:
                    latent_masks = [_masks, 1-_masks] # this is used for combining latents

            else: ## this is the original version
                cls_labels = [image_embeds]
                cls_labels_aux = [_aux_latents]
                cls_masks = [_masks]
                latent_masks = [ torch.ones_like(latents)]
            
            for _cls_labels, _cls_labels_aux, _cls_masks, _latent_masks in zip(cls_labels, cls_labels_aux, cls_masks, latent_masks):

                latent_view = latents
                latent_model_input = torch.cat([latent_view] * 2) if do_classifier_free_guidance else latent_view
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t) #  (batch_size, 4, F, H, W)
                # ipdb.set_trace()
                ## if use all training embedding, _cls_labels: 2F,C -> reshape 2,F,C will get the correct result,
                ## NOTE the batch size 0 is the unconditional
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    class_labels=_cls_labels,
                    class_labels_aux=_cls_labels_aux,
                    masks=_cls_masks,
                    adapter_features=adapter_features,
                    # cross_attention_kwargs=cross_attention_kwargs,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                ).sample
                

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents_view_denoised = self.scheduler.step(noise_pred, t, latent_view, **extra_step_kwargs).prev_sample

                # ipdb.set_trace()
                ## _latent_masks 1,1,F,H,W
                value += latents_view_denoised * _latent_masks
                count += _latent_masks

            assert (count > 0).all()
            latents = torch.where(count > 0, value / count, value)
            if callback is not None and i % callback_steps == 0:
                callback(i, t, latents)

        # 9. Post-processing
        video = self.decode_latents(latents)

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        # Convert to tensor
        if output_type == "tensor":
            video = torch.from_numpy(video)

        if not return_dict:
            return video

        return TuneAVideoPipelineOutput(videos=video)
