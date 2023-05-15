import argparse
import datetime
import logging
import inspect
import math
import os
from typing import Dict, Optional, Tuple
from omegaconf import OmegaConf

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import numpy as np
from PIL import Image

import diffusers
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, DDIMScheduler, PNDMScheduler, ControlNetModel, PriorTransformer, UnCLIPScheduler
from diffusers.pipelines.stable_diffusion.stable_unclip_image_normalizer import StableUnCLIPImageNormalizer
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from tqdm.auto import tqdm
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, CLIPTextModelWithProjection

from makeaprotagonist.models.unet import UNet3DConditionModel
from makeaprotagonist.dataset.dataset import MakeAProtagonistDataset
from makeaprotagonist.util import save_videos_grid, ddim_inversion_unclip
from makeaprotagonist.pipelines.pipeline_stable_unclip_controlavideo import MakeAProtagonistStableUnCLIPPipeline, MultiControlNetModel

from einops import rearrange
from makeaprotagonist.args_util import DictAction, config_merge_dict
import ipdb
import random
from glob import glob
import sys



# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.15.0.dev0")

logger = get_logger(__name__, log_level="INFO")


def main(
    pretrained_model_path: str,
    controlnet_pretrained_model_path: str,
    output_dir: str,
    train_data: Dict,
    validation_data: Dict,
    validation_steps: int = 100,
    trainable_modules: Tuple[str] = (
        "attn1.to_q",
        "attn2.to_q",
        "attn_temp",
    ),
    trainable_params: Tuple[str] = (),
    train_batch_size: int = 1,
    max_train_steps: int = 500,
    learning_rate: float = 3e-5,
    scale_lr: bool = False,
    lr_scheduler: str = "constant",
    lr_warmup_steps: int = 0,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_weight_decay: float = 1e-2,
    adam_epsilon: float = 1e-08,
    max_grad_norm: float = 1.0,
    gradient_accumulation_steps: int = 1,
    gradient_checkpointing: bool = True,
    checkpointing_steps: int = 500,
    resume_from_checkpoint: Optional[str] = None,
    mixed_precision: Optional[str] = "fp16",
    use_8bit_adam: bool = False,
    enable_xformers_memory_efficient_attention: bool = True,
    seed: Optional[int] = None,
    adapter_config=None, # the config for adapter
    use_temporal_conv=False, ## use temporal conv in resblocks
):
    *_, config = inspect.getargvalues(inspect.currentframe())

    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if seed is not None:
        set_seed(seed)

    # Handle the output folder creation
    if accelerator.is_main_process:
        # now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        # output_dir = os.path.join(output_dir, now)
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/samples", exist_ok=True)
        os.makedirs(f"{output_dir}/inv_latents", exist_ok=True)
        OmegaConf.save(config, os.path.join(output_dir, 'config.yaml'))

    prior_model_id = "kakaobrain/karlo-v1-alpha"
    data_type = torch.float16
    prior = PriorTransformer.from_pretrained(prior_model_id, subfolder="prior", torch_dtype=data_type)

    prior_text_model_id = "openai/clip-vit-large-patch14"
    prior_tokenizer = CLIPTokenizer.from_pretrained(prior_text_model_id)
    prior_text_model = CLIPTextModelWithProjection.from_pretrained(prior_text_model_id, torch_dtype=data_type)
    prior_scheduler = UnCLIPScheduler.from_pretrained(prior_model_id, subfolder="prior_scheduler")
    prior_scheduler = DDPMScheduler.from_config(prior_scheduler.config)

    # image encoding components
    feature_extractor = CLIPImageProcessor.from_pretrained(pretrained_model_path, subfolder="feature_extractor")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(pretrained_model_path, subfolder="image_encoder")
    # image noising components
    image_normalizer = StableUnCLIPImageNormalizer.from_pretrained(pretrained_model_path, subfolder="image_normalizer")
    image_noising_scheduler = DDPMScheduler.from_pretrained(pretrained_model_path, subfolder="image_noising_scheduler")
    # regular denoising components
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
    unet = UNet3DConditionModel.from_pretrained_2d(pretrained_model_path, subfolder="unet", use_temporal_conv=use_temporal_conv)
    noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")

    # vae
    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")

    ## controlnet
    assert not isinstance(controlnet_pretrained_model_path, str)
    controlnet = MultiControlNetModel( [ControlNetModel.from_pretrained(_control_model_path) for _control_model_path in controlnet_pretrained_model_path] )
    
    # Freeze vae and text_encoder and adapter
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    ## freeze image embed
    image_encoder.requires_grad_(False)

    unet.requires_grad_(False)
    ## freeze controlnet
    controlnet.requires_grad_(False)

    ## freeze prior
    prior.requires_grad_(False)
    prior_text_model.requires_grad_(False)

    for name, module in unet.named_modules():
        if name.endswith(tuple(trainable_modules)):
            for params in module.parameters():
                params.requires_grad = True

    if len(trainable_params):
        for name, params in unet.named_parameters():
            if name.endswith(tuple(trainable_params)):
                params.requires_grad = True

    if enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
            controlnet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    if scale_lr:
        learning_rate = (
            learning_rate * gradient_accumulation_steps * train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        unet.parameters(),
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        weight_decay=adam_weight_decay,
        eps=adam_epsilon,
    )

    # Get the training dataset
    train_dataset = MakeAProtagonistDataset(**train_data)

    # Preprocessing the dataset
    train_dataset.prompt_ids = tokenizer(
        train_dataset.prompt, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
    ).input_ids[0]

    train_dataset.preprocess_img_embedding(feature_extractor, image_encoder)
    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_batch_size
    )

    # Get the validation pipeline
    # validation_pipeline = TuneAVideoPipeline(
    #     vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
    #     scheduler=DDIMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
    # )
    prior_val_scheduler = DDIMScheduler.from_config(prior_scheduler.config) if validation_data.get("prior_val_scheduler", "") == "DDIM" else prior_scheduler
    
    validation_pipeline = MakeAProtagonistStableUnCLIPPipeline(
        prior_tokenizer=prior_tokenizer,
        prior_text_encoder=prior_text_model,
        prior=prior,
        prior_scheduler=prior_val_scheduler,
        feature_extractor=feature_extractor,
        image_encoder=image_encoder,
        image_normalizer=image_normalizer,
        image_noising_scheduler=image_noising_scheduler,
        vae=vae, 
        text_encoder=text_encoder, 
        tokenizer=tokenizer, 
        unet=unet,
        controlnet=controlnet,
        scheduler=DDIMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
    )

        
    validation_pipeline.enable_vae_slicing()
    ddim_inv_scheduler = DDIMScheduler.from_pretrained(pretrained_model_path, subfolder='scheduler')
    ddim_inv_scheduler.set_timesteps(validation_data.num_inv_steps)

    # Scheduler
    lr_scheduler = get_scheduler(
        lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
        num_training_steps=max_train_steps * gradient_accumulation_steps,
    )

    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move models to gpu and cast to weight_dtype
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    image_encoder.to(accelerator.device, dtype=weight_dtype)
    ## note controlnet use the unet dtype
    controlnet.to(accelerator.device, dtype=weight_dtype)
    ## prior
    prior.to(accelerator.device, dtype=weight_dtype)
    prior_text_model.to(accelerator.device, dtype=weight_dtype)


    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("text2video-fine-tune")

    # Train!
    total_batch_size = train_batch_size * accelerator.num_processes * gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if resume_from_checkpoint:
        if resume_from_checkpoint != "latest":
            path = os.path.basename(resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1]
        accelerator.print(f"Resuming from checkpoint {path}")
        accelerator.load_state(os.path.join(output_dir, path))
        global_step = int(path.split("-")[1])

        first_epoch = global_step // num_update_steps_per_epoch
        resume_step = global_step % num_update_steps_per_epoch


    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    if not "noise_level" in validation_data:
        validation_data.noise_level = train_data.noise_level

    image_embed_drop = train_data.get("image_embed_drop", 0)

    for epoch in range(first_epoch, num_train_epochs):
        unet.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(unet):
                # Convert videos to latent space
                pixel_values = batch["pixel_values"].to(weight_dtype)
                video_length = pixel_values.shape[1]
                pixel_values = rearrange(pixel_values, "b f c h w -> (b f) c h w")
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)
                latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each video
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["prompt_ids"])[0]

                #
                # ipdb.set_trace()
                ref_imbed = batch["ref_imbed"].to(accelerator.device, dtype=weight_dtype)  # 1,1,768
                ##
                if train_data.noise_level >= 1000:
                    train_noise = random.randint(0, 999)
                else:
                    train_noise = train_data.noise_level

                image_embeds = validation_pipeline.noise_image_embeddings(
                                image_embeds=ref_imbed,
                                noise_level=train_noise,
                                generator=None,
                            )

                if random.random() < image_embed_drop:
                    image_embeds = torch.zeros_like(image_embeds)

                # Get the target for loss depending on the prediction type
                if noise_scheduler.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.prediction_type == "v_prediction": ## use this for unclip model
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.prediction_type}")

                # Predict the noise residual and compute loss
                # model_pred = unet(noisy_latents, timesteps, encoder_hidden_states, adapter_features=adapter_features).sample
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states, class_labels=image_embeds).sample

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(train_batch_size)).mean()
                train_loss += avg_loss.item() / gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                if global_step % validation_steps == 0:
                    if accelerator.is_main_process:

                        # ControlNet
                        conditions = [_condition.to(weight_dtype) for _, _condition in batch["conditions"].items()] # b f c h w
                        masks = batch["masks"].to(weight_dtype) # b,f,1,h,w
                        if not validation_data.get("use_masks", False):
                            masks = torch.ones_like(masks)

                        ddim_inv_latent = None
                        if validation_data.use_inv_latent: #
                            emb_dim = train_dataset.img_embeddings[0].size(0)
                            key_frame_embed = torch.zeros((1, emb_dim)).to(device=latents.device, dtype=latents.dtype) ## this is dim 0
                            ddim_inv_latent = ddim_inversion_unclip(
                                validation_pipeline, ddim_inv_scheduler, video_latent=latents,
                                num_inv_steps=validation_data.num_inv_steps, prompt="", image_embed=key_frame_embed, noise_level=validation_data.noise_level, seed=seed)[-1].to(weight_dtype)

                        set_noise = validation_data.pop("noise_level")
                        v_noise = set_noise

                        if not validation_data.get("interpolate_embed_weight", False):
                            validation_data.interpolate_embed_weight = 1.0


                        samples = []
                            
                        generator = torch.Generator(device=accelerator.device)
                        generator.manual_seed(seed)

                        for idx, prompt in enumerate(validation_data.prompts):

                            _ref_image = Image.open(validation_data.ref_images[idx])
                            image_embed = None
                            ## prior latents
                            prior_embeds = None
                            prior_denoised_embeds = None
                            if validation_data.get("source_background", False):
                                ## using source background and changing the protagonist
                                prior_denoised_embeds = train_dataset.img_embeddings[0][None].to(device=latents.device, dtype=latents.dtype) # 1, 768 for UnCLIP-small
                            
                            if validation_data.get("source_protagonist", False):
                                # using source protagonist and changing the background
                                sample_indices = batch["sample_indices"][0]
                                image_embed = [train_dataset.img_embeddings[idx] for idx in sample_indices]
                                image_embed = torch.stack(image_embed, dim=0).to(device=latents.device, dtype=latents.dtype) # F, 768 for UnCLIP-small # F,C
                                _ref_image = None

                            sample = validation_pipeline(image=_ref_image, prompt=prompt, control_image=conditions, generator=generator, latents=ddim_inv_latent, image_embeds=image_embed, noise_level=v_noise, masks=masks, prior_latents=prior_embeds, prior_denoised_embeds=prior_denoised_embeds, **validation_data).videos

                            save_videos_grid(sample, f"{output_dir}/samples/sample-{global_step}-seed{seed}/{idx}-{prompt}.gif")
                            samples.append(sample)

                        # 
                        samples = [sample.float() for sample in samples]
                        samples = torch.concat(samples)
                        save_path = f"{output_dir}/samples/sample-{global_step}-s{validation_data.start_step}-e{validation_data.end_step}-seed{seed}.gif" # noise level and noise level for inv
                        save_videos_grid(samples, save_path, n_rows=len(samples))
                        logger.info(f"Saved samples to {save_path}")


            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= max_train_steps:
                break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()

    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/tuneavideo.yaml")
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction, 
        help='Override some settings in the '
        'used config, the key-value pair in xxx=yyy format will be merged '
        'into config file. If the value to be overwritten is a list, it '
        'should be like key="[a,b]" or key=a,b It also allows nested '
        'list/tuple values, e.g. key="[(a,b),(c,d)]" Note that the quotation '
        'marks are necessary and that no white space is allowed.')

    args = parser.parse_args()

    ## read from cmd line
    # ipdb.set_trace()
    # Load the YAML configuration file
    config = OmegaConf.load(args.config)
    # Merge the command-line arguments with the configuration file
    if args.options is not None:
        # config = OmegaConf.merge(config, args.options)
        config_merge_dict(args.options, config)
    
    main(**config)
