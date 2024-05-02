# ref:
# - https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py#L566
# - https://huggingface.co/spaces/baulab/Erasing-Concepts-In-Diffusion/blob/main/train.py

from typing import List, Optional
import argparse
import ast
from pathlib import Path
import gc

import torch
from tqdm import tqdm


from lora import LoRANetwork, DEFAULT_TARGET_REPLACE, UNET_TARGET_REPLACE_MODULE_CONV
import train_util
import model_util
import prompt_util
from prompt_util import (
    PromptEmbedsCache,
    PromptEmbedsPair,
    PromptSettings,
    PromptEmbedsXL,
)
import debug_util
import config_util
from config_util import RootConfig

import wandb

NUM_IMAGES_PER_PROMPT = 1


def flush():
    torch.cuda.empty_cache()
    gc.collect()


def train(
    config: RootConfig,
    prompts: list[PromptSettings],
    device,
):
    metadata = {
        "prompts": ",".join([prompt.json() for prompt in prompts]),
        "config": config.json(),
    }
    save_path = Path(config.save.path)

    modules = DEFAULT_TARGET_REPLACE
    if config.network.type == "c3lier":
        modules += UNET_TARGET_REPLACE_MODULE_CONV

    weight_dtype = config_util.parse_precision(config.train.precision)
    save_weight_dtype = config_util.parse_precision(config.train.precision)

    (
        tokenizers,
        text_encoders,
        unet,
        noise_scheduler,
    ) = model_util.load_models_xl(
        config.pretrained_model.name_or_path,
        scheduler_name=config.train.noise_scheduler,
    )

    for text_encoder in text_encoders:
        text_encoder.to(device, dtype=weight_dtype)
        text_encoder.requires_grad_(False)
        text_encoder.eval()

    unet.to(device, dtype=weight_dtype)
    if config.other.use_xformers:
        unet.enable_xformers_memory_efficient_attention()
    unet.requires_grad_(False)
    unet.eval()

    network = LoRANetwork(
        root_module= text_encoders,
        rank=config.network.rank,
        multiplier=1.0,
        alpha=config.network.alpha,
        train_method=config.network.training_method,
        lora_type=config.lora_type,
    ).to(device, dtype=weight_dtype)

    optimizer_module = train_util.get_optimizer(config.train.optimizer)
    #optimizer_args
    optimizer_kwargs = {}
    if config.train.optimizer_args is not None and len(config.train.optimizer_args) > 0:
        for arg in config.train.optimizer_args.split(" "):
            key, value = arg.split("=")
            value = ast.literal_eval(value)
            optimizer_kwargs[key] = value
            
    optimizer = optimizer_module(network.prepare_optimizer_params(), lr=config.train.lr, **optimizer_kwargs)
    lr_scheduler = train_util.get_lr_scheduler(
        config.train.lr_scheduler,
        optimizer,
        max_iterations=config.train.iterations,
        lr_min=config.train.lr / 100,
    )
    criteria = torch.nn.MSELoss()

    print("Prompts")
    for settings in prompts:
        print(settings)

    # debug
    debug_util.check_requires_grad(network)
    debug_util.check_training_mode(network)


    pbar = tqdm(range(config.train.iterations))

    loss = None
    text_encoders[0].training=True
    text_encoders[0].requires_grad_=True
    text_encoders[1].training=True
    text_encoders[1].requires_grad_=True

    for i in pbar:
        # inference_1
        with torch.no_grad():
            noise_scheduler.set_timesteps(
                config.train.max_denoising_steps, device=device
            )

            optimizer.zero_grad()

            prompt_pair: PromptSettings = prompts[
                torch.randint(0, len(prompts), (1,)).item()
            ]

            
            # LoRA 在這才有權重，離開後就不是lora
            with network:
                # final_embedded = train_util.clip_xl(text_encoders,tokenizers, prompt_pair) # T*
                pass

            positive_embedding = train_util.clip_xl_conditional(text_encoders,tokenizers, prompt_pair, "positive") # T
            neutral_embedding = train_util.clip_xl_conditional(text_encoders, tokenizers,prompt_pair, "neutral")   # T
            negative_embedding = train_util.clip_xl_conditional(text_encoders, tokenizers,prompt_pair, "negative") # T
            pass
            
        # 
        with network:
            final_embedded = train_util.clip_xl(text_encoders,tokenizers, prompt_pair) # T*
            pass


        loss = train_util.clip_loss(
            final_embedded,
            positive_embedding,
            neutral_embedding,
            negative_embedding,
        )

        # 1000倍しないとずっと0.000...になってしまって見た目的に面白くない
        pbar.set_description(f"Loss*1k: {loss.item()*1000:.4f}")

        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        flush()
        
        if (
            i % config.save.per_steps == 0
            and i != 0
            and i != config.train.iterations - 1
        ):
            print("Saving...")
            save_path.mkdir(parents=True, exist_ok=True)
            network.save_weights(
                save_path / f"{config.save.name}_{i}steps.pt",
                dtype=save_weight_dtype,
            )

    print("Saving...")
    save_path.mkdir(parents=True, exist_ok=True)
    network.save_weights(
        save_path / f"{config.save.name}_last.pt",
        dtype=save_weight_dtype,
    )

    del (
        unet,
        noise_scheduler,
        loss,
        optimizer,
        network,
    )

    flush()

    print("Done.")


def main(args):
    config_file = args.config_file

    config = config_util.load_config_from_yaml(config_file)
    if args.name is not None:
        config.save.name = args.name
    attributes = []
    if args.attributes is not None:
        attributes = args.attributes.split(',')
        attributes = [a.strip() for a in attributes]
    config.lora_type = args.lora_type
    config.network.alpha = args.alpha
    config.network.rank = args.rank
    config.save.name += f'_alpha{args.alpha}'
    config.save.name += f'_rank{config.network.rank }'
    config.save.name += f'_{config.network.training_method}'
    config.save.name += f'_{config.lora_type}'
    config.save.path += f'/{config.save.name}'

    prompts = prompt_util.load_prompts_from_yaml(config.prompts_file, attributes)
    print(prompts)
    device = torch.device(f"cuda:{args.device}")
    train(config, prompts, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        required=False,
        default = './trainscripts/textsliders/data/config-xl.yaml',
        help="Config file for training.",
    )
    # --config_file 'data/config.yaml'
    parser.add_argument(
        "--alpha",
        type=float,
        required=False,
        help="LoRA weight.",
    )
    # --alpha 1.0
    parser.add_argument(
        "--rank",
        type=int,
        required=False,
        help="Rank of LoRA.",
        default=4,
    )
    # --rank 4
    parser.add_argument(
        "--device",
        type=int,
        required=False,
        default=0,
        help="Device to train on.",
    )
    # --device 0
    parser.add_argument(
        "--name",
        type=str,
        required=False,
        default='smile_slider',
        help="Name of trained concept",
    )
    # --name 'eyesize_slider'
    parser.add_argument(
        "--attributes",
        type=str,
        required=False,
        default='male, female',
        help="attritbutes to disentangle (comma seperated string)",
    )
    # --attributes 'male, female'
    parser.add_argument(
        "--lora_type",
        type=str,
        default="clip",
        help="Select which architechture lora should be implemented, argument should be in ['unet', 'clip', 'mix']"
    )
    
    args = parser.parse_args()

    main(args)
