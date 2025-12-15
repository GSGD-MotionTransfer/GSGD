# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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

"""Script to fine-tune Stable Video Diffusion."""
import argparse
import datetime
import logging
import math
import os
import random
import shutil
import sys
from pathlib import Path
from pprint import pprint
from urllib.parse import urlparse

import yaml

sys.path.append(os.path.join(os.path.dirname(__file__), "reenact-anything"))

import accelerate
import cv2
import diffusers
import numpy as np
import PIL
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import (
    AutoencoderKLTemporalDecoder,
    EulerDiscreteScheduler,
    StableVideoDiffusionPipeline,
    UNetSpatioTemporalConditionModel,
)
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.lora import LoRALinearLayer
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate, is_wandb_available, load_image
from diffusers.utils.import_utils import is_xformers_available
from einops import rearrange
from huggingface_hub import create_repo, upload_folder
from packaging import version
from PIL import Image, ImageDraw
from torch.utils.data import Dataset, RandomSampler
from tqdm.auto import tqdm
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

sys.path.append(os.path.join(os.getcwd(), "reenact-anything"))
from utils.embds_inversion_utils import (
    ImageEmbeddingWrapper,
    allow_motion_embedding,
    initialize_image_embedding,
    override_pipeline_call,
    pipeline_decode_latents,
)
from utils.video_utils import SimpleImagesDataset

import wandb  # Can avoid it here if we really want to
from saving import (
    get_available_views,
    get_supervision_output_path,
    load_motion_embedding,
    save_checkpoint_video,
    extract_names_from_path,
    infer_source_name,
)

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.24.0.dev0")

logger = get_logger(__name__, log_level="INFO")


# copy from https://github.com/crowsonkb/k-diffusion.git
def rand_log_normal(shape, loc=0.0, scale=1.0, device="cpu", dtype=torch.float32):
    """Draws samples from an lognormal distribution."""
    u = torch.rand(shape, dtype=dtype, device=device) * (1 - 2e-7) + 1e-7
    return torch.distributions.Normal(loc, scale).icdf(u).exp()


def export_to_video(video_frames, output_video_path, fps):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    h, w, _ = video_frames[0].shape
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps=fps, frameSize=(w, h))
    for i in range(len(video_frames)):
        img = cv2.cvtColor(video_frames[i], cv2.COLOR_RGB2BGR)
        video_writer.write(img)


def export_to_gif(frames, output_gif_path, fps):
    """
    Export a list of frames to a GIF.

    Args:
    - frames (list): List of frames (as numpy arrays or PIL Image objects).
    - output_gif_path (str): Path to save the output GIF.
    - duration_ms (int): Duration of each frame in milliseconds.

    """
    # Convert numpy arrays to PIL Images if needed
    pil_frames = [
        Image.fromarray(frame) if isinstance(frame, np.ndarray) else frame
        for frame in frames
    ]

    pil_frames[0].save(
        output_gif_path.replace(".mp4", ".gif"),
        format="GIF",
        append_images=pil_frames[1:],
        save_all=True,
        duration=500,
        loop=0,
    )


def tensor_to_vae_latent(t, vae):
    video_length = t.shape[1]

    t = rearrange(t, "b f c h w -> (b f) c h w")
    latents = vae.encode(t).latent_dist.sample()
    latents = rearrange(latents, "(b f) c h w -> b f c h w", f=video_length)
    latents = latents * vae.config.scaling_factor

    return latents


def parse_args():
    parser = argparse.ArgumentParser(
        description="Script to train Stable Video Diffusion."
    )
    parser.add_argument(
        "--base_folder",
        # required=True,
        type=str,
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--num_tokens_in_motion_features",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=25,
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--height",
        type=int,
        default=576,
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=1,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_images_path",
        type=str,
        default=None,
        help="Images to condition on during validation of the motion embeddings",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint_dir",
        type=str,
        default=None,
    )

    args = parser.parse_args()
    return args


def get_available_memory():
    """Returns the available memory (in GB) on the current GPU."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        total_memory = torch.cuda.get_device_properties(
            device
        ).total_memory  # Total VRAM in bytes
        allocated_memory = torch.cuda.memory_allocated(device)  # Used memory in bytes
        available_memory = (total_memory - allocated_memory) / (
            1024**3
        )  # Convert to GB
        return available_memory
    return float("inf")  # If no GPU, assume infinite RAM


def reenact_inference(first_frames_path, motion_embedding_path, output_dir, available_views, motion_epoch, args):
    available_memory = get_available_memory()
    print(f"Available memory: {available_memory} GB")
    if available_memory < 18:
        print("Not enough memory, setting environment variables to be slower")
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # Forces sync, useful for debugging
        os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"  # Disables caching
        # will be significantly slower, per image 2 minutes instead of 20s, roughly

    # args = parse_args()
    # Hack to add it to args with minimal changes to the script

    logging_dir = os.path.join(output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=output_dir, logging_dir=logging_dir
    )
    # ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        log_with=args.report_to,
        project_config=accelerator_project_config,
        # kwargs_handlers=[ddp_kwargs]
    )

    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError(
                "Make sure to install wandb if you want to use it for logging during training."
            )
        import wandb

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
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)

    val_dataset = SimpleImagesDataset(
        first_frames_path,
        width=args.width,
        height=args.height,
        max_images_n=len(available_views),
        device=accelerator.device,
    )

    is_global_embedding = not os.path.isdir(motion_embedding_path)

    # Load the pre-trained motion embeddings
    if is_global_embedding:
        image_embds_wrap = ImageEmbeddingWrapper(torch.Tensor([0]))
        image_embds_wrap.requires_grad_(False)
        image_embds_wrap.to(device=accelerator.device, dtype=torch.float32)
        image_embds_wrap.load_tensor(motion_embedding_path)
        embeddings = {"global": image_embds_wrap}
    else:
        embeddings = {}
        for val_img_idx in range(len(val_dataset)):
            sample = val_dataset[val_img_idx]
            angle = sample["name"]
            # step=-1 means it will take the last one. if you want specific checkpoint, change it!!
            angle_motion_embedding_path, motion_embedding_tensor = load_motion_embedding(
                motion_embedding_path, angle, step=motion_epoch
            )

            image_embds_wrap = ImageEmbeddingWrapper(torch.Tensor([0]))
            image_embds_wrap.requires_grad_(False)
            image_embds_wrap.to(device=accelerator.device, dtype=torch.float32)
            image_embds_wrap.load_tensor(angle_motion_embedding_path)
            embeddings[angle] = image_embds_wrap

    # global_step = int(dal.motion_embedding_path.split(".pt")[0].split("_")[-1])

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    # if accelerator.is_main_process:
    #     accelerator.init_trackers("reenact_inference", config=vars(args))

    # The models need unwrapping because for compatibility in distributed training mode.
    pipeline = StableVideoDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        revision=args.revision,
        torch_dtype=torch.float32,
    ).to(accelerator.device)

    allow_motion_embedding(pipeline.unet)

    pipeline.set_progress_bar_config(disable=True)
    pipeline.unet.requires_grad_(False)
    pipeline.vae.requires_grad_(False)
    pipeline.image_encoder.requires_grad_(False)

    # run inference
    inference_save_dir = os.path.join(output_dir, "inference_images")

    if not os.path.exists(inference_save_dir):
        os.makedirs(inference_save_dir)

    with torch.autocast(
        str(accelerator.device).replace(":0", ""),
        # enabled=accelerator.mixed_precision == "fp16",
    ):
        with torch.no_grad():
            for val_img_idx in tqdm(range(len(val_dataset))):
                sample = val_dataset[val_img_idx]
                cond_name = sample["name"]
                angle = "global" if is_global_embedding else cond_name
                cond_img = sample["frame"]
                print("[*] Running inference for angle {}".format(cond_name))

                num_frames = args.num_frames
                video_frames = override_pipeline_call(
                    pipeline,
                    cond_img,
                    height=args.height,
                    width=args.width,
                    num_frames=num_frames,
                    decode_chunk_size=2,  # 2 -> 17.4GB, 8 -> 40GB
                    motion_bucket_id=127,
                    fps=7,
                    noise_aug_strength=0.02,
                    motion_features=embeddings[angle](),
                    max_guidance_scale=1,  # Dont do classifeir free guidance for now
                    # generator=generator,
                ).frames[0]

                save_checkpoint_video(
                    video_frames,
                    output_dir,
                    "inference",
                    figure_name=cond_name,
                    angle=None,
                    fps=10,
                )

                # log_decoded_video(
                #     global_step,
                #     num_frames,
                #     val_img_idx,
                #     inference_save_dir,
                #     video_frames,
                #     video_desc="val_" + cond_name,
                #     report_to=args.report_to,
                # )

    accelerator.wait_for_everyone()
    accelerator.end_training()


def log_decoded_video(
    global_step,
    num_frames,
    val_img_idx,
    val_save_dir,
    video_frames,
    video_desc="val_img",
    report_to=None,
):
    out_suff = f"{video_desc}_{val_img_idx}"
    filename = f"step_{global_step}_{out_suff}.gif"
    out_file = os.path.join(
        val_save_dir,
        filename,
    )
    for i in range(num_frames):
        img = video_frames[i]
        video_frames[i] = np.array(img)
    export_to_gif(video_frames, out_file, 8)

    if report_to == "wandb":
        downsampled_out_file = os.path.join(
            val_save_dir,
            "downsampled_" + filename,
        )
        # resize to save space in wandb
        orig_h, orig_w = video_frames[0].shape[-3:-1]
        downscale_by = 0.2
        new_size = (int(downscale_by * orig_w), int(downscale_by * orig_h))
        resized_video = np.array(
            [
                cv2.resize(frame, dsize=new_size, interpolation=cv2.INTER_CUBIC)
                for frame in video_frames
            ]
        )
        export_to_gif(resized_video, downsampled_out_file, 8)
        # wandb.log(
        #     {out_suff: wandb.Video(downsampled_out_file, fps=8)}, step=global_step
        # )
        os.remove(downsampled_out_file)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/MT3D.yaml",
        required=False,
        help="Path to config file",
    )
    parser.add_argument(
        "--output_path",
        required=True,
        type=str,
        help="The current run folder",
    )
    parser.add_argument(
        "--first_frames_path",
        required=True,
        help="Path to first frames directory",
    )
    parser.add_argument(
        "--motion_embedding_path",
        required=True,
        help="Path to pretrained global motion embedding (.pt) or dir of motion embeddings",
    )
    parser.add_argument(
        "--motion_epoch",
        default="3000",
        required=False,
        help="Epoch of the motion embedding to use",
    )
    args = parser.parse_args(argv)

    # Load YAML configuration
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    config["config"] = args.config
    config["output_path"] = args.output_path
    config["first_frames_path"] = args.first_frames_path
    config["motion_embedding_path"] = args.motion_embedding_path
    config["motion_epoch"] = args.motion_epoch
    config["stage"] = 3

    args = argparse.Namespace(**config)

    # Extract source_name and target_name from first_frames_path
    source_name, target_name = extract_names_from_path(args.first_frames_path)
    
    # Get available views from motion embedding (never use source_path)
    available_views = get_available_views(args.motion_embedding_path)
    
    # Get output path using inferred source name from embedding
    synthetic_source_path = os.path.join("Outputs", "MotionEmbeddings", source_name)
    synthetic_target_path = os.path.join("assets", "targets", f"{target_name}.ply")
    output_dir = get_supervision_output_path(
        synthetic_source_path,
        synthetic_target_path,
        args.output_path
    )
    os.makedirs(output_dir, exist_ok=True)
    
    reenact_inference(
        args.first_frames_path,
        args.motion_embedding_path,
        output_dir,
        available_views,
        args.motion_epoch,
        args
    )



if __name__ == "__main__":
    main()


def run_reenact_target(
    config: str,
    output_path: str,
    first_frames_path: str,
    motion_embedding_path: str,
    motion_epoch: str = "3000",
) -> None:
    argv = [
        "--config",
        config,
        "--output_path",
        output_path,
        "--first_frames_path",
        first_frames_path,
        "--motion_embedding_path",
        motion_embedding_path,
        "--motion_epoch",
        str(motion_epoch),
    ]
    main(argv)
