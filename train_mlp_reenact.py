import argparse
import gc
import math
import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.checkpoint
import yaml
from accelerate.logging import get_logger
from diffusers.utils import check_min_version
from torch.utils.data import RandomSampler
from tqdm.auto import tqdm

import wandb  # Can avoid it here if we really want to
from saving import (
    get_available_views,
    get_motion_embeddings_output_path,
    load_multi_angle_source_videos,
    load_multi_angle_first_frames,
    save_checkpoint_video,
    save_motion_embedding,
    save_angular_embeddings,
)
from multiview_video_utils import (
    GlobalMotionInversionDataset,
    GlobalSimpleImagesDataset,
)

# Utils of reenact_anything. It isn't written as a package, so we need to import it like this
sys.path.append(os.path.join(os.getcwd(), "reenact-anything"))
from train_reenact import (
    gather_function_args,
    gpu_stats,
    init_optimizer_and_scheduler,
    load_models,
    offload_training,
    reload_training,
    train_step,
)

from embds_inversion_utils import (
    ImageEmbeddingWrapper3D,
    initialize_image_embedding,
    override_pipeline_call,
)

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.24.0.dev0")

logger = get_logger(__name__, log_level="INFO")


def inference_val(
    pipeline,
    output_dir,
    cond_img,
    cond_name,
    cond_angle,
    image_embds_wrap,
    global_step,
    num_frames,
    height,
    width,
    generator,
    report_to,
):
    angle = torch.tensor(float(cond_angle)).to(image_embds_wrap.device)
    video_frames = override_pipeline_call(
        pipeline,
        cond_img,
        height=height,
        width=width,
        num_frames=num_frames,
        decode_chunk_size=8,
        motion_bucket_id=127,
        fps=7,
        noise_aug_strength=0.02,
        motion_features=image_embds_wrap(angle),
        max_guidance_scale=1,  # Dont do classifeir free guidance for now
        generator=generator,
    ).frames[0]

    save_checkpoint_video(
        video_frames,
        output_dir,
        global_step,
        figure_name=cond_name,
        angle=cond_angle,
        fps=10,
    )


def extract_motion(
    source_path,
    output_dir,
    available_views,
    num_anchors,
    first_frames_path=None,
    validation_steps=500,
    checkpointing_steps=500,
    max_train_steps=5000,
    validation_images_path=None,
    num_tokens_in_motion_features=5,
    pretrained_model_name_or_path="stabilityai/stable-video-diffusion-img2vid",
    num_frames=14,
    width=1024,
    height=576,
    max_num_validation_images=12,
    seed=0,
    per_gpu_batch_size=1,
    num_train_epochs=100,
    gradient_accumulation_steps=1,
    gradient_checkpointing=True,
    enable_xformers_memory_efficient_attention=True,
    report_to=None,
    allow_tf32=False,
    mixed_precision="fp16",
    num_workers=0,
    use_8bit_adam=False,
    mlp_lr=0.01,
    learning_rate=0.01,
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_weight_decay=0.01,
    adam_epsilon=1e-8,
    lr_warmup_steps=500,
    scale_lr=False,
    lr_scheduler="constant",
):

    os.makedirs(output_dir, exist_ok=True)

    accelerator, image_encoder, feature_extractor, vae, unet, weight_dtype = (
        load_models(
            pretrained_model_name_or_path,
            output_dir,
            gradient_accumulation_steps,
            mixed_precision,
            report_to,
            enable_xformers_memory_efficient_attention,
            seed,
            allow_tf32,
            gradient_checkpointing,
        )
    )

    # ============ DataLoaders ============
    global_batch_size = per_gpu_batch_size * accelerator.num_processes

    videos, available_views = load_multi_angle_source_videos(source_path, available_views)

    # Load first frames for validation if path provided
    if first_frames_path and os.path.exists(first_frames_path):
        first_frames, available_views_val, figure_names = load_multi_angle_first_frames(
            first_frames_path, available_views
        )
        multi_angle_val_dataset = GlobalSimpleImagesDataset(
            (first_frames, available_views_val, figure_names),
            width=width,
            height=height,
            device=accelerator.device,
        )
    else:
        # Create empty dataset if no first frames provided
        multi_angle_val_dataset = GlobalSimpleImagesDataset(
            ([], [], []),
            width=width,
            height=height,
            device=accelerator.device,
        )

    multi_angle_val_dataloader = torch.utils.data.DataLoader(
        multi_angle_val_dataset,
        batch_size=1,
        num_workers=num_workers,
    )

    multi_angle_train_dataset = GlobalMotionInversionDataset(
        videos,  # A list of source videos from different angles
        available_views,
        size=(num_frames, width, height),
        device=accelerator.device,
    )

    sampler = RandomSampler(multi_angle_train_dataset)
    multi_angle_train_dataloader = torch.utils.data.DataLoader(
        multi_angle_train_dataset,
        sampler=sampler,
        batch_size=1,
        num_workers=num_workers,
    )

    # Initialize motion token
    print("[*] Initializing the motion features ")
    multi_angle_image_embeddings = {}
    for angle in tqdm(multi_angle_train_dataloader, desc="Encoding angles"):
        image_embedding = initialize_image_embedding(
            angle["frames"].squeeze(0),
            pretrained_model_name_or_path,
            num_tokens_in_motion_features,
        )

        multi_angle_image_embeddings[angle["angle_name"][0]] = (
            image_embedding.unsqueeze(0)
        )

    global_embds_wrap = ImageEmbeddingWrapper3D(
        multi_angle_image_embeddings,
        mlp_lr,
        learning_rate,
        accelerator.device,
        number_of_anchors=num_anchors,
    )

    del multi_angle_image_embeddings
    gc.collect()
    torch.cuda.empty_cache()

    if scale_lr:
        learning_rate = (
            learning_rate
            * gradient_accumulation_steps
            * per_gpu_batch_size
            * accelerator.num_processes
        )

    # ============ Optimizer and Scheduler ============
    optimizer, lr_scheduler, overrode_max_train_steps = init_optimizer_and_scheduler(
        accelerator,
        multi_angle_train_dataloader,
        use_8bit_adam,
        learning_rate,
        adam_beta1,
        adam_beta2,
        adam_weight_decay,
        adam_epsilon,
        gradient_accumulation_steps,
        max_train_steps,
        num_train_epochs,
        lr_scheduler,
        lr_warmup_steps,
        global_embds_wrap,
    )

    print("Parameters in optimizer:")
    for param_group in optimizer.param_groups:
        for param in param_group["params"]:
            print(param.shape)
    # ============ Training setup ============

    # Prepare everything with our `accelerator`
    global_embds_wrap, unet, optimizer, lr_scheduler, multi_angle_train_dataloader = (
        accelerator.prepare(
            global_embds_wrap,
            unet,
            optimizer,
            lr_scheduler,
            multi_angle_train_dataloader,
        )
    )

    # attribute handling for models using DDP
    if isinstance(
        unet, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)
    ):
        unet = unet.module

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(multi_angle_train_dataloader) / gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        max_train_steps = num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("GlobalEmbedding", config=gather_function_args(main, locals()))

    save_motion_embedding(global_embds_wrap, output_dir, "initial")
    embeddings = global_embds_wrap.get_all_angular_embeddigns()
    save_angular_embeddings(embeddings, output_dir, "initial")

    # Train!
    total_batch_size = (
        per_gpu_batch_size * accelerator.num_processes * gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(multi_angle_train_dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {per_gpu_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    gpu_stats(logger)
    global_step = 0
    first_epoch = 0

    generator = torch.Generator(device=accelerator.device).manual_seed(seed)

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(global_step, max_train_steps),
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")
    gate_scalar = []
    for epoch in range(first_epoch, num_train_epochs):
        unet.train()
        train_loss = 0.0
        for step, batch in enumerate(multi_angle_train_dataloader):
            angle = torch.tensor(float(batch["angle_name"][0])).to(accelerator.device)
            loss, train_loss_addition, denoised_latents = train_step(
                batch,
                weight_dtype,
                global_embds_wrap(angle),
                unet,
                vae,
                feature_extractor,
                image_encoder,
                optimizer,
                accelerator,
                per_gpu_batch_size,
                gradient_accumulation_steps,
                global_step,
                lr_scheduler,
                loss_type="l2",
            )
            train_loss += train_loss_addition
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)

                train_loss = 0.0

                if accelerator.is_main_process:
                    # save checkpoints!
                    if global_step % checkpointing_steps == 0:
                        save_motion_embedding(global_embds_wrap, output_dir, global_step)

                    if global_step % checkpointing_steps == 0 or global_step % 50 == 0:
                        embeddings = global_embds_wrap.get_all_angular_embeddigns()

                        # Save embeddings only at checkpointing_steps
                        if global_step % checkpointing_steps == 0:
                            save_angular_embeddings(embeddings, output_dir, global_step)

                        del embeddings

                    # sample images!
                    if (global_step % validation_steps == 0) or (global_step == 1):
                        logger.info(
                            f"Running validation... \n Generating {len(available_views)} videos."
                        )

                        pipeline = offload_training(
                            image_encoder,
                            vae,
                            unet,
                            pretrained_model_name_or_path,
                            weight_dtype,
                            accelerator,
                        )

                        with torch.autocast(
                            str(accelerator.device).replace(":0", ""),
                            enabled=accelerator.mixed_precision == "fp16",
                        ):
                            with torch.no_grad():
                                # See the training samples:
                                for sample in tqdm(
                                    multi_angle_train_dataloader,
                                    desc="Training samples",
                                ):
                                    inference_val(
                                        pipeline,
                                        output_dir,
                                        sample["frame_0"],
                                        "train_cond",
                                        sample["angle_name"][0],
                                        global_embds_wrap,
                                        global_step,
                                        num_frames,
                                        height,
                                        width,
                                        generator,
                                        report_to,
                                    )

                                # Now validation samples:
                                if len(multi_angle_val_dataset) > 0:
                                    for sample in tqdm(
                                        multi_angle_val_dataloader,
                                        desc="Validation samples",
                                    ):
                                        for cond_img, cond_name in tqdm(
                                            zip(
                                                sample["first_frames"],
                                                sample["figures"],
                                            )
                                        ):

                                            inference_val(
                                                pipeline,
                                                output_dir,
                                                cond_img.squeeze(0),
                                                cond_name[0],
                                                sample["angle_name"][0],
                                                global_embds_wrap,
                                                global_step,
                                                num_frames,
                                                height,
                                                width,
                                                generator,
                                                report_to,
                                            )

                        reload_training(
                            pipeline,
                            image_encoder,
                            vae,
                            unet,
                            accelerator,
                        )

            logs = {
                "step_loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
            }
            progress_bar.set_postfix(**logs)

            if global_step >= max_train_steps:
                break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        global_embds_wrap = accelerator.unwrap_model(global_embds_wrap)
        save_motion_embedding(global_embds_wrap, output_dir, global_step)

    accelerator.end_training()


def seed_everything(seed):
    # Set environment variables
    os.environ["PYTHONHASHSEED"] = str(
        seed
    )  # Affects hash-based operations (if set early enough)

    # Seed Python, NumPy, and PyTorch RNGs
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

    # Configure CuDNN for determinism
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    # torch.autograd.set_detect_anomaly(True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/MT3D.yaml",
        required=False,
        help="Path to config file",
    )
    parser.add_argument(
        "--output_path",
        default="Expirements/test_for_push",
        type=str,
        help="Desired output dir",
    )
    parser.add_argument(
        "--num_anchors",
        default=5,
        type=int,
        help="Number of anchors",
    )
    parser.add_argument(
        "--source_path",
        # default="MixamoData/Source/Brian_Walking",
        default="hope/source/Brian_BreakdanceReady",
        required=False,
        help="Path to multi-view videos directory",
    )

    args = parser.parse_args()

    # Load YAML configuration
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    config["output_path"] = args.output_path
    config["source_path"] = args.source_path
    config["num_anchors"] = args.num_anchors
    config["stage"] = 1

    args = argparse.Namespace(**config)
    seed_everything(0)

    # Get available views
    available_views = get_available_views(args.source_path)
    
    # Get output path
    output_dir = get_motion_embeddings_output_path(args.output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # Get first_frames_path if available (for validation)
    first_frames_path = getattr(args, "first_frames_path", None)
    
    extract_motion(
        source_path=args.source_path,
        output_dir=output_dir,
        available_views=available_views,
        num_anchors=args.num_anchors,
        first_frames_path=first_frames_path,
        validation_steps=args.validation_steps,
        checkpointing_steps=args.checkpointing_steps,
        max_train_steps=args.max_train_steps,
    )


if __name__ == "__main__":
    main()
