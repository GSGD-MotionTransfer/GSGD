import argparse
import datetime
import os
import sys
import time

import cv2
import matplotlib.pyplot as plt
import pandas as pd
import yaml

sys.path.append(os.path.join(os.path.dirname(__file__), "SC4D"))

import glob
import time
from collections import namedtuple
from pathlib import Path

import imageio
import numpy as np
import rembg
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from knn_cuda import KNN
from PIL import Image
from torchvision.utils import save_image
from tqdm import tqdm

import pytorch3d.ops as ops
from chamferdist import ChamferDistance
from saving import (
    get_available_views,
    get_first_frames_output_path,
    save_target_image,
    infer_source_name,
)
from gs_renderer import (
    MiniCam,
    Renderer,
    initialize_weights,
    initialize_weights_one,
    initialize_weights_zero,
)
from SC4D.cam_utils import OrbitCamera, orbit_camera


def render_target_images(target_path, output_dir, available_views, args):
    # Load the gaussians of the target
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    renderer = Renderer(sh_degree=args.sh_degree)
    cam = OrbitCamera(args.W, args.H, r=args.radius, fovy=args.fovy)

    g = renderer.gaussians
    renderer.gaussians._timenet = renderer.gaussians._timenet.to(device)

    g.load_ply(target_path)
    num_pts = g._xyz.shape[0]

    for i in tqdm(available_views):
        circle_azi = float(i)

        circle_pose = orbit_camera(0, circle_azi, args.radius)
        color = torch.ones((num_pts, 3), device=device) * 0.1
        circle_cur_cam = MiniCam(
            circle_pose,
            args.W,
            args.H,
            cam.fovy,
            cam.fovx,
            cam.near,
            cam.far,
        )

        # circle_out = renderer.render(
        #     circle_cur_cam, override_color=color, time=0.0, stage="s1"
        # )
        circle_out = renderer.render(circle_cur_cam, time=0.0, stage="s1")
        circle_img = (
            circle_out["image"].detach().cpu().permute(1, 2, 0).numpy() * 255
        )  # 1,2,0
        circle_img = circle_img.astype("uint8")[..., [2, 1, 0]]  # RGB to BGR

        save_target_image(circle_img, output_dir, str(float(i)))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/MT3D.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--output_path",
        required=True,
        type=str,
        help="Desired output dir",
    )
    parser.add_argument(
        "--target_path",
        required=True,
        help="Path to pretrained gaussians",
    )
    parser.add_argument(
        "--motion_embedding_path",
        required=True,
        help="Path to motion embedding folder to infer angles and source name.",
    )

    args = parser.parse_args(argv)

    # Load YAML configuration
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    config["config"] = args.config
    config["output_path"] = args.output_path
    config["target_path"] = args.target_path
    config["motion_embedding_path"] = args.motion_embedding_path
    config["stage"] = 2

    args = argparse.Namespace(**config)

    # Determine source name and views from motion embedding
    source_name = infer_source_name(args.motion_embedding_path)
    available_views = get_available_views(args.motion_embedding_path)
    
    # Get output path using inferred source name
    synthetic_source_path = os.path.join("Outputs", "MotionEmbeddings", source_name)
    output_dir = get_first_frames_output_path(
        synthetic_source_path,
        args.target_path,
        args.output_path
    )
    os.makedirs(output_dir, exist_ok=True)

    os.environ["NUMBA_DISABLE_CACHING"] = (
        "1"  # to avoid conflicts when running many slurm jobs in parallel
    )
    print("NUMBA_DISABLE_CACHING", os.environ["NUMBA_DISABLE_CACHING"])
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
        "expandable_segments:True"  # hope it will avoid the fragmentation issues we have when running many slurm jobs, of trying to allocate 131GB
    )
    print("PYTORCH_CUDA_ALLOC_CONF", os.environ["PYTORCH_CUDA_ALLOC_CONF"])
    render_target_images(args.target_path, output_dir, available_views, args)


def run_render_target(
    config: str,
    output_path: str,
    target_path: str,
    motion_embedding_path: str,
) -> None:
    argv = [
        "--config",
        config,
        "--output_path",
        output_path,
        "--target_path",
        target_path,
        "--motion_embedding_path",
        motion_embedding_path,
    ]
    main(argv)


if __name__ == "__main__":
    main()
