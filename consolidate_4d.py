import datetime
import json
import os
import shutil
import sys

import cv2
import matplotlib.pyplot as plt
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "SC4D"))

import argparse
import dataclasses
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
import tqdm
import yaml
from diffusers import StableVideoDiffusionPipeline
from knn_cuda import KNN
from omegaconf import OmegaConf
from PIL import Image
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchvision.utils import save_image

import pytorch3d.ops as ops
import wandb
from chamferdist import ChamferDistance
from saving import (
    get_reconstruction_output_paths,
    load_reconstruction_images,
    load_and_preprocess_input,
    save_visualization_video,
    save_mask,
    extract_names_from_path,
)
from deform_utils import cal_connectivity_from_points
from gs_renderer import (
    MiniCam,
    Renderer,
    initialize_weights,
    initialize_weights_one,
    initialize_weights_zero,
)
from losses import SDSSVDLoss
from SC4D.cam_utils import OrbitCamera, orbit_camera
from visual_helpers import save_concatenated_images

# Utils of reenact_anything. It isn't written as a package, so we need to import it like this
sys.path.append(os.path.join(os.path.dirname(os.getcwd()), "reenact-anything"))
from utils.embds_inversion_utils import ImageEmbeddingWrapper, allow_motion_embedding
from utils.video_utils import MotionSingleVideoDataset, SimpleImagesDataset

from losses import SDSSVDLoss

ViewPose = namedtuple("ViewPose", ["view_angle", "pose"])


def is_number(s):
    try:
        float(s)  # Check if it can be converted to a float
        return True
    except ValueError:
        return False


def optimizer_step_and_log_grads(mlp, optimizer):
    """
    NOTE: IT RUNS OPTIMIZER.STEP() INSIDE!
    """
    # Choose params to log:
    mlp_params = {
        name: param for name, param in mlp.named_parameters() if param.requires_grad
    }
    # Add the optimizer params that contain only 1 param (to add stuff like opacity, r, f_dc... in addition to the timenet. And the timenet we took directly from the timenet to still have the namings of their layers)
    optim_additional_params = {
        param_group["name"]: param_group["params"][0]
        for param_group in optimizer.param_groups
        if len(param_group["params"]) == 1
    }
    for param_group in optimizer.param_groups:
        # Assert we log all og the gradients, everything the optimizer optimizes over, including the deform that we log separately
        assert (
            param_group["name"] in ["deform", "deform_rot"]
            or len(param_group["params"]) == 1
        ), "We didnt support logging grads for param {} since it didnt exist in the past. You should add its logging as well.".format(
            param_group["name"]
        )
    params_to_log = {**mlp_params, **optim_additional_params}

    # Store the parameters values before the step
    before_step_data = {
        name: param.data.clone().detach() for name, param in params_to_log.items()
    }
    before_step_grad = {
        name: (param.grad.clone().detach() if param.grad is not None else None)
        for name, param in params_to_log.items()
    }

    # PERFORM THE OPTIMIZER STEP
    optimizer.step()
    log_dicts = []

    # Iterate over parameters to log information
    for name, param in params_to_log.items():
        # Skip biases (you can refine this logic if needed)
        if "bias" in name:
            continue

        cur_data = before_step_data[name]
        cur_grad = before_step_grad[name]
        # Get the update applied to the parameter
        update = param.data - cur_data

        if cur_grad is None:
            print(f"Warning: Grad is None for parameter '{name}'")
            cur_grad = torch.full_like(cur_data, torch.nan)

        # Log dictionary
        log_dicts.append(
            {
                "param_name": name,
                # Weights themselves:
                # User param_beofore_step (instead of param.data) as the update was done against it, and not against the new value..
                "data_median": torch.median(cur_data).item(),
                "data_mean": torch.mean(cur_data).item(),
                "data_std": torch.std(cur_data).item(),
                # Grads:
                "grad_median": torch.median(cur_grad).item(),
                "grad_mean": torch.mean(cur_grad).item(),
                "grad_std": torch.std(cur_grad).item(),
                # Update of weights:
                "update_median": torch.median(update).item(),
                "update_mean": torch.mean(update).item(),
                "update_std": torch.std(update).item(),
                # Update vs Weights:
                "update_vs_data_median": (torch.median(update) / torch.median(cur_data))
                .log10()
                .item(),
                "update_vs_data_mean": (torch.mean(update) / torch.mean(cur_data))
                .log10()
                .item(),
                "update_vs_data_std": (torch.std(update) / torch.std(cur_data))
                .log10()
                .item(),
            }
        )

    # To avoid later "optimizer.step()" mistakenly doing double steps:
    optimizer.zero_grad()

    return log_dicts


class GUI:
    def __init__(self, supervision_path, target_path, final_output_path, checkpoints_path, args):
        self.args = args  # shared with the trainer's args to support in-place modification of rendering parameters.
        self.target_path = target_path
        self.final_output_path = final_output_path
        self.checkpoints_path = checkpoints_path
        self.W = args.W
        self.H = args.H
        self.cam = OrbitCamera(args.W, args.H, r=args.radius, fovy=args.fovy)

        self.seed = 0
        self.seed_everything()

        # models
        self.device = torch.device("cuda")
        self.bg_remover = None

        # Supervision Images
        image_list, selected_views = load_reconstruction_images(supervision_path, args.num_frames)

        self.num_frames = len(image_list[0])

        self.source_poses = []
        self.source_images = []
        self.source_masks = []
        self.source_time = []
        for i, v_angle_str in enumerate(selected_views):
            v_angle = float(v_angle_str)
            v_pose = orbit_camera(0, v_angle, self.args.radius)
            cur_images = []
            cur_masks = []
            cur_time = []

            for j in range(self.num_frames):
                mask, image = load_and_preprocess_input(
                    image_list[i][j], self.args.W, self.args.H, self.args.ref_size
                )
                mask = mask.to(self.device)
                image = image.to(self.device)
                cur_images.append(image)
                cur_masks.append(mask)
                cur_time.append(j / self.num_frames)

            self.source_poses.append(ViewPose(v_angle, v_pose))
            self.source_images.append(cur_images)
            self.source_masks.append(cur_masks)
            self.source_time.append(cur_time)

        # renderer
        self.renderer = Renderer(
            sh_degree=self.args.sh_degree,
            override_deformation_rotation=args.override_deformation_rotation,
        )
        self.test_renderer = Renderer(
            sh_degree=self.args.sh_degree,
            override_deformation_rotation=args.override_deformation_rotation,
        )

        # training stuff
        self.argsimizer = None
        self.step = 0
        self.train_steps = 1  # steps per rendering loop
        self.stage = "s1"

        if self.args.train_dynamic:
            # override if provide a checkpoint
            if self.args.load_stage == "s1":
                save_path = os.path.join(
                    self.final_output_path, "s1/point_cloud.ply"
                )
                g = self.renderer.gaussians
                g.load_ply(save_path)
                self.renderer.initialize(
                    num_pts=g._xyz.shape[0], num_cpts=g._xyz.shape[0]
                )
            else:
                # initialize gaussians to a blob
                self.renderer.initialize(
                    num_pts=self.args.num_cpts, num_cpts=self.args.num_cpts
                )

        self.chamferDist = ChamferDistance()
        self.cpts_s1 = []

        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).to(
            self.device
        )
        self.low_pass_filter = T.GaussianBlur(
            kernel_size=args.low_pass_kernel, sigma=args.low_pass_sigma
        ).to(self.device)

        # SDS loss will be initialized later if needed (requires first_frames_path and motion_embedding_path)
        # Initialize as None, will be set up if use_sds is True and paths are available
        self.sds_loss = None
        self._sds_initialized = False

    def seed_everything(self):
        try:
            seed = int(self.seed)
        except:
            print("Seed is not a number, generating a random one")
            seed = np.random.randint(0, 1000000)

        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        # torch.use_deterministic_algorithms(True) # RuntimeError: reflection_pad2d_backward_cuda does not have a deterministic implementation

        self.last_seed = seed

    def prepare_train_s1(self):

        self.step = 0
        self.stage = "s1"
        # self.args.position_lr_max_steps = 500

        # setup training
        self.renderer.gaussians.training_setup(self.args)
        # do not do progressive sh-level
        self.renderer.gaussians.active_sh_degree = self.renderer.gaussians.max_sh_degree
        self.optimizer = self.renderer.gaussians.optimizer

        self.init_control_points_and_optimizer()

        # Do not optimize over the canonical control points (TODO remove from the optimizers instead of this patch here. But for now we keep it in case we will want to optimize over c_xyz later)
        if self.stage == "s1":
            for param_group in self.optimizer.param_groups:
                if param_group["name"] in [
                    "c_xyz",
                    "xyz",
                ]:  # probably can change 'in ["c_xyz", "xyz"]' to '="xyz"'
                    param_group["lr"] = 0.0

    def init_control_points_and_optimizer(self):
        g = self.renderer.gaussians
        g.load_ply(self.target_path)
        g.max_radii2D = torch.zeros(
            (g.get_xyz.shape[0]), device="cuda"
        )  # fill since it isn't filled by default and is kept with the GUI.init() initialization
        # Optimizer initialization:
        g.training_setup(self.args)
        self.optimizer = self.renderer.gaussians.optimizer
        # Now FPS
        self.FPS(num_pts=self.args.num_cpts)

    def init_object(self, g):
        num_cp = g._xyz.shape[0]
        pretrained_object = self.pretrained_renderer.gaussians
        pretrained_object.load_ply(self.target_path)
        random_idx = torch.randperm(pretrained_object._xyz.shape[0])[:num_cp]
        new_cp = pretrained_object._xyz[random_idx]

        with torch.no_grad():
            g._c_xyz.copy_(new_cp)
            g._scaling.copy_(g._r.expand_as(g._xyz))
            g._c_radius.copy_(g._r.expand_as(g._c_radius))

        return g

    def init_object_around_cpts(self):
        g_cpts_only = self.renderer.gaussians
        with torch.no_grad():
            g_cpts_only._c_xyz.copy_(g_cpts_only._xyz)
            g_cpts_only._c_radius.copy_(g_cpts_only._r)

        g_cpts_only.load_ply(self.target_path)
        g_intialized_around_cpts = g_cpts_only
        return g_intialized_around_cpts

    def prepare_train_s2(self):

        self.stage = "s2"
        self.step = 0

        g = self.init_object_around_cpts()

        # update training
        self.renderer.gaussians.training_setup(self.args)
        self.renderer.gaussians.active_sh_degree = self.renderer.gaussians.max_sh_degree

        # Like in scgs, arap_deform.py: 48
        trajectory = self.renderer.gaussians._c_xyz[:, None].repeat(
            1, 16, 1
        )  # mimic trajectory from scgs when no time deformation is applied. Do it only to stay as close as possible to scgs, I think it is relevant because of the radius.
        node_radius = torch.full((self.renderer.gaussians._c_xyz.shape[0],), 0.1205)
        self.renderer.gaussians.c_nn_K = 16
        ii_c, jj_c, nn_c, weight_c = cal_connectivity_from_points(
            self.renderer.gaussians._c_xyz,
            0.1715,
            self.renderer.gaussians.c_nn_K,
            trajectory=trajectory,
            node_radius=node_radius,
        )
        self.renderer.gaussians.ii_c = ii_c
        self.renderer.gaussians.jj_c = jj_c
        self.renderer.gaussians.nn_c = nn_c
        self.renderer.gaussians.weight_c = weight_c

        self.optimizer = self.renderer.gaussians.optimizer

        g._r = torch.tensor([], device="cuda")
        for i in reversed(range(len(self.optimizer.param_groups))):
            param_group = self.optimizer.param_groups[i]
            if param_group["name"] not in ["deform", "deform_rot"]:

                self.optimizer.param_groups.pop(i)

        self.test_cpts(test_stage=self.stage, render_type="fixed")

    def get_global_step(self):
        return self.step + (0 if self.stage == "s1" else self.args.iters_s1)

    def train_step(self, i):
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()

        # if self.stage == "s1" and self.step == self.args.FPS_iter:
        #     self.FPS(num_pts=self.args.num_cpts)

        if self.stage == "s2" and self.step == 0:
            c_means3D = self.renderer.gaussians._c_xyz

            # Here we assume all the different views are with the same frames number
            for t in self.source_time[0]:
                means3D_deform, _ = self.renderer.gaussians._timenet(c_means3D, t)
                self.cpts_s1.append(c_means3D + means3D_deform)

        for _ in range(self.train_steps):

            self.step += 1
            log_dict = {}

            if self.stage == "s1":
                iters = self.args.iters_s1
            elif self.stage == "s2":
                iters = self.args.iters_s2
            else:
                assert ValueError(
                    "Video-to-4D generation of SC4D only contain two stages!!!"
                )
            step_ratio = self.step / iters

            self.renderer.gaussians.update_learning_rate(self.step, self.stage)

            # find knn
            if self.stage >= "s2":
                # Currently we re-run it every step and it isn't necessary. If we will optimize over the gaussians themselves (and not only the timenet) during s2 then this line can be significant
                self.find_knn(g=self.renderer.gaussians, k=4)

            loss = 0

            # random reference view and frame index
            v_idx = np.random.randint(0, len(self.source_poses))
            max_f_idx = int(
                len(self.source_images[v_idx])
                * min(self.get_global_step() / float(self.args.all_frames_steps), 1)
            )  # gradually advance in frames over the training in case of onyl stage 2 training.
            f_idx = np.random.randint(0, max_f_idx) if max_f_idx > 0 else 0

            self.cur_pose = self.source_poses[v_idx].pose
            self.cur_pose_angle = self.source_poses[v_idx].view_angle

            self.input_img_torch = self.source_images[v_idx][f_idx]
            self.input_mask_torch = self.source_masks[v_idx][f_idx]
            self.timestamp = self.source_time[v_idx][f_idx]

            if self.stage == "s2":
                # used later for the "gaussian alignment" loss
                self.cpts_ori = self.cpts_s1[f_idx]

            if self.args.use_sds:
                assert (
                    self.W == 1024 and self.H == 576
                ), "SDS using Stable Video Diffusion 576x1024 is not supported for non-512x512 images"
                
                # Initialize SDS loss if not already initialized
                if not self._sds_initialized and self.sds_loss is None:
                    # This should have been initialized in __init__, but handle case where it wasn't
                    pass
                
                if self.sds_loss is not None:
                    img_w, img_h = self.W, self.H

                    cur_cam = MiniCam(
                        self.cur_pose,
                        img_w,
                        img_h,
                        self.cam.fovy,
                        self.cam.fovx,
                        self.cam.near,
                        self.cam.far,
                    )
                    out_lists = self.renderer.render(cur_cam, time=None, stage=self.stage)
                    out_vid = torch.stack(out_lists["image"], dim=0)
                    sds_loss = self.sds_loss(
                        out_vid.unsqueeze(0) * 2 - 1,  # SVD expects values between -1 and 1
                        angle=str(self.cur_pose_angle),
                        grad_scale=self.args.sds_grad_scale,
                    )
                    loss += sds_loss
                    log_dict["loss/sds"] = sds_loss.item()

            else:
                img_w, img_h = 512, 512
                ### known view
                cur_cam = MiniCam(
                    self.cur_pose,
                    img_w,
                    img_h,
                    self.cam.fovy,
                    self.cam.fovx,
                    self.cam.near,
                    self.cam.far,
                )
                out = self.renderer.render(
                    cur_cam, time=self.timestamp, stage=self.stage
                )

                # image loss
                image = out["image"].unsqueeze(0)  # [1, 3, H, W] in [0, 1]
                input_img_torch = F.interpolate(
                    self.input_img_torch,
                    (img_w, img_h),
                    mode="bilinear",
                    align_corners=False,
                )
                image_loss = self.args.lambda_mse * F.mse_loss(image, input_img_torch)
                loss = loss + image_loss
                log_dict["loss/image"] = image_loss.item()

                # mask loss
                mask = out["alpha"].unsqueeze(0)  # [1, 1, H, W] in [0, 1]
                input_mask_torch = F.interpolate(
                    self.input_mask_torch,
                    (img_w, img_h),
                    mode="bilinear",
                    align_corners=False,
                )
                mask_loss = self.args.lambda_mask * F.mse_loss(mask, input_mask_torch)
                loss = loss + mask_loss
                log_dict["loss/mask"] = mask_loss.item()

                # lpips loss
                lpips_loss = self.args.lambda_lpips * self.lpips(image, input_img_torch)
                loss = loss + lpips_loss
                log_dict["loss/lpips"] = lpips_loss.item()

                # low_freq_motion loss
                low_pass_filtered_pred = self.low_pass_filter(image)
                low_pass_filtered_gt = self.low_pass_filter(input_img_torch)
                diff_func = (
                    self.lpips
                    if self.args.diff_func_low_freq_motion == "lpips"
                    else F.mse_loss
                )
                low_freq_motion_loss = self.args.lambda_low_freq_motion * diff_func(
                    low_pass_filtered_pred, low_pass_filtered_gt
                )
                loss += low_freq_motion_loss
                log_dict["loss/low_freq_motion"] = low_freq_motion_loss

            if self.args.use_arap and self.stage == "s2":  # and self.step < 2000:
                loss_arap, conns = self.renderer.arap_loss_v2(stage=self.stage)
                arap_loss_weighted = self.args.lambda_arap * loss_arap
                loss += arap_loss_weighted
                log_dict["loss/arap"] = arap_loss_weighted.item()

            # Canonical c_pts alignment to target ply alignment
            # if self.stage == "s1":
            #     canonical_alignment_loss = (
            #         self.args.lambda_ca * self.renderer.cannoical_alignment_loss()
            #     )
            #     loss += canonical_alignment_loss
            #     log_dict["loss/ca"] = canonical_alignment_loss.item()

            with torch.no_grad():
                if (
                    self.args.do_inference
                    and (self.step - 1) % self.args.check_inter == 0
                ):
                    self.test_3d(render_type=self.args.render_type)

            # optimize step
            loss.backward()

            log_dict["f_idx"] = f_idx
            log_dict["pose"] = self.source_poses[v_idx].view_angle
            log_dict["loss/total"] = loss.item()

            # Logs:

            if self.args.report_to_wandb:
                # Loss
                wandb.log(log_dict, step=self.get_global_step())

            # Log things that are quick to log
            if (self.step - 1) % self.args.log_quick_iters == 0:
                # Gradients
                grads_log_dicts = optimizer_step_and_log_grads(
                    self.renderer.gaussians._timenet, self.optimizer
                )
                if self.args.report_to_wandb:
                    grad_table = None
                    dict_cols = sorted(list(grads_log_dicts[0].keys()))
                    if grad_table is None:
                        grad_table = wandb.Table(columns=["step"] + dict_cols)
                    for grads_dict in grads_log_dicts:
                        grad_table.add_data(
                            *(
                                [self.get_global_step()]
                                + [grads_dict.get(k, None) for k in dict_cols]
                            )
                        )
                    wandb.log({"grad_table": grad_table}, step=self.get_global_step())

                if self.args.use_sds and self.sds_loss is not None:
                    self.sds_loss.log_latents(
                        step=str(self.get_global_step())
                        + "_"
                        + str(self.cur_pose_angle)
                    )
                else:
                    # Supervision Vs Predicted Visually
                    savename = "gt_vs_pred"
                    gt_vs_pred_output_path = os.path.join(
                        self.final_output_path,
                        "{}_step_{}.png".format(savename, self.get_global_step()),
                    )
                    save_concatenated_images(
                        [
                            input_img_torch,
                            image,
                            input_mask_torch,
                            mask,
                            low_pass_filtered_gt,
                            low_pass_filtered_pred,
                        ],
                        output_path=gt_vs_pred_output_path,
                    )
                    if self.args.report_to_wandb:
                        wandb.log(
                            {savename: wandb.Image(gt_vs_pred_output_path)},
                            step=self.get_global_step(),
                        )
                    print("Saved {}".format(gt_vs_pred_output_path))

            self.optimizer.step()  # if ran "optimizer_step_and_log_grads" before, won't make an effect
            self.optimizer.zero_grad()

            if (
                self.step % self.args.save_inter == 0
                or
                # next step we will FPS the point cloud
                self.stage == "s1"
                and self.step == self.args.FPS_iter - 1
            ):
                save_path = os.path.join(
                    self.final_output_path, "checkpoints", self.stage
                )
                path2 = (
                    os.path.join(save_path, "point_cloud_c_{}.ply".format(self.step))
                    if self.stage >= "s2"
                    else None
                )
                # self.renderer.gaussians.save_ply(
                #     os.path.join(
                #         self.dal.checkpoints_path, f"point_cloud_{self.step}.ply"
                #     ),
                #     path2,
                # )
                # self.renderer.gaussians.save_model(
                #     self.dal.checkpoints_path, step=self.step
                # )

            # densify and prune
            # if self.stage == "s1":
            #     if (
            #         self.step >= self.opt.density_start_iter
            #         and self.step <= self.opt.density_end_iter
            #     ):
            #         viewspace_point_tensor, visibility_filter, radii = (
            #             out["viewspace_points"],
            #             out["visibility_filter"],
            #             out["radii"],
            #         )
            #         self.renderer.gaussians.max_radii2D[visibility_filter] = torch.max(
            #             self.renderer.gaussians.max_radii2D[visibility_filter],
            #             radii[visibility_filter],
            #         )
            #         self.renderer.gaussians.add_densification_stats(
            #             viewspace_point_tensor, visibility_filter
            #         )

            #         if self.step % self.opt.densification_interval == 0:
            #             self.renderer.gaussians.densify_and_prune(
            #                 self.opt.densify_grad_threshold,
            #                 min_opacity=0.01,
            #                 extent=4,
            #                 max_screen_size=1,
            #             )
            #             print(
            #                 "Num of gaussians: ", self.renderer.gaussians._xyz.shape[0]
            #             )

            #         if self.step % self.opt.opacity_reset_interval == 0:
            #             self.renderer.gaussians.reset_opacity()

            #         if self.step % self.args.opacity_reset_interval == 0:
            #             self.renderer.gaussians.reset_opacity()

        ender.record()
        torch.cuda.synchronize()
        t = starter.elapsed_time(ender)

    def find_knn(self, g, k=4):
        control_pts = g._c_xyz.detach()
        gaussian_pts = g._xyz.detach()
        knn = KNN(k=k, transpose_mode=True)
        dist, indx = knn(
            control_pts.unsqueeze(0), gaussian_pts.unsqueeze(0)
        )  # 32 x 50 x 10
        dist, indx = dist[0], indx[0]
        g.neighbor_dists = dist
        g.neighbor_indices = indx

    def FPS(self, num_pts):
        g = self.renderer.gaussians
        _, idxs = ops.sample_farthest_points(points=g._xyz.unsqueeze(0), K=num_pts)
        idxs = idxs[0]
        mask_to_prune = torch.ones_like(g._xyz, dtype=torch.bool)[:, 0]
        mask_to_prune[idxs] = False
        g.prune_points(mask_to_prune)

    def load_model(self, g):
        load_stage = self.args.load_stage or self.args.test_stage
        path1 = os.path.join(self.checkpoints_path, "point_cloud.ply")
        path2 = os.path.join(self.checkpoints_path, "point_cloud_c.ply")

        model_dir = self.checkpoints_path
        if self.args.test_step:
            path1 = path1.split(".")[0] + "_{}".format(self.args.test_step) + ".ply"
            if test_stage > "s1":
                path2 = path2.split(".")[0] + "_{}".format(self.args.test_step) + ".ply"
        if load_stage < "s2":
            path2 = None
        g.load_ply(path1, path2)
        g.load_model(model_dir, self.args.test_step)

    def test_3d(self, test_cpts=True, render_type="fixed"):
        video_save_dir = self.args.output_path
        if not os.path.exists(video_save_dir):
            os.makedirs(video_save_dir)
        frames_0 = []
        frames_0_subsampled = []
        frames_90 = []
        init_ver = 0
        if test_cpts:
            self.test_cpts(test_stage=self.stage, render_type=render_type)
        for i in range(32):
            pose = orbit_camera(0, init_ver, self.args.radius)
            cur_cam = MiniCam(
                pose,
                self.W,
                self.H,
                self.cam.fovy,
                self.cam.fovx,
                self.cam.near,
                self.cam.far,
            )
            out = self.renderer.render(cur_cam, time=i / 32, stage=self.stage)
            img = out["image"].detach().cpu().permute(1, 2, 0).numpy() * 255
            img = img.astype("uint8")
            frames_0.append(img)
        # compose video

        save_visualization_video(frames_0, self.final_output_path, "render", self.get_global_step(), fps=self.args.fps_save)

        for i in range(32):
            pose = orbit_camera(0, init_ver, self.args.radius)
            cur_cam = MiniCam(
                pose,
                self.W,
                self.H,
                self.cam.fovy,
                self.cam.fovx,
                self.cam.near,
                self.cam.far,
            )
            out = self.renderer.render(
                cur_cam, time=i / 32, stage=self.stage, subsample_ratio=0.05
            )
            img = out["image"].detach().cpu().permute(1, 2, 0).numpy() * 255
            img = img.astype("uint8")
            frames_0_subsampled.append(img)
        # compose video

        save_visualization_video(frames_0_subsampled, self.final_output_path, "render_subsampled", self.get_global_step(), fps=self.args.fps_save)

        # Now for 90 degrees, for animals mainly
        for i in range(32):
            pose = orbit_camera(0, 90, self.args.radius)
            cur_cam = MiniCam(
                pose,
                self.W,
                self.H,
                self.cam.fovy,
                self.cam.fovx,
                self.cam.near,
                self.cam.far,
            )
            out = self.renderer.render(cur_cam, time=i / 32, stage=self.stage)
            img = out["image"].detach().cpu().permute(1, 2, 0).numpy() * 255
            img = img.astype("uint8")
            frames_90.append(img)
        # compose video

        save_visualization_video(frames_90, self.final_output_path, "render_90", self.get_global_step(), fps=self.args.fps_save)

        # if self.args.report_to_wandb:
        #     log_video_wandb("render", video_name, frames, self.get_global_step())

    def test_cpts(self, test_stage="s1", render_type="fixed", sh_degree=0):
        renderer = Renderer(sh_degree=sh_degree)
        if test_stage > "s1":
            renderer.initialize(num_pts=self.renderer.gaussians._c_xyz.shape[0])
            renderer.gaussians._xyz = self.renderer.gaussians._c_xyz
        else:
            renderer.initialize(num_pts=self.renderer.gaussians._xyz.shape[0])
            renderer.gaussians._xyz = self.renderer.gaussians._xyz
        renderer.gaussians._r = (
            torch.ones((1), device="cuda", requires_grad=True) * -5.0
        )
        renderer.gaussians._timenet = self.renderer.gaussians._timenet
        num_pts = renderer.gaussians._xyz.shape[0]
        device = renderer.gaussians._xyz.device
        renderer.gaussians._scaling = (
            torch.ones((num_pts, 3), device=device, requires_grad=True) * -5.0
        )
        renderer.gaussians._opacity = (
            torch.ones((num_pts, 1), device=device, requires_grad=True) * 2.0
        )
        color = torch.ones((num_pts, 3), device=device) * 0.1
        frames_0, frames_90 = [], []
        init_ver = 0
        ###
        cpts_tra = 0
        for i in range(32):
            pose_0 = orbit_camera(0, 0, self.args.radius)
            pose_90 = orbit_camera(0, 90, self.args.radius)
            cam_0 = MiniCam(
                pose_0,
                self.W,
                self.H,
                self.cam.fovy,
                self.cam.fovx,
                self.cam.near,
                self.cam.far,
            )
            cam_90 = MiniCam(
                pose_90,
                self.W,
                self.H,
                self.cam.fovy,
                self.cam.fovx,
                self.cam.near,
                self.cam.far,
            )
            out_0 = renderer.render(
                cam_0, override_color=color, time=i / 32, stage="s1"
            )
            out_90 = renderer.render(
                cam_90, override_color=color, time=i / 32, stage="s1"
            )
            img_0 = out_0["image"].detach().cpu().permute(1, 2, 0).numpy() * 255
            img_0 = img_0.astype("uint8")
            frames_0.append(img_0)
            img_90 = out_90["image"].detach().cpu().permute(1, 2, 0).numpy() * 255
            img_90 = img_90.astype("uint8")
            frames_90.append(img_90)

            ###
            if i == 0:
                cpts_tmp = out_0["cpts_t"]
            cpts_t = out_0["cpts_t"]
            cpts_tra += torch.dist(cpts_t, cpts_tmp, p=2)
            cpts_tmp = cpts_t
        print("cpts average moving length: ", cpts_tra.item())
        if self.args.report_to_wandb:
            wandb.log(
                {"cpts_average_moving_length": cpts_tra.item()},
                step=self.get_global_step(),
            )
        ###

        save_visualization_video(frames_0, self.final_output_path, "cpts_fixed", self.get_global_step(), fps=self.args.fps_save)
        save_visualization_video(frames_90, self.final_output_path, "cpts_fixed_90", self.get_global_step(), fps=self.args.fps_save)

        # if self.args.report_to_wandb:
        #     log_video_wandb(
        #         "render_cpts", circle_video_name, circle_frames, self.get_global_step()
        #     )

    def test(self, test_cpts=True, render_type="fixed", save_ply_path=None):
        video_save_dir = self.args.output_path
        test_stage = self.args.test_stage
        if not os.path.exists(video_save_dir):
            os.makedirs(video_save_dir)
        fixed_frames = []
        g = self.renderer.gaussians
        self.load_model(g=g)
        if test_stage >= "s2":
            self.find_knn(g)
        if test_cpts:
            self.test_cpts(test_stage=self.args.test_stage, render_type=render_type)
        for i in range(32):
            fixed_azi = self.args.test_azi
            circle_azi = 360 / 32 * i
            fixed_pose = orbit_camera(0, fixed_azi, self.args.radius)
            circle_pose = orbit_camera(0, circle_azi, self.args.radius)
            fixed_cur_cam = MiniCam(
                fixed_pose,
                self.W,
                self.H,
                self.cam.fovy,
                self.cam.fovx,
                self.cam.near,
                self.cam.far,
            )
            circle_cur_cam = MiniCam(
                circle_pose,
                self.W,
                self.H,
                self.cam.fovy,
                self.cam.fovx,
                self.cam.near,
                self.cam.far,
            )

            fixed_out = self.renderer.render(
                fixed_cur_cam,
                time=i / 32,
                stage=test_stage,
                save_ply_path=(
                    os.path.join(save_ply_path, f"fixed_{i}.ply")
                    if save_ply_path is not None
                    else None
                ),
            )
            fixed_img = fixed_out["image"].detach().cpu().permute(1, 2, 0).numpy() * 255
            fixed_img = fixed_img.astype("uint8")
            fixed_frames.append(fixed_img)
            circle_frames = []
            for j in range(2, 32, 32 // 14)[:-1]:
                circle_out = self.renderer.render(
                    circle_cur_cam, time=j / 32, stage=test_stage
                )
                circle_img = (
                    circle_out["image"].detach().cpu().permute(1, 2, 0).numpy() * 255
                )
                circle_img = circle_img.astype("uint8")
                circle_frames.append(circle_img)
                # save the mask only if first frame
                if j == 2:
                    mask = (
                        circle_out["alpha"].detach().cpu().permute(1, 2, 0).numpy()
                        * 255
                    ).astype("uint8")
                    save_mask(mask, self.final_output_path, "final_output", circle_azi)

            save_visualization_video(circle_frames, self.final_output_path, "final_output", circle_azi, fps=self.args.fps_save, save_mp4=True)
        # self.dal.vis_figure("final_fixed_render", fixed_frames, self.get_global_step())

    def train_dynamic(self, iters_s1=1500, iters_s2=5000, load_stage=""):
        g = self.renderer.gaussians

        if self.args.report_to_wandb:
            wandb.watch(self.renderer.gaussians._timenet, log="all")

        ### Stage 1: coarse stage of video-to-4D generation
        self.prepare_train_s1()
        self.renderer.gaussians.lr_setup(self.args)
        if iters_s1 > 0:
            for i in tqdm.trange(iters_s1):
                self.train_step(i)
            print("Num of cpts after s1: ", self.renderer.gaussians._c_xyz.shape[0])
            # save s1
            save_path = os.path.join(self.final_output_path, "checkpoints/s1")
            # g.save_ply(os.path.join(save_path, "point_cloud.ply"))
            # g.save_model(save_path)

        ### Stage 2: fine stage of video-to-4D generation
        self.prepare_train_s2()
        self.renderer.gaussians.lr_setup(self.args)
        if iters_s2 > 0:
            for i in tqdm.trange(iters_s2):
                self.train_step(i)
            # save s2
            g.save_ply(
                os.path.join(self.checkpoints_path, "point_cloud.ply"),
                os.path.join(self.checkpoints_path, "point_cloud_c.ply"),
            )
            g.save_model(self.checkpoints_path)


def log_all(opt):

    config = OmegaConf.to_container(opt, resolve=True)
    # Ensure output path exists
    run_log_path = os.path.join(config["output_path"], "run_log")
    os.makedirs(run_log_path, exist_ok=True)

    # Save code files manually
    code_files = ["consolidate_4d.py", "gs_renderer.py", "saving.py"]
    code_save_path = os.path.join(run_log_path, "code_files")
    os.makedirs(code_save_path, exist_ok=True)

    for file in code_files:
        shutil.copy(file, code_save_path)

    print(f"Saved code files to {code_save_path}")

    # Save command-line arguments
    args_file_path = os.path.join(run_log_path, "config.json")

    with open(args_file_path, "w") as f:
        json.dump(config, f, indent=4)

    print(f"Saved arguments to {args_file_path}")

    # Save command used for execution
    cmd_args_file = os.path.join(run_log_path, "cmd_args.txt")
    cmd_args = " ".join(os.sys.argv)  # Capture full command used to run the script

    with open(cmd_args_file, "w") as f:
        f.write(cmd_args + "\n")

    print(f"Saved command arguments to {cmd_args_file}")


def init_wandb(args):
    config_dict = OmegaConf.to_container(args, resolve=True)
    json_friendly_config = {
        k: v
        for k, v in config_dict.items()
        if isinstance(v, (str, int, float, list, dict, bool, type(None)))
    }

    wandb.init(
        project="motionTransfer3D",
        config=json_friendly_config,
        group=args.get("run_group", None),
    )

    artifact = wandb.Artifact("code_files", type="code")
    artifact.add_file("./consolidate_4d.py")
    artifact.add_file("./gs_renderer.py")
    artifact.add_file("./saving.py")
    wandb.log_artifact(artifact)


def log_video_wandb(vid_desc, vid_path, frames, step):
    vid_suff = vid_path.split(".")[-1]
    vid_file_no_suff = ".".join(vid_path.split(".")[:-1])
    downsampled_out_file = vid_file_no_suff + "downsampled_" + "." + vid_suff

    # resize to save space in wandb
    orig_h, orig_w = frames[0].shape[-3:-1]
    downscale_by = 0.6
    new_size = (int(downscale_by * orig_w), int(downscale_by * orig_h))
    resized_video = np.array(
        [
            cv2.resize(frame, dsize=new_size, interpolation=cv2.INTER_CUBIC)
            for frame in frames
        ]
    )

    imageio.mimwrite(
        downsampled_out_file,
        resized_video,
        fps=10,
        quality=8,
        macro_block_size=1,
        loop=0,
    )
    wandb.log(
        {vid_desc: wandb.Video(downsampled_out_file, fps=8)},
        step=step,
    )


def add_timestamp_suff(dirname):
    now = datetime.datetime.now()
    return dirname + "_" + now.strftime("%Y-%m-%d_%H-%M-%S")


def initialize_sds_loss(opt, target_path, final_output_path, first_frames_path, motion_embedding_path, available_views, motion_epoch):
    from saving import load_motion_embedding
    
    first_frames_dataset = SimpleImagesDataset(
        first_frames_path,
        width=opt.width,
        height=opt.height,
        max_images_n=len(available_views),
        device="cuda",
    )
    first_frame_cond_per_view = {
        sample["name"]: sample["frame"] for sample in first_frames_dataset
    }

    motion_embeddings = {}
    for angle, _ in first_frame_cond_per_view.items():
        # step=-1 means it will take the last one. if you want specific checkpoint, change it!!
        angle_motion_embedding_path, _ = load_motion_embedding(
            motion_embedding_path, angle, step=motion_epoch
        )

        image_embds_wrap = ImageEmbeddingWrapper(torch.Tensor([0]))
        image_embds_wrap.requires_grad_(False)
        image_embds_wrap.to(device="cuda", dtype=torch.float32)
        image_embds_wrap.load_tensor(angle_motion_embedding_path)
        motion_embeddings[angle] = image_embds_wrap()
    print("loaded {} motion embeddings".format(len(motion_embeddings)))

    pipeline = StableVideoDiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid",
        torch_dtype=torch.float32,
    ).to(
        "cuda"
    )  # 10424 GB RAM

    allow_motion_embedding(pipeline.unet)

    pipeline.set_progress_bar_config(disable=True)
    pipeline.unet.requires_grad_(False)
    pipeline.vae.requires_grad_(False)
    pipeline.image_encoder.requires_grad_(False)

    # Now the sds part:
    @dataclasses.dataclass
    class sds_config:
        report_to_wandb: bool

        use_xformers: bool = False
        del_text_encoders: bool = False
        batch_size: int = 1
        num_iter: int = 1000
        save_vid_iter: int = 50
        same_noise_for_frames: bool = False
        sds_timestep_low: int = 900
        timesteps: int = 1000
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        output_folder: str = os.path.join(opt.output_path, "sds_debug")

    cfg = sds_config(
        report_to_wandb=False,
        # grad_scale=1.0,
        # lr_base=1e-3,
    )

    sds_loss = SDSSVDLoss(
        pipeline,
        first_frame_cond_per_view=first_frame_cond_per_view,
        motion_embedding_per_view=motion_embeddings,
        cfg=cfg,
        device=pipeline.device,
        run_folder="./",  # TODO: change to output path
        debug=True,
    )

    return sds_loss


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/MT3D.yaml",
        required=False,
        help="Path to config file",
    )
    parser.add_argument(
        "--target_path",
        required=True,
        help="Path to pretrained gaussians",
    )
    parser.add_argument(
        "--specify_supervision",
        required=True,
        help="Path to generated supervision videos",
    )
    parser.add_argument(
        "--motion_embedding_path",
        required=True,
        help="Path to pretrained global motion embedding (.pt) or dir of motion embeddings",
    )
    parser.add_argument(
        "--first_frames_path",
        default=None,
        required=False,
        help="Path to first frames (required if using SDS loss)",
    )
    parser.add_argument(
        "--motion_epoch",
        default="3000",
        required=False,
        help="Epoch of the motion embedding to use",
    )
    args, extras = parser.parse_known_args(argv)

    # Load YAML configuration
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    # Extract source_name, target_name, and output_path from supervision path
    # Supervision path format: ./Outputs/{source_name}/{target_name}/{output_path}/Supervision/
    source_name, target_name = extract_names_from_path(args.specify_supervision)
    
    # Extract output_path from supervision path
    supervision_dir = args.specify_supervision.rstrip("/")
    if supervision_dir.endswith("/Supervision"):
        supervision_dir = supervision_dir[:-12]  # Remove "/Supervision"
    output_path = os.path.basename(supervision_dir)
    
    # Construct synthetic source path for path building
    synthetic_source_path = os.path.join("Outputs", "MotionEmbeddings", source_name)

    # Get output paths
    final_output_path, checkpoints_path = get_reconstruction_output_paths(
        synthetic_source_path,
        args.target_path,
        output_path
    )
    os.makedirs(final_output_path, exist_ok=True)
    os.makedirs(checkpoints_path, exist_ok=True)

    # Merge config
    config["config"] = args.config
    config["output_path"] = output_path
    config["target_path"] = args.target_path
    config["specify_supervision"] = args.specify_supervision
    config["motion_embedding_path"] = args.motion_embedding_path
    config["motion_epoch"] = args.motion_epoch
    config["first_frames_path"] = args.first_frames_path
    config["stage"] = 4

    opt = OmegaConf.merge(config, vars(args))
    opt = OmegaConf.merge(opt, OmegaConf.from_cli(extras))

    # Set default values for missing config keys
    if "use_sds" not in opt:
        # Default to True only if dimensions match SDS requirements (W=1024, H=576)
        # Otherwise set to False to avoid assertion errors
        opt.use_sds = (opt.get("W") == 1024 and opt.get("H") == 576)

    # Initialize SDS loss if needed
    sds_loss = None
    if opt.use_sds:
        if not opt.first_frames_path:
            raise ValueError("first_frames_path is required when using SDS loss (use_sds=True)")
        if not opt.motion_embedding_path:
            raise ValueError("motion_embedding_path is required when using SDS loss (use_sds=True)")
        from saving import get_available_views
        available_views = get_available_views(opt.motion_embedding_path)
        sds_loss = initialize_sds_loss(
            opt, opt.target_path, final_output_path,
            opt.first_frames_path, opt.motion_embedding_path,
            available_views, opt.get("motion_epoch", "3000")
        )
    
    gui = GUI(args.specify_supervision, opt.target_path, final_output_path, checkpoints_path, opt)
    if sds_loss is not None:
        gui.sds_loss = sds_loss
        gui._sds_initialized = True
    
    log_all(opt)

    if opt.report_to_wandb:
        import wandb

        init_wandb(opt)

    if opt.train_dynamic:
        gui.train_dynamic(opt.iters_s1, opt.iters_s2, opt.load_stage)
    gui.test(
        render_type=opt.render_type,
        save_ply_path=os.path.join(final_output_path, "plys") if opt.save_plys else None,
    )

    if opt.report_to_wandb:
        wandb.finish()


def run_consolidate_4d(
    config: str,
    target_path: str,
    specify_supervision: str,
    motion_embedding_path: str,
    first_frames_path: str | None = None,
    motion_epoch: str = "3000",
) -> None:
    argv = [
        "--config",
        config,
        "--target_path",
        target_path,
        "--specify_supervision",
        specify_supervision,
        "--motion_embedding_path",
        motion_embedding_path,
        "--motion_epoch",
        str(motion_epoch),
    ]
    if first_frames_path:
        argv += ["--first_frames_path", first_frames_path]
    main(argv)


if __name__ == "__main__":
    main()
