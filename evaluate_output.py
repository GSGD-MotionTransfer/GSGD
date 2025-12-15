import argparse
import math
import os
import pickle
import subprocess
import time
from glob import glob
from pathlib import Path

import clip
import cv2
import imageio
import lpips
import numpy as np
import pandas as pd
import rembg
import torch
import torchmetrics
import yaml
from cotracker.predictor import CoTrackerPredictor
from cotracker.utils.visualizer import Visualizer, read_video_from_path
from einops import rearrange
from PIL import Image, ImageSequence
from tqdm import tqdm


class MotionFidelity:
    def __init__(self, args, cotracker_model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.args = args
        self.cotracker = CoTrackerPredictor(checkpoint=cotracker_model_path).to(
            self.device
        )
        self.vis = Visualizer(linewidth=1, mode="rainbow", tracks_leave_trace=-1)

    def load_video_and_mask(self, video_path, mask_path, data_type):
        video = read_video_from_path(video_path)  # shape: (T, H, W, C)
        if video.shape[1:] != (800, 800):
            video = np.array([cv2.resize(frame, (800, 800)) for frame in video])
        video = (
            torch.from_numpy(video).permute(0, 3, 1, 2)[None].float().to(self.device)
        )

        if video.squeeze().shape[0] > 14:
            video = video.squeeze()[:14].unsqueeze(0)
        if video.squeeze().shape[0] < 14:
            while video.squeeze().shape[0] < 14:
                video = torch.cat(
                    [video.squeeze(), video.squeeze()[-1:]], dim=0
                ).unsqueeze(0)

        for mask_path in [mask_path, mask_path.replace(".png", "0.png")]:
            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
                mask = mask if mask.ndim == 2 else mask[..., -1:]

        if (
            self.eval_type == "ground_truth" or self.eval_type == "unsupervised"
        ) and data_type == "pred":
            mask = (mask > 128).astype(np.uint8) * 255

        # Resize the mask
        mask = cv2.resize(mask, (800, 800), interpolation=cv2.INTER_NEAREST)

        mask = torch.from_numpy(mask).float().to(self.device)[None, None]

        # if self.angle == "0.0":
        #     cv2.imwrite(
        #         f"{self.args.output_path}/{self.eval_type}_mask_{data_type}.png",
        #         mask.cpu().numpy()[0, 0],
        #     )

        # Convert to PyTorch tensor
        return video, mask

    def get_tracklets(self, video, mask=None):
        pred_tracks_small, pred_visibility_small = self.cotracker(
            video, grid_size=55, segm_mask=mask
        )
        # shape: (B, T, L, 2) -> rearranged to (B*L, T, 2)
        pred_tracks_small = rearrange(pred_tracks_small, "b t l c -> (b l) t c")
        return pred_tracks_small

    def get_similarity_matrix(self, tracklets1, tracklets2):
        displacements1 = tracklets1[:, 1:] - tracklets1[:, :-1]
        displacements1 = displacements1 / displacements1.norm(dim=-1, keepdim=True)

        displacements2 = tracklets2[:, 1:] - tracklets2[:, :-1]
        displacements2 = displacements2 / displacements2.norm(dim=-1, keepdim=True)
        similarity_matrix = torch.einsum(
            "ntc, mtc -> nmt", displacements1, displacements2
        ).mean(dim=-1)
        return similarity_matrix

    def get_score(self, similarity_matrix):
        if similarity_matrix.shape[0] == 0 or similarity_matrix.shape[1] == 0:
            return {"average_score": 0.0}
        max_similarity, _ = similarity_matrix.max(dim=1)
        non_nan_mask = ~torch.isnan(max_similarity)
        average_score = max_similarity[non_nan_mask].mean()
        return {"average_score": average_score.item()}

    def compute_similarity(self, gt_path, gt_mask_path, pred_path, pred_mask_path):
        gt_video, gt_mask = self.load_video_and_mask(gt_path, gt_mask_path, "gt")
        pred_video, pred_mask = self.load_video_and_mask(
            pred_path, pred_mask_path, "pred"
        )

        gt_tracklets = self.get_tracklets(gt_video, gt_mask)
        pred_tracklets = self.get_tracklets(pred_video, pred_mask)

        # if self.angle == "0.0":
        # if True:
        #     vis_gt_video = self.vis.visualize(
        #         video=gt_video,
        #         tracks=gt_tracklets.permute(1, 0, 2).unsqueeze(0),
        #         save_video=False,
        #     )
        #     vis_pred_video = self.vis.visualize(
        #         video=pred_video,
        #         tracks=pred_tracklets.permute(1, 0, 2).unsqueeze(0),
        #         save_video=False,
        #     )

        #     vis_gt_video = vis_gt_video.squeeze(0).permute(0, 2, 3, 1).cpu().numpy()
        #     vis_pred_video = vis_pred_video.squeeze(0).permute(0, 2, 3, 1).cpu().numpy()

        #     combined_video = np.concatenate(
        #         [vis_gt_video, vis_pred_video], axis=2
        #     ).astype(np.uint8)
        #     os.makedirs(self.args.output_path, exist_ok=True)
        #     imageio.mimwrite(
        #         f"{self.args.output_path}/{self.eval_type}_{self.angle}.gif",
        #         combined_video[9:],
        #         fps=11,
        #         quality=8,
        #         loop=0,
        #     )

        similarity_matrix = self.get_similarity_matrix(gt_tracklets, pred_tracklets)
        return self.get_score(similarity_matrix)


class MotionEvaluator:
    def __init__(
        self, args, pred_dir, gt_dir, pred_angle, gt_angle, metrics_fn, eval_type
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.args = args
        self.psnr = (
            metrics_fn["psnr_fn"].to(self.device)
            if eval_type == "ground_truth" or eval_type == "sanity"
            else None
        )
        self.lpips_fn = (
            metrics_fn["lpips_fn"].to(self.device)
            if eval_type == "ground_truth" or eval_type == "sanity"
            else None
        )
        self.motion_fidelity = metrics_fn["motion_fidelity"]
        self.clip_model, self.preprocess = metrics_fn["clip_model"]

        self.motion_fidelity.angle = gt_angle
        self.motion_fidelity.eval_type = eval_type

        # Construct paths using the angles
        if eval_type == "unsupervised" or eval_type == "baseline":
            self.gt_path = os.path.join(gt_dir, f"{gt_angle}.mp4")
            self.gt_mask_path = os.path.join(gt_dir, "masks", f"{gt_angle}.png")
        else:
            self.gt_path = os.path.join(gt_dir, f"{gt_angle}.mp4")
            self.gt_mask_path = os.path.join(gt_dir, gt_angle, f"0001.png")

        if eval_type == "baseline" or eval_type == "sanity":
            self.pred_path = os.path.join(pred_dir, f"{pred_angle}.mp4")
            self.pred_mask_path = os.path.join(pred_dir, pred_angle, f"0001.png")
        else:
            self.pred_path = os.path.join(pred_dir, f"step_{pred_angle}.mp4")
            self.pred_mask_path = os.path.join(
                pred_dir, "masks", f"mask_{pred_angle}.png"
            )

        # Preprocess frames and store as attributes
        self.gt_tensor, self.pred_tensor = self.preprocess_frames(
            self.gt_path, self.pred_path
        )

        # Store first frame for CLIP-I metric
        # Try multiple angle formats to find the matching first frame
        angle_variants = [
            gt_angle,
            -1 * float(gt_angle),
            float(gt_angle),
            str(float(gt_angle)),
            str(-1 * float(gt_angle)),
        ]
        
        self.first_frame = None
        for angle in angle_variants:
            self.first_frame_path = os.path.join(
                args.target_views, f"{float(angle)}.png"
            )
            if os.path.exists(self.first_frame_path):
                self.first_frame = (
                    torch.from_numpy(cv2.imread(self.first_frame_path))
                    .permute(2, 0, 1)
                    .float()
                    / 255.0
                )
                break

    def preprocess_frames(self, gt_path, pred_path):
        gt_cap = cv2.VideoCapture(gt_path)
        pred_cap = cv2.VideoCapture(pred_path)

        gt_frames = []
        pred_frames = []

        while True:
            gt_ret, gt_frame = gt_cap.read()
            pred_ret, pred_frame = pred_cap.read()
            if not gt_ret and not pred_ret:
                break
            if gt_ret:
                gt_frames.append(
                    cv2.cvtColor(cv2.resize(gt_frame, (800, 800)), cv2.COLOR_BGR2RGB)
                )
            if pred_ret:
                pred_frames.append(
                    cv2.cvtColor(cv2.resize(pred_frame, (800, 800)), cv2.COLOR_BGR2RGB)
                )

        gt_tensor = torch.from_numpy(np.stack(gt_frames)).float() / 255.0
        pred_tensor = torch.from_numpy(np.stack(pred_frames)).float() / 255.0

        # Ensure frames are in [B,C,H,W] format
        gt_tensor = gt_tensor.permute(0, 3, 1, 2).to(self.device)
        pred_tensor = pred_tensor.permute(0, 3, 1, 2).to(self.device)

        # Cut tensors to first 14 frames if longer
        if gt_tensor.shape[0] > 14:
            gt_tensor = gt_tensor[:14]
        if pred_tensor.shape[0] > 14:
            pred_tensor = pred_tensor[:14]

        return gt_tensor, pred_tensor

    def compute_psnr(self):
        psnr_value = self.psnr(self.gt_tensor, self.pred_tensor)
        return psnr_value.item()

    def compute_clip_score(self):
        batch_size = self.gt_tensor.shape[0]
        total_score = 0.0
        scores = []

        # Create debug directory if it doesn't exist
        debug_dir = "clip_score_debug"
        os.makedirs(debug_dir, exist_ok=True)

        for i, (gt_frame, pred_frame) in enumerate(
            zip(self.gt_tensor, self.pred_tensor)
        ):
            # Use the normalize_image helper function for consistency
            gt_image = self.normalize_image(gt_frame)
            pred_image = self.normalize_image(pred_frame)

            # Save images
            gt_image.save(os.path.join(debug_dir, f"gt_frame_{i}.png"))
            pred_image.save(os.path.join(debug_dir, f"pred_frame_{i}.png"))

            gt_image_preprocess = self.preprocess(gt_image).unsqueeze(0).to(self.device)
            pred_image_preprocess = (
                self.preprocess(pred_image).unsqueeze(0).to(self.device)
            )

            gt_features = self.clip_model.encode_image(gt_image_preprocess)
            pred_features = self.clip_model.encode_image(pred_image_preprocess)

            similarity = torch.cosine_similarity(
                gt_features, pred_features, dim=1
            ).item()
            # Map from [-1,1] to [0,1]
            similarity = (similarity + 1) / 2

            total_score += similarity
            scores.append(similarity)

        # Save scores to text file
        with open(os.path.join(debug_dir, "clip_scores.txt"), "w") as f:
            for i, score in enumerate(scores):
                f.write(f"Frame {i}: {score}\n")
            f.write(f"Average score: {total_score / batch_size}")

        return total_score / batch_size

    def normalize_image(self, tensor):
        image_array = tensor.permute(1, 2, 0).cpu().numpy()

        # If values are in [0,1], scale to [0,255]
        if image_array.max() <= 1.0:
            image_array = (image_array * 255).astype(np.uint8)
        else:
            image_array = np.clip(image_array, 0, 255).astype(np.uint8)

        return Image.fromarray(image_array)

    def compute_temporal_clip_score(self):
        batch_size = self.gt_tensor.shape[0]
        if batch_size < 2:
            return 0.0  # Not enough frames to compute a temporal score

        total_score = 0.0
        for i in range(batch_size - 1):
            gt_t = self.normalize_image(self.gt_tensor[i])
            gt_t1 = self.normalize_image(self.gt_tensor[i + 1])
            pred_t = self.normalize_image(self.pred_tensor[i])
            pred_t1 = self.normalize_image(self.pred_tensor[i + 1])

            gt_t_pre = self.preprocess(gt_t).unsqueeze(0).to(self.device)
            gt_t1_pre = self.preprocess(gt_t1).unsqueeze(0).to(self.device)
            pred_t_pre = self.preprocess(pred_t).unsqueeze(0).to(self.device)
            pred_t1_pre = self.preprocess(pred_t1).unsqueeze(0).to(self.device)

            gt_t_feat = self.clip_model.encode_image(gt_t_pre)
            gt_t1_feat = self.clip_model.encode_image(gt_t1_pre)
            pred_t_feat = self.clip_model.encode_image(pred_t_pre)
            pred_t1_feat = self.clip_model.encode_image(pred_t1_pre)

            # Check if consecutive frames are too similar
            if torch.allclose(
                gt_t_feat, gt_t1_feat, rtol=1e-4, atol=1e-4
            ) or torch.allclose(pred_t_feat, pred_t1_feat, rtol=1e-4, atol=1e-4):
                # Skip this pair of frames as they are too similar
                continue
            gt_diff = gt_t1_feat - gt_t_feat
            pred_diff = pred_t1_feat - pred_t_feat

            similarity = torch.cosine_similarity(gt_diff, pred_diff, dim=1).item()
            # Map from [-1,1] to [0,1]
            similarity = (similarity + 1) / 2

            if math.isnan(similarity):
                similarity = 0.0

            total_score += similarity

        return total_score / (batch_size - 1)

    def compute_lpips(self):
        batch_size = self.gt_tensor.shape[0]
        total_score = 0.0
        for gt_frame, pred_frame in zip(self.gt_tensor, self.pred_tensor):
            score = self.lpips_fn(gt_frame.unsqueeze(0), pred_frame.unsqueeze(0))
            total_score += score.item()

        return total_score / batch_size

    def compute_motion_fidelity(self):
        return self.motion_fidelity.compute_similarity(
            self.gt_path, self.gt_mask_path, self.pred_path, self.pred_mask_path
        )

    def compute_I_clip(self):
        batch_size = self.pred_tensor.shape[0]
        if batch_size < 2:
            return 0.0  # Not enough frames to compute a temporal score
        
        # Check if first_frame is available
        if not hasattr(self, 'first_frame') or self.first_frame is None:
            # Try to find first frame using the angle
            angle = self.motion_fidelity.angle
            angle_variants = [
                angle,
                -1 * float(angle),
                float(angle),
            ]
            
            for angle_var in angle_variants:
                first_frame_path = os.path.join(
                    self.args.target_views, f"{float(angle_var)}.png"
                )
                if os.path.exists(first_frame_path):
                    self.first_frame = (
                        torch.from_numpy(cv2.imread(first_frame_path))
                        .permute(2, 0, 1)
                        .float()
                        / 255.0
                    )
                    break
            
            if self.first_frame is None:
                return 0.0  # Cannot compute CLIP-I without first frame

        total_score = 0.0

        first_frame_norm = self.normalize_image(self.first_frame)
        first_frame_pre = self.preprocess(first_frame_norm).unsqueeze(0).to(self.device)
        first_frame_feat = self.clip_model.encode_image(first_frame_pre)

        for i in range(batch_size - 1):
            pred_t = self.normalize_image(self.pred_tensor[i])
            pred_t_pre = self.preprocess(pred_t).unsqueeze(0).to(self.device)
            pred_t_feat = self.clip_model.encode_image(pred_t_pre)

            similarity = torch.cosine_similarity(
                first_frame_feat, pred_t_feat, dim=1
            ).item()
            # Map from [-1,1] to [0,1]
            similarity = (similarity + 1) / 2

            if math.isnan(similarity):
                similarity = 0.0

            total_score += similarity

        return total_score / (batch_size - 1)


def evaluate_angle(evaluator, eval_type):
    if eval_type == "ground_truth" or eval_type == "sanity":
        psnr = evaluator.compute_psnr()
        lpips_ = evaluator.compute_lpips()
        clip_score = evaluator.compute_clip_score()
    else:
        psnr = None
        lpips_ = None
        clip_score = None
    temp_clip = evaluator.compute_temporal_clip_score()
    motion_fid = evaluator.compute_motion_fidelity()["average_score"]
    clip_i = evaluator.compute_I_clip()  # Add CLIP-I computation

    metrics = {
        "psnr": psnr,
        "lpips": lpips_,
        "clip_score": clip_score,
        "temporal_clip_score": temp_clip,
        "motion_fidelity": motion_fid,
        "clip_i": clip_i,  # Add CLIP-I to metrics
    }

    return metrics


def evaluate_all_angles(args, pred_dir, gt_dir, angles, eval_type, metrics_fn):
    if eval_type in ["ground_truth", "sanity"]:
        avg_metrics = {
            "psnr": 0.0,
            "lpips": 0.0,
            "clip_score": 0.0,
            "temporal_clip_score": 0.0,
            "motion_fidelity": 0.0,
            "clip_i": 0.0,  # Add CLIP-I
        }
    else:
        avg_metrics = {
            "temporal_clip_score": 0.0,
            "motion_fidelity": 0.0,
            "clip_i": 0.0,  # Add CLIP-I
        }

    data = []

    messages = {
        "ground_truth": "Evaluating prediction against ground truth.",
        "baseline": "Evaluating ground truth against source.",
        "unsupervised": "Evaluating prediction against source",
        "sanity": "Evaluating ground truth against ground truth.",
    }

    for angle in tqdm(angles, desc=messages[eval_type], total=len(angles)):
        evaluator = MotionEvaluator(
            args, pred_dir, gt_dir, angle[0], angle[1], metrics_fn, eval_type=eval_type
        )
        metrics = evaluate_angle(evaluator, eval_type)

        # Accumulate metrics
        for key in avg_metrics:
            if metrics[key] is not None:
                avg_metrics[key] += metrics[key]

        # Create row data
        row = {"angle": float(angle[0]) % 360}
        for key in avg_metrics:
            row[key] = metrics[key]
        data.append(row)

    # Compute averages
    for key in avg_metrics:
        avg_metrics[key] /= len(angles)

    # Add average row
    avg_row = {"angle": "AVERAGE"}
    for key in avg_metrics:
        avg_row[key] = avg_metrics[key]
    data.append(avg_row)

    return data


def get_angle_pairs(list_a, list_b):
    """Find equivalent angle pairs between two lists (modulo 360)."""
    if list_a is None or list_b is None:
        return []
    
    # Create {normalized_angle: original_angle} mappings
    norm_a = {float(ang) % 360: ang for ang in list_a}
    norm_b = {float(ang) % 360: ang for ang in list_b}

    # Find matching normalized angles
    return [(norm_a[k], norm_b[k]) for k in norm_a if k in norm_b]


def evaluate(cotracker_model_path, args):
    print(f"\n\n\nInitializing evaluator with CoTracker model: {cotracker_model_path}")

    psnr = torchmetrics.PeakSignalNoiseRatio()
    lpips_fn = lpips.LPIPS(net="alex")
    cotracker = MotionFidelity(args, cotracker_model_path)
    clip_model = clip.load(
        "ViT-B/32", device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    metrics_fn = {
        "psnr_fn": psnr,
        "lpips_fn": lpips_fn,
        "motion_fidelity": cotracker,
        "clip_model": clip_model,
    }

    # Determine angles to evaluate
    pred_angles = [
        angle.replace("step_", "").replace(".mp4", "")
        for angle in os.listdir(args.pred_dir)
        if angle.endswith("mp4")
    ]
    source_angles = [
        angle
        for angle in os.listdir(args.source_dir)
        if (os.path.isdir(os.path.join(args.source_dir, angle)) and angle != "masks")
    ]
    gt_angles = None
    if args.gt_dir and os.path.exists(args.gt_dir):
        gt_angles = [
            angle
            for angle in os.listdir(args.gt_dir)
            if os.path.isdir(os.path.join(args.gt_dir, angle))
        ]

    pred_gt_pairs = get_angle_pairs(pred_angles, gt_angles) if args.gt_dir else None
    pred_source_pairs = get_angle_pairs(pred_angles, source_angles)
    gt_source_pairs = get_angle_pairs(gt_angles, source_angles) if args.gt_dir else None
    gt_gt_pairs = get_angle_pairs(gt_angles, gt_angles) if args.gt_dir else None
    print(f"\nPred-GT pairs: {pred_gt_pairs}")
    print(f"\nPred-Source pairs: {pred_source_pairs}")
    print(f"\nGT-Source pairs: {gt_source_pairs}")
    print(f"\nGT-GT pairs: {gt_gt_pairs}")

    columns_mapping = {
        "ground_truth": [
            "angle",
            "psnr",
            "lpips",
            "clip_score",
            "temporal_clip_score",
            "motion_fidelity",
            "clip_i",  # Add CLIP-I
        ],
        "sanity": [
            "angle",
            "psnr",
            "lpips",
            "clip_score",
            "temporal_clip_score",
            "motion_fidelity",
            "clip_i",  # Add CLIP-I
        ],
        "baseline": [
            "angle",
            "temporal_clip_score",
            "motion_fidelity",
            "clip_i",  # Add CLIP-I
        ],
        "unsupervised": [
            "angle",
            "temporal_clip_score",
            "motion_fidelity",
            "clip_i",  # Add CLIP-I
        ],
    }

    output_files = {
        "ground_truth": "ground_truth_metrics.csv",
        "sanity": "sanity_metrics.csv",
        "baseline": "baseline_metrics.csv",
        "unsupervised": "unsupervised_metrics.csv",
    }

    if args.gt_dir and pred_gt_pairs:
        # Supervised evaluation (pred vs GT)
        gt_data = evaluate_all_angles(
            args=args,
            pred_dir=args.pred_dir,
            gt_dir=args.gt_dir,
            angles=pred_gt_pairs,
            eval_type="ground_truth",
            metrics_fn=metrics_fn,
        )
        df = pd.DataFrame(gt_data)[columns_mapping["ground_truth"]]
        df.to_csv(
            os.path.join(args.output_path, output_files["ground_truth"]), index=False
        )
        print(
            "GT data saved to: ",
            os.path.join(args.output_path, output_files["ground_truth"]),
        )

        # # Baseline evaluation (GT vs source)
        # baseline_data = evaluate_all_angles(
        #     args=args,
        #     pred_dir=args.gt_dir,
        #     gt_dir=args.source_dir,
        #     angles=gt_source_pairs,
        #     eval_type="baseline",
        #     metrics_fn=metrics_fn,
        # )
        # df = pd.DataFrame(baseline_data)[columns_mapping["baseline"]]
        # df.to_csv(os.path.join(args.output_path, output_files["baseline"]), index=False)

        # # Sanity check (GT vs GT)
        # sanity_data = evaluate_all_angles(
        #     args=args,
        #     pred_dir=args.gt_dir,
        #     gt_dir=args.gt_dir,
        #     angles=gt_gt_pairs,
        #     eval_type="sanity",
        #     metrics_fn=metrics_fn,
        # )
        # df = pd.DataFrame(sanity_data)[columns_mapping["sanity"]]
        # df.to_csv(os.path.join(args.output_path, output_files["sanity"]), index=False)

    # Unsupervised evaluation (pred vs source)
    unsupervised_data = evaluate_all_angles(
        args=args,
        pred_dir=args.pred_dir,
        gt_dir=args.source_dir,
        angles=pred_source_pairs,
        eval_type="unsupervised",
        metrics_fn=metrics_fn,
    )
    df = pd.DataFrame(unsupervised_data)[columns_mapping["unsupervised"]]
    df.to_csv(os.path.join(args.output_path, output_files["unsupervised"]), index=False)
    print(
        "Unsupervised data saved to: ",
        os.path.join(args.output_path, output_files["unsupervised"]),
    )


def seed_everything(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


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
        "--gt_dir",
        default=None,
        required=False,
        type=str,
        help="Directory containing ground truth videos (optional)",
    )
    parser.add_argument(
        "--pred_dir",
        required=True,
        type=str,
        help="Directory containing predicted videos",
    )
    parser.add_argument(
        "--source_dir",
        default=None,
        required=False,
        type=str,
        help="Directory containing source videos (optional, needed for motion metrics)",
    )
    parser.add_argument(
        "--target_views",
        type=str,
        required=True,
        help="Directory containing target views (REQUIRED for CLIP-I metric). Should contain PNG files named by angle (e.g., '0.0.png', '22.5.png', etc.). Typically: ./Outputs/<motion>/<target>/<run_name>/FirstFrames",
    )
    parser.add_argument(
        "--prompt",
        default="",
        type=str,
        help="Prompt (deprecated, kept for compatibility)",
    )

    args = parser.parse_args(argv)

    # Validate target_views directory
    if not args.target_views:
        print("\n" + "="*70)
        print("ERROR: --target_views is required but was not provided.")
        print("="*70)
        print("\nThe target_views directory should contain first frame images")
        print("generated by the render_target.py script.")
        print("\nExample path: ./Outputs/horse_run/beagle/example_run_20250101_120000/FirstFrames")
        print("\nPlease run render_target.py first to generate the first frames, or")
        print("provide the path to an existing FirstFrames directory.")
        print("="*70 + "\n")
        exit(1)
    
    if not os.path.exists(args.target_views):
        print("\n" + "="*70)
        print(f"ERROR: Target views directory does not exist: {args.target_views}")
        print("="*70)
        print("\nPlease check that the path is correct.")
        print("The directory should contain PNG files named by angle (e.g., '0.0.png', '22.5.png').")
        print("\nIf you haven't generated first frames yet, run:")
        print("  python render_target.py --config configs/MT3D.yaml \\")
        print("    --output_path example_run_20250101_120000 \\")
        print("    --target_path assets/targets/beagle.ply \\")
        print("    --source_path assets/motions/horse_run")
        print("="*70 + "\n")
        exit(1)
    
    # Check if directory has any PNG files
    png_files = [f for f in os.listdir(args.target_views) if f.endswith('.png')]
    if not png_files:
        print("\n" + "="*70)
        print(f"WARNING: No PNG files found in target_views directory: {args.target_views}")
        print("="*70)
        print("\nThe directory should contain PNG files named by angle.")
        print("CLIP-I metric will not be computed without first frame images.")
        print("="*70 + "\n")
    else:
        print(f"Found {len(png_files)} PNG files in target_views directory.")

    # Create output directory if it doesn't exist
    os.makedirs(args.output_path, exist_ok=True)

    seed_everything(0)

    # Load YAML configuration
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    config["config"] = args.config
    config["output_path"] = args.output_path
    config["gt_dir"] = args.gt_dir if args.gt_dir else None
    config["pred_dir"] = args.pred_dir
    config["source_dir"] = args.source_dir
    config["target_views"] = args.target_views
    config["stage"] = 5

    print("output path: ", args.output_path)
    print("gt_dir: ", args.gt_dir if args.gt_dir else "None (unsupervised evaluation)")
    print("pred_dir: ", args.pred_dir)
    print("source_dir: ", args.source_dir)
    print("target_views: ", args.target_views)

    args = argparse.Namespace(**config)

    evaluate(
        cotracker_model_path="co-tracker/checkpoints/scaled_offline.pth",
        args=args,
    )


def run_evaluate_output(
    config: str,
    output_path: str,
    pred_dir: str,
    target_views: str,
    source_dir: str | None = None,
    prompt: str = "",
) -> None:
    argv = [
        "--config",
        config,
        "--output_path",
        output_path,
        "--pred_dir",
        pred_dir,
        "--target_views",
        target_views,
        "--prompt",
        prompt,
    ]
    if source_dir:
        argv += ["--source_dir", source_dir]
    main(argv)


if __name__ == "__main__":
    main()
