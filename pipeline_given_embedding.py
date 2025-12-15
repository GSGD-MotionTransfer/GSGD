"""
End-to-end helper to run MotionTransfer3D when you already have a motion
embedding. It stitches together:
1) render_target.py   -> renders first frames for the target
2) reenact_target.py  -> generates supervision gifs using the motion embedding
3) consolidate_4d.py  -> reconstructs the final motion-transferred 3DGS

Example:
    python pipeline_given_embedding.py \\
        --config configs/MT3D.yaml \\
        --target_path data/web_crawled/targets/human_like_objects/basic_crop_top_and_pants.ply \\
        --motion_embedding_path Outputs/MotionEmbeddings/Human_BreakdanceReady \\
        --output_path quickstart_breakdance_on_basicpants
"""

import argparse
import datetime
import os
from saving import (
    get_base_output_path,
    get_first_frames_output_path,
    get_supervision_output_path,
    infer_source_name,
)
from render_target import run_render_target
from reenact_target import run_reenact_target
from consolidate_4d import run_consolidate_4d


def main():
    parser = argparse.ArgumentParser(
        description="Run the MotionTransfer3D pipeline with a pre-trained motion embedding."
    )
    parser.add_argument(
        "--config",
        default="configs/MT3D.yaml",
        help="Path to config file shared by all stages.",
    )
    parser.add_argument(
        "--target_path",
        required=True,
        help="Target 3D Gaussian Splatting .ply file.",
    )
    parser.add_argument(
        "--motion_embedding_path",
        required=True,
        help="Directory with per-angle motion_embedding_*.pt files or a single embedding .pt.",
    )
    parser.add_argument(
        "--motion_epoch",
        default="3000",
        help="Checkpoint epoch to load for the motion embedding.",
    )
    parser.add_argument(
        "--output_path",
        default=None,
        help="Run name. Default: given_embedding_<timestamp>.",
    )
    parser.add_argument(
        "--run_evaluation",
        action="store_true",
        help="If set, run evaluate_output.py after reconstruction.",
    )
    parser.add_argument(
        "--source_dir",
        default=None,
        help="Optional source videos directory for evaluation (if available).",
    )
    args = parser.parse_args()

    print("\n" + "="*80)
    print("[pipeline] Starting MotionTransfer3D pipeline")
    print("="*80)
    
    run_name = args.output_path or f"given_embedding_{datetime.datetime.now():%Y%m%d_%H%M%S}"
    source_name = infer_source_name(args.motion_embedding_path)
    
    print(f"[pipeline] Run name: {run_name}")
    print(f"[pipeline] Source name: {source_name}")
    print(f"[pipeline] Target path: {args.target_path}")
    print(f"[pipeline] Motion embedding path: {args.motion_embedding_path}")
    print(f"[pipeline] Motion epoch: {args.motion_epoch}")

    synthetic_source_path = os.path.join("Outputs", "MotionEmbeddings", source_name)

    first_frames_dir = get_first_frames_output_path(
        synthetic_source_path, args.target_path, run_name
    )
    supervision_dir = get_supervision_output_path(
        synthetic_source_path, args.target_path, run_name
    )
    base_output_dir = get_base_output_path(synthetic_source_path, args.target_path, run_name)
    
    print(f"[pipeline] First frames directory: {first_frames_dir}")
    print(f"[pipeline] Supervision directory: {supervision_dir}")
    print(f"[pipeline] Base output directory: {base_output_dir}")

    # Stage 1: render target first frames
    print("\n" + "-"*80)
    print("[pipeline] Stage 1: Rendering target first frames")
    print("-"*80)
    run_render_target(
        config=args.config,
        output_path=run_name,
        target_path=args.target_path,
        motion_embedding_path=args.motion_embedding_path,
    )
    print("[pipeline] Stage 1 complete: First frames rendered")

    # Stage 2: reenact using the provided motion embedding
    print("\n" + "-"*80)
    print("[pipeline] Stage 2: Reenacting using motion embedding")
    print("-"*80)
    print(f"[pipeline] Using first frames from: {first_frames_dir}")
    run_reenact_target(
        config=args.config,
        output_path=run_name,
        first_frames_path=first_frames_dir,
        motion_embedding_path=args.motion_embedding_path,
        motion_epoch=str(args.motion_epoch),
    )
    print("[pipeline] Stage 2 complete: Supervision videos generated")

    eval_output = None

    # Stage 3: reconstruct final motion-transferred 3DGS
    print("\n" + "-"*80)
    print("[pipeline] Stage 3: Reconstructing final motion-transferred 3DGS")
    print("-"*80)
    print(f"[pipeline] Using supervision from: {supervision_dir}")
    print(f"[pipeline] Using first frames from: {first_frames_dir}")
    run_consolidate_4d(
        config=args.config,
        target_path=args.target_path,
        specify_supervision=supervision_dir,
        motion_embedding_path=args.motion_embedding_path,
        first_frames_path=first_frames_dir,
        motion_epoch=str(args.motion_epoch),
    )
    print("[pipeline] Stage 3 complete: 3DGS reconstruction finished")

    print("\n" + "="*80)
    print("[pipeline] Pipeline Complete!")
    print("="*80)
    print(f"  First frames:   {first_frames_dir}\n"
        f"  Supervision:    {supervision_dir}\n"
        f"  Final output:   {os.path.join(base_output_dir, 'Final_Output/final_output')}\n"
        + (f"  Evaluation:    {eval_output}\n" if args.run_evaluation else "  Evaluation:    skipped\n"))


if __name__ == "__main__":
    main()

