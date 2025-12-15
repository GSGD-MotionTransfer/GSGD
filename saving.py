"""
Saving and loading utilities for MotionTransfer3D pipeline.
Replaces DAL class with pure functions.
"""
import os
import glob
from typing import Optional
from PIL import Image
import imageio
import numpy as np
import torch
import cv2


def get_available_views(directory_path: str, exclude_dirs: Optional[list[str]] = None) -> list[str]:
    """
    Extract available viewing angles from a directory with angle-named subdirectories.
    
    Args:
        directory_path: Path to directory with angle-named subdirectories (e.g., source_path or motion_embedding_path)
        exclude_dirs: List of directory names to exclude (e.g., ["masks"])
    
    Returns:
        List of sorted angle strings (e.g., ["0.0", "22.5", "-112.5", ...])
    
    Raises:
        ValueError: If directory_path is not a valid directory or no angles are found
    """
    if exclude_dirs is None:
        exclude_dirs = ["masks"]
    
    if not os.path.isdir(directory_path):
        raise ValueError(f"Directory path does not exist or is not a directory: {directory_path}")
    
    angles = []
    for entry in os.listdir(directory_path):
        full = os.path.join(directory_path, entry)
        if os.path.isdir(full) and entry not in exclude_dirs:
            try:
                float(entry)  # Verify it's a numeric angle
                angles.append(entry)
            except ValueError:
                continue
    
    if not angles:
        raise ValueError(
            f"No valid angle directories found in {directory_path}. "
            f"Expected subdirectories with numeric names (e.g., '0.0', '22.5', '-112.5')."
        )
    
    return sorted(angles, key=lambda x: float(x))


def infer_source_name(path: str) -> str:
    """
    Infer source_name from a path (source motion directory or motion embedding directory).
    
    Args:
        path: Path to source motion directory or motion embedding directory
    
    Returns:
        Source name extracted from the basename of the provided path
    """
    return os.path.basename(path.rstrip("/"))


def _extract_source_and_target_names(source_path=None, target_path=None):
    """
    Extract source_name and target_name from paths.
    
    Args:
        source_path: Path to source motion directory
        target_path: Path to target .ply file
    
    Returns:
        Tuple of (source_name, target_name)
    """
    source_name = None
    target_name = None
    
    if source_path:
        source_name = os.path.basename(source_path.rstrip('/'))
    if target_path:
        target_name = os.path.basename(target_path).replace('.ply', '')
    
    return source_name, target_name


def get_base_output_path(source_path, target_path, output_path):
    """
    Construct base output path.
    
    Args:
        source_path: Path to source motion directory
        target_path: Path to target .ply file
        output_path: Base output path (may be base name or full path)
    
    Returns:
        Path: ./Outputs/{source_name}/{target_name}/{output_path}/
    """
    source_name, target_name = _extract_source_and_target_names(source_path, target_path)
    
    if not source_name or not target_name:
        raise ValueError(f"Could not extract source_name and target_name from source_path={source_path}, target_path={target_path}")
    
    return os.path.join("./Outputs", source_name, target_name, output_path)


def get_motion_embeddings_output_path(output_path: str) -> str:
    """
    Construct path for motion embeddings output.
    
    Args:
        output_path: Base output path (e.g., "example_run_20250101_120000")
    
    Returns:
        Path: ./Outputs/MotionEmbeddings/{output_path}/
    """
    return os.path.join("./Outputs", "MotionEmbeddings", output_path)


def get_first_frames_output_path(source_path: str, target_path: str, output_path: str) -> str:
    """
    Construct path for first frames output.
    
    Args:
        source_path: Path to source motion directory
        target_path: Path to target .ply file
        output_path: Base output path
    
    Returns:
        Path: ./Outputs/{source_name}/{target_name}/{output_path}/FirstFrames/
    """
    base_path = get_base_output_path(source_path, target_path, output_path)
    return os.path.join(base_path, "FirstFrames")


def get_supervision_output_path(source_path: str, target_path: str, output_path: str) -> str:
    """
    Construct path for supervision videos output.
    
    Args:
        source_path: Path to source motion directory
        target_path: Path to target .ply file
        output_path: Base output path
    
    Returns:
        Path: ./Outputs/{source_name}/{target_name}/{output_path}/Supervision/
    """
    base_path = get_base_output_path(source_path, target_path, output_path)
    return os.path.join(base_path, "Supervision")


def extract_names_from_path(path: str) -> tuple[str, str]:
    """
    Extract source_name and target_name from output path.
    
    Args:
        path: Path like ./Outputs/{source_name}/{target_name}/...
    
    Returns:
        Tuple of (source_name, target_name)
    """
    # Remove ./ prefix if present
    if path.startswith("./"):
        path = path[2:]
    
    # Split by /
    parts = path.split("/")
    
    # Find "Outputs" and extract next two components
    if "Outputs" in parts:
        idx = parts.index("Outputs")
        if idx + 2 < len(parts):
            source_name = parts[idx + 1]
            target_name = parts[idx + 2]
            return source_name, target_name
    
    raise ValueError(f"Could not extract source_name and target_name from path: {path}")


def load_motion_embedding(motion_embedding_path: str, angle: Optional[str] = None, step: int | str = -1) -> tuple[str, torch.Tensor]:
    """
    Load motion embedding checkpoint.
    
    Args:
        motion_embedding_path: Path to embedding directory or file
        angle: Angle string if per-angle embeddings (optional)
        step: Checkpoint step (-1 for latest) (can be int or string)
    
    Returns:
        Tuple of (embedding_path, embedding_tensor)
    """
    # Convert step to int if string
    if isinstance(step, str):
        if step == "-1":
            step = -1
        else:
            try:
                step = int(step)
            except ValueError:
                # If step is not a number (e.g., "initial"), use it as-is
                pass
    
    # Handle file path (global embedding)
    if os.path.isfile(motion_embedding_path):
        return motion_embedding_path, torch.load(motion_embedding_path)
    
    # Handle directory path
    motion_embeddings_checkpoints_dir = motion_embedding_path
    if angle is not None:
        motion_embeddings_checkpoints_dir = os.path.join(motion_embeddings_checkpoints_dir, str(angle))
    
    if step == -1 or (isinstance(step, int) and step > 0):
        # Find available checkpoints
        available_files = [
            f for f in os.listdir(motion_embeddings_checkpoints_dir)
            if f.endswith(".pt")
        ]
        
        if not available_files:
            raise ValueError(f"No checkpoint found in {motion_embeddings_checkpoints_dir}")
        
        # Extract step numbers
        embeddings = []
        for f in available_files:
            if "motion_embedding_" in f:
                step_str = f.split("motion_embedding_")[-1].replace(".pt", "")
                if step_str.isdigit():
                    embeddings.append(int(step_str))
                elif step_str in ["initial", "final"]:
                    embeddings.append(step_str)
        
        if not embeddings:
            raise ValueError(f"No valid checkpoint found in {motion_embeddings_checkpoints_dir}")
        
        # If step is specified and exists, use it; otherwise use latest
        if isinstance(step, int) and step > 0:
            if step in embeddings:
                use_step = step
            else:
                # Use latest if specified step doesn't exist
                numeric_steps = [s for s in embeddings if isinstance(s, int)]
                if numeric_steps:
                    use_step = max(numeric_steps)
                    print(f"Warning: Step {step} not found, using latest step {use_step}")
                else:
                    use_step = max(embeddings) if isinstance(max(embeddings), int) else embeddings[-1]
        else:
            # Use latest
            numeric_steps = [s for s in embeddings if isinstance(s, int)]
            if numeric_steps:
                use_step = max(numeric_steps)
            else:
                use_step = embeddings[-1]
        
        step = use_step
    
    motion_embedding_path = os.path.join(
        motion_embeddings_checkpoints_dir, f"motion_embedding_{step}.pt"
    )
    
    if not os.path.exists(motion_embedding_path):
        raise FileNotFoundError(f"Motion embedding not found: {motion_embedding_path}")
    
    return motion_embedding_path, torch.load(motion_embedding_path)


def save_target_image(image_array: np.ndarray, output_dir: str, angle: str) -> None:
    """
    Save rendered target image.
    
    Args:
        image_array: Image as numpy array (BGR format)
        output_dir: Output directory
        angle: Angle string
    """
    _ensure_directory(output_dir)
    target_img_path = os.path.join(output_dir, f"{angle}.png")
    cv2.imwrite(target_img_path, image_array)


def load_multi_angle_source_videos(source_path: str, available_views: list[str]):
    """
    Load multiview videos from source directory.
    
    Args:
        source_path: Path to source directory
        available_views: List of angle strings
    
    Returns:
        Tuple of (videos, available_views) where videos is list of lists of PIL Images
    """
    source_videos_paths = []
    for angle in available_views:
        # Try both {angle}.mp4 and {angle}/*.mp4
        angle_path = os.path.join(source_path, f"{angle}.mp4")
        if os.path.exists(angle_path):
            source_videos_paths.append(angle_path)
        else:
            # Try subdirectory
            angle_dir = os.path.join(source_path, str(angle))
            if os.path.isdir(angle_dir):
                mp4_files = glob.glob(os.path.join(angle_dir, "*.mp4"))
                if mp4_files:
                    source_videos_paths.append(mp4_files[0])
                else:
                    raise FileNotFoundError(f"No .mp4 file found for angle {angle} in {angle_dir}")
            else:
                raise FileNotFoundError(f"Video not found for angle {angle} at {angle_path}")
    
    source_videos = [
        imageio.get_reader(source_video_path)
        for source_video_path in source_videos_paths
    ]
    source_videos = [
        [Image.fromarray(f) for f in reader] for reader in source_videos
    ]
    return source_videos, available_views


def load_multi_angle_first_frames(first_frames_path, available_views):
    """
    Load first frames for validation.
    
    Args:
        first_frames_path: Path to first frames directory
        available_views: List of angle strings
    
    Returns:
        Tuple of (first_frames, available_views, figure_names)
        where first_frames is list of lists (one list per angle, each containing PIL Images)
    """
    first_frames = []
    for angle in available_views:
        frame_path = os.path.join(first_frames_path, f"{angle}.png")
        if os.path.exists(frame_path):
            # Wrap in list to match expected format (list of lists)
            first_frames.append([Image.open(frame_path)])
        else:
            raise FileNotFoundError(f"First frame not found for angle {angle} at {frame_path}")
    
    # Return empty figure_names for now (can be extended if needed)
    return first_frames, available_views, []


def _convert_frames_to_pil(frames):
    """Convert frames to PIL Images."""
    return [
        Image.fromarray(frame) if isinstance(frame, np.ndarray) else frame
        for frame in frames
    ]


def _ensure_directory(path):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def save_video_as_gif(frames, output_path, fps=10, duration_ms=500):
    """
    Save video frames as GIF (low-level function).
    
    Args:
        frames: List of frames (numpy arrays or PIL Images)
        output_path: Output path WITHOUT .gif extension (function adds it)
        fps: Frames per second (for metadata)
        duration_ms: Duration per frame in milliseconds
    """
    pil_frames = _convert_frames_to_pil(frames)
    _ensure_directory(os.path.dirname(output_path) if os.path.dirname(output_path) else ".")
    
    pil_frames[0].save(
        f"{output_path}.gif",
        format="GIF",
        append_images=pil_frames[1:],
        save_all=True,
        duration=duration_ms,
        loop=0,
    )


def save_checkpoint_video(video_frames, output_dir, global_step, figure_name=None, angle=None, fps=10):
    """
    Save checkpoint video during training (high-level function).
    
    Args:
        video_frames: List of video frames
        output_dir: Base output directory
        global_step: Step identifier (str/int)
        figure_name: Figure name (optional)
        angle: Angle string (optional)
        fps: Frames per second
    """
    if angle is not None:
        video_output_path = os.path.join(output_dir, str(angle))
    else:
        video_output_path = output_dir
    
    if figure_name:
        save_path = os.path.join(video_output_path, f"{global_step}_{figure_name}")
    else:
        save_path = os.path.join(video_output_path, str(global_step))
    
    _ensure_directory(video_output_path)
    save_video_as_gif(
        [np.array(img) for img in video_frames],
        save_path,
        fps=fps,
    )


def save_motion_embedding(image_embds_wrap, output_dir, global_step, angle=None):
    """
    Save motion embedding or MLP checkpoint.
    
    Args:
        image_embds_wrap: ImageEmbeddingWrapper object
        output_dir: Output directory
        global_step: Step identifier (str/int)
        angle: Angle string for per-angle embeddings (optional)
    """
    _ensure_directory(output_dir)
    
    # Save MLP if present
    if hasattr(image_embds_wrap, "MLP"):
        mlp_save_path = os.path.join(
            output_dir, f"model_checkpoint_epoch_{global_step}.pth"
        )
        torch.save(image_embds_wrap.MLP.state_dict(), mlp_save_path)
    else:
        # Prepare directory for motion embedding
        embedding_dir = output_dir
        if angle is not None:
            embedding_dir = os.path.join(embedding_dir, str(angle))
            _ensure_directory(embedding_dir)
        
        # Save motion embedding
        save_path = os.path.join(
            embedding_dir, f"motion_embedding_{global_step}.pt"
        )
        torch.save(image_embds_wrap.get_parameters()[0], save_path)
        print(f"Saved motion embedding to {save_path}")


def save_angular_embeddings(embeddings_dict, output_dir, global_step):
    """
    Save multiple angular embeddings.
    
    Args:
        embeddings_dict: Dict mapping angle strings to embedding tensors
        output_dir: Output directory
        global_step: Step identifier (str/int)
    """
    for angle in embeddings_dict:
        embedding = embeddings_dict[angle]
        embedding_dir = os.path.join(output_dir, str(angle))
        _ensure_directory(embedding_dir)
        save_path = os.path.join(
            embedding_dir, f"motion_embedding_{global_step}.pt"
        )
        torch.save(embedding, save_path)


def load_reconstruction_images(supervision_path, num_frames):
    """
    Load supervision images for reconstruction.
    
    Args:
        supervision_path: Path to supervision directory
        num_frames: Expected number of frames
    
    Returns:
        Tuple of (image_list, selected_views)
        where image_list is list of lists of image paths, one list per view
    """
    views_list = [
        name.split("_")[-1].replace(".gif", "")
        for name in os.listdir(supervision_path)
        if name.endswith("gif")
    ]
    views_list = sorted(views_list, key=float)
    n_views_in_dir = len(views_list)
    print(f"Found {n_views_in_dir} views: {views_list}")
    
    selected_views = views_list
    print(f"Selected {len(selected_views)} views: {str(selected_views)}")
    
    # Extract frames from GIFs if needed
    from tqdm import tqdm
    for v in tqdm(selected_views):
        angle_dir = os.path.join(supervision_path, v)
        png_files = glob.glob(os.path.join(angle_dir, "*.png"))
        if len(png_files) != num_frames:
            gif_path = os.path.join(supervision_path, f"inference_{v}.gif")
            if os.path.exists(gif_path):
                with Image.open(gif_path) as gif:
                    for frame_number in range(gif.n_frames):
                        _ensure_directory(angle_dir)
                        gif.seek(frame_number)
                        frame = gif.copy()
                        frame = frame.resize((800, 800))
                        frame.save(os.path.join(angle_dir, f"{frame_number:04d}.png"))
    
    image_list = [
        glob.glob(os.path.join(supervision_path, v, "*.png")) for v in selected_views
    ]
    
    # Sort them
    image_list = [
        sorted(image_list[i], key=lambda x: int(os.path.basename(x).split(".")[0]))
        for i in range(len(selected_views))
    ]
    
    return image_list, selected_views


def load_and_preprocess_input(file_path, W, H, ref_size):
    """
    Load and preprocess input image for reconstruction.
    
    Args:
        file_path: Path to image file
        W: Target width
        H: Target height
        ref_size: Reference size for interpolation
    
    Returns:
        Tuple of (input_mask, input_img) as torch tensors
    """
    import torch.nn.functional as F
    
    print(f"[INFO] load image from {file_path}...")
    img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    
    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    
    input_mask = img[..., :1]
    input_img = img[..., :3]
    input_img = input_img[..., ::-1].copy()  # bgr to rgb
    
    # to torch tensors
    input_img_torch = torch.from_numpy(input_img).permute(2, 0, 1).unsqueeze(0)
    input_img = F.interpolate(
        input_img_torch,
        (ref_size, ref_size),
        mode="bilinear",
        align_corners=False,
    )
    input_mask_torch = torch.from_numpy(input_mask).permute(2, 0, 1).unsqueeze(0)
    input_mask = F.interpolate(
        input_mask_torch,
        (ref_size, ref_size),
        mode="bilinear",
        align_corners=False,
    )
    
    return input_mask, input_img


def get_reconstruction_output_paths(source_path, target_path, output_path):
    """
    Construct paths for reconstruction outputs.
    
    Args:
        source_path: Path to source motion directory
        target_path: Path to target .ply file
        output_path: Base output path (may be full path or base name)
    
    Returns:
        Tuple of (final_output_path, checkpoints_path)
    """
    # Check if output_path is already a full path
    if output_path.startswith("./Outputs/") or output_path.startswith("Outputs/"):
        # Already full path, use directly
        if output_path.startswith("./"):
            base_path = output_path
        else:
            base_path = "./" + output_path
    else:
        # Base name, construct full path
        base_path = get_base_output_path(source_path, target_path, output_path)
    
    final_output_path = os.path.join(base_path, "Final_Output")
    checkpoints_path = os.path.join(final_output_path, "Checkpoints")
    
    return final_output_path, checkpoints_path


def save_visualization_video(video_frames, output_dir, vis_type, step, fps=10, save_mp4=False):
    """
    Save visualization video/GIF (high-level function).
    
    Args:
        video_frames: List of video frames
        output_dir: Base output directory
        vis_type: Visualization type (e.g., "final_output", "render")
        step: Step identifier (int/str)
        fps: Frames per second
        save_mp4: Also save as MP4
    """
    import imageio
    import imageio.v3 as iio
    
    video_save_path = os.path.join(output_dir, vis_type, f"step_{step}.gif")
    _ensure_directory(os.path.dirname(video_save_path))
    
    imageio.mimwrite(
        video_save_path,
        video_frames,
        fps=fps,
        quality=8,
        macro_block_size=1,
        loop=0,
    )
    
    if save_mp4 or vis_type == "final_output":
        mp4_save_path = video_save_path.replace(".gif", ".mp4")
        iio.imwrite(
            mp4_save_path,
            video_frames,
            fps=fps,
            quality=8,
            macro_block_size=1,
            codec="mpeg4",
        )


def save_mask(mask, output_dir, vis_type, frame_idx):
    """
    Save mask image.
    
    Args:
        mask: Mask array (numpy)
        output_dir: Base output directory
        vis_type: Visualization type (e.g., "final_output")
        frame_idx: Frame index
    """
    mask_save_path = os.path.join(
        output_dir, vis_type, "masks", f"mask_{frame_idx}.png"
    )
    _ensure_directory(os.path.dirname(mask_save_path))
    cv2.imwrite(mask_save_path, mask)


def load_prompt(target_path):
    """
    Load text prompt from .txt file.
    
    Args:
        target_path: Path to target .ply file
    
    Returns:
        Prompt text string
    """
    prompt_path = target_path.replace(".ply", ".txt")
    with open(prompt_path, "r") as file:
        prompt = file.read()
    return prompt

