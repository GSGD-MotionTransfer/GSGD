import os
import subprocess

import torch

# from moviepy.editor import VideoFileClip, clips_array
from torchvision.utils import save_image


def save_concatenated_images(imgs_or_mask_lst, output_path):
    """
    imgs_or_mask_lst expects list of either:
    1. image type: Tensor of shape [1, 3, H, W] with values in [0, 1]
    2. mask type: Tensor of shape [1, 1, H, W] with values in [0, 1].
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    imgs_lst = [
        (img.expand(-1, 3, -1, -1) if img.shape[1] == 1 else img)
        for img in imgs_or_mask_lst
    ]
    concatenated = torch.cat(imgs_lst, dim=-1)
    save_image(concatenated, output_path)


def composite_videos_side_by_side(directory_path, output_video_filename):
    output_gif_filename = output_video_filename.replace(".mp4", ".gif")
    # Select relevant MP4 files
    files = [
        f for f in os.listdir(directory_path) if f.endswith(".mp4") and "circle_" in f
    ]
    front_view = os.path.join(directory_path, files[0])
    selected_files = files[::4]  # This slices the list to include every third file
    video_paths = [os.path.join(directory_path, f) for f in selected_files]
    video_paths = (
        video_paths[len(video_paths) // 2 :]
        + [front_view]
        + video_paths[: len(video_paths) // 2]
    )

    # Load video clips and composite them side by side
    clips = [VideoFileClip(f) for f in video_paths]
    final_clip = clips_array([clips])  # Adjust the array shape as needed
    final_clip.write_videofile(
        output_video_filename,
        codec="libx264",
        temp_audiofile="temp-audio.m4a",
        remove_temp=True,
    )

    # Close all clips to free up resources
    for clip in clips:
        clip.close()

    # Convert the video to a high-quality GIF using ffmpeg
    convert_video_to_gif(output_video_filename, output_gif_filename)


def convert_video_to_gif(input_video, output_gif):
    # Generate a palette for the GIF
    palette_file = "palette.png"
    subprocess.run(
        [
            "ffmpeg",
            "-i",
            input_video,
            "-vf",
            "fps=15,scale=800:-1:flags=lanczos,palettegen",
            palette_file,
        ],
        check=True,
    )

    # Create the GIF using the generated palette
    subprocess.run(
        [
            "ffmpeg",
            "-i",
            input_video,
            "-i",
            palette_file,
            "-lavfi",
            "fps=15,scale=800:-1:flags=lanczos [x]; [x][1:v] paletteuse",
            output_gif,
        ],
        check=True,
    )

    # Clean up the palette file
    os.remove(palette_file)


def stack_videos_in_a_dir_vertically(input_directory, output_filename):
    files = [f for f in os.listdir(input_directory) if f.endswith(".mp4")]
    video_paths = [os.path.join(input_directory, f) for f in files]
    clips = [VideoFileClip(f) for f in video_paths]

    if not clips:
        print("No videos found to stack.")
        return

    min_width = min(clip.size[0] for clip in clips)
    resized_clips = [clip.resize(width=min_width) for clip in clips]
    final_clip = clips_array(
        [[clip] for clip in resized_clips]
    )  # Each clip in its own row
    final_clip.write_videofile(output_filename, codec="libx264")

    # Close all clips to free up resources
    for clip in clips:
        clip.close()


if __name__ == "__main__":
    # Example usage
    # composite_videos_side_by_side("outputs/reenact/synthetic/flying_ironman/2_views_defult", "visuals/fixed_input_object/2_views_defult.mp4")
    composite_videos_side_by_side(
        "outputs/GT/flying_ironman/8_views_defult",
        "visuals/canon_mis_alignment/GT_8_views.mp4",
    )

    list_of_dirs = [
        "outputs/reenact/synthetic/flying_ironman_5_views",
        "outputs/reenact/synthetic/flying_ironman_5_views_vanila_swap",
        "outputs/reenact/synthetic/flying_ironman_5_views_vanila_circular",
        "outputs/reenact/synthetic/flying_ironman_5_views_vanila_tuned",
        "outputs/reenact/synthetic/flying_ironman_5_views_vanila_freeze_cp",
    ]

    # composite_videos_side_by_side("outputs/reenact/synthetic/flying_ironman_5_views_vanila_freeze_cp","visuals/changes_5_views_cp_swap/flying_ironman_5_views_vanila_freeze_cp.mp4")

    # for dir in list_of_dirs:
    #     name = dir.split("/")[-1]
    #     composite_videos_side_by_side(dir, "visuals/changes_5_views_cp_swap/" + name + ".mp4")

    # # # Example usage
    # stack_videos_in_a_dir_vertically("visuals/changes_5_views_cp_swap", "visuals/changes_5_views_cp_swap/changes_5_views_cp_swap.mp4")
