import sys

import PIL.Image
import PIL.ImageOps
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.utils import save_image

# utils of reenact_anything. It isn't written as a package, so we need to import it like this
sys.path.append("reenact-anything")
from utils.video_utils import export_to_video, pil_to_pt, pt_to_pil


class GlobalMotionInversionDataset(Dataset):
    def __init__(
        self,
        multi_source_videos_frames,
        available_views,
        size=(25, 1024, 576),  # (n_frames, width, height)
        flip_p=0.5,
        set="train",
        device="cpu",
    ):
        self.available_views = available_views
        self.size = size
        self.n_frames, self.width, self.height = size
        self.flip_p = flip_p
        self.device = device
        self.angle_videos_frames = []
        self.angle_names = []
        # Store the number of angles for cycling
        self._length = len(
            self.available_views
        )  # Dataset length is the number of angles

        self.flip_transform = transforms.RandomHorizontalFlip(p=self.flip_p)

        # Prepare videos for each angle
        for angle_name, single_angle_video in zip(
            available_views, multi_source_videos_frames
        ):
            resized_frames = [
                f.resize((self.width, self.height)) for f in single_angle_video
            ]
            # Ensure frame count is n_frames, up\downsample if necessary
            if len(resized_frames) > self.n_frames:
                print(
                    f"Warning: Video has {len(resized_frames)} frames. Taking the first {self.n_frames}"
                )
                resized_frames = resized_frames[: self.n_frames]
            elif len(resized_frames) < self.n_frames:
                print(
                    f"Warning: Video has {len(resized_frames)} frames. Upsampling to {self.n_frames}"
                )
                resized_frames = resized_frames + [resized_frames[-1]] * (
                    self.n_frames - len(resized_frames)
                )

            self.angle_videos_frames.append(pil_to_pt(resized_frames).to(self.device))
            self.angle_names.append(angle_name)

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        # Cycle through the angles
        angle_index = i % len(self.available_views)
        frames = self.angle_videos_frames[angle_index]
        angle_name = self.angle_names[angle_index]

        sample = {}
        sample["frame_0"] = frames[0]  # First frame of the selected angle
        sample["frames"] = frames
        sample["angle_name"] = angle_name

        return sample

    @staticmethod
    def save_batch(batch):
        export_to_video(pt_to_pil(batch["frames"]), "./debug_frames.mp4")
        pt_to_pil(batch["frame_0"].unsqueeze(0))[0].save(
            "./debug_frame_0.png",
        )


class GlobalSimpleImagesDataset(Dataset):
    def __init__(
        self,
        orig_images,
        width=1024,
        height=576,
        device="cpu",
    ):
        self.width = width
        self.height = height

        first_frames, angle_names, images_names = orig_images

        white_bg = PIL.Image.new(
            "RGBA", (self.width, self.height), (255, 255, 255, 255)
        )

        self.images_pil = []
        for frames in first_frames:
            processed_frames = []
            for img in frames:
                resized_img = img.convert("RGBA").resize((self.width, self.height))
                white_bg = PIL.Image.new("RGBA", resized_img.size, (255, 255, 255, 255))
                processed_frames.append(
                    PIL.Image.alpha_composite(white_bg, resized_img).convert("RGB")
                )
            self.images_pil.append(processed_frames)

        # pil_to_pt returns [1,chn,w,h] so squeeze to [3,w,h] (remove alpha channel)
        self.angle_first_frames = [
            [pil_to_pt(img).to(device)[:, :3, :, :] for img in angle_frames]
            for angle_frames in self.images_pil
        ]

        self.images_names = images_names
        self.angle_names = angle_names

        self._length = len(self.angle_first_frames)

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        angle_index = i % len(self.angle_names)
        angle_name = self.angle_names[angle_index]
        first_frames = self.angle_first_frames[angle_index]

        return {
            "angle_name": angle_name,
            "first_frames": first_frames,
            "figures": self.images_names,
        }

    @staticmethod
    def save_sample(sample: torch.Tensor, savename=None):
        if savename is None:
            savename = "./debug_frame_0.png"
        pt_to_pil(sample)[0].save(savename)
