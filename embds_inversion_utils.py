import math
import os
import sys

import torch

# Import override functions from reenact-anything submodule
# Note: sys.path.append is needed because:
# 1. The directory name contains a hyphen (reenact-anything), which can't be used as a Python module name
# 2. It's not structured as a proper Python package (no __init__.py files) rather a simple research code
# This pattern is consistent with other files in the codebase (see consolidate_4d.py, train_mlp_reenact.py, etc.)
sys.path.append(os.path.join(os.path.dirname(__file__), "reenact-anything"))
from utils.embds_inversion_utils import (
    ImageEmbeddingWrapper as BaseImageEmbeddingWrapper,
    allow_motion_embedding,
    forward_downUpMidBlock_TransformerSpatioTemporalModel,
    forward_UNetSpatioTemporalConditionModel,
    initialize_image_embedding,
    override_pipeline_call,
    pipeline_decode_latents,
)

# from motion_neural_field import SimpleMLP


class ImageEmbeddingWrapper3D(BaseImageEmbeddingWrapper):
    SAVE_NAME = "img_embds_wrapper.pt"

    def __init__(
        self, initial_clip_embeddings, mlp_lr, lr, device, number_of_anchors=4
    ):
        """
        Args:
            initial_clip_embeddings (dict): Keys are angles (as strings), values are embeddings.
            mlp_lr (float): Learning rate for the MLP.
            lr (float): Learning rate for other parts.
            device (str): Device to run the computations on.
            number_of_anchors (int): Number of anchor angles to use.
        """
        # Don't call super().__init__() as we don't use the base class's tensor-based initialization
        torch.nn.Module.__init__(self)
        self.device = device
        self.mlp_lr = mlp_lr
        self.lr = lr

        # Compute the angular gap between anchors and the list of anchor angles.
        self.number_of_anchors = number_of_anchors
        self.anchor_gap = (
            360.0 / number_of_anchors
        )  # For example, 90 if number_of_anchors==4
        # For example, for 4 anchors: [0, 90, 180, 270]
        self.anchor_angles = [i * self.anchor_gap for i in range(number_of_anchors)]
        # self.anchor_angles_rad = [torch.deg2rad(a) for a in self.anchor_angles]

        # ParameterDict to store anchor embeddings.
        self.anchors = torch.nn.ParameterDict()
        self.init_anchors(initial_clip_embeddings)
        print(f"\n\n\nInitialized {len(self.anchors)} anchor embeddings.")
        print(f"Anchor angles: {self.anchor_angles}\n\n\n")

    def grad_hook(self, name):
        def hook(grad):

            if not torch.all(torch.isfinite(grad)):
                print(f"Non-finite gradient for {name} detected:", torch.norm(grad))
            return grad

        return hook

    def _anchor_key(self, angle):
        return str(angle).replace(".", "_")

    def init_anchors(self, initial_clip_embeddings):
        for anchor_angle in self.anchor_angles:
            selected_embeddings = []  # To store embeddings close to the anchor.
            spherical_weights = (
                []
            )  # To store computed weights based on angular distance.

            for key in initial_clip_embeddings:
                # Convert key to a float angle in [0, 360)
                angle = float(key) % 360
                diff = abs(angle - anchor_angle)
                # Compute the minimum angular distance (taking wrap-around into account)
                distance = min(diff, 360 - diff)

                # Select embeddings that are within half the anchor gap.
                # (e.g., for gap=90°, this means within 45°)
                if distance <= 45:
                    # Weight decays exponentially with distance.
                    # For a gap of 90, this gives a decay factor of distance/(90/4)=distance/22.5 (as before).
                    weight = 1.0 / (2 ** (distance / (self.anchor_gap / 4)))
                    selected_embeddings.append(initial_clip_embeddings[key])
                    spherical_weights.append(weight)

            # Compute weighted average if any embeddings were selected.
            if spherical_weights:
                # weights_tensor: shape [N] where N is the number of selected embeddings.
                weights_tensor = torch.tensor(spherical_weights, device=self.device)
                weights_tensor /= (
                    weights_tensor.sum()
                )  # Normalize so that weights sum to 1.
                # embeddings_stack: shape [N, embedding_dim]
                embeddings_stack = (
                    torch.stack(selected_embeddings).squeeze(1).squeeze(1)
                )
                # avg_embedding: shape [embedding_dim]
                avg_embedding = torch.sum(
                    embeddings_stack * weights_tensor.view(-1, 1, 1, 1), dim=0
                )
            else:
                avg_embedding = (
                    None  # Handle the case if no embeddings are close enough.
                )

            # Save the computed average as a learnable parameter.
            self.anchors[self._anchor_key(anchor_angle)] = torch.nn.Parameter(
                avg_embedding
            )

    def get_closest_anchors(self, angle):
        angle = angle % 360.0
        gap = self.anchor_gap
        lower_index = int(angle // gap)
        lower_anchor = self.anchor_angles[lower_index]
        upper_anchor = self.anchor_angles[(lower_index + 1) % self.number_of_anchors]
        # for anchor in self.anchor_angles:
        #     if anchor in {lower_anchor, upper_anchor}:
        #         self.anchors[self._anchor_key(anchor)].requires_grad_(
        #             True
        #         )  # Enable for selected anchors
        #     else:
        #         self.anchors[self._anchor_key(anchor)].requires_grad_(
        #             False
        #         )  # Disable for others

        return lower_anchor, upper_anchor

    def linear_interpolation(self, angle, anchor0, anchor1):
        # Get anchor embeddings: shape [15, 5, 1024]
        # Use the sanitized keys to retrieve the parameters.
        a0 = self.anchors[self._anchor_key(anchor0)].squeeze()
        if a0.requires_grad:
            a0.register_hook(self.grad_hook(anchor0))
        a1 = self.anchors[self._anchor_key(anchor1)].squeeze()
        if a1.requires_grad:
            a1.register_hook(self.grad_hook(anchor1))

        # Calculate interpolation weight t from the given angle.
        # (Assumes anchors are at angles such that t is in [0, 1])
        t = (torch.abs(anchor0 - angle)) / self.anchor_gap

        # Perform linear interpolation.
        interpolated_embedding = (1 - t) * a0 + t * a1
        if interpolated_embedding.requires_grad:
            interpolated_embedding.register_hook(self.grad_hook("Interpolation"))

        return interpolated_embedding  # Shape: [15, 5, 1024]

    def spherical_interpolation(self, theta_t, theta1, theta2):
        # Get anchor embeddings: shape [15, 5, 1024]
        # Use the sanitized keys to retrieve the parameters.
        anchor1 = self.anchors[self._anchor_key(theta1)].squeeze()
        if anchor1.requires_grad:
            anchor1.register_hook(self.grad_hook("anchor1"))
        anchor2 = self.anchors[self._anchor_key(theta2)].squeeze()
        if anchor2.requires_grad:
            anchor2.register_hook(self.grad_hook("anchor2"))

        # Convert degrees to radians
        theta1_rad = math.radians(theta1)
        theta2_rad = math.radians(theta2)
        theta_t_rad = math.radians(theta_t)

        # Compute angular difference
        omega = theta2_rad - theta1_rad

        # Avoid division by zero in case of identical angles
        if abs(omega) < 1e-6:
            return anchor1  # If the angles are the same, return the vector.

        # Compute interpolation parameter t
        t = (theta_t_rad - theta1_rad) / omega

        # Compute SLERP coefficients
        p = math.sin((1 - t) * omega) / math.sin(omega)
        q = math.sin(t * omega) / math.sin(omega)

        interpolated_motion_embedding = p * anchor1 + q * anchor2
        if interpolated_motion_embedding.requires_grad:
            interpolated_motion_embedding.register_hook(self.grad_hook("interpolated"))

        return interpolated_motion_embedding

    def forward(self, angle, *inputs):

        anchor0, anchor1 = self.get_closest_anchors(angle)
        anchor0 = 360 + anchor0 if anchor0 < 0 else anchor0
        anchor1 = 360 + anchor1 if anchor1 < 0 else anchor1
        angle = 360 + angle if angle < 0 else angle

        interpolated_motion_embedding = self.spherical_interpolation(
            angle, anchor0, anchor1
        )
        interpolated_motion_embedding = interpolated_motion_embedding.unsqueeze(
            0
        )  # Shape: [1, 15, 5, 1024]
        return interpolated_motion_embedding

    def get_parameters(self):
        anchor_params = {"params": list(self.anchors.values()), "lr": self.lr}
        return [anchor_params]

    def get_anchors_embeddings(self):
        return torch.stack([self.anchors[str(a)] for a in self.anchor_angles])

    def get_all_angular_embeddigns(self):
        embeddings = {}
        num_views = 16
        for angle in range(0, num_views):
            angle = (360 / num_views) * angle
            angle = angle - 360 if angle >= 180 else angle
            angle_tensor = torch.tensor(angle, device=self.device)
            with torch.no_grad():
                interpolated = self.forward(angle_tensor, "residual")
                embeddings[str(angle)] = interpolated

            del angle_tensor, interpolated

        return embeddings

    def save_pretrained(self, save_path):
        if os.path.isdir(save_path):
            file_path = os.path.join(
                save_path, self.SAVE_NAME
            )  # Save inside the directory
        else:
            file_path = save_path  # Save directly to the specified file path
            save_dir = os.path.dirname(file_path)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir)  # Create directories if they don’t exist

        # Save anchors instead of global_motion
        torch.save(self.anchors, file_path)
        print(f"[ImageEmbeddingWrapper3D] Anchors saved to {file_path}")

    def load_tensor(self, load_path):
        if os.path.isfile(load_path):
            file_path = load_path  # Use the provided file
        else:
            file_path = os.path.join(
                load_path, self.SAVE_NAME
            )  # Use SAVE_NAME in the given directory

        loaded_anchors = torch.load(file_path, weights_only=True)
        # Load anchors into ParameterDict
        if isinstance(loaded_anchors, dict):
            for key, value in loaded_anchors.items():
                if key in self.anchors:
                    self.anchors[key] = torch.nn.Parameter(value.to(self.device))
        else:
            # Handle legacy format if needed
            raise ValueError("Loaded file format not supported. Expected ParameterDict.")
        print(f"[ImageEmbeddingWrapper3D] Anchors loaded from {file_path}")
        return self


