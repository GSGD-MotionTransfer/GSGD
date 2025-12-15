import math

import torch
import torch.nn as nn

from SC4D.pos_enc import Embedder


class SimpleMLP(nn.Module):
    def __init__(
        self,
        angle_tensor_dim: int = 1,
        embedding_dim: int = 1024,
        projection_dim: int = 16,
        hidden_dim: int = 2048,
    ):
        """
        Args:
            other_tensor_dim: Input dimension of the secondary tensor (default: 1)
            num_tokens: Number of tokens in final output dimension
            projection_dim: Dimension for projecting secondary tensor
            hidden_dim: Hidden dimension for main MLP
        """
        super().__init__()

        # Projection network for the secondary tensor
        self.angle_proj = nn.Sequential(
            nn.Linear(angle_tensor_dim, projection_dim),
            nn.Tanh(),
            nn.Linear(projection_dim, projection_dim),
        )
        # self.angle_proj = Embedder(
        #     include_input=False,
        #     input_dims=angle_tensor_dim,
        #     max_freq_log2=9,
        #     num_freqs=projection_dim//2,
        #     log_sampling=True,
        #     periodic_fns=[torch.sin, torch.cos],
        # ).embed

        # Main MLP processing concatenated features
        self.spatial_mlp = nn.Sequential(
            nn.Linear(embedding_dim + projection_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, embedding_dim),
        )

        self.temporal_mlp = nn.Sequential(
            nn.Linear(embedding_dim + projection_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, embedding_dim),
        )

        # self.gate = nn.Parameter(torch.zeros(1) + torch.randn(1) * 1e-4)
        # print("A Random gate to test the seed:", self.gate.item())
        self.alpha = nn.Sequential(
            nn.Linear(projection_dim, projection_dim),
            nn.Tanh(),
            nn.Linear(projection_dim, angle_tensor_dim),
        )

        self.iteration = 0

        self.spatial_mlp.apply(self.init_weights)
        self.temporal_mlp.apply(self.init_weights)
        self.alpha.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=1e-2)
            nn.init.constant_(m.bias, 0.0)

    def forward(
        self, global_embedding: torch.Tensor, angle_tensor: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            global_embedding: Tensor of shape (1,15,5,1024)
            angle_tensor: Tensor of shape (1,)

        Returns:
            Output tensor of shape (1, 15, 5, 1024)
        """
        # turn angle to radians
        angle_tensor = torch.deg2rad(angle_tensor)
        # Project angle tensor and expand to match embedding
        angle_proj = self.angle_proj(angle_tensor)  # (1, projection_dim)
        num_tokens = global_embedding.shape[2]
        angle_proj = angle_proj.expand(num_tokens, -1)

        spatial_embeddings = []
        for frame_embedding in range(global_embedding.shape[1] - 1):
            # Concatenate features
            frame_embedding = global_embedding[:, frame_embedding, :, :].squeeze(0)
            spatial_frame_embed = torch.cat(
                [frame_embedding, angle_proj], dim=-1
            )  # (5,1040)
            angular_spatial_frame_embed = self.spatial_mlp(
                spatial_frame_embed
            )  # (5,1024)
            spatial_embeddings.append(angular_spatial_frame_embed)

        temporal_embedding = global_embedding[:, -1, :, :].squeeze(0)
        temporal_embedding = torch.cat(
            [temporal_embedding, angle_proj], dim=-1
        )  # (5,1040)
        angular_temporal_embed = self.spatial_mlp(temporal_embedding)  # (5,1024)
        spatial_embeddings.append(angular_temporal_embed)

        out = torch.stack(spatial_embeddings).unsqueeze(0)  # (1,15,5,1024)

        # Normalize the output
        # out = out / out.norm(dim=-1, keepdim=True)
        # alpha = torch.sigmoid(self.gate)

        self.iteration += 1
        alpha = self.alpha(angle_proj)
        # angular_final_condition = (1 - alpha) * global_embedding + alpha * out
        angular_final_condition = global_embedding + alpha * out

        # return angular_final_condition, alpha, out
        return angular_final_condition, alpha, out
