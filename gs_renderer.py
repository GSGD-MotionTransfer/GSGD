import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from plyfile import PlyData, PlyElement
from torch import nn

import pytorch3d
from deform_utils import (
    cal_arap_error,
    cal_connectivity_from_points_v2,
    produce_edge_matrix_nfmt,
)
from SC4D.gs_renderer import GaussianModel as origGaussianModel
from SC4D.gs_renderer import GaussianRasterizationSettings, GaussianRasterizer, MiniCam
from SC4D.gs_renderer import Renderer as OrigRenderer
from SC4D.gs_renderer import TimeNet as origTimeNet
from SC4D.gs_renderer import (
    build_rotation_3d,
    eval_sh,
    initialize_weights,
    initialize_weights_one,
    initialize_weights_zero,
    quat_mul,
)
from SC4D.pos_enc import get_embedder


class TimeNet(origTimeNet):
    pass
    # def __init__(self, D=8, W=256, skips=[4], device="cuda"):
    #     # Overriden to increase "pts|times_ch" i.e. positional and time encoding dimension
    #     super(TimeNet, self).__init__()
    #     self.pts_ch = 20  # 10
    #     self.times_ch = 12  # 6
    #     self.pts_emb_fn, pts_out_dims = get_embedder(self.pts_ch, 3)
    #     self.times_emb_fn, times_out_dims = get_embedder(self.times_ch, 1)
    #     self.input_ch = pts_out_dims + times_out_dims
    #     self.skips = skips
    #     self.deformnet = nn.ModuleList(
    #         [nn.Linear(self.input_ch, W)]
    #         + [
    #             (
    #                 nn.Linear(W, W)
    #                 if i not in self.skips
    #                 else nn.Linear(W + self.input_ch, W)
    #             )
    #             for i in range(D - 1)
    #         ]
    #     )
    #     self.pts_layers = nn.Sequential(nn.Linear(W, W), nn.ReLU(), nn.Linear(W, 3))
    #     self.rot_layers = nn.Sequential(nn.Linear(W, W), nn.ReLU(), nn.Linear(W, 4))
    #     self.device = device
    #     self.deformnet.apply(initialize_weights)
    #     self.pts_layers.apply(initialize_weights)
    #     self.rot_layers.apply(initialize_weights)
    #     self.pts_layers[-1].apply(initialize_weights_zero)
    #     self.rot_layers[-1].apply(initialize_weights_one)

    # def forward(self, pts, t, nobatch=False, t_apply=False):
    #     # Overriden to allow override the output for frame 0 to be the canonical ouput.
    #     if len(pts.shape) == 2:
    #         nobatch = True
    #         pts = pts.unsqueeze(0)
    #     if t_apply:
    #         times = t
    #         pts = pts.repeat(times.shape[0], 1, 1)
    #     else:
    #         times = (
    #             torch.tensor([t])[:, None, None]
    #             .repeat(1, pts.shape[1], 1)
    #             .to(self.device)
    #         )  # B * N * 1
    #     pts_emb = self.pts_emb_fn(pts)
    #     times_emb = self.times_emb_fn(times)
    #     pts_emb = torch.cat([pts_emb, times_emb], dim=-1)  # B * N * (p + t)
    #     h = pts_emb
    #     for i, l in enumerate(self.deformnet):
    #         h = self.deformnet[i](h)
    #         h = F.relu(h)
    #         if i in self.skips:
    #             h = torch.cat([pts_emb, h], dim=-1)
    #     pts_t, rot_t = self.pts_layers(h), self.rot_layers(h)

    #     # Force it to have the canonical pose aligned with the frame 0 supervision
    #     # TODO might hurt us with non-consistent supervision. Should remove this after we understand the robot bug (or remove the warmup entirely and then they are aligned by definition)
    #     # if t_apply:
    #     #     pts_t[(times == 1).expand_as(pts_t)] = 0
    #     #     rot_t[(times == 1).expand_as(rot_t)] = 0
    #     # elif t == 1:
    #     #     pts_t = torch.zeros_like(pts_t)
    #     #     rot_t = torch.zeros_like(rot_t)

    #     if nobatch:
    #         pts_t, rot_t = pts_t[0], rot_t[0]
    #     return pts_t, rot_t


class GaussianModel(origGaussianModel):
    def __init__(self, sh_degree: int):
        # Overriden just to use our patched timenet
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()
        #
        self._timenet = TimeNet()
        self._c_xyz = torch.empty(0)
        self._c_radius = torch.empty(0)
        self._r = torch.empty(0)

    @staticmethod
    @torch.no_grad()
    def save_ply_from_raw(
        path1, means3D, shs, opacity, scales, rotations, list_of_attributes
    ):
        # Used to save the deformed point cloud manually for later use
        os.makedirs(os.path.dirname(path1), exist_ok=True)

        xyz = means3D.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        # Assume no f_rest
        f_dc = (
            shs.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        )
        f_rest = (
            torch.zeros([shs.shape[0], 0, 3])
            .detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )

        opacities = opacity.detach().cpu().numpy()
        if False:  # len(self._r) > 0:
            scale = self._r.expand_as(self._xyz).detach().cpu().numpy()
        else:
            scale = scales.detach().cpu().numpy()
        rotation = rotations.detach().cpu().numpy()

        dtype_full = [
            (attribute, "f4") for attribute in list_of_attributes
        ]  # self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate(
            (xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1
        )
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path1)


class Renderer(OrigRenderer):
    def __init__(
        self,
        sh_degree=3,
        white_background=True,
        radius=1,
        delta_t=1 / 32,
        override_deformation_rotation=False,
    ):
        self.sh_degree = sh_degree
        self.white_background = white_background
        self.radius = radius
        # Only change here:
        self.override_deformation_rotation = override_deformation_rotation

        self.gaussians = GaussianModel(sh_degree)

        self.bg_color = torch.tensor(
            [1, 1, 1] if white_background else [0, 0, 0],
            dtype=torch.float32,
            device="cuda",
        )

        self.delta_t = delta_t

    def render(
        self,
        viewpoint_camera,
        scaling_modifier=1.0,
        bg_color=None,
        override_color=None,
        compute_cov3D_python=False,
        convert_SHs_python=False,
        time=0.0,
        stage="s1",
        rot_as_res=True,
        xyz_detach=False,
        local_frame=True,
        #
        direct_deform=False,
        vertices_deform=None,
        subsample_ratio=1.0,
        save_ply_path=None,
        num_frames=14,
    ):
        is_single_frame = isinstance(time, float)
        # Get the deformation
        c_means3D = self.gaussians.get_c_xyz if stage == "s2" else self.gaussians._xyz

        if time is None:
            q_times = torch.linspace(0, 1, num_frames).to(c_means3D.device)
            q_times = (
                q_times[:, None, None]
                .repeat(1, c_means3D.shape[0], 1)
                .to(c_means3D.device)
            )
        elif is_single_frame:
            q_times = torch.tensor([time]).to(c_means3D.device)
            q_times = (
                q_times[:, None, None]
                .repeat(1, c_means3D.shape[0], 1)
                .to(c_means3D.device)
            )
        else:
            raise ValueError("Invalid time input")

        means3D_deform_lst, rots_deform_lst = self.gaussians._timenet(
            c_means3D.unsqueeze(0), q_times, t_apply=True
        )

        renders_dict = {}
        for i in range(len(means3D_deform_lst)):
            means3D_deform = means3D_deform_lst[i]
            rots_deform = rots_deform_lst[i]
            cpts_t = c_means3D + means3D_deform

            cur_render_dict = {
                **self._render(
                    c_means3D,
                    means3D_deform,
                    rots_deform,
                    stage,
                    viewpoint_camera,
                    scaling_modifier,
                    bg_color,
                    override_color,
                    compute_cov3D_python,
                    convert_SHs_python,
                    rot_as_res,
                    xyz_detach,
                    local_frame,
                    direct_deform,
                    vertices_deform,
                    subsample_ratio,
                    save_ply_path,
                ),
                **{"cpts_t": cpts_t},
            }

            for k, v in cur_render_dict.items():
                renders_dict[k] = renders_dict.get(k, []) + [cur_render_dict[k]]

        if is_single_frame:
            return {k: v[0] for k, v in renders_dict.items()}
        return renders_dict

    def _render(
        self,
        c_means3D,
        means3D_deform,
        rots_deform,
        stage,
        viewpoint_camera,
        scaling_modifier=1.0,
        bg_color=None,
        override_color=None,
        compute_cov3D_python=False,
        convert_SHs_python=False,
        rot_as_res=True,
        xyz_detach=False,
        local_frame=True,
        direct_deform=False,
        vertices_deform=None,
        subsample_ratio=1.0,  # for debugging
        save_ply_path=None,  # For debugging, ugly that it is here but will do
    ):
        # Overriden, no change
        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = (
            torch.zeros_like(
                self.gaussians.get_xyz,
                dtype=self.gaussians.get_xyz.dtype,
                requires_grad=True,
                device=self.gaussians.get_xyz.device,
            )
            + 0
        )
        try:
            screenspace_points.retain_grad()
        except:
            pass

        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=self.bg_color if bg_color is None else bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=self.gaussians.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=False,
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        means3D = self.gaussians.get_xyz
        means2D = screenspace_points
        opacity = self.gaussians.get_opacity

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = None
        if compute_cov3D_python:
            cov3D_precomp = self.gaussians.get_covariance(scaling_modifier)
        else:
            scales = self.gaussians.get_scaling
            # rotations = self.gaussians.get_rotation
            rotations = self.gaussians._rotation

        if stage >= "s2":
            eps = 1e-7
            c_radius = self.gaussians.get_c_radius(stage)
            neighbor_dists = self.gaussians.neighbor_dists
            neighbor_indices = self.gaussians.neighbor_indices
            c_radius_n = c_radius[neighbor_indices]
            w = torch.exp(-1.0 * neighbor_dists**2 / (2.0 * (c_radius_n[:, :, 0] ** 2)))
            w = w + eps
            w = F.normalize(w, p=1)
            means3D_n = c_means3D[neighbor_indices]  # N*4*3
            means3D_n_deform = means3D_deform[neighbor_indices]  # N*4*3

            rots3D_n_deform = rots_deform[neighbor_indices]  # N*4*4
            if self.override_deformation_rotation:
                with torch.no_grad():
                    derived_rotatations = derive_rotation_from_control_points(
                        c_means3D,
                        means3D_deform,
                        self.gaussians.ii_c,
                        self.gaussians.jj_c,
                        self.gaussians.nn_c,
                        self.gaussians.weight_c,
                        self.gaussians.c_nn_K,
                    )
                    rots3D_n_deform = derived_rotatations[neighbor_indices]  # N*4*4

            if local_frame:
                rot_addon = (
                    build_rotation_3d(rots3D_n_deform)
                    @ (means3D[:, None] - means3D_n)[..., None]
                ).squeeze(-1)
                if torch.isnan(rot_addon).sum() >= 1:
                    # If no rotation (0 rotation) it will get nan. So just dont use this part.
                    rot_addon = 0

                pts3D = (w[..., None] * (rot_addon + means3D_n + means3D_n_deform)).sum(
                    dim=1
                )
            else:
                pts3D = means3D + (w[..., None] * means3D_n_deform).sum(dim=1)
            new_rots3D = (w[..., None] * rots3D_n_deform).sum(dim=1)

            means3D = pts3D
            rotations = quat_mul(new_rots3D, rotations)
            # rotations = rotations + rots3D
        elif stage == "s1":
            means3D = means3D + means3D_deform
        else:
            assert ValueError("Nonexistent stage!!!")

        if xyz_detach:
            means3D = means3D.detach()

        rotations = self.gaussians.rotation_activation(rotations)

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = None
        # if colors_precomp is None:
        if override_color is None:
            if convert_SHs_python:
                shs_view = self.gaussians.get_features.transpose(1, 2).view(
                    -1, 3, (self.gaussians.max_sh_degree + 1) ** 2
                )
                dir_pp = self.gaussians.get_xyz - viewpoint_camera.camera_center.repeat(
                    self.gaussians.get_features.shape[0], 1
                )
                dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
                sh2rgb = eval_sh(
                    self.gaussians.active_sh_degree, shs_view, dir_pp_normalized
                )
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            else:
                shs = self.gaussians.get_features  # [N, 1, 3]
        else:
            colors_precomp = override_color

        # Rasterize visible Gaussians to image, obtain their radii (on screen).
        if subsample_ratio < 1.0:
            generator = torch.Generator()
            generator.manual_seed(0)
            num_gaussians = means3D.shape[0]
            num_samples = int(num_gaussians * subsample_ratio)
            sampled_indices = torch.randperm(num_gaussians, generator=generator)[
                :num_samples
            ]
            means3D = means3D[sampled_indices]
            rotations = rotations[sampled_indices]
            scales = scales[sampled_indices]
            opacity = opacity[sampled_indices]
            if shs is not None:
                shs = shs[sampled_indices]
            if colors_precomp is not None:
                colors_precomp = colors_precomp[sampled_indices]
            if cov3D_precomp is not None:
                cov3D_precomp = cov3D_precomp[sampled_indices]

        if save_ply_path is not None:
            self.gaussians.save_ply_from_raw(
                save_ply_path,
                means3D=means3D,
                shs=shs,
                opacity=self.gaussians._opacity,
                scales=self.gaussians._scaling,
                rotations=rotations,
                list_of_attributes=self.gaussians.construct_list_of_attributes(),
            )

        rendered_image, radii, rendered_depth, rendered_alpha = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
        )

        rendered_image = rendered_image.clamp(0, 1)

        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        return {
            "image": rendered_image,
            "depth": rendered_depth,
            "alpha": rendered_alpha,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
            "pts_t": means3D,
        }

    def arap_loss_v2(self, delta_t=0.05, t_samp_num=8, stage="s1"):
        q_times = torch.rand(t_samp_num).to("cuda")
        if stage == "s1":
            means3D = self.gaussians._xyz[None]
        else:
            means3D = self.gaussians._c_xyz[None]
        q_times = (
            q_times[:, None, None].repeat(1, means3D.shape[1], 1).to(means3D.device)
        )
        means3D_deform, _ = self.gaussians._timenet(means3D, q_times, t_apply=True)
        means3D_t = means3D.repeat(t_samp_num, 1, 1).detach() + means3D_deform
        # Ball query
        ii, jj, nn, _ = cal_connectivity_from_points_v2(means3D_t, K=10)
        error = cal_arap_error(means3D_t, ii, jj, nn, reference_pose=means3D.squeeze(0))
        return error, (ii, jj, nn, _)


def derive_rotation_from_neighborhood_all_gaussians(
    means3D, means3D_n, means3D_n_deform, w
):
    """
    NOTE: it is a highly non-efficient calculation for every gaussian. In SCGS they calculate it per control point, and then apply it on the gaussians themselvs. Let's see if it is so inefficient and we'll change it if so.
    Derive a new rotation quaternion for each Gaussian by examining
    how its neighbors moved, in an ARAP-like manner.

    Args:
        means3D:         (N, 3)     - The "center" positions of your Gaussians.
        means3D_n:       (N, K, 3)  - The original neighbor positions (for each Gaussian).
        means3D_n_deform:(N, K, 3)  - The *deformation* of the neighbor positions
                                     (often means3D_n + some offset).
        w:               (N, K)     - weights precomputed via e.g. exp(-(dist^2)/2sigma^2)):
    Returns:
        rots3D:          (N, 4)     - A quaternion [x, y, z, w] for each Gaussian,
                                      representing the best-fit local rotation.
    """

    import torch

    device = means3D.device
    N, K, _ = means3D_n.shape

    # -------------------------------------------------------------------------
    # 1) Construct original edge vectors (P) and new edge vectors (P_prime).
    #
    #    - Original edges (P):   neighbors - center
    #    - New edges (P_prime): (neighbors_deformed) - center
    #
    # -------------------------------------------------------------------------
    new_positions_n = means3D_n + means3D_n_deform
    # Original edges
    P = means3D_n - means3D.unsqueeze(1)  # (N, K, 3)
    # New (deformed) edges
    P_prime = new_positions_n - means3D.unsqueeze(1)  # (N, K, 3)

    # -------------------------------------------------------------------------
    # 2) Construct diagonal weight matrix D for each batch element
    #    w.shape = (N, K) => D.shape = (N, K, K)
    # -------------------------------------------------------------------------
    # "D" is the diagonal block for each row. In ARAP, you'd do something like:
    # We'll build "D" for batch-matrix multiply:
    D = torch.diag_embed(w)  # (N, K, K)

    # -------------------------------------------------------------------------
    # 3) Compute S_i = P_i^T * D_i * P'_i for each Gaussian (batch style).
    #
    #    S has shape (N, 3, 3). This is the 'covariance'-like matrix that
    #    we factor with SVD to extract the best-fit rotation.
    # -------------------------------------------------------------------------
    # Reshape P -> (N, 3, K) so we can do the matrix multiplication properly
    P_t = P.permute(0, 2, 1)  # shape (N, 3, K)
    # Multiply:  S = P^T * D * P'
    # Step-by-step: temp = D * P' => shape (N, K, 3)
    temp = torch.bmm(D, P_prime)  # (N, K, 3)
    # Now multiply P_t with "temp", but watch dimensions:
    S = torch.bmm(P_t, temp)  # shape (N, 3, 3)

    # -------------------------------------------------------------------------
    # 4) Handle any "unchanged" entries if needed (like your reference code does)
    #    If P == P_prime => no deflection => set S=0 => R=Identity
    # -------------------------------------------------------------------------
    unchanged_mask = (P == P_prime).all(dim=2)  # (N, K) True if each edge is same
    unchanged_rows = torch.where(unchanged_mask.all(dim=1))[0]  # all edges same
    if len(unchanged_rows) > 0:
        S[unchanged_rows] = (
            0  # SVD should result in U and W results of the Identity matrix in these cases, should validate it
        )

    # -------------------------------------------------------------------------
    # 5) SVD of S to get R = W * U^T
    # -------------------------------------------------------------------------
    U, sig, W = torch.svd(S)  # each S[i] => SVD => U[i], sig[i], W[i]
    R_mat = torch.bmm(W, U.permute(0, 2, 1))  # (N, 3, 3)

    # -------------------------------------------------------------------------
    # 6) Flip columns if det(R) <= 0 to ensure a proper rotation with +1 determinant
    # -------------------------------------------------------------------------
    detR = torch.det(R_mat)
    flip_indices = torch.where(detR <= 0)[0]
    if len(flip_indices) > 0:
        # Flip the column of U corresponding to smallest singular value
        Umod = U.clone()
        # Argmin along the singular values for each row
        cols_to_flip = torch.argmin(sig[flip_indices], dim=1)
        for idx, col in zip(flip_indices, cols_to_flip):
            Umod[idx, :, col] *= -1.0
        R_mat[flip_indices] = torch.bmm(
            W[flip_indices], Umod[flip_indices].permute(0, 2, 1)
        )

    # -------------------------------------------------------------------------
    # 7) Convert the 3x3 rotation to a quaternion [x, y, z, w]
    #    (Common function: matrix_to_quaternion or something similar.)
    # -------------------------------------------------------------------------
    rots3D = matrix_to_quaternion(R_mat)  # shape (N, 4)

    return rots3D


def derive_rotation_from_control_points(
    means3D, means3D_deform, ii_c, jj_c, nn_c, weight_c, K
):
    """
    Derive a new rotation quaternion for each Gaussian by examining
    how its neighbors moved, in an ARAP-like manner.
    """
    deformed_control_points = means3D_deform + means3D

    P_prime = produce_edge_matrix_nfmt(
        deformed_control_points,
        (means3D.shape[0], K, 3),
        ii_c,
        jj_c,
        nn_c,
        device=means3D.device,
    )  # [Nv, K, 3]
    P = produce_edge_matrix_nfmt(
        means3D, (means3D.shape[0], K, 3), ii_c, jj_c, nn_c, device=means3D.device
    )  # [Nv, K, 3]

    D = torch.diag_embed(weight_c, dim1=1, dim2=2)  # [Nv, K, K]

    ### Calculate covariance matrix in bulk
    S = torch.bmm(P.permute(0, 2, 1), torch.bmm(D, P_prime))  # [Nv, 3, 3]

    ## in the case of no deflection, set S = 0, such that R = I. This is to avoid numerical errors
    unchanged_verts = torch.unique(
        torch.where((P == P_prime).all(dim=1))[0]
    )  # any verts which are undeformed
    S[unchanged_verts] = 0

    U, sig, W = torch.svd(S)
    R = torch.bmm(W, U.permute(0, 2, 1))  # compute rotations

    return pytorch3d.transforms.matrix_to_quaternion(R)


def matrix_to_quaternion(R):
    import torch

    device = R.device
    N = R.shape[0]
    quat = torch.zeros(N, 4, device=device)

    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    mask_t = trace > 0

    #
    # 1) Handle trace > 0
    #
    if mask_t.any():
        t = 0.5 * torch.sqrt(trace[mask_t] + 1.0)
        quat[mask_t, 0] = t
        quat[mask_t, 1] = (R[mask_t, 2, 1] - R[mask_t, 1, 2]) / (4.0 * t)
        quat[mask_t, 2] = (R[mask_t, 0, 2] - R[mask_t, 2, 0]) / (4.0 * t)
        quat[mask_t, 3] = (R[mask_t, 1, 0] - R[mask_t, 0, 1]) / (4.0 * t)

    #
    # 2) Handle trace <= 0
    #
    mask_not = ~mask_t
    if mask_not.any():
        # Indices in the original batch
        idx_not = mask_not.nonzero(as_tuple=True)[0]  # e.g. a 1D tensor of indices

        R_not = R[mask_not]
        diag = torch.stack([R_not[:, 0, 0], R_not[:, 1, 1], R_not[:, 2, 2]], dim=1)
        max_diag_indices = diag.argmax(dim=1)  # which of x,y,z is largest diag?

        # We iterate over each "negative trace" matrix individually
        for i in range(R_not.shape[0]):
            idx_max = max_diag_indices[i].item()
            # The row in the original 'quat' we need to fill
            row_idx = idx_not[i].item()
            R_sub = R_not[i]

            if idx_max == 0:
                # x is biggest diagonal
                x = 0.5 * torch.sqrt(1 + R_sub[0, 0] - R_sub[1, 1] - R_sub[2, 2])
                quat[row_idx, 1] = x
                quat[row_idx, 0] = (R_sub[2, 1] - R_sub[1, 2]) / (4.0 * x)
                quat[row_idx, 2] = (R_sub[0, 1] + R_sub[1, 0]) / (4.0 * x)
                quat[row_idx, 3] = (R_sub[0, 2] + R_sub[2, 0]) / (4.0 * x)

            elif idx_max == 1:
                # y is biggest diagonal
                y = 0.5 * torch.sqrt(1 - R_sub[0, 0] + R_sub[1, 1] - R_sub[2, 2])
                quat[row_idx, 2] = y
                quat[row_idx, 0] = (R_sub[0, 2] - R_sub[2, 0]) / (4.0 * y)
                quat[row_idx, 1] = (R_sub[0, 1] + R_sub[1, 0]) / (4.0 * y)
                quat[row_idx, 3] = (R_sub[1, 2] + R_sub[2, 1]) / (4.0 * y)

            else:
                # z is biggest diagonal
                z = 0.5 * torch.sqrt(1 - R_sub[0, 0] - R_sub[1, 1] + R_sub[2, 2])
                quat[row_idx, 3] = z
                quat[row_idx, 0] = (R_sub[1, 0] - R_sub[0, 1]) / (4.0 * z)
                quat[row_idx, 1] = (R_sub[0, 2] + R_sub[2, 0]) / (4.0 * z)
                quat[row_idx, 2] = (R_sub[1, 2] + R_sub[2, 1]) / (4.0 * z)

    # Now normalize
    norm_quat = torch.norm(quat, dim=1, keepdim=True) + 1e-8
    quat = quat / norm_quat
    return quat


def save_image_locally(image: torch.Tensor, path: str = "rendered_image.png"):
    import torch
    import torchvision.transforms as transforms
    from PIL import Image

    # Assuming rendered_image is your tensor with shape (3, 256, 256)
    # Convert to PIL Image
    to_pil = transforms.ToPILImage()
    image = to_pil(image)

    # Save locally
    image.save(path)

    print(f"Image saved as {path}")
