import numpy as np
import torch

from SC4D.deform_utils import (
    arap_deformation_loss,
    cal_connectivity_from_points,
    cal_connectivity_from_points_v2,
    estimate_rotation,
    produce_edge_matrix_nfmt,
)


def cal_arap_error(
    nodes_sequence, ii, jj, nn, K=10, weight=None, sample_num=512, reference_pose=None
):
    # input: nodes_sequence: [Nt, Nv, 3]; ii, jj, nn: [Ne,], weight: [Nv, K]
    # output: arap error: float
    Nt, Nv, _ = nodes_sequence.shape
    # laplacian_mat = cal_laplacian(Nv, ii, jj, nn)  # [Nv, Nv]
    # laplacian_mat_inv = invert_matrix(laplacian_mat)
    arap_error = 0
    if weight is None:
        weight = torch.zeros(Nv, K).cuda()
        weight[ii, nn] = 1
    reference_pose = reference_pose if reference_pose is not None else nodes_sequence[0]
    source_edge_mat = produce_edge_matrix_nfmt(
        reference_pose, (Nv, K, 3), ii, jj, nn
    )  # [Nv, K, 3]
    sample_idx = torch.arange(Nv).cuda()
    if Nv > sample_num:
        sample_idx = torch.from_numpy(np.random.choice(Nv, sample_num)).long().cuda()
    else:
        source_edge_mat = source_edge_mat[sample_idx]
    weight = weight[sample_idx]
    for idx in range(1, Nt):
        # t1 = time.time()
        with torch.no_grad():
            rotation = estimate_rotation(
                reference_pose,
                nodes_sequence[idx],
                ii,
                jj,
                nn,
                K=K,
                weight=weight,
                sample_idx=sample_idx,
            )  # [Nv, 3, 3]
        # Compute energy
        target_edge_mat = produce_edge_matrix_nfmt(
            nodes_sequence[idx], (Nv, K, 3), ii, jj, nn
        )  # [Nv, K, 3]
        target_edge_mat = target_edge_mat[sample_idx]
        rot_rigid = torch.bmm(
            rotation, source_edge_mat[sample_idx].permute(0, 2, 1)
        ).permute(
            0, 2, 1
        )  # [Nv, K, 3]
        stretch_vec = target_edge_mat - rot_rigid  # stretch vector
        stretch_norm = torch.norm(stretch_vec, dim=2) ** 2  # norm over (x,y,z) space
        arap_error += (weight * stretch_norm).sum()
    return arap_error
