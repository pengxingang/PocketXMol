import numpy as np
import torch
from torch.nn import functional as F
from scipy.stats import vonmises, norm
from torch_scatter import scatter_add


def quat_1ijk_to_mat(quat):
    quat_1 = torch.ones(quat.shape[0], 1, device=quat.device)
    quat_1ijk = torch.cat([quat_1, quat], dim=-1)
    quat_1ijk = F.normalize(quat_1ijk, dim=-1)
    return quat_to_mat(quat_1ijk)


def quat_to_mat(quat):
    """
    quat: (N, 4), quaternion
    """
    # check norm
    assert torch.allclose(quat.norm(dim=-1), torch.ones_like(quat[:, 0])), 'quaternion is not normalized'
    
    R_mat = torch.zeros(quat.shape[0], 3, 3, device=quat.device)
    R_mat[:, 0, 0] = 1 - 2*quat[:, 2]**2 - 2*quat[:, 3]**2  # 1 - 2c^2 - 2d^2
    R_mat[:, 0, 1] = 2*quat[:, 1]*quat[:, 2] - 2*quat[:, 0]*quat[:, 3]  # 2bc - 2ad
    R_mat[:, 0, 2] = 2*quat[:, 1]*quat[:, 3] + 2*quat[:, 0]*quat[:, 2]  # 2bd + 2ac
    R_mat[:, 1, 0] = 2*quat[:, 1]*quat[:, 2] + 2*quat[:, 0]*quat[:, 3]  # 2bc + 2ad
    R_mat[:, 1, 1] = 1 - 2*quat[:, 1]**2 - 2*quat[:, 3]**2  # 1 - 2b^2 - 2d^2
    R_mat[:, 1, 2] = 2*quat[:, 2]*quat[:, 3] - 2*quat[:, 0]*quat[:, 1]  # 2cd - 2ab
    R_mat[:, 2, 0] = 2*quat[:, 1]*quat[:, 3] - 2*quat[:, 0]*quat[:, 2] # 2bd - 2ac
    R_mat[:, 2, 1] = 2*quat[:, 2]*quat[:, 3] + 2*quat[:, 0]*quat[:, 1] # 2cd + 2ab
    R_mat[:, 2, 2] = 1 - 2*quat[:, 1]**2 - 2*quat[:, 2]**2  # 1 - 2b^2 - 2c^2
    return R_mat


# def apply_torsional_rotation_multiple_domains(position, edge_index,
#                                               tor_edge, tor_angle, i_bond_for_tor_edge,
#                                               twisted_edge, i_tor_for_twisted_edge):
def apply_torsional_rotation_multiple_domains(positions, tor_order, tor_bonds, tor_angles,
                                                twisted_nodes, index_tor):
    """
    positions: (N, 3)
    tor_order: (n_rot,)
    tor_bonds: (n_rot[able bonds], 2)
    tor_angles: (n_rot,)
    twisted_nodes: (T,)
    index_tor: (T,)
    """
    # i_bond_for_twisted_edge = i_bond_for_tor_edge[i_tor_for_twisted_edge]
    twisted_order = tor_order[index_tor]
    largest_order = tor_order.max() + 1
    for curr_order in range(largest_order):
        # fetch torsional pairs related to current order
        ind_curr_tor = (tor_order == curr_order)  # (n_tor_edge)
        tor_bonds_curr = tor_bonds[ind_curr_tor]  # (sub_n_tor_edge)
        angles_curr = tor_angles[ind_curr_tor]  # (sub_n_tor_edge)
        
        # fetch twisted pairs related to current
        ind_curr_twisted = (twisted_order == curr_order)  # (n_twisted_edge)
        twisted_nodes_curr = twisted_nodes[ind_curr_twisted]  # (sub_n_twisted_edge)
        index_tor_curr = index_tor[ind_curr_twisted]  # (sub_n_twisted_edge)
        index_tor_curr -= (~ind_curr_tor).cumsum(dim=0)[index_tor_curr]
        assert index_tor_curr.unique().shape[0] == index_tor_curr.max()+1, 'index_tor_curr is wrong'
        positions = apply_torsional_rotation(positions, tor_bonds_curr, angles_curr,
                                            twisted_nodes_curr, index_tor_curr)
    return positions


# def apply_torsional_rotation(positions, edge_index, tor_edge, tor_angle,
#                              twisted_edge, i_tor_for_twisted):
def apply_torsional_rotation(positions, tor_bonds, tor_angles,
                             twisted_nodes, index_tor):
    """
    positions: (N, 3)
    tor_bonds: (2, n_rot[able bonds])
    tor_angles: (n_rot,)
    twisted_nodes: (T,)
    index_tor: (T,)
    """
    # # get nodes
    node_tor_left, node_tor_right = tor_bonds.T  # (n_rot,)
    assert (twisted_nodes.unique().shape[0] == twisted_nodes.shape[0]), 'twisted node appears in over one torsion'
    
    # # get the positions of nodes and vectors of edges
    pos_tor_left = positions[node_tor_left]  # (n_rot, 3)
    pos_tor_right = positions[node_tor_right]  # (n_rot, 3)
    pos_twisted = positions[twisted_nodes]  # (T, 3)
    
    vec_tor_edge = (pos_tor_left - pos_tor_right)  # (n_rot, 3)
    unit_tor_edge_expand = F.normalize(vec_tor_edge, dim=-1)[index_tor]  # (T, 3)
    vec_twisted_edge = pos_twisted - pos_tor_left[index_tor]  # (T, 3)
    
    # # calculate rotation-related parameters
    rot_axes = unit_tor_edge_expand  # (T, 3)
    rot_angles = tor_angles[index_tor]  # (T, )
    radius_vec = vec_twisted_edge - unit_tor_edge_expand * \
        (vec_twisted_edge * unit_tor_edge_expand).sum(-1, keepdims=True)  # (T, 3)
    rot_center = pos_twisted - radius_vec  # (T, 3)
    
    # # apply rotation
    pos_twisted_rot = apply_axis_angle_rotation(
        radius_vec, rot_axes, rot_angles) + rot_center  # (T, 3)
    
    positions_new = positions.clone()
    positions_new[twisted_nodes] = pos_twisted_rot
    return positions_new


def apply_axis_angle_rotation(positions, rot_axes, rot_angles):
        """
        Apply Rodrigues rotation formula.
        rotate position vectors by angle `rot_angles` around axes `rot_axes`.
        positions: (N, 3)
        rot_axes: (N, 3), NOTE: must be unit vectors
        rot_angles: (N, 1), (N,) or (N, 2) as (sin, cos)
        """
        if rot_angles.dim() < rot_axes.dim():
            rot_angles = rot_angles[:, None]
        
        if rot_angles.size(-1) == 2:
            sin_angle = rot_angles[..., 0:1]
            cos_angle = rot_angles[..., 1:2]
        else:
            sin_angle = torch.sin(rot_angles)
            cos_angle = torch.cos(rot_angles)
        rot_vec = (
            positions * cos_angle + 
            torch.linalg.cross(rot_axes, positions) * sin_angle +
            rot_axes * (rot_axes * positions).sum(-1, keepdims=True) * (1 - cos_angle)
        )
        return rot_vec


def sample_uniform_angle(sigmas):
    angles = torch.rand_like(sigmas) * 2 * torch.pi - torch.pi
    return angles


def robust_sample_angle(sigmas, sigma_th=0.1):
    # # sample from gaussian distribution
    samples = torch.randn_like(sigmas) * sigmas
    samples = samples.clamp(-torch.pi, torch.pi)
    # # sample from von mises distribution
    from_vonmises = (sigmas > sigma_th)
    if from_vonmises.any():
        kappa = (1 / sigmas[from_vonmises]**2)
        if isinstance(kappa, torch.Tensor):
            kappa = kappa.cpu().numpy()
        samples_vonmises = vonmises(kappa=kappa).rvs()
        # combine samples
        samples[from_vonmises] = torch.tensor(samples_vonmises, dtype=samples.dtype, device=samples.device)
    return samples

class RobustAngleSO3Distribution(torch.nn.Module):
    def __init__(self, sigma_th=4.e-3, n_bins=1000, n_L=1001):
        super().__init__()
        self.sigma_th = sigma_th
        self.n_bins = n_bins
        self.n_L = n_L
        
        bin_width = torch.pi / n_bins
        bins = torch.linspace(0, torch.pi, n_bins+1)[:-1] + bin_width / 2
        bins_expand = bins[None, :, None] # (1, n_bins, 1)
        
        ls = torch.arange(n_L)
        ls = ls[None, None, :]  # (1, 1, n_L)
        
        c0 = ((1 - torch.cos(bins_expand)) / torch.pi).squeeze(-1)
        c2 = (2*ls+1) * torch.sin((ls+0.5) * bins_expand) / torch.sin(bins_expand/2)
        
        self.bin_width = bin_width
        self.register_buffer('ls', ls)
        self.register_buffer('bins', bins)
        self.register_buffer('c0', c0)
        self.register_buffer('c2', c2)

    @torch.no_grad()
    def sample(self, sigma, is_uniform=False):
        """
        sigm: N
        """
        n = len(sigma)
        sigma_expand = sigma[:, None, None]  # (N_sigma, 1, 1)

        # angle distribution
        if not is_uniform:
            c1 = torch.exp(-self.ls*(self.ls+1) * (sigma_expand**2))
            probs = self.c0 * (c1 * self.c2).sum(-1) # (N, n_bins)
        else:
            probs = self.c0.repeat_interleave(n, dim=0) # (N, n_bins)
        probs = probs.clamp(min=0)
        idx_bins = torch.multinomial(probs, num_samples=1).squeeze(-1) # (N,)
        angles = self.bins[idx_bins] # (N,)
        angles = angles + self.bin_width * (torch.rand_like(angles) - 0.5)
        
        if not is_uniform:
            # gaussian distribution as approximation when sigma is too small
            # mean = 2 * sigma, std = sigma
            idx_gaussian = sigma < self.sigma_th
            angles[idx_gaussian] = (sigma[idx_gaussian] * 2 +
                torch.randn_like(angles[idx_gaussian]) * sigma[idx_gaussian])
        return angles
    
