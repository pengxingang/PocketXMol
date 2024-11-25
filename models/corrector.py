import torch
import torch.nn as nn
import torch.linalg as la
from torch.nn import functional as F
from torch_scatter import scatter_mean, scatter_sum

try:
    from utils.motion import apply_torsional_rotation_multiple_domains
except:
    pass


def grad_len_to_pos(pos, edge_index, config):
    pos_left = pos[edge_index[0]]
    pos_right = pos[edge_index[1]]
    value_min, value_max = config['min'], config['max']
    value_std = config['std']
    lens = torch.linalg.norm(pos_left - pos_right, dim=-1).unsqueeze(-1)
    grad_min = 2 * (lens - value_min) / value_std**2 / lens * (pos_left - pos_right)
    grad_max = 2 * (lens - value_max) / value_std**2 / lens * (pos_left - pos_right)
    grad = torch.where(lens < value_min, grad_min, 
                       torch.where(lens > value_max, grad_max, 0))
    grad_pos = scatter_mean(grad, edge_index[0], dim=0, dim_size=pos.shape[0])
    return grad_pos
    
    
def correct_pos_by_fixed_dist_batch(batch, outputs, use_pos='in', config=None):
    if use_pos == 'in':
        pos_in = batch['pos_in'].detach().clone()
        fixed_halfdist = batch['fixed_halfdist'].bool()
    elif use_pos == 'gt':
        pos_in = batch['gt_node_pos'].detach().clone()
        fixed_halfdist = batch['fixed_halfdist_flex'].bool()
    pred_pos = outputs['pred_pos'].detach().clone()
    
    # get parameter
    if config is None:
        iters = 10
        lr = 0.5
        # lamb = 0
    else:
        iters = config['iters']
        lr = config['lr']

    # get bond index
    halfedge_index = batch['halfedge_index']
    edge_index = torch.cat([halfedge_index, halfedge_index.flip(0)], dim=-1)
    fixed_dist = torch.cat([fixed_halfdist, fixed_halfdist], dim=-1)
    fixed_edge_index = edge_index[:, fixed_dist]
    
    # fetch relevant pos
    fixed_pos = batch['fixed_pos'].bool()
    
    # spring the pos
    with torch.enable_grad():
        dist_in = torch.linalg.norm(pos_in[fixed_edge_index[0]] - pos_in[fixed_edge_index[1]], dim=-1)
        for i in range(iters):
            dist_pred = torch.linalg.norm(pred_pos[fixed_edge_index[0]] - pred_pos[fixed_edge_index[1]], dim=-1)
            grad = 2 * ((dist_pred - dist_in) / dist_in).unsqueeze(-1) * (pred_pos[fixed_edge_index[0]] - pred_pos[fixed_edge_index[1]])
            grad_pos = scatter_mean(grad, fixed_edge_index[0], dim=0, dim_size=pos_in.shape[0])
            # grad = torch.autograd.grad(loss, in_pos)[0]
            pred_pos[~fixed_pos] = pred_pos[~fixed_pos] - grad_pos[~fixed_pos] * lr

            loss = F.mse_loss(torch.linalg.norm(pred_pos[fixed_edge_index[0]] - pred_pos[fixed_edge_index[1]], dim=-1), dist_in)
            if loss.isnan():
                pred_pos = outputs['pred_pos'].detach().clone()
                break
    # print(loss)
    return pred_pos


@torch.no_grad()
def correct_pos_batch_no_tor(batch, outputs):
    pos_in = batch['pos_in']
    pred_pos = outputs['pred_pos']
    
    pred_pos_corr = correct_pos(
        no_tor=True,
        pos_in=pos_in.clone(), pos_out=pred_pos.clone(),
        sin_in=None, cos_in=None,
        sin_out=None, cos_out=None,
        domain_node_index=batch['domain_node_index'],
        # domain_center_nodes=batch['domain_center_nodes'],
        tor_bonds_anno=None,
        twisted_nodes_anno=None,
        dihedral_pairs_anno=None,
    )
    return pred_pos_corr

@torch.no_grad()
def correct_pos_batch(batch, outputs, use_pos='in'):
    if use_pos == 'in':
        pos_in = batch['pos_in']
    elif use_pos == 'gt':
        pos_in = batch['gt_node_pos']
    pred_pos = outputs['pred_pos']
    
    if 'dih_sin' not in outputs:
        sin_out, cos_out = get_dihedral_batch(pred_pos, batch['tor_bonds_anno'], batch['dihedral_pairs_anno'])
    else:
        sin_out, cos_out = outputs['dih_sin'], outputs['dih_cos']
    sin_in, cos_in = get_dihedral_batch(pos_in, batch['tor_bonds_anno'], batch['dihedral_pairs_anno'])
    pred_pos_corr = correct_pos(
        pos_in=pos_in.clone(), pos_out=pred_pos.clone(),
        sin_in=sin_in, cos_in=cos_in,
        sin_out=sin_out,
        cos_out=cos_out,
        domain_node_index=batch['domain_node_index'],
        # domain_center_nodes=batch['domain_center_nodes'],
        tor_bonds_anno=batch['tor_bonds_anno'],
        twisted_nodes_anno=batch['twisted_nodes_anno'],
        dihedral_pairs_anno=batch['dihedral_pairs_anno'],
    )
    return pred_pos_corr


@torch.no_grad()
def correct_pos(pos_in, pos_out, 
                sin_in, cos_in,
                sin_out, cos_out,
                
                domain_node_index,
                tor_bonds_anno,
                twisted_nodes_anno, dihedral_pairs_anno,
                no_tor=False):
    """
    Input:
        pos_in/out: (N, 3)
        domain_node_index: (2, N)
        tor_bonds_anno: (n_tor, 3)
        twisted_nodes_anno: (n_twisted, 2)
        dihedral_pairs_anno: (n_dih, 3)
    """
    
    if not no_tor:
        # # get dihedral torsion
        index_tor = dihedral_pairs_anno[:, 0]
        # tor = out - in, (n_dih, 2)
        cos_tor = cos_out * cos_in + sin_out * sin_in
        sin_tor = sin_out * cos_in - cos_out * sin_in
        angles_tri = torch.cat([sin_tor, cos_tor], dim=1)  # (n_dih, 2)
        angles_tri = scatter_mean(angles_tri, index_tor, dim=0)
        angles_tri = F.normalize(angles_tri, dim=-1)
        
        # # apply torsion to pos_in
        tor_order = tor_bonds_anno[:, 0]
        tor_bonds = tor_bonds_anno[:, 1:]
        index_tor_twisted = twisted_nodes_anno[:, 0]
        twisted_nodes = twisted_nodes_anno[:, 1]
        pos_tor = apply_torsional_rotation_multiple_domains(pos_in, 
                                tor_order, tor_bonds, angles_tri,
                                twisted_nodes, index_tor_twisted)
    else:
        pos_tor = pos_in.clone()
        
    # # global rotation and translation (use center_nodes). not around center
    domain_index, node_index = domain_node_index
    # pos_center_tor = pos_tor[domain_center_nodes]
    # pos_center_out = pos_out[domain_center_nodes]
    # global_rot, global_trans = kabsch_batch(pos_center_tor, pos_center_out)
    global_rot, global_trans = kabsch_flatten(pos_tor[node_index], pos_out[node_index], domain_index)
    
    # # apply global rotation and translation
    pos_corrected_expand = torch.matmul(
        pos_tor[node_index, None, :],
        global_rot.transpose(1, 2)[domain_index]
    ) + global_trans[domain_index]

    pos_corrected = pos_out.clone()
    pos_corrected[node_index] = pos_corrected_expand.squeeze(1)
    return pos_corrected
    



def get_dihedral_batch(pos, tor_bonds_anno, dihedral_pairs_anno):
    index_tor = dihedral_pairs_anno[:, 0]
    dihedral_ends = dihedral_pairs_anno[:, 1:]  # (n_dih, 2)
    dihedral_tor_nodes = tor_bonds_anno[:, 1:][index_tor]  # (n_dih, 2)

    sin, cos = get_dihedral(pos[dihedral_ends[:, 0]], pos[dihedral_tor_nodes[:, 0]],
                    pos[dihedral_tor_nodes[:, 1]], pos[dihedral_ends[:, 1]])
    return sin, cos


def get_dihedral(p0, p1, p2, p3):
    """from https://stackoverflow.com/questions/20305272/dihedral-torsion-angle-from-four-points-in-cartesian-coordinates-in-python
    Praxeolitic formula
    1 sqrt, 1 cross product"""
    b0 = p0 - p1
    # b1 = p2 - p1
    b1 = p1 - p2
    b2 = p3 - p2

    # normalize b1 so that it does not influence magnitude of vector
    # rejections that come next
    b1 = F.normalize(b1, dim=-1)

    # vector rejections
    # v = projection of b0 onto plane perpendicular to b1
    #   = b0 minus component that aligns with b1
    # w = projection of b2 onto plane perpendicular to b1
    #   = b2 minus component that aligns with b1
    v = b0 - (b0 * b1).sum(-1, keepdim=True) * b1
    w = b2 - (b2 * b1).sum(-1, keepdim=True) * b1
    v = F.normalize(v, dim=-1)
    w = F.normalize(w, dim=-1)

    # angle between v and w in a plane is the torsion angle
    # v and w is normalized
    cos = (v * w).sum(-1, keepdim=True)
    # sin = (torch.cross(v, w) * b1).sum()
    sin = (torch.cross(w, v) * b1).sum(-1, keepdim=True)
    return sin, cos


def kabsch_flatten(X, Y, domain_index):
    """
    Align X to Y using rigid transformation
    see https://en.wikipedia.org/wiki/Kabsch_algorithm
    X, Y: (N, 3)
    domain_index: (N,)
    """
    # Normalize the data by centering at the origin
    n_domain = domain_index.max() + 1
    X_mean = scatter_mean(X, domain_index, dim=0) # (n_domain, 3)
    Y_mean = scatter_mean(Y, domain_index, dim=0) # (n_domain, 3)
    X_centered = (X - X_mean[domain_index])  # (N, 3)
    Y_centered = (Y - Y_mean[domain_index])  # (N, 3)
    
    # Compute the covariance matrix
    covariance_matrix = torch.zeros([n_domain, 3, 3], device=X.device, dtype=X.dtype)  # (n_domain, 3, 3)
    for i_domain in range(n_domain):
        this_domain = (domain_index == i_domain)
        covariance_matrix[i_domain] = torch.matmul(X_centered[this_domain].transpose(0, 1),
                                                   Y_centered[this_domain])  # (3, 3)
    
    # Perform Singular Value Decomposition, use float64 to avoid numerical issue
    U, _, Vt = torch.linalg.svd(covariance_matrix.to(torch.float32)) # (n_domain, 3, 3)
    d = torch.sign(torch.det(torch.matmul(U, Vt).to(torch.float32)))  # (n_domain,)

    # Compute the rotation matrix
    diag_mat = torch.eye(3, device=X.device, dtype=d.dtype).unsqueeze(0).repeat(n_domain, 1, 1)
    diag_mat[:, -1, -1] = d # (n_domain, 3, 3)
    R = torch.matmul(torch.matmul(Vt.transpose(1, 2), diag_mat), U.transpose(1, 2)).to(X.dtype) # (n_domain, 3, 3)
    
    
    # Compute the translation vector
    # translation = Y_mean - X_mean
    # translation = Y_mean - torch.matmul(R, X_mean)  # (N, 1, 3)
    translation = Y_mean.unsqueeze(1) - torch.matmul(X_mean.unsqueeze(1), R.transpose(1, 2))  # (N, 1, 3)
    
    return R, translation


def kabsch_batch(X, Y):
    """
    Align X to Y using rigid transformation
    see https://en.wikipedia.org/wiki/Kabsch_algorithm
    X, Y: (B, N, 3)
    """
    # Normalize the data by centering at the origin
    X_mean = torch.mean(X, dim=1, keepdim=True) # (B, 1, 3)
    Y_mean = torch.mean(Y, dim=1, keepdim=True) # (B, 1, 3)
    X_centered = (X - X_mean)  # (B, N, 3)
    Y_centered = (Y - Y_mean)  # (B, 3, N)
    
    # Compute the covariance matrix
    covariance_matrix = torch.matmul(X_centered.transpose(1, 2), Y_centered)  # (B, 3, 3)
    
    # Perform Singular Value Decomposition, use float64 to avoid numerical issue
    U, _, Vt = torch.linalg.svd(covariance_matrix.to(torch.float32)) # (B, 3, 3)
    d = torch.sign(torch.det(torch.matmul(U, Vt).to(torch.float32)))

    # Compute the rotation matrix
    diag_mat = torch.eye(3, device=X.device, dtype=d.dtype).unsqueeze(0).repeat(X.shape[0], 1, 1)
    diag_mat[:, -1, -1] = d
    R = torch.matmul(torch.matmul(Vt.transpose(1, 2), diag_mat), U.transpose(1, 2)).to(X.dtype) # (B, 3, 3)
    
    
    # Compute the translation vector
    # translation = Y_mean - X_mean
    # translation = Y_mean - torch.matmul(R, X_mean)  # (N, 1, 3)
    translation = Y_mean - torch.matmul(X_mean, R.transpose(1, 2))  # (N, 1, 3)
    
    return R, translation


def procrustes_analysis_batch(X, Y):
    """
    Align X to Y using rigid transformation
    see https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem
    and https://en.wikipedia.org/wiki/Procrustes_analysis
    X, Y: (B, N, 3)
    """
    # Normalize the data by centering at the origin
    X_mean = torch.mean(X, dim=1, keepdim=True) # (B, 1, 3)
    Y_mean = torch.mean(Y, dim=1, keepdim=True) # (B, 1, 3)
    X_centered_data_T = (X - X_mean)  # (B, N, 3)
    Y_centered_data = (Y - Y_mean).transpose(1, 2)  # (B, 3, N)
    
    # Compute the covariance matrix
    covariance_matrix = torch.matmul(Y_centered_data, X_centered_data_T)  # (B, 3, 3)
    
    # Perform Singular Value Decomposition, use float64 to avoid numerical issue
    U, _, Vt = torch.linalg.svd(covariance_matrix.to(torch.float64)) # (B, 3, 3)
    
    # Compute the rotation matrix
    R = torch.matmul(U, Vt).to(X.dtype) # (B, 3, 3)
    
    # Compute the translation vector
    # translation = Y_mean - X_mean
    # translation = Y_mean - torch.matmul(R, X_mean)  # (N, 1, 3)
    translation = Y_mean - torch.matmul(X_mean, R.transpose(1, 2))  # (N, 1, 3)
    
    return R, translation


def procrustes_analysis_one_sample(X, Y):
    """
    Align X to Y using rigid transformation
    see https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem
    and https://en.wikipedia.org/wiki/Procrustes_analysis
    X, Y: (N, 3)
    """
    # Normalize the data by centering at the origin
    X_mean = torch.mean(X, dim=0) # (3,)
    Y_mean = torch.mean(Y, dim=0) # (3,)
    X_centered_data = (X - X_mean).T  # (3, N)
    Y_centered_data = (Y - Y_mean).T  # (3, N)
    
    # Compute the covariance matrix
    covariance_matrix = torch.matmul(Y_centered_data, X_centered_data.T)  # (3, 3)
    
    # Perform Singular Value Decomposition
    U, _, Vt = la.svd(covariance_matrix) # (3, 3)
    
    # Compute the rotation matrix
    R = torch.matmul(U, Vt) # (3, 3)
    
    # Compute the translation vector
    # translation = Y_mean - X_mean
    translation = Y_mean - torch.matmul(R, X_mean)  # (3,)
    
    return R, translation


if __name__ == '__main__':
    
    # Original points
    # X = torch.tensor([[1.0, 1.0, 1.0],
    #                 [2.0, 2.0, 2.0],
    #                 [3.0, 3.0, 3.0]])
    X = torch.tensor([[[1., 0, 0],
                      [1, 0, 2],
                      [0, 0, 2],
                      [0, -5, 0]],
                      [[1., 0, 0],
                      [1, 0, 2],
                      [0, 0, 2],
                      [0, -5, 0]]])

    # Transformed points
    Y = torch.tensor([[[0., 1, 0],
                      [0, 1, 2],
                      [0, 0, 2],
                      [5, 0, 0]],
                      [[0., 1, 0],
                      [0, 1, 2],
                      [0, 0, 2],
                      [5, 0, 0]]])
    # Y = torch.tensor([[2.0, 3.0, 0.0],
    #                   [3.0, 3.0, 1.0],
    #                   [4.0, 4.0, 2.0]])
    # Y = torch.tensor([[ 1.8453,  2.7560, -0.1547],
    #         [ 3.0000,  3.3333,  1.0000],
    #         [ 4.1547,  3.9107,  2.1547]])

    # Perform Procrustes analysis
    # R, translation = procrustes_analysis_batch(X, Y)

    # Perform Kabsch algorithm
    R, translation = kabsch_batch(X, Y)


    print("Rotation Matrix:")
    print(R)
    print("\nTranslation Vector:")
    print(translation)

    print('Apply transformation to X:')
    # print(torch.matmul(R, X.T).T + translation)
    Y_app = torch.matmul(X, R.transpose(1, 2)) + translation
    print(Y_app)
    
    print('Delta:')
    print((Y - Y_app).abs().sum())

            