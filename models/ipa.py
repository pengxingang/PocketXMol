import torch
import torch.nn as nn
from torch.nn import Module
import torch.nn.functional as F
import numpy as np
from torch_scatter import scatter_softmax, scatter_sum, scatter_mean
from torch_geometric.nn import knn

from models.common import MLP, GaussianSmearing
from utils.motion import quat_1ijk_to_mat


################ geometry

def global_to_local(R, t, q):
    """
    Description:
        Convert global (external) coordinates q to local (internal) coordinates p.
        p <- R^{T}(q - t)
    Args:
        R:  (N, 3, 3).
        t:  (N, 3).
        q:  Global coordinates, (N, ..., 3).
    Returns:
        p:  Local coordinates, (N, ..., 3).
    """
    q_size = q.size()
    assert q_size[-1] == 3
    N = q_size[0]

    q = q.reshape(N, -1, 3).transpose(-1, -2)   # (N, *, 3) -> (N, 3, *)
    p = torch.matmul(R.transpose(-1, -2), (q - t.unsqueeze(-1)))  # (N, 3, *)
    p = p.transpose(-1, -2).reshape(q_size)     # (N, 3, *) -> (N, *, 3) -> (N, ..., 3)
    return p


def local_to_global(R, t, p):
    """
    Description:
        Convert local (internal) coordinates to global (external) coordinates q.
        q <- Rp + t
    Args:
        R:  (N, 3, 3).
        t:  (N, 3).
        p:  Local coordinates, (N, ..., 3).
    Returns:
        q:  Global coordinates, (N, ..., 3).
    """
    p_size = p.size()
    assert p_size[-1] == 3
    N = p_size[0]

    p = p.view(N, -1, 3).transpose(-1, -2)   # (N, *, 3) -> (N, 3, *)
    q = torch.matmul(R, p) + t.unsqueeze(-1)    # (N, 3, *)
    q = q.transpose(-1, -2).reshape(p_size)     # (N, 3, *) -> (N, *, 3) -> (N, ..., 3)
    return q



def _alpha_from_logits(logits, edge_index):
    """
    Args:
        logits: Logit matrices, (E, num_heads).
        edge_index: Edge indices, (2, E).
    Returns:
        alpha:  Attention weights.
    """
    alpha = scatter_softmax(logits, edge_index[0], dim=0)  # (N, num_heads)
    return alpha


def _heads(x, n_heads, n_ch):
    """
    Args:
        x:  (..., num_heads * num_channels)
    Returns:
        (..., num_heads, num_channels)
    """
    s = list(x.size())[:-1] + [n_heads, n_ch]
    return x.view(*s)


class GABlock(nn.Module):

    def __init__(self, node_dim, edge_dim, value_dim=32, query_key_dim=32, num_query_points=8,
                 num_value_points=8, num_heads=12, bias=False):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.value_dim = value_dim
        self.query_key_dim = query_key_dim
        self.num_query_points = num_query_points
        self.num_value_points = num_value_points
        self.num_heads = num_heads

        # Node
        self.proj_query = nn.Linear(node_dim, query_key_dim * num_heads, bias=bias)
        self.proj_key = nn.Linear(node_dim, query_key_dim * num_heads, bias=bias)
        self.proj_value = nn.Linear(node_dim, value_dim * num_heads, bias=bias)

        # Pair
        self.proj_pair_bias = nn.Linear(edge_dim, num_heads, bias=bias)

        # Spatial
        self.spatial_coef = nn.Parameter(torch.full([1, self.num_heads], fill_value=np.log(np.exp(1.) - 1.)),
                                         requires_grad=True)
        self.proj_query_point = nn.Linear(node_dim, num_query_points * num_heads * 3, bias=bias)
        self.proj_key_point = nn.Linear(node_dim, num_query_points * num_heads * 3, bias=bias)
        self.proj_value_point = nn.Linear(node_dim, num_value_points * num_heads * 3, bias=bias)

        # Output
        self.out_transform = nn.Linear(
            in_features=(num_heads * edge_dim) + (num_heads * value_dim) + (
                    num_heads * num_value_points * (3 + 3 + 1)),
            out_features=node_dim,
        )

        self.layer_norm_1 = nn.LayerNorm(node_dim)
        self.mlp_transition = nn.Sequential(nn.Linear(node_dim, node_dim), nn.ReLU(),
                                            nn.Linear(node_dim, node_dim), nn.ReLU(),
                                            nn.Linear(node_dim, node_dim))
        self.layer_norm_2 = nn.LayerNorm(node_dim)

    def _node_logits(self, x, edge_index):
        query_l = _heads(self.proj_query(x), self.num_heads, self.query_key_dim)  # (N, n_heads, qk_ch)
        key_l = _heads(self.proj_key(x), self.num_heads, self.query_key_dim)  # (N, n_heads, qk_ch)
        logits_node = (query_l[edge_index[0]] * key_l[edge_index[1]] *
                        (1 / np.sqrt(self.query_key_dim))).sum(-1)  # (E, n_heads)
        return logits_node  # (E, n_heads)

    def _pair_logits(self, z):
        logits_pair = self.proj_pair_bias(z)
        return logits_pair

    def _spatial_logits(self, R, t, x, edge_index):
        N = x.size(0)

        # Query
        query_points = _heads(self.proj_query_point(x), self.num_heads * self.num_query_points,
                              3)  # (N, n_heads * n_pnts, 3)
        query_points = local_to_global(R, t, query_points)  # Global query coordinates, (N, n_heads * n_pnts, 3)
        query_s = query_points.reshape(N, self.num_heads, -1)  # (N, n_heads, n_pnts*3)

        # Key
        key_points = _heads(self.proj_key_point(x), self.num_heads * self.num_query_points,
                            3)  # (N, 3, n_heads * n_pnts)
        key_points = local_to_global(R, t, key_points)  # Global key coordinates, (N, n_heads * n_pnts, 3)
        key_s = key_points.reshape(N, self.num_heads, -1)  # (N, n_heads, n_pnts*3)

        # Q-K Product
        sum_sq_dist = ((query_s[edge_index[0]] - key_s[edge_index[1]]) ** 2).sum(-1)  # (E, n_heads)
        gamma = F.softplus(self.spatial_coef)
        logits_spatial = sum_sq_dist * ((-1 * gamma * np.sqrt(2 / (9 * self.num_query_points)))
                                        / 2)  # (E, n_heads)
        return logits_spatial

    def _pair_aggregation(self, alpha, z, edge_index, n_nodes):
        feat_p2n = scatter_sum(alpha.unsqueeze(-1) * z.unsqueeze(-2),
                               edge_index[0], dim=0, dim_size=n_nodes)  # (n_nodes, n_heads, C)
        return feat_p2n.reshape(n_nodes, -1)

    def _node_aggregation(self, alpha, x, edge_index, n_nodes):
        value_l = _heads(self.proj_value(x), self.num_heads, self.query_key_dim)  # (N, n_heads, v_ch)
        feat_node = alpha.unsqueeze(-1) * value_l[edge_index[1]]  # (E, n_heads, v_ch)
        feat_node = scatter_sum(feat_node, edge_index[0], dim=0, dim_size=n_nodes)  # (N, n_heads, v_ch)
        return feat_node.reshape(n_nodes, -1)

    def _spatial_aggregation(self, alpha, R, t, x, edge_index, n_nodes):
        N = x.size(0)
        value_points = _heads(self.proj_value_point(x), self.num_heads * self.num_value_points,
                              3)  # (N, n_heads * n_v_pnts, 3)
        value_points = local_to_global(R, t, value_points.reshape(N, self.num_heads, self.num_value_points,
                                                                  3))  # (N, n_heads, n_v_pnts, 3)
        aggr_points = alpha[..., None, None] * value_points[edge_index[1]]  # (E, n_heads, n_pnts, 3)
        aggr_points = scatter_sum(aggr_points, edge_index[0], dim=0, dim_size=n_nodes)  # (N, n_heads, n_pnts, 3)

        feat_points = global_to_local(R[:n_nodes], t[:n_nodes], aggr_points)  # (N, n_heads, n_pnts, 3)
        feat_distance = feat_points.norm(dim=-1)  # (N, n_heads, n_pnts)
        feat_direction = F.normalize(feat_points, dim=-1, eps=1e-8)  # (N, n_heads, n_pnts, 3)

        feat_spatial = torch.cat([
            feat_points.reshape(n_nodes, -1),
            feat_distance.reshape(n_nodes, -1),
            feat_direction.reshape(n_nodes, -1),
        ], dim=-1)

        return feat_spatial

    def forward(self, R, t, x, z, edge_index, n_nodes=None):
        """
        Args:
            R:  Frame basis matrices, (N, 3, 3_index).
            t:  Frame external (absolute) coordinates, (N, 3).
            x:  Node-wise features, (N, F).
            z:  Pair-wise features, (E, C).
            edge_index: Edge indices, (2, E).
        Returns:
            x': Updated node-wise features, (N, F).
        """
        n_nodes = x.size(0) if n_nodes is None else n_nodes
        # Attention logits
        logits_node = self._node_logits(x, edge_index)  # (E, n_heads)
        logits_pair = self._pair_logits(z)  # (E, n_heads)
        logits_spatial = self._spatial_logits(R, t, x, edge_index)  # (E, n_heads)
        
        # Summing logits up and apply `softmax`.
        logits_sum = logits_node + logits_pair + logits_spatial
        alpha = _alpha_from_logits(logits_sum * np.sqrt(1 / 3), edge_index)  # (N, n_heads)

        # Aggregate features
        feat_p2n = self._pair_aggregation(alpha, z, edge_index, n_nodes)
        feat_node = self._node_aggregation(alpha, x, edge_index, n_nodes)
        feat_spatial = self._spatial_aggregation(alpha, R, t, x, edge_index, n_nodes)

        # Finally
        feat_all = self.out_transform(torch.cat([feat_p2n, feat_node, feat_spatial], dim=-1))  # (N, F)
        x_updated = self.layer_norm_1(x[:n_nodes] + feat_all)
        x_updated = self.layer_norm_2(x_updated + self.mlp_transition(x_updated))
        return x_updated


class GAEncoder(nn.Module):

    def __init__(self, node_dim, num_blocks, ga_block_opt={}, dist_cfg={}, **kwargs):
        super(GAEncoder, self).__init__()
        self.num_blocks = num_blocks
        self.distance_expansion = GaussianSmearing(**dist_cfg)
        edge_dim = dist_cfg['num_gaussians']

        self.ga_blocks = nn.ModuleList()
        self.frame_update = nn.ModuleList()
        for _ in range(num_blocks):
            self.ga_blocks.append(GABlock(node_dim, edge_dim, **ga_block_opt))
            self.frame_update.append(MLP(node_dim, 3, node_dim//2))

    def forward(self, h_node, pos_node, edge_index, **kwargs):
        R = torch.eye(3, device=pos_node.device).unsqueeze(0).repeat(pos_node.size(0), 1, 1)
        for i in range(self.num_blocks):
            h_edge, relative_vec, distance = self._build_edges_dist(pos_node, edge_index)
            h_node = self.ga_blocks[i](R, pos_node, h_node, h_edge, edge_index)

            # apply frame update
            quaternion = self.frame_update[i](h_node)
            rot_mat = quat_1ijk_to_mat(quaternion)
            R = R @ rot_mat
            
        return h_node, R

    def _build_edges_dist(self, pos, edge_index):
        # distance
        relative_vec = pos[edge_index[0]] - pos[edge_index[1]]
        distance = torch.norm(relative_vec, dim=-1, p=2)
        h_dist = self.distance_expansion(distance)
        return h_dist, relative_vec, distance


class ContextGAEdgeNet(Module):
    def __init__(self, node_dim, edge_dim, context_dim, num_blocks, knn, dist_cfg, dist_cfg_ctx, ga_cfg={}, **kwargs):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.context_dim = context_dim
        self.num_blocks = num_blocks
        self.knn = knn
        self.kwargs = kwargs

        self.context_emb = nn.Linear(context_dim, node_dim)
        self.distance_expansion = GaussianSmearing(**dist_cfg)
        self.dist_exp_ctx = GaussianSmearing(**dist_cfg_ctx)

        self.edge_emb = nn.ModuleList()
        self.edge_emb_mol = nn.ModuleList()
        self.edge_emb_ctx = nn.ModuleList()
        self.ga_blocks = nn.ModuleList()
        self.frame_update = nn.ModuleList()
        for _ in range(num_blocks):
            self.edge_emb_mol.append(nn.Linear(edge_dim+dist_cfg['num_gaussians'], edge_dim))
            self.edge_emb_ctx.append(nn.Linear(dist_cfg_ctx['num_gaussians'], edge_dim))
            self.edge_emb.append(nn.Linear(edge_dim, edge_dim))
            self.ga_blocks.append(GABlock(node_dim, edge_dim, **ga_cfg))
            self.frame_update.append(MLP(node_dim, 3+3, node_dim//2))
        self.edge_out = MLP(edge_dim + 2*node_dim, edge_dim, edge_dim, num_layer=3)


    def forward(self, h_node, pos_node, h_edge, edge_index,
                node_extra, edge_extra, batch_node=None,
                h_ctx=None, pos_ctx=None, batch_ctx=None):
        """
        graph node/edge features
            h_node: (n_node, node_dim)
            pos_node: (n_node, 3)
            h_edge: (n_edge, edge_dim)
            edge_index: (2, n_edge)
            node_extra: (n_node, node_extra_dim)
            edge_extra: (n_edge, edge_extra_dim)
            batch_node: (n_node, )
        context node features
            h_ctx: (n_ctx, ctx_dim)
            pos_ctx: (n_ctx, 3)
            batch_ctx: (n_ctx, )
        Output:
            h_node: (n_node, node_dim)
            h_edge: (n_edge, edge_dim)
            pos_node: (n_node, 3)
        """
        # is_mol = torch.cat([
        #     torch.ones_like(batch_node, dtype=torch.bool),
        #     torch.zeros_like(batch_ctx, dtype=torch.bool),
        # ], dim=0)
        n_nodes = h_node.size(0)
        
        h_ctx, R_ctx = h_ctx
        h_ctx = self.context_emb(h_ctx)
        
        # initialize frame
        R_node = torch.eye(3, device=pos_node.device).unsqueeze(0).repeat(n_nodes, 1, 1)
        for i in range(self.num_blocks):
            # # prepare edge features and index
            h_dist_mol, relative_vec, distance = self._build_edges_dist(pos_node, edge_index)
            h_edge_mol = torch.cat([h_edge, h_dist_mol], dim=-1)
            h_edge_ctx, vec_ctx, dist_ctx, ctx_knn_edge_index = self._build_context_edges_dist(
                pos_node, pos_ctx, batch_node, batch_ctx)

            ctx_knn_edge_index_new = torch.stack([ctx_knn_edge_index[0], ctx_knn_edge_index[1]+n_nodes], dim=0)
            edge_index_all = torch.cat([edge_index, ctx_knn_edge_index_new], dim=-1)
            h_edge_all = self.edge_emb[i](torch.cat([
                self.edge_emb_mol[i](h_edge_mol),
                self.edge_emb_ctx[i](h_edge_ctx)],
            dim=0))
            
            # # node feature updates
            pos_all = torch.cat([pos_node, pos_ctx], dim=0)
            h_node_all = torch.cat([h_node, h_ctx], dim=0)
            R_all = torch.cat([R_node, R_ctx], dim=0)
            h_node = self.ga_blocks[i](R_all, pos_all, h_node_all, h_edge_all, edge_index_all, n_nodes)

            # # predict frame update
            update = self.frame_update[i](h_node)
            translation, quaternion = torch.split(update, [3, 3], dim=-1)
            rot_mat = quat_1ijk_to_mat(quaternion)

            # # apply frame update
            pos_node = pos_node + local_to_global(R_node, torch.zeros_like(pos_node), translation)
            R_node = R_node @ rot_mat
            
        # # pos processing 
        h_edge = self.edge_out(torch.cat([
            h_edge_all[:h_edge.size(0)], h_node[edge_index[0]], h_node[edge_index[1]],
        ], dim=-1))
        return h_node, pos_node, h_edge

    def _build_edges_dist(self, pos, edge_index):
        # distance
        relative_vec = pos[edge_index[0]] - pos[edge_index[1]]
        distance = torch.norm(relative_vec, dim=-1, p=2)
        h_dist = self.distance_expansion(distance)
        return h_dist, relative_vec, distance
    
    def _build_context_edges_dist(self, pos, pos_ctx, batch_node, batch_ctx):
        
        # build knn edge index
        ctx_knn_edge_index = knn(y=pos, x=pos_ctx, k=self.knn,
                                batch_x=batch_ctx, batch_y=batch_node)

        relative_vec = pos[ctx_knn_edge_index[0]] - pos_ctx[ctx_knn_edge_index[1]]
        distance = torch.norm(relative_vec, dim=-1, p=2)
        h_dist = self.dist_exp_ctx(distance)
        return h_dist, relative_vec, distance, ctx_knn_edge_index
        
