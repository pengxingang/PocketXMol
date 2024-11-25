import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Sequential, Linear, Conv1d, ModuleList
from torch_scatter import scatter_sum, scatter_softmax
from torch_geometric.nn import knn
try:
    from models.common import GaussianSmearing, MLP, NONLINEARITIES
except:
    import sys
    sys.path.append('.')
    from models.common import GaussianSmearing, MLP, NONLINEARITIES


class GVP(nn.Module):
    '''
    from https://github.com/drorlab/gvp-pytorch/blob/main/gvp/__init__.py
    Geometric Vector Perceptron. See manuscript and README.md
    for more details.
    
    :param in_dims: tuple (n_scalar, n_vector)
    :param out_dims: tuple (n_scalar, n_vector)
    :param h_dim: intermediate number of vector channels, optional
    :param activations: tuple of functions (scalar_act, vector_act)
    :param vector_gate: whether to use vector gating.
                        (vector_act will be used as sigma^+ in vector gating if `True`)
    '''
    def __init__(self, in_dims, out_dims, h_dim=None,
                 activations=(F.relu, torch.sigmoid), vector_gate=True):
        super(GVP, self).__init__()
        self.si, self.vi = in_dims
        self.so, self.vo = out_dims
        self.vector_gate = vector_gate
        if self.vi: 
            self.h_dim = h_dim or max(self.vi, self.vo) 
            self.wh = nn.Linear(self.vi, self.h_dim, bias=False)
            self.ws = nn.Linear(self.h_dim + self.si, self.so)
            if self.vo:
                self.wv = nn.Linear(self.h_dim, self.vo, bias=False)
                if self.vector_gate: self.wsv = nn.Linear(self.so, self.vo)
        else:
            self.ws = nn.Linear(self.si, self.so)
        
        self.scalar_act, self.vector_act = activations
        # self.dummy_param = nn.Parameter(torch.empty(0))
        
    def forward(self, x):
        '''
        :param x: tuple (s, V) of `torch.Tensor`, 
                  or (if vectors_in is 0), a single `torch.Tensor`
        :return: tuple (s, V) of `torch.Tensor`,
                 or (if vectors_out is 0), a single `torch.Tensor`
        '''
        if self.vi:
            s_in, v_in = x
            # v = torch.transpose(v, -1, -2).contiguous()
            vh = self.wh(v_in.transpose(-1, -2).contiguous())
            vn = _norm_no_nan(vh, axis=-2)
            s = self.ws(torch.cat([s_in, vn], -1))
            if self.vo: 
                v = self.wv(vh).transpose(-1, -2).contiguous()
                # v = torch.transpose(v, -1, -2).contiguous()
                if self.vector_gate: 
                    if self.vector_act:
                        gate = self.wsv(self.vector_act(s))
                    else:
                        gate = self.wsv(s)
                    v = v * torch.sigmoid(gate).unsqueeze(-1).contiguous()
                elif self.vector_act:
                    v = v * self.vector_act(
                        _norm_no_nan(v, axis=-1, keepdims=True))
        else:
            s = self.ws(x)
            if self.vo:
                v = torch.zeros(s.shape[0], self.vo, 3,
                                device=self.dummy_param.device)
        if self.scalar_act:
            s = self.scalar_act(s)
        
        return (s, v) if self.vo else s

def _norm_no_nan(x, axis=-1, keepdims=False, eps=1e-5, sqrt=True):
    '''
    L2 norm of tensor clamped above a minimum value `eps`.
    
    :param sqrt: if `False`, returns the square of the L2 norm
    '''
    out = torch.clamp(torch.sum(torch.square(x), axis, keepdims), min=eps)
    return torch.sqrt(out) if sqrt else out
    # return torch.linalg.norm(x, ord=2, dim=axis, keepdim=keepdims)

class GVLinear(GVP):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, activations=(None, torch.sigmoid))
        


class Fuser(Module):
    def __init__(self, type_):
        super().__init__()
        self.type_ = type_
        
    def forward(self, left, right, left_index=None, right_index=None):
        if left_index is not None:
            left = (left[0][left_index], left[1][left_index])
        if right_index is not None:
            right = (right[0][right_index], right[1][right_index])
        scalar = left[0] * right[0]
        dot_prod = torch.sum(left[1] * right[1], dim=-1)
        scalar = torch.cat([scalar, dot_prod], dim=-1)
        if self.type_ == 'cross':
            # vector = left[1]  # torch.cat([left[1], right[1]], dim=-2)
            vector = (left[1] + right[1]) / 2
            dtype = vector.dtype
            # use float to calculate cross product
            cross_prod = torch.cross(left[1].float(), right[1].float(), dim=-1).to(dtype)
            vector = torch.cat([vector, cross_prod], dim=-2)
        elif self.type_ == 'concat':
            vector = torch.cat([left[1], right[1]], dim=-2)
        else:
            raise NotImplementedError('Fuser type not implemented')
        return scalar, vector
        
    def get_fuser_dim(self, hidden):
        if self.type_ in ['cross', 'concat']:
            return hidden[0] + hidden[1], hidden[1] * 2
            # return hidden[0] + hidden[1], hidden[1]
        
    def scatter_sum(self, inputs, index, *args, **kwargs):
        scalar, vector = inputs
        scalar_out = scatter_sum(scalar, index, *args, **kwargs)
        vector_out = scatter_sum(vector, index, *args, **kwargs)
        return scalar_out, vector_out
    
    def index(self, inputs, index):
        scalar, vector = inputs
        scalar_out = scalar[index]
        vector_out = vector[index]
        return scalar_out, vector_out
    
    def add(self, *inputs):
        outputs = inputs[0]
        for inp in inputs[1:]:
            outputs = (outputs[0] + inp[0], outputs[1] + inp[1])
        return outputs

class NodeBlockGVP(Module):

    def __init__(self, node_dims, edge_dims, hidden_dims, fuse_type='cross', right_node_dims=(0, 0)):
        super().__init__()
        self.node_dims = node_dims
        
        in_node_dims = right_node_dims if right_node_dims[0] > 0 else node_dims
        self.node_net = nn.Sequential(
            GVP(in_node_dims, hidden_dims),
            GVLinear(hidden_dims, hidden_dims)
        )
        self.edge_net = nn.Sequential(
            GVP(edge_dims, hidden_dims),
            GVLinear(hidden_dims, hidden_dims)
        )
        
        self.fuser = Fuser(fuse_type)
        self.msg_net = GVLinear(self.fuser.get_fuser_dim(hidden_dims), hidden_dims)
    
        self.centroid_lin = GVLinear(node_dims, hidden_dims)
        fusion_dim = self.fuser.get_fuser_dim(hidden_dims)
        self.layer_norms_sca = nn.LayerNorm(node_dims[0])
        self.layer_norms_vec = nn.LayerNorm(node_dims[1], elementwise_affine=False)
        self.out_transform = nn.Sequential(
            GVP(fusion_dim, hidden_dims),
            GVLinear(hidden_dims, node_dims)
        )

    def forward(self, x, edge_index, edge_attr, x_right=None):
        """
        Args:
            x:  Node features, ((N, H), (N, H', 3))
            edge_index: (2, E).
            edge_attr:  ((E, H), (E, H', 3))
        """
        N = x[0].size(0)
        row, col = edge_index   # (E,) , (E,)


        # Compose messages
        h_edge = self.edge_net(edge_attr)  # (E, H_per_head)
        
        x_right = x_right if x_right is not None else x
        h_node_right = self.node_net(x_right)
        msg_j = self.msg_net(self.fuser(h_edge, h_node_right, right_index=col))

        # Aggregate messages
        aggr_msg = self.fuser.scatter_sum(msg_j, row, dim=0, dim_size=N)
        out = self.fuser(self.centroid_lin(x), aggr_msg)
        # out = self.centroid_lin(x) + aggr_msg

        out = self.out_transform(out)
        out = self.fuser.add(out, x)
        out = (self.layer_norms_sca(out[0]),
               self.layer_norms_vec(out[1].transpose(-1, -2)).transpose(-1, -2))
        return out



class BondFFNGVP(Module):
    def __init__(self, bond_dims, node_dims, inter_dims, fuse_type, out_dims=None):
        super().__init__()
        out_dims = bond_dims if out_dims is None else out_dims
        self.fuse_type = fuse_type
        self.fuser = Fuser(fuse_type)
        
        self.bond_linear = GVLinear(bond_dims, inter_dims)
        self.node_linear = GVLinear(node_dims, inter_dims)
        self.inter_module = nn.Sequential(
            GVP(self.fuser.get_fuser_dim(inter_dims), inter_dims),
            GVLinear(inter_dims, out_dims)
        )

    def forward(self, bond_feat_input, node_feat_input, node_index):
        bond_feat = self.bond_linear(bond_feat_input)
        node_feat = self.node_linear(node_feat_input)
        inter_feat = self.fuser(bond_feat, node_feat, right_index=node_index)
        inter_feat = self.inter_module(inter_feat)
        return inter_feat



class EdgeBlockGVP(Module):
    def __init__(self, edge_dims, node_dims, hidden_dims, fuse_type):
        super().__init__()
        self.fuser = Fuser(fuse_type)
        inter_dims = (edge_dims[0]*2, edge_dims[1]*2) if hidden_dims is None else hidden_dims

        self.bond_ffn_left = BondFFNGVP(edge_dims, node_dims, inter_dims=inter_dims, fuse_type=fuse_type)
        self.bond_ffn_right = BondFFNGVP(edge_dims, node_dims, inter_dims=inter_dims, fuse_type=fuse_type)

        self.node_ffn_left = GVLinear(node_dims, edge_dims)
        self.node_ffn_right = GVLinear(node_dims, edge_dims)

        self.self_ffn = GVLinear(edge_dims, edge_dims)
        self.layer_norm_sca = nn.LayerNorm(edge_dims[0])
        self.layer_norm_vec = nn.LayerNorm(edge_dims[1], elementwise_affine=False)
        self.out_transform = nn.Sequential(
            GVP(edge_dims, edge_dims),
            GVLinear(edge_dims, edge_dims)
        )
        # self.act = nn.ReLU()

    def forward(self, h_bond, bond_index, h_node):
        """
        h_bond: (b, bond_dim)
        bond_index: (2, b)
        h_node: (n, node_dim)
        """
        N = h_node[0].size(0)
        left_node, right_node = bond_index

        h_bond_in = h_bond
        # message from neighbor bonds
        msg_bond_left = self.bond_ffn_left(h_bond, h_node, node_index=left_node)
        msg_bond_left = self.fuser.scatter_sum(msg_bond_left, right_node, dim=0, dim_size=N)
        msg_bond_left = self.fuser.index(msg_bond_left, left_node)

        msg_bond_right = self.bond_ffn_right(h_bond, h_node, node_index=right_node)
        msg_bond_right = self.fuser.scatter_sum(msg_bond_right, left_node, dim=0, dim_size=N)
        msg_bond_right = self.fuser.index(msg_bond_right, right_node)
        
        h_bond = self.fuser.add(
            msg_bond_left, msg_bond_right, 
            self.node_ffn_left(self.fuser.index(h_node, left_node)), 
            self.node_ffn_right(self.fuser.index(h_node, right_node)),
            self.self_ffn(h_bond)
        )
        h_bond = self.out_transform(h_bond)

        h_bond = self.fuser.add(h_bond_in, h_bond)
        h_bond = (self.layer_norm_sca(h_bond[0]),
                  self.layer_norm_vec(h_bond[1].transpose(-1, -2)).transpose(-1, -2))
        return h_bond


class ContextNodeEdgeNetGVP(Module):
    def __init__(self, node_dim, edge_dim, num_blocks, 
                 node_dim_vec, edge_dim_vec, dist_cfg,
                 context_dim=0, context_dim_vec=0, context_cfg=None, 
                 node_only=False, fuse_type='cross', **kwargs):
        super().__init__()
        self.node_dims = [node_dim, node_dim_vec]
        self.edge_dims = [edge_dim, edge_dim_vec]
        self.ctx_dims = [context_dim, context_dim_vec]
        self.num_blocks = num_blocks
        self.dist_cfg = dist_cfg
        self.fuse_type = fuse_type
        self.kwargs = kwargs
        self.fuser = Fuser(fuse_type)
        self.pos_last = kwargs.get('pos_last', False)

        # edge distance expansion and dim
        self.distance_expansion = GaussianSmearing(**dist_cfg)
        num_gaussians = dist_cfg['num_gaussians']
        self.node_only = node_only
        if node_only:
            input_edge_dims = (num_gaussians, 2)
        else:
            input_edge_dims = (num_gaussians + self.edge_dims[0], self.edge_dims[1]+2)
        

        # for context
        self.context_cfg = context_cfg
        if context_cfg is not None:
            context_edge_dims = (context_cfg['edge_dim'], context_cfg['edge_dim_vec'])
            self.knn = context_cfg['knn']
            self.dist_exp_ctx = GaussianSmearing(**context_cfg['dist_cfg'])
            input_context_edge_dims = (context_cfg['dist_cfg']['num_gaussians'], 2)
            assert context_dim >0, 'context_dim should be larger than 0'
            assert not node_only, 'node_only should be False when context is used'
        
        # node network
        self.node_blocks_with_edge = ModuleList()
        self.edge_embs = ModuleList()
        if not node_only:
            self.edge_blocks = ModuleList()
            self.pos_blocks = ModuleList()
            if self.context_cfg is not None:
                self.ctx_node_blocks = ModuleList()
                self.ctx_edge_embs = ModuleList()
                self.ctx_pos_blocks = ModuleList()
        for i in range(num_blocks):
            self.node_blocks_with_edge.append(NodeBlockGVP(
                node_dims=self.node_dims, edge_dims=self.edge_dims, hidden_dims=self.node_dims,
                fuse_type=fuse_type
            ))
            self.edge_embs.append(GVLinear(input_edge_dims, self.edge_dims))
            if self.node_only:
                continue
            self.edge_blocks.append(EdgeBlockGVP(
                edge_dims=self.edge_dims, node_dims=self.node_dims, hidden_dims=None,
                fuse_type=fuse_type
            ))
            self.pos_blocks.append(PosUpdateGVP(
                node_dims=self.node_dims, edge_dims=self.edge_dims, hidden_dims=self.node_dims,
                fuse_type=fuse_type
            ) if (not self.pos_last) or (i == num_blocks-1) else None)
            if self.context_cfg is not None:
                self.ctx_node_blocks.append(NodeBlockGVP(
                    node_dims=self.node_dims, edge_dims=context_edge_dims, hidden_dims=self.node_dims,
                    fuse_type=fuse_type, right_node_dims=self.ctx_dims
                ))
                self.ctx_edge_embs.append(GVLinear(input_context_edge_dims, context_edge_dims))
                self.ctx_pos_blocks.append(PosUpdateGVP(
                    self.node_dims, context_edge_dims, hidden_dims=self.node_dims,
                    fuse_type=fuse_type
                ) if (not self.pos_last) or (i == num_blocks-1) else None)

        if not self.node_only:
            # last layers
            self.node_last = GVLinear(self.node_dims, [self.node_dims[0], 0])
            self.edge_last = GVLinear(self.edge_dims, [self.edge_dims[0], 0])

    def forward(self, h_node, pos_node, h_edge, edge_index,
                node_extra, edge_extra, batch_node=None,
                h_ctx=None, pos_ctx=None, batch_ctx=None):
        # make vector features
        device = h_node.device
        h_node_vec = torch.zeros([h_node.size(0), self.node_dims[1], 3], dtype=h_node.dtype, device=device)
        h_node = (h_node, h_node_vec)
        if not self.node_only:
            h_edge_vec = torch.zeros([h_edge.size(0), self.edge_dims[1], 3], dtype=h_edge.dtype, device=device)
            h_edge = (h_edge, h_edge_vec)
        
        for i in range(self.num_blocks):
            # edge fetures before each block
            if (i==0) or (not self.node_only):
                h_edge_dist = self._build_edges_dist(pos_node, edge_index)
            if not self.node_only:
                h_edge_sca = torch.cat([h_edge[0], h_edge_dist[0]], dim=-1)
                h_edge_vec = torch.cat([h_edge[1], h_edge_dist[1]], dim=-2)
                h_edge = (h_edge_sca, h_edge_vec)
            else:
                h_edge = h_edge_dist
            h_edge = self.edge_embs[i](h_edge)

            # edge with context
            if h_ctx is not None:
                h_ctx_edge, ctx_knn_edge_index = self._build_context_edges_dist(
                    pos_node, pos_ctx, batch_node, batch_ctx)
                h_ctx_edge = self.ctx_edge_embs[i](h_ctx_edge)
            # else:
            #     h_ctx_edge = None
            #     ctx_knn_edge_index = None
                
            # node feature updates
            h_node = self.node_blocks_with_edge[i](h_node, edge_index, h_edge,)
            # h_node = self.fuser.add(h_node, h_node_update)
            if self.node_only:
                continue
            h_node = self.ctx_node_blocks[i](h_node, ctx_knn_edge_index, h_ctx_edge, h_ctx)
            # h_node = self.fuser.add(h_node, h_node_update_ctx)

            # edge feature updates
            # h_edge = self.fuser.add(h_edge, self.edge_blocks[i](h_edge, edge_index, h_node,))
            h_edge = self.edge_blocks[i](h_edge, edge_index, h_node,)
            # h_edge = self.fuser.add(h_edge, h_edge_update)

            # pos updates
            if (self.pos_last) and (i != self.num_blocks-1):
                continue
            pos_node = (pos_node + self.pos_blocks[i](h_node, h_edge, edge_index)
                            + self.ctx_pos_blocks[i](h_node, h_ctx_edge, ctx_knn_edge_index))

        if self.node_only:
            return h_node
        else:
            h_node = self.node_last(h_node)
            h_edge = self.edge_last(h_edge)
            return h_node, pos_node, h_edge

    def _build_edges_dist(self, pos, edge_index):
        # distance
        relative_vec = pos[edge_index[0]] - pos[edge_index[1]]
        distance = torch.norm(relative_vec, dim=-1, p=2)
        edge_dist = self.distance_expansion(distance)

        edge_unit = relative_vec / (distance.unsqueeze(-1) + 1.e-8)
        edge_vec = edge_unit / (distance.unsqueeze(-1) + 1)
        edge_vec = torch.stack([edge_vec, edge_unit], dim=-2)
        return (edge_dist, edge_vec)

    def _build_context_edges_dist(self, pos, pos_ctx, batch_node, batch_ctx):
        # build knn edge index
        ctx_knn_edge_index = knn(y=pos, x=pos_ctx, k=self.knn,
                                batch_x=batch_ctx, batch_y=batch_node)

        relative_vec = pos[ctx_knn_edge_index[0]] - pos_ctx[ctx_knn_edge_index[1]]
        distance = torch.norm(relative_vec, dim=-1, p=2)
        edge_dist = self.dist_exp_ctx(distance)

        edge_unit = relative_vec / (distance.unsqueeze(-1) + 1.e-3)
        edge_vec = edge_unit / (distance.unsqueeze(-1) + 3.)
        edge_vec = torch.stack([edge_vec, edge_unit], dim=-2)
        return (edge_dist, edge_vec), ctx_knn_edge_index


class PosUpdateGVP(Module):
    def __init__(self, node_dims, edge_dims, hidden_dims, fuse_type):
        super().__init__()
        self.fuser = Fuser(fuse_type)
        self.edge_lin = nn.Sequential(
            GVP(edge_dims, node_dims),
            GVLinear(node_dims, node_dims)
        )
        self.node_lin = nn.Sequential(
            GVP(self.fuser.get_fuser_dim(node_dims), hidden_dims),
            GVLinear(hidden_dims, (1, 1))
        )

    def forward(self, h_node, h_edge, edge_index):
        row, col = edge_index
        h_edge = self.edge_lin(h_edge)
        msg = self.fuser.scatter_sum(h_edge, row, dim=0, dim_size=h_node[0].shape[0])
        
        h_node = self.fuser(h_node, msg)
        delta_pos = self.node_lin(h_node)[1]
        return delta_pos.squeeze(-2)



if __name__ == '__main__':
    node_dim = 128
    edge_dim = 64
    node_dim_vec = 64
    edge_dim_vec = 32
    update_edge = True
    device = 'cuda'
    # test the NodeEdgeNetGVP
    model = NodeEdgeNetGVP(
        node_dim=node_dim, edge_dim=edge_dim, num_blocks=6, cutoff=20, 
        node_dim_vec=node_dim_vec, edge_dim_vec=edge_dim_vec, fuse_type='cross', update_edge=update_edge
    ).to(device)
    print('Num of trainable parameters: ', sum(p.numel() for p in model.parameters() if p.requires_grad))
    n_nodes = 100
    n_edges = 1000
    h_node = torch.randn(n_nodes, node_dim).to(device)
    pos_node = torch.randn(n_nodes, 3).to(device)
    edge_index = torch.randint(0, n_nodes, (2, n_edges)).to(device)
    h_edge = torch.randn(n_edges, edge_dim).to(device)
    out = model(h_node, pos_node, h_edge, edge_index)
    print(out)