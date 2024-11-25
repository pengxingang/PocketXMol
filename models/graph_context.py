from itertools import permutations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Sequential, Linear, Conv1d, ModuleList
from torch_scatter import scatter_mean, scatter_sum, scatter_softmax
from torch_geometric.nn import radius_graph, knn_graph, knn, radius
from models.common import GaussianSmearing, MLP, NONLINEARITIES
from utils.motion import apply_axis_angle_rotation, apply_torsional_rotation_multiple_domains
from utils.data import edge_index_to_index_of_edge


class ContextNodeBlock(Module):

    def __init__(self, node_dim, edge_dim, hidden_dim, gate_dim,
                 context_dim=0, context_edge_dim=0, layernorm_before=False):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.gate_dim = gate_dim
        self.context_dim = context_dim
        self.context_edge_dim = context_edge_dim
        self.layernorm_before = layernorm_before
        
        self.node_net = MLP(node_dim, hidden_dim, hidden_dim)
        self.edge_net = MLP(edge_dim, hidden_dim, hidden_dim)
        self.msg_net = Linear(hidden_dim, hidden_dim)

        if self.gate_dim > 0:
            self.gate = MLP(edge_dim+(node_dim+gate_dim)*2, hidden_dim, hidden_dim)

        self.centroid_lin = Linear(node_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        # self.act = nn.ReLU()
        # self.out_transform = Linear(hidden_dim, node_dim)
        self.out_layer = MLP(hidden_dim, node_dim, hidden_dim)
        
        if self.context_dim > 0:
            self.ctx_node_net = MLP(context_dim, hidden_dim, hidden_dim)
            self.ctx_edge_net = MLP(context_edge_dim, hidden_dim, hidden_dim)
            self.ctx_msg_net = Linear(hidden_dim, hidden_dim)
            self.ctx_gate = MLP(context_dim+context_edge_dim+(node_dim+gate_dim), hidden_dim, hidden_dim)

    def forward(self, x, edge_index, edge_attr, node_extra,
                ctx_x=None, ctx_edge_index=None, ctx_edge_attr=None):
        """
        Args:
            x:  Node features, (N, H).
            edge_index: (2, E).
            edge_attr:  (E, H)
        """
        N = x.size(0)
        row, col = edge_index   # (E,) , (E,)

        h_node = self.node_net(x)  # (N, H)

        # Compose messages
        h_edge = self.edge_net(edge_attr)  # (E, H_per_head)
        msg_j = self.msg_net(h_edge + h_node[col] + h_node[row])

        if self.gate_dim > 0:
            gate = self.gate(torch.cat([edge_attr, x[col], node_extra[col], x[row], node_extra[row]], dim=-1))
            msg_j = msg_j * torch.sigmoid(gate)

        # Aggregate messages
        aggr_msg = scatter_sum(msg_j, row, dim=0, dim_size=N)
        out = self.centroid_lin(x) + aggr_msg
        
        # context messages
        if ctx_x is not None:
            row, col = ctx_edge_index
            h_ctx = self.ctx_node_net(ctx_x)
            h_ctx_edge = self.ctx_edge_net(ctx_edge_attr)
            msg_ctx = self.ctx_msg_net(h_ctx_edge * h_ctx[col])
            if self.gate_dim > 0:
                gate = self.ctx_gate(torch.cat([ctx_edge_attr, ctx_x[col], x[row], node_extra[row]], dim=-1))
                msg_ctx = msg_ctx * torch.sigmoid(gate)
            aggred_ctx_msg = scatter_sum(msg_ctx, row, dim=0, dim_size=N)
            out = out + aggred_ctx_msg

        # output. skip connection
        out = self.out_layer(out)
        if not self.layernorm_before:
            out = self.layer_norm(out + x)
        else:
            out = self.layer_norm(out) + x
        return out


class NodeEncoder(Module):
    
    def __init__(self, node_dim=256, edge_dim=64, key_dim=128, num_heads=4, 
                    num_blocks=6, k=48, cutoff=10.0, use_atten=True, use_gate=True,
                    dist_version='new'):
        super().__init__()

        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.key_dim = key_dim
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.k = k
        self.cutoff = cutoff
        self.use_atten = use_atten
        self.use_gate = use_gate

        if dist_version == 'new':
            self.distance_expansion = GaussianSmearing(stop=cutoff, num_gaussians=20)
            self.edge_emb = Linear(self.additional_edge_feat+20, edge_dim)
        elif dist_version == 'old':
            self.distance_expansion = GaussianSmearing(stop=cutoff, num_gaussians=edge_dim-self.additional_edge_feat)
            self.edge_emb = Linear(edge_dim, edge_dim)
        else:
            raise NotImplementedError('dist_version notimplemented')
        self.node_blocks = ModuleList()
        for _ in range(num_blocks):
            block = NodeBlock(
                node_dim=node_dim,
                edge_dim=edge_dim,
                key_dim=key_dim,
                num_heads=num_heads,
                use_atten=use_atten,
                use_gate=use_gate,
            )
            self.node_blocks.append(block)

    @property
    def out_channels(self):
        return self.node_dim

    def forward(self, h, pos, edge_index, is_mol):
        #NOTE in the encoder, the edge dose not change since the position of mol and protein is fixed
        # edge_index = radius_graph(pos, self.cutoff, batch=batch, loop=False)
        edge_attr = self._add_edge_features(pos, edge_index, is_mol)
        for interaction in self.node_blocks:
            h = h + interaction(h, edge_index, edge_attr)
        return h

    @property
    def additional_edge_feat(self,):
        return 2

    def _add_edge_features(self, pos, edge_index, is_mol):
        edge_length = torch.norm(pos[edge_index[0]] - pos[edge_index[1]], dim=1)
        edge_attr = self.distance_expansion(edge_length)
        # 2-vector represent the two node types (atoms of protein or mol)
        edge_src_feat = is_mol[edge_index[0]].float().view(-1, 1)
        edge_dst_feat = is_mol[edge_index[1]].float().view(-1, 1)
        edge_attr = torch.cat([edge_attr, edge_src_feat, edge_dst_feat], dim=1)
        edge_attr = self.edge_emb(edge_attr)
        return edge_attr


class BondFFN(Module):
    def __init__(self, bond_dim, node_dim, inter_dim, gate_dim, out_dim=None):
        super().__init__()
        out_dim = bond_dim if out_dim is None else out_dim
        self.gate_dim = gate_dim
        self.bond_linear = Linear(bond_dim, inter_dim, bias=False)
        self.node_linear = Linear(node_dim, inter_dim, bias=False)
        self.inter_module = MLP(inter_dim, out_dim, inter_dim)
        if self.gate_dim > 0:
            self.gate = MLP(bond_dim+node_dim+gate_dim, out_dim, 32)

    def forward(self, bond_feat_input, node_feat_input, extra):
        bond_feat = self.bond_linear(bond_feat_input)
        node_feat = self.node_linear(node_feat_input)
        inter_feat = bond_feat + node_feat
        inter_feat = self.inter_module(inter_feat)
        if self.gate_dim > 0:
            gate = self.gate(torch.cat([bond_feat_input, node_feat_input, extra], dim=-1))
            inter_feat = inter_feat * torch.sigmoid(gate)
        return inter_feat


class EdgeBlock(Module):
    def __init__(self, edge_dim, node_dim, hidden_dim=None, gate_dim=0, layernorm_before=False):
        super().__init__()
        self.gate_dim = gate_dim
        inter_dim = edge_dim * 2 if hidden_dim is None else hidden_dim
        self.layernorm_before = layernorm_before

        self.bond_ffn_left = BondFFN(edge_dim, node_dim, inter_dim=inter_dim, gate_dim=gate_dim)
        self.bond_ffn_right = BondFFN(edge_dim, node_dim, inter_dim=inter_dim, gate_dim=gate_dim)

        self.msg_left = Linear(edge_dim, edge_dim)
        self.msg_right = Linear(edge_dim, edge_dim)

        self.node_ffn_left = Linear(node_dim, edge_dim)
        self.node_ffn_right = Linear(node_dim, edge_dim)

        self.self_ffn = Linear(edge_dim, edge_dim)
        self.layer_norm = nn.LayerNorm(edge_dim)
        self.out_layer = MLP(edge_dim, edge_dim, edge_dim)

    def forward(self, h_bond, bond_index, h_node, bond_extra):
        """
        h_bond: (b, bond_dim)
        bond_index: (2, b)
        h_node: (n, node_dim)
        """
        N = h_node.size(0)
        left_node, right_node = bond_index

        # message from neighbor bonds
        msg_bond_left = self.bond_ffn_left(h_bond, h_node[left_node], bond_extra)
        msg_bond_left = scatter_sum(msg_bond_left, right_node, dim=0, dim_size=N)
        msg_bond_left = msg_bond_left[left_node]

        msg_bond_right = self.bond_ffn_right(h_bond, h_node[right_node], bond_extra)
        msg_bond_right = scatter_sum(msg_bond_right, left_node, dim=0, dim_size=N)
        msg_bond_right = msg_bond_right[right_node]
        
        h_bond_update = (
            self.msg_left(msg_bond_left)
            + self.msg_right(msg_bond_right)
            + self.node_ffn_left(h_node[left_node])
            + self.node_ffn_right(h_node[right_node])
            + self.self_ffn(h_bond)
        )
        h_bond_update = self.out_layer(h_bond_update)

        # skip connection
        if not self.layernorm_before:
            h_bond = self.layer_norm(h_bond_update + h_bond)
        else:
            h_bond = self.layer_norm(h_bond_update) + h_bond
        return h_bond


class ContextNodeEdgeNet(Module):
    def __init__(self, node_dim, edge_dim, hidden_dim,
                 num_blocks, dist_cfg, gate_dim=0,
                 context_dim=0, context_cfg=None,
                 node_only=False, **kwargs):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.num_blocks = num_blocks
        self.dist_cfg = dist_cfg
        self.gate_dim = gate_dim
        self.node_only = node_only
        self.kwargs = kwargs
        self.downsample_context = kwargs.get('downsample_context', False)
        self.layernorm_before = kwargs.get("layernorm_before", False)

        self.distance_expansion = GaussianSmearing(**dist_cfg)
        num_gaussians = dist_cfg['num_gaussians']
        input_edge_dim = num_gaussians + (0 if node_only else edge_dim)
            
        # for context
        self.context_cfg = context_cfg
        if context_cfg is not None:
            context_edge_dim = context_cfg['edge_dim']
            self.knn = context_cfg['knn']
            self.dist_exp_ctx = GaussianSmearing(**context_cfg['dist_cfg'])
            input_context_edge_dim = context_cfg['dist_cfg']['num_gaussians']
            assert context_dim > 0, 'context_dim should be larger than 0 if context_cfg is not None'
            assert not node_only, 'not support node_only with context'
        else:
            context_edge_dim = 0
        
        # node network
        self.edge_embs = ModuleList()
        self.node_blocks_with_edge = ModuleList()
        if not node_only:
            self.edge_blocks = ModuleList()
            self.pos_blocks = ModuleList()
            if self.context_cfg is not None:
                self.ctx_edge_embs = ModuleList()
                self.ctx_pos_blocks = ModuleList()
        for _ in range(num_blocks):
            # edge emb
            self.edge_embs.append(Linear(input_edge_dim, edge_dim))
            # node update
            self.node_blocks_with_edge.append(ContextNodeBlock(
                node_dim, edge_dim, hidden_dim, gate_dim,
                context_dim, context_edge_dim, layernorm_before=self.layernorm_before
            ))
            if node_only:
                continue
            # edge update
            self.edge_blocks.append(EdgeBlock(
                edge_dim=edge_dim, node_dim=node_dim, gate_dim=gate_dim, layernorm_before=self.layernorm_before
            ))
            # pos update
            self.pos_blocks.append(PosUpdate(
                node_dim, edge_dim, hidden_dim=edge_dim, gate_dim=gate_dim*2,
            ))
            if self.context_cfg is not None:
                self.ctx_edge_embs.append(Linear(input_context_edge_dim, context_edge_dim))
                self.ctx_pos_blocks.append(PosUpdate(
                    node_dim, context_edge_dim, hidden_dim=edge_dim, gate_dim=gate_dim,
                    node_dim_right=context_dim,
                ))
        
        self.local_update = kwargs.get('local_update', False)
        if self.local_update:
            self.local_net = LocalPosUpdate(node_dim, edge_dim, node_dim, cutoff=3.8)
        
                
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

        for i in range(self.num_blocks):
            # # remake edge fetures (distance have been changed in each iteration)
            if (i==0) or (not self.node_only):
                h_dist, relative_vec, distance = self._build_edges_dist(pos_node, edge_index)
            if not self.node_only:
                h_edge = torch.cat([h_edge, h_dist], dim=-1)
            else:
                h_edge = h_dist
            h_edge = self.edge_embs[i](h_edge)
            
            # # edge with context
            if h_ctx is not None:
                h_ctx_edge, vec_ctx, dist_ctx, ctx_knn_edge_index = self._build_context_edges_dist(
                    pos_node, pos_ctx, batch_node, batch_ctx)
                h_ctx_edge = self.ctx_edge_embs[i](h_ctx_edge)
            else:
                ctx_knn_edge_index = None
                h_ctx_edge = None

            # # node feature updates
            h_node = self.node_blocks_with_edge[i](h_node, edge_index, h_edge, node_extra,
                                        h_ctx, ctx_knn_edge_index, h_ctx_edge)
            if self.node_only:
                continue
            
            # # edge feature updates
            h_edge = self.edge_blocks[i](h_edge, edge_index, h_node, edge_extra)

            # # pos updates
            pos_node = pos_node + self.pos_blocks[i](h_node, h_edge, edge_index, relative_vec, distance, node_extra, edge_extra)
            if h_ctx is not None:
                pos_node = pos_node + self.ctx_pos_blocks[i](
                    h_node, h_ctx_edge, ctx_knn_edge_index, vec_ctx, dist_ctx, node_extra,
                    edge_extra=None, h_node_right=h_ctx)

        if self.local_update:
            pos_node = pos_node + self.local_net(h_node, pos_node, h_edge, edge_index, batch_node)

        if self.node_only:
            return h_node
        else:
            return h_node, pos_node, h_edge

    def _build_edges_dist(self, pos, edge_index):
        # distance
        relative_vec = pos[edge_index[0]] - pos[edge_index[1]]
        distance = torch.norm(relative_vec, dim=-1, p=2)
        h_dist = self.distance_expansion(distance)
        return h_dist, relative_vec, distance
    
    def _build_context_edges_dist(self, pos, pos_ctx, batch_node, batch_ctx):
        # build knn edge index
        if self.knn < 100:
            if self.downsample_context:
                pos_ctx_noised = pos_ctx + torch.randn_like(pos_ctx) * 5  # works like masked position information
            else:
                pos_ctx_noised = pos_ctx
            ctx_knn_edge_index = knn(y=pos, x=pos_ctx_noised, k=self.knn,
                                    batch_x=batch_ctx, batch_y=batch_node)
        else: # fully connected x-yf
            device = pos.device
            ctx_knn_edge_index = []
            cum_node = 0
            cum_ctx = 0
            for i_batch in range(batch_ctx.max()+1):
                num_ctx = (batch_ctx==i_batch).sum()
                num_node = (batch_node==i_batch).sum()
                ctx_knn_edge_index_this = torch.stack(
                    torch.meshgrid(
                        torch.arange(num_node, device=device) + cum_node,
                        torch.arange(num_ctx, device=device) + cum_ctx,
                    )).view(2, -1)
                cum_node += num_node
                cum_ctx += num_ctx
                ctx_knn_edge_index.append(ctx_knn_edge_index_this)
            ctx_knn_edge_index = torch.cat(ctx_knn_edge_index, dim=-1)

        relative_vec = pos[ctx_knn_edge_index[0]] - pos_ctx[ctx_knn_edge_index[1]]
        distance = torch.norm(relative_vec, dim=-1, p=2)
        h_dist = self.dist_exp_ctx(distance)
        return h_dist, relative_vec, distance, ctx_knn_edge_index
        


class PosUpdate(Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, gate_dim, node_dim_right=None):
        super().__init__()
        self.left_lin_edge = MLP(node_dim, node_dim, hidden_dim)
        node_dim_right = node_dim if node_dim_right is None else node_dim_right
        self.right_lin_edge = MLP(node_dim_right, node_dim, hidden_dim)
        self.edge_lin = BondFFN(edge_dim, node_dim*2, node_dim, gate_dim, out_dim=1)
        self.pos_scale_net = nn.Sequential(MLP(node_dim+1+2, 1, hidden_dim), nn.Sigmoid())

    def forward(self, h_node, h_edge, edge_index, relative_vec, distance, node_extra, edge_extra=None, h_node_right=None):
        edge_index_left, edge_index_right = edge_index
        
        left_feat = self.left_lin_edge(h_node[edge_index_left])
        h_node_right = h_node if h_node_right is None else h_node_right
        right_feat = self.right_lin_edge(h_node_right[edge_index_right])
        both_extra = node_extra[edge_index_left]
        if edge_extra is not None:
            both_extra = torch.cat([both_extra, edge_extra], dim=-1)
        weight_edge = self.edge_lin(h_edge,
                            torch.cat([left_feat, right_feat], dim=-1),
                            both_extra)
        
        force_edge = weight_edge * relative_vec / (distance.unsqueeze(-1) + 1e-6) / (distance.unsqueeze(-1) + 5.) * 5
        delta_pos = scatter_sum(force_edge, edge_index_left, dim=0, dim_size=h_node.shape[0])
        delta_pos = delta_pos * self.pos_scale_net(torch.cat([h_node, node_extra,
                                        torch.norm(delta_pos, dim=-1, keepdim=True)], dim=-1))
        return delta_pos

class LocalPosUpdate(Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, cutoff=3.5):
        super().__init__()
        self.cutoff = cutoff
        self.dist_exp = GaussianSmearing(stop=cutoff, num_gaussians=16)
        self.dist_lin = MLP(edge_dim+16, edge_dim, edge_dim)
        self.node_lin = MLP(node_dim*3, node_dim, hidden_dim)
        self.edge_lin = MLP(edge_dim*3, edge_dim, edge_dim)
        self.weight_lin = MLP(node_dim+edge_dim, 3, node_dim)
    def forward(self, h_node, node_pos, h_edge, edge_index, batch_node):
        # find only edges within a threshold
        is_short_edge, vector, distance = self._find_short_edges(node_pos, edge_index)
        # drop nodes with no short edges
        node_degree = scatter_sum(is_short_edge.long(), edge_index[0], dim=-1)
        is_short_node = (node_degree > 1)
        is_short_edge = is_short_edge & (is_short_node[edge_index[0]]) & (is_short_node[edge_index[1]])
        short_edge_index = edge_index[:, is_short_edge]

        node_triples = []
        for node in torch.argwhere(is_short_node).flatten():
            node_neighs = short_edge_index[1, short_edge_index[0] == node]
            node_pairs = permutations(node_neighs, 2)
            node_triples.extend([torch.stack([node, pair[0], pair[1]]) for pair in node_pairs])
        node_triples_tensor = torch.tensor(node_triples)
        node_triples_stack = torch.stack(node_triples)
        
        edge_ids = [
            edge_index_to_index_of_edge(node_triples[:, 0:2].T, batch_node),
            edge_index_to_index_of_edge(node_triples[:, [0,2]].T, batch_node),
            edge_index_to_index_of_edge(node_triples[:, 1:3].T, batch_node)
        ]
        
        # feats
        # h_node = self.node_lin(h_node)
        dist_feat = self.dist_exp(distance)
        h_edge = self.dist_lin(torch.cat([h_edge, dist_feat], dim=-1))
        
        h_node_triplets = self.node_lin(torch.cat(h_node[node_triples], dim=-1))
        h_edge_triplets = self.edge_lin(torch.cat(h_edge[edge_ids], dim=-1))
        weight = self.weight_lin(torch.cat([h_node_triplets, h_edge_triplets], dim=-1))
        
        vector_triplets = vector[edge_ids[:2]]
        vector_cross = torch.linalg.cross(vector_triplets[0], vector_triplets[1], dim=-1)
        vector_triplets = torch.cat([vector_triplets, vector_cross], dim=-1)
        delta_pos = (weight * vector_triplets).sum(dim=1)
        delta_pos = scatter_sum(delta_pos, node_triples[:, 0], dim=0, dim_size=h_node.shape[0])
        
        return delta_pos

    def _find_short_edges(self, node_pos, edge_index):
        vec = node_pos[edge_index[0]] - node_pos[edge_index[1]]
        distance = torch.norm(vec, dim=-1)
        unit = vec / (distance.unsqueeze(-1) + 1e-8)
        is_short_edge = distance < self.cutoff
        return is_short_edge, unit, distance



class RigidNet(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, cutoff, use_gate):
        super().__init__()
        edge_dim = edge_dim // 2
        hidden_dim = hidden_dim // 2
        self.distance_expansion = GaussianSmearing(start=0, stop=cutoff, num_gaussians=edge_dim)
        self.domain_net = NodeBlock(node_dim, edge_dim, hidden_dim, use_gate, is_heter=True)
        self.translation_net = MLP(node_dim*2+edge_dim, 1, hidden_dim, 2)
        self.torque_net = MLP(node_dim*2+3, 1, hidden_dim, 2)
        self.angle_net = nn.Sequential(
            MLP(node_dim+1, 1, hidden_dim//2, 2),
            nn.Sigmoid(),
        )
    
    def forward(self, h_node, pos_node, delta_pos,
                domain_node_index_0, domain_node_index_1):
        
        h_node_in_domain = h_node[domain_node_index_1]
        pos_node_in_domain = pos_node[domain_node_index_1]
        delta_pos_in_domain = delta_pos[domain_node_index_1]
        
        # # domain feature
        n_domain = domain_node_index_0.max() + 1
        pos_domain = scatter_mean(pos_node_in_domain, index=domain_node_index_0,
                                  dim=0, dim_size=n_domain)[domain_node_index_0]
        radius_vec = pos_node_in_domain - pos_domain
        dist_domain = torch.norm(radius_vec, dim=-1, p=2)
        h_edge = self.distance_expansion(dist_domain)
        h_domain = self.domain_net(h_node, torch.stack([domain_node_index_0, domain_node_index_1], dim=0),
                                   h_edge, torch.zeros_like(h_node[..., 0:1]))
        
        # # domain translation
        translation_weight = self.translation_net(torch.cat([
            h_domain[domain_node_index_0], h_node_in_domain, h_edge], dim=-1))
        force_edge = translation_weight * delta_pos_in_domain
        translation_domain = scatter_mean(force_edge, index=domain_node_index_0, dim=0, dim_size=n_domain)
        translation_domain = translation_domain[domain_node_index_0]

        # # torque for each node (for node in domain)
        torque = torch.linalg.cross(radius_vec, delta_pos_in_domain, dim=-1)
        h_torque = torch.cat([h_node_in_domain, h_domain[domain_node_index_0], 
                              dist_domain[..., None],
                              torch.norm(delta_pos_in_domain, dim=-1, p=2, keepdim=True),
                              torch.norm(torque, dim=-1, p=2, keepdim=True),
                              ], dim=-1)
        scalar = self.torque_net(h_torque)
        scaled_torque = torque * scalar
        
        # torque for each domain
        torque_domain = scatter_mean(scaled_torque, index=domain_node_index_0, dim=0, dim_size=n_domain)

        # get rotatio axis (unit vector of torque) and rotation angle
        torque_norm = torch.norm(torque_domain, dim=-1, p=2, keepdim=True)
        rot_axis = torque_domain / torque_norm
        rot_angle = self.angle_net(torch.cat([h_domain, torque_norm], dim=-1)) * torch.pi
        # NOTE: can prdict cos and sin for stability in the feature (like AF2)
        # apply rotation
        pos_update = pos_domain + translation_domain + apply_axis_angle_rotation(
            radius_vec, rot_axis[domain_node_index_0], rot_angle[domain_node_index_0])
        
        pos_out = pos_node + delta_pos
        pos_out[domain_node_index_1] = pos_update
        return pos_out, rot_axis, rot_angle
        

class TorsionNet(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, cutoff, use_gate):
        super().__init__()
        self.radius_net = GaussianSmearing(start=0, stop=cutoff, num_gaussians=edge_dim)
        self.torque_net = MLP(node_dim*2+edge_dim*3+3, 1, hidden_dim, 2)

        self.torsional_left_net = NodeBlock(node_dim, edge_dim, hidden_dim//2, use_gate, )
        self.angle_net = nn.Sequential(
            MLP(node_dim+1, 1, hidden_dim//2, 2),
            nn.Sigmoid(),
        )
    
    def forward(self, h_node, pos_node, force,
                h_edge, edge_index, 
                torsional_edge_anno, twisted_edge_anno):
        """
        h_node: (N, node_dim)
        pos_node: (N, 3)
        force [delta_pos]: (N, 3)
        h_edge: (E, edge_dim)
        edge_index: (2, E)
        torsional_edge_anno: (2, n_tor)
        twisted_edge_anno: (2, T)
        """
        
        i_bond_for_tor_edge, tor_edge =torsional_edge_anno
        i_tor_edge_for_twisted_edge, twisted_edge = twisted_edge_anno
        
        # # fetch torsional bond/node features and vec
        tor_left, tor_right = edge_index[:, tor_edge]
        h_tor_edge = h_edge[tor_edge]  # (n_tor, edge_dim)
        h_tor_left = h_node[tor_left]  # (n_tor, node_dim)
        h_tor_right = h_node[tor_right]  # (n_tor, node_dim)
        vec_tor_bond = pos_node[tor_left] - pos_node[tor_right]  # (n_tor, 3)
        len_tor_bond = torch.norm(vec_tor_bond, dim=-1, p=2, keepdim=True)  # (n_tor, 1)
        unit_tor_bond = vec_tor_bond / (len_tor_bond + 1e-6)  # (n_tor, 3)
        unit_tor_bond_expand = unit_tor_bond[i_tor_edge_for_twisted_edge]  #  (T, 3)
        
        # # fetch twisted node and edge feature
        twisted_node, tor_end = edge_index[:, twisted_edge]
        assert (tor_end == tor_left[i_tor_edge_for_twisted_edge]).all(), "torsional end must be the same as torsional left"
        h_twisted_edge = h_edge[twisted_edge]  # (T, edge_dim)
        h_twisted_node = h_node[twisted_node]  # (T, node_dim)
        force_twisted_node = force[twisted_node]  # (T, 3)
        vec_twisted_edge = pos_node[twisted_node] - pos_node[tor_end]  # (T, 3)
        
        # # calculate torque
        vec_radius = vec_twisted_edge - torch.sum(vec_twisted_edge * unit_tor_bond_expand, dim=-1, keepdim=True) * unit_tor_bond_expand
        len_radius = torch.norm(vec_radius, dim=-1, p=2)
        h_radius = self.radius_net(len_radius)
        force_tangent = force_twisted_node - torch.sum(force_twisted_node * unit_tor_bond_expand, dim=-1, keepdim=True) * unit_tor_bond_expand
        torque = torch.linalg.cross(vec_radius, force_tangent, dim=-1)  # (T, 3)
        
        # # calculate torque weight
        h_torque = torch.cat([h_twisted_node, h_twisted_edge, h_tor_left[i_tor_edge_for_twisted_edge],
                              h_tor_edge[i_tor_edge_for_twisted_edge],
                              h_radius, 
                              torch.norm(force[twisted_node], dim=-1, p=2, keepdim=True),
                              torch.norm(force_tangent, dim=-1, p=2, keepdim=True),
                              torch.norm(torque, dim=-1, p=2, keepdim=True),], dim=-1)
        torque_weight = self.torque_net(h_torque)  # (T, 1)
        torque = torque * torque_weight
        
        # # aggregate torque to calculate angles
        torque_tor = scatter_mean(torque, index=i_tor_edge_for_twisted_edge, dim=0)  # (n_tor, 3)
        len_torque = torch.norm(torque_tor, dim=-1, p=2, keepdim=True)  # (n_tor, 1)
        assert torch.linalg.cross(torque_tor, unit_tor_bond, dim=-1).abs().max() < 1e-2, "torque must be in parallel with torsional bond"
        h_node = self.torsional_left_net(h_node, edge_index[:, twisted_edge].flip(0),
                                        h_edge[twisted_edge], torch.zeros_like(h_node[..., 0:1]))
        h_node_tor = h_node[tor_left]
        angles = self.angle_net(torch.cat([
            len_torque, h_node_tor,
        ], dim=-1)) * torch.pi
        direction = torch.sum(torque_tor * unit_tor_bond, dim=-1, keepdim=True)
        angles = angles * torch.sign(direction) # (n_tor, 1)
        # angles = angles[tor_left]
        
        # # apply rotation
        pos_update = apply_torsional_rotation_multiple_domains(
            pos_node, edge_index, 
            tor_edge, angles, i_bond_for_tor_edge,
            twisted_edge, i_tor_edge_for_twisted_edge
        )
        return pos_update, angles


