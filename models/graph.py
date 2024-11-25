import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Sequential, Linear, Conv1d, ModuleList
from torch_scatter import scatter_mean, scatter_sum, scatter_softmax
from torch_geometric.nn import radius_graph, knn_graph
from models.common import GaussianSmearing, MLP, NONLINEARITIES
from utils.motion import apply_axis_angle_rotation, apply_torsional_rotation_multiple_domains

class NodeBlock(Module):

    def __init__(self, node_dim, edge_dim, hidden_dim, use_gate, is_heter=False):
        super().__init__()
        self.use_gate = use_gate
        self.node_dim = node_dim
        self.is_heter = is_heter
        
        self.node_net = MLP(node_dim, hidden_dim, hidden_dim)
        self.edge_net = MLP(edge_dim, hidden_dim, hidden_dim)
        self.msg_net = Linear(hidden_dim, hidden_dim)

        if self.use_gate:
            self.gate = MLP(edge_dim+node_dim+1, hidden_dim, hidden_dim) # add 1 for time

        if not is_heter:
            self.centroid_lin = Linear(node_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.act = nn.ReLU()
        self.out_transform = Linear(hidden_dim, node_dim)

    def forward(self, x, edge_index, edge_attr, node_time):
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
        msg_j = self.msg_net(h_edge * h_node[col])

        if self.use_gate:
            gate = self.gate(torch.cat([edge_attr, x[col], node_time[col]], dim=-1))
            msg_j = msg_j * torch.sigmoid(gate)

        # Aggregate messages
        if not self.is_heter:
            aggr_msg = scatter_sum(msg_j, row, dim=0, dim_size=N)
            out = self.centroid_lin(x) + aggr_msg
        else:
            aggr_msg = scatter_sum(msg_j, row, dim=0)
            out = aggr_msg

        out = self.layer_norm(out)
        out = self.out_transform(self.act(out))
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
    def __init__(self, bond_dim, node_dim, inter_dim, use_gate, out_dim=None):
        super().__init__()
        out_dim = bond_dim if out_dim is None else out_dim
        self.use_gate = use_gate
        self.bond_linear = Linear(bond_dim, inter_dim, bias=False)
        self.node_linear = Linear(node_dim, inter_dim, bias=False)
        self.inter_module = MLP(inter_dim, out_dim, inter_dim)
        if self.use_gate:
            self.gate = MLP(bond_dim+node_dim+1, out_dim, 32)  # +1 for time

    def forward(self, bond_feat_input, node_feat_input, time):
        bond_feat = self.bond_linear(bond_feat_input)
        node_feat = self.node_linear(node_feat_input)
        inter_feat = bond_feat * node_feat
        inter_feat = self.inter_module(inter_feat)
        if self.use_gate:
            gate = self.gate(torch.cat([bond_feat_input, node_feat_input, time], dim=-1))
            inter_feat = inter_feat * torch.sigmoid(gate)
        return inter_feat


class QKVLin(Module):
    def __init__(self, h_dim, key_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.q_lin = Linear(h_dim, key_dim)
        self.k_lin = Linear(h_dim, key_dim)
        self.v_lin = Linear(h_dim, h_dim)

    def forward(self, inputs):
        n = inputs.size(0)
        return [
            self.q_lin(inputs).view(n, self.num_heads, -1),
            self.k_lin(inputs).view(n, self.num_heads, -1),
            self.v_lin(inputs).view(n, self.num_heads, -1),
        ]


class BondBlock(Module):
    def __init__(self, bond_dim, node_dim, use_gate=True, use_atten=False, num_heads=2, key_dim=128):
        super().__init__()
        self.use_atten = use_atten
        self.use_gate = use_gate
        inter_dim = bond_dim * 2

        self.bond_ffn_left = BondFFN(bond_dim, node_dim, inter_dim=inter_dim, use_gate=use_gate)
        self.bond_ffn_right = BondFFN(bond_dim, node_dim, inter_dim=inter_dim, use_gate=use_gate)
        if self.use_atten:
            # key_dim = bond_dim // 2
            assert bond_dim % num_heads == 0
            assert key_dim % num_heads == 0
            # linear transformation for attention 
            self.qkv_left = QKVLin(bond_dim, key_dim, num_heads)
            self.qkv_right = QKVLin(bond_dim, key_dim, num_heads)
            # self.q_lin = Linear(bond_dim, key_dim)
            # self.k_lin = Linear(bond_dim, key_dim)
            # self.v_lin = Linear(bond_dim, bond_dim)
            self.layer_norm_atten1 = nn.LayerNorm(bond_dim)
            self.layer_norm_atten2 = nn.LayerNorm(bond_dim)
        
        self.node_ffn_left = Linear(node_dim, bond_dim)
        self.node_ffn_right = Linear(node_dim, bond_dim)

        self.self_ffn = Linear(bond_dim, bond_dim)
        self.layer_norm = nn.LayerNorm(bond_dim)
        self.out_transform = Linear(bond_dim, bond_dim)
        self.act = nn.ReLU()

    def forward(self, h_bond, bond_index, h_node, atten_index=None):
        """
        h_bond: (b, bond_dim)
        bond_index: (2, b)
        h_node: (n, node_dim)
        pos_node: (n, 3)
        """
        N = h_node.size(0)
        left_node, right_node = bond_index

        # message from neighbor bonds
        msg_bond_left = self.bond_ffn_left(h_bond, h_node[left_node])
        msg_bond_left = scatter_sum(msg_bond_left, right_node, dim=0, dim_size=N)
        msg_bond_left = msg_bond_left[left_node]

        msg_bond_right = self.bond_ffn_right(h_bond, h_node[right_node])
        msg_bond_right = scatter_sum(msg_bond_right, left_node, dim=0, dim_size=N)
        msg_bond_right = msg_bond_right[right_node]
        
        h_bond = (
            msg_bond_left + msg_bond_right
            + self.node_ffn_left(h_node[left_node])
            + self.node_ffn_right(h_node[right_node])
            + self.self_ffn(h_bond)
        )
        h_bond = self.layer_norm(h_bond)

        if self.use_atten:
            index_query_bond_left, index_key_bond_left, index_query_bond_right, index_key_bond_right = atten_index

            # left node
            h_queries, h_keys, h_values = self.qkv_left(h_bond)
            queries_i = h_queries[index_query_bond_left]
            keys_j = h_keys[index_key_bond_left]
            qk_ij = (queries_i * keys_j).sum(-1)
            alpha = scatter_softmax(qk_ij, index_query_bond_left, dim=0)
            values_j = h_values[index_key_bond_left]
            num_attns = len(index_key_bond_left)
            h_bond = scatter_sum((alpha.unsqueeze(-1) * values_j).view(num_attns, -1), 
                                        index_query_bond_left, dim=0, dim_size=h_bond.size(0))
            h_bond = self.layer_norm_atten1(h_bond)

            # right node
            h_queries, h_keys, h_values = self.qkv_right(h_bond)
            queries_i = h_queries[index_query_bond_right]
            keys_j = h_keys[index_key_bond_right]
            qk_ij = (queries_i * keys_j).sum(-1)
            alpha = scatter_softmax(qk_ij, index_query_bond_right, dim=0)
            values_j = h_values[index_key_bond_right]
            num_attns = len(index_key_bond_right)
            h_bond = scatter_sum((alpha.unsqueeze(-1) * values_j).view(num_attns, -1), 
                                        index_query_bond_right, dim=0, dim_size=h_bond.size(0))
            h_bond = self.layer_norm_atten2(h_bond)

        h_bond = self.out_transform(self.act(h_bond))
        return h_bond




class EdgeBlock(Module):
    def __init__(self, edge_dim, node_dim, hidden_dim=None, use_gate=True):
        super().__init__()
        self.use_gate = use_gate
        inter_dim = edge_dim * 2 if hidden_dim is None else hidden_dim

        self.bond_ffn_left = BondFFN(edge_dim, node_dim, inter_dim=inter_dim, use_gate=use_gate)
        self.bond_ffn_right = BondFFN(edge_dim, node_dim, inter_dim=inter_dim, use_gate=use_gate)

        self.node_ffn_left = Linear(node_dim, edge_dim)
        self.node_ffn_right = Linear(node_dim, edge_dim)

        self.self_ffn = Linear(edge_dim, edge_dim)
        self.layer_norm = nn.LayerNorm(edge_dim)
        self.out_transform = Linear(edge_dim, edge_dim)
        self.act = nn.ReLU()

    def forward(self, h_bond, bond_index, h_node, bond_time):
        """
        h_bond: (b, bond_dim)
        bond_index: (2, b)
        h_node: (n, node_dim)
        """
        N = h_node.size(0)
        left_node, right_node = bond_index

        # message from neighbor bonds
        msg_bond_left = self.bond_ffn_left(h_bond, h_node[left_node], bond_time)
        msg_bond_left = scatter_sum(msg_bond_left, right_node, dim=0, dim_size=N)
        msg_bond_left = msg_bond_left[left_node]

        msg_bond_right = self.bond_ffn_right(h_bond, h_node[right_node], bond_time)
        msg_bond_right = scatter_sum(msg_bond_right, left_node, dim=0, dim_size=N)
        msg_bond_right = msg_bond_right[right_node]
        
        h_bond = (
            msg_bond_left + msg_bond_right
            + self.node_ffn_left(h_node[left_node])
            + self.node_ffn_right(h_node[right_node])
            + self.self_ffn(h_bond)
        )
        h_bond = self.layer_norm(h_bond)

        h_bond = self.out_transform(self.act(h_bond))
        return h_bond


class NodeEdgeNet(Module):
    def __init__(self, node_dim, edge_dim, num_blocks, cutoff, use_gate, **kwargs):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.num_blocks = num_blocks
        self.cutoff = cutoff
        self.use_gate = use_gate
        self.kwargs = kwargs

        if 'num_gaussians' not in kwargs:
            num_gaussians = 16
        else:
            num_gaussians = kwargs['num_gaussians']
        if 'start' not in kwargs:
            start = 0
        else:
            start = kwargs['start']
        self.distance_expansion = GaussianSmearing(start=start, stop=cutoff, num_gaussians=num_gaussians)
        if ('update_edge' in kwargs) and (not kwargs['update_edge']):
            self.update_edge = False
            input_edge_dim = num_gaussians
        else:
            self.update_edge = True  # default update edge
            input_edge_dim = edge_dim + num_gaussians
            
        if ('update_pos' in kwargs) and (not kwargs['update_pos']):
            self.update_pos = False
        else:
            self.update_pos = True  # default update pos
        
        # node network
        self.node_blocks_with_edge = ModuleList()
        self.edge_embs = ModuleList()
        self.edge_blocks = ModuleList()
        self.pos_blocks = ModuleList()
        for _ in range(num_blocks):
            self.node_blocks_with_edge.append(NodeBlock(
                node_dim=node_dim, edge_dim=edge_dim, hidden_dim=node_dim, use_gate=use_gate,
            ))
            self.edge_embs.append(Linear(input_edge_dim, edge_dim))
            if self.update_edge:
                self.edge_blocks.append(EdgeBlock(
                    edge_dim=edge_dim, node_dim=node_dim, use_gate=use_gate,
                ))
            if self.update_pos:
                self.pos_blocks.append(PosUpdate(
                    node_dim=node_dim, edge_dim=edge_dim, hidden_dim=edge_dim, use_gate=use_gate,
                ))
                
        # self.motion_update = kwargs.get('motion_update', False)
        # if self.motion_update:
        #     self.rigid_net = RigidNet(node_dim, edge_dim, edge_dim, cutoff, use_gate)
        #     self.torsion_net = TorsionNet(node_dim, edge_dim, edge_dim, cutoff, use_gate)

    def forward(self, h_node, pos_node, h_edge, edge_index, node_time, edge_time):
        # pos_node_input = pos_node
        for i in range(self.num_blocks):
            # edge fetures before each block
            if self.update_pos or (i==0):
                h_edge_dist, relative_vec, distance = self._build_edges_dist(pos_node, edge_index)
            if self.update_edge:
                h_edge = torch.cat([h_edge, h_edge_dist], dim=-1)
            else:
                h_edge = h_edge_dist
            h_edge = self.edge_embs[i](h_edge)
                
            # node and edge feature updates
            h_node_with_edge = self.node_blocks_with_edge[i](h_node, edge_index, h_edge, node_time)
            if self.update_edge:
                h_edge = h_edge + self.edge_blocks[i](h_edge, edge_index, h_node, edge_time)
            h_node = h_node + h_node_with_edge
            # pos updates
            if self.update_pos:
                pos_node = pos_node + self.pos_blocks[i](h_node, h_edge, edge_index, relative_vec, distance, edge_time)
        # if self.motion_update:
        #     delta_pos = pos_node - pos_node_input
        #     pos_node_rigid, rot_axis, rot_angle = self.rigid_net(h_node, pos_node_input, delta_pos,
        #                                domain_node_index_0, domain_node_index_1)
        #     delta_pos = pos_node - pos_node_rigid
        #     pos_node, tor_angle = self.torsion_net(h_node, pos_node_rigid, delta_pos, h_edge, edge_index,
        #                                 torsional_edge_anno, twisted_edge_anno)
        return h_node, pos_node, h_edge

    def _build_edges_dist(self, pos, edge_index):
        # distance
        relative_vec = pos[edge_index[0]] - pos[edge_index[1]]
        distance = torch.norm(relative_vec, dim=-1, p=2)
        edge_dist = self.distance_expansion(distance)
        return edge_dist, relative_vec, distance


class PosUpdate(Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, use_gate):
        super().__init__()
        self.left_lin_edge = MLP(node_dim, edge_dim, hidden_dim)
        self.right_lin_edge = MLP(node_dim, edge_dim, hidden_dim)
        self.edge_lin = BondFFN(edge_dim, edge_dim, node_dim, use_gate, out_dim=1)

    def forward(self, h_node, h_edge, edge_index, relative_vec, distance, edge_time):
        edge_index_left, edge_index_right = edge_index
        
        left_feat = self.left_lin_edge(h_node[edge_index_left])
        right_feat = self.right_lin_edge(h_node[edge_index_right])
        weight_edge = self.edge_lin(h_edge, left_feat * right_feat, edge_time)
        
        # relative_vec = pos_node[edge_index_left] - pos_node[edge_index_right]
        # distance = torch.norm(relative_vec, dim=-1, keepdim=True)
        force_edge = weight_edge * relative_vec / distance.unsqueeze(-1) / (distance.unsqueeze(-1) + 1.)
        delta_pos = scatter_sum(force_edge, edge_index_left, dim=0, dim_size=h_node.shape[0])

        return delta_pos

class PosPredictor(Module):
    def __init__(self, node_dim, edge_dim, bond_dim, use_gate):
        super().__init__()
        self.left_lin_edge = MLP(node_dim, edge_dim, hidden_dim=edge_dim)
        self.right_lin_edge = MLP(node_dim, edge_dim, hidden_dim=edge_dim)
        self.edge_lin = BondFFN(edge_dim, edge_dim, node_dim, use_gate, out_dim=1)

        self.bond_dim = bond_dim
        if bond_dim > 0:
            self.left_lin_bond = MLP(node_dim, bond_dim, hidden_dim=bond_dim)
            self.right_lin_bond = MLP(node_dim, bond_dim, hidden_dim=bond_dim)
            self.bond_lin = BondFFN(bond_dim, bond_dim, node_dim, use_gate, out_dim=1)

    def forward(self, h_node, pos_node, h_bond, bond_index, h_edge, edge_index, is_frag):
        # 1 pos update through edges
        is_left_frag = is_frag[edge_index[0]]
        edge_index_left, edge_index_right = edge_index[:, is_left_frag]
        
        left_feat = self.left_lin_edge(h_node[edge_index_left])
        right_feat = self.right_lin_edge(h_node[edge_index_right])
        weight_edge = self.edge_lin(h_edge[is_left_frag], left_feat * right_feat)
        force_edge = weight_edge * (pos_node[edge_index_left] - pos_node[edge_index_right])
        delta_pos = scatter_sum(force_edge, edge_index_left, dim=0, dim_size=h_node.shape[0])

        # 2 pos update through bonds
        if self.bond_dim > 0:
            is_left_frag = is_frag[bond_index[0]]
            bond_index_left, bond_index_right = bond_index[:, is_left_frag]

            left_feat = self.left_lin_bond(h_node[bond_index_left])
            right_feat = self.right_lin_bond(h_node[bond_index_right])
            weight_bond = self.bond_lin(h_bond[is_left_frag], left_feat * right_feat)
            force_bond = weight_bond * (pos_node[bond_index_left] - pos_node[bond_index_right])
            delta_pos = delta_pos + scatter_sum(force_bond, bond_index_left, dim=0, dim_size=h_node.shape[0])
        
        pos_update = pos_node + delta_pos / 10.
        return pos_update #TODO: use only frag pos instead of all pos to save memory


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


