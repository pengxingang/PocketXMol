import numpy as np
import torch
from torch.nn import Module, Linear, Sequential, LayerNorm
from torch_geometric.nn import radius, knn
from torch_geometric.utils import sort_edge_index
from torch_scatter import scatter_add, scatter_softmax, scatter_sum

from math import pi as PI


from ..common import ShiftedSoftplus, GaussianSmearing, EdgeExpansion, MLP
from ..invariant import GVLinear, GVPerceptronVN, MessageModule
from ..position import SingleAtomPredictor
# from utils.profile import lineprofile

class SpatialClassifierVN(Module):

    def __init__(self, num_classes, in_sca, in_vec, num_filters, edge_channels, cutoff=10.0, **kwargs):
        super().__init__()
        self.message_module = MessageModule(in_sca, in_vec, edge_channels, edge_channels, num_filters[0], num_filters[1], cutoff, **kwargs)

        self.classifier = Sequential(
            GVPerceptronVN(num_filters[0], num_filters[1], num_filters[0], num_filters[1], **kwargs),
            GVLinear(num_filters[0], num_filters[1], num_classes, 1)
        )

        self.distance_expansion = GaussianSmearing(stop=cutoff, num_gaussians=edge_channels)
        self.vector_expansion = EdgeExpansion(edge_channels)  # Linear(in_features=1, out_features=edge_channels, bias=False)
        self.cutoff = cutoff

    def forward(self, pos_query, pos_compose, node_attr_compose, edge_index_q_cps_knn):
        # (self, pos_query, edge_index_query, pos_ctx, node_attr_ctx, is_mol_atom, batch_query, batch_edge, batch_ctx):
        """
        Args:
            pos_query:   (N_query, 3)
            edge_index_query: (2, N_q_c, )
            pos_ctx:     (N_ctx, 3)
            node_attr_ctx:  (N_ctx, H)
            is_mol_atom: (N_ctx, )
            batch_query: (N_query, )
            batch_ctx:   (N_ctx, )
        Returns
            (N_query, num_classes)
        """

        # Pairwise distances and contextual node features
        vec_ij = pos_query[edge_index_q_cps_knn[0]] - pos_compose[edge_index_q_cps_knn[1]]
        dist_ij = torch.norm(vec_ij, p=2, dim=-1).view(-1, 1)  # (A, 1)
        edge_ij = self.distance_expansion(dist_ij), self.vector_expansion(vec_ij)
        
        # node_attr_ctx_j = [node_attr_ctx_[edge_index_q_cps_knn[1]] for node_attr_ctx_ in node_attr_ctx]  # (A, H)
        h = self.message_module(node_attr_compose, edge_ij, edge_index_q_cps_knn[1], dist_ij, annealing=True)

        # Aggregate messages
        y = [scatter_add(h[0], index=edge_index_q_cps_knn[0], dim=0, dim_size=pos_query.size(0)), # (N_query, F)
                scatter_add(h[1], index=edge_index_q_cps_knn[0], dim=0, dim_size=pos_query.size(0))]

        # element prediction
        y_cls, _ = self.classifier(y)  # (N_query, num_classes)

        return y_cls


class SimpleEdgePredictor(Module):

    def __init__(self, hidden_dim, num_edge_types, max_fragment, num_r_gaussian, cutoff, **kwargs):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_edge_types = num_edge_types
        self.max_fragment = max_fragment

        # linear transformation for attention 
        dim_input_pred = 2 * hidden_dim + num_r_gaussian + 1  # emb of two nodes + edge dist rbf + type indcators of two nodes
        self.edge_pred = MLP(dim_input_pred, 2 * hidden_dim, num_edge_types, 1)
        self.distance_expansion = GaussianSmearing(0., cutoff, num_gaussians=num_r_gaussian)


    def forward(self, h_mol, pos_mol, h_frag, pos_frag, batch_mol, batch_frag,
        return_index=False,
    ):
        # build frag pairwise index
        device = pos_mol.device
        # edge_index_ff_i, edge_index_ff_j = [], []  # edges of fragment atoms - fragment atoms
        # acc_num_nodes = 0
        # for frag in torch.arange(batch_frag.max() + 1):
        #     num_nodes = (batch_frag == frag).sum()
        #     index_edge_i = torch.arange(num_nodes, dtype=torch.long, device=device) + acc_num_nodes
        #     index_edge_i, index_edge_j = torch.meshgrid(index_edge_i, index_edge_i)
        #     index_edge_i, index_edge_j = index_edge_i.flatten(), index_edge_j.flatten()
        #     edge_index_ff_i.append(index_edge_i)
        #     edge_index_ff_j.append(index_edge_j)
        #     acc_num_nodes += num_nodes
        # edge_index_ff_i = torch.cat(edge_index_ff_i, dim=0)
        # edge_index_ff_j = torch.cat(edge_index_ff_j, dim=0)
        max_fragment  = self.max_fragment  # (batch_frag==0).sum()
        index_edge_i = torch.arange(max_fragment, dtype=torch.long, device=device)
        index_edge_i, index_edge_j = torch.meshgrid(index_edge_i, index_edge_i)
        index_edge_i, index_edge_j = index_edge_i.flatten(), index_edge_j.flatten()
        edge_index_ff_i = torch.cat([index_edge_i+i*max_fragment for i in range(batch_frag.max()+1)], dim=0)
        edge_index_ff_j = torch.cat([index_edge_j+i*max_fragment for i in range(batch_frag.max()+1)], dim=0)


        # predict for the edges of fragment atoms
        node_i_feat = h_frag[edge_index_ff_i]
        node_j_feat = h_frag[edge_index_ff_j]
        dist_ij = torch.norm(pos_frag[edge_index_ff_i] - pos_frag[edge_index_ff_j], p=2, dim=-1).view(-1, 1)
        dist_ij = self.distance_expansion(dist_ij)
        types_ij = torch.zeros(edge_index_ff_i.size(0), dtype=torch.long, device=device)  # 0 for frag-frag edges
        edge_ij = torch.cat([node_i_feat, node_j_feat, dist_ij, types_ij.view(-1, 1)], dim=1)
        edge_pred_ff = self.edge_pred(edge_ij)

        # Build edge index between frag and mol
        edge_index_mf_i, edge_index_mf_j = [], []
        acc_num_nodes_mol = 0
        acc_num_nodes_frag = 0
        for frag in torch.arange(batch_frag.max() + 1):
            num_nodes_mol = (batch_mol == frag).sum()
            num_nodes_frag = (batch_frag == frag).sum()
            index_edge_mol = torch.arange(num_nodes_mol, dtype=torch.long, device=device) + acc_num_nodes_mol
            index_edge_frag = torch.arange(num_nodes_frag, dtype=torch.long, device=device) + acc_num_nodes_frag
            index_edge_mol, index_edge_frag = torch.meshgrid(index_edge_mol, index_edge_frag)
            edge_index_mf_i.append(index_edge_mol.flatten())
            edge_index_mf_j.append(index_edge_frag.flatten())
            acc_num_nodes_mol += num_nodes_mol
            acc_num_nodes_frag += num_nodes_frag
        edge_index_mf_i = torch.cat(edge_index_mf_i, dim=0)
        edge_index_mf_j = torch.cat(edge_index_mf_j, dim=0)  #TODO: can not use loop

        # Predict for the edges between fragment atoms and molecule atoms
        node_i_feat = h_mol[edge_index_mf_i]
        node_j_feat = h_frag[edge_index_mf_j]
        dist_ij = torch.norm(pos_mol[edge_index_mf_i] - pos_frag[edge_index_mf_j], p=2, dim=-1).view(-1, 1)
        dist_ij = self.distance_expansion(dist_ij)
        types_ij = torch.ones(edge_index_mf_i.size(0), dtype=torch.long, device=device)  # 1 for frag-mol edges
        edge_ij = torch.cat([node_i_feat, node_j_feat, dist_ij, types_ij.view(-1, 1)], dim=1)
        edge_pred_mf = self.edge_pred(edge_ij)

        edge_pred_ff = edge_pred_ff.reshape(-1, self.max_fragment, self.num_edge_types)
        edge_pred_mf = edge_pred_mf.reshape(-1, self.max_fragment, self.num_edge_types)
        
        if return_index:
            edge_index_ff = torch.stack([edge_index_ff_i, edge_index_ff_j], dim=0)
            edge_index_mf = torch.stack([edge_index_mf_i, edge_index_mf_j], dim=0)
            return edge_pred_ff, edge_pred_mf, edge_index_ff, edge_index_mf
        else:
            return edge_pred_ff, edge_pred_mf

class EdgePredictorVN(Module):

    def __init__(self, in_sca, in_vec, key_channels, num_heads=1, num_bond_types=3, **kwargs):
        super().__init__()
        hidden_channels = [in_sca, in_vec]
        assert (hidden_channels[0] % num_heads == 0) and (hidden_channels[1] % num_heads == 0)
        assert (key_channels[0] % num_heads == 0) and (key_channels[1] % num_heads == 0)

        self.hidden_channels = hidden_channels
        self.key_channels = key_channels
        self.num_heads = num_heads

        # linear transformation for attention 
        self.q_lin = GVLinear(hidden_channels[0], hidden_channels[1], key_channels[0], key_channels[1])
        self.k_lin = GVLinear(hidden_channels[0], hidden_channels[1], key_channels[0], key_channels[1])
        self.v_lin = GVLinear(hidden_channels[0], hidden_channels[1], hidden_channels[0], hidden_channels[1])

        self.edge_pred = GVLinear(2 * hidden_channels[0], 2 * hidden_channels[1], num_bond_types + 1, 1)

        self.layernorm_sca = LayerNorm([hidden_channels[0]])
        self.layernorm_vec = LayerNorm([hidden_channels[1], 3])

    def forward(self, h_compose, idx_ligand, batch_ligand, return_index=False):
        h_ligand = [h_compose[0][idx_ligand], h_compose[1][idx_ligand]]
        scalar, vector = h_ligand
        N = scalar.size(0)

        # Build keys/queries index
        index_edge_i_list, index_edge_j_list = [], []
        acc_num_edges = 0
        for mol in torch.arange(batch_ligand.max() + 1):
            num_edges = (batch_ligand == mol).sum()
            index_edge_i = torch.arange(num_edges, dtype=torch.long, device=scalar.device) + acc_num_edges
            index_edge_i, index_edge_j = torch.meshgrid(index_edge_i, index_edge_i)
            index_edge_i, index_edge_j = index_edge_i.flatten(), index_edge_j.flatten()
            index_edge_i_list.append(index_edge_i)
            index_edge_j_list.append(index_edge_j)
            acc_num_edges += num_edges
        index_edge_i_list = torch.cat(index_edge_i_list, dim=0)
        index_edge_j_list = torch.cat(index_edge_j_list, dim=0)

        # Project to multiple key, query and value spaces
        h_queries = self.q_lin(h_ligand)
        h_queries = (h_queries[0].view(N, self.num_heads, -1),  # (N, heads, K_per_head)
                    h_queries[1].view(N, self.num_heads, -1, 3))  # (N, heads, K_per_head, 3)
        h_keys = self.k_lin(h_ligand)
        h_keys = (h_keys[0].view(N, self.num_heads, -1),  # (N, heads, K_per_head)
                    h_keys[1].view(N, self.num_heads, -1, 3))  # (N, heads, K_per_head, 3)
        h_values = self.v_lin(h_ligand)
        h_values = (h_values[0].view(N, self.num_heads, -1),  # (N, heads, K_per_head)
                    h_values[1].view(N, self.num_heads, -1, 3))  # (N, heads, K_per_head, 3)

        # Attention
        queries_i = [h_queries[0][index_edge_i_list], h_queries[1][index_edge_i_list]]
        keys_j = [h_keys[0][index_edge_j_list], h_keys[1][index_edge_j_list]]
        qk_ij = [
            (queries_i[0] * keys_j[0]).sum(-1) / np.sqrt(h_queries[0].size(-1)), 
            (queries_i[1] * keys_j[1]).sum(-1).sum(-1) / np.sqrt(h_queries[1].size(-1) * h_queries[1].size(-2))
        ]
        alpha = [
            scatter_softmax(qk_ij[0], index_edge_i_list, dim=0),  # (N', heads)
            scatter_softmax(qk_ij[1], index_edge_i_list, dim=0)  # (N', heads)
        ] 
        values_j = [h_values[0][index_edge_j_list], h_values[1][index_edge_j_list]]
        num_attens = len(index_edge_j_list)
        h_ligand_attened =[
            scatter_sum((alpha[0].unsqueeze(-1) * values_j[0]).view(num_attens, -1), index_edge_i_list, dim=0, dim_size=N),   # (N, H, 3)
            scatter_sum((alpha[1].unsqueeze(-1).unsqueeze(-1) * values_j[1]).view(num_attens, -1, 3), index_edge_i_list, dim=0, dim_size=N)   # (N, H, 3)
        ]
        h_ligand_attened = [  # skip connection
            h_ligand_attened[0] + h_ligand[0],
            h_ligand_attened[1] + h_ligand[1]
        ]
        h_ligand_attened = [
            self.layernorm_sca(h_ligand_attened[0]),
            self.layernorm_vec(h_ligand_attened[1])
        ]

        # Build edge index
        edge_index = []
        acc_num_edges = 0
        for mol in torch.arange(batch_ligand.max() + 1):
            num_edges = (batch_ligand == mol).sum()
            index_this_mol = torch.triu_indices(num_edges, num_edges, offset=1, device=scalar.device) + acc_num_edges
            edge_index.append(index_this_mol)
            acc_num_edges += num_edges
        edge_index = torch.cat(edge_index, dim=1)

        # Predict edge type
        edge_features = [
            torch.cat([h_ligand_attened[0][edge_index[0]], h_ligand_attened[0][edge_index[1]]], dim=1),  # (N, 2 * H)
            torch.cat([h_ligand_attened[1][edge_index[0]], h_ligand_attened[1][edge_index[1]]], dim=1),  # (N, 2 * H, 3)
        ]
        edge_types = self.edge_pred(edge_features)[0]  # (N, num_bond_types + 1)
        if return_index:
            return edge_types, edge_index
        else:
            return edge_types


class FragmentDecoder(Module):
    def __init__(self, num_elements, in_sca, in_vec, num_filters, edge_channels, max_fragments, cutoff=10.0):
        super(FragmentDecoder, self).__init__()
        self.max_fragments = max_fragments
        self.num_elements = num_elements
        
        self.message_module = MessageModule(in_sca, in_vec, edge_channels, edge_channels, num_filters[0], num_filters[1], cutoff)
        self.processor = Sequential(
            GVPerceptronVN(num_filters[0], num_filters[1], num_filters[0], num_filters[1]),
            GVLinear(num_filters[0], num_filters[1], num_filters[0], num_filters[1])
        )
        self.out_net = GVLinear(num_filters[0], num_filters[1], num_elements+1, 1)  # plus one for the END token

        self.distance_expansion = GaussianSmearing(stop=cutoff, num_gaussians=edge_channels)
        self.vector_expansion = EdgeExpansion(edge_channels)  # Linear(in_features=1, out_features=edge_channels, bias=False)

    def forward(self, h_compose, pos_query, pos_compose, q_cps_edge_index, num_gen_atoms):

        # Pairwise distances and contextual node features
        vec_ij = pos_query[q_cps_edge_index[0]] - pos_compose[q_cps_edge_index[1]]
        dist_ij = torch.norm(vec_ij, p=2, dim=-1).view(-1, 1)  # (A, 1)
        edge_ij = self.distance_expansion(dist_ij), self.vector_expansion(vec_ij)
        
        # node_attr_ctx_j = [node_attr_ctx_[edge_index_q_cps_knn[1]] for node_attr_ctx_ in node_attr_ctx]  # (A, H)
        h = self.message_module(h_compose, edge_ij, q_cps_edge_index[1], dist_ij, annealing=True)

        # Aggregate messages
        hidden = [scatter_add(h[0], index=q_cps_edge_index[0], dim=0, dim_size=pos_query.size(0)), # (N_query, F)
                scatter_add(h[1], index=q_cps_edge_index[0], dim=0, dim_size=pos_query.size(0))]

        # Process
        elements_pred = torch.zeros([pos_query.size(0), self.max_fragments+1, self.num_elements+1], device=pos_query.device, dtype=pos_query.dtype)
        pos_pred = torch.zeros([pos_query.size(0), self.max_fragments+1, 3], device=pos_query.device, dtype=pos_query.dtype)
        # pos_std = torch.zeros([pos_query.size(0), self.max_fragments+1, 3], device=pos_query.device, dtype=pos_query.dtype)
        if num_gen_atoms is not None:
            if len(num_gen_atoms) > 0:
                num_gen = min(torch.max(num_gen_atoms, dim=0)[0], self.max_fragments)
            else:
                num_gen = 0
        else:
            num_gen = self.max_fragments
        for step in range(num_gen+1):
            hidden = self.processor(hidden)
            pred_atom = self.out_net(hidden)
            elements_pred[:, step, :] = pred_atom[0]

            # pred_pos = self.out_pos(hidden)[1]
            pos_pred[:, step, :] = pos_query + pred_atom[1][:, 0, :]  # predicted position (relative pos + center pos)
            # pred_std = self.out_std(hidden)[1]
            # pos_std[:, step, :] = torch.exp(pred_std[:, 0, :])  # predicted standard deviation = exp(log_std)

        return elements_pred, pos_pred[:, :-1, :] # the END token does not need pos prediction


class FragmentPosDecoder(Module):
    def __init__(self, in_sca, in_vec, num_filters, edge_channels, max_fragments, n_components, knn, cutoff=10.0):
        super().__init__()
        self.max_fragments = max_fragments
        self.n_components = n_components
        self.knn = knn
        self.singel_atom_decoder = SingleAtomPredictor(in_sca, in_vec, num_filters, n_components)
        self.singel_atom_decoder_from_protein = SingleAtomPredictor(in_sca, in_vec, num_filters, n_components)
        self.message_module = MessageModule(in_sca, in_vec, edge_channels, edge_channels, num_filters[0], num_filters[1], cutoff)
        self.update_module = MessageModule(in_sca, in_vec, edge_channels, edge_channels, num_filters[0], num_filters[1], cutoff)

        self.distance_expansion = GaussianSmearing(stop=cutoff, num_gaussians=edge_channels)
        self.vector_expansion = EdgeExpansion(edge_channels)  # Linear(in_features=1, out_features=edge_channels, bias=False)
    
    def generate_one_frag(self, stage, h_compose, index_query, pos_compose, n_samples=3):
        # predict distributions
        h_query_sca, h_query_vec = h_compose[0][index_query:index_query+1], h_compose[1][index_query:index_query+1]
        pos_query = pos_compose[index_query:index_query+1]
        
        pos_gen_list = []
        for step in range(self.max_fragments):
            if (stage == 'protein') and (step == 0):
                mu, sigma, pi, hasatom = self.singel_atom_decoder_from_protein([h_query_sca, h_query_vec], pos_query)
            else:
                mu, sigma, pi, hasatom = self.singel_atom_decoder([h_query_sca, h_query_vec], pos_query)
            if not (hasatom > 0).all():
                break

            # sample from distributions
            pos_gen = self.singel_atom_decoder.sample_batch(mu, sigma, pi, n_samples)  # (1, n_samples, 3)
            pos_gen = pos_gen.view(-1, 3)
            # scoring 
            pdf = self.singel_atom_decoder.get_mdn_probability(mu.repeat_interleave(n_samples, dim=0),
                                                    sigma.repeat_interleave(n_samples, dim=0),
                                                    pi.repeat_interleave(n_samples, dim=0),
                                                    pos_gen)
            idx_max = torch.argmax(pdf.view(-1))
            # idx_max = 0
            pos_gen = pos_gen[idx_max:idx_max+1]
            pos_gen_list.append(pos_gen)

            if step == self.max_fragments-1:
                break
            # update focal hidden
            q_cps_edge_index = knn(x=pos_compose, y=pos_gen, k=self.knn)
            h_generated = self.get_gen_hidden(h_compose, pos_compose, pos_gen, q_cps_edge_index)
            if (stage == 'protein') and (step == 0):  # change the focal position and hidden
                pos_query = pos_gen
                h_query_sca, h_query_vec = h_generated[0], h_generated[1]
            else:
                h_query_sca, h_query_vec = self.update_focal([h_query_sca, h_query_vec], pos_query, h_generated, pos_gen)

        return torch.cat(pos_gen_list, dim=0)


    def generate(self, stage, h_compose, index_query, pos_compose, n_gen=100):
        n_query = len(index_query)
        # n_gen = n_query * n_gen_per_query
        pos_generation = torch.zeros([n_gen, self.max_fragments, 3], device=pos_compose.device, dtype=pos_compose.dtype)
        n_atoms_gen = torch.zeros([n_gen], device=pos_compose.device, dtype=torch.int64)
        
        index_query = index_query[np.random.choice(range(n_query), n_gen, replace=True)]
        for i ,idx in enumerate(index_query):
            pos_gen_frag = self.generate_one_frag(stage, h_compose, idx, pos_compose)
            n_atoms = pos_gen_frag.size(0)
            pos_generation[i, :n_atoms, :] = pos_gen_frag
            n_atoms_gen[i] = n_atoms
        return pos_generation, n_atoms_gen
    

    def forward(self, stage, h_compose, index_query, pos_compose, pos_gen_true, pos_frag_mask, batch_compose, batch_query):
        n_query = len(index_query)
        mu_pred= torch.zeros([n_query, self.max_fragments, self.n_components, 3], device=pos_compose.device, dtype=pos_compose.dtype)
        sigma_pred= torch.zeros([n_query, self.max_fragments, self.n_components, 3], device=pos_compose.device, dtype=pos_compose.dtype)
        pi_pred= torch.zeros([n_query, self.max_fragments, self.n_components], device=pos_compose.device, dtype=pos_compose.dtype)
        hasatom_pred = torch.zeros([n_query, self.max_fragments, 1], device=pos_compose.device, dtype=pos_compose.dtype)

        # for the first atom
        h_focal_sca, h_focal_vec = h_compose[0][index_query], h_compose[1][index_query]
        pos_focal = pos_compose[index_query]
        for step in range(self.max_fragments):
            idx_valid_focal = torch.nonzero(pos_frag_mask[:, step]).reshape(-1) # [:, 0]
            h_focal_valid = h_focal_sca[idx_valid_focal], h_focal_vec[idx_valid_focal]
            pos_focal_valid = pos_focal[idx_valid_focal]
            # generate new atoms
            if (stage == 'protein') and (step == 0):#!changed here
            # if (stage == 'protein'):
                mu, sigma, pi, hasatom = self.singel_atom_decoder_from_protein(h_focal_valid, pos_focal_valid)
            else:
                mu, sigma, pi, hasatom = self.singel_atom_decoder(h_focal_valid, pos_focal_valid)
            mu_pred[idx_valid_focal, step, ...] = mu
            sigma_pred[idx_valid_focal, step, ...] = sigma
            pi_pred[idx_valid_focal, step, ...] = pi
            hasatom_pred[idx_valid_focal, step, ...] = hasatom
            # get the hidden of generated positions
            if step < self.max_fragments - 1:
                idx_continue = torch.nonzero(pos_frag_mask[:, step+1])[:, 0]
                if len(idx_continue) == 0:
                    break
            else:
                break
            h_focal_continue = h_focal_sca[idx_continue], h_focal_vec[idx_continue]
            pos_atoms = pos_gen_true[idx_continue, step, :]
            q_cps_edge_index = knn(x=pos_compose, y=pos_atoms, k=self.knn, batch_x=batch_compose, batch_y=batch_query[idx_continue])
            h_generated = self.get_gen_hidden(h_compose, pos_compose, pos_atoms, q_cps_edge_index)
            # update h_focal
            if (stage == 'protein') and (step == 0):  # change the focal position and hidden #!changed here
            # if False:  # change the focal position 、and hidden
                pos_focal = pos_gen_true[:, 0, :]
                h_focal_sca[idx_continue] = h_generated[0].to(h_focal_sca.dtype)
                h_focal_vec[idx_continue] = h_generated[1].to(h_focal_vec.dtype)
            else:
                h_focal_continue = self.update_focal(h_focal_continue, pos_focal[idx_continue], h_generated, pos_atoms)
                h_focal_sca[idx_continue] = h_focal_continue[0].to(h_focal_sca.dtype)
                h_focal_vec[idx_continue] = h_focal_continue[1].to(h_focal_vec.dtype)
            
        return mu_pred, sigma_pred, pi_pred, hasatom_pred

    def get_gen_hidden(self, h_compose, pos_compose, pos_gen, gen_cps_edge_index):
        vec_ij = pos_gen[gen_cps_edge_index[0]] - pos_compose[gen_cps_edge_index[1]]
        dist_ij = torch.norm(vec_ij, p=2, dim=-1).view(-1, 1)  # (A, 1)
        edge_ij = self.distance_expansion(dist_ij), self.vector_expansion(vec_ij)
        h = self.message_module(h_compose, edge_ij, gen_cps_edge_index[1], dist_ij, annealing=True)

        # Aggregate messages
        hidden = [scatter_add(h[0], index=gen_cps_edge_index[0], dim=0, dim_size=pos_gen.size(0)), # (N_query, F)
                scatter_add(h[1], index=gen_cps_edge_index[0], dim=0, dim_size=pos_gen.size(0))]

        return hidden
    
    def update_focal(self, h_focal, pos_focal, h_gen, pos_gen):
        vec_ij = pos_focal - pos_gen
        dist_ij = torch.norm(vec_ij, p=2, dim=-1).view(-1, 1)
        edge_ij = self.distance_expansion(dist_ij), self.vector_expansion(vec_ij)
        edge_index_0 = torch.arange(len(pos_gen), device=pos_focal.device, dtype=torch.long)
        h = self.update_module(h_gen, edge_ij, edge_index_0, dist_ij, annealing=True)
        
        h_focal_new = [h_focal[0] + h[0], h_focal[1] + h[1]]
        return h_focal_new

    def get_mdn_probability(self, *args, **kwargs):
        return self.singel_atom_decoder.get_mdn_probability(*args, **kwargs)


class AttentionEdges(Module):

    def __init__(self, hidden_channels, key_channels, num_heads=1, num_bond_types=3, **kwargs):
        super().__init__()
        
        assert (hidden_channels[0] % num_heads == 0) and (hidden_channels[1] % num_heads == 0)
        assert (key_channels[0] % num_heads == 0) and (key_channels[1] % num_heads == 0)

        self.hidden_channels = hidden_channels
        self.key_channels = key_channels
        self.num_heads = num_heads

        # linear transformation for attention 
        self.q_lin = GVLinear(hidden_channels[0], hidden_channels[1], key_channels[0], key_channels[1])
        self.k_lin = GVLinear(hidden_channels[0], hidden_channels[1], key_channels[0], key_channels[1])
        self.v_lin = GVLinear(hidden_channels[0], hidden_channels[1], hidden_channels[0], hidden_channels[1])

        if 'use_atten_bias' in kwargs:
            self.use_atten_bias = kwargs['use_atten_bias']
        else:  # default use
            self.use_atten_bias = True
        if self.use_atten_bias:
            self.atten_bias_lin = AttentionBias(self.num_heads, hidden_channels, num_bond_types=num_bond_types)

    def forward(self, edge_attr, edge_index, pos_compose, 
                          index_real_cps_edge_for_atten, tri_edge_index, tri_edge_feat,):
        """
        Args:
            x:  edge features: scalar features (N, feat), vector features(N, feat, 3)
            edge_attr:  (E, H)
            edge_index: (2, E). the row can be seen as batch_edge
        """
        scalar, vector = edge_attr
        N = scalar.size(0)
        row, col = edge_index   # (N,) 

        # Project to multiple key, query and value spaces
        h_queries = self.q_lin(edge_attr)
        h_queries = (h_queries[0].view(N, self.num_heads, -1),  # (N, heads, K_per_head)
                    h_queries[1].view(N, self.num_heads, -1, 3))  # (N, heads, K_per_head, 3)
        h_keys = self.k_lin(edge_attr)
        h_keys = (h_keys[0].view(N, self.num_heads, -1),  # (N, heads, K_per_head)
                    h_keys[1].view(N, self.num_heads, -1, 3))  # (N, heads, K_per_head, 3)
        h_values = self.v_lin(edge_attr)
        h_values = (h_values[0].view(N, self.num_heads, -1),  # (N, heads, K_per_head)
                    h_values[1].view(N, self.num_heads, -1, 3))  # (N, heads, K_per_head, 3)
        
        # build query and key index
        # index_edge_i_list, index_edge_j_list = [], []
        # acc_num_edges = 0
        # for node in torch.arange(row.max() + 1):
        #     num_edges = (row == node).sum()
        #     index_edge_i = torch.arange(num_edges, dtype=torch.long, device=scalar.device) + acc_num_edges
        #     index_edge_i, index_edge_j = torch.meshgrid(index_edge_i, index_edge_i)
        #     index_edge_i, index_edge_j = index_edge_i.flatten(), index_edge_j.flatten()
        #     index_edge_i_list.append(index_edge_i)
        #     index_edge_j_list.append(index_edge_j)
        #     acc_num_edges += num_edges
        # index_edge_i_list = torch.cat(index_edge_i_list, dim=0)
        # index_edge_j_list = torch.cat(index_edge_j_list, dim=0)

        # assert (index_edge_i_list == index_real_cps_edge_for_atten[0]).all()
        # assert (index_edge_j_list == index_real_cps_edge_for_atten[1]).all()
        index_edge_i_list, index_edge_j_list = index_real_cps_edge_for_atten

        # # get nodes of triangle edges
        # node_a_cps_tri_edge = col[index_edge_i_list]
        # node_b_cps_tri_edge = col[index_edge_j_list]
        # assert (node_a_cps_tri_edge == tri_edge_index[0]).all()
        # assert (node_b_cps_tri_edge == tri_edge_index[1]).all()
        if self.use_atten_bias:
            atten_bias = self.atten_bias_lin(
                tri_edge_index,
                tri_edge_feat,
                pos_compose,
            )

        # query * key
        queries_i = [h_queries[0][index_edge_i_list], h_queries[1][index_edge_i_list]]
        keys_j = [h_keys[0][index_edge_j_list], h_keys[1][index_edge_j_list]]

        qk_ij = [
            (queries_i[0] * keys_j[0]).sum(-1),  # (N', heads)
            (queries_i[1] * keys_j[1]).sum(-1).sum(-1)  # (N', heads)
        ]

        if self.use_atten_bias:
            alpha = [
                atten_bias[0] + qk_ij[0],
                atten_bias[1] + qk_ij[1]
            ]
        else:
            alpha = [
                qk_ij[0],
                qk_ij[1]
            ]

        alpha = [
            scatter_softmax(alpha[0], index_edge_i_list, dim=0),  # (N', heads)
            scatter_softmax(alpha[1], index_edge_i_list, dim=0)  # (N', heads)
        ] 

        values_j = [h_values[0][index_edge_j_list], h_values[1][index_edge_j_list]]
        num_attens = len(index_edge_j_list)
        output =[
            scatter_sum((alpha[0].unsqueeze(-1) * values_j[0]).view(num_attens, -1), index_edge_i_list, dim=0, dim_size=N),   # (N, H, 3)
            scatter_sum((alpha[1].unsqueeze(-1).unsqueeze(-1) * values_j[1]).view(num_attens, -1, 3), index_edge_i_list, dim=0, dim_size=N)   # (N, H, 3)
        ]

        # output 
        output = [edge_attr[0] + output[0], edge_attr[1] + output[1]]
        return output


class AttentionBias(Module):

    def __init__(self, num_heads, hidden_channels, cutoff=10., num_bond_types=3): #TODO: change the cutoff
        super().__init__()
        num_edge_types = num_bond_types + 1
        self.num_bond_types = num_bond_types
        self.distance_expansion = GaussianSmearing(stop=cutoff, num_gaussians=hidden_channels[0] - num_edge_types-1)  # minus 1 for self edges (e.g. edge 0-0)
        self.vector_expansion = EdgeExpansion(hidden_channels[1])  # Linear(in_features=1, out_features=hidden_channels[1], bias=False)
        self.gvlinear = GVLinear(hidden_channels[0], hidden_channels[1], num_heads, num_heads)

    def forward(self,  tri_edge_index, tri_edge_feat, pos_compose):
        node_a, node_b = tri_edge_index
        pos_a = pos_compose[node_a]
        pos_b = pos_compose[node_b]
        vector = pos_a - pos_b
        dist = torch.norm(vector, p=2, dim=-1)
        
        dist_feat = self.distance_expansion(dist)
        sca_feat = torch.cat([
            dist_feat,
            tri_edge_feat,
        ], dim=-1)
        vec_feat = self.vector_expansion(vector)
        output_sca, output_vec = self.gvlinear([sca_feat, vec_feat])
        output_vec = (output_vec * output_vec).sum(-1)
        return output_sca, output_vec
