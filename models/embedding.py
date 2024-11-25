import torch
from torch.nn import Module, Linear, Embedding
from torch.nn import functional as F
from models.common import GaussianSmearing, MLP, NONLINEARITIES


class ContinuousEmbedding(Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.emb_layer = Linear(in_dim, out_dim, bias=False)


    def forward(self, inputs):
        return self.emb_layer(inputs)

class AtomEmbeddingVN(Module):
    def __init__(self, in_scalar, in_vector,
                 out_scalar, out_vector, vector_normalizer=20.):
        super().__init__()
        assert in_vector == 1
        self.in_scalar = in_scalar
        self.vector_normalizer = vector_normalizer
        self.emb_sca = Linear(in_scalar, out_scalar)
        self.emb_vec = Linear(in_vector, out_vector)

    def forward(self, scalar_input, vector_input):
        vector_input = vector_input / self.vector_normalizer
        assert vector_input.shape[1:] == (3, ), 'Not support. Only one vector can be input'
        sca_emb = self.emb_sca(scalar_input[:, :self.in_scalar])  # b, f -> b, f'
        vec_emb = vector_input.unsqueeze(-1)  # b, 3 -> b, 3, 1
        vec_emb = self.emb_vec(vec_emb).transpose(1, -1)  # b, 1, 3 -> b, f', 3
        return sca_emb, vec_emb
        
class BondEmbedding(Module):
    def __init__(self, in_dim, out_dim, cutoff):
        super().__init__()
        self.in_dim = in_dim
        self.distance_expansion = GaussianSmearing(stop=cutoff, num_gaussians=20)
        self.emb_layer = Linear(in_dim+20, out_dim)

    def forward(self, bond_feat, bond_index, pos_nodes):
        distance = torch.norm(pos_nodes[bond_index[0]] - pos_nodes[bond_index[1]], p=2, dim=-1)
        dist_feat = self.distance_expansion(distance)
        bond_feat = torch.cat([
            bond_feat, dist_feat
        ], dim=-1)
        return self.emb_layer(bond_feat)