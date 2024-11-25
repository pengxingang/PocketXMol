import torch
import torch.nn.functional as F
from torch.nn import Module, Sequential, ModuleList, Linear, Conv1d, LeakyReLU, LayerNorm
from torch_geometric.nn import radius_graph, knn_graph
# from torch_geometric.utils import sort_edge_index
from torch_scatter import scatter_sum, scatter_softmax
from math import pi as PI

from models.common import GaussianSmearing, ShiftedSoftplus, EdgeExpansion
from models.invariant import GVLinear, GVPerceptronVN, VNLinear, VNLeakyReLU, MessageModule
# from utils.profile import lineprofile

class AttentionInteractionBlock(Module):

    def __init__(self, hidden_channels, edge_channels):
        super().__init__()

        self.v_lin = Linear(hidden_channels, hidden_channels, bias=False)
        self.weight_v_net = Sequential(
            Linear(edge_channels, hidden_channels),
            ShiftedSoftplus(),
            Linear(hidden_channels, hidden_channels),
        )
        self.weight_v_lin = Linear(hidden_channels, hidden_channels)

        self.centroid_lin = Linear(hidden_channels, hidden_channels)
        self.act = ShiftedSoftplus()
        self.out_transform = Linear(hidden_channels, hidden_channels)

    def forward(self, x, edge_index, edge_attr):
        """
        Args:
            x:  Node features, (N, H).
            edge_index: (2, E).
            edge_attr:  (E, H)
        """
        N = x.size(0)
        row, col = edge_index   # (E,) , (E,)
        h_values = self.v_lin(x)

        # Compose messages
        W_v = self.weight_v_net(edge_attr)  # (E, H)
        msg_j = self.weight_v_lin(W_v * h_values[col])  # (E, H)
        
        # Aggregate messages
        aggr_msg = scatter_sum(msg_j, row, dim=0, dim_size=N) # (N, H)
        out = self.centroid_lin(x) + aggr_msg

        out = self.out_transform(self.act(out))
        return out


class CFTransformerEncoder(Module):
    
    def __init__(self, hidden_channels=256, edge_channels=64, num_interactions=6, k=32, cutoff=10.0, **kwargs):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.edge_channels = edge_channels
        # self.key_channels = key_channels
        # self.num_heads = num_heads
        self.num_interactions = num_interactions
        # self.k = k
        # self.cutoff = cutoff


        self.distance_expansion = GaussianSmearing(stop=cutoff, num_gaussians=edge_channels)
        self.interactions = ModuleList()
        for _ in range(num_interactions):
            block = AttentionInteractionBlock(
                hidden_channels=hidden_channels,
                edge_channels=edge_channels,
                # key_channels=key_channels,
                # num_heads=num_heads,
            )
            self.interactions.append(block)

    @property
    def out_channels(self):
        return self.hidden_channels

    def forward(self, node_attr, pos, edge_index):
        # edge_index = radius_graph(pos, self.cutoff, batch=batch, loop=False)
        # edge_index = knn_graph(pos, k=self.k, batch=batch, flow='target_to_source')
        edge_length = torch.norm(pos[edge_index[0]] - pos[edge_index[1]], dim=1, p=2)
        edge_attr = self.distance_expansion(edge_length)

        h = node_attr
        for interaction in self.interactions:
            h = h + interaction(h, edge_index, edge_attr)
        return h


class CFTransformerEncoderVN(Module):
    
    def __init__(self, hidden_channels=[256, 64], edge_channels=64, num_interactions=6, k=32, cutoff=10.0, **kwargs):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.edge_channels = edge_channels
        # self.key_channels = key_channels  # not use
        # self.num_heads = num_heads  # not use
        self.num_interactions = num_interactions
        self.k = k
        self.cutoff = cutoff

        self.interactions = ModuleList()
        for _ in range(num_interactions):
            block = AttentionInteractionBlockVN(
                hidden_channels=hidden_channels,
                edge_channels=edge_channels,
                # num_edge_types=num_edge_types,
                # key_channels=key_channels,
                # num_heads=num_heads,
                cutoff = cutoff,
                **kwargs
            )
            self.interactions.append(block)

    @property
    def out_sca(self):
        return self.hidden_channels[0]
    
    @property
    def out_vec(self):
        return self.hidden_channels[1]

    def forward(self, node_attr, pos, edge_index):

        # edge_index_bt = knn_graph(pos, k=self.k, batch=batch, flow='target_to_source')
        # assert torch.all(sort_edge_index(edge_index)[0] == sort_edge_index(edge_index_bt)[0])
        edge_vector = pos[edge_index[0]] - pos[edge_index[1]]

        h = list(node_attr)
        for interaction in self.interactions:
            delta_h = interaction(h, edge_index, edge_vector)
            h[0] = h[0] + delta_h[0]
            h[1] = h[1] + delta_h[1]
        return h


class AttentionInteractionBlockVN(Module):

    def __init__(self, hidden_channels, edge_channels, cutoff=10., **kwargs):
        super().__init__()
        # self.use_atten = False

        # edge features
        self.distance_expansion = GaussianSmearing(stop=cutoff, num_gaussians=edge_channels)
        self.vector_expansion = EdgeExpansion(edge_channels)  # Linear(in_features=1, out_features=edge_channels, bias=False)
        ## compare encoder and classifier message passing

        # edge weigths and linear for values
        self.message_module = MessageModule(hidden_channels[0], hidden_channels[1], edge_channels, edge_channels,
                                                                                hidden_channels[0], hidden_channels[1], cutoff, **kwargs)

        # centroid nodes and finall linear
        self.centroid_lin = GVLinear(hidden_channels[0], hidden_channels[1], hidden_channels[0], hidden_channels[1])
        self.act_sca = LeakyReLU()
        self.act_vec = VNLeakyReLU(hidden_channels[1])
        self.out_transform = GVLinear(hidden_channels[0], hidden_channels[1], hidden_channels[0], hidden_channels[1])

        self.layernorm_sca = LayerNorm([hidden_channels[0]])
        self.layernorm_vec = LayerNorm([hidden_channels[1], 3])


    def forward(self, x, edge_index, edge_vector):
        """
        Args:
            x:  Node features: scalar features (N, feat), vector features(N, feat, 3)
            edge_index: (2, E).
            edge_attr:  (E, H)
        """
        scalar, vector = x
        N = scalar.size(0)
        row, col = edge_index   # (E,) , (E,)

        # Compute edge features
        edge_dist = torch.norm(edge_vector, dim=-1, p=2)
        edge_sca_feat = self.distance_expansion(edge_dist)
        edge_vec_feat = self.vector_expansion(edge_vector) 
        msg_j_sca, msg_j_vec = self.message_module(x, (edge_sca_feat, edge_vec_feat), col, edge_dist, annealing=True)

        # Aggregate messages
        aggr_msg_sca = scatter_sum(msg_j_sca, row, dim=0, dim_size=N)  #.view(N, -1) # (N, heads*H_per_head)
        aggr_msg_vec = scatter_sum(msg_j_vec, row, dim=0, dim_size=N)  #.view(N, -1, 3) # (N, heads*H_per_head, 3)
        x_out_sca, x_out_vec = self.centroid_lin(x)
        out_sca = x_out_sca + aggr_msg_sca
        out_vec = x_out_vec + aggr_msg_vec

        out_sca = self.layernorm_sca(out_sca)
        out_vec = self.layernorm_vec(out_vec)
        out = self.out_transform((self.act_sca(out_sca), self.act_vec(out_vec)))
        return out


if __name__ == '__main__':
    from torch_geometric.data import Data, Batch

    hidden_channels = 64
    edge_channels = 48
    key_channels = 32
    num_heads = 4

    data_list = []
    for num_nodes in [11, 13, 15]:
        data_list.append(Data(
            x = torch.randn([num_nodes, hidden_channels]),
            pos = torch.randn([num_nodes, 3]) * 2
        ))
    batch = Batch.from_data_list(data_list)

    model = CFTransformerEncoder(
        hidden_channels = hidden_channels,
        edge_channels = edge_channels,
        key_channels = key_channels,
        num_heads = num_heads,
    )
    out = model(batch.x, batch.pos, batch.batch)

    print(out)
    print(out.size())
