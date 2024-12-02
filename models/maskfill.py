from easydict import EasyDict
from tqdm import tqdm
import torch
from torch.nn import Module
from torch.nn import functional as F
# from torch_cluster import radius_graph
# from torch_geometric.nn import radius, knn

# from models.diffusion import extract
# from models.egnn import EGNN, EGNNEncodeer
# from models.transition import CategoricalTransition, ContigousTransition, GeneralCategoricalTransition
from models.graph import NodeEdgeNet
from models.graph_context import ContextNodeEdgeNet
from models.graph_gvp import ContextNodeEdgeNetGVP
from models.ipa import ContextGAEdgeNet, GAEncoder

# from .encoders import get_encoder_vn
# from .fields import SimpleEdgePredictor, get_field_vn, FragmentPosDecoder
from .common import *
from .corrector import correct_pos, get_dihedral_batch
# from .embedding import AtomEmbedding, BondEmbedding
# from .position import PositionPredictor
# from .sample_grid import get_grids
from .diffusion import *
# from .debug import check_true_bonds_len, check_pred_bonds_len
# from utils.misc import unique



class PMAsymDenoiser(Module):
    def __init__(self,
        config,
        num_node_types,
        num_edge_types,  # explicit bond type: 0, 1, 2, 3, 4
        pocket_in_dim,
        **kwargs
    ):
        super().__init__()
        self.config = config
        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types
        gvp = getattr(config, 'gvp', False)
        
        # # pocket encoder
        pocket_dim = config.pocket_dim
        self.pocket_embedder = nn.Linear(pocket_in_dim, pocket_dim)
        pocket_name = getattr(config.pocket, 'name', 'default')
        if pocket_name == 'default':
            pocket_encoder_bb = ContextNodeEdgeNet if not gvp else ContextNodeEdgeNetGVP
        elif pocket_name == 'ipa':
            pocket_encoder_bb = GAEncoder
        self.pocket_encoder = pocket_encoder_bb(pocket_dim, node_only=True, **config.pocket)
        
        # # mol embedding
        self.addition_node_features = getattr(config, 'addition_node_features', [])
        node_dim = config.node_dim
        edge_dim = config.edge_dim
        node_emb_dim = node_dim - 2 - len(self.addition_node_features)  # 2 for fixed node and pos
        self.nodetype_embedder = nn.Embedding(num_node_types, node_emb_dim)
        self.edgetype_embedder = nn.Embedding(num_edge_types, edge_dim-2)  # 2 for fixed edgetype and dist
        
        # # denoiser
        denoiser_name = getattr(config.denoiser, 'name', 'default')
        if denoiser_name == 'default':
            denoiser_bb = ContextNodeEdgeNet if not gvp else ContextNodeEdgeNetGVP
        elif denoiser_name == 'ipa':
            denoiser_bb = ContextGAEdgeNet
        self.denoiser = denoiser_bb(node_dim, edge_dim,
                            context_dim=pocket_dim, **config.denoiser)

        # # decoder
        self.node_decoder = MLP(node_dim, num_node_types, node_dim)
        self.edge_decoder = MLP(edge_dim, num_edge_types, edge_dim)
        
        # additional output
        self.add_output = getattr(config, 'add_output', [])
        if 'confidence' in self.add_output:  # condidence
            self.node_cfd = MLP(node_dim, 1, node_dim//2)
            self.pos_cfd = MLP(node_dim, 1, node_dim//2)
            self.edge_cfd = MLP(edge_dim, 1, edge_dim//2)
            

    def forward(self, batch, **kwargs):
        """
        Predict Mol at t=0 given perturbed Mol at t with hidden dims and time
        Predict the position t=0 and node/edge type reconstruction v_0, given perturbed pos and nodes (r_t, v_t)
        """

        # # 1. prepare embedding 
        pos_in = batch['pos_in']
        h_node_in = self.nodetype_embedder(batch['node_in'])
        h_halfedge_in = self.edgetype_embedder(batch['halfedge_in'])
        # pos_in = batch['node_pos']
        # h_node_in = self.nodetype_embedder(batch['node_type'])
        # h_halfedge_in = self.edgetype_embedder(batch['halfedge_type'])
        
        # add fixed indicator as extra features
        node_extra = torch.stack([batch['fixed_node'], batch['fixed_pos']], dim=1).to(pos_in.dtype)
        # node_extra = torch.ones_like(node_extra)
        halfedge_extra = torch.stack([batch['fixed_halfedge'], batch['fixed_halfdist']], dim=1).to(pos_in.dtype)
        # halfedge_extra = torch.ones_like(halfedge_extra)
        h_node_in = torch.cat([h_node_in, node_extra], dim=-1)
        h_halfedge_in = torch.cat([h_halfedge_in, halfedge_extra], dim=-1)

        # break symmetry
        n_halfedges = h_halfedge_in.shape[0]
        halfedge_index = batch['halfedge_index']
        edge_index = torch.cat([halfedge_index, halfedge_index.flip(0)], dim=1)
        h_edge_in = torch.cat([h_halfedge_in, h_halfedge_in], dim=0)
        edge_extra = torch.cat([halfedge_extra, halfedge_extra], dim=0)
        
        # additonal node features
        if 'is_peptide' in self.addition_node_features:
            is_peptide = batch['is_peptide'].unsqueeze(-1).to(pos_in.dtype)
            h_node_in = torch.cat([h_node_in, is_peptide], dim=-1)
        
        # # encode pocket
        h_pocket = self.pocket_embedder(batch['pocket_atom_feature'])
        h_pocket = self.pocket_encoder(
            h_node=h_pocket,
            pos_node=batch['pocket_pos'],
            edge_index=batch['pocket_knn_edge_index'],
            h_edge=None,
            node_extra=None,
            edge_extra=None,
        )

        # # 2 diffuse to get the updated node embedding and bond embedding
        # device = h_node_in.device
        h_node, pos_node, h_edge = self.denoiser(
            h_node=h_node_in,
            pos_node=pos_in, 
            h_edge=h_edge_in, 
            edge_index=edge_index,
            node_extra=node_extra,
            edge_extra=edge_extra,
            batch_node=batch['node_type_batch'],
            # pocket
            h_ctx=h_pocket,
            pos_ctx=batch['pocket_pos'],
            batch_ctx=batch['pocket_pos_batch'],
        )
        
        pred_node = self.node_decoder(h_node)
        pred_halfedge = self.edge_decoder(h_edge[:n_halfedges]+h_edge[n_halfedges:])  # NOTE why not divide by 2?
        pred_pos = pos_node
        
        additional_outputs = {}
        if 'confidence' in self.add_output:
            pred_node_cfd = self.node_cfd(h_node)
            pred_pos_cfd = self.pos_cfd(h_node)  # use the node hidden
            pred_edge_cfd = self.edge_cfd(h_edge[:n_halfedges]+h_edge[n_halfedges:])  # NOTE why not divide by 2?
            additional_outputs = {'confidence_node': pred_node_cfd, 'confidence_pos': pred_pos_cfd, 'confidence_halfedge': pred_edge_cfd}
        # elif 'dihedral' in self.add_output:
        #     sin_out, cos_out = get_dihedral_batch(pred_pos, batch['tor_bonds_anno'], batch['dihedral_pairs_anno'])
        #     additional_outputs.update({'dih_sin': sin_out, 'dih_cos': cos_out})
        # # add last emb (temprary hack for pep node emb)
        # additional_outputs.update({'emb_node': h_node})
        
        # if kwargs.get('pos_corr', False):
        #     sin_in, cos_in = get_dihedral_batch(pos_in, batch['tor_bonds_anno'], batch['dihedral_pairs_anno'])
        #     pred_pos_corr = correct_pos(
        #         pos_in=pos_in.clone(), pos_out=pos_node.clone(),
        #         sin_in=sin_in, cos_in=cos_in,
        #         sin_out=sin_out, cos_out=cos_out,
        #         domain_node_index=batch['domain_node_index'],
        #         domain_center_nodes=batch['domain_center_nodes'],
        #         tor_bonds_anno=batch['tor_bonds_anno'],
        #         twisted_nodes_anno=batch['twisted_nodes_anno'],
        #         dihedral_pairs_anno=batch['dihedral_pairs_anno'],
        #     )
        #     additional_outputs.update({'pred_pos_corr': pred_pos_corr})

        return {
            'pred_node': pred_node,
            'pred_pos': pred_pos,
            'pred_halfedge': pred_halfedge,
            **additional_outputs,
        }
