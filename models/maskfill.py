"""
PocketXMol - Asymmetric Denoiser for Pocket-Molecule Interaction Modeling

This module (PMAsymDenoiser) implements the core denoising model for pocket-aware 3D molecule generation.

Key components:
    - Pocket encoder: Encodes protein pocket residues as context
    - Molecule embedder: Embeds atom types and bond types
    - Denoiser network: Remove noise from the noisy molecular hidden representations.
    - Output decoders: Predict clean atom types, positions, and bond types from denoised representations.
"""

# Third-party imports
import torch
from easydict import EasyDict
from torch.nn import Module
from torch.nn import functional as F
from tqdm import tqdm

# Local imports
from models.graph import NodeEdgeNet
from models.graph_context import ContextNodeEdgeNet
from models.graph_gvp import ContextNodeEdgeNetGVP
from models.ipa import ContextGAEdgeNet, GAEncoder

from models.common import *
from models.corrector import correct_pos, get_dihedral_batch
from models.diffusion import *


class PMAsymDenoiser(Module):
    """
    Pocket-Molecule Asymmetric Denoiser for 3D molecule generation.
    
    This model predicts clean molecular structures (atom types, 3D coordinates, and bonds) 
    from noisy inputs conditioned on the protein pocket and the task prompt variables.
    
    Args:
        config: Model configuration containing architecture parameters
        num_node_types: Number of atom types in the vocabulary
        num_edge_types: Number of edge types (typically: 0=none, 1=single, 2=double, 3=triple, 4=aromatic, and 5=MASK if use_edge_mask==True)
        pocket_in_dim: Input feature dimension for pocket atoms
    """
    
    def __init__(self,
        config,
        num_node_types,
        num_edge_types,
        pocket_in_dim,
        **kwargs
    ):
        super().__init__()
        self.config = config
        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types
        gvp = getattr(config, 'gvp', False)
        
        # Pocket encoder: processes protein context
        pocket_dim = config.pocket_dim
        self.pocket_embedder = nn.Linear(pocket_in_dim, pocket_dim)
        pocket_name = getattr(config.pocket, 'name', 'default')
        if pocket_name == 'default':
            pocket_encoder_bb = ContextNodeEdgeNet if not gvp else ContextNodeEdgeNetGVP
        elif pocket_name == 'ipa':
            pocket_encoder_bb = GAEncoder
        self.pocket_encoder = pocket_encoder_bb(pocket_dim, node_only=True, **config.pocket)
        
        # Molecule embedding layers
        self.addition_node_features = getattr(config, 'addition_node_features', [])
        node_dim = config.node_dim
        edge_dim = config.edge_dim
        # Reserve 2 dimensions for task prompt (fixed_node, fixed_pos)
        node_emb_dim = node_dim - 2 - len(self.addition_node_features)
        self.nodetype_embedder = nn.Embedding(num_node_types, node_emb_dim)
        # Reserve 2 dimensions for task prompt (fixed_edge, fixed_dist)
        self.edgetype_embedder = nn.Embedding(num_edge_types, edge_dim-2)
        
        # Denoiser network: remove noise from molecule representations
        denoiser_name = getattr(config.denoiser, 'name', 'default')
        if denoiser_name == 'default':
            denoiser_bb = ContextNodeEdgeNet if not gvp else ContextNodeEdgeNetGVP
        elif denoiser_name == 'ipa':
            denoiser_bb = ContextGAEdgeNet
        self.denoiser = denoiser_bb(node_dim, edge_dim,
                            context_dim=pocket_dim, **config.denoiser)

        # Output decoders
        self.node_decoder = MLP(node_dim, num_node_types, node_dim)
        self.edge_decoder = MLP(edge_dim, num_edge_types, edge_dim)
        
        # Additional outputs (e.g., confidence scores)
        self.add_output = getattr(config, 'add_output', [])
        if 'confidence' in self.add_output:
            self.node_cfd = MLP(node_dim, 1, node_dim//2)
            self.pos_cfd = MLP(node_dim, 1, node_dim//2)
            self.edge_cfd = MLP(edge_dim, 1, edge_dim//2)
            

    def forward(self, batch, **kwargs):
        """
        Forward pass: predicts clean molecule structure from noisy inputs.
        
        Given a noisy molecular graph at diffusion timestep t and a protein pocket,
        predicts the clean molecular structure at t=0 (atom types, 3D positions, bonds).
        
        Args:
            batch: Dictionary containing:
                - pos_in: Noisy 3D coordinates [N_atoms, 3]
                - node_in: Noisy node (atom) types [N_atoms]
                - halfedge_in: Noisy edge (bond) types. [N_edges]. Half means i<j only for all e_ij.
                - halfedge_index: Edge connectivity [2, N_edges]
                - fixed_node, fixed_pos: Binary indicators for fixed node_type/node_pos [N_atoms]
                - fixed_halfedge, fixed_halfdist: Binary indicators for fixed edge_type/edge_dist [N_edges]
                - pocket_atom_feature: Pocket atom features [N_pocket_atoms, D_pocket]
                - pocket_pos: Pocket coordinates [N_pocket_atoms, 3]
                - pocket_knn_edge_index: Pocket connectivity [2, N_pocket_edges]
                - is_peptide: composing amino acids or not [N_atoms]
                - node_type_batch, pocket_pos_batch: Batch indices for graph pooling
                
        Returns:
            Dictionary containing:
                - pred_node: Predicted node (atom) type logits [N_atoms, num_node_types]
                - pred_pos: Predicted 3D coordinates [N_atoms, 3]
                - pred_halfedge: Predicted edge (bond) type logits [N_edges, num_edge_types]
                - confidence_* (optional): Self-confidence scores for predictions
        """

        # Step 1: Prepare embeddings from noisy inputs
        pos_in = batch['pos_in']
        h_node_in = self.nodetype_embedder(batch['node_in'])
        h_halfedge_in = self.edgetype_embedder(batch['halfedge_in'])
        
        # Concatenate fixed indicators (task prompt) as additional features
        # These tell the model which nodes/edges should remain unchanged
        node_extra = torch.stack([batch['fixed_node'], batch['fixed_pos']], dim=1).to(pos_in.dtype)
        halfedge_extra = torch.stack([batch['fixed_halfedge'], batch['fixed_halfdist']], dim=1).to(pos_in.dtype)
        h_node_in = torch.cat([h_node_in, node_extra], dim=-1)
        h_halfedge_in = torch.cat([h_halfedge_in, halfedge_extra], dim=-1)

        # Convert half-edges to full bidirectional edges for message passing
        n_halfedges = h_halfedge_in.shape[0]
        halfedge_index = batch['halfedge_index']
        edge_index = torch.cat([halfedge_index, halfedge_index.flip(0)], dim=1)
        h_edge_in = torch.cat([h_halfedge_in, h_halfedge_in], dim=0)
        edge_extra = torch.cat([halfedge_extra, halfedge_extra], dim=0)
        
        # Add additional node features (e.g., peptide indicator)
        if 'is_peptide' in self.addition_node_features:
            is_peptide = batch['is_peptide'].unsqueeze(-1).to(pos_in.dtype)
            h_node_in = torch.cat([h_node_in, is_peptide], dim=-1)
        
        # Step 2: Encode protein pocket as context
        h_pocket = self.pocket_embedder(batch['pocket_atom_feature'])
        h_pocket = self.pocket_encoder(
            h_node=h_pocket,
            pos_node=batch['pocket_pos'],
            edge_index=batch['pocket_knn_edge_index'],
            h_edge=None,
            node_extra=None,
            edge_extra=None,
        )

        # Step 3: Denoise molecule conditioned on pocket
        h_node, pos_node, h_edge = self.denoiser(
            h_node=h_node_in,
            pos_node=pos_in, 
            h_edge=h_edge_in, 
            edge_index=edge_index,
            node_extra=node_extra,
            edge_extra=edge_extra,
            batch_node=batch['node_type_batch'],
            # Pocket context
            h_ctx=h_pocket,
            pos_ctx=batch['pocket_pos'],
            batch_ctx=batch['pocket_pos_batch'],
        )
        
        # Step 4: Decode predictions
        pred_node = self.node_decoder(h_node)
        # Average bidirectional edge features before decoding
        pred_halfedge = self.edge_decoder(h_edge[:n_halfedges] + h_edge[n_halfedges:])
        pred_pos = pos_node
        
        # Optional: predict self-confidence scores
        additional_outputs = {}
        if 'confidence' in self.add_output:
            pred_node_cfd = self.node_cfd(h_node)
            pred_pos_cfd = self.pos_cfd(h_node)
            pred_edge_cfd = self.edge_cfd(h_edge[:n_halfedges] + h_edge[n_halfedges:])
            additional_outputs = {
                'confidence_node': pred_node_cfd, 
                'confidence_pos': pred_pos_cfd, 
                'confidence_halfedge': pred_edge_cfd
            }
        
        return {
            'pred_node': pred_node,
            'pred_pos': pred_pos,
            'pred_halfedge': pred_halfedge,
            **additional_outputs,
        }

