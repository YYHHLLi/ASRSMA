import torch
import torch.nn as nn
import torch.nn.functional as F

from .MolTransformer import MolTransformer
from .layers.triangle_attn import PairUpdate, Pair2mol
from .layers.utils import NonLinearHead, DistanceHead


class DockingPoseModel(nn.Module):


    def __init__(self, args):
        super().__init__()
        self.args = args


        self.mol_model = MolTransformer(args.mol_atom_type,
                                        args.mol_node_feats, args)
        self.poc_model = MolTransformer(args.poc_atom_type,
                                        args.poc_node_feats, args)


        self.inter_blocks = args.pair_blocks
        self.mlp_z = NonLinearHead(args.node_embed_dim, args.edge_embed_dim)
        self.inter_model = nn.ModuleList([
            PairUpdate(args.edge_embed_dim, args.pair_hidden_dim,
                       args.pair_attn_size, args.pair_attn_heads,
                       args.pair_drop_ratio)
            for _ in range(self.inter_blocks)
        ])
        self.pair2mol = nn.ModuleList([
            Pair2mol(args.edge_embed_dim, args.pair_hidden_dim,
                     args.pair_attn_size, args.pair_attn_heads,
                     args.pair_drop_ratio, False)
            for _ in range(self.inter_blocks)
        ])

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=args.node_embed_dim,
            num_heads=args.cross_attn_heads,
            dropout=0.1,
            batch_first=True
        )
        self.value_proj = nn.Linear(args.node_embed_dim, args.node_embed_dim)


        self.poc_atom_linear = nn.Linear(args.node_embed_dim,
                                         args.edge_embed_dim)
        self.mol_atom_linear = nn.Linear(args.node_embed_dim,
                                         args.edge_embed_dim)
        self.aff_linear = nn.Linear(args.edge_embed_dim * 3, 1)
        self.gate_aff = nn.Linear(args.edge_embed_dim * 3, 1)
        self.leaky = nn.LeakyReLU()
        self.bias = nn.Parameter(torch.ones(1))

        self.cross_distance_project = NonLinearHead(args.edge_embed_dim, 1)
        self.holo_distance_project = DistanceHead(args.edge_embed_dim)


    def encode(self, batch):

        _, mol_rep,  _, _ = self.mol_model(batch[0])
        _, poc_rep,  _, _ = self.poc_model(batch[1])

        attn_output, attn_weights = self.cross_attention(
            query=poc_rep,
            key=mol_rep,
            value=mol_rep,
            need_weights=True
        )
        A = attn_weights  
        V_proj = self.value_proj(mol_rep)
        z = A.unsqueeze(-1) * V_proj.unsqueeze(1)

        return z                        #


    def forward(self, batch_data1, batch_data2=None):

        z1 = self.encode(batch_data1)
        if batch_data2 is None:
            return z1

        z2 = self.encode(batch_data2)
        return z1, z2
