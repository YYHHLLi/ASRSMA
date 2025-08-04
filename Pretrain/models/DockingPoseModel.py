import sys
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
        self.mol_model = MolTransformer(args.mol_atom_type, args.mol_node_feats, args)
        self.poc_model = MolTransformer(args.poc_atom_type, args.poc_node_feats, args)

        self.inter_blocks = args.pair_blocks
        self.mlp_z = NonLinearHead(args.node_embed_dim, args.edge_embed_dim)
        self.inter_model = nn.ModuleList([PairUpdate(args.edge_embed_dim,
                                                     args.pair_hidden_dim,
                                                     args.pair_attn_size,
                                                     args.pair_attn_heads,
                                                     args.pair_drop_ratio) for _ in range(self.inter_blocks)])

        self.pair2mol = nn.ModuleList([Pair2mol(args.edge_embed_dim,
                                                args.pair_hidden_dim,
                                                args.pair_attn_size,
                                                args.pair_attn_heads,
                                                args.pair_drop_ratio,
                                                False) for _ in range(self.inter_blocks)])

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=args.node_embed_dim,
            num_heads=args.cross_attn_heads,
            dropout=0.1,
            batch_first=True
        )
        self.value_proj = nn.Linear(args.node_embed_dim, args.node_embed_dim)
        self.poc_atom_linear = nn.Linear(args.node_embed_dim, args.edge_embed_dim)
        self.mol_atom_linear = nn.Linear(args.node_embed_dim, args.edge_embed_dim)
        self.aff_linear = nn.Linear(args.edge_embed_dim * 3, 1)
        self.gate_aff = nn.Linear(args.edge_embed_dim * 3, 1)
        self.leaky = nn.LeakyReLU()
        self.bias = nn.Parameter(torch.ones(1))
        self.cross_distance_project = NonLinearHead(args.edge_embed_dim, 1)
        self.holo_distance_project = DistanceHead(args.edge_embed_dim)

    def forward(self, batch_data1, batch_data2):
        _, mol_encoder_rep1, mol_encoder_pair_rep1, mol_padding_mask1 = self.mol_model(batch_data1[0], drop_prob=self.args.ligand_mask_prob)
        _, pocket_encoder_rep1, pocket_encoder_pair_rep1, poc_padding_mask1 = self.poc_model(batch_data1[1], drop_prob=self.args.pocket_mask_prob)
        attn_output1, attn_weights1 = self.cross_attention(
            query=pocket_encoder_rep1,
            key=mol_encoder_rep1,
            value=mol_encoder_rep1,
            need_weights=True
        )
        A1 = attn_weights1
        V_proj1 = self.value_proj(mol_encoder_rep1)
        z1 = A1.unsqueeze(-1) * V_proj1.unsqueeze(1)

        _, mol_encoder_rep2, mol_encoder_pair_rep2, mol_padding_mask2 = self.mol_model(batch_data2[0],drop_prob=self.args.ligand_mask_prob)
        _, pocket_encoder_rep2, pocket_encoder_pair_rep2, poc_padding_mask2 = self.poc_model(batch_data2[1],drop_prob=self.args.pocket_mask_prob)
        attn_output2, attn_weights2 = self.cross_attention(
            query=pocket_encoder_rep2,
            key=mol_encoder_rep2,
            value=mol_encoder_rep2,
            need_weights=True
        )
        A2 = attn_weights2
        V_proj2 = self.value_proj(mol_encoder_rep2)
        z2 = A2.unsqueeze(-1) * V_proj2.unsqueeze(1)
        return z1, z2