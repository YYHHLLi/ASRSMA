import argparse


def set_args():
    parser = argparse.ArgumentParser(description='')

    # Hardware specifications
    parser.add_argument('--seed', type=int, default=42)

    # Data specifications
    parser.add_argument('--data_dir', type=str, default='')
    parser.add_argument('--poc_max_len', type=int, default=256)
    parser.add_argument('--dist_threshold', type=float, default=8.0)
    parser.add_argument('--conf_size', type=int, default=10)

    # Model specifications
    # embed
    parser.add_argument('--mol_atom_type', type=int, default=12)
    parser.add_argument('--mol_node_feats', type=int, default=36)
    parser.add_argument('--poc_atom_type', type=int, default=6)
    parser.add_argument('--poc_node_feats', type=int, default=52)
    parser.add_argument('--edge_feats', type=int, default=7)

    # attn
    parser.add_argument('--encoder_layers', type=int, default=3)
    # node attn
    parser.add_argument('--node_embed_dim', type=int, default=256)
    parser.add_argument('--node_emb_dropout', type=float, default=0.2)
    parser.add_argument('--node_ffn_dim', type=int, default=256)
    parser.add_argument('--node_attn_heads', type=int, default=4)
    parser.add_argument('--node_attn_dropout', type=float, default=0.1)
    parser.add_argument('--node_ffn_dropout', type=float, default=0.2)

    # edge attn
    parser.add_argument('--edge_embed_dim', type=int, default=64)
    parser.add_argument('--edge_attn_heads', type=int, default=2)
    parser.add_argument('--edge_attn_size', type=int, default=32)
    parser.add_argument('--edge_dropout', type=float, default=0.2)

    # inter attn
    parser.add_argument('--pair_hidden_dim', type=int, default=64)
    parser.add_argument('--pair_attn_size', type=int, default=32)
    parser.add_argument('--pair_attn_heads', type=int, default=4)
    parser.add_argument('--pair_drop_ratio', type=int, default=0.2)
    parser.add_argument('--pair_blocks', type=int, default=3)

    # Training specifications
    parser.add_argument('--name', type=str, default='ASRSMA')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--early_stop', type=int, default=20)
    parser.add_argument('--accumulation_steps', type=int, default=4)
    parser.add_argument('--max_norm', type=float, default=1.0)
    parser.add_argument('--lr', type=float, default=5e-3)
    parser.add_argument('--affi_weight', type=float, default=0.05)

    parser.add_argument("--cross_attn_heads", type=int, default=4)
    # parser.add_argument('--use_random_mask', action='store_true')
    # parser.add_argument('--ligand_mask_prob', type=float, default=0.1, help='Ligand atom drop ratio')
    # parser.add_argument('--pocket_mask_prob', type=float, default=0.1, help='Pocket atom drop ratio')
    #

    parser.add_argument('--weight_decay', type=float, default=1e-5, help='L2 regularization strength')

    # args = parser.parse_args()

    return parser