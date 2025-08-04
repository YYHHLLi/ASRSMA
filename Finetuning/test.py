import os
import logging
import warnings
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.stats import spearmanr
from utils import set_global_seed, set_logger
from data.graph_dataset import DockingDataset
from data.collator import collator_3d
from option import set_args
from models.DockingPoseModel import DockingPoseModel

warnings.filterwarnings("ignore")
torch.multiprocessing.set_sharing_strategy("file_system")


class CNN3D_LBA(nn.Module):
    def __init__(self,
                 in_channels,
                 spatial_size,
                 conv_drop_rate,
                 fc_drop_rate,
                 conv_filters,
                 conv_kernel_size,
                 max_pool_positions,
                 max_pool_sizes,
                 max_pool_strides,
                 fc_units,
                 batch_norm=True,
                 dropout=False):
        super().__init__()
        layers = []
        if batch_norm:
            layers.append(nn.BatchNorm2d(in_channels))

        for i in range(len(conv_filters)):
            layers += [
                nn.Conv2d(in_channels,
                          conv_filters[i],
                          kernel_size=conv_kernel_size,
                          padding=conv_kernel_size // 2,
                          bias=True),
                nn.ReLU()
            ]
            if max_pool_positions[i]:
                layers.append(
                    nn.MaxPool2d(max_pool_sizes[i], max_pool_strides[i])
                )
            if batch_norm:
                layers.append(nn.BatchNorm2d(conv_filters[i]))
            if dropout:
                layers.append(nn.Dropout(conv_drop_rate))
            in_channels = conv_filters[i]

        layers += [
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten()
        ]

        in_features = in_channels
        for units in fc_units:
            layers += [
                nn.Linear(in_features, units),
                nn.ReLU()
            ]
            if batch_norm:
                layers.append(nn.BatchNorm1d(units))
            if dropout:
                layers.append(nn.Dropout(fc_drop_rate))
            in_features = units

        layers.append(nn.Linear(in_features, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, P, M, D) → (B, D, P, M)
        x = x.permute(0, 3, 1, 2)
        return self.model(x).view(-1)


# ------------------------------------------------------------
# helper
# ------------------------------------------------------------
def move_batch_to_device(batch, device):
    for g in batch[:2]:            # graph_pocket, graph_ligand
        for k in g:
            g[k] = g[k].to(device)
    return batch


# ------------------------------------------------------------
# main
# ------------------------------------------------------------
def run(args, device):
    set_global_seed(args.seed)

    # --------------- logger ---------------
    root   = os.path.dirname(os.path.realpath(__file__))
    ts     = datetime.now().strftime("%Y%m%d%H%M%S")
    outdir = os.path.join(root, f"test_{ts}")
    os.makedirs(outdir, exist_ok=True)
    set_logger(outdir, name="eval")
    logging.info("Results will be saved to %s", outdir)
    logging.info(args)

    # --------------- build networks ---------------
    num_conv          = 4
    conv_filters      = [32 * (2 ** n) for n in range(num_conv)]
    conv_kernel_size  = 3
    max_pool_positions = [0, 1] * ((num_conv + 1) // 2)
    max_pool_sizes     = [2] * num_conv
    max_pool_strides   = [2] * num_conv
    fc_units = [512]

    model_3d = CNN3D_LBA(
        in_channels=256,
        spatial_size=23,
        conv_drop_rate=0.1,
        fc_drop_rate=0.25,
        conv_filters=conv_filters,
        conv_kernel_size=conv_kernel_size,
        max_pool_positions=max_pool_positions,
        max_pool_sizes=max_pool_sizes,
        max_pool_strides=max_pool_strides,
        fc_units=fc_units,
        batch_norm=False,
        dropout=True).to(device)

    model = DockingPoseModel(args).to(device)


    finetune_root = "models/models_ft"

    dpm_ckpts = sorted(glob.glob(os.path.join(finetune_root, "dpm_ep.pt")))
    cnn_ckpts = sorted(glob.glob(os.path.join(finetune_root, "cnn3d_ep.pt")))

    assert len(dpm_ckpts) == 1, f"Expected exactly one dpm_ep*.pt file, but found {len(best_dpm_ckpts)}"
    assert len(cnn_ckpts) == 1, f"Expected exactly one cnn3d_ep*.pt file, but found {len(best_cnn_ckpts)}"

    dpm_ckpt = dpm_ckpts[0]
    cnn_ckpt = cnn_ckpts[0]

    # DockingPoseModel
    dpm_state = torch.load(dpm_ckpt, map_location="cpu")
    new_state = {}
    for k, v in dpm_state.items():
        new_state[k[7:]] = v        # 去掉多卡前缀
    model.load_state_dict({**model.state_dict(), **new_state}, strict=False)
    logging.info("Loaded DockingPoseModel weights from %s", dpm_ckpt)

    # CNN3D
    model_3d.load_state_dict(torch.load(cnn_ckpt, map_location="cpu"))
    logging.info("Loaded 3D-CNN weights from %s", cnn_ckpt)

    model.eval()
    model_3d.eval()

    # --------------- dataset ---------------
    args.data_dir = "data/Finetuning_data"


    def create_dataset(pkl_path):
        return DockingDataset(
            pkl_path,
            None,
            args.poc_max_len,
            normalize_labels=False,
        )

    test_dataset = create_dataset(
        os.path.join(args.data_dir, "test_data")
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collator_3d
    )

    # --------------- inference ---------------
    y_true, y_pred, losses = [], [], []

    with torch.no_grad():
        for batch in tqdm(test_loader, total=len(test_loader)):
            batch = move_batch_to_device(batch, device)
            labels = torch.tensor(batch[2]["label"]).float().to(device)

            z1 = model(batch)           # 单通路
            o1 = model_3d(z1)

            loss = F.mse_loss(o1, labels, reduction="mean")
            losses.append(loss.item())

            y_pred.extend(o1.cpu().tolist())
            y_true.extend(labels.cpu().tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    rmse     = np.sqrt(np.mean(np.square(y_pred - y_true)))
    mae      = np.mean(np.abs(y_pred - y_true))
    pearson  = np.corrcoef(y_true, y_pred)[0, 1]
    spearman = spearmanr(y_true, y_pred)[0]

    label_min = y_true.min()
    label_max = y_true.max()
    label_range = label_max - label_min

    print(f"y_true: {y_true} | y_pred: {y_pred} | ")

    print(f"Test RMSE: {rmse:.7f} | MAE: {mae:.7f} | "
          f"Pearson R: {pearson:.7f} | Spearman R: {spearman:.7f}")
    logging.info("Test RMSE %.7f, MAE %.7f, Pearson %.7f, Spearman %.7f",
                 rmse, mae, pearson, spearman)

# ------------------------------------------------------------
if __name__ == "__main__":
    parser = set_args()
    args   = parser.parse_args()

    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)

    run(args, device)
