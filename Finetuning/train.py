import os, logging, warnings
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import torch.distributed as dist
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr
from datetime import datetime
from utils import set_global_seed, set_logger
from data.graph_dataset import DockingDataset
from data.collator import collator_3d
from option import set_args
from models.DockingPoseModel import DockingPoseModel

warnings.filterwarnings("ignore")
torch.multiprocessing.set_sharing_strategy("file_system")


# ---------------- 3-D CNN -----------------
class CNN3D_LBA(nn.Module):
    def __init__(self, in_channels, conv_drop_rate, fc_drop_rate, conv_filters,
                 conv_kernel_size, max_pool_positions, max_pool_sizes,
                 max_pool_strides, fc_units, batch_norm=True, dropout=False):
        super().__init__()
        layers = [nn.BatchNorm2d(in_channels)] if batch_norm else []
        for i, out_ch in enumerate(conv_filters):
            layers += [nn.Conv2d(in_channels, out_ch, conv_kernel_size,
                                 padding=conv_kernel_size // 2),
                       nn.ReLU()]
            if max_pool_positions[i]:
                layers.append(nn.MaxPool2d(max_pool_sizes[i], max_pool_strides[i]))
            if batch_norm: layers.append(nn.BatchNorm2d(out_ch))
            if dropout:    layers.append(nn.Dropout(conv_drop_rate))
            in_channels = out_ch
        layers += [nn.AdaptiveMaxPool2d((1, 1)), nn.Flatten()]
        for units in fc_units:
            layers += [nn.Linear(in_channels, units), nn.ReLU()]
            if batch_norm: layers.append(nn.BatchNorm1d(units))
            if dropout:    layers.append(nn.Dropout(fc_drop_rate))
            in_channels = units
        layers.append(nn.Linear(in_channels, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x.permute(0, 3, 1, 2)).view(-1)


# ---------------- helper -----------------
def move_batch_to_device(batch, device):
    for g in batch[:2]:
        for k in g:
            g[k] = g[k].to(device, non_blocking=True)
    return batch



# ---------------- main -------------------
def run(args, device):
    set_global_seed(args.seed)
    best_rmse = float("inf")
    best_epoch = None
    best_ckpt_path = None
    window_start = args.epochs - 49  # 例如 epochs=100 → 51；epochs=200 → 151
    # folders / logger
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    save_root = os.path.join(os.path.dirname(__file__), f"run_{timestamp}")
    model_dir = os.path.join(save_root, "models_ft")
    if dist.get_rank() == 0:
        os.makedirs(model_dir, exist_ok=True)
        set_logger(save_root, name=f"{args.name}_{timestamp}")
        logging.info("Files will be saved to %s", save_root)
        logging.info(args)

    # models
    num_conv = 4
    model_3d = CNN3D_LBA(
        in_channels=256,
        conv_drop_rate=0.1,
        fc_drop_rate=0.25,
        conv_filters=[32 * (2 ** n) for n in range(num_conv)],
        conv_kernel_size=3,
        max_pool_positions=[0, 1] * ((num_conv + 1) // 2),
        max_pool_sizes=[2] * num_conv,
        max_pool_strides=[2] * num_conv,
        fc_units=[512],
        batch_norm=False,
        dropout=True).to(device)

    model = DockingPoseModel(args).to(device)


    ckpt = "ASRSMA/Pretrain/model_pt/epoch.pt"

    if os.path.exists(ckpt):
        state = torch.load(ckpt, map_location="cpu")
        model.load_state_dict({**model.state_dict(),
                               **{k[7:]: v for k, v in state.items()}}, strict=False)
        if dist.get_rank() == 0:
            logging.info("Loaded pretrained %s", ckpt)

   # ckpt = "ASRSMA/Pretrain/model_pt/epoch"

    if os.path.exists(ckpt):
        state = torch.load(ckpt, map_location="cpu")


        cleaned_state = {k.split("module.", 1)[-1]: v for k, v in state.items()}


        load_info = model.load_state_dict(
            {**model.state_dict(), **cleaned_state}, strict=False
        )


        if dist.get_rank() == 0:
            logging.info(
                "Loaded pretrained %s | missing=%d | unexpected=%d",
                ckpt,
                len(load_info.missing_keys),
                len(load_info.unexpected_keys),
            )
            if load_info.missing_keys:
                logging.debug("Missing keys: %s", load_info.missing_keys[:20])
            if load_info.unexpected_keys:
                logging.debug("Unexpected keys: %s", load_info.unexpected_keys[:20])

    model = nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    optimizer = torch.optim.Adam(model_3d.parameters(), lr=args.lr,
                                 betas=(0.9, 0.99), eps=1e-6)

    # data
    args.data_dir = "data/Finetuning_data"


    def make_ds(name):
        return DockingDataset(os.path.join(args.data_dir, name),
                              None, args.poc_max_len,
                              normalize_labels=False)

    def make_ds(name):
        return DockingDataset(os.path.join(args.data_dir, name), None, args.poc_max_len,
                              normalize_labels=False)

    train_ds   = make_ds("train_data")
    val_ds     = make_ds("val_data")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, num_workers=args.num_workers,
                              collate_fn=collator_3d)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=args.num_workers,
                            collate_fn=collator_3d)
    test_path = os.path.join(args.data_dir, "test_data")
    test_ds = make_ds("test_data") if os.path.exists(test_path) else None
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                             num_workers=args.num_workers, collate_fn=collator_3d) if test_ds else None

    scaler = torch.cuda.amp.GradScaler()


    for epoch in range(args.epochs):
        # ---------- train ----------
        model.train();
        model_3d.train()
        t_preds, t_labels, t_loss = [], [], 0.0

        for idx, batch in enumerate(tqdm(train_loader, disable=(dist.get_rank() != 0))):
            batch = move_batch_to_device(batch, device)
            labels = torch.tensor(batch[2]["label"]).float().to(device)

            with torch.cuda.amp.autocast():
                pred = model_3d(model(batch))
                loss = F.mse_loss(pred, labels)


            scaler.scale(loss).backward()
            if ((idx + 1) % args.accumulation_steps == 0) or (idx + 1 == len(train_loader)):
                dist.barrier()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model_3d.parameters(), max_norm=args.max_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            t_loss += loss.detach()
            t_preds.append(pred.detach().cpu().numpy())
            t_labels.append(labels.cpu().numpy())


        t_preds = np.concatenate(t_preds)
        t_labels = np.concatenate(t_labels)

        train_metrics = dict(
            loss=(t_loss / len(train_loader)).item(),
            rmse=np.sqrt(mean_squared_error(t_labels, t_preds)),
            mae=mean_absolute_error(t_labels, t_preds),
            r2=r2_score(t_labels, t_preds),
            pcc=pearsonr(t_labels, t_preds)[0],
            spcc=spearmanr(t_labels, t_preds)[0])

        # ---------- val ----------
        model.eval(); model_3d.eval()
        v_preds, v_labels, v_loss = [], [], 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = move_batch_to_device(batch, device)
                labels = torch.tensor(batch[2]["label"]).float().to(device)
                pred = model_3d(model(batch))
                v_loss += F.mse_loss(pred, labels, reduction="mean").detach()
                # v_loss += composite_loss(pred, labels, epoch, reduction_steps=1).detach()
                v_preds.append(pred.cpu().numpy()); v_labels.append(labels.cpu().numpy())
        v_preds = np.concatenate(v_preds); v_labels = np.concatenate(v_labels)
        val_metrics = dict(
            loss=(v_loss / len(val_loader)).item(),
            rmse=np.sqrt(mean_squared_error(v_labels, v_preds)),
            mae=mean_absolute_error(v_labels, v_preds),
            r2=r2_score(v_labels, v_preds),
            pcc=pearsonr(v_labels, v_preds)[0],
            spcc=spearmanr(v_labels, v_preds)[0])

        # ---------- test ----------
        test_metrics = {k: np.nan for k in ["loss", "rmse", "mae", "r2", "pcc", "spcc"]}
        if test_loader:
            te_preds_l, te_labels_l, te_loss = [], [], 0.0
            with torch.no_grad():
                for batch in test_loader:
                    batch = move_batch_to_device(batch, device)
                    labels = torch.tensor(batch[2]["label"]).float().to(device)
                    pred = model_3d(model(batch))
                    te_loss += F.mse_loss(pred, labels, reduction="mean").detach()
                    te_preds_l.append(pred.cpu().numpy())
                    te_labels_l.append(labels.cpu().numpy())
            te_preds = np.concatenate(te_preds_l)
            te_labels = np.concatenate(te_labels_l)
            test_metrics = dict(
                loss=(te_loss / len(test_loader)).item(),
                rmse=np.sqrt(mean_squared_error(te_labels, te_preds)),
                mae=mean_absolute_error(te_labels, te_preds),
                pcc=pearsonr(te_labels, te_preds)[0],
                spcc=spearmanr(te_labels, te_preds)[0])

        # ---------- summary log & save ----------
        # ---------- summary log & save ----------
        if dist.get_rank() == 0:
            lr = optimizer.param_groups[0]["lr"]
            logging.info(
                f"Epoch {epoch + 1:03d} | "
                # f"train_loss {train_metrics['loss']:.4f} | val_loss {val_metrics['loss']:.4f} | "
                # f"RMSE t/v {train_metrics['rmse']:.4f}/{val_metrics['rmse']:.4f} | "
                # f"lr {lr:.3e}"
            )

            if (epoch + 1) >= window_start:
                current_rmse = val_metrics["rmse"]

                if current_rmse < best_rmse:  # 发现更优 RMSE
                    # 删除旧的最佳文件（两份）
                    if best_epoch is not None:
                        old_dock = os.path.join(model_dir, f"best_docking_ep{best_epoch}.pt")
                        old_cnn = os.path.join(model_dir, f"best_cnn3d_ep{best_epoch}.pt")
                        for f in (old_dock, old_cnn):
                            if os.path.exists(f):
                                os.remove(f)

                    # 更新记录
                    best_rmse = current_rmse
                    best_epoch = epoch + 1

                    # 分开保存
                    dock_path = os.path.join(model_dir, f"dpm_ep.pt")
                    cnn_path = os.path.join(model_dir, f"cnn3d_ep.pt")

                    torch.save(model.state_dict(), dock_path)
                    torch.save(model_3d.state_dict(), cnn_path)



# -------------- entry -------------------
if __name__ == "__main__":
    parser = set_args()
    args = parser.parse_args()
    dist.init_process_group(backend="nccl")
    local_rank = dist.get_rank()
    torch.cuda.set_device(local_rank)
    run(args, torch.device("cuda", local_rank))
