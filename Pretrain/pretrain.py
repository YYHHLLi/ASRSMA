import sys
from datetime import datetime
import logging
import os
import pandas as pd
import torch.distributed as dist
import warnings
import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn

from tqdm import tqdm
from torch.utils.data import DataLoader
from utils import set_global_seed, set_logger, Dock_loss
from data.graph_dataset import DockingDataset, DockingValDataset
from data.collator import collator_3d
from option import set_args
from models.DockingPoseModel import DockingPoseModel
from torch.utils.data.distributed import DistributedSampler
from itertools import zip_longest

warnings.filterwarnings("ignore")
torch.multiprocessing.set_sharing_strategy('file_system')


def random_drop_atoms(node_feats, drop_prob=0.10):
    if drop_prob <= 0:
        return node_feats
    mask = (torch.rand(node_feats.size()[:2], device=node_feats.device) > drop_prob).float()
    return node_feats * mask.unsqueeze(-1)


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= dist.get_world_size()
    return rt


class NTXentLoss(nn.Module):
  def __init__(self, temperature=0.1):
    super().__init__()
    self.tau = temperature

  def _global_pool(self, z):
    if z.dim() == 4:  # (B,C,H,W)
      z = F.adaptive_avg_pool2d(z, 1).flatten(1)
    elif z.dim() == 3:  # (B,L,D)
      z = z.mean(dim=1)
    elif z.dim() == 2:  # (B,D)
      pass
    else:
      raise ValueError(f"Unsupported z shape {z.shape}")
    return z


  @staticmethod
  def _match_dims(a: torch.Tensor, b: torch.Tensor):
    if a.size(1) == b.size(1):
      return a, b
    if a.size(1) < b.size(1):
      pad = b.size(1) - a.size(1)
      a = F.pad(a, (0, pad))
    else:
      pad = a.size(1) - b.size(1)
      b = F.pad(b, (0, pad))
    return a, b


  def forward(self, z1, z2):
    z1 = self._global_pool(z1)
    z2 = self._global_pool(z2)
    z1, z2 = self._match_dims(z1, z2)  # NEW

    z = torch.cat([z1, z2], dim=0)  # (2B, D)
    z = F.normalize(z, dim=1)

    sim = torch.matmul(z, z.T) / self.tau
    sim.fill_diagonal_(float('-inf'))

    B = z1.size(0)
    idx = torch.arange(B, device=z.device)
    pos = torch.cat([idx + B, idx])

    loss = - (sim[torch.arange(2 * B, device=z.device), pos]
              - torch.logsumexp(sim, dim=1)).mean()
    return loss


def run(args, device):
    set_global_seed(args.seed)
    best_loss = None
    code_root = os.path.dirname(os.path.realpath(__file__))
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    save_folder_name = 'run_' + timestamp
    save_path = os.path.join(code_root, save_folder_name)
    model_path = os.path.join(save_path, 'models')
    code_path = os.path.join(save_path, 'code')

    if dist.get_rank() == 0:
        os.makedirs(save_path)
        os.makedirs(model_path)
        os.makedirs(code_path)
        os.system(f'cp -r models data docking {code_path} && cp *.py {code_path}')
        set_logger(save_path, name=args.name + f'_{timestamp}')
        logging.info('File saved in ----> {}'.format(save_path))
        logging.info(args)


    model = DockingPoseModel(args).cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    optimizer = torch.optim.Adam(model.parameters(), args.lr, betas=(0.9, 0.99), eps=1e-06,
                                 weight_decay=args.weight_decay)

    train_dataset_1 = DockingDataset(os.path.join(args.data_dir, 'pretrain1'),
                                     os.path.join(args.data_dir, 'pretrain1/train.txt'), args.poc_max_len) # Data after a single rotation
    train_dataset_2 = DockingDataset(os.path.join(args.data_dir, 'pretrain2'),
                                     os.path.join(args.data_dir, 'pretrain2/train.txt'), args.poc_max_len)

    if dist.get_rank() == 0:
        logging.info(
            "number of parameters in model: {}".format(sum([param.nelement() for param in model.parameters()])))
        logging.info(
            f'train samples from augmentation1: {len(train_dataset_1)}, train samples from augmentation2: {len(train_dataset_2)}')


    train_dataloader_1 = DataLoader(train_dataset_1, batch_size=args.batch_size, num_workers=args.num_workers,
                                    collate_fn=collator_3d)
    train_dataloader_2 = DataLoader(train_dataset_2, batch_size=args.batch_size, num_workers=args.num_workers,
                                    collate_fn=collator_3d)
    train_dataloader_2_shuffle = DataLoader(train_dataset_2, batch_size=args.batch_size,shuffle=True, num_workers=args.num_workers,
                                    collate_fn=collator_3d)

    model.train()
    scaler = torch.cuda.amp.GradScaler()
    ntxent_loss = NTXentLoss(temperature=0.1)

    for epoch in range(args.epochs):

        total_loss = 0

        for idx, (batch_data_1, batch_data_2) in tqdm(enumerate(zip(train_dataloader_1, train_dataloader_2)),
                                                       total=len(train_dataloader_1)):
            for dicts_1, dicts_2 in zip(batch_data_1[:2], batch_data_2[:2]):
                for key in dicts_1.keys():
                    dicts_1[key] = dicts_1[key].to(device)
                for key in dicts_2.keys():
                    dicts_2[key] = dicts_2[key].to(device)
            with torch.cuda.amp.autocast():
                z1, z2 = model(batch_data_1, batch_data_2)
                loss = ntxent_loss(z1, z2) / args.accumulation_steps


            scaler.scale(loss).backward()

            if (idx + 1) % args.accumulation_steps == 0 or (idx + 1) == len(train_dataloader_1):
                dist.barrier()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_norm, norm_type=2)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.detach()

        for idx, (batch_data_1, batch_data_2) in tqdm(enumerate(zip(train_dataloader_1, train_dataloader_2_shuffle)),
                                                       total=len(train_dataloader_1)):

            for dicts_1, dicts_2 in zip(batch_data_1[:2], batch_data_2[:2]):
                for key in dicts_1.keys():
                    dicts_1[key] = dicts_1[key].to(device)
                for key in dicts_2.keys():
                    dicts_2[key] = dicts_2[key].to(device)

            with torch.cuda.amp.autocast():
                z1, z2 = model(batch_data_1, batch_data_2)
                loss = ntxent_loss(z1, z2) / args.accumulation_steps
            scaler.scale(loss).backward()

            if (idx + 1) % args.accumulation_steps == 0 or (idx + 1) == len(train_dataloader_1):
                dist.barrier()  # Ensure synchronization across processes
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_norm, norm_type=2)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.detach()


        total_loss /= len(train_dataloader_1)
        lr = optimizer.state_dict()['param_groups'][0]['lr']
        reduced_loss = reduce_tensor(total_loss.data)

        if dist.get_rank() == 0:
            logging.info(f"epoch {epoch + 1:<4d}, train, loss: {reduced_loss:6.3f}, lr: {lr:10.9f}")

            # 第一次直接保存，之后只保存更优模型
            if best_loss is None or reduced_loss < best_loss:
                best_loss = reduced_loss
                checkpoints = model.state_dict()
                torch.save(checkpoints, os.path.join(model_path, f'epoch.pt'))
                logging.info(f"Saved best model at epoch {epoch + 1} with loss {best_loss:.4f}")



def evaluate(args, model, dataloader):
    with torch.no_grad():
        model.eval()
        total_cross_loss = 0
        total_mol_loss = 0
        total_cross_len = 0
        total_mol_len = 0
        total_affi_loss = 0

        for idx, batch_data in tqdm(enumerate(dataloader), total=len(dataloader)):

            for dicts in batch_data[:2]:
                for key in dicts.keys():
                    dicts[key] = dicts[key].to(device)

            with torch.cuda.amp.autocast():
                pred = model(batch_data)
                cross_loss, mol_loss, distance_mask, holo_distance_mask, affi_loss = Dock_loss(batch_data, pred,
                                                                                               args.dist_threshold)

            total_cross_loss += cross_loss * distance_mask.sum()
            total_mol_loss += mol_loss * holo_distance_mask.sum()
            total_cross_len += distance_mask.sum()
            total_mol_len += holo_distance_mask.sum()
            total_affi_loss += affi_loss

        total_cross_loss = total_cross_loss / total_cross_len
        total_mol_loss = total_mol_loss / total_mol_len
        total_affi_loss = total_affi_loss / (idx + 1)
        total_loss = total_cross_loss + total_mol_loss + total_affi_loss * args.affi_weight

    return total_loss, total_cross_loss, total_mol_loss, total_affi_loss


if __name__ == '__main__':
    parser = set_args()
    args = parser.parse_args()

    dist.init_process_group(backend="nccl")
    local_rank = dist.get_rank()
    # local_rank = 0
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    run(args, device)
