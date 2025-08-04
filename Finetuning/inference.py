import pickle
import sys
from datetime import datetime
import logging
import os

import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader

from utils import set_global_seed, set_logger, Dock_loss
from data.graph_dataset import DockingDataset, DockingValDataset, getpkl
from data.collator import collator_3d
from option import set_args
from models.DockingPoseModel import DockingPoseModel

from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from itertools import zip_longest
import warnings

warnings.filterwarnings("ignore")
torch.multiprocessing.set_sharing_strategy('file_system')
import numpy as np
import torch.nn as nn

from lba.util.voxelize import dotdict, get_center, gen_rot_matrix, get_grid

class CNN3D_LBA(nn.Module):
    def __init__(self, in_channels, spatial_size,
                 conv_drop_rate, fc_drop_rate,
                 conv_filters, conv_kernel_size,
                 max_pool_positions, max_pool_sizes, max_pool_strides,
                 fc_units,
                 batch_norm=True,
                 dropout=False):
        super(CNN3D_LBA, self).__init__()
        # 1, 512, 275, 12
        layers = []
        if batch_norm:
            layers.append(nn.BatchNorm2d(in_channels))

        # Convs
        for i in range(len(conv_filters)):
            layers.extend([
                nn.Conv2d(in_channels, conv_filters[i],
                          kernel_size=conv_kernel_size,
                          padding=int(conv_kernel_size/2),
                          bias=True),
                nn.ReLU()
                ])
            spatial_size -= (conv_kernel_size - 1)
            if max_pool_positions[i]:
                layers.append(nn.MaxPool2d(max_pool_sizes[i], max_pool_strides[i]))
                spatial_size = int(np.floor((spatial_size - (max_pool_sizes[i]-1) - 1)/max_pool_strides[i] + 1))
            if batch_norm:
                layers.append(nn.BatchNorm3d(conv_filters[i]))
            if dropout:
                layers.append(nn.Dropout(conv_drop_rate))
            in_channels = conv_filters[i]

        layers.append(nn.AdaptiveMaxPool2d((1,1)))
        layers.append(nn.Flatten())
        in_features = in_channels
        # FC layers
        for units in fc_units:
            layers.extend([
                nn.Linear(in_features, units),
                nn.ReLU()
                ])
            if batch_norm:
                layers.append(nn.BatchNorm2d(units))
            if dropout:
                layers.append(nn.Dropout(fc_drop_rate))
            in_features = units

        # Final FC layer
        layers.append(nn.Linear(in_features, 1))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.permute(0, 3, 1,2)
        b, c, h, w = x.shape
        size = h if h > w else w
        x = F.interpolate(x, size=(size, size))
        x = self.model(x).view(-1)
        return  x


class CNN3D_TransformLBA(object):
    def __init__(self, random_seed=None, **kwargs):
        self.random_seed = random_seed
        self.grid_config =  dotdict({
            # Mapping from elements to position in channel dimension.
            'element_mapping': {
                #'H': 0,
                'C': 0,
                'O': 1,
                'N': 2,
                'P': 3,
                'F': 4,
                'Cl': 5,
                'Br': 6,
                'I': 7,
            },
            # Radius of the grids to generate, in angstroms.
            'radius': 20.0,
            # Resolution of each voxel, in angstroms.
            'resolution': 1.0,
            # Number of directions to apply for data augmentation.
            'num_directions': 20,
            # Number of rolls to apply for data augmentation.
            'num_rolls': 20,
        })
        # Update grid configs as necessary
        self.grid_config.update(kwargs)

    def _voxelize(self, atoms_pocket, atoms_ligand):
        # Use center of ligand as subgrid center
        elements = atoms_pocket['element'].values
        #print(elements.shape)
        ligand_pos = atoms_ligand[['x', 'y', 'z']].astype(np.float32)
        ligand_center = get_center(ligand_pos)
        # Generate random rotation matrix
        rot_mat = gen_rot_matrix(self.grid_config, random_seed=self.random_seed)
        # Transform protein/ligand into voxel grids and rotate
        grid = get_grid(pd.concat([atoms_pocket, atoms_ligand]),
                        ligand_center, config=self.grid_config, rot_mat=rot_mat)
        # Last dimension is atom channel, so we need to move it to the front
        # per pytroch style
        grid = np.moveaxis(grid, -1, 0)
        return grid

    def __call__(self, item):
        # Transform protein/ligand into voxel grids.
        # Apply random rotation matrix.
        transformed = {
            'feature': self._voxelize(item['atoms_pocket'], item['atoms_ligand']),
            'label': item['scores']['neglog_aff'],
            'id': item['id']
        }
        return transformed



def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= dist.get_world_size()
    return rt


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin  # margin用于控制负样本的最小距离

    def forward(self, z1, z2, labels=1):
        """
        z1, z2: 输入的两个张量，通常是经过数据增强（如旋转）后的RNA-分子复合物的特征表示
        labels: 标记对的标签，1表示相似对，0表示不相似对
        """
        # 计算z1和z2之间的欧氏距离
        z1 = z1.permute(0, 3, 1,2)
        z2 = z2.permute(0, 3, 1,2)
        z2 = F.interpolate(z2, size=(z1.size(2), z1.size(3)))
        euclidean_distance = F.pairwise_distance(z1, z2, p=2)

        # 正样本损失: 当标签为1时，希望欧氏距离尽量小
        positive_loss = labels * torch.pow(euclidean_distance, 2)

        # 负样本损失: 当标签为0时，希望欧氏距离大于margin
        negative_loss = (1 - labels) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)

        # 总损失是正样本损失和负样本损失的加权和
        loss = torch.mean(positive_loss + negative_loss)

        return loss


def run(args, device):
    set_global_seed(args.seed)
    code_root = os.path.dirname(os.path.realpath(__file__))
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    save_folder_name = 'run_' + timestamp
    save_path = os.path.join(code_root, save_folder_name)
    model_path = os.path.join(save_path, 'models_ft')
    os.makedirs(save_path,exist_ok=True)
    code_path = os.path.join(save_path, 'code')

    os.makedirs(save_path,exist_ok=True)
    os.makedirs(model_path,exist_ok=True)
    os.makedirs(code_path,exist_ok=True)
    os.system(f'cp -r models data docking {code_path} && cp *.py {code_path}')
    set_logger(save_path, name=args.name + f'_{timestamp}')
    logging.info('File saved in ----> {}'.format(save_path))
    logging.info(args)

    num_conv = 4
    conv_filters = [32 * (2 ** n) for n in range(num_conv)]
    conv_kernel_size = 3
    max_pool_positions = [0, 1] * int((num_conv + 1) / 2)
    max_pool_sizes = [2] * num_conv
    max_pool_strides = [2] * num_conv
    fc_units = [512]

    model_3d = CNN3D_LBA(
        512, 23,
        0.1,
        0.25,
        conv_filters, conv_kernel_size,
        max_pool_positions,
        max_pool_sizes, max_pool_strides,
        fc_units,
        batch_norm=False,
        dropout=True).cuda()
    model = DockingPoseModel(args).cuda()
    # model_pat = 'run_20250301003217/models_ft/epoch_28.pt'
    model_pat = "run_20250306120923/models_ft/epoch_30.pt"
    status = torch.load(model_pat)
    model_status = model.state_dict()
    new_state_dict = {}
    for key in status.keys():
        name = key[7:]
        new_state_dict[name] = status[key]
    model_status.update(new_state_dict)
    model.load_state_dict(model_status)
    model = model.to(device)
    print('load weight succeed!')

    m_root,m_name = os.path.dirname(model_pat), os.path.basename(model_pat)
    model_3d_pat = os.path.join(m_root,'model_3d_'+m_name)
    status = torch.load(model_3d_pat)
    model_status = model_3d.state_dict()
    new_state_dict = {}
    for key in status.keys():
        # name = key[7:]
        name = key
        new_state_dict[name] = status[key]
    model_status.update(new_state_dict)
    model_3d.load_state_dict(model_status)
    model_3d = model_3d.to(device)
    print('load weight succeed!')

    args.data_dir = 'data/Finetuning_data'
    data_path = 'data/Finetuning_data/test_augmentation1_pkl/4znp_ATOM.pkl'
    batch_data_1 = collator_3d([getpkl(data_path,args.poc_max_len)])

    with torch.no_grad():
        for dicts_1, dicts_2 in zip(batch_data_1[:2], batch_data_1[:2]):
            for key in dicts_1.keys():
                dicts_1[key] = dicts_1[key].to(device)
            for key in dicts_2.keys():
                dicts_2[key] = dicts_2[key].to(device)
        z1, z2 = model(batch_data_1, batch_data_1)
        o1 = model_3d(z1)
        o2 = model_3d(z2)
        s = (o1+o2)/2
        print(s.item())

if __name__ == '__main__':
    parser = set_args()
    args = parser.parse_args()

    local_rank = 0
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    run(args, device)
