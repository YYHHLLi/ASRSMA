import torch
import logging
import os
import numpy as np
import random
import torch.nn.functional as F
from torch import nn


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2):

        z1 = F.normalize(z1, p=2, dim=-1)
        z2 = F.normalize(z2, p=2, dim=-1)
        similarity_matrix = torch.matmul(z1, z2.transpose(0, 1)) / self.temperature
        labels = torch.arange(z1.size(0)).to(z1.device)

        loss = F.cross_entropy(similarity_matrix, labels)
        return loss



def set_global_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def set_logger(save_path, name=None, print_on_screen=True):
    if name != None:
        log_file = os.path.join(save_path, name + '.log')
    else:
        log_file = os.path.join(save_path, 'train.log')
    print("logger =>" + log_file)
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.DEBUG,
        datefmt='%Y-%m-%d %H:%M:%S ',
        filename=log_file,
        filemode='a'
    )
    if print_on_screen:
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)


def Dock_loss(data, pred, dist_threshold=8.0):

    lig_dict, poc_dict, infos = data[0], data[1], data[2]
    poc_coord, poc_mask = poc_dict['pos'], poc_dict['x'][:, :, 0].ne(0)
    holo_coord, holo_mask = lig_dict['holo_pos'], lig_dict['x'][:, :, 0].ne(0)
    mask_2d = poc_mask.unsqueeze(1).int() * holo_mask.unsqueeze(2).int()
    distance_target = (poc_coord.unsqueeze(1) - holo_coord.unsqueeze(2)).norm(dim=-1)
    distance_target = distance_target * mask_2d

    holo_mask_2d = holo_mask.unsqueeze(1).int() * holo_mask.unsqueeze(2).int()
    holo_distance_target = (holo_coord.unsqueeze(1) - holo_coord.unsqueeze(2)).norm(dim=-1)
    holo_distance_target = holo_distance_target * holo_mask_2d

    cross_distance_predict, holo_distance_predict = pred[0], pred[1]
    distance_mask = distance_target != 0
    distance_mask &= (distance_target < dist_threshold)
    cross_distance_predict = cross_distance_predict.permute(0, 2, 1)
    distance_predict = cross_distance_predict[distance_mask]
    distance_target = distance_target[distance_mask]
    distance_target = distance_target.to(distance_predict.device)

    distance_loss = F.mse_loss(distance_predict.float(), distance_target.float(), reduction="mean")

    holo_distance_mask = holo_distance_target != 0
    holo_distance_predict_train = holo_distance_predict[holo_distance_mask]
    holo_distance_target = holo_distance_target[holo_distance_mask]
    holo_distance_target = holo_distance_target.to(holo_distance_predict_train.device)
    holo_distance_loss = F.smooth_l1_loss(
        holo_distance_predict_train.float(),
        holo_distance_target.float(),
        reduction="mean",
        beta=1.0,
    )

    pred_affi = pred[-1]
    affinity = infos['y'].to(pred_affi.device)
    affinity_loss = F.smooth_l1_loss(pred_affi.float(), affinity.float(), reduction="mean", beta=1.0)

    return distance_loss, holo_distance_loss, distance_mask, holo_distance_mask, affinity_loss