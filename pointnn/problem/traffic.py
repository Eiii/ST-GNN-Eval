from .base import Problem
from ..data.traffic import METRDataset, collate

import torch.nn.functional as F


class TrafficMETR(Problem):
    def __init__(self, base, normalize):
        all_fn = lambda date: True
        valid_fn = lambda date: date.week % 10 == 0 and all_fn(date)
        train_fn = lambda date: not valid_fn(date) and all_fn(date)
        self.train_dataset = METRDataset(base, train_fn, normalize)
        self.valid_dataset = METRDataset(base, valid_fn, normalize)
        print(len(self.train_dataset), len(self.valid_dataset))
        self.collate_fn = collate

    def loss(self, item, pred):
        return element_loss(item, pred).mean()

def element_loss(item, pred, norm_info=None):
    tgt = item['tgt_data']
    mask = item['tgt_mask']
    pred = pred.squeeze(-1)
    if norm_info:
        mean, std = norm_info
        pred = pred*std + mean
        tgt = tgt*std + mean
    loss_mat = F.mse_loss(pred, tgt, reduction='none')
    return loss_mat[mask]
