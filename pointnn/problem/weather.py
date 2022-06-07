from .base import Problem
from ..data.weather import WeatherDataset, collate

import torch
import torch.nn.functional as F

DEF_TARGETS = ['RELH', 'TAIR', 'WSPD', 'PRES']


class Weather(Problem):
    def __init__(self, data_path,
                 targets=DEF_TARGETS,
                 time_off=5, hist_count=12, sample_count=1,
                 drop=None, seed=1337, HACK_SNIP=None):
        self.base_dataset = WeatherDataset(base=data_path,
                                           target_cols=targets,
                                           time_off=time_off,
                                           hist_count=hist_count,
                                           sample_count=sample_count,
                                           seed=seed,
                                           drop=drop,
                                           HACK_SNIP=HACK_SNIP)
        self.train_dataset = self.base_dataset.train_dataset
        self.valid_dataset = self.base_dataset.test_dataset
        self.collate_fn = collate

    def loss(self, item, pred):
        return scaled_loss(self.base_dataset, item['target'], pred).mean()


def scaled_loss(base_dataset, target, pred):
    ranges = [torch.tensor(x) for x in base_dataset.target_ranges()]
    expand = lambda t: t.view(1, 1, -1).to(pred.device)
    low, high = [expand(t) for t in ranges]
    diff = high-low
    rescale = lambda t: ((t-low)/diff)-0.5
    scaled_target = rescale(target)
    scaled_pred = rescale(pred)
    loss = F.mse_loss(scaled_target, scaled_pred, reduction='none')
    return loss


def flat_loss(base_dataset, target, pred):
    loss = F.mse_loss(target, pred, reduction='none')
    return loss
