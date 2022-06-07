import torch
import pickle

from ...data.starcraft import collate
from ...problem.starcraft import sc2_frame_loss, get_pred_ts, get_alive

from torch.utils.data import DataLoader
from pathlib import Path

def seq_to_device(d, device):
    if isinstance(d, dict):
        return {k: v.to(device) for k, v in d.items()}
    elif isinstance(d, list):
        return [v.to(device) for v in d]

def pred_safe(model, ds, bs, device='cpu'):
    while bs != 1:
        print(f'batch size={bs}')
        try:
            return pred(model, ds, bs, device)
        except RuntimeError as e:
            print(e)
            bs = bs//2
    raise RuntimeError()

def pred(model, ds, bs, device='cpu'):
    losses = []
    pred_ts = []
    alive = []
    loader = DataLoader(ds, batch_size=bs, collate_fn=collate, num_workers=8)
    for item in loader:
        item = seq_to_device(item, device)
        args = model.get_args(item)
        pred = model(*args)
        ls = sc2_frame_loss(item, pred, reduction='none')
        ls = ls.detach()
        flat_pred_ts = get_pred_ts(item)
        flat_alive = get_alive(item)
        losses.append(ls)
        alive.append(flat_alive)
        pred_ts.append(flat_pred_ts)
    result = {'losses': torch.cat(losses).cpu(),
              'ts': torch.cat(pred_ts).cpu()
             }
    return result

def run_net(net, ds, max_bs):
    with torch.no_grad():
        net = net.eval().cuda()
        return pred_safe(net, ds, max_bs, device='cuda')

def param_count(path):
    nt = net_type(path)
    lut = {'Small': 100, 'Med': 1000, 'Large': 5000}
    for k, v in lut.items():
        if k in nt:
            return v
    return -1


def arch_type(path):
    with Path(path).open('rb') as fd:
        return pickle.load(fd)['net_type']


def net_type(path):
    return Path(path).stem.split(':')[0]
