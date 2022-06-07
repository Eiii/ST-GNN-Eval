from ...data.starcraft import StarcraftDataset, collate, parse_frame
from .. import common
from ...nets.pointconv import PointConvAttn, closest_pts_to_keys, _gather_neighbor_info
from ...nets.sc2 import dists
from . import plot
from functools import partial
import argparse
import torch
import numpy as np

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def net_predict(scene, net):
    net = net.eval()
    args = net.get_args(scene)
    p = net(*args)
    return p

def pick_pred(p, idx):
    def sel(x):
        return x[:, [idx], :, :]
    return {k:sel(v) for k, v in p.items()}

def plot_attn(item, net, out, ftype):
    pred = net_predict(item, net)
    #
    frame_sel = item['ts'] == 0
    in_frame = item['data'][frame_sel]
    in_mask = item['mask'][frame_sel]
    #
    pf = parse_frame(in_frame)
    in_batch = in_frame.unsqueeze(0)
    ts = item['ts'][frame_sel]
    pos = pf['pos']
    pos_t = torch.cat((pos, ts.unsqueeze(-1)), dim=-1).unsqueeze(0)
    dist_fn = partial(dists.space, in_mask.unsqueeze(0), ts.unsqueeze(0))
    nei_idxs_tuple = closest_pts_to_keys(pos_t, pos_t, 8, dist_fn)
    nei_idxs = nei_idxs_tuple[0]
    _, nei_info, _ = _gather_neighbor_info(in_batch, *nei_idxs_tuple)
    first_conv = net.tpc.space_convs[0]
    attn = first_conv.calc_attn(in_batch, nei_info).squeeze(0)
    #attn = attn ** 0.5
    num_units = in_frame.size(0)
    for unit_num in range(num_units):
        if pf['alive'][unit_num] != 1:
            continue
        fig, ax1 = plt.subplots(1, 1)
        fig.set_size_inches(8, 8)
        fig.set_dpi(200)
        plot.setup_frame_ax(ax1)
        plot.draw_scene(ax1, in_frame, in_mask, colors=plot.altcolors, alpha=1)
        #
        unit_pos = pf['pos'][unit_num]
        others = nei_idxs[0, unit_num]
        other_poss = [pf['pos'][n] for n in others]
        circ = plt.Circle(unit_pos.tolist(), 0.05, color='r', fill=False)
        ax1.add_artist(circ)
        nei_attns = attn[unit_num]
        nei_attns = nei_attns ** 0.5
        for other_pos, other_attn in zip(other_poss, nei_attns):
            attn_color = other_attn.tolist()
            circ = plt.Circle(other_pos.tolist(), 0.05, color=attn_color)
            ax1.add_artist(circ)
        fig.savefig(out/f'{unit_num:02d}.{ftype}')
        plt.close(fig)

def main_single(net_path, data_path, frame, out, ftype):
    if net_path.is_dir():
        net_path = list(net_path.glob('*.pkl'))[0]
        print(net_path)
    net = common.make_net(common.load_result(net_path))
    ds = StarcraftDataset(data_path, max_files=1, max_hist=5, num_pred=4,
                          hist_dist='fixed', hist_dist_args={'ts': [0]},
                          pred_dist='fixed', pred_dist_args={'ts': [1,2,4,7]})
    item = collate([ds[frame]])
    plot_attn(item, net, out, ftype)

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', type=Path)
    parser.add_argument('--data', type=Path)
    parser.add_argument('--out', type=Path, default=Path('scattn'))
    parser.add_argument('--frame', type=int, default=50)
    parser.add_argument('--ftype', default='png')
    return parser

if __name__ == '__main__':
    args = make_parser().parse_args()
    if not args.out.exists():
        args.out.mkdir()
    main_single(args.net, args.data, args.frame, args.out, args.ftype)
