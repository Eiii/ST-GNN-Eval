from ...data.starcraft import StarcraftDataset, collate
from .. import common
from . import plot
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

def plot_pred(item, net, num, title=None):
    pred = net_predict(item, net)
    num_preds = item['pred_ts'].size(1)
    rows = num_preds
    cols = 2
    fig, axs = plt.subplots(rows, cols, gridspec_kw={'width_ratios': [2, 1]})
    fig.set_size_inches(4*cols, 4*rows)
    fig.set_dpi(200)
    for pred_idx in range(num_preds):
        ax1, ax2 = axs[pred_idx, :]
        plot.setup_frame_ax(ax1)
        single_goal = item['pred_data'][0, pred_idx, :, :]
        pred_mask = item['pred_ids_mask'][0]
        single_pred = pick_pred(pred, pred_idx)
        print('x')
        plot.draw_scene(ax1, single_goal, pred_mask)
        plot.draw_scene_pred(ax1, single_goal, single_pred, pred_mask)
        plot.draw_deltas(ax1, single_goal, single_pred, pred_mask)
        #
        ax2.set_title('Health & Shields')
        plot.setup_val_ax(ax2)
        plot.plot_value_pred(ax2, single_goal, single_pred, item['pred_ids_mask'], 'health', 'red')
        plot.plot_value_pred(ax2, single_goal, single_pred, item['pred_ids_mask'], 'shields', 'blue')
    if title:
        fig.suptitle(title)
    fig.savefig(f'{num:03d}.png')

def plot_ep_frame(item, net, num, img):
    pred = net_predict(item, net)
    rows = 2
    cols = 2
    fig = plt.figure()
    gs = fig.add_gridspec(2, 1, height_ratios=[5, 1])
    gs1 = gs[0].subgridspec(1, 2, width_ratios=[2, 1])
    ax1, ax2 = [fig.add_subplot(g) for g in gs1]
    fig.set_size_inches(8, 6)
    fig.set_dpi(200)
    #
    plot.setup_frame_ax(ax1)
    ax1.set_title('Predicted Unit Positions')
    single_goal = item['preds'][0, 0, :, :]
    single_pred = pick_pred(pred, 0)
    plot.draw_scene(ax1, single_goal, item['unit_mask'])
    plot.draw_scene_pred(ax1, single_goal, single_pred, item['unit_mask'])
    plot.draw_deltas(ax1, single_goal, single_pred, item['unit_mask'])
    #
    ax2.set_title('Health & Shields')
    plot.setup_val_ax(ax2)
    plot.plot_value_pred(ax2, single_goal, single_pred, item['unit_mask'], 'health', 'red')
    plot.plot_value_pred(ax2, single_goal, single_pred, item['unit_mask'], 'shields', 'blue')
    #
    ax3 = fig.add_subplot(gs[1])
    ax3.set_title('Episode Timeline')
    plot.setup_timeline(ax3)
    plot.plot_timeline(ax3, img)
    #
    fig.tight_layout()
    fig.savefig(f'{num:03d}.png')

def main_single(net_path, data_path, ep):
    if net_path.is_dir():
        net_path = list(net_path.glob('*.pkl'))[0]
        print(net_path)
    net = common.make_net(common.load_result(net_path))
    ds = StarcraftDataset(data_path, max_files=2, max_hist=5, num_pred=4,
                          hist_dist='uniform', hist_dist_args={'max': 10},
                          pred_dist='fixed', pred_dist_args={'ts': [1, 2, 4, 7]})
    i = 50
    item = collate([ds[i]])
    plot_pred(item, net, i)

def make_timeline(entry, max_time):
    img = np.ones((1, max_time, 3), dtype=float)
    img[0, entry.hist_idxs, :] = [0, 0, 1]
    img[0, entry.pred_idxs, :] = [1, 0, 0]
    return img

def main_ep_anim(net_path, data_path, ep, frame):
    if net_path.is_dir():
        net_path = list(net_path.glob('*.pkl'))[0]
        print(net_path)
    net = common.make_net(common.load_result(net_path))
    ds = StarcraftDataset(data_path, max_files=1, max_hist=10, num_pred=1, dist_type='tail')
    ep_entries = [e for e in ds.all_entries if e.ep_idx==ep]
    orig = ep_entries[frame]
    ep_idx = orig.ep_idx
    num_frames = len(ds.episodes[ep_idx])
    for i in range(num_frames):
        # Make new entry - HACK
        ol = list(orig)
        dt = i-orig.pred_idxs[0]
        ol[3] = [dt]
        ol[4] = [i]
        new_entry = orig._make(ol)
        timeline_img = make_timeline(new_entry, num_frames)
        # Get frame
        fr = ds._get_entry(new_entry)
        fr = collate([fr])
        # Plot
        plot_ep_frame(fr, net, i, timeline_img)

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', type=Path)
    parser.add_argument('--data', type=Path)
    parser.add_argument('--episode', type=int, default=0)
    parser.add_argument('--frame', type=int, default=0)
    return parser

if __name__ == '__main__':
    args = make_parser().parse_args()
    main_single(args.net, args.data, args.episode)
    #main_ep_anim(args.net, args.data, args.episode, args.frame)
