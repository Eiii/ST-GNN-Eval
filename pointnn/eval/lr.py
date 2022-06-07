import argparse
import json
import numpy as np

from sys import argv
from math import ceil
from sklearn.metrics import f1_score

from . import common

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def get_loss(tl):
    return [t['loss'] for t in tl]


def get_lrs(tl):
    return [t['lr'] for t in tl]


def plot_lr_curves(measures, out_path, normalize):
    thresh_data = {}
    size = (8, 5)
    fig, ax = plt.subplots(figsize=size, dpi=200)
    ax.set_xlabel('Learning Rate')
    ax.set_ylabel('Loss Change')
    ax.set_xscale('log')
    all_names = list({m.name for m in measures if m.name is not None})
    print(f'Plotting {all_names}...')
    for name in all_names:
        print(f'{name}:')
        # LR plot
        tls = [m._training_loss for m in measures if m.name == name]
        xs = ys = None
        l_xs = [get_lrs(tl) for tl in tls]
        min_xl = min(len(x) for x in l_xs)
        max_xl = min(len(x) for x in l_xs)
        if max_xl - min_xl > 1:
            print('Misaligned training data')
        l_xs = [xs[:min_xl] for xs in l_xs]
        l_ys = [get_loss(tl)[:min_xl] for tl in tls]
        xs = np.vstack(l_xs)
        ys = np.vstack(l_ys)
        xs = xs[0]
        print(f'Steps: {len(xs)}')
        ys = np.mean(ys, axis=0)
        if normalize:
            plot_ys = ((ys / ys[0]) - 1)
            max_loss = 1
            plot_ys[plot_ys > max_loss] = max_loss
            plot_ys *= 100
        else:
            plot_ys = ys
        lr_thresh = calc_thresh(xs, ys)
        thresh_data[name] = lr_thresh
        ax.plot(xs, plot_ys, label=name)
    if normalize:
        ax.axhline(0, color='k')
    ax.legend()
    return thresh_data


def calc_thresh(xs, ys):
    fact = 4
    thresh = 1.2
    lr_floor = 1e-8
    min_f1 = 0.75
    seq_len = len(xs)
    early_idxs = ceil(seq_len/fact)
    early_losses = ys[:early_idxs]
    min_loss = early_losses.min()
    max_loss = early_losses.max()
    baseline_loss = early_losses[0]
    min_thresh = baseline_loss + (min_loss - baseline_loss)
    max_thresh = baseline_loss + (max_loss - baseline_loss)*thresh

    over_max = ys > max_thresh
    under_min = ys < min_thresh

    idx_scores = [(score_max_thresh(i, over_max), i) for i in range(seq_len)]
    _, max_thresh_idx = max(idx_scores)
    max_lr = xs[max_thresh_idx-1]

    idx_scores = [(score_min_thresh(i, max_thresh_idx, under_min), i) for i in range(seq_len)]
    min_thresh_score, min_thresh_idx = max(idx_scores)
    use_min = min_thresh_score > min_f1
    min_lr = xs[min_thresh_idx] if use_min else lr_floor
    print(f"{min_thresh_score:.2} {min_lr:.2e} {max_lr:.2e}")
    return min_lr, max_lr


def score_max_thresh(thresh_idx, goal):
    model = np.zeros_like(goal)
    model[thresh_idx:] = True
    score = f1_score(goal, model, zero_division=0)
    return score


def score_min_thresh(thresh_idx, max_thresh_idx, goal):
    goal = goal[:max_thresh_idx]
    model = np.zeros_like(goal)
    model[thresh_idx:] = True
    score = f1_score(goal, model, zero_division=0)
    return score


def make_plots(folder, out_path, lr_out, filter, normalize):
    measures = [r['measure'] for r in common.load_any(folder)]
    measures = [m for m in measures if not m.name.startswith('Pre')]
    if filter:
        filters = filter.split(',')
        measures = [m for m in measures if m.name in filters]
    thresh_data = plot_lr_curves(measures, out_path, normalize)
    with open(lr_out, 'w') as fd:
        json.dump(thresh_data, fd)
    plt.tight_layout()
    plt.savefig(out_path)


def set_lrs(lrs, files, dry):
    with open(lrs, 'r') as fd:
        lrs = json.load(fd)
    for exp_file in files:
        if 'lrtest' in exp_file:
            continue
        with open(exp_file, 'r') as fd:
            exp = json.load(fd)
        modified = False
        for entry in exp['entries']:
            name = entry['name']
            new_lrs = lrs.get(name, None)
            if new_lrs is None:
                print(f'No LRs for {name}')
                continue
            req_keys = ['lr', 'min_lr']
            train_args = entry['train_args']
            if all(k in train_args for k in req_keys):
                old_lrs = [train_args[k] for k in req_keys]
                print(f'Updating {name}: {old_lrs} {new_lrs}')
            else:
                print(f'Updating {name}: {new_lrs}')
            modified = True
            train_args['min_lr'], train_args['lr'] = new_lrs
        if modified:
            print(f'Updating {exp_file}')
            if not dry:
                with open(exp_file, 'w') as fd:
                    json.dump(exp, fd, indent=4)


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('folder')
    parser.add_argument('--output', default='out.png')
    parser.add_argument('--lroutput', default='lrs.json')
    parser.add_argument('--filter', default=None)
    parser.add_argument('--normalize', action='store_true')
    return parser


def make_apply_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lrs')
    parser.add_argument('--dryrun', action='store_true')
    parser.add_argument('experiments', nargs='+')
    return parser


if __name__ == '__main__':
    if argv[1] == 'apply':
        args = make_apply_parser().parse_args(argv[2:])
        set_lrs(args.lrs, args.experiments, args.dryrun)
    else:
        args = make_parser().parse_args()
        make_plots(args.folder, args.output, args.lroutput, args.filter, args.normalize)
