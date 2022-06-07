import argparse

from . import common

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot_curves(measures, out_path, valid, smooth):
    size = (8, 5)
    fig, ax = plt.subplots(figsize=size, dpi=200)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_yscale('log')
    all_names = list({m.name for m in measures if m.name is not None})
    print(f'Plotting {all_names}...')
    for name in all_names:
        # Training loss
        tls = [m._training_loss for m in measures if m.name == name]
        xs = []
        ys = []
        ts = []
        for tl in tls:
            x = [v['epoch'] for v in tl]
            y = [v['loss'] for v in tl]
            t = tl[-1]['time']
            y_array = np.array([y])
            if np.isnan(y_array).any():
                print('NaN loss')
            xs.append(x)
            ys.append(y)
            ts.append(t)
        max_len = max(len(x) for x in xs)
        xs = np.stack([x for x in xs if len(x) == max_len])
        ys = np.stack([y for y in ys if len(y) == max_len])
        ts = [t for t in ts if len(x) == max_len]
        xs = xs.mean(0)
        ys = np.mean(ys, axis=0)
        if smooth > 1:
            vs = []
            for i in range(ys.size-smooth):
                vs.append(np.mean(ys[i:i+smooth]))
            ys = np.array(vs)
            xs = xs[:ys.size]
        main_plot = ax.plot(xs, ys, label=name)
        main_color = main_plot[-1].get_color()

        # Training loss
        if valid:
            vls = [m._valid_stats for m in measures if m.name == name]
            xs = []
            ys = []
            for vl in vls:
                x = [t['epoch'] for t in vl]
                y = [t['loss'] for t in vl]
                y_array = np.array([y])
                if np.isnan(y_array).any():
                    continue
                xs.append(x)
                ys.append(y)
            max_len = max(len(x) for x in xs)
            xs = np.stack([x for x in xs if len(x) == max_len])
            ys = np.stack([y for y in ys if len(y) == max_len])
            xs = xs.mean(0)
            ys = np.mean(ys, axis=0)
            ax.plot(xs, ys, linestyle='--', color=main_color, alpha=0.5)
    ax.legend()


def make_plots(folder, out_path, filter, filter_all, filter_ignore, valid, smooth):
    measures = [r['measure'] for r in common.load_any(folder)]
    if filter:
        filters = filter.split(',')
        measures = [m for m in measures
                    if any(f in m.name for f in filters)]
    if filter_all:
        filters = filter_all.split(',')
        measures = [m for m in measures
                    if all(f in m.name for f in filters)]
    if filter_ignore:
        filters = filter_ignore.split(',')
        measures = [m for m in measures
                    if not any(f in m.name for f in filters)]
    plot_curves(measures, out_path, valid, smooth)
    plt.tight_layout()
    plt.savefig(out_path)


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('folder')
    parser.add_argument('--output', default='out.png')
    parser.add_argument('--valid', action='store_true')
    parser.add_argument('--smooth', type=int, default=0)
    parser.add_argument('--filter-any', default=None)
    parser.add_argument('--filter-all', default=None)
    parser.add_argument('--filter-ignore', default=None)
    parser.add_argument('--training', action='store_true')
    parser.add_argument('--skip-first', action='store_true')
    return parser


if __name__ == '__main__':
    args = make_parser().parse_args()
    make_plots(args.folder, args.output, args.filter_any, args.filter_all, args.filter_ignore, args.valid, args.smooth)

