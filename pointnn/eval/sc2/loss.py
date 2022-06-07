from ...data.starcraft import StarcraftDataset
from .. import common
from .common import run_net, param_count, net_type
from pathlib import Path
from collections import defaultdict
from sys import argv
import matplotlib.pyplot as plt
import math
import argparse
import torch
import pickle


def make_dataset(data_path, ts):
    if ts is None:
        ts = [1, 2, 4, 7]
    ds = StarcraftDataset(data_path, num_pred=len(ts), max_hist=5, hist_dist='uniform',
                          hist_dist_args={'max': 10}, pred_dist='fixed',
                          pred_dist_args={'ts': ts}, frame_skip=1)
    return ds


def batch(net_paths, ds, max_bs, out_path):
    out_path = Path(out_path)
    if out_path.exists():
        with out_path.open('rb') as fd:
            loss_dict = pickle.load(fd)
    else:
        loss_dict = {}
    for net_path in net_paths:
        print(net_path)
        if net_path in loss_dict:
            print('Skipping...')
            continue
        net = common.make_net(common.load_result(net_path))
        pred_result = run_net(net, ds, max_bs)
        loss_dict[net_path] = pred_result
        print(pred_result['losses'].mean().item())
        with out_path.open('wb') as fd:
            pickle.dump(loss_dict, fd)


def group_result(paths):
    result = {}
    data = {}
    for p in paths:
        with Path(p).open('rb') as fd:
            d = pickle.load(fd)
            data.update(d)
    net_types = {net_type(p) for p in data}
    for n_type in net_types:
        net_data = {k: v for k, v in data.items() if net_type(k) == n_type}
        net_counts = {k: param_count(k) for k in net_data}
        for net_count in set(net_counts.values()):
            ns = {n for n, c in net_counts.items() if c == net_count}
            lcl_nets = {k: v for k, v in data.items() if k in ns}
            losses = [x['losses'] for x in lcl_nets.values()]
            ts = [x['ts'] for x in lcl_nets.values()]
            alive = [x['alive'] for x in lcl_nets.values()]
            all_loss = torch.cat(losses, dim=0)
            all_t = torch.cat(ts, dim=0)
            all_alive = torch.cat(alive, dim=0)
            result[(n_type, net_count)] = {'loss': all_loss, 't': all_t, 'alive': all_alive}
    return result


def plot(paths):
    fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
    result = group_result(paths)
    net_types = {k[0] for k in result}
    for net_type in net_types:
        counts = {k[1] for k in result if k[0] == net_type}
        plot_data = []
        for count in sorted(counts):
            d_key = (net_type, count)
            loss = result[d_key]['loss'].mean(dim=-1)
            ts = result[d_key]['t']
            alive = result[d_key]['alive']
            mean = loss.mean()
            mean = min(mean, 0.5)
            std = loss.std() / math.sqrt(len(loss))
            plot_data.append((count, mean, std))
            #
            logloss = loss.log()
            uniq_ts = ts.unique().tolist()
            hist_xs = []
            for t in uniq_ts:
                mask = (ts == t)
                hist_xs.append(logloss[mask].tolist())
            fig2, ax2 = plt.subplots(figsize=(8,5), dpi=200)
            ax2.hist(hist_xs, bins=250,
                     histtype='stepfilled', stacked=True,
                     label=[str(int(t)) for t in uniq_ts])
            ax2.legend()
            fig2.savefig(f'{net_type}-{count}-t.png')
            #
            uniq_alive = alive.unique().tolist()
            hist_as = []
            for a in uniq_alive:
                mask = (alive == a)
                hist_as.append(logloss[mask].tolist())
            fig3, ax3 = plt.subplots(figsize=(8,5), dpi=200)
            ax3.hist(hist_as, bins=250,
                     histtype='stepfilled', stacked=True,
                     label=[str(int(t)) for t in uniq_alive])
            ax3.legend()
            fig3.savefig(f'{net_type}-{count}-alive.png')
        xs, ys, errs = zip(*plot_data)
        ax.errorbar(xs, ys, label=net_type, fmt='x:')
    ax.set_yscale('log')
    fig.legend()
    fig.savefig('out.png')


def plot_old(paths, filter, out):
    plot_data = defaultdict(lambda: defaultdict(list))
    for path in paths:
        with open(path, 'rb') as fd:
            data = pickle.load(fd)
            for net_path, loss_info in data.items():
                net_name = net_path[net_path.rfind('/')+1:net_path.index(':')]
                if filter and filter not in net_name:
                    continue
                if 'loss' not in loss_info:
                    losses = loss_info['losses'].sum(dim=-1)
                else:
                    losses = loss_info['loss']
                ts = loss_info['ts']
                uniq_ts = ts.unique()
                for t in uniq_ts:
                    t_losses = losses[ts==t]
                    plot_data[net_name][t.item()].append(t_losses)
    for net_name in plot_data:
        for t in plot_data[net_name]:
            mean = torch.cat(plot_data[net_name][t]).mean().item()
            plot_data[net_name][t] = mean
    fig, ax = plt.subplots()
    width = 0.9/3
    for i, (name, t_data) in enumerate(plot_data.items()):
        ts = list(range(1, 1+12))
        means = [min(t_data[t],3) for t in ts]
        print(means)
        plot_ts = [t+i*width for t in ts]
        ax.bar(plot_ts, means, width, label=name)
    ax.legend()
    #ax.set_yscale('log')
    ax.set_xlabel('Time Delta')
    ax.set_ylabel('Avg. Loss')
    fig.savefig(out)


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='./data/sc2scene')
    parser.add_argument('--t', type=int, default=None)
    parser.add_argument('--out', default='sc-loss.pkl')
    parser.add_argument('--net', nargs='+')
    parser.add_argument('--bs', type=int, default=32)
    return parser


def make_plot_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('paths', nargs='+')
    parser.add_argument('--filter', default=None)
    parser.add_argument('--out', default='out.png')
    return parser


if __name__ == '__main__':
    if len(argv) > 1 and argv[1] == 'plot':
        args = make_plot_parser().parse_args(argv[2:])
        plot_old(args.paths, args.filter, args.out)
    else:
        args = make_parser().parse_args()
        if args.t is None:
            ds = make_dataset(args.data, None)
            batch(args.net, ds, args.bs, args.out)
        else:
            ds = make_dataset(args.data, [args.t])
            batch(args.net, ds, args.bs, args.out)
