import pickle
import argparse
import torch
import itertools

from collections import defaultdict


def get_net_type(net_path):
    start = net_path.rfind('/')+1
    end = net_path.index(':')
    name = net_path[start:end]
    return name


def calc_stats(losses):
    t = torch.cat(losses)
    t[t.isnan()] = float('inf')
    quants = [torch.quantile(t, x).item() for x in (0.25, 0.5, 0.75)]
    fails = (t > quants[-1]*100)
    mean = t[~fails].mean().item()
    return {'quants': quants, 'mean': mean}


def table(stat_dict):
    all_nets = [d.keys() for _, d in stat_dict.items()]
    all_nets = set(itertools.chain(*all_nets))

    all_files = set(stat_dict.keys())

    # Order
    def order_nets(n):
        vs = [('Drop2', 1000), ('TGC', 1), ('TInt', 2), ('TPC', 3)]
        return sum(m * (s in n) for s, m in vs)
    all_nets = sorted(all_nets, key=order_nets)

    def file_to_drop(n):
        s = n[n.rfind('-')+1:n.rfind('.')]
        return int(s)/10
    all_files = sorted(all_files, key=file_to_drop)

    # Print
    s = 100
    for net in all_nets:
        row = []
        row.append(net)
        for fpath in all_files:
            item = stat_dict[fpath].get(net)
            if item:
                med = s*item['quants'][1]
                row.append(f'{med:.2f}')
                mean = s*item['mean']
                row.append(f'{mean:.2f}')
            else:
                row += ['x']*4
        print(' & '.join(row) + r' \\')


def main(files):
    all_files = {}
    for f in files:
        all_losses = defaultdict(list)
        with open(f, 'rb') as fd:
            data = pickle.load(fd)
            for net_path, result in data.items():
                net_type = get_net_type(net_path)
                all_losses[net_type].append(result['loss'])
        all_stats = {k: calc_stats(v) for k, v in all_losses.items()}
        all_files[f] = all_stats
    table(all_files)


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('files', nargs='+')
    return parser


if __name__ == '__main__':
    args = make_parser().parse_args()
    main(args.files)
