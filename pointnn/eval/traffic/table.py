import pickle
import argparse
import torch
from collections import defaultdict


def main(files):
    data = {}
    for f in files:
        with open(f, 'rb') as fd:
            data.update(pickle.load(fd))
    stats = group_results(data)
    table(stats)


def table(data):
    all_nets = list(data.keys())

    # Order
    def order_nets(n):
        vs = [('Small', 10), ('Med', 100), ('Large', 1000),
              ('TGC', 1), ('TInt', 2), ('TPC', 3), ('TPCA', 4)]
        return sum(m * (s in n) for s, m in vs)
    all_nets = sorted(all_nets, key=order_nets)

    # Print
    for net in all_nets:
        row = []
        row.append(net)
        item = data.get(net)
        if item:
            qs = [q for q in item['quants']]
            for q in qs:
                row.append(f'{q:.2f}')
            mean = item['mean']
            row.append(f'{mean:.2f}')
        else:
            row += ['x']*4
        print(' & '.join(row) + r' \\')


def get_net_type(net_path):
    if ':' in net_path:
        start = net_path.rfind('/')+1
        end = net_path.index(':')
        name = net_path[start:end]
        return name
    else:
        return net_path


def calc_stats(losses):
    t = torch.cat(losses)
    quants = [torch.quantile(t, x).item() for x in (0.25, 0.5, 0.75)]
    t[t.isnan()] = float('inf')
    fails = t.isinf()
    mean = t[~fails].mean().item()
    return {'quants': quants, 'mean': mean}


def group_results(result_dict):
    if not result_dict:
        return None
    all_errs = defaultdict(list)
    for net_path, result in result_dict.items():
        net_type = get_net_type(net_path)
        err = result['mse']**0.5
        all_errs[net_type].append(err)
    all_stats = {k: calc_stats(v) for k, v in all_errs.items()}
    return all_stats


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('files', nargs='+')
    return parser


if __name__ == '__main__':
    args = make_parser().parse_args()
    main(args.files)
