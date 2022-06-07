import pickle
import itertools
import argparse
import torch
from collections import defaultdict
from sys import argv


def main(weather_path, sc2_path, traffic_path, out_file):
    def _load(p):
        if p is None:
            return None
        with open(p, 'rb') as fd:
            return pickle.load(fd)
    weather = _load(weather_path)
    sc2 = _load(sc2_path)
    traffic = _load(traffic_path)
    weather_stats = group_results(weather)
    traffic_stats = group_results(traffic)
    sc2_stats = group_results(sc2)
    results = {'sc2': sc2_stats, 'weather': weather_stats, 'traffic': traffic_stats}
    with open(out_file, 'wb') as fd:
        pickle.dump(results, fd)


def table(data_path):
    with open(data_path, 'rb') as fd:
        data = pickle.load(fd)
    domain_order = ['sc2', 'weather', 'traffic']
    domain_scales = {'sc2': 10, 'weather': 100, 'traffic': 100}
    all_nets = [v.keys() for _, v in data.items()]
    all_nets = set(itertools.chain(*all_nets))

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
        for domain in domain_order:
            item = data[domain].get(net)
            s = domain_scales[domain]
            if item:
                qs = [s*q for q in item['quants']]
                for q in qs:
                    row.append(f'{q:.2f}')
                mean = s*item['mean']
                row.append(f'{mean:.2f}')
                fails = 100*item['fails']
                if fails > 0 and fails < 0.1:
                    row.append(f'<0.1')
                else:
                    row.append(f'{fails:.1f}')
            else:
                row += ['x']*5
        print(' & '.join(row) + r' \\')


def get_net_type(net_path):
    start = net_path.rfind('/')+1
    end = net_path.index(':')
    name = net_path[start:end]
    # HACK
    hack_rename = {'GC-Large': 'TGC-Large', 'GC-Med': 'TGC-Med', 'GC-Small': 'TGC-Small'}
    return hack_rename.get(name, name)


def calc_stats(losses):
    t = torch.cat(losses)
    t[t.isnan()] = float('inf')
    quants = [torch.quantile(t, x).item() for x in (0.25, 0.5, 0.75)]
    fails = (t > quants[-1]*100)
    fail_pct = (fails).sum().item() / len(t)
    mean = t[~fails].mean().item()
    return {'quants': quants, 'fails': fail_pct, 'mean': mean}


def group_results(result_dict):
    if not result_dict:
        return None
    all_losses = defaultdict(list)
    for net_path, result in result_dict.items():
        net_type = get_net_type(net_path)
        # SC2 hack...
        if 'loss' not in result and 'losses' in result:
            losses = result['losses'].sum(dim=-1)
        else:
            losses = result['loss']
        all_losses[net_type].append(losses)
    all_stats = {k: calc_stats(v) for k, v in all_losses.items()}
    return all_stats


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weather', default=None)
    parser.add_argument('--sc2', default=None)
    parser.add_argument('--traffic', default=None)
    parser.add_argument('--out', default='loss-stats.pkl')
    return parser


def make_table_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('file')
    return parser


if __name__ == '__main__':
    if len(argv) > 1 and argv[1] == 'table':
        args = make_table_parser().parse_args(argv[2:])
        table(args.file)
    else:
        args = make_parser().parse_args()
        main(args.weather, args.sc2, args.traffic, args.out)
