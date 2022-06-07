from .. import common
from ...data.traffic import METRDataset

from pathlib import Path
import argparse
import torch
import matplotlib.pyplot as plt


def calc_weights_tpc(dist_data, net, layer=0):
    xs, ys, _, costs = zip(*dist_data)
    xs, ys, costs = [torch.tensor(x) for x in (xs, ys, costs)]
    in_ = torch.stack((xs, ys, costs), dim=-1)
    wc = net.tpc.space_convs[layer].weight_conv
    weights = wc(in_).detach()
    return weights


def get_sensor_ids(ds):
    return [int(x) for x in ds.meta_df.index.tolist()]


def get_sensor_pos(ds, center_idx):
    meta = ds.meta_df
    idxs = [x for x in get_sensor_ids(ds)]
    xs = meta.loc[idxs]['x'].tolist()
    ys = meta.loc[idxs]['y'].tolist()
    connected = ds.sensor_connected[center_idx].tolist()
    cost = ds.sensor_cost[center_idx].tolist()
    return list(zip(xs, ys, connected, cost))


def plot_sensor_weights(ax, dist_data, weights, idx):
    disc_data = [(a, b, c, d) for (a, b, c, d) in dist_data if not c]
    all_xs, all_ys, _, all_costs = zip(*disc_data)
    ax.scatter(all_xs, all_ys, c='grey', s=5)

    connected_data = [(a, b, w) for ((a, b, c, d), w) in zip(dist_data, weights) if c]
    all_xs, all_ys, all_weights = zip(*connected_data)
    ax.scatter(all_xs, all_ys, c=all_weights, s=25)

    center_x, center_y, _, _ = dist_data[idx]
    ax.scatter([center_x], [center_y], color='magenta', s=30)


def main(base, net_path, out):
    all_fn = lambda _: True
    ds = METRDataset(base, all_fn, True, False)
    net = common.make_net(common.load_result(net_path)).eval()
    for i in range(10):
        sensor_dist = get_sensor_pos(ds, i)
        for l in range(2):
            weights = calc_weights_tpc(sensor_dist, net, l)
            for w in range(8):
                plot_weights = weights[:, w]
                fix, ax = plt.subplots(figsize=(10, 10), dpi=100)
                plot_sensor_weights(ax, sensor_dist, plot_weights, i)
                plt.savefig(out/f'w_{i}_{l}_{w}.png')
                plt.close()


def make_parser():
    def_data = './data/traffic/METR-LA/'
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default=Path(def_data))
    parser.add_argument('--net')
    parser.add_argument('--out', type=Path, default=Path('tweight'))
    return parser


if __name__ == '__main__':
    args = make_parser().parse_args()
    if not args.out.exists():
        args.out.mkdir()
    main(args.data, args.net, args.out)
