from .. import common
from pathlib import Path
import argparse
import torch
import matplotlib.pyplot as plt


def calc_weights_tpc(net, layer=0):
    steps = 150
    xs = torch.linspace(-1, 1, steps=steps)
    ys = torch.linspace(-1, 1, steps=steps)
    weight_grid = torch.cartesian_prod(xs, ys)
    wc = net.tpc.space_convs[layer].weight_conv
    weights = wc(weight_grid).detach()
    weights = weights.reshape(xs.size(0), ys.size(0), -1)
    return (xs, ys), weights


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--net')
    parser.add_argument('--out', type=Path, default=Path('.'))
    parser.add_argument('--ftype', default='png')
    parser.add_argument('--layer', type=int, default=0)
    return parser


def main(net_path, layer, out, ftype):
    net = common.make_net(common.load_result(net_path)).eval()
    pos, weights = calc_weights_tpc(net, layer)
    for i in range(16):
        fig, ax = plt.subplots()
        xs, ys = pos
        ax.pcolormesh(xs, ys, weights[:, :, i], shading='auto')
        fig.savefig(out/f'weight_{layer}_{i}.{ftype}')
        plt.close(fig)


if __name__ == '__main__':
    args = make_parser().parse_args()
    main(args.net, args.layer, args.out, args.ftype)
