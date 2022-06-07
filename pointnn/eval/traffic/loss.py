from argparse import ArgumentParser

from ...problem.traffic import TrafficMETR, element_loss, collate
from ...nets.traffic.baseline import Random, Nearest, Mean
from .. import common

import pickle
import torch
from torch.utils.data import DataLoader


def to_cuda(d):
    return {k: v.cuda() if v is not None else None for k, v in d.items()}


def main(data_path, net_paths, baseline, bs, out_path):
    loss_dict = {}
    if baseline:
        models = [('Nearest', Nearest()), ('Mean', Mean())]
        for model_name, model in models:
            prob = TrafficMETR(data_path, normalize=True)
            ds = prob.valid_dataset
            result = eval_model(ds, model, bs)
            loss_dict[model_name] = result
    else:
        for net_path in net_paths:
            print(net_path)
            prob = TrafficMETR(data_path, normalize=True)
            ds = prob.valid_dataset
            net = make_net(net_path)
            result = eval_model(ds, net, bs)
            loss_dict[net_path] = result
    with open(out_path, 'wb') as fd:
        pickle.dump(loss_dict, fd)


def pred_safe(net, ds, bs):
    while bs != 1:
        print(f'batch size={bs}')
        try:
            return pred(net, ds, bs)
        except RuntimeError as e:
            print(e)
            bs = bs // 2
    raise RuntimeError()


@torch.no_grad()
def pred(net, ds, bs):
    losses = []
    mses = []
    loader = DataLoader(ds, batch_size=bs, collate_fn=collate)
    for item in loader:
        item = to_cuda(item)
        args = net.get_args(item)
        pred = net(*args)
        mse = element_loss(item, pred, ds.norm_info).cpu()
        loss = element_loss(item, pred).cpu()
        losses.append(loss)
        mses.append(mse)
    return torch.cat(losses), torch.cat(mses)


def eval_model(ds, net, bs):
    losses, mses = pred_safe(net, ds, bs)
    out = {'loss': losses, 'mse': mses}
    return out


def make_net(path):
    net = common.make_net(common.load_result(path))
    net = net.cuda().eval()
    return net


def make_baseline(type_):
    if type_ == 'random':
        return Random([0, 70])
    elif type_ == 'nearest':
        return Nearest()
    elif type_ == 'mean':
        return Mean()


def make_parser():
    parser = ArgumentParser()
    parser.add_argument('--data', default='data/traffic/METR-LA')
    parser.add_argument('--nets', nargs='+')
    parser.add_argument('--baseline', action='store_true')
    parser.add_argument('--out', default='traffic-loss.pkl')
    parser.add_argument('--bs', type=int, default=128)
    return parser


if __name__ == '__main__':
    args = make_parser().parse_args()
    main(args.data, args.nets, args.baseline, args.bs, args.out)
