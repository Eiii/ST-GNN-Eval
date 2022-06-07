from ...data.traffic import METRDataset, collate
from ...nets.traffic.baseline import Nearest
from .. import common
import matplotlib
import matplotlib.pyplot as plt

from pathlib import Path
import argparse
import torch


def to_cuda(d):
    return {k: v.cuda() if v is not None else None for k, v in d.items()}


def make_parser():
    def_data = './data/traffic/METR-LA/'
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default=Path(def_data))
    parser.add_argument('--out', type=Path, default=Path('day'))
    parser.add_argument('--net', nargs='+')
    parser.add_argument('--month', type=int)
    parser.add_argument('--day', type=int)
    parser.add_argument('--sensors', nargs='+', type=int)
    parser.add_argument('--xstart', type=int)
    parser.add_argument('--xend', type=int)
    parser.add_argument('--ftype', default='png')
    return parser


def plot_traffic(ax, ds, sensor, norm_info):
    first_x = None
    mean, std = norm_info
    xs = []
    ys = []
    for entry in ds.entries:
        ref = entry.target_rows[0]
        first_x = first_x or ref
        col = ds.raw_df.loc[ref]
        if type(sensor) == int:
            sensor = str(sensor)
        val = col[str(sensor)]
        val = val*std + mean
        xs.append(ref-first_x)
        ys.append(val)
    ax.plot(xs, ys, label='actual', c='k')


def plot_preds(ax, ds, preds, sensor_id, norm_info, name='pred'):
    mean, std = norm_info
    xs = range(len(preds))
    ys = []
    for item, p in zip(ds, preds):
        x = collate([item])
        mask = x['tgt_id'] == sensor_id
        val = p[mask].item()
        val = val*std + mean
        ys.append(val)
    ax.plot(xs, ys, label=name)
    ax.set_ylim([0, 70])


@torch.no_grad()
def predict(net, ds):
    preds = []
    for item in ds:
        x = collate([item])
        x = to_cuda(x)
        args = net.get_args(x)
        pred = net(*args).cpu()
        preds.append(pred)
    return preds


def make_net(path):
    net = common.make_net(common.load_result(path))
    net = net.cuda().eval()
    return net


def get_net_type(net_path):
    start = net_path.rfind('/')+1
    end = net_path.index(':')
    return net_path[start:end]


def single(base, net_paths, month, day, out, sensors, xstart, xend, ftype):
    filter_fn = lambda date: date.week % 10 == 0 and date.month == month and date.day == day
    ds = METRDataset(base, filter_fn, True, False)
    assert len(ds) == 288, "Date isn't in the test dataset"
    norm_info = ds.norm_info
    all_ids = ds.raw_df.columns.tolist()
    models = [make_net(path) for path in net_paths]
    preds = [predict(model, ds) for model in models]
    names = [get_net_type(p) for p in net_paths]
    base_model = Nearest()
    mean_preds = predict(base_model, ds)
    for sensor_id in sensors if sensors else all_ids:
        fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
        print(sensor_id)
        plot_traffic(ax, ds, sensor_id, norm_info)
        sensor_idx = ds.sensor_id_to_idx[int(sensor_id)]
        for pred, name in zip(preds, names):
            plot_preds(ax, ds, pred, sensor_idx, norm_info, name)
        plot_preds(ax, ds, mean_preds, sensor_idx, norm_info, 'mean')
        ax.legend()
        fig.savefig(out/f'{sensor_id}.{ftype}')
        plt.close(fig)


if __name__ == '__main__':
    args = make_parser().parse_args()
    if not args.out.exists():
        args.out.mkdir()
    single(args.data, args.net, args.month, args.day, args.out, args.sensors, args.xstart, args.xend, args.ftype)
