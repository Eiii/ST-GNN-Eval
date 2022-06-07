from ...data.traffic import METRDataset
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.cm import ScalarMappable
from matplotlib import collections as coll
from matplotlib.animation import ArtistAnimation

from pathlib import Path
import argparse


def make_parser():
    def_data = './data/traffic/METR-LA/'
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default=Path(def_data))
    return parser


def plot_raw_values(ax, meta, row, time, norm, cmap):
    idxs = [int(x) for x in row.index.tolist()]
    vals = row.tolist()
    xs = meta.loc[idxs]['x'].tolist()
    ys = meta.loc[idxs]['y'].tolist()
    points = ax.scatter(xs, ys, alpha=0.5, c=vals, norm=norm, cmap=cmap)
    label = ax.text(0, 0, str(time), fontsize=24, transform=ax.transAxes)
    return [points, label]


def plot_sensor_pos(ax, meta):
    idxs = [int(x) for x in meta.index.tolist()]
    xs = meta.loc[idxs]['x'].tolist()
    ys = meta.loc[idxs]['y'].tolist()
    points = ax.scatter(xs, ys, s=25, edgecolors='none')


def plot_connect(ax, meta, adj, dist):
    ids = list(meta.index)
    close = dist < 2500
    adj_pairs = (adj*close).nonzero(as_tuple=False)
    lines = []
    for row in adj_pairs:
        from_idx, to_idx = row
        from_id = ids[from_idx]
        to_id = ids[to_idx]
        from_pos = meta.loc[from_id][['x', 'y']].tolist()
        to_pos = meta.loc[to_id][['x', 'y']].tolist()
        line = [from_pos, to_pos]
        lines.append(line)
    lc = coll.LineCollection(lines)
    ax.add_collection(lc)


def anim(base):
    all_fn = lambda _: True
    ds = METRDataset(base, all_fn, True, False)
    all_arts = []
    fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
    cnorm = colors.Normalize(vmin=0, vmax=70)
    cmap = 'rainbow'
    start = 60
    day = 288
    for i in range(start, start+day*2):
        row = ds.raw_df.loc[i]
        time = ds.raw_times.loc[i]
        art = plot_raw_values(ax, ds.meta_df, row, time, cnorm, cmap)
        all_arts.append(art)
    fig.colorbar(ScalarMappable(cnorm, cmap))
    anim = ArtistAnimation(fig, all_arts, interval=50)
    anim.save('out.mp4')


def single(base):
    all_fn = lambda _: True
    ds = METRDataset(base, all_fn, True, False)
    fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
    plot_sensor_pos(ax, ds.meta_df)
    # plot_connect(ax, ds.meta_df, ds.sensor_connected, ds.sensor_cost)
    ax.set_xlim([-25000, 15000])
    ax.set_ylim([-20000, 20000])
    plt.show(block=True)
    fig.savefig('out.png')


if __name__ == '__main__':
    args = make_parser().parse_args()
    single(args.data)
