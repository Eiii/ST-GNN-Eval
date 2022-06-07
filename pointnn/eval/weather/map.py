import itertools
import argparse

import numpy as np

from ...data.weather import WeatherDataset

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def get_stations(ds):
    pos = ds.raw_metadata[['x_m', 'y_m']]
    names = list(ds.raw_metadata.index)
    return names, pos.to_numpy()


COLORS = ['r', 'g', 'yellow', 'b']
def plot_loc(ax, names, pts, show_names=True, highlight_groups=None):
    ax.axis('equal')
    xy = zip(*pts)
    ax.scatter(*xy, c='black', s=15)
    if highlight_groups is not None:
        for group, color in zip(highlight_groups, itertools.cycle(COLORS)):
            group_pts = pts[group]
            xy = zip(*group_pts)
            ax.scatter(*xy, c=color)
    if show_names:
        for name, pt in zip(names, pts):
            ax.annotate(name, pt)
    # Set up ax
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("X")
    ax.set_ylabel("Y", rotation=0, labelpad=20)


def sample_space_groups(pts, num_groups=1, group_size=12):
    p1 = np.expand_dims(pts, 0)
    p2 = np.expand_dims(pts, 1)
    sqr_dists = ((p1-p2)**2).sum(-1)
    closest = np.argsort(sqr_dists, 1)
    np.random.shuffle(closest)
    closest = closest[:num_groups]
    closest = closest[:, :group_size]
    return closest


def make_3d(pts, timesteps=3, dist=1):
    ids = np.arange(pts.shape[0])
    zs = np.zeros((pts.shape[0], 1))
    pts = np.concatenate((pts, zs), 1)
    pts = np.broadcast_to(pts, (timesteps, pts.shape[0], pts.shape[1])).copy()
    ids = np.broadcast_to(ids, (timesteps, ids.shape[0])).copy()
    for i in range(timesteps):
        pts[i, :, -1] = -5 * i
    pts = pts.reshape(-1, pts.shape[-1])
    ids = ids.reshape(-1)
    return pts, ids


def plot_time(ax, pts, ids, num_groups=4):
    ax.view_init(elev=10, azim=-65)
    # Choose IDs to display
    all_ids = np.unique(ids)
    np.random.shuffle(all_ids)
    grp_ids = all_ids[:num_groups]
    other_ids = all_ids[num_groups:]
    other_pts = pts[np.isin(ids, other_ids)]
    xs, ys, ts = zip(*other_pts)
    ax.scatter(ts, xs, ys, depthshade=False, c='black', alpha=0.8, s=10,
               edgecolors='none')
    for i, color in zip(grp_ids, COLORS):
        grp_pts = pts[ids == i]
        xs, ys, ts = zip(*grp_pts)
        ax.scatter(ts, xs, ys, depthshade=False, c=color, s=40,
                   edgecolors='none')
        ax.plot(ts, xs, ys, c=color)
    # Set up axes
    ax.set_xlabel("Time", labelpad=30)
    ax.set_ylabel("X")
    ax.set_zlabel("Y")
    ax.set_xticks([0, -5, -10])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = True
    ax.grid(True)


def config_mpl():
    plt.rc('font', size=24)
    plt.rc('font', family='serif')
    plt.rc('axes', labelsize=28)


def main(path):
    config_mpl()
    ds = WeatherDataset(path, HACK_SNIP=2)
    names, pts = get_stations(ds)
    groups = sample_space_groups(pts, num_groups=3)
    fig = plt.figure(figsize=(10, 5), dpi=200)
    ax = fig.add_subplot()
    plot_loc(ax, names, pts, show_names=False, highlight_groups=groups)
    plt.tight_layout()
    plt.savefig('map_space.png')

    fig = plt.figure(figsize=(10, 5), dpi=200)
    ax = fig.add_subplot(projection='3d')
    pts_3d, ids = make_3d(pts, 3)
    plot_time(ax, pts_3d, ids, num_groups=3)
    plt.tight_layout()
    plt.savefig('map_time.png')


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='./data/weather')
    return parser


if __name__ == '__main__':
    args = make_parser().parse_args()
    main(args.data)
