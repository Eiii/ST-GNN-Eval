from ...data.starcraft import parse_frame

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as lines

import torch

colors = ('green', 'blue')
altcolors = ('m', 'y')
lightcolors = ('lightgreen', 'lightblue')
markers = 'os^'

def draw_scene(ax, scene, unit_mask, colors=lightcolors, alpha=0.5):
    pf = parse_frame(scene)
    alive_mask = pf['alive'].bool().squeeze(1)
    # Display all ALIVE UNITS
    mask = unit_mask * alive_mask
    mask_idx = mask.nonzero(as_tuple=True)[0]
    pos = pf['pos'][mask_idx]
    team = pf['owner'][mask_idx].squeeze(1)
    type_ = pf['type'][mask_idx].nonzero()[:,1]
    for t, color in zip((0, 1), colors):
        for y, ms in zip((0, 1, 2), markers):
            sel = (type_==y)&(team==t)
            p = pos[sel]
            r = ax.scatter(*p.unbind(1), marker=ms, alpha=alpha, c=color, zorder=15)

def draw_deltas(ax, scene, pred, unit_mask):
    pf = parse_frame(scene)
    pred_alive = pred['alive'][0, 0]
    pred_alive = (pred_alive[:, 1] > pred_alive[:, 0])
    actual_alive = pf['alive'].squeeze(1).bool()
    # Display all alive units we predicted to be alive
    # Draw position deltas
    mask = unit_mask * pred_alive * actual_alive
    mask_idx = mask.nonzero(as_tuple=True)[0]
    pre_pos = pf['pos'][mask_idx]
    next_pos = pred['pos'][0, 0, mask_idx].detach()
    for i in range(pre_pos.size(0)):
        p = pre_pos[i].tolist()
        n = next_pos[i].tolist()
        xs, ys = zip(p, n)
        l = lines.Line2D(xs, ys, color='red', linewidth=1, zorder=9)
        ax.add_line(l)
    # Highlight all alive units we predicted to be dead
    mask = unit_mask * actual_alive * pred_alive.logical_not()
    mask_idx = mask.nonzero(as_tuple=True)[0]
    pre_pos = pf['pos'][mask_idx]
    type_ = pf['type'][mask_idx].nonzero()[:,1]
    for y, ms in zip((0, 1, 2), markers):
        sel = (type_==y)
        p = pre_pos[sel]
        r = ax.scatter(*p.unbind(1), marker=ms, alpha=0.5, edgecolors='red',
                       facecolors='none', linewidth=1, zorder=2)
    # Highlight all dead units we predicted to be alive
    mask = unit_mask * actual_alive.logical_not() * pred_alive
    mask_idx = mask.nonzero(as_tuple=True)[0]
    pre_pos = pred['pos'][0, 0, mask_idx].detach()
    type_ = pf['type'][mask_idx].nonzero()[:,1]
    for y, ms in zip((0, 1, 2), markers):
        sel = (type_==y)
        p = pre_pos[ sel]
        r = ax.scatter(*p.unbind(1), marker=ms, alpha=1, edgecolors='red',
                       facecolors='none', linewidth=2, zorder=2)

def draw_scene_pred(ax, scene, pred, unit_mask, colors=colors):
    pf = parse_frame(scene)
    #
    alive = pred['alive'][0, 0]
    alive_mask = (alive[:, 1] > alive[:, 0])
    mask = unit_mask * alive_mask
    mask_idxs = mask.nonzero(as_tuple=True)[0]
    pos = pred['pos'][0, 0, mask_idxs].detach()
    team = pf['owner'][mask_idxs].squeeze(1)
    type_ = pf['type'][mask_idxs].nonzero()[:, 1]
    for t, color in zip((0, 1), colors):
        for y, ms in zip((0, 1, 2), markers):
            sel = (type_==y)&(team==t)
            p = pos[sel]
            r = ax.scatter(*p.unbind(1), marker=ms, alpha=0.5, c=color, zorder=18)

def plot_value_pred(ax, actual, pred, unit_mask, value, c='red'):
    pf = parse_frame(actual)
    alive = pf['alive'].bool().squeeze(1)
    pred_alive = pred['alive'][0, 0]
    pred_alive = (pred_alive[:, 1] > pred_alive[:, 0])
    mask = unit_mask.squeeze(0) * (pred_alive + alive)
    mask_idx = mask.nonzero(as_tuple=True)[0]
    actual = pf[value][mask_idx, 0].tolist()
    pred = pred[value][0, 0, mask_idx, 0].tolist()
    ax.scatter(actual, pred, marker='+', color=c)

def plot_timeline(ax, img):
    ax.imshow(img)

def plot_red_latent(ax, scene, lat, all_lats):
    all_lats = torch.stack(all_lats).squeeze(1).squeeze(1)
    for trace in all_lats.unbind(1):
        ax.plot(*trace.unbind(1), color='black', alpha=0.3)
    pf = parse_frame(scene)
    mask = pf['alive'].bool()
    mask_idx = mask.nonzero(as_tuple=True)[1]
    pos = pf['pos'][:, mask_idx]
    team = pf['owner'][0, mask_idx]
    type_ = pf['type'][:, mask_idx].nonzero()[:,0]
    for t, color in zip((0, 1), colors):
        for y, ms in zip((0, 1, 2), markers):
            sel = (type_.squeeze(0)==y)&(team==t)
            l = lat[mask_idx, :][sel, :]
            r = ax.scatter(*l.unbind(1), marker=ms, c=color, zorder=15)

def setup_frame_ax(ax):
    ax.set_xlim(-1.75, 1.5)
    ax.set_ylim(-1.1, 1.6)
    ax.set_aspect('equal')

def setup_val_ax(ax):
    ax.set_xlabel('Actual value')
    ax.set_ylabel('Predicted value')
    ax.plot([0,1], [0,1], alpha=0.25, color='black', zorder=-1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')

def setup_lat_ax(ax):
    pass

def setup_timeline(ax):
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)

