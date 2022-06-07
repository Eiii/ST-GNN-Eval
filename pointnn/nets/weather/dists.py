import torch


def cross(t1, t2=None, dims=(2, 1)):
    if t2 is None:
        t2 = t1
    _t1 = t1.unsqueeze(dims[0])
    _t2 = t2.unsqueeze(dims[1])
    return _t1, _t2


def space(mask, station_idx, times, allowed_neighbors, keys, points):
    # mask, station_idx, times:
    # BS x Entries
    # keys, points:
    # BS x Entries x 3
    # Calculate relative distances
    keys_e, pts_e = cross(keys, points)
    dist_vec = (keys_e - pts_e)
    sqr_dist = (dist_vec**2).sum(dim=-1)
    # Calculate valid neighbors
    times1, times2 = cross(times)
    same_time = (times1 == times2)
    sidx1, sidx2 = cross(station_idx)
    diff_stat = (sidx1 != sidx2)
    mask1, mask2 = cross(mask)
    cross_mask = mask1 * mask2
    valid = cross_mask * diff_stat * same_time
    if allowed_neighbors is not None:
        valid *= allowed_neighbors
    return valid, dist_vec, sqr_dist


def time(mask, station_idx, allowed_neighbors, time_enc, keys, points):
    # mask, station_idx, times:
    # BS x Entries
    # keys, points:
    # BS x Entries x 3
    # Calculate relative distances
    keys_e, pts_e = cross(keys, points)
    dist_vec = (keys_e - pts_e).unsqueeze(-1)
    sqr_dist = (dist_vec**2).sum(dim=-1)
    # Calculate valid neighbors
    diff_time = (keys_e != pts_e)
    sidx1, sidx2 = cross(station_idx)
    same_stat = (sidx1 == sidx2)
    mask1, mask2 = cross(mask)
    cross_mask = mask1 * mask2
    valid = cross_mask * same_stat * diff_time
    if allowed_neighbors is not None:
        valid *= allowed_neighbors
    return valid, time_enc(dist_vec), sqr_dist


def target(mask, times, allowed_neighbors, keys, points):
    # mask, station_idx, times:
    # BS x Entries
    # keys:
    # BS x Targets x 3
    # points:
    # BS x Entries x 3
    # Calculate relative distances
    keys_e, pts_e = cross(keys, points)
    dist_vec = (keys_e - pts_e)
    sqr_dist = (dist_vec**2).sum(dim=-1)
    # Calculate valid neighbors
    times1, times2 = cross(times)
    at_target = (times.unsqueeze(1) == 0)
    mask = mask.unsqueeze(1)
    valid = mask * at_target
    if allowed_neighbors is not None:
        valid *= allowed_neighbors
    valid, _ = torch.broadcast_tensors(valid, sqr_dist)
    return valid, dist_vec, sqr_dist
