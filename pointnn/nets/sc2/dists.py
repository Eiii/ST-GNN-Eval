import torch
from .. import encodings

def space(mask, ts, keys, points):
    key_pos = keys.unsqueeze(2)
    point_pos = points.unsqueeze(1)
    key_ts = ts.unsqueeze(2)
    point_ts = ts.unsqueeze(1)
    # Calculate spatial distances
    dist_vec = key_pos - point_pos
    sqr_dist = (dist_vec**2).sum(dim=-1)
    # Calculate mask
    # Units in a single timestep cannot be their own neighbor
    same_unit = torch.eye(sqr_dist.size(1), dtype=torch.bool, device=sqr_dist.device).unsqueeze(0)
    # Units must be in the same timestep to be neighbors
    same_t = (key_ts == point_ts)
    # Dead units cannot be anyone's neighbor
    # The unit must not be disabled by the batch's time mask or unit mask
    # Combine the above
    valid = same_unit.logical_not() * same_t
    valid *= mask.unsqueeze(1) * mask.unsqueeze(2)
    return valid, dist_vec, sqr_dist

def spacetime(mask, time_factor, keys, points):
    expand_keys = keys.unsqueeze(2)
    expand_points = points.unsqueeze(1)
    # Separate spatial distances from time distance
    xy_idxs = [0, 1]
    t_idxs = 2
    key_pos = expand_keys
    point_pos = expand_points
    # Calculate spatial distances
    dist_vec = key_pos - point_pos
    dist_vec[:, :, :, t_idxs] *= time_factor
    sqr_dist = (dist_vec**2).sum(dim=-1)
    # Calculate mask
    # Units in a single timestep cannot be their own neighbor
    same_unit = torch.eye(sqr_dist.size(1), dtype=torch.bool, device=sqr_dist.device).unsqueeze(0)
    # Dead units cannot be anyone's neighbor
    # The unit must not be disabled by the batch's time mask or unit mask
    # Combine the above
    exp_mask = mask.unsqueeze(1) * mask.unsqueeze(2)
    valid = same_unit.logical_not() * exp_mask
    return valid, dist_vec, sqr_dist

def time(mask, ids, time_enc, keys, points):
    key_ts = keys.unsqueeze(2)
    point_ts = points.unsqueeze(1)
    key_ids = ids.unsqueeze(2)
    point_ids = ids.unsqueeze(1)
    # Calculate spatial distances
    dist_vec = key_ts - point_ts
    dist = dist_vec.abs()
    dist_vec = dist_vec.unsqueeze(-1)
    # Determine which distances matter
    # Units in a single timestep cannot be their own neighbor
    same_unit = torch.eye(dist.size(1), dtype=torch.bool, device=dist.device).unsqueeze(0)
    same_id = (key_ids == point_ids)
    # Combine the above
    valid = same_unit.logical_not() * same_id
    valid *= mask.unsqueeze(1) * mask.unsqueeze(2)
    return valid, time_enc(dist_vec), dist

def target(point_mask, key_ids, point_ids, time_enc, keys, points):
    key_ts = keys.unsqueeze(2)
    key_ids = key_ids.unsqueeze(2)
    point_ts = points.unsqueeze(1)
    point_ids = point_ids.unsqueeze(1)
    # Calculate spatial distances
    dist_vec = key_ts - point_ts
    dist = dist_vec.abs()
    dist_vec = dist_vec.unsqueeze(-1)
    # Determine which distances matter
    # Units in a single timestep cannot be their own neighbor
    same_id = (key_ids == point_ids)
    # Combine the above
    valid = same_id * point_mask.unsqueeze(1)
    return valid, time_enc(dist_vec), dist

