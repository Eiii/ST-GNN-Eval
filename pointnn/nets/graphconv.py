import torch
import torch.nn as nn
import torch.nn.functional as F

from .pointconv import calc_neighbor_info


class GraphConv(nn.Module):
    def __init__(self, in_size, out_size, rel_size, neighbor_count):
        super().__init__()
        self.neighbor_count = neighbor_count
        self.w = nn.Linear(in_size+rel_size, out_size)

    def forward(self, keys, points, feats, dist_fn=None, dist_data=None):
        if dist_data is not None:
            n_idxs, neighbor_rel, neighbor_valid = dist_data
            bb = torch.arange(n_idxs.size(0), device=n_idxs.device).view(-1, 1, 1)
            neighbor_feats = feats[bb, n_idxs]
        else:
            neighbor_rel, neighbor_feats, neighbor_valid = \
                calc_neighbor_info(keys, points, feats, self.neighbor_count,
                                   dist_fn or self.dist_fn)
        # Zero out invalid feats
        comb_feats = torch.cat((neighbor_feats, neighbor_rel), dim=-1)
        cleared_feats = comb_feats * neighbor_valid.unsqueeze(-1)
        neighbor_counts = neighbor_valid.sum(-1)
        inv_n_counts = neighbor_counts.float() ** -1
        inv_n_counts[neighbor_counts == 0] = 0
        sum_feats = cleared_feats.sum(dim=2)
        scaled_feats = sum_feats * inv_n_counts.unsqueeze(-1)
        out_feats = F.relu(self.w(scaled_feats))
        return out_feats
