import torch
import torch.nn as nn

from .pointnet import SetTransform
from .pointconv import calc_neighbor_info


class InteractionNetworkNeighborhood(nn.Module):
    def __init__(self,
                 neighbors,
                 c_in,
                 edge_hidden,
                 c_out,
                 dim=3,
                 dist_fn=None,
                 key_feats='self',
                 pos_encoder=None
                 ):
        super().__init__()
        self.dist_fn = dist_fn
        self.neighbor_count = neighbors
        assert key_feats in ('self', 'query')
        self.key_feats = key_feats
        self.pos_encoder = pos_encoder
        if key_feats == 'self':
            in_sz = c_in*2
        elif key_feats == 'query':
            in_sz = c_in+dim
        self.edge_conv = SetTransform(in_size=in_sz, out_size=c_out,
                                      hidden_sizes=list(edge_hidden),
                                      reduction='none')

    def forward(self, keys, points, feats, dist_fn=None, dist_data=None):
        if dist_data is not None:
            n_idxs, neighbor_rel, neighbor_valid = dist_data
            bb = torch.arange(n_idxs.size(0), device=n_idxs.device).view(-1, 1, 1)
            neighbor_feats = feats[bb, n_idxs]
        else:
            neighbor_rel, neighbor_feats, neighbor_valid = \
                calc_neighbor_info(keys, points, feats, self.neighbor_count,
                                   dist_fn or self.dist_fn)
        edges = self.calc_edges(feats, keys, neighbor_feats)
        neighbor_valid = neighbor_valid.unsqueeze(-1)
        masked_edges = edges * neighbor_valid
        reduced_edges = masked_edges.sum(dim=2)
        return reduced_edges

    def calc_edges(self, feats, keys, neighbor_feats):
        if self.key_feats == 'self':
            exp_feats = feats.unsqueeze(2).expand(-1, -1, neighbor_feats.size(2), -1)
        elif self.key_feats == 'query':
            if self.pos_encoder is not None:
                enc_keys = self.pos_encoder.encode(keys.view([*keys.shape, 1, 1]))
            else:
                enc_keys = keys.unsqueeze(-2)
            exp_feats = enc_keys.expand(-1, -1, neighbor_feats.size(2), -1)
        pairs = torch.cat((exp_feats, neighbor_feats), dim=-1)
        edges = self.edge_conv(pairs)
        return edges
