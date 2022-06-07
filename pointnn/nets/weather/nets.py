from .. import pointnet
from .. import encodings
from .. import tpc
from ..base import Network

from . import dists

from functools import partial

import torch

HIST_FEAT_SIZE = 12
METADATA_SIZE = 19


class _WeatherBase(Network):
    def forward(self, hist, mask, times, station_meta, station_pos,
                station_idxs, target_pos, force_dropout=None):
        bs = hist.size(0)
        num_pts = hist.size(1)
        device = hist.device
        if force_dropout:
            dropout = force_dropout
        else:
            dropout = 0.5*torch.rand(1) if self.station_dropout == 'rand' else self.station_dropout
        if (self.training or force_dropout) and dropout > 0:
            sz = (bs, num_pts)
            prob = (1-dropout) * torch.ones(sz)
            ns_mask = torch.bernoulli(prob).bool().to(device)
            allowed = ns_mask.unsqueeze(1) * ns_mask.unsqueeze(2)
            target_allowed = ns_mask.unsqueeze(1)
        else:
            allowed = target_allowed = None
        tgt_latent = self.encode(hist, mask, times, station_meta,
                                 station_pos, station_idxs, allowed,
                                 target_allowed, target_pos)
        pred = self.pred(tgt_latent)
        return pred

    def make_predictors(self, target_size, decode_hidden, num_preds):
        args = {'in_size': target_size, 'out_size': num_preds,
                'hidden_sizes': decode_hidden, 'reduction': 'none'}
        self.pred = pointnet.SetTransform(**args)

    def get_args(self, item):
        return item['hist'], item['hist_mask'], \
               item['times'], \
               item['station_metadata'], item['station_pos'], \
               item['station_idxs'], item['target_pos']

    @staticmethod
    def _get_by_idx(idx, a):
        idxs_expand = idx.unsqueeze(-1).expand(-1, -1, a.size(-1))
        g = torch.gather(a, 1, idxs_expand)
        return g


class WeatherTPC(_WeatherBase):
    def __init__(self,
                 weight_hidden=[16, 32],
                 c_mid=32,
                 final_hidden=[256, 128, 128],
                 combine_hidden=[256, 128, 128],
                 neighborhood_sizes=[16, 16, 16],
                 latent_sizes=[32, 64, 64],
                 target_size=64,
                 decode_hidden=[64, 64, 64],
                 neighbors=8,
                 timesteps=12,
                 station_dropout=0,
                 heads=0):
        super().__init__()
        self.station_dropout = station_dropout
        # Setup
        in_size = HIST_FEAT_SIZE + METADATA_SIZE
        num_preds = 4
        self.make_encoders(in_size, weight_hidden, c_mid, final_hidden,
                           combine_hidden, neighborhood_sizes,
                           latent_sizes, neighbors, timesteps, target_size, heads)
        self.make_predictors(target_size, decode_hidden, num_preds)

    def make_encoders(self, feat_size, weight_hidden, c_mid, final_hidden,
                      combine_hidden, neighborhood_sizes,
                      latent_sizes, neighbors, timesteps, target_size, heads):
        pos_dim = 3
        self.time_encoder = encodings.PeriodEncoding(8, 20)
        self.tpc = tpc.TemporalPointConv(feat_size, weight_hidden, c_mid,
                                         final_hidden, latent_sizes, neighborhood_sizes,
                                         neighbors, timesteps, neighbors, combine_hidden,
                                         target_size, pos_dim, self.time_encoder, heads, 'space')

    def encode(self, hist, mask, times, station_meta, station_pos,
               station_idxs, allowed, target_allowed, target_pos):
        hist_pts = self._get_by_idx(station_idxs, station_pos)
        hist_meta = self._get_by_idx(station_idxs, station_meta)
        in_ = torch.cat([hist, hist_meta], dim=2)
        space_dist_fn = partial(dists.space, mask, station_idxs, times, allowed)
        time_dist_fn = partial(dists.time, mask, station_idxs, allowed, self.time_encoder.encode)
        target_dist_fn = partial(dists.target, mask, times, target_allowed)
        tgt_feats = self.tpc(in_, None, hist_pts, times, target_pos,
                             space_dist_fn, time_dist_fn, target_dist_fn)
        return tgt_feats


class WeatherGC(_WeatherBase):
    def __init__(self,
                 combine_hidden=[256, 128, 128],
                 neighborhood_sizes=[16, 16, 16],
                 latent_sizes=[32, 64, 64],
                 target_size=64,
                 decode_hidden=[64, 64, 64],
                 neighbors=8,
                 timesteps=12,
                 station_dropout=0):
        super().__init__()
        self.station_dropout = station_dropout
        # Setup
        in_size = HIST_FEAT_SIZE + METADATA_SIZE
        num_preds = 4
        self.make_encoders(in_size,
                           combine_hidden, neighborhood_sizes,
                           latent_sizes, neighbors, timesteps, target_size)
        self.make_predictors(target_size, decode_hidden, num_preds)

    def make_encoders(self, feat_size,
                      combine_hidden, neighborhood_sizes,
                      latent_sizes, neighbors, timesteps, target_size):
        pos_dim = 3
        self.time_encoder = encodings.PeriodEncoding(8, 20)
        self.tgc = tpc.TemporalGraphConv(feat_size, latent_sizes, neighborhood_sizes,
                                         combine_hidden, target_size, neighbors, timesteps,
                                         pos_dim, self.time_encoder, 'space')

    def encode(self, hist, mask, times, station_meta, station_pos,
               station_idxs, allowed, target_allowed, target_pos):
        hist_pts = self._get_by_idx(station_idxs, station_pos)
        hist_meta = self._get_by_idx(station_idxs, station_meta)
        in_ = torch.cat([hist, hist_meta], dim=2)
        space_dist_fn = partial(dists.space, mask, station_idxs, times, allowed)
        time_dist_fn = partial(dists.time, mask, station_idxs, allowed, self.time_encoder.encode)
        target_dist_fn = partial(dists.target, mask, times, target_allowed)
        tgt_feats = self.tgc(in_, None, hist_pts, times, hist_pts, target_pos,
                             space_dist_fn, time_dist_fn, target_dist_fn)
        return tgt_feats


class WeatherInteraction(_WeatherBase):
    def __init__(self,
                 edge_hidden=[16, 32],
                 combine_hidden=[256, 128, 128],
                 neighborhood_sizes=[16, 16, 16],
                 latent_sizes=[32, 64, 64],
                 target_size=64,
                 decode_hidden=[64, 64, 64],
                 neighbors=8,
                 timesteps=12,
                 station_dropout=0):
        super().__init__()
        self.station_dropout = station_dropout
        # Setup
        in_size = HIST_FEAT_SIZE + METADATA_SIZE
        num_preds = 4
        self.make_encoders(in_size, edge_hidden,
                           combine_hidden, neighborhood_sizes,
                           latent_sizes, neighbors, timesteps, target_size)
        self.make_predictors(target_size, decode_hidden, num_preds)

    def make_encoders(self, feat_size, edge_hidden,
                      combine_hidden, neighborhood_sizes,
                      latent_sizes, neighbors, timesteps, target_size):
        pos_dim = 3
        self.time_encoder = encodings.PeriodEncoding(8, 20)
        self.int = tpc.TemporalInteraction(feat_size, edge_hidden,
                                           latent_sizes, neighborhood_sizes,
                                           neighbors, timesteps, combine_hidden,
                                           target_size, pos_dim, self.time_encoder, 'space')

    def encode(self, hist, mask, times, station_meta, station_pos,
               station_idxs, allowed, target_allowed, target_pos):
        hist_pts = self._get_by_idx(station_idxs, station_pos)
        hist_meta = self._get_by_idx(station_idxs, station_meta)
        in_ = torch.cat([hist, hist_meta], dim=2)
        space_dist_fn = partial(dists.space, mask, station_idxs, times, allowed)
        time_dist_fn = partial(dists.time, mask, station_idxs, allowed, self.time_encoder.encode)
        target_dist_fn = partial(dists.target, mask, times, target_allowed)
        tgt_feats = self.int(in_, None, hist_pts, times, target_pos,
                             space_dist_fn, time_dist_fn, target_dist_fn)
        return tgt_feats
