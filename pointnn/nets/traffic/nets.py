from .. import pointnet
from .. import encodings
from .. import tpc
from ..base import Network

from . import dists

import torch
from functools import partial


class _TrafficCommon(Network):
    def get_args(self, item):
        xs = ['hist_t', 'hist_id', 'hist_pos', 'hist_data', 'id_dist', 'id_adj', 'hist_mask',
              'tgt_t', 'tgt_id', 'tgt_mask']
        args = [item[x] for x in xs]
        return args

    def _calc_means(self, tgt_id, hist_id, hist_data):
        all_ids = tgt_id.unique()
        means = torch.zeros_like(tgt_id, dtype=torch.float)
        for id_ in all_ids:
            hist_mask = (hist_id == id_)
            id_data = hist_data * hist_mask.unsqueeze(-1)
            id_avg = id_data.sum(dim=-1).sum(dim=-1) / hist_mask.sum(dim=-1)
            tgt_idxs = (tgt_id == id_).nonzero(as_tuple=True)
            tgt_means = id_avg[tgt_idxs[0]]
            means[tgt_idxs] = tgt_means
        return means


class TrafficTPC(_TrafficCommon):
    tpc_class = tpc.TemporalPointConv

    def __init__(self,
                 neighborhood_sizes=[2**4, 2**5, 2**5],
                 latent_sizes=[2**5, 2**6, 2**6],
                 target_size=2**6,
                 combine_hidden=[2**6, 2**6],
                 weight_hidden=[2**5, 2**5],
                 c_mid=2**5,
                 final_hidden=[2**6, 2**6],
                 decode_hidden=[2**6, 2**6, 2**6],
                 neighbors=8, timesteps=12,
                 mean_delta=False,
                 heads=0):
        super().__init__()
        feat_size = 1
        pos_dim = 2+1
        self.mean_delta = mean_delta
        self.time_encoder = encodings.DirectEncoding()
        self.neighbors = neighbors
        self.timesteps = timesteps
        self.tpc = self.tpc_class(feat_size, weight_hidden, c_mid, final_hidden,
                                  latent_sizes, neighborhood_sizes, neighbors,
                                  timesteps, timesteps, combine_hidden,
                                  target_size, pos_dim, self.time_encoder, heads)
        self.make_decoders(target_size, decode_hidden)

    def make_decoders(self, in_size, decode_hidden):
        args = {'in_size': in_size, 'out_size': 1,
                'hidden_sizes': decode_hidden, 'reduction': 'none'}
        self.pred_net = pointnet.SetTransform(**args)

    def decode_queries(self, feats):
        pred = self.pred_net(feats)
        return pred

    def forward(self, hist_t, hist_id, hist_pos, hist_data, id_dist, id_adj, hist_mask,
                tgt_t, tgt_id, tgt_mask):
        space_dist_data = dists.space_neighbors(self.neighbors, hist_t, hist_id, hist_pos, hist_mask, id_dist, id_adj)
        time_dist_data = dists.time_neighbors(self.timesteps, 5, hist_t, hist_id, hist_mask)
        query_dist_fn = partial(dists.query, self.time_encoder.encode, tgt_mask, hist_mask,
                                tgt_id, hist_id)
        hist_data = hist_data.unsqueeze(-1)
        query_feats = self.tpc(hist_data, hist_id, hist_pos, hist_t, tgt_t,
                               space_dist_data=space_dist_data, time_dist_data=time_dist_data, target_dist_fn=query_dist_fn)
        pred = self.decode_queries(query_feats)
        if self.mean_delta:
            means = self._calc_means(tgt_id, hist_id, hist_data)
            pred = pred+means.unsqueeze(-1)
        return pred


class TrafficInteraction(_TrafficCommon):
    int_class = tpc.TemporalInteraction

    def __init__(self,
                 neighborhood_sizes=[2**4, 2**5, 2**5],
                 latent_sizes=[2**5, 2**6, 2**6],
                 target_size=2**6,
                 combine_hidden=[2**6, 2**6],
                 edge_hidden=[2**5, 2**5],
                 decode_hidden=[2**6, 2**6, 2**6],
                 neighbors=8, timesteps=12,
                 mean_delta=False):
        super().__init__()
        feat_size = 1
        pos_dim = 2
        self.mean_delta = mean_delta
        self.time_encoder = encodings.DirectEncoding()
        self.neighbors = neighbors
        self.timesteps = timesteps
        self.int = self.int_class(feat_size, edge_hidden,
                                  latent_sizes, neighborhood_sizes, neighbors,
                                  timesteps, combine_hidden, target_size, pos_dim,
                                  self.time_encoder)
        self.make_decoders(target_size, decode_hidden)

    def make_decoders(self, in_size, decode_hidden):
        args = {'in_size': in_size, 'out_size': 1,
                'hidden_sizes': decode_hidden, 'reduction': 'none'}
        self.pred_net = pointnet.SetTransform(**args)

    def decode_queries(self, feats):
        pred = self.pred_net(feats)
        return pred

    def forward(self, hist_t, hist_id, hist_pos, hist_data, id_dist, id_adj, hist_mask,
                tgt_t, tgt_id, tgt_mask):
        space_dist_data = dists.space_neighbors(self.neighbors, hist_t, hist_id, hist_pos, hist_mask, id_dist, id_adj)
        time_dist_data = dists.time_neighbors(self.timesteps, 5, hist_t, hist_id, hist_mask)
        query_dist_fn = partial(dists.query, self.time_encoder.encode, tgt_mask, hist_mask,
                                tgt_id, hist_id)
        hist_data = hist_data.unsqueeze(-1)
        query_feats = self.int(hist_data, hist_id, hist_pos, hist_t, tgt_t,
                               space_dist_data=space_dist_data,
                               time_dist_data=time_dist_data,
                               target_dist_fn=query_dist_fn)
        pred = self.decode_queries(query_feats)
        if self.mean_delta:
            means = self._calc_means(tgt_id, hist_id, hist_data)
            pred = pred+means.unsqueeze(-1)
        return pred


class TrafficGraphConv(Network):
    def __init__(self,
                 neighborhood_sizes=[2**4, 2**5, 2**5],
                 latent_sizes=[2**5, 2**6, 2**6],
                 target_size=2**6,
                 combine_hidden=[2**6, 2**6],
                 decode_hidden=[2**6, 2**6, 2**6],
                 neighbors=8, timesteps=12):
        super().__init__()
        feat_size = 1
        pos_dim = 2+1
        self.time_encoder = encodings.DirectEncoding()
        self.neighbors = neighbors
        self.timesteps = timesteps
        self.tgc = tpc.TemporalGraphConv(feat_size, latent_sizes, neighborhood_sizes,
                                         combine_hidden, target_size, neighbors, timesteps,
                                         pos_dim, self.time_encoder, 'time')
        self.make_decoders(target_size, decode_hidden)

    def get_args(self, item):
        xs = ['hist_t', 'hist_id', 'hist_pos', 'hist_data', 'id_dist',
              'id_adj', 'hist_mask', 'tgt_t', 'tgt_id', 'tgt_mask']
        return [item[x] for x in xs]

    def make_decoders(self, in_size, decode_hidden):
        args = {'in_size': in_size, 'out_size': 1,
                'hidden_sizes': decode_hidden, 'reduction': 'none'}
        self.pred_net = pointnet.SetTransform(**args)

    def decode_queries(self, feats):
        pred = self.pred_net(feats)
        return pred

    def forward(self, hist_t, hist_id, hist_pos, hist_data, id_dist, id_adj, hist_mask,
                tgt_t, tgt_id, tgt_mask):
        space_dist_data = dists.space_neighbors(self.neighbors, hist_t, hist_id, hist_pos, hist_mask, id_dist, id_adj)
        time_dist_data = dists.time_neighbors(self.timesteps, 5, hist_t, hist_id, hist_mask)
        query_dist_fn = partial(dists.query, self.time_encoder.encode, tgt_mask, hist_mask,
                                tgt_id, hist_id)
        hist_data = hist_data.unsqueeze(-1)
        query_feats = self.tgc(hist_data, hist_id, hist_pos, hist_t, hist_t, tgt_t,
                               space_dist_data=space_dist_data,
                               time_dist_data=time_dist_data,
                               target_dist_fn=query_dist_fn)
        pred = self.decode_queries(query_feats)
        return pred
