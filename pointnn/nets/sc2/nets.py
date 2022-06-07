from . import dists
from .. import tpc
from .. import encodings
from ..base import Network
from ...data.starcraft import parse_frame
from ..pointnet import SetTransform

import torch

from functools import partial


class _SC2Common(Network):
    def make_decoders(self, in_size, decode_hidden):
        h_args = {'in_size': in_size, 'out_size': 1,
                  'hidden_sizes': decode_hidden, 'reduction': 'none'}
        o_args = {'in_size': in_size, 'out_size': 7,
                  'hidden_sizes': decode_hidden, 'reduction': 'none'}
        p_args = {'in_size': in_size, 'out_size': 2,
                  'hidden_sizes': decode_hidden, 'reduction': 'none'}
        a_args = {'in_size': in_size, 'out_size': 2,
                  'hidden_sizes': decode_hidden, 'reduction': 'none'}
        self.health_net = SetTransform(**h_args)
        self.shield_net = SetTransform(**h_args)
        self.ori_net = SetTransform(**o_args)
        self.pos_net = SetTransform(**p_args)
        self.alive_net = SetTransform(**a_args)

    def get_args(self, item):
        keys = ('data', 'ids', 'ts', 'mask', 'pred_ids', 'pred_ts')
        return [item[k] for k in keys]

    def forward(self, data, ids, ts, mask, pred_ids, pred_ts):
        target_feats = self.encode_targets(data, ids, ts, mask, pred_ids, pred_ts)
        preds = self.predict(target_feats)
        return self.assemble_frame(*preds)

    def encode_targets(self, data, ids, ts, mask, pred_ids, pred_ts):
        enc_data = self.encode(data, ids, ts, mask)
        target_feats = self.calc_targets(enc_data, ids, ts, mask, pred_ids, pred_ts)
        return target_feats

    def predict(self, unit_feats):
        healths = self.health_net(unit_feats)
        shields = self.shield_net(unit_feats)
        oris = self.ori_net(unit_feats)
        poss = self.pos_net(unit_feats)
        alive = self.alive_net(unit_feats)
        return [healths, shields, oris, poss, alive]

    def assemble_frame(self, health, shield, ori, pos, alive):
        return {'health': health,
                'shields': shield,
                'ori': ori,
                'pos': pos,
                'alive': alive}


class SC2TPC(_SC2Common):
    def __init__(self,
                 neighborhood_sizes,
                 latent_sizes,
                 target_size,
                 combine_hidden,
                 weight_hidden,
                 c_mid,
                 final_hidden,
                 decode_hidden,
                 neighbors, timesteps, heads=0):
        super().__init__()
        feat_size = 16
        pos_dim = 2
        self.time_encoder = encodings.PeriodEncoding(8, 10)
        self.make_tpc(feat_size, weight_hidden, c_mid, final_hidden,
                      latent_sizes, neighborhood_sizes, neighbors,
                      timesteps, combine_hidden, target_size, pos_dim,
                      self.time_encoder, heads)
        self.make_decoders(target_size, decode_hidden)

    def make_tpc(self, feat_size, weight_hidden, c_mid, final_hidden,
                 latent_sizes, neighborhood_sizes, neighbors,
                 timesteps, combine_hidden, target_size, pos_dim,
                 time_encoder, heads):
        self.tpc = tpc.TemporalPointConv(feat_size, weight_hidden, c_mid, final_hidden,
                                         latent_sizes, neighborhood_sizes, neighbors,
                                         timesteps, timesteps, combine_hidden,
                                         target_size, pos_dim, time_encoder, heads)

    def run_tpc(self, data, ids, pos, ts, flat_pred_ts, space_dist_fn, time_dist_fn, target_dist_fn):
        out = self.tpc(data, ids, pos, ts, flat_pred_ts, space_dist_fn, time_dist_fn, target_dist_fn)
        return out

    def encode_targets(self, data, ids, ts, mask, pred_ids, pred_ts):
        pf = parse_frame(data)
        pos = pf['pos']
        space_dist_fn = partial(dists.space, mask, ts)
        time_dist_fn = partial(dists.time, mask, ids, self.time_encoder.encode)
        # Target info
        expand_pred_ts = pred_ts.unsqueeze(-1)
        expand_pred_ids = pred_ids.unsqueeze(-2)
        expand_pred_ts, expand_pred_ids = torch.broadcast_tensors(expand_pred_ts, expand_pred_ids)
        flat_pred_ts = expand_pred_ts.reshape(expand_pred_ts.size(0), -1)
        flat_pred_ids = expand_pred_ids.reshape(expand_pred_ids.size(0), -1)
        target_dist_fn = partial(dists.target, mask, flat_pred_ids, ids, self.time_encoder.encode)
        out = self.run_tpc(data, ids, pos, ts, flat_pred_ts, space_dist_fn, time_dist_fn, target_dist_fn)
        target_feats = out.view(expand_pred_ts.size()+(out.size(-1),))
        return target_feats


class SC2Interaction(_SC2Common):
    def __init__(self,
                 neighborhood_sizes,
                 latent_sizes,
                 target_size,
                 combine_hidden,
                 edge_hidden,
                 decode_hidden,
                 neighbors, timesteps):
        super().__init__()
        feat_size = 16
        pos_dim = 2
        self.time_encoder = encodings.PeriodEncoding(8, 10)
        self.make_xxx(feat_size, edge_hidden,
                      latent_sizes, neighborhood_sizes, neighbors,
                      timesteps, combine_hidden, target_size, pos_dim,
                      self.time_encoder)
        self.make_decoders(target_size, decode_hidden)

    def make_xxx(self, feat_size, edge_hidden,
                 latent_sizes, neighborhood_sizes, neighbors,
                 timesteps, combine_hidden, target_size, pos_dim,
                 time_encoder):
        self.int = tpc.TemporalInteraction(feat_size, edge_hidden,
                                           latent_sizes, neighborhood_sizes, neighbors,
                                           timesteps, combine_hidden, target_size, pos_dim,
                                           time_encoder)

    def run_tpc(self, data, ids, pos, ts, flat_pred_ts, space_dist_fn, time_dist_fn, target_dist_fn):
        out = self.int(data, ids, pos, ts, flat_pred_ts, space_dist_fn, time_dist_fn, target_dist_fn)
        return out

    def encode_targets(self, data, ids, ts, mask, pred_ids, pred_ts):
        pf = parse_frame(data)
        pos = pf['pos']
        space_dist_fn = partial(dists.space, mask, ts)
        time_dist_fn = partial(dists.time, mask, ids, self.time_encoder.encode)
        # Target info
        expand_pred_ts = pred_ts.unsqueeze(-1)
        expand_pred_ids = pred_ids.unsqueeze(-2)
        expand_pred_ts, expand_pred_ids = torch.broadcast_tensors(expand_pred_ts, expand_pred_ids)
        flat_pred_ts = expand_pred_ts.reshape(expand_pred_ts.size(0), -1)
        flat_pred_ids = expand_pred_ids.reshape(expand_pred_ids.size(0), -1)
        target_dist_fn = partial(dists.target, mask, flat_pred_ids, ids, self.time_encoder.encode)
        out = self.run_tpc(data, ids, pos, ts, flat_pred_ts, space_dist_fn, time_dist_fn, target_dist_fn)
        target_feats = out.view(expand_pred_ts.size()+(out.size(-1),))
        return target_feats


class SC2GC(_SC2Common):
    def __init__(self,
                 neighborhood_sizes,
                 latent_sizes,
                 target_size,
                 combine_hidden,
                 decode_hidden,
                 neighbors, timesteps):
        super().__init__()
        feat_size = 16
        pos_dim = 2
        self.time_encoder = encodings.PeriodEncoding(8, 10)
        self.tgc = tpc.TemporalGraphConv(feat_size, latent_sizes, neighborhood_sizes,
                                         combine_hidden, target_size, neighbors, timesteps,
                                         pos_dim, self.time_encoder, 'time')
        self.make_decoders(target_size, decode_hidden)

    def encode_targets(self, data, ids, ts, mask, pred_ids, pred_ts):
        pf = parse_frame(data)
        pos = pf['pos']
        space_dist_fn = partial(dists.space, mask, ts)
        time_dist_fn = partial(dists.time, mask, ids, self.time_encoder.encode)
        # Target info
        expand_pred_ts = pred_ts.unsqueeze(-1)
        expand_pred_ids = pred_ids.unsqueeze(-2)
        expand_pred_ts, expand_pred_ids = torch.broadcast_tensors(expand_pred_ts, expand_pred_ids)
        flat_pred_ts = expand_pred_ts.reshape(expand_pred_ts.size(0), -1)
        flat_pred_ids = expand_pred_ids.reshape(expand_pred_ids.size(0), -1)
        target_dist_fn = partial(dists.target, mask, flat_pred_ids, ids, self.time_encoder.encode)
        out = self.tgc(data, ids, pos, ts, ts, flat_pred_ts, space_dist_fn, time_dist_fn, target_dist_fn)
        target_feats = out.view(expand_pred_ts.size()+(out.size(-1),))
        return target_feats
