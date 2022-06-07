import torch
import torch.nn as nn
import torch.nn.functional as F
from .pointnet import SetTransform


class PointConv(nn.Module):
    def __init__(self,
                 neighbors,
                 c_in,
                 weight_hidden,
                 c_mid,
                 final_hidden,
                 c_out,
                 dim=3,
                 dist_fn=None,
                 attn_heads=0):
        super().__init__()
        self.dist_fn = dist_fn
        self.neighbor_count = neighbors
        self.attn_heads = attn_heads
        if attn_heads:
            self.attn_conv = SetTransform(in_size=2*c_in, out_size=attn_heads,
                                          hidden_sizes=list(weight_hidden),
                                          reduction='none')
        self.weight_conv = SetTransform(in_size=dim, out_size=c_mid,
                                        hidden_sizes=list(weight_hidden),
                                        reduction='none')
        final_in = c_in*c_mid
        if attn_heads:
            final_in *= attn_heads
        self.final_conv = SetTransform(in_size=final_in, out_size=c_out,
                                       hidden_sizes=list(final_hidden),
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
        # These are still grouped in an extra dimension by keys
        # Since each key shares the same convolution, the extra dimension
        # can just be flattened and reconstructed later.
        m = self.calc_weights(feats, neighbor_feats, neighbor_rel)
        # Apply mask
        neighbor_valid = neighbor_valid.unsqueeze(-1)
        masked_m = m * neighbor_valid
        masked_n_feats = neighbor_feats * neighbor_valid
        # Transpose for matrix multiplication
        e = torch.matmul(masked_m.transpose(2, 3), masked_n_feats)
        # The resulting mxn feature matrix can just be flattened-- the
        # dimensions are meaningless.
        e = e.view(e.size(0), e.size(1), -1)
        final = self.final_conv(e)
        return final

    def calc_weights(self, feats, neighbor_feats, neighbor_rel):
        m = self.weight_conv(neighbor_rel)
        if self.attn_heads:
            attn = self.calc_attn(feats, neighbor_feats)
            ms = []
            for head in range(attn.size(-1)):
                head_attn = attn[:, :, :, [head]]
                ms.append(m * head_attn)
            m = torch.cat(ms, dim=-1)
        return m

    def calc_attn(self, feats, neighbor_feats):
        exp_feats = feats.unsqueeze(-2)
        feat_context = torch.cat(torch.broadcast_tensors(exp_feats, neighbor_feats), dim=-1)
        raw_attn = self.attn_conv(feat_context)
        attn = F.softmax(raw_attn, dim=-2)
        return attn


class PointConvAttn(nn.Module):
    def __init__(self,
                 neighbors,
                 c_in,
                 weight_hidden,
                 c_mid,
                 final_hidden,
                 c_out,
                 dim=3,
                 dist_fn=None,
                 heads=3):
        super().__init__()
        self.dist_fn = dist_fn
        self.neighbor_count = neighbors
        self.weight_conv = SetTransform(in_size=dim, out_size=c_mid,
                                        hidden_sizes=list(weight_hidden),
                                        reduction='none')
        final_in = c_in*c_mid*heads
        self.final_conv = SetTransform(in_size=final_in, out_size=c_out,
                                       hidden_sizes=list(final_hidden),
                                       reduction='none')

    def forward(self, keys, points, feats, attn_conv, dist_fn=None, dist_data=None):
        if dist_data is not None:
            n_idxs, neighbor_rel, neighbor_valid = dist_data
            bb = torch.arange(n_idxs.size(0), device=n_idxs.device).view(-1, 1, 1)
            neighbor_feats = feats[bb, n_idxs]
        else:
            neighbor_rel, neighbor_feats, neighbor_valid = \
                calc_neighbor_info(keys, points, feats, self.neighbor_count,
                                   dist_fn or self.dist_fn)
        # These are still grouped in an extra dimension by keys
        # Since each key shares the same convolution, the extra dimension
        # can just be flattened and reconstructed later.
        m = self.calc_weights(feats, neighbor_feats, neighbor_rel)
        # Apply mask
        neighbor_valid = neighbor_valid.unsqueeze(-1)
        masked_m = m * neighbor_valid
        masked_n_feats = neighbor_feats * neighbor_valid
        # Transpose for matrix multiplication
        e = torch.matmul(masked_m.transpose(2, 3), masked_n_feats)
        # The resulting mxn feature matrix can just be flattened-- the
        # dimensions are meaningless.
        e = e.view(e.size(0), e.size(1), -1)
        final = self.final_conv(e)
        return final

    def calc_weights(self, orig_feats, neighbor_orig, neighbor_rel, attn_conv):
        m = self.weight_conv(neighbor_rel)
        results = []
        all_attn = self.calc_attn(orig_feats, neighbor_orig, attn_conv)
        for head in range(all_attn.size(-1)):
            attn = all_attn[:, :, :, [head]]
            results.append(m * attn)
        all_m = torch.cat(results, dim=-1)
        return all_m

    @staticmethod
    def calc_attn(feats, neighbor_feats, attn_conv):
        exp_feats = feats.unsqueeze(-2)
        feat_context = torch.cat(torch.broadcast_tensors(exp_feats, neighbor_feats), dim=-1)
        raw_attn = attn_conv(feat_context)
        attn = F.softmax(raw_attn, dim=-2)
        return attn


def calc_neighbor_info(keys, points, feats, neighbors, dist_fn):
    # Get closest points, features, valid per key
    n_idxs, rel_pos, valid = closest_pts_to_keys(keys, points, neighbors, dist_fn)
    return _gather_neighbor_info(feats, n_idxs, rel_pos, valid)


def closest_pts_to_keys(keys, points, neighbor_count, dist_fn):
    # Get valid flag, relative vector, distance measure from custom
    # distance function
    valid, dist_vec, dist = dist_fn(keys, points)
    assert(dist_vec is not dist)
    # Add large value to masked out entries so the can't be sorted to the
    # top
    big_dist = dist.max() * 3e3
    dist += valid.logical_not()*big_dist
    # There might not be enough neighbors if we get really unlucky
    if neighbor_count != -1:
        k_count = min(neighbor_count, dist.size(2))
    else:
        k_count = dist.size(2)
    _, idxs = dist.topk(k_count, dim=2, largest=False, sorted=False)
    return idxs, dist_vec, valid


def _gather_neighbor_info(feats, n_idxs, rel_pos, valid):
    # torch.gather requires some extra work to get tensor sizes to line up
    # Use the indexes of entities in calculated neighborhood (n_idxs) to
    # get:
    # * Relative vector from key entity to neighbors (neighbor_rel)
    tmp_idx = n_idxs.unsqueeze(3).expand(-1, -1, -1, rel_pos.size(3))
    neighbor_rel = torch.gather(rel_pos, dim=2, index=tmp_idx)
    # * Features of each neighbor entity (neighbor_feats)
    tmp_idx = n_idxs.unsqueeze(3).expand(-1, -1, -1, feats.size(2))
    tmp_f = feats.unsqueeze(1).expand(-1, tmp_idx.size(1), -1, -1)
    neighbor_feats = torch.gather(tmp_f, dim=2, index=tmp_idx)
    # * Valid flag for each neighbor (neighbor_valid)
    neighbor_valid = torch.gather(valid, dim=2, index=n_idxs)
    return neighbor_rel, neighbor_feats, neighbor_valid
