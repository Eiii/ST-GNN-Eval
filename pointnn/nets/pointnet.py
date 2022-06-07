import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import pairwise


class SetTransform(nn.Module):
    def __init__(self, in_size, out_size, hidden_sizes,
                 reduction):
        super().__init__()
        assert reduction in ('none', 'max', 'sum')
        self.reduction = reduction
        self._create_layers(in_size, hidden_sizes, out_size)

    def _create_layers(self, in_size, hidden_sizes, out_size):
        self.mlps = nn.ModuleList()
        self.norms = nn.ModuleList()
        for in_, out_ in pairwise([in_size]+hidden_sizes+[out_size]):
            self.norms.append(nn.BatchNorm1d(in_))
            self.mlps.append(nn.Linear(in_, out_))

    def pointwise_transform(self, x):
        def _apply(x, mlp, norm, relu):
            # Linear can work with B x ... x C
            # BN1D requires B x C
            # Flatten the extra dimensions into B and restore them to keep
            # the layer happy
            flat_x, orig_prefix = flatten(x)
            norm_flat_x = norm(flat_x)
            norm_x = restore(norm_flat_x, orig_prefix)
            out_x = mlp(norm_x)
            if relu:
                out_x = F.relu(out_x)
            return out_x
        layer_groups = list(zip(self.mlps, self.norms))
        for mlp, norm in layer_groups[:-1]:
            x = _apply(x, mlp, norm, True)
        for mlp, norm in layer_groups[-1:]:
            x = _apply(x, mlp, norm, False)
        return x

    def reduce(self, x, reduction=None):
        reduction = reduction or self.reduction
        if reduction == 'none':
            return x
        elif reduction == 'max':
            return x.max(-2)[0]
        elif reduction == 'sum':
            return x.sum(-2)

    def forward(self, set_feats):
        xf_feats = self.pointwise_transform(set_feats)
        final_feats = self.reduce(xf_feats)
        return final_feats


def flatten(t):
    orig_prefix = t.size()[:-1]
    flat_t = t.view(-1, t.size(-1))
    return flat_t, orig_prefix


def restore(flat_t, orig_prefix):
    new_size = orig_prefix+(flat_t.size(-1),)
    return flat_t.view(*new_size)
