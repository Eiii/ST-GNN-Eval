from .. import nets

import pickle
from pathlib import Path


def load_any(in_path):
    in_path = Path(in_path)
    paths = in_path.glob('*.pkl') if in_path.is_dir() else [in_path]
    return [load_result(p) for p in paths]


def load_result(path):
    with open(path, 'rb') as fd:
        data = pickle.load(fd)
    return data


def make_net(result):
    nt = result['net_type']
    na = result['net_args']
    #HACK
    if nt in ('SC2TPC', 'SC2Interaction'):
        if 'neighbor_attn' in na:
            del na['neighbor_attn']
    net = nets.make_net_args(nt, na)
    net.load_state_dict(result['state_dict'])
    return net
