from .common import pad_tensors
import pickle
import itertools
import time

from collections import namedtuple
from pathlib import Path

import numpy as np
import scipy.stats as S

import torch
from torch.utils.data import Dataset

Entry = namedtuple('Entry', ['ep_idx', 'hist_idxs', 'hist_ts', 'pred_ts',
                             'pred_idxs'])


class StarcraftDataset(Dataset):
    def __init__(self, base,
                 max_hist=3,
                 num_pred=1,
                 hist_dist='fixed',
                 hist_dist_args=None,
                 pred_dist='fixed',
                 pred_dist_args=None,
                 max_files=None,
                 frame_skip=2):
        self.max_hist = max_hist
        self.num_pred = num_pred
        assert hist_dist in ('fixed', 'uniform', 'tail')
        self.hist_dist = hist_dist
        self.hist_dist_args = hist_dist_args
        if hist_dist == 'tail':
            max_ = hist_dist_args['max']
            scale = hist_dist_args.get('scale', 1)
            self.hist_tail_probs = np.array([S.norm.pdf(x, scale=num_pred/scale) for x in range(max_+1)])
            self.hist_tail_probs /= sum(self.hist_tail_probs)
        assert pred_dist in ('fixed', 'uniform', 'tail')
        self.pred_dist = pred_dist
        self.pred_dist_args = pred_dist_args
        if pred_dist == 'tail':
            max_ = pred_dist_args['max']
            scale = pred_dist_args.get('scale', 1)
            self.pred_tail_probs = np.array([S.norm.pdf(x, scale=num_pred/scale) for x in range(max_)])
            self.pred_tail_probs /= sum(self.pred_tail_probs)
        # Read the unprocessed episode frames
        print(f'Loading SC dataset @ {base}..')
        start = time.time()
        self.raw_episodes = self.load_episodes(Path(base), max_files)
        print(f'Raw episodes loaded. {time.time()-start:.2f}s')
        # Calculate all the entries in the dataset
        start = time.time()
        self.all_entries = self.calc_dataset(frame_skip)
        print(f'Done ({len(self.raw_episodes)} episodes, {len(self.all_entries)} entries).')
        print(f'{time.time()-start:.2f}s')

    def load_episodes(self, base_dir, max_files=None):
        files = sorted(list(base_dir.glob('*.pkl')))
        all_episodes = []
        if max_files is not None:
            print('WARNING: ONLY LOADING PARTIAL DATASET')
            files = itertools.islice(files, max_files)
        for f in files:
            with f.open('rb') as fd:
                all_episodes += pickle.load(fd)
        return all_episodes

    @staticmethod
    def _convert_frame(f):
        # ID - No change
        ids = f[0, :]
        # Owner - [1, 16] to onehot(1)
        @np.vectorize
        def _owner(x):
            return {1: 0, 16: 1}[x]
        owner = _owner(f[1, :])
        # Type - [48, 73, 105] to onehot(3)
        # NOTE - 1970 = fast zergling

        @np.vectorize
        def _type(x):
            return {48: 0, 73: 1, 105: 2, 1970: 2}[x]

        def _onehot(arr, max):
            return np.eye(max)[arr].T
        type = _onehot(_type(f[2, :]), 3)
        # Health - [0 - 100] to [0 - 1]
        @np.vectorize
        def _health(x):
            return x/100
        health = _health(f[3, :])
        # Sheilds - [0 - 50] to [0 - 0.5]
        sheilds = _health(f[4, :])
        # Orientation - [0 - 6] to onehot(7)
        ori = _onehot(f[5, :], 7)
        # Alive flag
        num_units = f.shape[1]
        alive = np.ones(num_units)
        # X,Y - [20 - 47] to [-1 - 1]
        @np.vectorize
        def _coord(x):
            return (x-33.5)/13.5
        x = _coord(f[6, :])
        y = _coord(f[7, :])
        c = np.vstack((owner[np.newaxis, :],
                       type,
                       health[np.newaxis, :],
                       sheilds[np.newaxis, :],
                       ori,
                       alive[np.newaxis, :],
                       x[np.newaxis, :],
                       y[np.newaxis, :]))
        Frame = namedtuple('Frame', ['tags', 'data'])
        return Frame(ids, c)

    def calc_dataset(self, frame_skip, num_workers=1, worker_id=0):
        all_entries = []
        worker_episodes = itertools.islice(self.raw_episodes, worker_id, None, num_workers)
        for ep_idx, ep in enumerate(worker_episodes):
            for ref_idx in range(0, len(ep), frame_skip):
                all_entries += self.make_entries(ep_idx, ref_idx)
        return all_entries

    def make_entries(self, ep_idx, ref_idx):
        hist_fn = getattr(self, f'_hist_{self.hist_dist}_dist')
        pred_fn = getattr(self, f'_pred_{self.pred_dist}_dist')
        hist_idxs = hist_fn(ep_idx, ref_idx)
        pred_idxs = pred_fn(ep_idx, ref_idx)
        hist_ts = [h-ref_idx for h in hist_idxs]
        pred_ts = [p-ref_idx for p in pred_idxs]
        if len(hist_idxs) > 0 and len(pred_idxs) > 0:
            e = Entry(ep_idx, hist_idxs, hist_ts, pred_ts, pred_idxs)
            return [e]
        else:
            return []

    def _hist_fixed_dist(self, ep_idx, ref_idx):
        ep_len = len(self.raw_episodes[ep_idx])
        idx_in_bounds = lambda i: 0 <= i < ep_len
        prev_ts = self.hist_dist_args['ts'] if self.hist_dist_args else range(0, -self.max_hist-1, -1)
        hist_idxs = (ref_idx+t for t in prev_ts)
        hist_idxs = [i for i in hist_idxs if idx_in_bounds(i)]
        return hist_idxs

    def _hist_uniform_dist(self, ep_idx, ref_idx):
        return self._hist_norm_dist(ep_idx, ref_idx, None)

    def _hist_tail_dist(self, ep_idx, ref_idx):
        return self._hist_norm_dist(ep_idx, ref_idx, self.hist_tail_probs)

    def _hist_norm_dist(self, ep_idx, ref_idx, ps):
        ep_len = len(self.raw_episodes[ep_idx])
        idx_in_bounds = lambda i: 0 <= i < ep_len
        max_ = self.hist_dist_args['max']
        prev_ts = list(range(0, -max_-1, -1))
        cand_idxs = [ref_idx+t for t in prev_ts]
        sel_idxs = np.random.choice(cand_idxs, size=self.num_pred, replace=False, p=ps)
        valid_idxs = [i for i in sel_idxs if idx_in_bounds(i)]
        return valid_idxs

    def _pred_fixed_dist(self, ep_idx, ref_idx):
        ep_len = len(self.raw_episodes[ep_idx])
        idx_in_bounds = lambda i: 0 <= i < ep_len
        pred_ts = self.pred_dist_args['ts'] if self.pred_dist_args else [0, 1, 3, 5]
        assert self.num_pred == len(pred_ts), "Hacked target times broke"
        pred_idxs = (ref_idx+t for t in pred_ts)
        pred_idxs = [i for i in pred_idxs if idx_in_bounds(i)]
        return pred_idxs

    def _pred_uniform_dist(self, ep_idx, ref_idx):
        return self._pred_norm_dist(ep_idx, ref_idx, None)

    def _pred_tail_dist(self, ep_idx, ref_idx):
        return self._pred_norm_dist(ep_idx, ref_idx, self.pred_tail_probs)

    def _pred_norm_dist(self, ep_idx, ref_idx, ps):
        ep_len = len(self.raw_episodes[ep_idx])
        idx_in_bounds = lambda i: 0 <= i < ep_len
        max_ = self.pred_dist_args['max']
        future_ts = list(range(1, max_+1))
        cand_idxs = [ref_idx+t for t in future_ts]
        sel_idxs = np.random.choice(cand_idxs, size=self.num_pred, replace=False, p=ps)
        valid_idxs = [i for i in sel_idxs if idx_in_bounds(i)]
        return valid_idxs

    def _fetch_entry(self, e):
        get_raw_frames = lambda f: (self.raw_episodes[e.ep_idx][i] for i in f)
        to_frames = lambda f: [self._convert_frame(i) for i in f]
        hists = to_frames(get_raw_frames(e.hist_idxs))
        preds = to_frames(get_raw_frames(e.pred_idxs))
        return self._build_item(e.ep_idx, hists, e.hist_ts, preds, e.pred_ts)

    def _build_item(self, ep_idx, hist, ts, preds, pred_ts):
        assert len(hist) == len(ts)
        all_ids = np.concatenate([h.tags for h in hist])
        all_data = np.concatenate([h.data for h in hist], 1).transpose()
        all_ts = np.concatenate([[t]*len(fr.tags) for t, fr in zip(ts, hist)])
        pred_ids = np.sort(np.unique(np.concatenate([h.tags for h in preds])))
        pred_mask = np.stack([np.isin(pred_ids, h.tags) for h in preds])
        pred_dat_size = (len(pred_ts), len(pred_ids), all_data.shape[1])
        pred_data = np.zeros(pred_dat_size)
        # Get pred_data set up - complicated because we have to worry about
        # order here.
        first = hist[np.argmin(ts)]
        for fr_idx, fr in enumerate(preds):
            for unit_idx, id_ in enumerate(pred_ids):
                idx = np.nonzero(id_ == fr.tags)[0]
                if idx.size > 0:
                    row = fr.data[:, idx[0]]
                    pred_data[fr_idx, unit_idx] = row
                else:
                    # The unit is dead! Fill in its state from the first frame
                    # instead
                    idx = np.nonzero(id_ == first.tags)[0]
                    assert len(idx) == 1
                    row = first.data[:, idx[0]]
                    row[4:] = 0  # Zero out non-static data
                    pred_data[fr_idx, unit_idx] = row
        # Make full mask
        d = {'data': all_data,
             'ids': all_ids,
             'ts': all_ts,
             'pred_data': pred_data,
             'pred_ids': pred_ids,
             'pred_ts': pred_ts}
        return d

    def __getitem__(self, idx):
        e = self.all_entries[idx]
        return self._fetch_entry(e)

    def __len__(self):
        return len(self.all_entries)


def parse_frame(frame, feat_idx=None):
    if feat_idx == None:
        feat_idx = frame.dim()-1

    def _get_cols(idx, size):
        i = torch.tensor(range(idx, idx+size), device=frame.device)
        f = frame.index_select(feat_idx, i)
        return f
    feat_idxs = {'owner': (0, 1),
                 'type': (1, 3),
                 'health': (4, 1),
                 'shields': (5, 1),
                 'ori': (6, 7),
                 'alive': (13, 1),
                 'pos': (14, 2)}
    return {feat: _get_cols(*idxs) for feat, idxs in feat_idxs.items()}


def collate(items):
    get_tensors = lambda k: [torch.tensor(i[k]) for i in items]
    padded_data = pad_tensors(get_tensors('data'), [0])
    padded_ids = pad_tensors(get_tensors('ids'), [0])
    padded_ts = pad_tensors(get_tensors('ts'), [0])
    padded_mask = pad_tensors([torch.ones(i['data'].shape[0], dtype=torch.bool) for i in items])
    return {'data': padded_data.float(),
            'ids': padded_ids,
            'ts': padded_ts.float(),
            'mask': padded_mask,
            **collate_pred(items)}


def collate_pred(items):
    get_tensors = lambda k: [torch.tensor(i[k]) for i in items]
    padded_pred_data = pad_tensors(get_tensors('pred_data'), [0, 1])
    padded_pred_ids = pad_tensors(get_tensors('pred_ids'), [0])
    padded_pred_ids_mask = pad_tensors([torch.ones(len(i['pred_ids']), dtype=torch.bool) for i in items])
    padded_pred_ts = pad_tensors(get_tensors('pred_ts'), [0])
    padded_pred_ts_mask = pad_tensors([torch.ones(len(i['pred_ts']), dtype=torch.bool) for i in items])
    return {'pred_data': padded_pred_data.float(),
            'pred_ids': padded_pred_ids,
            'pred_ts': padded_pred_ts.float(),
            'pred_ts_mask': padded_pred_ts_mask,
            'pred_ids_mask': padded_pred_ids_mask}
