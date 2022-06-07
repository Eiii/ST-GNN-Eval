import pandas
import csv
import pyproj
import math
import torch

from pathlib import Path
from collections import namedtuple
from itertools import product
from functools import lru_cache, partial

from .common import pad_tensors

METREntry = namedtuple('METREntry', ('hist_rows', 'target_rows', 'center'))


class METRDataset:
    def __init__(self, base, date_fn, normalize):
        self.normalize = normalize
        self._load_metr(Path(base))
        self._load_adj(Path(base))
        self._calc_entries(date_fn)

    def __getitem__(self, idx):
        e = self.entries[idx]
        return self._assemble_entry(e)

    def __len__(self):
        return len(self.entries)

    def _load_metr(self, base):
        self.raw_df = pandas.read_hdf(base/'metr-la.h5')
        self._normalize()
        self._time_to_offset()
        self.meta_df = pandas.read_csv(base/'graph_sensor_locations.csv')
        self._latlon_to_xy()
        self.meta_df.set_index(self.meta_df.sensor_id, inplace=True)
        self.meta_df.drop(columns=['index', 'sensor_id'], inplace=True)

    def _normalize(self):
        mean = self.raw_df.values.mean()
        std = self.raw_df.values.std()
        self.norm_info = (mean, std)
        if self.normalize:
            self.raw_df = (self.raw_df-mean)/std

    def _load_adj(self, base):
        path = base/'distances_la_2012.csv'
        with path.open('r') as fd:
            reader = csv.reader(fd)
            data = list(reader)
        # from, to, cost
        data = [(int(x), int(y), float(z)) for x, y, z in data[1:]]
        data = {(from_, to): cost for from_, to, cost in data}
        sensor_ids = list(self.meta_df.index)
        self.sensor_id_to_idx = {id_: idx for idx, id_ in enumerate(sensor_ids)}
        num_sensors = len(sensor_ids)
        sensor_cost = torch.zeros(num_sensors, num_sensors, dtype=torch.float)
        sensor_connected = torch.zeros(num_sensors, num_sensors, dtype=torch.bool)
        for (from_, to), cost in data.items():
            if not (from_ in sensor_ids and to in sensor_ids):
                continue
            from_idx = sensor_ids.index(from_)
            to_idx = sensor_ids.index(to)
            sensor_connected[from_idx, to_idx] = True
            sensor_cost[from_idx, to_idx] = cost
        self.sensor_connected = sensor_connected
        self.sensor_cost = sensor_cost

    def _filter_rows(self, fn):
        if fn is not None:
            valid_idx = [x for x in self.raw_df.index if fn(x)]
            self.raw_df = self.raw_df.loc[valid_idx]

    def _time_to_offset(self):
        start_time = min(self.raw_df.index)
        self.freq = start_time.freq
        to_offset = lambda x: int((x-start_time)/self.freq)
        new_index = self.raw_df.index.map(to_offset)
        self.raw_times = pandas.Series(self.raw_df.index, index=new_index)
        self.raw_df.index = new_index

    def _latlon_to_xy(self):
        utm = pyproj.CRS.from_epsg(3741)
        tf = pyproj.Transformer.from_crs(utm.geodetic_crs, utm)
        proj_row = lambda r: pandas.Series(tf.transform(*r), index=['x_m', 'y_m'])
        latlon = self.meta_df[['latitude', 'longitude']]
        xy = latlon.apply(proj_row, axis=1)
        xy -= xy.mean()
        self.meta_df[['x', 'y']] = xy

    def _calc_entries(self, filter_fn):
        entries = []
        for center, row in self.raw_df.iterrows():
            if not filter_fn(self.raw_times[center]):
                continue
            hists = self._calc_hist(center)
            tgts = self._calc_target(center)
            entry = METREntry(hists, tgts, center)
            if self._validate_entry(entry):
                entries.append(entry)
        self.entries = entries

    def _calc_hist(self, base):
        hist_amt = 12
        cand = range(base-hist_amt, base)
        valid = [x for x in cand if x in self.raw_df.index]
        return valid

    def _calc_target(self, base):
        off = 12
        tgt_amt = 1
        cand = range(base+off, base+off+tgt_amt)
        valid = [x for x in cand if x in self.raw_df.index]
        return valid

    def _assemble_entry(self, entry):
        # Locations
        # Times
        to_dt = lambda l: [(x-entry.center)*self.freq.n for x in l]
        hist_dt = to_dt(entry.hist_rows)
        tgt_dt = to_dt(entry.target_rows)
        period_times = self._period_encode(entry.hist_rows)
        # Values
        hist_table = self._get_rows(entry.hist_rows)
        tgt_table = self._get_rows(entry.target_rows)
        #
        hist_t, hist_ids, hist_pos, hist_data, hist_period_times = self._from_tables(hist_dt, hist_table, period_times)
        tgt_t, tgt_ids, tgt_pos, tgt_data, _ = self._from_tables(tgt_dt, tgt_table, None)
        hist_idx_ids = self.ids_to_idx(hist_ids)
        tgt_idx_ids = self.ids_to_idx(tgt_ids)
        id_dist = self.sensor_cost
        id_adj = self.sensor_connected
        return {'hist_t': hist_t,
                'hist_id': hist_idx_ids,
                'hist_pos': hist_pos,
                'hist_data': hist_data,
                'hist_period_times': hist_period_times,
                'id_dist': id_dist,
                'id_adj': id_adj,
                'tgt_t': tgt_t,
                'tgt_id': tgt_idx_ids,
                'tgt_pos': tgt_pos,
                'tgt_data': tgt_data,
                'norm_info': self.norm_info
                }

    def _period_encode(self, t):
        hour = 60 // 5
        day = 24 * hour
        week = 7 * day
        ps = torch.tensor([week, week/2, day, day/4, hour])
        ts = torch.tensor(t)
        xs = ts.unsqueeze(1) / ps.unsqueeze(0)
        sins = torch.sin(2*math.pi*xs)
        coss = torch.cos(2*math.pi*xs)
        return torch.cat((sins, coss), dim=-1)

    def ids_to_idx(self, ids):
        return [self.sensor_id_to_idx[i] for i in ids]

    @lru_cache
    def _calc_dist(self, keys_ids, points_ids):
        dist = torch.zeros(len(keys_ids), len(points_ids), dtype=torch.float)
        valid = torch.zeros(len(keys_ids), len(points_ids), dtype=torch.bool)
        ref_ids = self.meta_df.index.tolist()
        for (idx1, id1), (idx2, id2) in product(enumerate(keys_ids), enumerate(points_ids)):
            ref_idx1 = ref_ids.index(id1)
            ref_idx2 = ref_ids.index(id2)
            dist[idx1, idx2] = self.sensor_cost[ref_idx1, ref_idx2]
            valid[idx1, idx2] = self.sensor_connected[ref_idx1, ref_idx2]
        return dist, valid

    def _get_rows(self, rows):
        data = self.raw_df.loc[rows]
        return data

    def _from_tables(self, dt, data, period_times):
        assert len(dt) == len(data)
        all_t = []
        all_ids = []
        all_data = []
        all_pos = []
        all_period_times = []
        for i in range(len(dt)):
            row = data.iloc[i]
            t = dt[i]
            pos = self.meta_df.loc[[int(x) for x in row.index]][['x', 'y']]
            all_data += row.tolist()
            all_pos += pos.values.tolist()
            all_t += [t]*len(row)
            all_ids += [int(x) for x in row.index]
            if period_times is not None:
                all_period_times += [period_times[i]]*len(row)
        if period_times is not None:
            all_period_times = torch.stack(all_period_times)
        return all_t, all_ids, all_pos, all_data, all_period_times

    @staticmethod
    def _validate_entry(e):
        min_hist = 12
        min_tgt = 1
        valid_hist = len(e.hist_rows) >= min_hist
        valid_tgt = len(e.target_rows) >= min_tgt
        return valid_hist and valid_tgt


def get_tensors_full(items, k):
    def ensure_tensor(t):
        if isinstance(t, torch.Tensor):
            return t.clone().detach()
        else:
            return torch.tensor(t)
    return [ensure_tensor(i[k]) for i in items]


def collate(items):
    get_tensors = partial(get_tensors_full, items)
    hist_t = pad_tensors(get_tensors('hist_t'))
    hist_id = pad_tensors(get_tensors('hist_id'))
    hist_pos = pad_tensors(get_tensors('hist_pos'))
    hist_data = pad_tensors(get_tensors('hist_data'))
    hist_period_times = pad_tensors(get_tensors('hist_period_times'))
    id_dist = pad_tensors(get_tensors('id_dist'))
    id_adj = pad_tensors(get_tensors('id_adj'))
    hist_mask = pad_tensors([torch.ones(len(i['hist_t']), dtype=torch.bool) for i in items])
    tgt_t = pad_tensors(get_tensors('tgt_t'))
    tgt_id = pad_tensors(get_tensors('tgt_id'))
    tgt_pos = pad_tensors(get_tensors('tgt_pos'))
    tgt_data = pad_tensors(get_tensors('tgt_data'))
    tgt_mask = pad_tensors([torch.ones(len(i['tgt_t']), dtype=torch.bool) for i in items])
    norm_info = torch.stack(get_tensors('norm_info'))
    return {'hist_t': hist_t,
            'hist_id': hist_id,
            'hist_pos': hist_pos,
            'hist_data': hist_data,
            'hist_period_times': hist_period_times,
            'id_dist': id_dist,
            'id_adj': id_adj,
            'hist_mask': hist_mask,
            'tgt_t': tgt_t,
            'tgt_id': tgt_id,
            'tgt_pos': tgt_pos,
            'tgt_data': tgt_data,
            'tgt_mask': tgt_mask,
            'norm_info': norm_info
            }
