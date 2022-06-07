from .common import pad_tensors

import random
import datetime
import pickle
import functools

from pathlib import Path
from itertools import islice, product

import pandas
import pyproj

import torch
from torch.utils.data import Dataset


ALL_DATA_COLS = ['RELH', 'TAIR', 'WSPD', 'WVEC', 'WDIR', 'WDSD', 'WSSD',
                 'WMAX', 'PRES', 'SRAD', 'TA9M', 'WS2M']
# Removed TS/TB sensors becasue they're frequently -9
ALL_DATA_QS = [f'Q{col}' for col in ALL_DATA_COLS]

STATION_METADATA = ['elev', 'WCR05', 'WCS05', 'A05', 'N05', 'BULK5', 'GRAV5',
                    'SAND5', 'SILT5', 'CLAY5', 'WCR10', 'WCS10', 'A25', 'N25',
                    'BULK25', 'GRAV25', 'SAND25', 'SILT25', 'CLAY25']
POSITION_METADATA = ['x_m', 'y_m', 'elev']


@functools.lru_cache(maxsize=None)
def _parse_time_str(date, time):
    return datetime.datetime.strptime(f"{date} {time}", "%Y-%m-%d %H%M")


_mins = lambda x: datetime.timedelta(minutes=x)


class WeatherDataset:
    def __init__(self, base, target_cols=['RELH', 'TAIR', 'WSPD', 'PRES'],
                 time_off=5, hist_count=12,
                 sample_count=1, seed=1337,
                 load_train=True, load_test=True,
                 drop=None,
                 HACK_SNIP=None):
        assert time_off % 5 == 0 and time_off > 0
        self.target_cols = target_cols
        self.seed = seed
        self.time_off = _mins(time_off)
        self.hist_count = hist_count
        self.sample_count = sample_count
        self.drop = drop
        self.HACK_SNIP = HACK_SNIP

        base = Path(base)
        self._load_metadata(base/'stationMetadata.csv')
        self._project_latlon()
        self._load_data(base/'All_station_2008.csv')
        valid_stations = self._find_valid_stations()
        # Figure out which are train and which are test stations
        train_stations, self.test_stations = self._split_stations(valid_stations, 0.1)
        # Calculate indices for items
        if load_train:
            self.train_idxs = self._calc_train_keys(train_stations, len(self.test_stations))
            self.train_dataset = _TrainView(self)
        if load_test:
            self.test_idxs = self._calc_test_keys(train_stations, self.test_stations)
            self.test_dataset = _TestView(self)

    def _load_metadata(self, path):
        """ Reads the station metadata file from `path` into `self.raw_metadta`
        """
        print(f'Loading metadata @ {path}...')
        df = pandas.read_csv(path, index_col='stid')
        self.raw_metadata = df

    def _project_latlon(self):
        """ Add X and Y coordinates (in meters) to the station metadata.
        Lat/lon are bad at representing relative distances. Instead, use a map
        projection optimized to preserve distances at the area of interest.
        """
        utm = pyproj.CRS.from_epsg(3721)  # UTM 14N - Covers most of Oklahoma
        tf = pyproj.Transformer.from_crs(utm.geodetic_crs, utm)
        latlon = self.raw_metadata[['nlat', 'elon']]
        proj_row = lambda r: pandas.Series(tf.transform(*r), index=['x_m', 'y_m'])
        xy = latlon.apply(proj_row, axis=1)
        xy -= xy.mean()  # Recenter so mean position is 0 (note: only for numeric precision)
        self.raw_metadata = pandas.concat([self.raw_metadata, xy], axis=1)

    def _load_data(self, path, cache=True):
        """ Reads measurements CSV from `path` into `self.raw_data`
        This file can take a while to parse, so it can be cached for quick
        loading.
        """
        print(f'Loading data @ {path}...')
        cache_path = Path(str(path)+'.cache')
        if cache and cache_path.exists():
            print('Loading from cache')
            with cache_path.open('rb') as fd:
                self.raw_data = pickle.load(fd)
        else:
            df = pandas.read_csv(path)
            idxs = ['STID', 'Year', 'Month', 'Day', 'Time']
            df.set_index(idxs, inplace=True)
            if cache:
                with cache_path.open('wb') as fd:
                    pickle.dump(df, fd)
            self.raw_data = df
        # Drop idxs if requested
        if self.drop:
            idxs = list(self.raw_data.index)
            random.shuffle(idxs)
            count = int(len(idxs)*(1-self.drop))
            self.allowed_raw_idxs = set(idxs[:count])
        self.raw_data = self.raw_data.sort_index()

    def _calc_train_keys(self, stations, num_target_stations):
        """ Precalculate a list of all keys that can be used to quickly fetch &
        construct each training item.
        Since this function just works with training data, we have to do random
        train/target station splits manually to generate training examples.
        """
        print('Calculating train indexes...')
        keys = []
        periods = self._calc_periods()
        if self.HACK_SNIP:
            random.shuffle(periods)
            periods = islice(periods, self.HACK_SNIP)
        for hist_times, target_times in periods:
            for _ in range(self.sample_count):
                lcl_stat = list(stations)
                random.shuffle(lcl_stat)
                target_stations, hist_stations = \
                    lcl_stat[:num_target_stations], lcl_stat[num_target_stations:]
                keys += self._make_keys(hist_stations, target_stations,
                                        hist_times, target_times)
        return keys

    def _calc_test_keys(self, hist_stations, target_stations):
        """ Same as `_calc_train_keys`. The train/target split is provided. """
        print('Calculating test indexes...')
        keys = []
        periods = self._calc_periods(step=30)
        if self.HACK_SNIP:
            random.shuffle(periods)
            periods = islice(periods, self.HACK_SNIP)
        for hist_times, target_times in periods:
            keys += self._make_keys(hist_stations, target_stations, hist_times,
                                    target_times)
        return keys

    def _make_keys(self, hist_stations, target_stations, hist_times,
                   target_times):
        hist_keys = self._fetch_rows(hist_stations, hist_times)
        prevtarget_times = [t for t in hist_times if t not in target_times]
        prevtarget_keys = self._fetch_rows(target_stations, prevtarget_times)
        input_keys = hist_keys+prevtarget_keys
        input_keys = self._filter_quality(input_keys, ALL_DATA_COLS)
        input_stations = hist_stations+target_stations
        target_keys = self._fetch_rows(target_stations, target_times, drop_rows=False)
        target_keys = self._filter_quality(target_keys, self.target_cols)
        abs_times = [self._calc_time_key(t) for t in input_keys]
        last_time = max(abs_times)
        input_times = [(last_time-t).seconds//60 for t in abs_times]
        # We can deal with missing entries in hist, but missing targets
        # are harder. Just discard the instance in that case.
        if len(target_keys) == len(target_stations)*len(target_times):
            result = [(input_stations, input_times, input_keys,
                      target_stations, target_keys)]
            return result
        else:
            return []

    def _filter_quality(self, keys, cols):
        q = self._get_row_qual(keys, cols)
        bad_quals = [1, 2, 3, 8, 9, -9]
        is_bad = (q.isin(bad_quals)).any(axis=1)
        good = is_bad[~is_bad].index
        return list(good)

    def _fetch_rows(self, stations, times, drop_rows=True):
        raw_times = [self._time_to_tuple(t) for t in times]
        keys = [(a,)+b for a, b in product(stations, raw_times)]
        present_keys = [k for k in keys if k in self.raw_data.index]
        if self.drop and drop_rows:
            present_keys = [k for k in keys if k in self.allowed_raw_idxs]
        return present_keys

    @staticmethod
    def _time_to_tuple(time):
        clock = time.strftime("%H%M")
        return (time.year, time.month, time.day, int(clock))

    def _calc_periods(self, step=5):
        min_step = step*_mins(5)
        first = self.raw_data.iloc[0]
        last = self.raw_data.iloc[-1]
        time = self._calc_time(first)
        end_time = self._calc_time(last)
        periods = []
        while True:
            period = self._calc_period(time)
            _, tgt_ts = period
            last_target = tgt_ts[-1]
            if last_target > end_time:
                break
            periods.append(period)
            time += min_step
        return periods

    def _calc_period(self, start):
        # Make hist
        hist_times = [start+i*self.time_off for i in range(self.hist_count)]
        end = hist_times[-1]
        target_times = [end]
        return hist_times, target_times

    @staticmethod
    def _calc_time(row):
        date_str = row['Date']
        time = row.name[-1]
        time_str = '{:04d}'.format(time)
        return _parse_time_str(date_str, time_str)

    @staticmethod
    def _calc_time_key(key):
        _, yr, mth, day, time_num = key
        date_str = f"{yr}-{mth}-{day}"
        time_str = f"{time_num:04d}"
        return _parse_time_str(date_str, time_str)

    def _find_valid_stations(self):
        all_stations = {s[0] for s in self.raw_data.index}
        is_valid_md = lambda s: (self.raw_metadata.loc[s, STATION_METADATA] != -999).all()
        valid_metadata = [s for s in all_stations if is_valid_md(s)]
        return valid_metadata

    def _split_stations(self, stations, pct):
        assert 0 < pct < 1
        all_stats = sorted(list(stations))
        test_count = int(pct*len(all_stats))
        random.seed(self.seed)
        random.shuffle(all_stats)
        test, train = all_stats[:test_count], all_stats[test_count:]
        return train, test

    @staticmethod
    def _len(idxs):
        return len(idxs)

    def _getitem(self, idxs):
        stations, hist_times, hist_idxs, target_stations, target_idxs = idxs
        hist_data = self._get_row_data(hist_idxs)
        hist_q = self._get_row_qual(hist_idxs)
        target_data = self._get_row_data(target_idxs, self.target_cols)
        target_q = self._get_row_qual(target_idxs, self.target_cols)
        station_meta = self._get_metadata(stations)
        station_pos = self._get_station_position(stations)
        target_pos = self._get_station_position(target_stations)
        hist_stations = pandas.Series(i[0] for i in hist_data.index)
        station_idxs = hist_stations.map(station_meta.index.get_loc)
        tgt_station_idxs = pandas.Series(target_stations).map(station_meta.index.get_loc)
        return {'hist': hist_data,
                'station_metadata': station_meta,
                'station_pos': station_pos,
                'station_idxs': station_idxs,
                'times': hist_times,
                'target_pos': target_pos,
                'target': target_data,
                'tgt_station_idxs': tgt_station_idxs,
                'hist_q': hist_q,
                'target_q': target_q}

    def _get_row_data(self, row_keys, cols=None):
        rows = self.raw_data.loc[row_keys]
        cols = cols or ALL_DATA_COLS
        return rows[cols]

    def _get_row_qual(self, row_keys, cols=None):
        rows = self.raw_data.loc[row_keys]
        if cols is None:
            cols = ALL_DATA_QS
        else:
            cols = [f'Q{c}' for c in cols]
        return rows[cols]

    def _get_metadata(self, stations):
        rows = self.raw_metadata.loc[stations]
        cols = STATION_METADATA
        return rows[cols]

    def _get_station_position(self, stations):
        rows = self.raw_metadata.loc[stations]
        cols = POSITION_METADATA
        return rows[cols]

    def target_ranges(self):
        if not hasattr(self, '_target_ranges'):
            self._target_ranges = self._calc_target_ranges()
        return self._target_ranges

    def _calc_target_ranges(self):
        target_data = self.raw_data[self.target_cols]
        low = target_data.quantile(0.1)
        high = target_data.quantile(0.9)
        return low, high


class _TestView(Dataset):
    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return self.ds._len(self.ds.test_idxs)

    def __getitem__(self, i):
        return self.ds._getitem(self.ds.test_idxs[i])


class _TrainView(Dataset):
    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return self.ds._len(self.ds.train_idxs)

    def __getitem__(self, i):
        return self.ds._getitem(self.ds.train_idxs[i])


def make_mask(hists):
    masks = []
    for h in hists:
        count = h.size(0)
        masks.append(torch.ones(count))
    return masks


def collate(items):
    prep = lambda x: x.to_numpy() if type(x) == pandas.DataFrame else x
    get_tensors = lambda k: [torch.tensor(prep(i[k])) for i in items]
    hist = pad_tensors(get_tensors('hist'), [0])
    hist_q = pad_tensors(get_tensors('hist_q'), [0])
    mask = pad_tensors(make_mask(get_tensors('hist')))
    station_metadata = pad_tensors(get_tensors('station_metadata'), [0])
    station_pos = pad_tensors(get_tensors('station_pos'), [0])
    station_idxs = pad_tensors(get_tensors('station_idxs'))
    tgt_station_idxs = pad_tensors(get_tensors('tgt_station_idxs'))
    times = pad_tensors(get_tensors('times'))
    target = torch.stack(get_tensors('target'))
    target_q = torch.stack(get_tensors('target_q'))
    target_pos = torch.stack(get_tensors('target_pos'))
    result = {'hist': hist.float(),
              'hist_q': hist_q.float(),
              'hist_mask': mask.bool(),
              'station_metadata': station_metadata.float(),
              'station_pos': station_pos.float(),
              'station_idxs': station_idxs.long(),
              'tgt_station_idxs': tgt_station_idxs.long(),
              'times': times.float(),
              'target_pos': target_pos.float(),
              'target': target.float(),
              'target_q': target_q.float()}
    if 'is_anom' in items[0]:
        anom = pad_tensors(get_tensors('is_anom'))
        result['is_anom'] = anom.bool()
    return result
