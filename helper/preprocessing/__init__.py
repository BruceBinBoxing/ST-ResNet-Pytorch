import pandas as pd
import numpy as np
from copy import copy
import time
# from temporal_contrast_normalization import TemporalConstrastNormalization
# from personal_temporal_contrast_normalization import PersonalTemporalConstrastNormalization
from .minmax_normalization import MinMaxNormalization
from ..utils import string2timestamp


def timestamp2vec(timestamps):
    # tm_wday range [0, 6], Monday is 0
    # vec = [time.strptime(str(t[:8], encoding='utf-8'), '%Y%m%d').tm_wday for t in timestamps]  # python3
    vec = [time.strptime(str(t[:8], encoding='utf-8'), '%Y%m%d').tm_wday for t in timestamps]  # python2
    ret = []
    for i in vec:
        v = [0 for _ in range(7)]
        v[i] = 1
        if i >= 5: 
            v.append(0)  # weekend
        else:
            v.append(1)  # weekday
        ret.append(v)
    return np.asarray(ret)


def remove_incomplete_days(data, timestamps, T=48):
    # remove a certain day which has not 48 timestamps
    days = []  # available days: some day only contain some seqs
    days_incomplete = []
    i = 0
    while i < len(timestamps):
        if int(timestamps[i][8:]) != 1:
            i += 1
        elif i+T-1 < len(timestamps) and int(timestamps[i+T-1][8:]) == T:
            days.append(timestamps[i][:8])
            i += T
        else:
            days_incomplete.append(timestamps[i][:8])
            i += 1
    print("incomplete days: ", days_incomplete)
    days = set(days)
    idx = []
    for i, t in enumerate(timestamps):
        if t[:8] in days:
            idx.append(i)

    data = data[idx]
    timestamps = [timestamps[i] for i in idx]
    return data, timestamps


def split_by_time(data, timestamps, split_timestamp):
    # divide data into two subsets:
    # e.g., Train: ~ 2015.06.21 & Test: 2015.06.22 ~ 2015.06.28
    assert(len(data) == len(timestamps))
    assert(split_timestamp in set(timestamps))

    data_1 = []
    timestamps_1 = []
    data_2 = []
    timestamps_2 = []
    switch = False
    for t, d in zip(timestamps, data):
        if split_timestamp == t:
            switch = True
        if switch is False:
            data_1.append(d)
            timestamps_1.append(t)
        else:
            data_2.append(d)
            timestamps_2.append(t)
    return (np.asarray(data_1), timestamps_1), (np.asarray(data_2), timestamps_2)


def timeseries2seqs(data, timestamps, length=3, T=48):
    raw_ts = copy(timestamps)
    if type(timestamps[0]) != pd.Timestamp:
        timestamps = string2timestamp(timestamps, T=T)

    offset = pd.DateOffset(minutes=24 * 60 // T)

    breakpoints = [0]
    for i in range(1, len(timestamps)):
        if timestamps[i-1] + offset != timestamps[i]:
            print(timestamps[i-1], timestamps[i], raw_ts[i-1], raw_ts[i])
            breakpoints.append(i)
    breakpoints.append(len(timestamps))
    X = []
    Y = []
    for b in range(1, len(breakpoints)):
        print('breakpoints: ', breakpoints[b-1], breakpoints[b])
        idx = range(breakpoints[b-1], breakpoints[b])
        for i in range(len(idx) - length):
            x = np.vstack(data[idx[i:i+length]])
            y = data[idx[i+length]]
            X.append(x)
            Y.append(y)
    X = np.asarray(X)
    Y = np.asarray(Y)
    print("X shape: ", X.shape, "Y shape:", Y.shape)
    return X, Y

def timeseries2seqs_meta(data, timestamps, length=3, T=48):
    raw_ts = copy(timestamps)
    if type(timestamps[0]) != pd.Timestamp:
        timestamps = string2timestamp(timestamps, T=T)

    offset = pd.DateOffset(minutes=24 * 60 // T)

    breakpoints = [0]
    for i in range(1, len(timestamps)):
        if timestamps[i-1] + offset != timestamps[i]:
            print(timestamps[i-1], timestamps[i], raw_ts[i-1], raw_ts[i])
            breakpoints.append(i)
    breakpoints.append(len(timestamps))
    X = []
    Y = []
    avail_timestamps = []
    for b in range(1, len(breakpoints)):
        print('breakpoints: ', breakpoints[b-1], breakpoints[b])
        idx = range(breakpoints[b-1], breakpoints[b])
        for i in range(len(idx) - length):
            avail_timestamps.append(raw_ts[idx[i+length]])
            x = np.vstack(data[idx[i:i+length]])
            y = data[idx[i+length]]
            X.append(x)
            Y.append(y)
    X = np.asarray(X)
    Y = np.asarray(Y)
    print("X shape: ", X.shape, "Y shape:", Y.shape)
    return X, Y, avail_timestamps


def timeseries2seqs_peroid_trend(data, timestamps, length=3, T=48, peroid=pd.DateOffset(days=7), peroid_len=2):
    raw_ts = copy(timestamps)
    if type(timestamps[0]) != pd.Timestamp:
        timestamps = string2timestamp(timestamps, T=T)

    # timestamps index
    timestamp_idx = dict()
    for i, t in enumerate(timestamps):
        timestamp_idx[t] = i

    offset = pd.DateOffset(minutes=24 * 60 // T)

    breakpoints = [0]
    for i in range(1, len(timestamps)):
        if timestamps[i-1] + offset != timestamps[i]:
            print(timestamps[i-1], timestamps[i], raw_ts[i-1], raw_ts[i])
            breakpoints.append(i)
    breakpoints.append(len(timestamps))
    X = []
    Y = []
    for b in range(1, len(breakpoints)):
        print('breakpoints: ', breakpoints[b-1], breakpoints[b])
        idx = range(breakpoints[b-1], breakpoints[b])
        for i in range(len(idx) - length):
            # period
            target_timestamp = timestamps[i+length]

            legal_idx = []
            for pi in range(1, 1+peroid_len):
                if target_timestamp - peroid * pi not in timestamp_idx:
                    break
                legal_idx.append(timestamp_idx[target_timestamp - peroid * pi])
            # print("len: ", len(legal_idx), peroid_len)
            if len(legal_idx) != peroid_len:
                continue

            legal_idx += idx[i:i+length]

            # trend
            x = np.vstack(data[legal_idx])
            y = data[idx[i+length]]
            X.append(x)
            Y.append(y)
    X = np.asarray(X)
    Y = np.asarray(Y)
    print("X shape: ", X.shape, "Y shape:", Y.shape)
    return X, Y


def timeseries2seqs_3D(data, timestamps, length=3, T=48):
    raw_ts = copy(timestamps)
    if type(timestamps[0]) != pd.Timestamp:
        timestamps = string2timestamp(timestamps, T=T)

    offset = pd.DateOffset(minutes=24 * 60 // T)

    breakpoints = [0]
    for i in range(1, len(timestamps)):
        if timestamps[i-1] + offset != timestamps[i]:
            print(timestamps[i-1], timestamps[i], raw_ts[i-1], raw_ts[i])
            breakpoints.append(i)
    breakpoints.append(len(timestamps))
    X = []
    Y = []
    for b in range(1, len(breakpoints)):
        print('breakpoints: ', breakpoints[b-1], breakpoints[b])
        idx = range(breakpoints[b-1], breakpoints[b])
        for i in range(len(idx) - length):
            x = data[idx[i:i+length]].reshape(-1, length, 32, 32)
            y = np.asarray([data[idx[i+length]]]).reshape(-1, 1, 32, 32)
            X.append(x)
            Y.append(y)
    X = np.asarray(X)
    Y = np.asarray(Y)
    print("X shape: ", X.shape, "Y shape:", Y.shape)
    return X, Y


def bug_timeseries2seqs(data, timestamps, length=3, T=48):
    # have a bug
    if type(timestamps[0]) != pd.Timestamp:
        timestamps = string2timestamp(timestamps, T=T)

    offset = pd.DateOffset(minutes=24 * 60 // T)

    breakpoints = [0]
    for i in range(1, len(timestamps)):
        if timestamps[i-1] + offset != timestamps[i]:
            breakpoints.append(i)
    X = []
    Y = []
    for b in range(1, len(breakpoints)):
        print('breakpoints: ', breakpoints[b-1], breakpoints[b])
        idx = range(breakpoints[b-1], breakpoints[b])
        for i in range(len(idx) - 3):
            x = np.vstack(data[idx[i:i+3]])
            y = data[idx[i+3]]
            X.append(x)
            Y.append(y)
    X = np.asarray(X)
    Y = np.asarray(Y)
    print("X shape: ", X.shape, "Y shape:", Y.shape)
    return X, Y
