from __future__ import print_function
import pandas as pd
from datetime import datetime, timedelta
import time
import os


def timestamp_str_new(cur_timestampes, T=48):
    os.environ['TZ'] = 'Asia/Shanghai'
    # print cur_timestampes
    if '-' in cur_timestampes[0]:
        return cur_timestampes
    ret = []
    for v in cur_timestampes:
        '''TODO
        Bug here
        '''
        cur_sec = time.mktime(time.strptime("%04i-%02i-%02i" % (int(v[:4]), int(v[4:6]), int(v[6:8])), "%Y-%m-%d")) + (int(v[8:]) * 24. * 60 * 60 // T)
        curr = time.localtime(cur_sec)
        if v == "20151101288" or v == "2015110124":
            print(v, time.strftime("%Y-%m-%d-%H-%M", curr), time.localtime(cur_sec), time.localtime(cur_sec - (int(v[8:]) * 24. * 60 * 60 // T)), time.localtime(cur_sec - (int(v[8:]) * 24. * 60 * 60 // T) + 3600 * 25))
        ret.append(time.strftime("%Y-%m-%d-%H-%M", curr))
    return ret


def string2timestamp_future(strings, T=48):
    strings = timestamp_str_new(strings, T)
    timestamps = []
    for v in strings:
        year, month, day, hour, tm_min = [int(z) for z in v.split('-')]
        timestamps.append(pd.Timestamp(datetime(year, month, day, hour, tm_min)))

    return timestamps


def string2timestamp(strings, T=48):
    timestamps = []

    time_per_slot = 24.0 / T
    num_per_T = T // 24
    for t in strings:
        year, month, day, slot = int(t[:4]), int(t[4:6]), int(t[6:8]), int(t[8:])-1
        timestamps.append(pd.Timestamp(datetime(year, month, day, hour=int(slot * time_per_slot), minute=(slot % num_per_T) * int(60.0 * time_per_slot))))

    return timestamps


def timestamp2string(timestamps, T=48):
    # timestamps = timestamp_str_new(timestamps)
    num_per_T = T // 24
    return ["%s%02i" % (ts.strftime('%Y%m%d'),
                        int(1+ts.to_datetime().hour*num_per_T+ts.to_datetime().minute/(60 // num_per_T))) for ts in timestamps]
    # int(1+ts.to_datetime().hour*2+ts.to_datetime().minute/30)) for ts in timestamps]
