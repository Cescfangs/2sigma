import pickle
import pandas as pd
import numpy as np
from datetime import timedelta, datetime


def construct_data0(data, skip=6):
    volume_cols = ['volume_' + str(i) for i in range(skip)]
    vol_diff_cols = ['vol_diff_' + str(i) for i in range(skip)]
    other_cols = ['date_time', 'rain_level', 'holiday', 'weekdays', 'volume']

    i = 0
    array = []
    while i < data.shape[0]:
        row = []
        volume = data[i: i + skip, 0].sum()
        weekday = data[i, 1]
        holiday = data[i, 2]
        rain = data[i: i + skip, 3].mean()
        date_time = data[i, 5]
        row.extend([date_time, rain, holiday, weekday, volume])

        for j in range(skip):
            row.append(data[i + j, 0])
        for j in range(skip):
            row.append(data[i + j, 4])
        array.append(row)
        i += skip
    df = pd.DataFrame(data=np.array(array), columns=[other_cols + volume_cols + vol_diff_cols])
    df[df.columns[1:]].astype(int, inplace=True)
    # for col in df.columns[1:]:
    # df[col] = df[col].astype(int)
    df['date_time'] = pd.to_datetime(df.date_time)
    return df


test = pd.read_csv('../dataSets/testing_phase1/test1_features1.csv')
f = open('../etr.model', 'rb')
models = pickle.load(f)
f.close()
features = ['volume',
            'volume_0',
            'volume_1',
            'volume_2',
            'volume_3',
            'volume_4',
            'volume_5',
            'vol_diff_0',
            'vol_diff_1',
            'vol_diff_2',
            'vol_diff_3',
            'vol_diff_4',
            'vol_diff_5',
            'rain_level',
            'holiday',
            'weekdays']

pred = []
for case, model in models:
    tol, direc, time = case
    print('tol:', tol, '; direc:', direc, '; time:', time)
    df = test.loc[(test.tollgate_id == tol) & (test.direction == direc),
                  ['volume', 'weekdays', 'holidays', 'rain', 'volume_diff', 'date_time', 'start_time']]
    date = df.date_time.values[0]
    if time:
        df = df.loc[df.start_time >= 15]
        df = df[df.start_time < 17]
    else:
        df = df.loc[df.start_time >= 6]
        df = df[df.start_time < 8]

    df = construct_data0(
        np.array(df[['volume', 'weekdays', 'holidays', 'rain', 'volume_diff', 'date_time']]))
    print('samples:', len(df))
    pred.append(((tol, direc, time), model.predict(df[features])))

test_sub = pd.DataFrame(columns=['tollgate_id', 'date_time', 'direction', 'ampm', 'volume'])

day0 = datetime.strptime('2016-10-18', '%Y-%m-%d')
# print(type(day0))
for case, res in pred:
    print(case)
    tol, direc, time = case
    for ind, vol in enumerate(res):
        day = day0 + timedelta(days=ind)
        for p in range(6):
            time0 = day + timedelta(hours=17) if time else day + timedelta(hours=8)
            time_p = time0 + timedelta(minutes=p * 20)
            vol_p = vol * dist[(tol, direc, time, day.weekday())][p]
            #             print(vol_p)
            test_sub = test_sub.append({'tollgate_id': tol, 'date_time': time_p, 'direction': direc,
                                        'volume': vol_p,
                                        'ampm': time}, ignore_index=True)

test_sub.drop('ampm', axis=1, inplace=True)
test_sub['time_window'] = test_sub.date_time.apply(lambda x: '[' + str(x) + ',' + str(x + timedelta(minutes=20)) + ')')
test_sub[['tollgate_id', 'direction']] = test_sub[['tollgate_id', 'direction']].astype(int)
test_sub[['tollgate_id', 'time_window', 'direction', 'volume']].to_csv('../results/vol_submit_-1.csv', index=False)
