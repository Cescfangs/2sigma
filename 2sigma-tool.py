import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge,
from sklearn.metrics import r2_score


train = pd.read_hdf('train.h5')
test_data = train.loc[train.timestamp > 905]
train = train.loc[train.timestamp <= 905]
# Feature Selection #
null_labels = [
    'technical_21',
    'technical_19',
    'technical_27',
    'technical_36',
    'technical_35',
    'technical_17',
    'technical_43',
    'technical_13',
    'fundamental_33',
    'technical_14',
    'technical_33',
    'fundamental_18',
    'fundamental_48',
    'fundamental_59',
    'technical_9',
    'technical_16',
    'technical_42',
    'technical_18',
    'fundamental_42',
    'fundamental_0',
    'fundamental_7',
    'fundamental_41',
    'technical_41',
    'fundamental_21',
    'fundamental_19',
    'technical_29',
    'technical_24',
    'derived_0',
    'derived_1',
    'fundamental_17',
    'technical_3',
    'fundamental_20',
    'fundamental_32',
    'fundamental_62',
    'fundamental_25',
    'technical_1',
    'fundamental_58',
    'derived_3',
    'technical_5',
    'fundamental_52',
    'technical_10',
    'technical_31',
    'technical_25',
    'technical_44',
    'technical_28',
    'fundamental_40',
    'fundamental_27',
    'fundamental_29',
    'fundamental_43',
    'fundamental_15',
    'fundamental_30',
    'fundamental_60',
    'fundamental_16',
    'fundamental_50',
    'fundamental_44',
    'fundamental_37',
    'fundamental_14',
    'fundamental_23',
    'fundamental_55',
    'fundamental_8',
    'fundamental_63',
    'fundamental_39',
    'fundamental_54',
    'derived_2',
    'derived_4',
    'fundamental_35',
    'fundamental_34',
    'fundamental_47',
    'fundamental_51',
    'fundamental_31',
    'fundamental_49',
    'fundamental_22',
    'fundamental_9',
    'fundamental_24',
    'fundamental_57',
    'fundamental_28',
    'fundamental_61',
    'fundamental_1',
    'fundamental_6',
    'fundamental_38',
    'fundamental_5']

etr_features = ['y_past',
                'tec20-30',
                'technical_30',
                'tec123',
                'technical_43',
                'technical_43_diff',
                'tec123_past',
                'technical_11_diff',
                'technical_2_diff',
                'technical_11',
                'technical_20',
                'technical_2',
                'fundamental_25_nan',
                'technical_14_diff',
                'technical_21_diff',
                'technical_9_nan',
                'technical_40',
                'technical_30_diff',
                'technical_6_diff',
                'technical_6',
                'technical_17_diff',
                'technical_17',
                'technical_14',
                'technical_7',
                'technical_19',
                'technical_44_nan',
                'fundamental_27_nan',
                'technical_18_nan',
                'technical_28_nan',
                'technical_21',
                'technical_42_nan',
                'technical_29_diff',
                'technical_20_diff',
                'technical_31_nan',
                'fundamental_53',
                'technical_24_nan',
                'technical_36',
                'technical_19_diff',
                'technical_27',
                'technical_29',
                'technical_35',
                'technical_22',
                'technical_41_nan',
                'fundamental_8',
                'fundamental_21',
                'fundamental_17_nan',
                'technical_34',
                'technical_16_nan',
                'technical_27_diff',
                'fundamental_33_nan',
                'fundamental_58',
                'derived_1_nan',
                'technical_10',
                'technical_25_nan',
                'fundamental_18',
                'fundamental_59',
                'technical_40_diff',
                'null_count',
                'fundamental_5_nan',
                'fundamental_48',
                'fundamental_47_nan',
                'technical_36_diff',
                'fundamental_41_nan',
                'fundamental_42_nan',
                'fundamental_0_nan',
                'fundamental_50',
                'fundamental_40',
                'technical_3_nan',
                'fundamental_23',
                'fundamental_49_nan',
                'fundamental_36',
                'technical_44',
                'fundamental_2',
                'fundamental_0',
                'technical_41',
                'fundamental_62_diff',
                'technical_38_diff',
                'fundamental_22_nan',
                'technical_12',
                'fundamental_62',
                'technical_37_diff',
                'fundamental_44',
                'technical_29_nan',
                'fundamental_24_nan',
                'technical_10_nan',
                'fundamental_46',
                'technical_1',
                'fundamental_54_nan',
                'fundamental_0_diff',
                'technical_12_diff',
                'technical_35_diff',
                'derived_3_nan',
                'fundamental_63_nan',
                'fundamental_31_nan',
                'fundamental_40_nan',
                'fundamental_35_nan',
                'technical_3',
                'fundamental_13']
seed = 17
excl = ['id', 'timestamp', 'y']
origin_features = [c for c in train.columns if c not in excl]
origin_features_exclude_y = [c for c in train.columns if c not in ['y']]
diff_features = [feature + '_diff' for feature in origin_features]
linear_features = ['technical_20_diff', 'tec20-30']
# End of Feature Selection #
d_mean = train.median(axis=0)

last_stamp = train.loc[train.timestamp == train.timestamp.max(), origin_features_exclude_y]

# add diffs #


def add_diff(data):
    data.sort_values(['id', 'timestamp'], inplace=True)
    data['id_diff'] = data.id.diff()
    for feature in origin_features:
        diff_tag = feature + '_diff'
        data[diff_tag] = data[feature].diff()
        d_mean[diff_tag] = 0
    data.loc[data.id_diff != 0, diff_features] = 0
# end of diffs #


# add Nan tags #
def add_nan(data):
    Nan_counts = data.isnull().sum(axis=1)
    for feature in null_labels:
        data[feature + '_nan'] = pd.isnull(data[feature])
        d_mean[feature + '_nan'] = 0
    data['null_count'] = Nan_counts
# end of Nan tags #


def predict_y_past(x):
    w = np.array([-8.36489105, 9.19544792]).T
    return x.dot(w) - 0.00020021698404804056


ymean_dict = dict(train.groupby(["id"])["y"].median())

print('Processing data...')

train = train
add_diff(train)
add_nan(train)
train = train.fillna(d_mean)
train['tec20-30'] = train.technical_20 - train.technical_30
train['tec123'] = train['tec20-30'] + train.technical_13
train['tec123_past'] = train.tec123.shift()
train['y_past'] = train.y.shift()
train.loc[train.id_diff != 0, ['tec123_past', 'y_past']] = 0

low_y_cut = -0.075
high_y_cut = 0.075
y_above_cut = (train.y > high_y_cut)
y_below_cut = (train.y < low_y_cut)
y_within_cut = (~y_above_cut & ~y_below_cut)

# Generate models...
ridge_1 = Ridge()
ridge_2 = Ridge()
etr = ExtraTreesRegressor(n_estimators=1, max_depth=6, min_samples_leaf=30,
                          max_features=0.6, n_jobs=-1, random_state=seed, verbose=0)

print('Training Linear Model...\n', len(linear_features), 'features')
ridge_2.fit(train.loc[y_within_cut, linear_features], train.loc[y_within_cut, 'y'])
ridge_1.fit(np.array(train.loc[y_within_cut, linear_features[0]]
                     ).reshape(-1, 1), train.loc[y_within_cut, 'y'])

print('Training ETR Model...\n', len(etr_features), 'features')
etr.fit(train.loc[y_within_cut, etr_features], train.loc[y_within_cut, 'y'])
# end of Generate models.
# full_df = pd.read_hdf('../input/train.h5')

reward = []
# predicting...
print('Predicting...')
r_true = []
y_lr_1_p = 0
y_lr_2_p = 0
a = 0.5
b = 0.5
timestamp = 906
y_pred_acc = []
y_act_acc = []
while timestamp <= 1812:
    test = test_data.loc[test_data.timestamp == timestamp]
    test = pd.concat([test, last_stamp])

    add_diff(test)
    test['tec20-30'] = test.technical_20 - test.technical_30
    test['tec123'] = test['tec20-30'] + test.technical_13
    test['tec123_past'] = test.tec123.shift()
    test.loc[test.id_diff != 0, 'tec123_past'] = 0
    test = test.loc[test.timestamp == timestamp]
    test.sort_index(inplace=True)
    add_nan(test)
    test.fillna(d_mean, inplace=True)

    test['y_past'] = predict_y_past(test[['tec123_past', 'tec123']])
    test.fillna(0, inplace=True)
    last_stamp = test.loc[test.timestamp == timestamp, origin_features_exclude_y]

    y_etr = etr.predict(test[etr_features])
    y_act = test.loc[test.timestamp == timestamp, 'y']
    y_lr_2 = ridge_2.predict(test[linear_features]).clip(low_y_cut, high_y_cut)
    y_lr_1 = ridge_1.predict(np.array(test[linear_features[0]]
                                      ).reshape(-1, 1)).clip(low_y_cut, high_y_cut)
    y = y_lr_1 * 0.05 + y_lr_2 * 0.2 + y_etr * 0.75
    r_1 = r2_score(y_act, y_lr_1)
    r_2 = r2_score(y_act, y_lr_2)
    r_etr = r2_score(y_act, y_etr)
    R = r2_score(y_act, y)
    y_pred_acc.extend(list(y))
    y_act_acc.extend(list(y_act))
    score = r2_score(np.array(y_act_acc), np.array(y_pred_acc))
    reward.append(score)
    timestamp += 1
    if timestamp % 100 == 0:
        print('timestamp:', timestamp, '---->', score)


print('final r2_score:', reward[-1])
plt.figure(figsize=(15, 8))
plt.plot(range(906, test.timestamp.max() + 1), reward)
plt.plot([905, 1801], [0, 0], 'k--')
plt.show()
