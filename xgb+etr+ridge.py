import kagglegym
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge
from xgboost import XGBRegressor

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
    Nan_counts = train.isnull().sum(axis=1)
    for feature in null_labels:
        data[feature + '_nan_'] = pd.isnull(data[feature])
        d_mean[feature + '_nan_'] = 0
    data['null_count'] = Nan_counts
# end of Nan tags #


# Kaggle Environment #
env = kagglegym.make()
observation = env.reset()
# End of Kaggle Environment #

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

diff_features = [
'technical_20_diff',
 'technical_30_diff',
 'technical_2_diff',
 'technical_14_diff',
 'technical_11_diff',
 'technical_17_diff',
 'fundamental_57_diff',
 'technical_6_diff',
 'technical_1_diff',
 'technical_44_diff',
 'technical_40_diff',
 'fundamental_53_diff',
 'technical_41_diff',
 'technical_43_diff',
 'fundamental_49_diff',
 'fundamental_41_diff',
 'technical_27_diff',
 'technical_21_diff',
 'technical_7_diff',
 'technical_36_diff',
 'fundamental_18_diff',
 'technical_28_diff',
 'technical_10_diff',
 'fundamental_36_diff',
 'technical_31_diff',
 'fundamental_45_diff',
 'technical_24_diff',
 'fundamental_7_diff',
 'fundamental_52_diff',
 'technical_35_diff',
 'technical_19_diff',
 'technical_33_diff',
 'fundamental_42_diff',
 'fundamental_20_diff',
 'fundamental_30_diff',
 'fundamental_26_diff',
 'derived_2_diff',
 'fundamental_15_diff',
 'technical_13_diff',
 'fundamental_8_diff',
 'derived_4_diff',
 'technical_42_diff',
 'fundamental_13_diff',
 'fundamental_48_diff',
 'fundamental_54_diff',
 'technical_38_diff',
 'fundamental_22_diff',
 'technical_5_diff',
 'fundamental_10_diff',
 'technical_34_diff',
 'fundamental_6_diff',
 'fundamental_9_diff',
 'fundamental_39_diff',
 'fundamental_51_diff',
 'fundamental_55_diff',
 'fundamental_61_diff',
 'technical_29_diff',
 'derived_1_diff',
 'technical_3_diff',
 'technical_37_diff',
 'technical_18_diff',
 'fundamental_12_diff',
 'technical_0_diff',
 'technical_25_diff',
 'fundamental_50_diff',
 'fundamental_59_diff',
 'derived_3_diff',
 'technical_12_diff',
 'fundamental_0_diff',
 'fundamental_19_diff',
 'fundamental_23_diff',
 'technical_39_diff',
 'fundamental_14_diff',
 'fundamental_16_diff',
 'fundamental_17_diff',
 'fundamental_21_diff',
 'fundamental_24_diff',
 'fundamental_25_diff',
 'fundamental_27_diff',
 'fundamental_28_diff',
 'fundamental_11_diff',
 'fundamental_29_diff',
 'derived_0_diff',
 'fundamental_1_diff',
 'fundamental_2_diff',
 'fundamental_3_diff',
 'fundamental_5_diff',
 'fundamental_62_diff',
 'fundamental_63_diff',
 'technical_16_diff',
 'technical_22_diff',
 'technical_32_diff',
 'fundamental_60_diff',
 'fundamental_31_diff',
 'fundamental_58_diff',
 'fundamental_32_diff',
 'fundamental_33_diff',
 'fundamental_34_diff',
 'fundamental_35_diff',
 'fundamental_37_diff',
 'fundamental_38_diff',
 'fundamental_40_diff',
 'fundamental_43_diff',
 'fundamental_44_diff',
 'fundamental_46_diff',
 'fundamental_47_diff',
 'fundamental_56_diff',
 'technical_9_diff']

 nan_features = ['fundamental_18_nan',
 'fundamental_44_nan',
 'fundamental_14_nan',
 'fundamental_31_nan',
 'fundamental_8_nan',
 'fundamental_41_nan',
 'fundamental_0_nan',
 'fundamental_7_nan',
 'fundamental_33_nan',
 'fundamental_62_nan',
 'fundamental_34_nan',
 'derived_0_nan',
 'technical_5_nan',
 'fundamental_24_nan',
 'fundamental_35_nan',
 'technical_3_nan',
 'technical_24_nan',
 'derived_1_nan',
 'fundamental_25_nan',
 'technical_31_nan',
 'fundamental_39_nan',
 'technical_18_nan',
 'fundamental_21_nan',
 'fundamental_9_nan',
 'fundamental_47_nan',
 'fundamental_50_nan',
 'technical_9_nan',
 'fundamental_52_nan',
 'fundamental_16_nan',
 'derived_3_nan',
 'fundamental_23_nan',
 'technical_16_nan',
 'fundamental_48_nan',
 'fundamental_28_nan',
 'fundamental_17_nan',
 'fundamental_40_nan',
 'technical_1_nan',
 'fundamental_54_nan',
 'fundamental_32_nan',
 'fundamental_57_nan',
 'fundamental_49_nan',
 'fundamental_5_nan',
 'fundamental_55_nan',
 'fundamental_37_nan',
 'fundamental_60_nan',
 'fundamental_30_nan',
 'fundamental_15_nan',
 'fundamental_43_nan',
 'fundamental_29_nan',
 'fundamental_27_nan',
 'technical_28_nan',
 'technical_44_nan',
 'fundamental_63_nan',
 'derived_2_nan',
 'fundamental_38_nan',
 'fundamental_6_nan',
 'fundamental_1_nan',
 'fundamental_61_nan',
 'fundamental_22_nan',
 'fundamental_51_nan',
 'derived_4_nan',
 'technical_25_nan',
 'technical_10_nan',
 'technical_21_nan',
 'technical_42_nan',
 'fundamental_59_nan',
 'technical_33_nan',
 'technical_14_nan',
 'technical_13_nan',
 'technical_43_nan',
 'technical_17_nan',
 'technical_35_nan',
 'technical_36_nan',
 'technical_27_nan',
 'technical_19_nan',
 'fundamental_42_nan',
 'fundamental_58_nan',
 'fundamental_20_nan',
 'technical_29_nan',
 'fundamental_19_nan',
 'technical_41_nan']

seed = 17
excl = [env.ID_COL_NAME, env.SAMPLE_COL_NAME, env.TARGET_COL_NAME, env.TIME_COL_NAME]
origin_features = [c for c in observation.train.columns if c not in excl]
origin_features_exclude_y = [c for c in observation.train.columns if c not in ['y']]
diff_features = [feature + '_diff' for feature in origin_features]
xgb_features = nan_features[:20] + diff_features + origin_features
etr_features = nan_features + diff_features[:10] + origin_features
linear_features = ['technical_20', 'technical_20_diff', 'technical_30', 'technical_30_diff']
# End of Feature Selection #


ymean_dict = dict(observation.train.groupby(["id"])["y"].median())
print('Processing data...')

d_mean = observation.train.median(axis=0)
last_stamp = observation.train.loc[observation.train.timestamp ==
                                   observation.train.timestamp.max(), origin_features_exclude_y]
train = observation.train
add_diff(train)
add_nan(train)
train = train.fillna(d_mean)
low_y_cut = -0.075
high_y_cut = 0.075
y_above_cut = (train.y > high_y_cut)
y_below_cut = (train.y < low_y_cut)
y_within_cut = (~y_above_cut & ~y_below_cut)

# Generate models...
lr = Ridge()
etr = ExtraTreesRegressor(n_estimators=100, max_depth=4, n_jobs=-1, random_state=seed, verbose=0)
xgb = XGBRegressor(n_estimators=100, nthread=-1, max_depth=3, seed=seed)

print('Training Linear Model...')
lr.fit(train.loc[y_within_cut, linear_features], train.loc[y_within_cut, 'y'])

print('Training XGBoost Model...')
xgb.fit(train[xgb_features], train.y)

print('Training ETR Model...')
etr.fit(train[etr_features], train.y)
# end of Generate models.


train = 0
w_etr = 0.32
w_lr = 0.22
w_xgb = 1 - w_etr - w_lr

# predicting...
print('Predicting...')
while True:
    timestamp = observation.features.timestamp[0]
    test = observation.features
    test = pd.concat([test, last_stamp])
    last_stamp = test.loc[test.timestamp == timestamp, origin_features_exclude_y]

    add_diff(test)
    test = test.loc[test.timestamp == timestamp]
    test.sort_index(inplace=True)
    add_nan(test)
    
    pred = observation.target
    y_etr = etr.predict(test[etr_features])
    y_xgb = xgb.predict(test[xgb_features]).clip(low_y_cut, high_y_cut)
    y_lr = lr.predict(test[linear_features]).clip(low_y_cut, high_y_cut)

    pred['y'] = w_etr * y_etr + w_lr * y_lr + w_xgb * y_xgb
    pred['y'] = pred.apply(lambda r: 0.96 * r['y'] + 0.04 * ymean_dict[r['id']]
                           if r['id'] in ymean_dict else r['y'], axis=1)
    pred['y'] = [float(format(x, '.6f')) for x in pred['y']]

    observation, reward, done, info = env.step(pred)
    if done:
        print("R score ...", info["public_score"])
        break
    if timestamp % 100 == 0:
        print(timestamp, '---->', reward)
