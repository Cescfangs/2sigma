import kagglegym
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression

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


excl = [env.ID_COL_NAME, env.SAMPLE_COL_NAME, env.TARGET_COL_NAME, env.TIME_COL_NAME]
origin_features = [c for c in observation.train.columns if c not in excl]
origin_features_exclude_y = [c for c in observation.train.columns if c not in ['y']]
# End of Feature Selection #

print('Processing data...')
d_mean = observation.train.median(axis=0)
last_stamp = observation.train.loc[observation.train.timestamp ==
                                   observation.train.timestamp.max(), origin_features_exclude_y]

train = observation.train
Nan_counts = train.isnull().sum(axis=1)
train.sort_values(['id', 'timestamp'], inplace=True)
tec20_diff = train.technical_20.diff()
y = train.y
train = train[origin_features]

for feature in null_labels:
    train[feature + '_nan_'] = pd.isnull(train[feature])
    d_mean[feature + '_nan_'] = 0

train = train.fillna(d_mean)
train['znull'] = Nan_counts

linear_data = pd.DataFrame({'tec20': train.technical_20, 'diff': tec20_diff})
linear_data.fillna(0, inplace=True)
rfr = ExtraTreesRegressor(n_estimators=101, max_depth=4, n_jobs=-1, random_state=17, verbose=0)
model1 = rfr.fit(train, y)
print('Training Linear Model...')
# https://www.kaggle.com/bguberfain/two-sigma-financial-modeling/univariate-model-with-clip/run/482189
low_y_cut = -0.075
high_y_cut = 0.075
y_is_above_cut = (y > high_y_cut)
y_is_below_cut = (y < low_y_cut)
y_is_within_cut = (~y_is_above_cut & ~y_is_below_cut)
model2 = LinearRegression(n_jobs=-1)
model2.fit(linear_data.loc[y_is_within_cut], y.loc[y_is_within_cut])
train = []

# https://www.kaggle.com/ymcdull/two-sigma-financial-modeling/ridge-lb-0-0100659
ymean_dict = dict(observation.train.groupby(["id"])["y"].median())

print('Predicting...')
while True:
    timestamp = observation.features.timestamp[0]
    test = observation.features
    test_diff = test[origin_features_exclude_y]
    test_diff = pd.concat([test_diff, last_stamp])
    test_diff.sort_values(['id', 'timestamp'], inplace=True)
    test_diff['tec20_diff'] = test_diff.technical_20.diff()
    test_diff.sort_index(inplace=True)
    test_diff = test_diff.loc[test_diff.timestamp == timestamp, 'tec20_diff']
    last_stamp = test.loc[test.timestamp == timestamp, origin_features_exclude_y]
    linear_data = pd.DataFrame({'tec20': test.technical_20, 'diff': test_diff})
    linear_data = linear_data.fillna(0)

    Nan_counts = test.isnull().sum(axis=1)
    test = test[origin_features]
    for c in test.columns:
        test[c + '_nan_'] = pd.isnull(test[c])
    test = test.fillna(d_mean)
    test['znull'] = Nan_counts
    pred = observation.target
    pred['y'] = model2.predict(linear_data).clip(low_y_cut, high_y_cut) * 0.36 + \
        (model1.predict(test).clip(low_y_cut, high_y_cut) * 0.64)
    pred['y'] = pred.apply(lambda r: 0.96 * r['y'] + 0.04 * ymean_dict[r['id']]
                           if r['id'] in ymean_dict else r['y'], axis=1)
    pred['y'] = [float(format(x, '.6f')) for x in pred['y']]
    o, reward, done, info = env.step(pred)
    if done:
        print("R score ...", info["public_score"])
        break
    if timestamp % 100 == 0:
        print(timestamp, '---->', reward)
