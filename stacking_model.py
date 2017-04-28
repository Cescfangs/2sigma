import kagglegym
import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge, LinearRegression


# Kaggle Environment #
env = kagglegym.make()
observation = env.reset()
# End of Kaggle Environment #

# Feature Selection #
etr_features = [
    't22_0.0',
    't22_0.5',
    't34_0.0',
    't34_0.5',
    'y_lr1',
    'y_lr2',
    'y_past',
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
    # 'technical_22',
    'technical_41_nan',
    'fundamental_8',
    'fundamental_21',
    'fundamental_17_nan',
    # 'technical_34',
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
    # 'fundamental_63_nan',
    # 'fundamental_31_nan',
    # 'fundamental_40_nan',
    # 'fundamental_35_nan',
    # 'technical_3',
    # 'fundamental_13'
]
seed = 17
excl = [env.ID_COL_NAME, env.SAMPLE_COL_NAME, env.TARGET_COL_NAME, env.TIME_COL_NAME]
origin_features = [c for c in observation.train.columns if c not in excl]
origin_features_exclude_y = [c for c in observation.train.columns if c not in ['y']]
to_add_diff = [feature[:-5] for feature in etr_features if feature.endswith('_diff')]
to_add_nan = [feature[:-4] for feature in etr_features if feature.endswith('_nan')]
diff_features = [feature + '_diff' for feature in to_add_diff]
linear_features = ['technical_20_diff', 'tec20-30']
# End of Feature Selection #

d_mean = observation.train.median(axis=0)
last_stamp = observation.train.loc[observation.train.timestamp ==
                                   observation.train.timestamp.max(), origin_features_exclude_y]

# add diffs #


def additional_features(feature_base, suffix):
    return [col + suffix for col in feature_base]


# add diffs #
def add_diffs(data, features_to_add=None, verbose=False):
    features_to_add = features_to_add if features_to_add else origin_features(data)
    diff_features = additional_features(features_to_add, '_diff')
    data.sort_values(['id', 'timestamp'], inplace=True)
    data['id_diff'] = data.id.diff()
    for feature, diff_feature in zip(features_to_add, diff_features):
        data[diff_feature] = data[feature].diff()
    data.loc[data.id_diff != 0, diff_features] = 0.0
    if verbose:
        print('successfully add', len(features_to_add), 'diff features')
# end of diffs #


# add Nan tags #
def add_nans(data, features_to_add=None, verbose=False):
    features_to_add = features_to_add if features_to_add else origin_features(data)
    nan_features = additional_features(features_to_add, '_nan')
    for feature, nan_feature in zip(features_to_add, nan_features):
        data[nan_feature] = pd.isnull(data[feature])
        data[nan_feature] = data[nan_feature].astype(int)
    data['null_count'] = data.isnull().sum(axis=1)
    if verbose:
        print('successfully add', len(features_to_add), 'nan features')
# end of Nan tags #


def R_score(y_pred, y):
    '''
    input: ypred, y
    return: R2 score of prediction
    '''
    u = np.mean(y)
    R2 = 1 - np.sum(np.square(y - y_pred)) / np.sum(np.square(y - u))
    R = np.sign(R2) * np.sqrt(np.abs(R2))
    return R


def quantile(folds=3):
    return [1 / folds * i for i in range(folds + 1)]


def predict_y_past(x):
    w = np.array([-8.36489105, 9.19544792]).T
    return x.dot(w) - 0.00020021698404804056


print('Processing data...')
train = observation.train
add_nans(train, to_add_nan, 1)
train = train.fillna(d_mean)
add_diffs(train, to_add_diff, 1)
train['tec20-30'] = train.technical_20 - train.technical_30
train['tec123'] = train['tec20-30'] + train.technical_13
train['tec123_past'] = train.tec123.shift()
train.loc[train.id_diff != 0, ['tec123_past']] = 0
train['y_past'] = predict_y_past(train[['tec123_past', 'tec123']])
train.loc[train.id_diff != 0, ['y_past']] = 0
# print(train.isnull().sum().sort_values(ascending=0))
low_y_cut = -0.08
high_y_cut = 0.08
low_y_clip = -0.08
high_y_clip = 0.08
y_above_cut = (train.y > high_y_cut)
y_below_cut = (train.y < low_y_cut)
y_within_cut = (~y_above_cut & ~y_below_cut)

# Generate models...
clf_1 = Ridge()
clf_2 = Ridge()
etr = ExtraTreesRegressor(n_estimators=128, max_depth=6, min_samples_leaf=30,
                          max_features=0.6, n_jobs=-1, random_state=seed, verbose=0)

print('Training Linear Model...\n', len(linear_features), 'features')
# stacking y_lr
train['y_lr1'] = 0
train['y_lr2'] = 0
folds = 10
nb_fold = int(np.ceil(train.shape[0] / folds) + 0.5)
ix = list(train.index)
np.random.shuffle(ix)
indexs = []
for i in range(folds):
    index = ix[(i * nb_fold): ((i + 1) * nb_fold)]
    indexs.append(pd.Index(index))

x = train[linear_features + ['y']]
train['y_lr1'] = 0
train['y_lr2'] = 0
print('train samples:', x.shape[0])
for i, index in enumerate(indexs):
    print('...train fold', i, ', size', len(index))
    X_train = x.drop(index)
    clip = (X_train.y > low_y_cut) & (X_train.y < high_y_cut)
    clf_1.fit(np.array(X_train.loc[clip, linear_features[0]]).reshape(-1, 1), X_train.loc[clip, 'y'])
    clf_2.fit(X_train.loc[clip, linear_features], X_train.loc[clip, 'y'])
    train.loc[index, 'y_lr1'] = clf_1.predict(np.array(train.loc[index, linear_features[0]]).reshape(-1, 1)).clip(low_y_cut, high_y_cut)
    train.loc[index, 'y_lr2'] = clf_2.predict(train.loc[index, linear_features]).clip(low_y_cut, high_y_cut)


# recurrent ridge_1:
better = True
r_best_train = -99999
quant = 0.995
ix = train[y_within_cut].index
limit_len = len(ix) * 0.9
while better:
    print('training on', len(ix), 'samples')
    clf_2.fit(train.loc[ix, linear_features], train.loc[ix, 'y'])
    y_pred_residual = clf_2.predict(train.loc[ix, linear_features]).clip(low_y_cut, high_y_cut)
    y_train_pred = clf_2.predict(train[linear_features]).clip(low_y_cut, high_y_cut)
    r_train = R_score(y_train_pred, train.y)
    print('r_score on train:', r_train, ', improving:', r_train - r_best_train)
    if r_train > r_best_train:
        ridge_2 = clf_2
        r_best_train = r_train
        residual = np.abs(y_pred_residual - train.loc[ix, 'y'])
        ix = residual[abs(residual) <= abs(residual).quantile(quant)].index
    else:
        break
    if len(ix) < limit_len:
        break

r_best_train = -99999
ix = train[y_within_cut].index
while better:
    print('training on', len(ix), 'samples')
    clf_1.fit(np.array(train.loc[ix, linear_features[0]]).reshape(-1, 1), train.loc[ix, 'y'])
    y_pred_residual = clf_1.predict(
        np.array(train.loc[ix, linear_features[0]]).reshape(-1, 1)).clip(low_y_cut, high_y_cut)
    y_train_pred = clf_1.predict(np.array(train[linear_features[0]]).reshape(-1, 1)).clip(low_y_cut, high_y_cut)
    r_train = R_score(y_train_pred, train.y)
    print('r_score on train:', r_train, ', improving:', r_train - r_best_train)
    if r_train > r_best_train:
        ridge_1 = clf_1
        r_best_train = r_train
        residual = np.abs(y_pred_residual - train.loc[ix, 'y'])
        ix = residual[abs(residual) <= abs(residual).quantile(quant)].index
    else:
        break
    if len(ix) < limit_len:
        break

print('Training t22 Models...\n')
discrete_values = [-0.5, 0.0, 0.5]
t22_features = []
t22_models = []
t22_names = []
for t22 in discrete_values:
    meta_name = str(t22) + 'lr1_t22'
    ind = train.loc[y_within_cut & (train.technical_22 == t22)].index
    print('...size:', len(ind), 't22:', t22)
    clf_1 = LinearRegression(n_jobs=-1, normalize=True)
    clf_1.fit(train.loc[ind, linear_features[0]].values.reshape(-1, 1), train.loc[ind, 'y'])
    # clf_2 = LinearRegression(n_jobs=-1, normalize=True)
    # clf_2.fit(train.loc[ind, linear_features], train.loc[ind, 'y'])
    t22_models.append(clf_1)
    train[meta_name] = clf_1.predict(train[linear_features[0]].values.reshape(-1, 1)).clip(low_y_cut, high_y_cut)
    # train[meta_name_2] = clf_2.predict(train[linear_features]).clip(low_y_cut, high_y_cut)
    t22_names.append(meta_name)

t22_dummy = pd.get_dummies(train.technical_22, prefix='t22')
t34_dummy = pd.get_dummies(train.technical_34, prefix='t34')
train = pd.concat([train, t34_dummy, t22_dummy], axis=1)
# train.drop(['t22_-0.5', 't34_-0.5'], axis=1, inplace=True)
# meta according to t22
train['0.5lr1_t22'] = train['0.5lr1_t22'] * train['t22_0.5']
train['-0.5lr1_t22'] = train['-0.5lr1_t22'] * train['t22_-0.5']
train['0.0lr1_t22'] = train['0.0lr1_t22'] * train['t22_0.0']
etr_features.extend(t22_names)



print('Training ETR Model...\n', len(etr_features), 'features')
# train['y_lr1'] = ridge_1.predict(np.array(train[linear_features[0]]).reshape(-1, 1)).clip(low_y_cut, high_y_cut)
# train['y_lr2'] = ridge_2.predict(train[linear_features]).clip(low_y_cut, high_y_cut)

etr.fit(train[etr_features], train['y'])
# end of Generate models.

train = 0
# predicting...
print('Predicting...')
while True:
    timestamp = observation.features.timestamp[0]
    test = observation.features
    test = pd.concat([test, last_stamp])
    add_nans(test, to_add_nan)
    test.fillna(d_mean, inplace=True)
    add_diffs(test, to_add_diff)
    test.sort_index(inplace=True)

    test['tec20-30'] = test.technical_20 - test.technical_30
    test['tec123'] = test['tec20-30'] + test.technical_13
    test['tec123_past'] = test.tec123.shift()
    test.loc[test.id_diff != 0, 'tec123_past'] = 0
    test = test.loc[test.timestamp == timestamp]
    test['y_past'] = predict_y_past(test[['tec123_past', 'tec123']])
    test.fillna(0, inplace=True)

    # Residual group
    for i, t22_model in enumerate(t22_models):
        test[t22_names[i]] = t22_model.predict(test[linear_features[0]
                                                    ].values.reshape(-1, 1)).clip(low_y_cut, high_y_cut)

    t22_dummy = pd.get_dummies(test.technical_22, prefix='t22')
    t34_dummy = pd.get_dummies(test.technical_34, prefix='t34')
    test = pd.concat([test, t34_dummy, t22_dummy], axis=1)
    test['0.5lr1_t22'] = test['0.5lr1_t22'] * test['t22_0.5']
    test['-0.5lr1_t22'] = test['-0.5lr1_t22'] * test['t22_-0.5']
    test['0.0lr1_t22'] = test['0.0lr1_t22'] * test['t22_0.0']
    # test.drop(['t22_-0.5', 't34_-0.5'], axis=1, inplace=True)
    last_stamp = test.loc[test.timestamp == timestamp, origin_features_exclude_y]

    pred = observation.target
    y_lr_2 = ridge_2.predict(test[linear_features]).clip(low_y_cut, high_y_cut)
    y_lr_1 = ridge_1.predict(np.array(test[linear_features[0]]).reshape(-1, 1)).clip(low_y_cut, high_y_cut)

    test['y_lr1'] = y_lr_1
    test['y_lr2'] = y_lr_2

    y_etr = etr.predict(test[etr_features])  # .clip(low_y_cut, high_y_cut)
    pred['y'] = 0.8 * y_etr + 0.07 * y_lr_1 + 0.13 * y_lr_2
    pred['y'] = [float(format(x, '.6f')) for x in pred['y']]

    observation, reward, done, info = env.step(pred)
    if done:
        print("R score ...", info["public_score"])
        break
    if timestamp % 100 == 0:
        print('timestamp:', timestamp, '---->', reward)
