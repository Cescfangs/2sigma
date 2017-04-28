import kagglegym
import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.linear_model import Ridge
from copy import deepcopy


# Kaggle Environment #
env = kagglegym.make()
observation = env.reset()
# End of Kaggle Environment


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

etr_features = [
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

model_features = ['technical_20',
                  'tec20-30',
                  'tec123',
                  'technical_20_diff',
                  'y_lr1',
                  'y_lr2',
                  'technical_22',
                  'tec123_past',
                  'fundamental_25_nan',
                  'technical_7',
                  'technical_40',
                  'y_etr',
                  'technical_30',
                  'fundamental_27_nan',
                  'y_past',
                  'technical_30_diff',
                  'fundamental_17_nan',
                  'technical_2',
                  'fundamental_63_nan',
                  'fundamental_59',
                  'fundamental_47_nan',
                  'fundamental_5_nan',
                  'technical_17',
                  'technical_9_nan',
                  'technical_11',
                  'technical_21']

low_y_cut = -0.08
high_y_cut = 0.08
seed = 17
excl = [env.ID_COL_NAME, env.SAMPLE_COL_NAME, env.TARGET_COL_NAME, env.TIME_COL_NAME]
origin_features = [c for c in observation.train.columns if c not in excl]
origin_features_exclude_y = [c for c in observation.train.columns if c not in ['y']]
diff_features = [feature + '_diff' for feature in origin_features]
linear_features = ['technical_20_diff', 'tec20-30']
# End of Feature Selection #

d_mean = observation.train.median(axis=0)
last_stamp = observation.train.loc[observation.train.timestamp ==
                                   observation.train.timestamp.max(), origin_features_exclude_y]

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


def R_score(y_pred, y):
    '''
    input: ypred, y
    return: R2 score of prediction
    '''
    u = np.mean(y)
    R2 = 1 - np.sum(np.square(y - y_pred)) / np.sum(np.square(y - u))
    R = np.sign(R2) * np.sqrt(np.abs(R2))
    return R


def predict_y_past(x):
    w = np.array([-8.36489105, 9.19544792]).T
    return x.dot(w) - 0.00020021698404804056


def recurrent_training(clf, data, features, ix, quant=0.995, subsample_least=0.9):
    r_best = -99999
    limit_size = len(ix) * subsample_least
    clf_best = deepcopy(clf)
    features = [features] if isinstance(features, str) else features
    while True:
        print('\ntraining on', len(ix), 'samples')

        if len(features) == 1:
            clf.fit(np.array(train.loc[ix, features]).reshape(-1, 1), train.loc[ix, 'y'])
            # y_pred_residual = clf.predict(
            #     np.array(data.loc[ix, features]).reshape(-1, 1)).clip(low_y_cut, high_y_cut)
            y_pred = clf.predict(
                np.array(data[features]).reshape(-1, 1)).clip(low_y_cut, high_y_cut)
        else:
            clf.fit(train.loc[ix, features], train.loc[ix, 'y'])
            y_pred = clf.predict(data[features]).clip(low_y_cut, high_y_cut)
        r = R_score(y_pred, data.y)
        y_pred_residual = y_pred[ix]
        print('r_score on train:', r, ', improving:', r - r_best)
        if r > r_best:
            r_best = r
            clf_best = deepcopy(clf)
            residual = np.abs(y_pred_residual - data.loc[ix, 'y'])
            ix = residual[abs(residual) <= abs(residual).quantile(quant)].index
            if len(ix) < limit_size:
                break
        else:
            break
    return clf_best


# ymean_dict = dict(observation.train.groupby(["id"])["y"].median())
print('Processing data...')
train = observation.train
add_diff(train)
add_nan(train)
train = train.fillna(d_mean)
train['tec20-30'] = train.technical_20 - train.technical_30
train['tec123'] = train['tec20-30'] + train.technical_13
train['tec123_past'] = train.tec123.shift()
train['y_past'] = train.y.shift()
train.loc[train.id_diff != 0, ['tec123_past', 'y_past']] = 0


y_above_cut = (train.y > high_y_cut)
y_below_cut = (train.y < low_y_cut)
y_within_cut = (~y_above_cut & ~y_below_cut)

# Generate models...
# ridge_1 = Ridge()
# ridge_2 = Ridge()
print('Training Linear Model...\n', len(linear_features), 'features')
# recurrent ridge_1:

ix = train[y_within_cut].index
ridge_1 = recurrent_training(clf=Ridge(), data=train, features=linear_features[0], ix=ix)
ridge_2 = recurrent_training(clf=Ridge(), data=train, features=linear_features, ix=ix)

# print('Training XGBoost Model...\n', len(xgb_features), 'features')
# xgb.fit(train[xgb_features], train.y)
print('Training ETR Model...\n', len(etr_features), 'features')

train['y_lr1'] = ridge_1.predict(
    np.array(train[linear_features[0]]).reshape(-1, 1)).clip(low_y_cut, high_y_cut)
train['y_lr2'] = ridge_2.predict(train[linear_features]).clip(low_y_cut, high_y_cut)

models = [(ridge_1, [linear_features[0]], np.abs(train.y - train.y_lr1)),
          (ridge_2, linear_features, np.abs(train.y - train.y_lr2))]
etr = ExtraTreesRegressor(n_estimators=1, max_depth=6, min_samples_leaf=30,
                          max_features=0.6, n_jobs=-1, random_state=seed, verbose=0)

etr.fit(train[etr_features], train['y'])
train['y_etr'] = etr.predict(train[etr_features])

# choose best 8 trees:
etr_trees = []
tree_num = 8
for tree in etr.estimators_:
    y_tree = tree.predict(train[etr_features])
    residual = np.abs(y_tree - train.y)
    r_tree = R_score(y_tree, train.y)
    etr_trees.append((tree, r_tree, residual))
best_trees = [(x[0], etr_features, x[2]) for x in sorted(etr_trees, key=lambda x: x[1])][: tree_num]
models.extend(best_trees)
# end of Generate models.
# full_df = pd.read_hdf('../input/train.h5')
print('Best model numbers:', len(models))

print('Training Weights Model...\n', len(model_features), 'features')

residual = pd.DataFrame({
    'residual_lr1': np.abs(train['y_lr1'] - train.y),
    'residual_lr2': np.abs(train['y_lr2'] - train.y),
})
for ind, tree in enumerate(best_trees):
    residual['tree_' + str(ind)] = tree[2]

residual['best'] = residual.idxmin(axis=1)
model_etc = ExtraTreesClassifier(n_jobs=-1, n_estimators=1, max_depth=5, max_features=0.8,
                                 random_state=seed, verbose=0)
model_etc.fit(train[model_features], residual.best)


train = 0
reward = -1
# predicting...
print('Predicting...')
r_true = []
y_lr_1_p = 0
y_lr_2_p = 0
while True:
    timestamp = observation.features.timestamp[0]
    test = observation.features
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
    pred = observation.target
    y_lr_2 = ridge_2.predict(test[linear_features]).clip(low_y_cut, high_y_cut)
    y_lr_1 = ridge_1.predict(np.array(test[linear_features[0]]
                                      ).reshape(-1, 1)).clip(low_y_cut, high_y_cut)

    # add pred features
    test['y_lr1'] = y_lr_1
    test['y_lr2'] = y_lr_2

    y_etr = etr.predict(test[etr_features])  # .clip(low_y_cut, high_y_cut)
    test['y_etr'] = y_etr
    probs = model_etc.predict_proba(test[model_features])
    for ind, model in enumerate(models):
        clf, feature, _ = model
        x = test[feature] if len(feature) > 1 else np.array(test[feature]).reshape(-1, 1)
        pred['y'] += clf.predict(x).clip(low_y_cut, high_y_cut) * probs[:, ind]
    # pred['y']=(probs[:, 0] + 0.75) / 2 * y_etr + (probs[:, 2] + 0.2) / \
    #     2 * y_lr_2 + (probs[:, 1] + 0.05) / 2 * y_lr_1
    # pred['y'] = pred.apply(lambda r: 0.97 * r['y'] + 0.03 * ymean_dict[r['id']]
    #                       if r['id'] in ymean_dict else r['y'], axis=1)
    pred['y'] = [float(format(x, '.6f')) for x in pred['y']]

    observation, reward, done, info = env.step(pred)
    if done:
        print("R score ...", info["public_score"])
        # print("R True...", np.array(r_true).mean())
        break
    if timestamp % 100 == 0:
        print('timestamp:', timestamp, '---->', reward)
