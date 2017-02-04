import kagglegym
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression

env = kagglegym.make()
o = env.reset()
np.corrcoef
excl = [env.ID_COL_NAME, env.SAMPLE_COL_NAME, env.TARGET_COL_NAME, env.TIME_COL_NAME]
col = [c for c in o.train.columns if c not in excl]
train = pd.read_hdf('../input/train.h5')
train = train[col]
d_mean = train.median(axis=0)

train = o.train[col]
n = train.isnull().sum(axis=1)
for c in train.columns:
    train[c + '_nan_'] = pd.isnull(train[c])
    d_mean[c + '_nan_'] = 0
train = train.fillna(d_mean)
train['znull'] = n
n = []

rfr = ExtraTreesRegressor(n_estimators=100, max_depth=4, n_jobs=-1, random_state=17, verbose=0)
model1 = rfr.fit(train, o.train['y'])

# https://www.kaggle.com/bguberfain/two-sigma-financial-modeling/univariate-model-with-clip/run/482189
low_y_cut = -0.075
high_y_cut = 0.075
y_is_above_cut = (o.train.y > high_y_cut)
y_is_below_cut = (o.train.y < low_y_cut)
y_is_within_cut = (~y_is_above_cut & ~y_is_below_cut)
model2 = LinearRegression(n_jobs=-1)
model2.fit(np.array(o.train[col].fillna(d_mean).loc[y_is_within_cut][
           ['technical_20', 'technical_30']]), o.train.loc[y_is_within_cut, 'y'])
train = []

# https://www.kaggle.com/ymcdull/two-sigma-financial-modeling/ridge-lb-0-0100659
ymean_dict = dict(o.train.groupby(["id"])["y"].median())

model_weights = [0.6, 0.65, 0.7]
dict_weights = [0.02, 0.05, 0.1]
case_no = len(model_weights) * len(dict_weights)
for i, model_weight in enumerate(model_weights):
    for j, dict_weight in enumerate(dict_weights):
        print('\n#############################################\nProcess case ', i * len(dict_weights) + j, 'of ', case_no)
        test = o.features[col]
        n = test.isnull().sum(axis=1)
        for c in test.columns:
            test[c + '_nan_'] = pd.isnull(test[c])
        test = test.fillna(d_mean)
        test['znull'] = n
        pred = o.target
        test2 = np.array(o.features[col].fillna(d_mean)[['technical_20', 'technical_30']])
        pred['y'] = (model1.predict(test).clip(low_y_cut, high_y_cut) * model_weight) + \
            (model2.predict(test2).clip(low_y_cut, high_y_cut) * (1 - model_weight))
        pred['y'] = pred.apply(lambda r: (1 - dict_weight) * r['y'] + dict_weight * ymean_dict[r['id']]
                               if r['id'] in ymean_dict else r['y'], axis=1)
        pred['y'] = [float(format(x, '.6f')) for x in pred['y']]
        o, reward, done, info = env.step(pred)
        if done:
            print("el fin ...", info["public_score"])
            break
        if o.features.timestamp[0] % 100 == 0:
            print(reward)