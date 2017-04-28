import numpy as np
import pandas as pd


def corrcoef(col_1, col_2):
    corref = np.corrcoef(col_1, col_2)[0][1]
    return corref


def feature_corref(data, target_col, cols, verbose=False):
    corref = {}
    if target_col in cols:
        cols.remove(target_col)
    for i, col in enumerate(cols):
        if verbose:
            print(i, 'process', col)
        clean_cols = data.loc[data[target_col].notnull() & data[col].notnull(), [target_col, col]]
        corref[col] = np.corrcoef(clean_cols[col], clean_cols[target_col])[0][1]
    return sorted(corref.items(), key=lambda x: np.abs(x[1]), reverse=True)


def scatter_plot(data, col_1, col_2, show=False):
    import matplotlib.pyplot as plt
    clean_cols = data.loc[data[col_1].notnull() & data[col_2].notnull(), [col_1, col_2]]
    plt.figure(figsize=(15, 8))
    plt.scatter(clean_cols[col_1], clean_cols[col_2])
    plt.xlabel(col_1)
    plt.ylabel(col_2)
    if show:
        plt.show()


def R_score(y_pred, y):
    '''
    input: ypred, y
    return: R2 score of prediction
    '''
    u = np.mean(y)
    R2 = 1 - np.sum(np.square(y - y_pred)) / np.sum(np.square(y - u))
    R = np.sign(R2) * np.sqrt(np.abs(R2))
    return R


def r_score(y, y_pred):
    '''
    input: ypred, y
    return: R2 score of prediction
    '''
    u = np.mean(y)
    R2 = 1 - np.sum(np.square(y - y_pred)) / np.sum(np.square(y - u))
    R = np.sign(R2) * np.sqrt(np.abs(R2))
    return R


def preprocess(path='train.h5'):
    data = pd.read_hdf(path)
    excl = ['id', 'timestamp', 'y']
    feature_cols = [col for col in data.columns if col not in excl]
    return data, feature_cols


def split_data(data, features='all', target='y'):
    if features == 'all':
        features = data.columns
    X_train = data.loc[data.timestamp <= 905, features]
    y_train = data.loc[data.timestamp <= 905, target]
    X_test = data.loc[data.timestamp > 905, features]
    y_test = data.loc[data.timestamp > 905, target]
    return X_train, y_train, X_test, y_test


def sort_dict(dict_to_sort, key=lambda x : x[1], reverse=True):
    return sorted(dict_to_sort.items(), key=key, reverse=reverse)


def origin_features(data, excl=['id', 'timestamp', 'y']):
    return [col for col in data.columns if col not in excl]


def additional_features(feature_base, suffix):
    return [col + suffix for col in feature_base]


def excl_features():
    return ['id', 'timestamp', 'y']


def add_nans(data, features_to_add=None, fillna=False, nan_type='int'):
    features_to_add = features_to_add if features_to_add else origin_features(data)
    nan_features = additional_features(features_to_add, '_nan')
    for feature, nan_feature in zip(features_to_add, nan_features):
        data[nan_feature] = pd.isnull(data[feature])
        if nan_type == 'int':
            data[nan_feature] = data[nan_feature].astype(int)
    data['null_count'] = data.isnull().sum(axis=1)
    print('sucessfully add', len(features_to_add), 'nan features')


def add_diffs(data, features_to_add=None, resort=False):
    features_to_add = features_to_add if features_to_add else origin_features(data)
    diff_features = additional_features(features_to_add, '_diff')
    data.sort_values(['id', 'timestamp'], inplace=True)
    data['id_diff'] = data.id.diff()
    for feature, diff_feature in zip(features_to_add, diff_features):
        data[diff_feature] = data[feature].diff()
    data.loc[data.id_diff != 0, diff_features] = 0.0
    # data.drop('id_diff', inplace=True, axis=1)
    print('sucessfully add', len(features_to_add), 'diff features')


def predict_y_past(x):
    w = np.array([-8.08872128, 8.89837742]).T
    return x.dot(w) - 0.00026617524826966221
