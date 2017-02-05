import numpy as np
import pandas as pd


def corrcoef(col_1, col_2):
    corref = np.corrcoef(col_1, col_2)[0][1]
    return corref


def feature_corref(data, target_col, cols):
    corref = {}
    if target_col in cols:
        cols.remove(target_col)
    for i, col in enumerate(cols):
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


def preprocess(path='train.h5'):
    data = pd.read_hdf(path)
    excl = ['id', 'timestamp', 'y']
    feature_cols = [col for col in data.columns if col not in excl]
    return data, feature_cols


def split_data(data, features='all'):
    if features == 'all':
        features = data.columns
    X_train = data.loc[data.timestamp <= 905, features]
    y_train = data.loc[data.timestamp <= 905, 'y']
    X_test = data.loc[data.timestamp > 905, features]
    y_test = data.loc[data.timestamp > 905, 'y']
    return X_train, y_train, X_test, y_test


def sort_dict(dict_to_sord, key=None, reverse=True):
    return sorted(dict_to_sord.items(), key=key, reverse=reverse)


def origin_features(data):
    return [col for col in data.columns if col not in ['id', 'timestamp', 'y']]

def addtional_features(feature_base, suffix):
    return [col + suffix for col in feature_base]

def excl_feature():
    return ['id', 'timestamp', 'y']