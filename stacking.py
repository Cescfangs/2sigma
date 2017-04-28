import numpy as np

def split_value(a):
    a = np.array(a).reshape(-1, 1)
    best = a[0]
    mse = 1000
    for i in range(1, len(a) - 1):
        mse_t = np.sum(np.square(a[:i] - a[:i].mean())) + np.sum(np.square(a[i:] - a[i:].mean()))
        if mse_t < mse:
            mse = mse_t
            best = i
    return best, mse

a = [4.5, 4.75, 4.91, 5.34, 5.8, 7.05, 7.9, 8.23, 8.7, 9.0]
a_1 = [x for x in a if x >= 5]
print(split_value(a[1:4]))