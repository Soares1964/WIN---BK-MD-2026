import numpy as np


def cross_signal(fast, slow):

    f = fast.values
    s = slow.values

    sig = np.zeros(len(f))

    sig[f > s] = 1
    sig[f < s] = -1

    return sig