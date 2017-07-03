import numpy as np

from itertools import compress


def find_crossing(vec, th):
    """returns index of vec crossing threshold th

    :param vec: signal
    :type vec: list of floats
    :param th: threshold
    :type th: float
    :returns: [description]
    :rtype: {[type]}
    """

    vec = vec - th
    v_bool = (vec[:-1] * vec[1:]) <= 0
    return list(compress(xrange(len(v_bool)), v_bool))


def create_mask_for_edges(length, start, stop):
    mask = np.zeros(length, dtype=bool)
    if (stop < 0) ^ (stop > length):
        return mask
    if stop > start:
        mask[:stop] = True
    return mask


def disc_edges(signal, high_th, low_th):
    hi_cross = find_crossing(signal, high_th)
    low_cross = find_crossing(signal, low_th)
    if not hi_cross:
        return np.zeros(len(signal), dtype=bool)
    mask = create_mask_for_edges(len(signal), hi_cross[0], low_cross[0])
    return mask
