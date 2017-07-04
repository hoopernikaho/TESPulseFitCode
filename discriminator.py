import numpy as np

from itertools import compress


def find_crossing(vec, th):
    """returns index of vec crossing threshold th

    :param vec: signal
    :type vec: list of floats
    :param th: threshold
    :type th: float
    :returns: list of index of crossings
    :rtype: list of int
    """

    vec = vec - th
    v_bool = (vec[:-1] * vec[1:]) <= 0
    return list(compress(xrange(len(v_bool)), v_bool))


def intervals_no_edges(signal, high_th, low_th):
    """SET-RESET trigger

    matches start and stop indexes into ordered pairs, removing partial pulses
    at the edge of the trace
    :param signal: signal trace
    :type signal: array of float
    :param high_th: SET threshold
    :type high_th: float
    :param low_th: RESET threshold
    :type low_th: float
    :returns: two arrays, one for the starting indexes, the other for the stops
    :rtype: arrays of int
    """
    hi_cross = find_crossing(signal, high_th)
    low_cross = find_crossing(signal, low_th)

    starts = []
    stops = []

    try:
        start = next(x for x in hi_cross if x > low_cross[0])
        stop = next(x for x in low_cross if x > start)
    except StopIteration:
        return np.array(starts), np.array(stops)
    starts.append(start)
    stops.append(stop)
    while True:
        try:
            start = next(x for x in hi_cross if x > stop)
            stop = next(x for x in low_cross if x > start)
        except StopIteration:
            break
        stops.append(stop)
        starts.append(start)
        starts = starts[: len(stops)]

    return np.array(starts), np.array(stops)


def intervals_w_edges(signal, high_th, low_th):
    """SET-RESET trigger

    matches start and stop indexes into ordered pairs ignoring eges effect
    :param signal: signal trace
    :type signal: array of float
    :param high_th: SET threshold
    :type high_th: float
    :param low_th: RESET threshold
    :type low_th: float
    :returns: two arrays, one for the starting indexes, the other for the stops
    :rtype: arrays of int
    """
    hi_cross = find_crossing(signal, high_th)
    low_cross = find_crossing(signal, low_th)

    starts = []
    stops = []
    stop = 0
    while True:
        try:
            start = next(x for x in hi_cross if x > stop)
            stop = next(x for x in low_cross if x > start)
        except StopIteration:
            break
        stops.append(stop)
        starts.append(start)
        # starts = starts[: len(stops)]

    return np.array(starts), np.array(stops)


def create_mask_for_peak(length, starts, stops):
    """ Peak mask creator

    Generates a boolean mask from matching start and stop indexes
    :param length: length of the mask
    :type length: int
    :param starts: array of starting indexes
    :type starts: numpy array of int
    :param stops: array of stopping indexes
    :type stops: numpy array of int
    :returns: boolean mask
    :rtype: numpy array of bool
    """
    starts = starts[(starts >= 0) & (stops <= length)]
    stops = stops[(starts >= 0) & (stops <= length)]

    mask = np.zeros(length, dtype=bool)
    for start, stop in zip(starts, stops):
        mask[start:stop] = True
    return mask


def disc_peak(signal, high_th, low_th, edges=False):
    """from trace to mask

    high level function: from the trace and the SET-RESET threshold,
        generates a mask singling out the peaks
    :param signal: trace
    :type signal: array of float
    :param high_th: SET threshold
    :type high_th: float
    :param low_th: RESET threshold
    :type low_th: float
    :param edges: if include also partial pulses
    :type edges: bool
    :returns: boolean mask
    :rtype: numpy array of bool
    """
    func = intervals_no_edges
    if edges:
        func = intervals_w_edges
    starts, stops = func(signal, high_th, low_th)
    mask = create_mask_for_peak(len(signal), starts, stops)
    return mask