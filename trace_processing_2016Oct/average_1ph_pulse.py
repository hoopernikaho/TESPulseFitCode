#!/usr/bin/env python2
"""
starting from pulse traces taken from the oscilloscope (binary format),
derive the shape of the signal photon
"""

import numpy as np
import peakutils
import thres_poiss

from lmfit import Model
from lmfit import Parameters

from scipy.signal import savgol_filter


def shift_corrected_pulse(traces, smooth=201):
    """ single ph pulse model
    :param traces: array of traces for single photon events
    :param smooth: number of smoothing points for the filter
    """
    ph1_shifted = [time_offset(time_v, t) for t in traces]
    ph1_ave = np.nanmean(ph1_shifted, 0)

    # reduce to trace length to avoid edge effects
    n = int(len(time_v) / 10)
    return time_v[n:-n], savgol_filter(ph1_ave, smooth, 3)[n:-n]


def shift(xs, n):
    """ shifting array xs by n positions """
    if n == 0:
        return xs
    e = np.empty_like(xs)
    if n >= 0:
        e[:n] = np.nan
        e[n:] = xs[:-n]
    else:
        e[n:] = np.nan
        e[:n] = xs[-n:]
    return e


def time_offset(time_v, trace):
    """ shift a trace so that the steepest point is at time 0
    """
    # start by finding the max of the derivative
    signal = trace
    dt = np.diff(time_v)[0]

    # derivative of the signal, soothed
    d_signal = savgol_filter(np.diff(signal), 301, 3)

    # find peaks in the derivative to find the steepest point
    idx = peakutils.indexes(d_signal, .5, 3000)
    if len(idx) == 0:
        return [np.nan for _ in time_v]
    idx_s = np.flipud(idx[d_signal[idx].argsort()])[0]

    try:
        time_p = peakutils.interpolate(time_v[:-1], d_signal, [idx_s])[0]
    except (RuntimeError, ValueError) as e:
        time_p = time_v[idx_s]

    n_shift = int(time_p / dt)
    return shift(trace, - n_shift)


def fit_shift(time_s, signal, fit_model):

    dt = np.diff(time_s)[0]
    d_signal = savgol_filter(np.diff(signal), 301, 3)
    idx = peakutils.indexes(d_signal, .5, 3000)
    idx_s = np.flipud(idx[d_signal[idx].argsort()])

    p = Parameters()
    p.add('x_offset', time_s[np.argmax(signal)] - 4e-7)
    if len(idx_s) > 0:
        p.add('x_offset', time_s[idx_s[0]])
    p.add('amplitude', 1, vary=1)

    result = fit_model.fit(signal,
                           x=time_s,
                           params=p,
                           weights=1 / 0.001
                           )

    n_shift = int(result.best_values['x_offset'] / dt)
    return shift(signal, -n_shift)


def fit_corrected_pulse(time_v, ph1_trc, fit_model):

    ph1_shifted = [fit_shift(time_v, s, fit_model)
                   for s
                   in ph1_trc]

    ph1_ave = np.nanmean(ph1_shifted, 0)
    n = int(len(time_v) / 10)
    return time_v[n:-n], ph1_ave[n:-n]


def ph1_traces(trc, thresholds):

    time_v = trc[0, 0].mat[:, 0]
    th = thresholds

    # select only the traces corresponding to 1 photon
    ph1 = trc[:, np.logical_and(trc[3, :] > th[0], trc[3, :] < th[1])]

    # Correct the traces for the vertical offset
    ph1_corrected = [t[0].mat[:, 1] - t[1]
                     for t
                     in ph1.transpose()]
    return time_v, ph1_corrected


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    data_folder = '../data/20160914_TES5_MAGSQ1_4.4ns_double_pulse/'
    single_pulse_folder = 'single/'
    results_folder = 'analysis/20161018_g2_heralded_singlesrate_approx_40k_results/'
    bg = 1300

    # extract the raw traces
    trc_single = thres_poiss.traces(data_folder + single_pulse_folder, bg)

    # form the area histogram find the optimal threshold for 1 photon pulses
    pnr = np.histogram(trc_single[3], 200)
    # result = thres.gauss_fit_interp(pnr, .7e-8, weighted=True)
    th = thres_poiss.thresholds_N(pnr, .7e-8, True)

    # filter the traces to obtain 1 photon traces only
    time_v, ph1_trc = ph1_traces(np.array(trc_single), th)

    # shift and average the 1 photon traces
    time_m, ph1_m = shift_corrected_pulse(data_folder + single_pulse_folder)

    # save average pulse
    np.save(data_folder + results_folder + 'ph1_model.npy',
            np.array(zip(time_m, ph1_m)))

    def one_pulse(x, x_offset=0, amplitude=1):
        """convert the sample single photon pulse into a function
        that can be used in a fit
        """
        x = x - x_offset
        return amplitude * np.interp(x, time_m, ph1_m)

    fit_model = Model(one_pulse)
    time_f, ph1_f = fit_corrected_pulse(time_v, ph1_trc, fit_model)

    plt.plot(time_m, ph1_m)
    plt.plot(time_f, ph1_f)
    plt.show()
