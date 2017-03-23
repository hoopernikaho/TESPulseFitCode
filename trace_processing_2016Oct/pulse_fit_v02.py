#!/usr/bin/env python
"""
space for comments and authorship
"""
import heralded_pulses_analysis as hps
import numpy as np

from scipy.signal import savgol_filter

import peakutils

from lmfit import Model
from lmfit import Parameters

time_f, = np.load(results_folder + 'ph1_model.npy')


def one_pulse(x, x_offset=0, amplitude=1):
    """convert the sample single photon pulse into a function
    that can be used in a fit
    """
    x = x - x_offset
    return amplitude * np.interp(x, time_f, signal_fs)


two_pulse_fit = Model(one_pulse, prefix='one_') + \
    Model(one_pulse, prefix='two_')


def fit_two(time, signal):

    idx_s = peakutils.indexes(savgol_filter(np.diff(signal), 301, 3),
                              thres=0.7,
                              min_dist=20)
    idx_s = np.flipud(idx_s[signal[idx_s].argsort()])

    p = Parameters()
    # print(len(idx_s))
    if len(idx_s) == 0:
        p.add('one_x_offset', time[np.argmax(signal)])
        p.add('two_x_offset', time[np.argmax(signal)] + 10e-9)
    else:
        if len(idx_s) > 0:
            p.add('one_x_offset', time[idx_s[0]])
            p.add('two_x_offset', time[idx_s[0]] + 10e-9)
        if len(idx_s) > 1:
            p.add('two_x_offset', time[idx_s[1]])
    p.add('one_amplitude', 1, min=0, vary=1)
    p.add('two_amplitude', .9, min=0
          # expr='one_amplitude'
          )
    result = two_pulse_fit.fit(signal,
                               x=time,
                               params=p,
                               # weights=1 / yerr,
                               # method=method
                               )
    return result


def time_diff(time, signal):
    result = fit_two(time, signal)
    return np.abs(result.best_values['two_x_offset'] -
                  result.best_values['one_x_offset'])


def g2(filelist, t_initial=None, t_final=None):
    return [time_diff(*hps.trace_extr(file, t_initial, t_final))
            for file
            in filelist]
