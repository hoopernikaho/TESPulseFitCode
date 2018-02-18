"""
Similar to pulse_averaging.py but for cw traces.
"""
import numpy as np
import peakutils
from lmfit import Model
from lmfit import Parameters
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import lecroy

import heralded_pulses_analysis as hpa
import pulse_utils as pu
import pulse_averaging as pa
import trace_param as tp

def positive_time_axis(time_v):
    return time_v-time_v[0]

def time_offset(time_v, signal, high_th, low_th, offset):
    """ 
    shift a trace so that the steepest point is at time 0
    steepest points must exist within discriminator window
    """
    # start by finding the max of the derivative
    dt = np.diff(time_v)[0]

    # derivative of the signal, smoothed
    d_signal = savgol_filter(np.diff(signal), 41, 1)
    # ax = plt.gca()
    # ax2 = ax.twinx()
    # ax2.plot(d_signal/np.max(d_signal), color='green')

    # use peaks only if they are legitimate edges identified by the set-reset switch 
    mask = pu.disc_peak_full(signal, high_th, low_th, offset)
    # print np.where(mask==True)
    # print peakutils.indexes(d_signal, .5, 3000)

    # find peaks in the derivative to find the steepest point
    idx = np.array([i for i in peakutils.indexes(d_signal, .5, 3000) if mask[i]==True])
    # print idx

    if len(idx) == 0:
        return [np.nan for _ in time_v]

    idx_s = np.flipud(idx[d_signal[idx].argsort()])[0]
    try:
        time_p = peakutils.interpolate(time_v[:-1], d_signal, [idx_s])[0]
    except (RuntimeError, ValueError) as e:
        time_p = time_v[idx_s]

    n_shift = int(time_p / dt)
    return pa.shift(signal, - n_shift+int(len(signal)/2))

def trace_ave(filelist, high_th, low_th, offset, smooth=41):
    time = positive_time_axis(pu.time_vector(filelist[0]))
    dt = np.diff(time)[0]
    a = [time_offset(time, tp.trace_extr(file), high_th, low_th, offset)
         for file
         in filelist]
    # reduce to trace length to avoid edge effects

    idx_0 = int(len(time) / 30)
    v_len = len(time) - 2 * idx_0
    time = time - int(len(time)/2)*dt  # zero average trace to t=0
    time = time[idx_0:idx_0 + v_len]
    a = [line[idx_0:idx_0 + v_len] for line in a]

    return time, savgol_filter(np.nanmean(a, 0), smooth, 1)

def fit_shift(time_s, signal, fit_model, high_th, low_th, offset):
    """ 
    fit the trace with a sample pulse and shift it to match the staritngtime
    """
    dt = np.diff(time_s)[0]
    d_signal = savgol_filter(np.diff(signal), 41, 1)
    # ax = plt.gca()
    # ax2 = ax.twinx()
    # ax2.plot(d_signal/np.max(d_signal), color='green')

    # use peaks only if they are legitimate edges identified by the set-reset switch 
    mask = pu.disc_peak_full(signal, high_th, low_th, offset)

    # find peaks in the derivative to find the steepest point
    idx = np.array([i for i in peakutils.indexes(d_signal, .5, 3000) if mask[i]==True])
    # print(time_s[idx])
    # print time_v[left_edges], time_v[idx]

    if len(idx) == 0:
        return [np.nan for _ in time_s]

    idx_s = np.flipud(idx[d_signal[idx].argsort()])
    # print(time_s[idx_s[0]])

    p = Parameters()
    p.add('x_offset', time_s[np.argmax(signal)])
    if len(idx_s) > 0:
        p.add('x_offset', time_s[idx_s[0]])
    p.add('amplitude', 1, vary=1)

    result = fit_model.fit(signal,
                           x=time_s,
                           params=p,
                           # weights=1 / 0.001
                           )

    n_shift = int(result.best_values['x_offset'] / dt)

    # print(result.fit_report())
    # result.plot_fit()
    # print n_shift*dt
    return pa.shift(signal, -n_shift+int(len(signal)/2))


def fit_corrected_pulse(filelist, high_th, low_th, offset, fit_model):
    time = positive_time_axis(pu.time_vector(filelist[0]))
    dt = np.diff(time)[0]
    a = [fit_shift(time, tp.trace_extr(file), fit_model, high_th, low_th, offset)
         for file
         in filelist]

    # reduce to trace length to avoid edge effects
    idx_0 = int(len(time) / 30)
    v_len = len(time) - 2 * idx_0
    time = time - int(len(time)/2)*dt  # zero average trace to t=0
    time = time[idx_0:idx_0 + v_len]
    a = [line[idx_0:idx_0 + v_len] for line in a]
    a_avg = np.nanmean(a, 0)
    a_std = np.nanstd(a, 0)
    # bg = np.median(a[a<0.5*np.max(a)])
    # a_fs = savgol_filter(a_avg, 101, 5)
    hist_fs = np.histogram(a_avg[a_avg<np.max(a_avg/2)], 500)
    a_avg = a_avg - hist_fs[1][np.argmax(hist_fs[0])]

    return time, a_avg, a_std 

def time_fitted(time_s, signal, fit_model, high_th, low_th, offset):
    """ 
    fit the trace with a sample pulse and return the detection time
    """
    try:
        dt = np.diff(time_s)[0]
        d_signal = savgol_filter(np.diff(signal), 41, 1)
        # ax = plt.gca()
        # ax2 = ax.twinx()
        # ax2.plot(d_signal/np.max(d_signal), color='green')

        # use peaks only if they are legitimate edges identified by the set-reset switch 
        mask = pu.disc_peak_full(signal, high_th, low_th, offset)

        # find peaks in the derivative to find the steepest point
        idx = np.array([i for i in peakutils.indexes(d_signal, .5, 3000) if mask[i]==True])
        # print(time_s[idx])
        # print time_v[left_edges], time_v[idx]

        # if len(idx) == 0:
        #     return [np.nan for _ in time_s]

        idx_s = np.flipud(idx[d_signal[idx].argsort()])
        # print(time_s[idx_s[0]])

        p = Parameters()
        p.add('x_offset', time_s[np.argmax(signal)])
        if len(idx_s) > 0:
            p.add('x_offset', time_s[idx_s[0]])
        p.add('amplitude', 1, vary=1)

        result = fit_model.fit(signal,
                               x=time_s,
                               params=p,
                               # weights=1 / 0.001
                               )

        toa = result.best_values['x_offset']
        # print(toa)
        return toa
    except:
        return 0