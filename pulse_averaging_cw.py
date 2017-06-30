"""
Similar to pulse_averaging.py but for cw traces.
"""
import numpy as np
import peakutils
from lmfit import Model
from lmfit import Parameters
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

import heralded_pulses_analysis as hps
import pulse_discrimination as pd
import pulse_averaging as pa

def disc_shift(time_s, signal, height_th):
    """ 
    use the set reset switch to locate the pulse edge and shift it to the center of the acquisition window.
    if all traces have the same acquisition window, they will be synchronised.
    """
    center = int(len(time_s)/2)

    [mask, clamp, edges, left_edges, right_edges] = pd.discriminator(
        time_s, 
        signal, 
        height_th=height_th,
        method=3)   
    n_shift = left_edges[0]
    print n_shift
    return pa.shift(signal, -n_shift+center)

def time_offset(time_v, trace, height_th):
    """ shift a trace so that the steepest point is at time 0
    """
    # start by finding the max of the derivative
    zero = int(len(time_v)/4)*0
    signal = trace
    dt = np.diff(time_v)[0]

    # derivative of the signal, smoothed
    d_signal = savgol_filter(np.diff(signal), 301, 1)
    # ax = plt.gca()
    # ax2 = ax.twinx()
    # ax2.plot(np.array(time_v)[1:],d_signal*100)

    # use peaks only if they are legitimate edges identified by the set-reset switch 
    [mask, clamp, edges, left_edges, right_edges] = pd.discriminator(
    time_v, 
    signal,
    dt_left=100e-9,
    dt_right=0, 
    height_th=height_th,
    method=2) 

    # find peaks in the derivative to find the steepest point
    idx = left_edges[0]+peakutils.indexes(d_signal[(mask&clamp)[1:]], .5, 3000)
    # print time_v[left_edges], time_v[idx]

    if len(idx) == 0:
        return [np.nan for _ in time_v]
    # else:
    #     idx=(mask&clamp)[idx]*idx

    idx_s = np.flipud(idx[d_signal[idx].argsort()])[0]
    try:
        time_p = peakutils.interpolate(time_v[:-1], d_signal, [idx_s])[0]
    except (RuntimeError, ValueError) as e:
        time_p = time_v[idx_s]

    n_shift = int(time_p / dt)
    return pa.shift(trace, - n_shift + zero)

def trace_ave(filelist, height_th, t_initial=None, t_final=None, smooth=201):
    time, _ = hps.trace_extr(filelist[0], height_th, t_initial, t_final)

    a = [time_offset(*hps.trace_extr(file, height_th, t_initial, t_final),height_th=height_th)
         for file
         in filelist]
    # reduce to trace length to avoid edge effects

    idx_0 = int(len(time) / 30)
    v_len = len(time) - 2 * idx_0
    time = time[idx_0:idx_0 + v_len]
    a = [line[idx_0:idx_0 + v_len] for line in a]

    return time, savgol_filter(np.nanmean(a, 0), smooth, 3)

def fit_shift(time_s, signal, fit_model, height_th):
    """ fit the trace with a sample pulse and shift it to match the staritng
    time
    """
    zero = int(len(time_s)/4)*0
    dt = np.diff(time_s)[0]
    d_signal = savgol_filter(np.diff(signal), 301, 1)
    # ax = plt.gca()
    # ax2 = ax.twinx()
    # ax2.plot(np.array(time_v)[1:],d_signal*100)

    # use peaks only if they are legitimate edges identified by the set-reset switch 
    [mask, clamp, edges, left_edges, right_edges] = pd.discriminator(
    time_s, 
    signal,
    dt_left=100e-9,
    dt_right=0, 
    height_th=height_th,
    method=2) 

    # find peaks in the derivative to find the steepest point
    idx = left_edges[0]+peakutils.indexes(d_signal[(mask&clamp)[1:]], .5, 3000)
    # print time_v[left_edges], time_v[idx]

    if len(idx) == 0:
        return [np.nan for _ in time_v]
    # else:
    #     idx=(mask&clamp)[idx]*idx

    idx_s = np.flipud(idx[d_signal[idx].argsort()])

    p = Parameters()
    p.add('x_offset', time_s[np.argmax(signal)] - 4e-7 - zero*dt)
    if len(idx_s) > 0:
        p.add('x_offset', time_s[idx_s[0]] - zero*dt)
    p.add('amplitude', 1, vary=1)
    result = fit_model.fit(signal,
                           x=time_s,
                           params=p,
                           weights=1 / 0.001
                           )
    n_shift = int(result.best_values['x_offset'] / dt)
    # print n_shift*dt
    return pa.shift(signal, -n_shift)

# def fit_shift(time_s, signal, fit_model):
#     """ fit the trace with a sample pulse and shift it to match the staritng
#     time
#     """
#     zero = int(len(time_s)/4)*0
#     dt = np.diff(time_s)[0]
#     d_signal = savgol_filter(np.diff(signal), 301, 1)
#     idx = peakutils.indexes(d_signal, .5, 3000)
#     idx_s = np.flipud(idx[d_signal[idx].argsort()])

#     p = Parameters()
#     p.add('x_offset', time_s[np.argmax(signal)] - 4e-7 - zero*dt)
#     if len(idx_s) > 0:
#         p.add('x_offset', time_s[idx_s[0]] - zero*dt)
#     p.add('amplitude', 1, vary=1)
#     result = fit_model.fit(signal,
#                            x=time_s,
#                            params=p,
#                            weights=1 / 0.001
#                            )
#     n_shift = int(result.best_values['x_offset'] / dt)
#     # print n_shift*dt
#     return pa.shift(signal, -n_shift)


def fit_corrected_pulse(filelist, height_th, fit_model, t_initial=None, t_final=None):
    time, _ = hps.trace_extr(filelist[0], height_th, t_initial, t_final)
    a = [fit_shift(*hps.trace_extr(file, height_th, t_initial, t_final),
                   fit_model=fit_model,
                   height_th=height_th)
         for file
         in filelist]

    # reduce to trace length to avoid edge effects
    idx_0 = int(len(time) / 30)
    v_len = len(time) - 2 * idx_0
    time = time[idx_0:idx_0 + v_len]
    a = [line[idx_0:idx_0 + v_len] for line in a]
    a_avg = np.nanmean(a, 0)
    a_std = np.nanstd(a, 0)
    # bg = np.median(a[a<0.5*np.max(a)])
    # a_fs = savgol_filter(a_avg, 101, 5)
    hist_fs = np.histogram(a_avg[a_avg<np.max(a_avg/2)], 500)
    a_avg = a_avg - hist_fs[1][np.argmax(hist_fs[0])]

    return time, a_avg, a_std 

# def fit_corrected_pulse(filelist, height_th, fit_model, t_initial=None, t_final=None):
#     time, _ = hps.trace_extr(filelist[0], height_th, t_initial, t_final)
#     a = [fit_shift(*hps.trace_extr(file, height_th, t_initial, t_final),
#                    fit_model=fit_model)
#          for file
#          in filelist]

#     # reduce to trace length to avoid edge effects
#     idx_0 = int(len(time) / 30)
#     v_len = len(time) - 2 * idx_0
#     time = time[idx_0:idx_0 + v_len]
#     a = [line[idx_0:idx_0 + v_len] for line in a]
#     a_avg = np.nanmean(a, 0)
#     a_std = np.nanstd(a, 0)
#     # bg = np.median(a[a<0.5*np.max(a)])
#     # a_fs = savgol_filter(a_avg, 101, 5)
#     hist_fs = np.histogram(a_avg[a_avg<np.max(a_avg/2)], 500)
#     a_avg = a_avg - hist_fs[1][np.argmax(hist_fs[0])]

#     return time, a_avg, a_std 