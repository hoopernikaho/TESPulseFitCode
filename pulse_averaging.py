import heralded_pulses_analysis as hps
import numpy as np
import peakutils

from lmfit import Model
from lmfit import Parameters

from scipy.signal import savgol_filter

def find_bg(signal):
    freq, ampl = np.histogram(signal, 50)
    freq_f = savgol_filter(freq, 11, 3)
    return ampl[np.argmax(freq_f)]

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


def trace_ave(filelist, t_initial=None, t_final=None, smooth=201):
    time, _ = hps.trace_extr(filelist[0], t_initial, t_final)

    a = [time_offset(*hps.trace_extr(file, t_initial, t_final))
         for file
         in filelist]
    # reduce to trace length to avoid edge effects

    idx_0 = int(len(time) / 30)
    v_len = len(time) - 2 * idx_0
    time = time[idx_0:idx_0 + v_len]
    a = [line[idx_0:idx_0 + v_len] for line in a]

    return time, savgol_filter(np.nanmean(a, 0), smooth, 3)


def fit_shift(time_s, signal, fit_model):
    """ fit the trace with a sample pulse and shift it to macth the staritng
    time
    """
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


def fit_corrected_pulse(filelist, fit_model, t_initial=None, t_final=None):
    time, _ = hps.trace_extr(filelist[0], t_initial, t_final)
    a = [fit_shift(*hps.trace_extr(file, t_initial, t_final),
                   fit_model=fit_model)
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

if __name__ == '__main__':

    time_p, signal_p = trace_ave(filelist[mask_1ph], t_initial, t_final)

    def one_pulse(x, x_offset=0, amplitude=1):
        """convert the sample single photon pulse into a function
        that can be used in a fit
        """
        x = x - x_offset
        return amplitude * np.interp(x, time_p, signal_p)

    fit_model = Model(one_pulse)

    

    time_f, signal_f = fit_corrected_pulse(filelist[mask_1ph], fit_model,
                                           t_initial,
                                           t_final)

    signal_fs = savgol_filter(signal_f, 101, 5)
    hist_fs = np.histogram(signal_fs, 300)
    signal_fs = signal_fs - hist_fs[1][np.argmax(hist_fs[0])]

    # results_folder = ('/workspace/projects/TES/analysis/20161116_TES5_20MHz_bwl_diode_n012_height_optimised_results/')
    # save average pulse
    np.save(results_directory + 'ph1_model.npy',
            np.array(zip(time_f, signal_fs)))
