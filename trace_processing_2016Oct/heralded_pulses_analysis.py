import lecroy
import numpy as np

from scipy.signal import savgol_filter


def find_idx(time_v, t0):
    return np.argmin(np.abs(time_v - t0))


def find_bg(signal):
    freq, ampl = np.histogram(signal, 200)
    freq_f = savgol_filter(freq, 11, 3)
    return ampl[np.argmax(freq_f)]


def param_extr(filename, t_initial=None, t_final=None, h_th=-10):
    """extract relevant parameters from a trace stored in a file
    """
    trc = lecroy.LecroyBinaryWaveform(filename)
    time = trc.mat[:, 0]
    signal = trc.mat[:, 1]
    idx_0 = 0
    idx_1 = -1
    if t_initial is not None:
        idx_0 = find_idx(time, t_initial)
    if t_final is not None:
        idx_1 = find_idx(time, t_final)
    time = time[idx_0:idx_1]
    signal = signal[idx_0:idx_1]

    # add check for length of bg_points against length of signal
    # bg_points = np.min([bg_points, len(signal)])
    # bg = np.median(signal[:bg_points])

    bg = find_bg(signal)
    signal = signal - bg
    height = np.max(signal)
    area_th = np.sum(signal[signal > h_th])
    area_abs = np.sum(np.abs(signal))

    return np.array([area_th, area_abs, height, bg])


def trace_extr(filename, t_initial=None, t_final=None):
    """extract relevant parameters from a trace stored in a file
    """
    trc = lecroy.LecroyBinaryWaveform(filename)
    time = trc.mat[:, 0]
    signal = trc.mat[:, 1]
    idx_0 = 0
    idx_1 = -1
    if t_initial is not None:
        idx_0 = find_idx(time, t_initial)
    if t_final is not None:
        idx_1 = find_idx(time, t_final)
    time = time[idx_0:idx_1]
    signal = signal[idx_0:idx_1]

    # add check for length of bg_points against length of signal
    # bg_points = np.min([bg_points, len(signal)])
    # bg = np.median(signal[:bg_points])
    bg = find_bg(signal)
    signal = signal - bg

    return np.array(time), np.array(signal)


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


# def time_offset(time_v, trace):
#     """ shift a trace so that the steepest point is at time 0
#     """
#     # start by finding the max of the derivative
#     signal = trace
#     dt = np.diff(time_v)[0]

#     # derivative of the signal, soothed
#     d_signal = savgol_filter(np.diff(signal), 301, 3)

#     # find peaks in the derivative to find the steepest point
#     idx = peakutils.indexes(d_signal, .5, 3000)
#     if len(idx) == 0:
#         return [np.nan for _ in time_v]
#     idx_s = np.flipud(idx[d_signal[idx].argsort()])[0]

#     try:
#         time_p = peakutils.interpolate(time_v[:-1], d_signal, [idx_s])[0]
#     except (RuntimeError, ValueError) as e:
#         time_p = time_v[idx_s]

#     n_shift = int(time_p / dt)
#     return shift(trace, - n_shift)


# def trace_ave(filelist, t_initial, t_final, bg_points=1000, smooth=201):
#     time, _ = trace_extr(filelist[0], t_initial, t_final, bg_points=1000)

#     a = [time_offset(*trace_extr(file, t_initial, t_final, bg_points=1000))
#          for file
#          in filelist]
#     # reduce to trace length to avoid edge effects
#     idx_0 = int(len(time) / 30)
#     v_len = len(time) - 2 * idx_0
#     time = time[idx_0:idx_0 + v_len]
#     a = [line[idx_0:idx_0 + v_len] for line in a]

#     return time,\
#        savgol_filter(np.nanmean(a, 0), smooth, 3)


if __name__ == '__main__':
    directory_name = data_folder + trace_folder
    filelist = np.array(glob.glob(directory_name + '*.trc'))

    # we limit the temporal length of the traces
    t_initial = -1.38e-6
    t_final = 8.82e-6
    min_peak_sep = 5
    height_th = 0.01

    # reads the traces one by one and extract relevant parameters into
    # numpy structure
    data = np.array([param_extr(f, t_initial, t_final, h_th=height_th)
                     for f
                     in filelist])

    areas = data[:, 0]
    # use the area to count the number of photons
    pnr = np.histogram(areas, 400)

    # remove the first bin.
    # because of the threshold filtering it only makes life
    # complicated
    pnr = [pnr[0][1:], pnr[1][1:]]

    # find the thresholds by fitting the distribution with gaussian peaks
    th = thres_poiss.thresholds_N(pnr, min_peak_sep, weighted=True)

    # select only 1-photon traces
    mask_1ph = (areas > pnr[1][1]) & (areas < th[0])

    # select only 2-photon traces
    mask_2ph = (areas > th[0]) & (areas < th[1])
