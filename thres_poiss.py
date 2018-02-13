#!/usr/bin/env python2

"""
find the optimal threshold between peaks distributions
"""
import glob
import lecroy
import numpy as np
import peakutils
import matplotlib.pyplot as plt
from thres import solve, histogram

from scipy.stats import norm

from lmfit import Parameters
from lmfit.models import GaussianModel

# from scipy.signal import savgol_filter


def traces(directory_name, bg_points=1000, indexleft=0, indexright=0):
    """ from a trace stored in binary format
    creates a waveform object.
    It returns a list with useful statistics
    """

    # reads the files into waveform objects
    filelist = np.array(glob.glob(directory_name + '*.trc'))
    if indexright == 0:
        indexleft = 0
        indexright = len(filelist)

    trcs = [lecroy.LecroyBinaryWaveform(f)
            for f
            in filelist[indexleft:indexright]]
    # uses the median as an estimator of the vertical offset of the trace
    bg = [np.median(t.mat[:bg_points, 1]) for t in trcs]

    # max height
    heights = [np.max(t.mat[:, 1] - b) for t, b in zip(trcs, bg)]

    # the area of the pulses after subtracting the vertical offset
    # dt = trcs[0].metadata['HORIZ_INTERVAL']
    areas_abs = [np.sum(np.abs(t.mat[:, 1] - b))
                 for t, b
                 in zip(trcs, bg)]
    areas = [np.sum((t.mat[:, 1] - b))
             for t, b
             in zip(trcs, bg)]

    return np.array(trcs), np.array(bg), np.array(heights),\
        np.array(areas_abs), np.array(areas)


def peaks(pnr, min_peak_sep=1, threshold=None):
    """
    Only the peaks with amplitude higher than the threshold will be detected.
    The peak with the highest amplitude is preferred to satisfy minimum
    distance constraint (in units of indices).

    :param pnr: 2D histogram, output of np.histogram()
    :param min_peak_sep: Minimum distance between each detected peak.
    :param threshold: Normalized threshold, float between [0., 1.]
    """
    # unpack the histogram into x and y values
    frequencies = pnr[0]

    # match the number of bins to the frequencies, find the step size
    x_val = pnr[1]
    step = np.diff(x_val)[0]
    x_val = x_val[:-1] + step / 2.

    # heuristic value for the threshold. Input one if you can do better!
    if threshold is None:
        threshold = np.median(frequencies) / np.max(frequencies) + .05

    # the dirty work
    indexes = peakutils.indexes(frequencies,
                                thres=threshold,
                                min_dist=int(min_peak_sep / step))
    indexes.sort()
    return x_val[indexes], frequencies[indexes]


def gauss_fit_interp(pnr, min_peak_sep, threshold=None, weighted=False, plot=False):
    """
    improve the precision in the location of the peaks by fitting them
    using a sum of Gaussian distributions
    :param pnr: 2D histogram, output of np.histogram
    :param min_peak_sep: Minimum distance between each detected peak.
    :param threshold: Normalized threshold, float between [0., 1.]
    :param weighted: if True, it associate a poissonian error to the
        frequencies
    """

    # unpack the histogram into x and y values
    frequencies = pnr[0]

    # match the number of bins to the frequencies, find the step size
    x_val = pnr[1]
    step = np.diff(x_val)[0]
    x_val = x_val[:-1] + step / 2.

    # find a first approximation of the peak location using local differences
    peaks_pos, peak_height = peaks(pnr, min_peak_sep, threshold)
    print peaks_pos
    # build a fitting model with a number of gaussian distributions matching
    # the number of peaks
    fit_model = np.sum([GaussianModel(prefix='g{}_'.format(k))
                        for k, _
                        in enumerate(peaks_pos)])

    # Generate the initial conditions for the fit
    p = Parameters()

    p.add('A', np.max(peak_height) * min_peak_sep, min=0)
    p.add('n_bar', 0.4 , min=0)
    # p.add('Delta_E', peaks_pos[-1] - peaks_pos[-2])
    p.add('g0_sigma', min_peak_sep / 5, min=0)
    p.add('sigma_p', min_peak_sep / np.sqrt(2) / np.pi, min=0)

    # Centers
    p.add('g0_center', peaks_pos[0], min=0)
    p.add('g1_center', peaks_pos[1], min=0)

    # amplitudes
    p.add('g0_amplitude'.format(k),
           min_peak_sep / np.sqrt(2),
           expr='A * exp(-n_bar)',
           min=0)

    p.add('g1_amplitude'.format(k),
           min_peak_sep / np.sqrt(2),
           expr='A * (1-exp(-n_bar))',
           min=0)

    # fixed width
    [p.add('g{}_sigma'.format(k + 1),
           min_peak_sep / np.sqrt(2) / np.pi,
           min=0,
           # expr='sigma_p * sqrt({})'.format(k + 1)
           expr='sigma_p'
           )
     for k, _
     in enumerate(peak_height[1:])]

    if weighted:
        # generates the poissonian errors, correcting for zero values
        err = np.sqrt(frequencies)
        err[frequencies == 0] = 1
        # err[frequencies < 200] = 1e5

        result = fit_model.fit(frequencies,
                               x=x_val,
                               params=p,
                               weights=1 / err,
                               # method='powell'
                               )
    else:
        result = fit_model.fit(frequencies, x=x_val, params=p)

    # amplitudes = np.array([result.best_values['g{}_amplitude'.format(k)]
    #                        for k, _
    #                        in enumerate(peaks_pos)])
    # centers = np.array([result.best_values['g{}_center'.format(k)]
    #                     for k, _
    #                     in enumerate(peaks_pos)])
    # sigmas = np.array([result.best_values['g{}_sigma'.format(k)]
    #                    for k, _
    #                    in enumerate(peaks_pos)])
    # s_vec = centers.argsort()
    if plot:
        plt.figure()
        plt.plot(x_val,frequencies)
        [plt.plot(x_val, result.eval_components(x=x_val)['g{}_'.format(k)]) for k,_ in enumerate(result.components)]
    return result

def thresholds(pnr, min_peak_sep, threshold=None, weighted=False):
    """
    halfway point between peaks
    :param pnr: 2d list of histogram
    """
    result = gauss_fit_interp(pnr, min_peak_sep, threshold, weighted)
    centers = np.array([result.best_values['g{}_center'.format(k)]
                        for k, _
                        in enumerate(result.components)])
    s_vec = centers.argsort()
    peaks_pos = centers[s_vec]

    return np.diff(peaks_pos) / 2 + peaks_pos[:-1]


def min_overlap(x0, x1, sigma0, sigma1, samples=1000):
    """
    return threshold value that minimizes the overlap between the given
    gaussian distributions
    """
    x_vec = np.linspace(x0, x1, samples)
    noise = 1 - norm.cdf(x_vec, loc=x0, scale=sigma0)
    sgn = norm.cdf(x_vec, loc=x1, scale=sigma1)
    snr = sgn + noise
    sgn[np.argmin(snr)]
    noise[np.argmin(snr)]
    return x_vec[np.argmin(snr)]


def thresholds_N(pnr, min_peak_sep, threshold=None, weighted=False):
    """
    thresholds between peaks assuming gaussian distributions
    """
    result = gauss_fit_interp(pnr, min_peak_sep, threshold, weighted)

    centers = np.array([result.best_values['g{}_center'.format(k)]
                        for k, _
                        in enumerate(result.components)])
    sigmas = np.array([result.best_values['g{}_sigma'.format(k)]
                       for k, _
                       in enumerate(result.components)])
    s_vec = centers.argsort()

    xs = centers[s_vec]
    ss = sigmas[s_vec]
    N = len(xs)
    return [min_overlap(xs[j], xs[j + 1], ss[j], ss[j + 1])
            for j
            in range(N - 1)]

def thresholds_N_unnormed(pnr, min_peak_sep, threshold=None, weighted=False):
    """
    thresholds between peaks assuming gaussian distributions
    unlike thresholds_N, does not take each gaussian as being normalised on its own,
    but also takes into account the distribution of counts in each gaussian distribution.
    :param pnr: 2D histogram, output of np.histogram. 
    """
    x_val = pnr[1]
    step = np.diff(x_val)[0]
    x_val = x_val[:-1] + step / 2.

    # note that all functions here return only properties of g1 and above.
    # g0 is not computed since its distribution is not computed as gaussian using our discriminator

    result = gauss_fit_interp(pnr, min_peak_sep, threshold, weighted)
    
    comps = result.eval_components(x=x_val)

    centers = np.array([result.best_values['g{}_center'.format(k)]
                        for k, _
                        in enumerate(result.components)])

    N = len(centers)

    return [solve(x_val,
        comps['g{}_'.format(j)],
        comps['g{}_'.format(j+1)],
        (x_val>centers[j])&(x_val<centers[j+1]))
    for j
    in range(N-1)]


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    data_folder = '../data/'
    single_pulse_folder = '20161011_g2_signal_unheralded_singlesrate_5.6k/'
    bg = 2000

    trc_single = traces(data_folder + single_pulse_folder, bg)
    pnr = np.histogram(trc_single[2], 200)
    result = gauss_fit_interp(pnr, .7e-8, weighted=True)
    th = thresholds_N(pnr, .7e-8, weighted=True)
    print(result.fit_report())
    print(th)
    result.plot()
    plt.show()
