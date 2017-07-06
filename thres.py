#!/usr/bin/env python2

"""
find the optimal threshold between peaks distributions without assuming distributions are Poisson distributed.
borrows some functions from thres_poiss.
"""
import glob
import lecroy
import numpy as np
import peakutils
import matplotlib.pyplot as plt
import thres_poiss as tp
from scipy.stats import norm

from lmfit import Parameters
from lmfit.models import GaussianModel, RectangleModel

"""
A version of gauss_fit_poiss_ph that begins fitting from the minimum between n=0 and n=1 area distributions.
This version is for lower height thresholds of the SET-RESET switch, used to discrminate pulse regions during area calculations. 
Lower height thresholds result in accidentals, hence the trailing n=0 tail.
"""
def find_idx(array, value):
    return np.argmin(np.abs(array-value))

def gauss_fit_poiss_ph_region(pnr, min_peak_sep, threshold=None, weighted=False, plot=False):
    """
    improve the precision in the location of the peaks by fitting them
    using a sum of Gaussian distributions
    'poiss_ph' naming convention because it only fits the amplitudes to poissonian stats for n>=1
    :param pnr: 2D histogram, output of np.histogram, assumes it's a rectangular function (first bin) + sum of gaussians, but discards the first bin.
    :param min_peak_sep: Minimum distance between each detected peak.
    :param threshold: Normalized threshold, float between [0., 1.]
    :param weighted: if True, it associate a poissonian error to the
        frequencies
    """

    # unpack the histogram into x and y values
    # first bin corresponds to n=0 traces with no photon detection events, thus having zero area.
    f0 = pnr[0][0] #n=0 freq
    frequencies = pnr[0][1:]

    # match the number of bins to the frequencies, find the step size
    x_val = pnr[1][1:]
    step = np.diff(x_val)[0]
    x0 = pnr[1][0] + step / 2 #n=0 bin
    x_val = x_val[:-1] + step / 2.

    # find a first approximation of the peak location using local differences
    peaks_pos, peak_height = tp.peaks([frequencies,x_val], min_peak_sep, threshold)
    print 'est peak pos = {}\nest peak hts = {}'.format(peaks_pos, peak_height)

    # detect min between n=0 and n=1
    th01 = x_val[find_idx(x_val,peaks_pos[0]) + np.argmin(frequencies[(x_val>peaks_pos[0])&(x_val<peaks_pos[1])])]
    print 'th01 = {}'.format(th01)

    # constrain fitting region:
    x_val_ = x_val
    frequencies_=frequencies

    mask = x_val>th01
    x_val = x_val[mask]
    frequencies = frequencies[mask]
    peaks_pos = peaks_pos[peaks_pos>th01]
    peak_height = peak_height[peaks_pos>th01]

    # build a fitting model with a number of gaussian distributions matching
    # the number of peaks
    fit_model = np.sum([GaussianModel(prefix='g{}_'.format(k+1))
                        for k, _
                        in enumerate(peaks_pos)])

    # Generate the initial conditions for the fit
    p = Parameters()
    p.add('n_bar', 0.2)

    p.add('A', np.max(peak_height) * min_peak_sep)
    p.add('Delta_E', peaks_pos[-1] - peaks_pos[-2])
    # p.add('g1_sigma', min_peak_sep / 5, min=0)
    p.add('sigma_p', min_peak_sep / np.sqrt(2) / np.pi, min=0)
    # n>=1 Centers
    p.add('g1_center', peaks_pos[0], min=0)
    p.add('g2_center', peaks_pos[1], min=0)
    [p.add('g{}_center'.format(k+3),
           j,
           # expr='g{}_center + Delta_E'.format(k + 1)
           )
     for k, j
     in enumerate(peaks_pos[2:])]

    # n>=1 amplitudes
    [p.add('g{}_amplitude'.format(k+1),
           j * min_peak_sep / np.sqrt(2),
           expr='A * exp(-n_bar) * n_bar**{} / factorial({})'.format(k+1, k+1),
           min=0)
     for k, j
     in enumerate(peak_height)]

    # n>=1 fixed widths
    [p.add('g{}_sigma'.format(k + 1),
           min_peak_sep / np.sqrt(2) / np.pi,
           min=0,
           expr='sigma_p * sqrt({})'.format(k + 1)
           # expr='sigma_p'
           )
     for k, _
     in enumerate(peak_height)]

    if weighted:
        # generates the poissonian errors, correcting for zero values
        err = np.sqrt(frequencies)
        err[frequencies == 0] = 1

        result = fit_model.fit(frequencies,
                               x=x_val,
                               params=p,
                               weights=1 / err,
                               method='powell'
                               )
    else:
        result = fit_model.fit(frequencies, x=x_val, params=p)

    if plot:
        plt.figure(figsize=(10,5))
        plt.errorbar(pnr[1][:-1]+step/2,pnr[0],yerr=np.sqrt(pnr[0]),linestyle='',ecolor='black',color='black')
        [plt.plot(x_val, result.eval_components(x=x_val)['g{}_'.format(k+1)]) for k,_ in enumerate(result.components)]
        plt.axvline(th01,color='black', label='th01')
        plt.legend()
        print result.fit_report()
    return result

"""
Implemented code that 
-does not take into account the n=0 delta peak...
-take into account the amplitude contraints to follow Poission
-uses gauss_fit_results to extract centers and sigmas
"""
def gauss_fit_poiss_ph(pnr, min_peak_sep, threshold=None, weighted=False, plot=False):
    """
    improve the precision in the location of the peaks by fitting them
    using a sum of Gaussian distributions
    'poiss_ph' naming convention because it only fits the amplitudes to poissonian stats for n>=1
    :param pnr: 2D histogram, output of np.histogram, assumes it's a rectangular function (first bin) + sum of gaussians, but discards the first bin.
    :param min_peak_sep: Minimum distance between each detected peak.
    :param threshold: Normalized threshold, float between [0., 1.]
    :param weighted: if True, it associate a poissonian error to the
        frequencies
    """

    # unpack the histogram into x and y values
    # first bin corresponds to n=0 traces with no photon detection events, thus having zero area.
    f0 = pnr[0][0] #n=0 freq
    frequencies = pnr[0][1:]

    # match the number of bins to the frequencies, find the step size
    x_val = pnr[1][1:]
    step = np.diff(x_val)[0]
    x0 = pnr[1][0] + step / 2 #n=0 bin
    x_val = x_val[:-1] + step / 2.

    # find a first approximation of the peak location using local differences
    peaks_pos, peak_height = tp.peaks([frequencies,x_val], min_peak_sep, threshold)
    print peaks_pos, peak_height

    # build a fitting model with a number of gaussian distributions matching
    # the number of peaks
    fit_model = np.sum([GaussianModel(prefix='g{}_'.format(k+1))
                        for k, _
                        in enumerate(peaks_pos)])

    # Generate the initial conditions for the fit
    p = Parameters()
    p.add('n_bar', 0.2)

    p.add('A', np.max(peak_height) * min_peak_sep)
    p.add('Delta_E', peaks_pos[-1] - peaks_pos[-2])
    # p.add('g1_sigma', min_peak_sep / 5, min=0)
    p.add('sigma_p', min_peak_sep / np.sqrt(2) / np.pi, min=0)
    # n>=1 Centers
    p.add('g1_center', peaks_pos[0], min=0)
    p.add('g2_center', peaks_pos[1], min=0)
    [p.add('g{}_center'.format(k+3),
           j,
           expr='g{}_center + Delta_E'.format(k + 1)
           )
     for k, j
     in enumerate(peaks_pos[2:])]

    # n>=1 amplitudes
    [p.add('g{}_amplitude'.format(k+1),
           j * min_peak_sep / np.sqrt(2),
           expr='A * exp(-n_bar) * n_bar**{} / factorial({})'.format(k+1, k+1),
           min=0)
     for k, j
     in enumerate(peak_height)]

    # n>=1 fixed widths
    [p.add('g{}_sigma'.format(k + 1),
           min_peak_sep / np.sqrt(2) / np.pi,
           min=0,
           expr='sigma_p * sqrt({})'.format(k + 1)
           # expr='sigma_p'
           )
     for k, _
     in enumerate(peak_height)]

    if weighted:
        # generates the poissonian errors, correcting for zero values
        err = np.sqrt(frequencies)
        err[frequencies == 0] = 1

        result = fit_model.fit(frequencies,
                               x=x_val,
                               params=p,
                               weights=1 / err,
                               method='powell'
                               )
    else:
        result = fit_model.fit(frequencies, x=x_val, params=p)
    return result

"""
Some older code that 
-takes completely into account the n=0 delta peak
-take into account the amplitude contraints to follow Poission
"""

def gauss_fit_interp_disc(pnr, min_peak_sep, threshold=None, weighted=False):
    """
    improve the precision in the location of the peaks by fitting them
    using a sum of Gaussian distributions
    'disc' naming convention because it takes into account the n=0 rectangular function 
    in the first bin as a result of calculting 0 area in a trace where the discriminator does not trigger.
    :param pnr: 2D histogram, output of np.histogram(area under discriminator), assumes it's a rectangular function + sum of gaussians
    :param min_peak_sep: Minimum distance between each detected peak.
    :param threshold: Normalized threshold, float between [0., 1.]
    :param weighted: if True, it associate a poissonian error to the
        frequencies
    """

    # unpack the histogram into x and y values
    # first bin corresponds to n=0 traces with no photon detection events, thus having zero area.
    f0 = pnr[0][0] #n=0 freq
    frequencies = pnr[0][1:]

    # match the number of bins to the frequencies, find the step size
    x_val = pnr[1][1:]
    step = np.diff(x_val)[0]
    x0 = pnr[1][0] + step / 2 #n=0 bin
    x_val = x_val[:-1] + step / 2.

    # find a first approximation of the peak location using local differences
    peaks_pos, peak_height = tp.peaks([frequencies,x_val], min_peak_sep, threshold)
    print peaks_pos, peak_height

    # build a fitting model with a number of gaussian distributions matching
    # the number of peaks
    fit_model = RectangleModel(prefix='g0_',form='linear') + np.sum([GaussianModel(prefix='g{}_'.format(k+1))
                        for k, _
                        in enumerate(peaks_pos)])

    # Generate the initial conditions for the fit
    p = Parameters()
    p.add('n_bar', 0.2)
    # n=0 params
    p.add('g0_amplitude', f0, expr='{}*exp(-n_bar)'.format(np.sum(pnr[0])))
    p.add('g0_center1',x0-step/2,vary=0)
    p.add('g0_center2',x0+step/2,vary=0)
    p.add('g0_sigma1',0)
    p.add('g0_sigma2',0)

    p.add('A', np.max(peak_height) * min_peak_sep)
    # p.add('Delta_E', peaks_pos[-1] - peaks_pos[-2])
    # p.add('g1_sigma', min_peak_sep / 5, min=0)
    p.add('sigma_p', min_peak_sep / np.sqrt(2) / np.pi, min=0)
    # n>=1 Centers
    p.add('g1_center', peaks_pos[0], min=0)
    p.add('g2_center', peaks_pos[1], min=0)
    [p.add('g{}_center'.format(k+3),
           j,
           # expr='g{}_center + Delta_E'.format(k + 1)
           )
     for k, j
     in enumerate(peaks_pos[2:])]

    # n>=1 amplitudes
    [p.add('g{}_amplitude'.format(k+1),
           j * min_peak_sep / np.sqrt(2),
           expr='A * exp(-n_bar) * n_bar**{} / factorial({})'.format(k+1, k+1),
           min=0)
     for k, j
     in enumerate(peak_height)]

    # n>=1 fixed widths
    [p.add('g{}_sigma'.format(k + 1),
           min_peak_sep / np.sqrt(2) / np.pi,
           min=0,
           # expr='sigma_p * sqrt({})'.format(k + 1)
           expr='sigma_p'
           )
     for k, _
     in enumerate(peak_height)]

    if weighted:
        # generates the poissonian errors, correcting for zero values
        err = np.sqrt(pnr[0])
        err[pnr[0] == 0] = 1

        result = fit_model.fit(pnr[0],
                               x=pnr[1][:-1]+step/2,
                               params=p,
                               weights=1 / err,
                               method='powell'
                               )
    else:
        result = fit_model.fit(pnr[0], x=pnr[1][:-1]+step/2, params=p)

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
    return result

def gauss_fit_disc_results(pnr, min_peak_sep, threshold=None, weighted=False):
    """
    extract results of fitting to gaussian distributions
    """
    result = gauss_fit_interp_disc(pnr, min_peak_sep, threshold, weighted)

    centers = np.array([result.best_values['g{}_center'.format(k+1)]
                        for k, _
                        in enumerate(result.components[1:])])

    sigmas = np.array([result.best_values['g{}_sigma'.format(k+1)]
                       for k, _
                       in enumerate(result.components[1:])])
    s_vec = centers.argsort()

    xs = centers[s_vec]
    ss = sigmas[s_vec]
    N = len(xs)
    return np.array(centers), np.array(sigmas)

"""
Some older code that 
-does not take into account the n=0 delta peak...
-does not take into account the amplitude contraints to follow Poission
"""

def gauss_fit(pnr, min_peak_sep, threshold=None, weighted=False):
    """
    improve the precision in the location of the peaks by fitting them
    using a sum of Gaussian distributions
    NOTE: unlike gauss_fit_interp, it does not assume a Poissonian distribution
    :param pnr: 2D histogram, output of np.histogram, assumes it's a sum of gaussians
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
    peaks_pos, peak_height = tp.peaks(pnr, min_peak_sep, threshold)

    # build a fitting model with a number of gaussian distributions matching
    # the number of peaks
    fit_model = np.sum([GaussianModel(prefix='g{}_'.format(k))
                        for k, _
                        in enumerate(peaks_pos)])

    # Generate the initial conditions for the fit
    p = Parameters()

    p.add('A', np.max(peak_height) * min_peak_sep)
    p.add('g0_sigma', min_peak_sep / 5, min=0)
    p.add('sigma_p', min_peak_sep / np.sqrt(2) / np.pi, min=0)

    # Centers
    p.add('g0_center', peaks_pos[0], min=0)
    p.add('g1_center', peaks_pos[1], min=0)
    [p.add('g{}_center'.format(k + 2),
           j,
           # expr='g{}_center + Delta_E'.format(k + 1)
           )
     for k, j
     in enumerate(peaks_pos[2:])]

    # amplitudes
    [p.add('g{}_amplitude'.format(k),
           j * min_peak_sep / np.sqrt(2),
           # expr='A * exp(-n_bar) * n_bar**{} / factorial({})'.format(k, k),
           min=0)
     for k, j
     in enumerate(peak_height)]

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

        result = fit_model.fit(frequencies,
                               x=x_val,
                               params=p,
                               weights=1 / err,
                               # method='nelder'
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
    return result

def gauss_fit_results(pnr, min_peak_sep, threshold=None, weighted=False, plot=False):
    """
    extract results of fitting to gaussian distributions
    """
    result = gauss_fit_poiss_ph_region(pnr, min_peak_sep, threshold, weighted, plot)

    centers = np.array([result.best_values['g{}_center'.format(k+1)]
                        for k, _
                        in enumerate(result.components)])
    sigmas = np.array([result.best_values['g{}_sigma'.format(k+1)]
                       for k, _
                       in enumerate(result.components)])
    s_vec = centers.argsort()

    xs = centers[s_vec]
    ss = sigmas[s_vec]
    N = len(xs)
    return np.array(centers), np.array(sigmas)

# def thresholds_N(pnr, min_peak_sep, threshold=None, weighted=False):
#     """
#     thresholds between peaks assuming gaussian distributions
#     """
#     result = gauss_fit(pnr, min_peak_sep, threshold, weighted)

#     centers = np.array([result.best_values['g{}_center'.format(k)]
#                         for k, _
#                         in enumerate(result.components)])
#     sigmas = np.array([result.best_values['g{}_sigma'.format(k)]
#                        for k, _
#                        in enumerate(result.components)])
#     s_vec = centers.argsort()

#     xs = centers[s_vec]
#     ss = sigmas[s_vec]
#     N = len(xs)
#     return [tp.min_overlap(xs[j], xs[j + 1], ss[j], ss[j + 1])
#             for j
#             in range(N - 1)]

def min_overlap(x0, x1, sigma0, sigma1, samples=1000):
    """
    return threshold value that minimizes the overlap between the given
    gaussian distributions
    """
    x_vec = np.linspace(x0, x1, samples)
    noise = 1 - norm.cdf(x_vec, loc=x0, scale=sigma0)
    sgn = norm.cdf(x_vec, loc=x1, scale=sigma1)
    snr = sgn + noise
    print 'at threshold {} between {} and {}:'.format(x_vec[np.argmin(snr)], x0, x1)
    print 'prob signal lost = {}'.format(sgn[np.argmin(snr)])
    print 'prob noise enters = {}'.format(noise[np.argmin(snr)])

    return x_vec[np.argmin(snr)]

def thresholds_N(pnr, min_peak_sep, threshold=None, weighted=False):
    """
    thresholds between peaks assuming gaussian distributions
    """
    result = gauss_fit_poiss_ph_region(pnr, min_peak_sep, threshold, weighted)

    centers = np.array([result.best_values['g{}_center'.format(k+1)]
                        for k, _
                        in enumerate(result.components)])
    sigmas = np.array([result.best_values['g{}_sigma'.format(k+1)]
                       for k, _
                       in enumerate(result.components)])
    s_vec = centers.argsort()

    xs = centers[s_vec]
    ss = sigmas[s_vec]
    N = len(xs)
    return [min_overlap(xs[j], xs[j + 1], ss[j], ss[j + 1])
            for j
            in range(N - 1)]