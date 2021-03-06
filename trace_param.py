#!/usr/bin/env python
""" Reads LeCroy trc binay files

Convert binary .trc files generated by LeCroy oscilloscopes into python arrays
based on
https://github.com/yetifrisstlama/readTrc
"""
from readTrc import *
import numpy as np
from scipy.signal import savgol_filter
# import pyximport
# pyximport.install()

try:
    import htest
    cyt = True
except ImportError:
    cyt = False

import matplotlib.pyplot as plt

def trace_extr(filename, bg_correction=True):
    """ extract the trace with option vertical offset correction

    :param filename: .trc file containing the trace
    :type filename: file path
    :param bg_correction: offset correction flag, defaults to True
    :type bg_correction: bool, optional
    :returns: trace
    :rtype: array of float
    """

    _, signal, _ = readTrc(filename)
    # plt.figure(); plt.hist(signal,21,label='before',range=(-0.01,0.01)); plt.xlim(-0.01,0.01); plt.legend(); plt.show()
    if bg_correction:
        # if cyt:
            # return signal - find_bg_c(signal)
        # plt.figure(); plt.hist(signal - find_bg(signal,21),21,label='after',range=(-0.01,0.01)); plt.xlim(-0.01,0.01); plt.legend(); plt.show()
        return signal - find_bg(signal)
    return signal


def full_extr(filename):

    time, signal, d = readTrc(filename)
    return time, signal, d


def find_bg(signal, bins=51):
    """find vertical offsets

    :param signal: trace
    :type signal: array of float
    :param bins: number of bins, defaults to 21
    :type bins: int, optional
    :returns: peak of the histogram
    :rtype: float
    """
    freq, ampl = np.histogram(signal, bins, range=(-0.01,0.01))
    freq_f = savgol_filter(freq,9,1)

    return (ampl[:-1]+(ampl[1]-ampl[0])/2)[np.argmax(freq_f)]


def find_bg_c(signal, bins=61):
    """find vertical offsets

    :param signal: trace
    :type signal: array of float
    :param bins: number of bins, defaults to 21
    :type bins: int, optional
    :returns: peak of the histogram
    :rtype: float
    """
    # print 'find_bg_c'
    h = htest.hist1d(bins, np.min(signal), np.max(signal))
    # h = htest.hist1d(bins, -0.01, 0.01)
    h.fillcy(signal, np.ones(len(signal)))
    # h.fillcywithcall(signal, np.ones(bins))
    return h.xaxis.values()[np.argmax(h.data)]


if __name__ == '__main__':
    import glob

    # path = "./"
    filelist = glob.glob(path + "*.trc")

    data = [trace_extr(f) for f in filelist]

