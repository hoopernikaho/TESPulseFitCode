#!/usr/bin/env python
""" test script for SET-RESET discriminator
"""

import glob
import matplotlib.pyplot as plt
import numpy as np
import random

from discriminator import *
from discriminator_edges import *
from trace_param import *

data_dir = '/mnt/disk1/analysis/20170523_TES5_75mK_95uA_trise_115ns_results/peak_disc_challenging_traces/'
filelist = glob.glob(data_dir + '*.trc')
high_th = 0.00773176320829
low_th = 0.00183298303024

filename = filelist[7]
# filename = 'interesting_traces/C2doubles02442.trc'
print(filename)

signal = trace_extr(filename)


# backward masks
mask_b = np.flipud(disc_peak(np.flipud(signal), high_th, 0))
mask_b_edge = np.flipud(disc_edges(np.flipud(signal), high_th, 0))

# forward masks
mask_f = disc_peak(signal, high_th, low_th)
mask_f_edge = disc_edges(signal, high_th, low_th)

# final mask, a combination of the two previous ones
mask = mask_f + mask_b + mask_b_edge + mask_f_edge

area = np.sum(np.abs(signal[mask]))
print('area: {}'.format(area))
print('max height outside of peaks: {}'.format(np.max(signal[~mask])))

plt.figure()
# plt.clf()
plt.plot(signal / np.max(signal))
plt.plot([0, 5000], [high_th / np.max(signal), high_th / np.max(signal)], '--')
plt.plot([0, 5000], [low_th / np.max(signal), low_th / np.max(signal)], '--')
plt.plot(mask_b, label='Backward')
plt.plot(mask_f, label='Forward')
plt.plot(mask_b_edge, label='Backward_edge')
plt.plot(mask_f_edge, label='Forward_edge')
plt.plot(mask)
plt.legend()

plt.show()
