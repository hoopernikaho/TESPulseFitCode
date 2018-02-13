from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from lmfit import Model, Parameters
import math
from itertools import compress
import trace_param as trcp
import pulse_utils as pu

from itertools import compress

def find_crossing(vec, th, edge):
    """returns index of vec crossing threshold th

    :param vec: signal
    :type vec: list of floats
    :param th: threshold
    :type th: float
    :param edge: 'left' or 'right'
    :type edge: string
    :returns: list of index of crossings
    :rtype: list of int
    """

    vec = vec - th
    if edge=='left':
        v_bool = (vec[1:]-vec[:-1]) > 0
    if edge=='right': 
        v_bool = (vec[1:]-vec[:-1]) < 0
    return np.array(list(compress(xrange(len(v_bool)), v_bool)))

def fit_two_cw(time, signal,
            two_pulse_fit,
            pulse_params,
            height_th,
            sigma0):
    # signal = signal - np.median(signal[signal<height_th])
    (sum_a,sum_mu,sum_b,sum_tau,diff_a,diff_b,diff_tau) = pulse_params
    mask = pu.disc_peak_full(signal,height_th,0,0)
    # signal = signal - trcp.find_bg(signal[~mask])
    # mask = pu.disc_peak_full(signal,height_th,0,0)
    # plt.plot(time,mask*np.max(signal),linestyle='--')
    left_edges = find_crossing(mask, 0.5, 'left')
    right_edges = find_crossing(mask, 0.5, 'right')

    one_x_offset_init_min = time[left_edges[0]]
    # one_x_offset_init_max = time[right_edges[0]]
    one_x_offset_init_max = time[left_edges[0]+np.argmax(signal[left_edges[0]:right_edges[0]])]

    if len(left_edges) == 1:
        two_x_offset_init_min = one_x_offset_init_min
        two_x_offset_init_max = time[right_edges[0]]

        one_x_offset_init = one_x_offset_init_min
        two_x_offset_init = two_x_offset_init_max - 0.5e-6
    else:
        two_x_offset_init_min = time[left_edges[1]]
        two_x_offset_init_max = time[left_edges[1]+np.argmax(signal[left_edges[1]:right_edges[1]])]

        one_x_offset_init = (one_x_offset_init_min+one_x_offset_init_max)/2 
        two_x_offset_init = (two_x_offset_init_min+two_x_offset_init_max)/2

    one_amplitude_init = sum_mu/2
    two_amplitude_init = sum_mu/2

    """Compulsory LMFIT for both overlapping and non-overlapping cases"""
    p = Parameters()
    p.add('one_x_offset', 
          one_x_offset_init, 
          min=one_x_offset_init_min, 
          max=one_x_offset_init_max, 
          vary=True)
    
    p.add('two_x_offset', 
          two_x_offset_init, 
          min=two_x_offset_init_min, 
          max=two_x_offset_init_max, 
          vary=True)

    p.add('sum_amplitudes', 
        (one_amplitude_init+two_amplitude_init)*1.01, #not sure why, but some assymetry is required... sum_mu can't be avg of sum_a and sum_b
        min=sum_a, 
        max=sum_b, 
        vary=True)

    p.add('diff_amplitudes', 
        one_amplitude_init-two_amplitude_init, 
        min=diff_a, 
        max=diff_b, 
        vary=True)

    p.add('one_amplitude', 
          one_amplitude_init,
          expr= '(sum_amplitudes + diff_amplitudes)/2',
          vary=True) #warning: max >= 2 causes n=2 & noise to be fitted on a tau~0 2ph trace. 

    p.add('two_amplitude', 
          two_amplitude_init,
          expr= '(sum_amplitudes - diff_amplitudes)/2',
          vary=True) #warning: max >= 2 causes n=2 & noise to be fitted on a tau~0 2ph trace. 

    result = two_pulse_fit.fit(np.array(signal),
                               x=np.array(time),
                               params=p,
                               weights=1/sigma0,
                               method='powell'
                               )
    return result