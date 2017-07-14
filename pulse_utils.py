import discriminator as disc
import discriminator_edges as disc_edges
import matplotlib.pyplot as plt
import lecroy
import numpy as np

def time_vector(filename):
    trc = lecroy.LecroyBinaryWaveform(filename)
    time = trc.mat[:, 0]
    return time

def disc_peak_full(signal, high_th, low_th, offset):
    """from trace to mask

    high level function: from the trace and the SET-RESET threshold,
        generates a mask singling out the peaks
    :param signal: trace
    :type signal: array of float
    :param high_th: SET threshold
    :type high_th: float
    :param low_th: RESET threshold
    :type low_th: float
    :returns: boolean mask
    :rtype: numpy array of bool
    """
    l_signal = len(signal)
    starts_, stops_ = disc.intervals_no_edges(np.flipud(signal), high_th, low_th)

    stops = l_signal-starts_
    starts = l_signal-stops_

    # [plt.axvline(s, color='green') for s in starts]
    # [plt.axvline(s) for s in stops]

    stops = stops + offset
    stops = stops[(stops <= l_signal)]
    try:
        """
        checks if the extension of the last pulse overlaps an omitted partial pulse
        if true, omit both pulses
        """
        last_stop = stops[-1]
        hi_cross = disc.find_crossing(signal[last_stop-offset:last_stop], high_th)
        if len(hi_cross)>0:
            stops = np.delete(stops,0)
    except:
        pass #no last stop exists

    starts = starts[:len(stops)]
    mask = disc.create_mask_for_peak(len(signal), starts, stops)
    
    return np.array(mask)

# def disc_peak_full(signal, height_th, noise_th):
#     forward = disc_peak_extend(signal, height_th, 0, 450)
#     backward = np.flipud(disc.disc_peak(np.flipud(signal), height_th, 0))
#     mask = forward + backward
#     return mask

def disc_edges_full(signal, heigh_th, low_th):
    forward = disc_edges.disc_edges(signal, heigh_th, low_th)
    backward = np.flipud(disc_edges.disc_edges(np.flipud(signal), heigh_th, low_th))
    mask = forward + backward
    return mask

def rise_time(time,signal,plot=True):
    """returns 10% to 90% rise timing"""
    amplitude = np.max(signal)
    idxpeak = np.argmax(signal)
    def find_idx(array,value):
        array=np.array(array)
        return np.argmin(np.abs(array-value))
    t10 = time[find_idx(signal[:idxpeak],0.1*amplitude)]
    t90 = time[find_idx(signal[:idxpeak],0.9*amplitude)]
    risetime = t90-t10
    if plot:
        plt.figure()
        plt.plot(time,signal)
        plt.axvline(t10,linestyle='--')
        plt.axvline(t90,linestyle='--')
    return risetime
