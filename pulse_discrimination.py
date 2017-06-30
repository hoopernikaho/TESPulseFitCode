#!/usr/bin/env python
"""Collection of functions used to discriminate between
pulses of different areas
"""

import numpy as np
import matplotlib.pyplot as plt
# import pulse_utils as pu

height_th = 0.0103


def fill_between(initial, final):
    """
    generates a list of numbers between initial and final.
    eg.
    input fill_between(0,10)
    output [1,2,...,9]
    """
    return np.arange(initial + 1, final)


def array_fill_between(_array, initial, final):
    return np.sort(np.unique(np.append(_array, fill_between(initial, final))))


def discriminator(time,
                  signal,
                  height_th=height_th,
                  dt_left=200e-9,
                  dt_right=1000e-9,
                  Plot=False,
                  method=0):
    """
    find area above threshold, including a dt_left duration of the siganl
    """
    dt = time[1] - time[0]
    bins_left = int(dt_left / dt)
    bins_right = int(dt_right / (time[1] - time[0]))
    bins_final = len(time) - 1

    # where signal is above threshold
    initial_window = np.array(np.where(signal > height_th)[0])
    mask = initial_window
    if len(mask) > 0:
        if method == 0:
            # MASK METHOD 1: include regions left and right of signal above
            # threshold.
            for i in np.arange(len(initial_window)) - 1:
                if (initial_window[i + 1] - initial_window[i] < bins_right + bins_left):
                    mask = array_fill_between(
                        mask, initial_window[i], initial_window[i + 1])
                else:
                    mask = array_fill_between(
                        mask, initial_window[i], initial_window[i] + bins_right)
                    mask = array_fill_between(
                        mask, initial_window[i + 1] - bins_left, initial_window[i + 1])
            # right most pulse close to edge but doesn't exceed
            if mask[-1] + bins_right <= bins_final:
                mask = array_fill_between(
                    mask, mask[-1], mask[-1] + bins_right)
            else:
                # right most pulse exceeds right edge of trace: include mask up to end of pulse
                # bins_final+1 term includes one more index since fill_between
                # fills up to bins_final
                mask = array_fill_between(mask, mask[-1], bins_final + 1)
            if mask[0] - bins_left >= 0:  # left most pulse close to edge but doesn't exceed
                mask = array_fill_between(mask, mask[0] - bins_left, mask[0])
            else:
                # left most pulse exceeds left edge of trace
                # 0-1 term includes one more index '-1' since fill_between
                # fills from 0
                mask = array_fill_between(mask, 0 - 1, mask[0])

        if method == 1:
            # MASK METHOD 2: SR LATCH
            mask = srlatch_full(signal, 0, height_th)
        if method == 2:
            # MASK METHOD 3: rev SR LATCH
            initial_window = np.where(srlatch_rev(signal, 0, height_th))[0]
            mask = initial_window
            for i in np.arange(len(initial_window)) - 1:
                if (initial_window[i + 1] - initial_window[i] < bins_right + bins_left):
                    mask = array_fill_between(
                        mask, initial_window[i], initial_window[i + 1])
                else:
                    mask = array_fill_between(
                        mask, initial_window[i], initial_window[i] + bins_right)
                    mask = array_fill_between(
                        mask, initial_window[i + 1] - bins_left, initial_window[i + 1])
            # right most pulse close to edge but doesn't exceed
            if mask[-1] + bins_right <= bins_final:
                mask = array_fill_between(
                    mask, mask[-1], mask[-1] + bins_right)
            else:
                # right most pulse exceeds right edge of trace: include mask up to end of pulse
                # bins_final+1 term includes one more index since fill_between
                # fills up to bins_final
                mask = array_fill_between(mask, mask[-1], bins_final + 1)
            if mask[0] - bins_left >= 0:  # left most pulse close to edge but doesn't exceed
                mask = array_fill_between(mask, mask[0] - bins_left, mask[0])
            else:
                # left most pulse exceeds left edge of trace
                # 0-1 term includes one more index '-1' since fill_between
                # fills from 0
                mask = array_fill_between(mask, 0 - 1, mask[0])

    # transforms index mask into boolean mask
    mask_boolean = np.zeros(len(time), dtype=int)
    mask_boolean[mask] = 1
    mask = mask_boolean

    edges = np.diff(mask * 1)  # left edge = 1, right edge = -1

    # find indices of left and right edges of pulses
    right_edges = np.array(np.where(edges < 0), dtype='int64').flatten()
    left_edges = np.array(np.where(edges > 0), dtype='int64').flatten()
    clamp = np.zeros(len(time))

    if (len(left_edges) > 0)and(len(right_edges > 0)):
        for i in np.arange(len(time)):
            if ((i >= left_edges[0])and(i <= right_edges[-1])):
                clamp[i] = 1
    clamp = np.array(clamp, dtype='bool')
    mask = np.array(mask, dtype='bool')
    # print clamp,edges,left_edges,right_edges

    if Plot:
        plt.figure()
        plt.plot(time, signal)
        plt.hlines(height_th, time[0], time[1], linestyle='--')
        plt.plot(time, mask * np.max(signal), label='mask')
        plt.plot(time[1:], edges * np.max(signal), label='edges')
        plt.plot(time, (clamp & mask) * np.max(signal), label='clamp&mask')
        plt.scatter(time[(clamp & mask)], signal[
                    (clamp & mask)], label='', color='red', marker='o')
        plt.legend()

    return np.array([mask, clamp, edges, left_edges, right_edges])


def area_windowed(time,
                  signal,
                  height_th=height_th,
                  dt_left=200e-9,
                  dt_right=1000e-9,
                  Plot=False,
                  method=0):
    [mask, clamp, edges, left_edges, right_edges] = discriminator(
        time, signal,
        height_th=height_th,
        dt_left=dt_left,
        dt_right=dt_right,
        Plot=Plot,
        method=method)
    return np.sum(np.abs(signal[clamp & mask]))


def area_breakdown(time,
                   signal,
                   height_th=height_th,
                   dt_left=200e-9,
                   dt_right=1000e-9,
                   Plot=False):
    [mask, clamp, edges, left_edges, right_edges] = discriminator(
        time,
        signal,
        height_th=height_th,
        dt_left=dt_left,
        dt_right=dt_right,
        Plot=Plot)
    parity = np.sum(edges)

    if len(left_edges) and len(right_edges):  # at least 1 full pulse
        if parity > 0:  # has 1/2 pulse on right
            areas = [np.sum(signal[l:r])
                     for l, r in zip(left_edges[1:], right_edges)]
        if parity < 0:  # has 1/2 pulse on left
            areas = [np.sum(signal[l:r])
                     for l, r in zip(left_edges, right_edges[:-1])]
        if parity == 0:  # has integer pulses
            areas = [np.sum(signal[l:r])
                     for l, r in zip(left_edges, right_edges)]
    else:
        areas = []

    return np.array(areas), mask, clamp, edges, left_edges, right_edges


def srlatch(_signal, reset_th, set_th=height_th):
    """
    Implements a two level discriminator, returning a mask
    """
    s = [_signal > set_th][0]
    r = [_signal < reset_th][0]
    L = len(_signal)
    q = np.zeros(L)
    # qrev = np.zeros(L)
    for i in np.arange(L):
        if ((s[i] == 0) & (r[i] == 0) & (i > 0)):
            q[i] = q[i - 1]
        elif ((s[i] == 0) & (r[i] == 1)):
            q[i] = 0
        elif ((s[i] == 1) & (r[i] == 0)):
            q[i] = 1
        else:
            q[i] == -1  # error
    return np.array(q)


def srlatch_full(_signal, reset_th, set_th=height_th):
    q = srlatch(_signal, reset_th, set_th)
    qrev = np.flipud(srlatch(np.flipud(_signal), reset_th, set_th))
    qnet = np.logical_or(q, qrev)
    return np.array(qnet)


def srlatch_rev(_signal, reset_th, set_th=height_th):
    qrev = np.flipud(srlatch(np.flipud(_signal), reset_th, set_th))
    return np.array(qrev, dtype='bool')
