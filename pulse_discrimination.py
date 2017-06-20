"""Collection of functions used to discriminate between pulses of different areas."""
import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
# import pulse_utils as pu

def fill_between(initial,final):
    """
    generates a list of numbers between initial and final.
    eg. 
    input fill_between(0,10)
    output [1,2,...,9]
    """
    return np.arange(initial+1,final)

def array_fill_between(_array, initial, final):
    return np.sort(np.unique(np.append(_array, fill_between(initial,final))))

def discriminator(time,signal,height_th,dt_left=300e-9,dt_right=1400e-9,Plot=False, method=0):
    """
    find area above threshold, including a dt_left duration of the siganl 
    """
    dt = time[1]-time[0]
    bins_left = int(dt_left/dt)
    bins_right = int(dt_right/(time[1]-time[0]))
    bins_final = len(time)-1

    initial_window = np.array(np.where(signal > height_th)[0]) #where signal is above threshold
    mask = initial_window
    if len(mask) > 0:
        if method == 0:
            # MASK METHOD 1: include regions left and right of signal above threshold.
            for i in np.arange(len(initial_window))-1:
                    if (initial_window[i+1]-initial_window[i] < bins_right+bins_left):
                        mask = array_fill_between(mask, initial_window[i], initial_window[i+1])
                    else:
                        mask = array_fill_between(mask, initial_window[i], initial_window[i]+bins_right)
                        mask = array_fill_between(mask, initial_window[i+1]-bins_left, initial_window[i+1])
            if mask[-1]+bins_right <= bins_final: #right most pulse close to edge but doesn't exceed
                mask = array_fill_between(mask, mask[-1], mask[-1]+bins_right)
            else: 
                #right most pulse exceeds right edge of trace: include mask up to end of pulse
                #bins_final+1 term includes one more index since fill_between fills up to bins_final
                mask = array_fill_between(mask, mask[-1], bins_final+1)
            if mask[0]-bins_left >= 0: #left most pulse close to edge but doesn't exceed
                mask = array_fill_between(mask, mask[0]-bins_left, mask[0])
            else: 
                #left most pulse exceeds left edge of trace
                #0-1 term includes one more index '-1' since fill_between fills from 0
                mask = array_fill_between(mask, 0-1, mask[0])
        
        if method == 1:
            # MASK METHOD 2: SR LATCH FULL
            mask = srlatch_full(signal, 0, height_th)
            
        if method == 2:
            # MASK METHOD 3: rev SR LATCH
            initial_window = np.where(srlatch_rev(signal, 0, height_th))[0]
            
            # MANUAL INCLUSION OF ADDITIONAL PULSE REGIONS, since srlatch_rev does not include the full pulse 
            mask = initial_window
            for i in np.arange(len(initial_window))-1:
                    if (initial_window[i+1]-initial_window[i] < bins_right+bins_left):
                        mask = array_fill_between(mask, initial_window[i], initial_window[i+1])
                    else:
                        mask = array_fill_between(mask, initial_window[i], initial_window[i]+bins_right)
                        mask = array_fill_between(mask, initial_window[i+1]-bins_left, initial_window[i+1])
                        
            if mask[-1]+bins_right <= bins_final: #right most pulse close to edge but doesn't exceed
                mask = array_fill_between(mask, mask[-1], mask[-1]+bins_right)
            else: 
                #right most pulse exceeds right edge of trace: include mask up to end of pulse
                #bins_final+1 term includes one more index since fill_between fills up to bins_final
                mask = array_fill_between(mask, mask[-1], bins_final+1)
            if mask[0]-bins_left >= 0: #left most pulse close to edge but doesn't exceed
                mask = array_fill_between(mask, mask[0]-bins_left, mask[0])
            else: 
                #left most pulse exceeds left edge of trace
                #0-1 term includes one more index '-1' since fill_between fills from 0
                mask = array_fill_between(mask, 0-1, mask[0])

        if method == 3:
            # MASK METHOD 4: SR LATCH (CONVENTIONAL)
            mask = srlatch(signal, 0, height_th)

    #transforms index mask into boolean mask
    mask_boolean = np.zeros(len(time), dtype=int)
    mask_boolean[mask] = 1
    mask = mask_boolean

    edges = np.diff(mask*1) #left edge = 1, right edge = -1

    #checks for half pulses: traces with half pulses on the left(right) should have negative(positive) parity.
    # parity = np.sum(mask_boolean_diff)

    #find indices of left and right edges of pulses
    right_edges = np.array(np.where(edges<0),dtype='int64').flatten()
    left_edges = np.array(np.where(edges>0),dtype='int64').flatten()

    # print mask,edges,right_edges,left_edges

    #restrict from the first left edge to the last right edge
    # clamp = [((i >= left_edges[0])and(i <= right_edges[-1])) for i in np.arange(len(time))]
    clamp = np.zeros(len(time))

    if (len(left_edges) > 0)and(len(right_edges > 0)):
        for i in np.arange(len(time)):
            if left_edges[0] <= i <= right_edges[-1]:
                clamp[i] = 1
            # else:
                # clamp = np.ones(len(time))     
                # print 'error: no index between first left and last right \n{}'.format([left_edges, right_edges])
    # else:
        # clamp = np.ones(len(time))
        # print 'error: left or right index zero length \n{}'.format([left_edges, right_edges])
    clamp = np.array(clamp,dtype='bool')
    mask = np.array(mask,dtype='bool')
    
    # print clamp,edges,left_edges,right_edges

    if Plot:
        plt.figure(figsize=(10,5))
        plt.plot(time,signal)
        plt.hlines(height_th,time[0],time[1],linestyle='--')
        plt.plot(time,mask*np.max(signal),label='mask')
        plt.plot(time[1:],edges*np.max(signal),label='edges')
        plt.plot(time,(clamp&mask)*np.max(signal),label='clamp&mask')
        plt.scatter(time[(clamp&mask)],signal[(clamp&mask)],label='',color='red',marker='o')
        plt.legend()

    return np.array([mask, clamp, edges, left_edges, right_edges])

def area_windowed(time,signal,height_th,dt_left=200e-9,dt_right=1000e-9,Plot=False,method=0):
    [mask, clamp, edges, left_edges, right_edges] = discriminator(time, signal, height_th=height_th,dt_left=dt_left,dt_right=dt_right,Plot=Plot, method=method)
    return np.sum(np.abs(signal[clamp&mask]))

def area_breakdown(time,signal,height_th,dt_left=200e-9,dt_right=1000e-9,Plot=False):
    [mask, clamp, edges, left_edges, right_edges] = discriminator(time, signal, height_th=height_th,dt_left=dt_left,dt_right=dt_right,Plot=Plot)
    parity = np.sum(edges)

    if len(left_edges) and len(right_edges): #at least 1 full pulse
        if parity > 0: #has 1/2 pulse on right
            areas = [np.sum(signal[l:r]) for l,r in zip(left_edges[1:],right_edges)]
        if parity < 0: #has 1/2 pulse on left
            areas = [np.sum(signal[l:r]) for l,r in zip(left_edges,right_edges[:-1])]
        if parity == 0: #has integer pulses
            areas = [np.sum(signal[l:r]) for l,r in zip(left_edges,right_edges)]
    else: areas=[]

    return np.array(areas), mask, clamp, edges, left_edges, right_edges

# def clamp_window(_time,_signal,height_th=.8*height_th):
#     """
#     removes the first and last 1us of trace window if heightt at the edges of the trace exceeds a certain threshold.
#     this minimises the effects of half pulses.
#     """
#     _dt = _time[1]-_time[0]
#     _bins = int(2e-6/_dt)
#     if _signal[0] > height_th:
#         _signal = np.concatenate([np.zeros(_bins), _signal[_bins:]])
#     if _signal[-1] > height_th:
#         _signal = np.concatenate([_signal[:-_bins], np.zeros(_bins)])
#     return _signal

# def matched_filter(_signal,signal_fs=signal_fs_pad):
#     return np.convolve(_signal,np.conjugate(signal_fs),mode='valid')/np.sum(signal_fs)

def srlatch(_signal,reset_th, set_th):
    """
    Implements a two level discriminator, returning a mask
    """
    s=[_signal>set_th][0]
    r=[_signal<reset_th][0]
    L = len(_signal)
    q=np.zeros(L)
    qrev=np.zeros(L)
    # print s,r,q
    for i in np.arange(L):
        if ((s[i]==0)&(r[i]==0)&(i>0)):
            q[i]=q[i-1]
        elif ((s[i]==0)&(r[i]==1)):
            q[i]=0
        elif ((s[i]==1)&(r[i]==0)):
            q[i]=1
        else:
            q[i]==-1 #error
    return np.array(q)

def srlatch_full(_signal,reset_th, set_th):
    q=srlatch(_signal, reset_th, set_th)
    qrev=np.flipud(srlatch(np.flipud(_signal), reset_th, set_th))
    qnet=np.logical_or(q,qrev)
    return np.array(qnet)

def srlatch_rev(_signal,reset_th, set_th):
    qrev=np.flipud(srlatch(np.flipud(_signal), reset_th, set_th))
    return np.array(qrev,dtype='bool')

def savdisc(time,signal,numpts):
    # disc = savgol_filter(np.diff(signal), numpts, 1)*savgol_filter(signal[1:], numpts, 1)
    disc = savgol_filter(np.diff(signal), numpts, 1)
    normed_disc = disc/np.max(disc)
    return time[1:], normed_disc

def savdischt(time,signal,numpts,height_th):
    _ , normed_disc = savdisc(time, signal, numpts)


def savdiscopp(time,signal,numpts):
    disc = np.diff(savgol_filter(signal, numpts, 1))
    normed_disc = disc/np.max(disc)
    return time[1:], normed_disc

# def srlatch_full_plot(_time,_signal,reset_th, set_th=height_th):
#     plt.figure()
#     plt.plot(_time,_signal)
#     plt.plot(_time,np.max(_signal)*srlatch_full(_signal,reset_th,set_th))

# def plot_filter(_time,_signal):
#     f, (ax1, ax2) = plt.subplots(1, 2, sharex=True)
#     ax1.plot(_time*1e6,_signal)
#     ax2.plot(_time*1e6,matched_filter(_signal))
#     plt.show()

# filtered=np.array([matched_filter(*trace_extr(f)) for f in tqdm.tqdm(filelist[:10000])])

# def area_windowed(time,signal,height_th=height_th,dt_left=200e-9,dt_right=1000e-9,Plot=False):
#     """
#     find area above threshold, including a dt_left duration of the siganl 
#     """
#     signal=savgol_filter(signal,301,3)
#     mask = srlatch_full(signal, 0, height_th)
#     edges = np.diff(mask*1) #left edge = 1, right edge = -1

#     #checks for half pulses: traces with half pulses on the left(right) should have negative(positive) parity.
#     # parity = np.sum(mask_boolean_diff)

#     #find indices of left and right edges of pulses
#     right_edges = np.array(np.where(edges<0),dtype='int64').flatten()
#     left_edges = np.array(np.where(edges>0),dtype='int64').flatten()

#     # print mask,edges,right_edges,left_edges

#     #restrict from the first left edge to the last right edge
#     # clamp = [((i >= left_edges[0])and(i <= right_edges[-1])) for i in np.arange(len(time))]
#     clamp = np.zeros(len(time))

#     if (len(left_edges) > 0)and(len(right_edges > 0)):
#         for i in np.arange(len(time)):
#             if ((i >= left_edges[0])and(i <= right_edges[-1])):
#                 clamp[i] = 1
#     clamp = np.array(clamp,dtype='bool')
#     mask = np.array(mask,dtype='bool')
#     # print clamp,edges,left_edges,right_edges

#     if Plot:
#         # plt.figure()
#         plt.plot(time,signal)
#         plt.hlines(height_th,time[0],time[1],linestyle='--')
#         plt.plot(time,mask*np.max(signal),label='mask')
#         plt.plot(time[1:],edges*np.max(signal),label='edges')
#         plt.plot(time,(clamp&mask)*np.max(signal),label='clamp&mask')
#         plt.scatter(time[(clamp&mask)],signal[(clamp&mask)],label='',color='red',marker='o')
#         plt.legend()

#     return np.sum(np.abs(signal[clamp&mask]))

# def area_windowed(time,signal,height_th=height_th,dt_left=200e-9,dt_right=1000e-9,Plot=False,Clamp=True):
#     """
#     find area above threshold, including a dt_left duration of the siganl 
#     """
#     bins_left = int(dt_left/dt)
#     bins_right = int(dt_right/(time[1]-time[0]))
#     bins_final = len(time)-1
#     # if Clamp:
#     #     signal=clamp_window(time, signal,.5*height_th)
#     initial_window = np.array(np.where(signal > height_th)[0]) #where signal is above threshold
#     mask = initial_window
#     if len(mask) > 0:
#         for i in np.arange(len(initial_window))-1:
#                 if (initial_window[i+1]-initial_window[i] < bins_right+bins_left):
#                     mask = array_fill_between(mask, initial_window[i], initial_window[i+1])
#                 else:
#                     mask = array_fill_between(mask, initial_window[i], initial_window[i]+bins_right)
#                     mask = array_fill_between(mask, initial_window[i+1]-bins_left, initial_window[i+1])
#         if mask[-1]+bins_right <= bins_final: #right most pulse close to edge but doesn't exceed
#             mask = array_fill_between(mask, mask[-1], mask[-1]+bins_right)
#         else: #right most pulse exceeds edge
#             mask = array_fill_between(mask, mask[-1], bins_final)
#         if mask[0]-bins_left >= 0: #left most pulse close to edge but doesn't exceed
#             mask = array_fill_between(mask, mask[0]-bins_left, mask[0])
#         else: #left most pulse exceeds edge
#             mask = array_fill_between(mask, 0, mask[0])


#         #safety checks...
#         mask = mask[mask<bins_final]
#         mask = np.unique(mask)

#         #transforms index mask into boolean mask
#         mask_boolean = np.zeros(len(time), dtype=int)
#         mask_boolean[mask] = 1

#         #uses boolean mask to detect left and right edges of pulse
#         mask_boolean_diff = np.diff(mask_boolean)

#         #checks for half pulses: traces with half pulses on the left(right) should have negative(positive) parity.
#         parity = np.sum(mask_boolean_diff)

#         #find indices of left and right edges of pulses
#         right_edges = np.where(mask_boolean_diff==-1)
#         left_edges = np.where(mask_boolean_diff==1)

#         #extra right edge, that only comes from half a pulse on left margin of trace
#         if parity < 0:
#             mask = mask[mask>=right_edges[0]]
#         #extra left edge, that only comes from half a pulse on right margin of trace
#         if parity > 0:
#             mask = mask[mask<=left_edges[-1]]

#     # print mask
#     if Plot:
#         # plt.figure()
#         plt.plot(time,signal)
#         plt.scatter(time[mask],signal[mask],color='red')
#         plt.hlines(height_th,time[0],time[1],linestyle='--')
#         plt.plot(time[1:],np.max(signal)*mask_boolean_diff)
#     # print(mask,signal)
#     return np.sum(signal[mask])

# def area_windowed(time,signal,height_th=height_th,dt_left=200e-9,dt_right=1000e-9,Plot=False,Clamp=True):
#     """
#     find area above threshold, including a dt_left duration of the siganl 
#     """
#     bins_left = int(dt_left/dt)
#     bins_right = int(dt_right/(time[1]-time[0]))
#     bins_final = len(time)-1
#     if Clamp:
#         signal=clamp_window(time, signal,.5*height_th)
#     initial_window = np.array(np.where(signal > height_th)[0]) #where signal is above threshold
#     mask = initial_window
#     if len(mask) > 0:
#         for i in np.arange(len(initial_window))-1:
#                 if (initial_window[i+1]-initial_window[i] < bins_right+bins_left):
#                     mask = array_fill_between(mask, initial_window[i], initial_window[i+1])
#                 else:
#                     mask = array_fill_between(mask, initial_window[i], initial_window[i]+bins_right)
#                     mask = array_fill_between(mask, initial_window[i+1]-bins_left, initial_window[i+1])
#         if mask[-1]+bins_right <= bins_final: #right most pulse close to edge but doesn't exceed
#             mask = array_fill_between(mask, mask[-1], mask[-1]+bins_right)
#         else: #right most pulse exceeds edge
#             mask = array_fill_between(mask, mask[-1], bins_final)
#         if mask[0]-bins_left >= 0: #left most pulse close to edge but doesn't exceed
#             mask = array_fill_between(mask, mask[0]-bins_left, mask[0])
#         else: #left most pulse exceeds edge
#             mask = array_fill_between(mask, 0, mask[0])


#         #safety checks...
#         mask = mask[mask<bins_final]
#         mask = np.unique(mask)

#         #transforms index mask into boolean mask
#         mask_array = np.zeros(len(time), dtype=int)
#         mask_array[mask] = 1

#         #uses boolean mask to detect left and right edges of pulse
#         mask_array_diff = np.diff(mask_array)
#     # print mask
#     if Plot:
#         # plt.figure()
#         plt.plot(time,signal)
#         plt.scatter(time[mask],signal[mask],color='red')
#         plt.hlines(height_th,time[0],time[1],linestyle='--')
#         plt.plot(time[1:],np.max(signal)*mask_array_diff)
#     # print(mask,signal)
#     return np.sum(signal[mask])

# def area_windowed(time,signal,height_th=height_th,dt_left=200e-9,dt_right=500e-9,Plot=False,Clamp=True):
#     """
#     find area above threshold, including a dt_left duration of the siganl 
#     """
#     # bins_left = int(dt_left/dt)
#     bins_right = int(dt_right/(time[1]-time[0]))
#     bins_final = len(time)-1
#     if Clamp:
#         signal=clamp_window(time, signal,.5*height_th)
#     initial_window = np.array(np.where(signal > height_th)[0]) #where signal is above threshold
#     mask = initial_window
#     if len(mask) > 0:
#         for i in np.arange(len(initial_window))-1:
#                 if (initial_window[i+1]-initial_window[i] < bins_right):
#                     mask = array_fill_between(mask, initial_window[i], initial_window[i+1])
#                 else:
#                     mask = array_fill_between(mask, initial_window[i], initial_window[i]+bins_right)
#         if mask[-1]+bins_right < bins_final:
#             mask = array_fill_between(mask, mask[-1], mask[-1]+bins_right)
#         else:
#             mask = array_fill_between(mask, mask[-1], bins_final)

#         #safety checks...
#         mask = mask[mask<bins_final]
#         mask = np.unique(mask)
#     # print mask
#     if Plot:
#         # plt.figure()
#         plt.plot(time,signal)
#         plt.scatter(time[mask],signal[mask],color='red')
#         plt.hlines(height_th,time[0],time[1],linestyle='--')
#     # print(mask,signal)
#     return np.sum(signal[mask])








































    # return np.sum(signal[window_left:initial_window[0]])+\
    # np.sum(signal[initial_window])+\
    # np.sum(signal[initial_window[-1]:window_right]) 

    # initial_window_diff = np.diff(initial_window)
    # gaps = np.array(np.where(initial_window_diff > bins_right)) -1

    # for g in gaps:

        # window_left = initial_window[0] - bins_left #include extra signal to the left
    # window_right = initial_window[-1] + bins_right #include extra signal to the right

    # def extend(mask,bins_left=bins_left,bins_right=bins_right):
    #   new_left = mask[0] - bins_left
    #   new_right = mask[-1] + bins_right

    #   if new_left < 0:
    #       new_left = mask[0] #ensure extra signals exist

    #   if (new_right > len(time) -1):
    #       new_right = mask[-1]