#!/usr/bin/env python
"""
version 4: 
hysterysis implemented for fit_two (lmfit using differentiated signal for edge detection)
MCMC implemented for traces
applied to cw coherent source g2

version 5: 
same as v4, applied to '/workspace/projects/TES/data/20160914_TES5_MAGSQ1_4.4ns_double_pulse/'

7 Oct 2016 Jianwei
"""
from __future__ import division

# import heralded_pulses_analysis as hps
import numpy as np
import pymc

from scipy.signal import savgol_filter

import peakutils

from lmfit import Model
from lmfit import Parameters

import random

from multiprocessing import Pool, cpu_count
import tqdm
from mpl_toolkits.mplot3d import Axes3D

def one_pulse(x, x_offset=0, amplitude=1):
        """convert the sample single photon pulse into a function
        that can be used in a fit
        """
        x = x - x_offset
        return amplitude * np.interp(x, time_f, signal_fs)

def hysterysis(time,signal,candidates,height_th=height_th,hysterysis=300e-9):
    """
    Removes edges, retaining those with the highest signal height,  within hysterysis window
    Used with fit_two because defining the min distance between differentiated peaks (signal edges) results in retaining indices where the signal height is lower if these are detected first.
    This results in rejection of legitimate signal edges.
    The current workaround uses a small minimum distance to detect multiple edges including noise, but reject them later using this hysterysis function.
    """
    idx_edge = []
    # candidates = np.array([c for c in candidates if signal[c] > height_th])
    candidates = np.sort(candidates)
    for i,idx in enumerate(candidates):
        if len(idx_edge) == 0: #check for empty list
            idx_edge.append(idx)
        else:
            if np.min(np.abs(time[idx] - time[idx_edge])) > hysterysis:
                #immedietely retain indices above hysterysis
                    idx_edge.append(idx)
            else:
                #within the hysterysis window, retain the index that has the higher value
                #look for the element that has the nearest signal value to signal[idx]
                # nearest_idx = find_idx(signal[idx],signal[idx_edge])
                previous_idx = candidates[i-1]
                if signal[idx]-signal[previous_idx] > 0:
                    # idx_edge[-1] = idx
                    idx_edge[-1] = int((idx+idx_edge[-1])/2)

    return np.array(np.sort(idx_edge))

def fit_two_mcmc(time, signal, amplitude_init_min=0.7, amplitude_init_max=1.5, one_x_offset_init=None, Plot=False, debug=False, sampling=1e2):
    
    if ((t_initial==None) or (t_final==None)):
        t_mean = np.mean(time)
        _t_initial=time[0]
        _t_final=time[-1]
    else:
        t_mean = (t_initial+t_final)/2
        _t_initial=t_initial
        _t_final=t_final
    if one_x_offset_init == None:
        one_x_offset_init = t_mean

    def model(x, f): 
        #priors
        y_err = pymc.Uniform("sig", 0.0, .004, value=.002)
        # y_err = 0.002
        print (_t_initial,_t_final, one_x_offset_init)
        one_x_offset = pymc.Uniform("one_x_offset", _t_initial, _t_final, value=one_x_offset_init)
        two_x_offset = pymc.Uniform("two_x_offset", _t_initial, _t_final, value=t_mean)
        one_x_amplitude = pymc.Uniform("one_x_amplitude", amplitude_init_min, amplitude_init_max, value=1.0)
        two_x_amplitude = pymc.Uniform("two_x_amplitude", amplitude_init_min, amplitude_init_max, value=1.0)

        #model
        @pymc.deterministic(plot=False)
        def mod_two_pulse(x=time, one_x_offset=one_x_offset, two_x_offset=two_x_offset, one_x_amplitude=one_x_amplitude, two_x_amplitude=two_x_amplitude):
              return one_pulse(x, x_offset=one_x_offset, amplitude=one_x_amplitude)+one_pulse(x, x_offset=two_x_offset, amplitude=two_x_amplitude)

        #likelihoodsy
        y = pymc.Normal("y", mu=mod_two_pulse, tau= 1.0/y_err**2, value=signal, observed=True)
        return locals()

    MDL = pymc.MCMC(model(time,signal), db='pickle') # The sample is stored in a Python serialization (pickle) database
    MDL.use_step_method(pymc.AdaptiveMetropolis,MDL.y_err) # use AdaptiveMetropolis to "learn" how to step
    MDL.sample(iter=sampling, burn=int(sampling/2), thin=2)  # run chain longer since there are more dimensions?

    if Plot:
        y_fit = MDL.mod_two_pulse.value #get mcmc fitted values
        plt.figure()
        plt.plot(time, signal, 'b', marker='o', ls='-', lw=1, label='Observed')
        plt.plot(time,y_fit,'k', marker='+', ls='--', ms=5, mew=2, label='Bayesian Fit Values')

        plt.legend()
        plt.show()
    if debug:
        pymc.Matplot.plot(MDL)
        plt.show()

    return MDL #usage: MDL.one_x_offset.value for fitted result

def edges_by_diff(time, signal, height_th=height_th, filterpts=301, order=3, thres=0.7, min_dist=20, debug=False):
    idx_s = peakutils.indexes(savgol_filter(np.diff(signal), 301, 3),
                              thres=0.7,
                              min_dist=20)
    idx_s = np.flipud(idx_s[signal[idx_s].argsort()])

    if debug:
        plt.figure()
        plt.plot(time,signal)
        plt.vlines(time[idx_s[0:2]],0,height_th,label='idxs[0,2]', linestyle='--')
        plt.hlines(height_th,time[0],time[-1],linestyle='--')
        print idx_s

    return idx_s

def edges_by_diff_hys(time, signal, height_th=height_th,filterpts=301, order=3, thres=0.7, min_dist=20, debug=False, save=False):
    """obtain edges in trace by differentiation"""
    idx_s_0 = peakutils.indexes(savgol_filter(np.diff(signal), filterpts, order),
                          thres=thres,
                          min_dist=min_dist)
    idx_s = idx_s_0
    if len(idx_s) >= 2: #hysterysis required when # edges detected exceeds 1
        idx_s = hysterysis(time, signal, idx_s_0) #removes edges within hysterysis window (see function comment)
    if len(idx_s) > 1: #flip if # edges > 1
        idx_s_2 = np.flipud(idx_s[signal[idx_s].argsort()])
    else:
        idx_s_2 = idx_s

    if debug:
        try:
            plt.figure()
            plt.plot(time,signal)
            filteredD = savgol_filter(np.diff(signal), filterpts, order)
            plt.plot(time[:-1], filteredD/np.max(filteredD)*np.max(signal),label='filtered and differentiated signal')
            # plt.vlines(time[idx_s_0],0,np.max(signal),label='idxs', linestyle='--')
            print idx_s
            # plt.vlines(time[idx_s],height_th,2*height_th,label='idxs_hys')
            plt.vlines(time[idx_s_2][0:2],2*height_th,3*height_th,color='blue',label='top two idxs_hys')
            # plt.vlines(time[idx_s_3],2*height_th,3*height_th,label='idxs_hys_thres',linestyle='-.')
            plt.hlines(height_th,time[0],time[-1], linesyle='--', label='threshold')
            plt.legend()
            plt.show()
        except:
            pass
    if save:
        np.savetxt(results_directory+'200ns_dsig.dat', zip(time[:-1], filteredD/np.max(filteredD)*np.max(signal)),header='time\tnormed_dsignal')
        np.savetxt(results_directory+'200ns_sig.dat', zip(time,signal))
    return idx_s_2

def fit_two(time, signal, height_th=height_th,filterpts=301, order=3, thres=0.7, min_dist=20, sampling=3e2):
    """
    Uses a discriminator to identify bulk pulse regions based on a height threshold, and some extention to the left and right of the region.
    Uses 'edges_by_diff_hys' peak identification of a filtered, differentiated pulse, with hysterysis ~50 ns to identify pulse edges, and uses the edges with the heighest 2 peaks.
        Weakness: the second highest peak might still be a noise peak. To reduce this effect, the index of the edges are checked with the discriminator window.
        This is still insufficienct as it does not filter out noise peaks along the tail.
        A proper treatment requires an analysis of reduced chisq, based on the overlapping noise models of two single photon pulses.
    Tested with 20160914_TES5_MAGSQ1_4.4ns_double_pulse/200ns pulses:
        for 200 ns seperated n=1 pulses, showed a symmetric distribution, with +10ns offset.
        for overlapping n=2 pulses, the min seperation between fitted pulses depends on the amplitude limits given to the mcmc fit.
            min = 0 amplitude is unadvisable since it tends to fit n=2 pulses to n=.3, n=1.3, and n=.3 pulses tend to follow noise fluctuations.
            a range of .7:1.3 varying all amps and offsets results in an almost symm dist around 100ns with the same width.
    @Jianwei 25 Jan 2017
    """
    idx_s = edges_by_diff_hys(time, signal)
    p = Parameters()

    [mask, clamp, edges, left_edges, right_edges] = discriminator(time, signal, dt_left=200e-9,dt_right=800e-9, height_th=h_th, Plot=False, method=2)
    window = mask&clamp
    idx_s = np.array([i for i in idx_s if window[i]])

    if len(idx_s) < 2: #does mcmc fit to initialise lmfit - duplicate work yes, but recoding the results array is more work :P
        result_mcmc = fit_two_mcmc(time[window][::10], signal[window][::10], sampling=sampling)

        p.add('one_x_offset', result_mcmc.one_x_offset.value, vary=1)
        p.add('two_x_offset', result_mcmc.two_x_offset.value, vary=1)
        p.add('one_amplitude', result_mcmc.one_x_amplitude.value, min=0.7, max=1.3, vary=1) #warning: max >= 2 causes n=2 & noise to be fitted on a tau~0 2ph trace. 
        p.add('two_amplitude', result_mcmc.two_x_amplitude.value, min=0.7, max=1.3, vary=1)

    if len(idx_s) >= 2:
        p.add('one_x_offset', time[idx_s[0]])
        p.add('two_x_offset', time[idx_s[1]])
        p.add('one_amplitude', 1, min=0.7, max=1.5, vary=1) #warning: max >= 2 causes n=2 & noise to be fitted on a tau~0 2ph trace. 
        p.add('two_amplitude', .9, min=0.7, max=1.5, vary=1
          # expr='one_amplitude'
          )
    # yerr = 0.001 + 0.001*(1-signal/np.max(signal))
    result = two_pulse_fit.fit(signal,
                               x=time,
                               params=p,
                               # weights=1 / yerr,
                               # method=method
                               )
    return result, idx_s

def fit_two_with_noise(time, signal, signal_var):
    """
    takes initial fit result from fit_two
    performs lmfit again with signal variance of a model pulse
    """
    result, idx_s = fit_two(time, signal)
    p = Parameters()
    yerr = signal_var + signal_var

"""
MULTITHREADING
"""
def extract_and_fit_two(f):
    """
    extracts trace and run fit_two
    a seperate function is required for pickling in multithread processing.
    """
    return fit_two(*hps.trace_extr(file, t_initial, t_final))

def results_mtrack(file_list):
    nCores = cpu_count() #to find number of cores
    p = Pool(processes=8) #initialise pool cores
    r = p.map(extract_and_fit_two, tqdm.tqdm(file_list))
    return np.array(r)
# results = results_mtrack(filelist[mask_2ph])

"""
LMFIT USING EDGES TO INITIALISE
"""
def idx_com(signal, height_th=0.008):
    """Computes centre of mass index for signal"""
    signal = np.abs(signal)

    moment = np.sum([i*s for i,s in enumerate(signal) if s > height_th])
    mass = np.sum(signal[s > height_th])
    idx = int(moment*1./mass)

    if idx <= len(signal): 
        return idx
    else: 
        return random.random()*(len(signal))

def fit_two_edges(time, signal, pulsewidth=500e-9, height_th=0.008):

        # idx_s = peakutils.indexes(savgol_filter(np.diff(signal), 1001, 3), thres=0.7, min_dist=50)
        # idx_s = np.flipud(idx_s[signal[idx_s].argsort()])
        
        idx_e = edges(time, signal)
        com = time[idx_com(signal)] #centre of mass

        p = Parameters()
        # print(len(idx_s))
        if len(idx_e) == 0:
                p.add('one_x_offset', time[np.argmax(signal)], min=t_initial, max=t_final)
                p.add('two_x_offset', 2*(com-pulsewidth/2) - time[np.argmax(signal)], min=t_initial, max=t_final) #assume centre of mass is offset from edge about 500 ns
        else:
                if len(idx_e) > 0:
                        p.add('one_x_offset', time[idx_e[0]], min=t_initial, max=t_final)
                        p.add('two_x_offset', time[np.argmax(signal)]-pulsewidth/2, min=t_initial, max=t_final) #since idx_e is sorted, the second edge should come later, beloinging to the maximum of the combined pulse shape
                        # p.add('two_x_offset', 2*com - time[idx_e[0]], min=t_initial, max=t_final)
                if len(idx_e) > 1:
                        p.add('two_x_offset', time[idx_e[1]], min=t_initial, max=t_final)
        p.add('one_amplitude', 1, min=0.8, max=1.2, vary=1)
        p.add('two_amplitude', 1, min=0.8, max=1.2, vary=1
                    # expr='one_amplitude'
                    )
        yerr = 0.002
        # yerr = 0.001 + 0.001*(1+np.exp(-signal/height_th)) #weigh errors with height of signal
        # yerr = 0.001 + 0.001*(1-signal/np.max(signal))
        # if len(idx_e) == 0:
        #     yerr = 0.001+0.005*np.exp(np.abs((time-time[np.argmax(signal)]))/pulsewidth*20)
        # elif len(idx_e) == 1:
        #     yerr = yerr = 0.001+0.005*np.exp(np.abs((time-time[idx_e[0]]))/pulsewidth*20)
        # elif len(idx_e) >= 2:
        #     yerr = yerr = 0.001+0.005*np.exp(np.abs((time-time[idx_e[0]]))/pulsewidth*20)+0.005*np.exp(np.abs((time-time[idx_e[1]]))/pulsewidth*20)

        result = two_pulse_fit.fit(signal,
                                                             x=time,
                                                             params=p,
                                                             weights=1 / yerr,
                                                             # method=method
                                                             )
        return result

def valid_result(time,signal,result,tolerance=.8):
    """
    flags fit where a pulse was assigned to a position in a 2ph trace, but no pulse was actually there
    compares the fitted pulse with signal at one_x_offset and two_x_offset
    works well provided amplitudes are not allowed to vary much
    :params tolerance:accept if fitted amplitudes match signal at amplitudes below this
    """

    one_x_offset, two_x_offset = result.best_values['one_x_offset'],result.best_values['two_x_offset']
    one_amplitude, two_amplitude = result.best_values['one_amplitude'],result.best_values['two_amplitude']
    idx_one, idx_two = find_idx(time, one_x_offset), find_idx(time, two_x_offset)

    condition1 = np.abs(signal[idx_one] - one_amplitude*np.max(signal_fs))/(one_amplitude*np.max(signal_fs)) < tolerance
    condition2 = np.abs(signal[idx_two] - two_amplitude*np.max(signal_fs))/(two_amplitude*np.max(signal_fs)) < tolerance
    
    if (condition1 & condition2):
        return True
    else: 
        return False

mask_valid = []
for f,r in zip(filelist[mask_2ph],results):
    t,s = hps.trace_extr(f, t_initial, t_final)
    mask_valid.append(
        valid_result(t,s,r,.5)
            )


"""VISUALISATION FOR DIAGNOSISING FITS"""

"""
EXTRACT RESULTS
"""
class extract():
    # global one_amplitudes, two_amplitudes, one_amplitudes_init, two_amplitudes_init
    # global one_x_offsets, two_x_offsets, one_x_offsets_init, two_x_offsets_init
    # global tau, chisqs, redchis
    def __init__(self,results_):
        results=results_
        # results=results_[:,0]
        # idxs=results_[:,1]

        self._one_amplitudes = np.array([r.best_values['one_amplitude'] for r in results])
        self._two_amplitudes = np.array([r.best_values['two_amplitude'] for r in results])
        self._one_amplitudes_init = np.array([r.init_values['one_amplitude'] for r in results])
        self._two_amplitudes_init = np.array([r.init_values['two_amplitude'] for r in results])

        self._one_x_offsets = np.array([r.best_values['one_x_offset'] for r in results])
        self._two_x_offsets = np.array([r.best_values['two_x_offset'] for r in results])
        self._one_x_offsets_init = np.array([r.init_values['one_x_offset'] for r in results])
        self._two_x_offsets_init = np.array([r.init_values['two_x_offset'] for r in results])

        self._tau = self._two_x_offsets - self._one_x_offsets
        self._chisqs=[r.chisqr for r in results]
        self._redchis=[r.redchi for r in results] 

        """
        INTERESTING TRACES
        """
        #some often used masks
        self._c1 = (self._one_amplitudes >= .8)&(self._one_amplitudes < 1.2)
        self._c2 = (self._two_amplitudes >= .8)&(self._two_amplitudes < 1.2) 
        self._c3 = (self._tau>-2000e-9)&(self._tau<0e-9) #interesting taus
        self._c4 = (self._one_x_offsets>0)&(self._two_x_offsets>0) #rejects accidentals before 0us, preventing extra accidental coincidences contributed by -ve arrival times, since -ve and +ve time ranges are not equal.
        self._c5 = (self._one_x_offsets<7e-6)&(self._two_x_offsets<7e-6) #rejects last 1us 
        self._c6 = np.logical_or((self._one_x_offsets<200e-9),(self._two_x_offsets<200e-9)) #contains pulse correlated with trigger
        # self._c7 = self._numedges >=3
        self._c8 = self._chisqs <= 1.3*np.median(self._chisqs)
        self._c9 = (self._tau < 300e-9)&(self._tau > 30e-9)

        self._mask_amp = np.where(self._c1&self._c2)
        self._mask_tau = np.where(self._c1&self._c2&self._c4&self._c9)
        self._mask2 = np.where(self._c1&self._c2&self._c4&self._c8)
        self._mask3 = np.where(~(self._c1&self._c2&self._c3&self._c4))

        self._mask = self._mask_amp

        self._bins = 200
    @property
    def interesting_traces(self):
        self._interesting_trcs = np.array(
            [
                hps.trace_extr(f, t_initial, t_final) 
                for f 
                in filelist[mask_2ph][self._mask]
            ]
        )
        plt.figure('interesting traces');[plt.plot(t[::20]*1e9,s[::20],alpha=.2) for t,s in self._interesting_trcs];plt.show()
        return self._interesting_trcs

    @property
    def interesting_results(self,lims=[0,10]):
        self._interesting_results = results[self._mask]
        plt.figure();[self._interesting_results[i].plot_fit() for i in np.arange(lims[0],lims[1])]
        return self._interesting_results

    @property
    def numedges(self):
        """NUMBER OF EDGES IN INTERESTING TRACES"""
        self._numedges=np.array([len(i) for i in idxs])
        freq, num=np.histogram(_numedges[self._mask],100,range=([0,100])) #number of edges histogram of 2ph traces
        freq_i, num_i=np.histogram(_numedges[self._mask2], 100,range=([0,100])) #number of edges histogram of interesting 2ph traces
        plt.figure()
        ax = plt.subplot(211)
        ax.bar(num[:-1],freq, alpha=0.5,label='all 2ph')
        ax.bar(num[:-1],freq_i,alpha=0.5,label='odd peak 2ph',color='red')
        ax.legend()

        # ax2 = plt.subplot(212)
        # ax2.bar(num[:-1],freq_i/freq, alpha=0.5,label='odd_peak 2ph / all 2ph')
        # plt.show()
        return self._numedges

    def g2(self,label=''):
        plt.hist(np.abs(self._tau[self._mask])*1e9,self._bins,alpha=0.5,label=label,range=lims)
        return hist(self._tau[self._mask]*1e9, self._bins,alpha=0.5,label=label)
"""
G2
"""
def hist(data,bins=400,lims=None, label='', Plot=True, alpha=.5):
    """Creates and Plots numpy histogram, removing the last bin""" 
    y, binEdges = np.histogram(data,bins,range=(lims))
    if Plot:
        plt.figure()
        y_err = np.sqrt(y)
        width = binEdges[1]-binEdges[0]
        plt.bar(binEdges[:-1],y,yerr=y_err,alpha=alpha, width=width, label=label)
        plt.xlim(lims)
    return y, binEdges[:-1]

"""
CHISQ vs TAU
"""
def plot_chqsqr_tau(tau=tau,chisqs=chisqs):
    plt.figure();plt.scatter(tau*1e9,chisqs)
    plt.figure();plt.scatter(tau*1e9,redchis);plt.ylim([np.min(redchis),np.max(redchis)])

    amplitudes_min=np.min(one_amplitudes,two_amplitudes)
    plt.figure();plt.scatter(tau*1e9,amplitudes_min);plt.ylim([np.min(amplitudes_min),np.max(amplitudes_min)])

"""
ARRIVAL TIMES HISTOGRAM
"""
def plot_offsets(one_x_offsets=one_x_offsets,two_x_offsets=one_x_offsets):
    fig_arr = plt.figure()
    ax = fig_arr.add_subplot(211)
    ax.hist(one_x_offsets*1e9,400,
        range=([t_initial*1e9,t_final*1e9]),
        alpha=.5,
        label='one_x_offset'); plt.legend()

    ax1 = fig_arr.add_subplot(212)
    ax1.hist(two_x_offsets*1e9,400,
        range=([t_initial*1e9,t_final*1e9]),
        alpha=.5,
        label='two_x_offset'); plt.legend()
    plt.xlabel('time(ns)')
    plt.show()

"""
TAU vs OFFSETS
"""
def plot_tau_vs_offsets(one_x_offsets=one_x_offsets,two_x_offsets=two_x_offsets,tau=tau):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(one_x_offsets[mask],
        two_x_offsets[mask],
        tau[mask]*1e6
        ,marker=".")

    ax.set_xlabel('one_offset');ax.set_ylabel('two_offset');ax.set_zlabel('tau(us)')

    ax.set_xlim([np.min(one_x_offsets),np.max(one_x_offsets)])
    ax.set_xlim([np.min(two_x_offsets),np.max(two_x_offsets)])
    ax.set_zlim([0,0.7])

def misc():
    """
    ONE OFFSET vs TWO OFFSET 
    """
    plt.figure()
    plt.title('Arrival times')
    plt.scatter(one_x_offsets*1e9,two_x_offsets*1e9,
        marker=".")
    plt.xlabel('one_x_offset(ns)')
    plt.ylabel('two_x_offset(ns)')

    """
    ONE AMPLITUDE TWO AMPLITUDE vs TAU
    """
    x = one_amplitudes
    y = two_amplitudes
    z = tau*1e9

    x = one_amplitudes[mask]
    y = two_amplitudes[mask]
    z = tau[mask]*1e9

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x,y,z,marker=".")

    ax.set_xlabel('one_ampl')
    ax.set_ylabel('two_ampl')
    ax.set_zlabel('tau(ns)')

    ax.set_xlim([0,3])
    ax.set_ylim([0,3])
    # ax.set_zlim([-8e3,8e3])

    """
    ONE AMPLITUDE TWO AMPLITUDE at TAU slice
    """
    mask = np.where((np.abs(tau)>0e-9)&(np.abs(tau)<1000e-9))
    plt.figure(); plt.scatter(one_amplitudes[mask], two_amplitudes[mask],marker=".")

    """Compute signal photon number for 2 photon traces"""
    arrival_times_flatten = np.append(one_x_offsets, two_x_offsets)
    arrival_times_hist = np.histogram(arrival_times_flatten,400,range=([t_initial,t_final]))
    # mask_signal = [((t1>0) & (t1<100e-9)) or ((t2>0) & (t2<100e-9)) for t1,t2 in arrival_times]
    mask_signal = [((t1>-1e-6) & (t1<1e-6)) or ((t2>-1e-6) & (t2<1e-6)) for t1,t2 in arrival_times]
    print np.sum(mask_signal)

    """AREA DISTRIBUTION OF INTERESTING TRACES"""
    # data_hopt = data_hopt=np.array([hps.param_extr(f,t_initial,t_final,h_th=0.008) for f in tqdm.tqdm(filelist[:15000])])


    n2ph,a2ph = np.histogram(data_hopt[:,0][mask_2ph][mask],100,range=([0,14]))
    n2phm,a2phm = np.histogram(data_hopt[:,0][mask_2ph][mask2],100,range=([0,14]))
    ratio = np.array(n2phm)/np.array(n2ph) 
    plt.figure();plt.bar(a2phm[:-1],ratio,width=a2phm[1]-a2phm[0])

    """
    FITTED vs INITIAL
    """
    fig = plt.figure()
    plt.scatter(one_x_offsets_init[mask]*1e9, one_x_offsets[mask]*1e9)
    plt.scatter(two_x_offsets_init[mask]*1e9, two_x_offsets[mask]*1e9)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(two_x_offsets_init*1e9, two_x_offsets*1e9, np.array(chisqs)/1000)


def time_diff(time, signal):
        result = fit_two(time, signal)
        return np.abs(result.best_values['two_x_offset'] -
                                    result.best_values['one_x_offset'])

def g2(filelist, t_initial=None, t_final=None):
        return [time_diff(*hps.trace_extr(file, t_initial, t_final))
                        for file
                        in filelist]

def persistence(file_list, sampling=10, density=20):
    plt.figure()
    for f in tqdm.tqdm(file_list[::sampling]):
        time, signal = hps.trace_extr(f)
        plt.plot(time[::density]*1e9,signal[::density],alpha=0.2)
        plt.xlabel('time(ns)')


# def fit_two(time, signal):

#         idx_s = peakutils.indexes(savgol_filter(np.diff(signal), 1001, 3), thres=0.7, min_dist=50)
#         idx_s = np.flipud(idx_s[signal[idx_s].argsort()])
#         print(idx_s)

#         com = time[idx_com(signal)] #centre of mass

#         p = Parameters()
#         # print(len(idx_s))
#         if len(idx_s) == 0:
#                 p.add('one_x_offset', time[np.argmax(signal)], min=t_initial, max=t_final)
#                 p.add('two_x_offset', 2*com - time[np.argmax(signal)], min=t_initial, max=t_final)
#                 print ('0')
#         else:
#                 if len(idx_s) > 0:
#                         p.add('one_x_offset', time[idx_s[0]], min=t_initial, max=t_final)
#                         # p.add('two_x_offset', time[idx_s[0]] + 10e-9, min=t_initial, max=t_final)
#                         p.add('two_x_offset', 2*com - time[idx_s[0]], min=t_initial, max=t_final)
#                 if len(idx_s) > 1:
#                         p.add('two_x_offset', time[idx_s[1]], min=t_initial, max=t_final)
#         p.add('one_amplitude', 1, min=1,max=1.2, vary=1)
#         p.add('two_amplitude', 1, min=1,max=1.2, vary=1
#                     # expr='one_amplitude'
#                     )

#         yerr = 0.001 + 0.001*(1-signal/np.max(signal)) #weigh errors with height of signal

#         result = two_pulse_fit.fit(signal,
#                                                              x=time,
#                                                              params=p,
#                                                              weights=1 / yerr,
#                                                              # method=method
#                                                              )
#         return result
def edges(time,signal,height_th=0.008, hysterysis=200e-9):
    """
    finds edge indexes at height threshold, with a hysterysis = rise time of 1 ph pulse.
    can be used to initialise the edge positions when the pulses are not overlapped.
    for overlapped pulses, the number of detected edges should be 1, then centre-of-mass can be used to predict the second, hidden edge.
    :params hysterysis: average 10% to 90% rise time 
    """
    dt = time[1]-time[0]
    n = int(hysterysis/dt/2) #hysterysis window length = 2n or 2n+1
    candidates = np.argsort(np.abs(signal - height_th))
    idx_edge = []

    peak_ht_est = 2*height_th
    lower_threshold = 0.3*peak_ht_est
    upper_threshold = 0.7*peak_ht_est

    for idx in candidates:
        try:
            if (
                (signal[idx-n]<lower_threshold) & (signal[idx+n]>upper_threshold) & 
                (time[idx] < time[-1] - 1e-6)
                ): 
                #hysterysis qualifier and omit pulses occuring at the last 1us of the trace
                #hysterysis should already omit pulses that have thier edges cut off at the beginning of the trace
                if len(idx_edge) == 0: 
                    idx_edge.append(idx)
                elif np.min(np.abs(time[idx] - time[idx_edge])) > hysterysis: #prevent duplicates
                    idx_edge.append(idx)

        except:
            pass

    return np.array(np.sort(idx_edge))

if __name__ == '__main__':
    """
    Import single photon model
    """
    ph1_model = np.load(results_directory + 'ph1_model.npy')
    time_f, signal_fs = ph1_model[:,0], ph1_model[:,1]
    time_f, signal_fs = time_f[:-100], signal_fs[:-100]

    two_pulse_fit = Model(one_pulse, prefix='one_') + \
        Model(one_pulse, prefix='two_')

    """Import SPDC source traces"""
    directory_name_source = '/workspace/projects/TES/data/20170126_TES5_n012_distinguishibility_20MHz_tau_vs_offset/200ns/'
    filelist_source = np.array(glob.glob(directory_name_source + '*.trc'))
    data_source = np.array([param_extr(f, t_initial=None, t_final=None, h_th=height_th)
                     for f
                     in tqdm.tqdm(filelist_source[:10000])])

    results_= np.array([fit_two(*trace_extr(file, t_initial, t_final))
                        for file
                        in tqdm.tqdm(filelist_source[mask_2ph_source])])

    idxs = results_[:,1]
    results = results_[:,0]
    res = extract(results_)
    """
    acquire height@time - change time to sample from params_extr function
    """
    data_source_2ph = np.array([param_extr(f, t_initial, t_final, h_th=height_th)
                 for f
                 in tqdm.tqdm(filelist_source[mask_2ph])])

# """diagnose EDGES: check of all 2ph traces have at least one detected edge"""
# edges_2ph = np.array([edges(*hps.trace_extr(file, t_initial, t_final))
#                         for file
#                         in filelist[mask_2ph]])
# count_edges_2ph = np.array([len(edgs) for edgs in edges_2ph])

# """
# Free running coherent state g2 (expected)
#     Fixed Trace length T of bins
#     g2 length g2_T of g2_bins
# """
# g2_T = 2000e-9
# # 2000e-9
# g2_bins = 20
# # 200
# T = 10e-6 #trace length
# dT = g2_T/g2_bins #g2 scope / bins in g2 scope
# bins = T/dT #bins in a trace
# # nbar = lambda bins: 40e3/.47*.872*dT #APD rates/ APD eff / TES eff / trace length
# # poiss = lambda k, nbar: nbar**k*np.exp(-1*nbar)/math.factorial(k) 
# prob = lambda tau,bins: 2/bins**2*(T-tau)/(dT)

# taus = np.linspace(0,g2_T,g2_bins)#g2 scope, bins in g2 scope
# prob(taus,bins)*952
# plt.plot(taus*1e9,prob(taus,bins)*952)