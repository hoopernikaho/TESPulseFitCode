"""
Implements the same thing as pulse fit v06 except using individual pulse amplitudes
"""
import pymc
import numpy as np
import matplotlib.pyplot as plt
from lmfit import Model, Parameters
import math

import heralded_pulses_analysis as hpa
import pulse_averaging_cw as pacw
import pulse_fit_v05 as pf
import single_photon_model_no_shift as noshift
import pulse_averaging as pa
import thres_poiss
import pulse_discrimination as pd
import thres
import stats_proc as sp
"""
Monte-Carlo Markov Chain
"""
def fit_two_mcmc(time, 
                 signal, 
                 height_th,
                 one_pulse,
                 sigma0,    # signal noise
                 amp_mu,
                 amp_tau,
                 amp_a,
                 amp_b,
                 sampling, 
                 burn, 
                 thin,
                 Plot=False, 
                 debug=False, 
                 auto=False):
    
    # LIMIT SEARCH FOR OFFSETS
    _t_initial=time[pd.srlatch_rev(signal,0,height_th)][0] 
    _t_final=time[pd.srlatch_rev(signal,0,height_th)][-1] 
    
    def model(x, f): 
        # PRIORS
        y_err = sigma0
        # print (_t_initial,_t_final, one_x_offset_init)
        one_x_offset = pymc.Uniform("one_x_offset", _t_initial, time[np.argmax(signal)], value=_t_initial)
        two_x_offset = pymc.Uniform("two_x_offset", _t_initial, _t_final, value=_t_final)
        one_x_amplitude = pymc.TruncatedNormal("one_x_amplitude", 
                                           mu=amp_mu, 
                                           tau=amp_tau, 
                                           a=amp_a, 
                                           b=amp_b, 
                                           value=amp_mu) #sigma/mu is the n=1 std deviation in units of n=1 amplitude
        two_x_amplitude = pymc.TruncatedNormal("two_x_amplitude", 
                                           mu=amp_mu, 
                                           tau=amp_tau, 
                                           a=amp_a, 
                                           b=amp_b, 
                                           value=amp_mu) #sigma/mu is the n=1 std deviation in units of n=1 amplitude
        # MODEL
        @pymc.deterministic(plot=False)
        def mod_two_pulse(x=time, 
                          one_x_offset=one_x_offset, 
                          two_x_offset=two_x_offset, 
                          one_x_amplitude=one_x_amplitude, 
                          two_x_amplitude=two_x_amplitude):
              return one_pulse(x, x_offset=one_x_offset, amplitude=one_x_amplitude)+\
            one_pulse(x, x_offset=two_x_offset, amplitude=two_x_amplitude)

        #likelihoodsy
        y = pymc.Normal("y", mu=mod_two_pulse, tau= 1.0/y_err**2, value=signal, observed=True)
        return locals()

    MDL = pymc.MCMC(model(time,signal), db='pickle') # The sample is stored in a Python serialization (pickle) database
    MDL.use_step_method(pymc.AdaptiveMetropolis, 
        [MDL.one_x_amplitude, MDL.two_x_amplitude],
        scales={MDL.one_x_amplitude:np.sqrt(1/amp_tau), 
                MDL.two_x_amplitude:np.sqrt(1/amp_tau)}, 
        )
    if auto: 
        # uses Raftery Lewis to determine fit Parameters per trace: 
        # https://pymc-devs.github.io/pymc/modelchecking.html#convergence-diagnostics
        
        # pilot run
        InitSamples = 4*len(time)
        InitMDL = MDL
        InitMDL.sample(iter=InitSamples, burn=int(InitSamples*.5), thin=10)
        pymc_diagnostic = pymc.raftery_lewis(InitMDL, q=0.025, r=0.02, verbose=0) 
        [EstBurn, EstSampling, EstThin] = np.max(
            np.array(
                [pymc_diagnostic[i] for i in pymc_diagnostic.keys()[1:]] # first key: mod_two_pulse irrelavent
            ),
            axis=0)[2:] # first 2 diagnostics: 1st order Markov Chain irrelavent
        # print [EstBurn, EstSampling, EstThin]
        # actual run
        MDL.sample(iter=EstSampling, burn=EstBurn, thin=EstThin, verbose=0)
    else:
        MDL.sample(iter=sampling, burn=burn, thin=thin, verbose=0)  
    # thin: consider every 'thin' samples
    # burn: number of samples to discard: decide by num of samples to run till parameters stabilise at desired precision
    if Plot:
        y_fit = MDL.mod_two_pulse.value #get mcmc fitted values
        plt.figure()
        plt.plot(time, signal, 'b', marker='o', ls='-', lw=1, label='Observed')
        plt.plot(time,y_fit,'k', marker='+', ls='--', ms=5, mew=2, label='Bayesian Fit Values')
        plt.legend()
        pymc.Matplot.plot(MDL)      
    if debug:
        for i in np.arange(10):
            MDL.sample(iter=sampling, burn=burn, thin=thin, verbose=0)
            pymc.gelman_rubin(MDL)
            pymc.Matplot.summary_plot(MDL)
    return MDL #usage: MDL.one_x_offset.value for fitted result

def fit_two_cw(time, signal,
            two_pulse_fit,
            one_pulse,
                 sigma0,    # signal noise
                 amp_mu,
                 amp_tau,
                 amp_a,
                 amp_b,
            sampling, 
            burn, 
            thin,
            height_th,
            Plot=False, 
            debug=False):
    
    # Identify Pulse Region
    [mask, clamp, edges, left_edges, right_edges] = pd.discriminator(time, signal, 
                                                                     dt_left=0,
                                                                     dt_right=0, 
                                                                     height_th=height_th, 
                                                                     Plot=False, 
                                                                     method=2)
    # Use method 2 sr latch reversed with extra time dt_right to manually adjust for additional pulse regions to be considered for the fit.
    # dt_right limit must be SMALLER than limit set for 2 photon selection
    
    # Some flags for debugging later
    mcmc_flag = False
    unequal_edges = False
    
    # Raise flags based on number of edges
    if len(left_edges)==len(right_edges):
        """full pulses"""
        if len(left_edges)>=2:
            """2 or more edges"""
            mcmc_flag = False
        if len(left_edges)==1:
            """1 edge comprising of overlapping photons"""
            mcmc_flag = True
    else:
        mcmc_flag = True
        unequal_edges = True
    
    if mcmc_flag==True:
        # Use MCMC
        result_mcmc = fit_two_mcmc(time[::1], signal[::1], height_th=height_th,
                                one_pulse=one_pulse,
                               sigma0=sigma0,    # signal noise
                                   amp_mu=amp_mu,
                                   amp_tau=amp_tau,
                                   amp_a=amp_a,
                                   amp_b=amp_b,
                               sampling=sampling, 
                               burn=burn,
                               thin=thin,
                               Plot=Plot,
                               debug=debug)
        
        # Use MCMC results to initialise least squares fit
        # one_x_offset_init = np.median(result_mcmc.trace('one_x_offset')[:])
        # two_x_offset_init = np.median(result_mcmc.trace('two_x_offset')[:])
        one_x_offset_init = result_mcmc.one_x_offset.value
        two_x_offset_init = result_mcmc.two_x_offset.value
        one_x_offset_init_min = None; one_x_offset_init_max = None
        two_x_offset_init_min = None; two_x_offset_init_max = None
        
        one_amplitude_init = result_mcmc.one_x_amplitude.value
        two_amplitude_init = result_mcmc.two_x_amplitude.value
        
    if mcmc_flag==False:
        one_x_offset_init_min = time[left_edges][0]
        one_x_offset_init_max = time[left_edges[0]+np.argmax(signal[left_edges[0]:right_edges[0]])]
        two_x_offset_init_min = time[left_edges][1]
        two_x_offset_init_max = time[left_edges[1]+np.argmax(signal[left_edges[1]:right_edges[1]])]
        one_x_offset_init = (one_x_offset_init_min+one_x_offset_init_max)/2 
        two_x_offset_init = (two_x_offset_init_min+two_x_offset_init_max)/2
        one_amplitude_init = amp_mu
        two_amplitude_init = amp_mu

    # Use Least Squares on all cases
    p = Parameters()
    p.add('one_x_offset', 
          one_x_offset_init, 
          min=one_x_offset_init_min , 
          max=one_x_offset_init_max, 
          vary=True)
    
    p.add('two_x_offset', 
          two_x_offset_init, 
          min=two_x_offset_init_min , 
          max=two_x_offset_init_max, 
          vary=True)

    p.add('one_amplitude', 
          one_amplitude_init,
          min=amp_a,
          max=amp_b,
          vary=True) #warning: max >= 2 causes n=2 & noise to be fitted on a tau~0 2ph trace. 

    p.add('two_amplitude', 
          two_amplitude_init,
          min=amp_a,
          max=amp_b,
          vary=True) #warning: max >= 2 causes n=2 & noise to be fitted on a tau~0 2ph trace. 

    result = two_pulse_fit.fit(np.array(signal),
                               x=np.array(time),
                               params=p,
                               weights=1/sigma0
                               )
    # print left_edges, right_edges
    return result, mcmc_flag, unequal_edges

def testcw(fname, t_initial, t_final, two_pulse_fit, one_pulse,
  sigma0, 
                 amp_mu,
                 amp_tau,
                 amp_a,
                 amp_b,
  sampling, burn, thin, height_th, Plot=True, debug=False):
    result, mcmc_flag, unequal_edges = fit_two_cw(*hpa.trace_extr(fname, height_th,
                                                  t_initial=t_initial, t_final=t_final), 
                               sigma0=sigma0,    # signal noise
                                   amp_mu=amp_mu,
                                   amp_tau=amp_tau,
                                   amp_a=amp_a,
                                   amp_b=amp_b,
                                  two_pulse_fit=two_pulse_fit,  
                                  one_pulse=one_pulse,
                                  height_th=height_th, 
                                  sampling=sampling,
                                  burn=burn,
                                  thin=thin,
                                  Plot=Plot,
                                  debug=debug)
    # pymc.gelman_rubin() #https://pymc-devs.github.io/pymc/modelchecking.html
    if Plot:
        plt.figure()
        plt.plot(*hpa.trace_extr(fname,height_th,t_initial,t_final), label='unwindowed', color='grey')
        result.plot_fit(data_kws={'alpha':0.5, 'marker':'.'})
        # plt.savefig(results_directory+'testmcmccw.pdf')
        print result.fit_report()
        print ('arrival time difference = {:.2f}ns'.\
               format((result.init_values['one_x_offset']-result.init_values['two_x_offset'])*1e9))
    return result, mcmc_flag, unequal_edges

# def testcw(i, sampling, burn, thin, height_th, Plot=True):
#     result, mcmc_flag, unequal_edges = fit_two_cw(*hpa.trace_extr(filelist_cont[:10000][mask_2ph_cont][i], 
#                                                   t_initial, t_final), 
                            
#                                   height_th=height_th, 
#                                   sampling=sampling,
#                                  burn=burn,
#                                  thin=thin,
#                                 debug=True)
#     # pymc.gelman_rubin() #https://pymc-devs.github.io/pymc/modelchecking.html
#     if Plot:
#         plt.figure()
#         plt.plot(*hpa.trace_extr(filelist_cont[mask_2ph_cont][i]), label='unwindowed', color='grey', alpha=0.5)
#         result.plot_fit(data_kws={'alpha':0.5, 'marker':'.'})
#         plt.savefig(results_directory+'testmcmccw.pdf')
#         print result.fit_report()
#         print ('arrival time difference = {:.2f}ns'.\
#                format((result.init_values['one_x_offset']-result.init_values['two_x_offset'])*1e9))
#     return result, mcmc_flag, unequal_edges