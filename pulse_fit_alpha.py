"""
Pulse Fit Code.
Priors are pulse area distributions obtained from CW traces.
"""
# MCMC
def fit_two_mcmc_adp(time, signal, 
                 amplitude_init_min=0.7, amplitude_init_max=1.5, 
                 one_x_offset_init=None, 
                 Plot=False, debug=False, 
                 sampling=7e3, burn=20, thin=6,
                auto=False):
    
    # LIMIT SEARCH FOR OFFSETS
    _t_initial=time[pd.srlatch_rev(signal,0)][0] 
#     _t_final=time[pd.srlatch_rev(signal,np.min(signal))][-1] 
    _t_final=time[pd.srlatch_rev(signal,0)][-1] 
    
    def model(x, f): 
        # PRIORS
        y_err = sigma0
        print (_t_initial,_t_final, one_x_offset_init)
        one_x_offset = pymc.Uniform("one_x_offset", _t_initial, _t_final, value=_t_initial)
        two_x_offset = pymc.Uniform("two_x_offset", _t_initial, _t_final, value=_t_final)
        one_x_amplitude = pymc.TruncatedNormal("one_x_amplitude", 
                                               mu=area_mus[1]/2/area_mus[0],
                                               tau=1.0/((area_sigmas[0]/area_mus[0])**2), 
                                               a=th_areas[0][0]/area_mus[0], 
                                               b=th_areas[0][1]/area_mus[0], 
                                               value=area_mus[1]/2/area_mus[0]) #sigma/mu is the n=1 std deviation in units of n=1 amplitude
        two_x_amplitude = pymc.TruncatedNormal("two_x_amplitude", 
                                               mu=area_mus[1]/2/area_mus[0],
                                               tau=1.0/((area_sigmas[0]/area_mus[0])**2), 
                                               a=th_areas[0][0]/area_mus[0], 
                                               b=th_areas[0][1]/area_mus[0], 
                                               value=area_mus[1]/2/area_mus[0])
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
        scales={MDL.one_x_amplitude:area_sigmas[0]/area_mus[0], 
                MDL.two_x_amplitude:area_sigmas[0]/area_mus[0]}, 
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
        print [EstBurn, EstSampling, EstThin]
        # actual run
        MDL.sample(iter=EstSampling, burn=EstBurn, thin=EstThin, verbose=0)
    else:
        MDL.sample(iter=sampling, burn=burn, thin=thin, verbose=0)  
    # thin: consider every 'thin' samples
    # burn: number of samples to discard: decide by num of samples to run till parameters stabilise at desired precision
    if Plot:
        y_fit = MDL.mod_two_pulse.value #get mcmc fitted values
        plt.plot(time, signal, 'b', marker='o', ls='-', lw=1, label='Observed')
        plt.plot(time,y_fit,'k', marker='+', ls='--', ms=5, mew=2, label='Bayesian Fit Values')
        plt.legend()      
    if debug:
        pymc.Matplot.plot(MDL)
    return MDL #usage: MDL.one_x_offset.value for fitted result

# LEAST SQUARES - performs also MCMC within it
# Use average pulse to generate single photon model
def one_pulse(x, x_offset=0, amplitude=1):
    """convert the sample single photon pulse into a function
    that can be used in a fit
    """
    x = x - x_offset
    return amplitude * np.interp(x, time_f, signal_f)

two_pulse_fit = Model(one_pulse, prefix='one_') + \
        Model(one_pulse, prefix='two_')

def fit_two_cw_adp(time, signal, 
            sampling, burn, thin,
            height_th=height_th, 
            debug=False):
    
    # Identify Pulse Region
    [mask, clamp, edges, left_edges, right_edges] = pd.discriminator(time, signal, 
                                                                     dt_left=0*200e-9,
                                                                     dt_right=0*700e-9, 
                                                                     height_th=height_th, 
                                                                     Plot=True, 
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
        result_mcmc = fit_two_mcmc_adp(time[::1], signal[::1], 
                               sampling=sampling, 
                               burn=burn,
                               thin=thin,
                               debug=debug)
        
        # Use MCMC results to initialise least squares fit
        one_x_offset_init = result_mcmc.one_x_offset.value
        two_x_offset_init = result_mcmc.two_x_offset.value
        one_x_offset_init_min = None; one_x_offset_init_max = None
        two_x_offset_init_min = None; two_x_offset_init_max = None
        
        one_amplitude_init = result_mcmc.one_x_amplitude.value
        two_amplitude_init = result_mcmc.two_x_amplitude.value
        
    if mcmc_flag==False:
        one_x_offset_init = time[left_edges][0]
        two_x_offset_init = time[left_edges][1]
        one_x_offset_init_min = time[left_edges][0]; one_x_offset_init_max = time[right_edges][0]
        two_x_offset_init_min = time[left_edges][1]; two_x_offset_init_max = time[right_edges][1]
        one_amplitude_init = area_mus[1]/2/area_mus[0]
        two_amplitude_init = area_mus[1]/2/area_mus[0]
        
    # Use Least Squares on all cases
    p = Parameters()
    p.add('one_x_offset', 
          one_x_offset_init, 
          min=one_x_offset_init_min , 
          max=one_x_offset_init_max, 
          vary=1)
    
    p.add('two_x_offset', 
          two_x_offset_init, 
          min=two_x_offset_init_min , 
          max=two_x_offset_init_max, 
          vary=1)
    
    p.add('one_amplitude', 
          one_amplitude_init, 
          min=(area_mus[1]-4*area_sigmas[1])/2/area_mus[0],
          max=(area_mus[1]+4*area_sigmas[1])/2/area_mus[0], 
          vary=1) #warning: max >= 2 causes n=2 & noise to be fitted on a tau~0 2ph trace. 
    
    p.add('two_amplitude', 
          two_amplitude_init, 
          min=(area_mus[1]-4*area_sigmas[1])/2/area_mus[0], 
          max=(area_mus[1]+4*area_sigmas[1])/2/area_mus[0], 
          vary=1)

    result = two_pulse_fit.fit(signal,
                               x=time,
                               params=p,
                               )
    
    return result, mcmc_flag, unequal_edges

def testcwadp(i, sampling, burn, thin, height_th, Plot=True):
    result, mcmc_flag, unequal_edges = fit_two_cw_adp(*hpa.trace_extr(filelist_cont[:10000][mask_2ph_cont][i], 
                                                  t_initial, t_final), 
                            
                                  height_th=height_th, 
                                  sampling=sampling,
                                 burn=burn,
                                 thin=thin,
                                debug=True)
    # pymc.gelman_rubin() #https://pymc-devs.github.io/pymc/modelchecking.html
    if Plot:
        plt.figure()
        plt.plot(*hpa.trace_extr(filelist_cont[mask_2ph_cont][i]), label='unwindowed', color='grey', alpha=0.5)
        result.plot_fit(data_kws={'alpha':0.5, 'marker':'.'})
        plt.savefig(results_directory+'testmcmccw.pdf')
        print result.fit_report()
        print ('arrival time difference = {:.2f}ns'.\
               format((result.init_values['one_x_offset']-result.init_values['two_x_offset'])*1e9))
    return result, mcmc_flag, unequal_edges