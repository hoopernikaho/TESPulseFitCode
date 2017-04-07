"""
Performs a fit on deltat=0 2-photon traces,
    Compares mcmc fit with least squares
"""
import sys
import numpy as np
import glob
import tqdm
sys.path.append('/workspace/projects/TES/scripts/TESPulseFitCode')
sys.path.insert(0,'/workspace/projects/TES/scripts/')
import heralded_pulses_analysis as hpa
import pulse_discrimination as pf


height_th = 0.0075
mask_2ph = (areas>0.9) & (areas<1.6)
mask_2ph = np.load(results_directory+'mask_2ph.npy')

ph1_model = np.load(results_directory + 'ph1_model.npy')
time_f, signal_fs = ph1_model[:,0], ph1_model[:,1]
time_f, signal_fs = time_f[:-100], signal_fs[:-100]

def fit_two_mcmc(time, signal, amplitude_init_min=0.7, amplitude_init_max=1.5, one_x_offset_init=None, Plot=False, debug=False, sampling=1e2):
    
    _t_initial=time[srlatch_rev(signal,np.min(signal))][0] #limit search for offset
    _t_final=time[srlatch_rev(signal,np.min(signal))][-1]
    def model(x, f): 
        #priors
        y_err = pymc.Uniform("sig", 0.0005, .00141552889104*3, value=.00141552889104)
        # y_err = .00141552889104
        print (_t_initial,_t_final, one_x_offset_init)
        one_x_offset = pymc.Uniform("one_x_offset", _t_initial, _t_final, value=_t_initial)
        two_x_offset = pymc.Uniform("two_x_offset", _t_initial, _t_final, value=_t_final)
        # amp_sum = pymc.Normal("amp_sum", mu=2*1.0267, tau=1.0/(2*0.15693**2), value=2*1.0267)
        # amp_diff = pymc.Normal("amp_diff", mu=0, tau=1.0/(2*0.15693**2), value=0)
        # one_x_amplitude = (amp_sum+amp_diff)/2
        # two_x_amplitude = (amp_sum-amp_diff)/2
        # one_x_amplitude = pymc.Uniform("one_x_amplitude", amplitude_init_min, amplitude_init_max, value=1.0)
        # two_x_amplitude = pymc.Uniform("two_x_amplitude", amplitude_init_min, amplitude_init_max, value=1.0)
        one_x_amplitude = pymc.Normal("one_x_amplitude", mu=1.0267, tau=1.0/((0.15693)**2), value=1.0267)
        two_x_amplitude = pymc.Normal("two_x_amplitude", mu=1.0267, tau=1.0/((0.15693)**2), value=1.0267)
        #model
        @pymc.deterministic(plot=False)
        def mod_two_pulse(x=time, one_x_offset=one_x_offset, two_x_offset=two_x_offset, one_x_amplitude=one_x_amplitude, two_x_amplitude=two_x_amplitude):
              return one_pulse(x, x_offset=one_x_offset, amplitude=one_x_amplitude)+one_pulse(x, x_offset=two_x_offset, amplitude=two_x_amplitude)

        #likelihoodsy
        y = pymc.Normal("y", mu=mod_two_pulse, tau= 1.0/y_err**2, value=signal, observed=True)
        return locals()

    MDL = pymc.MCMC(model(time,signal), db='pickle') # The sample is stored in a Python serialization (pickle) database
    # MDL.use_step_method(pymc.Metropolis,MDL.amp_sum,proposal_sd=1.41*0.15693)
    # MDL.use_step_method(pymc.Metropolis,MDL.amp_diff,proposal_sd=1.41*0.15693)
    MDL.use_step_method(pymc.AdaptiveMetropolis, 
        [MDL.one_x_amplitude, MDL.two_x_amplitude, MDL.y_err],
        scales={MDL.one_x_amplitude:0.15693, MDL.two_x_amplitude:0.15693, MDL.y_err:0.00141552889104*3}, 
        # [MDL.one_x_amplitude, MDL.two_x_amplitude, MDL.one_x_offset, MDL.two_x_offset],
        # scales={MDL.one_x_amplitude:0.15693, MDL.two_x_amplitude:0.15693, MDL.one_x_offset:1e-6, MDL.two_x_offset:1e-6}, 
        # scale: https://github.com/pymc-devs/pymc-doc/blob/master/_sources/modelfitting.txt 
        # [MDL.one_x_amplitude, MDL.two_x_offset,MDL.y_err,MDL.one_x_offset,MDL.two_x_offset] 
        # cov=np.array(0.15693**2*np.matrix([[1,-.7],[-.8,1]])),
        # delay=1#steps to delay before using empirical cov matrix. keep this small to start discovering anticorr between amplitudes early. 
        )
    MDL.sample(iter=sampling, burn=int(sampling*.95), thin=10)  
    #thin: consider every 'thin' samples
    #burn: number of samples to discard - decide by num of samples to run till parameters stabilise at desired precision

    if Plot:
        y_fit = MDL.mod_two_pulse.value #get mcmc fitted values
        plt.figure()
        plt.plot(time, signal, 'b', marker='o', ls='-', lw=1, label='Observed')
        plt.plot(time,y_fit,'k', marker='+', ls='--', ms=5, mew=2, label='Bayesian Fit Values')

        plt.legend()
        plt.show()
    if debug:
        # pymc.Matplot.plot(MDL)
        
        samples = np.array([MDL.one_x_amplitude.trace(), MDL.two_x_amplitude.trace(),
                    MDL.one_x_offset.trace(), MDL.two_x_offset.trace(),
                    # MDL.y_err.trace()
                    ]).T
        samples=samples[0]
        import triangle
        tmp=triangle.corner(samples[:,:], labels=['one_amp','two_amp','one_offset','two_offset','y_err'])
        plt.show()

    return MDL #usage: MDL.one_x_offset.value for fitted result

def fit_two(time, signal, height_th,filterpts=301, order=3, thres=0.7, min_dist=20, sampling=2e2, debug=False):
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
    # idx_s = edges_by_diff_hys(time, signal)
    p = Parameters()

    [mask, clamp, edges, left_edges, right_edges] = discriminator(time, signal, dt_left=300e-9,dt_right=800e-9, height_th=h_th, Plot=False, method=2)
    window = mask&clamp
    # idx_s = np.array([i for i in idx_s if window[i]])

    # if len(idx_s) < 2: #does mcmc fit to initialise lmfit - duplicate work yes, but recoding the results array is more work :P
    # result_mcmc = fit_two_mcmc(time[window][::10], signal[window][::10], sampling=sampling, debug=debug)
    result_mcmc = fit_two_mcmc(time[::10], signal[::10], sampling=sampling, debug=debug)

    p.add('one_x_offset', result_mcmc.one_x_offset.value, vary=1)
    p.add('two_x_offset', result_mcmc.two_x_offset.value, vary=1)
    p.add('one_amplitude', result_mcmc.one_x_amplitude.value, min=0.5, max=1.5, vary=1) #warning: max >= 2 causes n=2 & noise to be fitted on a tau~0 2ph trace. 
    p.add('two_amplitude', result_mcmc.two_x_amplitude.value, min=0.5, max=1.5, vary=1)

    # if len(idx_s) >= 2:
    #     p.add('one_x_offset', time[idx_s[0]])
    #     p.add('two_x_offset', time[idx_s[1]])
    #     p.add('one_amplitude', .7, min=0.5, max=1, vary=1) #warning: max >= 2 causes n=2 & noise to be fitted on a tau~0 2ph trace. 
    #     p.add('two_amplitude', 1.3, min=1, max=1.5, vary=1
    #       # expr='one_amplitude'
    #       )
    # yerr = 0.001 + 0.001*(1-signal/np.max(signal))
    result = two_pulse_fit.fit(signal,
                               x=time,
                               params=p,
                               # weights=1 / yerr,
                               # method=method
                               )
    return result, result_mcmc

"""Import source traces"""
directory_name_source = '/workspace/projects/TES/data/20170126_TES5_n012_distinguishibility_20MHz_tau_vs_offset/200ns/'
filelist_200ns = np.array(glob.glob(directory_name_source + '*.trc'))
data_200ns = np.array([hpa.param_extr(f, t_initial=None, t_final=None, h_th=height_th)
                 for f
                 in tqdm.tqdm(filelist_200ns[:10000])])

"""Testing mcmc on one trace"""
mdl = fit_two_mcmc(*hpa.trace_extr(filelist_200ns[mask_2ph_200ns][0], t_initial, t_final), sampling=1.5e4)
def testmcmc(i, iters, height_th, Plot=True):
    result, result_mcmc = fit_two(*trace_extr(filelist_200ns[mask_2ph_200ns][i], t_initial, t_final, height_th=height_th), sampling=iters)
    # pymc.gelman_rubin() #https://pymc-devs.github.io/pymc/modelchecking.html
    if Plot:
        plt.figure()
        plt.plot(*trace_extr(filelist_200ns[mask_2ph_200ns][i]), label='unwindowed')
        plt.plot(result_mcmc.time, result_mcmc.signal, 'b', marker='o', ls='-', lw=1, label='Observed')
        plt.plot(result_mcmc.time, result_mcmc.mod_two_pulse.value,'k', marker='+', ls='--', ms=5, mew=2, label='Bayesian Fit Values')
        plt.legend()
        plt.show()
        print result.fit_report()
        pymc.Matplot.plot(result_mcmc)
    return result_mcmc
testmcmcsame = [testmcmc(0,1e4,Plot=False) for i in tqdm.tqdm(np.arange(20))] #repeat over the same signal to check for consistency
maxamp=np.maximum([i.one_x_amplitude.value for i in testmcmcsame_20ns_yerr],
    [i.two_x_amplitude.value for i in testmcmcsame_20ns_yerr])
minamp=np.minimum([i.one_x_amplitude.value for i in testmcmcsame_20ns_yerr],
    [i.two_x_amplitude.value for i in testmcmcsame_20ns_yerr])
plt.figure();plt.hist(maxamp,alpha=0.5);plt.hist(minamp,alpha=0.5)
plt.figure();plt.scatter(taus,maxamp)
plt.figure();plt.scatter([i.one_x_amplitude.value for i in testmcmcsame_20ns_yerr],
[i.two_x_amplitude.value for i in testmcmcsame_20ns_yerr], marker='.');plt.xlim(0.7,1.5);plt.ylim(0.7,1.5)

results_single_balamps_mcmc_norm_amp= np.array([fit_two(*trace_extr(file, t_initial, t_final), sampling=1.5e4)#1.5e4 for 10us trace, 2ns per pt, sampled every 20 pts.
                    for file
                    in tqdm.tqdm(filelist_200ns[mask_2ph_200ns][::5])])

processed_single_balamps_mcmc_norm_amp = extract(results_single_balamps_mcmc_norm_amp[:,0])
# results_directory = '/workspace/projects/TES/analysis/20170126_TES5_n012_distinguishibility_20MHz_tau_vs_offset_results/g2_single_diode_pulse_2photon_pulse/mcmc_norm_amps_8Mar/'
results_directory = '/workspace/projects/TES/analysis/20170126_TES5_n012_distinguishibility_20MHz_tau_vs_offset_results/fit_accuracy/200ns_8Mar/'

plt.figure()
plt.hist(processed_single_balamps_mcmc_norm_amp._one_x_offsets*1e9,400, range=(0,500))
plt.figure()
plt.hist(processed_single_balamps_mcmc_norm_amp._two_x_offsets*1e9,400, range=(0,500))
plt.figure()
plt.hist(processed_single_balamps_mcmc_norm_amp._one_amplitudes_init,40)
plt.savefig(results_directory+'single_photon_pulse_mcmc_one_amp.pdf')
plt.figure()
plt.hist(processed_single_balamps_mcmc_norm_amp._two_amplitudes_init,40)
plt.savefig(results_directory+'single_photon_pulse_mcmc_two_amp.pdf')
plt.figure()
plt.plot(processed_single_balamps_mcmc_norm_amp._one_amplitudes_init
    ,processed_single_balamps_mcmc_norm_amp._two_amplitudes_init,'o')
plt.savefig(results_directory+'single_photon_pulse_mcmc_one_vs_two_amp.pdf')
plt.figure()
plt.plot(np.maximum(processed_single_balamps_mcmc_norm_amp._one_amplitudes_init,processed_single_balamps_mcmc_norm_amp._two_amplitudes_init),
    np.abs(processed_single_balamps_mcmc_norm_amp._one_x_offsets_init
    -processed_single_balamps_mcmc_norm_amp._two_x_offsets_init)*1e9,'o')
plt.xlabel('maximum amplitude')
plt.ylabel('time separation (ns)')
plt.savefig(results_directory+'single_photon_pulse_mcmc_maxamp_vs_g2.pdf')
plt.figure()
plt.hist(np.abs(processed_single_balamps_mcmc_norm_amp._one_x_offsets_init
    -processed_single_balamps_mcmc_norm_amp._two_x_offsets_init)*1e9,40, range=(0,350))
plt.savefig(results_directory+'single_photon_pulse_mcmc_g2.pdf')

plt.show()

processed_single_balamps_mcmc_norm_amp_less = np.array(
zip(processed_single_balamps_mcmc_norm_amp._one_x_offsets_init,
processed_single_balamps_mcmc_norm_amp._two_x_offsets_init,
processed_single_balamps_mcmc_norm_amp._one_amplitudes_init,
processed_single_balamps_mcmc_norm_amp._two_amplitudes_init),
dtype=[('one_x_offsets','float64'),
('two_x_offsets','float64'),
('one_amplitudes','float64'),
('two_amplitudes','float64')]
)

np.savetxt(results_directory+'200ns_separation_offset_amplitude_data_mcmc.txt',
    processed_single_balamps_mcmc_norm_amp_less, 
    header = '\t'.join(processed_single_balamps_mcmc_norm_amp_less.dtype.names))

processed_single_less = np.array(
zip(processed_single_balamps_mcmc_norm_amp._one_x_offsets,
processed_single_balamps_mcmc_norm_amp._two_x_offsets,
processed_single_balamps_mcmc_norm_amp._one_amplitudes,
processed_single_balamps_mcmc_norm_amp._two_amplitudes),
dtype=[('one_x_offsets','float64'),
('two_x_offsets','float64'),
('one_amplitudes','float64'),
('two_amplitudes','float64')]
)
np.savetxt(results_directory+'200ns_separation_offset_amplitude_data.txt',
    processed_single_less, 
    header = '\t'.join(processed_single_less.dtype.names))