"""
This script compares the fitting accuracy of the single photon pulse model with and without horizontal shift correction.
"""
import sys
sys.path.insert(0,'/workspace/projects/TES/scripts/')
from lmfit import Model
from lmfit import Parameters
# import pulse_fit_v05 as pf 
import heralded_pulses_analysis as hpa
import tqdm
import numpy as np
import matplotlib.pyplot as plt
def FWHM(X,Y,plot=True):
    X=np.array(X)
    Y=np.array(Y)
    half_max = np.max(Y) / 2
    #find when function crosses line half_max (when sign of diff flips)
    #take the 'derivative' of signum(half_max - Y[])
    d = np.sign(half_max - np.array(Y[0:-1])) - np.sign(half_max - np.array(Y[1:]))
    #plot(X,d) #if you are interested
    #find the left and right most indexes
    
    left_idx = np.where(d > 0)
    right_idx = np.where(d < 0)

    left = X[left_idx][0]
    right = X[right_idx][-1]
    print left, right
    fwhm = right-left
    if plot:
        plt.annotate('', (left, half_max), (right, half_max), arrowprops={'arrowstyle':'<->'})
        plt.text((left+right)/2, half_max, '{} ns'.format(fwhm), ha='center')
    return right-left #return the difference (full width)
def trace_simple_ave(filelist, height_th, t_initial=None, t_final=None):
    time, _ = hpa.trace_extr(filelist[0], t_initial, t_final)

    a = [hpa.trace_extr(file, t_initial, t_final)[1]
         for file
         in filelist]
    # reduce to trace length to avoid edge effects

    idx_0 = int(len(time) / 30)
    v_len = len(time) - 2 * idx_0
    time = time[idx_0:idx_0 + v_len]
    a = [line[idx_0:idx_0 + v_len] for line in a]
    amean = np.nanmean(a, 0)
    bg = np.median(amean[amean<height_th])
    return time, amean-bg, np.std(amean-bg)

if __name__ == '__main__':
    directory_name = '/workspace/projects/TES/data/20170126_TES5_n012_distinguishibility_20MHz_tau_vs_offset/single/'
    results_directory = '/workspace/projects/TES/analysis/20170126_TES5_n012_distinguishibility_20MHz_tau_vs_offset_results/'
    filelist = np.array(glob.glob(directory_name + '*.trc'))

    # we limit the temporal length of the traces
    t_initial = None
    t_final = None
    height_th = 0.0075
    mask_1ph = np.load(results_directory+'mask_1ph.npy')
    time_s, signal_s = trace_simple_ave(filelist[np.where(mask_1ph)])

    def fit_one(time,signal,height_th, time_model, signal_model):
        def one_pulse(x, x_offset=0, amplitude=1):
            """convert the sample single photon pulse into a function
            that can be used in a fit
            """
            x = x - x_offset
            return amplitude * np.interp(x, time_model, signal_model)
        try:
            p = Parameters()
            p.add('x_offset', time[hpa.find_idx(signal[:np.argmax(signal)],height_th)], vary=1)
            p.add('amplitude', 1, vary=1)
            result = Model(one_pulse).fit(signal, x=time, params=p)
            return result
        except:
            pass

    results = [fit_one(*hps.trace_extr(f,h_th=height_th),
        height_th=height_th,
        time_model=time_s,
        signal_model=signal_s) for f in tqdm.tqdm(filelist[np.where(mask_1ph)][:])]

    results_f = [fit_one(*hps.trace_extr(f,h_th=height_th),
    height_th=height_th,
    time_model=time_f,
    signal_model=signal_fs) for f in tqdm.tqdm(filelist[np.where(mask_1ph)][:])]

    """Extract Data"""
    chisqrs = [r.chisqr for r in results]
    chisqrs_f = [r.chisqr for r in results_f]
    amps = [r.best_values['amplitude'] for r in results]
    amps_f = [r.best_values['amplitude'] for r in results_f]
    shifts = [r.best_values['x_offset'] for r in results]
    shifts_f = [r.best_values['x_offset'] for r in results_f]

    """Chisqrs"""
    plt.figure()
    plt.tight_layout
    plt.hist(chisqrs, bins=60, range=[0.005,0.02], alpha=0.5, label='without shift', histtype='step', linestyle='--')
    plt.hist(chisqrs_f, bins=60, range=[0.005,0.02], alpha=0.5, label='with shift', histtype='step')
    plt.title('chisq distribution of single photon pulse fitting\ncomparison between shift-corrected and no-corrected model pulse')
    plt.xlabel('chisq')
    plt.legend()
    plt.savefig(results_directory+'shift_corrected_pulse_model_comparison.pdf')
    plt.show()

    """Amplitudes"""
    plt.figure()
    plt.hist(amps, bins=60, range=[0,2], alpha=0.5, label='without shift', histtype='step', linestyle='--')
    plt.hist(amps_f, bins=60, range=[0,2], alpha=0.5, label='with shift', histtype='step')
    plt.title('')
    plt.xlabel('amps')
    plt.legend()
    plt.show()

    """Arrival Times"""
    plt.figure()
    plt.hist(np.array(shifts)*1e9, bins=200, range=[-200,700], alpha=0.5, label='without shift', histtype='step', linestyle='--')
    plt.hist(np.array(shifts_f)*1e9, bins=200, range=[-200,700], alpha=0.5, label='with shift', histtype='step')
    plt.title('')
    plt.xlabel('offsets')
    plt.legend()
    
    t_hist = np.histogram(np.array(shifts_f)*1e9, bins=200, range=[-200,700])
    FWHM(t_hist[1]+np.diff(t_hist[1])[0]/2, t_hist[0])
    plt.show()
# class extract_single():
#     def __init__(self,results):
#         self.results
