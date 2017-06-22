import lecroy
import numpy as np
import glob
import matplotlib.pyplot as plt
import tqdm
from scipy.signal import savgol_filter
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
from pulse_discrimination import discriminator

def find_idx(time_v, t0):
    return np.argmin(np.abs(time_v - t0))

def find_bg(signal):
    freq, ampl = np.histogram(signal, 500)
    freq_f = savgol_filter(freq, 31, 1)
    # print 'binsize = {}'.format(np.diff(ampl)[0])
    # plt.plot(ampl[:-1],freq_f)
    return ampl[np.argmax(freq_f)]

def param_extr(filename, t_initial=None, t_final=None, h_th=0.0075, t0=.56e-6):
    """extract relevant parameters from a trace stored in a file
    """
    trc = lecroy.LecroyBinaryWaveform(filename)
    # time = trc.mat[:, 0][::10]
    # signal = trc.mat[:, 1][::10]
    time = trc.mat[:, 0][::1]
    signal = trc.mat[:, 1][::1]
    
    """
    Consider only signal between t_initial and t_final
    """
    idx_0 = 0
    idx_1 = -1
    if t_initial is not None:
        idx_0 = find_idx(time, t_initial)
    if t_final is not None:
        idx_1 = find_idx(time, t_final)
    time = time[idx_0:idx_1]
    signal = signal[idx_0:idx_1]
    """
    Background Correction
    """
    # bg = np.median(signal[signal<h_th])
    bg = find_bg(signal)
    signal = signal - bg
 
    """
    Mask traces to reject noise
    Clamp traces to reject half pulses at edges
    """
    [mask, clamp, edges, left_edges, right_edges] = discriminator(time, signal, 
                                                                  dt_left=0*300e-9,dt_right=1300e-9, 
                                                                  height_th=h_th, 
                                                                  Plot=False, 
                                                                  method=2)
    window = mask&clamp

    """
    Extract properties
    """
    height = np.max(signal)

    # obtain height within region not containing fractional pulses
    if (len(left_edges)>len(right_edges)):
        height_clamped = np.max(signal[:left_edges[-1]])

    if (len(left_edges)<len(right_edges)):
        height_clamped = np.max(signal[right_edges[0]:])
     # detect max height of trace excluding edges of trace
    if (len(left_edges)==len(right_edges)):
        if len(left_edges)==0:
            height_clamped = np.max(signal)
        else:
            if left_edges[0]<right_edges[-1]:
                height_clamped = np.max(signal[clamp])
            else:
                height_clamped = np.max(signal[right_edges[-1]:left_edges[0]])



    area_win = np.sum(np.abs(signal[mask&clamp]))
    # if np.sum(clamp) == 0:
    #     area_win = np.sum(np.abs(signal))
    # else:
    #     area_win = np.sum(np.abs(signal[clamp]))
    # try:
    #     t_left = time[left_edges[0]]
    #     t_right = time[right_edges[-1]]
    #     area_win = np.sum(np.abs(signal[(window)&(t_left<time)&(time<t_right)]))
    # except:
    #     area_win = 0
    """
    Other parameters tried previously...
    """
    # area_abs = np.sum(np.abs(signal[signal>h_th]))
    area_abs = np.sum(np.abs(signal))
    heightattime=signal[find_idx(time,t0)]
    timeofarrival=time[find_idx(signal,h_th)]

    return np.array((area_win, 
                     area_abs, 
                     height, 
                     height_clamped, 
                     heightattime, 
                     bg),
        dtype=[('area_win','float64'),
        ('area_abs','float64'),
        ('height','float64'),
        ('height_clamped','float64'),
        ('heightattime','float64'),
        ('bg','float64')]
        )

def trace_extr(filename, h_th, t_initial=None, t_final=None, zero=True):
    """extract relevant parameters from a trace stored in a file
    """
    trc = lecroy.LecroyBinaryWaveform(filename)
    time = trc.mat[:, 0]
    signal = trc.mat[:, 1]
    # _,signal = butter_lowpass_filter(trc,1000e3)
    
    idx_0 = 0
    idx_1 = -1
    if t_initial is not None:
        idx_0 = find_idx(time, t_initial)
    if t_final is not None:
        idx_1 = find_idx(time, t_final)
    time = time[idx_0:idx_1]
    signal = signal[idx_0:idx_1]

    bg = find_bg(signal)
    # bg = np.median(signal[signal<h_th])
    signal = signal - bg

    if zero:
        time=time-np.mean(time)

    return np.array(time), np.array(signal)

def std_extr(filename,height_th,t_initial=None, t_final=None):
    _, signal = trace_extr(filename, height_th, t_initial, t_final)
    return np.std(signal)


def shift(xs, n):
    """ shifting array xs by n positions """
    if n == 0:
        return xs
    e = np.empty_like(xs)
    if n >= 0:
        e[:n] = np.nan
        e[n:] = xs[:-n]
    else:
        e[n:] = np.nan
        e[:n] = xs[-n:]
    return e

def baseline_als(y, lam=10**9, p=.00001, niter=10):
    """"
    Returns baseline of slowly varying pulse. NOT a low-pass filter.
    Asymmetric Least Squares Smoothing" by P. Eilers and H. Boelens in 2005 
    lam for smoothness 10**2< = lam < = 10**9
    p for asymmetry 0.001 <= p <= 0.1 is a good choice (for a signal with positive peaks)
    """
    L = len(y)
    D = sparse.csc_matrix(np.diff(np.eye(L), 2))
    w = np.ones(L)
    for i in xrange(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = spsolve(Z, w*y)
        w = p * (y > z) + (1-p) * (y < z)
    return z

def baseline_correction(y, lam=10**8, p=.001, niter=10):
    y = y - baseline_als(y, lam=lam, p=p, niter=niter)
    y = y - find_bg(y) 
    return y

# def pplot(filelist,density=10,plot_every=5):
#   """generates a lightweight plot of some sample traces
#   WARNING: does not automatically remove trace dc offset
#   :params density: plot every 'density' number of traces
#   """
#   plt.figure()
#   for f in filelist[::density]:
#       trc = lecroy.LecroyBinaryWaveform(f)
#       time = trc.mat[:,0]
#       signal = trc.mat[:,1]
#       # plot_every = int(len(time)/100)
#       plt.plot(time[::plot_every]*1e6,signal[::plot_every],alpha=0.2)
#       plt.xlabel('time(us)')
def pplot(filelist, height_th, t_initial=None, t_final=None, density=10,plot_every=5):
    """generates a lightweight plot of some sample traces
    WARNING: does not automatically remove trace dc offset
    :params density: plot every 'density' number of traces
    """
    plt.figure()
    for f in filelist[::density]:
        time, signal = trace_extr(f, t_initial=t_initial, t_final=t_final, h_th=height_th)
        # plot_every = int(len(time)/100)
        plt.plot(time[::plot_every]*1e6,signal[::plot_every],alpha=0.2)
        plt.xlabel('time(us)')

def save_sample_traces(filelist,sample_trace_directory_name=None):
    for f in tqdm.tqdm(filelist):
        trc = lecroy.LecroyBinaryWaveform(f)
        time = trc.mat[:,0]
        signal = trc.mat[:,1]
        fname = f.strip('.trc').split('/')[-1]
        np.savetxt(sample_trace_directory_name+fname+'.txt', np.array(zip(time,signal)))

def load_sample_traces(sample_trace_directory_name):
    sample_traces = []
    for f in glob.glob(sample_trace_directory_name+'*.txt'):
        trc = np.loadtxt(f)
        time = trc[:,0]
        signal = trc[:,1]
        # print time,signal
        sample_traces.append(np.array(zip(time,signal)))
    return np.array(sample_traces)

if __name__ == '__main__':
    
    directory_name = '/workspace/projects/TES/data/20170126_TES5_n012_distinguishibility_20MHz_tau_vs_offset/single/'
    results_directory = '/workspace/projects/TES/analysis/20170126_TES5_n012_distinguishibility_20MHz_tau_vs_offset_results/'
    filelist = np.array(glob.glob(directory_name + '*.trc'))

    # we limit the temporal length of the traces
    t_initial = None
    t_final = None
    height_th = 0.0075
    min_peak_sep = 5

    # reads the traces one by one and extract relevant parameters into
    # numpy structure
    data = np.array([param_extr(f, t_initial, t_final, h_th=height_th)
                     for f
                     in tqdm.tqdm(filelist[:10000])])

    heights = data['height']
    areas = data['area_win']
    # use the area to count the number of photons
    pnr_height = np.histogram(heights, 400)
    pnr = np.histogram(areas, 400)

    # remove the first bin.
    # because of the threshold filtering it only makes life
    # complicated
    pnr = [pnr[0][1:], pnr[1][1:]]

    # find the thresholds by fitting the distribution with gaussian peaks
    th = thres_poiss.thresholds_N(pnr, min_peak_sep, weighted=True)

    # select only 1-photon traces
    mask_1ph = (areas > pnr[1][1]) & (areas < th[0])

    # select only 2-photon traces
    mask_2ph = (areas > th[0]) & (areas < th[1])

    """diagnostics tools 
    Jianwei
    """
    trc_double = np.array([trace_extr(f, t_initial, t_final) for f in filelist[mask_2ph]])
