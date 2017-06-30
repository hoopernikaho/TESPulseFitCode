# Step 1: Create Spectral Probability Density Function from Noise Spectrum
# Step 1a: Create discrete spectral probability distribution
def bg_fft(bg_filelist,height_th,numfiles=1000):
    """
    imports files from filelist containing background traces
    traces may have stray photons: these are removed by detecting traces with max smaller than height_th
    """
    for f in tqdm.tqdm(bg_filelist[:numfiles]):
        t, s = hpa.trace_extr(f)
        if np.max(s) < height_th:
            freq, prob = FFT(t, s)
            probs.append(prob)
    probs_mean=np.nanmean(probs,axis=0)
    freq,_ = FFT(*hpa.trace_extr(bg_filelist[0]))
    return freq, probs_mean
# Step 1b: Create continuous spectral probability density function
