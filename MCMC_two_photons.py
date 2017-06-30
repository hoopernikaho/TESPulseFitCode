import pymc
import numpy as np
import pd

tau = (mu / sigma)**2
sigma_n = sigma / mu
th1 = th[0][0] / mu
th2 = th[1][0] / mu


def one_pulse(x, x_offset, amplitude):
    pass


def weight_noise(signal):
    pass


tau_rms = weight_noise(signal)


def fit_two_mcmc(time,
                 signal,
                 sigma_ampl,
                 tau_rms,
                 ampl_limit=2.5,
                 sampling=7e3,
                 burn=20,
                 thin=6,
                 f_model):
    """[summary]

    [description]

    Arguments:
        time {[type]} -- [description]
        signal {[type]} -- [description]

    Keyword Arguments:
        one_x_offset_init {[type]} -- [description] (default: {None})
        Plot {bool} -- [description] (default: {False})
        debug {bool} -- [description] (default: {False})
        sampling {number} -- [description] (default: {7e3})
        burn {number} -- number of samples to discard: decide by num of samples
             untill parameters stabilise at desired precision (default: {20})
        thin {number} -- consider every 'thin' samples (default: {6})
        auto {bool} -- [description] (default: {False})

    Returns:
        [MCMC object] -- usage: MDL.one_x_offset.value for fitted result
    """

    # LIMIT SEARCH FOR OFFSETS
    mask = time[pd.srlatch_rev(signal, 0)]
    _t_initial, _t_final = mask[0], mask[-1]
    _t_peak = time(np.argmax(signal[mask]))

    tau = (1 / sigma_ampl)**2

    def model(x, f):
        """ priors distributions
        """
        one_x_offset = pymc.Uniform("one_x_offset",
                                    _t_initial,
                                    _t_peak,
                                    value=_t_initial)
        two_x_offset = pymc.Uniform("two_x_offset",
                                    _t_initial,
                                    _t_final,
                                    value=_t_final)

        one_amplitude = pymc.TruncatedNormal("one_amplitude",
                                             mu=1,
                                             tau=tau,
                                             a=1 - ampl_limit * sigma_ampl,
                                             b=1 + ampl_limit * sigma_ampl,
                                             value=1)
        two_amplitude = pymc.TruncatedNormal("two_amplitude",
                                             mu=1,
                                             tau=tau,
                                             a=1 - ampl_limit * sigma_ampl,
                                             b=1 + ampl_limit * sigma_ampl,
                                             value=1)

        @pymc.deterministic(plot=False)
        def mod_two_pulse(x=time,
                          one_x_offset=one_x_offset,
                          two_x_offset=two_x_offset,
                          one_amplitude=one_amplitude,
                          two_amplitude=two_amplitude):
            return f_model(x, x_offset=one_x_offset,
                           amplitude=one_amplitude) +\
                f_model(x, x_offset=two_x_offset,
                        amplitude=two_amplitude)

        y = pymc.Normal("y",
                        mu=mod_two_pulse,
                        tau=tau_rms,
                        value=signal,
                        observed=True)
        return locals()

    # The sample is stored in a Python serialization (pickle) database
    MDL = pymc.MCMC(model(time, signal), db='pickle')
    MDL.use_step_method(pymc.AdaptiveMetropolis,
                        [MDL.one_x_amplitude, MDL.two_x_amplitude],
                        scales={MDL.one_x_amplitude: sigma_n,
                                MDL.two_x_amplitude: sigma_n},
                        )

    MDL.sample(iter=sampling, burn=burn, thin=thin, verbose=0)
    return MDL

# if __name__ == '__main__':
#     main()

    # if Plot:
    #     y_fit = MDL.mod_two_pulse.value  # get mcmc fitted values
    #     plt.plot(time, signal, 'b', marker='o', ls='-', lw=1, label='Observed')
    #     plt.plot(time, y_fit, 'k', marker='+', ls='--',
    #              ms=5, mew=2, label='Bayesian Fit Values')
    #     plt.legend()
    # if debug:
    #     pymc.Matplot.plot(MDL)
