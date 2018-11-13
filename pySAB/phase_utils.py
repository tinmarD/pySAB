import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.signal as signal
import tqdm


def bp_filter_1d(x, fs, ftype, wn, order=4, do_plot=1, ax_list=[]):
    """ Band Pass Filtering for 1D input data

    Parameters
    ----------
    x : list | numpy array
        input array to filter
    fs : int
        Sampling Frequency (Hz)
    ftype : str
        Filter type - 'butter', 'elliptic' or 'fir'
    wn : list | numpy array
        Normalized cut off frequency
    order : int
        Filter order
    do_plot : int | bool
        If 1, plot the frequency response
    ax_list : list | numpy array
        List of axis for plotting

    Returns
    -------
    y : numpy array
        Output filtered

    (b, a) : tuple
        Filter coefficients

    """

    ax_list = np.array(ax_list)
    wn = np.array(wn).squeeze()
    if ftype == 'butter':
        # z, p, k = signal.butter(order, wn, btype='bandpass', output='zpk')
        b, a = signal.butter(order, wn, btype='bandpass')
        y = signal.filtfilt(b, a, x)
    elif ftype == 'elliptic':
        # z, p, k = signal.ellip(order, rp=0.3, rs=40, Wn=wn, btype='bandpass', output='zpk')
        b, a = signal.ellip(order, rp=0.3, rs=40, Wn=wn, btype='bandpass')
        y = signal.filtfilt(b, a, x)
    elif ftype == 'fir':
        b = signal.firwin(order, wn, pass_zero=False)
        a = 1
        # z, p, k = [], [], []
        y = signal.lfilter(b, a, x)
    else:
        raise ValueError('Wrong Argument ftype')
    if do_plot:
        if not ax_list.size == 2:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax2 = ax.twinx()
            lw, alphaval = 2, 1
        else:
            lw, alphaval = 1, 0.25
            ax, ax2 = ax_list[0], ax_list[1]
        w, h = signal.freqz(b, a)
        freq = fs/(2*np.pi) * w
        ax.plot(freq, 20 * np.log10(abs(h)), 'b', lw=lw, alpha=alphaval)
        ax.set(ylabel='Gain (dB)', xlabel='Freq (rad/sample)', title='Filter Frequency Response')
        angles = np.unwrap(np.angle(h))
        ax2.plot(freq, angles, 'g')
        ax2.set(ylabel='Angle (radians)')
        ax2.grid(False)
        plt.axis('tight')
        plt.show()
    return y, (b, a)


def compute_robust_estimation(x_raw, fs, fmin, fmax, f_tolerance, noise_tolerance, n_monte_carlo=20, do_plot=0,
                              superpose=1, ftype='elliptic', forder=4, do_fplot=0, return_errors=0):
    """ Compute the robust estimation of the phase using the method described in [1]_.

    Parameters
    ----------
    x_raw : array
        Input raw signal
    fs : int
        Sampling Frequency (Hz)
    fmin : int
        Low cut-off frequency (Hz)
    fmax : int
        High cut-off frequency (Hz)
    f_tolerance : float
        Tolerance of the cut-off frequencies. A random number in the interval [-f_tolerance/2, +f_tolerance/2] will be
        added to the cut-off frequencies
    noise_tolerance : float
        Random noise from a uniform distribution in [-noise_tolerance/2, +noise_tolerance/2] will be added to the signal
        x_raw
    n_monte_carlo : int
        Number of monte carlo repetition
    do_plot : int | bool
        If True, plot the results
    superpose : int | bool
        If True, superpose all the phase estimates
    ftype : str
        Filter type:

        * 'butter' : Butterworth filter
        * 'elliptic' (default) : Elliptic filter
        * 'fir' : Finite Impulse Response filter

    forder : int
        Filter order - default : 4
    do_fplot : int | bool
        If 1, plot the frequency response
    return_errors : int | bool
        If 1, returns also the errors measures

    Returns
    -------
    ie : numpy array
        Instantaneous Enveloppe
    ip : numpy array
        Instantaneous Phase :math:`\phi(t)`
    ifreq : numpy array
        Instantaneous Frequency
    ip_wrap : numpy array
        Instantaneous Phase wrapped between [-pi, pi]
    var_ratio : numpy array
        Ratio of the variation of the phase with the variation of the enveloppe - returned if ``return_errors=True``
    error_sig : numpy array
        Error signal defined as :math:`H[cos(\phi(t))] - sin(\phi(t))` - returned if ``return_errors=True``

    References
    ----------
    .. [1] Esmaeil Seraj and Reza Sameni. Robust electroencephalogram phase estimation with applications in brain-computer
       interface systems. 9 February 2017.

    """
    ie, ip, ip_wrap = np.zeros((3, n_monte_carlo, len(x_raw)))
    ifreq = np.zeros((n_monte_carlo, len(x_raw)))
    x_filt = np.zeros((n_monte_carlo, len(x_raw)))
    x_raw_noisy = np.zeros((n_monte_carlo, len(x_raw)))
    if return_errors:
        var_ratio, error_sig = np.zeros((2, n_monte_carlo, len(x_raw)))
    fax_list = []
    if do_fplot:
        f = plt.figure()
        fax1 = f.add_subplot(111)
        fax2 = fax1.twinx()
        fax_list = [fax1, fax2]
    for i in range(0, n_monte_carlo):
        fmin_i = fmin - f_tolerance/2 + f_tolerance * np.random.rand(1)
        fmax_i = fmax - f_tolerance/2 + f_tolerance * np.random.rand(1)
        x_raw_noisy[i, :] = x_raw - noise_tolerance/2 + noise_tolerance * np.random.rand(len(x_raw))
        x_filt[i, :], _ = bp_filter_1d(x_raw_noisy[i, :], fs, ftype=ftype, wn=2/fs*np.array([fmin_i, fmax_i]),
                                       order=forder, do_plot=do_fplot, ax_list=fax_list)
        if return_errors:
            ie[i, :], ip[i, :], ifreq[i, :], ip_wrap[i, :], var_ratio[i, :], error_sig[i, :] = \
                compute_analytical_signal(x_filt[i, :], fs, return_errors=1)
        else:
            ie[i, :], ip[i, :], ifreq[i, :], ip_wrap[i, :] = compute_analytical_signal(x_filt[i, :], fs)
    ie_mean, ip_mean, ifreq_mean = np.mean(ie, 0), np.mean(ip, 0), np.mean(ifreq, 0)
    # plot results
    if do_plot:
        alphaval = 0.5 if n_monte_carlo > 1 else 1
        colpal = sns.color_palette()
        t = np.linspace(0, len(x_raw)/fs, len(x_raw))
        f = plt.figure()
        ax = plt.subplot2grid((4, 1), (0, 0), rowspan=1, colspan=1)
        ax.plot(t, x_raw_noisy.T, alpha=alphaval) if superpose else ax.plot(t, x_raw_noisy.mean(0))
        ax.set(ylabel='Amplitude (uV)', title='Raw Signal + noise - N Monte Carlo = {}'.format(n_monte_carlo))
        ax2 = plt.subplot2grid((4, 1), (1, 0), rowspan=1, colspan=1, sharex=ax)
        ax2.plot([0, t[-1]], [0, 0], color='k', zorder=0, alpha=alphaval)
        lg1 = ax2.plot(t, x_filt.T, lw=1, color=colpal[0], alpha=alphaval) if superpose else \
            ax2.plot(t, x_filt.mean(axis=0), color=colpal[0])
        lg2 = ax2.plot(t, ie.T, lw=1, color=colpal[1], alpha=alphaval) if superpose else \
            ax2.plot(t, ie_mean, ls='-', color=colpal[1])
        plt.legend([lg1[0]] + [lg2[0]], ['Filtered signal', 'Inst. Amplitude'], loc='upper left', frameon=True,
                   framealpha=0.5)
        ax2.set(ylabel='Amplitude (uV)', title='Filtering [{}-{}]Hz - f_tolerance = {} Hz - noise_tolerance = {} uV'.
                format(fmin, fmax, f_tolerance, noise_tolerance))
        ax3 = plt.subplot2grid((4, 1), (2, 0), rowspan=1, colspan=1, sharex=ax)
        if superpose:
            ax3.plot(t, ip_wrap.T, lw=1, color=colpal[1], alpha=alphaval)
        else:
            ax3.plot(t, ip_wrap.mean(0), color=colpal[1])
            ax3.fill_between(t, ip_wrap.mean(0) - ip_wrap.std(0), ip_wrap.mean(0) + ip_wrap.std(0), color=colpal[1],
                             alpha=alphaval)
            ax3.legend(['Mean', 'Std'], loc='upper left', frameon=True, framealpha=0.5)
        ax3.set(ylabel='Instantaneous Phase')
        ax4 = plt.subplot2grid((4, 1), (3, 0), rowspan=1, colspan=1, sharex=ax)
        if superpose:
            ax4.plot(t, ifreq.T, lw=1, color=colpal[0], alpha=alphaval)
        else:
            ax4.plot(t, ifreq_mean, color=colpal[0])
            ax4.fill_between(t[1:], ifreq_mean - ifreq.std(0), ifreq_mean + ifreq.std(0), color=colpal[0], alpha=alphaval)
            ax4.legend(['Mean', 'Std'], loc='upper left', frameon=True, framealpha=0.5)
        ax4.set(xlabel='Time (s)', ylabel='Instantaneous Frequency')
        ax.autoscale(axis='x', tight=True)
    if return_errors:
        return ie, ip, ifreq, ip_wrap, var_ratio, error_sig
    else:
        return ie, ip, ifreq, ip_wrap


def compute_analytical_signal(x_filtered, fs, return_errors=[]):
    """ Compute the analytical signal of the band-pass filter x_filtered and return the analytical amplitude, phase
    (wrap and unwrap) and frequency.

    Parameters
    ----------
    x_filtered : numpy array
        Filtered signal (should be narrow-band) - Real signal from which the analytical signal is calculated
    fs : int
        Sampling frequency (Hz)
    return_errors : int | bool | None
        If 1, return the error measures

    Returns
    -------
    ie : numpy array
        Instantaneous Enveloppe
    ip : numpy array
        Instantaneous Phase :math:`\phi(t)`
    ifreq : numpy array
        Instantaneous Frequency
    ip_wrap : numpy array
        Instantaneous Phase wrapped between [-pi, pi]
    var_ratio : numpy array
        Ratio of the variation of the phase with the variation of the enveloppe - returned if ``return_errors=True``
    error_sig : numpy array
        Error signal defined as :math:`H[cos(\phi(t))] - sin(\phi(t))` - returned if ``return_errors=True``

    """
    x_analytic = signal.hilbert(x_filtered)
    ie = np.abs(x_analytic)
    ip_wrap = np.angle(x_analytic)
    ip = np.unwrap(ip_wrap)
    ifreq = (np.diff(ip) / (2.0*np.pi) * fs)
    ifreq = np.hstack([ifreq[0], ifreq])
    if return_errors:
        var_ratio = np.abs(np.diff(ip) * fs) / (np.abs(np.diff(ie) * fs / ie[1:]) +1e-7)
        var_ratio = np.hstack([0, var_ratio])
        error_sig = np.imag(signal.hilbert(np.cos(ip))) - np.sin(ip)
        return ie, ip, ifreq, ip_wrap, var_ratio, error_sig
    else:
        return ie, ip, ifreq, ip_wrap


def itpc(x_trials, fs, filt_cf, filt_bw, f_tolerance=[], noise_tolerance=[], n_monte_carlo=20, ftype='elliptic',
        forder=4, do_plot=0, contour_plot=1, n_contours=10):
    """ Compute and plot the Inter-Trial Phase Clustering

    Parameters
    ----------
    x_trials : numpy array - shape (n_pnts, n_trials)
        2D numpy array containing the trials for one channel
    fs : int
        Sampling frequency (Hz)
    filt_cf : numpy array
        Filters center frequencies
    filt_bw : numpy array | int
        Filters bandwidth
    f_tolerance : float | array | None (default: none)
        Tolerance of the cut-off frequencies. A random number in the interval [-f_tolerance/2, +f_tolerance/2] will be
        added to the cut-off frequencies. If none, f_tolerance is set to ``filt_bw / 100`` for each filter.
    noise_tolerance : float | None (default: none)
        Random noise from a uniform distribution in [-noise_tolerance/2, +noise_tolerance/2] will be added to the signal
        x_raw. If none, noise_tolerance is set to ``np.std(x_trials) / 30``.
    n_monte_carlo : int
        Number of monte carlo repetition
    ftype : str
        Filter type:

        * 'butter' : Butterworth filter
        * 'elliptic' (default) : Elliptic filter
        * 'fir' : Finite Impulse Response filter

    forder : int
        Filter order - default : 4
    do_plot : int | bool
        If 1, plot the ITPC (default: 0)
    contour_plot : int | bool
        If 1 , plot the contours (contourf function) else use the pcolormesh function (default: 1)
    n_contours : int
        If ``contour_plot=1``, number of levels in contourf function

    Returns
    -------
    itpc_mat : array [n_freqs*n_pnts]
        ITPC result matrix
    ip_var_mean : array [n_freqs*n_pnts]
        Mean of the variance of the instantaneous phase across the Monte Carlo repetitions
    ie_mat_mean : array [n_freqs*n_pnts]
        Mean of the instantaneous enveloppe across the Monte Carlo repetitions
    """
    filt_cf, filt_bw = np.array(filt_cf), np.array(filt_bw)
    f_tolerance, noise_tolerance = np.array(f_tolerance), np.array(noise_tolerance)
    n_freqs = filt_cf.size
    if filt_bw.size == 1:
        filt_bw = filt_bw * np.ones(n_freqs)
    elif not filt_bw.size == n_freqs:
        print(filt_bw.size)
        raise ValueError('filt_bw argument should be a scalar or an array of the same length than filt_cf')
    if f_tolerance.size == 0:
        f_tolerance = 0.01 * filt_bw
    elif f_tolerance.size == 1:
        f_tolerance = f_tolerance * np.ones(f_tolerance)
    elif not f_tolerance.size == n_freqs:
        raise ValueError('f_tolerance argument should be a scalar or an array of the same length than filt_cf')
    if noise_tolerance.size == 0:
        noise_tolerance = np.std(x_trials) / 30
    elif not noise_tolerance.size == 1:
        raise ValueError('noise_tolerance argument should be scalar')
    (n_pnts, n_trials) = x_trials.shape
    ip_wrap_mat, ie_mat, ip_var = np.zeros((3, n_freqs, n_pnts, n_trials))
    for i_trial in tqdm.tqdm(range(0, n_trials)):
        for j_freq in range(0, n_freqs):
            ie, ip_wrap, _, _, = compute_robust_estimation(x_trials[:, i_trial], fs,
                                                           fmin=filt_cf[j_freq] - filt_bw[j_freq] / 2,
                                                           fmax=filt_cf[j_freq] + filt_bw[j_freq] / 2,
                                                           f_tolerance=f_tolerance[j_freq], noise_tolerance=noise_tolerance,
                                                           n_monte_carlo=n_monte_carlo, do_plot=0, ftype=ftype,
                                                           forder=forder, return_errors=0)
            ie_mat[j_freq, :, i_trial] = ie.mean(0)
            ip_wrap_mat[j_freq, :, i_trial] = ip_wrap.mean(0)
            ip_var[j_freq, :, i_trial] = 1 - np.abs(np.mean(np.exp(1j*ip_wrap), axis=0))
    ie_mat_mean = np.mean(ie_mat, axis=2)
    analytic_mat = 1 * np.exp(ip_wrap_mat*1j)
    itpc_mat = np.abs(np.mean(analytic_mat, axis=2))
    ip_var_mean = np.mean(ip_var, axis=2)
    # Plot
    if do_plot:
        if n_monte_carlo == 1:
            n_subplots, plot_var_phase = 2, 0
        else:
            n_subplots, plot_var_phase = 3, 1
        t = np.linspace(0, n_pnts/fs, n_pnts)
        f = plt.figure()
        ax = f.add_subplot(1, n_subplots, 1)
        im = plt.contourf(t, filt_cf, itpc_mat, n_contours) if contour_plot else \
            plt.pcolormesh(t, filt_cf, itpc_mat)
        ax.set(title='Inter Trial Phase Clustering', xlabel='Time (s)', ylabel='Frequency (Hz)')
        plt.colorbar(im)
        if plot_var_phase:
            ax2 = f.add_subplot(132, sharex=ax, sharey=ax)
            im2 = plt.contourf(t, filt_cf, ip_var_mean, 2*n_contours) if contour_plot else \
                plt.pcolormesh(t, filt_cf, ip_var_mean)
            plt.colorbar(im2)
            ax2.set(title='Phase Angle Variation', xlabel='Time (s)', ylabel='Frequency (Hz)')
        ax3 = f.add_subplot(1, n_subplots, 2+plot_var_phase, sharex=ax, sharey=ax)
        im3 = plt.contourf(t, filt_cf, ie_mat_mean, 2*n_contours) if contour_plot else \
            plt.pcolormesh(t, filt_cf, ie_mat_mean)
        plt.colorbar(im3, ax=ax3)
        ax3.set(title='Mean Instantaneous Amplitude', xlabel='Time (s)', ylabel='Frequency (Hz)')
        plt.tight_layout(pad=1, w_pad=0)
    return itpc_mat, ip_var_mean, ie_mat_mean


def plot_complex_trajectory(x, ax=[]):
    """ Plot the complex trajectory of real input signal x. This may help visualize the narrow-band
    behaviour of the signal.

    Parameters
    ----------
    x : array
        input real signal
    ax : list | none
        axis list

    """
    x_analytic = signal.hilbert(x)
    x_real = np.real(x_analytic)
    x_imag = np.imag(x_analytic)
    if not ax:
        f = plt.figure()
        ax = f.add_subplot(111)
    ax.plot(x_real, x_imag)
    ax.scatter(x_real, x_imag, marker='x', color='g')
    ax.set(xlabel='$x(t)$', ylabel='$H[x(t)]$', title='Complex Trajectory of signal x(t)')


def plot_analytical_signal(x, fs):
    """ Compute the analytical signal from the signal x and plot the instantaneous enveloppe, phase
    and frequency

    Parameters
    ----------
    x : array
        input signal (should be narrow-band)
    fs : int
        sampling frequency (Hz)

    """
    t = np.linspace(0, len(x) / fs, len(x))
    x_analytic = signal.hilbert(x)
    x_imag = np.imag(x_analytic)
    ie, ip = np.abs(x_analytic), np.angle(x_analytic)
    ifreq = np.diff(np.unwrap(ip)) / (2.0 * np.pi) * fs
    col_palette = sns.color_palette()
    # Plot
    f = plt.figure()
    ax1 = f.add_subplot(211)
    ax1.plot(t, x, color=col_palette[0])
    ax1.plot(t, np.imag(x_analytic), ls=':', lw=1, color=col_palette[0])
    ax1.plot(t, ie, color=col_palette[1])
    ax1.legend(['Input Signal', 'Hilbert Transform', 'Instantaneous Amplitude'], loc='upper left', frameon=True,
               framealpha=0.5)
    ax1.set(title='Analytical Signal', ylabel='Amplitude')
    ax2 = f.add_subplot(212, sharex=ax1)
    lg1 = ax2.plot(t, ip)
    ax2.set(xlabel='time (s)', ylabel='Phase Angle (rad)')
    ax3 = ax2.twinx()
    ax3.grid(False)
    lg2 = ax3.plot(t[1:], ifreq, color=col_palette[1])
    ax3.set(ylabel='Instantaneous Frequency (Hz)')
    plt.legend(lg1+lg2, ['Instantaneous Phase', 'Instantaneous Frequency'], loc='upper left', frameon=True,
               framealpha=0.5)
    ax3.autoscale(axis='x', tight=True)

