import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import pywt
import scipy.signal as signal
import scipy.interpolate as interpolate
from mne.time_frequency import tfr_array_morlet
import tqdm
from operator import add
import warnings
import phase_utils
import utils_sigprocessing


class FeatureExtracter:
    """
    Feature Extracter Class. Each Time Features object possess one instance. Used to extracted feature from the original
    data (amplitude usually). For frequency-based features, frequency bands of interest are defined (freq_bands) and
    the mean of the feature on these bands is computed.

    Attributes
    ----------
    data_ori : array
        Represents the amplitude data. Features are computed from this. Size : [n_chan, n_pnts, n_trials]
    srate : float
        Sampling rate (Hz)
    n_chan : int
        Number of channels in the data
    n_pnts : int
        Number of time points
    n_trials : int
        Number of trials
    tmin : float (default 0)
        Starting time (s)
    tmax : float | None (default: None)
        Ending time (s). If none, is computed from the number of points and the sampling rate.
    channel_names : array
        Name of each channel
    freq_bands : array
        Array of shape (n_freq_bands * 2) which defined the frequency bands of interest. Mean value of the
        frequency-based features are computed over these bands.
        Default value is : [[2, 4], [4, 8], [8, 12], [15, 30], [30, 80]]
    freq_band_names : array
        Name of the frequency bands. Default value is ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']

    """
    def __init__(self, data_ori, srate, n_chan, n_pnts, n_trials, channel_names, tmin=0, tmax=[],
                 freq_bands=[[2, 4], [4, 8], [8, 12], [15, 30], [30, 80]],
                 freq_band_names=['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']):
        data_ori = np.array(data_ori)
        self.srate = srate
        self.n_chan = n_chan
        self.n_pnts = n_pnts
        self.n_trials = n_trials
        self.tmin = tmin
        self.tmax = tmax if tmax else n_pnts / srate
        self.channel_names = np.atleast_1d(channel_names)
        self.freq_bands = np.array(freq_bands)
        self.freq_band_names = np.array(freq_band_names)
        if data_ori.ndim == 1:
            if not len(data_ori) == n_pnts:
                raise ValueError('1D data_ori argument must have a length of n_pnts')
            if n_chan != 1 or n_trials != 1:
                raise ValueError('Cannot find more than 1 channel and trial')
            self.data_ori = np.array([[data_ori]]).swapaxes(1, 2)
        elif data_ori.ndim == 2:
            if not data_ori.shape == (n_chan, n_pnts):
                raise ValueError('2D data_ori argument must have a shape [n_chan, n_pnts]')
            if n_trials != 1:
                raise ValueError('Cannot find more than trial')
            self.data_ori = np.array([data_ori]).T.swapaxes(0, 1)
        elif data_ori.ndim == 3:
            if not data_ori.shape == (n_chan, n_pnts, n_trials):
                raise ValueError('3D data_ori argument must have a shape [n_chan, n_pnts, n_trials]')
            self.data_ori = data_ori
        else:
            raise ValueError('Argument data_ori must be a 3D matrix or less')

    def bandpower_on_data(self, filt_type='butterworth', filt_order=3, scale_type=[], base_tstart=[], base_tend=[]):
        """

        Parameters
        ----------
        filt_type :
        filt_order :
        scale_type :
        base_tstart :
        base_tend :

        Returns
        -------

        """
        n_fbands = len(self.freq_band_names)
        n_features = n_fbands*self.n_chan
        feature_data = np.zeros((n_features, self.n_pnts, self.n_trials))
        for i_chan in tqdm.tqdm(range(0, self.n_chan)):
            feat_pos_i = np.arange(i_chan*n_fbands, (i_chan+1)*n_fbands)
            for i_trial in range(0, self.n_trials):
                # Size bandpower_i : [n_fbands*n_pnts]
                bandpower_i = bandpower_1d(self.data_ori[i_chan, :, i_trial], self.srate, self.freq_bands, filt_type,
                                           filt_order, scale_type=scale_type, base_tstart=base_tstart,
                                           base_tend=base_tend)
                feature_data[feat_pos_i, :, i_trial] = 10*np.log10(bandpower_i)
                feature_channame = self.channel_names.repeat(n_fbands)
        feature_type = np.tile(['BandPower {}'.format(fband_i) for fband_i in self.freq_band_names] , self.n_chan)
        feature_name = ['{} - {}'.format(feat_chan_i, feat_type_i) for (feat_chan_i, feat_type_i) in zip(feature_channame, feature_type)]
        return feature_data, feature_name, feature_type, feature_channame

    def cwt_on_data(self, wav_name=[], pfreqs=[], scale_type=[], base_tstart=[], base_tend=[]):
        """ Apply the Continous Wavelet Transform to the data ``data_ori``, return both power and phase average over the
        frequency bands. Uses the `PyWavelets module <https://pywavelets.readthedocs.io/en/latest/>`_

        Parameters
        ----------
        wav_name : str
            Wavelet name
        pfreqs : array
            Pseudo frequency array to use for computing power and phase
        scale_type : str
            Scaling type. Can be :

            * 'db_ratio' : :math:`P_{norm} = 10 \\cdot log10(\\frac{P}{mean(P_{baseline})})`
            * 'percent_change' : :math:`P_{norm} = 100 \\cdot \\frac{P - mean(P_{baseline})}{mean(P_{baseline})}`
            * 'z_transform' : :math:`P_{norm} = \\frac{P - mean(P_{baseline})}{std(P_{baseline})}`
        base_tstart : int
            Baseline starting point (in sample)
        base_tend : int
            Baseline ending point (in sample)

        Returns
        -------
        """
        n_fbands = len(self.freq_band_names)
        n_features = n_fbands*self.n_chan
        # cwt_bandpower, cwt_bandphase = np.zeros((2, self.n_chan, self.n_pnts, self.n_trials, n_fbands))
        cwt_bandpower, cwt_bandphase = np.zeros((2, n_features, self.n_pnts, self.n_trials))
        for i_chan in tqdm.tqdm(range(0, self.n_chan)):
            feat_pos_i = np.arange(i_chan*n_fbands, (i_chan+1)*n_fbands)
            for i_trial in range(0, self.n_trials):
                _, cwt_bandpower_i, _, cwt_bandphase_i = cwt_1d(self.data_ori[i_chan, :, i_trial], self.srate,
                                                                self.freq_bands, scale_type=scale_type,
                                                                base_tstart=base_tstart, base_tend=base_tend)
                cwt_bandpower[feat_pos_i, :, i_trial] = 10*np.log10(cwt_bandpower_i)
                cwt_bandphase[feat_pos_i, :, i_trial] = cwt_bandphase_i
        feature_channame = self.channel_names.repeat(n_fbands)
        feature_type_power = np.tile(['CWT BandPower {}'.format(fband_i) for fband_i in self.freq_band_names], self.n_chan)
        feature_type_phase = np.tile(['CWT Phase {}'.format(fband_i) for fband_i in self.freq_band_names], self.n_chan)
        feature_name_power = ['{} - {}'.format(feat_chan_i, feat_type_i) for (feat_chan_i, feat_type_i) in
                              zip(feature_channame, feature_type_power)]
        feature_name_phase = ['{} - {}'.format(feat_chan_i, feat_type_i) for (feat_chan_i, feat_type_i) in
                              zip(feature_channame, feature_type_phase)]

        return [cwt_bandpower, cwt_bandphase], [feature_name_power, feature_name_phase], \
               [feature_type_power, feature_type_phase], [feature_channame, feature_channame]

    def dwt_on_data(self, wav_name, scale_type=[], base_tstart=[], base_tend=[]):
        """ Apply the Discrete Wavelet Transform on the data

        Parameters
        ----------
        wav_name
            scale_type : str
        scale_type : str
            Scaling type. Can be :

            * 'db_ratio' : :math:`P_{norm} = 10 \\cdot log10(\\frac{P}{mean(P_{baseline})})`
            * 'percent_change' : :math:`P_{norm} = 100 \\cdot \\frac{P - mean(P_{baseline})}{mean(P_{baseline})}`
            * 'z_transform' : :math:`P_{norm} = \\frac{P - mean(P_{baseline})}{std(P_{baseline})}`
        base_tstart : int
            Baseline starting point (in sample)
        base_tend : int
            Baseline ending point (in sample)

        """
        wav = pywt.Wavelet(wav_name)
        n_max_level = pywt.dwt_max_level(self.n_pnts, wav.dec_len)
        coeff_mat = np.zeros((self.n_chan, self.n_pnts, self.n_trials, n_max_level+1))
        for i_chan in tqdm.tqdm(range(0, self.n_chan)):
            for i_trial in range(0, self.n_trials):
                coeff_i, pseudo_freq_band = dwt_1d(self.data_ori[i_chan, :, i_trial], srate=self.srate, wav_name=wav_name,
                                                   scale_type=scale_type, base_tstart=base_tstart, base_tend=base_tend)
                coeff_mat[i_chan, :, i_trial, :] = coeff_i.T
        coeff_mat_reshaped = coeff_mat.swapaxes(0, 2).reshape((self.n_trials, self.n_pnts,
                                                               self.n_chan * (n_max_level+1))).T
        feature_chan_name = self.channel_names.repeat(n_max_level+1)
        feature_type = np.tile(np.array(pseudo_freq_band)[::-1], self.n_chan)
        feature_type = np.array(['DWT {}'.format(feat_type_i) for feat_type_i in feature_type])
        feature_name = np.array(list(map(add, list(feature_type), list(np.repeat(' ', len(feature_type))))))
        feature_name = np.array(list(map(add, list(feature_name), list(feature_chan_name))))
        return coeff_mat_reshaped, feature_name, feature_type, feature_chan_name

    def stft_on_data(self, win_name='hamming', win_dur=0.2, overlap=0.85, nfft=[], scale_type=[],
                     base_tstart=[], base_tend=[]):
        """

        Parameters
        ----------
        win_name : str (default: 'hamming')
            Window's name
        win_dur : float (default: 0.2)
            Window's duration (s)
        overlap : float
            Overlap - must be between 0 and 1 - Default: 0.85
        nfft : int | None (default: None)
            Number of frequencies used in the FFT
        scale_type : str
            Scaling type. Can be :

            * 'db_ratio' : :math:`P_{norm} = 10 \\cdot log10(\\frac{P}{mean(P_{baseline})})`
            * 'percent_change' : :math:`P_{norm} = 100 \\cdot \\frac{P - mean(P_{baseline})}{mean(P_{baseline})}`
            * 'z_transform' : :math:`P_{norm} = \\frac{P - mean(P_{baseline})}{std(P_{baseline})}`
        base_tstart : int
            Baseline starting point (in sample)
        base_tend : int
            Baseline ending point (in sample)

        Returns
        -------
        """
        n_fbands = len(self.freq_band_names)
        n_features = n_fbands*self.n_chan
        stft_bandpower, stft_phase = np.zeros((2, n_features, self.n_pnts, self.n_trials))
        for i_chan in tqdm.tqdm(range(0, self.n_chan)):
            feat_pos_i = np.arange(i_chan*n_fbands, (i_chan+1)*n_fbands)
            for i_trial in range(0, self.n_trials):
                _, stft_bandpower_i, stft_phase_i = stft_1d(self.data_ori[i_chan, :, i_trial], self.srate,
                                                            self.freq_bands, win_name, win_dur, overlap, nfft,
                                                            scale_type=scale_type, base_tstart=base_tstart,
                                                            base_tend=base_tend)
                stft_bandpower[feat_pos_i, :, i_trial] = 10*np.log10(stft_bandpower_i)
                stft_phase[feat_pos_i, :, i_trial] = stft_phase_i

        feature_channame = self.channel_names.repeat(n_fbands)
        feature_type_power = np.tile(['STFT BandPower {}'.format(fband_i) for fband_i in self.freq_band_names], self.n_chan)
        feature_type_phase = np.tile(['STFT Phase {}'.format(fband_i) for fband_i in self.freq_band_names], self.n_chan)
        feature_name_power = ['{} - {}'.format(feat_chan_i, feat_type_i) for (feat_chan_i, feat_type_i) in
                              zip(feature_channame, feature_type_power)]
        feature_name_phase = ['{} - {}'.format(feat_chan_i, feat_type_i) for (feat_chan_i, feat_type_i) in
                              zip(feature_channame, feature_type_phase)]

        return [stft_bandpower, stft_phase_i], [feature_name_power, feature_name_phase], \
               [feature_type_power, feature_type_phase], [feature_channame, feature_channame]

    def filter_hilbert_on_data(self, center_freq, bandwidth, ftype='elliptic', forder=4, f_tolerance=[],
                               noise_tolerance=[], n_monte_carlo=5):
        """ Estimate the phase of the data ``data_ori`` using band-pass filtering and the Hilbert transform. The cos
        and sin of the phase angle are returned, the phase angle is 2-pi periodic, thus cannot be used directly as a
        feature in a classification or regression task.
        see :func:`phase_utils.compute_robust_estimation`

        Parameters
        ----------
        center_freq : array
            Center frequency (in Hz) of the band-pass filters
        bandwidth : array
            Bandwidth (in Hz) of the band-pass filters
        ftype : str
            Filter type:

            * 'butter' : Butterworth filter
            * 'elliptic' (default) : Elliptic filter
            * 'fir' : Finite Impulse Response filter
        forder : int
            Filter order - default : 4
        f_tolerance : float
            Tolerance of the cut-off frequencies. A random number in the interval [-f_tolerance/2, +f_tolerance/2] will be
            added to the cut-off frequencies
        noise_tolerance : float
            Random noise from a uniform distribution in [-noise_tolerance/2, +noise_tolerance/2] will be added to the signal
            x_raw
        n_monte_carlo : int
            Number of monte carlo repetition

        """
        center_freq, bandwidth = np.array(center_freq), np.array(bandwidth)
        n_freqs, n_freq_bands = len(center_freq), len(self.freq_band_names)
        if bandwidth.size == 1:
            bandwidth = bandwidth * np.ones(n_freqs)
        elif not bandwidth.size == n_freqs:
            raise ValueError('filt_bw argument should be a scalar or an array of the same length than filt_cf')
        if not f_tolerance:
            f_tolerance = min(bandwidth) / 30
        if not noise_tolerance:
            noise_tolerance = 0.5
        ip_wrap_mat = np.zeros((self.n_chan, self.n_pnts, self.n_trials, n_freqs))
        for i_chan in tqdm.tqdm(range(0, self.n_chan)):
            for i_trial in range(0, self.n_trials):
                data_chan_trial = self.data_ori[i_chan, :, i_trial].squeeze()
                for j_freq in range(0, n_freqs):
                    _, _, _, ip_wrap_i, = phase_utils.compute_robust_estimation(
                        data_chan_trial, self.srate, fmin=center_freq[j_freq] - bandwidth[j_freq] / 2,
                        fmax=center_freq[j_freq] + bandwidth[j_freq] / 2, f_tolerance=f_tolerance,
                        noise_tolerance=noise_tolerance, n_monte_carlo=n_monte_carlo, ftype=ftype, forder=forder)
                    ip_wrap_mat[i_chan, :, i_trial, j_freq] = np.mean(ip_wrap_i, axis=0)
        # Mean over frequency bands
        n_freq_bands = self.freq_bands.shape[0]
        ip_wrap_mat_mean = np.zeros((self.n_chan, self.n_pnts, self.n_trials, n_freq_bands))
        print('Compute the mean over frequency bands')
        for i_chan in tqdm.tqdm(range(0, self.n_chan)):
            for i_trial in range(0, self.n_trials):
                ip_wrap_mat_mean[i_chan, :, i_trial, :] = compute_band_mean(ip_wrap_mat[i_chan, :, i_trial, :].T,
                                                                            center_freq, self.freq_bands).T
        # Shape phase data
        ip_reshaped = ip_wrap_mat_mean.swapaxes(0, 2).reshape((self.n_trials, self.n_pnts, self.n_chan*n_freq_bands)).T
        feature_chan_name = self.channel_names.repeat(n_freq_bands)
        feature_type = np.tile(np.array(self.freq_band_names), self.n_chan)
        feature_type_cos = np.array(['Phase Hilbert (cos) {}'.format(feat_type_i) for feat_type_i in feature_type])
        feature_type_sin = np.array(['Phase Hilbert (sin) {}'.format(feat_type_i) for feat_type_i in feature_type])
        feature_name_cos = np.array(list(map(add, list(feature_type_cos), list(np.repeat(' ', len(feature_type_cos))))))
        feature_name_cos = np.array(list(map(add, list(feature_name_cos), list(feature_chan_name))))
        feature_name_sin = np.array(list(map(add, list(feature_type_sin), list(np.repeat(' ', len(feature_type_sin))))))
        feature_name_sin = np.array(list(map(add, list(feature_name_sin), list(feature_chan_name))))
        return np.vstack([np.cos(ip_reshaped), np.sin(ip_reshaped)]), np.hstack([feature_name_cos, feature_name_sin]), \
               np.hstack([feature_type_cos, feature_type_sin]), np.hstack(([feature_chan_name, feature_chan_name]))


def bandpower_1d(x, srate, freq_bands, filt_type='butterworth', filt_order=3, scale_type=[], base_tstart=[],
                 base_tend=[]):
    """ Compute the power of input signal ``x`` in the different frequency bands defined by ``freq_bands``. Signal ``x``
    is first passed through a band-pass filter specified by ``filt_type`` and ``filt_order``, output amplitude is then
    squared to get the power and the envelope is computed using the Hilbert transform

    Outputs
    -------
    band_power : array (n_freq_bands*n_pnts)
        Band power for each frequency band
    """
    freq_bands, base_tstart, base_tend = np.atleast_2d(freq_bands), np.atleast_1d(base_tstart), np.atleast_1d(base_tend)
    n_fbands, n_pnts = freq_bands.shape[0], x.size
    amp_filt_squared, band_power = np.zeros((2, n_fbands, n_pnts))
    # Compute filter coefficients
    filt_coeffs = [get_bandpass_filter(srate, cutoff_freq, filt_type, filt_order) for cutoff_freq in freq_bands]
    # Filter input signal x
    for i, fband_i in enumerate(freq_bands):
        amp_filt_squared[i, :] = signal.filtfilt(filt_coeffs[i][0], filt_coeffs[i][1], x, padlen=100) ** 2
        band_power[i, :] = utils_sigprocessing.get_signal_envelope(amp_filt_squared[i, :],
                                                                   N=max(2, int(0.5*srate/freq_bands[i][1])),
                                                                   wn_lp=2/srate*freq_bands[i][0])
    # Baseline normalization if asked
    if scale_type and base_tstart.size == 1 and base_tend.size == 1:
        band_power = tf_scaling(band_power, int(srate*base_tstart), int(srate*base_tend), scale_type)

    return band_power


def cwt_1d(x, srate, freq_bands, freqs=np.logspace(np.log10(3), np.log10(90), 40),
           n_cycles=np.logspace(np.log10(1), np.log10(10), 40), scale_type=[], base_tstart=[], base_tend=[]):
    """ Compute the Continuous Wavelet Transform for a 1 dimension input array ``x``

    Parameters
    ----------
    x : array
        Input array - must have 1 dimension
    srate : int
        Sampling frequency
    freq_bands : array (size: n_freq_bands * 2)
        Frequency bands of interest. Mean value of both power and phase are calculated on these bands.
    freqs : array
        Pseudo frequency array to use for computing power and phase
    n_cycles : int | array
        Number of cycles for each wavelet
    scale_type : str
        Scaling type. Can be :

        * 'db_ratio' : :math:`P_{norm} = 10 \\cdot log10(\\frac{P}{mean(P_{baseline})})`
        * 'percent_change' : :math:`P_{norm} = 100 \\cdot \\frac{P - mean(P_{baseline})}{mean(P_{baseline})}`
        * 'z_transform' : :math:`P_{norm} = \\frac{P - mean(P_{baseline})}{std(P_{baseline})}`

    base_tstart : int
        Baseline starting point (in sample)
    base_tend : int
        Baseline ending point (in sample)

    Returns
    -------
    coeffs_power : array (size: n_freqs * n_pnts)
        Power extracted from the wavelet coefficients
    coeffs_power_bands : array (size: n_freq_bands * n_pnts)
        Mean of ``coeffs_power`` over the frequency bands defined in ``freq_bands``
    phase : array (size: n_freqs * n_pnts)
        Phase angle
    phase_bands : array (size: n_freq_bands * n_pnts)
        Mean of ``phase`` over the frequency bands defined in ``freq_bands``
    """
    freq_bands, base_tstart, base_tend = np.atleast_1d(freq_bands), np.atleast_1d(base_tstart), np.atleast_1d(base_tend)
    x_3d = np.array([[x.squeeze()]])
    cwt_complex = tfr_array_morlet(x_3d, srate, freqs, n_cycles, output='complex').squeeze()
    cwt_power, cwt_phase = np.abs(cwt_complex)**2, np.angle(cwt_complex)
    # Compute mean over frequency bands
    cwt_bandpower = compute_band_mean(cwt_power, freqs, freq_bands, interp_method='linear')
    cwt_bandphase = compute_band_mean(cwt_phase, freqs, freq_bands, interp_method='linear')
    # Normalization / Scaling
    if scale_type and base_tstart.size == 1 and base_tend.size == 1:
        cwt_power = tf_scaling(cwt_power, int(base_tstart * srate), int(base_tend * srate), scale_type)
        cwt_bandpower = tf_scaling(cwt_bandpower, int(base_tstart * srate), int(base_tend * srate), scale_type)
    return cwt_power, cwt_bandpower, cwt_phase, cwt_bandphase


def dwt_1d(x, srate, wav_name, do_plot=0, scale_type=[], base_tstart=[], base_tend=[]):
    """ Discrete wavelet transform for input 1D array.

    Parameters
    ----------
    x : array
        Input 1D array
    srate : float
         Sampling rate (Hz)
    wav_name : str
        Wavelet name
    do_plot : bool
        If True, plot the results
    scale_type : str
         Normalization type - can be 'db_ratio', 'percent_change', or 'z_transform'
    base_tstart : float
         Baseline starting time (s)
    base_tend : float
        Baseline ending time (s)

    Returns
    -------
    coeffs_upsampled, pfreq_bands

    """
    base_tstart, base_tend = np.atleast_1d(base_tstart), np.atleast_1d(base_tend)
    wav = pywt.Wavelet(wav_name)
    coeffs = pywt.wavedec(x, wav)
    n_bands = len(coeffs)
    n_pnts = len(x)
    coeffs_upsampled = np.zeros((n_bands, n_pnts))
    pfreq_bands = []
    # Upsample the wavelet coefficients
    for i in range(0, n_bands):
        coeffs_i = coeffs[i]
        y_ind = np.round((len(coeffs_i) - 1) / n_pnts * np.arange(0, n_pnts))
        coeffs_upsampled[i, :] = coeffs_i[y_ind.astype(int)]
        pfreq_bands.append('{:.0f}-{:.0f} Hz'.format(2**-(i+2)*srate, 2**-(i+1)*srate))
    # Normalization / Scaling
    if scale_type and base_tstart.size == 1 and base_tend.size == 1:
        coeffs_upsampled = tf_scaling(coeffs_upsampled, int(base_tstart * srate), int(base_tend * srate), scale_type)
    return coeffs_upsampled, pfreq_bands


def stft_1d(x, srate, freq_bands, win_name='hamming', win_dur=[], overlap=0.9, nfft=256,
            fmin=[], scale_type=[], base_tstart=[], base_tend=[]):
    """ Compute the short-time Fourier transform

    Parameters
    ----------
    x : array
        Input vector
    srate : float
        Sampling frequency (Hz)
    freq_bands : array (size: n_freq_bands * 2)
        Frequency bands of interest. Mean value of both power and phase are calculated on these bands.
    win_name :  str (default: 'hamming')
        Window's name
    win_dur : float (default: [])
        Window's duration (s)
   overlap : float
        Overlap - must be between 0 and 1 - Default: 0.9
    nfft : int | None (default: 256)
        Number of frequencies used in the FFT
    fmin : float
        Starting frequency
    scale_type : str
        Normalization type - can be 'db_ratio', 'percent_change', or 'z_transform'
    base_tstart : float
        Baseline starting time (s)
    base_tend : float
        Baseline ending time (s)

    Returns
    -------
    pxx, pxx_bands_interp, phase_bands_interp
    """
    freq_bands, base_tstart, base_tend = np.atleast_1d(freq_bands), np.atleast_1d(base_tstart), np.atleast_1d(base_tend)
    if not win_dur:
        win_dur = 1 / freq_bands.min()
    win_len = int(win_dur * srate)
    if not nfft:
        nfft = win_len
    if not fmin:
        fmin = freq_bands[0, 0]
    if overlap < 0 or overlap > 1:
        raise ValueError('overlap argment should be between 0 and 1')
    n_overlap = int(win_len * overlap)
    fft_win = signal.get_window(win_name, win_len)
    # Short Time Fourier Transform
    freqs, t_fft, zxx = signal.stft(x, fs=srate, window=fft_win, nperseg=win_len, noverlap=n_overlap, nfft=nfft)
    phase = np.angle(zxx)
    freq_sel_ind = freqs > fmin
    freqs, zxx = freqs[freq_sel_ind], zxx[freq_sel_ind, :]
    pxx = abs(zxx)**2
    n_freqs, n_pnts, n_fbands = len(freqs), len(x), freq_bands.shape[0]
    # Interpolate in time the psd pxx and the phase
    pxx_interp, phase_interp = np.zeros((2, n_freqs, n_pnts))
    for i in range(n_freqs):
        pxx_interp[i, :] = interpolate.interp1d(t_fft, pxx[i, :], kind='linear')(np.linspace(0, n_pnts/srate, n_pnts))
        phase_interp[i, :] = interpolate.interp1d(t_fft, phase[i, :], kind='linear')(np.linspace(0, n_pnts/srate, n_pnts))
    # Compute mean over frequency bands
    pxx_bands_interp = compute_band_mean(pxx_interp, freqs, freq_bands, interp_method='linear')
    phase_bands_interp = compute_band_mean(phase_interp, freqs, freq_bands, interp_method='linear')
    # Normalization / Scaling
    if scale_type and base_tstart.size == 1 and base_tend.size == 1:
        phase_bands_interp = tf_scaling(phase_bands_interp, int(base_tstart * srate), int(base_tend * srate), scale_type)

    return pxx, pxx_bands_interp, phase_bands_interp


def tf_scaling(tf_power_mat, base_ind_start, base_ind_end, scale_type):
    """ Scale / Normalize the time-frequency map given the baseline time.

    Parameters
    ----------
    tf_power_mat : array (size n_channels * n_pnts)
        Time frequency matrix
    base_ind_start : int
        Baseline starting point (in sample)
    base_ind_end : int
        Baseline ending point (in sample).
    scale_type : str
        Scaling type. Can be :

        * 'db_ratio' : :math:`P_{norm} = 10 \\cdot log10(\\frac{P}{mean(P_{baseline})})`
        * 'percent_change' : :math:`P_{norm} = 100 \\cdot \\frac{P - mean(P_{baseline})}{mean(P_{baseline})}`
        * 'z_transform' : :math:`P_{norm} = \\frac{P - mean(P_{baseline})}{std(P_{baseline})}`

    Returns
    -------
    tf_power_mat_norm : array
        Normalized time-frequency map

    """
    n_pnts = tf_power_mat.shape[1]
    baseline_ind = np.arange(base_ind_start, base_ind_end)
    baseline_mean = tf_power_mat[:, baseline_ind].mean(1)
    baseline_mean_mat = np.tile(baseline_mean, (n_pnts, 1)).T
    if scale_type.lower() == 'db_ratio':
        tf_power_mat_norm = 10*np.log10(tf_power_mat / baseline_mean_mat)
    elif scale_type.lower() == 'percent_change':
        tf_power_mat_norm = 100*(tf_power_mat - baseline_mean_mat) / baseline_mean_mat
    elif scale_type.lower() == 'z_transform':
        baseline_std = tf_power_mat[:, baseline_ind].std(1)
        baseline_std_mat = np.tile(baseline_std, (n_pnts, 1)).T
        tf_power_mat_norm = (tf_power_mat - baseline_mean_mat) / baseline_std_mat
    else:
        raise ValueError('Unkown value for scale_type : {}'.format(scale_type))
    return tf_power_mat_norm


def compute_band_mean(data, freqs, freq_bands, interp_method='linear', freqs_interp=[]):
    """ Compute the mean of ``data`` over the frequency bands defined by ``freq_bands``. The original ``data`` array
    of shape [n_freqs * n_pnts] is first interpolated along the frequency axis before computing the mean.
    The new frequency values can be specified by the ``freqs_interp`` parameter. By default the ``freqs_interp``
    parameter add 3 equally spaced points between each original frequency value.

    Parameters
    ----------
    data : array (size: n_freqs * n_pnts)
        Time-frequency data
    freqs : array (size: n_freqs)
        Frequencies in ``data`` (in Hz)
    freq_bands : array (size: n_freq_bands * 2)
        Frequency bands of interest. Mean value are calculated on these bands.
    interp_method : str (default: 'linear')
        Specifies the kind of interpolation as a string ('linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic')
        See :func: `scipy.interpolated.interp1d`
    freqs_interp : array | None (default: none)
        Frequencies (in Hz) used for interpolation. If none, 3 equally spaced frequency points are added between each
        original frequency value.

    Returns
    -------
    data_band_mean : array (size: n_freq_bands * n_pnts)
        Mean of the data over the frequency bands

    """
    data, freq_bands, freqs = np.array(data), np.array(freq_bands), np.array(freqs)
    if not data.ndim == 2:
        raise ValueError('data argument should be 2D: [n_freqs * n_pnts]')
    if freq_bands.min() < freqs.min():
        warnings.warn('freq_bands minimum value is outside freqs range')
    if freq_bands.max() > freqs.max():
        warnings.warn('freq_bands maximum value is outside freqs range')
    (n_freqs, n_pnts), n_fbands = data.shape, freq_bands.shape[0]
    data_band_mean = np.zeros((n_fbands, n_pnts))
    # Interpolation
    if interp_method:
        if not freqs_interp:
            freqs_interp = np.sort(np.hstack([freqs, freqs[:-1] + 0.25*np.diff(freqs), freqs[:-1] +
                                              0.5*np.diff(freqs), freqs[:-1] + 0.75*np.diff(freqs)]))
        data_interp = np.zeros((len(freqs_interp), n_pnts))
        for t in np.arange(n_pnts):
            f_interp = interpolate.interp1d(freqs, data[:, t], kind=interp_method)
            data_interp[:, t] = f_interp(freqs_interp)
    else:
        data_interp, freqs_interp = data, freqs
    for i, fband_i in enumerate(freq_bands):
        find = (freqs_interp >= fband_i[0]) & (freqs_interp <= fband_i[1])
        if not find.any():
            continue
        data_band_mean[i, :] = np.mean(data_interp[find, :], axis=0)
    return data_band_mean


def zero_tf_edges(tf_mat, fs, freqs, n_period=2):
    if tf_mat.ndim > 2 or not tf_mat.shape[0] == len(freqs):
        raise ValueError('Argument tf_mat should have a shape [n_freqs, n_pnts]')
    for i, f in enumerate(freqs):
        period_i_pnts = int(n_period * np.ceil(fs/f))
        tf_mat[i, 0:period_i_pnts] = 0
        tf_mat[i, -period_i_pnts:] = 0
    return tf_mat


def get_bandpass_filter(srate, cutoff_freqs, filt_type='butterworth', filt_order=[]):
    if filt_type.lower() == 'fir':
        fir_order = 500 if not filt_order else filt_order
        bandwidth = float(np.diff(cutoff_freqs))
        b_filt = signal.firwin2(fir_order, gain=[0, 0, 1, 1, 0, 0],
                                freq=2/srate*np.array([0, cutoff_freqs[0]-bandwidth/10, cutoff_freqs[0],
                                                       cutoff_freqs[1], cutoff_freqs[1]+bandwidth/10, srate/2]))
        a_filt = 1
    elif filt_type.lower() in ['iir', 'butter', 'butterworth']:
        iir_order = 4 if not filt_order else filt_order
        b_filt, a_filt = signal.iirfilter(iir_order, Wn=2/srate*cutoff_freqs, btype='bandpass')
    return b_filt, a_filt
