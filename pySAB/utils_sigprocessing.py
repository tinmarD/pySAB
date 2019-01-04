import numpy as np
from scipy import signal


def movingaverage1d(x, win_length):
    x = np.array(x)
    n_pnts = x.size
    if not x.ndim == 1:
        raise ValueError('Argument x should be a 1D array')
    if not np.isscalar(win_length):
        raise ValueError('Argument win_length must be a scalar')
    if win_length > n_pnts:
        raise ValueError('Smoothing window length must be inferior to vector length')

    if win_length % 2 is 0:
        print('Length of smoothing window should be odd, increase the length of one')
        win_length += 1

    n_half_conv = int(np.floor(win_length/2))
    y_smooth = np.zeros(n_pnts)
    # y_smooth = np.convolve(x, np.ones(win_length)/win_length, 'same')
    y_smooth[n_half_conv:-n_half_conv] = np.convolve(x, np.ones(win_length)/win_length, 'valid')
    for i in range(n_half_conv):
        y_smooth[i] = np.mean(x[:i])
    for i in range(n_pnts-n_half_conv,n_pnts):
        y_smooth[i] = np.mean(x[i-n_half_conv:])

    return y_smooth


def get_signal_baseline(x, N, wn_lp=[], forder=3, remove_negative_values=0):
    """ Compute the baseline level of signal x. Peaks are removed if N is larger than the width of the peaks.
     Adapted from C. Yarne [2017] - Simple empirical algorithm to obtain signal envelope in three step.
     ( 1st step : Take the absolute value )
     2nd step : Divide signal x in k segments of N samples, then the MINimum value is taken from
     the signal in each segment.
     3rd step : Low pass filter the previous signal to get rid of the remaining staircase ripples

    Parameters
    ----------
    x : array
        Input signal
    N : integer
        Segment length
    wn_lp : float
        Normalized cutoff

    Returns
    -------

    """
    # 1st step
    # x_abs = np.abs(x)
    # 2nd step
    n_splits = int(x.size / N)
    x_split = np.array_split(x, n_splits)
    x_min = np.zeros(x.size)
    k = 0
    for x_split_i in x_split:
        x_min[k:k + x_split_i.size] = x_split_i.min()
        k += x_split_i.size
    x_min[0], x_min[-1] = x[0], x[-1]
    # 3rd step
    if wn_lp:
        [b, a] = signal.butter(forder, wn_lp, 'low')
        x_min_filt = signal.filtfilt(b, a, x_min, padlen=50)
    else:
        x_min_filt = x_min
    if remove_negative_values:
        x_min_filt_pos = x_min_filt[x_min_filt > 0]
        x_min_filt[x_min_filt < 0] = x_min_filt_pos.min()

    return x_min_filt


def get_signal_envelope(x, N, wn_lp, forder=3, remove_negative_values=1):
    """ Compute the envelope of signal x using the method described in :
     C. Yarne [2017] - Simple empirical algorithm to obtain signal envelope in three step.
     1st step : Take the absolute value
     2nd step : Divide signal x in k segments of N samples, then the maximum value is taken from 
     the signal in each segment.
     3rd step : Low pass filter the previous signal to get rid of the remaining staircase ripples
    
    Parameters
    ----------
    x : array
        Input signal
    N : integer
        Segment length
    wn_lp : float
        Normalized cutoff

    Returns
    -------

    """
    # 1st step
    x_abs = np.abs(x)
    # 2nd step
    n_splits = int(x.size / N)
    x_split = np.array_split(x_abs, n_splits)
    x_max = np.zeros(x.size)
    k = 0
    for x_split_i in x_split:
        x_max[k:k+x_split_i.size] = x_split_i.max()
        k += x_split_i.size
    # 3rd step
    [b, a] = signal.butter(forder, wn_lp, 'low')
    x_max_filt = signal.filtfilt(b, a, x_max, padlen=50)
    if remove_negative_values:
        x_max_filt_pos = x_max_filt[x_max_filt>0]
        x_max_filt[x_max_filt<0] = x_max_filt_pos.min()
    return x_max_filt

    # plt.figure()
    # plt.plot(x, 'k')
    # plt.plot(x_abs, 'k', ls='--')
    # plt.plot(x_max)
    # plt.plot(x_max_filt)

