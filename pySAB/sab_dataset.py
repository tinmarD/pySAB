import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.io as sio
import scipy.signal as signal
import os
import re
from datetime import datetime
from mne.time_frequency import psd_array_welch
import _pickle
import timefeatures
import phase_utils
sns.set()
sns.set_context('paper')


class ChannelInfo:
    """ Contains informations about channels

    Attributes
    ----------
    n_chan : int
        Number of channels
    chan_names : numpy array (str)
        Array containing the channels' name
    chan_names_no_eeg : numpy array (str)
        Channel names without the 'EEG' prefix
    chan_elec_names : numpy array (str)
        Name of the electrode for each channel
    elec_names : numpy array (str)
        Name of all different electrodes
    elec_eeg_names : numpy array (str)
        Name of all different EEG electrodes
    n_elec : int
        Number of electrodes
    # n_elec_eeg : int
        Number of EEG electrodes

    """

    def __init__(self, channel_names):
        self.chan_names = np.array(channel_names)
        self.n_chan = len(channel_names)
        self.eeg_chan_ind, self.n_eeg_chan = self.get_eeg_channels()
        self.chan_names_no_eeg = np.array([str.strip(re.sub('EEG', '', chan_name)) for chan_name in channel_names])
        self.chan_elec_names = self.get_electrode_names()
        _, idx = np.unique(self.chan_elec_names, return_index=True)
        self.elec_names = self.chan_elec_names[np.sort(idx)]
        _, idx = np.unique(self.chan_elec_names[self.eeg_chan_ind], return_index=True)
        self.elec_eeg_names = self.chan_elec_names[self.eeg_chan_ind][np.sort(idx)]
        self.n_elec, self.n_eeg_elec = len(self.elec_names), len(self.elec_eeg_names)

    def __str__(self):
        desc_str = 'Channel Info : {} channels and {} electrodes\n'.format(self.n_chan, self.n_elec)
        desc_str += '{} EEG channels - {} non-EEG channels\n'.format(self.n_eeg_chan, self.n_chan - self.n_eeg_chan)
        desc_str += '{} EEG electrodes - {} non-EEG electrodes'.format(self.n_eeg_elec, self.n_elec - self.n_eeg_elec)
        return desc_str

    def __getitem__(self, item):
        return ChannelInfo(self.chan_names[item])

    def get_channel_pos(self, chan_sel_name, true_match=True):
        """ Return the channel position from the channel's name

        Parameters
        ---------
        chan_sel_name : str
            Channel's name
        true_match : bool (default: True)
            If True, search an exact match between ``chan_sel_name`` parameter and ``chan_names`` attribute. If False
            returns, every channel position containing ``chan_sel_name``

        Return
        -------
        chan_sel_pos : int
            Channel's position
        """
        chan_sel_name = np.array(chan_sel_name)
        if chan_sel_name.size > 1:
            chan_sel_pos = []
            for chan_sel_name_i in chan_sel_name:
                chan_sel_pos_i = self.get_channel_pos(chan_sel_name_i)
                chan_sel_pos.append(chan_sel_pos_i)
        else:
            if true_match:
                if chan_sel_name in self.chan_names:
                    chan_sel_pos = int(np.where(self.chan_names == chan_sel_name)[0])
                elif chan_sel_name in self.chan_names_no_eeg:
                    chan_sel_pos = int(np.where(self.chan_names_no_eeg == chan_sel_name)[0])
                else:
                    print('No channel named {}'.format(chan_sel_name))
                    chan_sel_pos = []
            else:
                chan_sel_pos = np.where([str(chan_sel_name) in chan_i for chan_i in self.chan_names])[0]
        return np.array(chan_sel_pos)

    def get_channel_name(self, chan_sel_pos):
        """ Get the channel name from the channel's position

        Parameters
        ----------
        chan_sel_pos : int | array
            Channel's position

        Returns
        -------
        chan_names : str | list
            Channel's names

        """
        return self.chan_names[chan_sel_pos]

    def get_electrode_names(self):
        """ Used to get electrode names from the channel names

        Returns
        -------
        Electrode names : array
        """
        # chan_elec_names = [re.search('^\D+', chan_name_i)[0] for chan_name_i in self.chan_names_no_eeg]
        chan_elec_names = [re.search('^\D+', chan_name_i).group() for chan_name_i in self.chan_names_no_eeg]
        return np.array(chan_elec_names)

    def get_electrode_channels_pos(self, elec_desc):
        """ Get the channels position of the selected electrode

        Parameters
        ----------
        elec_desc : int | str | array
            Either the number of the electrode, or the name of the electrode to select

        Returns
        -------
        elec_chan_pos : array
            Position of the channels of the selected electrode

        """
        elec_desc = np.array(elec_desc)
        if elec_desc.size > 1:
            elec_chan_pos, elec_name = [], []
            for elec_desc_i in elec_desc:
                elec_chan_pos_i, elec_name_i = self.get_electrode_channels_pos(elec_desc_i)
                elec_chan_pos.append(elec_chan_pos_i)
                elec_name.append(elec_name_i)
            return np.hstack(elec_chan_pos), np.hstack(elec_name)
        else:
            if elec_desc.dtype.char == 'U':
                if elec_desc not in self.elec_names:
                    raise ValueError('No electrode named {}'.format(elec_desc))
                elec_pos, elec_name = np.where(self.elec_names == elec_desc)[0], elec_desc
                elec_chan_pos = np.where(self.chan_elec_names == elec_name)[0]
            else:
                try:
                    elec_pos, elec_name = int(elec_desc), []
                except ValueError:
                    raise ValueError('Argument should either be a string or a number')
                if elec_pos < 0 or elec_pos > self.n_elec:
                    raise ValueError('Wrong electrode position. Only {} electrodes'.format(self.n_elec))
                else:
                    elec_name = self.elec_names[elec_pos]
                    elec_chan_pos = np.where(self.chan_elec_names == elec_name)[0]
            return elec_chan_pos, elec_name

    def get_eeg_channels(self):
        """ Used to find the EEG channels

        .. note:: EEG channels' name should start with the prefix 'EEG'

        .. note:: If no EEG channel can be found, all channels are considered to be EEG channels

        Returns
        -------
        eeg_channels_ind : array [boolean]
            Boolean array containing 1 if channel is an EEG channel, 0 otherwise
        n_eeg_channels : int
            Number of eeg channels

        """
        is_eeg_channel = lambda x: True if re.match('EEG', x) else False
        eeg_channels_ind = np.array(list(map(is_eeg_channel, self.chan_names)))
        if np.sum(eeg_channels_ind) == 0:
            print('Could not find any EEG channels (with EEG prefix). All channels will be considered EEG')
            eeg_channels_ind = np.ones(len(self.chan_names), dtype=bool)
        return eeg_channels_ind, np.sum(eeg_channels_ind)


class SabDataset:
    """Class representing a SAB dataset - can be from the encoding phase or recognition phase

    Attributes
    ----------
    matlab_dataset_dirpath : str
        Path of the Matlab dataset
    subject_id : str
        Subject identifier
    dataset_type : str
        either 'rec' or 'enc' for recognition and encoding datasets respectively
    srate : int
        Sampling Rate (Hz)
    data : numpy array [3D]
        Data matrix 3D - size [n_chan, n_pnts, n_trials]
    times : numpy array
        time vector
    n_chan, n_pnts, n_trials : int
        number of channels / time points / trials
    tmin, tmax : num
        minimum / maximum time
    channel_info : :class:`ChannelInfo` instance
        Contains information about the channels
    For 'rec' datasets:
    hits, correct_rejects, false_alarms, omissions : list | array
        binary vectors representing each conditions [size: (n_samples)]
    reaction_times : list | array
        reaction times vector [size: (n_samples)]
    """
    def __init__(self, matlab_dataset_dirpath, subject_id, dataset_type='rec', colors_dict=[], **kwargs):
        plt.rcParams['image.cmap'] = 'viridis'
        self.matlab_dataset_dirpath = matlab_dataset_dirpath
        self.subject_id = subject_id
        sns.set()
        sns.set_context('paper')
        if dataset_type.lower() not in ['rec', 'enc']:
            raise ValueError('dataset_type argument must be either rec or enc')
        self.dataset_type = dataset_type.lower()
        # Create class from a matlab dataset
        if matlab_dataset_dirpath:
            # Load Recognition dataset
            if self.dataset_type == 'rec':
                mat_struct = sio.loadmat(os.path.join(matlab_dataset_dirpath, 'EEGrec.mat'), squeeze_me=True)['EEGrec']
            elif self.dataset_type == 'enc':
                mat_struct = sio.loadmat(os.path.join(matlab_dataset_dirpath, 'EEGenc.mat'), squeeze_me=True)['EEGenc']
            self.srate = int(mat_struct['srate'])
            self.data = mat_struct['data'].item()
            self.times = mat_struct['times'].item()
            (self.n_chan, self.n_pnts, self.n_trials) = self.data.shape
            self.tmin, self.tmax = float(mat_struct['xmin']), float(mat_struct['xmax'])
            self.channel_info = ChannelInfo(mat_struct['chanlocs'].item()['labels'])
            if self.dataset_type == 'rec':
                self.hits = sio.loadmat(os.path.join(matlab_dataset_dirpath, 'hits.mat'),
                                        squeeze_me=True)['hits'].astype(bool)
                self.correct_rejects = sio.loadmat(os.path.join(matlab_dataset_dirpath, 'correctRejects.mat'),
                                                   squeeze_me=True)['correctRejects'].astype(bool)
                try:
                    self.false_alarms = sio.loadmat(os.path.join(matlab_dataset_dirpath, 'falseAlarms.mat'),
                                                    squeeze_me=True)['falseAlarms'].astype(bool)
                except IOError:
                    print('Could not load falseAlarms.mat in {}'.format(matlab_dataset_dirpath))
                    self.false_alarms = np.zeros(self.n_trials, dtype=bool)
                try:
                    self.omissions = sio.loadmat(os.path.join(matlab_dataset_dirpath, 'omissions.mat'),
                                                 squeeze_me=True)['omissions'].astype(bool)
                except IOError:
                    print('Could not load omissions.mat in {}'.format(matlab_dataset_dirpath))
                    self.omissions = np.zeros(self.n_trials, dtype=bool)
                self.n_hits, self.n_correct_rejects = np.sum(self.hits), np.sum(self.correct_rejects)
                self.n_false_alarms, self.n_omissions = np.sum(self.false_alarms), np.sum(self.omissions)
                self.reaction_times = sio.loadmat(os.path.join(matlab_dataset_dirpath, 'reactionTimes.mat'),
                                                  squeeze_me=True)['reactionTimes']
                try:
                    self.image_names = sio.loadmat(os.path.join(matlab_dataset_dirpath, 'imageNames.mat'),
                                                   squeeze_me=True)['imageNames']
                except IOError:
                    print('Could not find the variable imageNames.mat in directory {}'.format(matlab_dataset_dirpath))
                    self.image_names = []
        # Create dataset from arguments : must provide the followings arguments :
        # {'srate', 'data', 'tmin', 'tmax', 'channel_info'}.
        # If rec dataset also : {'hits', 'correct_rejects', 'false_alarms', 'omissions'}
        else:
            self.srate = kwargs['srate']
            self.data = kwargs['data']
            if not self.data.ndim == 3:
                raise ValueError('data argument must be 3D : [n_chan, n_pnts, n_trials]')
            (self.n_chan, self.n_pnts, self.n_trials) = self.data.shape
            self.tmin, self.tmax = kwargs['tmin'], kwargs['tmax']
            self.times = np.linspace(self.tmin, self.tmax, self.n_pnts)
            self.channel_info = kwargs['channel_info']
            if self.dataset_type == 'rec':
                self.hits = kwargs['hits']
                self.correct_rejects = kwargs['correct_rejects']
                self.false_alarms = kwargs['false_alarms']
                self.omissions = kwargs['omissions']
                self.n_hits, self.n_correct_rejects = np.sum(self.hits), np.sum(self.correct_rejects)
                self.n_false_alarms, self.n_omissions = np.sum(self.false_alarms), np.sum(self.omissions)
                self.reaction_times = kwargs['reaction_times']
        # Color dictionnary for different conditions
        self.colors_dict = colors_dict if colors_dict else {'hits': 'g', 'cr': 'r', 'omission': 'y', 'fa': 'm'}

    def __str__(self):
        """ Return a description of the SAB dataset

        Examples
        --------
            >>> print(rec_dataset) # doctest: +SKIP
            >>> SAB dataset REC - 001_CC # doctest: +SKIP
            >>> 112 channels, 180 points [-0.10, 0.60s], sampling rate 256.0 Hz # doctest: +SKIP
            >>> 540 trials : 183 hits, 173 correct rejects, 87 omissions, 97 false alarms # doctest: +SKIP
            >>> Channel Info : 112 channels and 17 electrodes # doctest: +SKIP
            >>> 108 EEG channels - 4 non-EEG channels # doctest: +SKIP
            >>> 13 EEG electrodes - 4 non-EEG electrodes # doctest: +SKIP
        """
        desc_str = 'SAB dataset {} - {}\n'.format(self.dataset_type.upper(), self.subject_id)
        desc_str += '{} channels, {} points [{:.2f}, {:.2f}s], sampling rate {} Hz\n'.\
            format(self.n_chan, self.n_pnts, self.tmin, self.tmax, self.srate)
        if self.dataset_type == 'rec':
            desc_str += '{} trials : {} hits, {} correct rejects, {} omissions, {} false alarms\n'.format\
                (self.n_trials, self.n_hits, self.n_correct_rejects, self.n_omissions, self.n_false_alarms)
        desc_str += self.channel_info.__str__()
        return desc_str

    def __getitem__(self, item):
        if type(item) == tuple:
            if len(item) == 3:
                chan_item, pnts_item, trial_item = item
            elif len(item) == 2:
                chan_item, pnts_item = item
                trial_item = slice(None, None, None)
            elif len(item) > 4:
                raise ValueError('Wrong argument item : {}'.format(item))
        else:
            chan_item = item
            pnts_item = slice(None, None, None)
            trial_item = slice(None, None, None)
        data_sel = self.data[chan_item, :, :][:, pnts_item, :][:, :, trial_item]
        channel_info_sel = self.channel_info[chan_item]
        tmin_sel = self.tmin if pnts_item.start is None else pnts_item.start / self.srate
        tmax_sel = self.tmax if pnts_item.stop is None else pnts_item.stop / self.srate
        if self.dataset_type == 'rec':
            return SabDataset(matlab_dataset_dirpath=[], subject_id=self.subject_id, dataset_type=self.dataset_type,
                              srate=self.srate, data=data_sel, tmin=tmin_sel, tmax=tmax_sel, channel_info=channel_info_sel,
                              hits=self.hits[trial_item], correct_rejects=self.correct_rejects[trial_item],
                              false_alarms=self.false_alarms[trial_item], omissions=self.omissions[trial_item],
                              reaction_times=self.reaction_times[trial_item])
        else:
            return SabDataset(matlab_dataset_dirpath=[], subject_id=self.subject_id, dataset_type=self.dataset_type,
                              srate=self.srate, data=data_sel, tmin=tmin_sel, tmax=tmax_sel, channel_info=channel_info_sel)

    def save(self, dir_path='.', filename=[]):
        """ Save the SabDataset instance to a pickle file using the pickle module.

        Parameters
        ----------
        dir_path : str
            Path of the directory where it will be saved
        filename : str
            Filename
        """
        if not os.path.isdir(dir_path):
            print('Creating save directory : {}'.format(dir_path))
            os.mkdir(dir_path)
        if not filename:
            filename = 'sab_dataset_{}_{}_{}.p'.format(self.dataset_type, self.subject_id,
                                                       datetime.strftime(datetime.now(), '%d%m%y_%H%M'))
        with open(os.path.join(dir_path, filename), 'wb') as f:
            _pickle.dump(self, f)

    def save_sig_to_file(self, chanpos, trialpos=[], output_dir=r'.'):
        """ Save the signal to a file

        Parameters
        ----------
        chanpos : int | str
            Channel position
        trialpos : int | None (default: None)
            Trial position. If none, will save all the trials in a 3D data matrix.
        output_dir : str (default: '.')
            Output directory path

        """
        trialpos = np.array(trialpos)
        if not trialpos.size == 1:
            raise ValueError('Argument trialpos should be a scalar')
        if type(chanpos) == str:
            chanpos = self.channel_info.get_channel_pos(chanpos)
        channame = re.sub(' ', '_', self.channel_info.chan_names[chanpos])
        channame = re.sub('\'', 'p', channame)
        out_filename = '{}_{}_chan_{}_trial_{}_{:.0f}Hz.dat'.format(self.dataset_type, self.subject_id, channame,
                                                                    trialpos, self.srate)
        self.data[chanpos, :, trialpos].tofile(os.path.join(output_dir, out_filename))

    def downsample(self, decimate_order):
        """ Downsample the data along the time axis.

        Parameters
        ----------
        decimate_order : int
            Order of the down-sampling. Sampling frequency will be divided by this number

        """
        if not np.isscalar(decimate_order):
            raise ValueError('Argument decimate_order must be a scalar')
        if decimate_order > 12:
            raise Exception('Argument decimate order should not be higher than 12. Apply downsample multiples times')
        decimate_order = int(decimate_order)
        self.data = signal.decimate(self.data, decimate_order, axis=1)
        self.n_pnts = self.data.shape[1]
        self.times = np.linspace(self.times[0], self.times[-1], self.n_pnts)
        self.srate = self.srate / decimate_order
        print('New sampling rate is {}'.format(self.srate))

    def plot_erp(self, channel_desc, plot_ci=1, plot_hits=1, plot_cr=1, plot_omissions=0, plot_fa=0, ax=[], colors_dict=[]):
        """ Plot the evoked response averaged over trials

        Parameters
        ----------
        channel_desc : str | int | array
            Channel's name or position
        plot_ci : bool (default: True)
            If True, plot the confidence interval around the mean
        plot_hits : bool (default: True)
            If True, plot the ERP for the 'Hit' condition
        plot_cr : bool (default: True)
            If True, plot the ERP for the 'Correct Reject' condition
        plot_omissions : bool (default: False)
            If True, plot the ERP for the 'Omission' condition
        plot_fa : bool (default: False)
            If True, plot the ERP for the 'False Alarme' condition
        ax : list | none (default: none)
            List of axis where to plot the ERP

        Returns
        -------
        ax_list : list
            List of axis where to plot the ERP
        leg_h : array
            Array of legends handles
        """
        channel_desc = np.array(channel_desc)
        if channel_desc.dtype.char == 'U':
            channel_num = self.channel_info.get_channel_pos(channel_desc, true_match=False)
        else:
            channel_num = channel_desc
        if type(colors_dict) is not dict:
            colors_dict = self.colors_dict
        ax_list, leg_h = [], []
        if channel_num.size > 1:
            for channel_num_i in channel_num:
                ax_i, leg_h_i = self.plot_erp(channel_num_i, plot_ci, plot_hits, plot_cr, plot_omissions, plot_fa, ax)
                ax_list.append(ax_i)
                leg_h.append(leg_h_i)
        else:
            channel_num = channel_num.squeeze()
            if not ax:
                f = plt.figure()
                ax = f.add_subplot(111)
            if self.dataset_type == 'rec':
                legend_str = []
                if plot_hits:
                    h = self.plot_erp_subplot(self.data[channel_num, :, self.hits], ax, colors_dict['hits'], plot_ci)
                    legend_str.append('Hits')
                    leg_h.append(h)
                if plot_cr:
                    h = self.plot_erp_subplot(self.data[channel_num, :, self.correct_rejects], ax, colors_dict['cr'], plot_ci)
                    legend_str.append('Correct Rejects')
                    leg_h.append(h)
                if plot_omissions:
                    h = self.plot_erp_subplot(self.data[channel_num, :, self.omissions], ax, colors_dict['omission'], plot_ci)
                    legend_str.append('Omissions')
                    leg_h.append(h)
                if plot_fa:
                    h = self.plot_erp_subplot(self.data[channel_num, :, self.false_alarms], ax, colors_dict['fa'], plot_ci)
                    legend_str.append('False Alarms')
                    leg_h.append(h)
                ax.legend(legend_str)
            elif self.dataset_type == 'enc':
                self.plot_erp_subplot(self.data[channel_num, :, :].T, ax, 'b', plot_ci)
                ax.legend(['Encoding'])
            ax.autoscale(axis='x', tight=True)
            ax.set(xlabel='Time (s)', ylabel='Amplitude', title=self.channel_info.get_channel_name(channel_num))
            ax.plot(ax.get_xlim(), [0, 0], 'k', lw=1, zorder=1)
            ax.plot([0, 0], ax.get_ylim(), 'k', lw=1, zorder=1)
            ax.autoscale(axis='y', tight=True)
            ax_list.append(ax)
        return ax_list, np.hstack(leg_h)

    def plot_erp_subplot(self, data, ax, color_str, plot_ci):
        """ Internal rountine used by plot_erp method
        """
        leg_h = ax.plot(self.times, np.mean(data, 0), color=color_str)
        if plot_ci:
            ci_up = np.mean(data, 0) + 1.96 * np.std(data, 0) / np.sqrt(data.shape[0])
            ci_low = np.mean(data, 0) - 1.96 * np.std(data, 0) / np.sqrt(data.shape[0])
            ax.fill_between(self.times, ci_up, ci_low, facecolor=color_str, alpha=0.2)
        return leg_h

    def plot_electrode_erps(self, elec_desc, plot_ci=0, plot_hits=1, plot_cr=0, plot_omissions=0, plot_fa=0):
        """ Plot on the same figure the ERPs for each channel of the selected electrode

        Parameters
        ----------
        elec_desc : str
            Electrode's name
        plot_ci : bool (default: False)
            If True, plot the confidence interval around the mean
        plot_hits : bool (default: True)
            If True, plot the ERP for the 'Hit' condition
        plot_cr : bool (default: False)
            If True, plot the ERP for the 'Correct Reject' condition
        plot_omissions : bool (default: False)
            If True, plot the ERP for the 'Omission' condition
        plot_fa : bool (default: False)
            If True, plot the ERP for the 'False Alarme' condition

        """
        channel_pos, elec_name = self.channel_info.get_electrode_channels_pos(elec_desc)
        chan_names, n_chan = self.channel_info.get_channel_name(channel_pos), len(channel_pos)
        f, leg_h = plt.figure(), []
        ax = f.add_subplot(111)
        if (plot_hits + plot_cr + plot_omissions + plot_fa) > 1:
            colors_hits = sns.light_palette('g', n_chan)
            colors_cr = sns.light_palette('r', n_chan)
            colors_omissions = sns.light_palette('y', n_chan)
            colors_fa = sns.light_palette('m', n_chan)
        else:
            colors = sns.color_palette("muted", n_chan)
        for i, channel_pos_i in enumerate(channel_pos):
            if (plot_hits + plot_cr + plot_omissions + plot_fa) > 1:
                color_dict = {'hits': colors_hits[i], 'cr': colors_cr[i], 'omissions': colors_omissions[i],
                              'fa': colors_fa[i]}
            else:
                color_dict = {'hits': colors[i], 'cr': colors[i], 'omissions': colors[i], 'fa': colors[i]}
            _, leg_h_i = self.plot_erp(channel_pos_i, plot_ci, plot_hits, plot_cr, plot_omissions, plot_fa, ax,
                                       color_dict)
            leg_h.append(leg_h_i)
        ax.set(title='Electrode {}'.format(elec_name))
        legend_str = []
        legend_str.append(np.tile(np.array('Hits '), n_chan) + chan_names) if plot_hits else legend_str
        legend_str.append(np.tile(np.array('CR '), n_chan) + chan_names) if plot_cr else legend_str
        legend_str.append(np.tile(np.array('Omission '), n_chan) + chan_names) if plot_omissions else legend_str
        legend_str.append(np.tile(np.array('FA '), n_chan) + chan_names) if plot_fa else legend_str
        legend_str = np.hstack(np.array(legend_str).reshape(plot_hits + plot_cr + plot_omissions + plot_fa, n_chan).T)
        ax.legend(np.hstack(leg_h), legend_str)

    def plot_mean_spectrum(self, channel_desc, fmin=2, fmax=90, plot_hits=1, plot_cr=1, plot_omissions=0, plot_fa=0):
        channel_desc = np.array(channel_desc)
        if channel_desc.dtype.char == 'U':
            channel_num = np.atleast_1d(self.channel_info.get_channel_pos(channel_desc))
        else:
            channel_num = np.atleast_1d(channel_desc)

        def plot_mean_spectrum_subplot(freqs, psds, color, ax, plot_ci=1):
            psd_mean, psd_std = (10 * np.log10(psds)).mean(0), (10 * np.log10(psds)).std(0)
            ci95_up_pre = psd_mean - 2 * psd_std / psds.shape[0]
            ci95_down_pre = psd_mean + 2 * psd_std / psds.shape[0]
            ax.plot(freqs, psd_mean, c=color)
            if plot_ci:
                ax.fill_between(freqs, ci95_up_pre, ci95_down_pre, color=color, alpha=0.4)

        for chan_num_i in channel_num:
            data_i = self.data[chan_num_i, :, :]
            nfft = min(self.n_pnts, 2048)
            f = plt.figure()
            ax = f.add_subplot(111)
            legend_str = []
            if plot_hits:
                psds_i, freqs = psd_array_welch(data_i[:, self.hits].T, self.srate, fmin=fmin, fmax=fmax, n_fft=nfft)
                plot_mean_spectrum_subplot(freqs, psds_i, self.colors_dict['hits'], ax)
                legend_str.append('Hits')
            if plot_cr:
                psds_i, freqs = psd_array_welch(data_i[:, self.correct_rejects].T, self.srate, fmin=fmin, fmax=fmax, n_fft=nfft)
                plot_mean_spectrum_subplot(freqs, psds_i, self.colors_dict['cr'], ax)
                legend_str.append('Correct Rejects')
            if plot_omissions:
                psds_i, freqs = psd_array_welch(data_i[:, self.omissions].T, self.srate, fmin=fmin, fmax=fmax, n_fft=nfft)
                plot_mean_spectrum_subplot(freqs, psds_i, self.colors_dict['omissions'], ax)
                legend_str.append('Omissions')
            if plot_fa:
                psds_i, freqs = psd_array_welch(data_i[:, self.false_alarms].T, self.srate, fmin=fmin, fmax=fmax, n_fft=nfft)
                plot_mean_spectrum_subplot(freqs, psds_i, self.colors_dict['fa'], ax)
                legend_str.append('False Alarms')
            chan_name_i = self.channel_info.get_channel_name(chan_num_i)
            ax.legend(legend_str)
            ax.autoscale(axis='both', tight=True)
            ax.set(xlabel='Frequency (Hz)', ylabel='Gain (dB)', title='Mean Power Spectral Density - {}'.format(chan_name_i))


    def plot_itpc(self, channel_desc, trial_pos=[], filt_cf=np.logspace(np.log10(3), np.log10(70), 20),
                  filt_bw=np.logspace(np.log10(1.5), np.log10(20), 20), f_tolerance=[], noise_tolerance=[],
                  n_monte_carlo=20, ftype='elliptic', forder=4, do_plot=1, contour_plot=1, n_contours=20):
        """ Plot the ITPC, Inter-Trial Phase Coherence

        Parameters
        ----------
        channel_desc : int | str | array
            Channel's position or name
        trial_pos : array | None (default: None)
            Trial to select before computing the ITPC. If none, select all trials
        filt_cf : array
            Filter's center frequency for computing the ITPC (in Hz)
        filt_bw : array | int
            Filter's bandwidth. Can be fixed or evolve with the center frequency. (In Hz)
        f_tolerance : float | array | None (default: none)
            Tolerance of the cut-off frequencies. A random number in the interval [-f_tolerance/2, +f_tolerance/2] will
            be added to the cut-off frequencies. If none, f_tolerance is set to ``filt_bw / 100`` for each filter.
        noise_tolerance : float | None (default: none)
            Random noise from a uniform distribution in [-noise_tolerance/2, +noise_tolerance/2] will be added to
            the signal x_raw. If none, noise_tolerance is set to ``np.std(data_sel) / 30``.
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
            If 1 , plot the contours (contourf function) else use the imshow function (default: 1)
        n_contours : int
            If ``contour_plot=1``, number of levels in contourf function


        See Also
        --------
        phase_utils.itpc

        """
        channel_desc, trial_pos = np.array(channel_desc), np.array(trial_pos)
        if channel_desc.dtype.char == 'U':
            channel_num = self.channel_info.get_channel_pos(channel_desc)
        else:
            channel_num = channel_desc
        if trial_pos.size == 0:
            trial_pos = np.arange(self.n_trials)
        x_trials = self.data[channel_num, :, :]
        phase_utils.itpc(x_trials, self.srate, filt_cf, filt_bw, f_tolerance=f_tolerance, noise_tolerance=noise_tolerance,
                         n_monte_carlo=n_monte_carlo, ftype=ftype, forder=forder, do_plot=do_plot,
                         contour_plot=contour_plot, n_contours=n_contours)

    def create_features(self, chan_sel=[], electrode_sel=[], trial_sel=[]):
        """ Create a TimeFeatures instance from the current dataset

        Parameters
        ----------
        chan_sel : int | str | array | None (default: None)
            Channel selection to compute the feature only on these channels. If none, select all channels
        electrode_sel : int | str | array | None (default: None)
            Electrode selection to compute the feature only on these electrodes. If none, the selection is done based
            on ``chan_sel`` parameter
        trial_sel : array | None (default: None)
            Trial selection to compute the features only with these trials. If none, select all trials

        Returns
        -------
        TimeFeatures : Instance of TimeFeatures
            TimeFeatures instance generated from the amplitude of the current dataset, given the selected channels

        .. note:: If both chan_sel and electrode_sel are provided, the selection is only done given the electrode
        selection

        """
        chan_sel, electrode_sel, trial_sel = np.array(chan_sel), np.array(electrode_sel), np.array(trial_sel)
        if chan_sel.size == 0:
            chan_sel_pos = np.where(self.channel_info.eeg_chan_ind)[0]
        elif chan_sel.dtype.char == 'U':
            chan_sel_pos = self.channel_info.get_channel_pos()
        else:
            chan_sel_pos = chan_sel
        if electrode_sel.size > 0:
            chan_sel_pos, _ = self.channel_info.get_electrode_channels_pos(electrode_sel)
        if trial_sel.size == 0:
            trial_sel_pos = np.arange(self.n_trials, dtype=int)
        elif trial_sel.dtype == bool:
            trial_sel_pos = np.where(trial_sel)[0]
        else:
            trial_sel_pos = trial_sel
        feature_data = self.data[np.ix_(chan_sel_pos, np.arange(self.n_pnts), trial_sel_pos)]
        feature_names = self.channel_info.chan_names[chan_sel_pos]
        if self.dataset_type == 'enc':
            labels = np.ones(trial_sel_pos.size)
            label_names = 'Image'
        elif self.dataset_type == 'rec':
            # Check that each trial is assign in only one condition
            if not np.sum(self.hits & self.correct_rejects & self.omissions & self.false_alarms) == 0:
                raise ValueError('Each trial should be assigned to only one condition')
            labels = 1 * self.hits + 2 * self.correct_rejects + 3 * self.omissions + 4 * self.false_alarms
            label_names = np.array(['Hits', 'Correct rejects', 'Omissions', 'False Alarms'])
            reaction_time_sel = self.reaction_times[trial_sel_pos]
        else:
            raise ValueError('Wrong dataset_type value : {}'.format(self.dataset_type))
        labels_sel = labels[trial_sel_pos]
        label_names_sel = label_names[np.unique(labels_sel)-1]
        time_feature_name = '{}_{}'.format(self.subject_id, self.dataset_type)
        return timefeatures.TimeFeatures(feature_data, labels_sel, self.tmin, self.tmax, feature_names, label_names_sel,
                                         self.srate, reaction_time_sel, name=time_feature_name)


def load_sab_dataset(filepath):
    """ Load a time SabDataset instance previously saved using the pickle module

    Parameters
    ----------
    filepath : str
        Path of the pickle file

    Returns
    -------
    sab_dataset : SabDataset instance
        The SabDataset instance
    """
    with open(filepath, 'rb') as f:
        return _pickle.load(f)

