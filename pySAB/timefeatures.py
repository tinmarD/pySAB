import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.axes import Axes
import seaborn as sns
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from scipy.stats import pearsonr, spearmanr
import inspect
import re
import tqdm
from datetime import datetime
import pickle
from os import path, mkdir
from sab_clustering import *
from featureextracter import FeatureExtracter
sns.set()
sns.set_context('paper')


class TimeFeatures:
    """ Class for the extraction and visualization of features across time from the raw amplitude data.
    Visualize the evolution of different features over a time course ``[tmin, tmax]``, for multiples trials, for
    multiple conditions. The condition of each trial is defined by the ``labels`` vector.
    The original data is given by the 3D array ``data_ori`` (size: [n_features, n_pnts, n_trials]) which usually
    represents the amplitude of each channel. From these original data, other feature can be computed using the
    ``feature_extracter`` attribute, which will be stored in the ``data`` attribute.

    Attributes
    ----------
    data_ori : array (size: n_channels*n_pnts*n_trials)
        Original amplitude data. Other features will be computed from this.
    data : array (size: n_features*n_pnts*n_trials)
        Features data. Contain all the features concatenated.
    tmin : float
        Starting time (s)
    tmax : float
        Ending time (s)
    srate : float
        Sampling frequency (Hz)
    name : str
        A string identifying the time features
    times : array
        Time vector from ``tmin`` to ``tmax``
    n_chan : int
        Number of channels
    n_pnts : int
        Number of time points
    n_trials : int
        Number of trials
    n_features : int
        Number of features
    n_labels : int
        Number of different conditions
    labels : array (size: n_trials)
        Condition of each trial represented by a number. Name of the associated condition can be accessed from
        the ``label_dict`` attribute
    channel_names : array (size: n_features)
        Name of each channel
    feature_names : array (size: n_features)
        Name of each feature
    feature_type : array (size: n_features)
        Type of each feature (e.g. : 'amplitude', 'phase alpha', ...)
    feature_channame : array (size: n_features)
        Name of the channel associated with each feature
    label_dict : dict
        Dictionnary containing the condition name
    reaction_times : array (size: n_trials)
        Can be defined for some trials involving a timed response
    preprocessing_done : bool
        True if preprocessing has been done, False otherwise
    feature_extracter : ``featureextracter.FeatureExtracter`` instance
        Used to compute feature from the original data
    """
    def __init__(self, data, labels, tmin, tmax, channel_names, label_names, srate, reaction_times=[], name=[]):
        self.data_ori = np.array(data)
        self.data = np.array(data)
        if data.ndim == 2:
            self.n_features, self.n_trials = data.shape
            self.n_pnts = 1
            if not tmin == tmax:
                raise ValueError('Arguments tmin and tmax should be equal for features with one time point')
        elif data.ndim == 3:
            (self.n_features, self.n_pnts, self.n_trials) = data.shape
        self.tmin, self.tmax = tmin, tmax
        self.srate = srate
        self.times = np.linspace(tmin, tmax, self.n_pnts)
        self.labels = np.array(labels)
        self.n_chan = self.n_features
        self.channel_names = np.array(channel_names)
        self.feature_names = np.array(['Amp {}'.format(chan_name) for chan_name in channel_names])
        self.feature_type = np.repeat('Amp', self.n_chan)
        self.feature_channame = np.array(channel_names)
        _, idx = np.unique(label_names, return_index=True)
        label_names = np.array(label_names)[np.sort(idx)]
        label_values = np.unique(labels)
        self.n_labels = len(label_values)
        if not len(label_values) == len(label_names):
            raise ValueError('The number of label values is different from the number of label names')
        self.label_dict = dict(zip(label_values, label_names))
        self.reaction_times = np.array(reaction_times)
        self.preprocessing_done = False
        self.feature_extracter = FeatureExtracter(self.data_ori, self.srate, self.n_chan, self.n_pnts, self.n_trials,
                                                  channel_names)
        self.name = name if name else 'Unnamed'
        sns.set_context('paper')
        plt.rcParams['image.cmap'] = 'viridis'

    def __str__(self):
        desc_str = 'Time Features {} - {} features, {} time points, {} trials\n'.format(self.name, self.n_features,
                                                                                        self.n_pnts, self.n_trials)
        desc_str += '{} labels : {}\n'.format(self.n_labels, self.label_dict)
        _, indexes = np.unique(self.feature_type, return_index=True)
        unique_types = self.feature_type[np.sort(indexes)]
        desc_str += 'Feature types : {}'.format(', '.join(unique_types))
        return desc_str

    def save(self, dir_path='.', filename=[], filename_desc=[]):
        """ Save the time features instance to a pickle file using the pickle module.

        Parameters
        ----------
        dir_path : str
            Path of the directory where it will be saved
        filename : str
            Filename
        """
        if not path.isdir(dir_path):
            print('Creating save directory : {}'.format(dir_path))
            mkdir(dir_path)
        if not filename:
            filename = 'time_feature_{}_{}.p'.format(self.name, datetime.strftime(datetime.now(), '%d%m%y_%H%M'))
        if filename_desc:
            filename = '{}_{}.p'.format(filename[:-2], filename_desc)
        with open(path.join(dir_path, filename), 'wb') as f:
            pickle.dump(self, f)

    def get_feature_pos(self, feature_pos=[], feature_type=[], feature_channame=[], join=0):
        """ Get the feature position from feature type or/and feature channel name

        Parameters
        ----------
        feature_pos : int | array | None (default: none)
            If provided, this method do nothing
        feature_type : int | array | None (default: none)
            Feature type selection
        feature_channame : int | array | None (default: none)
            Feature channel name selection
        join : bool (default: False)
            If True, selection the union ``[OR]`` between the 2 selection criteria (type and channame), otherwise
            select the intersection ``[AND]``

        Returns
        -------
        feature_pos : int | array
            Feature position
        """
        feature_pos, feature_type = np.atleast_1d(feature_pos), np.atleast_1d(feature_type)
        feature_channame = np.atleast_1d(feature_channame)
        if feature_pos.size == 0 and feature_type.size > 0 or feature_channame.size > 0:
            feature_sel_type = np.zeros(self.n_features, dtype=bool)
            feature_sel_channame = np.zeros(self.n_features, dtype=bool)
            for feat_type in feature_type:
                feat_type_ind = np.array([feat_type in feat_type_i for feat_type_i in self.feature_type])
                if feat_type_ind.astype(int).sum() == 0:
                    print('No feature type named {}'.format(feat_type))
                feature_sel_type = feature_sel_type | feat_type_ind
            for feat_channame_i in feature_channame:
                if type(feature_channame) is int:
                    feat_channame_i = self.channel_names[feat_channame_i]
                if type(feat_channame_i) is str or type(feat_channame_i) is np.str_:
                    feat_channame_ind = np.array([feat_channame_i in channame_i for channame_i in self.feature_channame])
                    if feat_channame_ind.astype(int).sum() == 0:
                        print('No channel named {}'.format(feat_channame_i))
                    feature_sel_channame = feature_sel_channame | feat_channame_ind
                else:
                    raise ValueError('Unknown type feat_channame : {}'.format(feat_channame_i))
            if feature_type.size > 0 and feature_channame.size > 0 and not join:
                feature_sel_ind = feature_sel_type & feature_sel_channame
            else:
                feature_sel_ind = feature_sel_type | feature_sel_channame
            feature_pos = np.where(feature_sel_ind)[0]
        elif feature_pos.size == 0 and feature_type.size == 0 and feature_channame.size == 0:
            feature_pos = np.arange(0, self.n_features)
        return feature_pos

    def get_data(self, **kwargs):
        """ Select the data corresponding to selected feature and/or labels and/or time points. The selection
        parameter can be the feature position, channel's name, type and/or the condition (label_key) and/or the
        time points.

        Parameters
        ----------
        feature_pos : array, optional
            position of the features to select
        feature_type : array, optional
            type of the features to select
        feature_channame : array, optional
            Channel name of the feature to select
        label_keys : int | array, optional
            Label keys, if not specified, select all labels condition
        time_points : array, optional
            Time points to select

        Returns
        -------
        data_feature : array (size n_features_sel, n_pnts_sel, n_trials_sel)
            Selected features array. The returned array is always 3D, even if the a single a feature or time point
            is selected.
        feature_names : array
            Names of the selected features
        label_vect : array
            Labels of the selected trials
        n_selected_trials : int
            Number of selected trials

        """
        kwargs_keys = list(kwargs.keys())
        select_args = ['feature_pos', 'label_keys', 'time_points', 'feature_type', 'feature_channame']
        for key in kwargs_keys:
            if key not in select_args:
                raise ValueError('Wrong argument for selecting data : {}. Possible values are {}'.format(key, select_args))

        feature_pos = kwargs['feature_pos'] if 'feature_pos' in kwargs_keys else []
        label_keys = kwargs['label_keys'] if 'label_keys' in kwargs_keys else []
        time_points = kwargs['time_points'] if 'time_points' in kwargs_keys else []
        feature_type = kwargs['feature_type'] if 'feature_type' in kwargs_keys else []
        feature_channame = kwargs['feature_channame'] if 'feature_channame' in kwargs_keys else []
        feature_pos, label_keys, time_points = np.atleast_1d(feature_pos), np.atleast_1d(label_keys), np.atleast_1d(time_points)
        feature_type, feature_channame = np.atleast_1d(feature_type), np.atleast_1d(feature_channame)
        feature_pos = self.get_feature_pos(feature_pos, feature_type, feature_channame)
        if label_keys.size == 0:
            label_keys = np.array(list(self.label_dict.keys()))
        if time_points.size == 0:
            time_points = np.arange(0, self.n_pnts, 1)
        label_keys = np.array([label_keys]) if label_keys.ndim == 0 else np.array(label_keys)
        label_ind = np.zeros(self.n_trials, dtype=bool)
        label_names = []
        for label_key in label_keys:
            if label_key not in self.label_dict.keys():
                raise ValueError('Label key {} not in known labels keys : {}'.format(label_key, self.label_dict))
            label_ind = label_ind | (self.labels == label_key)
            label_names.append(self.label_dict[label_key])
        label_pos = np.array([np.where(label_ind)[0]]) if np.sum(label_ind) == 1 else np.where(label_ind)[0]
        time_points = np.array([time_points]) if time_points.ndim == 0 else time_points
        label_vect = self.labels[label_pos]
        n_selected_trials = len(label_vect)
        # Data selection
        data_feature = self.data[feature_pos, :, :]
        data_feature = data_feature[:, :, label_pos]
        data_feature = data_feature[:, time_points, :]
        return data_feature, feature_pos, label_vect, n_selected_trials

    def sort_features(self, sorting_variable='feature_type'):
        """ Sort the features according to sorting_variable

        Parameters
        ----------
        sorting_variable : str
            Sorting variable. Can be 'feature_type' or 'feature_channame'

        """
        sorting_var_possible = ['feature_type', 'feature_channame']
        if type(sorting_variable) is not str or sorting_variable not in sorting_var_possible:
            raise ValueError('sorting_variable argument must be a string. Possible values are {}'.format(sorting_var_possible))
        if sorting_variable is 'feature_type':
            sorting_vect = np.argsort(self.feature_type)
        elif sorting_variable is 'feature_channame':
            sorting_vect = np.argsort(self.feature_channame)
        self.data = self.data[sorting_vect, :, :]
        self.feature_type = self.feature_type[sorting_vect]
        self.feature_channame = self.feature_channame[sorting_vect]
        self.feature_names = self.feature_names[sorting_vect]

    def sort_trials(self, sorting_variable='label', direction='ascend'):
        """ Sort the data trial order given the sorting variable. Sort data_ori and data given the new trial order

        Parameters
        ----------
        sorting_variable : str
            Sorting variable, can be either 'label' or 'reaction times'
        direction : str
            Possible values are 'ascend' or 'descend'

        """
        sorting_var_possible = ['label', 'reaction-times', 'rt', 'reactiontimes']
        if type(sorting_variable) is not str or sorting_variable not in sorting_var_possible:
            raise ValueError('sorting_variable argument must be a string. Possible values are {}'.format(sorting_var_possible))
        if type(direction) is not str or direction not in ['ascend', 'descend']:
            raise ValueError('direction argument must be a string. Possible values are {}'.format(['ascend', 'descend']))
        if sorting_variable is 'label':
            sorting_vect = np.argsort(self.labels) if direction is 'ascend' else np.argsort(self.labels)[::-1]
        elif sorting_variable in ['reaction-times', 'rt', 'reactiontimes']:
            sorting_vect = np.argsort(self.reaction_times) if direction is 'ascend' else np.argsort(self.reaction_times)[::-1]
        self.data_ori = self.data_ori[:, :, sorting_vect]
        self.data = self.data[:, :, sorting_vect]
        self.labels = self.labels[sorting_vect]
        self.reaction_times = self.reaction_times[sorting_vect]

    def extract_feature(self, feature_type=[]):
        """ Compute features from the original data ``data_ori`` and add them to the feature array ``data``

        Parameters
        ----------
        feature_type : str | array
            Feature type to compute. To choose in ``['dwt', 'phase_hilbert', 'stft_pxx', 'cwt_coefs', 'cwt_phase']``

        """
        possibles_features = ['filt_bandpower', 'dwt', 'stft_bandpower', 'stft_phase', 'cwt_bandpower', 'cwt_phase', 'phase_hilbert']
        if not feature_type:
            print('Possible features to compute : {}'.format(possibles_features))
        elif type(feature_type) == str:
            feature_type = [feature_type]
        feature_type = [feat_type.lower() for feat_type in feature_type]
        for feat_type_i in feature_type:
            if feat_type_i not in possibles_features:
                raise ValueError('Wrong feature type : {}. Possible features are {}'.format(feat_type_i, possibles_features))
        # default_pfreqs = np.logspace(np.log10(2), np.log10(90), 40)
        if 'filt_bandpower' in feature_type:
            feat_mat, feat_name, feat_type, feat_channame = self.feature_extracter.bandpower_on_data()
            self.add_feature(feat_mat, feat_name, feat_type, feat_channame)
        if 'dwt' in feature_type:
            feat_mat, feat_name, feat_type, feat_channame = self.feature_extracter.dwt_on_data(wav_name='db4')
            self.add_feature(feat_mat, feat_name, feat_type, feat_channame)
        if 'stft_bandpower' in feature_type or 'stft_phase' in feature_type:
            feat_mat, feat_name, feat_type, feat_channame = self.feature_extracter.stft_on_data()
            if 'stft_bandpower' in feature_type:
                self.add_feature(feat_mat[0], feat_name[0], feat_type[0], feat_channame[0])
            if 'stft_phase' in feature_type:
                self.add_feature(feat_mat[1], feat_name[1], feat_type[1], feat_channame[1])
        if 'cwt_bandpower' in feature_type or 'cwt_phase' in feature_type:
            feat_mat, feat_name, feat_type, feat_channame = self.feature_extracter.cwt_on_data()
            if 'cwt_bandpower' in feature_type:
                self.add_feature(feat_mat[0], feat_name[0], feat_type[0], feat_channame[0])
            if 'cwt_phase' in feature_type:
                self.add_feature(feat_mat[1], feat_name[1], feat_type[1], feat_channame[1])
        if 'phase_hilbert' in feature_type:
            center_freq = np.logspace(np.log10(4), np.log10(65), 20)
            bandwidth = np.logspace(np.log10(2), np.log10(15), 20)
            feat_mat, feat_name, feat_type, feat_channame = self.feature_extracter.filter_hilbert_on_data(
                center_freq, bandwidth, ftype='elliptic', forder=4)
            self.add_feature(feat_mat, feat_name, feat_type, feat_channame)

    def add_trials(self, trials_mat, label_id, label_name, reaction_times=[]):
        """ Add trials related to a new or already existing label

        Parameters
        ----------
        trials_mat : array (size: n_features*n_pnts*n_new_trials)
            New trials array, to add to the existing trials
        label_id : int
            New trials label id
        label_name : str
            New trials label name
        reaction_times : array (size: n_new_trials) | None (default)
            If reaction times are associated with the new trials, supply it here

        """
        reaction_times = np.array(reaction_times)
        if not self.data.shape[0] == trials_mat.shape[0] or not self.data.shape[1] == trials_mat.shape[1]:
            raise ValueError('Argument trials_mat must have the same number of feature and the same number of time'
                             ' points the the data attribute')

        if label_id in self.label_dict.keys():
            if not self.label_dict[label_id] == label_name:
                raise ValueError('label_id {} already exists but the label name does not correspond'.format(label_id))
        else:
            # Add the new label to the label_dict attribute
            self.label_dict[label_id] = label_name
            self.n_labels += 1

        # Add the data:
        n_new_trials = trials_mat.shape[2]
        self.data = np.concatenate([self.data, trials_mat], axis=2)
        self.labels = np.hstack([self.labels, label_id*np.ones(n_new_trials)])
        self.n_trials += n_new_trials
        if reaction_times.size > 0:
            if not reaction_times.size == n_new_trials:
                raise ValueError('Argument reaction_times size must be equal to the number of new trials')
            self.reaction_times = np.hstack([self.reaction_times, reaction_times])

    def add_feature(self, feat_mat, feat_name, feat_type, feat_channame):
        """ Add the feature matrix ``feat_mat`` to the existing features in ``data`` attribute. Update ``feature_name``,
        ``feature_channame`` and ``feature_type`` attributes.

        Parameters
        ----------
        feat_mat : array (size: n_new_features*n_pnts*n_trials)
            New feature array, to add to the existing features
        feat_name : array (size: n_new_features)
            Name of the new features
        feat_type : array (size: n_new_features)
            Type of the new features
        feat_channame : array (size: n_new_features)
            Channel's name of the new features
        """
        # Concatenate previous and new features
        self.data = np.vstack([self.data, feat_mat])
        self.feature_names = np.hstack([self.feature_names, feat_name])
        self.feature_type = np.hstack([self.feature_type, feat_type])
        self.feature_channame = np.hstack([self.feature_channame, feat_channame])
        self.n_features += feat_mat.shape[0]

    def plot_feature_erp(self, plot_traces=0, ax=[], **kwargs):
        """ Plot the feature evolution across time. Selection parameters can be passed.

        Parameters
        ----------
        plot_traces : Bool
            If True plot all the trials, otherwise plot the mean and tthe 95% CI
        kwargs : Optional
            Data selection parameters. Can be ``feature_pos``, ``feature_type``, ``feature_channame``,
            ``label_keys``, ``time_points``. See :func:`get_data`

        """
        data_sel, feature_pos, target, _ = self.get_data(**kwargs)
        feature_names = self.feature_names[feature_pos]
        n_features, n_pnts, n_trials = data_sel.shape
        label_sel_key = np.array(np.unique(target))
        if label_sel_key.size == 0:
            label_sel_key = np.array(list(self.label_dict.keys()))
        elif label_sel_key.size == 1:
            label_sel_key = np.atleast_1d(label_sel_key)
        label_colors = sns.color_palette(n_colors=label_sel_key.size)
        n_subplots = 1 if not plot_traces else label_sel_key.size
        for i, feature_name_i in enumerate(feature_names):
            if not ax:
                f = plt.figure()
                ax = f.add_subplot(1, n_subplots, 1)
            data_i = data_sel[i, :, :]
            for j, label_key in enumerate(label_sel_key):
                if j > 0 and plot_traces and not ax:
                    ax = f.add_subplot(1, n_subplots, j+1, sharex=ax, sharey=ax)
                data_i_label = data_i[:, target == label_key]
                ci_up = np.mean(data_i_label, 1) + 1.96 * np.std(data_i_label, 1) / np.sqrt(n_trials)
                ci_low = np.mean(data_i_label, 1) - 1.96 * np.std(data_i_label, 1) / np.sqrt(n_trials)
                if plot_traces:
                    ax.plot(self.times, data_i_label, c=label_colors[j], alpha=0.1)
                else:
                    ax.fill_between(self.times, ci_up.squeeze(), ci_low.squeeze(), alpha=0.2)
                ax.plot(self.times, np.mean(data_i_label, 1).squeeze(), c=label_colors[j], lw=2, label=self.label_dict[label_key])
                ax.autoscale(axis='both', tight=True)
                ax.plot(ax.get_xlim(), [0, 0], 'k', lw=1, zorder=1)
                ax.plot([0, 0], ax.get_ylim(), 'k', lw=1, zorder=1)
                ax.legend()
                ax.set(xlabel='Time (s)', ylabel='Amplitude', title=feature_name_i)

    def plot_feature_erpimage(self, ax=[], cax=[], **kwargs):
        """ Plot the erp-image of the selected features

        kwargs : Optional
            Data selection parameters. Can be ``feature_pos``, ``feature_type``, ``feature_channame``,
            ``label_keys``, ``time_points``. See :func:`get_data`

        """
        data_sel, feature_pos, target, _ = self.get_data(**kwargs)
        feature_names = self.feature_names[feature_pos]
        n_features, n_pnts, n_trials = data_sel.shape
        # label_sel_key = np.array(np.unique(target))
        # if label_sel_key.size == 0:
        #     label_sel_key = np.array(list(self.label_dict.keys()))
        # elif label_sel_key.size == 1:
        #     label_sel_key = np.array([label_sel_key])
        for i, feature_name_i in enumerate(feature_names):
            if not ax:
                f = plt.figure()
                ax = f.add_subplot(111)
            else:
                f = ax.figure
            data_i = data_sel[i, :, :]
            im = ax.imshow(data_i.T, aspect='auto', origin='lower', extent=(self.tmin, self.tmax, 0, n_trials))
            ax.set(xlabel='Time (s)', ylabel='Trial', title='ERP Image - {}'.format(feature_name_i))
            f.colorbar(im, cax=cax) if cax else f.colorbar(im, ax=ax)
            ax.grid(False)
            ax.plot([0, 0], [0, n_trials], 'k')

    def plot_feature_distribution(self, ax=[], **kwargs):
        """ Plot the distribution of features at a certain time point for selected labels. The time_points parameter
        should be specified.

        Parameters
        ----------
        kwargs : Optional
            Data selection parameters. Can be ``feature_pos``, ``feature_type``, ``feature_channame``,
            ``label_keys``, ``time_points``. The ``time_points`` parameter must be specified. See :func:`get_data`
        """
        if 'time_points' not in kwargs.keys():
            raise ValueError('Parameter \'time_points\' must be specified and contain only 1 time point')
        elif not np.array(kwargs['time_points']).size == 1:
            raise ValueError('Parameter \'time_points\' must be specified and contain only 1 time point')
        else:
            time_point = int(kwargs['time_points'])
        data_sel, feature_pos, labels, _ = self.get_data(**kwargs)
        label_keys = np.unique(labels)
        label_names = [self.label_dict[key] for key in label_keys]
        for i, feat_pos_i in enumerate(feature_pos):
            data_feat_i = data_sel[i, :, :].squeeze()
            if not ax:
                f = plt.figure()
                ax = f.add_subplot(111)
            data = []
            for key in label_keys:
                data.append(data_feat_i[labels == key])
            pal = sns.cubehelix_palette(len(label_keys), rot=-.5, dark=.3)
            sns.violinplot(data=data, palette=pal, inner='box', ax=ax)
            ax.set(title='Feature {} - time : {:.3f} s'.format(self.feature_names[feat_pos_i], time_point / self.srate +
                                                               self.tmin), xlabel='Condition', ylabel='Feature Value')
            ax.set_xticklabels(label_names)

    def plot_feature_hits_reaction_time(self, scaling=[], feature_range=(0, 1), transform=[],
                                        ax_list=[], **kwargs):
        """ Plot the reaction time in function of the feature. Used to visualize any correlation between the feature
        and the reaction time (if it is defined). The 'time_points' parameter must be set and contain only 1 value.
        Only one feature must be selected.

        Parameters
        ----------
        scaling : str | None (default: none)
            Scaling method. Can be :

            * 'standardization' : compute the z-score
            * 'normalization' : normalization betwenn 0 and 1 (default) or set by ``feature_range``
            * 'robust' : robust z-score
        feature_range : tuple | array (default: (0, 1))
            Feature range when scaling is set to normalization
        transform : lambda | none (default: none)
            Function to transform the feature. Use lambdas, e.g. : ``lambda x: x**2``.
        ax_list : list | None (default: none)
            List of axes to plot on
        kwargs : Optional (except ``time_points``)
            Data selection parameters. Can be ``feature_pos``, ``feature_type``, ``feature_channame``,
            ``label_keys``, ``time_points``. The ``time_points`` parameter must be specified and only 1 feature should
            be selected. The ``label_keys`` is automatically set to the 'Hits' key. See :func:`get_data`

        """
        if self.reaction_times.size == 0:
            print('No reaction time variable')
            return
        if 'time_points' not in kwargs.keys():
            raise ValueError('Parameter \'time_points\' must be specified and contain only 1 time point')
        elif not np.array(kwargs['time_points']).size == 1:
            raise ValueError('Parameter \'time_points\' must be specified and contain only 1 time point')
        else:
            time_point = int(kwargs['time_points'])
        if scaling and scaling.lower() not in ['standardization', 'normalization', 'robust']:
            raise ValueError('scaling possible values are : standardization, normalization or robust')
        ax_list = np.atleast_1d(ax_list)
        kwargs['label_keys'] = self.get_label_key_from_value('Hits')
        data_feature, feature_pos, labels, _ = self.get_data(**kwargs)
        if not feature_pos.size == 1:
            raise ValueError('Only one feature should be selected')
        if scaling:
            data_feature = scale_data(data_feature.squeeze(), scaling, feature_range)
        else:
            scaling = 'None'
        if transform:
            data_feature = transform(data_feature)
            transform_str = re.search('lambda .+,', inspect.getsource(transform))
            if not transform_str:
                transform_str = re.search('lambda .+\)', inspect.getsource(transform))
            transform_str = transform_str.group()[7:-1]
        else:
            transform_str = 'None'
        if not ax_list.size == 3 and not ax_list.size == 1:
            f = plt.figure()
            ax1 = plt.subplot2grid((5, 5), loc=(0, 0), rowspan=4, colspan=4)
            ax2 = plt.subplot2grid((5, 5), loc=(4, 0), rowspan=1, colspan=4)
            ax3 = plt.subplot2grid((5, 5), loc=(0, 4), rowspan=4, colspan=1)
        else:
            if ax_list.size == 1:
                ax1, ax2, ax3 = ax_list[0], [], []
            else:
                ax1, ax2, ax3 = ax_list[0], ax_list[1], ax_list[2]
        data_x = data_feature.squeeze()
        rt_hits = self.reaction_times[self.labels == self.get_label_key_from_value('Hits')]
        corr_pearson, _ = pearsonr(data_x, rt_hits)
        corr_spearman, _ = spearmanr(data_x, rt_hits)
        ax1 = sns.regplot(data_x, rt_hits, ci=None, ax=ax1)
        ax1.set(title='feature : {} - Time : {:.3f}s - Transform : {} - Scaling : {}'.format
                (self.feature_names[feature_pos], self.sample2time(time_point), transform_str, scaling),
                ylabel='Reaction Time (ms)')
        ax1_xlims, ax1_ylims = ax1.get_xlim(), ax1.get_ylim()
        ax1.text(x=0.05, y=0.90, s='Pearson Corr. : {:.3f}\nSpearman Corr. : {:.3f}'.format(corr_pearson, corr_spearman),
                 transform=ax1.transAxes)
        ax1.axhline(y=1000*self.sample2time(time_point), color='r', alpha=0.3, zorder=1)
        ax1.set_xlim(ax1_xlims)
        ax1.set_ylim(ax1_ylims)
        if ax2 and ax3:
            sns.distplot(data_x, ax=ax2)
            sns.distplot(rt_hits, vertical=True, ax=ax3)
            ax2.set(xlabel='Feature Distribution')
            ax3.set_ylabel('Reaction Time Distribution', rotation=-90, labelpad=10)
            ax3.yaxis.set_label_position('right')
            ax3.set_xlim((0, 0.04))

    def compute_feature_importance(self, label_keys, n_decision_trees=250):
        """ Compute feature importance using forest of decitions trees. See :func:`sklearn.ensemble.ExtraTreesClassifier`

        Parameters
        ----------
        label_keys : array
            Labels to use for the classification. Should contain at least 2 conditions.
        n_decision_trees : int (default: 250)
            Number of decision trees used in the classifier.

        """
        label_keys = np.array(label_keys)
        if label_keys.size < 2:
            raise ValueError('Argument labels keys should contain at least 2 elements')
        label_names = [self.label_dict[key] for key in label_keys]
        feature_importance_ev = np.zeros((self.n_features, self.n_pnts))
        for t in tqdm.tqdm(range(0, self.n_pnts)):
            data, _, labels, _ = self.get_data(time_points=t, label_keys=label_keys)
            data = data.squeeze().T
            forest = ExtraTreesClassifier(n_estimators=n_decision_trees, random_state=0)
            forest.fit(data, labels)
            feature_importance_ev[:, t] = forest.feature_importances_
        f = plt.figure()
        ax = f.add_subplot(111)
        im = ax.imshow(feature_importance_ev, aspect='auto', origin='lower', extent=(self.tmin, self.tmax, 0,
                                                                                     self.n_features))
        plt.set_cmap('viridis')
        plt.colorbar(im)
        ax.plot([0, 0], ax.get_ylim(), 'k')
        ax.yaxis.set_ticks(np.arange(0.5, self.n_features+0.5, 1))
        ax.yaxis.set_ticklabels(self.feature_names[np.arange(0, self.n_features, 1)])
        ax.grid(False)
        ax.set(xlabel='Time (s)', ylabel='Feature name', title='Feature Importance using forest of trees - '
                                                               'Classification : {}'.format(' / '.join(label_names)))

    def cluster_data(self, cluster_algo, do_plot=False, ax=[], cb_ax=[], **kwargs):
        """ Cluster the data selected by **kwargs  - see :func:`timefeatures.get_data` -
        Return the cross-tabulation of the predicted versus true labels

        Parameters
        ----------
        cluster_algo : str
            Clustering algorithm name.
            Possible values: ['kmeans', 'affinitypropagation', 'spectralclustering', 'agglomerativeclustering', 'dbscan']
        do_plot : bool
            If True, plot the clustering results
        ax : matplotlib axis | None
            Axis to plot on
        cb_ax : matplotlib axis | None
            Axis for the colorbar
        kwargs :
            Data selection parameters. Can be ``feature_pos``, ``feature_type``, ``feature_channame``,
            ``label_keys``, ``time_points``. See :func:`get_data`

        Returns
        -------
        ct : pandas DataFrame
            Output cross-tabulation DataFrame

        """
        if 'time_points' not in kwargs.keys():
            raise ValueError('\'time_points\' argument must be set')
        data_sel, _, y_true, _ = self.get_data(**kwargs)
        data_sel_2d = np.atleast_2d(data_sel.squeeze()).T[:, :]
        ct = cluster_data_2d(cluster_algo, data_sel_2d, y_true, self.label_dict)
        if do_plot:
            if not ax:
                f, ax = plt.subplots()
            else:
                f = ax.figure
            im = ax.imshow(ct, origin='lower', aspect='auto', norm=mpl.colors.Normalize(vmin=0, vmax=1), cmap="viridis")
            ax.grid(False)
            f.colorbar(im, cax=cb_ax) if cb_ax else f.colorbar(im, ax=ax)
            ax.yaxis.set_ticks(ct.index)
            ax.xaxis.set_ticks(range(len(ct.columns)))
            ax.xaxis.set_ticklabels(list(ct.columns))
            # Print values:
            for (i, j), val in np.ndenumerate(np.array(ct)):
                ax.text(j, i, '{:.2f}'.format(val))
        return ct

    def compute_correlation_feature_target(self, **kwargs):
        """ Compute and plot the correlation between the selected features for 2 specified conditions. These conditions
        must be specified by providing a `label_keys` parameter. Feature selections parameters can be used to select
        features - see :func:`get_data` -

        Parameters
        ----------
        kwargs : Optional (except for ``label_keys``)
            Can be ``feature_pos``, ``feature_type``, ``feature_channame``, ``label_keys``, ``time_points``.
            ``label_keys`` must be set. See :func:`get_data`

        """
        if 'label_keys' not in kwargs.keys():
            raise ValueError('Argument label_keys must be specified and contain 2 values')
        else:
            label_keys = np.array(kwargs['label_keys'])
        if not label_keys.size == 2:
            raise ValueError('Argument label_keys must contain 2 values')
        data_sel, feature_pos, target, _ = self.get_data(**kwargs)
        feature_names = self.feature_names[feature_pos]
        n_features, n_pnts, n_trials = data_sel.shape
        label_names = [self.label_dict[key] for key in label_keys]
        pearson_corr = np.zeros((n_features, n_pnts))
        for t in range(0, n_pnts):
            for i in range(0, n_features):
                pearson_corr[i, t] = pearsonr(data_sel[i, t, :], target)[0]
        f = plt.figure()
        ax = f.add_subplot(111)
        plt.set_cmap('viridis')
        im = ax.imshow(np.abs(pearson_corr), aspect='auto', origin='lower', extent=(self.tmin, self.tmax, 0, n_features))
        plt.colorbar(im)
        ax.plot([0, 0], ax.get_ylim(), 'k')
        ax.yaxis.set_ticks(np.arange(0.5, n_features+0.5, 1))
        ax.yaxis.set_ticklabels(feature_names[np.arange(0, n_features, 1)])
        ax.grid(False)
        ax.set(xlabel='Time (s)', ylabel='Feature name', title='Absolute Pearson Correlation between feature and target'
                                                               ' value - {}/{}'.format(label_names[0], label_names[1]))

    def compute_correlation_hits_reaction_times(self, transform=[],  method='pearson', scaling='',
                                                feature_range=(0, 1), ax_list=[], **kwargs):
        """ Compute and plot the correlation (Pearson or Spearman) between selected features and reaction times (if
        defined).

        Parameters
        ----------
        transform : lambda
            Transform applied to the selected features. Use lambdas (e.g. : ``lambda x: np.exp(x)``)
        method : str (default 'pearson')
            Correlation method to use. Can be ``'pearson'`` or ``'spearman'``
        scaling : str | None (default: none)
            Scaling method. Can be :

            * 'standardization' : compute the z-score
            * 'normalization' : normalization betwenn 0 and 1 (default) or set by ``feature_range``
            * 'robust' : robust z-score
        feature_range : tuple | array (default: (0, 1))
            Feature range when scaling is set to normalization
        ax_list : list | None (default: none)
            Axis list where to plot the results
        kwargs : Optional
            Feature selection parameters. Can be ``feature_pos``, ``feature_type``, ``feature_channame``,
            ``label_keys``, ``time_points``.
            If ``label_keys`` is not specified, will be set to 1 (hits). See :func:`get_data`.

        Returns
        -------
        correlation : array
            Correlation array between the feature and the reaction times
        feature_pos : array
            Position of the selected features

        """
        ax_list = np.array(ax_list)
        if scaling and scaling.lower() not in ['standardization', 'normalization', 'robust']:
            raise ValueError('scaling possible values are : standardization, normalization or robust')
        if method.lower() not in ['pearson', 'spearman']:
            raise ValueError('method possible values are : pearson or spearman')
        if self.reaction_times.size == 0:
            print('No reaction time variable')
            return []
        if 'label_keys' not in kwargs.keys():
            kwargs['label_keys'] = 1
        else:
            if not kwargs['label_keys'] == 1:
                print('If specified label_key in **kwargs, must be equal to 1 [Hits]')
                kwargs['label_keys'] = 1
        data_sel, feature_pos, _, _ = self.get_data(**kwargs)
        feature_names = self.feature_names[feature_pos]
        n_features, n_pnts, n_trials = data_sel.shape
        correlation, p_values = np.zeros((n_features, n_pnts)), np.zeros((n_features, n_pnts))
        trial_pos = (self.labels == self.get_label_key_from_value('Hits'))
        hits_reaction_times = self.reaction_times[trial_pos]
        for t in range(0, n_pnts):
            data_t = data_sel[:, t, :].squeeze() if n_features > 1 else data_sel[:, t, :]
            data_scaled_t = scale_data(data_t.T, scaling, feature_range=feature_range).T if scaling else data_t
            for i in range(0, n_features):
                correlation[i, t], p_values[i, t] = correlation_1d(data_scaled_t[i, :], hits_reaction_times,
                                                                   method=method, transform_x=transform)
        if not ax_list.size == 2:
            plt.figure()
            ax = plt.subplot2grid((4, 1), loc=(0, 0), rowspan=3)
            ax2 = plt.subplot2grid((4, 1), loc=(3, 0), rowspan=1, sharex=ax, sharey=ax)
        else:
            ax, ax2 = ax_list[0], ax_list[1]
        plt.set_cmap('viridis')
        im = ax.imshow(np.abs(correlation), aspect='auto', origin='lower', extent=(self.tmin, self.tmax, 0, n_features))
        plt.colorbar(im, ax=ax)
        ax.plot([0, 0], ax.get_ylim(), 'k')
        ax.yaxis.set_ticks(np.arange(0.5, n_features+0.5, 1))
        ax.yaxis.set_ticklabels(feature_names[np.arange(0, n_features, 1)])
        ax.grid(False)
        if not scaling:
            scaling = 'None'
        if transform:
            transform_str = re.search('lambda .+,', inspect.getsource(transform))
            if not transform_str:
                transform_str = re.search('lambda .+\)', inspect.getsource(transform))
            transform_str = transform_str.group()[7:-1]
            titre = 'Absolute {} correlation between transformed feature [{}] and reaction ' \
                    'times - Scaling : {}'.format(method.capitalize(), transform_str, scaling)
            ax.set(ylabel='Feature name', title=titre)
        else:
            ax.set(ylabel='Feature name', title='Absolute {} correlation between feature and reaction times - '
                                                'Scaling : {}'.format(method.capitalize(), scaling))

        plt.set_cmap('viridis')
        p_values_binned = np.ceil(np.log10(p_values))
        im_pval = ax2.imshow(p_values_binned, aspect='auto', origin='lower', extent=(self.tmin, self.tmax, 0,
                                                                                     n_features))
        plt.colorbar(im_pval, ax=ax2)
        ax2.plot([0, 0], ax2.get_ylim(), 'k')
        ax2.yaxis.set_ticks(np.arange(0.5, n_features+0.5, 1))
        ax2.yaxis.set_ticklabels(feature_names[np.arange(0, n_features, 1)])
        ax2.grid(False)
        ax2.set(xlabel='Time (s)', ylabel='Feature name', title=r'$\log_{10}(p\_val)$')
        return correlation, feature_pos

    def get_label_key_from_value(self, dict_value):
        """ Return the label key from the value of the dictionnary attribute `label_dict`

        Parameters
        ----------
        dict_value : str
            Dictionnary value

        Returns
        -------
        dict_key : int
            Corresponding dictionnary key
        """
        return list(self.label_dict.keys())[list(self.label_dict.values()).index(dict_value)]

    def sample2time(self, sample):
        """ Return the time in seconds from the sample index

        Parameters
        ----------
        sample : int
            Sample index

        Returns
        -------
        t_sec : float
            Corresponding time in seconds
        """
        if sample < 0:
            raise ValueError('sample argment cannot be inferior to 0')
        return np.array(sample) / self.srate + self.tmin

    def time2sample(self, t_sec):
        """ Return the sample index from the time in seconds

        Parameters
        ----------
        t_sec : float
            Time in seconds

        Returns
        -------
        sample : int
            Correponding sample index

        """
        if t_sec < self.tmin or t_sec > self.tmax:
            print('t_sec argument must range between tmin and tmax [{}, {}]'.format(self.tmin, self.tmax))
            t_sec = self.tmin if t_sec < self.tmin else self.tmax
        return int(np.round(np.array(t_sec - self.tmin) * self.srate))

    def feature_name2pos(self, feature_sel_names):
        """ Return the feature names from the feature positions

        Parameters
        ----------
        feature_sel_names : str | array
            Feature names

        Returns
        -------
        feature_pos : int | array
            Corresponding feature positions

        """
        feature_sel_names = np.array(feature_sel_names)
        if feature_sel_names.size == 1:
            feature_sel_names = np.array([feature_sel_names])
        feature_pos = []
        for name in feature_sel_names:
            if name not in self.feature_names:
                print('Feature {} does not exist'.format(name))
            else:
                feature_pos.append(np.where(self.feature_names == name))
        feature_pos = np.atleast_1d(feature_pos).squeeze()
        return feature_pos

    def feature_pos2name(self, feature_pos):
        """ Return the feature position from the feature names

        Parameters
        ----------
        feature_pos : int | array
            Feature position

        Returns
        -------
        feature_name : str | array
            Corresponding feature names

        """
        return self.feature_names[np.array(feature_pos)]

    def interactive_feature_rt_correlation(self, transform=[],  method='pearson', scaling=[], feature_range=(0, 1),
                                           **kwargs):
        """ Interactive plot of the correlation between feature and reaction times (for hits trials)

        Parameters
        ----------
        transform : lambda
            Transform applied to the selected features. Use lambdas (e.g. : ``lambda x: np.exp(x)``)
        method : str
            Correlation method to use. Can be ``'pearson'`` or ``'spearman'``
        scaling : str | None (default: none)
            Scaling method. Can be :

            * 'standardization' : compute the z-score
            * 'normalization' : normalization betwenn 0 and 1 (default) or set by ``feature_range``
            * 'robust' : robust z-score
        feature_range : tuple | array (default: (0, 1))
            Feature range when scaling is set to normalization
        kwargs : Optional
            Feature selection parameters. Can be ``feature_pos``, ``feature_type``, ``feature_channame``,
            ``label_keys``, ``time_points``.
            If ``label_keys`` is not specified, will be set to 1 (hits). See :func:`get_data`.
        """
        f = plt.figure()
        ax1 = plt.subplot2grid((8, 8), loc=(0, 0), rowspan=5, colspan=4)
        ax2 = plt.subplot2grid((8, 8), loc=(5, 0), rowspan=3, colspan=4, sharex=ax1, sharey=ax1)
        if 'time_points' in kwargs.keys():
            kwargs['time_points'] = []
        corr_mat, feature_pos = self.compute_correlation_hits_reaction_times(transform, method, scaling,
                                                                             ax_list=[ax1, ax2], **kwargs)
        ax3 = plt.subplot2grid((8, 8), loc=(1, 4), rowspan=5, colspan=3)
        ax4 = plt.subplot2grid((8, 8), loc=(6, 4), rowspan=1, colspan=3, sharex=ax3)
        ax5 = plt.subplot2grid((8, 8), loc=(1, 7), rowspan=5, colspan=1, sharey=ax3)
        _, feature_pos_sel, _, _ = self.get_data(**kwargs)
        self.plot_feature_hits_reaction_time(ax_list=[ax3, ax4, ax5], transform=transform, scaling=scaling,
                                             feature_range=feature_range, feature_pos=feature_pos_sel[0], time_points=0)

        def onclick(event):
            if event.inaxes == ax1 or event.inaxes == ax2:
                time_sample, feature_sel_pos = int(self.time2sample(event.xdata)), int(event.ydata)
                ax3.clear()
                ax4.clear()
                ax5.clear()
                map(Axes.clear, [ax3, ax4, ax5])
                feature_pos_click = feature_pos[feature_sel_pos]
                self.plot_feature_hits_reaction_time(scaling=scaling, feature_range=feature_range, transform=transform,
                                                     ax_list=[ax3, ax4, ax5], feature_pos=feature_pos_click,
                                                     time_points=time_sample)
                                                    # corr_val=corr_mat[feature_sel_pos, time_sample])
                ax3.figure.canvas.draw()
        f.canvas.mpl_connect('button_press_event', onclick)


def load_time_features(filepath):
    """ Load a time feature instance previously saved using the pickle module

    Parameters
    ----------
    filepath : str
        Path of the pickle file

    Returns
    -------
    time_features : TimeFeatures
        The TimeFeatures instance
    """
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def scale_data(data, scaling, feature_range=(0, 1)):
    """ Scale the data using scikit-learn scalers

    Parameters
    ----------
    data : array
        data to scale
    scaling : str | None (default: none)
        Scaling method. Can be :

        * 'standardization' : compute the z-score
        * 'normalization' : normalization betwenn 0 and 1 (default) or set by ``feature_range``
        * 'robust' : robust z-score
    feature_range : tuple | array (default: (0, 1))
        Feature range when scaling is set to normalization


    Returns
    -------
    scaled_data : array
        Scaled data

    """
    data = np.array(data)
    if data.ndim == 1:
        data = np.array([data]).T
    if scaling == 'standardization':
        scaler = StandardScaler()
    elif scaling == 'normalization':
        scaler = MinMaxScaler(feature_range=feature_range)
    elif scaling == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError('scaling possible values are : standardization, normalization or robust')
    return scaler.fit_transform(data)


def correlation_1d(vect_x, vect_y, method='pearson', transform_x=[], transform_y=[]):
    """ Correlation measure between 1D arrays

    Parameters
    ----------
    vect_x : array
        First array
    vect_y : array
        Second array
    method : str (default: 'pearson')
        Correlation methods -  Can be ``'pearson'`` or ``'spearman'``
    transform_x : lambda
        Transformation for first array. Use lambda, e.g. : ``lambda x: x**2``
    transform_y : lambda
        Transformation for second array. Use lambda, e.g. : ``lambda x: x**2``

    Returns
    -------
    corr_value : float
        Correlation value
    p_value : float
        Associated p-value

    """
    vect_x = transform_x(vect_x) if transform_x else vect_x
    vect_y = transform_x(vect_y) if transform_y else vect_y
    if method == 'pearson':
        (corr, pval) = pearsonr(vect_x, vect_y)
    elif method == 'spearman':
        (corr, pval) = spearmanr(vect_x, vect_y)
    else:
        raise ValueError('Argument method must be either pearson or spearman')
    return corr, pval

