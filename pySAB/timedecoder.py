import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import multiprocessing as mp
from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, NearMiss
from imblearn.combine import SMOTEENN, SMOTETomek
from functools import wraps
import _pickle
import time
import tqdm
import re
from collections import Counter
from timedecoder_utils import *


def timethis(func):
    """ Timing Wrapper"""
    @wraps(func)
    def add_timer(*args, **kwargs):
        """ Add a timer to the function"""
        t_start = time.time()
        res = func(*args, **kwargs)
        print('Function : {} - Time elapsed : {}'.format(func.__name__, time.time() - t_start))
        return res
    return add_timer


class TimeDecoder:
    """ Class representing a Decoding/MVPA/Classification pipeline
    This decoding works with 3D data having a temporal dimension.

    Attributes
    ----------
    clf : sklearn model
        Scikit learn classification model
    clf_name : str
        Classification model name
    scaler_name : str
        Scaling method. Possibles values :
          * 'standardization' : :class:`sklearn.preprocessing.StandardScaler` - Standardize features by removing the mean and scaling to unit variance
          * 'normalization' : :class:`sklearn.preprocessing.MinMaxScaler` - Transforms features by scaling each feature to a given range
          * 'robust' : :class:`sklearn.preprocessing.RobustScaler` - Scale features using statistics that are robust to outliers
    scaler : sklearn Scaler
        Scikit-Learn scaler
    do_pca : bool
        If True, will run a PCA and keep only the first components given the ``n_components`` attribute
    n_components : int, float, None or string
        Number of components to keep.
        if n_components is not set all components are kept::
            n_components == min(n_samples, n_features)
        If ``0 < n_components < 1`` and ``svd_solver == 'full'``, select the
        number of components such that the amount of variance that needs to be
        explained is greater than the percentage specified by n_components.
        If ``svd_solver == 'arpack'``, the number of components must be
        strictly less than the minimum of n_features and n_samples.
        Hence, the None case results in::
            n_components == min(n_samples, n_features) - 1

    """
    def __init__(self, classifier, scaling="normalization", do_pca=0, n_components=0.95):
        self.clf = classifier
        self.clf_name = re.search('.+\(', classifier.__str__())[0][:-1]
        self.scaler_name = scaling
        if scaling.lower() == 'standardization':
            self.scaler = StandardScaler()
        elif scaling.lower() == 'normalization':
            self.scaler = MinMaxScaler()
        elif scaling.lower() == 'robust':
            self.scaler = RobustScaler()
        if do_pca:
            self.pca = PCA(n_components=n_components)
            self.n_components = n_components
        else:
            self.pca = []
            self.n_components = []

    def __str__(self):
        desc_str = 'TimeDecoder instance - Method used : {}\n'.format(self.clf_name)
        desc_str += 'Scaling : {}\n'.format(self.scaler_name)
        if self.pca:
            if 0 < self.n_components < 1:
                desc_str += 'PCA on : keep {}% of variance\n'.format(int(100*self.n_components))
            else:
                desc_str += 'PCA on : keep {}% components\n'.format(int(self.n_components))
        else:
            desc_str += 'PCA off\n'
        return desc_str

    @timethis
    def temporal_generalization(self, X, y, y_dict, n_splits=5, n_iter=3, resample='None', smote_kind='regular',
                                near_miss_ver=1, permute_train_labels=False, n_processes=6):
        """ Temporal Generalisation.
        For each time points, t_i, t_j, train the classifier at t_i, test at time t_j

        Parameters
        ----------
        X : 3D array [n_features, n_pnts, n_trials]
            Temporal feature array
        y : array [n_trials]
            Label array
        y_dict : dict
            Label/Class dictionnary
        n_splits : int
            Number of folds for the Stratified K-Folds cross-validation
        n_iter : int
            Number of times the whole decoding pipeline is repeated
        resample : str (default: 'rus')
            Resampling method. Possible values : [``'ros'``, ``'rus'``, ``'nearmiss'``, ``'smote'``'', ``'adasyn'``'',
            ``'smoteenn'``'', ``'smotomek'``''] - See :func:`timedecoder.get_sampler`
        smote_kind : str (default: 'regular')
            The type of SMOTE algorithm to use one of the following options: ``'regular'``, ``'borderline1'``,
            ``'borderline2'``,  ``'svm'``. Only used if ``method=='SMOTE'``.
        near_miss_ver : int
            Version of the NearMiss to use. Possible values are 1, 2 or 3.
        permute_train_labels : bool (default: False)
            If True, permute the training labels. Used for computing a chance level.
        n_processes : int (default: 6)
            Number of processes to use in parralel

        Returns
        -------
        accuracy : array
            Accuracy score. Size [n_iter, n_pnts, n_pnts]
        auc : array
            Area Under the Curve score. Size [n_iter, n_pnts, n_pnts]
        """
        if not X.ndim == 3:
            raise ValueError('X argument should have 3 dimensions [n_features, n_pnts, n_trials')
        (n_features, n_pnts, n_trials) = X.shape
        if not y.size == n_trials:
            raise ValueError('Size of label vector y not consistent with the shape of X argument')
        # Check there are only 2 classes
        n_classes = np.unique(y).size
        if not n_classes == 2:
            raise ValueError('Decoding only works with 2 classes')
        n_trials_per_class = list(Counter(y).values())
        # If the number of trial per class is not the same and there is no resampling strategy : Warning
        if not np.unique(n_trials_per_class).size == 1 and resample is 'None':
            print('WARNING : the number of trials is not the same for all class')
        sampler = get_sampler(resample, smote_kind, near_miss_ver) if resample is not 'None' else []
        print_classes_composition(y, y_dict)
        accuracy, auc = np.zeros((2, n_iter, n_pnts, n_pnts))
        for i_iter in tqdm.tqdm(range(n_iter)):
            pool = mp.Pool(processes=n_processes)
            async_output = [pool.apply_async(temporal_generalization_step, args=(X, i_time, y, y_dict, self.clf, self.scaler,
                                                                                 n_splits, sampler, self.pca, self.n_components,
                                                                                 permute_train_labels))
                            for i_time in range(n_pnts)]
            async_res = [p.get() for p in async_output]

            accuracy[i_iter, :, :] = np.array([async_res[i][0] for i in range(n_pnts)])
            auc[i_iter, :, :] = np.array([async_res[i][1] for i in range(n_pnts)])
        score_map = {'accuracy': accuracy.mean(0), 'auc': auc.mean(0)}
        for score in score_map.keys():
            f, ax = plt.subplots()
            im = ax.imshow(score_map[score], aspect='auto', origin='lower')
            plt.colorbar(im)
            ax.grid(False)
            ax.set(xlabel='Generalization time', ylabel='Training Time', title=score)
        return accuracy, auc

    @timethis
    def decode(self, X, y, y_dict, n_splits=5, n_iter=1, resample='rus', smote_kind='regular',
               near_miss_ver=1, do_plot=1, score_to_plot=['auc', 'accuracy'], compute_auc=True,
               permute_train_labels=False):
        """ Do the temporal decoding of 3D data ``X`` given labels ``y``
        For each time point, run

        Parameters
        ----------
        X : 3D array [n_features, n_pnts, n_trials]
            Temporal feature array
        y : array [n_trials]
            Label array
        y_dict : dict
            Label dictionnary
        n_splits : int
            Number of folds for the Stratified K-Folds cross-validation
        n_iter : int
            Number of times the whole decoding pipeline is repeated
        resample : str (default: 'rus')
            Resampling method. Possible values : [``'ros'``, ``'rus'``, ``'nearmiss'``, ``'smote'``'', ``'adasyn'``'',
            ``'smoteenn'``'', ``'smotomek'``''] - See :func:`timedecoder.get_sampler`
        smote_kind : str (default: 'regular')
            The type of SMOTE algorithm to use one of the following options: ``'regular'``, ``'borderline1'``,
            ``'borderline2'``,  ``'svm'``. Only used if ``method=='SMOTE'``.
        near_miss_ver : int
            Version of the NearMiss to use. Possible values are 1, 2 or 3.
        do_plot : bool (default: True)
            Plot the decoding results
        score_to_plot : str | list
            The different scores to plot. Possible values :
             * 'accuracy'
             * 'precision'
             * 'recall'
             * 'f1'
             * 'class_accuracy'
             * 'auc'
        permute_train_labels : bool (default: False)
            If True, permute the training labels. Used for computing a chance level.
        compute_auc : bool (default: True)
            If True, compute the Area Under the Curve score.

        Returns
        -------
        scores : dict
            Dictionnary containing the different scores : 'accuracy', 'precision', 'recall', 'f1', 'class_accuracy'
            and 'auc'

        """
        if not X.ndim == 3:
            raise ValueError('X argument should have 3 dimensions [n_features, n_pnts, n_trials]')
        (n_features, n_pnts, n_trials) = X.shape
        if not y.size == n_trials:
            raise ValueError('Size of label vector y not consistent with the shape of X argument')
        # Check there are only 2 classes
        n_classes = np.unique(y).size
        if not n_classes == 2:
            raise ValueError('Decoding only works with 2 classes')
        n_trials_per_class = list(Counter(y).values())
        # If the number of trial per class is not the same and there is no resampling strategy : Warning
        if not np.unique(n_trials_per_class).size == 1 and resample is 'None':
            print('WARNING : the number of trials is not the same for all class')
        print_classes_composition(y, y_dict)

        sampler = get_sampler(resample, smote_kind, near_miss_ver) if resample is not 'None' else []
        precision, recall, f1, accuracy, auc = np.zeros((5, n_iter, n_pnts))
        class_acc, class_acc_i = np.zeros((n_iter, n_pnts, n_classes)), np.zeros((n_splits, n_classes))
        imbalanced_class = False
        # Decoding is repeated n_iter times
        for i_iter in tqdm.tqdm(range(n_iter)):
            for i in range(n_pnts):
                accuracy[i_iter, i], f1[i_iter, i], precision[i_iter, i], recall[i_iter, i], class_acc[i_iter, i, :],
                auc[i_iter, i] = decode_time_step(np.atleast_2d(X[:, i, :].squeeze()).T[:, :], y, y_dict, self.clf, self.scaler,
                                                  n_splits, sampler, permute_train_labels, compute_auc)
        if imbalanced_class:
            print('WARNING : Imbalanced classes were detected !!!!')
        scores = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1, 'class_accuracy': class_acc,
                  'auc': auc}

        if do_plot:
            self.plot_scores(scores, y_dict, n_iter, n_splits, n_pnts, resample, score_to_plot)

        return scores

    def plot_scores(self, scores, y_dict, n_iter, n_splits, n_pnts, resample, score_to_plot=[]):
        """ Plot the decoding performances measures listed in ``scores``

        Parameters
        ----------
        scores : dict

        y_dict : dict
            Label / Class dictionnary
        n_splits : int
            Number of folds for the Stratified K-Folds cross-validation
        n_iter : int
            Number of times the whole decoding pipeline is repeated
        n_pnts : int
            Number of time points
        resample : str
            Name of the resampling method used for the decoding
        score_to_plot : list
            Scores to plot

        """
        if not score_to_plot:
            score_to_plot = list(scores.keys())
        f = plt.figure()
        ax = f.add_subplot(111)
        legend_str = []
        for score_i in score_to_plot:
            score_mean, score_std = scores[score_i].mean(0), scores[score_i].std(0)
            if score_i is 'class_accuracy':
                for i, y_i in enumerate(y_dict.keys()):
                    ax.plot(score_mean[:, i], lw=1)
                    legend_str.append('{} - {}'.format(score_i, y_dict[y_i]))
            else:
                ax.plot(score_mean)
                legend_str.append('{} mean'.format(score_i))
                if n_iter > 1 and len(score_to_plot) == 1:
                    ax.fill_between(range(n_pnts), score_mean+score_std, score_mean-score_std, alpha=0.7)
                    legend_str.append('{} Std'.format(score_i))
            plt.autoscale(axis='x', tight=True)
        plt.legend(legend_str)
        titre = 'Decoding  with {} - {} outer folds - Resampling method : {} - {} iterations'.format(
                 self.clf_name, n_splits, resample, n_iter)
        ax.set(title=titre, ylabel='Decoding', xlabel='Sample')


    @timethis
    def decode_mpver(self, X, y, y_dict, n_splits=5, n_iter=3, resample='rus', smote_kind='regular',
                     near_miss_ver=1, do_plot=1, score_to_plot=['auc', 'accuracy'],
                     compute_auc=True, permute_train_labels=False, n_processes=6):
        """ Do the temporal decoding of 3D data ``X`` given labels ``y`` - Multiprocesses version

        Parameters
        ----------
        X : 3D array [n_features, n_pnts, n_trials]
            Temporal feature array
        y : array [n_trials]
            Label array
        y_dict : dict
            Label/Class dictionnary
        n_splits : int
            Number of folds for the Stratified K-Folds cross-validation
        n_iter : int
            Number of times the whole decoding pipeline is repeated
        resample : str (default: 'rus')
            Resampling method. Possible values : [``'ros'``, ``'rus'``, ``'nearmiss'``, ``'smote'``'', ``'adasyn'``'',
            ``'smoteenn'``'', ``'smotomek'``''] - See :func:`timedecoder.get_sampler`
        smote_kind : str (default: 'regular')
            The type of SMOTE algorithm to use one of the following options: ``'regular'``, ``'borderline1'``,
            ``'borderline2'``,  ``'svm'``. Only used if ``method=='SMOTE'``.
        near_miss_ver : int
            Version of the NearMiss to use. Possible values are 1, 2 or 3.
        do_plot : bool (default: True)
            Plot the decoding results
        score_to_plot : str | list
            The different scores to plot. Possible values :
             * 'accuracy'
             * 'precision'
             * 'recall'
             * 'f1'
             * 'class_accuracy'
             * 'auc'
        compute_auc : bool (default: True)
            If True, compute the Area Under the Curve score.
        permute_train_labels : bool (default: False)
            If True, permute the training labels. Used for computing a chance level.
        n_processes : int (default: 6)
            Number of processes to use in parralel

        Returns
        -------
        scores : dict
            Dictionnary containing the different scores : 'accuracy', 'precision', 'recall', 'f1', 'class_accuracy'
            and 'auc'

        """
        if not X.ndim == 3:
            raise ValueError('X argument should have 3 dimensions [n_features, n_pnts, n_trials')
        (n_features, n_pnts, n_trials) = X.shape
        if not y.size == n_trials:
            raise ValueError('Size of label vector y not consistent with the shape of X argument')
        # Check there are only 2 classes
        n_classes = np.unique(y).size
        if not n_classes == 2:
            raise ValueError('Decoding only works with 2 classes')
        n_trials_per_class = list(Counter(y).values())
        # If the number of trial per class is not the same and there is no resampling strategy : Warning
        if not np.unique(n_trials_per_class).size == 1 and resample is 'None':
            print('WARNING : the number of trials is not the same for all class')
        print_classes_composition(y, y_dict)

        sampler = get_sampler(resample, smote_kind, near_miss_ver) if resample is not 'None' else []
        precision, recall, f1, accuracy, auc = np.zeros((5, n_iter, n_pnts))
        class_acc, class_acc_i = np.zeros((n_iter, n_pnts, n_classes)), np.zeros((n_splits, n_classes))
        imbalanced_class = False
        pool = mp.Pool(processes=n_processes)
        # Decoding is repeated n_iter times
        for i_iter in tqdm.tqdm(range(n_iter)):
            # For each time point
            async_output = [pool.apply_async(decode_time_step, args=(np.atleast_2d(X[:, i, :].squeeze()).T[:, :], y, y_dict, self.clf,
                                                                     self.scaler, n_splits, sampler, self.pca,
                                                                     self.n_components, permute_train_labels,
                                                                     compute_auc))
                            for i in range(n_pnts)]
            async_res = [p.get() for p in async_output]
            (accuracy[i_iter], f1[i_iter], precision[i_iter], recall[i_iter], class_acc[i_iter], auc[i_iter]) = \
                [[async_res[i][j] for i in range(n_pnts)] for j in range(6)]
            if imbalanced_class:
                print('WARNING : Imbalanced classes were detected !!!!')
        scores = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1, 'class_accuracy': class_acc,
                  'auc': auc}
        if do_plot:
            self.plot_scores(scores, y_dict, n_iter, n_splits, n_pnts, resample, score_to_plot)

        return scores

    def single_feature_decoding(self, X, y, y_dict, n_splits=5, n_iter=3, resample='Rus', smote_kind='regular',
                                near_miss_ver=1, do_plot=1, score_to_plot=['auc', 'accuracy'], compute_auc=True,
                                permute_train_labels=False, n_processes=6, features_names=[]):
        """ Apply the classifier on each feature separately. Plot the classification results as an image.

        Parameters
        ----------
        X : 3D array [n_features, n_pnts, n_trials]
            Temporal feature array
        y : array [n_trials]
            Label array
        y_dict : dict
            Label dictionnary
        n_splits : int
            Number of folds for the Stratified K-Folds cross-validation
        n_iter : int
            Number of times the whole decoding pipeline is repeated
        resample : str (default: 'rus')
            Resampling method. Possible values : [``'ros'``, ``'rus'``, ``'nearmiss'``, ``'smote'``'', ``'adasyn'``'',
            ``'smoteenn'``'', ``'smotomek'``''] - See :func:`timedecoder.get_sampler`
        smote_kind : str (default: 'regular')
            The type of SMOTE algorithm to use one of the following options: ``'regular'``, ``'borderline1'``,
            ``'borderline2'``,  ``'svm'``. Only used if ``method=='SMOTE'``.
        near_miss_ver : int
            Version of the NearMiss to use. Possible values are 1, 2 or 3.
        do_plot : bool (default: True)
            Plot the decoding results
        score_to_plot : str | list
            The different scores to plot. Possible values :
             * 'accuracy'
             * 'precision'
             * 'recall'
             * 'f1'
             * 'class_accuracy'
             * 'auc'
        compute_auc : bool (default: True)
            If True, compute the Area Under the Curve score.
        permute_train_labels : bool (default: False)
            If True, permute the training labels. Used for computing a chance level.
        n_processes : int (default: 6)
            Number of processes to use in parralel
        features_names : list | None
            Name of each feature, used for the results figure.

        Returns
        -------
        scores : dict
            Dictionnary containing the different scores : 'accuracy', 'precision', 'recall', 'f1', 'class_accuracy'
            and 'auc'

        """
        if not X.ndim == 3:
            raise ValueError('X argument should have 3 dimensions [n_features, n_pnts, n_trials')
        (n_features, n_pnts, n_trials) = X.shape
        if not y.size == n_trials:
            raise ValueError('Size of label vector y not consistent with the shape of X argument')
        n_classes = len(y_dict)
        precision, recall, f1, accuracy, auc = np.zeros((5, n_features, n_pnts))
        class_acc = np.zeros((n_features, n_pnts, n_classes))
        for i in tqdm.tqdm(range(n_features)):
            X_i = np.moveaxis(np.atleast_3d(X[i, :, :]), 2, 0)
            scores_i = self.decode_mpver(X_i, y, y_dict, n_splits, n_iter, resample, smote_kind, near_miss_ver,
                                         permute_train_labels, n_processes, 0, [], compute_auc)
            precision[i, :], recall[i, :] = scores_i['precision'].mean(0), scores_i['recall'].mean(0)
            f1[i, :], accuracy[i, :] = scores_i['f1'].mean(0), scores_i['accuracy'].mean(0)
            auc[i, :], class_acc[i, :, :] = scores_i['auc'].mean(0), scores_i['class_accuracy'].mean(0)
        scores = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1, 'class_accuracy': class_acc,
                  'auc': auc}
        # Plot
        if do_plot:
            for score in score_to_plot:
                f, ax = plt.subplots()
                im = ax.imshow(scores[score], aspect='auto', origin='lower')
                plt.colorbar(im)
                ax.set_yticks(np.arange(n_features))
                ax.set_yticklabels(features_names)
                ax.grid(False)
                ax.set(xlabel='time', ylabel='Feature', title='Decoding - {}'.format(score))
        f, ax = plt.subplots()
        for score in score_to_plot:
            ax.plot(scores[score].max(0))
        ax.legend(score_to_plot)
        ax.autoscale(axis='x', tight=True)

        return scores


def apply_pca(X, n_components=0.95):
    """ Not used ? - Should not be here
    """
    if not X.ndim == 3:
        raise ValueError('X argument should have 3 dimensions [n_features, n_pnts, n_trials')
    (n_features, n_pnts, n_trials) = X.shape
    std_scaler = StandardScaler()
    pca = PCA(n_components)
    feature_var_explained = np.zeros((n_features, n_pnts))
    # For each time point
    for i in tqdm.tqdm(range(n_pnts)):
        X_i = X[:, i, :].squeeze().T[:, :]
        # Scale data (standardization)
        X_scaled = std_scaler.fit_transform(X_i)
        pca.fit_transform(X_scaled)
        feature_var_explained[:, i] = np.matmul(pca.explained_variance_, np.abs(pca.components_))
    # Plot results
    f = plt.figure()
    ax = f.add_subplot(111)
    im = ax.imshow(feature_var_explained, origin='lower', aspect='auto', extent=(0, n_pnts, 0, n_features))
    plt.colorbar(im)
    return feature_var_explained


def print_classes_composition(y, y_dict):
    """ Print the number of trials for the different classes

    Parameters
    ----------
    y : array
        Trial labels
    y_dict : dict
        Dictionnary containing the name of each class

    """
    classes_composition_str = ''
    y_unique = np.unique(y)
    for y_i in y_unique:
        classes_composition_str += '{} {} - '.format(sum(y == y_i), y_dict[y_i])
    print(classes_composition_str[:-3])


def get_sampler(method, smote_kind='regular', near_miss_ver=1):
    """ Return a resampler instance from the imbalanced-learn API
    Resampling is used when the number of trials is not the same across the 2 classes.
    For more details see https://imbalanced-learn.readthedocs.io/en/stable/user_guide.html

    Parameters
    ----------
    method :  str
        Resampling method. Possible values :
          * 'ros', 'randomoversampler' : :class:`imblearn.over_sampling.RandomOverSampler` - Object to over-sample the
            minority class(es) by picking samples at random with replacement.
          * 'rus', 'randomundersampler' : :class:`imblearn.under_sampling.RandomUnderSampler` - Under-sample the
            majority class(es) by randomly picking samples with or without replacement.
          * 'smote' : :class:`imblearn.over_sampling.RandomOverSampler` - Class to perform over-sampling using SMOTE,
            Synthetic Minority Over-sampling Technique.
          * 'adasyn' : :class:`imblearn.over_sampling.ADASYN` - Perform over-sampling using Adaptive Synthetic (ADASYN)
            sampling approach for imbalanced datasets.
          * 'nearmiss' : :class:`imblearn.over_sampling.RandomOverSampler` - Class to perform under-sampling based
            on NearMiss methods.
          * 'smoteenn' : :class:`imblearn.combine.SMOTEENN` - Combine over- and under-sampling using
            SMOTE and Edited Nearest Neighbours.
          * 'smotetomek' : :class:`imblearn.combine.SMOTETomek` - Combine over- and under-sampling
            using SMOTE and Tomek links.
    smote_kind : str (default: 'regular')
        The type of SMOTE algorithm to use one of the following options: ``'regular'``, ``'borderline1'``,
         ``'borderline2'``,  ``'svm'``. Only used if ``method=='SMOTE'``.
    near_miss_ver : int
        Version of the NearMiss to use. Possible values are 1, 2 or 3.

    Returns
    -------
    sampler : imblearn sampler
        The sampler from the imbalanced-learn API

    """
    method_low = method.lower()
    if method_low in ['ros' or 'randomoversampler']:
        sampler = RandomOverSampler()
    elif method_low == 'smote':
        if smote_kind not in ['regular', 'borderline1', 'borderline2', 'svm']:
            raise ValueError('Wrong argument smote_kind {}. Choices are (\'regular\', \'borderline1\', \'borderline2\''
                             ', \'svm\')'.format(smote_kind))
        sampler = SMOTE(kind=smote_kind)
    elif method_low == 'adasyn':
        sampler = ADASYN()
    elif method_low in ['rus', 'randomundersampler']:
        sampler = RandomUnderSampler()
    elif method_low == 'nearmiss':
        sampler = NearMiss(version=near_miss_ver)
    elif method_low == 'smoteenn':
        sampler = SMOTEENN()
    elif method_low == 'smotetomek':
        sampler = SMOTETomek()
    else:
        raise ValueError('Wrong argument for resampling method : {}'.format(method))
    return sampler

