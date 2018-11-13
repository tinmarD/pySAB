from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.decomposition import PCA
from collections import Counter
import numpy as np


def decode_time_step(X_i, y, y_dict, clf, scaler, n_splits=5, sampler=[],
                     do_pca=False, n_components=0.95, permute_train_labels=False,
                     compute_auc=False):
    """ Run a classification of data ``X_i`` with labels ``y`` using a Stratified K-Folds cross-validation
    where ``n_splits`` is the number of folds.
    Classification pipeline, for each one of the Stratified K-Folds :
        1. If ``do_pca`` is True, first do a standardization scaling and then apply PCA and keep only the first ``n_components``
        2. Apply the ``scaler``. Fit the transform on the training set and apply it on the test set
        3. Resample the train set so that the train set is balanced between classes, using the ``sampler`` argument
        4. Train the classifier on the train set (If ``permute_train_labels`` is True, permute the training labels first)
        5. Predict labels on the test set
        6. Evaluate the performances with several scores

    See also :func:`timedecoder.TimeDecoder.decode`

    Parameters
    ----------
    X_i : 2D array [n_features, n_trials]
        Feature array
    y : array [n_trials]
        Label of each trial in ``X_i``
    y_dict : dict
        Label dictionnary
    clf : sklearn classifier
        Scikit-Learn classifier
    scaler : sklearn scaler
        Scikit-Learn scaler
    n_splits : int
        Number of folds for the Stratified K-Folds cross-validation
    sampler : sklearn sampler
        Scikit learn sampler - see :func:`timedecoder.get_sampler`
    do_pca : bool (default: False)
        If True, apply a PCA decomposition and keep only a certain amount of components given ``n_components``
    n_components : int | float
        Number of components to keep when running the PCA
        If ``0 < n_components < 1`` and ``svd_solver == 'full'``, select the
        number of components such that the amount of variance that needs to be
        explained is greater than the percentage specified by n_components.
    permute_train_labels : bool (default: False)
        If True, permute the training labels. Used for computing the chance level.
    compute_auc : bool (default: False)
        If True, will try to compute the Area Under the Curve score

    Returns
    -------
    accuracy_mean : float
        Accuracy, mean over the K-folds
    f1_mean : float
        F1-score, mean over the K-folds
    precision_mean : float
        Precision, mean over the K-folds
    recall_mean : float
        Recall, mean over the K-folds
    class_acc_mean : array [n_classes]
        Class-accuracy for each class, mean over the K-folds
    auc_mean : float
        Area under the curve, mean over the K-folds
    """
    if X_i.ndim != 2:
        raise ValueError('X_i argument must have 2 dimensions [n_features, n_trials]')
    # Check there are only 2 classes
    y_unique = np.unique(y)
    n_classes = y_unique.size
    if not n_classes == 2:
        raise ValueError('Decoding only works with 2 classes')
    if sampler and hasattr(clf, 'class_weight') and clf.class_weight is not None:
        raise ValueError('Cannot combine class_weight with resampling methods')
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
    # Get train and test sets
    precision, recall, f1, accuracy, auc = np.zeros((5, n_splits))
    class_acc = np.zeros((n_splits, n_classes))
    imbalanced_class = False
    if do_pca:
        pca = PCA(n_components=n_components)
    for i_split, (train_idx, test_idx) in enumerate(skf.split(X_i, y)):
        X_train, X_test = X_i[train_idx, :], X_i[test_idx, :]
        y_train, y_test = y[train_idx], y[test_idx]
        if permute_train_labels:
            y_train = np.random.permutation(y_train)
        # If PCA, do standardization scaling and PCA
        if do_pca:
            std_scaler = StandardScaler()
            X_train = std_scaler.fit_transform(X_train)
            X_test = std_scaler.transform(X_test)
            X_train = pca.fit_transform(X_train)
            X_test = pca.transform(X_test)
        # Scale train set
        X_train_scaled = scaler.fit_transform(X_train, y_train)
        X_test_scaled = scaler.transform(X_test)
        # Resample X_train so that the train set is balanced between classes and fit the model
        if sampler:
            X_train_scaled_re, y_train_re = sampler.fit_sample(X_train_scaled, y_train)
        else:
            X_train_scaled_re, y_train_re = X_train_scaled, y_train
        if not imbalanced_class and np.unique(list(Counter(y_train_re).values())).size > 1:
            imbalanced_class = True
        clf.fit(X_train_scaled_re, y_train_re)
        # Test the model on the test set
        y_test_predict = clf.predict(X_test_scaled)
        accuracy[i_split] = accuracy_score(y_test, y_test_predict)
        class_acc[i_split] = [np.sum((y_test == y_i) & (y_test_predict == y_i)) / np.sum(y_test == y_i) for y_i in y_unique]
        try:
            f1[i_split] = f1_score(y_test, y_test_predict)
            recall[i_split] = recall_score(y_test, y_test_predict)
            precision[i_split] = precision_score(y_test, y_test_predict)
        except:
            print('Could not compute f1 or recall or precision score')
        if compute_auc:
            y_prob = clf.predict_proba(X_test_scaled)
            auc[i_split] = roc_auc_score(y_test == clf.classes_[0], y_prob[:, 0])
    return accuracy.mean(), f1.mean(), precision.mean(), recall.mean(), class_acc.mean(0), auc.mean(0)


def temporal_generalization_step(X, i_time, y, y_dict, clf, scaler, n_splits=5, sampler=[], do_pca=False,
                                 n_components=0.95, permute_train_labels=False):
    """ Temporal generalization step.
    Train at time ``i_time`` and test for all time points. Use a stratified K-folds cross validation.

    Pipeline, for each stratified K-folds:
      1. If ``do_pca`` is True, first do a standardization scaling and then apply PCA and keep only the first ``n_components`` on the train set
      2. Apply the ``scaler``. Fit the transform on the training set
      3. Resample the train set so that the train set is balanced between classes, using the ``sampler`` argument
      4. Train the classifier on the train set (If ``permute_train_labels`` is True, permute the training labels first)
      5. For each time point :
        * If ``do_pca`` is True, do a standardization scaling and then apply PCA on the test set
        * Apply the ``scaler`` on the test set
        * Predict labels on the test set
        * Evaluate the performances with several scores

    See also :func:`timedecoder.TimeDecoder.temporal_generalization`

    Parameters
    ----------
    X : 3D array [n_features, n_pnts, n_trials]
        Temporal feature array
    i_time : int
        Training time point
    y : array [n_trials]
        Label array
    y_dict : dict
        Label dictionnary
    clf : sklearn classifier
        Scikit-Learn classifier
    scaler : sklearn scaler
        Scikit-Learn scaler
    n_splits : int
        Number of folds for the Stratified K-Folds cross-validation
    sampler : sklearn sampler
        Scikit learn sampler - see :func:`timedecoder.get_sampler`
    do_pca : bool (default: False)
        If True, apply a PCA decomposition and keep only a certain amount of components given ``n_components``
    n_components : int | float
        Number of components to keep when running the PCA
        If ``0 < n_components < 1`` and ``svd_solver == 'full'``, select the
        number of components such that the amount of variance that needs to be
        explained is greater than the percentage specified by n_components.
    permute_train_labels : bool (default: False)
        If True, permute the training labels. Used for computing the chance level.

    Returns
    -------
    accuracy_mean : float
        Accuracy, mean over the K-folds
    class_acc_mean : array [n_classes]
        Class-accuracy for each class, mean over the K-folds
    """
    (n_features, n_pnts, n_trials) = X.shape
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
    # Get train and test sets
    accuracy, auc = np.zeros((2, n_splits, n_pnts))
    X_i = np.atleast_2d(X[:, i_time, :].squeeze()).T[:, :]
    if do_pca:
        pca = PCA(n_components=n_components)
    for i_split, (train_idx, test_idx) in enumerate(skf.split(X_i, y)):
        # --- TRAINING of the model at time i ---
        X_train, y_train, y_test = X_i[train_idx, :], y[train_idx], y[test_idx]
        if permute_train_labels:
            y_train = np.random.permutation(y_train)
        # If PCA, do standardization scaling and PCA
        if do_pca:
            std_scaler = StandardScaler()
            X_train = std_scaler.fit_transform(X_train)
            X_train = pca.fit_transform(X_train)
        # Scaling
        X_train_scaled = scaler.fit_transform(X_train, y_train)
        # Resample X_train so that the train set is balanced between classes and fit the model
        if sampler:
            X_train_scaled, y_train = sampler.fit_sample(X_train_scaled, y_train)
        clf.fit(X_train_scaled, y_train)
        # --- TEST for each time ----
        for j_time in range(n_pnts):
            X_test_j = np.atleast_2d(X[:, j_time, test_idx].squeeze()).T[:, :]
            # If PCA, do standardization scaling and PCA
            if do_pca:
                X_test_j = std_scaler.transform(X_test_j)
                X_test_j = pca.transform(X_test_j)
            # Scale train set
            X_test_scaled_j = scaler.transform(X_test_j)
            # Test the model on the test set
            y_test_predict_j = clf.predict(X_test_scaled_j)
            y_prob_j = clf.predict_proba(X_test_scaled_j)
            accuracy[i_split, j_time] = accuracy_score(y_test, y_test_predict_j)
            auc[i_split, j_time] = roc_auc_score(y_test == clf.classes_[0], y_prob_j[:, 0])
    return accuracy.mean(0), auc.mean(0)

