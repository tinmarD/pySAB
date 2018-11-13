from sklearn.cluster import KMeans, AffinityPropagation, SpectralClustering, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler
import pandas as pd


def cluster_data_2d(model, X, y_true, y_dict):
    """ Apply the clustering ``model`` on the 2D data ``X``
    Cluster trials between a fixed number of clusters, defined in the model.

    Parameters
    ----------
    model : sklearn.cluster model
        The scikit learn clustering model. Use the :func:`sab_clustering.get_clustering_algo` function to get the model.
    X : 2D array [n_trials, n_features]
        Input array to cluster
    y_true : 1D array [n_trials]
        True label of each trial
    y_dict : dict
        Dictionnary giving the label names

    Returns
    -------
    ct : pandas DataFrame
        Output cross-tabulation DataFrame
    """
    if not X.ndim == 2:
        raise ValueError('X argument should have 2 dimensions [n_trials, n_features]')
    (n_trials, n_features) = X.shape
    if not y_true.size == n_trials:
        raise ValueError('Size of label vector y not consistent with the shape of X argument')
    y_str = [y_dict[y_i] for y_i in y_true]
    # Scale the features
    std_scaler = StandardScaler()
    X_scaled = std_scaler.fit_transform(X)
    # Apply the clustering algorithm
    y_pred = model.fit_predict(X_scaled)
    df = pd.DataFrame({'y_pred': y_pred, 'Class': y_str})
    ct = pd.crosstab(df['y_pred'], df['Class'], normalize='index')
    return ct


def get_clustering_algo(algo_name, n_clusters):
    """ Return the clustering model given its name

    Parameters
    ----------
    algo_name : str
        Must be in :
            * 'kmeans'
            * 'affinitypropagation'
            * 'spectralclustering'
            * 'agglomerativeclustering'
            * 'dbscan'
    n_clusters : int
        Number of clusters

    Returns
    -------
    model : sklearn.cluster model
        The Scikit learn clustering model

    """
    algo_name = algo_name.lower().replace(" ", "")
    if algo_name in ['kmeans', 'k-means', 'kmean']:
        model = KMeans(n_clusters=n_clusters)
    elif algo_name in ['affinitypropagation', 'affinity-propagation']:
        model = AffinityPropagation()
    elif algo_name in ['spectralclustering', 'spectral-clustering']:
        model = SpectralClustering(n_clusters=n_clusters)
    elif algo_name in ['agglomerativeclustering', 'agglomerative-clustering']:
        model = AgglomerativeClustering(n_clusters=n_clusters)
    elif algo_name in ['dbscan']:
        model = DBSCAN()
    else:
        raise ValueError('Wrong algo_name argument : {}'.format(algo_name))
    return model



