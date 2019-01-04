====================
    API Reference
====================

Reference for the `pySAB` API

.. contents::
   :local:
   :depth: 2

SabDataset
==========

**Class representing a SAB dataset**

:py:mod:`sab_dataset.SabDataset`:


.. currentmodule:: sab_dataset.SabDataset

.. autosummary::

    __init__
    downsample
    plot_erp
    plot_electrode_erps
    plot_itpc
    create_features
    save
    save_sig_to_file


TimeFeatures
=============

**Class for the visualization of features across time from the raw amplitude data. Interface between the other classes**

:py:mod:`timefeatures.TimeFeatures`:

.. currentmodule:: timefeatures.TimeFeatures

Extract feature, order them, access to data :

.. autosummary::

    extract_feature
    get_data
    sort_features
    sort_trials
    feature_name2pos
    feature_pos2name
    get_feature_pos
    get_label_key_from_value
    sample2time
    time2sample


Correlation analysis, feature importance, clustering

.. autosummary::

    compute_feature_importance
    compute_correlation_hits_reaction_times
    compute_correlation_feature_target
    interactive_feature_rt_correlation
    cluster_data

Visualization

.. autosummary::

    plot_feature_erp
    plot_feature_erpimage
    plot_feature_distribution

Clustering internal functions

:py:mod:`sab_clustering`:

.. autosummary::

    cluster_data_2d
    get_clustering_algo


Interactive visualization of features

:py:mod:`sab_tkinterwindow.TimeFeatureWindow` : Tkinter GUI for visualizing features evolution and analysis


Feature extraction
==================

**Class for computing features from the raw amplitude**

:py:mod:`featureextracter.FeatureExtracter`

.. currentmodule:: featureextracter.FeatureExtracter

.. autosummary::

    __init__
    stft_on_data
    dwt_on_data
    cwt_on_data
    filter_hilbert_on_data
    bandpower_on_data


:py:mod:`featureextracter`

.. currentmodule:: featureextracter

.. autosummary::

    stft_1d
    dwt_1d
    cwt_1d
    bandpower_1d
    tf_scaling
    compute_band_mean


Decoding / MVPA
=================

**Class for running decoding/MVPA/Classification analyses from the time features**

:py:mod:`time_decoder.TimeDecoder`

.. currentmodule:: time_decoder.TimeDecoder

.. autosummary::

    __init__
    decode
    decode_mpver
    single_feature_decoding
    temporal_generalization
    plot_scores


:py:mod:`timedecoder_utils`

.. currentmodule:: timedecoder_utils

.. autosummary::

    decode_time_step
    temporal_generalization_step


Phase Utils functions
======================

**Various functions for extracting and analyzing the phase of iEEG signals**

:py:mod:`phase_utils`

.. currentmodule:: phase_utils

.. autosummary::

    itpc
    compute_robust_estimation
    compute_analytical_signal
    bp_filter_1d
    plot_analytical_signal
    plot_complex_tracjectory



