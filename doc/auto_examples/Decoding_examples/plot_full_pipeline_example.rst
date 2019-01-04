

.. _sphx_glr_auto_examples_Decoding_examples_plot_full_pipeline_example.py:


============================
From SAB Dataset to Decoding
============================

This example shows how to use the different modules of SAB_main from the creation of the SAB Dataset to decoding EEG




.. code-block:: python

    import sab_dataset
    from timedecoder import *
    from sklearn import svm
    import os
    import seaborn as sns
    sns.set()
    sns.set_context('paper')


    sab_dataset_dirpath = os.path.join('pySAB', 'sample_data') if os.path.isdir('pySAB') else os.path.join('..', '..', 'pySAB', 'sample_data')
    sab_dataset_filename = 'sab_dataset_name.p'

    load_dataset = 1






Dataset creation from the matlab dataset
This first step create the python dataset of the SAB data from the matlab dataset. All is needed is the path towards
the matlab dataset directory and the id of the subject.
The matlab directory must contain the variables EEGrec.mat, hits.mat, correctRejects.mat, falseAlarms.mat,
omissions.mat, reactionTimes.mat



.. code-block:: python


    if not load_dataset:
        sab_dataset = sab_dataset.SabDataset(sab_dataset_dirpath, 'subject_id')
        ###########################################
        # If you want to specify colors for the different conditions, use the color_dict argument. It must be a dictionnary
        # with the condition names as keys and the colors as values. For instance :
        # sab_dataset = sab_dataset.SabDataset('path_to_the_matlab_directory', 'subject_id',
        #                          colors_dict={'hits': 'g', 'cr': 'r', 'omission': 'y', 'fa': 'm'})
        # To save the newly created dataset use the save method which takes 2 arguments : the directory where the dataset
        # will be saved and the filename. By default the filename will be similar to sab_dataset_rec_001AA_150218_1542.p
        sab_dataset.save('path_to_save_the_dataset', 'sab_dataset_name.p')
    else:
        ###########################################
        # If you have previously created the dataset, you can load it using the load_sab_dataset function. The function
        # takes as argument the path of the file
        sab_dataset = sab_dataset.load_sab_dataset(os.path.join(sab_dataset_dirpath, sab_dataset_filename))







You can print the informations of the dataset using the print function



.. code-block:: python

    print(sab_dataset)





.. rst-class:: sphx-glr-script-out

 Out::

    SAB dataset REC - subject_id
    112 channels, 1434 points [-0.10, 0.60s], sampling rate 2048 Hz
    540 trials : 183 hits, 173 correct rejects, 87 omissions, 97 false alarms
    Channel Info : 112 channels and 17 electrodes
    108 EEG channels - 4 non-EEG channels
    13 EEG electrodes - 4 non-EEG electrodes


If needed you can downsample the data using the downsample method.



.. code-block:: python

    help(sab_dataset.downsample)

    # Downsample the dataset to 256Hz
    sab_dataset.downsample(8)





.. rst-class:: sphx-glr-script-out

 Out::

    Help on method downsample in module sab_dataset:

    downsample(decimate_order) method of sab_dataset.SabDataset instance
        Downsample the data along the time axis.
    
        Parameters
        ----------
        decimate_order : int
            Order of the down-sampling. Sampling frequency will be divided by this number

    New sampling rate is 256.0


TimeFeature Creation from the sab_dataset
For this step you just need to call the create_features method of SabDataset class which returns a TimeFeature instance



.. code-block:: python

    time_features = sab_dataset.create_features()







It is possible to select some channels, electrodes or trials using respectively the chan_sel, electrode_sel or trial_sel
arguments. See the help for more details :



.. code-block:: python

    help(sab_dataset.create_features)





.. rst-class:: sphx-glr-script-out

 Out::

    Help on method create_features in module sab_dataset:

    create_features(chan_sel=[], electrode_sel=[], trial_sel=[]) method of sab_dataset.SabDataset instance
        Create a TimeFeatures instance from the current dataset
    
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


You can check the time_features information using again the print function



.. code-block:: python

    print(time_features)





.. rst-class:: sphx-glr-script-out

 Out::

    Time Features subject_id_rec - 108 features, 180 time points, 540 trials
    4 labels : {1: 'Hits', 2: 'Correct rejects', 3: 'Omissions', 4: 'False Alarms'}
    Feature types : Amp


TimeDecoder Class.
In this step, the time_decoder instance is created from the TimeDecoder class. This class allow to run classification on
the time_features data.
First we need to created a classifier to the time_decoder instance. It can be Support Vector Machine, LDA, ... as long
as it support 2-classes classification with fit and predict methods
In this example, we use a SVM classifier, with C parameter set to 1



.. code-block:: python

    svm_clf = svm.SVC(kernel='linear', probability=True)

    # We can now create the time_decoder instance.
    time_decoder = TimeDecoder(svm_clf)







Decoding.
We can now start to classify the EEG data (stored in time_features), given 2 conditions (e.g. 'hits' and 'Correct rejects')
by using the decode method.



.. code-block:: python

    data, _, labels, _ = time_features.get_data(label_keys=[1, 2])
    # label_keys 1, 2 are the labels used for hits and correct_rejects respectively. You can see it with the label_dict argument
    print(time_features.label_dict)
    # Run the decoding using multiples processors :
    scores = time_decoder.decode_mpver(data, labels, time_features.label_dict)




.. image:: /auto_examples/Decoding_examples/images/sphx_glr_plot_full_pipeline_example_001.png
    :align: center


.. rst-class:: sphx-glr-script-out

 Out::

    {1: 'Hits', 2: 'Correct rejects', 3: 'Omissions', 4: 'False Alarms'}
    183 Hits - 173 Correct rejects
    Function : decode_mpver - Time elapsed : 32.78383207321167


You can select data using the arguments of get_data method (see help(time_features.get_data)).
The possible selection arguments are feature_pos, feature_type, feature_channame, label_keys and time_points.
The current time_features instance contains the following channels :



.. code-block:: python

    print(time_features.channel_names)





.. rst-class:: sphx-glr-script-out

 Out::

    ["EEG TP'1-TP'2" "EEG TP'2-TP'3" "EEG TP'3-TP'4" "EEG TP'4-TP'5"
     "EEG TP'5-TP'6" "EEG TP'6-TP'7" "EEG TP'7-TP'8" "EEG TP'8-TP'9"
     "EEG TP'9-TP'10" "EEG A'1-A'2" "EEG A'2-A'3" "EEG A'4-A'5" "EEG A'5-A'6"
     "EEG B'1-B'2" "EEG B'2-B'3" "EEG B'4-B'5" "EEG B'5-B'6" "EEG C'1-C'2"
     "EEG C'2-C'3" "EEG C'3-C'4" "EEG C'4-C'5" "EEG C'5-C'6" "EEG C'6-C'7"
     "EEG C'7-C'8" "EEG C'8-C'9" "EEG C'9-C'10" "EEG C'10-C'11"
     "EEG C'11-C'12" "EEG C'12-C'13" "EEG TB'1-TB'2" "EEG TB'2-TB'3"
     "EEG TB'3-TB'4" "EEG TB'4-TB'5" "EEG TB'5-TB'6" "EEG TB'6-TB'7"
     "EEG TB'7-TB'8" "EEG TB'8-TB'9" "EEG TB'9-TB'10" "EEG TB'10-TB'11"
     "EEG TB'11-TB'12" "EEG TB'12-TB'13" "EEG TB'13-TB'14" "EEG T'1-T'2"
     "EEG T'2-T'3" "EEG T'3-T'4" "EEG T'4-T'5" "EEG T'5-T'6" "EEG T'6-T'7"
     "EEG GC'1-GC'2" "EEG GC'2-GC'3" "EEG GC'16-GC'17" "EEG GC'17-GC'18"
     "EEG GC'17-GC'18" "EEG OR'1-OR'2" "EEG OR'3-OR'4" "EEG OR'4-OR'5"
     "EEG OR'5-OR'6" "EEG OR'6-OR'7" "EEG OR'7-OR'8" "EEG OR'8-OR'9"
     "EEG OR'9-OR'10" "EEG OR'10-OR'11" "EEG OR'11-OR'12" "EEG OR'12-OR'13"
     "EEG OR'13-OR'14" "EEG OR'14-OR'15" "EEG OR'15-OR'16" "EEG OR'16-OR'17"
     'EEG TP1-TP2' 'EEG TP2-TP3' 'EEG TP3-TP4' 'EEG TP4-TP5' 'EEG TP5-TP6'
     'EEG TP6-TP7' 'EEG TP7-TP8' 'EEG TP8-TP9' 'EEG TP9-TP10' 'EEG TP10-TP11'
     'EEG A 1-A 2' 'EEG A 2-A 3' 'EEG A 4-A 5' 'EEG A 5-A 6' 'EEG C 1-C 2'
     'EEG C 2-C 3' 'EEG C 4-C 5' 'EEG C 5-C 6' 'EEG TB1-TB2' 'EEG TB2-TB3'
     'EEG TB3-TB4' 'EEG TB4-TB5' 'EEG TB5-TB6' 'EEG TB6-TB7' 'EEG TB7-TB8'
     'EEG TB8-TB9' 'EEG TB9-TB10' 'EEG TB10-TB11' 'EEG TB11-TB12'
     'EEG TB12-TB13' 'EEG TB13-TB14' 'EEG OR1-OR2' 'EEG OR2-OR3' 'EEG OR3-OR4'
     'EEG OR4-OR5' 'EEG OR5-OR6' 'EEG OR6-OR7' 'EEG OR7-OR8' 'EEG OR13-OR14'
     'EEG OR14-OR15']


If we want to select only some channel, you can specify it with the feature_channame argument :



.. code-block:: python

    data, _, labels, _ = time_features.get_data(feature_channame=['C\'1-C\'2'], label_keys=[1, 2])
    scores = time_decoder.decode_mpver(data, labels, time_features.label_dict)



.. image:: /auto_examples/Decoding_examples/images/sphx_glr_plot_full_pipeline_example_002.png
    :align: center


.. rst-class:: sphx-glr-script-out

 Out::

    183 Hits - 173 Correct rejects
    Function : decode_mpver - Time elapsed : 7.542262077331543


**Total running time of the script:** ( 1 minutes  1.463 seconds)



.. only :: html

 .. container:: sphx-glr-footer


  .. container:: sphx-glr-download

     :download:`Download Python source code: plot_full_pipeline_example.py <plot_full_pipeline_example.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: plot_full_pipeline_example.ipynb <plot_full_pipeline_example.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.readthedocs.io>`_
