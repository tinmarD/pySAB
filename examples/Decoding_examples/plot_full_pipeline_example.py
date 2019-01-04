"""
============================
From SAB Dataset to Decoding
============================

This example shows how to use the different modules of SAB_main from the creation of the SAB Dataset to decoding EEG

"""
# import matplotlib
# matplotlib.use('TkAgg')
import sab_dataset
from timedecoder import *
from sklearn import svm
import os
import seaborn as sns
sns.set()
sns.set_context('paper')


sab_dataset_dirpath = os.path.join('pySAB', 'sample_data') if os.path.isdir('pySAB') else os.path.join('..', '..', 'pySAB', 'sample_data')
sab_dataset_filename = 'sab_dataset_rec_subject_id_040119_1153.p'

load_dataset = 1
###########################################
# Dataset creation from the matlab dataset
# This first step create the python dataset of the SAB data from the matlab dataset. All is needed is the path towards
# the matlab dataset directory and the id of the subject.
# The matlab directory must contain the variables EEGrec.mat, hits.mat, correctRejects.mat, falseAlarms.mat,
# omissions.mat, reactionTimes.mat

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

###########################################
# You can print the informations of the dataset using the print function
print(sab_dataset)

###########################################
# If needed you can downsample the data using the downsample method.
help(sab_dataset.downsample)

# Downsample the dataset to 256Hz
sab_dataset.downsample(2)

###########################################
# TimeFeature Creation from the sab_dataset
# For this step you just need to call the create_features method of SabDataset class which returns a TimeFeature instance
time_features = sab_dataset.create_features()

###########################################
# It is possible to select some channels, electrodes or trials using respectively the chan_sel, electrode_sel or trial_sel
# arguments. See the help for more details :
help(sab_dataset.create_features)

###########################################
# You can check the time_features information using again the print function
print(time_features)

###########################################
# TimeDecoder Class.
# In this step, the time_decoder instance is created from the TimeDecoder class. This class allow to run classification on
# the time_features data.
# First we need to created a classifier to the time_decoder instance. It can be Support Vector Machine, LDA, ... as long
# as it support 2-classes classification with fit and predict methods
# In this example, we use a SVM classifier, with C parameter set to 1
svm_clf = svm.SVC(kernel='linear', probability=True)

# We can now create the time_decoder instance.
time_decoder = TimeDecoder(svm_clf)

###########################################
# Decoding.
# We can now start to classify the EEG data (stored in time_features), given 2 conditions (e.g. 'hits' and 'Correct rejects')
# by using the decode method.
data, _, labels, _ = time_features.get_data(label_keys=[1, 2])
# label_keys 1, 2 are the labels used for hits and correct_rejects respectively. You can see it with the label_dict argument
print(time_features.label_dict)
# Run the decoding using multiples processors :
scores = time_decoder.decode_mpver(data, labels, time_features.label_dict)

###########################################
# You can select data using the arguments of get_data method (see help(time_features.get_data)).
# The possible selection arguments are feature_pos, feature_type, feature_channame, label_keys and time_points.
# The current time_features instance contains the following channels :
print(time_features.channel_names)

###########################################
# If we want to select only some channel, you can specify it with the feature_channame argument :
data, _, labels, _ = time_features.get_data(feature_channame=['EEG TP\'2-TP\'3'], label_keys=[1, 2])
scores = time_decoder.decode_mpver(data, labels, time_features.label_dict)
