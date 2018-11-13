"""
============================================
      Feature Evolution and Distribution
============================================

This example shows how to plot feature evolution, or the distribution of a feature over trials
at a certain time point.

"""

from os.path import isdir, join
import sab_dataset
import seaborn as sns
sns.set()
sns.set_context('paper')

############################
# Load the data : sab dataset
sab_dataset_dirpath = join('pySAB', 'sample_data') if isdir('pySAB') else join('..', '..', 'pySAB', 'sample_data')
sab_dataset_filename = 'sab_dataset_name.p'
rec_dataset = sab_dataset.load_sab_dataset(join(sab_dataset_dirpath, sab_dataset_filename))

###########################
# Downsample the data
rec_dataset.downsample(8)

###################################################
# Construct the features from the SabDataset object - Select only 'hits' and 'correct rejects' trials and keep only
# 2 electrodes of interest :
time_features = rec_dataset.create_features(electrode_sel=['TB\'', 'C\''],
                                            trial_sel=(rec_dataset.hits | rec_dataset.correct_rejects))
print(time_features)

#####################################
# Extract features, if called without any parameter, the function return the possible feature to extract
time_features.extract_feature()

#####################################
# Extract the discrete wavelet transform coefficients, The new features are automatically added
time_features.extract_feature('dwt')
print(time_features)

####################################
# Plot the evolution of feature 19 :
time_features.plot_feature_erp(feature_pos=19)

####################################
# Plot the evolution of feature DWT 16-32 Hz for channel TB'10-TB'11 :
time_features.plot_feature_erp(feature_type='DWT 16-32', feature_channame='TB\'10-TB\'11')

####################################
# Plot the distribution of a feature at a certain time point. The 'time_points' parameter must be passed and contain
# only 1 time point
time_point_sel = time_features.time2sample(0.55)
time_features.plot_feature_distribution(time_points=time_point_sel, feature_pos=19)
