"""
============================================
         Extract and Save Features
============================================

This example shows how to create a TimeFeature instance from a SabDataset instance,
 how to extract features from the amplitude data and save them so that it can be used later on.

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
# sab_dataset_dirpath = 'sample_data_whole' if isdir('sample_data_whole') else join('..', '..', 'sample_data_whole')
# subject_id = '042'
# rec_dataset = sab_dataset.SabDataset(sab_dataset_dirpath, subject_id, 'rec')

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
# Extract the phase
# time_features.extract_feature('cwt_phase')

#####################################
# Save the time features instance so that it can be used later without having to re-compute the features
# time_features.save(dir_path=sab_dataset_dirpath)

