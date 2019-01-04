"""
===================================================
      Feature Importance using Forest of Trees
===================================================

This example shows how to compute feature importance for a classification task using forest of trees

"""

# import matplotlib
# matplotlib.use('TkAgg')
from os.path import isdir, join
import sab_dataset
import seaborn as sns
sns.set()
sns.set_context('paper')

############################
# Load the data : sab dataset
sab_dataset_dirpath = join('pySAB', 'sample_data') if isdir('pySAB') else join('..', '..', 'pySAB', 'sample_data')
sab_dataset_filename = 'sab_dataset_rec_subject_id_040119_1153.p'
rec_dataset = sab_dataset.load_sab_dataset(join(sab_dataset_dirpath, sab_dataset_filename))

###########################
# Downsample the data
rec_dataset.downsample(2)

###################################################
# Construct the features from the SabDataset object - Select only 'hits' and 'correct rejects' trials and keep only
# 1 electrode of interest :
time_features = rec_dataset.create_features(trial_sel=(rec_dataset.hits | rec_dataset.correct_rejects))
print(time_features)

############################################################
# Compute feature importance using forest of decision trees
time_features.compute_feature_importance([1, 2])

