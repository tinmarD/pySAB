"""
=====================
ERP Analyses example
=====================

This example shows how to use the ERP functions

"""

from sab_dataset import *
from os.path import isdir, join
import seaborn as sns
sns.set()
sns.set_context('paper')

############################
# Load the data : sab dataset
sab_dataset_dirpath = join('pySAB', 'sample_data') if isdir('pySAB') else join('..', '..', 'pySAB', 'sample_data')
sab_dataset_filename = 'sab_dataset_rec_subject_id_040119_1153.p'
rec_dataset = load_sab_dataset(join(sab_dataset_dirpath, sab_dataset_filename))

###############################
# See information about this dataset
print(rec_dataset)

##################################
# Plot the evoked response of channel 6 for 'hits' and 'correct reject' conditions (default)
rec_dataset.plot_erp(6)

#################################
# Plot the ERPs of channels containing `'TB'4'` one for hits condition
rec_dataset.plot_erp('EEG TP\'3-TP\'4', plot_hits=1, plot_cr=0, plot_fa=0, plot_omissions=0)

#################################
# Plot ERPs of the first electrode
rec_dataset.plot_electrode_erps(0)




