"""
==============================
 Inter-Trial Phase Clustering
==============================

This example shows how to compute and plot the ITPC : Inter-Trial Phase Coherence from a SabDataset instance

"""

import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
from os.path import isdir, join
plt.rcParams['image.cmap'] = 'viridis'
from sab_dataset import *
import seaborn as sns
sns.set()
sns.set_context('paper')

############################
# Load the data : sab dataset
sab_dataset_dirpath = join('pySAB', 'sample_data') if isdir('pySAB') else join('..', '..', 'pySAB', 'sample_data')
sab_dataset_filename = 'sab_dataset_small.p'
rec_dataset = load_sab_dataset(join(sab_dataset_dirpath, sab_dataset_filename))

############################
# Print dataset information
print(rec_dataset)

############################
# Downsample the dataset
rec_dataset.downsample(2)

#######################################################
# Select filter center frequencies and filter bandwidth
n_filters = 30
filt_cf = np.logspace(np.log10(3), np.log10(70), n_filters)
filt_bw = np.logspace(np.log10(1.5), np.log10(20), n_filters)
f = plt.figure()
ax = f.add_subplot(111)
ax.scatter(np.arange(n_filters), filt_cf, zorder=2)
ax.vlines(np.arange(n_filters), filt_cf-filt_bw/2, filt_cf+filt_bw/2, zorder=1)
ax.set(title='Filters center frequency and bandwidth', xlabel='Filter index', ylabel='(Hz)')
plt.legend(['Center frequency', 'Bandwidth'])

##########################################################
# Compute and plot the ITPC of channel 4, for hits trials
rec_dataset.plot_itpc(4, rec_dataset.hits, filt_cf, filt_bw, n_monte_carlo=1, contour_plot=1)

