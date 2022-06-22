import os
import sys
sys.path.insert(1, '../core/')
#os.environ["PYTHONBREAKPOINT"] = "0"

import numpy as np
import scipy.stats as stats
import copy

from core_spike_slab import *
from unsupervised_hyperelasticity import *
from post_process import *
from helper import *
from preprocess_data import extractSystemOfEquations

from config import *

#torch
import torch
torch.manual_seed(0)
torch.set_default_dtype(torch.float64)
torch.autograd.set_detect_anomaly(True)

#numpy
np.random.seed(0)

#matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

#scipy
import scipy
from scipy import sparse

#pandas
import pandas as pd

#others:
from contextlib import contextmanager
import shutil
from distutils.dir_util import copy_tree

#custom:
from data_definitions import *
from load_fem_data import *
from utilities import *

# show current time
show_current_time()

# rng
rng = np.random.default_rng(seed)

# argv, the first argument in the function call is material name, second is noise level, third argument indicates whether data is to be drawn from dynamic simulations or not.
prefix = '/res='
fig_prefix = 'Quasi-static '
dynamic = False
if (len(sys.argv)>1):
	if(len(sys.argv)>=3):
		fem_material = sys.argv[1]
		noise_level = sys.argv[2]
	if(len(sys.argv)==4):
		dynamic = True
		fem_dir = '../dyn-euclid-master-data'
		prefix = '/dyn_res='
		fig_prefix = 'Dynamic '
	if(len(sys.argv)>4):
		raise ValueError('Wrong number of command line arguments')
check_data_name_validity(fem_material,noise_level)

#----------------------------------------------------------------------------------------
# data
#----------------------------------------------------------------------------------------
filtered_data_set = []
loadsteps = get_loadsteps(fem_material)
feature_filter = get_feature_filter(fem_material)

#----------------------------------------------------------------------------------------
#preprocess_data

fem_dir_pp = fem_dir+prefix+fem_res+'_noise='+noise_level+'/'+fem_material;
extractSystemOfEquations(fem_dir_pp,loadsteps)

#----------------------------------------------------------------------------------------

for loadstep in loadsteps:

	raw_data = get_data(rng, fem_dir, prefix, fem_material, noise_level, loadstep, feature_filter)

	filtered_data = filter_raw_data(rng, raw_data, filter_value)

	filtered_data_set.append(filtered_data)

data = process_raw_data(filtered_data_set, lambda_r)

theta_gt = get_theta_gt(fem_material, feature_filter)

if feature_filter:
 	theta_gt = theta_gt[feature_filter]

#----------------------------------------------------------------------------------------
# setup
#----------------------------------------------------------------------------------------

params = Params(N = data.D.shape[0],
				P = data.D.shape[1],
				a_v = a_v,
				b_v = b_v,
				a_p = a_p,
				b_p = b_p,
				a_sigma = a_sigma,
				b_sigma = b_sigma)

# Run parallel chains
all_chains = []
for chain_iter in range(0,parallel_chains):
	chain = run_spike_slab(rng, data, params, chain_length, burn, chain_iter, verbose=True)
	all_chains.append(chain)

# Combine chains for analysis
chain = all_chains[0]
for chain_iter in range(1,parallel_chains):
	chain.combine_chain(all_chains[chain_iter])

def getFigTitle(fem_material):
	""" Depending on the fem_material value, any custom title can be assigned to the feature coefficient plots for the chain.
	"""
	return fem_material

fig_title2 = fem_material+', noise='+noise_level
if 'high' in noise_level:
	fig_title = 'Benchmark: '+getFigTitle(fem_material)+', '+noise_level+' noise ($\sigma_u =10^{-3}$)'
elif 'low' in noise_level:
	fig_title = 'Benchmark: '+getFigTitle(fem_material)+', '+noise_level+' noise ($\sigma_u =10^{-4}$)'
else:
	fig_title = 'Benchmark: '+getFigTitle(fem_material)+', '+noise_level+' noise ($\sigma_u =0$)'

os.makedirs(fig_dir,exist_ok=True)
post_proc(chain, theta_gt, feature_filter, fem_material, predict_energy_path, fig_prefix+fig_title, fig_title2, fig_dir, plotting=True, interactive_job=interactive_job)

print('\n\n....... ending .......')

# show current time
show_current_time()
