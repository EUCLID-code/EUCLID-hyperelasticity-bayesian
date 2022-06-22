import os
import numpy as np
import scipy.stats as stats
from core_spike_slab import *
from helper import *
from features_library import *
from config import *

class RawData:
    def __init__(self, A1, b1, A2, b2, dof_x, dof_y):
        self.A1 = A1
        self.b1 = b1
        self.A2 = A2
        self.b2 = b2
        self.dof_x = dof_x
        self.dof_y = dof_y


def check_data_name_validity(fem_material,noise_level):
    material_list = ['NeoHookeanJ2', 'NeoHookeanJ4', 'Isihara', 'HainesWilson', 'GentThomas','ArrudaBoyce','Ogden','Ogden3','Anisotropy','PDMS','Holzapfel']
    noise_list = ['none','low','high']
    if fem_material not in material_list:
        raise ValueError('Incorrect fem_material provided')
    if noise_level not in noise_list:
        raise ValueError('Incorrect noise_level provided')
    print('fem_material: ',fem_material)
    print('noise_level: ',noise_level)


def get_data(rng, fem_dir, prefix, fem_material, noise_level, loadstep, feature_filter):

    prefix = fem_dir+prefix+fem_res+'_noise='+noise_level+'/'+fem_material;
    if(prefix[-1]=='/'):
        prefix = prefix[0:-1]

    fem_path = prefix+'/'+str(loadstep)+'/';

    A1 = np.loadtxt(fem_path+'A1.csv', dtype = np.float64, delimiter=",");
    b1 = np.loadtxt(fem_path+'b1.csv', dtype = np.float64, delimiter=",");

    A2 = np.loadtxt(fem_path+'A2.csv', dtype = np.float64, delimiter=",");
    b2 = np.loadtxt(fem_path+'b2.csv', dtype = np.float64, delimiter=",");

    dof_x = np.loadtxt(fem_path+'dof_x.csv', dtype = np.float64, delimiter=",");
    dof_y = np.loadtxt(fem_path+'dof_y.csv', dtype = np.float64, delimiter=",");

    if feature_filter:
        A1 = A1[:,feature_filter]
        A2 = A2[:,feature_filter]

    return RawData(A1, b1, A2, b2, dof_x, dof_y)

def filter_raw_data(rng, raw_data, filter_value):
    """
    Randomly sub-sampling `filter_value` degrees of freedom from the data available at all quadrature points in `raw_data`

    _Input Arguments_

    -`rng` - Random number generator

    -`raw_data` - Contains `D` (derivatives of feature library) and `y` (inertia terms and reaction forces) evaluated at all quadrature points

    -`filter_value` - Number of degrees of freedom subsampled from the original data

    _Output Arguments_

    - `RawData(A1, b1, A2, b2, dof_x, dof_y)` - Object of class `RawData` which is constructed from subsampled data
    
    ---
    
    """

    if not filter_value:
        #filter_value is empty
        return raw_data

    A1 = raw_data.A1
    b1 = raw_data.b1

    A2 = raw_data.A2
    b2 = raw_data.b2

    dof_x = raw_data.dof_x
    dof_y = raw_data.dof_y

    # A total of 'filter_value' number of points are randomly sub-sampled
    idx = np.arange(A1.shape[0])
    idx = rng.permutation(idx)
    arr_idx = idx[0:filter_value]

    A1 = A1[arr_idx,:]
    b1 = b1[arr_idx]

    dof_x = dof_x[arr_idx]
    dof_y = dof_y[arr_idx]

    print(str(A1.shape[0])+'\n')
    print(str(b1.shape[0])+'\n')
    print(str(dof_x.shape[0])+'\n')
    print(str(dof_y.shape[0])+'\n')
    print(str(len(arr_idx))+'\n')
    if A1.shape[0] != filter_value:
        raise ValueError('Something went wrong in data filtering')
    if b1.shape[0] != filter_value:
        raise ValueError('Something went wrong in data filtering')
    if dof_x.shape[0] != filter_value:
        raise ValueError('Something went wrong in data filtering')
    if dof_y.shape[0] != filter_value:
        raise ValueError('Something went wrong in data filtering')

    return RawData(A1, b1, A2, b2, dof_x, dof_y)



def process_raw_data(raw_data_set, lambda_r):

    D = raw_data_set[0].A1
    y = raw_data_set[0].b1

    # Careful: start iteration from 1
    for i in range(1,len(raw_data_set)):
        D = np.concatenate((D, raw_data_set[i].A1), axis=0)
        y = np.concatenate((y, raw_data_set[i].b1), axis=0)

    # Careful: start iteration from 0
    for i in range(0,len(raw_data_set)):
        D = np.concatenate((D, lambda_r * raw_data_set[i].A2), axis=0)
        y = np.concatenate((y, lambda_r * raw_data_set[i].b2), axis=0)

    return Data(D, y)



def predict_energy_path(chain, theta_gt, fem_mat, feature_filter, deformation):
    """
    Predicts the energy deformation evolution for predicted and true material models along six deformation paths:
    i.) uniaxial tension, ii.) uniaxial compression, iii.) simple shear, iv.) biaxial tension, v.) biaxial compression, vi.) pure shear

    _Input Arguments_

    -`chain` - Object of class `Chain` (see `core_spike_slab` file)

    -`theta_gt` - The true set of feature coefficients for the benchmark material

    -`fem_mat` - The name of the benchmark material to be tested

    -`feature_filter` - The list of features to retain for constructing the Markov chain. Suppressed features will be highlighted with a red patch in the plot

    -`deformation` - Name of deformation path to be evaluated

    _Output Arguments_

    -`gamma` - Numpy array of different values for the deformation parameter

    -`energy_mean` - Energy corresponding to mean value of `theta` (feature coefficients) across different members of the Markov chain

    -`energy_plus` - 97.5 percentile energy branch across different members of the Markov chain

    -`energy_minus` - 2.5 percentile energy branch across different members of the Markov chain

    -`energy_gt` - Energy corresponding to the true feature coefficients

    -`energy` - Numpy array containing energy of all chain members at all deformation parameters

    ---
    
    """

    gamma = np.linspace(0.,1.,num=50)

    F11 = np.zeros_like(gamma) + 1.
    F12 = np.zeros_like(gamma)
    F21 = np.zeros_like(gamma)
    F22 = np.zeros_like(gamma) + 1.

    if deformation == 'tension':
        F11 = 1.+gamma

    elif deformation == 'simple_shear':
        F12 = gamma

    elif deformation == 'pure_shear':
        F11 = 1.+gamma
        F22 = 1./(1.+gamma)

    elif deformation == 'compression':
        F11 = 1./(1.+gamma)

    elif deformation == 'biaxial_tension':
        F11 = 1.+gamma
        F22 = 1.+gamma

    elif deformation == 'biaxial_compression':
        F11 = 1./(1.+gamma)
        F22 = 1./(1.+gamma)

    else:
        raise ValueError('Wrong input for deformation path')

    features = computeFeatures_numpy(F11, F12, F21, F22)
    features_temp = features ##Remove if not working
    features = features[:,feature_filter]

    energy = features @ chain.theta.T

    energy_mean = np.mean(energy,axis=1)
    energy_std = np.std(energy,axis=1)

    energy_plus  = np.percentile(energy, 97.5, axis=1)
    energy_minus = np.percentile(energy, 2.5, axis=1)

    energy_gt = features_temp @ (get_theta_gt(fem_mat,feature_filter)) ##Remove if not working

    return gamma, energy_mean, energy_plus, energy_minus, energy_gt, energy


#
