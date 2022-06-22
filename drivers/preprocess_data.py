#=====================================================================
# INITIALIZATIONS:
#=====================================================================
#torch
import torch
torch.manual_seed(0)
torch.set_default_dtype(torch.float64)
torch.autograd.set_detect_anomaly(True)

#numpy
import numpy as np
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
import os
import sys
sys.path.insert(1, '../core/')
from contextlib import contextmanager
import shutil
from distutils.dir_util import copy_tree

#custom:
from data_definitions import *
from load_fem_data import *
from utilities import *

fem_material = sys.argv[1];
noise = sys.argv[2];


# Get FE data
def extractSystemOfEquations(fem_dir,loadsteps):
    """
    Processes nodal and element data at various load steps to assemble the linear equation [A1;lambda_r*A2]*theta = [b1;lambda_r*b2]

    _Input Arguments_

    -`fem_dir` - File path location for data from FEM simulations

    -`loadsteps` - The deformation steps from the FEM simulations used to discover the material properties
    
    ---
    
    """
    for loadstep in loadsteps:
        print(loadstep)
        data = loadFemData(fem_dir+'/'+str(loadstep), AD=True, noiseLevel = 0., noiseType = 'displacement')

        d_features_dI1 = data.featureSet.d_features_dI1 #numElements x n_f
        d_features_dI3 = data.featureSet.d_features_dI3 #numElements x n_f
        d_features_dIa = data.featureSet.d_features_dIa #numElements x n_f
        d_features_dIb = data.featureSet.d_features_dIb #numElements x n_f

        dI1dF = data.dI1dF #numElements x 4
        dI3dF = data.dI3dF #numElements x 4
        dIadF = data.dIadF #numElements x 4
        dIbdF = data.dIbdF #numElements x 4

        d_features_dF11 = d_features_dI1 * dI1dF[:,0:1] + d_features_dI3 * dI3dF[:,0:1] + d_features_dIa * dIadF[:,0:1] + d_features_dIb * dIbdF[:,0:1]
        d_features_dF12 = d_features_dI1 * dI1dF[:,1:2] + d_features_dI3 * dI3dF[:,1:2] + d_features_dIa * dIadF[:,1:2] + d_features_dIb * dIbdF[:,1:2]
        d_features_dF21 = d_features_dI1 * dI1dF[:,2:3] + d_features_dI3 * dI3dF[:,2:3] + d_features_dIa * dIadF[:,2:3] + d_features_dIb * dIbdF[:,2:3]
        d_features_dF22 = d_features_dI1 * dI1dF[:,3:4] + d_features_dI3 * dI3dF[:,3:4] + d_features_dIa * dIadF[:,3:4] + d_features_dIb * dIbdF[:,3:4]

        num_nodes_per_element = len(data.gradNa)
        dim = data.x_nodes.shape[1]


        #Assemble A_nodes
        A_nodes = torch.zeros(data.numNodes, dim, getNumberOfFeatures()) #numNodes x dim x n_f

        for a in range(num_nodes_per_element):

            lhs1 = d_features_dF11 * data.gradNa[a][:,0:1] + d_features_dF12 * data.gradNa[a][:,1:2]
            lhs1 = lhs1 * data.qpWeights.unsqueeze(-1) #numElements x n_f

            lhs2 = d_features_dF21 * data.gradNa[a][:,0:1] + d_features_dF22 * data.gradNa[a][:,1:2]
            lhs2 = lhs2 * data.qpWeights.unsqueeze(-1) #numElements x n_f

            lhs = torch.stack((lhs1,lhs2),dim=1) #numElements x dim x n_f

            #Assemble over nodes:
            A_nodes.index_add_(0, data.connectivity[a], lhs) #numNodes x dim x nf

        #Assemble b_nodes
        mass_type = get_mass_type()
        if mass_type == 'lumped':
            b_nodes = - data.lumped_mass_acceleration
        else:
            raise ValueError('Incorrect mass_type option')

        #Rearrange as per dofs
        A_dofs = torch.flatten(A_nodes.clone(),start_dim=0,end_dim=1)
        b_dofs = torch.flatten(b_nodes.clone(),start_dim=0,end_dim=1)

        #Get free-dof equations
        dirichlet_dofs = torch.flatten(data.dirichlet_nodes.clone(),start_dim=0,end_dim=1)

        #Assemble free-dof equations
        A_free = A_dofs[~dirichlet_dofs,:]
        b_free = b_dofs[~dirichlet_dofs]

        #Assemble fixed-dof equations
        A_fix = torch.zeros(len(data.reactions), getNumberOfFeatures())
        b_fix = torch.zeros(len(data.reactions))

        for r in range(len(data.reactions)):

            bc_dofs = torch.flatten(data.reactions[r].dofs.clone(),start_dim=0,end_dim=1)

            A_fix[r,:] = torch.sum(A_dofs[bc_dofs,:],dim=0)

            b_fix[r] = torch.sum(b_dofs[bc_dofs],dim=0) + data.reactions[r].force

        #Extracting position information for free-dofs only
        dof_x = torch.flatten(torch.cat((data.x_nodes[:,0:1],data.x_nodes[:,0:1]),dim=1),start_dim=0,end_dim=1)
        dof_y = torch.flatten(torch.cat((data.x_nodes[:,1:2],data.x_nodes[:,1:2]),dim=1),start_dim=0,end_dim=1)

        dof_x = dof_x[~dirichlet_dofs]
        dof_y = dof_y[~dirichlet_dofs]

        #exporting to csv files

        np.savetxt(fem_dir+'/'+str(loadstep)+'/A1.csv', A_free.cpu().detach().numpy(), delimiter=",")
        np.savetxt(fem_dir+'/'+str(loadstep)+'/b1.csv', b_free.cpu().detach().numpy(), delimiter=",")

        np.savetxt(fem_dir+'/'+str(loadstep)+'/A2.csv', A_fix.cpu().detach().numpy(), delimiter=",")
        np.savetxt(fem_dir+'/'+str(loadstep)+'/b2.csv', b_fix.cpu().detach().numpy(), delimiter=",")

        np.savetxt(fem_dir+'/'+str(loadstep)+'/dof_x.csv', dof_x.cpu().detach().numpy(), delimiter=",")
        np.savetxt(fem_dir+'/'+str(loadstep)+'/dof_y.csv', dof_y.cpu().detach().numpy(), delimiter=",")
