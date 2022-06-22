import torch
import numpy as np
import pandas as pd
from utilities import *
from features_library import *
from config import *
from data_definitions import *

def loadFemData(path, AD = True, noiseLevel = 0., noiseType = 'displacement', denoisedDisplacements = None):
    """
    Load finite element data and add noise (optional).
    Note that the loaded finite element data might already be perturbed by noise.
    In that case, adding additional noise is not necessary.

    _Input Arguments_

    - `path` - path to finite element data

    - `AD` - specify if automatic differention is needed

    - `noiseLevel` - noise level

    - `noiseType` - a string list specifying whether noise should be added to 'displacement','strain','acceleration','velocity'

    - `denoisedDisplacements` - pass denoised displacement data if available

    _Output Arguments_

    - `dataset` - finite element dataset

    ---

    """
    print('\n-----------------------------------------------------')
    print('Loading data: ', path)
    if(path[-1]=='/'):
        path=path[0:-1]
    numNodesPerElement = 3
    #nodal data
    df = pd.read_csv(path+'/output_nodes.csv',dtype=np.float64)
    numNodes = df.shape[0]
    x_nodes = torch.tensor(df[['x','y']].values)
    u_nodes = torch.tensor(df[['ux','uy']].values)

    if (('vx' in df.columns) and ('vy' in df.columns)):
        v_nodes = torch.tensor(df[['vx','vy']].values)
    else:
        v_nodes = torch.zeros_like(u_nodes)

    if (('ax' in df.columns) and ('ay' in df.columns)):
        a_nodes = torch.tensor(df[['ax','ay']].values)
    else:
        a_nodes = torch.zeros_like(u_nodes)

    if(denoisedDisplacements is not None):
        u_nodes = denoisedDisplacements

    bcs_nodes = torch.tensor(df[['bcx','bcy']].round().astype(int).values)
    #convert bcs_nodes to booleans
    dirichlet_nodes = (bcs_nodes!=0)

    #noise
    if(('displacement' in noiseType) or ('strain' in noiseType) or ('velocity' in noiseType) or ('acceleration' in noiseType)):
        pass
    else:
        raise ValueError('Incorrect noiseType argument!')

    noise_u_nodes = noiseLevel * torch.randn_like(u_nodes)
    noise_u_nodes[dirichlet_nodes] = 0.

    if('displacement' in noiseType):
        u_nodes += noise_u_nodes
        print('Applying noise to displacements:',noiseLevel)
    noise_a_nodes = noiseLevel * torch.randn_like(a_nodes)
    noise_a_nodes[dirichlet_nodes] = 0.
    if('acceleration' in noiseType):
        a_nodes += noise_a_nodes
        print('Applying noise to accelerations:',noiseLevel)

    noise_v_nodes = noiseLevel * torch.randn_like(v_nodes)
    noise_v_nodes[dirichlet_nodes] = 0.
    if('velocity' in noiseType):
        v_nodes += noise_v_nodes
        print('Applying noise to velocities:',noiseLevel)

    #reaction forces data
    numReactions = torch.max(bcs_nodes).item()
    df = pd.read_csv(path+'/output_reactions.csv',dtype=np.float64)
    reactions = []
    for i in range(numReactions):
        reactions.append(Reaction(bcs_nodes == (i+1),df['forces'][i])) # These reaction forces are force/thickness in dimension.

    #element data
    df = pd.read_csv(path+'/output_elements.csv',dtype=np.float64)
    numElements = df.shape[0]
    connectivity = []
    for i in range(numNodesPerElement):
        connectivity.append(torch.tensor(df['node'+str(i+1)].round().astype(int).tolist()))

    #integrator/shape-function data
    df = pd.read_csv(path+'/output_integrator.csv',dtype=np.float64)
    gradNa = []
    for i in range(numNodesPerElement):
        gradNa.append(torch.tensor(df[['gradNa_node'+str(i+1)+'_x','gradNa_node'+str(i+1)+'_y']].values))
    qpWeights = torch.tensor(df['qpWeight'].values)

    #element-wise rearrangement of displacements
    u = []
    for i in range(numNodesPerElement):
        u.append(u_nodes[connectivity[i],:])

    #computing deformation gradient at quadrature points
    dim=2
    voigtMap = [[0,1],[2,3]]

    F=torch.zeros(numElements,4)
    for a in range(numNodesPerElement):
        for i in range(dim):
            for j in range(dim):
                F[:,voigtMap[i][j]] += u[a][:,i] * gradNa[a][:,j]
    F[:,0] += 1.0
    F[:,3] += 1.0

    if('strain' in noiseType):
        F += noiseLevel * torch.randn_like(F)
        print('Applying noise to strains:',noiseLevel)

    #computing detF
    J = computeJacobian(F)

    #computing Cauchy-Green strain: C = F^T F
    C = computeCauchyGreenStrain(F)

    #computing strain invariants
    I1, I2, I3 = computeStrainInvariants(C)

    # a_mult is defined in config
    Ia = C @ a_mult
    Ib = C @ b_mult

    #activate gradients
    I1.requires_grad = True
    I2.requires_grad = True
    I3.requires_grad = True
    Ia.requires_grad = True
    Ib.requires_grad = True

    #computing invariant derivaties
    dI1dF = computeStrainInvariantDerivatives(F,1)
    dI2dF = computeStrainInvariantDerivatives(F,2)
    dI3dF = computeStrainInvariantDerivatives(F,3)
    dIadF = computeStrainInvariantDerivatives(F,4)
    dIbdF = computeStrainInvariantDerivatives(F,5)

    #computing extended set of nonlinear features
    featureSet = FeatureSet()
    featureSet.features = computeFeatures_torch(I1, I2, I3, Ia, Ib) #don't detach, need it for autograd

    if(AD==True):
        def differentiateFeaturesWithInvariants(features,I):
            """
            Compute derivatives of the features with respect to the invariants of the Cauchy-Green strain tensor using automatic differentiation.

            _Input Arguments_

            - `features` - features

            - `I` - invariant

            _Output Arguments_

            - `d_feature_dI` - derivative

            ---

            """
            d_feature_dI = torch.zeros(features.shape[0],features.shape[1])
            for i in range(features.shape[1]):
                temp = torch.autograd.grad(features[:,i:(i+1)],I,torch.ones(I.shape[0],1),create_graph=True,allow_unused=True)[0]
                if(type(temp)!=type(None)):
                    d_feature_dI[:,i:(i+1)] = temp
            return d_feature_dI

        featureSet.d_features_dI1 = differentiateFeaturesWithInvariants(featureSet.features,I1)
        featureSet.d_features_dI2 = differentiateFeaturesWithInvariants(featureSet.features,I2)
        featureSet.d_features_dI3 = differentiateFeaturesWithInvariants(featureSet.features,I3)
        featureSet.d_features_dIa = differentiateFeaturesWithInvariants(featureSet.features,Ia)
        featureSet.d_features_dIb = differentiateFeaturesWithInvariants(featureSet.features,Ib)

        #detach features now:
        featureSet.features = featureSet.features.detach()
        featureSet.d_features_dI1 = featureSet.d_features_dI1.detach()
        featureSet.d_features_dI2 = featureSet.d_features_dI2.detach()
        featureSet.d_features_dI3 = featureSet.d_features_dI3.detach()
        featureSet.d_features_dIa = featureSet.d_features_dIa.detach()
        featureSet.d_features_dIb = featureSet.d_features_dIb.detach()

    #density
    if os.path.exists(path+'/output_density.csv'):
        df = pd.read_csv(path+'/output_density.csv',dtype=np.float64)
        density = df.density[0]
    else:
        print('\nWARNING: density file not found\n')
        density = 0.


    lumped_mass = torch.zeros(numNodes, 2)

    for e in range(numElements):
        for a1 in range(numNodesPerElement):
            lumped_mass[connectivity[a1][e]][0] += density * qpWeights[e]/numNodesPerElement
            lumped_mass[connectivity[a1][e]][1] += density * qpWeights[e]/numNodesPerElement

    # lumped mass_acceleration | size: numNodes x 2
    lumped_mass_acceleration = torch.mul(lumped_mass,a_nodes)


    dataset = FemDataset(path,
        x_nodes, u_nodes, v_nodes, a_nodes, dirichlet_nodes,
        reactions,
        connectivity, gradNa, qpWeights,
        F, J, C,
        I1, I2, I3, Ia, Ib,
        dI1dF, dI2dF, dI3dF, dIadF, dIbdF,
        featureSet,
        density,
        lumped_mass_acceleration)

    print('-----------------------------------------------------\n')

    return dataset
