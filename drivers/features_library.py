import os
import numpy as np
import scipy.stats as stats
import torch
from core_spike_slab import *
from helper import *
from config import *


def get_mass_type():
    """
    
    """

    mass_type = "lumped" #Can be made 'consistent' otherwise

    return mass_type


def computeFeatures_torch(I1, I2, I3, Ia, Ib):
    """
    Compute the features dependent on the right Cauchy-Green strain invariants.
    Note that the features only depend on I1 and I3 for plane strain assumption.

    _Input Arguments_

    - `I1` - 1st invariant

    - `I2` - 2nd invariant

    - `I3` - 3rd invariant

    - `Ia` - Invariant along anisotropy direction.

    - `Ib` - Invariant along second anisotropy direction.

    _Output Arguments_

    - `x` - features

    ---

    """

    numFeatures = 28
    x = torch.zeros(I1.shape[0],numFeatures)

    # Some standard terms
    K1 = I1 * torch.pow(I3,-1/3) - 3.0
    K2 = (I1 + I3 - 1) * torch.pow(I3,-2/3) - 3.0
    J = torch.sqrt(I3)
    Iat = Ia * torch.pow(J,-2./3.)
    Ibt = Ib * torch.pow(J,-2./3.)

    i=-1;

    ####################################################
    # Generalized Mooney-Rivlin:

    #linear
    i+=1; x[:,i:(i+1)] = K1
    i+=1; x[:,i:(i+1)] = K2
    #quadratic
    i+=1; x[:,i:(i+1)] = K1**2
    i+=1; x[:,i:(i+1)] = K1 * K2
    i+=1; x[:,i:(i+1)] = K2**2
    #cubic
    i+=1; x[:,i:(i+1)] = K1**3
    i+=1; x[:,i:(i+1)] = (K1**2) * K2
    i+=1; x[:,i:(i+1)] = K1 * (K2**2)
    i+=1; x[:,i:(i+1)] = K2**3
    #quartic
    i+=1; x[:,i:(i+1)] = K1**4
    i+=1; x[:,i:(i+1)] = (K1**3) * K2
    i+=1; x[:,i:(i+1)] = (K1**2) * (K2**2)
    i+=1; x[:,i:(i+1)] = K1 * (K2**3)
    i+=1; x[:,i:(i+1)] = K2**4

    ######################################################
    #volumetric term:
    i+=1; x[:,i:(i+1)] = (J-1)**2

    # ####################################################
    # Gent-Thomas:
    i+=1; x[:,i:(i+1)] = torch.log((K2+3.0)/3.0)

    ####################################################
    #Arruda Boyce
    lambda_chain = torch.sqrt(I1 * torch.pow(J,-2./3.)/3.)
    xtilde = lambda_chain / torch.sqrt(torch.tensor([28.]))
    s1 = (torch.sign(0.841-xtilde)+1.)/2.
    s2 = (torch.sign(xtilde-0.841)+1.)/2.
    beta_chain1 = 1.31*torch.tan(1.59*xtilde)+0.91*xtilde
    beta_chain2 = torch.tensor([1.])/(torch.sign(xtilde) - xtilde)
    beta_chain = s1 * beta_chain1 + s2*beta_chain2

    R1 = beta_chain*lambda_chain + torch.sqrt(torch.tensor([28.]))*torch.log(beta_chain/torch.sinh(beta_chain))
    i+=1; x[:,i:(i+1)] = torch.tensor([10.])* (torch.sqrt(torch.tensor([28.]))*R1 - 1.5164)


    ####################################################
    #Ogden
    I1_tilde = J**(-2/3)*I1 + 1e-13
    I1t_0 =torch.tensor([3]) + 1e-13
    J_0 = torch.tensor([1]) + 1e-13
    mu_ogden = 1
    alpha_ogden = 0.65
    W_offset = 1/alpha_ogden * 2. * (0.5**alpha_ogden*(I1t_0  +  torch.sqrt(  (I1t_0-1/(J_0**(2./3.)))**2 - 4*J_0**(2./3.)) - 1/(J_0**(2./3.)) )**alpha_ogden+( 0.5*I1t_0 - 0.5*torch.sqrt(  (I1t_0-1/(J_0**(2./3.)))**2 - 4*J_0**(2./3.))  - 0.5/(J_0**(2./3.)) )**alpha_ogden + J_0**(-alpha_ogden*2./3.) ) * mu_ogden
    W_truth = 1/alpha_ogden * 2. * (0.5**alpha_ogden*(I1_tilde  +  torch.sqrt(  (I1_tilde-1/(J**(2./3.)))**2 - 4*J**(2./3.)) - 1/(J**(2./3.)) )**alpha_ogden+( 0.5*I1_tilde - 0.5*torch.sqrt(  (I1_tilde-1/(J**(2./3.)))**2 - 4*J**(2./3.))  - 0.5/(J**(2./3.)) )**alpha_ogden + J**(-alpha_ogden*2./3.) ) * mu_ogden - W_offset
    i+=1; x[:,i:(i+1)] = W_truth
    ####################################################
    #Ogden2ndterm
    I1_tilde = J**(-2/3)*I1 + 1e-13
    I1t_0 =torch.tensor([3]) + 1e-13
    J_0 = torch.tensor([1]) + 1e-13
    mu_ogden = 1
    alpha_ogden = 2.5
    W_offset = 1/alpha_ogden * 2. * (0.5**alpha_ogden*(I1t_0  +  torch.sqrt(  (I1t_0-1/(J_0**(2./3.)))**2 - 4*J_0**(2./3.)) - 1/(J_0**(2./3.)) )**alpha_ogden+( 0.5*I1t_0 - 0.5*torch.sqrt(  (I1t_0-1/(J_0**(2./3.)))**2 - 4*J_0**(2./3.))  - 0.5/(J_0**(2./3.)) )**alpha_ogden + J_0**(-alpha_ogden*2./3.) ) * mu_ogden
    W_truth = 1/alpha_ogden * 2. * (0.5**alpha_ogden*(I1_tilde  +  torch.sqrt(  (I1_tilde-1/(J**(2./3.)))**2 - 4*J**(2./3.)) - 1/(J**(2./3.)) )**alpha_ogden+( 0.5*I1_tilde - 0.5*torch.sqrt(  (I1_tilde-1/(J**(2./3.)))**2 - 4*J**(2./3.))  - 0.5/(J**(2./3.)) )**alpha_ogden + J**(-alpha_ogden*2./3.) ) * mu_ogden - W_offset
    i+=1; x[:,i:(i+1)] = W_truth
    ####################################################
    #Ogden3rdTerm
    I1_tilde = J**(-2/3)*I1 + 1e-13
    I1t_0 =torch.tensor([3]) + 1e-13
    J_0 = torch.tensor([1]) + 1e-13
    mu_ogden = 1
    alpha_ogden = 1
    W_offset = 1/alpha_ogden * 2. * (0.5**alpha_ogden*(I1t_0  +  torch.sqrt(  (I1t_0-1/(J_0**(2./3.)))**2 - 4*J_0**(2./3.)) - 1/(J_0**(2./3.)) )**alpha_ogden+( 0.5*I1t_0 - 0.5*torch.sqrt(  (I1t_0-1/(J_0**(2./3.)))**2 - 4*J_0**(2./3.))  - 0.5/(J_0**(2./3.)) )**alpha_ogden + J_0**(-alpha_ogden*2./3.) ) * mu_ogden
    W_truth = 1/alpha_ogden * 2. * (0.5**alpha_ogden*(I1_tilde  +  torch.sqrt(  (I1_tilde-1/(J**(2./3.)))**2 - 4*J**(2./3.)) - 1/(J**(2./3.)) )**alpha_ogden+( 0.5*I1_tilde - 0.5*torch.sqrt(  (I1_tilde-1/(J**(2./3.)))**2 - 4*J**(2./3.))  - 0.5/(J**(2./3.)) )**alpha_ogden + J**(-alpha_ogden*2./3.) ) * mu_ogden - W_offset
    i+=1; x[:,i:(i+1)] = W_truth
    ####################################################
    #Anisotropy 1 fibre
    i+=1; x[:,i:(i+1)] = torch.pow(Iat - 1.,2.)
    i+=1; x[:,i:(i+1)] = torch.pow(Iat - 1.,3.)
    i+=1; x[:,i:(i+1)] = torch.pow(Iat - 1.,4.)
    ####################################################
    #Anisotropy 2 fibre
    i+=1; x[:,i:(i+1)] = torch.pow(Ibt - 1.,2.)
    i+=1; x[:,i:(i+1)] = torch.pow(Ibt - 1.,3.)
    i+=1; x[:,i:(i+1)] = torch.pow(Ibt - 1.,4.)
    ####################################################
    # Holzapfel features (These features are filtered out during chain formation. They are used only to evaluate ground-truth Holzapfel model)
    k1h = 0.9;
    k2h = 0.8;
    i+=1; x[:,i:(i+1)] = k1h/2./k2h*(torch.exp(k2h*torch.pow(Iat - 1.,2.))-1.);
    i+=1; x[:,i:(i+1)] = k1h/2./k2h*(torch.exp(k2h*torch.pow(Ibt - 1.,2.))-1.);

    # . . .  add more if needed;
    # don't forget to update the numFeatures above

    if(i!=numFeatures-1):
        raise ValueError('You forgot to change numFeatures in computeFeatures_NN!')

    return x

def get_theta_gt(fem_path, feature_filter):
    """
    Returns the true feature coefficients corresponding to the benchmark material.
    
    """

    n_f = getNumberOfFeatures()

    nh2 = np.zeros(n_f);
    nh2[0] = 0.5;
    nh2[14] = 1.5;

    ish = np.zeros(n_f);
    ish[0] = 0.5;
    ish[1] = 1;
    ish[2] = 1;
    ish[14] = 1.5;

    hw = np.zeros(n_f);
    hw[0] = 0.5;
    hw[1] = 1;
    hw[3] = 0.7;
    hw[5] = 0.2;
    hw[14] = 1.5;

    gth = np.zeros(n_f);
    gth[0] = 0.5;
    gth[14] = 1.5;
    gth[15] = 1.0;

    ab = np.zeros(n_f)
    ab[16] = 0.25;
    ab[14] = 1.5;

    og = np.zeros(n_f)
    og[17] = 0.65;
    og[14] = 1.5;

    og3 = np.zeros(n_f)
    og3[17] = 0.4;
    og3[18] = 0.0012;
    og3[19] = 0.1;
    og3[14] = 1.5;


    hol = np.zeros(n_f)
    hol[0] = 0.5;
    hol[14] = 1.0;
    hol[26] = 1.0;
    hol[27] = 1.0;

    theta_gt = []

    if 'NeoHookeanJ2' in fem_path:
        theta_gt = nh2
    elif 'NeoHookeanJ4' in fem_path:
        theta_gt = nh4
    elif 'Isihara' in fem_path:
        theta_gt = ish
    elif 'HainesWilson' in fem_path:
        theta_gt = hw
    elif 'GentThomas' in fem_path:
        theta_gt = gth
    elif 'ArrudaBoyce' in fem_path:
        theta_gt = ab
    elif 'Holzapfel' in fem_path:
        theta_gt = hol;
    elif 'Ogden' in fem_path:
        if '3' in fem_path:
            theta_gt = og3;
        else:
            theta_gt = og;
    else:
        raise ValueError('Can\'t detect GT material model')

    return theta_gt


def getNumberOfFeatures():
    """
    Compute number of features.

    _Input Arguments_

    - _none_

    _Output Arguments_

    - `features.shape[1]` - number of features

    ---

    """
    features = computeFeatures_torch(torch.zeros(1,1),torch.zeros(1,1),torch.zeros(1,1),torch.zeros(1,1),torch.zeros(1,1))
    return features.shape[1]



def computeFeatures_numpy(F11, F12, F21, F22):
    """
    
    """
    #Adapted from first EUCLID paper's code with modifications
    #(replacing torch with numpy)

    if(len(F11.shape) == 1):
        F11 = np.expand_dims(F11, axis=-1)
        F12 = np.expand_dims(F12, axis=-1)
        F21 = np.expand_dims(F21, axis=-1)
        F22 = np.expand_dims(F22, axis=-1)

    C11 = F11**2 + F21**2
    C12 = F11*F12 + F21*F22
    C21 = F11*F12 + F21*F22
    C22 = F12**2 + F22**2

    I1 = C11 + C22 + 1.0
    I2 = C11 + C22 - C12*C21 + C11*C22
    I3 = C11*C22 - C12*C21
    Ia = C11*a1**2.0 + 2.0*C12*a1*a2 + C22*a2**2.0
    Ib = C11*b1**2.0 + 2.0*C12*b1*b2 + C22*b2**2.0

    I1 = torch.tensor(I1)
    I2 = torch.tensor(I2)
    I3 = torch.tensor(I3)
    Ia = torch.tensor(Ia)
    Ib = torch.tensor(Ib)


    x = computeFeatures_torch(I1, I2, I3, Ia, Ib).detach().cpu().numpy()

    return x
