import torch

from config import *

def computeJacobian(F):
    """
    Compute Jacobian from deformation gradient.

    _Input Arguments_

    - `F` - deformation gradient in Voigt notation

    _Output Arguments_

    - `J` - Jacobian

    ---

    """
    F11 = F[:,0:1]
    F12 = F[:,1:2]
    F21 = F[:,2:3]
    F22 = F[:,3:4]

    J = F11*F22 - F12*F21
    return J

def computeCauchyGreenStrain(F):
    """
    Compute right Cauchy-Green strain tensor from deformation gradient.

    _Input Arguments_

    - `F` - deformation gradient in Voigt notation

    _Output Arguments_

    - `C` - Cauchy-Green strain tensor in Voigt notation

    ---

    """
    F11 = F[:,0:1]
    F12 = F[:,1:2]
    F21 = F[:,2:3]
    F22 = F[:,3:4]

    C11 = F11**2 + F21**2
    C12 = F11*F12 + F21*F22
    C21 = F11*F12 + F21*F22
    C22 = F12**2 + F22**2

    C = torch.cat((C11,C12,C21,C22),dim=1)
    return C


def computeStrainInvariants(C):
    """
    Compute invariants of the Cauchy-Green strain tensor.
    Plane strain is assumed.

    _Input Arguments_

    - `C` - Cauchy-Green strain tensor in Voigt notation

    _Output Arguments_

    - `I1` - 1st invariant

    - `I2` - 2nd invariant

    - `I3` - 3rd invariant

    ---

    """
    C11 = C[:,0:1]
    C12 = C[:,1:2]
    C21 = C[:,2:3]
    C22 = C[:,3:4]

    I1 = C11 + C22 + 1.0
    I2 = C11 + C22 - C12*C21 + C11*C22
    I3 = C11*C22 - C12*C21
    return I1, I2, I3


def computeStrainInvariantDerivatives(F,i,secondDerivative=False):
    """
    Compute derivatives of the invariants of the Cauchy-Green strain tensor with respect to the deformation gradient.
    Plane strain is assumed.

    _Input Arguments_

    - `F` - deformation gradient in Voigt notation

    - `i` - specify the invariant that should be differentiated- I1, I2, I3, Ia or Ib

    - `secondDerivative` - specify if second derivative should be computed

    _Output Arguments_

    - `dIdF` - derivative (note that the size of `dIdF` depends on the choice of `secondDerivative`)

    ---

    """
    F11 = F[:,0:1]
    F12 = F[:,1:2]
    F21 = F[:,2:3]
    F22 = F[:,3:4]
    if not(secondDerivative):
        dIdF = torch.zeros(F.shape[0],F.shape[1])
        if(i==1):
            # dI1/dF:
            dIdF = 2.0*F
        elif(i==2):
            # dI2/dF:
            dIdF11 = 2.0*F11 - 2.0*F12*F21*F22 + 2.0*F11*(F22**2)
            dIdF12 = 2.0*F12 + 2.0*F12*(F21**2) - 2.0*F11*F21*F22
            dIdF21 = 2.0*F21 + 2.0*(F12**2)*F21 - 2.0*F11*F12*F22
            dIdF22 = 2.0*F22 - 2.0*F11*F12*F21 + 2.0*(F11**2)*F22
            dIdF = torch.cat((dIdF11,dIdF12,dIdF21,dIdF22),dim=1)
        elif(i==3):
            # dI3/dF:
            J = F11*F22 - F12*F21
            dIdF11 = 2.0*F22 * J
            dIdF12 = -2.0*F21 * J
            dIdF21 = -2.0*F12 * J
            dIdF22 = 2.0*F11 * J
            dIdF = torch.cat((dIdF11,dIdF12,dIdF21,dIdF22),dim=1)
        elif(i==4):
            dIdF11 = 2.0*F11*a1**2.0 + 2.0*F12*a1*a2
            dIdF12 = 2.0*F11*a1*a2 + 2.0*F12*a1**2.0
            dIdF21 = 2.0*F22*a1*a2 + 2.0*F21*a1**2.0
            dIdF22 = 2.0*F22*a2**2.0 + 2.0*F21*a1*a2
            dIdF = torch.cat((dIdF11,dIdF12,dIdF21,dIdF22),dim=1)
        elif(i==5):
            dIdF11 = 2.0*F11*b1**2.0 + 2.0*F12*b1*b2
            dIdF12 = 2.0*F11*b1*b2 + 2.0*F12*b1**2.0
            dIdF21 = 2.0*F22*b1*b2 + 2.0*F21*b1**2.0
            dIdF22 = 2.0*F22*b2**2.0 + 2.0*F21*b1*b2
            dIdF = torch.cat((dIdF11,dIdF12,dIdF21,dIdF22),dim=1)
        else:
            raise ValueError('Incorrect invariant index')
    if secondDerivative:
        dIdF = torch.zeros(F.shape[1],F.shape[1])
        if(i==1):
            # d(dI1/dF)/dF:
            dIdF = 2.0*torch.eye(F.shape[1])
        elif(i==3):
            # d(dI3/dF)/dF:
            J = F11*F22 - F12*F21
            dJdF11 = F22
            dJdF12 = - F21
            dJdF21 = - F12
            dJdF22 = F11
            # d(dI3/dF)/dF11:
            dIdF[0,0] = 2.0 * F22 * dJdF11
            dIdF[0,1] = -2.0 * F21 * dJdF11
            dIdF[0,2] = -2.0 * F12 * dJdF11
            dIdF[0,3] = 2.0 * J + 2.0 * F11 * dJdF11
            # d(dI3/dF)/dF12:
            dIdF[1,0] = 2.0 * F22 * dJdF12
            dIdF[1,1] = -2.0 * F21 * dJdF12
            dIdF[1,2] = -2.0 * J -2.0 * F12 * dJdF12
            dIdF[1,3] = 2.0 * F11 * dJdF12
            # d(dI3/dF)/dF21:
            dIdF[2,0] = 2.0 * F22 * dJdF21
            dIdF[2,1] = -2.0 * J + -2.0 * F21 * dJdF21
            dIdF[2,2] = -2.0 * F12 * dJdF21
            dIdF[2,3] = 2.0 * F11 * dJdF21
            # d(dI3/dF)/dF22:
            dIdF[3,0] = 2.0 * J + 2.0 * F22 * dJdF22
            dIdF[3,1] = -2.0 * F21 * dJdF22
            dIdF[3,2] = -2.0 * F12 * dJdF22
            dIdF[3,3] = 2.0 * F11 * dJdF22
        elif(i==4):
            dIdF[0,0]  =  2.0*pow(a1,2.0) ;
            dIdF[0,1]  =  2.0*a1*a2 ;
            dIdF[0,2]  =  0.0 ;
            dIdF[0,3]  =  0.0 ;


            dIdF[1,0]  =  2.0*a1*a2 ;
            dIdF[1,1]  =  2.0*pow(a2,2.0) ;
            dIdF[1,2]  =  0.0 ;
            dIdF[1,3]  =  0.0 ;


            dIdF[2,0]  =  0.0 ;
            dIdF[2,1]  =  0.0 ;
            dIdF[2,2]  =  2.0*pow(a1,2.0) ;
            dIdF[2,3]  =  2.0*a1*a2 ;


            dIdF[3,0]  =  0.0 ;
            dIdF[3,1]  =  0.0 ;
            dIdF[3,2]  =  2.0*a1*a2 ;
            dIdF[3,3]  =  2.0*pow(a2,2.0) ;

        elif(i==5):
            dIdF[0,0]  =  2.0*pow(b1,2.0) ;
            dIdF[0,1]  =  2.0*b1*b2 ;
            dIdF[0,2]  =  0.0 ;
            dIdF[0,3]  =  0.0 ;


            dIdF[1,0]  =  2.0*b1*b2 ;
            dIdF[1,1]  =  2.0*pow(b2,2.0) ;
            dIdF[1,2]  =  0.0 ;
            dIdF[1,3]  =  0.0 ;


            dIdF[2,0]  =  0.0 ;
            dIdF[2,1]  =  0.0 ;
            dIdF[2,2]  =  2.0*pow(b1,2.0) ;
            dIdF[2,3]  =  2.0*b1*b2 ;


            dIdF[3,0]  =  0.0 ;
            dIdF[3,1]  =  0.0 ;
            dIdF[3,2]  =  2.0*b1*b2 ;
            dIdF[3,3]  =  2.0*pow(b2,2.0) ;

        else:
            raise ValueError('Incorrect invariant index')
    return dIdF


def computeTriangleArea(p1, p2, p3):
    """
    Computing the area of a triangle with vertices at points p1, p2, p3 (2D).

    """
    vec_a = p2 - p1
    vec_b = p3 - p1

    a = torch.sqrt(vec_a[:,0:1]**2 + vec_a[:,1:2]**2)

    b = torch.sqrt(vec_b[:,0:1]**2 + vec_b[:,1:2]**2)

    dotprod = vec_a[:,0:1]*vec_b[:,0:1] + vec_a[:,1:2]*vec_b[:,1:2]

    cos_theta = dotprod/(a * b)

    sin_theta = torch.sqrt( 1 - cos_theta**2 )

    area = (0.5 * a * b * sin_theta).squeeze()

    return area
