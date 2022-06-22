#### `computeFeatures_numpy(F11, F12, F21, F22):`

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


#### `computeFeatures_torch(I1, I2, I3, Ia, Ib):`

Compute the features dependent on the right Cauchy-Green strain invariants.
Note that the features only depend on I1 and I3 for plane strain assumption.

_Input Arguments_

- `I1` - 1st invariant
   
- `I2` - 2nd invariant
   
- `I3` - 3rd invariant
   
- 'Ia' - Invariant along anisotropy direction.
   
- 'Ib' - Invariant along second anisotropy direction.
   
_Output Arguments_

- `x` - features
   
---
   

#### `get_mass_type():`

mass_type = "lumped" #Can be made 'consistent' otherwise

return mass_type


computeFeatures_torch(I1, I2, I3, Ia, Ib):

#### `get_theta_gt(fem_path, feature_filter):`

Returns the true feature coefficients corresponding to the benchmark material.


#### `getNumberOfFeatures():`

Compute number of features.

_Input Arguments_

- _none_
   
_Output Arguments_

- `features.shape[1]` - number of features
   
---
   