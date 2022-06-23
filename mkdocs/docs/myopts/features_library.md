#### `computeFeatures_numpy(F11, F12, F21, F22):`



#### `computeFeatures_torch(I1, I2, I3, Ia, Ib):`

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
   

#### `get_mass_type():`



#### `get_theta_gt(fem_path, feature_filter):`

Returns the true feature coefficients corresponding to the benchmark material.


#### `getNumberOfFeatures():`

Compute number of features.

_Input Arguments_

- _none_
   
_Output Arguments_

- `features.shape[1]` - number of features
   
---
   