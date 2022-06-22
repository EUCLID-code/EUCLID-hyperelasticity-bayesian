#### `check_data_name_validity(fem_material,noise_level):`



#### `filter_raw_data(rng, raw_data, filter_value):`

Randomly sub-sampling `filter_value` degrees of freedom from the data available at all quadrature points in `raw_data`

_Input Arguments_

-`rng` - Random number generator
   
-`raw_data` - Contains `D` (derivatives of feature library) and `y` (inertia terms and reaction forces) evaluated at all quadrature points
   
-`filter_value` - Number of degrees of freedom subsampled from the original data
   
_Output Arguments_

- `RawData(A1, b1, A2, b2, dof_x, dof_y)` - Object of class `RawData` which is constructed from subsampled data
   
---
   

#### `get_data(rng, fem_dir, prefix, fem_material, noise_level, loadstep, feature_filter):`



#### `predict_energy_path(chain, theta_gt, fem_mat, feature_filter, deformation):`

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
   

#### `process_raw_data(raw_data_set, lambda_r):`

