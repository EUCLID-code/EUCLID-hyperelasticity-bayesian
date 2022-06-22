#### `post_proc(chain, theta_gt, feature_filter, fem_mat, energy_func, fig_title, fig_title2, fig_dir = None, plotting=True, interactive_job=True):`

Making output plots containing a summary of the chains, and the corresponding predicted energies

_Input Arguments_

-`chain` - object of `Chain` class (see `core_spike_slab` file)
   
-`theta_gt` - The true set of feature coefficients for the benchmark material
   
-`feature_filter` - The list of features to retain for constructing the Markov chain. Suppressed features will be highlighted with a red patch in the plot
   
-`fem_mat` - The name of the benchmark material to be tested
   
-`energy_func` - The label of the function used to predict energy evolution for the discovered material along 6 different deformation paths
   
-`fig_title` - Title displayed on the figure
   
-`fig_title2` - Filename of the saved figure (.png format)
   
---
   

#### `print_solution(theta):`

