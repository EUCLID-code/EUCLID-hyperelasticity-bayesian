#File description

This file contains the parameters required for material discovery using Bayesian-EUCLID framework. Section 2 and Table A.2 in the [paper](https://arxiv.org/abs/2203.07422) discusses these parameters.

#Methods

- `get_loadsteps(...):` - Returns the loadsteps to be considered from the FEM data for the discovery process.

- `get_feature_filter(...):` - Returns `feature_filter`, which is a list of features to retain for constructing the Markov chain. Suppressed features will be highlighted with a red patch in the plot
