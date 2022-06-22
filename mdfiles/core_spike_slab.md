# `class Chain:`

Refer to the hierarchical Bayesian model discussed in [the Bayesian-EUCLID paper](https://arxiv.org/abs/2203.07422) (Fig. 2) for details regarding the parameters.

_Attributes_

-`p0` - Numpy array of length `chain_length` containing different values of $p_0$ in the chain

-`vs` - Numpy array of length `chain_length` containing different values of $\nu_s$ in the chain

-`sig2` - Numpy array of length `chain_length` containing different values of $\sigma^2$ in the chain

-`z` - Numpy array of dimension `chain_length`X`numFeatures` (see `features_library.py`) containing different values of z (_activity_) in the chain

-`theta` - Numpy array of dimension `chain_length`X`numFeatures` (see `features_library.py`) containing different values of theta (_feature coefficients_) in the chain

-`chain_length`

-`burn` - Number of elements of the chain discarded as _burn in_ in sampling the posterior probability distribution.

_Methods_

-`__init__(...):` - Generates an object of class `Chain`

-`update_state(...):` - Populates the chain with newly sampled state variables

-`combine_chain(...):` - Combines different parallelly generated Markov chains

-`burn_chain(...):` - Deletes the first `burn` number of elements of the chains

---

#`class Data:`

#`class Params:`

#`class State:`
