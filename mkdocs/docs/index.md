# Bayesian-EUCLID Readme

This document provides guidelines to use the Bayesian-EUCLID code. It describes:
i.) the input syntax, ==files and parameters needed to discover the material model from data==,
ii.) The ==format of input== required to discover the material model from displacement data,
iii.) ==Interpreting the outputs.==

An example is used to illustrate the use of the Bayesian-EUCLID code to discover the material parameters for the Arruda Boyce
benchmark material used in [the Bayesian-EUCLID paper](https://arxiv.org/abs/2203.07422).

## The File to run :

Run the main_hss.py file found in 'drivers' folder. The arguments used for running the file are:
###i.) Benchmark material name,
The benchmark material name can be any of the following:
=="Holzapfel" "ArrudaBoyce" "Ogden3" "NeoHookeanJ2" "Isihara" "HainesWilson" "GentThomas" "Ogden"==
###ii.) Noise added to nodal displacements of the benchmark data,
This can be =="high"== or =="low"==
###iii.) A string indicating whether data is to be taken from dynamic simulations (dyn-euclid-master-data folder) or from
quasistatic simulations (euclid-master-data). This is an optional argument

The format of input

The format of output - Plots and mean theta.
