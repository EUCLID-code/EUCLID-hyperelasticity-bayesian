#File description

This is the main file to run launch the EUCLID-Bayesian code for material model discovery.

The arguments to run the file are:
1. Benchmark material Name. Can be any one of the following: **ArrudaBoyce, Holzapfel, ArrudaBoyce, Ogden3, NeoHookeanJ2, Isihara, HainesWilson, GentThomas, Ogden**)
2. Noise conditioning of the data (can be **high**, **low** or **none**)
3. Whether the data is to be taken from the `euclid-master-data` folder (quasistatic) or the `dyn-euclid-master-data` folder (dynamic). Any string provided in this argument will result in the code selecting the _dynamic_ data. Leave this argument blank if _quasistatic_ data is to be selected for further analysis.
