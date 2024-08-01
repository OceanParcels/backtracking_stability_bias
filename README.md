# Stability Bias in Lagrangian Backtracking in Divergent Flows
Code accompanying the manuscript "Stability Bias in Lagrangian Backtracking in Divergent Flows" (authors: Daan Reijnders, Michael C. Denes, Siren Rühs, Øyvind Breivik, Tor Nordam and Erik van Sebille).

## Directory structure
 - `atlantic`: code for the experiments in the Atlantic sector of the [PSY4V3R1 product from Mercator Ocean International](https://www.mercator-ocean.eu/en/ocean-science/operational-systems/operational-systems/). Data is presumed to be locally available. `moi_run_divergence_Atlantic.py` provides the particle simulations (see also the `jobs` directory), and two of the notebooks contain the analysis for the FB and BF experiments (see manuscript). 
 - `channel`: code for the runs and analyses in the idealized ACC channel. Data is presumed to be locally available (the dataset can be generated with MITgcm using the [code from Reijnders et al. (2022)](https://doi.org/10.24416/UU01-RXA2PB)). `Lagrangian_model.py` contains the bespoke Lagrangian model code, with RK4 and analytical scheme codes. `channel_analysis.ipynb` contains the analyses, including figure code.
 - `idealized`: plots for the idealized analyses/plots from the 'Theory'-section. 
 - `jobs`: SLURM scripts with python commands for executing the simulations.

Do not hesitate to reach out to [@daanreijnders](https://github.com/daanreijnders) for questions.