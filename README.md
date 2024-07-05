# Tremor Regimes

+++ EDIT: Python update and associated adaptions to packages implemented, but not yet modified on github, which may impact functionality. July 2024. Repo needs to be updated to account for second publication. +++

This is a beta version of an algorithm designed to detect subtle changes in seismicity in volcanic environments. It is subject to modifications and may thus not work as intended on every OS.

Each folder and the codes within were assigned an individual copyright licence. Please refer to the resepective folder in this repository to get information on further use and distribution.

Last update: March 2023
Download of data | SOM | Regime Detection fully available. Feature analysis partly available.

Development of a similar approach with more options for data pre-processing and analysis is in progress - stay tuned.

Publication: 
Steinke, B., Jolly, A. D., Carniel, R., Dempsey, D. E., & Cronin, S. J. Identification of seismo‐volcanic regimes at Whakaari/White Island (New Zealand) via systematic tuning of an unsupervised classifier. Journal of Geophysical Research: Solid Earth, e2022JB026221.

Available at: 
https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/2022JB026221

For any enquiries, please contact *bastian.steinke@auckland.ac.nz*

## Installation
Ensure you have Anaconda Python 3.8 installed.

1. Clone the repo

```bash
git clone https://github.com/bste426/TremorRegimes
```

2. Open new Terminal window within each sub-section and create the corresponding conda environment

```bash

conda env create -f environment.yml

conda activate [name]
```

The installation has been developed and tested on Mac operating systems.

## General Disclaimers
1. This algorithm is not guaranteed to forecast future eruptions, it only may indicate varying volcanic states. In our paper, we discuss the conditions under which the forecast model is likely to perform poorly.

2. This software is not guaranteed to be free of bugs or errors. Most codes have a few errors and, provided these are minor, they will have only a marginal effect on accuracy and performance. That being said, if you discover a bug or error, please report this to *bastian.steinke@auckland.ac.nz*

## Acknowledgments
The algorithm presented here is a joint effort, specifically supported by David Dempsey, Roberto Carniel and Luca Barbui.
Underlying seismic data curtesy of GeoNet.
