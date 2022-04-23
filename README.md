# Tremor Regimes
This is a beta version of an algorithm designed to detect subtle changes in seismicity in volcanic environments. It is subject to modifications and may thus not work as intended on every OS.
Each folder and the codes within were assigned an individual copyright licence. Please refer to the resepective folder in this repository to get information for further use and distribution.

More details and updates soon! Steinke et al. (submitted)
For any enquiries, please contact *bastian.steinke@auckland.ac.nz*

## Installation
Ensure you have Anaconda Python 3.8 installed.

1. Clone the repo

```bash
git clone https://github.com/bste426/TremorRegimes
```

2. CD into each repo and create the corresponding conda environment

```bash
cd whakaari

conda env create -f environment.yml

conda activate [name]
```

The installation has been developed and tested on Mac operating systems.

## Disclaimers
1. This algorithm is not guaranteed to forecast future eruptions, it only may indicate varying volcanic states. In our paper, we discuss the conditions under which the forecast model is likely to perform poorly.

2. This software is not guaranteed to be free of bugs or errors. Most codes have a few errors and, provided these are minor, they will have only a marginal effect on accuracy and performance. That being said, if you discover a bug or error, please report this to *bastian.steinke@auckland.ac.nz*

## Acknowledgments
The resulting algorithm presented here is a joint effort, specifically supported by David Dempsey, Roberto Carniel and Luca Barbui.
Underlying seismic data curtesy of GeoNet.