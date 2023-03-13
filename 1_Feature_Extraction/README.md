# Feature Extraction

featbatch.py has 3 functions: 
(1) It downloads and processes miniseed earthquake data.
(2) It concatenates individual miniseed files to a continuous time stream.
(3) It generates feature matrices, which can be used as input for SOM classifications.

_init_.py is an accompaying file supporting featbatch.py. Both codes need to be executed within the correct conda envrionment (environment.yml).

## Disclaimers
General disclaimers are valid for all folders in this repository.

## Acknowledgments
This part of the algorithm was developed by David Dempsey and modified by Bastian Steinke. Please refer to David Demspey's MIT licence for codes in this folder.
Seismic data can be obtained using the GeoNet FDSN server.