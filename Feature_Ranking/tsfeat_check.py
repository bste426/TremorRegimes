#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 11:17:56 2020

@author: bst
"""

import pandas as pd
import numpy as np
import tsfresh

### LOAD DATA (for spec. time)
print('Initialising...')

cl_vector = pd.read_csv('Clusters.csv')
VCL = cl_vector.iloc[:,0]

filename = 'whakaari_043200.00wndw_rsam_10_5.00-10.00_features.csv'
df = pd.read_csv(filename,header=None, skiprows=1)

startdate = '2008-05-27'
enddate = '2020-11-18'

time_np = df[0]
dates = list(time_np)

newdates = [x[:-9] for x in dates]
for row in newdates:
    if startdate == row:
        to_start = np.array(newdates.index(startdate))+1
        break
for row in newdates:
    if enddate == row:
        to_end = np.array(newdates.index(enddate))
        break
to_end = to_end - to_start
df = pd.read_csv(filename, header=None, skiprows=(int(to_start)), nrows=(int(to_end)))
header = pd.read_csv(filename)
header = header.columns

### CHECK WHETHER COLUMN CONTAINS ONLY IDENTICAL FEATURE VALUES

def is_unique(s):
    a = s.to_numpy()  # s.values (pandas<0.24)
    return (a[0] == a).all()


### CALCULATE P-VALUES
print('Calculating P-Values...')

RMAT = []
names = []
rowname = ['p-value']

for t in range(np.shape(df)[1]-1):
    if is_unique(df[t+1]):
        continue
    else:
        FTS = df.iloc[:,t+1]
        features = header[t+1]
        names.append(features)
        Rval = tsfresh.feature_selection.significance_tests.target_binary_feature_real_test(FTS, VCL, 'mann')
        RMAT.append(Rval)


### SAVE TABLE

RMAT = sorted(RMAT, reverse = True)
RMAT = pd.DataFrame(RMAT, columns=rowname, index=names)
RMAT = RMAT[1:20]
RMAT.to_csv("RMAT.csv")

