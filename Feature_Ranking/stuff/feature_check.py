#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 11:17:56 2020

@author: bst
"""

import matplotlib.pyplot as ppl
import matplotlib.dates as mdates
import pandas as pd
import numpy as np

filename = '/Users/bst/Documents/All/PhD/Data/Codes/SOM_Carniel/csv2som/feature_matrices/12h/whakaari_043200.00wndw_rsam_10_5.00-10.00_features.csv'
df = pd.read_csv(filename,header=None, skiprows=1)

startdate = '2016-04-29'
enddate = '2020-11-17'

time_np = df[0]
#time_np = 41*365.25*24*60*60
#print(time_np)
#time_np = datetime.utcfromtimestamp(time_np)
dates = list(time_np)
#print(time_np)
try:
    newdates = [x[:-9] for x in dates]
    for row in newdates:
        if startdate == row:
            to_start = np.array(newdates.index(startdate))
            break
    for row in newdates:
        if enddate == row:
            to_end = np.array(newdates.index(enddate))
            break
    to_end = to_end - to_start
except:
    newdates = dates
    for row in newdates:
        if startdate == row:
            to_start = np.array(newdates.index(startdate))
            print(to_start)
            break
    for row in newdates:
        if enddate == row:
            to_end = np.array(newdates.index(enddate))
            print(to_end)
            break
    to_end = to_end - to_start
df = pd.read_csv(filename, header=None, skiprows=(int(to_start)), nrows=(int(to_end)))




for i in range(np.shape(df)[1]):
    y = np.array(df[i])
    time_np = np.array(df[0])
    tminimo = 0
    tmassimo = time_np.shape[0]
    
    fig, axarr = ppl.subplots(sharex=False, sharey=False)
    axarr.set_yticklabels([])
    axarr.axis('off')
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(time_np,y)
    ppl.title('x', loc='right')
    ppl.show()

