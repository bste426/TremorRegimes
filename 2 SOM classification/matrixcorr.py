#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 15:49:03 2020

@author: bst
"""

import pandas as pd
import glob, os
import numpy as np

#decide whether correction based on value (auto) or date list (manual)
automatic = True
cutoff_min = 0.01
cutoff_max = 1E10
station = 'RDT'
#filename='/Users/bst/Desktop/whakaari_3600.00wndw_rsam_10_5.00-10.00_features.csv'
#df = pd.read_csv(filename,header=None, low_memory=False)

PATH = os.getcwd()
fl = '*.csv'
files = glob.glob(fl)


for f in range(len(files)):

    ###   READING DATA   ###
    print('reading file...')
    print(files[f])

    df = pd.read_csv(files[f], header=None, low_memory=False)

    ###   MANUALLY REMOVING DATA   ###
    if automatic is False:
        print('removing data...')
        with open(('shit_dates/{:s}.txt').format(station), 'r') as fp:
           deldates = [ln.rstrip() for ln in fp.readlines()]
        indices = []
        newdates = [x[:-9] for x in list(df[0])]

        for z in range(len(deldates)):
            indices2 = [i for i, x in enumerate(newdates) if x == deldates[z]]
            indices.append(indices2)

        print(df)
        print(indices)

        for x in range(len(indices)):
            df = df.drop(df.index[indices[x]])

    ###   AUTOMATICALLY REMOVING DATA   ###
    else:
        print('removing data...')
        df = pd.read_csv(files[f], low_memory=False)
        col_name = df.columns[1]
        df = df.drop(df.index[df[col_name] > cutoff_max])
        df = df.drop(df.index[df[col_name] < cutoff_min])



    ###   SAVING DATA   ###
    print('saving corrected file...')

    if not os.path.isdir('corrected_matrices'):
        os.makedirs('corrected_matrices')
    path = PATH+'/corrected_matrices/'+files[f][:-4]+'_corr.csv'
    df.to_csv(path, header=False, index = False)