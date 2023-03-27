#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 11:17:56 2020

@author: bst
"""

import pandas as pd
import numpy as np
import tsfresh
from imblearn.under_sampling import RandomUnderSampler
import logging
import matplotlib.pyplot as ppl
from datetime import datetime
import seaborn as sns
import os
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
from pandas.plotting import register_matplotlib_converters

pd.options.mode.chained_assignment = None
logging.getLogger('tsfresh').setLevel(logging.ERROR)
register_matplotlib_converters()

'''This script returns a list of statistically significant features for each of the regimes in the previous step. This could help interpreting regime changes 
as well well as discerning regimes that are similar or distinct from each other.'''

# Provide path to feature matrix:
path_to_matrix = '../1_Feature_Extraction/features/whakaari_043200.00wndw_rsam_10_2.00-5.00_data_features.csv'

# Provide path to cluster vector:
cl_path = '../2_SOM_classification/OUTPUT/cluster_vectors/clusters_whakaari_043200.00wndw_rsam_10_2.00-5.00_data_features_5cl_5x5_2011-06-01_2012-06-01.csv'

# Provide path to regime list:
path_to_regime_list = '../3_Regime_Detection/OUTPUT/regimes/regimes_corr_2011-06-01_2012-05-31_bw0.05.csv'


#############################
###### S E T T I N G S ######
#############################

# Feature ranking?
'''This is the main function.'''
feat_check = True

# Correlation matrix?
'''Optional: Get info on correlation between features.'''
corrmat = False #currrently not implemented

# Would you like to plot individual feature values as TS? Which one?
'''Optional: Visualise individual features to compare with other time series, e.g. RSAM.'''
plot_fval = False
if plot_fval is True:
    feature_number = range(759)     # range(759) for all features

# Change start and end date of analysis:
startdate = '2011-06-01'
enddate = '2012-06-01'  # the first time window excluded after the set of windows of interest

n_clusters = 5      # total number of clusters


###################################################################################################################################################
'''More settings to adapt analysis to specific needs.'''

# Plot correlation matrices and dendrogram?
plot_featureclusters = False

# threshold for feature clustering (can be max. distance value or max. no. of clusters, depends on criterion)
threshold = 0.5
criterion = 'distance' # criterion = 'inconsistent', 'distance', 'maxclust', 'monocrit', 'maxclust_monocrit'
method = 'ward' # methods = 'single','complete','average','median','centroid','ward','weighted'
    
# Chose X for top_X features (consider only top X features in feature ranking):
top_X = 30

# Do you want to include Top X features in the dendrogram based on value (< 0.01) or label ('True')?
value_based = False

# For comparison of two regimes with similar Feature Ranks:
FeatValComp = False
feature_comparison = [195,167]   # compare these features with each other (mean and std)

# Apply undersampling of data during BHT?
undersampling = True

###################################################################################################################################################

'''
### CHECK WHETHER COLUMN CONTAINS ONLY IDENTICAL FEATURE VALUES
def is_unique(s):
    a = s.to_numpy()  # s.values (pandas before version 0.24)
    return (a[0] == a).all()
'''

###################################
### I N I T I A L I S A T I O N ###
###################################

### LOAD DATA ###
print('Initialising matrix...')

df = pd.read_csv(path_to_matrix, header=None, skiprows=1)

if plot_fval is True:

    time_np = df[0]
    dates = list(time_np)

    newdates = [x[:-9] for x in dates]
    for row in newdates:
        if startdate == row:
            to_start = np.array(newdates.index(startdate)) + 1
            break
    for row in newdates:
        if enddate == row:
            to_end = np.array(newdates.index(enddate))
            break

    to_end = to_end - to_start
    data = pd.read_csv(path_to_matrix, header=None, skiprows=(int(to_start)), nrows=(int(to_end)))

    data = pd.DataFrame.to_numpy(data)
    time = data[:,0]
    time_np2 = []
    for x in range(len(time)):
        time_np2.append(datetime.strptime(str(time[x]), '%Y-%m-%d %H:%M:%S'))
    time_np = np.array(time_np2)

    # PLOT

    for f in range(len(feature_number)):

        ppl.figure(figsize=(15,5))
        valuedata = data[:,feature_number[f]]
        ppl.plot(time_np,valuedata,color='cyan',label=('FeatValue_'+str(feature_number[f])))
        ppl.xlabel('Time')
        ppl.ylabel('FeatValue')
        ppl.margins(x=0)

        # add events to plot
        with open('../1_Feature_Extraction/RSAM/eruptive_periods.txt', 'r') as fp:
            tes = [ln.rstrip() for ln in fp.readlines()]
        xcoords = tes
        for xc in xcoords:
            ppl.axvline(x = xc, color='k', linestyle='-', linewidth=2, label='_')
        
        with open('../1_Feature_Extraction/RSAM/activity.txt', 'r') as fp:
            act = [ln.rstrip() for ln in fp.readlines()]
        cords = act
        for co in cords:
            ppl.axvline(x = co, color='dimgrey', linestyle='--', linewidth=2, label='_')

        ppl.axvline(x='2012-08-04 16:52:00', color='k', linestyle='-', linewidth=2, label='eruption')
        ppl.axvline(x='2012-09-02 00:00:00', color='dimgrey', linestyle='--', linewidth=2, label='ash emission')
        ppl.axvline(x='2012-11-24 00:00:00', color='dimgrey', linestyle=':', linewidth=2, label='observation of lava dome')

        # FINISH #
        ppl.title('Feat_value', loc='center')
        ppl.legend()
        ppl.savefig('RSAMs/featvalue_'+str(feature_number[f])+'.png', dpi=400)

if corrmat is True:

    print('Calculating Correlation Matrix...')

    # - visualise ranks of all features/feature classes
    # - based on visualisation: assign TYPE to regimes

    # Compute correlation matrix
    df_corr = pd.read_csv(path_to_matrix, low_memory=False)
    df_corr = pd.DataFrame(df_corr)
    d = df_corr.loc[:, (df_corr != df_corr.iloc[0]).any()]  # only non-constant features, drop the rest

    corr = d.corr().abs()  # calculates correlation matrix; methods = ‘pearson’, ‘kendall’, ‘spearman’
    corr = pd.DataFrame(corr)

    # Find out ideal cl_number for features:
    # cl_det(som, cl, mapname, plot_curves, degree)

    # Plot correlation matrix

    if plot_featureclusters is True:
        if not os.path.isdir('OUTPUT/corrmx'):
            os.makedirs('OUTPUT/corrmx')
        corr.to_csv('OUTPUT/corrmx/corrmx.csv')

        # Generate a mask for the upper triangle
        mask = np.triu(np.ones_like(corr, dtype=bool))

        # Set up the matplotlib figure
        f, ax = ppl.subplots(figsize=(50, 50))

        # Generate a custom diverging colormap
        cmap = sns.diverging_palette(230, 20, as_cmap=True)

        # Draw the heatmap with the mask and correct aspect ratio
        sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, center=0,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5})

        ax.set_yticklabels([])
        ax.set_xticklabels([])

        ppl.savefig(
            'OUTPUT/corrmx/corrmat_all.png',
            dpi=400)
    
    # FEATURE CLUSTERING (https://www.kaggle.com/sgalella/correlation-heatmaps-with-hierarchical-clustering)

    correlations = corr
    dissimilarity = np.clip((1 - abs(correlations)), 0, 1)
    Z = linkage(squareform(dissimilarity), method)
    # methods = 'ward','median','centroid','weighted','average','complete','single'
    labels = fcluster(Z, threshold, criterion=criterion)  # here, all features are assigned to a cluster
    # criterion = 'inconsistent', 'distance', 'maxclust', 'monocrit', 'maxclust_monocrit'

    if plot_featureclusters is True:
        print('Plotting Dendrogram and Clustermap...')
        # Dendrogram
        ppl.figure(figsize=(20, 12))
        dendrogram(Z, labels=correlations.columns, orientation='top', leaf_rotation=90)
        ppl.tick_params(axis='x', which='major', labelsize=1)
        if not os.path.isdir('OUTPUT/feat_clusters'):
            os.makedirs('OUTPUT/feat_clusters')
        ppl.savefig('OUTPUT/feat_clusters/dendrogram.png', dpi=600)

        # Heatmap
        labels_order = np.argsort(labels)
        for idx, i in enumerate(correlations.columns[labels_order]):
            if idx == 0:
                clustered = pd.DataFrame(correlations[i])
            else:
                df_to_append = pd.DataFrame(correlations[i])
                clustered = pd.concat([clustered, df_to_append], axis=1)
        correlations2 = clustered.corr()
        fig, ax = ppl.subplots(figsize=(20,20))
        sns.set(font_scale=.1)
        # SELECT: CORRELATION HEATMAP (sns.heatmap) OR SORTED HEATMAP WITH DENDROGRAM (sns.clustermap)
        #ax = sns.heatmap(round(correlations2, 2), cmap='RdBu', annot=True, annot_kws={"size": 7}, vmin=-1, vmax=1)
        ax = sns.clustermap(correlations2, method=method, cmap='RdBu', figsize = (15, 12))#, row_cluster = False, dendrogram_ratio = (.1, .2), cbar_pos = (0, .2, .03, .4), vmin=-1, vmax=1)
            # methods = 'ward', 'median', 'centroid', 'weighted', 'average', 'complete', 'single'
        if not os.path.isdir('OUTPUT/feat_clusters'):
            os.makedirs('OUTPUT/feat_clusters')
        ppl.savefig('OUTPUT/feat_clusters/clustermap.png', dpi=300)

if feat_check is True:
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
    df = pd.read_csv(path_to_matrix, header=None, skiprows=(int(to_start)), nrows=(int(to_end))+1)
    header = pd.read_csv(path_to_matrix, index_col=0, nrows=0).columns.tolist()

    cl_vector = pd.read_csv(cl_path, header = None)

    cl_vector.columns = (['cluster','time'])
    data = cl_vector['cluster']
    time_np = df[0]
    dates = list(time_np)

    ### CLUSTER ADAPTION TO BINARY ###
    print('ADAPTING CLUSTER VECTOR...')

    regime_list = pd.read_csv('../3_Regime_Detection/OUTPUT/regimes/regimes_corr_2011-06-01_2012-05-31_bw0.05.csv')

    # make sure end date is included in regime_list
    if list(regime_list['time'])[-1]!=list(cl_vector['time'])[-1]:
        swap = list(np.array(cl_vector)[-1])
        swap[0],swap[1]=swap[1],swap[0]
        regime_list.loc[len(regime_list.index)] = swap 

    reg_times = regime_list['time']
    dom_cluster = regime_list['dom_cluster']

    true_cols = []
    F1_list = []
    F2_list = []
    F1_err_list = []
    F2_err_list = []
    top_X_list = []

    # prepare dataframe to fill with Top X features
    feature_rank_list = pd.DataFrame(columns = ['Regime {:d}'.format(x+1) for x in range(len(reg_times)-1)])

    # for each regime:
    for i in range(len(reg_times)-1):
        print('############# Setting up '+str(i+1)+'th regime #############')

        #### PREPARE BINARY CLUSTER VECTOR FOR BHT ####

        reg_start_date = reg_times[i]
        reg_end_date = reg_times[i+1]

        DKC = dom_cluster[i]  # dominant cluster

        # prepare list of clusters to set 0
        vector = [x for x in range(n_clusters) if x != DKC]    # contains all but dominant cluster
        data_mod = data.replace(DKC, n_clusters + 1).replace(vector, 0).replace(n_clusters + 1, 1) # sets dominant cluster to 1, all other to 0
        VCL = pd.DataFrame(data_mod)    # final cluster vector
        VCL.insert(1, "Time", time_np, True) # add time column to cluster column

        for row in dates:
            if reg_start_date == row:
                to_start_reg = np.array(dates.index(reg_start_date))      # to find regime start date in matrix
                break
        for row in dates:
            if reg_end_date == row:
                to_end_reg = np.array(dates.index(reg_end_date))          # to find regime end date in matrix
                break

        # all windows outside regime
        VCL_in = VCL[int(to_start_reg):int(to_end_reg)] # all windows inside the regime
        indexNames = VCL_in[VCL_in['cluster'] == 0].index
        VCL_in.drop(indexNames, inplace=True)           # discard all windows of non-dominant cluster inside the regime

        # all windows outside regime
        VCL_out = VCL
        VCL_out.drop(VCL_out[int(to_start_reg):int(to_end_reg)].index, inplace=True)    # all windows outside regime
        indexNames = VCL_out[VCL_out['cluster'] == 1].index
        VCL_out.drop(indexNames, inplace=True)          # discard all windows of dominant cluster outside the regime

        lists = [VCL_in, VCL_out]
        VCL_reg = pd.concat(lists, ignore_index=True).sort_values('Time')
        VCL_reg.reset_index(drop=True, inplace=True)                           # all windows for BHT, but need to assign values from feature matrix
        dlen = df.shape
        matrix = pd.DataFrame(data=df.iloc[0:dlen[0], 1:dlen[1]])              # feature matrix (values only)

        # need to kick out rows in matrix which are not in the cluster vector anymore
        visitors = df
        orders = VCL_reg
        nonorders = visitors.loc[~visitors[0].isin(orders['Time']),].index
        matrix = matrix.drop(nonorders)                     # matrix to use for BHT

        VCL_reg.drop('Time',inplace = True, axis = 1)       # delete time column in cluster vector
        test = pd.DataFrame(VCL_reg)
        VCL_fin = test.iloc[:,0]                            # final binary vector for BHT
        

        if FeatValComp is True:

            matrix = pd.DataFrame(data=df.iloc[0:dlen, 0:df.shape[1]])              # feature matrix
            matrix = matrix[int(to_start_reg):int(to_end_reg)]                      # focus on regime only
            nonorders = matrix.loc[~matrix[0].isin(VCL_in['Time']),].index          # kick out all non-cluster windows
            matrix_red = matrix.drop(labels = nonorders)                            # kicked out
            matrix_red = pd.DataFrame(data=matrix_red.iloc[0:dlen, 1:df.shape[1]])  # kick out time vector

            # Calculates the mean value of the two key features from each group alpha to epsilon
            F1 = np.mean(matrix_red[feature_comparison[0]])
            F2 = np.mean(matrix_red[feature_comparison[1]])
            F1_err = np.std(matrix_red[feature_comparison[0]])
            F2_err = np.std(matrix_red[feature_comparison[1]])
            F1_list.append(F1)
            F2_list.append(F2)
            F1_err_list.append(F1)
            F2_err_list.append(F2)

        ### CALCULATE P-VALUES ###
        print('Calculating P-Values')

        RMAT = []
        BHT = []
        names = []
        rowname = ['p-value']

        # Undersampling data (few active windows for specific clusters vs 100k windows produces high FPR)
        if undersampling is True:
            rus = RandomUnderSampler(0.5)
            vals, labels = rus.fit_resample(matrix, VCL_fin)
            labels = pd.Series(labels, index=range(len(labels)))
            vals.index = labels.index

            # Benjamini-Hochberg-Test
            BHval = tsfresh.feature_selection.relevance.calculate_relevance_table(vals, labels, ml_task='classification')
            BHT = pd.DataFrame(BHval)

        else:
            labels_o = pd.Series(VCL_fin, index=range(len(VCL_fin)))
            matrix.index = labels_o.index
            BHval = tsfresh.feature_selection.relevance.calculate_relevance_table(matrix, labels_o, ml_task='classification')
            BHT = pd.DataFrame(BHval)

        for z in range(len(header) + 1):
            BHT = BHT.rename(columns={'feature': 'feature_number'}, index={z: str(header[z - 1])})
        if not os.path.isdir('OUTPUT/BHT'):
            os.makedirs('OUTPUT/BHT')
        BHT.to_csv('OUTPUT/BHT/Regime_' + str(i + 1) + '.csv')

        # Append clusters of (Top X) features to list:
        feature_rank_list['Regime {:d}'.format(i+1)] = list(BHT)[:top_X]

        # Prepare list of Top-X features across ALL regimes (used below)
        for z in range(len(BHT.index[:top_X])):
            if value_based is False:
                if (BHT['relevant'][z]) == True:
                    top_X_list.append(BHT.index[:top_X][z])
            else:
                if (BHT['p_value'][z]) < 0.01:
                    top_X_list.append(BHT.index[:top_X][z])

        # Create list of "True" columns (for histogram plots)
        R = i+1
        if value_based is False:
            if (BHT['relevant'][0]) == True:
                string = 'Regime {:d}'.format(R)
                true_cols.append(string)
        else:
            if (BHT_cl['p_value'][0]) < 0.01:
                string = 'Regime {:d}'.format(R)
                true_cols.append(string)

    # Convert table for Top X features for each regime to csv
    ppl.close()
    ppl.close()
    ppl.close()
    if not os.path.isdir('OUTPUT/ranked_features'):
        os.makedirs('OUTPUT/ranked_features')
    feature_rank_list.to_csv('OUTPUT/ranked_features/FRL.csv', index = None)

    # Find out what the top X feature across all regimes are and find them in dendrogram
    '''
    uniques = []
    [uniques.append(x) for x in top_X_list if x not in uniques]
    ppl.figure(figsize=(20, 12))
    CC = list(correlations.columns)
    for y in range(len(CC)):
        if CC[y] not in uniques:
            CC[y] = '-'
    pd.Index(CC)
    dendrogram(Z, labels=CC, orientation='top', leaf_rotation=90)
    ppl.tick_params(axis='x', which='major', labelsize=1)
    ppl.axhline(y=threshold, color = 'black', ls = '--', linewidth=1)
    if not os.path.isdir('OUTPUT/dendrogram'):
        os.makedirs('OUTPUT/dendrogram')
    ppl.savefig('OUTPUT/dendrogram/dendrogram_top{:d}_features.png'.format(top_X), dpi=600)
    

    # Plot histograms of each regime based on distribution of feature clusters within the Top X
    bins_list = [103.5, 104.5, 105.5, 106.5, 107.5, 108.5, 109.5, 110.5, 111.5, 112.5]
    ax_list = feature_rank_list.hist(column = true_cols, grid = False, xlabelsize=4, ylabelsize=4, bins=bins_list, rwidth=0.9)
    [x.title.set_size(6) for x in ax_list.ravel()]
    for i in range((np.shape(ax_list))[0]):
        for j in range((np.shape(ax_list))[1]):
            try:
                ax_list[i][j].set_xlim((104, 112))
                ax_list[i][j].set_ylim((0, 30))
                ax_list[i][j].set_xticks(np.linspace(104, 112, 9))
                ax_list[i][j].set_yticks(np.linspace(0, 30, 7))
            except:
                continue
    ppl.tight_layout()
    if not os.path.isdir('OUTPUT/histogram'):
        os.makedirs('OUTPUT/histogram')
    ppl.savefig('OUTPUT/histogram/featcluster_hist.png', dpi=400)
    '''

    if FeatValComp is True:
        # Plot histograms of features values for the two key features in each group to distinguish regimes
        fig, ax1 = ppl.subplots(figsize=(3, 2))
        regs = ['Regime 8','Regime 15']#,'Regime 9','Regime 13','Regime 16','Regime 17']

        color = 'tab:red'
        ax1.set_ylabel('cwt_coeff_ \n w_7_20', color=color, fontsize=10)
        #ax1.scatter(regs, F1_list, c=color, marker='.', alpha=0.25)
        ax1.errorbar(regs, F1_list, yerr=F1_err_list, color=color, marker='.', markersize=10, linestyle="None", elinewidth=0.5, ecolor=color, alpha = 0.5)
        ppl.xticks(rotation=90)
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        color = 'tab:blue'
        ax2.set_ylabel('cwt_coeff_ \n w_14_20', color=color, fontsize=10)  # we already handled the x-label with ax1
        #ax2.scatter(regs, F2_list, c=color, marker='.', alpha=0.25)
        ax2.errorbar(regs, F2_list, yerr=F2_err_list, color=color, marker='.', markersize=10, linestyle="None", elinewidth=0.5, ecolor=color, alpha = 0.5)

        ppl.tight_layout()
        if not os.path.isdir('OUTPUT/featvalcomp'):
            os.makedirs('OUTPUT/featvalcomp')
        ppl.savefig('OUTPUT/featvalcomp/featvalcomp.png', dpi=400)
