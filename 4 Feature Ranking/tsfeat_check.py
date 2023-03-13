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

#from tsfresh.transformers import FeatureSelector
#import joblib

def main():

    # Suppress all kinds of super annoying but maybe important warnings:
    pd.options.mode.chained_assignment = None
    logging.getLogger('tsfresh').setLevel(logging.ERROR)
    register_matplotlib_converters()


    ############### SETTINGS ################


    # Correlation matrix?
    corrmat = True

    # Feature ranking?
    feat_check = True

    # Plot correlation matrices and dendrogram?
    plot_featureclusters = False

    # threshold for feature clustering (can be max. distance value or max. no. of clusters, depends on criterion)
    threshold = 0.5
    criterion = 'distance'
        # criterion = 'inconsistent', 'distance', 'maxclust', 'monocrit', 'maxclust_monocrit'
    method = 'ward'
        # methods = 'single','complete','average','median','centroid','ward','weighted'

    # Chose X for top_X features (consider only top X features in feature ranking):
    top_X = 30

    # Do you want to include Top X features in the dendrogram based on value (< 0.01) or label ('True')?
    value_based = False

    # Calculate histograms?
    FVC = False

    # Would you like to plot individual feature values as TS? Which one?
    plot_fval = False
    if plot_fval is True:
        feature_number = range(759)     # range(759) for all features

    # Apply undersampling of data during BHT?
    undersampling = False

    # Is the time period 2008-2020? If so, then 'False'.
    adapt_dates = True

    # Change start and end date of analysis:
    if adapt_dates is True:
        startdate = '2019-01-01'
        enddate = '2020-01-01'

    # Provide path to regime list:
    regime_list = pd.read_csv('/Users/bst/Documents/All/PhD/Data/Codes/regime_detection/kde/regimes/regimes_corr_2019-01-01_2020-01-01_bw0.19.csv', header=None)
    #regime_list = pd.read_csv('/Users/bst/Desktop/results/5.1 Automatic Regimes/2014_16_all/regimes/regimes_corr_2014-01-01_2016-06-01_bw0.08.csv', header=None)
    #regime_list = pd.read_csv('/Users/bst/Desktop/results/5.1 Automatic Regimes/2019_all/regimes/regimes_corr_2019-01-01_2020-01-01_bw0.13.csv', header=None)

    n_clusters = 5      # total number of clusters

    # Provide path to cluster vector:
    cl_path = '/Users/bst/Documents/All/PhD/Data/Codes/feature_ranking/cluster_vectors/Clusters 2016_20.csv'

    # For comparison of two regimes with similar Feature Ranks:
    FeatValComp = False
    feature_comparison = [195,167]   # compare these features with each other (mean and std)
    '''
    # 2019 g: [22,34,78,79,80,219,222,673,705,706,707,731]   # compare these features with each other (mean and std)
    # 2012 yg:[97,99,100,102,103,106,112,114,115,129,674]
    # 2012 g: [22,78,79,14,151,155,159,163,167,659,664,708]
    # 2016: [22,78,79,80,219,222,673,707,708,731]
    
    # Cross comparison (same clusters / FRs in different years):
    # 1 [1,22,78,79,80,167,219,222,673,706,707,708,731] (all R1s)
    # 2 [30,34,59,170,171,176,177,180,200,678,701,702] (R3-2012, R5-2019, R8-2019)
    # 3 [34,97,99,112,114,127,129,701,702,703,704] (R13-2012, R4-2019)
    # 4 [97,100,102,103,104,112,114,115,127,295] (R7-2012, R7-2016)
    '''

    '''
    # probably not needed anymore:
    # Apply undersampling to data?
    undersampling = False
    # Do you want to apply the feature ranking across all windows (if so: regime_all=True)?
    regime_all = False
    '''

    ### CHECK WHETHER COLUMN CONTAINS ONLY IDENTICAL FEATURE VALUES
    def is_unique(s):
        a = s.to_numpy()  # s.values (pandas before version 0.24)
        return (a[0] == a).all()

    ### LOAD DATA ###
    print('Initialising matrix...')

    filename = '/Users/bst/Documents/All/PhD/Data/Codes/SOM_Carniel/csv2som/feature_matrices/whakaari_043200.00wndw_rsam_10_2.00-5.00_data_features.csv'
    df = pd.read_csv(filename, header=None, skiprows=1)


    if plot_fval is True:

        if adapt_dates is True:
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
            data = pd.read_csv(filename, header=None, skiprows=(int(to_start)), nrows=(int(to_end)))

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

            # EVENTS DURING UNREST #

            # 2019
            #ppl.axvline(x='2019-12-09 01:11:00', color='r', linestyle='-', linewidth=2, label='eruption')

            # 2016
            ppl.axvline(x='2016-04-27 09:37:00', color='r', linestyle='-', linewidth=2, label='eruption')

            # 2012-2014
            '''
            ppl.axvline(x='2012-08-04 16:52:00', color='r', linestyle='-', linewidth=2)
            ppl.axvline(x='2013-08-19 16:52:00', color='r', linestyle='-', linewidth=2)
            ppl.axvline(x='2013-10-03 16:52:00', color='r', linestyle='-', linewidth=2)
            ppl.axvline(x='2013-10-11 16:52:00', color='r', linestyle='-', linewidth=2)
            ppl.axvline(x='2012-09-02 16:52:00', color='k', linestyle='--', linewidth=2)
            ppl.axvline(x='2013-01-15 16:52:00', color='k', linestyle='--', linewidth=2)
            ppl.axvline(x='2013-04-10 16:52:00', color='k', linestyle='--', linewidth=2)
            ppl.axvline(x='2012-11-24 00:00:00', color='grey', linestyle=':', linewidth=2)
            '''

            # FINISH #
            ppl.title('Feat_value', loc='center')
            ppl.legend()
            ppl.savefig('RSAMs/featvalue_'+str(feature_number[f])+'.png', dpi=400)

    if corrmat is True:

        print('Calculating Correlation Matrix...')

        # - visualise ranks of all features/feature classes
        # - based on visualisation: assign TYPE to regimes

        # Compute correlation matrix
        df_corr = pd.read_csv(filename, low_memory=False)
        df_corr = pd.DataFrame(df_corr)
        d = df_corr.loc[:, (df_corr != df_corr.iloc[0]).any()]  # only non-constant features, drop the rest

        corr = d.corr().abs()  # calculates correlation matrix; methods = ‘pearson’, ‘kendall’, ‘spearman’
        corr = pd.DataFrame(corr)

        # Find out ideal cl_number for features:
        # cl_det(som, cl, mapname, plot_curves, degree)

        # Plot correlation matrix

        if plot_featureclusters is True:
            if not os.path.isdir('corrmx'):
                os.makedirs('corrmx')
            corr.to_csv('/Users/bst/Documents/All/PhD/Data/Codes/regime_detection/correlation_matrix/corrmx/corrmx.csv')

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
                '/Users/bst/Documents/All/PhD/Data/Codes/regime_detection/correlation_matrix/corrmat_all/corrmat_all.png',
                dpi=400)
        


        # FEATURE CLUSTERING (https://www.kaggle.com/sgalella/correlation-heatmaps-with-hierarchical-clustering)

        correlations = corr
        dissimilarity = np.clip((1 - abs(correlations)), 0, 1)
        Z = linkage(squareform(dissimilarity), method)
        # methods = 'ward','median','centroid','weighted','average','complete','single'
        labels = fcluster(Z, threshold, criterion=criterion)  # here, all features are assigned to a cluster
        # criterion = 'inconsistent', 'distance', 'maxclust', 'monocrit', 'maxclust_monocrit'

        '''
        labels_list = pd.DataFrame(correlations.columns)
        labels_list['Cluster'] = labels
        labels_list.to_csv('labels_list.csv', index=False)  # List with clustered features
        '''

        if plot_featureclusters is True:
            print('Plotting Dendrogram and Clustermap...')
            # Dendrogram
            ppl.figure(figsize=(20, 12))
            dendrogram(Z, labels=correlations.columns, orientation='top', leaf_rotation=90)
            ppl.tick_params(axis='x', which='major', labelsize=1)
            if not os.path.isdir('OUTPUT'):
                os.makedirs('OUTPUT')
            ppl.savefig('OUTPUT/dendrogram.png', dpi=600)

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
            if not os.path.isdir('OUTPUT'):
                os.makedirs('OUTPUT')
            ppl.savefig('OUTPUT/clustermap.png', dpi=300)

    if feat_check is True:
        if adapt_dates is True:
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
            header = pd.read_csv(filename, index_col=0, nrows=0).columns.tolist()
        else:
            header = pd.read_csv(filename, index_col=0, nrows=0).columns.tolist()

        cl_vector = pd.read_csv(cl_path)
        data = cl_vector['0']

        time_np = df[0]
        dates = list(time_np)

        ### CLUSTER ADAPTION TO BINARY ###
        print('ADAPTING CLUSTER VECTOR...')

        reg_start = regime_list[0]
        reg_end = regime_list[1]
        key_cluster = regime_list[2]
        major_cluster = regime_list[3]
        minor_cluster = regime_list[4]

        true_cols = []

        F1_list = []
        F2_list = []
        F1_err_list = []
        F2_err_list = []

        top_X_list = []
        regime_cluster_lists = ['Regime {:d}'.format(x+1) for x in range(len(reg_start))]
        feature_rank_list = pd.DataFrame(columns = regime_cluster_lists)

        # for each regime:
        for i in range(len(reg_start)):
            print('############# Setting up '+str(i+1)+'th regime #############')

            reg_start_date = reg_start[i]
            reg_end_date = reg_end[i]

            DKC = (key_cluster[i])-1    # DKC...discrete key cluster for this regime (dominant cluster)
            MJC = (major_cluster[i])-1
            MNC = (minor_cluster[i])-1

            ### PREPARE CLUSTER VECTOR

            # prepare list of clusters to set 0
            vector = list(range(n_clusters))
            if major_cluster[i] == 0:
                vector = [x for x in vector if x != DKC]    # 1 dominant cluster
            else:
                if minor_cluster[i] == 0:
                    vector = [x for x in vector if x != DKC and x != MJC]    # 2 dominant clusters
                else:
                    vector = [x for x in vector if x != DKC and x != MJC and x != MNC]    # 3 dominant clusters

            vector = tuple(vector)

            # set all clusters to 0 apart from key_cluster (=> n_cluster+1): Depending on major/minor cluster exist
            if major_cluster[i] == 0:
                data_mod = data.replace(DKC, n_clusters + 1).replace(vector, 0).replace(n_clusters + 1, 1) # 1 dominant cluster
            else:
                if minor_cluster[i] == 0:
                    data_mod = data.replace(DKC, n_clusters + 1).replace(MJC, n_clusters + 1).replace(vector, 0).replace\
                        (n_clusters + 1, 1) # 2 dominant clusters
                else:
                    data_mod = data.replace(DKC, n_clusters + 1).replace(MJC, n_clusters + 1).replace(MNC, n_clusters + 1).\
                        replace(vector, 0).replace(n_clusters + 1, 1)  # 3 dominant clusters

            cl_vector2 = pd.DataFrame(data_mod)
            VCL = cl_vector2.iloc[:, 0]     # final cluster vector

            ### PREPARE BINARY CLUSTER VECTOR FOR BHT

            # kick out all windows assigned to discrete_key_cluster outside window and all other clusters inside window
            reg_dates = [x[:-9] for x in dates]

            for row in reg_dates:
                if reg_start_date == row:
                    to_start_reg = np.array(reg_dates.index(reg_start_date)) + 1
                    break
            for row in reg_dates:
                if reg_end_date == row:
                    to_end_reg = np.array(reg_dates.index(reg_end_date))
                    break

            VCL_reg = []
            VCL = pd.DataFrame(VCL)
            VCL.insert(1, "Time", time_np, True)    # add time column to cluster column


            # all windows inside the regime
            VCL_in = VCL[int(to_start_reg):int(to_end_reg)]
            indexNames = VCL_in[VCL_in['0'] == 0].index
            VCL_in.drop(indexNames, inplace=True)               # kick out all non-regime windows

            l = len(feature_comparison)
            feature_value_list = []
            dlen = df.shape[0]

            if FeatValComp is True:

                matrix = pd.DataFrame(data=df.iloc[0:dlen, 0:df.shape[1]])              # feature matrix
                matrix = matrix[int(to_start_reg):int(to_end_reg)]                      # focus on regime only
                nonorders = matrix.loc[~matrix[0].isin(VCL_in['Time']),].index          # kick out all non-cluster windows
                matrix_red = matrix.drop(labels = nonorders)                            # kicked out
                matrix_red = pd.DataFrame(data=matrix_red.iloc[0:dlen, 1:df.shape[1]])  # kick out time vector

                '''
                for f in range(len(feature_comparison)):
                    MEAN = np.mean(matrix_red[feature_comparison[f]])
                    STD = np.std(matrix_red[feature_comparison[f]])
                    new_array = ([feature_comparison[f],MEAN,STD])
                    feature_value_list.append(new_array)

                FVL = pd.DataFrame(feature_value_list)
                FVL.to_csv('FVLs/FVL_R' + str(i + 1) + '.csv')
                '''

                # Calculates the mean value of the two key features from each group alpha to epsilon
                F1 = np.mean(matrix_red[feature_comparison[0]])
                F2 = np.mean(matrix_red[feature_comparison[1]])
                F1_err = np.std(matrix_red[feature_comparison[0]])
                F2_err = np.std(matrix_red[feature_comparison[1]])
                F1_list.append(F1)
                F2_list.append(F2)
                F1_err_list.append(F1)
                F2_err_list.append(F2)

            else:
                matrix = pd.DataFrame(data=df.iloc[0:dlen, 1:df.shape[1]])
                matrix = matrix[int(to_start_reg):int(to_end_reg)]


                # all windows outside regime
                VCL.drop(VCL[int(to_start_reg):int(to_end_reg)].index, inplace=True)
                indexNames = VCL[VCL['0'] == 1].index
                VCL.drop(indexNames, inplace=True)                                      # kick out all windows assigned to same cluster as regime

                lists = [VCL_in, VCL]
                VCL_reg = pd.concat(lists, ignore_index=True).sort_values('Time')
                VCL_reg.reset_index(drop=True, inplace=True)                            # all windows for BHT, but need to assign values from feature matrix
                dlen = df.shape[0]
                matrix = pd.DataFrame(data=df.iloc[0:dlen, 1:df.shape[1]])              # feature matrix (values only)

                # need to kick out rows in matrix which are not in the cluster vector anymore
                visitors = df
                orders = VCL_reg
                nonorders = visitors.loc[~visitors[0].isin(orders['Time']),].index
                matrix = matrix.drop(nonorders)                     # matrix to use for BHT

                VCL_reg.drop('Time',inplace = True, axis = 1)       # delete time column in cluster vector
                test = pd.DataFrame(VCL_reg)
                VCL_fin = test.iloc[:,0]                            # final binary vector for BHT


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

                    for z in range(len(header) + 1):
                        BHT = BHT.rename(columns={'feature': 'feature_number'}, index={z: str(header[z - 1])})

                    BHT.to_csv('BHT/Regime_' + str(i + 1) + '.csv')

                else:
                    labels_o = pd.Series(VCL_fin, index=range(len(VCL_fin)))
                    matrix.index = labels_o.index
                    BHval = tsfresh.feature_selection.relevance.calculate_relevance_table(matrix, labels_o, ml_task='classification')
                    BHT = pd.DataFrame(BHval)
                    for z in range(len(header) + 1):
                        BHT = BHT.rename(columns={'feature': 'feature_number'}, index={z: str(header[z - 1])})

                BHT_cl = BHT
                for p in range(len(correlations.columns)):
                    BHT_cl.loc[(correlations.columns)[p], 'Cluster'] = labels[p]

                #BHT.to_csv('BHT/Regime_' + str(i + 1) + '.csv')
                BHT_cl.to_csv('BHT_Cl/Reg_Feature_Clusters_Regime' + str(i + 1) + '.csv')

                # Append clusters of (Top X) features to list:
                feature_rank_list['Regime {:d}'.format(i+1)] = list(BHT_cl['Cluster'])[:top_X]

                # Prepare list of Top-X features across ALL regimes (used below)
                for z in range(len(BHT_cl.index[:top_X])):
                    if value_based is False:
                        if (BHT_cl['relevant'][z]) == True:
                            top_X_list.append(BHT_cl.index[:top_X][z])
                    else:
                        if (BHT_cl['p_value'][z]) < 0.01:
                            top_X_list.append(BHT_cl.index[:top_X][z])

                # Create list of "True" columns (for histogram plots)
                R = i+1
                if value_based is False:
                    if (BHT_cl['relevant'][0]) == True:
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
        feature_rank_list.to_csv('FRL.csv', index = None)

        '''
        # Colourise cells depending on their value (i.e. visualise cluster distribution across all regimes)
        import imgkit
        FRLfig = feature_rank_list.style.background_gradient(cmap='Set1', axis=None)#, subset=['Regime 1','Regime 2','Regime 3','Regime 4','Regime 5','Regime 6','Regime 7','Regime 8', 'Regime 9', 'Regime 11', 'Regime 13', 'Regime 14','Regime 15','Regime 16','Regime 17'])
        html = FRLfig.render()  # pd styles cannot be exported as png directly, needs to be converted to HTML first
        imgkit.from_string(html, 'FRL_color.png')
            # imgkit needs to be installed via homebrew first:
            # https://stackoverflow.com/questions/45664519/export-pandas-styled-table-to-image-file/50097322
            # https://github.com/jarrekk/imgkit
        '''


        # Find out what the top X feature across all regimes are and find them in dendrogram
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
        ppl.savefig('OUTPUT/dendrogram_top{:d}_features.png'.format(top_X), dpi=600)

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
        ppl.savefig('OUTPUT/featcluster_hist.png', dpi=400)

        if FVC is True:
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
            ppl.savefig('OUTPUT/featvalcomp.png', dpi=400)


if __name__ == '__main__':
    main()