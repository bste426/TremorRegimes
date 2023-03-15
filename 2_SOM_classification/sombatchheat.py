#pip install git+https://github.com/sevamoo/SOMPY.git#egg=SOMPY

import matplotlib.pylab as plt
import glob
import sompy as sompy
import pandas as pd
import numpy as np
import pickle
import scipy.signal as ss
import datetime
from statistics import multimode
from pandas.plotting import register_matplotlib_converters
import matplotlib.dates as mdates
import logging
import os
import csv
from matplotlib.ticker import MaxNLocator
import matplotlib.animation as ani
import matplotlib.pyplot as ppl
import joblib
import seaborn as sns
from matplotlib.colors import LogNorm
from inspect import getfile, currentframe
from datetime import datetime, timedelta, date
import jsonpickle, json
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.cluster import KMeans
from itertools import combinations
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, v_measure_score

'''
This code creates a Self-Organising Map (SOM) based on features extracted in '1_Feature_Extraction'. 
SOM will reconstruct the input data structure in reduced dimensionality, before k-means assigns time windows to clusters.
'''


########################
#### INITIALISATION ####
########################

#Name of volcano, station:
volcano = 'Whakaari'
station = 'WIZ'

# What time-period would you like to train the SOM with? (train data)
startdate = '2012-08-01'    # min.: 2008-06-02 (2-day and 5-day matrix)
enddate = '2012-08-05'      # max.: 2020-12-26 (last common day in 2-day and 5-day matrix)

# Folder of feature matrices (default):
PATH = '../1_Feature_Extraction/features/'
files = glob.glob(PATH + '*.csv')

# Are training and test data sets the same? (This is to reproduce the structure of the training data itself.)
'This has to be on True now, alternative will be available very soon (March 2023) once the code is updated.'
TESTisTRAIN = True 

if TESTisTRAIN is False:
    # Specify time period of test data (data you want to be looking at in the end):
    startdate_test = '2011-06-01'  # min.: 2008-06-01 (5-day matrix)
    enddate_test = '2014-06-01'  # max.: 2020-12-31 (last day in data for original analysis of Whakaari data - see paper)

    # Do you already have a trained map?
    trainedmap = True
    # If so: Where can the trained map be found?
    trained_PATH = '/trained_maps/whakaari_43200.00wndw_rsam_10_2.00-5.00_data_features_5cl_5x5_2021-01-01_2021-11-21.pkl'

# Plot SOM visualisations?
plot_SOM = False

timewindow = 'all'  # ignore (could be specified for data nomenclature purposes if only a specific time window length is being used)

# Shall the trained map be saved (or not, to save disk space)?
save_trained_map = True
# Shall the maps be tested and plotted or just trained (not needed e.g. when you only want to train a SOM and then use it at 'trained_PATH' in 'TESTisTRAIN' = False )?
Test_and_Plot = True
monthinterval = 1   #interval of x-ticks in months (adapt for visual reasons depending on test length)
# Shall the maps be interactive for close-ups?
interactive = False
# Would you like to compute SOM errors? (default: False.This feature is only really useful when using a large set of consistent feature matrices, i.e. complete combinations of frequency bands and time window lengths)
heatmap_on = True



#######################
### HYPERPARAMETERS ###
#######################
'''Remember that two key parameters, data frequency band, RSAM interval and time window length, have been specified during feature extraction.
'''

#mapsize (number of neurons, for multiple SOM sizes):
mx = [5,10,15]#,10,16,20,30]   #  x-dimension of SOM
my = [5,10,15]#,40,25,20,30]   #  y-dimension of SOM

#map_lattice:
map_lattice = ['rect','rect','rect']#,'rect','rect','rect','rect'] # 'rect'(angular) or 'hexa'(gonal)

#number of clusters:
n_clusters = 5 # Initially, you can perform statistical tests to get an estimate of a suitable number of clusters (see below).

# Do you want to calculate a suggested number of cluster?
CL_Determination = True
cl_mode = []
# Create output from statistical tests?
plot_curves = True
# Use linear or polynomial detrending of curves to find maxima/minima performance scores?
Polynomial_Detrend = True
degree = 4

def cl_det(som, cl, mapname, plot_curves, degree):
    '''Performs statistical tests to return a suitable number of clusters to start with.
    More info: https://github.com/tirthajyoti/Machine-Learning-with-Python/blob/master/Clustering-Dimensionality-Reduction/Clustering_metrics.ipynb
    '''

    register_matplotlib_converters()  # to avoid warnings
    import logging
    from sklearn.mixture import GaussianMixture
    Graphics = False  # False disables graphics, e.g. for batch
    ppl.interactive(False)
    logging.getLogger('matplotlib.font_manager').disabled = True

    som_vectors = som.codebook.matrix

    # define the number of hypothetical clusters you want to test here
    Min_num_clusters = 2  # default: 2
    Max_num_clusters = 13  # default: 13

    y = cl
    n_samples = som_vectors.shape[0]
    n_features = som_vectors.shape[1]

    d1 = som_vectors
    d1.shape

    df1 = pd.DataFrame(data=d1, columns=['Feature_' + str(i) for i in range(1, n_features + 1)])
    df1.head()

    lst_vars = list(combinations(df1.columns, 2))
    len(lst_vars)

    '''
    ### PLOTTING SECTION ###

    total_num_subplots = len(lst_vars)

    start_num_subplot = 1
    end_num_subplot = total_num_subplots
    # per Lipari sono troppi, limitare a max circa 100, va fatta la scelta
    start_num_subplot = 101
    end_num_subplot = 150

    plot_num_columns = 3
    real_number_of_suplots = (end_num_subplot - start_num_subplot + 1)

    plot_num_rows = int(real_number_of_suplots / plot_num_columns) + 1
    ppl.figure(figsize=(20, plot_num_rows * 6))

    for i in range(1, real_number_of_suplots):
        ppl.subplot(plot_num_rows, plot_num_columns, i)
        dim1 = lst_vars[start_num_subplot + i - 1][0]
        dim2 = lst_vars[start_num_subplot + i - 1][1]
        ppl.scatter(df1[dim1], df1[dim2], c=y, edgecolor='k', s=150)

        ppl.xlabel(f"{dim1}", fontsize=13)
        ppl.ylabel(f"{dim2}", fontsize=13)

    num_of_boxplots = len(df1.columns)
    boxplots_num_columns = 3
    boxplots_num_rows = int(num_of_boxplots / boxplots_num_columns) + 1
    ppl.figure(figsize=(20, plot_num_rows * 6))
    for i, c in enumerate(df1.columns):
        ppl.subplot(boxplots_num_rows, boxplots_num_columns, i + 1)
        sns.boxplot(y=df1[c], x=y)
        ppl.xticks(fontsize=15)
        ppl.yticks(fontsize=15)
        ppl.xlabel("Cluster", fontsize=15)
        ppl.ylabel(c, fontsize=15)
        ppl.show()

    ### END PLOTTING SECTION ###
    '''

    X = df1
    X.head()
    
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    ####################################
    ### PERFORMING STATISTICAL TESTS ###
    ####################################

    km_silhouette = []
    vmeasure_score = []
    db_score = []
    all_cl_no = []

    for i in range(Min_num_clusters, Max_num_clusters):
        km = KMeans(n_clusters=i, random_state=0).fit(X_scaled)
        preds = km.predict(X_scaled)

        silhouette = silhouette_score(X_scaled, preds)
        km_silhouette.append(silhouette)

        db = davies_bouldin_score(X_scaled, preds)
        db_score.append(db)

        v_measure = v_measure_score(y, preds)
        vmeasure_score.append(v_measure)

    if Polynomial_Detrend is False:
        # (removes linear trend from score_values and calculates local maxima (Sil&V-Meas.) / minima (DB))
        loc_min_kms = ss.argrelextrema(ss.detrend(np.asarray(km_silhouette)), np.greater)
        loc_min_db = ss.argrelextrema(ss.detrend(np.asarray(db_score)), np.less)
        loc_min_v_meas = ss.argrelextrema(ss.detrend(np.asarray(vmeasure_score)), np.greater)

    else:
        # (removes polynomial trend from values and calculates local max/min)
        X = [i for i in range(Min_num_clusters, Max_num_clusters)]
        X = np.reshape(X, (len(X), 1))
        y1 = km_silhouette
        y2 = db_score
        y3 = vmeasure_score
        pf = PolynomialFeatures(degree=degree)
        Xp = pf.fit_transform(X)
        md2 = LinearRegression()
        md3 = LinearRegression()
        md4 = LinearRegression()
        md2.fit(Xp, y1)
        md3.fit(Xp, y2)
        md4.fit(Xp, y3)
        trendp1 = md2.predict(Xp)
        trendp2 = md3.predict(Xp)
        trendp3 = md4.predict(Xp)
        detrpoly1 = [y1[i] - trendp1[i] for i in range(0, len(y1))]
        detrpoly2 = [y2[i] - trendp2[i] for i in range(0, len(y2))]
        detrpoly3 = [y3[i] - trendp3[i] for i in range(0, len(y3))]
        loc_max_kms = ss.argrelextrema(np.asarray(detrpoly1), np.greater)
        loc_min_db = ss.argrelextrema(np.asarray(detrpoly2), np.less)
        loc_max_v_meas = ss.argrelextrema(np.array(detrpoly3), np.greater)

    LKM = [(i + Min_num_clusters) for i in list(loc_max_kms[0]) if (i + Min_num_clusters) > 3]
    LDB = [(i + Min_num_clusters) for i in list(loc_min_db[0]) if (i + Min_num_clusters) > 3]
    LVM = [(i + Min_num_clusters) for i in list(loc_max_v_meas[0]) if (i + Min_num_clusters) > 3]

    for x1 in range(len(LKM)):
        all_cl_no.append(list(LKM)[x1])  # +MinNumCL to display number of cluster, not index in list
    for x2 in range(len(LDB)):
        all_cl_no.append(list(LDB)[x2])
    for x3 in range(len(LVM)):
        all_cl_no.append(list(LVM)[x3])

    # votes = []
    all_cl = list(multimode(all_cl_no))

    if len(all_cl) == 1:
        for i in range(6):
            cl_mode.append(all_cl[0])
        # cl_mode.append(votes)
    elif len(all_cl) == 2:
        for j in range(3):
            for i in range(len(all_cl)):
                cl_mode.append(all_cl[i - 1])
        # cl_mode.append(votes)
    elif len(all_cl) == 3:
        for j in range(2):
            for i in range(len(all_cl)):
                cl_mode.append(all_cl[i - 1])
        # cl_mode.append(votes)
    else:
        all_cl = list(multimode(all_cl_no))[:3]
        for j in range(2):
            for i in range(len(all_cl)):
                cl_mode.append(all_cl[i - 1])
        # cl_mode.append(votes)

    ########################
    ### PLOTTING RESULTS ###
    ########################
    """This section returns visual output of the statistical test results. Individual section can be activated and deactivated depending on what is required."""

    if plot_curves is True:

        # Min_num_clusters_for_plot=2
        # Max_num_clusters_for_plot=15
        Min_num_clusters_for_plot = Min_num_clusters
        Max_num_clusters_for_plot = Max_num_clusters
        circlesize = 60  # size of circles in the graphs
        hsize = 16  # size of each graph (horizontal)
        vsize = 5  # size of each graph (vertical)

        ### VISUAL OUTPUTS ###

        logging.getLogger('matplotlib.font_manager').disabled = True

        if not os.path.isdir('cl_OUTPUT'):
            os.makedirs('cl_OUTPUT')

        '''
        # Elbow method
        ppl.figure(figsize=(hsize, vsize))
        ppl.title("Elbow: "+mapname, fontsize=16)
        ppl.scatter(x=[i for i in range(Min_num_clusters_for_plot, Max_num_clusters_for_plot)],
                    y=km_scores[
                        Min_num_clusters_for_plot - Min_num_clusters:Max_num_clusters_for_plot - Min_num_clusters],
                    s=circlesize, edgecolor='k')
        ppl.plot([i for i in range(Min_num_clusters_for_plot, Max_num_clusters_for_plot)],
                    km_scores[
                    Min_num_clusters_for_plot - Min_num_clusters:Max_num_clusters_for_plot - Min_num_clusters],
                    color='r')
        ppl.grid(True)
        ppl.xlabel("Number of clusters", fontsize=14)
        ppl.ylabel("K-means score", fontsize=15)
        ppl.xticks([i for i in range(Min_num_clusters_for_plot, Max_num_clusters_for_plot)], fontsize=14)
        ppl.yticks(fontsize=15)
        ppl.savefig('cl_OUTPUT/elbow_' + mapname + '.png', dpi=200)


        # Elbow method, Derivative
        derivative_km = np.diff(km_scores)
        ppl.figure(figsize=(hsize, vsize))
        ppl.title("The derivative of elbow method for determining number of clusters\n", fontsize=16)
        Max_num_clusters_for_plot_1 = Max_num_clusters_for_plot - 1
        ppl.scatter(x=[i for i in range(Min_num_clusters_for_plot, Max_num_clusters_for_plot_1)], y=derivative_km[
                                                                                                    Min_num_clusters_for_plot - Min_num_clusters:Max_num_clusters_for_plot_1 - Min_num_clusters],
                    s=circlesize, edgecolor='k')
        ppl.plot([i for i in range(Min_num_clusters_for_plot, Max_num_clusters_for_plot_1)],
                    derivative_km[Min_num_clusters_for_plot - Min_num_clusters:Max_num_clusters_for_plot_1 - Min_num_clusters],
                    color='r')
        ppl.grid(True)
        ppl.xlabel("Number of clusters", fontsize=14)
        ppl.ylabel("K-means score", fontsize=15)
        ppl.xticks([i for i in range(Min_num_clusters_for_plot, Max_num_clusters_for_plot_1)], fontsize=14)
        ppl.yticks(fontsize=15)
        ppl.savefig('cl_OUTPUT/elbow_derivative'+map+'.png', dpi=200)
        '''

        # V-Measure for Evaluating Clustering Performance (compared to known labels)
        ppl.figure(figsize=(hsize, vsize))
        ppl.title("V-Measure: " + mapname, fontsize=16)
        ppl.scatter(x=[i for i in range(Min_num_clusters_for_plot, Max_num_clusters_for_plot)],
                    y=vmeasure_score[
                        Min_num_clusters_for_plot - Min_num_clusters:Max_num_clusters_for_plot - Min_num_clusters],
                    s=circlesize, edgecolor='k')
        ppl.plot([i for i in range(Min_num_clusters_for_plot, Max_num_clusters_for_plot)],
                    vmeasure_score[
                    Min_num_clusters_for_plot - Min_num_clusters:Max_num_clusters_for_plot - Min_num_clusters],
                    color='r')
        ppl.grid(True)
        ppl.xlabel("Number of clusters", fontsize=14)
        ppl.ylabel("V-measure score", fontsize=15)
        ppl.xticks([i for i in range(Min_num_clusters_for_plot, Max_num_clusters_for_plot)],
                    fontsize=14)
        ppl.yticks(fontsize=15)
        ppl.savefig('cl_OUTPUT/V-measure_' + mapname + '.png', dpi=200)

        # Silhouette method
        ppl.figure(figsize=(hsize, vsize))
        ppl.title("Silhouette: " + mapname, fontsize=16)
        ppl.scatter(x=[i for i in range(Min_num_clusters_for_plot, Max_num_clusters_for_plot)],
                    y=km_silhouette[
                        Min_num_clusters_for_plot - Min_num_clusters:Max_num_clusters_for_plot - Min_num_clusters],
                    s=circlesize, edgecolor='k')
        ppl.plot([i for i in range(Min_num_clusters_for_plot, Max_num_clusters_for_plot)],
                    km_silhouette[
                    Min_num_clusters_for_plot - Min_num_clusters:Max_num_clusters_for_plot - Min_num_clusters],
                    color='r')
        ppl.grid(True)
        ppl.xlabel("Number of clusters", fontsize=14)
        ppl.ylabel("Silhouette score", fontsize=15)
        ppl.xticks([i for i in range(Min_num_clusters_for_plot, Max_num_clusters_for_plot)],
                    fontsize=14)
        ppl.yticks(fontsize=15)
        ppl.savefig('cl_OUTPUT/silhouette_' + mapname + '.png', dpi=200)

        # Davies-Bouldin score
        ppl.figure(figsize=(hsize, vsize))
        ppl.title("D-B Score: " + mapname, fontsize=16)
        ppl.scatter(x=[i for i in range(Min_num_clusters_for_plot, Max_num_clusters_for_plot)],
                    y=db_score[
                        Min_num_clusters_for_plot - Min_num_clusters:Max_num_clusters_for_plot - Min_num_clusters],
                    s=circlesize, edgecolor='k')
        ppl.plot([i for i in range(Min_num_clusters_for_plot, Max_num_clusters_for_plot)],
                    db_score[
                    Min_num_clusters_for_plot - Min_num_clusters:Max_num_clusters_for_plot - Min_num_clusters],
                    color='r')
        ppl.grid(True)
        ppl.xlabel("Number of clusters", fontsize=14)
        ppl.ylabel("Davies-Bouldin score", fontsize=15)
        ppl.xticks([i for i in range(Min_num_clusters_for_plot, Max_num_clusters_for_plot)],
                    fontsize=14)
        ppl.yticks(fontsize=15)
        ppl.savefig('cl_OUTPUT/Davies-Bouldin_' + mapname + '.png', dpi=200)

        '''
        # BIC score with a Gaussian Mixture Model
        ppl.figure(figsize=(hsize, vsize))
        ppl.title("The Gaussian Mixture model BIC (red line) and AIC (blue line)", fontsize=16)
        ppl.scatter(x=[i for i in range(Min_num_clusters_for_plot, Max_num_clusters_for_plot)],
                    y=np.log(
                        gm_bic[Min_num_clusters_for_plot - Min_num_clusters:Max_num_clusters_for_plot - Min_num_clusters]),
                    s=circlesize, edgecolor='k')
        ppl.plot([i for i in range(Min_num_clusters_for_plot, Max_num_clusters_for_plot)],
                    np.log(gm_bic[Min_num_clusters_for_plot - Min_num_clusters:Max_num_clusters_for_plot - Min_num_clusters]),
                    color='r')
        ppl.scatter(x=[i for i in range(Min_num_clusters_for_plot, Max_num_clusters_for_plot)],
                    y=np.log(
                        gm_aic[Min_num_clusters_for_plot - Min_num_clusters:Max_num_clusters_for_plot - Min_num_clusters]),
                    s=circlesize, edgecolor='k')
        ppl.plot([i for i in range(Min_num_clusters_for_plot, Max_num_clusters_for_plot)],
                    np.log(gm_aic[Min_num_clusters_for_plot - Min_num_clusters:Max_num_clusters_for_plot - Min_num_clusters]),
                    color='b')
        ppl.grid(True)
        ppl.xlabel("Number of clusters", fontsize=14)
        ppl.ylabel("Log of Gaussian mixture BIC score", fontsize=15)
        ppl.xticks([i for i in range(Min_num_clusters_for_plot, Max_num_clusters_for_plot)], fontsize=14)
        ppl.yticks(fontsize=15)
        ppl.savefig('cl_OUTPUT/BIC(GMM)'+mapname+'.png', dpi=200)


        # (Log-)likelihood score
        ppl.figure(figsize=(hsize, vsize))
        ppl.title('Log-L-Score: '+ mapname, fontsize=16)
        ppl.scatter(x=[i for i in range(Min_num_clusters_for_plot, Max_num_clusters_for_plot)],
                    y=gm_score[
                        Min_num_clusters_for_plot - Min_num_clusters:Max_num_clusters_for_plot - Min_num_clusters],
                    s=150, edgecolor='k')
        ppl.plot([i for i in range(Min_num_clusters_for_plot, Max_num_clusters_for_plot)],
                    gm_score[
                    Min_num_clusters_for_plot - Min_num_clusters:Max_num_clusters_for_plot - Min_num_clusters],
                    color='r')
        ppl.grid(True)
        ppl.xlabel("Number of clusters", fontsize=14)
        ppl.ylabel("Gaussian mixture GM_score", fontsize=15)
        ppl.xticks([i for i in range(Min_num_clusters_for_plot, Max_num_clusters_for_plot)], fontsize=14)
        ppl.yticks(fontsize=15)
        ppl.savefig('cl_OUTPUT/(Log-)Likelihood_score_' + mapname + '.png', dpi=200)
        '''
    ppl.close('all')

def heatmap(Errors, n_clusters, win_len, fbands, mapsize):
    
    columns = ['te', 'qe']
    map = str(mapsize[0])+'x'+str(mapsize[1])
    TE_QE = pd.DataFrame(Errors, columns=columns, index=None)
    TE_QE['window_length'] = [int(float(x)) for x in win_len]
    TE_QE['frequency_band'] = fbands
    TE_QE['lowpass'] = [float(x[:4]) for x in fbands]
    TE_QE = TE_QE.sort_values(by = ['window_length', 'lowpass'])

    TEnorm = TE_QE['te']
    QEnorm = TE_QE['qe'] 

    TE = np.asarray(TEnorm).reshape(len(set(win_len)), len(set(fbands)))
    norm = LogNorm(vmin=0.000001, vmax=1)
    title = 'Topographic Error      Cluster: ' + str(n_clusters) + '      Mapsize: ' + map
    sns.set(font_scale=1.2)
    fig, ax = plt.subplots(figsize=(20, 8))
    cmap = sns.color_palette("cool", as_cmap=True)
    ax = sns.heatmap(TE, linewidth=0.5, norm=norm, cmap=cmap, cbar=True, xticklabels=sorted(set(fbands)),
                    yticklabels=sorted(set(TE_QE['window_length'])), cbar_kws={'label': 'Topographic Error'})
    ax.set_ylabel('Window length', fontsize=20)
    ax.set_xlabel('Frequency band [Hz]', fontsize=20)
    plt.title(title, fontsize=25)
    if not os.path.isdir('__OUTPUT/heatmapsTE'):
        os.makedirs('__OUTPUT/heatmapsTE')
    plt.savefig('__OUTPUT/heatmapsTE/' + 'TE ' + volcano + ' ' + station + ' ' + map + '.png', dpi=300)

    # Quantization error
    QE = np.asarray(QEnorm).reshape(len(set(win_len)), len(set(fbands)))
    title = 'Quantisation Error      Cluster: ' + str(n_clusters) + '      Mapsize: ' + map
    sns.set(font_scale=1.2)
    fig, ax = plt.subplots(figsize=(20, 6))
    colours3 = sns.color_palette("cool", as_cmap=True)
    ax = sns.heatmap(QE, linewidth=0.5, norm=norm, cmap=cmap, cbar=True, xticklabels=sorted(set(fbands)),
                    yticklabels=sorted(set(TE_QE['window_length'])), cbar_kws={'label': 'Topographic Error'})
    ax.set_ylabel('Window length', fontsize=20)
    ax.set_xlabel('Frequency band [Hz]', fontsize=20)
    plt.title(title, fontsize=25)
    if not os.path.isdir('__OUTPUT/heatmapsQE'):
        os.makedirs('__OUTPUT/heatmapsQE')
    plt.savefig('__OUTPUT/heatmapsQE/' + 'QE ' + volcano + ' ' + station + ' ' + map + '.png',
                dpi=300)
    plt.close('all')

if TESTisTRAIN is True:

    for z in range(len(mx)):
        '''Perform analysis for each given map size/shape'''
        print('Calculating SOM with dimensions '+str(mx[z])+'x'+str(my[z]))

        mapsize = [mx[z], my[z]]
        lattice = map_lattice[z]

        fbands = [] # collects frequency band of SOM
        win_len = []# collects time window length
        Errors = [] # collects topographic and quantization errors of map (for heatmap)

        for file in files:

            ###########################################################
            ############## T R A I N I N G   S T A G E ################
            ###########################################################

            print(file) # provide info on where the analysis is at
            Graphics = False  # False disables graphics, e.g. for batch
            logging.getLogger('matplotlib.font_manager').disabled = True
            plt.interactive(False)

            # Prepare name and path to save SOM
            FILE = file[len(PATH):-4]  # removes ".csv"
            EXTNAME = '_{:s}cl_{:s}x{:s}_{:s}_{:s}.pkl'.format(str(n_clusters),str(mx[z]),str(my[z]),startdate,enddate) # save classification result under this name
            mapname = FILE+EXTNAME[:-4] # full path to save classification result

            # extract frequency band from filename (easier naming convention would make this easier; implemented in future)
            fband = file[([index for index, character in enumerate(file) if character == '_'][-3]+1):([index for index, character in enumerate(file) if character == '_'][-2])]
            fbands.append(fband)
            window = file[([index for index, character in enumerate(file) if character == '_'][2]+1):([index for index, character in enumerate(file) if character == '_'][3]-4)]
            win_len.append(window)

            # Load dataframe from startdate to enddate
            df = pd.read_csv(file, header=None, low_memory=False)
            time_np = df[0]
            dates = list(time_np)
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
            df = pd.read_csv(file, header=None, skiprows=(int(to_start)), nrows=(int(to_end)),low_memory=False)

            # preparation of training data
            dlen = df.shape[0]
            df.head()
            dfselection2=pd.DataFrame(data=df.iloc[0:dlen, 1:df.shape[1]])
            Traindata = dfselection2.values

            # This builds the SOM using pre-defined hyperparameters and other default parameters. Initialization can be 'random' or 'pca'.
            som = sompy.SOMFactory.build(Traindata, mapsize, mask=None, mapshape='planar', lattice=lattice, normalization='None', initialization='pca', neighborhood='gaussian', training='batch', name='sompy')  
            som.train(n_job=6, verbose='info') #verbose='info' medium. verbose='debug' will print more, and verbose=None wont print anything

            # Calculate and save topographic and quantization errors
            te = som.calculate_topographic_error()
            qe = np.mean(som._bmu[1])
            print ("Topographic error = %s; Quantization error = %s" % (te,qe))
            teqe = (te,qe)
            Errors.append(teqe)

            # Visualise internal structure of SOM
            if plot_SOM is True:
                v = sompy.mapview.View2DPacked(50, 50, 'test', text_size=8)
                fig = v.show(som, what='codebook', which_dim='all', cmap='jet', col_sz=6)  # which_dim='all' default
                plt.savefig('mapview.png', dpi=200)

                v = sompy.mapview.View2DPacked(2, 2, 'test', text_size=8)
                v.show(som, what='cluster')

                h = sompy.hitmap.HitMapView(10, 10, 'hitmap', text_size=8, show_text=True)
                h.show(som)

                vhts = sompy.visualization.bmuhits.BmuHitsView(40,40,"Hits Map",text_size=12)
                vhts.show(som, anotate=True, onlyzeros=False, labelsize=12, cmap="Greys", logaritmic=False)

                plts = sompy.visualization.dotmap.DotMapView(40, 40, "Dot Map", text_size=12)
                plts.show(som) #lentissimo

                u = sompy.umatrix.UMatrixView(50, 50, 'umatrix', show_axis=True, text_size=8, show_text=True)
                UMAT = u.build_u_matrix(som, distance=1, row_normalized=False)
                UMAT = u.show(som, distance2=1, row_normalized=False, show_data=True, contooor=True, blob=False)

            # Use built-in function of SOMPY to perform clustering (k-means)
            cl = som.cluster(n_clusters=n_clusters)

            # save SOM
            if save_trained_map is True:
                if not os.path.isdir('trained_maps'):
                    os.makedirs('trained_maps')
                joblib.dump(som, 'trained_maps/' + FILE + EXTNAME)

            # after creating the SOM, perform tests with hypothetical numbers of clusters to gain optimum
            if CL_Determination is True:
                cl_det(som,cl,mapname,plot_curves,degree)




            if Test_and_Plot is True:

                ###################################################
                ############## T E S T   S T A G E ################
                ###################################################

                output = som.project_data(Traindata)
                outputDF = pd.DataFrame(output) #for each data, outputDF contains its assigned SOM node (bmu)
                cl = som.cluster(n_clusters=n_clusters)
                cloutput = cl[output] #for each data, cloutput contains its assigned cluster
                cloutputDF = pd.DataFrame(cloutput)
                cloutputDF.to_csv('Clusters.csv', index=False, header=False)

                to_start, to_end = int(to_start), int(to_end)
                time_np = time_np[to_start:(to_start + to_end)]

                if not os.path.isdir('tested_maps'):
                    os.makedirs('tested_maps')

                with open('tested_maps/' + FILE + EXTNAME, 'wb') as f:
                    pickle.dump([time_np, cloutput], f)



                #########################################################
                ############## P L O T T I N G   D A T A ################
                #########################################################

                register_matplotlib_converters()  # to avoid warnings

                # preparing time vector (conversion to timestamps)
                with open('tested_maps/' + FILE + EXTNAME,'rb') as f:
                    time_np, cloutput = pickle.load(f)
                time_np = np.asarray(time_np)
                time_np2 = []
                for x in range(len(time_np)):
                 try:
                  time_np2.append(datetime.strptime(str(time_np[x]), '%Y-%m-%d %H:%M:%S'))
                 except:
                  time_np2.append(datetime.strptime(str(time_np[x]), '%Y-%m-%d'))
                time_np = np.array(time_np2)


                # plot initialization
                fig, axarr = plt.subplots(1, sharex=True, sharey=False)
                if interactive is False:
                    fig.set_size_inches(16, 9)
                axarr.set_yticklabels([])
                axarr.axis('off')
                ax = fig.add_subplot(1, 1, 1, sharex=fig.axes[0])
                colormap = np.array(['r', 'purple', 'b', 'cyan', 'lime'])
                tminimo = 0
                tmassimo = time_np.shape[0]
                colours = cloutput[tminimo:tmassimo] % colormap.shape[0]
                ax.scatter(time_np[tminimo:tmassimo], cloutput[tminimo:tmassimo]+1, c=colormap[colours])
                ax.set_xlim(min(time_np), max(time_np))
                ax.set_ylabel('Cluster number', fontsize = 15)
                ax.grid(linestyle='dotted', which='both')
                ax.yaxis.set_major_locator(MaxNLocator(integer=True))
                fig.subplots_adjust(hspace=0.05)
                plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)

                # change interval of ticks on x-axis here depending on length of time-period covered by data
                ax.xaxis.set_major_locator(mdates.MonthLocator(interval=monthinterval))
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                fig.autofmt_xdate()
                plt.xticks(fontsize = 15)
                plt.yticks(fontsize = 15)

                # add events to plot
                with open('../1_Feature_Extraction/RSAM/eruptive_periods.txt', 'r') as fp:
                    tes = [ln.rstrip() for ln in fp.readlines()]
                xcoords = tes
                for xc in xcoords:
                    ax.axvline(x = xc, color='k', linestyle='-', linewidth=2, label='_')
                
                with open('../1_Feature_Extraction/RSAM/activity.txt', 'r') as fp:
                    act = [ln.rstrip() for ln in fp.readlines()]
                cords = act
                for co in cords:
                    ax.axvline(x = co, color='dimgrey', linestyle='--', linewidth=2, label='_')

                ax.axvline(x='2012-08-04 16:52:00', color='k', linestyle='-', linewidth=2, label='eruption')
                ax.axvline(x='2012-09-02 00:00:00', color='dimgrey', linestyle='--', linewidth=2, label='ash emission')
                ax.axvline(x='2012-11-24 00:00:00', color='dimgrey', linestyle=':', linewidth=2, label='observation of lava dome')


                # add legend and title
                ax.legend(bbox_to_anchor=(0.25, 1), fontsize = 15, ncol = 3)
                TITLE = str(volcano + ', ' + station + '   fband: ' + fband +'   Cluster: ' + str(n_clusters) + '   SOM size: '+str(mx[z])+'x'+str(my[z]))

                ppl.title(TITLE, loc='left')

                # save plot
                fig.set_size_inches(16, 6)
                if not os.path.isdir('__OUTPUT'):
                    os.makedirs('__OUTPUT')
                path = '__OUTPUT/' + FILE + EXTNAME + '.png'
                plt.savefig(path, dpi=600)

                if interactive is True:
                    fig.set_size_inches(15,8)
                    ppl.ion()
                    ppl.show()
                    ppl.pause(1000)

                plt.close()

        if heatmap_on:
            heatmap(Errors, n_clusters, win_len, fbands, mapsize)

    if CL_Determination is True:

        CLM = pd.DataFrame(list(cl_mode))
        CLM.columns = ['cluster']
        CL_fig = sns.displot(CLM, x='cluster', color='black', discrete=True, binwidth=1, stat='density')
        CL_fig.savefig('CL_distribution.png', dpi=300)

else:

    if trainedmap is True:
        Graphics = False  # False disables graphics, e.g. for batch
        logging.getLogger('matplotlib.font_manager').disabled = True
        plt.interactive(False)

        mapsize = [5,5]
        lattice = ['rect']

        Errors = []

        EXTNAME = '_{:s}cl_{:s}x{:s}_{:s}_{:s}.pkl'.format(str(n_clusters), str(mx[0]), str(my[0]), startdate_test, enddate_test)


        # GET TEST DATA

        PATH = '/Users/bste426/Documents/All/PhD/Data/Codes/SOM_Carniel/csv2som/feature_matrices/'
        fl = PATH + '*.csv'
        p = len(PATH)
        p2 = p + 31  # 31 symbols to remove from the file name to extract 'fband1-fband2'
        files = glob.glob(fl)

        fbands = [x[p2:-18] for x in files]  # removes everything before and after the frequency band info in the file name

        FILE = files[0][p:-4]  # removes ".csv"
        filename = PATH + FILE + '.csv'
        df = pd.read_csv(filename, header=None)
        time_np = df[0]

        dates = list(time_np)

        newdates = [x[:-9] for x in dates]
        for row in newdates:
            if startdate_test == row:
                to_start = np.array(newdates.index(startdate_test))
                break
        for row in newdates:
            if enddate_test == row:
                to_end = np.array(newdates.index(enddate_test))
                break
        to_end = to_end - to_start
        df = pd.read_csv(filename, header=None, skiprows=(int(to_start)), nrows=(int(to_end)))

        dlen = df.shape[0]
        time_np = df[0]
        dfselection2 = pd.DataFrame(
            data=df.iloc[0:dlen, 1:df.shape[1]])
        Testdata = dfselection2.values
        # LOAD TRAINED MAP

        ### Insert trained map here:
        som = joblib.load(trained_PATH)

        # TEST DATA ON TRAINED MAP

        output = som.project_data(Testdata)

        outputDF = pd.DataFrame(output)  # for each data, outputDF contains its assigned SOM node (bmu)
        cl = som.cluster(n_clusters=n_clusters)
        cloutput = cl[output]  # for each data, cloutput contains its assigned cluster
        cloutputDF = pd.DataFrame(cloutput)
        cloutputDF.to_csv('Clusters.csv', index=False, header=False)

        # SAVE MAP

        if not os.path.isdir('tested_maps'):
            os.makedirs('tested_maps')

        if Art_matrix is True:
            with open('tested_maps/' + FILE + EXTNAME, 'wb') as f:  # Python 3: open(..., 'wb')
                pickle.dump([time_np, cloutput], f)
        else:
            if extdrive is True:
                with open('/Volumes/TheBigOne/Data/Maps/2019 NO/tested_maps/' + FILE + EXTNAME, 'wb') as f:  # Python 3: open(..., 'wb')
                    pickle.dump([time_np, cloutput], f)
            else:
                with open('/Users/bste426/Documents/All/PhD/Data/Codes/SOM_Carniel/csv2som/tested_maps/' + FILE + EXTNAME, 'wb') as f:
                    pickle.dump([time_np, cloutput], f)

        ########### PLOTTING STAGE ###########

        register_matplotlib_converters()  # to avoid warnings
        if Art_matrix is True:
            with open('tested_maps/' + FILE + EXTNAME, 'rb') as f:  # Python 3: open(..., 'wb')
                time_np, cloutput = pickle.load(f)
        else:
            if extdrive is True:
                with open('/Volumes/TheBigOne/Data/Maps/2019 NO/tested_maps/' + FILE + EXTNAME,
                          'rb') as f:  # Python 3: open(..., 'wb')
                    time_np, cloutput = pickle.load(f)
            else:
                with open('/Users/bste426/Documents/All/PhD/Data/Codes/SOM_Carniel/csv2som/tested_maps/' + FILE + EXTNAME,
                          'rb') as f:
                    time_np, cloutput = pickle.load(f)

        time_np2 = []
        for x in range(len(time_np)):
            try:
                time_np2.append(datetime.strptime(str(time_np[x]), '%Y-%m-%d %H:%M:%S'))
            except:
                time_np2.append(datetime.strptime(str(time_np[x]), '%Y-%m-%d'))

        time_np = np.array(time_np2)

        # plot initialization
        fig, axarr = plt.subplots(1, sharex=True, sharey=False)
        if interactive is False:
            fig.set_size_inches(16, 9)
        axarr.set_yticklabels([])
        axarr.axis('off')
        ax = fig.add_subplot(1, 1, 1, sharex=fig.axes[0])
        colormap = np.array(['r', 'purple', 'b', 'cyan', 'lime'])
        tminimo = 0
        tmassimo = time_np.shape[0]
        colours = cloutput[tminimo:tmassimo] % colormap.shape[0]
        ax.scatter(time_np[tminimo:tmassimo], cloutput[tminimo:tmassimo] + 1, c=colormap[colours])
        ax.set_xlim(min(time_np), max(time_np))
        ax.set_ylabel('Cluster number')
        ax.grid(linestyle='dotted', which='both')
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        # plt.pause(0.001)
        fig.subplots_adjust(hspace=0.05)
        plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)

        # change interval of ticks on x-axis here depending on length of time-period covered by data
        test_startdate = datetime.strptime('2010-12-30 00:00:00', '%Y-%m-%d %H:%M:%S')
        if min(time_np) < test_startdate:
            ax.xaxis.set_major_locator(mdates.YearLocator(base=1, month=1, day=1, tz=None))
        else:
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        fig.autofmt_xdate()

        # add events to plot
        with open('act_log/eruptive_periods.txt', 'r') as fp:
            tes = [ln.rstrip() for ln in fp.readlines()]
        xcoords = tes
        for xc in xcoords:
            ax.axvline(x=xc, color='k', linestyle='-', linewidth=2, label='_')
        with open('act_log/activity.txt', 'r') as fp:
            act = [ln.rstrip() for ln in fp.readlines()]
        cords = act
        for co in cords:
            ax.axvline(x=co, color='dimgrey', linestyle='--', linewidth=2, label='_')
        ax.axvline(x='2012-08-04 16:52:00', color='k', linestyle='-', linewidth=2, label='eruption')
        ax.axvline(x='2012-09-02 00:00:00', color='dimgrey', linestyle='--', linewidth=2, label='ash emission')
        ax.axvline(x='2012-11-24 00:00:00', color='dimgrey', linestyle=':', linewidth=2, label='observation of lava dome')

        # add legend and title
        ax.legend(bbox_to_anchor=(0.79, 1))
        TITLE = str(FILE + '   Cluster: ' + str(n_clusters) + '   SOM size: ' + str(mx[0]) + 'x' + str(my[0]))
        ppl.title(TITLE, loc='left')

        # save plot
        fig.set_size_inches(16, 9)
        if not os.path.isdir('../__OUTPUT'):
            os.makedirs('../__OUTPUT')
        if Art_matrix is True:
            path = '../__OUTPUT/' + FILE + EXTNAME + '.png'
        else:
            if extdrive is True:
                path = '/Volumes/TheBigOne/Data/Maps/2019 NO/plots/' + FILE + EXTNAME + '.png'
            else:
                path = '/Users/bste426/Documents/All/PhD/Data/Codes/SOM_Carniel/__OUTPUT/plots/' + FILE + EXTNAME + '.png'
        plt.savefig(path, dpi=200)

        if interactive is True:
            fig.set_size_inches(15, 8)
            ppl.ion()
            ppl.show()
            ppl.pause(1000)

        plt.close()

    else:
        Errors = []

        PATH = '/Users/bste426/Documents/All/PhD/Data/Codes/SOM_Carniel/csv2som/feature_matrices/'
        fl = PATH + '*.csv'
        p = len(PATH)
        p2 = p + 31  # 31 symbols to remove from the file name to extract 'fband1-fband2'
        files = glob.glob(fl)

        fbands = [x[p2:-18] for x in files]  # removes everything before and after the frequency band info in the file name

        for z in range(len(mx)):

            mapsize = [mx[z], my[z]]
            lattice = map_lattice[z]

            for i in range(len(files)):


                # TRAIN MAP WITH TRAIN DATA RANGE


                Graphics = False  # False disables graphics, e.g. for batch
                logging.getLogger('matplotlib.font_manager').disabled = True
                plt.interactive(False)

                FILE = files[i][77:-4]  # removes ".csv"
                filename = PATH + FILE + '.csv'
                df = pd.read_csv(filename, header=None, skiprows=1)
                time_np = df[0]

                dates = list(time_np)
                print(startdate, enddate)

                if len(dates[1]) < 11:
                    newdates = dates
                    for row in newdates:
                        if startdate == row:
                            to_start = np.array(newdates.index(startdate))
                            break
                    for row in newdates:
                        if enddate == row:
                            to_end = np.array(newdates.index(enddate))
                            break
                    to_end = to_end - to_start

                else:
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

                df = pd.read_csv(filename, header=None, skiprows=(int(to_start)), nrows=(int(to_end)), low_memory=False)

                dlen = df.shape[0]
                df.head()
                dfselection2 = pd.DataFrame(
                    data=df.iloc[0:dlen, 1:df.shape[1]])  # columns from 1 to 5. Not using column 0 which is time
                Traindata = dfselection2.values
                som = sompy.SOMFactory.build(Traindata, mapsize, mask=None, mapshape='planar', lattice=lattice,
                                             normalization='var', initialization='pca', neighborhood='gaussian',
                                             training='batch',
                                             name='sompy')  # this will use the default parameters, but i can change the initialization and neighborhood methods
                # all_mapshapes = ['planar','toroid','cylinder'] ## Only planar is implemented!!!
                # all_lattices = ['hexa','rect']
                # all_initialization = ['random','pca']

                som.train(n_job=6, verbose='debug')
                # verbose='info' medium. verbose='debug' will print more, and verbose=None wont print anything

                te = som.calculate_topographic_error()
                qe = np.mean(som._bmu[1])
                print ("Topographic error = %s; Quantization error = %s" % (te, qe))

                teqe = (te, qe)
                Errors.append(teqe)

                EXTNAME = '_{:s}cl_{:s}x{:s}_{:s}_{:s}.pkl'.format(str(n_clusters), str(mx[z]), str(my[z]), startdate,
                                                                   enddate)
                cl = som.cluster(n_clusters=n_clusters)

                if plot_SOM is True:
                    #v = sompy.mapview.View2DPacked(50, 50, 'test', text_size=8)
                    #v.show(som, what='codebook', which_dim='all', cmap='jet', col_sz=6)  # which_dim='all' default

                    #v = sompy.mapview.View2DPacked(2, 2, 'test', text_size=8)
                    #v.show(som, what='cluster')

                    h = sompy.hitmap.HitMapView(10, 10, 'hitmap', text_size=8, show_text=True)
                    h.show(som)

                    #vhts = sompy.visualization.bmuhits.BmuHitsView(40, 40, "Hits Map", text_size=12)
                    #vhts.show(som, anotate=True, onlyzeros=False, labelsize=12, cmap="Greys", logaritmic=False)

                    #plts = sompy.visualization.dotmap.DotMapView(40, 40, "Dot Map", text_size=12)
                    #plts.show(som)  # lentissimo

                    u = sompy.umatrix.UMatrixView(50, 50, 'umatrix', show_axis=True, text_size=8, show_text=True)
                    UMAT = u.build_u_matrix(som, distance=1, row_normalized=False)
                    UMAT = u.show(som, distance2=1, row_normalized=False, show_data=True, contooor=True, blob=False)


                # if not os.path.isdir('trained_maps'):
                # os.makedirs('trained_maps')

                getattr(som, 'cluster_labels')

                if save_trained_map is True:
                    # joblib.dump(som, '/Volumes/TheBigOne/Data/Maps/'+FILE+EXTNAME)
                    if Art_matrix is True:
                        joblib.dump(som,
                                    '/Users/bste426/Documents/All/PhD/Data/Codes/SOM_Carniel/csv2som/trained_maps/' + FILE + EXTNAME)
                    else:
                        if extdrive is True:
                            joblib.dump(som, '/Volumes/TheBigOne/Data/Maps/2019 NO/trained_maps/' + FILE + EXTNAME)
                        else:
                            joblib.dump(som,
                                        '/Users/bste426/Documents/All/PhD/Data/Codes/SOM_Carniel/csv2som/trained_maps/' + FILE + EXTNAME)


                ########## CL DETERMINATION (OPTIONAL) ##########


                mapname = FILE + EXTNAME[:-4]

                if CL_Determination is True:

                    cl_det(som, cl, mapname, plot_curves, degree)


                ########## TESTING STAGE WITH TEST DATA RANGE ##########

                # load test data

                df = pd.read_csv(filename, header=None, skiprows=1)
                time_np = df[0]

                dates = list(time_np)

                if len(dates[1])<11:
                    newdates = dates
                    for row in newdates:
                        if startdate_test == row:
                            to_start = np.array(newdates.index(startdate_test))
                            break
                    for row in newdates:
                        if enddate_test == row:
                            to_end = np.array(newdates.index(enddate_test))
                            break
                    to_end = to_end - to_start
                else:
                    newdates = [x[:-9] for x in dates]
                    for row in newdates:
                        if startdate_test == row:
                            to_start = np.array(newdates.index(startdate_test))
                            break
                    for row in newdates:
                        if enddate_test == row:
                            to_end = np.array(newdates.index(enddate_test))
                            break
                    to_end = to_end - to_start


                df = pd.read_csv(filename, header=None, skiprows=(int(to_start)), nrows=(int(to_end)), low_memory=False)

                dlen = df.shape[0]
                time_np = df[0]
                dfselection2 = pd.DataFrame(
                    data=df.iloc[0:dlen, 1:df.shape[1]])
                Testdata = dfselection2.values

                # test SOM

                #with open('/Users/bst/Documents/All/PhD/Data/Codes/SOM_Carniel/csv2som/tested_maps/whakaari_043200.00wndw_rsam_10_2.00-5.00_data_features_5cl_5x5_2016-01-01_2020-01-01.pkl', 'rb') as f:  # Python 3: open(..., 'wb')
                #    time_np, cloutput = pickle.load(f)
                output = som.project_data(Testdata)
                outputDF = pd.DataFrame(output)  # for each data, outputDF contains its assigned SOM node (bmu)
                cl = som.cluster(n_clusters=n_clusters)
                cloutput = cl[output]  # for each data, cloutput contains its assigned cluster
                cloutputDF = pd.DataFrame(cloutput)
                cloutputDF.to_csv('Clusters.csv', index=False, header=False)

                time_np = df[0]

                if not os.path.isdir('tested_maps'):
                    os.makedirs('tested_maps')

                if Art_matrix is True:
                    with open('tested_maps/' + FILE + EXTNAME, 'wb') as f:  # Python 3: open(..., 'wb')
                        pickle.dump([time_np, cloutput], f)
                else:
                    if extdrive is True:
                        with open('/Volumes/TheBigOne/Data/Maps/2019 NO/tested_maps/' + FILE + EXTNAME,
                                  'wb') as f:  # Python 3: open(..., 'wb')
                            pickle.dump([time_np, cloutput], f)
                    else:
                        with open(
                                '/Users/bste426/Documents/All/PhD/Data/Codes/SOM_Carniel/csv2som/tested_maps/' + FILE + EXTNAME,
                                'wb') as f:
                            pickle.dump([time_np, cloutput], f)

                ########### PLOTTING STAGE ###########

                register_matplotlib_converters()  # to avoid warnings
                if Art_matrix is True:
                    with open('tested_maps/' + FILE + EXTNAME, 'rb') as f:  # Python 3: open(..., 'wb')
                        time_np, cloutput = pickle.load(f)
                else:
                    if extdrive is True:
                        with open('/Volumes/TheBigOne/Data/Maps/2019 NO/tested_maps/' + FILE + EXTNAME,
                                  'rb') as f:  # Python 3: open(..., 'wb')
                            time_np, cloutput = pickle.load(f)
                    else:
                        with open(
                                '/Users/bste426/Documents/All/PhD/Data/Codes/SOM_Carniel/csv2som/tested_maps/' + FILE + EXTNAME,
                                'rb') as f:
                            time_np, cloutput = pickle.load(f)

                time_np2 = []
                for x in range(len(time_np)):
                    try:
                        time_np2.append(datetime.strptime(str(time_np[x]), '%Y-%m-%d %H:%M:%S'))
                    except:
                        time_np2.append(datetime.strptime(str(time_np[x]), '%Y-%m-%d'))

                time_np = np.array(time_np2)

                # plot initialization
                fig, axarr = plt.subplots(1, sharex=True, sharey=False)
                if interactive is False:
                    fig.set_size_inches(16, 5)
                axarr.set_yticklabels([])
                axarr.axis('off')
                ax = fig.add_subplot(1, 1, 1, sharex=fig.axes[0])
                colormap = np.array(['r', 'purple', 'b', 'cyan', 'lime'])
                tminimo = 0
                tmassimo = time_np.shape[0]

                colours = cloutput[tminimo:tmassimo] % colormap.shape[0]
                ax.scatter(time_np[tminimo:tmassimo], cloutput[tminimo:tmassimo] + 1, c=colormap[colours])
                ax.set_xlim(min(time_np), max(time_np))
                ax.set_ylabel('Cluster number', fontsize = 15)
                ax.grid(linestyle='dotted', which='both')
                ax.yaxis.set_major_locator(MaxNLocator(integer=True))
                # plt.pause(0.001)
                fig.subplots_adjust(hspace=0.05)
                plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)

                # change interval of ticks on x-axis here depending on length of time-period covered by data
                test_startdate = datetime.strptime('2010-12-30 00:00:00', '%Y-%m-%d %H:%M:%S')
                if min(time_np) < test_startdate:
                    ax.xaxis.set_major_locator(mdates.YearLocator(base=1, month=1, day=1, tz=None))
                else:
                    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=monthinterval))
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                fig.autofmt_xdate()
                plt.yticks(fontsize = 15)
                plt.xticks(fontsize = 15)


                # add events to plot
                with open('act_log/eruptive_periods.txt', 'r') as fp:
                    tes = [ln.rstrip() for ln in fp.readlines()]
                xcoords = tes
                for xc in xcoords:
                    ax.axvline(x=xc, color='k', linestyle='-', linewidth=2, label='_')
                with open('act_log/activity.txt', 'r') as fp:
                    act = [ln.rstrip() for ln in fp.readlines()]
                cords = act
                for co in cords:
                    ax.axvline(x=co, color='dimgrey', linestyle='--', linewidth=2, label='_')
                ax.axvline(x='2012-08-04 16:52:00', color='k', linestyle='-', linewidth=2, label='eruption')
                ax.axvline(x='2012-09-02 00:00:00', color='dimgrey', linestyle='--', linewidth=2,
                           label='ash emission')
                ax.axvline(x='2012-11-24 00:00:00', color='dimgrey', linestyle=':', linewidth=2,
                           label='observation of lava dome')
                '''

                # include earthquake events into classification plot
                with open('act_log/tectonics.txt', 'r') as fp:
                    tecs = [ln.rstrip() for ln in fp.readlines()]
                xcoords = tecs
                for xc in xcoords:
                    ax.axvline(x=xc, color='green', linestyle='-', linewidth=3, label='_')
                '''

                # add legend and title
                ax.legend(bbox_to_anchor=(0.25, 1), fontsize = 15, ncol=3)
                TITLE = str(FILE + '   Cluster: ' + str(n_clusters) + '   SOM size: ' + str(mx[z]) + 'x' + str(my[z]))
                #ppl.title(TITLE, loc='left')

                # save plot
                fig.set_size_inches(16, 6)
                if not os.path.isdir('../__OUTPUT/plots'):
                    os.makedirs('../__OUTPUT/plots')
                if Art_matrix is True:
                    path = '../__OUTPUT/plots/' + FILE + EXTNAME + '.png'
                else:
                    if extdrive is True:
                        path = '/Volumes/TheBigOne/Data/Maps/2019 NO/plots/' + FILE + EXTNAME + '.png'
                    else:
                        path = '/Users/bste426/Documents/All/PhD/Data/Codes/SOM_Carniel/__OUTPUT/plots/' + FILE + EXTNAME + '.png'
                plt.tight_layout()
                plt.savefig(path, dpi=600)
                '''
                if interactive is True:
                    fig.set_size_inches(15, 8)
                    ppl.ion()
                    ppl.show()
                    ppl.pause(1000)
                '''
                plt.close()