#pip install git+https://github.com/sevamoo/SOMPY.git#egg=SOMPY

import matplotlib.pylab as plt
import matplotlib.pyplot as ppl
import matplotlib
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
from sklearn import preprocessing
from sklearn.cluster import KMeans
from itertools import combinations
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, v_measure_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# to avoid warnings:
register_matplotlib_converters()  
plt.set_loglevel('WARNING')

'''
This code creates a Self-Organising Map (SOM) based on features extracted in '1_Feature_Extraction'. 
SOM will reconstruct the input data structure in reduced dimensionality, before k-means assigns time windows to clusters.
'''



#####################################
#### I N I T I A L I S A T I O N ####
#####################################

#Name of volcano, station:
volcano = 'Copahue'
station = 'COP'

# What time-period would you like to train the SOM with? (train data)
startdate = '2020-01-01'    # min.: 2008-06-02 (2-day and 5-day matrix)
enddate = '2020-12-31'      # max.: 2020-12-26 (last common day in 2-day and 5-day matrix)

# Folder of feature matrices (default):
PATH = '../1_Feature_Extraction/features/{:s}/{:s}/'.format(volcano, station)
files = glob.glob(PATH + '*')

# Feature matrices should be normalised:
normalise_matrix = True



#################################
### A N A L Y S I S   M O D E ###
#################################

# Are training and test data sets the same? (This is to reproduce the structure of the training data itself.)
TESTisTRAIN = True 

if TESTisTRAIN is False:
    '''This is either for the case that you have a trained SOM that you want to use to test an independent dataset (e.g. from a different volcano), or
    for the case that the matrices provided in PATH are split up in different train and test sets. Initially, set trainedmap = False.
    '''

    # Specify time period of test data (data you want to be looking at in the end):
    startdate_test = '2012-06-01'
    enddate_test = '2013-06-01'
    # Do you already have a trained map?
    trainedmap = False
    # If so: Where can the trained map be found?
    trained_PATH = 'OUTPUT/trained_maps/whakaari_043200.00wndw_rsam_10_2.00-5.00_data_features_6cl_5x5_2008-06-01_2014-01-01.pkl'
 
# Plot visualised SOM structure?
plot_SOM = False # to prevent it from showing the plots, deactivate the plt.show() function in line 171 in sompy/visualization/mapview.py as well as line 94 in sompy/visualization/umatrix.py.

# Shall the trained map be saved for further analysis (or not, to save disk space)?
save_trained_map = False

# Shall the maps be tested and plotted or just trained (not needed e.g. when you only want to train a SOM and then use it at 'trained_PATH' in 'TESTisTRAIN' = False )?
Test_and_Plot = True
monthinterval = 1   #interval of x-ticks in months (adapt for visual reasons depending on test length)

# Shall the maps be interactive for close-ups? (shows immediately, turn off for batch)
interactive = True

# Would you like to compute SOM errors? (default: False.This feature is only really useful when using a large set of consistent feature matrices, i.e. complete combinations of frequency bands and time window lengths)
heatmap_on = False
if TESTisTRAIN is False and trainedmap is True:
     heatmap_on = False #only available when training SOMs



#####################################
### H Y P E R P A R A M E T E R S ###
#####################################

'''Remember that two key parameters, data frequency band, RSAM interval and time window length, have been specified during feature extraction.
'''

#mapsize (number of neurons, for multiple SOM sizes):
mx = [24]#,10,15,10,16,20,30]   #  x-dimension of SOM
my = [16]#,10,15,40,25,20,30]   #  y-dimension of SOM

#map_lattice:
lattice_shape = 'hexa'  #   rectangular, otherwise 'hexa' for hexagonal.
map_lattice = [lattice_shape for i in range(len(mx))]   # Can be adapted to individual maps, e.g. ['rect','hexa','rect','rect',...] 

#number of clusters:
n_clusters = 5 # Initially, you can perform statistical tests to get an estimate of a suitable number of clusters (see below).

# Do you want to calculate a suggested number of cluster?
CL_Determination = False #This is implemented for batch processing when TESTisTrain = 'True'. Otherwise, use individual matrices with pre-defined n_clusters.
cl_mode = []
# Create output from statistical tests?
plot_curves = False
# Use linear or polynomial detrending of curves to find maxima/minima performance scores?
Polynomial_Detrend = False
degree = 4



#########################
### F U N C T I O N S ###
#########################

def load_data(file, startdate, enddate):

    #load data
    try:
        df = pd.read_csv(file, header=None, low_memory=False)
        time_np = df[0] #time vector
        dates = list(time_np)
    except:
        # write time vector from parquet
        df = pd.read_parquet(file)
        dates = [datetime.strftime(x, '%Y-%m-%d %H:%M:%S') for x in df.index]
        dates.insert(0, 'time')
        time_np = pd.Series(dates)
        # prepare matrix from parquet
        df = df.reset_index()
        columns = df.columns
        df.columns = range(df.columns.size)
        df = pd.concat([pd.DataFrame(columns).T,df.loc[:]])

    #make sure time format is converted correctly
    if len(dates[1]) > 10:
        newdates = [x[:-9] for x in dates]
    else:
        newdates = [x for x in dates]

    #find rows of given start and end dates to use only given period of matrix
    for row in newdates:
        if startdate == row:
            to_start = int(np.array(newdates.index(startdate)))
            break
    for row in newdates:
        if enddate == row:
            to_end = int(np.array(newdates.index(enddate)))
            break

    #extract values
    dfselection2=pd.DataFrame(data=df.iloc[1:df.shape[0], 1:df.shape[1]])[to_start:to_end]
    if normalise_matrix is True:
        x = dfselection2.values #returns a numpy array
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        time_np = time_np[to_start:to_end].reset_index(drop=True)
        df = pd.DataFrame(x_scaled).dropna(axis=1)
        time_np = time_np.drop(df[df[0]>0.01].index)
        df = df.drop(df[df[0]>0.01].index).values
        #filter 0.01 [pd.DataFrame(x_scaled)[0][i] for i in range(pd.DataFrame(x_scaled)[0].shape[0]) if pd.DataFrame(x_scaled)[0][i]<0.01]
        #df = pd.DataFrame(x_scaled).values #normalized feature values cut down to period of interest
        #test = [pd.DataFrame(x_scaled)[i] for i in range(x_scaled.shape[1]) if pd.DataFrame(x_scaled)[i].isnull().values.any()==False] 
    else:
        x_nonscaled = np.array(dfselection2.values, dtype='float')
        df = pd.DataFrame(x_nonscaled).values #non-normalized feature values cut down to period of interest
        time_np = time_np[to_start:to_end].reset_index(drop=True)

    '''
    # Some nice additional plots:
    pca = PCA(n_components=2)
    scaler = preprocessing.StandardScaler().fit(df)
    data_sta = scaler.transform(df)
    YM = pca.fit_transform(data_sta)
    tsn = TSNE(n_components=2, init='random', perplexity=759).fit_transform(data_sta)
    fin = pd.DataFrame(tsn).set_index(time_np)
    plt.scatter(fin[fin.columns[0]],fin[fin.columns[1]], alpha = 0.35)
    YM = pd.DataFrame(YM).set_index(time_np)#.values for SOM 
    plt.scatter(YM[YM.columns[0]],YM[YM.columns[1]], c = 'darkblue', alpha = 0.35)
    for i in range(YM.shape[0]):
        plt.text(YM.iloc[i][0], YM.iloc[i][1], YM.index[i])
    
    RSAM = pd.read_parquet('/Users/bste426/Documents/All/PhD/Data/Codes/TremorRegimes/1_Feature_Extraction/RSAM/Augustine/AUW/rsam_Augustine_AUW_10_filtered.parquet')
    plt.plot(RSAM[3100000:4000000], alpha = 0.75)
    plt.legend(RSAM.columns)
    plt.axvspan('2006-01-11 00:00:00', '2006-01-28 00:00:00', alpha=0.25, color='red', label='EX')
    plt.axvspan('2006-01-28 00:00:00', '2006-02-10 00:00:00', alpha=0.25, color='blue', label='CONT')
    plt.axvspan('2006-03-03 00:00:00', '2006-03-16 00:00:00', alpha=0.25, color='yellow', label='EFF')
    plt.ylim(0, 10000)
    legend1 = plt.legend()
    plt.legend(RSAM.columns, loc = 'upper left')
    ppl.gca().add_artist(legend1)
    
    for i in range(8):
        plt.plot(df[df.columns[8+i]])
    '''

    return df, time_np

def training_stage(df, n_clusters, mapsize, lattice, plot_SOM, save_trained_map, CL_Determination):

    Traindata = df

    # This builds the SOM using pre-defined hyperparameters and other default parameters. Initialization can be 'random' or 'pca'.
    som = sompy.SOMFactory.build(Traindata, mapsize, mask=None, mapshape='planar', lattice=lattice, normalization='var', initialization='pca', neighborhood='gaussian', training='batch', name='sompy')  
    som.train(n_job=6, verbose='info') #verbose='info' medium. verbose='debug' will print more, and verbose=None wont print anything
    # Calculate and save topographic and quantization errors
    te = som.calculate_topographic_error()
    qe = som.calculate_quantization_error()
    print ("Topographic error = %s; Quantization error = %s" % (te,qe))
    teqe = (te,qe)
    Errors.append(teqe)

    # Visualise internal structure of SOM
    if plot_SOM is True:
        if not os.path.isdir('OUTPUT/insideSOM'):
            os.makedirs('OUTPUT/insideSOM')
   
        h = sompy.hitmap.HitMapView(30, 30, 'hitmap', text_size=8, show_text=True)
        h.show(som)
        plt.savefig('OUTPUT/insideSOM/hitmap_'+ FILE + EXTNAME[:-4] + '.png', dpi=200)
        plt.close()

        u = sompy.umatrix.UMatrixView(30, 50, 'umatrix', show_axis=True, text_size=8, show_text=True)
        u.show(som, distance2=1, row_normalized=False, show_data=True, contooor=True, blob=False)
        plt.savefig('OUTPUT/insideSOM/uMatrix_'+ FILE + EXTNAME[:-4] + '.png', dpi=200)
        plt.close()
        
    # Use built-in function of SOMPY to perform clustering (k-means)
    cl = som.cluster(n_clusters=n_clusters)

    # save SOM
    if save_trained_map is True:
        if not os.path.isdir('OUTPUT/trained_maps'):
            os.makedirs('OUTPUT/trained_maps')
        joblib.dump(som, 'OUTPUT/trained_maps/' + FILE + EXTNAME)

    # after creating the SOM, perform tests with hypothetical numbers of clusters to gain optimum
    if CL_Determination is True:
        cl_det(som,cl,mapname,plot_curves,degree)

    return Traindata, som

def test_stage(Traindata, n_clusters, time_np, FILE, EXTNAME):
    output = som.project_data(Traindata) # this is where the data (labelled 'Traindata', even though here it is the test data set) are introduced
    cl = som.cluster(n_clusters=n_clusters) # performs clustering
    cloutput = cl[output] #for each data, cloutput contains its assigned cluster
    cloutputDF = pd.DataFrame(cloutput)
    #to_start, to_end = int(to_start), int(to_end)
    #time_np = time_np[to_start:(to_start + to_end)]
    cloutputDF['Time'] = np.array(time_np)
    if not os.path.isdir('OUTPUT/cluster_vectors'):
        os.makedirs('OUTPUT/cluster_vectors')
    cloutputDF.to_csv('OUTPUT/cluster_vectors/clusters_'+ FILE + EXTNAME[:-4] +'.csv', index=False, header=False)

    if not os.path.isdir('OUTPUT/tested_maps'):
        os.makedirs('OUTPUT/tested_maps')

    with open('OUTPUT/tested_maps/' + FILE + EXTNAME, 'wb') as f:
        pickle.dump([time_np, cloutput], f)

def plot_data(FILE, EXTNAME, time_np, monthinterval, volcano, station, fband, n_clusters, map_x, map_y):
    
    # preparing time vector (conversion to timestamps)
    with open('OUTPUT/tested_maps/' + FILE + EXTNAME,'rb') as f:
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
    ax.scatter(time_np[tminimo:tmassimo], cloutput[tminimo:tmassimo]+1, s = 10, c=colormap[colours], alpha = 0.75)
    ax.set_xlim(min(time_np), max(time_np))
    ax.set_ylabel('Cluster number', fontsize = 12)
    ax.grid(linestyle='dotted', which='both')
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    fig.subplots_adjust(hspace=0.05)
    plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)

    # change interval of ticks on x-axis here depending on length of time-period covered by data
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=monthinterval))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    fig.autofmt_xdate()
    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)

    # add events to plot
    #'''
    with open('volcanic_activity_log/{:s}/eruptive_periods.txt'.format(volcano), 'r') as fp:
        tes = [ln.rstrip() for ln in fp.readlines()]
    xcoords = tes
    for xc in xcoords:
        ax.axvline(x = xc, color='k', linestyle='-', linewidth=2, label='_')
    
    '''
    with open('volcanic_activity_log/{:s}/activity.txt'.format(volcano), 'r') as fp:
        act = [ln.rstrip() for ln in fp.readlines()]
    cords = act
    for co in cords:
        ax.axvline(x = co, color='dimgrey', linestyle='--', linewidth=2, label='_')

    ax.axvline(x='2012-08-04 16:52:00', color='k', linestyle='-', linewidth=2, label='eruption')
    ax.axvline(x='2012-09-02 00:00:00', color='dimgrey', linestyle='--', linewidth=2, label='ash emission')
    ax.axvline(x='2012-11-24 00:00:00', color='dimgrey', linestyle=':', linewidth=2, label='observation of lava dome')
    '''

    # add legend and title
    ax.legend(bbox_to_anchor=(1.005, 1.075), loc='upper right', fontsize = 9, ncol = 3)
    TITLE = str(volcano + ', ' + station + '   fband: ' + fband +'   Cluster: ' + str(n_clusters) + '   SOM size: '+str(map_x)+'x'+str(map_y))

    ppl.title(TITLE, loc='left')

    # save plot
    fig.set_size_inches(16, 6)
    if not os.path.isdir('OUTPUT/classification_plots'):
        os.makedirs('OUTPUT/classification_plots')
    path = 'OUTPUT/classification_plots/' + FILE + EXTNAME + '.png'
    plt.savefig(path, dpi=600)

    if interactive is True:
        fig.set_size_inches(16,6)
        ppl.ion()
        ppl.show()
        ppl.pause(1000)

    plt.close()

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
        loc_max_kms = ss.argrelextrema(ss.detrend(np.asarray(km_silhouette)), np.greater)
        loc_min_db = ss.argrelextrema(ss.detrend(np.asarray(db_score)), np.less)
        loc_max_v_meas = ss.argrelextrema(ss.detrend(np.asarray(vmeasure_score)), np.greater)

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

        if not os.path.isdir('OUTPUT/cl_OUTPUT'):
            os.makedirs('OUTPUT/cl_OUTPUT')

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
        ppl.savefig('OUTPUT/cl_OUTPUT/elbow_' + mapname + '.png', dpi=200)


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
        ppl.savefig('OUTPUT/cl_OUTPUT/elbow_derivative'+map+'.png', dpi=200)
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
        ppl.savefig('OUTPUT/cl_OUTPUT/V-measure_' + mapname + '.png', dpi=200)

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
        ppl.savefig('OUTPUT/cl_OUTPUT/silhouette_' + mapname + '.png', dpi=200)

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
        ppl.savefig('OUTPUT/cl_OUTPUT/Davies-Bouldin_' + mapname + '.png', dpi=200)

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
        ppl.savefig('OUTPUT/cl_OUTPUT/BIC(GMM)'+mapname+'.png', dpi=200)


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
        ppl.savefig('OUTPUT/cl_OUTPUT/(Log-)Likelihood_score_' + mapname + '.png', dpi=200)
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
    if not os.path.isdir('OUTPUT/heatmaps/heatmapsTE'):
        os.makedirs('OUTPUT/heatmaps/heatmapsTE')
    plt.savefig('OUTPUT/heatmaps/heatmapsTE/' + 'TE ' + volcano + ' ' + station + ' ' + map + '.png', dpi=300)

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
    if not os.path.isdir('OUTPUT/heatmaps/heatmapsQE'):
        os.makedirs('OUTPUT/heatmaps/heatmapsQE')
    plt.savefig('OUTPUT/heatmaps/heatmapsQE/' + 'QE ' + volcano + ' ' + station + ' ' + map + '.png',
                dpi=300)
    plt.close('all')



###################################
### S T A R T   A N A L Y S I S ###
###################################

for z in range(len(mx)):
    '''Performs analysis for each given map size/shape'''
    print('Calculating SOM with dimensions '+str(mx[z])+'x'+str(my[z]))

    mapsize = [mx[z], my[z]]
    map_x, map_y = mx[z],my[z]
    lattice = map_lattice[z]

    fbands = [] # collects frequency band of SOM
    win_len = []# collects time window length
    Errors = [] # collects topographic and quantization errors of map (for heatmap)

    for file in files:
        print(file) # provide info on where the analysis is at
        Graphics = False  # False disables graphics, e.g. for batch
        logging.getLogger('matplotlib.font_manager').disabled = True
        #plt.interactive(False)

        # Prepare name and path to save SOM
        FILE = file[len(PATH):-4]  # removes ".csv"
        EXTNAME = '_{:s}cl_{:s}x{:s}_{:s}_{:s}.pkl'.format(str(n_clusters),str(map_x),str(map_y),startdate,enddate) # save classification result under this name
        mapname = FILE+EXTNAME[:-4] # full path to save classification result

        # extract frequency band from filename (better naming convention would make this easier; implemented in future)
        fband = file[([index for index, character in enumerate(file) if character == '_'][-3]+1):([index for index, character in enumerate(file) if character == '_'][-2])]
        fbands.append(fband)
        window = file[([index for index, character in enumerate(file) if character == '_'][2]+1):([index for index, character in enumerate(file) if character == '_'][3]-4)]
        win_len.append(window)

        # Load dataframe from startdate to enddate
        df, time_np = load_data(file, startdate, enddate)

        if TESTisTRAIN is True:
            # Train a SOM on each given file in the directory
            Traindata, som = training_stage(df, n_clusters, mapsize, lattice, plot_SOM, save_trained_map, CL_Determination)

            # Introduce test data and plot classification result
            if Test_and_Plot is True:
                test_stage(Traindata, n_clusters, time_np, FILE, EXTNAME)
                plot_data(FILE, EXTNAME, time_np, monthinterval, volcano, station, fband, n_clusters, map_x, map_y)

        else:
            if trainedmap is True:
                
                # load trained SOM
                som = joblib.load(trained_PATH)

                # preparation of test data
                Testdata, time_np = load_data(file, startdate_test, enddate_test)

                # Introduce test data and plot classification result
                EXTNAME = '_{:s}cl_{:s}x{:s}_{:s}_{:s}.pkl'.format(str(n_clusters),str(map_x),str(map_y),startdate_test,enddate_test)
                if Test_and_Plot is True:
                    test_stage(Testdata, n_clusters, time_np, FILE, EXTNAME)
                    plot_data(FILE, EXTNAME, time_np, monthinterval, volcano, station, fband, n_clusters, map_x, map_y)

            else:
                # train SOM on matrix
                _, som = training_stage(df, n_clusters, mapsize, lattice, plot_SOM, save_trained_map, CL_Determination)

                # preparation of test data
                Testdata, time_np = load_data(file, startdate_test, enddate_test)

                # Introduce test data and plot classification result
                EXTNAME = '_{:s}cl_{:s}x{:s}_{:s}_{:s}.pkl'.format(str(n_clusters),str(map_x),str(map_y),startdate_test,enddate_test)
                if Test_and_Plot is True:
                    test_stage(Testdata, n_clusters, time_np, FILE, EXTNAME)
                    plot_data(FILE, EXTNAME, time_np, monthinterval, volcano, station, fband, n_clusters, map_x, map_y)

    if heatmap_on:
        heatmap(Errors, n_clusters, win_len, fbands, mapsize)

if CL_Determination is True:
    try:
        # Concatenates all collected cluster numbers from statistical tests across all map sizes and matrices
        CLM = pd.DataFrame(list(cl_mode))
        CLM.columns = ['cluster']
        CL_fig = sns.displot(CLM, x='cluster', color='black', discrete=True, binwidth=1, stat='density')
        CL_fig.savefig('OUTPUT/cluster_distribution.png', dpi=300)
    except:
        print("REMINDER: Cluster determination can only be implemented when training the SOM, not when testing. Run again with 'TESTisTRAIN' = True or trainedmap = False.")