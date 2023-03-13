import numpy as np
from scipy.stats import gaussian_kde as KDE
import pandas as pd
import scipy.signal as ss
import matplotlib.pylab as plt
from pandas.plotting import register_matplotlib_converters
import matplotlib.dates as mdates
import os
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as ppl
from datetime import datetime, timedelta
from statistics import mean
from collections import Counter
from operator import itemgetter
import warnings
warnings.filterwarnings("ignore")

'''
This code implements a Kernel Density Estimation for the clusters generated during SOM training (see sombatchheat.py).
Once completed, dominant clusters are identified to detect regime changes, which can be performed with or without conditions (e.g., account for KDE inaccuracies and volcano-specific regime characteristics).
The KDE distribution over time and dominant clusters over classified time-windows can be plotted. Output is a list of regimes (.csv) for further analysis.
'''



########################
#### INITIALISATION ####
########################

# What episode would you like to analyse? (UTC)
startdate = '2019-11-09 00:00:00'
enddate = '2019-12-09 21:00:00'

# Location of file containing dates and clusters (classified time-windows)
cl_path = '../../SOM_Carniel/csv2som/Clusters.csv'

# Location of file containing feature
#path_to_featurematrix = '../../SOM_Carniel/csv2som/feature_matrices/whakaari_010800.00wndw_rsam_10_2.00-5.00_data_features.csv'

# What bandwidth would you like to apply to the KDE?
Band_list = np.arange(0.03,0.05,0.01)

# Do you want to run the KDE analysis in raw mode or apply conditions (e.g. modify dominant cluster)?
"""
Note: If no conditional editing of KDE analysis is applied, the number of regimes containing 0 windows (of the
automatically assigned dominant cluster) increases.
"""
corr_duration = False           # Part I (see line 172)
minimum_duration = 3            # minimum regime duration in days (e.g., 5 days for Whakaari in perennial time series analysis) or 'None' if no minimum_duration
consider_KDE_maxima = False     # If this is set to 'True', all regimes, which do not contain a local maximum of the KDE, will be discarded (split up into neighbouring regimes). 

corr_dominant_cluster = True    # Part II (see line 172) Check (and correct) if dominant cluster identified by KDE and the actual dominant cluster (most time windows in a regime) differ.

# Plot regimes over classification result?
plot = True


##############################
#### PERFORM KDE ANALYSIS ####
##############################

clusters_all = pd.read_csv(cl_path,header=None)
to_start = list(clusters_all[0]).index(startdate)
to_end = list(clusters_all[0]).index(enddate)
clusters_all.columns = (['time','cluster'])
clusters_all = clusters_all[to_start:to_end]
###clusters_all in plot!!!!###

time_np = clusters_all['time']  #time vector
cl_vector = (pd.DataFrame(clusters_all['cluster'])).set_index(time_np)

accuracies = [] #capture how good chosen bandwidth is (calculate accuracy of identified dominant clusters)
for B in range(len(Band_list)):

    #################################################
    #### CALCULATE AND PLOT KERNEL DENSITY ESTIMATION
    #################################################

    bandwidth = round(Band_list[B], 2)
    print('Now processing data for bandwidth = {:f}'.format(bandwidth))

    KDEs = {} #prep list to be filled with KDE function for each cluster
    cl = len(Counter(cl_vector['cluster']).values()) #number of clusters from SOM
    t = [datetime.timestamp(datetime.strptime(x, '%Y-%m-%d %H:%M:%S')) for x in time_np] #all timestamps

    plt.figure(figsize=(12,6))
    ax3 = plt.axes()
    colourlist = ['red','purple','blue','cyan','lime']

    for i in range(cl):
        #find indices of each cluster
        indices = cl_vector.index[cl_vector['cluster']==i].tolist() 
        #convert to to timestamps:
        timestamps = []
        for j in range(len(indices)): 
            timestamps.append(datetime.timestamp(datetime.strptime(indices[j], '%Y-%m-%d %H:%M:%S')))
        KDEs["kde{0}".format(i)] = KDE(timestamps, bandwidth)
        ax3.fill_between(t, 0, KDEs["kde{0}".format(i)](t), color=colourlist[i], alpha=0.5)

    x_ticks = ax3.get_xticks()
    xlabels = [datetime.fromtimestamp(float(x)).strftime('%d-%b-%y') for x in x_ticks]
    ax3.set_xticklabels(xlabels)
    ax3.margins(x=0, y=0)
    ax3.set_xlabel('time')
    ax3.set_ylabel('cluster density')
    ax3.text(0.95,0.95,'bandwith={:3.2f}'.format(bandwidth),ha='right',va='top',transform = ax3.transAxes)
    ax3.set_title('Cluster KDE', size=11)
    if not os.path.isdir('plots'):
        os.makedirs('plots')
    plt.savefig('plots/kde_{:s}_{:s}_bw{:s}.png'.format(startdate[:10],enddate[:10],str(bandwidth)), dpi=400)

    ############################################
    #### FIND DOMINANT CLUSTERS FOR REGIMES ####
    ############################################

    dom_CL = []
    date = []

    for i in range(len(time_np)):
        max_val = [max((KDEs["kde{0}".format(x)])(t[i]) for x in range(cl))]    # find maximum value in KDE at each time step
        max_index = [i for i, x in enumerate([max_val in ((KDEs["kde{0}".format(x)])(t[i])) for x in range(cl)]) if x] # which cluster does the max val belong to?
        dom_CL.append(max_index[0])                                             # keep in mind that plots show clusters 1,2,3,... whereas here we deal with 0,1,2,...
        date.append(datetime.strptime(list(time_np)[i],'%Y-%m-%d %H:%M:%S'))    # as timestamp

    reg_list = pd.DataFrame(pd.concat([(pd.DataFrame(date)),(pd.DataFrame(dom_CL))],axis = 1))    # list containing the dominant cluster according to the KDE at each time window date

    # looking for indices where value in 'dom_CL' changes:
    changes = []
    changes.append(np.array(reg_list)[0,:])                             # include episode start date in list
    for i in range(1,len(reg_list)):
        if (np.array(reg_list)[i-1,1])!=(np.array(reg_list)[i,1]):      # if cluster in following window is not the same
            changes.append(np.array(reg_list)[i,:])                     # put new cluster as new regime
        else:
            continue    
    if datetime.strftime((np.array(reg_list)[len(reg_list)-1, :][0]), "%Y-%m-%d") != datetime.strftime(changes[-1][0], "%Y-%m-%d"):
        changes.append(np.array(reg_list)[len(reg_list)-1, :])              # include episode end date in list if no regime change takes place
    else:
        pass

    ###########################################
    #### CONDITIONAL PROCESSING OF REGIMES ####
    ###########################################
    """
    DEFAULT: False
    TREAT WITH CARE - CONDITIONAL REGIME PROCESSING IS NOT ALWAYS APPLICABLE (DEPENDS ON INPUT DATA AND CONTEXT).
    Conditional adaption of regimes can be a neccessary step to optimise interpretability of regimes changes. 
    Adaption steps include forcing of minimum regime durations (e.g., 5 days for Whakaari/White Island) and correction for false detection of dominant cluster (due to KDE bandwidth bluring).
    """

    if corr_duration is False and corr_dominant_cluster is False:
        final_regimes = pd.DataFrame(changes)
        if not os.path.isdir('regimes'):
            os.makedirs('regimes')
        final_regimes.columns = ['time','dom_cluster']
        final_regimes.to_csv('regimes/regimes_nocorr_{:s}_{:s}_bw{:s}.csv'.format(startdate[:10],enddate[:10],str(bandwidth)), index=False)
    
    if corr_duration is True:
        ### CONDITIONAL PROCESSING STEP 1 - CHECK FOR REGIME DURATION AND LOCATION OF KDE MAXIMA ###
        '''
        This part identifies regime durations and corrects regimes if the MINIMUM DURATION CRITERION (specify in the INITIALISATION section) is not met.
        
        OPTIONAL:
        In some cases, KDE gaussian curves of similar clusters overlap, causing unwanted regime changes. To identify such artefacts, local maxima of the given cluster KDE are located.
        If there aren't any local maxima inside a given regime it will be discarded (previous and subsequent regimes extended).
        '''

        final_dates = []
        final_clusters = []
        final_dates.append(np.array(changes[0])[0])      # add starting date and cluster, then start analysis with subsequent changes   
        final_clusters.append(np.array(changes[0])[1])

        ### Check each regime individually:
        for i in range(1,len(changes),1):

            t_start = datetime.timestamp((np.array(changes[i-1])[0]))       # regime start time
            t_end = datetime.timestamp((np.array(changes[i])[0]))           # regime end time
            u = [x for x in t if x >= t_start and x <= t_end]               # times for specific regime (for kdeX(u))

            
            if minimum_duration != 'None':

                # check whether regime duration criterion is met:
                if ((np.array(changes[i])[0])-(np.array(changes[i-1])[0])).days > minimum_duration:        
                    DURATION_i = True
                    #min. duration criterion met
                else:
                    DURATION_i = False
                    #min. duration criterion not met   

            else:
                DURATION_i = True
                     

            if consider_KDE_maxima is True:

                # load KDE for dominant cluster
                x = (np.array(changes[i-1])[1])-1
                kdeX = KDEs['kde{0}'.format(x)]      
                
                # locate local maxima:
                KDE_max_vals = [list(kdeX(t))[x] for x in (np.array(ss.argrelextrema(kdeX(t), np.greater)))[0].tolist()]
                
                # check whether regime includes KDE maxima:
                if max(kdeX(u)) in KDE_max_vals:                                   
                    KDE_max_i = True
                    #maximum lays in regime
                else:
                    KDE_max_i = False
                    #no maximum detected in regime

            else:
                KDE_max_i = True

            # if both criteria met, include regime_date in final list
            if DURATION_i and KDE_max_i is True:
                final_dates.append(np.array(changes[i])[0])         
                final_clusters.append(np.array(changes[i])[1])

            # if min. duration criteria is not met:
            elif DURATION_i is False:
                if i != 1:
                    if (np.array(changes[i])[1])==(np.array(changes[i-2])[1]):              # if clusters of previous and following regime are the same...
                        final_dates.pop()                                                   # ... previous regime will be extended to include current one.
                        final_clusters.pop()
                    else:                                                                   
                        final_dates.pop()                                                   # ... otherwise split current regime up into the previous and the subsequent one (50% each)
                        final_clusters.pop()
                        middle_date = (np.array(changes[i])[0])-(((np.array(changes[i])[0])-(np.array(changes[i-1])[0]))/2).round('1d')    
                        final_dates.append(middle_date)                              
                        final_clusters.append(np.array(changes[i])[1])
                else:
                    pass       # just to make sure the first regime is longer than 5 days - otherwise go directly to the second
            
            # if no local KDE maximum was found inside regime:
            else:
                if i != 1:
                    if (np.array(changes[i])[1])==(np.array(changes[i-2])[1]):
                        final_dates.pop()
                        final_clusters.pop()

                    elif (np.array(changes[i])[1]) != (np.array(changes[i - 2])[1]):
                        slope_KDE = kdeX(t_end) - kdeX(t_start)                 # Determine slope of KDE curve.
                        if slope_KDE < 0:                                       # If KDE slope is negative...
                            final_dates.pop()
                            final_clusters.pop()
                            final_dates.append((np.array(changes[i])[0]))       # ... use start date of current regime...
                            final_clusters.append((np.array(changes[i])[1]))    # ... and merge with previous cluster.
                        elif slope_KDE > 0:                                     # If KDE slope is positive...
                            final_clusters.pop()
                            final_clusters.append((np.array(changes[i])[1]))     # ... use cluster of current regime for subsequent regime.
                else:
                    pass       # just to make sure the first regime contains a local KDE maximum - otherwise go directly to the second

        final_dates.append(changes[-1][0])      #add end date
        final_clusters.append(changes[-1][1])   #add last active cluster

        final_regimes = pd.DataFrame(pd.concat([pd.DataFrame(final_dates), pd.DataFrame(final_clusters)], axis=1)) #combined list of regime changes
        final_regimes.columns = ['time','dom_cluster']
        if corr_dominant_cluster is False:
            final_regimes.to_csv('regimes/regimes_corr_{:s}_{:s}_bw{:s}.csv'.format(startdate,enddate,str(bandwidth)), index=False)

    if corr_dominant_cluster is True:
        ### CONDITIONAL PROCESSING STEP 2 - CHECK WHETHER KDE-SUGGESTED DOMINANT CLUSTER MATCHES ACTUAL DOMINANT CLUSTER ###
        '''
        This part checks whether the dominant cluster based on the KDE and the actual dominant cluster (the cluster most time windows in a regime are assigned to).
        If this is not the case, the dominant cluster will be corrected.
    
        LIMITATION: There is a natural offset between the gaussian curve of the KDE and actual regime changes (i.e., a change in the dominant cluster occurs before the KDE detects it).
                    Correcting the dominant cluster based on potentially imprecise KDE-based regimes could therefore cause a mismatch between the 'corrected' dominant cluster and the actual dominant cluster.
    
        ADVICE: If your dataset is biased in terms of cluster distribution (some clusters occuring a lot more often than others), checking the KDE result is a good idea.
                Otherwise, especially for short datasets, set modify_dominant_cluster to 'False'.
        '''
        
        if corr_duration is False:
            final_regimes = pd.DataFrame(changes)
            final_regimes.columns = ['time','dom_cluster']

        for i in range(len(final_regimes)-1):

            reg_start = cl_vector.index.get_loc(str(final_regimes['time'][i]))
            reg_end = cl_vector.index.get_loc(str(final_regimes['time'][i+1]))

            # Determine cluster with most windows in regime (dom_cl):
            cluster_regime = cl_vector['cluster'][reg_start:reg_end]
            unique_clusters = np.vstack(np.unique(cluster_regime.values, return_counts=True)).T   # array with clusters and their occurence
            
            # Return the actual dominant cluster:
            dom_cl = max(unique_clusters,key=itemgetter(1))[0]
            # Dominant cluster identified by KDE:
            KDE_cl = final_regimes['dom_cluster'][i]

            # If they match, KDE did a good job. If not...
            if dom_cl != KDE_cl:
                
                # ...we can first check if actual dominant cluster is the cluster most time windows have been assigned to (refered to as 'base cluster').
                base_cluster = max(np.vstack(np.unique(cl_vector.values, return_counts=True)).T,key=itemgetter(1))[0]
                
                # If the base cluster occurs significantly more frequently than other clusters, then the actual dominant cluster might be the base cluster.
                # However, the KDE is supposed to detect changes in the relative distribution - so the KDE might be right picking out a less frequent cluster as the dominant.
                # It is therefore important to check whether the KDE might be right and, if it is, stick to the dominant cluster returned by the KDE.
                if dom_cl != base_cluster:
                    final_regimes['dom_cluster'][i] = dom_cl    # If not base cluster, change to actual dominant cluster (usually a good choice, especially for short regimes).
                
                # If base cluster IS the actual dominant cluster...
                elif dom_cl == base_cluster:   

                    # ...check whether there is a KDE low-point in the base cluster (KDE problem: base cluster still dominant although density decreased):
                    basecluster_lowpoints = (ss.argrelextrema(KDEs['kde{0}'.format(base_cluster)](t), np.less))[0].tolist()  # indices of local minima
                    lowpoint_dates = [date[x] for x in basecluster_lowpoints]    # dates of local minima (reminder: 'date' is the list of datetimes for entire dataset)

                    # If so, don't change to dominant cluster and stick to KDE cluster:
                    if any(item in lowpoint_dates for item in date[reg_start:reg_end]):
                        continue

                    # If no low point in base cluster, change to actual dominant cluster
                    else:
                        final_regimes['dom_cluster'][i] = dom_cl
                        
                        '''
                        The complicated way (old code - should be disregarded):

                        # If less than 12.5% (worked best) of regime windows belong to suggested cluster, change to dom_cl.
                        if np.count_nonzero(y == int(final_regimes[i][1])) < 0.125 * np.count_nonzero(y == dom_cl):
                            final_regimes['dom_cluster'][i] = dom_cl

                        # If balanced window number, check if there is another cluster with more windows != dom_cl.
                        else:
                            continue
                            
                            max_array = []
                            remaining_cluster = [x for x in [1,2,3,4,5] if x != dom_cl and x != int(final_regimes[i][1])]
                            for x in range(len(remaining_cluster)):
                                max_array.append(np.count_nonzero(y == int(remaining_cluster[x])))
                            dom_cl2 = (list(max_array).index(max(max_array)))+1
        
                            if (np.count_nonzero(y != dom_cl2)) > (np.count_nonzero(y != int(final_regimes[i][1]))):
                                final_regimes[i][1] = dom_cl2   # If so, change regime cluster to second dominant cluster.
                            else:
                                final_regimes[i][1] = dom_cl    # If not, change to base cluster.

                        '''
        
        final_regimes.to_csv('regimes/regimes_corr_{:s}_{:s}_bw{:s}.csv'.format(startdate[:10],enddate[:10],str(bandwidth)), index=False)


        '''
        OLD CODE:
        corrected_regimes = []

        for i in range(len(final_regimes)-1):
            if final_regimes[i][1] != final_regimes[i+1][1]:
                corrected_regimes.append(final_regimes[i+1])
        final_regimes[-1][1] = final_regimes[-2][1]
        corrected_regimes.append(final_regimes[-1])

        if corrected_regimes[-1][0] == corrected_regimes[-2][0]:  # just to make sure start and end date are not included twice...
            corrected_regimes.pop(-1)
        if corrected_regimes[0][0] == corrected_regimes[1][0]:
            corrected_regimes.pop(0)
        final_regimes = pd.DataFrame(corrected_regimes)

        # add end dates to list of start dates
        startdates = []
        enddates = []
        clusters = []
        delta = timedelta(days=1)

        startdates.append(datetime.strptime(str(corrected_regimes[0][0])[:10],'%Y-%m-%d'))
        clusters.append(corrected_regimes[0][1])
        for i in range(1,len(corrected_regimes)-1,1):
            startdates.append(datetime.strptime(str(corrected_regimes[i][0])[:10],'%Y-%m-%d'))
            enddates.append(datetime.strptime(str(corrected_regimes[i][0])[:10],'%Y-%m-%d'))
            #enddates.append(datetime.strptime(str(final_regimes2[i][0])[:10], '%Y-%m-%d') - delta)
            clusters.append(corrected_regimes[i][1])
        enddates.append(datetime.strptime(str(corrected_regimes[-1][0])[:10],'%Y-%m-%d'))
        MJC = [0 for x in range(len(corrected_regimes)-1)]
        MNC = [0 for x in range(len(corrected_regimes)-1)]

        mod_list = {'startdate': [datetime.strftime(x, '%Y-%m-%d') for x in startdates], 'enddates': [datetime.strftime(x, '%Y-%m-%d') for x in enddates], 'dominant clusters': clusters, 'major clusters': MJC, 'minor clusters': MNC}
        mod_list = pd.DataFrame(mod_list)
        if not os.path.isdir('regimes'):
            os.makedirs('regimes')
        mod_list.to_csv('regimes/regimes_corr_{:s}_{:s}_bw{:s}.csv'.format(startdate,enddate,str(bandwidth)), header = False, index=False)
        '''  

    ###################################################
    #### PLOT REGIMES OVER CLASSIFIED TIME WINDOWS ####
    ###################################################

    if plot is True:

        register_matplotlib_converters()  # to avoid warnings

        fig, axarr = plt.subplots(1, sharex=True, sharey=False)
        fig.set_size_inches(16, 9)
        axarr.set_yticklabels([])
        axarr.axis('off')
        ax = fig.add_subplot(1, 1, 1, sharex=fig.axes[0])
        colormap = np.array(['r', 'purple', 'b', 'cyan', 'lime', 'yellow', 'darkgrey'])
        t_min = 0
        t_max = np.shape(date)[0]
        colours = clusters_all['cluster'][t_min:t_max] % colormap.shape[0]
        ax.scatter(date[t_min:t_max], clusters_all['cluster'][t_min:t_max]+1, c=colormap[colours]) # we add 1 to the cluster number so that the plot shows clusters 0,1,2,... and not 1,2,3,...
        ax.set_xlim(min(date), max(date))
        ax.set_ylabel('Cluster number')
        ax.grid(linestyle='dotted', which='both')
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        # plt.pause(0.001)
        fig.subplots_adjust(hspace=0.05)
        plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)

        # ax.xaxis.set_major_locator(mdates.YearLocator(base=1, month=1, day=1, tz=None))
        # ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))

        xlabels = [datetime.fromtimestamp(float(x)).strftime('%d-%b-%y') for x in x_ticks]
        ax3.set_xticklabels(xlabels)
        ax.axvline(x='2019-12-09 01:11:00', color='k', linestyle='-', linewidth=2, label='eruption')
        for i in range(len(final_regimes)-1):
            plt.axvspan((final_regimes['time'][i]), (final_regimes['time'][i+1]), facecolor=np.array(colormap)[(final_regimes['dom_cluster'][i])], alpha=0.15)

        # add legend and title
        #ax.legend(bbox_to_anchor=(0.79, 1))
        TITLE = str('auto_regimes')
        ppl.title(TITLE, loc='left')

        # save plot
        fig.set_size_inches(16, 9)
        if not os.path.isdir('__OUTPUT'):
            os.makedirs('__OUTPUT')
        path = '__OUTPUT/auto_regimes_{:s}_{:s}_bw{:s}.png'.format(startdate[:10],enddate[:10],str(bandwidth))
        plt.savefig(path, dpi=200)

        plt.close()

    #########################
    #### BANDWIDTH CHECK ####
    #########################
    """
    For each regime, check whether the ratio between correct windows (i.e. windows belonging to the cluster predicted by the auto_regime)
    and false windows. Find best bandwidth to optimise the ratio of correctly identified dominant clusters in all regimes, i.e. find highest mean score.
    """

    scores = []
    for i in range(len(final_regimes)-1):
        reg_start = (cl_vector.index.tolist()).index(str(final_regimes['time'][i]))
        reg_end = (cl_vector.index.tolist()).index(str(final_regimes['time'][i+1]))
        cluster_regime = cl_vector['cluster'][reg_start:reg_end]
        y = np.array((cluster_regime))

        correct = np.count_nonzero(y == int(final_regimes['dom_cluster'][i]))  # number of windows assigned to predicted cluster (correct)
        not_correct = np.count_nonzero(y != int(final_regimes['dom_cluster'][i]))   # number of remaining windows assigned to other clusters (false)
        scores.append(correct/(correct+not_correct))

    mean_score = mean(scores)
    accuracies.append(mean_score*100) # in %


############################
#### CHECK KDE ACCURACY ####
############################
plt.close('all')
bws = [] #collects bandwidths as rounded values (otherwise doesn't work in the list)
for i in range(len(Band_list)):
    bw = round(Band_list[i],2)
    bws.append(bw)
plt.scatter(bws, accuracies, color = 'darkred') #plot distribution of classification accuracy across bandwidths
plt.vlines(bws, accuracies, 0, color='darkred', alpha=0.2, linewidth=5)
plt.xlabel('KDE bandwidth')

#indicate maximum accuracy values as good KDE performance
local_maxima = (np.array(ss.argrelextrema(np.array(accuracies), np.greater)))[0].tolist()
for i in range(len(local_maxima)):
    plt.scatter(bws[local_maxima[i]], accuracies[local_maxima[i]], s = 150, facecolors='none', edgecolors='black', label = 'peak performance' if i == 0 else "") #highlights maximum
    plt.legend(bbox_to_anchor=(1, 1), fontsize = 10)
    plt.annotate(str(bws[local_maxima[i]]), xy=(bws[local_maxima[i]], accuracies[local_maxima[i]]), xytext=(bws[local_maxima[i]], accuracies[local_maxima[i]]+1), ha="center", size = 8)

#plt.xticks(bws)    #activate if only few bandwidths
plt.ylabel('KDE accuracy [%]')
plt.ylim(np.min(accuracies)-10, np.max(accuracies)+10 if np.max(accuracies)+10<100 else 100)
plt.title('KDE performance', size=15)
path = 'regimes/regimes_accuracy_plot.png'
plt.savefig(path, dpi=400)
accuracies = pd.DataFrame(accuracies, index=bws)
accuracies.columns = ['KDE accuracy']
accuracies = accuracies.rename_axis('Time')
accuracies.to_csv('regimes/regimes_accuracy.csv')
