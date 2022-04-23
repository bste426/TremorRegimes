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
from KDEpy import FFTKDE



#### INITIALISATION ####

# Wanna see colorful images?
plot = True     # Sure!
# Do you want to run the KDE analysis in raw mode or apply conditions (e.g. modify dominant cluster)?
conditioning = True
"""
Note: If no conditional editing of KDE analysis is applied, the number of regimes containing 0 windows (of the
automatically assigned dominant cluster) increases.)
"""

# What bandwidth would you like to apply to the KDE?
Band_list = np.arange(0.21,0.22,0.01)

# What episode would you like to analyse?
startdate = '2019-01-01'  # min.: 2008-06-02 (2-day and 5-day matrix) --- further: 2011-05-28|2014-01-02|2018-12-27
enddate = '2020-01-01'

# Where 'dem clusterz at:
cl_path = '/Users/bst/Documents/All/PhD/Data/Codes/feature_ranking/cluster_vectors/Clusters 2016_20.csv'

# Empty list for scores (optimise bandwidth)
accuracies = []
"""
A KDE creates a continuous function by stacking normal distributions
at each data point. Choose the width of each distribution (bandwidth) so
there is a good amount of overlap between neighbouring points.

For your problem, start with a bandwidth that is maybe 10x your window
overlap, e.g., if overlap is 12 hours, try a bandwidth of 120 hours.

Modify the bandwidth until you think you are see helpful patterns in the 
clusters.
"""


for B in range(len(Band_list)):

    bandwidth = round(Band_list[B], 2)
    print('Now processing data for bandwidth = {:f}'.format(bandwidth))

    #### CALCULATE AND PLOT KERNEL DENSITY ESTIMATION

    if plot is True:
        plt.figure(figsize=(12,6))
        ax3 = plt.axes()

    clusters_all = pd.read_csv(cl_path,header=None)
    time_np = pd.read_csv('/Users/bst/Desktop/results/time_vector', header=None)[1]
    dates = list(time_np)

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

    time_np = pd.read_csv('/Users/bst/Desktop/results/time_vector', header=None, skiprows=(int(to_start)), nrows=(int(to_end)))[1]
    cl_vector = pd.DataFrame(time_np)
    cl_vector['cluster'] = clusters_all

    cl_lists = {}

    for i in range(5):
        row = cl_vector[cl_vector['cluster'].isin([i])]
        cl_lists["cluster_new{0}".format(i)] = []
        cl_lists["cluster_new{0}".format(i)].append(row[1])

    # This is a terrible seection:
    cl_1 = cl_lists["cluster_new0"]
    cl_2 = cl_lists["cluster_new1"]
    cl_3 = cl_lists["cluster_new2"]
    cl_4 = cl_lists["cluster_new3"]
    cl_5 = cl_lists["cluster_new4"]

    time_np2_1 = []
    time_np2_2 = []
    time_np2_3 = []
    time_np2_4 = []
    time_np2_5 = []

    te1 = np.array(cl_1).T
    te2 = np.array(cl_2).T
    te3 = np.array(cl_3).T
    te4 = np.array(cl_4).T
    te5 = np.array(cl_5).T

    for x in range(len(te1)):
        new = datetime.timestamp(datetime.strptime(str(te1[x])[2:-2], '%Y-%m-%d %H:%M:%S'))
        time_np2_1.append(float(new))
    for x in range(len(te2)):
        new = datetime.timestamp(datetime.strptime(str(te2[x])[2:-2], '%Y-%m-%d %H:%M:%S'))
        time_np2_2.append(int(new))
    for x in range(len(te3)):
        new = datetime.timestamp(datetime.strptime(str(te3[x])[2:-2], '%Y-%m-%d %H:%M:%S'))
        time_np2_3.append(int(new))
    for x in range(len(te4)):
        new = datetime.timestamp(datetime.strptime(str(te4[x])[2:-2], '%Y-%m-%d %H:%M:%S'))
        time_np2_4.append(int(new))
    for x in range(len(te5)):
        new = datetime.timestamp(datetime.strptime(str(te5[x])[2:-2], '%Y-%m-%d %H:%M:%S'))
        time_np2_5.append(int(new))

    times_1 = np.array(time_np2_1)
    times_2 = np.array(time_np2_2)
    times_3 = np.array(time_np2_3)
    times_4 = np.array(time_np2_4)
    times_5 = np.array(time_np2_5)

    # use a KDE to create a density function for both clusters and overlap them

    start = int((min(times_1)))
    end = int((max(times_1)))
    steps = (int(max(times_1))-int(min(times_1)))/43200

    t = np.arange(start, end, 43200)

    #kde1, y = FFTKDE(kernel='cosine', bw='ISJ').fit(times_1.T).evaluate()
    kde1 = KDE(times_1.T, bandwidth)
    kde2 = KDE(times_2.T, bandwidth)
    kde3 = KDE(times_3.T, bandwidth)
    kde4 = KDE(times_4.T, bandwidth)
    kde5 = KDE(times_5.T, bandwidth)

    if plot is True:
        #ax3.fill_between(kde1, y, color = 'red', alpha=0.5)
        ax3.fill_between(t, 0, kde1(t), color = 'red', alpha=0.5)
        ax3.fill_between(t, 0, kde2(t), color = 'purple', alpha=0.5)
        ax3.fill_between(t, 0, kde3(t), color = 'blue', alpha=0.5)
        ax3.fill_between(t, 0, kde4(t), color = 'cyan', alpha=0.5)
        ax3.fill_between(t, 0, kde5(t), color = 'lime', alpha=0.5)

        date_list = pd.date_range(datetime.fromtimestamp(start), periods=int(steps/2+1)).tolist()   # create list of days
        date_list = [d.strftime('%Y%m%d') for d in date_list]                                       # convert to str
        date_list = [x for x in date_list if '01' in x[-2:]]                                        # only months
        date_list = [datetime.strptime(d,'%Y%m%d') for d in date_list]                              # back to datetime
        new_dates = [datetime.timestamp(d) for d in date_list]                                      # list of months in timestamp
        date_array = np.array(new_dates)                                                            # timestamps to array
        plt.xticks(date_array[::3])                                                                 # new ticks
        x_ticks = ax3.get_xticks()
        xlabels = [datetime.fromtimestamp(float(x)).strftime('%b-%y') for x in x_ticks]               # convert tick labels to new format
        ax3.set_xticklabels(xlabels)

        ax3.margins(x=0, y=0)
        ax3.set_xlabel('time')
        ax3.set_ylabel('cluster density')
        ax3.text(0.95,0.95,'bandwith={:3.2f}'.format(bandwidth),ha='right',va='top',transform = ax3.transAxes)
        ax3.set_title('KDE for Clusters of the 2011-2014 episode', size=11)

        if not os.path.isdir('plots'):
            os.makedirs('plots')
        plt.savefig('/Users/bst/Documents/All/PhD/Data/Codes/regime_detection/kde/plots/kde_{:s}_{:s}_bw{:s}.png'.format(startdate,enddate,str(bandwidth)), dpi=400)


    #### CREATE REGIME LIST ####

    # detect highest density value for each window in time_np -> create list with DATE | CLUSTER w. highest value
    # based on new list: create txt file with START_DATE | END_DATE | CLUSTER

    dom_CL = []
    date = []

    for w in range(len(time_np)-1):
        max_val = max([kde1(t[w]),kde2(t[w]),kde3(t[w]),kde4(t[w]),kde5(t[w])])             # find maximum value in KDE
        max_index = [kde1(t[w]),kde2(t[w]),kde3(t[w]),kde4(t[w]),kde5(t[w])].index(max_val) # find index (date)
        dom_CL.append(max_index+1)
        date.append(datetime.strptime(time_np[w],'%Y-%m-%d %H:%M:%S')) # as weird timestamp

    dom_CL = pd.DataFrame(dom_CL)
    date = pd.DataFrame(date)
    reg_list = pd.concat([date,dom_CL],axis = 1)    # list containing the dominant cluster according to the KDE and date

    reg_list = pd.DataFrame(reg_list)

    # looking for index where dom_CL changes:
    changes = []

    changes.append(np.array(reg_list)[0,:])                            # include episode start date in list
    for i in range(1,len(reg_list)):
        if (np.array(reg_list)[i-1,1])!=(np.array(reg_list)[i,1]):      # if cluster in following window is not the same
            changes.append(np.array(reg_list)[i,:])                     # put new cluster as new regime
        else:
            continue
    changes.append(np.array(reg_list)[len(reg_list)-1, :])              # include episode end date in list

    if conditioning is False:
        final_regimes = changes
        final_regimes2 = final_regimes
        final_regimes = pd.DataFrame(final_regimes)
        if not os.path.isdir('regimes'):
            os.makedirs('regimes')
        final_regimes.to_csv('/Users/bst/Documents/All/PhD/Data/Codes/regime_detection/kde/regimes/regimes_nocorr_{:s}_{:s}_bw{:s}.csv'.format(startdate,enddate,str(bandwidth)), index=False)



    #### CONDITIONAL PROCESSING OF REGIMES ####

    else:

        final_dates = []
        final_clusters = []
        bool_matrix_KDE = []
        bool_matrix_DUR = []

        ### KICK OUT REGIMES < 5d
        for i in range(1,len(changes),1):

            t_start = datetime.timestamp((np.array(changes[i-1])[0]))
            t_end = datetime.timestamp((np.array(changes[i])[0]))
            u = [x for x in t if x >= t_start and x <= t_end]             # times for specific regime (for kdeX(u))

            if (np.array(changes[i-1])[1]) == 1:
                kdeX = kde1
            if (np.array(changes[i-1])[1]) == 2:
                kdeX = kde2
            if (np.array(changes[i-1])[1]) == 3:
                kdeX = kde3
            if (np.array(changes[i-1])[1]) == 4:
                kdeX = kde4
            if (np.array(changes[i-1])[1]) == 5:
                kdeX = kde5

            KDE_max_vals = [list(kdeX(t))[x] for x in (np.array(ss.argrelextrema(kdeX(t), np.greater)))[0].tolist()]

            if max(kdeX(u)) in KDE_max_vals:                                   # check whether regime includes KDEmax
                KDE_max_i = True
            else:
                KDE_max_i = False

            if ((np.array(changes[i])[0])-(np.array(changes[i - 1])[0])).days > 5:        # check whether regime duration > 5 days
                DURATION_i = True
            else:
                DURATION_i = False

            bool_matrix_DUR.append(DURATION_i)
            bool_matrix_KDE.append(KDE_max_i)

            # YES | YES
            #if DURATION_i and KDE_max_i is True:
            if DURATION_i is True:
                final_dates.append(np.array(changes[i])[0])         # include regime_date in final list
                final_clusters.append(np.array(changes[i])[1])      # include regime_cluster in final list

            # NO | YES
            #elif DURATION_i is False and KDE_max_i is True:
            elif DURATION_i is False:
                if (np.array(changes[i])[1])==(np.array(changes[i-2])[1]):              # if clusters of previous and following regime are equal...
                    final_dates.pop()                                             # ... don't include these regimes in final list and delete previous one
                    final_clusters.pop()
                else:                                                                   # if pre and post cluster differ...
                    final_dates.pop()                                             # ... don't include these regimes in final list and delete previous one
                    final_clusters.pop()
                    date = (np.array(changes[i])[0])-((np.array(changes[i])[0])-(np.array(changes[i-1])[0]))/2
                    if str(date)[11:13] == '18':
                        delta = timedelta(hours=6)
                        date = date + delta
                    if str(date)[11:13] == '06':
                        delta = timedelta(hours=6)
                        date = date - delta
                    final_dates.append(date)
                                                                                        # ... split up regime into both (drag start date of current regime 50% earlier)
                    final_clusters.append(np.array(changes[i])[1])
            '''
            # YES | NO  and  NO | NO
            else:
                slope_KDE = kdeX(t_end) - kdeX(t_start)
    
                if (np.array(changes[i])[1])==(np.array(changes[i-2])[1]):
                    final_dates.pop()
                    final_clusters.pop()
    
                elif (np.array(changes[i])[1]) != (np.array(changes[i - 2])[1]):
                    if slope_KDE < 0:                                       # if KDE slope is negative...
                        final_dates.pop()
                        final_clusters.pop()
                        final_dates.append((np.array(changes[i])[0]))       # ... use date of current regime and previous cluster
                        final_clusters.append((np.array(changes[i])[1]))
                    elif slope_KDE > 0:                                     # if KDE slope is positive...
                        final_clusters.pop()
                        final_clusters.append((np.array(changes[i])[1]))     # ... use cluster of current regime for previous date
            '''

        final_dates = pd.DataFrame(final_dates)
        final_clusters = pd.DataFrame(final_clusters)
        comb = pd.DataFrame(pd.concat([final_dates, final_clusters], axis=1))
        final_regimes = []
        final_regimes.append(np.array(reg_list)[0, :])
        for i in range(len(comb)):
            final_regimes.append(np.array(comb)[i,:])

        # check if suggested regime cluster correspond to actual dominant cluster (i.e.: Has the majority of the windows been assigned to the regime cluster?)
        for i in range(len(final_regimes)-1):

            # Determine cluster with most windows in regime (dom_cl):
            reg_start = int(list(cl_vector[:][1]).index(str(final_regimes[i][0])))
            reg_end = int(list(cl_vector[:][1]).index(str(final_regimes[i+1][0])))
            cluster_regime = cl_vector['cluster'][reg_start:reg_end]
            y = np.array((cluster_regime)+1)                # all clusters within regime
            N1 = np.count_nonzero(y == 1)
            N2 = np.count_nonzero(y == 2)
            N3 = np.count_nonzero(y == 3)
            N4 = np.count_nonzero(y == 4)
            N5 = np.count_nonzero(y == 5)
            var = {N1: "1", N2: "2", N3: "3", N4: "4", N5: "5"}
            dom_cl = int(var.get(max(var)))

            if dom_cl == final_regimes[i][1]:
                continue    # Dominant cluster was correctly determined.

            # If dominant cluster is not correct, check if actual dominant cluster is base cluster.
            # Base cluster is dominant (i.e. most windows assigned to it) most of the time, hence distorting KDE results.
            # Base cluster must be red cluster 1! If not, change code.

            else:
                if dom_cl != 1:
                    final_regimes[i][1] = dom_cl    # If not base cluster, change to dominant cluster (usually good choice).
                elif dom_cl == 1:   # If base cluster is dominant cluster (like in most cases)...

                    # ...check whether there is a low-point in the base cluster:
                    u_new = list(time_np[reg_start:reg_end])
                    kde1_all_lowpoints = np.array(ss.argrelextrema(kde1(t), np.less))[0].tolist()
                    lowpoint_dates = [time_np[x] for x in kde1_all_lowpoints]
                    # If so, don't change to dominant cluster.
                    # (KDE problem: base cluster still dominant although KDE goes down).
                    if any(item in lowpoint_dates for item in u_new):
                        continue

                    # If no low point in base cluster...
                    else:
                        # ...AND if less than 12.5% of regime windows belong to suggested cluster, change to dom_cl.
                        if np.count_nonzero(y == int(final_regimes[i][1])) < 0.125 * np.count_nonzero(y == dom_cl):
                            final_regimes[i][1] = dom_cl
                        # If balanced window number, check if there is another cluster with more windows != dom_cl.
                        else:
                            continue
                            '''
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

        final_regimes2 = []

        final_regimes2.append(np.array(reg_list)[0, :])
        for i in range(len(final_regimes)-1):
            if final_regimes[i][1] != final_regimes[i+1][1]:
                final_regimes2.append(final_regimes[i+1])
        final_regimes[-1][1] = final_regimes[-2][1]
        final_regimes2.append(final_regimes[-1])

        if final_regimes2[-1][0] == final_regimes2[-2][0]:  # just to make sure start and end date are not included twice...
            final_regimes2.pop(-1)
        if final_regimes2[0][0] == final_regimes2[1][0]:
            final_regimes2.pop(0)
        final_regimes = pd.DataFrame(final_regimes2)

        # add end dates to list of start dates
        startdates = []
        enddates = []
        clusters = []
        delta = timedelta(days=1)

        startdates.append(datetime.strptime(str(final_regimes2[0][0])[:10],'%Y-%m-%d'))
        clusters.append(final_regimes2[0][1])
        for i in range(1,len(final_regimes2)-1,1):
            startdates.append(datetime.strptime(str(final_regimes2[i][0])[:10],'%Y-%m-%d'))
            enddates.append(datetime.strptime(str(final_regimes2[i][0])[:10],'%Y-%m-%d'))
            #enddates.append(datetime.strptime(str(final_regimes2[i][0])[:10], '%Y-%m-%d') - delta)
            clusters.append(final_regimes2[i][1])
        enddates.append(datetime.strptime(str(final_regimes2[-1][0])[:10],'%Y-%m-%d'))
        MJC = [0 for x in range(len(final_regimes2)-1)]
        MNC = [0 for x in range(len(final_regimes2)-1)]

        mod_list = {'startdate': [datetime.strftime(x, '%Y-%m-%d') for x in startdates], 'enddates': [datetime.strftime(x, '%Y-%m-%d') for x in enddates], 'dominant clusters': clusters, 'major clusters': MJC, 'minor clusters': MNC}
        mod_list = pd.DataFrame(mod_list)
        if not os.path.isdir('regimes'):
            os.makedirs('regimes')
        mod_list.to_csv('/Users/bst/Documents/All/PhD/Data/Codes/regime_detection/kde/regimes/regimes_corr_{:s}_{:s}_bw{:s}.csv'.format(startdate,enddate,str(bandwidth)), header = False, index=False)


    #### PLOTTING STAGE ####

    if plot is True:

        register_matplotlib_converters()  # to avoid warnings

        regimes = final_regimes

        time_np2 = []
        for x in range(len(time_np)):
            try:
                time_np2.append(datetime.strptime(str(time_np[x]), '%Y-%m-%d %H:%M:%S'))
            except:
                time_np2.append(datetime.strptime(str(time_np[x]), '%Y-%m-%d'))

        time_np = np.array(time_np2)
        clusters_all = clusters_all[0].to_numpy()

        fig, axarr = plt.subplots(1, sharex=True, sharey=False)
        fig.set_size_inches(16, 9)
        axarr.set_yticklabels([])
        axarr.axis('off')
        ax = fig.add_subplot(1, 1, 1, sharex=fig.axes[0])
        colormap = np.array(['r', 'purple', 'b', 'cyan', 'lime'])
        tminimo = 0
        tmassimo = time_np.shape[0]
        colours = clusters_all[tminimo:tmassimo] % colormap.shape[0]
        ax.scatter(time_np[tminimo:tmassimo], clusters_all[tminimo:tmassimo] + 1, c=colormap[colours])
        ax.set_xlim(min(time_np), max(time_np))
        ax.set_ylabel('Cluster number')
        ax.grid(linestyle='dotted', which='both')
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        # plt.pause(0.001)
        fig.subplots_adjust(hspace=0.05)
        plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)

        # ax.xaxis.set_major_locator(mdates.YearLocator(base=1, month=1, day=1, tz=None))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        fig.autofmt_xdate()

        # add events to plot
        with open('/Users/bst/Documents/All/PhD/Data/Codes/SOM_Carniel/csv2som/act_log/eruptive_periods.txt', 'r') as fp:
            tes = [ln.rstrip() for ln in fp.readlines()]
        xcoords = tes
        for xc in xcoords:
            ax.axvline(x=xc, color='k', linestyle='-', linewidth=2, label='_')

        with open('/Users/bst/Documents/All/PhD/Data/Codes/SOM_Carniel/csv2som/act_log/activity.txt', 'r') as fp:
            act = [ln.rstrip() for ln in fp.readlines()]
        cords = act
        for co in cords:
            ax.axvline(x=co, color='dimgrey', linestyle='--', linewidth=2, label='_')

        ax.axvline(x='2012-08-04 16:52:00', color='k', linestyle='-', linewidth=2, label='eruption')
        ax.axvline(x='2012-09-02 00:00:00', color='dimgrey', linestyle='--', linewidth=2, label='ash emission')
        ax.axvline(x='2012-11-24 00:00:00', color='dimgrey', linestyle=':', linewidth=2, label='observation of lava dome')

        for i in range(len(regimes) - 1):
            plt.axvspan((np.array(regimes)[i][0]), (np.array(regimes)[i + 1][0]), facecolor=np.array(colormap)[(np.array(regimes)[i][1]) - 1], alpha=0.15)

        # add legend and title
        ax.legend(bbox_to_anchor=(0.79, 1))
        TITLE = str('auto_regimes')
        ppl.title(TITLE, loc='left')

        # save plot
        fig.set_size_inches(16, 9)
        if not os.path.isdir('__OUTPUT'):
            os.makedirs('__OUTPUT')
        path = '__OUTPUT/auto_regimes_{:s}_{:s}_bw{:s}.png'.format(startdate,enddate,str(bandwidth))
        plt.savefig(path, dpi=200)

        plt.close()


    #### BANDWIDTH CHECK ####
    """
    For each regime, check whether the ratio between correct windows (i.e. windows belonging to the cluster predicted by the auto_regime)
    and false windows. Find optimal bandwidth to optimise the ratios in all regimes, i.e. find highest mean score.
    """
    scores = []
    for i in range(len(final_regimes)-1):
        reg_start = int(list(cl_vector[:][1]).index(str(final_regimes2[i][0])))
        reg_end = int(list(cl_vector[:][1]).index(str(final_regimes2[i+1][0])))
        cluster_regime = cl_vector['cluster'][reg_start:reg_end]
        y = np.array((cluster_regime)+1)

        correct = np.count_nonzero(y == int(final_regimes2[i][1]))  # number of windows assigned to predicted cluster (correct)
        n_corr = np.count_nonzero(y != int(final_regimes2[i][1]))   # number of remaining windows assigned to other clusters (false)
        pre_score = correct/(correct+n_corr)
        print(pre_score)
        scores.append(pre_score)

    mean_score = mean(scores)
    accuracies.append(mean_score)

bws = []
for i in range(len(Band_list)):
    bw = round(Band_list[i],2)
    bws.append(bw)
accuracies = pd.DataFrame(accuracies, index=bws)
accuracies.to_csv('/Users/bst/Documents/All/PhD/Data/Codes/regime_detection/kde/regimes/regimes_accuracy.csv')