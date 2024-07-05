import numpy as np
from obspy import read
from obspy import UTCDateTime
import matplotlib.pyplot as plt
import os, sys
from functools import partial
import pandas as pd
from glob import glob
from multiprocessing import Pool
from obspy.signal.filter import bandpass, highpass
import pickle
from obspy.signal.trigger import classic_sta_lta as stl
from obspy.imaging.spectrogram import spectrogram as spec
from scipy.signal import detrend
from scipy.signal import argrelextrema
import math
from obspy.signal.freqattributes import central_frequency_unwindowed as cfu
import datetime
import obspy
from obspy.core.trace import Trace
from scipy.signal import welch
from scipy.signal import spectrogram as scispec
from matplotlib.pyplot import specgram as pltspec
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
from matplotlib import ticker
from astropy.timeseries import LombScargle
from mpl_toolkits.mplot3d import Axes3D
import statsmodels.api as sm


################################
### EXTRACT SPECS FOR PULSES ###
################################

extract_RSAM = False
extract_events = False

volcano = 'Whakaari'
station = 'WIZ'
interval = 10 #RSAM interval in seconds
fband = [[1, 6]]
frequency = 100


###################
### PERIODICITY ###
###################

periodicity = False
mode = 'LS' # 'ACF' for autocorrelation, 'LS' for Lomb-Scargle

freq_plot = np.arange(0.002, 0.015, 0.001) # this determined which periodicities you will be looking at in the end - can go full range, but takes a lot longer (change '.power(freq)' to 'autopower()' 
#--- might need some try and error, or an automatic approach to auto-tune plot with best setting (e.g. running multiple scans, then returns the one with highest periodicity)
#--- default: (0.002, 0.015, 0.001)

LF = 1 # default: 1Hz
HF = 10 # default: 10Hz
COMP = '1S' # default: 1S
data_window = 1800 # number of compressed windows to analyse (e.g., 3600 for a one hour data window if COMP = '1S')
non_overlap = 600 # if overlapping windows, determine the number of data points NOT overlapping (e.g. 600 for 10-min rolling window if COMP = '1S')

target_folder = '/Users/bste426/Desktop/run/*' # chose folder with data (miniseed or pkl)


def normalised_spectrogram():

    mode = 'single' # 'bulk' for multiple days, 'single' for a single day file

    if mode == 'bulk':
        folder = sorted(glob('/Users/bste426/Documents/All/PhD/Data/Codes/TremorRegimes/SeismicData/Whakaari/WIZ/*'))
        matrices = []
        n = len(folder)

        for i in range(n):
            cf = (i+1)/n
            print(f'processing files: [{"#"*round(50*cf)+"-"*round(50*(1-cf))}] {100.*cf:.2f}%\r', end='')
            # Load seismic data
            fp = open(folder[i], 'rb')
            data = pickle.load(fp)
            if i == 0: ST = pd.to_datetime(data.meta.starttime.timestamp, unit = 's').round('D')
            if i == n-1: ET = pd.to_datetime(data.meta.endtime.timestamp, unit = 's').round('D')
            # Filter data
            filtered_data = bandpass(data.data, 1, 10, 100, corners = 12) #[3600*100*17:3600*100*17+360000]

            # Calculate spectrogram
            #spec(testdata, 100, per_lap = 0.99, wlen = 20, cmap = 'jet') # ObsPy
            spectrum, _, _, _ = pltspec(filtered_data, NFFT = 16384*16, noverlap = 8192*16, Fs = 100, cmap = 'jet')
            plt.close()
            
            #'''
            # Normalise spectrogram - columnwise (norm time steps over frequency)
            time_vectors = spectrum.T # now we got the matrix as columns per time
            normed_time_vectors = [(x-np.min(x))/np.max(x) for x in time_vectors] # normalises each time column
            re_vectorise = np.array(normed_time_vectors).T
            matrices.append(normed_time_vectors)            
            #'''
            #matrices.append(spectrum)

        #all_spectra = np.concatenate(matrices, axis = 1) 
        all_spectra = np.concatenate(matrices, axis = 0)  #change to axis = 0 for normalising columns-wise
        all_spectra = all_spectra.T
        print('Normalising...')
        #normed = [(y-np.min(y))/np.max(y) for y in all_spectra] #normalises row-wise       
        print('Plotting...')

        x_lims = list(map(datetime.datetime.fromtimestamp, [ST.timestamp(), ET.timestamp()]))
        x_lims = mdates.date2num(x_lims)

        fig, ax = plt.subplots()
        ax.imshow(all_spectra, cmap='jet', origin = 'lower', extent = [x_lims[0],x_lims[1],0,50], vmax = 0.1, aspect='auto')
        # for non-normed specgra: vmax = 0.0000000000005
        ax.xaxis_date()
        date_format = mdates.DateFormatter('%d-%m-%Y %H:%M')
        ax.xaxis.set_major_formatter(date_format)
        fig.autofmt_xdate()
        ax.set_ylim(0,15)
        plt.savefig('/Users/bste426/Desktop/columnswise.png', format="png", dpi=1200)

    if mode == 'single':
        # Load seismic data
        fp = open('/Users/bst/Documents/All/PhD/Data/Codes/TremorRegimes/SeismicData/Whakaari/CTZ/2022-01-15.pkl', 'rb')
        data = pickle.load(fp)

        # Filter data
        filtered_data = bandpass(data.data, 0.1, 10, 100) [3*60*60*100:15*60*60*100] # select specific part of data: [3600*100*17:3600*100*17+360000]
        # For spectrogram
        #raw_data = data.data[1000:13000]
        # Calculate spectrogram
        #spec(testdata, 100, per_lap = 0.99, wlen = 20, cmap = 'jet') # ObsPy
        spectrum, _, _, _ = pltspec(filtered_data, NFFT = 2048, noverlap = 1024, Fs = 100, cmap = 'jet')
        plt.close()

        # Normalise spectrogram
        time_vectors = spectrum.T # now we got the matrix as columns per time
        normed_time_vectors = [(x-np.min(x))/np.max(x) for x in time_vectors] # normalises each time column
        re_vectorise = np.array(normed_time_vectors).T

        # prep time vector
        t_len = len(filtered_data)/100
        time = np.arange(0, t_len, 0.01)

        # Prepare plot
        _, (ax1, ax2) = plt.subplots(nrows=2, sharex = True)
        ax1.plot(time, filtered_data, c = 'k', linewidth = 0.2)
        ax1.set_ylabel('Amplitude [m/s]')
        ax1.set_xlim(0,t_len)
        plt.imshow(spectrum, cmap='gist_heat', origin = 'lower', extent=[0,t_len,0,50], vmax = 2e-12, aspect='auto')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Frequency (Hz)')
        ax2.set_ylim(1,5)
        #plt.show()
        #plt.pause(1000)

        '''
        # T waves
                fp = open('/Users/bst/Documents/All/PhD/Data/Codes/TremorRegimes/SeismicData/Whakaari/URZ/2022-01-15.pkl', 'rb')
        data = pickle.load(fp)
        filtered_data = bandpass(data.data, 0.1, 10, 100) [3*60*60*100:15*60*60*100] # select specific part of data: 
        spectrum, _, _, _ = pltspec(filtered_data, NFFT = 2048, noverlap = 1024, Fs = 100, cmap = 'jet')
        plt.close()
        time = np.arange(0, 24, 1/(60*6000)) [3*60*60*100:15*60*60*100]
        _, (ax1, ax2) = plt.subplots(nrows=2)
        ax1.set_title('URZ')
        ax1.plot(time, filtered_data, c = 'k', linewidth = 0.2)
        ax1.set_ylabel('Amplitude [m/s]')
        ax1.set_xlim(3,15)
        plt.imshow(spectrum, cmap='gist_heat', origin = 'lower', extent=[0,t_len,0,50], vmax = 5e-15, aspect='auto')
        ax2.set_xlabel('UTC')
        ax2.set_xticklabels([])
        ax2.set_ylabel('Frequency (Hz)')
        '''

def plot_acoustics():
    folder = sorted(glob('/Users/bst/Documents/All/PhD/Publications/paper 2 (whakaari pulses)/submission/writing stage/figures/Figure 3/periodic waveforms/for supfig 4/*'))
    for k in range(len(folder)):
        vars = exec(f'ax{k} = []')
    fig, vars = plt.subplots(len(folder), dpi = 200)
    for i in range(len(folder)):
        fp = open(folder[i], 'rb')
        data = pickle.load(fp)
        """
        ST = pd.to_datetime(data.meta.starttime.timestamp, unit = 's').round('ms') #start time
        ET = pd.to_datetime(data.meta.endtime.timestamp, unit = 's').round('ms')
        time = pd.date_range(ST.round('D'), ET.round('D'), periods = 100*60*60*24+1)
        time = [x.to_pydatetime() for x in time[:300000]]    
        """    
        datfilt = bandpass(data.data, 1, 10, 100)[:2000000]
        vars[i].plot(datfilt, c = 'k')
        vars[i].set_ylabel(folder[i][-14:-4], fontsize = 10)
        vars[i].set_ylim(-0.51e-5, 0.51e-5)
        #vars[i].xaxis.set_major_formatter(mdates.DateFormatter("%M"))
        #vars[i].set_xticklabels([str(x) for x in np.arange(1,11,1)])
    #vars[2].set_xlabel('time [min]')
    plt.show()
    plt.pause(1000)

def plot_spectrogram():
    mode = 'multi'
    norm = False

    if mode == 'multi':
        folder = glob('/Volumes/TheBigOne/Seismic data/Seismic Data/Whakaari/WIZ 2012/*')
        folder.sort()
        folder = folder[:10]
        all_data = []
        for file in folder:
            print(file)
            data = open(file, 'rb')
            seis = pickle.load(data)
            seis = bandpass(seis, 0.1, 20, 100, corners = 16)
            if len(seis.data) > 4320000: all_data.append(seis.data[:4320000])
            else: all_data.append(np.pad(seis.data, (0, 4320000-len(seis.data)), 'constant'))
            
        data = np.concatenate(all_data)
        spec(data, 50, per_lap = 0, mult = None, wlen = 2400, dbscale = False, show = True)
        plt.pause(1000)
        plt.savefig('spec.pdf', transparent = True)

def plot_event_PSD():

    # Load list of events from compile_events
    events = pd.read_csv('/Users/bste426/Documents/All/PhD/Data/Codes/Whakaari_pulses/events/event_stats_order_20.csv')
    event_starts = [datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S') for x in events['start_time']]

    # Load seismic data
    fp = open('/Users/bste426/Documents/All/PhD/Publications/paper 2 (whakaari pulses)/submission/1 writing stage/figures/Figure 3/periodic waveforms/for supfig 4/2022-03-22.pkl', 'rb')
    data = pickle.load(fp)

    # Prepare time vector
    ST = pd.to_datetime(data.meta.starttime.timestamp, unit = 's').round('ms') #start time
    ET = pd.to_datetime(data.meta.endtime.timestamp, unit = 's').round('ms')
    time = pd.date_range(ST.round('D'), ET.round('D'), periods = 100*60*60*24+1)
    time = [x.to_pydatetime() for x in time]

    # Cut seismic data to length
    all_raw = []
    if len(data)>100*60*60*24: [all_raw.append(x) for x in data[:100*60*60*24+1]]
    else: [all_raw.append(x) for x in data]
    # Bandpassfilter if needed
    #all_raw = bandpass(all_raw, 2, 5, frequency, corners = 4) # possible frequency bandpass filter

    # Slice continuous seismic data into time windows based on events (pulses) detected on that day and saved in 'events'-file
    events_on_day = list(set(event_starts).intersection(time))
    events_on_day.sort()

    # extract waveform and spectrum per event, process and save waveform for correlation later
    for i in range(3):
        
        #load event
        index_in    = time.index(events_on_day[i])
        index_out   = time.index(events_on_day[i+1])
        event       = all_raw[index_in:index_out] # raw waveform of one event
        event       = (event-np.mean(event))
        event       = event/(max(np.abs(event))) #demean and normalise

        #plot waveform and PSD
        f           = np.fft.fft(event)
        freq        = np.fft.fftfreq(len(event), 1/100)
        freq        = freq[1:int(len(freq)/2)-1]

        power_spectrum  = np.abs(f)**2
        power_spectrum  = power_spectrum[1:int(len(power_spectrum)/2)-1]
        power_norm      = np.array([(x-np.min(power_spectrum))/(np.max(power_spectrum-np.min(power_spectrum))) for x in power_spectrum])

        fig, axs = plt.subplots(2)
        axs[0].plot(event, c = 'k', linewidth = 0.5)
        axs[1].plot(freq, power_norm, alpha = 1, c = 'k', linewidth = 1)
        axs[1] = plt.xscale('log')
        axs[1] = plt.xlim(0.1, 10)
        axs[1] = plt.grid(visible = True, which = 'both', axis = 'both')

def load_dataframe(fl, index_col=None, parse_dates=False, usecols=None, infer_datetime_format=False, 
    nrows=None, header='infer', skiprows=None):
    if fl.endswith('.pkl'):
        fp = open(fl, 'rb')
        df = pickle.load(fp)
    else:
        raise ValueError('only csv and pkl file formats supported')

    if fl.endswith('.pkl') or fl.endswith('.hdf'):
        if usecols is not None:
            if len(usecols) == 1 and usecols[0] == df.index.name:
                df = df[df.columns[0]]
            else:
                df = df[usecols]
        if nrows is not None:
            if skiprows is None: skiprows = range(1,1)
            skiprows = list(skiprows)
            inds = sorted(set(range(len(skiprows)+nrows)) - set(skiprows))
            df = df.iloc[inds]
        elif skiprows is not None:
            df = df.iloc[skiprows:]
    return df

def get_rsam(interval, fband, fl):

    data = pd.read_pickle(fl) #read data from file
    ti = data.meta['starttime']
    Ns = interval
    tiday = UTCDateTime("{:d}-{:02d}-{:02d} 00:00:00".format(ti.year, ti.month, ti.day))
    ti = tiday+int(np.round((ti-tiday)/Ns))*Ns
    N = Ns*frequency  # number of samples
    Nm = int(N*np.floor(len(data)/N))

    # filter raw data into frequency band of given interval (RSAM)
    filtered_data = bandpass(data, fband[0][0], fband[0][1], frequency, corners = 4)
    filtered_data = abs(filtered_data[:Nm])
    filtered_data = filtered_data.reshape(-1,N).mean(axis=-1)*1.e9

    # write out temporary file
    time = [(ti+j*Ns).datetime for j in range(filtered_data.shape[0])]
    df = pd.Series(filtered_data, index=pd.Series(time))

    return df

def compile_rsam(interval, fband, ncpus=6):
    ''' process the downloaded seismic data for rsam at particular interval/freq bands

        Parameters:
        -----------
        interval : int
            Minute length of rsam averaging interval.
        fband : list
            Limits for bandpass. (2dp sensitivity)
        src : str
            Path to folder containing source files (downloaded from Geonet).
        recompile : bool
            Flag to delete file and start again.
    '''

    if not os.path.isdir('RSAM'):
        os.makedirs('RSAM')
    
    # parallel data collection - creates temporary files
    fls = glob('../TremorRegimes/SeismicData/Whakaari/WIZ/*')
    n = len(fls)

    # counts, how many days-data are in folder
    f = partial(get_rsam, interval, fband)     # hard code initial time and station arguments
    p = Pool(ncpus)
    outs = [None]*n
    for i, out in enumerate(p.imap(f, fls)):
        cf = (i+1)/n
        print(f'processing rsam: [{"#"*round(50*cf)+"-"*round(50*(1-cf))}] {100.*cf:.2f}%\r', end='')
        outs[i] = out
    outs = [out for out in outs if out is not None]

    p.close()
    p.join()
    out = pd.concat(outs)
    out.sort_index(inplace=True)
    out = out.loc[~out.index.duplicated(keep='last')]                       # remove duplicates
    out = out.resample('{:d}S'.format(interval)).interpolate('linear')      # fills in gaps with linear interpolation

    return out

def extract_waveforms(interval, fband, fl):
    print(fl)
    data = pd.read_pickle(fl) #read data from file
    # data = data.traces[0]

    ST = pd.to_datetime(data.meta.starttime.timestamp, unit = 's').round('ms') #start time
    ET = pd.to_datetime(data.meta.endtime.timestamp, unit = 's').round('ms')
    time = pd.date_range(ST.round('D'), ET.round('D'), periods = 100*60*60*24+1)

    all_raw = []
    if len(data)>100*60*60*24: [all_raw.append(x) for x in data[:100*60*60*24+1]]
    else: [all_raw.append(x) for x in data]
    mean = np.mean(all_raw)
    if len(all_raw)<8640001: all_raw += mean * (8640001 - len(all_raw)) # if file is shorter than a day, rest is filled up with mean value of that day
    all_raw = bandpass(all_raw, fband[0][0], fband[0][1], frequency, corners = 4)

    # Event detection (based on modified STALTA)
    abs_data = pd.Series(np.abs(all_raw), index=time) #take absolute values for stalta
    resampled_data = abs_data.resample('1S').sum() #compression to 1 second to make it faster
    stalta = stl(resampled_data,5,20)   #STA: 5 sec, LTA: 20 Sec (measured in re-sampled frequency)
    df = pd.DataFrame(stalta, columns=['data'])
    df.index = resampled_data.index
    n = 20  # determines how far left and right the next local minima at least has to be located
    plt.plot(stalta[:100])
    
    # Extraction of event data
    event_data = []
    event_starts = [x*100 for x in argrelextrema(df.values, np.less_equal,order=n)[0][19:]]  #indices of lowpoints in 100Hz-data (bandpassfiltered)
    for i in range(len(event_starts)-1):
        event = all_raw[event_starts[i]:event_starts[i+1]]

        f           = np.fft.fft(event)
        freq        = np.fft.fftfreq(len(event), 1/100)
        freq        = freq[1:int(len(freq)/2)-1]

        power_spectrum  = np.abs(f)**2
        power_spectrum  = power_spectrum[1:int(len(power_spectrum)/2)-1]

        df = pd.DataFrame({'vals': power_spectrum}, index = freq)
        peaks = argrelextrema(df.values, np.greater_equal)
        top5  = pd.DataFrame([df['vals'].iloc[x] for x in peaks[0]]).sort_values(by = [0])[-5:]

        event_data.append(
            {
                'start_time': time[event_starts[i]],
                '1st': df.index[df['vals'] == top5.iloc[0][0]].tolist()[0],
                '2nd': df.index[df['vals'] == top5.iloc[1][0]].tolist()[0],
                '3rd': df.index[df['vals'] == top5.iloc[2][0]].tolist()[0],
                '4th': df.index[df['vals'] == top5.iloc[3][0]].tolist()[0],
                '5th': df.index[df['vals'] == top5.iloc[4][0]].tolist()[0]
                #'duration': int(len(event)/100),
                #'max_amp': np.max(event),
                #'central_freq': cfu(event,100)
            }
        )

    out = {
    k: [d.get(k) for d in event_data]
    for k in set().union(*event_data)
    }

    return pd.DataFrame(out)#['start_time'], out['duration'], out['max_amp'], out['central_freq']

def compile_events(interval, fband, ncpus=6):

    if not os.path.isdir('events'):
        os.makedirs('events')
    
    # parallel data collection - creates temporary files
    fls = glob('../TremorRegimes/SeismicData/Whakaari/WIZ/*')
    n = len(fls)

    # counts, how many days-data are in folder
    f = partial(extract_waveforms, interval, fband)
    p = Pool(ncpus)
    outs = [None]*n
    for i, out in enumerate(p.imap(f, fls)):
        cf = (i+1)/n
        print(f'processing events: [{"#"*round(50*cf)+"-"*round(50*(1-cf))}] {100.*cf:.2f}%\r', end='')
        outs[i] = out
    outs = [out for out in outs if out is not None]

    p.close()
    p.join()
    event_stats = [pd.concat(outs)][0]
    event_stats = event_stats.reset_index().set_index('start_time')
    event_stats.sort_index(inplace=True)
    event_stats = event_stats.drop(['index'], axis=1)

    return event_stats

def plot_all():
    '''Plot key characteristics of pulses. Stats file from compile_events.'''

    data = pd.read_csv('/Users/bste426/Documents/All/PhD/Data/Codes/Whakaari_pulses/events/event_stats_order_10 minima.csv')
    data['start_time'] = [datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S') for x in data['start_time']]
    rsam = pd.read_csv('/Users/bste426/Documents/All/PhD/Data/Codes/Whakaari_pulses/RSAM/WIZ_RSAM_[[1, 5]].csv')
    rsam.columns = ['time', 'vals']
    rsam['time'] = [datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S') for x in rsam['time']]
    rsam.index = rsam['time']
    rsam.drop(rsam.columns[0], axis = 1)
    #res = pd.Series(rsam['vals'], index=rsam['time'])
    #rsam_res = res.resample('10T').sum()

    _, (ax1, ax2, ax3) = plt.subplots(3, sharex = True, figsize=(16,6), dpi = 75)
    
    #ax1.set_title('Temporal evolution of extracted pulses', fontsize = 14)
    ax1.scatter(data['start_time'], data['duration'], s = 1, c = 'k', alpha = 0.05)
    ax1.set_ylabel('Duration\n[s]', fontsize = 17, labelpad = 26)
    ax1.set_ylim(10,50)
    ax2.scatter(data['start_time'], data['max_amp'], s = 1, c = 'powderblue', alpha = 0.1)
    ax2.set_ylabel('Max. Ampl.\n[V/s]', fontsize = 17, labelpad = 10)
    ax2.set_ylim(1e-6,1e-5)
    ax3.scatter(data['start_time'], data['central_freq'], s = 1, c = 'salmon', alpha = 0.01)
    ax3.set_ylabel('Cent. freq.\n[Hz]', fontsize = 17, labelpad = 10)
    ax3.set_ylim(2.5,3.5)
    ax3.set_xlabel('Time', size = 17)
    #ax4.scatter(data['start_time'], data['skewness'], s = 1, c = 'brown', alpha = 0.01)
    #ax4.set_ylabel('Skewness', fontsize = 12, labelpad = 20)
    #ax4.set_ylim(-0.25,0.25)
    #ax5.scatter(rsam_res.index, rsam_res.values, s = 1, c = 'k', alpha = 0.25)
    #ax5.set_ylabel('RSAM\n[points]', fontsize = 12, labelpad = 12)
    #ax5.set_ylim(0,2000)
    plt.xlim(pd.to_datetime('2012-08-01 00:00:00'),pd.to_datetime('2013-02-01 00:00:00'))
    for i in (ax1, ax2, ax3): 
        i.axvline(pd.to_datetime('2012-08-04 00:00:00'), c = 'r', label = 'Aug 2012 eruption', linewidth = 3)
        i.axvline(pd.to_datetime('2012-09-02 00:00:00'), c = 'k', label = 'ash emission', linestyle = 'dashed', linewidth = 3)
        i.axvline(pd.to_datetime('2012-11-24 00:00:00'), c = 'red', label = 'dome observed', linewidth = 3.5, linestyle = 'dotted')
        #i.axvline(pd.to_datetime('2012-12-05 00:00:00'), c = 'blue', label = 'lake drying phase', linewidth = 2, linestyle = 'dotted')
        i.axvline(pd.to_datetime('2013-01-14 00:00:00'), c = 'blue', label = 'onset surface activity', linewidth = 3)
        i.axvline(pd.to_datetime('2012-09-11 12:00:00'), c = 'w', label = 'missing data', linewidth = 1)
        i.axvline(pd.to_datetime('2012-12-03 12:00:00'), c = 'w', label = '', linewidth = 1)
        #i.axvline(pd.to_datetime('2013-08-19 00:00:00'), c = 'r', label = '', linewidth = 2)
        #i.axvline(pd.to_datetime('2013-10-03 00:00:00'), c = 'r', label = '', linewidth = 2)
        #i.axvline(pd.to_datetime('2013-10-09 00:00:00'), c = 'r', label = '', linewidth = 2)
    
    #_, labels = ax1.get_legend_handles_labels()
    #ax1.legend(ncol=len(labels), bbox_to_anchor=(0, 1),loc='lower left', fontsize='small')
    plt.tight_layout()
    #plt.pause(1000)
    #plt.savefig('/Users/bste426/Desktop/columnswise.png', format="png", dpi=200)
    ax1.yaxis.set_major_locator(ticker.MaxNLocator(4))
    ax2.yaxis.set_major_locator(ticker.MaxNLocator(4))
    ax3.yaxis.set_major_locator(ticker.MaxNLocator(4))
    ax1.tick_params(axis='both', which='major', labelsize=15)
    ax2.tick_params(axis='both', which='major', labelsize=15)
    ax3.tick_params(axis='both', which='major', labelsize=15)
    plt.savefig('/Users/bste426/Documents/All/PhD/Publications/paper 2 (whakaari pulses)/submission/2 review stage/figures/Figure 1/essentials/event_stats.png', format="png", dpi=200)

def plot_monitoringTS():

    #RSAM
    rsam            = pd.read_csv('/Users/bst/Documents/All/PhD/Data/Codes/Whakaari_pulses/rsam.csv')
    rsam.columns    = ['time', 'vals']
    rsam['time']    = [datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S') for x in rsam['time']]
    rsam.index      = rsam['time']
    rsam            = rsam.drop(rsam.columns[0], axis = 1)

    #DSAR
    dsar            = pd.read_csv('/Users/bst/Documents/All/PhD/Data/Codes/Whakaari_pulses/dsar.csv')
    dsar.index      = rsam.index
    dsar            = dsar.drop(dsar.columns[0], axis = 1)

    #SSAM

    #TRANSIENTS
    events = ['VT','HF','LP','VLP']
    valuelists = ['VTval','HFval','LPval','VLPval']
    arraylists = ['VTarr','HFarr','LParr','VLParr']
    dct = {}
    dctval = {}
    dctarr = {}
    ALL = []

    for i in range(len(events)):
        filepath = ('/Users/bst/Documents/All/PhD/Data/Codes/SOM_Carniel/csv2som/act_log/{:s}s.txt'.format(events[i]))
        file = pd.read_csv(filepath)
        dct['%s' % events[i]] = file.values.tolist()

    for x in range(len(valuelists)):
        dctval['%s' % valuelists[x]] = []
        for j in range(len(dct["%s" % events[x]])):
            dctval['%s' % valuelists[x]].append(str(dct["%s" % events[x]][j])[2:-15])

    for y in range(len(arraylists)):
        dctarr['%s' % arraylists[y]] = pd.DataFrame(dctval['%s' % valuelists[y]])
        dctarr['%s' % arraylists[y]][1] = y
        ALL.append(dctarr['%s' % arraylists[y]])

    ALL = np.vstack(sorted((np.concatenate(ALL,axis=0)), key=lambda x: x[0], reverse=False))

    dates = []
    eventtype = []

    for z in range(np.shape(ALL)[0]):
        dates.append((datetime.datetime.strptime((str(ALL[z,0])), '%Y-%m-%d')).strftime("%Y-%m-%d %H:%M:%S"))
        eventtype.append(int(ALL[z,1]))
    dates = [datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S') for x in dates]
    eventtype = np.array(eventtype)

    #FEATURE RSV
    rsv2            = pd.read_csv('/Users/bst/Documents/All/PhD/Data/Codes/Whakaari_pulses/ratioseries.csv')
    rsv2.index      = [datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S') for x in rsv2[rsv2.columns[0]]]
    rsv2            = rsv2.drop(rsv2.columns[0], axis = 1)

    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, sharex = True, figsize=(50,7), dpi = 150)
    ax1.set_title('Monitoring parameters during Whakaari 2012/13 unrest phase', fontsize = 14)
    ax1.scatter(rsam.index, rsam.values, s = 1, c = 'k', alpha = 0.1)
    ax1.set_ylabel('RSAM\n[counts]', fontsize = 12, labelpad = 10)
    ax1.set_ylim(0,4000)
    ax2.set_ylabel('SSAM\n[Hz]', fontsize = 12, labelpad = 12)
    ax2.set_ylim(0,10)
    ax3.scatter(dsar.index, dsar.values, s = 1, c = 'lightblue', alpha = 0.1)
    ax3.set_ylabel('DSAR', fontsize = 12, labelpad = 25)
    ax3.set_ylim(0,4)
    colourmap = np.array(['magenta', 'lightgreen', 'purple', 'orange'])
    colours = eventtype % colourmap.shape[0]
    ax4.scatter(dates, eventtype, c = colourmap[colours], s = 15, alpha = 0.15)
    ax4.set_ylabel('Event type', fontsize = 12, labelpad = 12)
    ax4.set_ylim(-0.5,3.5)
    ax4.set_yticks([0,1,2,3],['VT', 'HF', 'LP', 'VLP'])
    ax5.plot(rsv2.index, rsv2, c = 'grey', alpha = 1, linewidth = 1)
    ax5.set_ylabel('RSV', fontsize = 12, labelpad = 25)
    ax5.set_ylim(0,3)
    plt.xlim(dsar.index[0], dsar.index[-1])
    for i in (ax1, ax2, ax3, ax4, ax5): 
        i.axvline(pd.Timestamp('2012-08-04 00:00:00'), c = 'r', label = 'Aug 2012 eruption', linewidth = 2)
        i.axvline(pd.Timestamp('2012-09-02 00:00:00'), c = 'k', label = 'ash emission', linestyle = 'dashed', linewidth = 2)
        i.axvline(pd.Timestamp('2012-11-24 00:00:00'), c = 'red', label = 'dome observed', linewidth = 2, linestyle = 'dotted')
        i.axvline(pd.Timestamp('2012-12-05 00:00:00'), c = 'blue', label = 'lake drying phase', linewidth = 2, linestyle = 'dotted')
        i.axvline(pd.Timestamp('2013-01-14 00:00:00'), c = 'blue', label = 'onset surface activity', linewidth = 2)
        i.axvline(pd.Timestamp('2013-08-19 00:00:00'), c = 'r', label = 'Aug 2013 eruption', linewidth = 2)
        i.axvline(pd.Timestamp('2013-10-03 00:00:00'), c = 'r', label = 'Oct 2013 eruption I', linewidth = 2)
        i.axvline(pd.Timestamp('2013-10-11 00:00:00'), c = 'r', label = 'Oct 2013 eruption II', linewidth = 2)
    #handles, labels = ax1.get_legend_handles_labels()
    #fig.legend(handles = handles, labels = labels, loc = 'upper right')
    plt.tight_layout()

def plot_periodicity(freq_plot, FILEPATH):

    matrix = pd.read_csv(FILEPATH) # load data

    freq = freq_plot # determine which periodicities shall be plotted (lags for ACF)
    index = np.arange(0,len(matrix),1)
    X,Y = np.meshgrid(freq, index)
    Z = matrix[matrix.columns[1:]].values

    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap = plt.cm.get_cmap('turbo'), shade=True, lw=5)

    ax.set_zlim(0,0.25)
    ax.tick_params(labelsize = 14)
    ax.set_yticklabels(ax.get_yticks(), verticalalignment='center', horizontalalignment='right', rotation_mode='anchor', rotation = 30)
    ax.tick_params(axis='y', which='major', pad=-5)

    ax.set_xlabel('Periodicity')
    ax.set_zlabel('Power')

    ax.set_xlabel('Periodicity', size = 15, labelpad = 10)
    ax.set_zlabel('Power', size = 15, labelpad = 10)

    ax.xaxis.pane.set_edgecolor('k')
    ax.yaxis.pane.set_edgecolor('k')
    ax.zaxis.pane.set_edgecolor('k')

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    ax.grid(False)

    #periods = 1/freq
    #ax.set_xticks(freq, [round(x) for x in periods])
    ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    dates = [pd.to_datetime(x).strftime('%m/%d/%Y %H:%M') for x in matrix[matrix.columns[0]]] # change to '%m/%d/%Y %H:%M' if only using a day of data or less   
    ax.set_yticks(index, dates)
    ax.yaxis.set_major_locator(plt.MaxNLocator(4))  # might be nicer with 5, depends on data
    ax.zaxis.set_major_locator(plt.MaxNLocator(4))

    ax.view_init(elev=45, azim=55)
    #plt.tight_layout()
    #plt.show()
    plt.savefig('periodic waveforms/periodicity visualisations/{:s}.png'.format(station), transparent=True)

def check_periodicity(folder, COMP, mode, freq_plot, data_window, non_overlap):

    all_data = []
    time_vector = []

    folder = glob(target_folder)
    folder.sort()

    for j in range(len(folder)):
        
        print('Loading...', folder[j])
        try: 
            data = pd.read_pickle(folder[j])
        except: 
            data = read(folder[j])
            data = data.traces[0]

        len_day = 100*60*60*24+1

        current_day = folder[j][-14:-4] # unfortunately neccessary, because start time might be after 12pm, so can't use metadata startime...
        actual_ST = pd.Timestamp(current_day+str(' 00:00:00.000000'))
        if j == 0: time_vector.append(actual_ST)
        if j == len(folder)-1: time_vector.append(actual_ST+datetime.timedelta(days = 1))

        time = pd.date_range(actual_ST, actual_ST+datetime.timedelta(days = 1), periods = len_day) #generic time vector (needed for compression)

        #Calculate lags towards start and end time of record to fill up gaps:
        ST = pd.to_datetime(data.meta.starttime.timestamp, unit = 's').round('ms') #start time
        lag_start = round(((actual_ST-ST).total_seconds())*100) #returns number of samples to be cut off (positive value) or to be added (negative value)
        ET = pd.to_datetime(data.meta.endtime.timestamp, unit = 's').round('ms')
        lag_end = round(((ET-(actual_ST+datetime.timedelta(days = 1))).total_seconds())*100)

        if len(data)>len_day: 
            all_raw = data[lag_start:lag_start+len_day] # standard case: use all data available, cut off data if overlap in records
            if len(all_raw) < len(time):
                if lag_start > 0 and lag_end < 0:
                    all_raw = data[lag_start:lag_start+len_day]
                    all_raw = np.append(all_raw, [0]*abs(lag_end)) # if file starts before and stops within current day -> need to pad zeros at end, data inside day window not enough
                elif lag_start < 0 and lag_end > 0:
                    all_raw = np.zeros(abs(lag_start)) # if file starts within current day and ends within next day -> need to pad zeros at start, data inside day window not enough
                    all_raw = np.append(all_raw, data[:len_day])
        if len(data)<len_day: 
            if lag_start > 0:
                all_raw = data[lag_start:] # if record started too early, cut off overlap
            if lag_start < 0:
                all_raw = np.zeros(abs(lag_start)) # if record started too late, fill up with zeros
                all_raw = np.append(all_raw, data) # then fill up with data
            if lag_start == 0: 
                all_raw = np.array([])
                all_raw = np.append(all_raw, data) # if record mathes, just fill up with data
            if lag_end > 0:
                all_raw = all_raw[:lag_end] # if record ended too late, cut off overlap
            if lag_end < 0:
                all_raw = np.pad(all_raw, (0,abs(lag_end)), constant_values = 0) # if record ended too early, fill up with zeros
        if len(data) == len_day:
            all_raw = data
        # Data prep complete.
            
        # Bandpass filtering if needed:
        filtered = bandpass(all_raw, LF, HF, 100, corners = 4)[:len_day]

        # NOTE: In a later real-time version, this could be skipped and the STALTA simply not calculated if data is missing. Lomb-Scargle takes care of missing data points (i.e., missing STALTA vals).

        # Calculate STALTA on compressed seismic data
        print('Compressing data...')
        abs_data = pd.Series(np.abs(filtered), index=time) #take absolute values for compression
        resampled_data = abs_data.resample(COMP).sum() #compression to 1 second to make it faster
        all_data.append(resampled_data[:-1])

    data = np.concatenate(all_data)

    data_window = data_window
    steps = non_overlap # non overlapping part 
    n_windows = int((len(data)/steps) - (data_window/steps - 1))
    time_ind = pd.date_range(time_vector[0], time_vector[1], int(len(data)/steps)+1)

    lags = 100
    OUTPUT = []

    print('Calculating autocorrelation...')
    for i in range(n_windows):
        snippet = data[i*steps:i*steps+data_window]
        if mode == 'ACF':
            outs = sm.tsa.acf(snippet, nlags = lags)
        else:
            outs = (LombScargle(np.arange(1,len(snippet)+1,1), snippet).power(freq_plot))
        OUTPUT.append(outs)

    master_matrix = pd.DataFrame(OUTPUT, index = time_ind[:-int(data_window/steps)])
    if mode == 'ACF':
        outliers = master_matrix.loc[master_matrix[master_matrix.columns[10]] > 0.35] # MANUAL: get rid of outliers - each window with AC>0.35 at lag 10 is kicked out (reset to 0)
        master_matrix.loc[outliers.index] = 0
    OUT = 'periodic waveforms/periodicity visualisations/{:s}.csv'.format(station)
    master_matrix.to_csv(OUT)

    print('Preparing 3D plot...')
    if mode == 'ACF': freq_plot = range(lags+1)
    plot_periodicity(freq_plot = freq_plot, FILEPATH = OUT)


if __name__ == '__main__':
    if extract_RSAM is True:
        RSAM = compile_rsam(interval, fband, ncpus=6)
        RSAM.to_csv('RSAM/WIZ_RSAM_{:s}.csv'.format(str(fband)), index=True, header=True)
    if extract_events is True:
        event_stats = compile_events(interval, fband, ncpus=6)
        event_stats.to_csv('events/event_stats.csv', index=True, header=True)
    if periodicity is True:
        #plot_periodicity(freq_plot, OUT)
        check_periodicity(target_folder, COMP, mode, freq_plot, data_window, non_overlap)


    # Other functions:

    #plot_all()              
    normalised_spectrogram()
    #plot_spectrogram()
    #plot_event_PSD()   
    #plot_acoustics() 
    #plot_monitoringTS()
            
    pass  