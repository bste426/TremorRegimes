import os, sys
from functools import partial
from obspy import read
import h5py

sys.path.insert(0, os.path.abspath('..'))
from functions import *
# tsfresh and sklearn dump a lot of warnings - these are switched off below, but should be
# switched back on when debugging
import logging
logger = logging.getLogger("tsfresh")
logger.setLevel(logging.ERROR)
import warnings
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.decomposition import PCA
from sklearn import preprocessing
from scipy.signal import decimate
import itertools

from sklearn.exceptions import FitFailedWarning
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=FitFailedWarning)
warnings.filterwarnings("ignore")

'''
MacOS: PRESS 'command' + K + 'command' + '0' (zero) TO COLLAPSE ALL SECTION IN THIS CODE.
'''

###################################
### A N A L Y S I S   M O D E S ### 
###################################

station = 'WIZ'
'''
Whakaari: WIZ
Ruapehu: FWVZ | MAVZ | WHVZ
Tongariro: OTVZ
Redoubt: REF | RDT | RSO
Augustine: AUH | AUI | AUL | AUW
Vulcano: IVGP
Colima: EZV4(SOMA) | EZV5(EFRE) (not for download)
'''

#Compute features from RSAM:
download = False #not available for Colima
RSAM     = False
filter   = False
features = False
#Compute features from raw data
from_raw = True
#Check long-term tremor oscillations at WIZ
tremor_drops_WIZ = False

file_format = 'csv'         #Choose 'parquet' or 'hdf5' (only for 'from_raw') for reduced file size and computation time. Otherwise 'csv' for quick data check.
PCA_features = False        #!!!For some reason only works when time window lengths are calculated separately.!!!
                            #Performs PCA on resulting feature matrix output as alternative dataset for SOM


#######################
### S E T T I N G S ### 
####### R S A M #######

if from_raw is False:
    #Time period for downloading data
    '''Note that missing dates will be interpolated.'''
    startdate = datetime(2018,8,1,0,0,0)
    enddate = datetime(2018,8,3,0,0,0)

    #Depending on file size, might need to be reduced by decimation factor.
    decimation_factor = None

    ### Choose RSAM interval (in seconds)
    intervals = ([10])#1, 5, 10, 60, 120, 300, 600])

    ### Choose frequency bands
    fbands = [[2,5]]

    #Specify time window length in seconds
    windows = ([3600])#,10800,21600,43200,86400,172800])

    # Here you can choose if time windows should overlap or not. If so, specify overlap rate.
    overlap = False
    if overlap is True:
        overlap_rate = ((3600*48-600)/(3600*48))    # can be expressed in relative (e.g. 0.75 for 75%) or in absolute quantities (e.g. [(3000/3600)] for 1 hour windows every 10 minutes (5/6 overlap))
    else:
        overlap_rate = 0
        
    FE_settings={}



#######################
### S E T T I N G S ### 
######## R A W ########

if from_raw is True:
    '''
    Specify Feature extraction settings below.
    'len_ov' is pairs of [1] length and [2] overlap of windows [in seconds]. I recommend not using overlap in the first stage as datasets become very large.
    'FE' is tuples of [1] tsfresh library to extract features from, [2] 'compute only'-features [None = all from library] and [3] features to be ignored from library.
    Libraries available: EfficientFCParameters(), ComprehensiveFCParameters(), MinimalFCParameters() [https://tsfresh.readthedocs.io/en/latest/text/feature_extraction_settings.html]
    NEW: FirstFCParameters and SecondFCParameters
    '''

    # Raw data are usually 100-200Hz. To reduce file size and memory, you can reduce the data load by a factor of...
    decimation_factor = 2

    # Would you like to filter the raw data? Filter: Butterworth Bandpass Filter
    fbands = [[0.1,25]]

    FE_settings={
    'Stage_I':{
        'len_ov':[10,0],
        'FE':[FirstFCParameters(),['maximum','minimum'], None]
        },
    'Stage_II':{
        'len_ov':[3600,3000],
        'FE':[SecondFCParameters(),['maximum','minimum'], None],
        },
    'Stage_III':{
        'len_ov':[(3600*12),(3600*11)],
        'FE':[None,['maximum','minimum'], None],
        },
    }

skip3rd = True

STATIONS={
    'WIZ':{
        'volcano':'Whakaari',
        'frequency':100,
        'client_name':"GEONET",
        'nrt_name':'https://service-nrt.geonet.org.nz',
        'location':'10',
        'channel':'HHZ',
        'network':'NZ'
        },
    'test':{
        'volcano':'Whakaari',
        'frequency':50,
        'client_name':"GEONET",
        'nrt_name':'https://service-nrt.geonet.org.nz',
        'location':'10',
        'channel':'HHZ',
        'network':'NZ'
        },
    'WIZ2019':{
        'volcano':'Whakaari',
        'frequency':50,
        'client_name':"GEONET",
        'nrt_name':'https://service-nrt.geonet.org.nz',
        'location':'10',
        'channel':'HHZ',
        'network':'NZ'
        },
    'WIZ2016':{
        'volcano':'Whakaari',
        'frequency':50,
        'client_name':"GEONET",
        'nrt_name':'https://service-nrt.geonet.org.nz',
        'location':'10',
        'channel':'HHZ',
        'network':'NZ'
        },
    'REF':{
        'volcano':'Redoubt',
        'frequency':100,
        'client_name':"IRIS",
        'nrt_name':'https://service.iris.edu',
        'location':'',
        'channel':'EHZ',
        'network':'AV'
        },
    'REF2019':{
        'volcano':'Redoubt',
        'frequency':100,
        'client_name':"IRIS",
        'nrt_name':'https://service.iris.edu',
        'location':'',
        'channel':'EHZ',
        'network':'AV'
        },
    'RSO':{
        'volcano':'Redoubt',
        'frequency':100,
        'client_name':"IRIS",
        'nrt_name':'https://service.iris.edu',
        'location':'',
        'channel':'EHZ',
        'network':'AV'
        },
    'RDT':{
        'volcano':'Redoubt',
        'frequency':100,
        'client_name':"IRIS",
        'nrt_name':'https://service.iris.edu',
        'location':'',
        'channel':'EHZ',
        'network':'AV'
        },
    'AUH':{
        'volcano':'Augustine',
        'frequency':100,
        'client_name':"IRIS",
        'nrt_name':'https://service.iris.edu',
        'location':'',
        'channel':'EHZ',
        'network':'AV'
        },
    'AUI':{
        'volcano':'Augustine',
        'frequency':100,
        'client_name':"IRIS",
        'nrt_name':'https://service.iris.edu',
        'location':'',
        'channel':'EHZ',
        'network':'AV'
        },
    'AUL':{
        'volcano':'Augustine',
        'frequency':100,
        'client_name':"IRIS",
        'nrt_name':'https://service.iris.edu',
        'location':'',
        'channel':'EHZ',
        'network':'AV'
        },
    'AUW':{
        'volcano':'Augustine',
        'frequency':100,
        'client_name':"IRIS",
        'nrt_name':'https://service.iris.edu',
        'location':'',
        'channel':'EHZ',
        'network':'AV'
        },
    'EZV4':{
        'volcano':'Colima',
        'frequency':100,
        'client_name':'',
        'nrt_name':'',
        'location':'',
        'channel':'',
        'network':'',
        'location':''
        },
    'EZV5':{
        'volcano':'Colima',
        'frequency':100,
        'client_name':'',
        'nrt_name':'',
        'location':'',
        'channel':'',
        'network':'',
        'location':''
        },
    'IVGP':{
        'volcano':'Vulcano',
        'frequency':100,
        'client_name':'https://webservices.ingv.it',
        'nrt_name':'https://webservices.ingv.it',
        'location':'',
        'channel':'HHZ',
        'network':'IV',
        'location':'*'
        },
    'MAVZ':{
        'volcano':'Ruapehu',
        'frequency':100,
        'client_name':"GEONET",
        'nrt_name':'https://service-nrt.geonet.org.nz',
        'location':'10',
        'channel':'HHZ',
        'network':'NZ'
        },
    'FWVZ':{
        'volcano':'Ruapehu',
        'frequency':100,
        'client_name':"GEONET",
        'nrt_name':'https://service-nrt.geonet.org.nz',
        'location':'10',
        'channel':'HHZ',
        'network':'NZ'
        },
    'WHVZ':{
        'volcano':'Ruapehu',
        'frequency':100,
        'client_name':"GEONET",
        'nrt_name':'https://service-nrt.geonet.org.nz',
        'location':'10',
        'channel':'HHZ',
        'network':'NZ'
        },
    'OTVZ':{
        'volcano':'Tongariro',
        'frequency':100,
        'client_name':"GEONET",
        'nrt_name':'https://service-nrt.geonet.org.nz',
        'location':'10',
        'channel':'HHZ',
        'network':'NZ'
        }
    }

# can be read from obspy file via e.g.: data.traces[0].meta['network']
volcano = STATIONS[station]['volcano']
frequency = STATIONS[station]['frequency']
client_name = STATIONS[station]['client_name']
nrt_name = STATIONS[station]['nrt_name']
location = STATIONS[station]['location']
channel = STATIONS[station]['channel']
network = STATIONS[station]['network']


# This is for feature extraction from RSAM.

def get_gdata_day(t0,i):
    """ Download WIZ data for given 24 hour period, writing data to temporary file.

        Parameters:
        -----------
        i : integer
            Number of days that 24 hour download period is offset from initial date.
        station : str
            Name of station to grab data from.
        t0 : datetime.datetime
            Initial date of data download period.

    """
    if not os.path.isdir('../SeismicData/{:s}/{:s}'.format(volcano,station)):
        os.makedirs('../SeismicData/{:s}/{:s}'.format(volcano,station), exist_ok=True)

    daysec = 24*3600
    t0 = UTCDateTime(t0) + i*daysec
    t1 = t0

    fl = "../SeismicData/{:s}/{:s}/{:d}-{:02d}-{:02d}.pkl".format(volcano, station, t0.year, t0.month, t0.day)
    if os.path.isfile(fl):
        return    

    client = FDSNClient(client_name)
    nrt_client = FDSNClient(nrt_name)
    
    # download data
    try:
        site = client.get_stations(starttime=t0, endtime=t1, station=station, level="response", channel=channel)
        '''
        starttime = UTCDateTime("2005-10-01")
        endtime = UTCDateTime("2005-10-01")
        inventory = client.get_stations(network="AV", station="*", channel='*', starttime = starttime, endtime = endtime)
        print(inventory)
        '''
    except FDSNNoDataException:
        try:
            site = nrt_client.get_stations(starttime=t0, endtime=t1, station=station, level="response", channel=channel)
        except FDSNNoDataException:
            print('No client found')
            pass

    try:
        data = client.get_waveforms(network ,station, location, channel, t0, t0+daysec)

        # if less than 1 day of data, try different client
        if len(data.traces[0].data) < 60*100:
            raise FDSNNoDataException('')
    except ObsPyMSEEDFilesizeTooSmallError:
        print('No data found - too small')
        return
    except FDSNNoDataException:
        try:
            data = client.get_waveforms(network ,station, location, channel, t0+i*daysec, t0+(i+1)*daysec)
        except FDSNNoDataException:
            print('No data found - could be missing on the server (e.g. station outage)')
            return

    data.remove_sensitivity(inventory=site)  # removes station response
    if decimation_factor is not None:
        data.traces[0].decimate(decimation_factor)       # downsample data otherwise its huge (adapt depending on max frequency to be analysed)
    save_dataframe(data.traces[0], fl)

def pull_geonet_data(ncpus=6):
    ''' pulls down all the geonet data for whakaari so you can reprocess different length RSAMs if you like

    '''
    # define range for downloading data
    ti = startdate
    tf = enddate
    ndays = (tf-ti).days+1

    # parallel data collection
    f = partial(get_gdata_day, ti)    
    p = Pool(ncpus)

    try:
        for i, _ in enumerate(p.imap(f, range(ndays))):
            cf = (i + 1) / ndays
            print(f'grabbing server data: [{"#" * round(50 * cf) + "-" * round(50 * (1 - cf))}] {100. * cf:.2f}%\r', end='')
    except ValueError:  # raised if `y` is empty.
        pass

    p.close()
    p.join()

def compile_rsam(interval, fband, src, ncpus=6):
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

    if not os.path.isdir('RSAM/{:s}/{:s}'.format(volcano, station)):
        os.makedirs('RSAM/{:s}/{:s}'.format(volcano, station), exist_ok=True)
    
    # parallel data collection - creates temporary files
    fls = glob('{:s}/*'.format(src))
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

def get_rsam(interval, fband, fl):

    #fl = '/Users/bste426/Documents/All/PhD/Data/SeismicData/Augustine/_tmp_AUW/2006-03-08.pkl'
    # load data
    try:
        tr = read(fl)
    except:
        try:
            tr = load_dataframe(fl) #for pkl or mseed files
        except:
            print('empty file --- ',fl)
            return
    if tr is None:
        return None
    
    # make sure obspy file is in the right format
    try:
        data = tr.traces[0]
    except:
        data = tr

    # extract start date from filename (rather than meta-data) because sometimes file starts a few seconds before midnight. ALSO: If no data is available at the start of a day, the remaining data of that day will be moved forward to start at 12.00am
    try:
        ti = UTCDateTime(str(fl)[-14:-4])
    except:
        try:
            ti = data.meta['starttime']
            #print('Warning: Could not read starttime from filename - reading from metadata instead. This may cause false reading of starttime as records begin before midnight.')
        except:
            print('Starttime of seismic records for given file could not be determined - chose different data format or check that naming convention corresponds to `YYYY-MM-DD.pkl`.')

    # prepare startdate, no of samples, etc...
    Ns = interval
    tiday = UTCDateTime("{:d}-{:02d}-{:02d} 00:00:00".format(ti.year, ti.month, ti.day))
    ti = tiday+int(np.round((ti-tiday)/Ns))*Ns
    N = Ns*frequency  # number of samples
    Nm = int(N*np.floor(len(data)/N))

    # earthquake filter based on STA/LTA: https://github.com/jemromerol/apasvo
    #list,func = stalta.sta_lta(data.data, frequency, method = 'iterative')
    #plt.plot(func)#print(list, func, (int(list)/360000))
    #plt.pause(1000)

    # filter raw data into frequency band of given interval (RSAM)
    filtered_data = bandpass(data, *fband, frequency)
    filtered_data = abs(filtered_data[:Nm])
    #median_var = np.min(filtered_data)
    #filtered_data = np.array([x if x < 5e-6 else 0 for x in filtered_data])
    filtered_data = filtered_data.reshape(-1,N).mean(axis=-1)*1.e9
    #plt.plot(filtered_data)

    #filtered_data_mean = filtered_data.reshape(-1,N).mean(axis=-1)*1.e9
    #filtered_data_median = np.median(filtered_data.reshape(-1,N)*1.e9,axis=-1)
    #plt.plot(filtered_data_mean)
    #plt.plot(filtered_data_median)

    # write out temporary file
    time = [(ti+j*Ns).datetime for j in range(filtered_data.shape[0])]
    df = pd.Series(filtered_data, index=pd.Series(time))

    return df

def run_PCA(data, file, window):
    '''Performs a PCA on the calculated feature matrix.'''
    print('Performing PCA......')
    pca = PCA(n_components=10)
    scaler = preprocessing.StandardScaler().fit(data)
    data_sta = scaler.transform(data)
    YM = pca.fit_transform(data_sta)
    YM = pd.DataFrame(YM).set_index(data.index)
    if file_format == 'csv':
        YM.to_csv('features/{:s}/{:s}/{:s}_{:s}_features_PCA.csv'.format(volcano, station, file[len('RSAM/{:s}/{:s}/rsam_'.format(volcano, station)):-len(file_format)-1], str(window)), index=True)
    if file_format == 'parquet':
        table = pa.Table.from_pandas(YM)
        pq.write_table(table, 'features/{:s}/{:s}/{:s}_{:s}_features_PCA.parquet'.format(volcano, station, file[len('RSAM/{:s}/{:s}/rsam_'.format(volcano, station)):-len(file_format)-1], str(window)))

def dump_features(PCA_features):
    ''' output feature matrix for data
    '''
    fl = glob('RSAM/{:s}/{:s}/*.{:s}'.format(volcano, station, file_format))

    if not os.path.isdir('features/{:s}/{:s}'.format(volcano, station)):
        os.makedirs('features/{:s}/{:s}'.format(volcano, station), exist_ok=True)

    for file in fl:
        if file_format == 'csv':
            df = pd.read_csv(file)
            skip_columns = 1
        else:
            df = pd.read_parquet(file)
            skip_columns = 0
        freq_bands = df.columns[skip_columns:]
        datastream = file[len('RSAM/{:s}/{:s}/'.format(volcano, station)):-len(file_format)-1]
        
        print(datastream)

        #read RSAM interval from file (to resample gaps)
        if "_600" in file:
            interpolation_window = '10T'
        if "_60" in file:
            interpolation_window = '1T'
        if "_10" in file:
            interpolation_window = '10S'
        if "_0.05" in file:
            interpolation_window = '0.05S'

        for window in windows:
            print(window)
            fm = ForecastModel(volcano=volcano, station=station, frequency = interpolation_window, window=window, overlap=overlap_rate, look_forward=0, data_streams=datastream, freq_bands = freq_bands, file_format = file_format)
            fm.n_jobs = 6
            FM = fm._extract_features(fm.ti_model, fm.tf_model, freq_bands)

            # writes one file per RSAM file per desired time window length
            if file_format == 'csv':
                FM.to_csv('features/{:s}/{:s}/{:s}_{:s}_features.csv'.format(volcano, station, file[len('RSAM/{:s}/{:s}/rsam_'.format(volcano, station)):-len(file_format)-1], str(window)), index=True)
            if file_format == 'parquet':
                table = pa.Table.from_pandas(FM)
                pq.write_table(table, 'features/{:s}/{:s}/{:s}_{:s}_features.parquet'.format(volcano, station, file[len('RSAM/{:s}/{:s}/rsam_'.format(volcano, station)):-len(file_format)-1], str(window)))
    
            if PCA_features == True:
                run_PCA(FM, file, window)
        
    return
        
def filter_RSAM(volcano = volcano, station = station, file_format = file_format):
    fl = glob('RSAM/{:s}/{:s}/*.{:s}'.format(volcano, station, file_format))
    for file in fl:
        if file_format == 'csv':
            data = pd.read_csv(file)
        elif file_format == 'parquet':
            data = pd.read_parquet(file)
    
    #find outliers (cutoff threshold can be set manually)
    outliers_up = sorted(set(itertools.chain.from_iterable([list([x-timedelta(seconds = 10), x, x+timedelta(seconds = 10)]) for x in data.index[data['[0.1, 1]'] > 3e13]])))
    outliers_down = sorted(set(itertools.chain.from_iterable([list([x-timedelta(seconds = 10), x, x+timedelta(seconds = 10)]) for x in data.index[data['[8, 16]'] < 300000]])))
    #filtered_data = data.drop(outliers)

    #interpolate data gaps 
    filtered_data = filtered_data.resample('10S').interpolate('linear')  
    '''
    #add DSAR:
    dsar = [data['[4.5, 8]'][i]/data['[8, 16]'][i] for i in range(len(data))]
    data['DSAR'] = dsar

    #normalise

    # Plot if needed:
    plt.plot(filtered_data, alpha=0.35)
    plt.plot(data, alpha=0.35)
    plt.legend(labels=filtered_data.columns)
    #plt.ylim(0,3000)
    plt.pause(1000)
    '''
    if file_format == 'csv':
        filtered_data.to_csv(str(fl)[2:-2], index=True, header=True)
    if file_format == 'parquet':
        table = pa.Table.from_pandas(filtered_data)
        pq.write_table(table, str(fl)[2:-2])


# This is for feature extraction from raw seismic data.

def construct_windows(input_time, N_windows, sample_rate, first_datum, window_size, overlap_size):

    dfi_id = []
    dft = []
    window_end_dates = []

    for i in range(N_windows):
        cf = i/N_windows
        print(f'Constructing windows: [{"#"*round(50*cf)+"-"*round(50*(1-cf))}] {100.*cf:.2f}%\r', end='')
        dfi_id.append(np.ones(int((window_size)*sample_rate), dtype=int)*i)
        dft.append((input_time[int(i*((window_size-overlap_size)*sample_rate)):int(i*((window_size-overlap_size)*sample_rate))+int(window_size*sample_rate)]))
        #window_end_dates.append(first_datum + timedelta(seconds = window_size*CF) + timedelta(seconds = i*(window_size-overlap_size)*CF))
        window_end_dates.append(first_datum + timedelta(seconds = window_size) + timedelta(seconds = i*(window_size-overlap_size)))


    dft_list = (itertools.chain.from_iterable(dft)) #times
    dfi_id_list = (itertools.chain.from_iterable(dfi_id)) #id

    new_df_raw = pd.DataFrame(dfi_id_list)
    new_df = new_df_raw.set_index(dft_list)
    new_df.columns = ['id']

    return new_df, window_end_dates

def FE_from_raw(input_data, raw_windows, raw_overlap, SR, Nw, first_window, time, cfp, compute_only_features, drop_features, OUT, file_format, stage):

    #FEATURE SELECTION:
    if compute_only_features:
        cfp = dict([(k, cfp[k]) for k in cfp.keys() if k in compute_only_features])
    if drop_features:
        [cfp.pop(d_f) for d_f in drop_features if d_f in list(cfp.keys())]

    #FEATURE EXTRACTION
    if isinstance(input_data, pd.DataFrame) is False:
        input_data = pd.DataFrame(input_data)
        input_data.columns=['raw']

    single_stream, wd = construct_windows(time, Nw, SR, first_window, raw_windows, raw_overlap)

    for feature in input_data.columns:
        #data = list(input_data['{:s}'.format(feature)])[:(Nw-1)*((raw_windows-raw_overlap)*SR)+raw_windows*SR]
        new_df = copy(single_stream)
        if raw_overlap == 0:
            data = list(input_data['{:s}'.format(feature)])[:int((Nw-1)*((raw_windows-raw_overlap)*SR)+raw_windows*SR)]
            new_df.insert(loc=0, column='{:s}'.format(feature), value=data)
        else:
            overlapping_data = []
            overlapping_data.append([list(input_data['{:s}'.format(feature)])[int(i*((raw_windows-raw_overlap)*SR)):int(i*((raw_windows-raw_overlap)*SR)+raw_windows*SR)] for i in range(Nw)])
            data = list(itertools.chain.from_iterable(list(itertools.chain.from_iterable(overlapping_data)))) 
            new_df.insert(loc=0, column='{:s}'.format(feature), value=data)
        new_df.columns = [sub.replace('__', '_') for sub in new_df.columns]
        feature = feature.replace("__", "_" ) #tsfresh doesn't like '__' ... looks like its face when you still do it.
        matrix = extract_features(new_df, column_id='id', n_jobs=6, default_fc_parameters=cfp, impute_function=impute)
        
        #SAVE FILES
        if stage == 'Stage_I':
            if not os.path.isdir('RSAM/{:s}/{:s}/Stage_I'.format(volcano, station)):
                os.makedirs('RSAM/{:s}/{:s}/Stage_I'.format(volcano, station), exist_ok = True)
            matrix.index = pd.Series(wd)
            matrix.index.name = 'time'

            if file_format == 'hdf5':
                hf = h5py.File('{:s}_standard.h5'.format(OUT), 'w')
                hf.create_dataset('data', data=matrix)
                hf.close()

            if file_format == 'parquet':
                table = pa.Table.from_pandas(matrix)
                pq.write_table(table, '{:s}.parquet'.format(OUT))

            if file_format == 'csv':
                matrix.to_csv('{:s}.csv'.format(OUT), index=True, header=True)

            return matrix
        
        else:
            OUT = 'RSAM/{:s}/{:s}/Stage_II/Stage2_{:s}s_{:s}_{:s}'.format(volcano, station, str(raw_windows), str('NO') if raw_overlap==0 else str(raw_overlap),feature)
            if not os.path.isdir('RSAM/{:s}/{:s}/Stage_II'.format(volcano, station)):
                os.makedirs('RSAM/{:s}/{:s}/Stage_II'.format(volcano, station), exist_ok = True)
            matrix.index = pd.Series(wd)
            matrix.index.name = 'time'

            if file_format == 'hdf5':
                hf = h5py.File('{:s}_standard.h5'.format(OUT), 'w')
                hf.create_dataset('data', data=matrix)
                hf.close()

            if file_format == 'parquet':
                table = pa.Table.from_pandas(matrix)
                pq.write_table(table, '{:s}.parquet'.format(OUT))

            if file_format == 'csv':
                matrix.to_csv('{:s}.csv'.format(OUT), index=True, header=True)
        
def prep_settings(stage):

    raw_windows             = int(FE_settings[stage]['len_ov'][0])
    raw_overlap             = int(FE_settings[stage]['len_ov'][1])
    cfp                     = FE_settings[stage]['FE'][0]
    compute_only_features   = FE_settings[stage]['FE'][1]
    drop_features           = FE_settings[stage]['FE'][2]

    return raw_windows, raw_overlap, cfp, compute_only_features, drop_features

def prep_other_settings(Ov0,Win1,Ov1,Win2,Ov2,input,stage):
    
    SR = 1/(Win1-Ov1) # for previous 10s window with no overlap, this would be 0.1Hz
    if stage == '2':
        first_window = input.index[0].round("D") if Ov1 == 0 else input.index[0]
    if stage == '3':
        first_window = input.index[0].round("D") if Ov0 == 0 and Ov1 == 0 else input.index[0]
    time = list(input.index)
    #subtract remaining overlaps at the end of the current series + at the beginning of the previous series from Nw_II (below)
    cutoff = int(Ov2/(Win2-Ov2)+(int(np.ceil((Win1-(Win1-Ov1))/(Win2-Ov2)))))
    Nw = int((input.index[-1]-input.index[0].round("D"))/timedelta(seconds = (Win2-Ov2))-cutoff)


    return SR, first_window, Nw, time

def extract_signature_features(stage):
    
    class_I = pd.read_csv('matrix FE I')
    selected_fts = extract_signature_features(matrix)
    selected_fts.to_csv('RSAM/selected_feats.csv', index=True, header=True)
    p_test = calculate_relevance_table(data)

    return p_test
    #load data
    '''
    folder = glob('RSAM/Stage_II/*')
    for file in folder:
        data_matrix = pd.read_csv(file)
        correlation_matrix = data_matrix.corr().abs()
        correlation_matrix = pd.DataFrame(correlation_matrix)
    '''

def raw(volcano = volcano, station = station, FE_settings = FE_settings, file_format = 'hdf5', SR = frequency, skip3rd = True, fband = fbands, decimation_factor = None):  
    
    folder = glob('../SeismicData/{:s}/{:s}/*'.format(volcano,station))

    if not os.path.isdir('RSAM'):
        os.makedirs('RSAM', exist_ok=True)

    all_raw = []    # complete (pre-processed) seismic signal for data available
    all_time = []   # complete time vector for the entire signal

    ###########################
    ### PRE-PROCESSING DATA ###
    ###########################

    for file in sorted(folder):
        print(file)
        # Read file and load velocity data, sampling rate (SR) and starttime (ST)
        try:
            data = read(file)   # read data from file
            raw_data = data.traces[0].data
            SR = int(data.traces[0].meta.sampling_rate) #sampling rate
            ST = pd.to_datetime(data.traces[0].meta.starttime.timestamp, unit = 's').round('ms') #start time
            ET = pd.to_datetime(data.traces[0].meta.endtime.timestamp, unit = 's').round('ms')
        except:
            try:
                data = pd.read_pickle(file) #read data from file
                raw_data = data.data
                SR = int(data.meta.sampling_rate) #sampling rate
                ST = pd.to_datetime(data.meta.starttime.timestamp, unit = 's').round('ms') #start time
                ET = pd.to_datetime(data.meta.endtime.timestamp, unit = 's').round('ms')
            except:
                print('Files should be in mseed or PKL - if they are, then they may be empty or broken.')

        # bandpass-filter data to frequency of interest      
        if fband is not None:
            raw_data = bandpass(raw_data, fband[0][0],fband[0][1], SR, corners = 4)

        # Decimate signal (downsample + anti-aliasing filter) to save memory
        if decimation_factor is not None:
            SR = int(SR/decimation_factor)
            raw_data = decimate(raw_data, decimation_factor)

        # Cut data to length if more than a day of data in one file
        if len(raw_data)>SR*60*60*24: [all_raw.append(x) for x in raw_data[:SR*60*60*24+1]]
        else: [all_raw.append(x) for x in raw_data]

        # Prepare time vector for data
        if len(raw_data)>SR*60*60*24: time = pd.date_range(ST.round('D'), ET.round('D'), periods = SR*60*60*24+1)
        else: time = pd.date_range(ST, ET, periods = (len(raw_data))) #'%Y-%m-%d %H:%M:%S.%f'
        #[all_time.append(datetime.strftime(x, '%Y-%m-%d %H:%M:%S.%f')) for x in time]
        [all_time.append(x) for x in time]
        if all_time.count(ST.round('D')) == 2: all_time.pop(all_time.index(ST.round('D'))), all_raw.pop(all_time.index(ST.round('D')))

    all_time = all_time[:720001]
    all_raw = all_raw[:720001]


    ######################################
    ### FIRST FEATURE EXTRACTION STAGE ###
    ######################################

    raw_windows_I, raw_overlap_I, cfp, compute_only_features, drop_features = prep_settings('Stage_I')

    Nw_I = int(np.floor(len(all_time)/((raw_windows_I-raw_overlap_I)*SR)))-int(raw_overlap_I/(raw_windows_I-raw_overlap_I))

    matrix = FE_from_raw(all_raw, raw_windows_I, raw_overlap_I, SR, Nw_I, all_time[0], all_time, cfp, compute_only_features, drop_features, 
                         'RSAM/{:s}/{:s}/Stage_I/Stage1_{:s}s_{:s}'.format(volcano, station, str(raw_windows_I), str('NO') if raw_overlap_I==0 else str(raw_overlap_I)), file_format, stage = 'Stage_I')

    #######################################
    ### SECOND FEATURE EXTRACTION STAGE ###
    #######################################

    raw_windows_II, raw_overlap_II, cfp, compute_only_features, drop_features = prep_settings('Stage_II')

    SR_II, first_window, Nw_II, time = prep_other_settings(None, raw_windows_I, raw_overlap_I, raw_windows_II, raw_overlap_II, matrix, '2')

    matrix_II = FE_from_raw(matrix, raw_windows_II, raw_overlap_II, SR_II, Nw_II, first_window, time, cfp, compute_only_features, drop_features, None, file_format, stage = 'Stage_II')

    ######################################
    ### THIRD FEATURE EXTRACTION STAGE ###
    ######################################

    if skip3rd is not True:

        raw_windows_III, raw_overlap_III, cfp, compute_only_features, drop_features = prep_settings('Stage_III')

        SR_III, first_window, Nw_III, time = prep_other_settings(raw_overlap_I, raw_windows_II, raw_overlap_II, raw_windows_III, raw_overlap_III, matrix_II, '3')

        FE_from_raw(matrix_II, raw_windows_III, raw_overlap_III, SR_III, Nw_III, first_window, time, cfp, compute_only_features, drop_features, 'RSAM/Stage3_{:s}h_{:s}'.format(str(int(raw_windows_III/3600)), str('NO') if raw_overlap_III==0 else str(int(raw_overlap_III/3600))), file_format, stage = 'Stage_III')


# This is for analysing tremor pulses at WIZ in 2019.

def tremor_drops():
    data_10 = pd.read_csv('/Users/bste426/Documents/All/PhD/Data/Codes/TremorRegimes/1_Feature_Extraction/RSAM/Whakaari/WIZ2016/rsam_Whakaari_WIZ2016_10.csv')
    data_60 = pd.read_csv('/Users/bste426/Documents/All/PhD/Data/Codes/TremorRegimes/1_Feature_Extraction/RSAM/Whakaari/WIZ2016/rsam_Whakaari_WIZ2016_60.csv')
    data_600 = pd.read_csv('/Users/bste426/Documents/All/PhD/Data/Codes/TremorRegimes/1_Feature_Extraction/RSAM/Whakaari/WIZ2016/rsam_Whakaari_WIZ2016_600.csv')

    dsar_10 = [data_10['[4.5, 8]'][i]/data_10['[8, 16]'][i] for i in range(len(data_10))]
    dsar_60 = [data_60['[4.5, 8]'][i]/data_60['[8, 16]'][i] for i in range(len(data_60))]
    dsar_600 = [data_600['[4.5, 8]'][i]/data_60['[8, 16]'][i] for i in range(len(data_600))]

    data_10['DSAR'] = dsar_10
    data_60['DSAR'] = dsar_60
    data_600['DSAR'] = dsar_600

    data_10.index = pd.to_datetime(data_10[data_10.columns[0]])
    data_60.index = pd.to_datetime(data_60[data_60.columns[0]])
    data_600.index = pd.to_datetime(data_600[data_600.columns[0]])


    plt.plot(data_10['DSAR'])
    plt.plot(data_60['DSAR'])
    plt.plot(data_600['DSAR'])
    plt.axvline(Timestamp('2019-12-09 01:10:00'), color = 'red')
    plt.xlabel('Time')
    plt.ylabel('DSAR')

    fig, ax = plt.subplots()
    ax.plot(dsar_60)

    def onclick(event):
        print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
            ('double' if event.dblclick else 'single', event.button,
            event.x, event.y, event.xdata, event.ydata))
        return event.xdata, event.ydata
    fig.canvas.mpl_connect('button_press_event', onclick)

    clicks = pd.read_csv('/Users/bste426/Desktop/clicks.csv')
    click_times = np.round(clicks['Time'])
    click_dates = [data_60.index[int(i)] for i in click_times]
    new_clicks = pd.DataFrame(clicks['Amplitude'])
    new_clicks.index = click_dates
    onset_times = [new_clicks.index[i*2] for i in range(int(len(new_clicks.index)/2))]
    for i in onset_times: plt.axvline(i, color = 'black', alpha = 0.3)
    plt.legend(['10s DSAR', '60s DSAR', '10min DSAR', 'eruption', 'tremor drop'], fontsize = 15)
    plt.xlabel('Time', fontsize = 15)
    plt.ylabel('DSAR', fontsize = 15)
    plt.title('Tremor drops at Whakaari 2019')



if __name__ == "__main__":

    if download is True:
        #GET DATA FROM FDSN SERVER / OFFLINE
        if volcano == 'Colima':
            pass
        else:
            try:
                pull_geonet_data(ncpus=6)
            except:
                pass

    if RSAM is True:
        #CREATE RAW MATRICES WITH DIFFERENT FREQUENCIES AND INTERVALS

        #Processing per desired RSAM interval:
        for interval in intervals:
            RSAM = pd.DataFrame()
            #calculate each RSAM frequency band individually
            for fband in fbands:
                print('Interval: ',interval, ' Frequency band: ',fband) #to keep track
                RSAM[str(fband)] = compile_rsam(interval, fband, '../SeismicData/{:s}/{:s}'.format(volcano,station), ncpus=6)
            #concatenate all fbands in one file and save file:
            if file_format == 'csv':
                RSAM.to_csv('RSAM/{:s}/{:s}/rsam_{:s}_{:s}_{:s}.csv'.format(volcano, station, volcano, station, str(interval)), index=True, header=True)
            if file_format == 'parquet':
                table = pa.Table.from_pandas(RSAM)
                pq.write_table(table, 'RSAM/{:s}/{:s}/rsam_{:s}_{:s}_{:s}.parquet'.format(volcano, station, volcano, station, str(interval)))

    if filter is True:
        filter_RSAM(volcano, station, file_format)

    if features is True:
        #CALCULATE FEATURES FOR EACH RSAM SERIES (AND PERFORMS PCA ON OUTPUT MATRIX IF DESIRED)
        dump_features(PCA_features)

    if from_raw is True:
        raw(volcano, station, FE_settings, file_format, frequency, skip3rd, fbands, decimation_factor)
        extract_signature_features('FE_I')

    if tremor_drops_WIZ is True:
        tremor_drops()

pass
