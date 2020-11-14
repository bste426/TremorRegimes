import os, sys
from functools import partial

sys.path.insert(0, os.path.abspath('..'))
from whakaari import *
# tsfresh and sklearn dump a lot of warnings - these are switched off below, but should be
# switched back on when debugging
import logging
logger = logging.getLogger("tsfresh")
logger.setLevel(logging.ERROR)
import warnings
from sklearn.exceptions import FitFailedWarning
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=FitFailedWarning)

def data_test():
    ''' tests data format and plotting
    '''
    # required directory structure (repeat as required)
    # /data/
    #   ./volcano1/
    #      ./eruptive_periods.txt
    #      ./dtype1_data.csv
    #      ./dtype2_data.csv
    #   ./volcano2/
    #      ./eruptive_periods.txt
    #      etc.
    #   etc.
    #
    # IMPORTANT
    # check data format for eruptive_periods.txt
    # if multiple data files, they must have the same time indices
    #
    # MOST IMPORTANT
    # any data value must be computed using ONLY data prior to its time index.
    # e.g., if 2015/08/20 is in the time column, then the data column
    # MUST only have been computed using data prior to 2015/08/20

    # test loading of the data
    td = TremorData(volcano='whakaari', frequency='10S')     # loading 10 minute interval tremor data
    
    # test plotting of the data
    # for each eruption, plot data 30 days prior
    for i,te in enumerate(td.tes):
        day = timedelta(days=1)
        td.plot(tlim=[te-1*day, te+1*day], save='../data/{:s}/eruption_{:02d}.png'.format(td.volcano, i))

def get_gdata_day(t0,station,i):
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
    daysec = 24*3600
    t0 = UTCDateTime(t0) + i*daysec
    t1 = t0
    fl = "../../_tmp_{:s}/{:d}-{:02d}-{:02d}.pkl".format(station, t0.year, t0.month, t0.day)
    if os.path.isfile(fl):
        return

    # open clients
    client = FDSNClient("GEONET")

    # download data
    try:
        site = client.get_stations(starttime=t0, endtime=t1, station=station, level="response", channel="HHZ")
    except FDSNNoDataException:
        pass
    '''
    try:
        WIZ = client.get_waveforms('NZ','WIZ', "10", "HHZ", t0+i*daysec, t0 + (i+1)*daysec)

        # if less than 1 day of data, try different client --- in "X*100" change X to data windows (in seconds)
        if len(WIZ.traces[0].data) < 60*100:
            raise FDSNNoDataException('')
    except ObsPyMSEEDFilesizeTooSmallError:
        return
    except FDSNNoDataException:
        try:
            WIZ = client_nrt.get_waveforms('NZ','WIZ', "10", "HHZ", t0+i*daysec, t0 + (i+1)*daysec)
        except FDSNNoDataException:
            return
    '''
    # process frequency bands
    from obspy import read, Stream
    t1c = str(t1)
    WIZ = read('/Volumes/Blau/Whakaari_all/' + t1c[0:23] + '.mseed')
    WIZ.remove_sensitivity(inventory=site)
    WIZ.traces[0].decimate(5)       # downsample data otherwise its huge
    ti = WIZ.traces[0].meta['starttime']
    save_dataframe(WIZ.traces[0], fl)
    
def pull_geonet_data(clean=False, station='WIZ', ncpus=1):
    ''' pulls down all the geonet data for whakaari so you can reprocess different length RSAMs if you like

        this one takes a few hours to run from scratch
    '''

    # pull raw geonet data
    makedir('../../_tmp_'+station)    # DO NOT PUT DATA IN REPOSITORY FOLDER
    
    # option to delete files and start from scratch
    if clean: _ = [os.remove(fl) for fl in glob('_tmp_{:s}/*.pkl'.format(station))]

    # default data range if not given 
    ti = datetime(2011,1,1,0,0,0)    # first reliable date for WIZ
    tf = datetime(2013,1,1,0,0,0)
    ndays = (tf-ti).days+1

    # parallel data collection - creates temporary files in ../../_tmp_*station*
    f = partial(get_gdata_day, ti, station)     # hard code initial time and station arguments
    p = Pool(ncpus)
    for i, _ in enumerate(p.imap(f, range(ndays))):
        cf = (i+1)/ndays
        print(f'grabbing geonet data: [{"#"*round(50*cf)+"-"*round(50*(1-cf))}] {100.*cf:.2f}%\r', end='') 
    
    p.close()
    p.join()

def compile_rsam(interval, fband, src, recompile=False, ncpus=1):
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
    # note, the naming convention in the file below adheres to the description in data_test()
    fl = '../data/whakaari/rsam_{:d}_{:3.2f}-{:3.2f}_data.csv'.format(interval, *fband)

    if os.path.isfile(fl) and not recompile:
        return

    # parallel data collection - creates temporary files in ../../_tmp_*station*
    fls = glob('{:s}/*.pkl'.format(src))
    n = len(fls)                               # counts, how many days-data are in folder
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
    save_dataframe(out, fl, index=True)

def get_rsam(interval, fband, fl):

    # load data
    tr = load_dataframe(fl)
    if tr is None:
        return None
    data = tr.data
    ti = tr.meta['starttime']
    Ns = int(interval*1)            # seconds in interval
    # round start time to nearest interval
    tiday = UTCDateTime("{:d}-{:02d}-{:02d} 00:00:00".format(ti.year, ti.month, ti.day))
    ti = tiday+int(np.round((ti-tiday)/Ns))*Ns
    N = Ns*50                        # number of samples (at 20 Hz)
    Nm = int(N*np.floor(len(data)/N))

    filtered_data = bandpass(data, *fband, 50)
    filtered_data = abs(filtered_data[:Nm])
    filtered_data = filtered_data.reshape(-1,N).mean(axis=-1)*1.e9

    # write out temporary file
    time = [(ti+j*Ns).datetime for j in range(filtered_data.shape[0])]
    df = pd.Series(filtered_data, index=pd.Series(time))
    return df

def dump_features():
    ''' output feature matrix for data
    '''
    #if sampling frequency changes, adapt line 90 in __init__.py // window/forward lengths in seconds
    #window = ([600,3600,10800,21600,43200,86400,172800,657429,2629746])
    window = ([21600,43200])
    fl = glob('../data/whakaari/*_data.csv')
    for x in range(len(fl)):
        flN = fl[x]
        flN = flN[17:35]
        dtypes = []
        dtypes.append(flN)
        data_streams = dtypes
        print(data_streams)
        for i in range(len(window)):
            print(window)
            fm = ForecastModel(volcano='whakaari', window=window[i], overlap=0, look_forward=0, data_streams=data_streams)
            fm.n_jobs = 6
            FM,ys = fm._extract_features(fm.ti_model, fm.tf_model)
    return

if __name__ == "__main__":

# STEP 1: GET DATA FROM FDSN SERVER / OFFLINE
    #pull_geonet_data(ncpus=7)

    '''
# STEP 2: CREATE RAW MATRICES WITH DIFFERENT FREQUENCIES AND INTERVALS
    interval = ([10,60,600])                 # interval of windows in seconds
    fband = [[1.2,25],[8,16]]    # add, change or remove frequency bands here (max. fq: 50Hz)
    for i in range(len(interval)):
        for x in range((len(fband))):
            compile_rsam(interval[i], fband[x], '../../_tmp_WIZ', ncpus=7, recompile=True)
    '''

# STEP 3: TEST WHETHER DATA HAS BEEN LOADED AND PLOTTED OVER ERUPTIONS CORRECTLY
    #data_test()                # only works for 1 feature band (for now)

# STEP 4: CALCULATE FEATURES FOR EACH MATRIX
    dump_features()             # only works for 1 feature band (for now)
pass
