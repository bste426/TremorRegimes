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
    if not os.path.isdir('../SeismicData'):
        os.makedirs('../SeismicData', exist_ok=True)
    daysec = 24*3600
    t0 = UTCDateTime(t0) + i*daysec
    t1 = t0

    fl = "../SeismicData/{:d}-{:02d}-{:02d}.pkl".format(t0.year, t0.month, t0.day)
    if os.path.isfile(fl):
        return

    # open clients
    client = FDSNClient("GEONET")

    # download data
    try:
        site = client.get_stations(starttime=t0, endtime=t1, station='WIZ', level="response", channel="HHZ")
    except FDSNNoDataException:
        pass

    try:
        WIZ = client.get_waveforms('NZ','WIZ', "10", "HHZ", t0+i*daysec, t0+(i+1)*daysec)

        # if less than 1 day of data, try different client
        if len(WIZ.traces[0].data) < 60*100:
            raise FDSNNoDataException('')
    except ObsPyMSEEDFilesizeTooSmallError:
        return
    except FDSNNoDataException:
        try:
            WIZ = client.get_waveforms('NZ','WIZ', "10", "HHZ", t0+i*daysec, t0 + (i+1)*daysec)
        except FDSNNoDataException:
            return

    WIZ.remove_sensitivity(inventory=site)  # removes station response
    WIZ.traces[0].decimate(2)       # downsample data otherwise its huge (adapt depending on max frequency to be analysed)
    save_dataframe(WIZ.traces[0], fl)

def pull_geonet_data(ncpus=6):
    ''' pulls down all the geonet data for whakaari so you can reprocess different length RSAMs if you like

    '''

    # define range for downloading data
    ti = datetime(2012,8,1,0,0,0)
    tf = datetime(2012,8,5,0,0,0)
    ndays = (tf-ti).days+1

    # parallel data collection - creates temporary files in ../../_tmp_*station*

    f = partial(get_gdata_day, ti)     # hard code initial time and station arguments
    p = Pool(ncpus)

    try:
        for i, _ in enumerate(p.imap(f, range(ndays))):
            cf = (i + 1) / ndays
            print(f'grabbing geonet data: [{"#" * round(50 * cf) + "-" * round(50 * (1 - cf))}] {100. * cf:.2f}%\r', end
                  ='')
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

    fl = 'RSAM/rsam_{:d}_{:3.2f}-{:3.2f}_data.csv'.format(interval, *fband)
    
    # parallel data collection - creates temporary files
    fls = glob('{:s}/*.pkl'.format(src))
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
    save_dataframe(out, fl, index=True)

def get_rsam(interval, fband, fl):

    # load data
    try:
        tr = load_dataframe(fl)
    except:
        print('empty file --- ',fl)
        return

    if tr is None:
        return None
    
    try:
        data = tr.traces[0]
    except:
        data = tr

    #ti = data.meta['starttime']
    ti = UTCDateTime(str(fl)[-14:-4])
    Ns = int(interval)            # seconds in interval
    # round start time to nearest interval
    tiday = UTCDateTime("{:d}-{:02d}-{:02d} 00:00:00".format(ti.year, ti.month, ti.day))
    ti = tiday+int(np.round((ti-tiday)/Ns))*Ns
    N = Ns*50                        # number of samples (at 50 Hz)
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
    #Specify time window length in seconds
    window = ([3600,10800,21600])#,43200,86400,172800,432000])
    
    # Here you can choose if time windows should overlap or not. If so, specify overlap rate.
    overlap = False
    if overlap is True:
        overlap_rate = 0.75     # can be expressed in relative (e.g. 0.75 for 75%) or in absolute quantities (e.g. [(3000/3600)] for 1 hour windows every 10 minutes (5/6 overlap))
    else:
        overlap_rate = 0

    fl = glob('RSAM/*.csv')

    for x in range(len(fl)):
        data_streams = []
        data_streams.append(fl[x][5:-4])
        print(data_streams)
        for i in range(len(window)):
            print(window[i])
            fm = ForecastModel(volcano='whakaari', window=window[i], overlap=overlap_rate, look_forward=0, data_streams=data_streams)
            fm.n_jobs = 6
            FM,ys = fm._extract_features(fm.ti_model, fm.tf_model)
    return

if __name__ == "__main__":

# STEP 1: GET DATA FROM FDSN SERVER / OFFLINE
    pull_geonet_data(ncpus=6)
    print('Time period for downloading data can be adjusted in lines 74/75.')

# STEP 2: CREATE RAW MATRICES WITH DIFFERENT FREQUENCIES AND INTERVALS
    ### Choose RSAM interval (in seconds)
    interval = ([10,600])
    ### Choose frequency bands
    fband = [[1,15],[2,5]]#,[10,15],[2,5],[6,8],[1,2],[0.2,15],[5,10],[0.5,1.1],[0.03,0.125],[1,2.9],[3.5,4],[0.2,5],[1.2,5],[2,5],[2,6.5],[6,8]] 
    for i in range(len(interval)):
        for x in range((len(fband))):
            print(fband[x])
            compile_rsam(interval[i], fband[x], '../SeismicData', ncpus=6)
    print('RSAM rate and frequency band can be adjusted in lines 201/203.')
    
# STEP 3: CALCULATE FEATURES FOR EACH MATRIX
    dump_features()
    print('Time window length can be adjusted in lines 171.')

pass
