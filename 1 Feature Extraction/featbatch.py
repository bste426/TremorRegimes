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
from datetime import timezone

from sklearn.exceptions import FitFailedWarning
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=FitFailedWarning)

volcano = 'Augustine' #'Vulcano' 'Whakaari' 'Redoubt'
station = '*'         #'IVGP'    'WIZ'      'REF'
channel = '*'         #chose EHZ for AV stations or * (use EHZ for reboubt REF, otherwise will download all 3 comps)

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

def get_data_from_stream(st, site):
    if len(st.traces) == 0:
        raise
    elif len(st.traces) > 1:
        try:
            st.merge(fill_value='interpolate').traces[0]
        except Exception:
            st.interpolate(100).merge(fill_value='interpolate').traces[0]

    st.remove_sensitivity(inventory=site)
    # st.detrend('linear')
    # return st.traces[0].data
    return st

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
    if not os.path.isdir('/Users/bste426/Documents/All/PhD/Data/SeismicData/{:s}/_tmp_{:s}'.format(volcano,station)):
        os.makedirs('/Users/bste426/Documents/All/PhD/Data/SeismicData/{:s}/_tmp_{:s}'.format(volcano,station))

    if station=='WIZ':
        daysec = 24*3600
        t0 = UTCDateTime(t0) + i*daysec
        t1 = t0

        fl = "/Users/bste426/Documents/All/PhD/Data/SeismicData/{:s}/_tmp_{:s}/{:d}-{:02d}-{:02d}.pkl".format(volcano, station, t0.year, t0.month, t0.day)
        if os.path.isfile(fl):
            return

        # open clients
        client = FDSNClient("GEONET")

        #'''
        starttime = UTCDateTime("2009-01-10")
        endtime = UTCDateTime("2009-01-20")
        inventory = client.get_stations(network="NZ", station="WIZ", channel='*', starttime = starttime, endtime = endtime)
        print(inventory)
        #'''

        # download data
        try:
            site = client.get_stations(starttime=t0, endtime=t1, station=station, level="response", channel="HHZ")
        except FDSNNoDataException:
            pass

        try:
            WIZ = client.get_waveforms('NZ','WIZ', "10", "HHZ", t0+i*daysec, t0 + (i+1)*daysec)

            # if less than 1 day of data, try different client --- in "X*100" change X to data windows (in seconds)
            if len(WIZ.traces[0].data) < 60*100:
                raise FDSNNoDataException('')
        except ObsPyMSEEDFilesizeTooSmallError:
            return
        except FDSNNoDataException:
            try:
                WIZ = client.get_waveforms('NZ','WIZ', "10", "HHZ", t0+i*daysec, t0 + (i+1)*daysec)
            except FDSNNoDataException:
                return

        WIZ.remove_sensitivity(inventory=site)  # deactivate?
        WIZ.traces[0].decimate(2)       # downsample data otherwise its huge
        save_dataframe(WIZ.traces[0], fl)

    if station=='IVGP':
        daysec = 24 * 3600
        t0 = UTCDateTime(t0) + i*daysec
        t1 = t0

        client = FDSNClient('https://webservices.ingv.it')

        fl = "/Users/bste426/Documents/All/PhD/Data/SeismicData/{:s}/_tmp_{:s}/{:d}-{:02d}-{:02d}.pkl".format(volcano, station, t0.year, t0.month, t0.day)

        if os.path.isfile(fl):
            return

        try:
            site = client.get_stations(starttime=t0, endtime=t1, station=station, level="response", channel="HHZ")
        except FDSNNoDataException:
            pass

        location = ''

        try:
            data = client.get_waveforms('IV', station, location, "HHZ", t0+i*daysec, t0 + (i+1)*daysec)

            # if less than 1 day of data, try different client --- in "X*100" change X to data windows (in seconds)
            if len(data.traces[0].data) < 60*100:
                raise FDSNNoDataException('')
        except ObsPyMSEEDFilesizeTooSmallError:
            return
        except FDSNNoDataException:
            try:
                data = client.get_waveforms('IV', station, location, "HHZ", t0+i*daysec, t0 + (i+1)*daysec)
            except FDSNNoDataException:
                return

        IVGP = get_data_from_stream(data, site)

        #data.remove_sensitivity(inventory=site)  # deactivate?
        #data.traces[0].decimate(6)       # downsample data otherwise its huge
        save_dataframe(IVGP, fl)
        '''
        data.remove_sensitivity(inventory=site)
        data.decimate(6)
        try:
            save_dataframe(data, fl)
        except:
            data = data.traces[0].decimate(2)
            save_dataframe(data.traces[0], fl)
        '''

    else:

        daysec = 24*3600
        t0 = UTCDateTime(t0) + i*daysec
        t1 = t0

        fl = "/Users/bste426/Documents/All/PhD/Data/SeismicData/{:s}/_tmp_{:s}/{:d}-{:02d}-{:02d}.pkl".format(volcano, station, t0.year, t0.month, t0.day)

        if os.path.isfile(fl):
            return

        client = FDSNClient('IRIS')

        #'''
        #Check station data availability for time period
        starttime = UTCDateTime("2005-06-01")
        endtime = UTCDateTime("2005-06-02")
        inventory = client.get_stations(network="AV", station="AU*", channel=channel, starttime = starttime, endtime = endtime)
        print(inventory)
        '''

        # download data
        try:
            site = client.get_stations(starttime=t0, endtime=t1, station=station, level="response", channel=channel)
        except FDSNNoDataException:
            print('error')
            return

        location = ''

        try:
            REF = client.get_waveforms('AV', station, location, channel, t0, t0+1*daysec)
        except FDSNNoDataException:
            print('No data...')
            return


        REF = get_data_from_stream(REF, site)
        save_dataframe(REF, fl)
        '''
        #REF.remove_sensitivity(inventory=site)  # deactivate?
        #REF.traces[0].decimate(2)       # downsample data otherwise its huge
        #save_dataframe(REF, fl)

def pull_geonet_data(clean=False, station = 'station', ncpus=6):
    ''' pulls down all the geonet data for whakaari so you can reprocess different length RSAMs if you like

        this one takes a few hours to run from scratch
    '''

    # pull raw geonet data
    #makedir('/Volumes/SaveTheDay/New_Data/SeismicData/Hawaii/_tmp_'+station)    # DO NOT PUT DATA IN REPOSITORY FOLDER

    # option to delete files and start from scratch
    if clean: _ = [os.remove(fl) for fl in glob('_tmp_{:s}/*.pkl'.format(station))]

    # default data range if not given 
    ti = datetime(2006,1,1,0,0,0)
    tf = datetime(2006,1,2,0,0,0)
    #ti = datetime.fromisoformat(datetime.strftime(datetime.now(timezone.utc), '%Y-%m-%d 00:00:00'))
    #tf = datetime.fromisoformat(datetime.strftime(datetime.now(timezone.utc), '%Y-%m-%d %H:%M:%S'))
    ndays = (tf-ti).days+1

    # parallel data collection - creates temporary files in ../../_tmp_*station*

    f = partial(get_gdata_day, ti)     # hard code initial time and station arguments
    p = Pool(ncpus)
    '''
    for i, _ in enumerate(p.imap(f, range(ndays))):
        cf = (i+1)/ndays
        print(f'grabbing geonet data: [{"#"*round(50*cf)+"-"*round(50*(1-cf))}] {100.*cf:.2f}%\r', end='') 
    '''
    try:
        for i, _ in enumerate(p.imap(f, range(ndays))):
            cf = (i + 1) / ndays
            print(f'grabbing geonet data: [{"#" * round(50 * cf) + "-" * round(50 * (1 - cf))}] {100. * cf:.2f}%\r', end
                  ='')
    except ValueError:  # raised if `y` is empty.
        pass
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
    #fl = '/Users/bste426/Documents/All/PhD/Data/SeismicData/{:s}/rsam_{:d}_{:3.2f}-{:3.2f}_data.csv'.format(volcano,interval, *fband)
    fl = '/Users/bste426/Documents/All/PhD/Data/Codes/Dempsey_new/data/whakaari/rsam_{:d}_{:3.2f}-{:3.2f}_data.csv'.format(interval, *fband)
    if os.path.isfile(fl) and not recompile:
        return
    # parallel data collection - creates temporary files in ../../_tmp_*station*
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
    #if sampling frequency changes, adapt line 90 in __init__.py // window/forward lengths in seconds
    window = ([3600,10800,21600,43200,86400,172800,432000])
    overlap = 0
    #overlap = ([0,(7200/10800),(18000/21600),(39600/43200),(82800/86400),(169200/172800),(428400/432000)])
    fl = glob('../data/whakaari/*.csv')
    #fl = glob('/Users/bste426/Documents/All/PhD/Data/SeismicData/{:s}/*.csv'.format(volcano))

    for x in range(len(fl)):
        flN = fl[x]
        flN = flN[17:-4]
        #flN = flN[-27:-4]
        dtypes = []
        dtypes.append(flN)
        data_streams = dtypes
        print(data_streams)
        for i in range(len(window)):
            print(window[i])
            fm = ForecastModel(volcano='whakaari', window=window[i], overlap=overlap, look_forward=0, data_streams=data_streams)
            fm.n_jobs = 6
            FM,ys = fm._extract_features(fm.ti_model, fm.tf_model)
    return

if __name__ == "__main__":

# STEP 1: GET DATA FROM FDSN SERVER / OFFLINE
    pull_geonet_data(ncpus=6)

    '''
# STEP 2: CREATE RAW MATRICES WITH DIFFERENT FREQUENCIES AND INTERVALS
    interval = ([10])               # interval of windows in seconds
    #fband = [[1,15]] #,[10,15],[2,5],[6,8],[1,2]] #[0.2,15],[5,10],[0.5,1.1],[0.03,0.125],[1,2.9],[3.5,4],[0.2,5],[1.2,5],[2,5],[2,6.5],[6,8]]    # add, change or remove frequency bands here (max. fq: 50Hz)
    fband = [[0.1,1],[1,2],[1,15],[2,5],[5,10],[6,8],[10,15]]    # add, change or remove frequency bands here (max. fq: 50Hz)

    for i in range(len(interval)):
        for x in range((len(fband))):
            print(fband[x])
            compile_rsam(interval[i], fband[x], '/Users/bste426/Documents/All/PhD/Data/SeismicData/{:s}/_tmp_{:s}'.format(volcano, station), ncpus=6, recompile=True)
    '''

# STEP 3: TEST WHETHER DATA HAS BEEN LOADED AND PLOTTED OVER ERUPTIONS CORRECTLY
    #data_test()                # only works for 1 feature band (for now)

# STEP 4: CALCULATE FEATURES FOR EACH MATRIX
    #dump_features()
pass
