__author__ = """David Dempsey, modified by Bastian Steinke"""
__email__ = 'bastian.steinke@auckland.ac.nz'
__version__ = '0.1.0'

# general imports
import os, sys, shutil, warnings, gc, joblib, pickle
import numpy as np
from datetime import datetime, timedelta, date
from copy import copy
from matplotlib import pyplot as plt
from inspect import getfile, currentframe
from glob import glob
import pandas as pd
from pandas._libs.tslibs.timestamps import Timestamp
from multiprocessing import Pool
from textwrap import wrap
from time import time
from scipy.integrate import cumtrapz
from scipy.signal import stft
from scipy.optimize import curve_fit
from corner import corner
from functools import partial
from fnmatch import fnmatch
import itertools

# ObsPy imports
try:
    from obspy.clients.fdsn import Client as FDSNClient 
    from obspy import UTCDateTime, read_inventory 
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        from obspy.signal.filter import bandpass
    from obspy.io.mseed import ObsPyMSEEDFilesizeTooSmallError
    from obspy.clients.fdsn.header import FDSNNoDataException
    failedobspyimport = False
except:
    failedobspyimport = True

# feature recognition imports
from tsfresh import extract_features, select_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.transformers import FeatureSelector
from tsfresh.feature_extraction.settings import ComprehensiveFCParameters, MinimalFCParameters, EfficientFCParameters, FirstFCParameters, SecondFCParameters
from imblearn.under_sampling import RandomUnderSampler

# classifier imports
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import GridSearchCV, ShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

datas = ['rsam','mf','hf','dsar']
all_classifiers = ["SVM","KNN",'DT','RF','NN','NB','LR']
FMTS = ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d', '%Y %m %d %H %M %S','%m/%d/%Y', '%m/%d/%Y %H:%M']
_MONTH = timedelta(days=365.25/12)
_DAY = timedelta(days=1.)

makedir = lambda name: os.makedirs(name, exist_ok=True)

class TremorData(object):
    """ Object to manage acquisition and processing of seismic data.
        
        Attributes:
        -----------
        volcano : str
            Name of volcano to read data for. Must correspond to directory in ../data
        df : pandas.DataFrame
            Time series of tremor data and transforms.
        t0 : datetime.datetime
            Beginning of data range.
        t1 : datetime.datetime
            End of data range.

        Methods:
        --------
        update
            Obtain latest GeoNet data.
        get_data
            Return tremor data in requested date range.
        plot
            Plot tremor data.
    """
    def __init__(self, volcano, station, frequency, data_streams,file_format):
        self.volcano = volcano
        self.frequency = frequency
        self.dir = os.sep.join(getfile(currentframe()).split(os.sep)[:-2]+['RSAM/{:s}/{:s}'.format(volcano, station)])
        self._assess(data_streams,file_format)
    def __repr__(self):
        if self.exists:
            tm = [self.ti.year, self.ti.month, self.ti.day, self.ti.hour, self.ti.minute]
            tm += [self.tf.year, self.tf.month, self.tf.day, self.tf.hour, self.tf.minute]
            return 'TremorData:{:d}/{:02d}/{:02d} {:02d}:{:02d} to {:d}/{:02d}/{:02d} {:02d}:{:02d}'.format(*tm)
        else:
            return 'no data'
    def _assess(self, data_streams,file_format):
        """ Load existing file and check date range of data.
        """
        # error checking
        if not os.path.isdir(self.dir):
            raise ValueError('no directory for volcano \'{:s}\''.format(self.dir))
        #fl = '{:s}/eruptive_periods.txt'.format(self.dir)
        #if not os.path.isfile(fl):
        #    raise ValueError('no eruption data file \'{:s}\''.format(fl))
        fl = '{:s}/*.{:s}'.format(self.dir, file_format)
        data_fls = glob(fl)
        if len(data_fls) == 0:
            raise ValueError('no data files \'{:s}\''.format(fl))
        # get eruptions
        #with open('{:s}/eruptive_periods.txt'.format(self.dir),'r') as fp:
        #    self.tes = [datetimeify(ln.rstrip()) for ln in fp.readlines()]
        # check if data file exists
        self._load_data(data_streams,file_format)
    def _load_data(self,data_streams,file_format):
        #data_streams = str(data_streams)
        #fl = '/Users/bst/Documents/All/PhD/Data/Codes/Dempsey_new/data/whakaari/rsam_600_0.01-0.50_data.csv'
        dir = '{:s}/{:s}.{:s}'
        fl = dir.format(self.dir, data_streams,file_format)
        if file_format == 'csv':
            d = pd.read_csv(fl)
            fmt = detect_timeformat(d[d.columns[0]][0])
            self.df = d.set_index([d.columns[0]], drop = True).rename_axis('Time')
            self.df.index = [datetimeify(date, fmt) for date in d[d.columns[0]]]
            self.ti = np.min(self.df.index)
            self.tf = np.max(self.df.index)
        elif file_format == 'parquet':
            d = pd.read_parquet(fl)
            fmt = detect_timeformat(str(d.index[0]))
            self.df = d.rename_axis('Time')
            self.ti = np.min(self.df.index)
            self.tf = np.max(self.df.index)
        else:
            print('Unrecognized format of RSAM file (should be csv or parquet).')

    def _compute_transforms(self):
        """ Compute data transforms.

            Notes:
            ------
            Naming convention is *transform_type*_*data_type*, so for example
            'inv_mf' is "inverse medium frequency or 1/mf. Other transforms are
            'diff' (derivative), 'log' (base 10 logarithm) and 'stft' (short-time
            Fourier transform averaged across 40-45 periods).
        """
        return
        for col in self.df.columns:
            if col == 'time': continue
            # inverse
            if 'inv_'+col not in self.df.columns:
                self.df['inv_'+col] = 1./self.df[col]
            # diff
            if 'diff_'+col not in self.df.columns:
                self.df['diff_'+col] = self.df[col].diff()
                self.df['diff_'+col][0] = 0.
            # log
            if 'log_'+col not in self.df.columns:
                self.df['log_'+col] = np.log10(self.df[col])
            # stft
            if 'stft_'+col not in self.df.columns:
                seg,freq = [12,16]
                data = pd.Series(np.zeros(seg*6-1))
                data = data.append(self.df[col], ignore_index=True)
                Z = abs(stft(data.values, window='nuttall', nperseg=seg*6, noverlap=seg*6-1, boundary=None)[2])
                self.df['stft_'+col] = np.mean(Z[freq:freq+2,:],axis=0)
    def _is_eruption_in(self, days, from_time):
        """ Binary classification of eruption imminence.

            Parameters:
            -----------
            days : float
                Length of look-forward.
            from_time : datetime.datetime
                Beginning of look-forward period.

            Returns:
            --------
            label : int
                1 if eruption occurs in look-forward, 0 otherwise
            
        """
        for te in self.tes:
            if 0 < (te-from_time).total_seconds()/(3600*24) < days:
                return 1.
        return 0.
    def get_data(self, ti=None, tf=None):
        """ Return tremor data in requested date range.

            Parameters:
            -----------
            ti : str, datetime.datetime
                Date of first data point (default is earliest data).
            tf : str, datetime.datetime
                Date of final data point (default is latest data).

            Returns:
            --------
            df : pandas.DataFrame
                Data object truncated to requsted date range.
        """
        # set date range defaults
        if ti is None:
            ti = self.ti
        if tf is None:
            tf = self.tf

        # convert datetime format
        ti = datetimeify(ti)
        tf = datetimeify(tf)

        # subset data
        inds = (self.df.index>=ti)&(self.df.index<tf)
        return self.df.loc[inds]
    def plot(self, data_streams=None, save='tremor_data.png', tlim = [None, None], ylim=[0, 5000]):
        """ Plot tremor data.

            Parameters:
            -----------
            save : str
                Name of file to save output.
            data_streams : str, list
                String or list of strings indicating which data or transforms to plot (see below). 
            ylim : list
                Two-element list indicating y-axis limits for plotting.
                
            data type options:
            ------------------
            rsam - 2 to 5 Hz (Real-time Seismic-Amplitude Measurement)
            mf - 4.5 to 8 Hz (medium frequency)
            hf - 8 to 16 Hz (high frequency)
            dsar - ratio of mf to hf, rolling median over 180 days

            transform options:
            ------------------
            inv - inverse, i.e., 1/
            diff - finite difference derivative
            log - base 10 logarithm
            stft - short-time Fourier transform at 40-45 min period

            Example:
            --------
            data_streams = ['dsar', 'diff_hf'] will plot the DSAR signal and the derivative of the HF signal.
        """
        if type(data_streams) is str:
            data_streams = [data_streams,]
        elif data_streams is None:
            data_streams = self.df.columns
        if any(['_' in ds for ds in data_streams]):
            self._compute_transforms()
        ti,tf = tlim
        if ti is None: ti = self.ti
        if tf is None: tf = self.tf

        # set up figures and axes
        f = plt.figure(figsize=(8,4))
        ax = plt.axes([0.1, 0.1, 0.8, 0.8])

       # plot data for each year
        data = self.get_data(ti, tf)
        xi = datetime(year=1,month=1,day=1,hour=0,minute=0,second=0)
        cols = ['c','m','y','g',[0.5,0.5,0.5],[0.75,0.75,0.75]]
        for data_stream, col in zip(data_streams,cols):
            ax.plot(data.index, data[data_stream], '-', color=col, label=data_stream)
        
        for te in self.tes:
            ax.axvline(te, color='r', linestyle='--', linewidth=2)
        ax.axvline(te, color='r', linestyle='--', linewidth=2, label='eruption')
        ax.set_xlim([ti,tf])
        ax.legend()
        
        plt.savefig(save, dpi=400)

class ForecastModel(object):
    """ Object for train and running forecast models.
        
        Constructor arguments:
        ----------------------
        window : float
            Length of data window in days.
        overlap : float
            Fraction of overlap between adjacent windows. Set this to 1. for overlap of entire window minus 1 data point.
        look_forward : float
            Length of look-forward in days.
        ti : str, datetime.datetime
            Beginning of analysis period. If not given, will default to beginning of tremor data.
        tf : str, datetime.datetime
            End of analysis period. If not given, will default to end of tremor data.
        data_streams : list
            Data streams and transforms from which to extract features. Options are 'X', 'diff_X', 'log_X', 'inv_X', and 'stft_X' 
            where X is one of 'rsam', 'mf', 'hf', or 'dsar'.            
        root : str 
            Naming convention for forecast files. If not given, will default to 'fm_*Tw*wndw_*eta*ovlp_*Tlf*lkfd_*ds*' where
            Tw is the window length, eta is overlap fraction, Tlf is look-forward and ds are data streams.
        savefile_type : str
            Extension denoting file format for save/load. Options are csv, pkl (Python pickle) or hdf.

        Attributes:
        -----------
        data : TremorData
            Object containing tremor data.
        dtw : datetime.timedelta
            Length of window.
        dtf : datetime.timedelta
            Length of look-forward.
        dt : datetime.timedelta
            Length between data samples (10 minutes).
        dto : datetime.timedelta
            Length of non-overlapping section of window.
        iw : int
            Number of samples in window.
        io : int
            Number of samples in overlapping section of window.
        ti_model : datetime.datetime
            Beginning of model analysis period.
        tf_model : datetime.datetime
            End of model analysis period.
        ti_train : datetime.datetime
            Beginning of model training period.
        tf_train : datetime.datetime
            End of model training period.
        ti_forecast : datetime.datetime
            Beginning of model forecast period.
        tf_forecast : datetime.datetime
            End of model forecast period.
        drop_features : list
            List of tsfresh feature names or feature calculators to drop during training.
            Facilitates manual dropping of correlated features.
        exclude_dates : list
            List of time windows to exclude during training. Facilitates dropping of eruption 
            windows within analysis period. E.g., exclude_dates = [['2012-06-01','2012-08-01'],
            ['2015-01-01','2016-01-01']] will drop Jun-Aug 2012 and 2015-2016 from analysis.
        use_only_features : list
            List of tsfresh feature names or calculators that training will be restricted to.
        compute_only_features : list
            List of tsfresh feature names or calcluators that feature extraction will be 
            restricted to.
        update_feature_matrix : bool
            Set this True in rare instances you want to extract feature matrix without the code
            trying first to update it.
        n_jobs : int
            Number of CPUs to use for parallel tasks.
        rootdir : str
            Repository location on file system.
        plotdir : str
            Directory to save forecast plots.
        modeldir : str
            Directory to save forecast models (pickled sklearn objects).
        featdir : str
            Directory to save feature matrices.
        featfile : str
            File to save feature matrix to.
        preddir : str
            Directory to save forecast model predictions.

        Methods:
        --------
        _detect_model
            Checks whether and what models have already been run.
        _construct_windows
            Create overlapping data windows for feature extraction.
        _extract_features
            Extract features from windowed data.
        _get_label
            Compute label vector.
        _load_data
            Load feature matrix and label vector.
        _drop_features
            Drop columns from feature matrix.
        _exclude_dates
            Drop rows from feature matrix and label vector.
        _collect_features
            Aggregate features used to train classifiers by frequency.
        _model_alerts
            Compute issued alerts for model consensus.
        get_features
            Return feature matrix and label vector for a given period.
        train
            Construct classifier models.
        forecast
            Use classifier models to forecast eruption likelihood.
        hires_forecast
            Construct forecast at resolution of data.
        plot_forecast
            Plot model forecast.
        plot_accuracy
            Plot performance metrics for model.
        plot_features
            Plot frequency of extracted features by most significant.
        plot_feature_correlation
            Corner plot of feature correlation.
    """
    def __init__(self, volcano, station, frequency, window, overlap, look_forward, ti=None, tf=None, data_streams = '', freq_bands=['rsam','mf','hf','dsar'], file_format = 'csv', root=None, savefile_type='csv'):
        self.volcano = volcano
        self.window = window
        self.overlap = overlap
        self.look_forward = look_forward
        self.data_streams = data_streams
        self.data = TremorData(self.volcano,station,frequency,data_streams,file_format)
        if any(['_' in ds for ds in freq_bands]):
            self.data._compute_transforms()
        if any([d not in self.data.df.columns for d in freq_bands]):
            print(self.data.df.columns,freq_bands)
            raise ValueError("data restricted to any of {}".format(self.data.df.columns))
        if ti is None: ti = self.data.ti
        if tf is None: tf = self.data.tf
        self.ti_model = datetimeify(ti)
        self.tf_model = datetimeify(tf)
        if self.tf_model > self.data.tf:
            t0,t1 = [self.tf_model.strftime('%Y-%m-%d %H:%M'), self.data.tf.strftime('%Y-%m-%d %H:%M')]
            raise ValueError("Model end date '{:s}' beyond data range '{:s}'".format(t0,t1))
        if self.ti_model < self.data.ti:
            t0,t1 = [self.ti_model.strftime('%Y-%m-%d %H:%M'), self.data.ti.strftime('%Y-%m-%d %H:%M')]
            raise ValueError("Model start date '{:s}' predates data range '{:s}'".format(t0,t1))
        self.dtw = timedelta(seconds=self.window)
        self.dtf = timedelta(seconds=self.look_forward)
        self.dt = (self.data.df.index[1] - self.data.df.index[0])
        dt = int(self.dt.total_seconds())
        self.dto = (1.-self.overlap)*self.dtw
        self.iw = int(self.window/dt)
        self.io = int(self.overlap*self.iw)
        if self.io == self.iw: self.io -= 1

        self.window = self.iw*dt
        self.dtw = timedelta(seconds=self.window)
        if self.ti_model - self.dtw < self.data.ti:
            self.ti_model = self.data.ti+self.dtw
        self.overlap = self.io*1./self.iw
        self.dto = (1.-self.overlap)*self.dtw

        self.drop_features = []
        self.exclude_dates = []
        self.use_only_features = []
        self.compute_only_features = []
        self.update_feature_matrix = True
        self.n_jobs = 6

        # naming convention and file system attributes
        self.savefile_type = savefile_type
        if root is None:
            self.root = '{:s}_{:s}_{:3.2f}'.format(self.volcano, station, self.window)
            self.root += '_'+((('{:s}-')*len(self.data_streams))[:-1]).format(*sorted(self.data_streams))
        else:
            self.root = root
        self.rootdir = os.sep.join(getfile(currentframe()).split(os.sep)[:-2])
        self.plotdir = r'{:s}/plots/{:s}'.format(self.rootdir, self.root)
        self.modeldir = r'{:s}/models/{:s}'.format(self.rootdir, self.root)
        self.featdir = r'{:s}/features'.format(self.rootdir, self.root)
        self.featfile = r'{:s}/{:s}/{:s}/{:s}_features.{:s}'.format(self.featdir, volcano, station, self.root, self.savefile_type)
        self.preddir = r'{:s}/predictions/{:s}'.format(self.rootdir, self.root)
    # private helper methods
    def _detect_model(self):
        """ Checks whether and what models have already been run.
        """
        fls = glob(self._use_model+os.sep+'*.fts')
        if len(fls) == 0:
            raise ValueError("no feature files in '{:s}'".format(self._use_model))

        inds = [int(float(fl.split(os.sep)[-1].split('.')[0])) for fl in fls if ('all.fts' not in fl)]
        if max(inds) != (len(inds) - 1):
            raise ValueError("feature file numbering in '{:s}' appears not consecutive".format(self._use_model))
        
        self.classifier = []
        for classifier in all_classifiers:
            model = get_classifier(classifier)[0]
            pref = type(model).__name__
            if all([os.path.isfile(self._use_model+os.sep+'{:s}_{:04d}.pkl'.format(pref,ind)) for ind in inds]):
                self.classifier = classifier
                return
        raise ValueError("did not recognise models in '{:s}'".format(self._use_model))
    def _construct_windows(self, Nw, ti, freqband, i0=0, i1=None):
        """ Create overlapping data windows for feature extraction.

            Parameters:
            -----------
            Nw : int
                Number of windows to create.
            ti : datetime.datetime
                End of first window.
            i0 : int
                Skip i0 initial windows.
            i1 : int
                Skip i1 final windows.

            Returns:
            --------
            df : pandas.DataFrame
                Dataframe of windowed data, with 'id' column denoting individual windows.
            window_dates : list
                Datetime objects corresponding to the beginning of each data window.
        """
        if i1 is None:
            i1 = Nw

        # get data for windowing period
        df = self.data.get_data(ti-self.dtw, ti+(Nw-1)*self.dto)[freqband]

        # create windows
        #dfs = pd.DataFrame()
        dfs = []
        dfi_id = []
        dfi_index = []
        print('Constructing windows...')
        for i in range(i0, i1):
            # Changed back to lists for faster processing
            #dfi = pd.DataFrame(df[:].iloc[i*(self.iw-self.io):i*(self.iw-self.io)+self.iw])
            dfi = df[:].iloc[i*(self.iw-self.io):i*(self.iw-self.io)+self.iw]
            try:
                #dfi['id'] = pd.DataFrame(np.ones(self.iw, dtype=int)*i, index=dfi.index)
                dfi_id.append(list(np.ones(self.iw, dtype=int)*i))
                dfi_index.append(list(dfi.index))
            except ValueError:
                print('this shouldn\'t be happening')
            dfs.append(list(dfi))
            #dfs = dfs.append(dfi)
        #df = pd.concat(dfs)
        dfi_index_list = list(itertools.chain.from_iterable(dfi_index))
        dfs_list = list(itertools.chain.from_iterable(dfs))
        dfi_id_list = list(itertools.chain.from_iterable(dfi_id))

        new_df = pd.DataFrame(dfs_list, index = dfi_index_list)
        new_df.columns = ([freqband])
        new_df['id'] = dfi_id_list
        df = new_df
        
        #df = dfs
        window_dates = [ti + i*self.dto for i in range(Nw)]

        return df, window_dates[i0:i1]
    def _extract_features(self, ti, tf, freq_bands):
        """ Extract features from windowed data.

            Parameters:
            -----------
            ti : datetime.datetime
                End of first window.
            tf : datetime.datetime
                End of last window.

            Returns:
            --------
            fm : pandas.Dataframe
                tsfresh feature matrix extracted from data windows.
            ys : pandas.Dataframe
                Label vector corresponding to data windows

            Notes:
            ------
            Saves feature matrix to $rootdir/features/$root_features.csv to avoid recalculation.
        """
        makedir(self.featdir)

        # number of windows in feature request
        Nw = int(np.floor(((tf-ti)/self.dt)/(self.iw-self.io)))+1

        # features to compute
        #cfp = MinimalFCParameters()
        cfp = SecondFCParameters()
        self.compute_only_features = None #,'sample_entropy','maximum','minimum','c3','autocorrelation', 'agg_autocorrelation', 'partial_autocorrelation','number_peaks','binned_entropy','approximate_entropy','linear_trend', 'agg_linear_trend','ar_coefficient']
        self.drop_features = ['permutation_entropy','multi_ACF_lag_stats','central_freqs','dsar']
        if self.compute_only_features:
            cfp = dict([(k, cfp[k]) for k in cfp.keys() if k in self.compute_only_features])
        else:
            # drop features if relevant
            _ = [cfp.pop(df) for df in self.drop_features if df in list(cfp.keys())]
        '''
        # check if feature matrix already exists and what it contains
        if os.path.isfile(self.featfile):
            # t = pd.to_datetime(pd.read_csv(self.featfile, index_col=0, parse_dates=['time'], usecols=['time'], infer_datetime_format=True).index.values)
            t = pd.to_datetime(load_dataframe(self.featfile, index_col=0, parse_dates=['time'], usecols=['time'], infer_datetime_format=True).index.values)
            ti0,tf0 = t[0],t[-1]
            Nw0 = len(t)
            # hds = pd.read_csv(self.featfile, index_col=0, nrows=1)
            hds = load_dataframe(self.featfile, index_col=0, nrows=1)
            hds = list(set([hd.split('__')[1] for hd in hds]))

            # option 1, expand rows
            pad_left = int((ti0-ti)/self.dto)# if ti < ti0 else 0
            pad_right = int(((ti+(Nw-1)*self.dto)-tf0)/self.dto)# if tf > tf0 else 0
            i0 = abs(pad_left) if pad_left<0 else 0
            i1 = Nw0 + max([pad_left,0]) + pad_right
            
            # option 2, expand columns
            existing_cols = set(hds)        # these features already calculated, in file
            new_cols = set(cfp.keys()) - existing_cols     # these features to be added
            more_cols = bool(new_cols)
            all_cols = existing_cols|new_cols
            cfp = ComprehensiveFCParameters()
            cfp = dict([(k, cfp[k]) for k in cfp.keys() if k in all_cols])

            # option 3, expand both
            if any([more_cols, pad_left > 0, pad_right > 0]) and self.update_feature_matrix:
                # fm = pd.read_csv(self.featfile, index_col=0, parse_dates=['time'], infer_datetime_format=True)
                fm = load_dataframe(self.featfile, index_col=0, parse_dates=['time'], infer_datetime_format=True)
                if more_cols:
                    # expand columns now
                    df0, wd = self._construct_windows(Nw0, ti0)
                    cfp0 = ComprehensiveFCParameters()
                    cfp0 = dict([(k, cfp0[k]) for k in cfp0.keys() if k in new_cols])
                    fm2 = extract_features(df0, column_id='id', n_jobs=self.n_jobs, default_fc_parameters=cfp0, impute_function=impute)
                    fm2.index = pd.Series(wd)
                    fm2.index.name = 'time'
                    
                    fm = pd.concat([fm,fm2], axis=1, sort=False)

                # check if updates required because training period expanded
                    # expanded earlier
                if pad_left > 0:
                    df, wd = self._construct_windows(Nw, ti, i1=pad_left)
                    fm2 = extract_features(df, column_id='id', n_jobs=self.n_jobs, default_fc_parameters=cfp, impute_function=impute)
                    fm2.index = pd.Series(wd)
                    fm2.index.name = 'time'
                    fm = pd.concat([fm2,fm], sort=False)
                    # expanded later
                if pad_right > 0:
                    df, wd = self._construct_windows(Nw, ti, i0=Nw - pad_right)
                    fm2 = extract_features(df, column_id='id', n_jobs=self.n_jobs, default_fc_parameters=cfp, impute_function=impute)
                    fm2.index = pd.Series(wd)
                    fm2.index.name = 'time'
                    fm = pd.concat([fm,fm2], sort=False)
                
                # write updated file output
                # fm.to_csv(self.featfile, index=True, index_label='time')
                save_dataframe(fm, self.featfile, index=True, index_label='time')
                # trim output
                fm = fm.iloc[i0:i1]    
            else:
                # read relevant part of matrix
                # fm = pd.read_csv(self.featfile, index_col=0, parse_dates=['time'], infer_datetime_format=True, header=0, skiprows=range(1,i0+1), nrows=i1-i0)
                fm = load_dataframe(self.featfile, index_col=0, parse_dates=['time'], infer_datetime_format=True, header=0, skiprows=range(1,i0+1), nrows=i1-i0)
        else:
            # create feature matrix from scratch  
            '''
        matrix = pd.DataFrame()
        for freqband in freq_bands:
            print('Now processing ', freqband)
            df, wd = self._construct_windows(Nw, ti, freqband)
            fm = extract_features(df, column_id='id', n_jobs=self.n_jobs, default_fc_parameters=cfp, impute_function=impute)
            #fm.index = pd.Series(wd)
            #fm.index.name = 'time'
            for c in fm.columns:
                matrix[c] = fm[c]
        matrix.index = pd.Series(wd)
        matrix.index.name = 'time'
        # fm.to_csv(self.featfile, index=True, index_label='time')
        #save_dataframe(fm, self.featfile, index=True, index_label='time')
        
        #ys = pd.DataFrame(self._get_label(fm.index.values), columns=['label'], index=fm.index)
        return matrix
    def _get_label(self, ts):
        """ Compute label vector.

            Parameters:
            -----------
            t : datetime like
                List of dates to inspect look-forward for eruption.

            Returns:
            --------
            ys : list
                Label vector.
        """
        return [self.data._is_eruption_in(days=self.look_forward, from_time=t) for t in pd.to_datetime(ts)]
    def _load_data(self, ti, tf):
        """ Load feature matrix and label vector.

            Parameters:
            -----------
            ti : str, datetime
                Beginning of period to load features.
            tf : str, datetime
                End of period to load features.

            Returns:
            --------
            fM : pd.DataFrame
                Feature matrix.
            ys : pd.DataFrame
                Label vector.
        """
        # return pre loaded
        try:
            if ti == self.ti_prev and tf == self.tf_prev:
                return self.fM, self.ys
        except AttributeError:
            pass

        # read from CSV file
        try:
            # t = pd.to_datetime(pd.read_csv(self.featfile, index_col=0, parse_dates=['time'], usecols=['time'], infer_datetime_format=True).index.values)
            t = pd.to_datetime(load_dataframe(self.featfile, index_col=0, parse_dates=['time'], usecols=['time'], infer_datetime_format=True).index.values)
            if (t[0] <= ti) and (t[-1] >= tf):
                self.ti_prev = ti
                self.tf_prev = tf
                fM,ys = self._extract_features(ti,tf)
                self.fM = fM
                self.ys = ys
                return fM,ys
        except FileNotFoundError:
            pass

        # range checking
        if tf > self.data.tf:
            raise ValueError("Model end date '{:s}' beyond data range '{:s}'".format(tf, self.data.tf))
        if ti < self.data.ti:
            raise ValueError("Model start date '{:s}' predates data range '{:s}'".format(ti, self.data.ti))
        
        # divide training period into years
        ts = [datetime(*[yr, 1, 1, 0, 0, 0]) for yr in list(range(ti.year+1, tf.year+1))]
        if ti - self.dtw < self.data.ti:
            ti = self.data.ti + self.dtw
        ts.insert(0,ti)
        ts.append(tf)
        for t0,t1 in zip(ts[:-1], ts[1:]):
            print('feature extraction {:s} to {:s}'.format(t0.strftime('%Y-%m-%d'), t1.strftime('%Y-%m-%d')))
            fM,ys = self._extract_features(ti,t1)
        self.ti_prev = ti
        self.tf_prev = tf
        self.fM = fM
        self.ys = ys
        return fM, ys
    def _drop_features(self, X, drop_features):
        """ Drop columns from feature matrix.

            Parameters:
            -----------
            X : pd.DataFrame
                Matrix to drop columns.
            drop_features : list
                tsfresh feature names or calculators to drop from matrix.

            Returns:
            --------
            Xr : pd.DataFrame
                Reduced matrix.
        """
        self.drop_features = drop_features
        if len(self.drop_features) != 0:
            cfp = ComprehensiveFCParameters()
            df2 = []
            for df in self.drop_features:
                if df in X.columns:
                    df2.append(df)          # exact match
                else:
                    if df in cfp.keys() or df in ['fft_coefficient_hann']:
                        df = '*__{:s}__*'.format(df)    # feature calculator
                    # wildcard match
                    df2 += [col for col in X.columns if fnmatch(col, df)]              
            X = X.drop(columns=df2)
        return X
    def _exclude_dates(self, X, y, exclude_dates):
        """ Drop rows from feature matrix and label vector.

            Parameters:
            -----------
            X : pd.DataFrame
                Matrix to drop columns.
            y : pd.DataFrame
                Label vector.
            exclude_dates : list
                List of time windows to exclude during training. Facilitates dropping of eruption 
                windows within analysis period. E.g., exclude_dates = [['2012-06-01','2012-08-01'],
                ['2015-01-01','2016-01-01']] will drop Jun-Aug 2012 and 2015-2016 from analysis.

            Returns:
            --------
            Xr : pd.DataFrame
                Reduced matrix.
            yr : pd.DataFrame
                Reduced label vector.
        """
        self.exclude_dates = exclude_dates
        if len(self.exclude_dates) != 0:
            for exclude_date_range in self.exclude_dates:
                t0,t1 = [datetimeify(dt) for dt in exclude_date_range]
                inds = (y.index<t0)|(y.index>=t1)
                X = X.loc[inds]
                y = y.loc[inds]
        return X,y
    def _collect_features(self, save=None):
        """ Aggregate features used to train classifiers by frequency.

            Parameters:
            -----------
            save : None or str
                If given, name of file to save feature frequencies. Defaults to all.fts
                if model directory.

            Returns:
            --------
            labels : list
                Feature names.
            freqs : list
                Frequency of feature appearance in classifier models.
        """
        makedir(self.modeldir)
        if save is None:
            save = '{:s}/all.fts'.format(self.modeldir)
        
        feats = []
        fls = glob('{:s}/*.fts'.format(self.modeldir))
        for i,fl in enumerate(fls):
            if fl.split(os.sep)[-1].split('.')[0] in ['all','ranked']: continue
            with open(fl) as fp:
                lns = fp.readlines()
            feats += [' '.join(ln.rstrip().split()[1:]) for ln in lns]               

        labels = list(set(feats))
        freqs = [feats.count(label) for label in labels]
        labels = [label for _,label in sorted(zip(freqs,labels))][::-1]
        freqs = sorted(freqs)[::-1]
        # write out feature frequencies
        with open(save, 'w') as fp:
            _ = [fp.write('{:d},{:s}\n'.format(freq,ft)) for freq,ft in zip(freqs,labels)]
        return labels, freqs
    def _model_alerts(self, t, y, threshold, ialert, dti):
        """ Compute issued alerts for model consensus.

            Parameters:
            -----------
            t : array-like
                Time vector corresponding to model consensus.
            y : array-like
                Model consensus.
            threshold : float
                Consensus value above which an alert is issued.
            ialert : int
                Number of data windows spanning an alert period.
            dti : datetime.timedelta
                Length of window overlap.

            Returns:
            --------
            false_alert : int
                Number of falsely issued alerts.
            missed : int
                Number of eruptions for which an alert not issued.
            true_alert : int
                Number of eruptions for which an alert correctly issued.
            true_negative : int
                Equivalent number of time windows in which no alert was issued and no eruption
                occurred. Each time window has the average length of all issued alerts.
            dur : float
                Total alert duration as fraction of model analysis period.
            mcc : float
                Matthews Correlation Coefficient.
        """
        # create contiguous alert windows
        inds = np.where(y>threshold)[0]

        if len(inds) == 0:
            return 0, len(self.data.tes), 0, int(1e8), 0, 0

        dinds = np.where(np.diff(inds)>ialert)[0]
        alert_windows = list(zip(
            [inds[0],]+[inds[i+1] for i in dinds],
            [inds[i]+ialert for i in dinds]+[inds[-1]+ialert]
            ))
        alert_window_lengths = [np.diff(aw) for aw in alert_windows]
        
        # compute true/false positive/negative rates
        tes = copy(self.data.tes)
        nes = len(self.data.tes)
        nalerts = len(alert_windows)
        true_alert = 0
        false_alert = 0
        inalert = 0.
        missed = 0
        total_time = (t[-1] - t[0]).total_seconds()

        for i0,i1 in alert_windows:

            inalert += ((i1-i0)*dti).total_seconds()
            # no eruptions left to classify, only misclassifications now
            if len(tes) == 0:
                false_alert += 1
                continue

            # eruption has been missed
            while tes[0] < t[i0]:
                tes.pop(0)
                missed += 1
                if len(tes) == 0:
                    break
            if len(tes) == 0:
                continue

            # alert does not contain eruption
            if not (tes[0] > t[i0] and tes[0] <= (t[i0] + (i1-i0)*dti)):
                false_alert += 1
                continue

            # alert contains eruption
            while tes[0] > t[i0] and tes[0] <= (t[i0] + (i1-i0)*dti):
                tes.pop(0)
                true_alert += 1
                if len(tes) == 0:
                    break

        # any remaining eruptions after alert windows have cleared must have been missed
        missed += len(tes)
        dur = inalert/total_time
        true_negative = int((len(y)-np.sum(alert_window_lengths))/np.mean(alert_window_lengths))-missed
        mcc = matthews_corrcoef(self._ys, (y>threshold)*1.)

        return false_alert, missed, true_alert, true_negative, dur, mcc
    # public methods
    def get_features(self, ti=None, tf=None, n_jobs=6, drop_features=[], compute_only_features=[]):
        """ Return feature matrix and label vector for a given period.

            Parameters:
            -----------
            ti : str, datetime.datetime
                Beginning of period to extract features (default is beginning of model analysis).
            tf : str, datetime.datetime
                End of period to extract features (default is end of model analysis).
            n_jobs : int
                Number of cores to use.
            drop_feautres : list
                tsfresh feature names or calculators to exclude from matrix.
            compute_only_features : list
                tsfresh feature names of calculators to return in matrix.
            
            Returns:
            --------
            fM : pd.DataFrame
                Feature matrix.
            ys : pd.Dataframe
                Label vector.
        """
        # initialise training interval
        self.drop_features = drop_features
        self.compute_only_features = compute_only_features
        self.n_jobs = n_jobs
        ti = self.ti_model if ti is None else datetimeify(ti)
        tf = self.tf_model if tf is None else datetimeify(tf)
        return self._load_data(ti, tf)
    def train(self, ti=None, tf=None, Nfts=20, Ncl=100, retrain=False, classifier="DT", random_seed=0,
            drop_features=[], n_jobs=6, exclude_dates=[], use_only_features=[]):
        """ Construct classifier models.

            Parameters:
            -----------
            ti : str, datetime.datetime
                Beginning of training period (default is beginning model analysis period).
            tf : str, datetime.datetime
                End of training period (default is end of model analysis period).
            Nfts : int
                Number of most-significant features to use in classifier.
            Ncl : int
                Number of classifier models to train.
            retrain : boolean
                Use saved models (False) or train new ones.
            classifier : str, list
                String or list of strings denoting which classifiers to train (see options below.)
            random_seed : int
                Set the seed for the undersampler, for repeatability.
            drop_features : list
                Names of tsfresh features to be dropped prior to training (for manual elimination of 
                feature correlation.)
            n_jobs : int
                CPUs to use when training classifiers in parallel.
            exclude_dates : list
                List of time windows to exclude during training. Facilitates dropping of eruption 
                windows within analysis period. E.g., exclude_dates = [['2012-06-01','2012-08-01'],
                ['2015-01-01','2016-01-01']] will drop Jun-Aug 2012 and 2015-2016 from analysis.

            Classifier options:
            -------------------
            SVM - Support Vector Machine.
            KNN - k-Nearest Neighbors
            DT - Decision Tree
            RF - Random Forest
            NN - Neural Network
            NB - Naive Bayes
            LR - Logistic Regression
        """
        self.classifier = classifier
        self.exclude_dates = exclude_dates
        self.use_only_features = use_only_features
        self.n_jobs = n_jobs
        makedir(self.modeldir)

        # initialise training interval
        self.ti_train = self.ti_model if ti is None else datetimeify(ti)
        self.tf_train = self.tf_model if tf is None else datetimeify(tf)
        if self.ti_train - self.dtw < self.data.ti:
            self.ti_train = self.data.ti+self.dtw
        
        # check if any model training required
        if not retrain:
            run_models = False
            pref = type(get_classifier(self.classifier)[0]).__name__ 
            for i in range(Ncl):         
                if not os.path.isfile('{:s}/{:s}_{:04d}.pkl'.format(self.modeldir, pref, i)):
                    run_models = True
            if not run_models:
                return # not training required
        else:
            # delete old model files
            _ = [os.remove(fl) for fl in  glob('{:s}/*'.format(self.modeldir))]

        # get feature matrix and label vector
        fM, ys = self._load_data(self.ti_train, self.tf_train)

        # manually drop features (columns)
        fM = self._drop_features(fM, drop_features)

        # manually select features (columns)
        if len(self.use_only_features) != 0:
            use_only_features = [df for df in self.use_only_features if df in fM.columns]
            fM = fM[use_only_features]
            Nfts = len(use_only_features)+1

        # manually drop windows (rows)
        fM, ys = self._exclude_dates(fM, ys, exclude_dates)
        if ys.shape[0] != fM.shape[0]:
            raise ValueError("dimensions of feature matrix and label vector do not match")
        
        # select training subset
        inds = (ys.index>=self.ti_train)&(ys.index<self.tf_train)
        fM = fM.loc[inds]
        ys = ys['label'].loc[inds]

        # set up model training
        if self.n_jobs > 1:
            p = Pool(self.n_jobs)
            mapper = p.imap
        else:
            mapper = map
        f = partial(train_one_model, fM, ys, Nfts, self.modeldir, self.classifier, retrain, random_seed)

        # train models with glorious progress bar
        for i, _ in enumerate(mapper(f, range(Ncl))):
            cf = (i+1)/Ncl
            print(f'building models: [{"#"*round(50*cf)+"-"*round(50*(1-cf))}] {100.*cf:.2f}%\r', end='') 
        if self.n_jobs > 1:
            p.close()
            p.join()
        
        # free memory
        del fM
        gc.collect()
        self._collect_features()
    def forecast(self, ti=None, tf=None, recalculate=False, use_model=None, n_jobs=None):
        """ Use classifier models to forecast eruption likelihood.

            Parameters:
            -----------
            ti : str, datetime.datetime
                Beginning of forecast period (default is beginning of model analysis period).
            tf : str, datetime.datetime
                End of forecast period (default is end of model analysis period).
            recalculate : bool
                Flag indicating forecast should be recalculated, otherwise forecast will be
                loaded from previous save file (if it exists).
            use_model : None or str
                Optionally pass path to pre-trained model directory in 'models'.
            n_jobs : int
                Number of cores to use.

            Returns:
            --------
            consensus : pd.DataFrame
                The model consensus, indexed by window date.
        """
        self._use_model = use_model
        makedir(self.preddir)

        #
        if n_jobs is not None: self.n_jobs = n_jobs 
        self.ti_forecast = self.ti_model if ti is None else datetimeify(ti)
        self.tf_forecast = self.tf_model if tf is None else datetimeify(tf)
        if self.tf_forecast > self.data.tf:
            self.tf_forecast = self.data.tf
        if self.ti_forecast - self.dtw < self.data.ti:
            self.ti_forecast = self.data.ti+self.dtw

        loadFeatureMatrix = True

        model_path = self.modeldir + os.sep
        if use_model is not None:
            self._detect_model()
            model_path = self._use_model+os.sep
            
        model,classifier = get_classifier(self.classifier)

        # logic to determine which models need to be run and which to be 
        # read from disk
        pref = type(model).__name__
        fls = glob('{:s}/{:s}_*.pkl'.format(model_path, pref))
        load_predictions = []
        run_predictions = []
        ys = []
        if recalculate:
            run_predictions = [(rp, rp.replace(model_path, self.preddir+os.sep).replace('.pkl','.{:s}'.format(self.savefile_type))) for rp in fls]
            for rp in run_predictions:
                if os.path.isfile(rp[-1]):
                    os.remove(rp[-1])
            ti = self.ti_forecast
        else:
            tis = []
            for fl in fls:
                num = fl.split(os.sep)[-1].split('_')[-1].split('.')[0]
                flp = '{:s}/{:s}_{:s}.{:s}'.format(self.preddir, pref, num, self.savefile_type)
                if not os.path.isfile(flp):
                    run_predictions.append([fl, flp])
                    tis.append(self.ti_forecast)
                else:
                    y = load_dataframe(flp, index_col=0, parse_dates=['time'], infer_datetime_format=True)
                    if y.index[-1] < self.tf_forecast:
                        run_predictions.append([fl, flp])
                        tis.append(y.index[-1])
                    else:
                        ys.append(y)
            if len(tis)>0:
                ti = np.min(tis)

        # ys = []            
        # # load existing predictions
        # for fl in load_predictions:
        #     # y = pd.read_csv(fl, index_col=0, parse_dates=['time'], infer_datetime_format=True)
        #     y = load_dataframe(fl, index_col=0, parse_dates=['time'], infer_datetime_format=True)
        #     ys.append(y)

        # generate new predictions
        if len(run_predictions)>0:
            # run_predictions = [(rp, rp.replace(model_path, self.preddir+os.sep).replace('.pkl','.{:s}'.format(self.savefile_type))) for rp in run_predictions]

            # load feature matrix
            fM,_ = self._extract_features(ti, self.tf_forecast)

            # setup predictor
            if self.n_jobs > 1:
                p = Pool(self.n_jobs)
                mapper = p.imap
            else:
                mapper = map
            f = partial(predict_one_model, fM, model_path, pref)

            # train models with glorious progress bar
            #f(run_predictions[0])
            for i, y in enumerate(mapper(f, run_predictions)):
                cf = (i+1)/len(run_predictions)
                print(f'forecasting: [{"#"*round(50*cf)+"-"*round(50*(1-cf))}] {100.*cf:.2f}%\r', end='') 
                ys.append(y)
            
            if self.n_jobs > 1:
                p.close()
                p.join()
        
        # condense data frames and write output
        ys = pd.concat(ys, axis=1, sort=False)
        consensus = np.mean([ys[col].values for col in ys.columns if 'pred' in col], axis=0)
        forecast = pd.DataFrame(consensus, columns=['consensus'], index=ys.index)
        # forecast.to_csv('{:s}/consensus.csv'.format(self.preddir), index=True, index_label='time')
        save_dataframe(forecast, '{:s}/consensus.{:s}'.format(self.preddir,self.savefile_type), index=True, index_label='time')
        
        # memory management
        if len(run_predictions)>0:
            del fM
            gc.collect()

        return forecast
    def hires_forecast(self, ti, tf, recalculate=True, save=None, root=None, nztimezone=False, 
        n_jobs=None, threshold=0.8, alt_rsam=None, xlim=None):
        """ Construct forecast at resolution of data.

            Parameters:
            -----------
            ti : str, datetime.datetime
                Beginning of forecast period.
            tf : str, datetime.datetime
                End of forecast period.
            recalculate : bool
                Flag indicating forecast should be recalculated, otherwise forecast will be
                loaded from previous save file (if it exists).
            save : None or str
                If given, plot forecast and save to filename.
            root : None or str
                Naming convention for saving feature matrix.
            nztimezone : bool
                Flag to plot forecast using NZ time zone instead of UTC.            
            n_jobs : int
                CPUs to use when forecasting in parallel.
            Notes:
            ------
            Requires model to have already been trained.
        """
        # error checking
        try:
            _ = self.ti_train
        except AttributeError:
            raise ValueError('Train model before constructing hires forecast.')
        
        if save == '':
            save = '{:s}/hires_forecast.png'.format(self.plotdir)
            makedir(self.plotdir)
        
        if n_jobs is not None: self.n_jobs = n_jobs
 
        # calculate hires feature matrix
        if root is None:
            root = self.root+'_hires'
        _fm = ForecastModel(self.volcano, self.window, 1., self.look_forward, ti, tf, self.data_streams, root=root, savefile_type=self.savefile_type)
        _fm.compute_only_features = list(set([ft.split('__')[1] for ft in self._collect_features()[0]]))
        _fm._extract_features(ti, tf)

        # predict on hires features
        ys = _fm.forecast(ti, tf, recalculate, use_model=self.modeldir, n_jobs=n_jobs)
        
        if save is not None:
            self._plot_hires_forecast(ys, save, threshold, nztimezone=nztimezone, alt_rsam=alt_rsam, xlim=xlim)

        return ys
    # plotting methods
    def plot_forecast(self, ys, threshold=0.75, save=None, xlim=['2019-12-01','2020-02-01']):
        """ Plot model forecast.

            Parameters:
            -----------
            ys : pandas.DataFrame
                Model forecast returned by ForecastModel.forecast.
            threshold : float
                Threshold consensus to declare alert.
            save : str
                File name to save figure.
            local_time : bool
                If True, switches plotting to local time (default is UTC).
        """
        makedir(self.plotdir)
        if save is None:
            save = '{:s}/forecast.png'.format(self.plotdir)
        # set up figures and axes
        f = plt.figure(figsize=(24,15))
        N = 10
        dy1,dy2 = 0.05, 0.05
        dy3 = (1.-dy1-(N//2)*dy2)/(N//2)
        dx1,dx2 = 0.37,0.04
        axs = [plt.axes([0.10+(1-i//(N/2))*(dx1+dx2), dy1+(i%(N/2))*(dy2+dy3), dx1, dy3]) for i in range(N)][::-1]
        
        for i,ax in enumerate(axs[:-1]):
            ti,tf = [datetime.strptime('{:d}-01-01 00:00:00'.format(2011+i), '%Y-%m-%d %H:%M:%S'),
                datetime.strptime('{:d}-01-01 00:00:00'.format(2012+i), '%Y-%m-%d %H:%M:%S')]
            ax.set_xlim([ti,tf])
            ax.text(0.01,0.95,'{:4d}'.format(2011+i), transform=ax.transAxes, va='top', ha='left', size=16)
            
        ti,tf = [datetimeify(x) for x in xlim]
        axs[-1].set_xlim([ti, tf])
        
        # model forecast is generated for the END of each data window
        t = ys.index

        # average individual model responses
        ys = np.mean(np.array([ys[col] for col in ys.columns]), axis=0)
        for i,ax in enumerate(axs):

            ax.set_ylim([-0.05, 1.05])
            ax.set_yticks([0,0.25,0.5, 0.75, 1.0])
            if i//(N/2) == 0:
                ax.set_ylabel('alert level')
            else:
                ax.set_yticklabels([])

            # shade training data
            ax.fill_between([self.ti_train, self.tf_train],[-0.05,-0.05],[1.05,1.05], color=[0.85,1,0.85], zorder=1, label='training data')            
            for exclude_date_range in self.exclude_dates:
                t0,t1 = [datetimeify(dt) for dt in exclude_date_range]
                ax.fill_between([t0, t1],[-0.05,-0.05],[1.05,1.05], color=[1,1,1], zorder=2)            
            
            # consensus threshold
            ax.axhline(threshold, color='k', linestyle=':', label='alert threshold', zorder=4)

            # modelled alert
            ax.plot(t, ys, 'c-', label='modelled alert', zorder=4)

            # eruptions
            for te in self.data.tes:
                ax.axvline(te, color='r', linestyle='--', zorder=5)
            ax.axvline(te, color='r', linestyle='--', label='eruption')

            ax_ = ax.twinx()
            for ds,c in zip(self.data_streams, ['k','b','g']):
                ax_.plot(self.data.df.index, self.data.df[ds], c+'-')
                ax.plot([],[],c+'-',label=ds)
                ax_.set_xlim(ax.get_xlim())

        for tii,yi in zip(t, ys):
            if yi > threshold:
                i = (tii.year-2011)
                axs[i].fill_between([tii, tii+self.dtf], [0,0], [1,1], color='y', zorder=3)
                j = (tii+self.dtf).year - 2011
                if j != i:
                    axs[j].fill_between([tii, tii+self.dtf], [0,0], [1,1], color='y', zorder=3)
                
                if tii > ti:
                    axs[-1].fill_between([tii, tii+self.dtf], [0,0], [1,1], color='y', zorder=3)
                
        for ax in axs:
            ax.fill_between([], [], [], color='y', label='eruption forecast')
        axs[-1].legend()
        
        plt.savefig(save, dpi=400)
        plt.close(f)
    def _plot_hires_forecast(self, ys, save, threshold=0.75, nztimezone=False, alt_rsam=None, xlim=None):
        """ Plot model hires version of model forecast (single axes).

            Parameters:
            -----------
            ys : pandas.DataFrame
                Model forecast returned by ForecastModel.forecast.
            threshold : float
                Threshold consensus to declare alert.
            save : str
                File name to save figure.
        """
        makedir(self.plotdir)
        # set up figures and axes
        f = plt.figure(figsize=(8,8))
        ax1 = plt.axes([0.1, 0.55, 0.8, 0.4])
        ax2 = plt.axes([0.1, 0.08, 0.8, 0.4])
        t = pd.to_datetime(ys.index.values)
        rsam = self.data.get_data(t[0], t[-1])['rsam']
        trsam = rsam.index
        if alt_rsam is not None:
            alt_trsam = alt_rsam.index
        if nztimezone:
            t = to_nztimezone(t)
            trsam = to_nztimezone(trsam)
            ax2.set_xlabel('Local time')
            if alt_rsam is not None:
                alt_trsam = to_nztimezone(alt_trsam)
        else:
            ax2.set_xlabel('UTC')
        y = np.mean(np.array([ys[col] for col in ys.columns]), axis=0)
        
        ts = [t[-1], trsam[-1]]
        if alt_rsam is not None: ts.append(alt_trsam[-1])
        tmax = np.max(ts)
        ax2.set_xlim([tmax-timedelta(days=7), tmax])
        ax1.set_xlim([t[0], tmax])
        ax1.set_title('Whakaari Eruption Forecast')
        for ax in [ax1,ax2]:
            ax.set_ylim([-0.05, 1.05])
            ax.set_yticks([0,0.25,0.50,0.75,1.00])
            ax.set_ylabel('ensemble mean')
        
            # consensus threshold
            ax.axhline(threshold, color='k', linestyle=':', label='alert threshold', zorder=4)

            # modelled alert
            ax.plot(t, y, 'c-', label='ensemble mean', zorder=4, lw=0.75)
            ax_ = ax.twinx()
            ax_.set_ylabel('RSAM [$\mu$m s$^{-1}$]')
            ax_.set_ylim([0,5])
            ax_.set_xlim(ax.get_xlim())
            if alt_rsam is not None:
                ax_.plot(alt_trsam, alt_rsam.values*1.e-3, '-', color=[0.5,0.5,0.5], lw=0.75)
            ax_.plot(trsam, rsam.values*1.e-3, 'k-', lw=0.75)

            for tii,yi in zip(t, y):
                if yi > threshold:
                    ax.fill_between([tii, tii+self.dtf], [0,0], [100,100], color='y', zorder=3)
                    
            ax.fill_between([], [], [], color='y', label='eruption forecast')
            ax.plot([],[],'k-', lw=0.75, label='RSAM')
            if alt_rsam is not None:
                ax.plot([],[],'-', color=[0.5,0.5,0.5], lw=0.75, label='RSAM (WSRZ-scaled)')
        ax1.legend(loc=1, ncol=2)
        if xlim is not None: 
            ax2.set_xlim(xlim)
            tmax = xlim[-1] 
        tf = tmax 
        t0 = tf.replace(hour=0, minute=0, second=0)
        xts = [t0 - timedelta(days=i) for i in range(7)][::-1]
        lxts = [xt.strftime('%d %b') for xt in xts]
        ax2.set_xticks(xts)
        ax2.set_xticklabels(lxts)
        tfi  = self.data.tf
        if nztimezone:
            tfi = to_nztimezone([tfi])[0]
        ax2.text(0.025, 0.95, 'model last updated {:s}'.format(tfi.strftime('%H:%M, %d %b %Y')), size = 12, ha = 'left', 
            va = 'top', transform=ax2.transAxes)
        
        t0 = datetimeify('2020-01-01')
        xts = [t0.replace(month=i) for i in range(1, tf.month+1)]
        lxts = [xt.strftime('%d %b') for xt in xts]
        ax1.set_xticks(xts)
        ax1.set_xticklabels(lxts)
        ax1.text(0.025, 0.95, t0.strftime('%Y'), size = 12, ha = 'left', 
            va = 'top', transform=ax1.transAxes)

        plt.savefig(save, dpi=400)
        plt.close(f)
    def plot_accuracy(self, ys, save=None):
        """ Plot performance metrics for model.

            Parameters:
            -----------
            ys : pandas.DataFrame
                Model forecast returned by ForecastModel.forecast.
            save : str
                File name to save figure.

        """
        makedir(self.plotdir)
        if save is None:
            save = '{:s}/accuracy.png'.format(self.plotdir)
        
        # time series
        t = ys.index
        
        self._ys = np.array([self.data._is_eruption_in(days=self.look_forward, from_time=ti) for ti in pd.to_datetime(ys.index)])
        thresholds = np.linspace(0.0,1.0,101)
        ialert = self.look_forward/((1-self.overlap)*self.window)
        dti = timedelta(days=(1-self.overlap)*self.window)
        FPs = [int(1e8)]; FNs = [0]; alert_duration = [1.]; TPs = [len(self.data.tes)]; TNs = [1]
        MCC = [0.]
        for j,threshold in enumerate(thresholds[1:]):
            
            FP, FN, TP, TN, dur, mcc = self._model_alerts(t, ys['consensus'], threshold, ialert, dti)

            FPs.append(FP)
            FNs.append(FN)
            TPs.append(TP)
            TNs.append(TN)
            alert_duration.append(dur)
            MCC.append(mcc)

        FPs = np.array(FPs); FNs = np.array(FNs); TPs = np.array(TPs); TNs = np.array(TNs); MCC = np.array(MCC)
        with open(save.replace('png','txt'),'w') as fp:
            fp.write('threshold,FP,FN,TP,TN,alert_fraction,MCC\n')
            _ = [fp.write('{:4.3f},{:d},{:d},{:d},{:d},{:4.3f},{:4.3f}\n'.format(*vals)) for vals 
                    in zip(thresholds,FPs,FNs,TPs,TNs,alert_duration,MCC)]
        MCC = (TPs*TNs-FPs*FNs)/np.sqrt((TPs+FPs)*(TPs+FNs)*(TNs+FPs)*(TNs+FNs))
        F1 = 2*TPs/(2*TPs+FPs+FNs)
        accuracy = (TPs+TNs)/(TPs+TNs+FPs+FNs)
        f = plt.figure(figsize=(10,10))
        ax1 = plt.axes([0.1, 0.55, 0.35, 0.35])
        ax2 = plt.axes([0.6, 0.55, 0.35, 0.35])
        ax3 = plt.axes([0.1, 0.08, 0.35, 0.35])
        ax4 = plt.axes([0.6, 0.08, 0.35, 0.35])
        ax1.plot(thresholds, FPs/(FPs+TPs), 'k-', label='false alert rate')
        ax1.set_ylabel('false alert rate / FP/(FP+TP)')
        ax1b = ax1.twinx()
        ax1b.plot(thresholds, FNs, 'b-')
        ax1.plot([],[],'b-',label='missed eruptions')
        ax1b.set_ylabel('missed eruptions / FN')
        ax1b.set_ylim([-.1, ax1b.get_ylim()[-1]])
        ax1.legend()
        ax2.plot(FPs/(FPs+TNs), TPs/(TPs+FNs), 'k-')
        ax2.set_title('ROC')
        ax2.set_ylabel('true positive rate')
        ax2.set_xlabel('false positive rate')
        ax2.plot([0,1],[0,1],'--', color=[0.5,0.5,0.5])
        ax3.plot(thresholds, alert_duration, 'k-')
        ax3.set_ylabel('alert duration')
        ax4.plot(thresholds, MCC, 'k-', label='MCC')
        ax4.plot(thresholds, F1, 'b-', label='F1')
        ax4.plot(thresholds, accuracy, 'g-', label='accuracy')
        ax4.set_ylabel('score')
        ax4.set_ylim([0,1])
        ax4.legend()
        for ax in [ax1,ax3,ax4]:
            ax.set_xlabel('alert threshold')
            ax.set_xlim([0.5,1])
        
        plt.savefig(save, dpi=300)
        plt.close(f)
    def plot_features(self, N=10, save=None):
        """ Plot frequency of extracted features by most significant.

            Parameters:
            -----------
            N : int
                Number of features to plot, ordered by most frequent amongst all classifiers.
            save : str
                File name to save figure.
        """
        makedir(self.plotdir)
        if save is None:
            save = '{:s}/features.png'.format(self.plotdir)
        
        feats = []
        fls = glob('{:s}/*.fts'.format(self.modeldir))
        for i,fl in enumerate(fls):
            with open(fl) as fp:
                lns = fp.readlines()
            feats += [' '.join(ln.rstrip().split()[1:]) for ln in lns]               

        f = plt.figure(figsize=(8, 16))
        ax = plt.axes([0.05, 0.05, 0.4, 0.9])
        height = 0.8
        # sort features in descending order by frequency of appearance
        labels = list(set(feats) - set(['']))
        freqs = [feats.count(label) for label in labels]
        labels = [label for _,label in sorted(zip(freqs,labels))][::-1]
        freqs = sorted(freqs)[::-1]
        fts = copy(labels)
        
        N = np.min([N, len(freqs)])
        labels = ['\n'.join(wrap(' '.join(l.split('_')), 40)) for l in labels ][:N]
        freqs = freqs[:N]
        inds = range(len(freqs))
        ax.barh(inds, np.array(freqs)/len(fls), height=height, color='#90EE90')
        ax2 = ax.twiny()
        ax.xaxis.grid()
        ax2.set_xlim(ax.get_xlim())
        xm = np.mean(ax.get_xlim())
        for ind,label in zip(inds,labels):
            ax.text(xm, ind, label, ha='center', va='center')
        plt.yticks([])
        for axi in [ax,ax2]: axi.set_ylim([inds[0]-0.5, inds[-1]+0.5])
        ax.invert_yaxis()
        
        # righthand feature plots
        axs = []
        dy1 = 0.9*height/N
        dy2 = 0.9*(1-height)/N
        for i in range(N):
            axi = plt.axes([0.5, 0.05+i*(dy1+dy2)+dy2/2, 0.4, dy1])
            axi.set_yticks([])
            axs.append(axi)
        axs = axs[::-1]

        fM,ys = self._extract_features(self.ti_forecast, self.tf_forecast)
        inds0 = np.where(ys['label']<1)
        
        inds = []
        for te in self.data.tes:
            inds.append(np.where((ys['label']>0)&(abs((te-ys.index).total_seconds()/(3600*24))<5.)))
        cols = ['b','g','r','m','c']
        
        N0 = int(np.sqrt(len(inds0[0])/2.))

        for axi, ft in zip(axs,fts):
            ft0 = np.log10(fM[ft].iloc[inds0]).replace([np.inf, -np.inf], np.nan).dropna()
            ft0_min = np.mean(ft0)-3*np.std(ft0)
            ft0 = ft0[ft0>ft0_min]
            y,e = np.histogram(ft0, N0)
            x = 0.5*(e[:-1]+e[1:])
            axi.fill_between(x, [0,]*len(x), y, color='#add8e6', label='all windows')
            ylim = axi.get_ylim()
            axi.set_ylim(ylim)
            ym = np.mean(ylim)
            dy = (ylim[1]-ylim[0])/(len(inds)+1)
            for i, ind, col in zip(range(-2,3), inds, cols):
                ft1 = np.log10(fM[ft].iloc[ind]).replace([np.inf, -np.inf], np.nan).dropna()
                te = self.data.tes[i+2]
                lbl = te.strftime('%b')+' '+('{:d}'.format(te.year))
                axi.scatter(ft1, [ym+dy*i,]*len(ft1), np.arange(1,len(ft1)+1)*6, col, marker='x', label=lbl)

        axs[0].legend(prop={'size':6})

        plt.savefig(save, dpi=300)
        plt.close(f)
    def plot_feature_correlation(self, N=20, save=None):
        """ Corner plot of feature correlation.

            Parameters:
            -----------
            N : int
                Number of features to plot, ordered by most frequent amongst all classifiers
            save : str
                File name to save figure.
        """
        makedir(self.plotdir)
        if save is None:
            save = '/feature_correlation.png'.format(self.plotdir)
        
        # compile feature frequencies
        feats = []
        fls = glob('{:s}/*.fts'.format(self.modeldir))
        for i,fl in enumerate(fls):
            with open(fl) as fp:
                lns = fp.readlines()
            feats += [' '.join(ln.rstrip().split()[1:]) for ln in lns]               
        labels = list(set(feats) - set(['']))
        freqs = [feats.count(label) for label in labels]

        labels = [label for _,label in sorted(zip(freqs,labels))][::-1]
        freqs = sorted(freqs)[::-1]
        fts = copy(labels)
        
        N = np.min([N, len(freqs)])
        labels = ['\n'.join(wrap(' '.join(l.split('_')), 30)) for l in labels ][:N]
        
        f = plt.figure(figsize=(8, 8))

        fM,ys = self._extract_features(self.ti_forecast, self.tf_forecast)
        filt_df = np.log10(fM[fts[:N]]).replace([np.inf, -np.inf], np.nan).dropna()
        low, high = [0.005, 0.995]
        quant_df = filt_df.quantile([low, high])
        filt_df.index = range(filt_df.shape[0])
        filt_df = filt_df.apply(lambda x: x[(x>quant_df.loc[low,x.name]) & (x < quant_df.loc[high,x.name])], axis=0)
        fM[fts[:N]]

        inds = []
        for te in self.data.tes:
            inds.append(np.where((ys['label']>0)&(abs((te-ys.index).total_seconds()/(3600*24))<5.)))
        cols = ['b','g','r','m','c']
        truths = []
        for i, ind, col in zip(range(-2,3), inds, cols):
            truths.append(np.log10(fM[fts[:N]].iloc[ind]).replace([np.inf, -np.inf], np.nan).dropna())
            #te = self.data.tes[i+2]
            
        fig = corner(filt_df.dropna(), labels=labels[:N])
        axes = np.array(fig.axes).reshape((N, N))
        for yi in range(N):
            for xi in range(yi):
                ax = axes[yi,xi]
                for col,truth in zip(cols,truths):
                    ax.scatter(truth[fts[xi]].values, truth[fts[yi]].values, np.arange(1,5)*6, col)
                            
        plt.savefig(save,dpi=300)
        plt.close(fig)

def save_dataframe(df, fl, index=True, index_label=None):
    if fl.endswith('.csv'):
        df.to_csv(fl, index=index, index_label=index_label)
    elif fl.endswith('.pkl'):
        fp = open(fl, 'wb')
        pickle.dump(df,fp)
    elif fl.endswith('.hdf'):
        df.to_hdf(fl, 'test', format='fixed', mode='w')
    else:
        raise ValueError('only csv, hdf and pkl file formats supported')

def load_dataframe(fl, index_col=None, parse_dates=False, usecols=None, infer_datetime_format=False, 
    nrows=None, header='infer', skiprows=None):
    if fl.endswith('.csv'):
        df = pd.read_csv(fl, index_col=index_col, parse_dates=parse_dates, usecols=usecols, infer_datetime_format=infer_datetime_format,
            nrows=nrows, header=header, skiprows=skiprows)
    elif fl.endswith('.pkl'):
        fp = open(fl, 'rb')
        df = pickle.load(fp)
    elif fl.endswith('.hdf'):
        df = pd.read_hdf(fl, 'test')
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

def get_classifier(classifier):
    """ Return scikit-learn ML classifiers and search grids for input strings.

        Parameters:
        -----------
        classifier : str
            String designating which classifier to return.

        Returns:
        --------
        model : 
            Scikit-learn classifier object.
        grid : dict
            Scikit-learn hyperparameter grid dictionarie.

        Classifier options:
        -------------------
        SVM - Support Vector Machine.
        KNN - k-Nearest Neighbors
        DT - Decision Tree
        RF - Random Forest
        NN - Neural Network
        NB - Naive Bayes
        LR - Logistic Regression
    """
    if classifier == 'SVM':         # support vector machine
        model = SVC(class_weight='balanced')
        grid = {'C': [0.001,0.01,0.1,1,10], 'kernel': ['poly','rbf','sigmoid'],
            'degree': [2,3,4,5],'decision_function_shape':['ovo','ovr']}
    elif classifier == "KNN":        # k nearest neighbour
        model = KNeighborsClassifier()
        grid = {'n_neighbors': [3,6,12,24], 'weights': ['uniform','distance'],
            'p': [1,2,3]}
    elif classifier == "DT":        # decision tree
        model = DecisionTreeClassifier(class_weight='balanced')
        grid = {'max_depth': [3,5,7], 'criterion': ['gini','entropy'],
            'max_features': ['auto','sqrt','log2',None]}
    elif classifier == "RF":        # random forest
        model = RandomForestClassifier(class_weight='balanced')
        grid = {'n_estimators': [10,30,100], 'max_depth': [3,5,7], 'criterion': ['gini','entropy'],
            'max_features': ['auto','sqrt','log2',None]}
    elif classifier == "NN":        # neural network
        model = MLPClassifier(alpha=1, max_iter=1000)
        grid = {'activation': ['identity','logistic','tanh','relu'],
            'hidden_layer_sizes':[10,100]}
    elif classifier == "NB":        # naive bayes
        model = GaussianNB()
        grid = {'var_smoothing': [1.e-9]}
    elif classifier == "LR":        # logistic regression
        model = LogisticRegression(class_weight='balanced')
        grid = {'penalty': ['l2','l1','elasticnet'], 'C': [0.001,0.01,0.1,1,10]}
    else:
        raise ValueError("classifier '{:s}' not recognised".format(classifier))
    
    return model, grid

def train_one_model(fM, ys, Nfts, modeldir, classifier, retrain, random_seed, random_state):
    # undersample data
    rus = RandomUnderSampler(0.75, random_state=random_state+random_seed)
    fMt,yst = rus.fit_resample(fM,ys)
    yst = pd.Series(yst, index=range(len(yst)))
    fMt.index = yst.index

    # find significant features
    select = FeatureSelector(n_jobs=0, ml_task='classification')
    select.fit_transform(fMt,yst)
    fts = select.features[:Nfts]
    pvs = select.p_values[:Nfts]
    fMt = fMt[fts]
    with open('{:s}/{:04d}.fts'.format(modeldir, random_state),'w') as fp:
        for f,pv in zip(fts,pvs): 
            fp.write('{:4.3e} {:s}\n'.format(pv, f))

    # get sklearn training objects
    ss = ShuffleSplit(n_splits=5, test_size=0.25, random_state=random_state+random_seed)
    model, grid = get_classifier(classifier)            
        
    # check if model has already been trained
    pref = type(model).__name__
    fl = '{:s}/{:s}_{:04d}.pkl'.format(modeldir, pref, random_state)
    if os.path.isfile(fl) and not retrain:
        return
    
    # train and save classifier
    model_cv = GridSearchCV(model, grid, cv=ss, scoring="balanced_accuracy",error_score=np.nan)
    model_cv.fit(fMt,yst)
    _ = joblib.dump(model_cv.best_estimator_, fl, compress=3)

def predict_one_model(fM, model_path, pref, flp):
    flp,fl = flp

    if os.path.isfile(fl):
        ypdf0 = load_dataframe(fl, index_col='time', infer_datetime_format=True, parse_dates=['time'])

    num = flp.split(os.sep)[-1].split('.')[0].split('_')[-1]
    model = joblib.load(flp)
    with open(model_path+'{:s}.fts'.format(num)) as fp:
        lns = fp.readlines()
    fts = [' '.join(ln.rstrip().split()[1:]) for ln in lns]            
    
    if not os.path.isfile(fl):
        # simulate predicton period
        yp = model.predict(fM[fts])
        # save prediction
        ypdf = pd.DataFrame(yp, columns=['pred{:s}'.format(num)], index=fM.index)
    else:
        fM2 = fM.loc[fM.index>ypdf0.index[-1], fts]
        if fM2.shape[0] == 0:
            ypdf = ypdf0
        else:
            yp = model.predict(fM2)
            ypdf = pd.DataFrame(yp, columns=['pred{:s}'.format(num)], index=fM2.index)
            ypdf = pd.concat([ypdf0, ypdf])

    # ypdf.to_csv(fl, index=True, index_label='time')
    save_dataframe(ypdf, fl, index=True, index_label='time')
    return ypdf

def datetimeify(t, fmt=None):
    """ Return datetime object corresponding to input string.

        Parameters:
        -----------
        t : str, datetime.datetime
            Date string to convert to datetime object.

        Returns:
        --------
        datetime : datetime.datetime
            Datetime object corresponding to input string.

        Notes:
        ------
        This function tries several datetime string formats, and raises a ValueError if none work.
    """
    if type(t) in [datetime, Timestamp]:
        return t
    if fmt is None:
        fmts = FMTS
    else:
        fmts = [fmt,]
    for fmt in fmts:
        try:
            return datetime.strptime(t, fmt)
        except ValueError:
            pass
    raise ValueError("time data '{:s}' not a recognized format".format(t))

def detect_timeformat(t):
    for fmt in FMTS:
        try:
            _ = datetime.strptime(t, fmt)
            return fmt
        except ValueError:
            pass
    raise ValueError("time data '{:s}' not a recognized format".format(t))

def to_nztimezone(t):
    """ Routine to convert UTC to NZ time zone.
    """
    from dateutil import tz
    utctz = tz.gettz('UTC')
    nztz = tz.gettz('Pacific/Auckland')
    return [ti.replace(tzinfo=utctz).astimezone(nztz) for ti in pd.to_datetime(t)]
