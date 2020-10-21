import os, sys
sys.path.insert(0, os.path.abspath('..'))
from whakaari import TremorData, ForecastModel, datetimeify
from datetime import timedelta, datetime

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

'''
def forecast_dec2019():
    #forecast model for Dec 2019 eruption
    
    # constants
    month = timedelta(days=365.25/12)
    day = timedelta(days=1)
    td = TremorData()
        
    # construct model object
    data_streams = ['rsam','mf','hf','dsar']
    fm = ForecastModel(ti='2011-01-01', tf='2020-01-01', window=2., overlap=0.75, 
        look_forward=2., data_streams=data_streams)
    
    # columns to manually drop from feature matrix because they are highly correlated to other 
    # linear regressors
    drop_features = ['linear_trend_timewise','agg_linear_trend']
    
    # set the available CPUs higher or lower as appropriate
    n_jobs = 6

    # train the model, excluding 2019 eruption
    # note: building the feature matrix may take several hours, but only has to be done once 
    # and will intermittantly save progress in ../features/
    # trained scikit-learn models will be saved to ../models/*root*/
    te = td.tes[-1]
    fm.train(ti='2011-01-01', tf='2020-01-01', drop_features=drop_features, retrain=True, 
        exclude_dates=[[te-month,te+month],], n_jobs=n_jobs)      

    # run forecast from 2011 to 2020
    # model predictions will be saved to ../predictions/*root*/ 
    ys = fm.forecast(ti='2011-01-01', tf='2020-01-01', recalculate=True, n_jobs=n_jobs)    

    # plot forecast and quality metrics
    # plots will be saved to ../plots/*root*/
    fm.plot_forecast(ys, threshold=0.8, xlim = [te-month/4., te+month/15.], 
        save=r'{:s}/forecast.png'.format(fm.plotdir))
    fm.plot_accuracy(ys, save=r'{:s}/accuracy.png'.format(fm.plotdir))

    # construct a high resolution forecast (10 min updates) around the Dec 2019 eruption
    # note: building the feature matrix might take a while
    fm.hires_forecast(ti=te-fm.dtw-fm.dtf, tf=te+month/30, recalculate=True, 
        save=r'{:s}/forecast_hires.png'.format(fm.plotdir), n_jobs=n_jobs)

forecast_dec2019()


def forecast_test():

    # constants
    month = timedelta(days=365.25/12)
        
    # set up model
    data_streams = ['rsam','mf','hf','dsar']
    fm = ForecastModel(ti='2013-08-02', tf='2013-08-18', window=2., overlap=0.75,
        look_forward=2., data_streams=data_streams, root='test')
    print(fm.plotdir)
    # set the available CPUs higher or lower as appropriate
    n_jobs = 6
    
    # train the model
    #drop_features = ['linear_trend_timewise','agg_linear_trend']
    fm.train(ti='2013-08-02', tf='2013-08-18', retrain=True, n_jobs=n_jobs)
    # plot a forecast for a future eruption
    te = fm.data.tes[1]
    fm.hires_forecast(ti=datetimeify('2013-08-15'), tf=datetimeify('2013-08-21'), save = None,
                      recalculate=True, n_jobs=n_jobs)
    #fm.hires_forecast(ti='2013-08-15', tf='2013-08-21', save=None,
    #                 recalculate=True, n_jobs=n_jobs)


'''
def forecast_now():
    #forecast model for present day 
    
    # constants
    month = timedelta(days=365.25/12)
    day = timedelta(days=1)
    print("Now updating")
    # pull the latest data from GeoNet
    td = TremorData()
    td.update(ti = '2019-11-24', tf = '2019-11-26')     # ti: start time    tf: end time
    print("Done with updating.")
    # model from 2011 to present day (td.tf)
    data_streams = ['rsam','mf','hf','dsar']
    '''
    fm = ForecastModel(ti='2011-01-30', tf='2011-02-01', window=2, overlap=0.75,
        look_forward=2, data_streams=data_streams, root='online_forecaster')
    
    # set the available CPUs higher or lower as appropriate
    n_jobs = 6
    print("Done setting up model - now training")
    
    # The online forecaster is trained using all eruptions in the dataset. It only
    # needs to be trained once, or again after a new eruption.
    # (Hint: feature matrices can be copied from other models to avoid long recalculations
    # providing they have the same window length and data streams. Copy and rename 
    # to *root*_features.csv)
    drop_features = ['linear_trend_timewise','agg_linear_trend']
    fm.train(ti='2011-02-01', tf='2011-02-03', drop_features=drop_features,
        retrain=False, n_jobs=n_jobs)
    print("Done training - now forecasting")
    # forecast the last 7 days at high resolution
    fm.hires_forecast(ti=fm.data.tf - 7*day, tf=fm.data.tf, recalculate=True, 
        save='current_forecast.png', nztimezone=True, n_jobs=n_jobs)
'''
if __name__ == "__main__":
    #forecast_dec2019()
    #forecast_test()
    forecast_now()