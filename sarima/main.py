from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from math import sqrt

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from datetime import datetime

from keras import optimizers
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Activation
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.layers import Dense
from keras.callbacks import CSVLogger
from keras.callbacks import Callback

import plotly.offline as pyoff
import warnings
import itertools
import statsmodels.api as sm
from utils.parse_data import parse_data

SPLIT_PERCENT = 0.80
data_set = pd.read_csv('new-york-city-zips.csv')
data_frames = parse_data(data_set)

def split_data(data, split_point):
    return data[:-split_point], data[-split_point:]

def rmse(actual, forecasts):
    return sqrt(mean_squared_error(actual, forecasts))

def fit_sarima_model(training_set, test_set):
    p = q = D = range(3)
    d = [12, 24, 36]
    orders = list(itertools.product(p, q, d))
    seasonal_orders = list(itertools.product(p, q, D))
    current_error = -1
    best_model = None
    
    #grid search for the best combination of parameters (p, q, d) (P, Q, D)
    for order in orders:
        for seasonal_order in seasonal_orders:
            model = sm.tsa.statespace.SARIMAX(training_set, order=order, seasonal_order=seasonal_order, 
                                     enforce_stationarity=False,
                                     enforce_invertibility=False)
            results = model.fit()
            predicted = results.get_forecast(len(test_set)).predicted_mean
            error = rmse(test_set, predicted)
            if current_error < 0 or error < current_error:
                best_model = results
                current_error = error
    return best_model

def get_forecast(training_set, test_set, future_steps):
    model = fit_sarima_model(training_set, test_set)
    predictions = model.get_forecast(future_steps).predicted_mean
    return predictions

def get_forecasts():
    for zipcode in data_frames:
        forecasts = {}
        data_frame = data_frames[zipcode]['Price'].values
        training_set, test_set = split_data(data_frame, SPLIT_PERCENT)
        forecast = get_forecast(training_set, test_set, len(test_set))
        info = {'forecasts': forecast, 'actual': test_set}
        forecasts[zipcode] = info
        return forecasts