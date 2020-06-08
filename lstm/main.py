from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
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
from keras.callbacks import LambdaCallback

from forecast import forecast
from utils.parse_data import parse_data
from fit_model import fit_model

SPLIT_PERCENT = 0.80
data_set = pd.read_csv('new-york-city-zips.csv')
data_frames = parse_data(data_set)

def split_data_set(data_set, split_percent):
    split = int(len(data_set) * split_percent)
    training_set = data_set[:split]
    test_set = data_set[split: ]
    return training_set, test_set

def plotData(df, actual, forecasts):
    plt.figure(figsize = (16, 9))
    plt.plot(forecasts, color = 'green', label = 'Predicted Prices')
    plt.plot(actual, color = 'black', label = 'Real Prices')
    plt.title('1 year forecast')
    plt.xticks(range(0, df.shape[0], 1), df['Date'].loc[::1], rotation=45)
    plt.xlabel('Month')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

def get_forecasts():
    for zipcode in data_frames:
        forecasts = {}
        data_frame = data_frames[zipcode]['Price'].values
        training_set, test_set = split_data_set(data_frame, SPLIT_PERCENT)
        model, time_steps = fit_model(training_set, test_set)
        forecast = forecast(training_set, model, time_steps, len(test_set))
        info = {'forecasts': forecast, 'actual': test_set}
        forecasts[zipcode] = info
        return forecasts
