from forecast import forecast
from hyperopt import Trials, STATUS_OK, tpe, fmin, hp
import numpy as np
import pandas as pd
import os
import sys
from math import sqrt
import time
import pandas as pd
from tqdm._tqdm_notebook import tqdm_notebook as tqdm
import tensorflow as tf
import pickle
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Embedding
from keras.layers import LSTM
from tensorflow import keras
from keras.callbacks import Callback
from keras import optimizers
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import CSVLogger
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler, normalize
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from keras.preprocessing.sequence import TimeseriesGenerator

import logging
import itertools as it

search_space = {

    'batch_size': hp.choice('batch_sizes', [1, 6, 12]),
    'time_steps': hp.choice('time_steps', [12 , 24, 36, 48, 60]),
    'lstm1_nodes':hp.choice('layer1_units', [100, 150, 200, 250]),
    'lstm1_dropouts':hp.uniform('layer1_dropouts', 0, 0.2, 0.5),
    'lstm_layers': hp.choice('num_layers',[
        {
            'layers': 1, 
        },
        {
            'layers': 2,
            'lstm2_nodes':hp.choice('layer2_units', [20, 30, 40, 50]),
            'lstm2_dropouts':hp.uniform('layer2_units', 0, 0.2, 0.5)  
        }
    ]),
    'dense_layers': hp.choice('num_dense_layers',[
        {
            'layers': 1
        },
        {
            'layers': 2,
            'dense2_nodes':hp.choice('dense2_units', [10, 20, 30, 40])
        }
    ]),
    "lr": hp.uniform('lr', 0, 1),
    "epochs": hp.choice('epochs', [50, 100, 150, 200, 250, 300, ]),
    "optimizer": hp.choice('optimizers',["adam", "rms"])
}

training_set = None
test_set = None
scaler = MinMaxScaler()

def rmse(actual, forecasts):
    return sqrt(mean_squared_error(actual, forecasts))

def fit_model(train, test):
    global test_set
    global training_set
    training_set = scaler.fit_transform(training_set.reshape(-1, 1))
    trials = Trials()
    best = fmin(get_best_model,
        space=search_space,
        algo=tpe.suggest,
        max_evals=200,
        trials=trials)
    time_steps = trials.results[np.argmin([r['loss'] for r in trials.results])]['time_steps']
    return best, time_steps

def get_best_model(params):
    batch_size = params["batch_size"]
    time_steps = params["time_steps"]
    epochs = params["epochs"]
    generator = TimeseriesGenerator(training_set, training_set, time_steps, batch_size)
    model = Sequential()
    model.add(LSTM(params["lstm1_nodes"], input_shape=(time_steps, 1), dropout=params["lstm1_dropouts"], return_sequences=True))  
    if params["lstm_layers"]["layers"] == 2:
        model.add(LSTM(params["lstm_layers"]["lstm2_nodes"], dropout=params["lstm_layers"]["lstm2_dropouts"]))
    else:
        model.add(Flatten())

    if params["dense_layers"]["layers"] == 2:
        model.add(Dense(params["dense_layers"]["dense2_nodes"], activation='relu'))
    
    model.add(Dense(1, activation='sigmoid'))

    lr = params["lr"]
    if params["optimizer"] == 'rms':
        optimizer = optimizers.RMSprop(lr=lr)
    else:
        optimizer = 'adam'
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    model.fit_generator(generator, epochs = epochs)
    forecasts = forecast(model, training_set, time_steps, len(test_set))
    forecasts = scaler.inverse_transform(forecasts.reshape(-1, 1))
    error = rmse(test_set, forecasts)
    return {'loss': error, 'status': STATUS_OK, 'model': model, 'time_steps': time_steps}