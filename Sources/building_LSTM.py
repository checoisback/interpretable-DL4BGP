# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 15:22:00 2021

@author: bio
"""

from keras.layers import Dense,Dropout, SimpleRNN, GRU,LSTM, Masking, Bidirectional,Conv1D,MaxPooling1D
from keras.models import Sequential
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from keras.optimizers import RMSprop, Adam
import keras.backend as K
from keras.utils import plot_model
import numpy as np



def build_LSTMmodel(X_train,Y_train,verbose = True):
    
    trainX = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    model = Sequential()
    model.add(LSTM(64, input_shape=(1,X_train.shape[1]))) #128
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, Y_train, epochs=100, batch_size=256, verbose=0)
    if verbose:
        model.summary()

    return model
