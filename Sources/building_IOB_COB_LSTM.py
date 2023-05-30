# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 09:04:53 2021

@author: bio
"""


from keras.layers import Dense,Dropout, SimpleRNN, GRU,LSTM, Masking, Bidirectional, Conv1D, MaxPooling1D, Lambda
from keras.models import Sequential
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from keras.optimizers import RMSprop, Adam
import keras.backend as K
import pdb
from keras.utils import plot_model
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.io
import joblib
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

scaler_filename = f'Sources/addsOn/scaler.save'
scaler = joblib.load(scaler_filename)

def myIOB_layer(input_tensor):
    

    
    x = input_tensor[:,0,1]
    a = -1
    b = 1
    Ts = 5
    
    """total insulin scaler from training set"""
    scaler_filename = f'Sources/addsOn/scaler.save'
    scaler = joblib.load(scaler_filename)
    insulin_train_scaler = scaler[1]
    
    min_ins = insulin_train_scaler.data_min_
    max_ins = insulin_train_scaler.data_max_
    insulin = ((x - a)*(max_ins-min_ins))/(b-a)+min_ins
    
    """IOB scaler from training set"""
    iob_train_scaler = scaler[3]
    min_iob = iob_train_scaler.data_min_
    max_iob = iob_train_scaler.data_max_
    #plt.plot(K.get_value(insulin))

    k1 = 0.0173
    k2 = 0.0116
    k3 = 6.75
    IOB_6h_curve = np.zeros(360,)
    for t in range(360):
        IOB_6h_curve[t]= 1-0.75*((-k3/(k2*(k1-k2))*(math.exp(-k2*(t)/0.75)-1) + k3/(k1*(k1-k2))*(math.exp(-k1*(t)/0.75)-1))/(2.4947e4));
    
    IOB_6h_curve = np.concatenate((np.zeros(18,), IOB_6h_curve),axis = 0)#`18 for 30 min net, 20 for 60 min
    IOB_6h_curve_sampled = K.reverse(K.constant(IOB_6h_curve[::Ts]),axes = 0)
    
    
    data = K.reshape(insulin,[1, K.shape(insulin)[0], 1])       
    kernel = K.reshape(IOB_6h_curve_sampled, [int(IOB_6h_curve_sampled.shape[0]), 1, 1])
    
    IOB = K.reshape(K.conv1d(data,kernel,strides=1,padding='same'),[-1])
    
    scaledIOB = a + ((IOB-min_iob)*(b-a))/(max_iob-min_iob)

    
    return scaledIOB


def myCOB_layer(input_tensor,dynamic_flag = 'fast'):
    x = input_tensor[:,0,2]
    a = -1
    b = 1
    Ts = 5
    """cho scaler from training set"""
    scaler_filename = f'Sources/addsOn/scaler.save'
    scaler = joblib.load(scaler_filename)
    cho_train_scaler = scaler[2]

    min_cho = cho_train_scaler.data_min_
    max_cho = cho_train_scaler.data_max_
    cho = ((x - a)*(max_cho-min_cho))/(b-a)+min_cho
    
    """COB scaler from training set"""
    cob_train_scaler = scaler[4]
    min_cob = cob_train_scaler.data_min_
    max_cob = cob_train_scaler.data_max_
 
    
    """COB curves"""
    mat_data = scipy.io.loadmat(f'Sources/addsOn/COB.mat')
    COB = mat_data.get('COB')
    
    if dynamic_flag == 'fast':
        fast_meals_curve = COB[:,0]
        my_absorption_curve = K.reverse(K.constant(fast_meals_curve[::Ts]),axes = 0);
    elif dynamic_flag == 'slow':
        slow_meals_curve = COB[:,1]
        my_absorption_curve = K.reverse(K.constant(slow_meals_curve[::Ts]),axes = 0);
        
    data = K.reshape(cho,[1, K.shape(cho)[0], 1])
    kernel = K.reshape(my_absorption_curve, [int(my_absorption_curve.shape[0]), 1, 1])
    COB = K.reshape(K.conv1d(data,kernel,strides=1,padding='same'),[-1])
    
    
    scaledCOB = a + ((COB-min_cob)*(b-a))/(max_cob-min_cob)
    
    return scaledCOB




def myPreprocessingLayer(X): 
    output_list = []
    iob_inp = myIOB_layer(X)
    cob_inp = myCOB_layer(X)
    output_list.append(K.reshape(X[:,0,0],[-1,1]))
    output_list.append(K.reshape(iob_inp,[-1,1]))
    output_list.append(K.reshape(cob_inp,[-1,1]))
    newX = K.stack(output_list, axis = 2) # shape: samples, current time, num features
    
    return  newX


def IobCob_LSTM(X_train,Y_train,verbose = True):
    
    trainX = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    model = Sequential()
    model.add(Lambda(myPreprocessingLayer))
    model.add(LSTM(64, input_shape=(1,X_train.shape[1]))) #128
    model.add(Dense(1,activation='tanh'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, Y_train, batch_size=256, epochs=100, verbose=0)
    
    if verbose:
        
        model.summary()

    return model