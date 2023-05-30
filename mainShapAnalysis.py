# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 16:43:34 2021

@author: bio
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import itertools as iter
import time
import argparse
import warnings
import scipy.io as sio
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import load_model
import keras.backend as K
from Sources.building_IOB_COB_LSTM import*
import math
warnings.filterwarnings("ignore")

import shap
from Sources.utils import *


""" SHAP analysis: np-LSTM(CGM, ins, cho) vs p-LSTM(CGM, ins, cho)"""

subj = '588';
lb = 1
num_models = 2
chose_iter = 0
features =  [['CGM','total_insulin','CHO','IOB','COB']]
PredictionHorizon = [6]
N_train_samples = 1000
N_test_samples = 500


saving_results = 1
saving_models = 1

for p in range(len(PredictionHorizon)):
    
    PH = PredictionHorizon[p]
    
    print(f"current prediction horizon: {5*PH} min.")
    
    for mod in range(num_models):
        
        cols = features[0]
        
        """ data preparation """
        # training data
        data_tr = pd.read_csv(F'data/ohio{subj}_Training.txt')
        data_tr = data_tr[1000:2000] # to speed up SHAP analysis
            
        data_tr.insert(5,'total_insulin',data_tr.basal_insulin + data_tr.bolus_insulin)
        Xtrain, Ytrain, train_scalers = make_Training_matrix(data_tr, cols, lb, PH)
        Xtr, Ytr, nan_idx_tr = handling_nan_values(Xtrain, Ytrain)
        Xtr = Xtr[:,:3] #keep CGM, ins and cho
        
        # test data
        data_te = pd.read_csv(F'data/ohio{subj}_Testing.txt')
        
        data_te.insert(5,'total_insulin',data_te.basal_insulin+data_te.bolus_insulin)
        Xtest, Ytest, cgm_scaler = make_Testing_matrix(data_te, cols, train_scalers, lb, PH)
        Xte, Yte, nan_idx_te = handling_nan_values(Xtest, Ytest)
        Xte = Xte[:,:3] #keep CGM, ins and cho
         
        """load trained lstm model"""
        if mod == 0:
            model = load_model(f'results/models/{cols}/lstm{mod}_ph_{5*PH}_iters_{chose_iter}.h5')
        else:
            model = load_model(f'results/models/{cols}/lstm{mod}_ph_{5*PH}_iters_{chose_iter}.h5', custom_objects={"myPreprocessingLayer":myPreprocessingLayer,"myIOB_layer":myIOB_layer,"myCOB_layer":myCOB_layer})
            
       
        """SHAP analysis"""
        trainX = np.reshape(Xtr, (Xtr.shape[0], 1, Xtr.shape[1]))
        testX = np.reshape(Xte, (Xte.shape[0], 1, Xte.shape[1]))
        N = np.size(Xtr,1)
        explainer = shap.DeepExplainer(model, trainX[0:N_train_samples])
        idx = np.random.permutation(np.arange(0,len(testX), 1))[:]
        shap_values = explainer.shap_values(testX[idx])[0]
        var_names = cols
        plt.figure()
        
        shap.summary_plot(shap_values.reshape(-1,N), testX[idx].reshape(-1,N), feature_names=var_names)