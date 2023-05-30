import numpy as np 
import pandas as pd
import pdb
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# =============================================================================
# how to handle with missing data
# =============================================================================
def find_nan_idx(X):
    """ Find all indices where X is nan """
    
    nan_idx = np.argwhere(np.isnan(X))
    nan_idx_row = nan_idx[:,0]
    return nan_idx_row

def handling_nan_values(X, y, method='remove'):
    """ Handles missing values in numpy matrix using specified method """
    
    # find nan values in y
    y_idx_nan = find_nan_idx(y)
    # find nan values in X
    X_nan_idx = find_nan_idx(X)
    # union of the 2 nan array indexes
    nan_idx = np.union1d(X_nan_idx,y_idx_nan)
    
    if method == 'remove':
        # remove nan values from X and y
        Xfin = np.delete(X, nan_idx, 0)
        Yfin = np.delete(y, nan_idx, 0)
    elif method == 'mean':
        # replace nan values with mean
        # TODO
        Xfin = X
        Yfin = y
    
    return Xfin, Yfin, nan_idx

def insert_row_nan(nan_array_idx, y):
    """ Re-inserts nan in their original position """
    
    for k in range(len(nan_array_idx)):
        y = np.insert(y, nan_array_idx[k], np.nan, 0)
    return y

# =============================================================================
# evaluation metrics
# =============================================================================
def compute_rmse(y, yhat, PH):
    nan_idx_target = find_nan_idx(y)
    y_wo_nan = np.delete(y, nan_idx_target, 0)
    yhat_wo_nan = np.delete(yhat, nan_idx_target, 0)
    RMSE_score = np.sqrt(mean_squared_error(y_wo_nan, yhat_wo_nan))
    return RMSE_score


# =============================================================================
# # --- prepare data for model ---
# =============================================================================

def roll_features(x, lb, PH):
    """
    creates shifted columns, e.g. x(t), x(t-1), ... x(t-lb)
    lb: lookback window length
    e.g. samples x(t), ..., x(t-lb)
    
    """
    xmatrix = np.zeros([len(x)-lb-PH+1,lb])
    #ymatrix = np.zeros([len(x)-lb-PH+1,])

    for i in range(0,lb):
        x_rolled = np.roll(x,-i)
        x_rolled = x_rolled[:-lb-PH+1]
        xmatrix[:,i] = x_rolled.T
    
    #ymatrix = x[lb+PH-1:]
    Xnew = xmatrix
    return Xnew
    

def get_target(data, lb, PH):
    """ creates target column, y(t+PH) """
    y = data['CGM'].values
    y_target = np.zeros([len(y)-lb-PH+1,])
    y_target = y[lb+PH-1:]
    
    y_target = MinMaxScaler(feature_range=(-1, 1)).fit_transform(y_target.reshape(-1,1)).flatten()
    return y_target



# =============================================================================
# # --- make Xtrain and Ytrain matrix using selected features ---
# =============================================================================
def make_Training_matrix(data, labels, lb, PH):
    """
    Prepares data for LSTM
    1) creates target column, y(t+PH)
    2) selects columns and rescales values
    3) creates shifted columns using lookback parameters
    
    lb: lookback window length
    e.g. samples x(t), ..., x(t-lb)
    
    lb_sample_rate: sample rate of lookback window
    e.g.
    with lb_sample_rate=1, x(t), x(t-1), x(t-2), ..., x(t-lb)
    with lb_sample_rate=3, x(t), x(t-3), x(t-6), ..., x(t-lb)
    """
    
    X = []
    training_scalers = []
    
    # get target
    y = get_target(data, lb, PH)
    
    # get X
    for col in labels:
        signal_values = data[col].values
        scaler = MinMaxScaler(feature_range=(-1, 1))
        signal_values = scaler.fit_transform(signal_values.reshape(-1,1))
        signal_values = signal_values.reshape(-1)
        # store rescaler
        training_scalers.append(scaler)
        
        # roll features
        X_tmp  = roll_features(signal_values, lb, PH)
        X.append(X_tmp)
        
    # make numpy matrix
    X = np.concatenate(X, axis=1)
    
    return X, y, training_scalers
    


# =============================================================================
# # --- make Xtest and Ytest matrix using training features    
# =============================================================================
def make_Testing_matrix(data, labels, training_scalers, lookback_wind, PH):
    
    
    # --- select variables from original dataset ---
    num_signals = len(labels)
    X = []
    
    # get target
    y = get_target(data, lookback_wind, PH)
    
    for u in range(num_signals):
        
        var_name = labels[u]
        signal_values = data[var_name].values
        
        scaler_tmp = training_scalers[u]
        
        if  var_name == 'CGM':
            cgm_scaler = scaler_tmp
        
        signal_values = scaler_tmp.fit_transform(signal_values.reshape(-1,1))
        signal_values = signal_values.reshape(-1)
            
        X_tmp  = roll_features(signal_values, lookback_wind, PH)
        X.append(X_tmp)
        
    # make numpy matrix
    X = np.concatenate(X, axis=1)
    
    return X,y,cgm_scaler
