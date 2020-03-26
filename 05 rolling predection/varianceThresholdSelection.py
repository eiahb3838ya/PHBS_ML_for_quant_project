#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 19:41:14 2020

@author: Trista
"""
from sklearn import preprocessing
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')

def varianceThresholdSelection(X_train, y_train, X_test,y_test, method = None,returnCoef = False):
    
    '''
    choose the model = 'VarianceThreshold'
    fit any feature_selection model with the X_train, y_train
    transform the X_train, X_test with the model
    do not use the X_test to build feature selection model
    
    return the selected X_train, X_test
    print info of the selecter
    return the coef or the score of each feature if asked
    
    '''
    threshold = 0.8
    coef = pd.Series(X_train.var(),index = X_train.columns)
    selectedX_train = (coef >= threshold * (1-threshold))
    selectedX_train = pd.DataFrame(selectedX_train)
    fs_X_name = selectedX_train[selectedX_train[0] == True].index.tolist()
    X_train = X_train[fs_X_name]
    X_test = X_test[fs_X_name]
    
    features = X_train.columns.tolist()
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = pd.DataFrame(scaler.transform(X_train))
    X_test = pd.DataFrame(scaler.transform(X_test))
    X_train.columns = features
    X_test.columns = features
    
    if method == True:
        print('The total feature number is '+ str(len(fs_X_name)))
        print('The selected feature name is ' + str(fs_X_name))
        
    if not returnCoef:
        return(X_train, X_test)
    else:
        return(X_train, X_test, coef)
    
if __name__ == '__main__':
    from FeatureEngineering import FeatureEngineering
    ROOT =  '/Users/mac/Desktop/ML_Quant/data'
    rawDf = pd.read_pickle(os.path.join(ROOT, 'cleanedFactor.pkl'))
    getFeatures = FeatureEngineering(ROOT)
    features = getFeatures.combine_feature()
    rawDf = pd.merge(features,rawDf,on = 'date')
    # rawDf = rawDf.fillna(method = 'ffill')
    rawXs, rawYs = rawDf.iloc[:, :-4], rawDf.iloc[:, -1].astype(bool)

    def split_train_test_data(X,y,test_size):
        num_train = int(len(X) - len(X) * test_size)
        X_train = X.iloc[:num_train,:]
        X_test = X.iloc[num_train:,:]
        y_train = y[:num_train]
        y_test = y[num_train:]
        return X_train,y_train,X_test,y_test

    X_train, y_train, X_test, y_test = split_train_test_data(rawXs,rawYs,test_size = 0.3)
    X_train, X_test = varianceThresholdSelection(X_train, y_train, X_test, y_test,method = True, returnCoef = False)
    