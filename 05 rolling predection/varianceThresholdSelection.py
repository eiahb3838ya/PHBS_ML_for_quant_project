#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 19:41:14 2020

@author: Trista
"""
from sklearn import preprocessing
import pandas as pd
import os
from FeatureEngineering import FeatureEngineering
import warnings
warnings.filterwarnings('ignore')

def varianceThresholdSelection(X_train, X_test, y_train, y_test, method = None,returnCoef = False):
    
    '''
    choose the model = 'VarianceThreshold'
    fit any feature_selection model with the X_train, y_train
    transform the X_train, X_test with the model
    do not use the X_test to build feature selection model
    
    return the selected X_train, X_test
    print info of the selecter
    return the coef or the score of each feature if asked
    
    '''
    #transform to norm 
    for item in [X_train,X_test]:
        quantile_transformer = preprocessing.QuantileTransformer(output_distribution = 'normal',
                                                                 random_state = 0)
        item = quantile_transformer.fit_transform(item).copy()
        
        threshold = 0.8
        coef = pd.Series(X_train.var(),index = X_train.columns)
        selectedX_train = (coef >= threshold * (1-threshold))
        selectedX_train = pd.DataFrame(selectedX_train)
        fs_X_name = selectedX_train[selectedX_train[0] == True].index.tolist()
        X_train = X_train[fs_X_name]
        X_test = X_test[fs_X_name]
    
    if method == True:
        print('The total feature number is '+ str(len(fs_X_name)))
        print('The selected feature name is ' + str(fs_X_name))
        
    if not returnCoef:
        return(X_train, X_test)
    else:
        return(X_train, X_test, coef)
    
#%%just for test 
def split_train_test_data(X,y,test_size):
    num_train = int(len(X) - len(X) * test_size)
    X_train = X.iloc[:num_train,:]
    X_test = X.iloc[num_train:,:]
    y_train = y[:num_train]
    y_test = y[num_train:]
    return X_train,X_test,y_train,y_test

if __name__ == '__main__':
    ROOT =  '/Users/mac/Desktop/ML_Quant/data'
    rawDf = pd.read_pickle(os.path.join(ROOT, 'cleanedFactor.pkl'))
    getFeatures = FeatureEngineering(ROOT)
    features = getFeatures.combine_feature()
    rawDf = pd.merge(features,rawDf,on = 'date')
    # rawDf = rawDf.fillna(method = 'ffill')
    rawXs, rawYs = rawDf.iloc[:, :-4], rawDf.iloc[:, -1].astype(bool)

    X_train, X_test, y_train, y_test = split_train_test_data(rawXs,rawYs,test_size = 0.3)
    X_train, X_test = varianceThresholdSelection(X_train, X_test, y_train, y_test,method = True, returnCoef = False)
    