#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 19:50:29 2020

@author: Trista
"""
from sklearn import preprocessing
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')
import numpy as np

def SVCL1Selection(X_train, y_train,X_test, y_test, verbal = None, returnCoef = False):
    '''
    choose the model = 'SVCL1'
    fit any feature_selection model with the X_train, y_train
    transform the X_train, X_test with the model
    do not use the X_test to build feature selection model
    
    return the selected X_train, X_test
    print info of the selecter
    return the coef or the score of each feature if asked
    
    '''
    #transform to standardscaler
    features = X_train.columns.tolist()
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = pd.DataFrame(scaler.transform(X_train))
    X_test = pd.DataFrame(scaler.transform(X_test))
    X_train.columns = features
    X_test.columns = features
        
    lsvc = LinearSVC(C=1, penalty="l1", dual=False).fit(X_train, y_train)
    model = SelectFromModel(lsvc, prefit=True)
    index = model.get_support()
    fs_X_name = pd.Series(index,index = X_train.columns)
    getSelectedName = fs_X_name[index==True].index.tolist()
    # print('The selected feature name is '+str(getSelectedName))
    X_train = X_train[getSelectedName]
    X_test = X_test[getSelectedName]
    coef = pd.Series(index)
    
    if verbal == True:
        print('The total feature number is '+ str(sum(index == True)))
        print('The selected feature name is '+ str(getSelectedName))
        
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
       return X_train,y_train,X_test, y_test
    X_train,y_train,X_test, y_test = split_train_test_data(rawXs,rawYs,test_size = 0.3)
    X_train, X_test = SVCL1Selection(X_train, y_train,X_test, y_test,method = True, returnCoef = False)
    
