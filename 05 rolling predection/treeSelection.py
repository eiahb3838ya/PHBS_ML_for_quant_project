#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 19:39:09 2020

@author: Trista
"""

from sklearn import preprocessing
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
import pandas as pd
import os

def treeSelection(X_train,y_train ,X_test , y_test, method = None, returnCoef = False):
    '''
    choose the model = 'Tree'
    fit any feature_selection model with the X_train, y_train
    transform the X_train, X_test with the model
    do not use the X_test to build feature selection model
    
    return the selected X_train, X_test
    print info of the selecter
    return the coef or the score of each feature if asked
    '''
    clf = ExtraTreesClassifier(n_estimators = 50)
    clf = clf.fit(X_train,y_train)
    coef = clf.feature_importances_
    model = SelectFromModel(clf,prefit = True)
    index = model.get_support()
    fs_X_name = pd.Series(index,index = X_train.columns)
    getSelectedName = fs_X_name[index==True].index.tolist()
    X_train = X_train[getSelectedName]
    X_test = X_test[getSelectedName]
    coef = pd.Series(coef)
    
    if method == True:
        print('The total feature number is '+ str(sum(index == True)))
        print('The selected feature name is '+ str(getSelectedName))
        
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
    
    from FeatureEngineering import FeatureEngineering
    ROOT =  '/Users/mac/Desktop/ML_Quant/data'
    rawDf = pd.read_pickle(os.path.join(ROOT, 'cleanedFactor.pkl'))
    getFeatures = FeatureEngineering(ROOT)
    features = getFeatures.combine_feature()
    rawDf = pd.merge(features,rawDf,on = 'date')
    # rawDf = rawDf.fillna(method = 'ffill')
    rawXs, rawYs = rawDf.iloc[:, :-4], rawDf.iloc[:, -1].astype(bool)

    X_train, X_test, y_train, y_test = split_train_test_data(rawXs,rawYs,test_size = 0.3)
    X_train, X_test = treeSelection(X_train, X_test, y_train, y_test,method = True, returnCoef = False)
    
