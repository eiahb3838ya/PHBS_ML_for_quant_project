# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 12:56:45 2020

@author: Evan & Trista
"""
import pandas as pd
import os
from FeatureEngineering import FeatureEngineering
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LassoCV
import numpy as np
from sklearn import linear_model
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import RFE
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
import warnings
warnings.filterwarnings('ignore')

def naiveSelection(X_train, X_test, y_train, y_test, method = None, returnCoef = False):
    
    '''
    choose the model type with 3 method('VarianceThreshold','SVCL1','Tree'),else pass
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
        
    if method == 'VarianceThreshold':
        threshold = 0.8
        coef = pd.Series(X_train.var(),index = X_train.columns)
        selectedX_train = (coef >= threshold * (1-threshold))
        selectedX_train = pd.DataFrame(selectedX_train)
        fs_X_name = selectedX_train[selectedX_train[0] == True].index.tolist()
        print('The total feature number is '+ str(len(fs_X_name)))
        # print('The selected feature name is ' + str(fs_X_name))
        X_train = X_train[fs_X_name]
        X_test = X_test[fs_X_name]
        
    elif method =='SVCL1':
        lsvc = LinearSVC(C=1, penalty="l1", dual=False).fit(X_train, y_train)
        model = SelectFromModel(lsvc, prefit=True)
        index = model.get_support()
        fs_X_name = pd.Series(index,index = X_train.columns)
        print('The total feature number is '+str(sum(index == True)))
        getSelectedName = fs_X_name[index==True].index.tolist()
        # print('The selected feature name is '+str(getSelectedName))
        X_train = X_train[getSelectedName]
        X_test = X_test[getSelectedName]
        coef = index
        
    elif method =='Tree':
        clf = ExtraTreesClassifier(n_estimators = 50)
        clf = clf.fit(X_train,y_train)
        coef = clf.feature_importances_
        model = SelectFromModel(clf,prefit = True)
        index = model.get_support()
        fs_X_name = pd.Series(index,index = X_train.columns)
        print('The total feature number is '+str(sum(index == True)))
        getSelectedName = fs_X_name[index==True].index.tolist()
        # print('The selected feature name is '+str(getSelectedName))
        X_train = X_train[getSelectedName]
        X_test = X_test[getSelectedName]
        
    else:
        pass
          
    if not returnCoef:
        return(X_train, X_test)
    else:
        return(X_train, X_test, coef)
    
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
    X_train, X_test = naiveSelection(X_train, X_test, y_train, y_test, method = 'Tree', returnCoef = False)
    