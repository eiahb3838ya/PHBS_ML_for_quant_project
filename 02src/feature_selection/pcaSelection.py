#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 17:48:50 2020

@author: Trista
"""
import pandas as pd
import os
from sklearn import preprocessing
import warnings
warnings.filterwarnings('ignore')
from sklearn.decomposition import PCA

def pcaSelection(X_train, y_train, X_test, y_test, verbal = None, returnCoef = False):
    '''
    choose the feature selection method = 'pca'
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
    
    pca = PCA(n_components = 3)
    X_train = pca.fit_transform(X_train)
    print ('The explained variance ratio is:')
    print(pca.explained_variance_ratio_)
    print('The total explained variance ratio is ')
    print(sum(pca.explained_variance_ratio_))
    print ('The explained variance is:')
    print(pca.explained_variance_)
    X_test = pca.transform(X_test)
    
    
    coef = pd.Series()
    # featureName = None
    
    if verbal == True:
       print('The total feature number is '+ str(X_train.shape[1]))
       # print('The selected feature name is '+ str(featureName))
       
    if not returnCoef:
        return(X_train, X_test)
    else:
        return(X_train, X_test, coef)
    
if __name__ == '__main__':
    import sys,os
    import pandas as pd
    ROOT =  '../'
    sys.path.append(os.path.join(ROOT, '05 rolling prediction'))
    from FeatureEngineering import FeatureEngineering
    ROOT = '../'
    DATA_PATH = os.path.join(ROOT, '00 data')
    CLEANED_FACTOR_PATH = os.path.join(ROOT, '02 data process')
    rawDf = pd.read_pickle(os.path.join(CLEANED_FACTOR_PATH, 'cleanedFactor.pkl'))
    INDEX_FACTOR_PATH = os.path.join(ROOT, '02 data process')
    indexDf = pd.read_pickle(os.path.join(INDEX_FACTOR_PATH, 'newIndexFactor.pkl'))
    rawDf = pd.merge(indexDf,rawDf,on = 'date',how='right')
    # rawDf = pd.concat([indexDf,rawDf],axis = 1)

#%%    
    # sys.path.append(os.path.join(ROOT, '04 select feature and build model'))
    from FeatureEngineering import FeatureEngineering
    getFeatures = FeatureEngineering(DATA_PATH)
    features = getFeatures.combine_feature()
    rawDf = pd.merge(features,rawDf,on = 'date',how = 'right')
    # rawDf = rawDf.iloc[58:,:]
    rawXs, rawYs = rawDf.iloc[:, :-4], rawDf.iloc[:, -1]

    def split_train_test_data(X,y,test_size):
        num_train = int(len(X) - len(X) * test_size)
        X_train = X.iloc[:num_train,:]
        X_test = X.iloc[num_train:,:]
        y_train = y[:num_train]
        y_test = y[num_train:]
        return X_train,y_train,X_test, y_test
    X_train,y_train,X_test, y_test = split_train_test_data(rawXs,rawYs,test_size = 0.3)
    X_train, X_test = pcaSelection(X_train,y_train,X_test, y_test, verbal = True, returnCoef = False)
    
    