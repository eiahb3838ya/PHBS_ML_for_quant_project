# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 10:57:36 2020

@author: Evan
"""
#%%
import pandas as pd 
import numpy as np
import  sys, os

from tqdm import notebook 
from tqdm import tqdm

from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

import matplotlib.pyplot as plt

from MyDecisionTreeClassifier import MyDecisionTreeClassifier
from MyClassifier import *
from MySVMClassifier import MySVMClassifier
from MyDeepLearningClassifier import MyDeepLearningClassifier


from FeatureEngineering import FeatureEngineering

from naiveSelection import naiveSelection
from treeSelection import treeSelection
from SVCL1Selection import SVCL1Selection
from varianceThresholdSelection import varianceThresholdSelection

#%%

class RollingSignalGenerator:
    def __init__(self, rawXs, rawYs, startDate = None, endDate = None, predictWindow = 20):
        self.rawXs = rawXs
        self.rawYs = rawYs 
        
        if startDate == None:
            self.startDate = str(self.rawYs.index[0].date())
        else:
            self.startDate = startDate
        if endDate == None:
            self.endDate = str(self.rawYs.index[-1].date())
        else:
            self.endDate = endDate
        
        try:
            self.changeHandDates = pd.date_range(self.startDate, self.endDate,freq = "{}B".format(predictWindow))
        except Exception as e:
            print('Please input startDate and endDate as format YYYY-mm-dd')
            print(e.args[0])
            
    def generateOnePeriodSignal(self, X_train, y_train, X_test, y_test, featureSelectionFunction, predictModel):
        X_train_selected, X_test_selected = featureSelectionFunction(X_train, y_train, X_test, y_test,method = True)
        # fit predict
        model = predictModel()
        model.fit(X_train_selected, y_train)
        
        y_true = y_test
        y_pred = model.predict(X_test_selected)
        
#        tqdm.write("precision:{}".format(metrics.precision_score(y_true, y_pred)))
#        tqdm.write("recall:{}".format(metrics.recall_score(y_true, y_pred)))
#        tqdm.write("f1:{}\n".format(metrics.f1_score(y_true, y_pred)))  
        
        # return(pd.Series(y_pred, index = y_test.index), model)
        y_pred = pd.Series(y_pred)
        y_pred.index = y_test.index
        return(pd.Series(y_pred), model)
        
            
    def generateSignal(self, predictModel, featureSelectionFunction, minTrainDays = 1800, trainMode = 'extention', recordModels = True):
        modelRecord = {}
        outputPrediction = pd.Series()
        
        for predictStartDate, predictEndDate in tqdm(zip(self.changeHandDates, self.changeHandDates[1:])):
#            check if we have enough data 
            tqdm.write('start predict from {} to {}'.format(predictStartDate, predictEndDate))
            trainDataDays = np.busday_count(np.datetime64(self.startDate), np.datetime64(predictStartDate.date()))
            if trainDataDays < minTrainDays:
                tqdm.write('We only have {} trainDataDays'.format(trainDataDays))
                continue
            
#            split the traing and testing set 
            if trainMode == 'extention':
                trainStartDate = self.startDate
            elif trainMode == 'rolling':
                trainStartDate = predictStartDate-pd.Timedelta(minTrainDays, unit = 'B')
            X_train, y_train = self.rawXs[trainStartDate:predictStartDate], self.rawYs[trainStartDate:predictStartDate]
            tqdm.write('train shape (X, y):{}'.format(X_train.shape, y_train.shape))
            X_test, y_test = self.rawXs[predictStartDate:predictEndDate], self.rawYs[predictStartDate:predictEndDate]
            tqdm.write('test  shape (X, y):{}'.format(X_test.shape, y_test.shape))
            
            y_predictSeries, model = self.generateOnePeriodSignal(X_train, y_train, X_test, y_test,\
                                                                  featureSelectionFunction, predictModel)
            #  concat outputs
            outputPrediction = pd.concat([outputPrediction, y_predictSeries])    
            if recordModels:
                modelRecord.update({
                    str(predictStartDate.date()):{
                        'trainStartDate' :trainStartDate,
                        'predictStartDate' :predictStartDate,
                        'predictEndDate' :predictEndDate, 
                        'model' :model 
                    }           
                })
    
        if recordModels:
            return(outputPrediction, modelRecord)
        else:
            return(outputPrediction)
    
        
#%%
if __name__ =='__main__':
    # ROOT = '../'
    # DATA_PATH = os.path.join(os.path.join(ROOT, '04 select feature and build model'), 'data')
    # CLEANED_FACTOR_PATH = os.path.join(ROOT, '03 data process')
    ROOT =  '/Users/mac/Desktop/ML_Quant/data'
    rawDf = pd.read_pickle(os.path.join(ROOT, 'cleanedFactor.pkl'))
    getFeatures = FeatureEngineering(ROOT)
    features = getFeatures.combine_feature()
    rawDf = pd.merge(features,rawDf,on = 'date')
    rawXs, rawYs = rawDf.iloc[:, :-4], rawDf.iloc[:, -1].astype(bool)
    
    MIN_TRAIN_DAYS = 1600
    TRAIN_MODE = 'rolling'
    # predictModel = MyLogisticRegClassifier
    recordModels = True
    selector = naiveSelection
    myPredictModel = MyDeepLearningClassifier
    # myPredictModel =  MyDecisionTreeClassifier
    
#%%
    sig = RollingSignalGenerator(rawXs, rawYs)
    outputPrediction, models = sig.generateSignal(predictModel = myPredictModel, featureSelectionFunction = selector)
#%%
    from load_data import load_data, plot_rts
    windADf = load_data(ROOT + '/881001.csv')
    indexClose = windADf.loc[:, ['date', 'close']].set_index('date')
    indexClose = indexClose[outputPrediction.index[0]:]
#%%
    plt.figure(figsize = (20, 8))
    plt.grid()
    plt.plot(indexClose.index, indexClose.close)
    plt.scatter(indexClose.loc[outputPrediction.index[outputPrediction]].index, indexClose.loc[outputPrediction.index[outputPrediction]], marker = '^', color = 'r', s = 8, alpha = 0.3)
    plt.scatter(indexClose.loc[outputPrediction.index[~outputPrediction]].index, indexClose.loc[outputPrediction.index[~outputPrediction]], color = 'g', s = 8 , alpha = 0.3)
    plt.show()
        
        
    
            
            
            
            

            
            
            
            
            
            
            
            
            
            
            
            