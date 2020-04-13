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
plt.style.use('ggplot')

ROOT = '../'
# from FeatureEngineering import FeatureEngineering
sys.path.append(os.path.join(ROOT, '03 feature selection'))
from naiveSelection import naiveSelection
from treeSelection import treeSelection
from SVCL1Selection import SVCL1Selection
from varianceThresholdSelection import varianceThresholdSelection
from pcaSelection import pcaSelection

sys.path.append(os.path.join(ROOT, '04 build classifier model'))

from MyDecisionTreeClassifier import MyDecisionTreeClassifier
from MyClassifier import *
from MySVMClassifier import MySVMClassifier
# from MyXGBoostClassifier import MyXGBoostClassifier
# <<<<<<< HEAD
# from MyDeepLearningClassifier import MyDeepLearningClassifier
# =======
from MyKNNClassifier import MyKNNClassifier
from sklearn.svm import SVC
# >>>>>>> cba755d863f0fcf468ecabcb8370f5eb74e79c83


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
        X_train_selected, X_test_selected = featureSelectionFunction(X_train, y_train, X_test, y_test, verbal = True)
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
        
        for predictStartDate, predictEndDate in zip(self.changeHandDates, self.changeHandDates[1:]):
#            check if we have enough data 
            print('start predict from {} to {}'.format(predictStartDate, predictEndDate))
            trainDataDays = np.busday_count(np.datetime64(self.startDate), np.datetime64(predictStartDate.date()))
            if trainDataDays < minTrainDays:
                print('We only have {} trainDataDays'.format(trainDataDays))
                continue
            
#            split the traing and testing set 
            if trainMode == 'extention':
                trainStartDate = self.startDate
            elif trainMode == 'rolling':
                trainStartDate = predictStartDate-pd.Timedelta(minTrainDays, unit = 'B')
            X_train, y_train = self.rawXs[trainStartDate:predictStartDate], self.rawYs[trainStartDate:predictStartDate]
            print('train shape (X, y):{}'.format(X_train.shape, y_train.shape))
            X_test, y_test = self.rawXs[predictStartDate:predictEndDate], self.rawYs[predictStartDate:predictEndDate]
            print('test  shape (X, y):{}'.format(X_test.shape, y_test.shape))
            
            y_predictSeries, model = self.generateOnePeriodSignal(X_train, y_train, X_test, y_test,\
                                                                  featureSelectionFunction, predictModel)
            #  concat outputs
            outputPrediction = pd.concat([outputPrediction, y_predictSeries])
            
            
            # tqdm.write("precision:{}".format(metrics.precision_score(y_true, y_pred)))
            # tqdm.write("recall:{}".format(metrics.recall_score(y_true, y_pred)))
            # tqdm.write("f1:{}\n".format(metrics.f1_score(y_true, y_pred)))  
        
            if recordModels:
                performance = {}
                
                performance.update({
                    'precision':metrics.precision_score(y_test, y_predictSeries),
                    'recall':metrics.recall_score(y_test, y_predictSeries),
                    'f1_score':metrics.f1_score(y_test, y_predictSeries)
                    })
                               
                modelRecord.update({
                    str(predictStartDate.date()):{
                        'trainStartDate' :trainStartDate,
                        'predictStartDate' :predictStartDate,
                        'predictEndDate' :predictEndDate, 
                        'model' :model,
                        'performance':performance
                    }           
                })
    
        if recordModels:
            return(outputPrediction, modelRecord)
        else:
            return(outputPrediction)
    
#%%
if __name__ =='__main__':
    ROOT = '../'
    DATA_PATH = os.path.join(ROOT, '00 data')
    CLEANED_FACTOR_PATH = os.path.join(ROOT, '02 data process')
    rawDf = pd.read_pickle(os.path.join(CLEANED_FACTOR_PATH, 'cleanedFactor.pkl'))
    INDEX_FACTOR_PATH = os.path.join(ROOT, '02 data process')
    indexDf = pd.read_pickle(os.path.join(INDEX_FACTOR_PATH, 'newIndexFactor.pkl'))
    rawDf = pd.merge(indexDf,rawDf,on = 'date',how='right')
    # rawDf = pd.concat([indexDf,rawDf],axis = 1)

#%%    
    sys.path.append(os.path.join(ROOT, '04 select feature and build model'))
    sys.path.append(os.path.join(ROOT, '05 rolling prediction'))
    from FeatureEngineering import FeatureEngineering
    getFeatures = FeatureEngineering(DATA_PATH)
    features = getFeatures.combine_feature()
    rawDf = pd.merge(features,rawDf,on = 'date',how = 'right')
    # rawDf = rawDf.iloc[58:,:]
    rawXs, rawYs = rawDf.iloc[:, :-4], rawDf.iloc[:, -1]
    
    MIN_TRAIN_DAYS = 1600
    TRAIN_MODE = 'extention'
    
    recordModels = True

    selector = SVCL1Selection
    # myPredictModel = MyDeepLearningClassifier
    myPredictModel =  MyKNNClassifier
    
#%%
    sig = RollingSignalGenerator(rawXs, rawYs)
    outputPrediction, models = sig.generateSignal(predictModel = myPredictModel, featureSelectionFunction = selector)
    
#%% 
    # outputPrediction
    y_true = rawYs[outputPrediction.index]
    print(metrics.precision_score(y_true, outputPrediction))
    
    
    outputPredictionFileName = str(selector.__name__) + '_' +str(myPredictModel.__name__)
    path = os.path.join(ROOT, '05 rolling prediction/outputResults/{}'.format(outputPredictionFileName))
    if not os.path.isdir(path):
        os.makedirs(path)
    
    outputPrediction.to_pickle(os.path.join(path,'{}_Value.pkl'.format(outputPredictionFileName)))
    np.save(os.path.join(path,'{}_models'.format(outputPredictionFileName)), models)
#%%
    from load_data import load_data, plot_rts
    windADf = load_data(DATA_PATH + '/881001.csv')
    indexClose = windADf.loc[:, ['date', 'close']].set_index('date')
    indexClose = indexClose[outputPrediction.index[0]:]
    
    # plt.figure(figsize = (20, 8))
    # plt.grid()
    # plt.plot(indexClose.index, indexClose.close)
    # outputPrediction = outputPrediction.astype(bool)
    # plt.scatter(indexClose.loc[outputPrediction.index[outputPrediction]].index, indexClose.loc[outputPrediction.index[outputPrediction]], marker = '^', color = 'r', s = 8, alpha = 0.3)
    # plt.scatter(indexClose.loc[outputPrediction.index[~outputPrediction]].index, indexClose.loc[outputPrediction.index[~outputPrediction]], color = 'g', s = 8 , alpha = 0.3)
    # plt.show()
    
#%% plot output prediction result
    y_pred = pd.read_pickle(path + '/{}_Value.pkl'.format(outputPredictionFileName)).astype(bool)
    y_indexClose = indexClose[y_pred.index[0]:]
    
    windAReturn = indexClose.pct_change().shift(-1).rename(columns = {'close':'return'})
    windACumprod = (windAReturn+1).cumprod()
    
    longDay = y_pred.index[y_pred]
    strategyLongReturn = windAReturn.loc[longDay]
    strategyLongCumprod = (strategyLongReturn+1).cumprod()
    
    shortDay = y_pred.index[~y_pred]
    strategyLongShortReturn = windAReturn.loc[y_pred.index]
    strategyLongShortReturn.loc[shortDay] = strategyLongShortReturn.loc[shortDay]*-1
    strategyLongShortCumprod = (strategyLongShortReturn+1).cumprod()
    
    #save result
    Result = pd.DataFrame()
    Result['windAReturn'] = windAReturn['return']    
    Result['windACumprod'] = windACumprod['return']
    Result = pd.merge(Result,strategyLongReturn['return'],
                      left_index = True,right_index = True,how='left')
    Result = pd.merge(Result,strategyLongCumprod['return'],
                      left_index = True,right_index = True,how='left')
    Result = pd.merge(Result,strategyLongShortReturn['return'],
                      left_index = True,right_index = True,how='left')
    Result = pd.merge(Result,strategyLongShortCumprod['return'],
                      left_index = True,right_index = True,how='left')
    Result = Result.fillna(method = 'ffill')
    Result.columns = ['windAReturn','windACumprod',
                      'LongReturn','LongCumprod',
                      'LongShortReturn','LongShortCumprod']
    Result.to_csv(path+'/{}_NAV.csv'.format(outputPredictionFileName))
    # Result.plot()
    
#%% plot buy and sell time
    plt.figure(figsize = (20, 6))
    # plt.style.use('dark_background') 
    plt.plot(y_indexClose.index, y_indexClose,color = 'blue')
    plt.scatter(shortDay, y_indexClose.loc[shortDay],color = 'red',s = 12)
    plt.scatter(longDay, y_indexClose.loc[longDay],color = 'green',s = 12)
    plt.title('buy and sell time')
    plt.savefig(os.path.join(path,'{}.png'.format(outputPredictionFileName+'_buy_sell_time')))
    plt.show()
    
#%% plot LS portfolio 
    plt.figure(figsize = (20, 20))
    plt.subplot(311)
    plt.plot(strategyLongCumprod.index, strategyLongCumprod, label = 'Long')
    plt.plot(strategyLongShortCumprod.index, strategyLongShortCumprod, label = 'Long-Short')
    plt.plot(windACumprod.loc[y_pred.index].index, windACumprod.loc[y_pred.index], label = 'Simple holding')
    plt.legend()
    plt.title('Timing Strategy Performance')
    
    plt.subplot(312)
    plt.bar(strategyLongReturn.index, strategyLongReturn['return'])
    plt.title('PureLong Strategy win vs fail time')
    
    plt.subplot(313)
    plt.bar(strategyLongShortReturn.index, strategyLongShortReturn['return'])
    plt.title('LongShort Strategy win vs fail time')
    
    plt.savefig(os.path.join(path,'{}.png'.format(outputPredictionFileName+'_performance')))
    plt.show()
    
    
#%% precision recall f1 

    ROOT = '../'
    outputPredictionFileName = str(selector.__name__) + '_' +str(myPredictModel.__name__)
    path = os.path.join(ROOT, '05 rolling prediction/outputResults/{}'.format(outputPredictionFileName))
    models = np.load(os.path.join(path,'{}_models.npy'.format(outputPredictionFileName)), allow_pickle = True).item()
    
    precision = []
    recall = []
    f1 = []
    dateList = []
    
    for k, v in models.items():
        print(k)
        print(v['performance'])
        dateList.append(k)
        precision.append(v['performance']['precision'])
        f1.append(v['performance']['f1_score'])
        
    plt.figure(figsize = (15, 8))
    plt.plot(dateList, precision, label = 'precision')
    plt.plot(dateList, f1,label = 'f1 score')
    plt.xticks(rotation = 90)
    plt.legend()
    plt.show()
    
    np.mean(precision)
            
            
            
            
            
            
            
            
            
            
            