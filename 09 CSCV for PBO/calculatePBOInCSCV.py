#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 15:41:47 2020

@author: Trista

calculate probability of backtest overffiting of an invest strategy
After we get best hyperprameter(like xgboost), 
we can calculate different combination of parameter to know whether the strategy is over fitting or not.
For the evidence in paper, timing strategy is likely overfitting.
"""
import pandas as pd
import os
import numpy as np
from itertools import combinations
import math
from tqdm import tqdm

#%%
# cut origin data into S subsamples and make paris of IS and OOS
# calculate ranks and lambda
def calculateLambda(df, column, S):
    df.sort_values(by = column, ascending = True, inplace = True)
    df.set_index(column, inplace = True)
    
    signal = list()
    for i in range(0,S):
        signal.append(i)
    
    trainingDataNoGroup = list(set(combinations(signal, int(S/2))))
   
    lamdas = list()
    for trainingDataNo in tqdm(trainingDataNoGroup):
        # for every case calculate the PBO
        trainData = pd.DataFrame(columns = list(df.columns))
        testData = pd.DataFrame(columns = list(df.columns))
        for i in range(S):
            sample = df.iloc[int(i*len(df)//S):int((i+1)*len(df)//S)]
            if i in trainingDataNo:
                trainData = pd.concat([trainData, sample], axis = 0)
            else:
                testData = pd.concat([testData, sample], axis = 0)
                
        # calculate SR or whatever other indicators.
        # this is case by case
        trainDataSharpeRatio = trainData.mean() / trainData.std() 
        # position is to get the highest SR ratio strategy n*
        position = trainDataSharpeRatio.sort_values(ascending = False).rank().index[0]
        testDataSharpeRatio = testData.mean() / testData.std()
        rank = testDataSharpeRatio.rank()[position] / (len(testDataSharpeRatio)+1)
        lamda = math.log(rank/(1-rank))
        print('lamda = ', lamda)
        # the higher the lambda, the better the strategy.
        lamdas = lamdas + [lamda]
    lamdas.sort()
    return lamdas

#%% calculate PBO
def calPBO(lamdas):
    count = len([num for num in lamdas if num < 0])
    PBO = count / len(lamdas)
    print('PBO is ',PBO)

    if PBO < 0.5:
        print ('it is very likely that this timing strategy is useful')
    else:
        print('useless timing strategy :( ')
    return PBO
 
if __name__ == '__main__':
    ROOT =  '../'
    df = pd.read_csv(os.path.join(ROOT,'07 CSCV for PBO/testData.csv'))
    df = df.iloc[-96:,:]
    column = 'date'
    lamdas = calculateLambda(df, column, S = 8)
    # dfForSave = pd.DataFrame(lamdas, columns = ['lamda'])
    # dfForSave.to_excel('lamda.xlsx')
    PBO = calPBO(lamdas)
    

