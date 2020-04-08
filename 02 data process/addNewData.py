#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 13:48:47 2020

@author: Trista
"""
import pandas as pd
import os

ROOT = '/Users/mac/Desktop/ML_Quant'
DATA_PATH = os.path.join(os.path.join(ROOT, '00 data'), 'AddNewData')
outputDir = os.path.join(os.path.join(ROOT, '03 data process'))
files = os.listdir(DATA_PATH)
files.sort()
files = files[1:]
# df1 = pd.read_csv(DATA_PATH + '/'+ files[0])
# df1.columns = df1.iloc[2,:]
# df1 = df1.iloc[3:,:]
# df1 = df1.set_index('Date')
# df1 = df1[['close']].rename(columns={'close':files[0][:-4]})
   
df = pd.DataFrame()
for i in files:
    oneDf = pd.read_csv(DATA_PATH + '/'+ i)
    oneDf.columns = oneDf.iloc[2,:]
    oneDf = oneDf.iloc[3:,:]
    oneDf.columns = ['date',i[:-4]]
    oneDf = oneDf.set_index('date')
    # oneDf = oneDf[['close']].rename(columns={'close':i[:-4]})
    df = pd.concat([df,oneDf],axis = 1)


subDf = df.copy() 
subDf = subDf.fillna(method = 'ffill')
subDf = subDf.astype(float)
subDf.index = pd.DatetimeIndex(subDf.index)
subDf = subDf.iloc[500:,:]
subDf.to_csv(os.path.join(outputDir, 'newIndexFactor.csv'))
subDf.to_pickle(os.path.join(outputDir, 'newIndexFactor.pkl'))
