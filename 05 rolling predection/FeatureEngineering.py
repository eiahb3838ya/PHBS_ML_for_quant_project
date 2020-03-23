#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 16:23:20 2020

@author: Trista
"""
import pandas as pd
from load_data import load_data
from datetime import datetime
import numpy as np

class FeatureEngineering():
    def __init__(self,ROOT,START_TIME = '2008-01-03',END_TIME = '2020-03-15'):
        self.path = ROOT
        self.df = pd.DataFrame(load_data(self.path + '/881001.csv'))
        self.df = self.df[self.df['date'] >= pd.Timestamp(START_TIME)] 
        self.df = self.df[self.df['date'] <= pd.Timestamp(END_TIME)]
        
        self.benchmark = pd.DataFrame(load_data(self.path + '/000300.csv'))
        self.benchmark = self.benchmark[self.benchmark['date'] >= pd.Timestamp(START_TIME)] 
        self.benchmark = self.benchmark[self.benchmark['date'] <= pd.Timestamp(END_TIME)]
        
        self.macro_factor = pd.read_pickle(self.path + '/cleanedFactor.pkl')
        self.date = self.df['date']
        self.rts = self.df['rts']
        
        self.open = self.df['open']
        self.high = self.df['high']
        self.low = self.df['low']
        self.close = self.df['close']
        self.pre_close = self.df['pre_close']
        self.volume = self.df['volume']
        self.amount = self.df['amt']
        
        self.benchmark_open = self.benchmark['open']
        self.benchmark_close = self.benchmark['close']
    
    
    def alpha002(self):
        result = ((self.close-self.low)-(self.high-self.close))/((self.high-self.low)).diff()
        result = result.fillna(method = 'ffill')
        return result
    
    def alpha014(self):
        result = self.close - self.close.shift(5)
        return result
    
    def alpha018(self):
        delay5 = self.close.shift(5)
        result = self.close / delay5
        return result
    
    def alpha020(self):
        delay6 = self.close.shift(6)
        result = (self.close - delay6) * 100 / delay6
        return result
    
    def alpha034(self):
        result = self.close.rolling(12).mean()/self.close
        return result
    
    def alpha060(self):
        result =((self.close - self.low) - (self.high - self.close))/(self.high - self.low) * self.volume
        result = result.rolling(20).sum()
        return result
    
    def alpha066(self):
        up = self.close - self.close.rolling(6).mean()
        down = self.close.rolling(6).mean()
        result = up / down * 100
        return result
    
    def alpha070(self):
        result = self.amount.rolling(6).std()
        return result
    
    def alpha106(self):
        result = self.close - self.close.shift(20)
        return result
    
    def combine_feature(self):
        X = pd.DataFrame()
        X['date'] = self.date
        X['rts'] = self.rts
        X['close'] = self.close
        X['benchmarkClose'] = self.benchmark_close
        
        X['alpha002'] = self.alpha002()
        X['alpha014'] = self.alpha014()
        X['alpha018'] = self.alpha018()
        X['alpha020'] = self.alpha020()
        X['alpha034'] = self.alpha034()
        # X['alpha060'] = self.alpha060()
        X['alpha066'] = self.alpha066()
        X['alpha070'] = self.alpha070()
        X['alpha106'] = self.alpha106()
        
        X = pd.merge(X,self.macro_factor,on = 'date')
        X.iloc[:,1:] = X.iloc[:,1:].astype(float)
        return X
       
if __name__ == '__main__':
    ROOT =  '/Users/mac/Desktop/ML_Quant/data'
    Klass = FeatureEngineering(ROOT)
    features = Klass.combine_feature()
    

