#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 16:23:20 2020

@author: Trista
"""
#%%
import pandas as pd
from abc import abstractmethod, ABCMeta
#%%
class TechnicalFeatureBase(object,metaclass=ABCMeta):
    def __init__(self, data, **kwargs):
        if isinstance(data, pd.DataFrame):
            self.df = data
        elif isinstance(data, str):
            try:
                self.df = pd.read_csv(data)
            except :
                print("please pass in csv file path")
                raise 
        else:
            print("please pass in pd.DataFrame or file path")
            raise Exception

        self.rts = self.df['return']
        self.open = self.df['open']
        self.high = self.df['high']
        self.low = self.df['low']
        self.close = self.df['close']
        self.pre_close = self.df['pre_close']
        self.volume = self.df['volume']
        self.amount = self.df['amt']
        self.params = kwargs
        
    @abstractmethod
    def get_feature(self):
        pass
