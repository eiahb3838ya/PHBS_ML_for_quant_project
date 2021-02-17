#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 16:23:20 2020

@author: Trista
"""
#%%
import pandas as pd
#%%
class AlphaFeature(object):
    def __init__(self, data):
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
    
    def get_feature(self, alpha_feature_list):
        X = pd.DataFrame()
        for a_feature in alpha_feature_list:
            X[a_feature] = self.__getattribute__(a_feature)()
        return X

#%%
if __name__ == '__main__':
    import os
    os.chdir(path="..")
    from util import custom_load_data
    DATA_PATH = 'C:\\Users\\eiahb\\Documents\\MyFiles\\PythonProject\\PHBS_machine_learning_for_finance\\PHBS_ML_for_quant_project\\00data'
    index_dict = {
        "windA":"881001.csv", 
        "hs300":"000300.csv"
    }
    alpha_feature_list = [
        'alpha002',
        'alpha014',
        'alpha018',
        'alpha020',
        'alpha034',
        'alpha066',
        'alpha070',
        'alpha106'

    ]
    test_index = "windA"
    index_data = custom_load_data(os.path.join(DATA_PATH, index_dict[test_index]))
    klass = AlphaFeature(index_data)
    features = klass.get_feature(alpha_feature_list)

    


# %%
