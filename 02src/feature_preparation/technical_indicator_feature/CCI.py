#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 16:23:20 2020

@author: Evan
"""
#%%

import talib
if __name__ == '__main__':
    from TechnicalFeatureBase import TechnicalFeatureBase
else:
    from .TechnicalFeatureBase import TechnicalFeatureBase

#%%
class CCI(TechnicalFeatureBase):
    # self.rts = self.df['return']
    # self.open = self.df['open']
    # self.high = self.df['high']
    # self.low = self.df['low']
    # self.close = self.df['close']
    # self.pre_close = self.df['pre_close']
    # self.volume = self.df['volume']
    # self.amount = self.df['amt']
    # self.params = kwargs
    def get_feature(self):
        if "param_n" in self.params:
            paraCCI_n = self.params.get("param_n")
        else:
            paraCCI_n = 20
        targetCCI = talib.CCI(self.high, self.low, self.close, timeperiod=paraCCI_n)
        return(targetCCI)

#%%  
if __name__ == '__main__':
    import os
    import numpy as np
    import statsmodels.api as sm
    os.chdir(path="../..")
    from util import custom_load_data
    DATA_PATH = 'C:\\Users\\eiahb\\Documents\\MyFiles\\PythonProject\\PHBS_machine_learning_for_finance\\PHBS_ML_for_quant_project\\00data'
    index_dict = {
        "windA":"881001.csv", 
        "hs300":"000300.csv"
    }
    
    test_index = "hs300"
    index_data = custom_load_data(os.path.join(DATA_PATH, index_dict[test_index]))
    klass = CCI(data = index_data, param_n = 20)
    features = klass.get_feature()
    print(features.head())
    features.hist()
    
    tomorrow_return = klass.rts.shift(-1)
    mask_features = np.ma.masked_invalid(features)
    mask_tomorrow_return = np.ma.masked_invalid(tomorrow_return)
    mask = mask_features.mask|mask_tomorrow_return.mask

    #相關係數
    print(np.corrcoef(mask_features[~mask], mask_tomorrow_return[~mask]))
    
    # 回歸
    model = sm.OLS(mask_tomorrow_return[~mask], sm.add_constant(mask_features[~mask]))
    results = model.fit()
    print(results.summary())

# %%
