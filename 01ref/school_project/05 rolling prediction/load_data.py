# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 16:20:26 2020

@author: alfre
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
plt.style.use('ggplot')

def load_data(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
    df.columns = df.iloc[2,:]
    df = df.iloc[3:,:]
    df.reset_index()
    # df.set_index(['Date'], inplace=True)  
    df = df.rename(columns = {'pct_chg':'return','Date':'date'})
    df = df.iloc[242:,:]
    df['date'] = pd.to_datetime(df['date'],format = '%Y-%m-%d')
    df.iloc[:,1:] = df.iloc[:,1:].astype(float)
    df['rts'] =df['close'] / df['pre_close'] - 1
    df = df.reset_index(drop = True)
    df.fillna(method = 'ffill')
    return df

def plot_rts(df):
    slice_data = df[['date','rts']]
    slice_data = slice_data.reset_index(drop = False)
    plt.figure(figsize = (24, 12))
    plt.plot(df['date'],df['rts'],'-o', ms = 3)
    plt.title('This is daily return of WindA.',fontsize = 20)
    
    plt.figure(figsize = (24 ,12))
    cumRts = np.cumprod(df['rts']+1)/(df['rts'].iloc[0]+1)
    plt.plot(df['date'],cumRts)
    plt.title('This is cum daily return of windA.',fontsize = 20)  

if __name__ == '__main__':
    DATA_PATH = '/Users/mac/Desktop/ML_Quant/data'
    df = load_data(DATA_PATH + '/881001.csv')
    plot_rts(df)