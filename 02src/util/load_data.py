# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 16:20:26 2020

@author: alfre
"""
#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
plt.style.use('ggplot')

#%%
def custom_load_data(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
    df.columns = df.iloc[2, :].rename("")
    df = df.iloc[3:, :]
    df = df.rename(columns={'Date': 'date'})
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    df = df.set_index('date')
    df= df.astype(float)
    df['return'] = df['close'] / df['pre_close'] - 1
    return df

#%%
def plot_rts(df):
    plt.figure(figsize=(24, 12))
    plt.plot(df['return'], '-o', ms=3)
    plt.title('Daily return ', fontsize=20)
    plt.figure(figsize=(24, 12))
    plt.hist(df['return'])
    plt.figure(figsize=(24, 12))
    cumRts = np.cumprod(df['return']+1)/(df['return'].iloc[0]+1)
    plt.plot(cumRts)
    plt.title('Cum daily return .', fontsize=20)

#%%
if __name__ == '__main__':
    DATA_PATH = 'C:\\Users\\eiahb\\Documents\\MyFiles\\PythonProject\\PHBS_machine_learning_for_finance\\PHBS_ML_for_quant_project\\00data\\881001.csv'
    df = custom_load_data(DATA_PATH)
    plot_rts(df)
