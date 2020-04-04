#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 00:58:02 2020

@author: Trista
"""

# from sklearn import preprocessing
import numpy as np
import pandas as pd
from keras.layers.recurrent import LSTM
from keras.layers import  Dropout
from keras import models,layers
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from keras.models import Sequential
from FeatureEngineering import FeatureEngineering
import os

def split_train_test_data(X,y,test_size):
    num_train = int(len(X) - len(X) * test_size)
    X_train = X.iloc[:num_train,:]
    X_test = X.iloc[num_train:,:]
    y_train = y[:num_train]
    y_test = y[num_train:]
    return X_train,y_train,X_test, y_test

ROOT =  '/Users/mac/Desktop/ML_Quant/data'
rawDf = pd.read_pickle(os.path.join(ROOT, 'cleanedFactor.pkl'))
getFeatures = FeatureEngineering(ROOT)
features = getFeatures.combine_feature()
rawDf = pd.merge(features,rawDf,on = 'date')
rawXs, rawYs = rawDf.iloc[:, :-4], rawDf.iloc[:, -1].astype(bool)
X_train,y_train,X_test, y_test = split_train_test_data(rawXs,rawYs,test_size = 0.3)

#%% for lstm data imput
# X_train = X_train.values
# X_test = X_test.values
# y_train = y_train.values
# y_test = y_test.values

# X_train = X_train.reshape (X_train. shape + (1,)) 
# X_test = X_test.reshape(X_test.shape + (1,))

#%%
from keras.models import Sequential
from keras.layers import Dense, Dropout
# tf.keras.backend.clear_session()
model = models.Sequential()
# model.add(LSTM(30, input_dim=X_train.shape[-1],input_length=55,
#                activation='relu',return_sequences=True))
model.add(layers.Dense(30,activation = 'relu',input_shape=(55,)))
model.add(Dropout(0.1))
model.add(layers.Dense(1,activation = 'sigmoid' ))
model.summary()

model.compile( loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'] )
history = model.fit( x=X_train, y=y_train, validation_split=0.2, epochs=50, batch_size=10, verbose=2)

train_result = model.evaluate(X_train,y_train)
test_result = model.evaluate(X_test,y_test)

print('TRAIN Accuracy:',train_result[1])
print('TEST Accuracy:',test_result[1])

'''evalution model'''
import matplotlib.pyplot as plt

def plot_metric(history, metric):
    train_metrics = history.history[metric]
    val_metrics = history.history['val_'+metric]
    epochs = range(1, len(train_metrics) + 1)
    plt.plot(epochs, train_metrics, 'bo--')
    plt.plot(epochs, val_metrics, 'ro-')
    plt.title('Training and validation '+ metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["train_"+metric, 'val_'+metric])
    plt.show()

plot_metric(history,"loss")
plot_metric(history,"accuracy")

model.evaluate(x = X_test,y = y_test)

#predict class
model.predict_classes(X_test[0:10])

#save model
model.save('/Users/mac/Desktop/ML_Quant/06 LSTM model/keras_model.h5')  
# del model  
# identical to the previous one
model = models.load_model('/Users/mac/Desktop/ML_Quant/06 LSTM model/keras_model.h5')
model.evaluate(X_test,y_test)