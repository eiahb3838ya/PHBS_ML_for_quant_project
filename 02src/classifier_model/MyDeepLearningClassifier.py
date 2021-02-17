#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 00:17:21 2020

@author: Trista
"""
from parametersRepo import *
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import models,layers
import pandas as pd

class MyDeepLearningClassifier:
    def __init__(self):
        self.parameter = None
        self.model = None
        
    def getPara(self):
        # do some how cv or things to decide the hyperparameter
        # return dict
        if self.parameter == None:
            print('Hi~ please first use fit function to get model :)')
        else:
            print('haha! We already trained deepLearning Model~')
            return self.parameter
        return self.parameter
        
    def fit(self, X, y):
        # do what ever plot or things you like 
        # just like your code
        self.parameter = len(X.columns)
        model = models.Sequential()
        model.add(Dense(30,activation = 'relu',input_shape=(len(X.columns),)))
        model.add(Dropout(0.1))
        model.add(Dense(1,activation = 'sigmoid' ))
        # model.summary()
        model.compile( loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'] )
        self.model = model
        return(self.model.fit(X, y,
                              validation_split=0.2, 
                              epochs=1, batch_size=10, verbose=2))
        
    def predict(self, X):
        return(pd.Series(self.model.predict_classes(X).flatten()).astype(bool))

if __name__ == '__main__':
    test = MyDeepLearningClassifier()
    test.fit(X_train,y_train)
    test.predict(X_test)
    # test.model.score(X_train,y_train)
    # test.model.score(X_test,y_test)
    # test.model.predict(X_test)
