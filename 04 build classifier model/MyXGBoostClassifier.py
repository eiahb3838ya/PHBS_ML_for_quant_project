#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 18:06:19 2020

@author: Trista
"""
from xgboost import XGBClassifier
from parametersRepo import *

class MyXGBoostClassifier:
    def __init__(self):
        self.parameter = self.getPara()
        self.model =  XGBClassifier(seed=self.parameter['model_seed'],
                                    n_estimators=self.parameter['n_estimators'],
                                    max_depth=self.parameter['max_depth'],
                                    learning_rate=self.parameter['learning_rate'],
                                    min_child_weight=self.parameter['min_child_weight'])
        
    def getPara(self):
        # do some how cv or things to decide the hyperparameter
        # n_neighbors = 15
        # weights = 'uniform'
        return(paraXGBoost)
        
    def fit(self, X, y):
        # do what ever plot or things you like 
        # just like your code
        return(self.model.fit(X, y))
        
    def predict(self, X):
        return(self.model.predict(X))
    
if __name__ == '__main__':
    test = MyXGBoostClassifier()
    test.fit(X_train,y_train)
    test.model.score(X_train,y_train)
    test.model.score(X_test,y_test)
