# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 16:33:40 2020

@author: Evan
"""

from sklearn import linear_model

class MyLogisticRegClassifier:
    def __init__(self):
        self.parameter = self.getPara()
        self.model = LogisticRegression(random_state=0)
        
    def getPara(self):
        # do some how cv or things to decide the hyperparameter
        return({})
        
    def fit(self, X, y):
        # do what ever plot or things you like 
        # just like your code
        return(self.model.fit(X, y))
        
    def predict(self, X):
        return(self.model.predict(X))