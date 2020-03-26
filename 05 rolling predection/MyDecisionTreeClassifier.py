# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 13:26:29 2020

@author: Evan
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

class MyDecisionTreeClassifier:
    def __init__(self):
        self.parameter = self.getPara()
        self.model = DecisionTreeClassifier()
        
    def getPara(self):
        # do some how cv or things to decide the hyperparameter
        return({})
        
    def fit(self, X, y):
        # do what ever plot or things you like 
        # just like your code
        return(self.model.fit(X, y))
        
    def predict(self, X):
        return(self.model.predict(X))
    