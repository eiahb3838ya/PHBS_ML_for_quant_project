# -*- coding: utf-8 -*-
"""
Created on Sat Apr 4 21:57:19 2020

@author: alfre
"""

from sklearn import neighbors

class MyKNNClassifier:
    def __init__(self):
        # self.parameter = self.getPara()
        self.model = neighbors.KNeighborsClassifier(n_neighbors = 3)
        
    def getPara(self):
        # do some how cv or things to decide the hyperparameter
        
        # n_neighbors = 15
        # weights = 'uniform'
        return(paraKNN)
        
    def fit(self, X, y):
        # do what ever plot or things you like 
        # just like your code
        return(self.model.fit(X, y))
        
    def predict(self, X):
        return(self.model.predict(X))