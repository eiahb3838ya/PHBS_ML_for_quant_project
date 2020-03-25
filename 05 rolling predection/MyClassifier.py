#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 21:37:31 2020

@author: Trista
"""
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn import neighbors
from sklearn.neural_network import MLPClassifier


#%% LogisticRegression
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

#%% NaiveBayesClassifier.
class MyNaiveBayesClassifier:
    def __init__(self):
        self.parameter = self.getPara()
        self.model = GaussianNB()
        
    def getPara(self):
        # do some how cv or things to decide the hyperparameter
        return({})
        
    def fit(self, X, y):
        # do what ever plot or things you like 
        # just like your code
        return(self.model.fit(X, y))
        
    def predict(self, X):
        return(self.model.predict(X))
    
#%% KNNClassifier
class MyKNNClassifier:
    def __init__(self):
        self.parameter = self.getPara()
        self.model = neighbors.KNeighborsClassifier(n_neighbors = self.parameter['n_neighbors'],
                                                    weights = self.parameter['weights'])
        
    def getPara(self):
        # do some how cv or things to decide the hyperparameter
        n_neighbors = 15
        weights = 'uniform'
        
        return({'n_neighbors':n_neighbors,
                'weights':weights})
        
    def fit(self, X, y):
        # do what ever plot or things you like 
        # just like your code
        return(self.model.fit(X, y))
        
    def predict(self, X):
        return(self.model.predict(X))
    
#%%
class MyNeuralNetworkClassifier:
    def __init__(self):
        self.parameter = self.getPara()
        self.model = MLPClassifier(solver = self.parameter['solver'],
                                   alpha = self.parameter['alpha'],
                                   hidden_layer_sizes = self.parameter['hidden_layer_sizes'],
                                   random_state = self.parameter['random_state'])
        
    def getPara(self):
        # do some how cv or things to decide the hyperparameter
        return({'solver':'lbfgs',
                'alpha':1e-5,
                'hidden_layer_sizes':(5,2),
                'random_state':1})
        
    def fit(self, X, y):
        # do what ever plot or things you like 
        # just like your code
        return(self.model.fit(X, y))
        
    def predict(self, X):
        return(self.model.predict(X))
    

