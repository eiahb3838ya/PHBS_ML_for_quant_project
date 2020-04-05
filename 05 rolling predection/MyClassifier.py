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
from parametersRepo import *

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

#%% NeuralNetwork
class MyNeuralNetworkClassifier:
    def __init__(self):
        self.parameter = self.getPara()
        self.model = MLPClassifier(solver = self.parameter['solver'],
                                   alpha = self.parameter['alpha'],
                                   hidden_layer_sizes = self.parameter['hidden_layer_sizes'],
                                   random_state = self.parameter['random_state'])
        
    def getPara(self):
        # do some how cv or things to decide the hyperparameter
        return(paraNeuralNetwork)
        
    def fit(self, X, y):
        # do what ever plot or things you like 
        # just like your code
        return(self.model.fit(X, y))
        
    def predict(self, X):
        return(self.model.predict(X))
    
#%%Perceptron
from sklearn.linear_model import Perceptron
class MyPerceptronClassifier:
    def __init__(self):
        self.parameter = self.getPara()
        self.model = Perceptron(tol = self.parameter['tol'],
                                random_state = self.parameter['random_state'])
        
    def getPara(self):
        # do some how cv or things to decide the hyperparameter
        return(paraPerceptron)
        
    def fit(self, X, y):
        # do what ever plot or things you like 
        # just like your code
        return(self.model.fit(X, y))
        
    def predict(self, X):
        return(self.model.predict(X))
    

