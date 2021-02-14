#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 12:06:25 2020

@author: Trista

This .py contain all the parameters we need for our base classifier model
"""
#Decision Tree
paraDecisionTree = {'max_depth':8}

#KNN
paraKNN = {'n_neighbors':15,
           'weights':'uniform'}

#NeuralNetwork
paraNeuralNetwork = {'solver':'lbfgs',
                     'alpha':1e-5,
                     'hidden_layer_sizes':(5,2),
                     'random_state':1}

#Perceptron
paraPerceptron = {'tol':1e-3,
                  'random_state':0}

paraXGBoost = {'model_seed':100,
               'n_estimators':100,
               'max_depth':3,
               'learning_rate':0.1,
               'min_child_weight':1}