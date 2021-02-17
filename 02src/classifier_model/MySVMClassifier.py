#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 01:40:36 2020

@author: Trista
"""
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

class MySVMClassifier:
    def __init__(self):
        self.parameter = None
        self.model = None
        
    def getPara(self):
        # do some how cv or things to decide the hyperparameter
        if self.parameter == None:
            print('Hi~ please first use fit function to get parameter :)')
        else:
            print('haha! We already do CV and find the best parameters~')
            return self.parameter
        
    def fit(self, X, y):
        # do what ever plot or things you like 
        # just like your code
        # Set the parameters by cross-validation
        tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3],
                             'C': [0.01, 0.1, 1]},]
        scores = ['precision']
        
        for score in scores:
            print("# Tuning hyper-parameters for %s" % score)
            clf = GridSearchCV(
                SVC(), tuned_parameters, scoring='%s_macro' % score
            )
            clf.fit(X, y)
            print("Best parameters set found on development set:")
            self.parameter = clf.best_params_
            self.model = clf
            print(clf.best_params_)
            print("Grid scores on development set:")
            means = clf.cv_results_['mean_test_score']
            stds = clf.cv_results_['std_test_score']
            for mean, std, params in zip(means, stds, clf.cv_results_['params']):
                print("%0.3f (+/-%0.03f) for %r"
                      % (mean, std * 2, params))
        
            print("Detailed classification report:")
            print("The model is trained on the full development set.")
            print("The scores are computed on the full evaluation set.")
        return(self.model.fit(X, y))
        
    def predict(self, X):
        return(self.model.predict(X))