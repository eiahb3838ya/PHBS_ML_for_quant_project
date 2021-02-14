# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 13:26:29 2020

@author: alfre
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from parametersRepo import *

class MyDecisionTreeClassifier:
    def __init__(self):
        self.parameter = self.getPara()
        self.model = DecisionTreeClassifier(max_depth = self.parameter['max_depth'])
        
    def getPara(self):
        # do some how cv or things to decide the hyperparameter
        # return dict
        return(paraDecisionTree)
        
    def fit(self, X, y):
        # do what ever plot or things you like 
        # just like your code
        return(self.model.fit(X, y))
        
    def predict(self, X):
        return(self.model.predict(X))
    
if __name__ == '__main__':
    
    test = MyDecisionTreeClassifier()
    test.fit(X_train,y_train)
    test.model.score(X_train,y_train)
    test.model.score(X_test,y_test)