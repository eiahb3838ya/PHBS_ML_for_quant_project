# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 12:56:45 2020

@author: Evan
"""

def naiveSelection(X_train, y_train, X_test, y_test, method = None, returnCoef = False):
    
    '''
    choose the model type with method
    fit any feature_selection model with the X_train, y_train
    transform the X_train, X_test with the model
    do not use the X_test to build feature selection model
    
    return the selected X_train, X_test
    print info of the selecter
    return the coef or the score of each feature if asked
    
    '''
    if not returnCoef:
        return(X_train, X_test)
    else:
        return(X_train, X_test, pd.Series())