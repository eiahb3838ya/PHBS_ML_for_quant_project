#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 19:00:16 2020

@author: Trista
"""
from FeatureEngineering import FeatureEngineering
from sklearn import preprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import linear_model,kernel_ridge
import math
from pylab import rcParams
plt.style.use('ggplot')
import matplotlib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import RFE,RFECV
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score
import xgboost as xgb
from sklearn.ensemble import GradientBoostingRegressor, \
    RandomForestRegressor, ExtraTreesRegressor,AdaBoostRegressor
from sklearn import tree

#%%helper function
def get_mape(y_true,y_pred):
    y_true,y_pred = np.array(y_true),np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred)/y_true)) * 100

#%%feature selection
def feature_selection(data,method = None):
    pre = data.iloc[:,:2]
    X = data.iloc[:,4:]
    y = data['rts'].shift(-1)
    features = np.array(pd.DataFrame(X.columns).T).tolist()
    
    #because y is shift -1
    X = X.iloc[:-1,:]
    y = y.iloc[:-1]
    
    #feature selection
    if method == 'GBDT':
        model = RFE(estimator = GradientBoostingRegressor(), n_features_to_select = 50)
        model.fit(X,y)
        coef = pd.DataFrame(model.support_,index = features,columns = ['fs_results'])
        ftselected = coef[coef['fs_results'] == True].copy()
        ftselected = pd.DataFrame(ftselected)
        ftselected = ftselected.reset_index()
        selectedFeaturesName = ftselected['level_0'].tolist()
        X = data.loc[:,selectedFeaturesName]
        sNum = len(selectedFeaturesName)
        eNum = len(coef)-len(selectedFeaturesName)
        print("GBDT picked " + str(sNum) + " variables and eliminated the other " +  str(eNum) + " variables")
        X = pd.merge(pre,X, left_index = True, right_index = True)
        return X
    elif method == 'LASSO':
         model_lasso = LassoCV(alphas = [0.000001,0.0001,0.001]).fit(X,y)
         model_lasso.alpha_
         model_lasso.coef_
         coef = pd.Series(model_lasso.coef_, index = X.columns)
         coef = pd.DataFrame(coef,index = coef.index,columns = ['fs_results'])
         ftselected = coef[coef['fs_results'] != 0].copy()
         selectedFeaturesName = ftselected.index.tolist()
         X = data.loc[:,selectedFeaturesName]
         print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
        
         matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
         coef.plot(kind = "barh")
         plt.title("Coefficients in the Lasso Model")
         plt.show() 
         X = pd.merge(pre,X, left_index = True, right_index = True)
         return X
    else:
        return data
    
#%%
'''train,validation,test'''
def model_preprocess(X):
    df = X
    df.dropna(axis = 1, how = 'all',inplace = True)
    df.fillna(0, inplace = True)
    test_size = 0.2
    cv_size = 0.2
    Nmax = 30
    num_cv = int(cv_size * len(X))
    num_test = int(test_size * len(X))
    num_train = len(X) - num_cv - num_test
    
    train = df[:num_train]
    cv = df[num_train:num_train + num_cv]
    train_cv = df[:(num_train + num_cv)]
    test = df[(num_train + num_cv) :]
    
    indicator = df.columns.tolist()
    indicator.remove('date')
    indicator.remove('rts')
    features = indicator.copy()
    target = ['rts']
    
    #get X and y
    X_train = train[features]
    quantile_transformer = preprocessing.QuantileTransformer(output_distribution = 'normal',
                                                             random_state = 0)
    X_train = quantile_transformer.fit_transform(X_train).copy()
    y_train = train[target]
    
    X_cv = cv[features]
    y_cv = cv[target]
    quantile_transformer = preprocessing.QuantileTransformer(output_distribution = 'normal',
                                                             random_state = 0)
    X_cv = quantile_transformer.fit_transform(X_cv).copy()

    X_test = test[features]
    y_test = test[target]
    quantile_transformer = preprocessing.QuantileTransformer(output_distribution = 'normal',
                                                             random_state = 0)    
    X_test = quantile_transformer.fit_transform(X_test).copy()
    return X_train,y_train,X_cv,y_cv,X_test,y_test,features,train,cv,test

#%% start build model
'''linear model with feature selection'''
def linear_model_fit(model,model_name,X_train,y_train,X_cv,y_cv,
                 X_test,y_test,features,train,cv,test):
    print('start build model!')
    #Constructor
    model.fit(X_train,y_train)
    
    #Corr
    if model_name == 'Linear_Regression' or 'Rdige_Regression':
        coef = pd.DataFrame(model.coef_,columns = features,index = ['features_importance'])
    elif model_name == 'RANSAC_Regression' or 'Nearest Neighbor Regression':
        coef = pd.DataFrame(model.score(X_train,y_train),columns = features, index = ['features_importance'])
    else:
        coef = pd.DataFrame(model.features_importance_,columns = features, index = ['features_importance'])
   
    model_coef = coef.T.copy()
    model_coef.sort_index(ascending = False, inplace = True)
    print(model_coef.head(10).round(4))
    
    #predict on train set
    y_estimation = model.predict(X_train)
    
    #RMSE
    print('%s' % model_name + '_in sample MSE = ' + str(mean_squared_error(y_train, y_estimation)))
    print('%s' % model_name + '_in sample RMSE = ' + str(math.sqrt(mean_squared_error(y_train, y_estimation))))
    print('%s' % model_name + '_in sample MAPE = ' + str(get_mape(y_train, y_estimation)))
    print('%s' % model_name + '_in sample R2 = ' + str(r2_score(y_train, y_estimation)))

    #Plot
    rcParams['figure.figsize'] = 10,8
    matplotlib.rcParams.update({'font.size': 14})
    
    est_df = pd.DataFrame({'est': y_estimation.T.tolist()[0],
                            'date': train['date']})
    
    ax = train.plot(x = 'date', y = 'rts', style = 'b-', grid = True)
    ax = cv.plot(x = 'date', y = 'rts', style = 'y-',grid = True, ax = ax)
    ax = test.plot(x = 'date', y = 'rts', style = 'g-', grid = True, ax = ax)
    ax = est_df.plot(x = 'date', y = 'est', style = 'r-',grid = True, ax = ax)
    ax.legend(['train','cv','test','est'])
    ax.set_xlabel('date')
    ax.set_ylabel('%')
    plt.show()
    
    #Predict on test set
    est_ = model.predict(X_test)
    
    #Calculate RMSE
    print('%s' % model_name + '_out sample MSE = ' + str(mean_squared_error(y_test, est_)))
    print('%s' % model_name + '_out sample RMSE = ' + str(math.sqrt(mean_squared_error(y_test, est_))))
    print('%s' % model_name + '_out sample MAPE = ' + str(get_mape(y_test, est_)))
    print('%s' % model_name + '_out sample R2 = ' + str(r2_score(y_test, est_)))
    
    rcParams['figure.figsize'] = 10, 8  # width 10, height 8
    matplotlib.rcParams.update({'font.size': 14})
    
    est_df_ = pd.DataFrame({'est': est_.T.tolist()[0],
                            'date': test['date']})

    # ax = train.plot(x='Date', y='return', style='b-', grid=True)
    # ax = cv.plot(x='Date', y='return', style='y-', grid=True, ax=ax)
    ax = test.plot(x='date', y='rts', style='g-', grid=True)
    ax = est_df_.plot(x='date', y='est', style='r-', grid=True, ax=ax)
    ax.legend(['test', 'predictions'])
    ax.set_xlabel("date")
    ax.set_ylabel("%")
    plt.show()
    return est_df, est_df_

'''model without feature selection'''
def model_fit(model, model_name,X_train, y_train,X_cv,y_cv,X_test, y_test, features_name,train,cv,test):

    # Train the regressor
    model.fit(X_train, y_train)
    # Do prediction on train set
    est = model.predict(X_train)
    # Calculate RMSE
    print('%s' % model_name + "_in-sample MSE = " + str(mean_squared_error(y_train, est)))
    print('%s' % model_name + "_in-sample RMSE = " + str(math.sqrt(mean_squared_error(y_train, est))))
    print('%s' % model_name + "_in-sample MAPE = " + str(get_mape(y_train, est)))
    print('%s' % model_name + "_in-sample R2 = " + str(r2_score(y_train, est)))
    # Plot adjusted close over time
    rcParams['figure.figsize'] = 10, 8  # width 10, height 8

    est_df = pd.DataFrame({'est': est,
                           'date': train['date']})

    ax = train.plot(x='date', y='rts', style='b-', grid=True)
    ax = cv.plot(x='date', y='rts', style='y-', grid=True, ax=ax)
    ax = test.plot(x='date', y='rts', style='g-', grid=True, ax=ax)
    ax = est_df.plot(x='date', y='est', style='r-', grid=True, ax=ax)
    ax.legend(['train', 'cv', 'test', 'est'])
    ax.set_xlabel("date")
    ax.set_ylabel("rts")
    plt.show()
    # Do prediction on test set
    est_ = model.predict(X_test)

    # Calculate RMSE
    print('%s' % model_name + "_out sample MSE = " + str(mean_squared_error(y_test, est_)))
    print('%s' % model_name + "_out sample RMSE = " + str(math.sqrt(mean_squared_error(y_test, est_))))
    print('%s' % model_name + "_out sample MAPE = " + str(get_mape(y_test, est_)))
    print('%s' % model_name + "_out sample R2 = " + str(r2_score(y_test, est_)))
    # Plot adjusted close over time
    rcParams['figure.figsize'] = 10, 8  # width 10, height 8
    matplotlib.rcParams.update({'font.size': 14})

    est_df_ = pd.DataFrame({'est': est_,
                            'date': test['date']})

    #ax = train.plot(x='Date', y='return', style='b-', grid=True)
    #ax = cv.plot(x='Date', y='return', style='y-', grid=True, ax=ax)
    ax = test.plot(x='date', y='rts', style='g-', grid=True)
    ax = est_df_.plot(x='date', y='est', style='r-', grid=True, ax=ax)
    ax.legend([ 'test', 'predictions'])
    ax.set_xlabel("date")
    ax.set_ylabel("rts")
    plt.show()
    return est_df, est_df_


if __name__ == '__main__':
    ROOT =  '/Users/mac/Desktop/ML_Quant/data'
    Klass = FeatureEngineering(ROOT)
    data = Klass.combine_feature()
    data = feature_selection(data,method ='GBDT')
    X_train,y_train,X_cv,y_cv,X_test,y_test,features,train,cv,test = model_preprocess(data)
    
    '''linear regression'''
    linreg = linear_model.LinearRegression()
    est_linreg,est_linreg_ = linear_model_fit(linreg, 'Linear_Regression',X_train, y_train,X_cv,y_cv,X_test, y_test, features,train,cv,test)

    '''ridge regression'''
    ridgereg = linear_model.Ridge()
    est_ridgereg,est_ridgereg_ = linear_model_fit(ridgereg, 'Rdige_Regression',X_train, y_train,X_cv,y_cv,X_test, y_test, features,train,cv,test)

    '''Bayesian Ridge Regression'''
    Bayesreg = linear_model.BayesianRidge()
    Bayesreg_model_fit = model_fit(Bayesreg, 'Bayesian_Ridge_Regression',X_train, y_train,X_cv,y_cv,X_test, y_test, features,train,cv,test)
    coef = pd.DataFrame(Bayesreg.coef_, index=features, columns=['features_importance'])
    coef.sort_index(ascending=False, inplace=True)
    print(coef.head(10).round(6))

    '''ARD Regression'''
    ardreg = linear_model.ARDRegression()
    ardreg_model_fit = model_fit(ardreg, 'ARD_Regression',X_train, y_train,X_cv,y_cv,X_test, y_test, features,train,cv,test)
    coef = pd.DataFrame(ardreg.coef_, index=features, columns=['features_importance'])
    coef.sort_index(ascending=False, inplace=True)
    print(coef.head(10).round(6))
    
    '''Decision Tree Regression'''
    treereg = tree.DecisionTreeRegressor()
    treereg_model_fit = model_fit(treereg, 'Decision Tree Regression',X_train, y_train,X_cv,y_cv,X_test, y_test, features,train,cv,test)
    coef = pd.DataFrame(treereg.feature_importances_, index=features, columns=['features_importance'])
    coef.sort_index(ascending=False, inplace=True)
    print(coef.head(10).round(3))
    
    '''Random Forest Regression'''
    rfref = RandomForestRegressor()
    rfref_model_fit = model_fit(rfref, 'Random Forest Regression',X_train, y_train,X_cv,y_cv,X_test, y_test, features,train,cv,test)
    coef = pd.DataFrame(rfref.feature_importances_, index=features, columns=['features_importance'])
    coef.sort_index(ascending=False, inplace=True)
    print(coef.head(10).round(3))
    
    '''AdoBoost Regression'''
    adgbreg = AdaBoostRegressor()
    adgbreg_model_fit = model_fit(adgbreg, 'AdoBoost Regression',X_train, y_train,X_cv,y_cv,X_test, y_test, features,train,cv,test)
    coef = pd.DataFrame(adgbreg.feature_importances_, index=features, columns=['features_importance'])
    coef.sort_index(ascending=False, inplace=True)
    print(coef.head(10).round(3))
    
    '''Extra Tree Regression'''
    extreg = ExtraTreesRegressor()
    extreg_model_fit = model_fit(extreg, 'Extra_Tree_Regression',X_train, y_train,X_cv,y_cv,X_test, y_test, features,train,cv,test)
    coef = pd.DataFrame(extreg.feature_importances_, index=features, columns=['features_importance'])
    coef.sort_index(ascending=False, inplace=True)
    print(coef.head(10).round(3))
    
    '''GBRT'''
    gbdtreg = GradientBoostingRegressor()
    gbdtreg_model_fit = model_fit(gbdtreg, 'GBRT',X_train, y_train,X_cv,y_cv,X_test, y_test, features,train,cv,test)
    coef = pd.DataFrame(gbdtreg.feature_importances_, index=features, columns=['features_importance'])
    coef.sort_index(ascending=False, inplace=True)
    print(coef.head(10).round(3))
    
    '''xgboost regression'''
    model_seed = 100
    n_estimators = 100
    max_depth = 3
    learning_rate = 0.1
    min_child_weight = 1

    # Create the model
    xgbreg = xgb.XGBRegressor(seed=model_seed,
                              n_estimators=n_estimators,
                              max_depth=max_depth,
                              learning_rate=learning_rate,
                              min_child_weight=min_child_weight)

    xgbreg_model_fit = model_fit(xgbreg, 'XGBoost',X_train, y_train,X_cv,y_cv,X_test, y_test, features,train,cv,test)
    coef = pd.DataFrame(xgbreg.feature_importances_, index=features, columns=['features_importance'])
    coef.sort_index(ascending=False, inplace=True)
    print(coef.head(10).round(3))
   