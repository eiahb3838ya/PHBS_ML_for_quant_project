这个function是用来衡量回测过拟合的概率。

之前的Machine Learning调参是为了衡量训练过拟合的概率。

需要的输入是一张data*n（n为策略的参数组合的个数）的表。目前这里的testData只是为了测试code。

这里的n，比如像xgboost，应该会有很多种排列组合，那么把参数都测了一下以后，可以保存每种可能的收益率。

会输出PBO（probability of overfitting），回测过拟合的概率，认为回测过拟合概率越小越好。

首先会计算lambda，再会计算PBO，PBO认为<0.5是比较合适的timing strategy。

参考华泰的研报case3：双均线 50ETF 择时策略，该策略遍历了参数组合，结论是择时策略大概率是一种回测过拟合。

