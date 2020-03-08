# Paper research: short-term markter timing strategy based on boosting ML algos
Member: Evan HU, Alfred CHEN, Trista FANG, Loralei WANG

Date: 2020.03

**😄first ddl:2020.03.01  aim:finish the draft about proposal**


---


# **Phase 1**
1. **Introduction**
1. **dataset desctiption**

According to the research report of Industrial Securities, we choose following factors, most of which can be acquired from Wind/Choice Database directly（Denoted by D in the table）.All factors are based on daily frequency data.

| 银行间同业拆借加权利率：1天 | Interbank Offered Rate: 1 day | D | 
|:----:|:----:|:----:|
| 银行间同业拆借加权利率：1天：过去五天的变化率 | Interbank Offered Rate: 1 day: Change rate in   the past five days | D | 
| 7天期回购利率 | 7-day repo rate | D   | 
| 7天期回购利率：过去五天的变化率 | 7-day repo rate: Change rate in the past five days | D | 
| 银行间回购加权利率：7天 | Interbank repo weighted interest rate: 7 days | D | 
| 银行间回购加权利率：7天：过去五天的变化率 | Interbank repo weighted interest rate: 7 days: Change rate in the past five days | D | 
| shibor利率（0N） | shibor（0N） | D | 
| shibor利率（1W） | shibor（1W） | D | 
| shibor利率（2W） | shibor（2W） | D  | 
| shibor利率（1M） | shibor（1M） | D | 
| shibor利率（3M） | shibor（3M） | D | 
| shibor利率（6M） | shibor（6M） | D | 
| shibor利率（0N）：过去五天的变化率 | shibor（0N）: Change rate in the past five days | D | 
| shibor利率（1W）：过去五天的变化率 | shibor（1W）: Change rate in the past five days | D | 
| shibor利率（2W）：过去五天的变化率 | shibor（2W）: Change rate in the past five days | D  | 
| shibor利率（1M）：过去五天的变化率 | shibor（1M）: Change rate in the past five days | D  | 
| shibor利率（3M）：过去五天的变化率 | shibor（3M）: Change rate in the past five days | D | 
| shibor利率（6M）：过去五天的变化率 | shibor（6M）: Change rate in the past five days | D | 
| 中债国债到期收益率（0年） | Government Bond YTM（0Y） | D | 
| 中债国债到期收益率（0年）：过去五天的变化率 | Government Bond YTM（0Y）: Change rate in the past five days | D | 
| 中债国债到期收益率（3年） | Government Bond YTM（3Y） | D | 
| 中债国债到期收益率（3年）：过去五天的变化率 | Government Bond YTM（3Y）: Change rate in the past five days | D | 
| 南华指数 | NHCI | D | 
| 南华指数：过去五天的变化率 | NHCI: Change rate   in the past five days | D | 
| CRB现货指数：综合 | CRB | D | 
| CRB现货指数：综合：过去五天的变化率 | CRB: Change rate in the past five days | D | 
| 期货收盘价（连续）：COMEX黄金 | Futures closing price (continuous): COMEX Gold | D | 
| 期货收盘价（连续）：COMEX黄金：过去五天的变化率 | Futures closing price (continuous): COMEX Gold: Change   rate in the past five days | D | 
| 期货结算价（连续）：WTI原油 | Futures settlement price (continuous): WTI Crude Oil | D | 
| 期货结算价（连续）：WTI原油：过去五天的变化率 | Futures settlement price (continuous): WTI Crude Oil: Change rate in the past five days | D | 
| COMEX黄金/WTI原油 | COMEX Gold/ WTI Crude Oil | D | 
| COMEX黄金/WTI原油：过去五天的变化率 | COMEX Gold/ WTI Crude Oil: Change rate in the   past five days | D | 
| 标普500 | S & P 500 | D | 
| 标普500：过去五天的变化率 | S & P 500: Change rate in the past five days | D | 
| 市场动量指标RSI | Market momentum indicator | RSI=Sum(Max(Close-LastClose,0),N,1)/Sum(ABS(Close-LastClose),N,1)*100 | 
| 市场动量指标：过去五天的收益率 | Market momentum indicator: Change rate in the past five days | D | 
| 市场交易活跃指标（成交量） | Volume | D | 
| 市场交易活跃指标：过去五天成交量的变化率 | Volume: Change rate in the past five days | D | 
| Beta分离度指标 | Beta resolution index | beta is calculated by CAPM, then calculate the difference between 90% percentile and 10% percentile of beta   | 
| Beta分离度指标：过去五天的变化率 | Beta resolution index: Change rate in the past five days | D | 
| 50ETF过去60日的波动率 | 50ETF volatility over the past 60 days | D | 
| 50ETF过去60日的波动率：过去五天的变化率 | 50ETF volatility over the past 60 days: Change   rate in the past five days | D | 
| 50ETF过去120日的波动率 | 50ETF volatility over the past 60 days | D | 
| 50ETF过去120日的波动率：过去五天的变化率 | 50ETF volatility over the past 60 days: Change rate in the past five days | D | 


1. **data cleaning details**
  1. Data fromats, structures
design a process for data cleaning. Remove NA values and make the format easy to slice according to time. Use dict and pandas dataframe to design our structure.
  2. Data Normalization
design a class for normalization work, which provides method including loading , calculating, saving data according to parameters given. At least implement two kinds of normalization, including min-max, z-score normalization.

1. **explora and analysis data**
  1. **Visualiztion**
to check if our data follow required statistical assumptions, we will visualize our data using seaborn or other tools. Draw a heat map to check the corr_coef of each factors. 
  2. **Feature selection**
to check which factors have better prediction power. We will apply feature selection methods including Cross Entropy, information gain, gini coef, LASSO. Draw the graph for each factor accordingly 
  3. **Decomposition (optional)**
We can try PCA method to avoid dimension disaster, pick the top 5, 10 vectors as our feature to input.
1. **single model(decision tree timing model)**

**a. ****D****ecision tree for prediction**

This paper use decision tree model - CART (classification and regression tree) - mainly as classification tools. We consider use either Gini coefficient or information entropy as inpurity level of the data to establish the decision tree. The advantages of using decision tree models include the ability to combine multiple factors' information, the ability to fit unlinear factors and the abilty to automatically select factors with strong predicting abilities. 

**b. ****Classification**

we use the daily data of the 51 selected factors as input to train the decision tree on the rise/fall of 50ETF in the previous 1800 days to pick some of the best classifying factors. Then we use the trained model on the test data (which is next day right after the train data) to predict whether 50ETF would rise or fall. If it is predicted to rise, then we long 50ETF at the end of today; if it is predicted to fall, then we short it at the end of today. We assume we already have had a position on the first day of the trading.

**c. ****Model updates**

We maintain the same trained model parameters for 20 consecutive trading days, and then retrain the model use the previous data plus the new 20 days' data following the above procedures, and so back and forth every 20 consecutive trading days. This way we can adjust the model with the latest information and more data.

**d. ****K-fold validation for hyperparameters**

This paper implements a 5-fold validation on every train dataset to select the model with best performance. The hyperparameter contains two parameters, namely the inpurity level of the decision trees and the depth of the decision trees. As mentioned above, the inpurity level can be measured by either Gini coefficient or entropy. And the depth of the decision tree can be set as 5, 10, 15, ..., 30 layers respectively.

**e. Maybe try other model****s**

Though the paper chooses decision tree as the basic classifier in AdaBoost algorithm, there may be some other algorithms to use for classification. We list some as follows so that we might try to implement and test their performance in the project:

**Logistic regression:**

Logistic regression is easy to understand, easy to implement and adjust and really efficient. But it cannot solve unlinear problems and may cause overfitting.

**Support Vector Machine(SVM):**

The decision function of SVM is determined by a small amount of support vectors so that it avoids the curse of demensionality. SVM has great generalization capabilities. It can also solve unlinear or high-dimensional problems. SVM also has impressive robustness. But it consumes large quantities of computation and relies heavily on fine-tuning.

**Naive Bayes:**

Naive Bayes algorithm has very stable classification efficiency and very robust. But the requirements of independent attributes is rare in practice so there might be some errors.

**KNN:**

KNN can be used for unlinear classification and the result is accurate. The time complexity of KNN is O(n). However, the computation is huge and requires large storage.


---


# **Phase 2**
1. **Booosting method **
### a. Adaboost  method to optimize result
Decision timing model 用多个单层的决策树，不容易过拟合，它比较简单，无法充分利用因子信息和捕捉因子的非线性特征。Boosting方法是ensemble method的一种，在实践中具有广泛应用。AdaBoost算法（Adaptive Boosting）作为其中一种boosting方法的实现，可以解决这些问题。AdaBoost的自适应性体现在在当前基分类器分类错误的样本权重会增大，而正确分类的样本权重会减小，从而在训练下一个基分类器时会着重拟合之前分类错误的样本。

Adaboost model的特点：

1)   多个弱分类器esemble，具有较高的精度

2）不容易出现过拟合，模型比较稳健

3）可以处理非线性因子

4）不需要人工进行feature selection，可以在模型中加入大量的因子

Adaboost model的算法：

we have N samples(x1,y1),(x2,y2),...(xn,yn),x is feature, y 是类别。假定基分类器的数量设定为M个多Adaboost算法步骤如下：

1）Initial sample weight，wi = 1/N，i=1,2,3...,N

2)  for i in range(0,M):

  * 利用样本权重训练一个基分类器G_m(x)
  * 计算基分类器的样本加权错误率：![图片](https://uploader.shimo.im/f/WWy9hVgtVJk9CoiK.png!thumbnail)
  * 计算基分类器的信心度：![图片](https://uploader.shimo.im/f/0pbNlzHbjA0DF89i.png!thumbnail)
  * 根据信心度更新样本权重：![图片](https://uploader.shimo.im/f/JKfDMn48DYMrTseW.png!thumbnail)

3）结合M个基分类器得到最终分类器：![图片](https://uploader.shimo.im/f/J4xQwo4gm2Qvjk5t.png!thumbnail)

我们利用xx种日频因子数据构造基于决策树的Adaboost分类器，对下一个交易日的wind全A指数涨（+1）跌（-1）做出预测。将基分类器决策树的参数设置为步骤 5.single model（decision tree timing model）。为提高模型的泛化能力，我们将决策树的深度设置为1，一个根节点和两个叶节点。在这种设置下，Adaboost的每一个基分类器都会选择xx个因子中的某一个作为决策树节点来对下一交易日wind全A的涨跌做出预测。下一个交易日的最终预测结果将由所有基分类器共同决定。

training➡️弱分类器➡️调整权重➡️强分类器

### b. Try with different base classilfier  in Adaboost method to test 
a.的Adaboost的基分类器树用决策树实现的。这一个部分将原来以decision tree类的基分类器换成时别的ML method下的基分类器。

在这个阶段我们尝试使用不同的base classifier model来作为Adaboost method的基分类器。logistic regision，svm，感知器，...etc。这个的work proof在于调每个基分类器的超参数。

基分类器中可以用regression类的model，如果预测出下一交易日的wind全A指数比today的高，那么就transform为1，否则就为-1。

### c.  Trianing model and predict trading signal
测试的数据来源是xxxx到xxxx，使用扩展窗口法（expanding window）来训练和交叉验证模型并发出交易信号。

1）training set at least 1800 days, using k-Fold(k=5) to do cross validation in order to select most highest accuracy hyperpramater to fit model. 超参数的搜索方式如下：

  * 决策树的不纯度（cross entropy，gini coef）
  * Adaboost 基分类器的数量（20，25，30，40）

2）利用训练完的最优模型来预测wind全A指数下一交易日的涨跌，实施timing的交易策略

  * long short strategy：如果预测下一日信号为涨，则假设在当日收盘前5min买入wind全A；如果预测下一日信号为跌，则在当日收盘前5min做空wind全A。
  * long strategy：如果预测下一日信号为涨，则假设在当日收盘前5min买入wind全A；如果预测下一日信号为跌，则当日不操作，即继续持仓

3）连续使用2）中的交易模型20 days后继续进入1），并将此20个交易日的数据加入到原来的training set中。

![图片](https://uploader.shimo.im/f/oMcfPNiwXBEl4Y68.png!thumbnail)
### d.  Adaboost backtest result
    首先为了测试Adaboost model能否产生交易信号，假设在换仓当日无交易成本，给出Decision tree 下的Adaboost timing model的long short strategy result 和 pure long strategy result。给出其他基分类器下的Adaboost timing model。baseline model 设置成 简单holding strategy的表现。

  * 报告多空策略【年化收益，年化波动率，换仓次数，sharp ratio】
  * 报告纯多头策略【年化收益，年化波动率，换仓次数，sharp ratio】

【图】Decision tree类基分类器下的Adaboost timing model 策略净值

（3条，long short，pure long，simple holding）

【图】其他基分类器下的Adaboost timing model 策略净值

（3条，long short，pure long，simple holding）

if 发现了某种Adaboost model确实比较有效后，调整timing model的交易策略：在换仓信号发出时，次日开盘进入新的仓位（T0策略，因为有底仓）;加入单边万5的交易成本。

新的情况下，New Adaboost timing model的表现有所下降（合理），但还是比pure long timing model好（合理，因为是多空2边 vs 多一边）和简单holidng model好。

【图】New Adaboost timing model 策略净值

（3条，long short，pure long，simple holding）

1. ** Xgboost**

**waiting for Robert.**

