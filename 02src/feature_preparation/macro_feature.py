# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# fixme:逆回購、淨回籠
# %%
def getPctChange(df, days = 5):
    df_pctChange5 = df.pct_change(days)
    df_pctChange5.columns = map(lambda x: x+'_pctChange5', df_pctChange5.columns )
    return(df_pctChange5)

def load_macro_feature():
    return(pd.read_csv(os.path.join(dataDir, 'macro_feature.csv'), index_col=0))

# %%
dataDir =("C:\\Users\\eiahb\\Documents\\MyFiles\\PythonProject\\PHBS_machine_learning_for_finance\\PHBS_ML_for_quant_project\\00data\\macro_feature")
# %%
rawRateDf = pd.read_csv(os.path.join(dataDir, '利率类20070901-20200308.csv'),engine='python',encoding='utf8', skiprows=3,dtype = str)
rawCommodityDf = pd.read_csv(os.path.join(dataDir, '商品类20070901-20200308.csv'),engine='python',encoding='utf8', skiprows=3,dtype = str)
rawStockIndexDf = pd.read_csv(os.path.join(dataDir, '股票指数类 20070901-20200308.csv'),engine='python', sep='\t', encoding='utf8',dtype = str)
rawWindADf = pd.read_csv(os.path.join(dataDir, '用wind全A算市场动量和交易活跃指标.csv'),engine='python',dtype = str)


# %%
rateDf = rawRateDf
rateDf = rateDf.rename(columns = {"指标名称":"date"})
rateDf = rateDf.set_index('date')
rateDf = rateDf.astype(float)
rateDf.index = pd.DatetimeIndex(rateDf.index)
rateDf_pctChange5 = getPctChange(rateDf)
rateDf_pctChange5.head(10)


# %%
commodityDf = rawCommodityDf
commodityDf = commodityDf.rename(columns={'指标名称':'date'})
commodityDf = commodityDf.set_index('date')
commodityDf = commodityDf.applymap(lambda x: x.replace(',', ''))
commodityDf = commodityDf.astype(float)
commodityDf.index = pd.DatetimeIndex(commodityDf.index)
commodityDf['COMEX黄金/WTI原油'] = commodityDf['期货收盘价(连续):COMEX黄金']/commodityDf['期货结算价(连续):布伦特原油']
commodityDf.loc[(commodityDf['期货收盘价(连续):COMEX黄金']==0) & (commodityDf['期货结算价(连续):布伦特原油']==0), 'COMEX黄金/WTI原油']=0

commodityDf_pctChange5 = getPctChange(commodityDf)
commodityDf_pctChange5.head(10)


# %%
stockIndexDf = rawStockIndexDf
stockIndexDf = stockIndexDf.set_index('date')
stockIndexDf = stockIndexDf.astype(float)
stockIndexDf.index = pd.DatetimeIndex(stockIndexDf.index)
stockIndexDf_pctChange5 = getPctChange(stockIndexDf)
stockIndexDf_pctChange5.head(10)


# %%
windADf = rawWindADf.set_index('date')
windADf = windADf.astype(float)
windADf.index = pd.DatetimeIndex(windADf.index)
windAReturn = windADf['close'].pct_change().rename('windAReturn')
windATomorrowUp = (windAReturn.shift(-1)>0).rename('windATomorrowUp').astype(int)


# %%
mktVolume = windADf['volume'].rename('mktVolume')
mktVolume_pctChange5 = mktVolume.pct_change(5).rename('mktVolume_pctChange5')
mktClose_pctChange5 = windADf['close'].pct_change(5).rename('mktClose_pctChange5')
mktClose_pctChange5
mktMomentumDf = pd.concat([mktVolume, mktVolume_pctChange5, mktClose_pctChange5], axis = 1)
mktMomentumDf.head()


# %%
output_df = pd.concat([rateDf, rateDf_pctChange5,
                commodityDf, commodityDf_pctChange5,
                stockIndexDf, stockIndexDf_pctChange5,
                mktMomentumDf
               ], axis = 1)

output_df.describe()
output_df.to_csv(os.path.join(dataDir, 'macro_feature.csv'))

# %%
if __name__ == "__main__":
    from pylab import * 
    import matplotlib
    nas_df = output_df.isna()
    nas_df.resample('Y').sum()
    matplotlib.rcParams['font.family'] = 'Microsoft YaHei'
    mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei'] #更新字体格式
    mpl.rcParams['font.size'] = 9 
    plt.figure(figsize = (15, 6))
    plt.title('NaN count in data')
    plt.xticks(rotation='vertical')
    plt.bar(nas_df.sum().index, nas_df.sum().values)
    plt.show()







# %%
