import numpy as np

def simple_strategy(factor_df, target_price):
    '''
    如果空倉並且發生買進信號則開倉
    如果持倉並且發生賣出信號則平倉
    紀錄 flag 買賣標記
    紀錄 position 持倉標記
    若我們計算開盤到開盤的收益，並且持有一天
    即
    t日計算信號 (flag = 1) 若發生開倉信號則在 t+1 開盤時間開倉 (position = 1)
    取得 t+1 - t+2 的收益
    '''
    data  = factor_df.copy()
    data['flag'] = 0 # 买卖标记
    data['position'] = 0 # 持仓标记

    position = 0 # 是否持仓，持仓：1，不持仓：0
    for i in range(1,data.shape[0]-1):
        
        # 开仓
        if data.ix[i,'signalBuy'] and position ==0:
            data.ix[i,'flag'] = 1
            data.ix[i+1,'position'] = 1
            position = 1
        # 平仓
        elif data.ix[i,'signalSell'] and position == 1: 
            data.ix[i,'flag'] = -1
            data.ix[i+1,'position'] = 0
            position = 0
        
        # 保持
        else:
            data.ix[i+1,'position'] = data.ix[i,'position']   
    data['open_to_open_return'] = target_price.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0)
    data['tomorrow_open_to_open_return'] = data['open_to_open_return'].shift(-1)
    data['signal_return'] = data['tomorrow_open_to_open_return']* data['position']
    
#     print(data['signalReturn'].isna().any())
    data['nav'] = (1+data['signal_return']).cumprod()
        
    return(data)
