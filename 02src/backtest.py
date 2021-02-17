#%%
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

if __name__ == '__main__':
    from util import custom_load_data
    from feature_preparation import load_macro_feature, AlphaFeature
    from feature_selection import pcaSelection, naiveSelection
    from classifier_model import MyNaiveBayesClassifier
    from rolling_prediction import RollingSignalGenerator
    from position_strategy import simple_strategy
else:
    raise ImportError
#%% parameter
DATA_PATH = 'C:\\Users\\eiahb\\Documents\\MyFiles\\PythonProject\\PHBS_machine_learning_for_finance\\PHBS_ML_for_quant_project\\00data'
START_DATE = "2010-01-01"
END_DATE = "2020-01-01"
index_dict = {
    "windA":"881001.csv", 
    "hs300":"000300.csv"
}
test_index = "windA"
alpha_feature_list = [
    'alpha002',
    'alpha014',
    'alpha018',
    'alpha020',
    'alpha034',
    'alpha066',
    'alpha070',
    'alpha106'
]

#%% define how to class
def get_return_class(index_data):
    tomorrow_class = (index_data['return'].shift(-1)>0)
    return tomorrow_class
#%%
def main():
    index_data = custom_load_data(os.path.join(DATA_PATH, index_dict[test_index]))
    macro_feature = load_macro_feature()
    macro_feature.index = pd.to_datetime(macro_feature.index)
    af = AlphaFeature(index_data)
    alpha_feature = af.get_feature(alpha_feature_list)

    y = get_return_class(index_data.loc[START_DATE:END_DATE])
    alpha_feature = alpha_feature.reindex(y.index)
    macro_feature = macro_feature.reindex(y.index)

    alpha_feature = alpha_feature.fillna(method="ffill")

    signal_generator = RollingSignalGenerator(alpha_feature, y)
    signal, modelRecord = signal_generator.generateSignal(predictModel = MyNaiveBayesClassifier, featureSelectionFunction = pcaSelection)
    print(signal)


    signalBuy = signal
    signalSell = ~signal
    factor_df = pd.DataFrame()
    factor_df['signalBuy'] = signalBuy.fillna(False)
    factor_df['signalSell'] = signalSell.fillna(False)
    factor_df.head()

    result = simple_strategy(factor_df = factor_df, target_price = index_data.close)
    plt.figure(figsize = (30, 16))

    result.nav.plot()
    (index_data.reindex(result.index).open/ index_data.reindex(result.index).open[0]).plot()
    plt.legend(labels=['strategy','benchmark'],loc='best')
    plt.grid()
# %%
if __name__ == "__main__":
    main()
# %%
