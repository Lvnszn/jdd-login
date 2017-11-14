import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
import lightgbm as lgb
train=pd.read_csv('../input/t_login.csv')
test=pd.read_csv('../input/t_trade.csv')
t_test = pd.read_csv('../input/t_trade_test.csv')

ttt = test.groupby(['id'])['is_risk'].sum().reset_index()
ttt.loc[ttt['is_risk']>0, 'is_risk'] = 1
res = t_test.merge(ttt[['id', 'is_risk']], how='left', on='id')
print t_test.shape
print test.is_risk.value_counts()
print res.is_risk.value_counts()
res = res.fillna(0)
res['is_risk'] = res.is_risk.astype(int)
res[['rowkey', 'is_risk']].to_csv("baseline.csv", index=False, header=False)

# 0.16 bad
