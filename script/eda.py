# coding:utf-8
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
import time
train=pd.read_csv('../input/t_login.csv', na_values=-1)
test=pd.read_csv('../input/t_trade.csv', na_values=-1)
# 登陆时间戳和登陆时间一样 is_sec都是false a表大


train=pd.read_csv('../input/t_login.csv')
test=pd.read_csv('../input/t_trade.csv')
t_test = pd.read_csv('../input/t_trade_test.csv')
t_train = pd.read_csv('../input/t_login_test.csv')

train_test = train[(train.time < '2015-06-01')]
test_test = test[(test.time > '2015-05-01') & (test.time < '2015-06-01')]

aa = test_test[['id', 'rowkey']].merge(train_test[train_test['time'] > '2015-05-01'], how='left', on='id')
ids = aa[np.isnan(aa['log_id']) == True]

ii = test_test[['id', 'is_risk']].merge(ids[['id']], on='id')
print ii[['id', 'is_risk']].drop_duplicates().is_risk.value_counts()

# 统计, 上个月登录,下个月消费,异常用户为1-2个左右,可忽略不计