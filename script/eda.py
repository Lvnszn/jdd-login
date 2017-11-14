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

print train.result.value_counts()
print train.type.value_counts()
all = train.merge(test, on='id')

def c_t(timestamp):
    time_local = time.localtime(timestamp)
    return time.strftime("%Y-%m-%d %H:%M:%S",time_local)

all['c_t'] = map(lambda x:c_t(x), all['timestamp'])
print all.head(50)