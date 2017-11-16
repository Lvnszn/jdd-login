import numpy as np
import pandas as pd

train=pd.read_csv('../input/t_login.csv')
test=pd.read_csv('../input/t_trade.csv')
# t_test = pd.read_csv('../input/t_trade_test.csv')
t_train = pd.read_csv('../input/t_login_test.csv')
t_test = pd.read_csv('../input/Risk_Detection_Upload_Sample.csv', header=-1)
t_test.columns=['rowkey', 'is_risk']

preds = pd.read_csv('preds.csv', header=-1)
preds.columns=['rowkey', 'is_risk']
t_test = t_test[['rowkey']].merge(preds[preds['is_risk']>0], how='left', on='rowkey')
t_test['is_risk'] = t_test.is_risk.fillna(0)
t_test['is_risk'] = t_test.is_risk.astype(int)
t_test.to_csv("baseline4.csv", index=None, sep=',', header=False)