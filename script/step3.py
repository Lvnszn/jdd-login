import numpy as np
import pandas as pd

train=pd.read_csv('../input/t_login.csv')
test=pd.read_csv('../input/t_trade.csv')
t_test = pd.read_csv('../input/t_trade_test.csv')
t_train = pd.read_csv('../input/t_login_test.csv')

preds = pd.read_csv('xgb_pred.csv')
preds.columns=['rowkey', 'is_risk']
t_test = t_test.merge(preds[preds['is_risk']>0], how='left', on='rowkey')
t_test['is_risk'] = t_test.is_risk.fillna(0)
t_test['is_risk'] = t_test.is_risk.astype(int)
t_test[['rowkey', 'is_risk']].drop_duplicates().to_csv("baseline4.csv", index=False, header=False)