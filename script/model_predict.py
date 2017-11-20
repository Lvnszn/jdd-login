import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
from sklearn.metrics import fbeta_score
from sklearn.metrics import recall_score, precision_score
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix

e = 0.8

def train_xgb(data, labels, test=None, test_label=None):
    param = {}
    param['objective'] = 'binary:logistic'
    param['eta'] = 0.25
    param['max_depth'] = 3
    param['eval_metric'] = "auc"
    param['min_child_weight'] = 3
    param['subsample'] = 0.8
    param['colsample_bytree'] = 0.8
    param['seed'] = 321
    param['silent'] = 1
    param['nthread'] = 8
    num_rounds = 10000
    print("start train...%d, %d" % (data.shape[0], len(labels)))

    member = test['rowkey']
    test = test.drop('rowkey', axis=1)
    data = data.drop('rowkey', axis=1)
    train_X, test_X, train_y, test_y = train_test_split(data,
                                                        labels,
                                                        test_size=0.2,
                                                        random_state=42)
    xgtrain = xgb.DMatrix(train_X, label=train_y)
    xgtest = xgb.DMatrix(test_X, label=test_y)

    watchlist = [(xgtrain, 'train'), (xgtest, 'eval')]
    bst = xgb.train(param, xgtrain, num_rounds, watchlist, maximize=True,  early_stopping_rounds=40)
    importance = bst.get_fscore()
    print(sorted(importance.items()))

    xgtrain = xgb.DMatrix(test)
    preds = bst.predict(xgtrain)
    test['is_risk'] = pd.DataFrame(preds)

    test.loc[test['is_risk'] > e, 'is_risk'] = 1
    test.loc[test['is_risk'] <= e, 'is_risk'] = 0
    test = pd.concat([test, member],axis=1)
    test = test_label.merge(test[test['is_risk'] > 0][['rowkey', 'is_risk']].drop_duplicates(), how='left', on='rowkey')
    test['is_risk'] = test.is_risk.fillna(0)
    test['is_risk'] = test.is_risk.astype(int)
    print test[['rowkey', 'is_risk']].drop_duplicates().is_risk.value_counts()
    test[['rowkey', 'is_risk']].to_csv("preds.csv", index=False, header=False)

def ff(preds, dtrain):
    label = dtrain.get_label()
    preds = map(lambda x: 1 if x > e else 0, preds)
    return 'fbeta_score', fbeta_score(label, preds, beta=0.1)

train=pd.read_csv('../input/t_login.csv')
test=pd.read_csv('../input/t_trade.csv')
t_test = pd.read_csv('../input/t_trade_test.csv')
t_train = pd.read_csv('../input/t_login_test.csv')

train_test = train[(train.time < '2015-06-01')]
test_test = test[(test.time >= '2015-05-01') & (test.time < '2015-06-01')]

print test_test.is_risk.value_counts()

all_train = pd.concat([train, t_train])

devices= train_test[['id','device']].drop_duplicates().groupby(['id'])['device'].count().reset_index()
all_devices = all_train[['id','device']].drop_duplicates().groupby(['id'])['device'].count().reset_index()
devices.columns=['id', 'divs']
all_devices.columns=['id', 'divs']
train_test = train_test.merge(devices, on='id')
all_train = all_train.merge(all_devices, on='id')

citys= train_test[['id','city']].drop_duplicates().groupby(['id'])['city'].count().reset_index()
all_citys = all_train[['id','city']].drop_duplicates().groupby(['id'])['city'].count().reset_index()
citys.columns=['id', 'citys']
all_citys.columns=['id', 'citys']
train_test = train_test.merge(citys, on='id')
all_train = all_train.merge(all_citys, on='id')

feature_to_use = ['rowkey', 'divs', 'citys', 'id']
# feature_to_use = ['rowkey', 'result','timelong','device','log_from','ip','city','id', 'is_scan', 'type']
# train = pd.concat([train, pd.get_dummies(train['is_scan'], prefix='scan'), pd.get_dummies(train['type'], prefix='type'), pd.get_dummies(train['result'], prefix='result')], axis=1)
# _train = pd.concat([t_train, pd.get_dummies(t_train['is_scan'], prefix='scan'), pd.get_dummies(t_train['type'], prefix='type'), pd.get_dummies(t_train['result'], prefix='result')], axis=1)

all = train_test.merge(test_test, on='id')
_all = all_train.merge(t_test, on='id')

all['is_risk'] = all.is_risk.astype(int)
train_xgb(all[feature_to_use], all['is_risk'], _all[feature_to_use], test_label=t_test)