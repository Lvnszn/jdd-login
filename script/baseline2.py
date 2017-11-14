import numpy as np
import pandas as pd

train=pd.read_csv('../input/t_login.csv')
test=pd.read_csv('../input/t_trade.csv')
t_test = pd.read_csv('../input/t_trade_test.csv')
t_train = pd.read_csv('../input/t_login_test.csv')


tt2 = t_train[['id','device']].drop_duplicates().groupby(['id'])['device'].count().reset_index()
tt2['is_risk'] = 1
all2 = t_test.merge(tt2[tt2.device>33], how='left', on='id')
all2 = all2.fillna(0)
all2['is_risk'] = all2.is_risk.astype(int)
print all2.is_risk.value_counts()
all2[['rowkey', 'is_risk']].to_csv("baseline2.csv", index=False, header=False)

# 0.47
