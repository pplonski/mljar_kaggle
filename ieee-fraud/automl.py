import pandas as pd
from supervised import AutoML

train_identity = pd.read_csv('train_identity.csv')
train_transaction = pd.read_csv('train_transaction.csv')

test_identity = pd.read_csv('test_identity.csv')
test_transaction = pd.read_csv('test_transaction.csv')

train = pd.merge(train_transaction,train_identity,on='TransactionID', how='left')
test = pd.merge(test_transaction, test_identity, on='TransactionID', how='left')

del train_identity, train_transaction, test_identity, test_transaction

# solving incosistent naming of columns

for i in range(1,39):
    if i < 10:
          test.rename(columns = {"id-0"+str(i) : "id_0"+str(i)}, inplace=True)
    else:
          test.rename(columns = {"id-"+str(i) : "id_"+str(i)}, inplace = True)

x_cols = train.columns[2:]

automl = AutoML(algorithms=["LightGBM"], mode="Compete", eval_metric="auc", total_time_limit=4*3600)
automl.fit(train[x_cols], train["isFraud"])

sub = pd.read_csv("sample_submission.csv")
sub[sub.columns[1:]] = automl.predict_proba(test)[:,1]
sub.to_csv("sub_1.csv", index=False)


