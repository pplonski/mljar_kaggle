import pandas as pd
from supervised import AutoML

train = pd.read_csv("train.csv")
test  = pd.read_csv("test.csv")
sub = pd.read_csv("sample_submission.csv")
x_cols = train.columns[2:]
print(x_cols)

automl = AutoML(mode="Compete", eval_metric="auc", total_time_limit=11)#4*3600)
automl.fit(train[x_cols], train["target"])



sub[sub.columns[1:]] = automl.predict_proba(test)[:,1]
sub.to_csv("sub_1.csv", index=False)


