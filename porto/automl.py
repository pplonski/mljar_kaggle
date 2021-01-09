import pandas as pd
from supervised import AutoML

train = pd.read_csv("train.csv")
test  = pd.read_csv("test.csv")
sub = pd.read_csv("sample_submission.csv")
x_cols = train.columns[2:]

automl = AutoML(mode="Compete", total_time_limit=4*3600, eval_metric="auc")
automl.fit(train[x_cols], train["target"])

sub["target"] = automl.predict_proba(test)[:,1]
sub.to_csv("sub_1.csv", index=False)