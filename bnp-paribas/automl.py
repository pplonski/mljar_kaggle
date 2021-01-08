import pandas as pd
from supervised import AutoML

train = pd.read_csv("train.csv")
test  = pd.read_csv("test.csv")
sub = pd.read_csv("sample_submission.csv")
x_cols = train.columns[2:]

automl = AutoML(mode="Compete", total_time_limit=12*3600)
automl.fit(train[x_cols], train["target"])

sub["PredictedProb"] = automl.predict_proba(test)[:,1]
sub.to_csv("sub_1.csv", index=False)