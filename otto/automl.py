import pandas as pd
from supervised import AutoML

train = pd.read_csv("train.csv")
test  = pd.read_csv("test.csv")
sub = pd.read_csv("sampleSubmission.csv")
x_cols = train.columns[1:-1]
print(x_cols)

automl = AutoML(mode="Compete", total_time_limit=4*3600)
automl.fit(train[x_cols], train["target"])

sub[sub.columns[1:]] = automl.predict_proba(test)
sub.to_csv("sub_1.csv", index=False)