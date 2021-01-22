import pandas as pd
from supervised import AutoML

train = pd.read_csv("train.csv")
test  = pd.read_csv("test.csv")
sub = pd.read_csv("sample_submission.csv")
print(train.shape, test.shape)
print(sub.shape)
x_cols = train.columns[2:]
print(x_cols)

print(train[x_cols])
print(train["TripType"])

automl = AutoML(mode="Compete", ml_task="multiclass_classification", total_time_limit=4*3600)
automl.fit(train[x_cols], train["TripType"])

sub[sub.columns[1:]] = automl.predict_proba(test)
sub.to_csv("sub_1.csv", index=False)


