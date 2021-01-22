import pandas as pd
from supervised import AutoML

train = pd.read_csv("train.csv")
test  = pd.read_csv("test.csv")
sub = pd.read_csv("sample_submission.csv")
x_cols = train.columns[2:]
print(x_cols)

automl = AutoML(results_path="AutoML_3", mode="Compete", total_time_limit=4*3600, eval_metric="r2")
automl.fit(train[x_cols], train["y"])

sub[sub.columns[1:]] = automl.predict(test)
sub.to_csv("sub_1.csv", index=False)


