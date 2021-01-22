import numpy as np
import pandas as pd
from supervised import AutoML

train = pd.read_csv("train.csv")
test  = pd.read_csv("test.csv")
sub = pd.read_csv("sample_submission.csv")
x_cols = train.columns[2:]
print(x_cols)
print(train.columns)
print(train["target"].min())
print(train["target"].max())

automl = AutoML(mode="Compete", eval_metric="rmse", total_time_limit=4*3600)
automl.fit(train[x_cols], np.log(train["target"]))



sub[sub.columns[1]] = np.exp(automl.predict(test))
sub.to_csv("sub_1.csv", index=False)


