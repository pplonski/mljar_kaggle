#!/bin/bash

kaggle competitions download -c porto-seguro-safe-driver-prediction
unzip porto-seguro-safe-driver-prediction.zip 
rm *zip
ls

python3 automl.py 

kaggle competitions submit -c porto-seguro-safe-driver-prediction -f sub_1.csv -m "AutoML"