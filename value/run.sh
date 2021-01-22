#!/bin/bash

kaggle competitions download -c santander-value-prediction-challenge
unzip santander-value-prediction-challenge.zip 
rm *zip
ls

python3 automl.py 

kaggle competitions submit -c santander-value-prediction-challenge -f sub_1.csv -m "AutoML"