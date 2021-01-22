#!/bin/bash

kaggle competitions download -c santander-customer-transaction-prediction
unzip santander-customer-transaction-prediction.zip 
rm *zip
ls

python3 automl.py 

kaggle competitions submit -c santander-customer-transaction-prediction -f sub_1.csv -m "AutoML"