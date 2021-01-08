#!/bin/bash

kaggle competitions download -c santander-customer-satisfaction
unzip santander-customer-satisfaction.zip 
rm *zip
ls
# Work4WalmarT

python3 automl.py 

kaggle competitions submit -c santander-customer-satisfaction -f sub_1.csv -m "AutoML"