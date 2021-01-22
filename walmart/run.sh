#!/bin/bash

kaggle competitions download -c walmart-recruiting-trip-type-classification
unzip walmart-recruiting-trip-type-classification.zip 
unzip sample_submission.csv.zip
unzip test.csv.zip
unzip train.csv.zip
rm *zip
ls
# Work4WalmarT

python3 automl.py 

kaggle competitions submit -c walmart-recruiting-trip-type-classification -f sub_1.csv -m "AutoML"