#!/bin/bash

kaggle competitions download -c bnp-paribas-cardif-claims-management
unzip bnp-paribas-cardif-claims-management.zip 
unzip sample_submission.csv.zip
unzip test.csv.zip
unzip train.csv.zip
rm *zip
ls

python3 automl.py 

kaggle competitions submit -c bnp-paribas-cardif-claims-management -f sub_1.csv -m "AutoML"