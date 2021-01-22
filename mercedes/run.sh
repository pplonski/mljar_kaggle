#!/bin/bash

kaggle competitions download -c mercedes-benz-greener-manufacturing
unzip mercedes-benz-greener-manufacturing.zip 
unzip sample_submission.csv.zip
unzip test.csv.zip
unzip train.csv.zip
rm *zip
ls

python3 automl.py 

kaggle competitions submit -c mercedes-benz-greener-manufacturing -f sub_1.csv -m "AutoML"