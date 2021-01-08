#!/bin/bash

kaggle competitions download -c otto-group-product-classification-challenge
unzip otto-group-product-classification-challenge.zip 
rm *zip
ls

python3 automl.py 

kaggle competitions submit -c otto-group-product-classification-challenge -f sub_1.csv -m "AutoML"