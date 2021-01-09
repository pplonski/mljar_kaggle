#!/bin/bash

kaggle competitions download -c ieee-fraud-detection
unzip ieee-fraud-detection.zip 
rm *zip
ls

python3 automl.py 

kaggle competitions submit -c ieee-fraud-detection -f sub_1.csv -m "AutoML"