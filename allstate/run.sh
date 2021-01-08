#!/bin/bash

kaggle competitions download -c allstate-claims-severity
unzip allstate-claims-severity.zip 
rm *zip
ls
# Work4WalmarT

python3 automl.py 

kaggle competitions submit -c allstate-claims-severity -f sub_1.csv -m "AutoML"