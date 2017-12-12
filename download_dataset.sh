#!/bin/bash
test -d dataset || mkdir dataset
cd dataset
wget https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip -O dataset.zip
unzip dataset -d .
rm  dataset.zip
mv  UCI\ HAR\ Dataset/* .
rm -rf UCI\ HAR\ Dataset
