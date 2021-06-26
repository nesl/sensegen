#!/bin/bash
cd dataset
wget 'https://data.mendeley.com/datasets/63zm778szb/3/files/f9190e5c-ea86-49e7-b7e2-0ddf40b146b9/ECG%20signals%20(744%20fragments).zip?dl=1' -O ecg_data.zip
unzip ecg_data
mv MLII ecg_data
rm ecg_data.zip