#!/bin/bash

wget https://swift.dkrz.de/v1/dkrz_035d8f6ff058403bb42f8302e6badfbc/clisight/data_climate_foresight.tar
tar -xvf data_climate_foresight.tar

mkdir -p ./data/natural_hazards/

mkdir -p ./data/population/
cd ./data/population
curl -O "https://population.un.org/wpp/Download/Files/1_Indicators%20(Standard)/CSV_FILES/WPP2022_Demographic_Indicators_Medium.zip"
unzip WPP2022_Demographic_Indicators_Medium.zip 