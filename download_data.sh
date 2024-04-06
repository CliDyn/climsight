#!/bin/bash
wget https://swift.dkrz.de/v1/dkrz_035d8f6ff058403bb42f8302e6badfbc/clisight/data_climate_foresight.tar
tar -xvf data_climate_foresight.tar
rm -r ./data/natural_earth # just for now to not have duplicates

mkdir -p ./data/natural_earth/coastlines
mkdir -p ./data/natural_earth/land
mkdir -p ./data/natural_earth/rivers
mkdir -p ./data/natural_earth/lakes
mkdir -p ./data/natural_hazards/
mkdir -p ./data/population

cd ./data/natural_earth/coastlines
wget https://naciscdn.org/naturalearth/10m/physical/ne_10m_coastline.zip 
unzip ne_10m_coastline.zip 
rm ne_10m_coastline.zip

cd ../land
wget https://naciscdn.org/naturalearth/10m/physical/ne_10m_land.zip
unzip ne_10m_land.zip
rm ne_1om_land.zip

cd ../rivers
wget https://naciscdn.org/naturalearth/10m/physical/ne_10m_rivers_lake_centerlines.zip
unzip ne_10m_rivers_lake_centerlines.zip
rm ne_10m_rivers_lake_centerlines.zip
wget https://naciscdn.org/naturalearth/10m/physical/ne_10m_rivers_australia.zip
unzip ne_10m_rivers_australia.zip
rm ne_10m_rivers_australia.zip
wget https://naciscdn.org/naturalearth/10m/physical/ne_10m_rivers_europe.zip
unzip ne_10m_rivers_europe.zip
rm ne_10m_rivers_europe.zip
wget https://naciscdn.org/naturalearth/10m/physical/ne_10m_rivers_north_america.zip 
unzip ne_10m_rivers_north_america.zip
rm ne_10m_rivers_north_america.zip

cd ../lakes
wget https://naciscdn.org/naturalearth/10m/physical/ne_10m_lakes.zip
unzip ne_10m_lakes.zip
rm ne_10m_lakes.zip
wget https://naciscdn.org/naturalearth/10m/physical/ne_10m_lakes_australia.zip
unzip ne_10m_lakes_australia.zip
rm ne_10m_lakes_australia.zip
wget https://naciscdn.org/naturalearth/10m/physical/ne_10m_lakes_europe.zip
unzip ne_10m_lakes_europe.zip
rm ne_10m_lakes_europe.zip
wget https://naciscdn.org/naturalearth/10m/physical/ne_10m_lakes_north_america.zip 
unzip ne_10m_lakes_north_america.zip
rm ne_10m_lakes_north_america.zip

cd ../../population
curl -O "https://population.un.org/wpp/Download/Files/1_Indicators%20(Standard)/CSV_FILES/WPP2022_Demographic_Indicators_Medium.zip"
unzip WPP2022_Demographic_Indicators_Medium.zip
rm WPP2022_Demographic_Indicators_Medium.zip