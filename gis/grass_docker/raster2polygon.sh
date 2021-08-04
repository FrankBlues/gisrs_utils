#!/bin/bash
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib
# export PATH=$PATH:/usr/lib

echo "Input raster: $1"
echo "Output shapefile: $2"
r.external input=$1 output=buidings --overwrite
g.region raster=buidings -p
r.to.vect -v input=buidings output=buildings_vec type=area --overwrite
v.out.ogr -e input=buildings_vec output=$2 format=ESRI_Shapefile --overwrite
# v.out.ogr -e input=buildings_vec output=buildings.gpkg format=GPKG --overwrite