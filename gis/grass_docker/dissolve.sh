#!/bin/bash
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib
# export PATH=$PATH:/usr/lib

echo "Input vector: $2"
echo "Output: $1"
echo "Layer name: $3"
echo "Field:$4"

v.import input=$2 layer=$3 output=todo_dissolve --overwrite
v.dissolve --overwrite input=todo_dissolve@PERMANENT column=$4 output=dissolved
v.out.ogr -e input=dissolved output=$1 format=ESRI_Shapefile