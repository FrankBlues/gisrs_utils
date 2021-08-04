# -*- coding: utf-8 -*-
import os
from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input_raster', required=True, help='输入影像文件')
    parser.add_argument('--output_shp', required=True, help='输出shapefile文件')

    args = parser.parse_args()
    # grass location
    location = '/grassdb/test_grass'
    # init grass
    os.system(f"grass --text -c -e {args.input_raster} {location}")
    # exec raster to polygon shell
    os.system(f"grass {location}/PERMANENT --exec sh raster2polygon.sh {args.input_raster} {args.output_shp}")