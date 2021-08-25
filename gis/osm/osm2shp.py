# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 18:21:02 2021

@author: DELL
"""

import os


if __name__ == '__main__':
    
    osm_file = r'D:\g_tiles\china-latest.osm.pbf'
    out_dir = r'D:\temp11\test_osm'
    out_base = 'osm_cn'
    out = os.path.join(out_dir, out_base)
    osm_features = ['points', 'lines', 'multilinestrings',
                    'multipolygons', 'other_relations']
    osm_features = ['lines']
    for fea in osm_features:
        out_shp = out + '_' + fea + '.shp'
        cmd = f"ogr2ogr {out_shp} {osm_file} {fea} -lco ENCODING=UTF-8"
        print(cmd)
        os.system(cmd)