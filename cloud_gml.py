# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 15:10:03 2018

利用gdal栅格化工具将哨兵2L1C数据中云掩膜（gml格式）栅格化
"""

import os
import glob
from image_process import get_gml_src_proj, rasterize


if __name__ == '__main__':
    rootdir = r'F:\SENTINEL\download\down0801'

    cld_gml = glob.iglob(os.path.join(rootdir, '*/MSK_CLOUDS_B00.gml'))

    for gml in cld_gml:
        basedir = os.path.dirname(gml)
        mask = os.path.join(basedir, 'cld_mask_gml.tif')
        refImage = os.path.join(basedir, 'B02.jp2')

        rasterize(gml, mask, refImage=refImage, pixel_size=10,
                  outputSRS=get_gml_src_proj(gml))
