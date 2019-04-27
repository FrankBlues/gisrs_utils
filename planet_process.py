# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 14:23:15 2018

@author: Administrator
"""

import rasterio
from image_process import gamma, linear_stretch, new_green_band

import numpy as np


if __name__ == '__main__':

    tif = r'I:\planet\planet_gz_20181003.tif'
    # Simplely process the image, ajdust the green band using infrared band.
    with rasterio.open(tif) as src:
        kargs = src.meta

        b = src.read(1)
        g = src.read(2)
        r = src.read(3)
        ir = src.read(4)

        gg = new_green_band(g, ir, ratio=0.07)
        del g
        orr = gamma(linear_stretch(r, percent=None, leftPercent=0.5,
                                   rightPercent=2, nodata=0), 1.3)
        orrr = orr*1.
        orrr[orrr == 0] = np.nan
        # show_hist(orrr,bins=50)

        ogg = gamma(linear_stretch(gg, percent=None, leftPercent=0.5,
                                   rightPercent=2, nodata=0), 1.5)
        oggg = ogg*1.
        oggg[oggg == 0] = np.nan
        # show_hist(oggg,bins=50)

        obb = gamma(linear_stretch(b, percent=None, leftPercent=0.5,
                                   rightPercent=3, nodata=0), 1.5)
        obbb = obb*1.
        obbb[obbb == 0] = np.nan
        # show_hist(obbb,bins=50)

        del r, gg, b
        kargs.update({
                'dtype': 'uint8',
                'count': 3,
                })
        with rasterio.open('d:/test_gd1.tif', 'w', **kargs) as dst:
            dst.write(orr, 1)
            dst.write(ogg, 2)
            dst.write(obb, 3)
