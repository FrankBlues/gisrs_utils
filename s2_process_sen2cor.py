# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 15:35:54 2019

Process sentinel 2 L1C data using sen2cor module then apply cira strech.

@author: Administrator
"""

import os
import glob
import rasterio
from image_process import cira_strech, new_green_band, bytscl


def get_raster_arr(r):
    """Read raster array.

    Args:
        r (str): Input raster file.

    Returns:
        numpy ndarray: The raster data array.

    """
    with rasterio.open(r) as src:
        return src.read(1)


def cira_strech_l2a(r, g, b, ir, out_tiff):
    """Apply CIRA strech to the L2A data .

    Args:
        r, g, b, ir (numpy ndarray): The sentinel L2A product band array.
        out_tiff (str): Output tifffile.

    """
    new_rarr = cira_strech(get_raster_arr(r) * 0.0001)
    new_garr = new_green_band(get_raster_arr(g) * 0.0001,
                              get_raster_arr(ir) * 0.0001, 0.07)
    new_garr = cira_strech(new_garr)

    with rasterio.open(b) as src:
        kargs = src.meta.copy()
        barr = src.read(1)
        new_barr = cira_strech(barr*0.0001)
        del barr
        kargs.update({
                'driver': 'GTiff',
                'count': 3,
                'dtype': 'uint8',
                })
        with rasterio.open(out_tiff, 'w', **kargs) as dst:
            dst.write(bytscl(new_rarr, 1, 0), 1)
            dst.write(bytscl(new_garr, 1, 0), 2)
            dst.write(bytscl(new_barr, 1, 0), 3)


def get_tiles(safe_dir, date):
    """ Get all the tiles on the given date from L2A product directory.

    Args:
        safe_dir (str): The L2A product directory.
        date (str): Date with the format 'YYYYMMDD'.

    Returns:
        list: MGRS tile list.

    """
    safes = glob.glob(os.path.join(safe_dir, '*{}*.SAFE'.format(date)))
    return [os.path.basename(s).split('_')[-2][1:] for s in safes]


if __name__ == '__main__':
    sen2cor_home = r'D:\Sen2Cor-02.05.05-win64'
    os.chdir(sen2cor_home)
    safe_dir = 'H:/s2_data/'
    inList = glob.glob(safe_dir + 'S2*L1C_201812*.SAFE')

    # Process using Sen2Cor module in commond line.
    for safe in inList:
        a = os.system("L2A_Process --resolution=10 {}".format(
                os.path.normpath(safe)))

    date = '20181221'
    tiles = get_tiles(safe_dir, date)
    for tile in tiles:
        blist = glob.glob(safe_dir +
                          ('S2*{date}*{tile}*.SAFE/GRANULE/*{date}*/IMG_DAT'
                           'A/R10m/*_B*.jp2'.format(date=date, tile=tile)))
        for band in blist:
            if 'B02' in os.path.basename(band):
                b = os.path.normpath(band)
            elif 'B03' in os.path.basename(band):
                g = os.path.normpath(band)
            elif 'B04' in os.path.basename(band):
                r = os.path.normpath(band)
            elif 'B08' in os.path.basename(band):
                ir = os.path.normpath(band)

        cira_strech_l2a(r, g, b, ir,
                        'H:/temp/S2_{0}_{1}.tif'.format(date, tile))
