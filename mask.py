# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 16:46:00 2018

Extract Raster data By Mask using python rasterio module.

@author: Administrator
"""

import os
import glob
import numpy as np
import rasterio
import rasterio.mask
from rasterio.windows import Window

from features import get_features_from_shp as get_features


def extract_by_mask_rio_ds(features, raster_dataset, out, nodata=0):
    """Extract raster by vector mask.

    Note:
        Mask and raster dataset should have the same projection.

    Args:
        features: a GeoJSON-like dict or an object that implements the Python
            geo interface protocol (such as a Shapely Polygon).
        raster_dataset (rasterio dataset): The raster dataset.
        out (str): Masked output raster file with geotiff format.
        nodata (float): Nodata value of output raster, default 0.

    Returns:
        str: The output raster filename.
        None: If error happens.

    Raises:
        ValueError: If input shapes do not overlap raster.

    """
    out_meta = raster_dataset.meta.copy()

    try:
        out_image, out_transform = rasterio.mask.mask(
                raster_dataset,
                features,
                crop=True,
                nodata=nodata)

        out_meta.update({"driver": "GTiff",
                         "nodata": nodata,
                         "height": out_image.shape[1],
                         "width": out_image.shape[2],
                         "transform": out_transform})
    except ValueError:  # ValueError: Input shapes do not overlap raster.
        return  # do nothing
    # 全部为空值则不进行输出
    if not np.all(out_image == nodata):
        with rasterio.open(out, "w", **out_meta) as dest:
            dest.write(out_image)
    return out


def extract_by_mask(maskshp, raster, out, nodata=0):
    """Same as the 'extractByMask_ds' function except that the input
    is raster file.

    """
    with rasterio.open(raster) as src:
        extract_by_mask_rio_ds(get_features(maskshp), src, out, nodata=nodata)


def window_list(window_size, width, height):
    """Split data into windows (square).

    Args:
        window_size (int): Window size.
        width, height (int): Data width and height.

    Returns:
        list: Rasterio Window object.

    """
    import math
    n_windows_x = math.ceil(width*1./window_size)
    n_windows_y = math.ceil(height*1./window_size)

    x_last_width = width % window_size
    y_last_width = height % window_size

    x_last_width = window_size if x_last_width == 0 else x_last_width
    y_last_width = window_size if y_last_width == 0 else y_last_width

    window_list = []
    for iy in range(n_windows_y):
        for ix in range(n_windows_x):
            # complete windows
            if ix != n_windows_x - 1 and iy != n_windows_y - 1:
                window_list.append(Window(
                        window_size * ix, window_size * iy,
                        window_size, window_size))
            # windows at the last column
            elif ix == n_windows_x - 1 and iy != n_windows_y - 1:
                window_list.append(Window(
                        window_size * ix, window_size * iy,
                        x_last_width, window_size))
            # window at the last row
            elif ix != n_windows_x - 1 and iy == n_windows_y - 1:
                window_list.append(Window(
                        window_size * ix, window_size * iy,
                        window_size, y_last_width))
            # window both in the last row and column
            else:
                window_list.append(Window(
                        window_size * ix, window_size * iy,
                        x_last_width, y_last_width))

    return window_list


def split_image(image, window_size=1280, use_block=False):
    """ Split image into many parts by window.

    Returns:
        image (str): Input raster file.
        window_size (int): Window size used if use_block is False, default 1280
        use_block (bool): If true, use the defalut block size to define
                windows, else use window_size parameters to define windows.

    Returns:
        generator: The output rasterio dataset.

    """
    with rasterio.open(image) as src:
        # use block size
        if use_block:
            windows = [window for ij, window in src.block_windows()]
        # use self defined window
        else:
            windows = window_list(window_size, src.width, src.height)

        print("Split into total {} windows.".format(len(windows)))

        for idx, window in enumerate(windows):
            print("current number:{}".format(idx))

            kwargs = src.meta.copy()

            kwargs.update({
                'height': window.height,
                'width': window.width,
                'transform': rasterio.windows.transform(window, src.transform)
                })

            # TODO using memery or tempfile to store temp data
            out = 'G:/temp/cropped_{}.tif'.format(str(idx + 1))

            with rasterio.open(out, 'w', **kwargs) as dst:
                dst.write(src.read(window=window))

            yield rasterio.open(out)
            # result_list.append(rasterio.open(out))
    # return result_list


def extract_by_mask_window(image, shp, outTiff, window_size=1280,
                           nodata=0, use_block=False):
    """Split raster file inot many parts, then extact each parts by mask
    defined by shapefile, then merge each parts to one file again. Used
    to reduce the memory use.

    Args:
        image (str): Input raster file.
        shp (esri shapefile): Shapefile used to define windows.
        outTiff (str): Output raster filename.
        window_size (int): Window size used if use_block is False, default 1280
        nodata (float): Nodata value of output raster， default 0.
        use_block (bool): If true, use the defalut block size to define
                windows, else use window_size parameters to define windows.

    """
    dss = split_image(image, window_size=window_size)
    valid_out_list = []
    for i, ds in enumerate(dss):
        # TODO using memery or tempfile to store temp data
        out = 'G:/temp/extracted_{}.tif'.format(str(i))
        # ds = rasterio.open(fi)
        extract_result = extract_by_mask_rio_ds(shp, ds, out, nodata=0)
        if extract_result is not None:
            valid_out_list.append(extract_result)
    del dss

    import mosaic
    print(valid_out_list)
    src_files_to_mosaic = [rasterio.open(f) for f in valid_out_list]
    mosaic.merge1(src_files_to_mosaic)
    del src_files_to_mosaic


if __name__ == '__main__':

    shp = r"G:\temp\test_mask_shapefile.shp"
    image = r"G:\temp\test_image.tif"
    window_size = 400
    # out = split_image(r'G:\temp\test_image.tif',400)
    # for ds in out:

    # extractByMask_window(image,shp,400)
    image = r'G:\mosaic_20181007_t49_wgs84.tif'
    shp = r'D:\test\SENTINEL\广东范围_84.shp'
    outTiff = "G:/temp/temp.tif"

    # extract_by_mask_window(image, shp, outTiff, 20000)

    mosaic_dir = r'I:\MODIS\mosaic'
    out_dir = r'I:\MODIS\mask'
    # mosaic_list = glob.glob(os.path.join(mosaic_dir, '*.tif'))

    r = r'D:\work\data\影像样例\610124.tif'
    shp = r'D:\test\test_fishnet.shp'
    ds = rasterio.open(r)
    nodata = 0
    # extract_by_mask(shp, r, 'd:/temp/ttttt.tif')
