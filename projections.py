# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 14:56:48 2018

Raster projection methods.

@author: Administrator
"""


from math import ceil

import numpy as np

import rasterio
from rasterio.warp import reproject, Resampling, calculate_default_transform

try:
    from osgeo import gdal, osr
except ImportError:
    import gdal
    import osr


def reproject_dataset(data, pixel_spacing=0.00035862, epsg_to=4326,
                      N=50, resample_method=gdal.GRA_NearestNeighbour):

    """ Reproject data using gdal.

    Args:
        data (str): Input raster data supported by gdal.
        pixel_spacing (float): Raster resolution.
        epsg_to (epsg code): Destinate projection.
        N (int): Number of edge point to calculate the destinate data bounds.
        resample_method : resample methods.
            GRA_NearestNeighbour Nearest neighbour (select on one input pixel)
            GRA_Bilinear Bilinear (2x2 kernel)
            GRA_Cubic Cubic Convolution Approximation (4x4 kernel)
            GRA_CubicSpline Cubic B-Spline Approximation (4x4 kernel)
            GRA_Lanczos Lanczos windowed sinc interpolation (6x6 kernel)
            GRA_Average Average (computes the average of all
                           non-NODATA contributing pixels)
            GRA_Mode Mode (selects the value which appears most often
                           of all the sampled points)

    Returns:
        projected gdal raster dataset.

    """
    try:

        dataset = gdal.Open(data)
        dataType = dataset.GetRasterBand(1).DataType

        # 创建坐标系及转换方法
        sr_to = osr.SpatialReference()
        sr_to.ImportFromEPSG(epsg_to)
        # 从数据源中获取投影
        sr_from = osr.SpatialReference()
        sr_from.ImportFromWkt(dataset.GetProjection())
        tx = osr.CoordinateTransformation(sr_from, sr_to)

        # Get the Geotransform vector
        geo_t = dataset.GetGeoTransform()
        x_size = dataset.RasterXSize  # Raster xsize
        y_size = dataset.RasterYSize  # Raster ysize

        # 计算投影后数据范围

        # 仅用左上角点和右下角点，投影后范围小于之前影像，因此将4个角点坐标都转换后，
        # 取4个方向的最大或最小值来计算新影像范围
        # 4个角点转换后的范围仍然小于数据范围, 尝试从4条边各取N个点（平均），
        # 用上下边线上的点确定最大最小Y值，左右边线上的点确定最大最小X值

        # 左上点及右下点坐标转换
        (ulx, uly, ulz) = tx.TransformPoint(geo_t[0], geo_t[3])
        (lrx, lry, lrz) = tx.TransformPoint(geo_t[0] + geo_t[1] * x_size,
                                            geo_t[3] + geo_t[5] * y_size)
        # 根据边线长度 取N个平均点
        x_space = np.linspace(geo_t[0], geo_t[0] + geo_t[1]*x_size, N)
        y_space = np.linspace(geo_t[3], geo_t[3] + geo_t[5]*y_size, N)
        # 初始化
        UX, UY, LX, LY = lrx, uly, ulx, lry
        for i in np.arange(N):
            # 上边线，用来确定最大Y坐标UY
            (ux, uy, uz) = tx.TransformPoint(x_space[i], geo_t[3])
            # 左边线，用来确定最小X坐标LX
            (lx, ly, lz) = tx.TransformPoint(geo_t[0], y_space[i])
            # 下边线，用来确定最小Y坐标LY
            (llx, lly, llz) = tx.TransformPoint(x_space[i],
                                                geo_t[3] + geo_t[5]*y_size)
            # 右边线，用来确定最大X坐标UX
            (rx, ry, rz) = tx.TransformPoint(geo_t[0] + geo_t[1] * x_size,
                                             y_space[i])
            if uy > UY:
                UY = uy
            if lx < LX:
                LX = lx
            if lly < LY:
                LY = lly
            if rx > UX:
                UX = rx
        # 投影后影像范围
        height = UY - LY
        width = UX - LX

        # 将数据写入内存
        # Now, we create an in-memory raster
        mem_drv = gdal.GetDriverByName('MEM')
        # pixel_spacing表示每个像元的大小
        # GDT_Byte表示8位无符号整型
        # 波段数量与源数据一致
        dest = mem_drv.Create('', ceil(width/pixel_spacing),
                              ceil(height/pixel_spacing),
                              dataset.RasterCount,
                              dataType)

        # Calculate the new geotransform
        new_geo = (LX, pixel_spacing, geo_t[2],
                   UY, geo_t[4], -pixel_spacing)
        # Set the geotransform
        dest.SetGeoTransform(new_geo)
        dest.SetProjection(sr_to.ExportToWkt())
        # Perform the projection/resampling
        res = gdal.ReprojectImage(dataset, dest,
                                  sr_from.ExportToWkt(),
                                  sr_to.ExportToWkt(),
                                  resample_method)
        return dest
    finally:
        del dataset


def reproject_rio(infile, outfile, dst_crs='EPSG:4326',
                  resample_method=Resampling.nearest,
                  resolution=None):
    """ Reproject data using rasterio , See
    https://rasterio.readthedocs.io/en/latest/topics/reproject.html

    Args:
        infile (str): Input raster file.
        outfile (str): Reprojected outfile.
        dst_crs : Coordinate spacial reference supported by rasterio,
                default 'EPSG:4326'
        resample_method: Resample method ,defalt Resampling.nearest
                Resampling.nearest,
                Resampling.bilinear,
                Resampling.cubic,
                Resampling.cubic_spline,
                Resampling.lanczos,
                Resampling.average,
                Resampling.mode,
                Resampling.max (GDAL >= 2.2),
                Resampling.min (GDAL >= 2.2),
                Resampling.med (GDAL >= 2.2),
                Resampling.q1 (GDAL >= 2.2),
                Resampling.q3 (GDAL >= 2.2)
        resolution (float): Resolution of projected raster,default None which
                means same as input.

    """
    with rasterio.open(infile) as src:
        if resolution is not None:
            trans, width, height = calculate_default_transform(
                    src.crs, dst_crs, src.width,
                    src.height, resolution=resolution, *src.bounds)
        else:
            trans, width, height = calculate_default_transform(
                src.crs, dst_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'driver': 'GTIFF',
            'crs': dst_crs,
            'transform': trans,
            'width': width,
            'height': height
        })

        with rasterio.open(outfile, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=trans,
                    dst_crs=dst_crs,
                    resampling=resample_method)


if __name__ == '__main__':
    """
    # 调用
    data = r"D:\test14.tif"
    reprojected_dataset = reproject_dataset(data, pixel_spacing=0.0005,
                                            dataType=gdal.GDT_UInt16)
    # Let's save it as a GeoTIFF.
    driver = gdal.GetDriverByName("GTiff")
    dst_ds = driver.CreateCopy(r"D:\test18.TIF", reprojected_dataset, 0)
    dst_ds = None  # Flush the dataset to disk
    """
    infile = r'I:\sentinel\0915\T49QCC_20180915T030539_TCI.jp2'
    outfile = 'd:/projectionwithrio.tif'

    # reprojected_dataset = reproject_dataset(infile, pixel_spacing=10,
    #                                         epsg_to=32650)

    reproject_rio(infile, outfile, dst_crs='EPSG:4326')

    # tiffname = os.path.join(tempdir, 'example.tif')
    # driver = gdal.GetDriverByName ( "GTiff" )
    # dst_ds = driver.CreateCopy( r"D:\test18.TIF", reprojected_dataset, 0 )
    # dst_ds = None # Flush the dataset to disk
