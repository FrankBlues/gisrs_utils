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
                  resolution=None, src_crs=None):
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
        crs = src.crs
        if crs is None:  # 原始数据中没有投影
            print("Origin image has not built-in projection.")
            if src_crs:  # 采用指定的投影参考
                crs = src_crs
            else:
                raise ValueError("Source image hasn't a projection define,"
                                 "Please check it or designate one.")
                return

        if resolution is not None:
            trans, width, height = calculate_default_transform(
                    crs, dst_crs, src.width,
                    src.height, resolution=resolution, *src.bounds)
        else:
            trans, width, height = calculate_default_transform(
                crs, dst_crs, src.width, src.height, *src.bounds)
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
                    src_crs=crs,
                    dst_transform=trans,
                    dst_crs=dst_crs,
                    resampling=resample_method)


# def reproject_rio(infile, outfile, dst_crs='EPSG:4326',
#                   resample_method=Resampling.nearest,
#                   resolution=None, src_crs=None):


def point2matrix(x, y, z, args):
    """定义函数返回系数矩阵 B,l;通过给定的同名点坐标列立误差方程B系数阵的部分.

    Args:
        x, y, z:原坐标值;
        args: 七参数误差值[Delta_X, Delta_Y, Delta_Z, theta_x, theta_y, theta_z, m].

    Returns:
        ndarray: W系数阵.

    """
    array = [
        [1, 0, 0, 0, -(1+args[6])*z, (1+args[6])*y, x+args[5]*y-args[4]*z],
        [0, 1, 0, (1+args[6])*z, 0, -(1+args[6])*x, -args[5]*x+y+args[3]*z],
        [0, 0, 1, -(1+args[6])*y, (1+args[6])*x, 0, args[4]*x-args[3]*y+z]
        ]
    return np.array(array)


def points2W(points, args):
    """通过同名点序列列立误差方程B系数阵的整体.

    Args:
        points (list): 同名点序列;
        args (list): 七参数误差值.

    Returns:
        big_mat (ndarray): W系数阵.

    """
    big_mat = None
    for (x, y, z) in points:
        mat = point2matrix(x, y, z, args)
        if big_mat is None:
            big_mat = mat
        else:
            big_mat = np.concatenate((big_mat, mat))
    return big_mat


def points2b(source, target, args):
    """通过同名点坐标转换关系列立误差方程B系数阵的整体.

    Args:
        source (list): 原始点对;
        target (list): 目标点对;
        args (list): 七参列表.

    Returns:
        ndarray: b系数阵.
    """
    big_mat = [0] * len(source) * 3

    for i, ((x1, y1, z1), (x2, y2, z2)) in enumerate(zip(source, target)):
        (x0, y0, z0) = ordinationConvert(x1, y1, z1, args)
        big_mat[3*i + 0] = x2 - x0
        big_mat[3*i + 1] = y2 - y0
        big_mat[3*i + 2] = z2 - z0
    return np.array(big_mat).transpose()


def ordinationConvert(x1, y1, z1, args):
    x2 = args[0] + (1+args[6])*(x1 + args[5]*y1 - args[4]*z1)
    y2 = args[1] + (1+args[6])*(-args[5]*x1 + y1 + args[3]*z1)
    z2 = args[2] + (1+args[6])*(args[4]*x1 - args[3]*y1 + z1)
    return (x2, y2, z2)


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

    # reproject_rio(infile, outfile, dst_crs='EPSG:4326')

    # tiffname = os.path.join(tempdir, 'example.tif')
    # driver = gdal.GetDriverByName ( "GTiff" )
    # dst_ds = driver.CreateCopy( r"D:\test18.TIF", reprojected_dataset, 0 )
    # dst_ds = None # Flush the dataset to disk


    # 使用4个同名点的原坐标(x, y, z)和目标坐标（x’, y’, z’)根据最小二乘来求解七参数
    source_points = [
        (3381400.980, 395422.030, 32.956),
        (3381404.344, 395844.239, 32.207),
        (3382149.810, 396003.592, 33.290),
        (3382537.793, 395985.359, 33.412)
        ]
    target_points = [
        (3380968.194, 539468.888, 13.875),
        (3380977.154, 539890.934, 13.179),
        (3381724.612, 540040.47,  14.273),
        (3381724.636, 540040.485, 14.282)
        ]

    #归一化处理便于模型的快速迭代
    ave_src = np.mean(np.array(source_points), axis=0)
    ave_tar = np.mean(np.array(target_points), axis=0)
    source_points -= ave_src
    target_points -= ave_tar
    
    # 组成法方程 (W’W) x = (W’b) 并利用最小二乘求解x
    Args = np.array([0, 0, 0, 0, 0, 0, 0], dtype='float64')
    parameters = np.array([1, 1, 1, 1, 1, 1, 1])

    # 当七参数的误差值之和大于1e-10时，迭代运算得到更精确的结果
    while np.fabs(np.array(parameters)).sum() > 1e-10 :
        W = points2W(source_points, Args)
        b = points2b(source_points, target_points, Args)
        qxx = np.linalg.inv(np.dot(W.transpose(), W))
        parameters = np.dot(np.dot(qxx, W.transpose()), b)
        Args += parameters

    # 打印七参数
    print("Args:")
    print(np.round(Args, 3))

    # 评定最小二乘模型的精度结果，并使用预留的已知坐标同名点来验证七参数模型的效果
    # 检查点坐标
    source_test_points = [
        (3381402.058, 395657.940, 32.728)
        ]
    
    target_test_points = [
        (3380972.424, 539704.811, 13.665)
        ]
    
    #归一化处理
    source_test_points = np.array(source_test_points - ave_src)
    
    # 单位权标准差，即所得模型的坐标精度
    sigma0 = np.sqrt((b*b).sum() / 2)
    # 参数标准差，即所得模型的七参数精度
    sigmax = sigma0 * np.sqrt(np.diag(qxx))
    print('单位权中误差: %.3f' % (sigma0))
    print('参数中误差:')
    print(np.round(sigmax,3))
    (x2, y2, z2) = ordinationConvert(source_test_points[0, 0], source_test_points[0, 1], source_test_points[0, 2], Args) + ave_tar
    print('模型预测结果: ')
    print('[(%.3f, %.3f, %.3f)]'%(x2, y2, z2))
    print('真实结果: ')
    print(target_test_points)



