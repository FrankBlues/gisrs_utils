# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 11:49:59 2018
Mosaic rasters.
@author: Administrator
"""

import rasterio
from rasterio.merge import merge
from rasterio.transform import Affine
from rasterio import windows

import glob
import os
import math

import sys


def merge_rio(src_datasets_to_mosaic, output, res=None, nodata=None,
              crs=None, compress='lzw', method='first'):
    """Merge raster datasets to one raster using rasterio.

    Args:
        src_datasets_to_mosaic (list): List of rasterio datasets.
        output (str): Output raster file.
        res (float): Resolution of output raster.
        nodata (float): Nodata value of output raster.

    Usage:
        >>> src_files_to_mosaic = [rasterio.open(f) for f in rfiles]
        >>> merge_rio(src_files_to_mosaic, output, res=10)

    """
    mosaic, out_trans = merge(src_datasets_to_mosaic, res=res, nodata=nodata, method=method)

    try:
        src = src_datasets_to_mosaic[0]
        out_meta = src.meta.copy()
        out_meta.update({
                "driver": "GTiff",
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": out_trans,
                "crs": src.crs if (crs is None) else crs,
                "nodata": nodata,
                "compress": compress,
                "BIGTIFF": 'YES'
                })

        with rasterio.open(output, 'w', **out_meta) as dst:
            dst.write(mosaic)
    finally:
        src.close()
        del mosaic
        del src_datasets_to_mosaic


def merge_one_by_one(datasets, out, nodata=None, compress='lzw'):
    """ Merge raster datasets to one raster,unlike the obove one, this function
    write dataset one by one to the designate file to reduce memory use.

    Args:
        datasets (list): List of input datasets.
        out (str): The output raster.
        compress (str): Compression method,(lzw,zip,None),default lzw.

    """
    first = datasets[0]
    first_res = first.res
    if nodata is None:
        nodataval = first.nodatavals[0]
    else:
        nodataval = nodata
    dtype = first.dtypes[0]

    # Determine output band count
    output_count = first.count

    # Extent from extent of all inputs
    xs = []
    ys = []
    for src in datasets:
        left, bottom, right, top = src.bounds
        xs.extend([left, right])
        ys.extend([bottom, top])
    dst_w, dst_s, dst_e, dst_n = min(xs), min(ys), max(xs), max(ys)

    # out trans
    output_transform = Affine.translation(dst_w, dst_n)

    # res
    res = first_res
    output_transform *= Affine.scale(res[0], -res[1])
    # print(output_transform)

    # Compute output array shape. We guarantee it will cover the output
    # bounds completely
    output_width = int(math.ceil((dst_e - dst_w) / res[0]))
    output_height = int(math.ceil((dst_n - dst_s) / res[1]))

    # Adjust bounds to fit
    dst_e, dst_s = output_transform * (output_width, output_height)

    kargs = {
            "driver": "GTIFF",
            "count": output_count,
            "nodata": nodataval,
            "dtype": dtype,
            "height": output_height,
            "width": output_width,
            "transform": output_transform,
            "crs": first.crs,
            "compress": compress,
            }

    # write out dataset by dataset
    with rasterio.open(out, 'w', BIGTIFF='YES', **kargs) as dst:
        print("Total {} windows.".format(len(datasets)))
        for i, src in enumerate(datasets):
            src_w, src_s, src_e, src_n = src.bounds

            # Compute the destination window
            dst_window = windows.from_bounds(
                    src_w, src_s, src_e, src_n, output_transform, precision=7)
            # print(src)
            print("Writing window {}..".format(i + 1))
            # print(dst_window)
            dst.write(src.read(), window=dst_window)


def merge_within_extent(files, out, extent=None, res=None, nodata=None,
                        compress='lzw'):
    """ Merge raster files within an extent to one raster.

    Args:
        files (list): List of input raster.
        out (str): The output raster.
        extent (list): an extent (left, bottom, right, top).
        res (tuple): out resolution (x_res, y_res).
        nodata (float): out nodata.
        compress (str): Compression method,(lzw,zip,None),default lzw.

    """
    print("Read metadata from first file.")
    datasets = [rasterio.open(f) for f in files]
    first = datasets[0]
    first_res = first.res
    if nodata is None:
        nodataval = first.nodatavals[0]
    else:
        nodataval = nodata
    dtype = first.dtypes[0]

    # Determine output band count
    output_count = first.count

    print("Calculate output geotransform.")
    # Extent from extent of all inputs
    xs = []
    ys = []
    if extent is None:
        for src in datasets:
            left, bottom, right, top = src.bounds
            xs.extend([left, right])
            ys.extend([bottom, top])
        dst_w, dst_s, dst_e, dst_n = min(xs), min(ys), max(xs), max(ys)
    else:
        dst_w, dst_s, dst_e, dst_n = extent

    # out trans
    output_transform = Affine.translation(dst_w, dst_n)

    # res
    if res is None:
        res = first_res

    if isinstance(res, (float, str, int)):
        res = (float(res), float(res))

    output_transform *= Affine.scale(res[0], -res[1])
    print("  Output transform: ", output_transform.to_gdal())

    # Compute output array shape. We guarantee it will cover the output
    # bounds completely
    output_width = int(math.ceil((dst_e - dst_w) / res[0]))
    output_height = int(math.ceil((dst_n - dst_s) / res[1]))
    print(f"Output width: {output_width}, height: {output_height}")

    # Adjust bounds to fit
    dst_e, dst_s = output_transform * (output_width, output_height)

    kargs = {
            "driver": "GTIFF",
            "count": output_count,
            "nodata": nodataval,
            "dtype": dtype,
            "height": output_height,
            "width": output_width,
            "transform": output_transform,
            "crs": first.crs,
            "compress": compress,
            }

    print("Mosaic files...")
    # write out dataset by dataset
    with rasterio.open(out, 'w+', BIGTIFF='YES', **kargs) as dst:
        print("Total {} files.".format(len(datasets)))
        for i, src in enumerate(datasets):
            current_file = files[i]
            print(f"  Process file {current_file}..")
            src_w, src_s, src_e, src_n = src.bounds

            # not intersect
            if src_w >= dst_e or src_s >= dst_n or src_e <= dst_w or src_n <= dst_s:
                print("    File not intersected with target extent, next.")
                continue

            print("    Calculate window for source and dest file.")
            # inter bounds
            inter_bound = (max(src_w, dst_w), max(src_s, dst_s), min(src_e, dst_e), min(src_n, dst_n))

            # Compute the source window
            src_window = windows.from_bounds(
                    inter_bound[0], inter_bound[1], inter_bound[2], inter_bound[3], src.transform, precision=15)

            # Compute the destination window
            dst_window = windows.from_bounds(
                    inter_bound[0], inter_bound[1], inter_bound[2], inter_bound[3], output_transform, precision=15)

            # only update nodata values
            if nodataval is not None:
                dst_array = dst.read(window=dst_window)
                # make sure source and dest data has the same shape.
                src_array = src.read(out_shape=dst_array.shape, window=src_window)

                mask = dst_array == nodataval
                dst_array[mask] = src_array[mask]
                mask = src_array = None

                print("    Writing out...")
                dst.write(dst_array, window=dst_window)

            else:
                print("    Writing out...")
                dst.write(src.read(window=src_window), window=dst_window)

if __name__ == '__main__':

    tiles_dir = r'D:\work\data\影像样例\445825_246658_pd2020008643\PD2020008643\IMG_PHR1B_MS_002'
    tiles = glob.glob(os.path.join(tiles_dir, '*.TIF'))
    file_dir = r'E:\S2'
    date = '20211203'
    files = glob.glob(os.path.join(file_dir, '*{0}*/GRANULE/*/IMG_DATA/*{0}*_TCI.jp2'.format(date)))
    print(files)

    tiles_dir = r'D:\tmp\mosaic'
    files = glob.glob(os.path.join(tiles_dir, '*.tiff'))

    # files = glob.glob("/mnt/cephfs/rsi/result/69/RSI202206061510577550000/defog/*/*/*/*.tiff")

    merge_within_extent(files, 'd:/aaaabcdefg.tif', extent=(110, 34, 110.5, 34.333333), res='5e-5')


    # datasets = [rasterio.open(f) for f in files]
    # out_file = os.path.join("E:/S2", f's2_{date}.tif')
    # out_file = "/mnt/cephfs/rsi/data/Test/mlm/mosaic444.tif"

    # merge_one_by_one([rasterio.open(f) for f in tiles], 'G:/temp/SN3_image_shanghai_test_bigtiff.tif')

    # merge_rio(datasets, out_file)
    # for x in os.listdir(tiles_dir):
    #     print(x)

    #     tiles = glob.glob(os.path.join(tiles_dir, x, '*.png'))
    #     merge_rio([rasterio.open(f) for f in tiles], os.path.join(tiles_dir, x + '.tif'))




# =============================================================================
#     rasterDir = r'D:\work\data\土地利用30'
#     output = r'D:\work\data\land_use_global_30.tif'
#     output_dir = r'D:\work\data'
#
#     rfiles = glob.glob(os.path.join(rasterDir, '*.tif'))
#     rfiles = rfiles[-16:]
#     import re
#     # 找到EW和NS间的数字,即经度.
#     pattern = re.compile(r'(?<=[EW])\d+\.?\d*(?=[NS])')
#
#     batch = []
#     flag = -1
#     name_suffix = ''
#
#     for r in rfiles:
#         lons = re.findall(pattern, r)
#         if len(lons) != 1:
#             print(r)
#             print("find more than 1.")
#         lon = lons[0]
#         if flag != int(lon):
#             if len(batch) > 0:
#                 print("正在镶嵌当前经度数据：{0},共找到{1}个文件.".format(name_suffix, len(batch)))
#                 src_files_to_mosaic = [rasterio.open(f) for f in batch]
#                 merge_rio(src_files_to_mosaic, os.path.join(output_dir, 'land_use_30m_' + name_suffix + '.tif'), nodata=0)
#                 # break
#
#             flag = int(lon)
#             batch = []
#             batch.append(r)
#             # print('E' + str(lon))
#             if 'E' + str(lon) in r:
#                 name_suffix = 'E' + str(lon)
#             elif 'W' + str(lon) in r:
#                 name_suffix = 'W' + str(lon)
#             # print(name_suffix)
#
#
#         else:
#             batch.append(r)
#     src_files_to_mosaic = [rasterio.open(f) for f in batch]
#     merge_rio(src_files_to_mosaic, os.path.join(output_dir, 'land_use_30m_' + name_suffix + '.tif'), nodata=0)
#
#
# =============================================================================





    # src_files_to_mosaic = [rasterio.open(f) for f in rfiles]
    # merge_one_by_one(src_files_to_mosaic, output, nodata=0, compress='lzw')
    # merge_rio(src_files_to_mosaic,output,nodata=0)
    # del src_files_to_mosaic

# =============================================================================
#     s2_files_dir = r'E:\liuan\S2'
#
#     dic = {}
#     for f in os.listdir(s2_files_dir):
#         if f.endswith('.jp2'):
#             s2_file = os.path.join(s2_files_dir,f)
#
#             filedate = f[7:15]
#             if filedate not in dic.keys():
#                 dic[filedate] = [s2_file]
#             else:
#                 dic[filedate].append(s2_file)
#
#     for k in dic:
#         if len(dic[k]) > 1:
#
#             output = os.path.join(s2_files_dir,
#                    'mosaic_T50SMA_T50SMB_' + k + '_TCI.tif')
#
#             src_files_to_mosaic = [rasterio.open(f) for f in dic[k]]
#             merge_rio(src_files_to_mosaic,output,res=10)
#         else:
#             print(dic[k][0])
# =============================================================================

# =============================================================================
#
#     #投影一致
#     outDir = r'I:\sentinel\处理\20181007_'
#     outMosaic = os.path.join(outDir,'mosaic_20181007_.tif')
#     print(outMosaic)
#
#     #src_files_to_mosaic = [rasterio.open(f) for f in rfiles]
#     resultlist = glob.glob(os.path.join(outDir,'S2_49*.tif'))
#     print(resultlist)
#     src_files_to_mosaic = [rasterio.open(f) for f in resultlist]
#     merge_rio(src_files_to_mosaic,outMosaic,res=10)#,nodata=-3000)
#
#     del src_files_to_mosaic
#
#     for f in resultlist:
#         os.remove(f)
#         if os.path.isfile(f.replace('.tif','.jpg')):
#             os.remove(f.replace('.tif','.jpg'))
# =============================================================================

    """
    with rasterio.open(rfiles[0]) as src:
        for f in rfiles:
            ds = rasterio.open(f)
            if ds.crs == src.crs:
                src_files_to_mosaic_proj1.append(ds)

            else:
                src_files_to_mosaic_proj2.append(ds)


    merge_rio(src_files_to_mosaic_proj1,output1,res=10)
    merge_rio(src_files_to_mosaic_proj2,output2,res=10)

    del src_files_to_mosaic_proj1
    del src_files_to_mosaic_proj2
    """
