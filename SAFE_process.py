# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 15:02:18 2018

对哨兵2数据中TCI数据简单拉伸用来显示.

Note:
    为了方便独立部署,其中很多函数有重复.

@author: Administrator
"""

import os
import shutil
from zipfile import ZipFile

import rasterio
from rasterio.warp import reproject, Resampling, calculate_default_transform

import glob
import numpy as np

import xml.etree.ElementTree as ET

from mosaic import merge_one_by_one

def gamma(image, gamma=1.0):
    """ Apply gamma correction to the channels of the image.

    Note:
        Only apply to 8 bit unsighn image.

    Args:
        image (numpy ndarray): The image array.
        gamma (float): The gamma value.

    Returns:
        Numpy ndarray: Gamma corrected image array.

    Raises:
        ValueError: If gamma value less than 0 or is nan.

    """
    if gamma <= 0 or np.isnan(gamma):
        raise ValueError("gamma must be greater than 0")

    norm = image/256.
    norm **= 1.0 / gamma
    return (norm * 255).astype('uint8')


def bytscl(argArry, maxValue=None, minValue=None, nodata=None, top=255):
    """将原数组指定范围(minValue ≤ x ≤ maxValue)数据拉伸至指定整型范围(0 ≤ x ≤ Top),
    输出数组类型为无符号8位整型数组.

    Note:
        Dtype of the output array is uint8.

    Args:
        argArry (numpy ndarray): 输入数组.
        maxValue (float): 最大值.默认为输入数组最大值.
        minValue (float): 最小值.默认为输入数组最大值.
        nodata (float or None): 空值，默认None，计算时排除.
        top (float): 输出数组最大值，默认255.

    Returns:
        Numpy ndarray: 线性拉伸后的数组.

    Raises:
        ValueError: If the maxValue less than or equal to the minValue.

    """
    mask = (argArry == nodata)
    retArry = np.ma.masked_where(mask, argArry)

    if maxValue is None:
        maxValue = np.ma.max(retArry)
    if minValue is None:
        minValue = np.ma.min(retArry)

    if maxValue <= minValue:
        raise ValueError("Max value must be greater than min value! ")

    retArry = (retArry - minValue) * float(top) / (maxValue - minValue)

    retArry[argArry < minValue] = 0
    retArry[argArry > maxValue] = top
    retArry = np.ma.filled(retArry, 0)
    return retArry.astype('uint8')


def rmdirs(para_dir):
    """Remove all directory under parent directory.

    Args:
        para_dir (str): Parental directory.

    """
    if not os.path.isdir(para_dir):
        print("Input params must be a directory!")
        return

    for f in os.listdir(para_dir):
        complete_dir = os.path.join(para_dir, f)
        if os.path.isdir(complete_dir):
            shutil.rmtree(complete_dir)


def unzip_files_in_safe_endswith(s2_safe_file, unzip_dir, ends_str):
    """ Extract sentinel 2 TCI file from s2 SAFE file(.zip) to a directory.

    Args:
        s2_safe_file (str): Origin S2 SAFE file.
        pci_dir (str): Destinate directory.
        ends_str (str): The end strings of the file to be extracted.

    """
    with ZipFile(s2_safe_file) as myzip:
        for m in myzip.namelist():
            if m.endswith(ends_str):
                print(m)
                outfile = myzip.extract(m, path=unzip_dir)
                shutil.move(outfile, os.path.join(unzip_dir,
                                                  os.path.basename(outfile)))
                break
    rmdirs(unzip_dir)


def unzip_pci_image(s2_safe_file, pci_dir):
    """Extract sentinel 2 TCI file from s2 SAFE file(.zip) to a directory.

    Args:
        s2_safe_file (str):  origin S2 SAFE file.
        pci_dir (str): Destinate directory.

    """
    unzip_files_in_safe_endswith(s2_safe_file, pci_dir, '_TCI.jp2')


def read_pci_from_zipfile(s2_safe_file):
    """ Directly read sentinel 2 TCI file from s2 SAFE file(.zip)
    using rasterio.

    Args:
        s2_safe_file (str): Original s2 SAFE file.

    Returns:
        rasterio dataset.

    """
    with ZipFile(s2_safe_file) as myzip:
        for m in myzip.namelist():
            # print(m)
            if 'MSIL2A' in s2_safe_file:
                ends = '_TCI_10m.jp2'
            else:
                ends = '_TCI.jp2'
            if m.endswith(ends):
                ds = rasterio.open('zip:{0}!/{1}'.format(s2_safe_file, m))
                break
    return ds


def process_pci_dataset(src, outfile, dst_crs='EPSG:4326',
                        resolution=None, resample_method=Resampling.nearest):
    """Simpely process sentinel 2 PCI image(line strech and apply gamma
    value) then projected to WGS 1984.

    Args:
        src (rasterio dataset): Input rasterio dataset.
        outfile (str): Result file (TIFF format).
        dst_crs : Target coordinate reference system.
        resolution (:tuple: x resolution, y resolution or float, optional):
            Target resolution, in units of target coordinate reference system.
        resample_method: Resampling method to use.  One of the following:
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
            Resampling.q3 (GDAL >= 2.2),

    """
    # calculate_default_transform
    if resolution is not None:
        transform, width, height = calculate_default_transform(
                src.crs, dst_crs, src.width, src.height,
                resolution=resolution, *src.bounds)
    else:
        transform, width, height = calculate_default_transform(
                src.crs, dst_crs, src.width, src.height, *src.bounds)

    kwargs = src.meta.copy()
    kwargs.update({
        'driver': 'GTIFF',
        'nodata': 0,
        'crs': dst_crs,
        'transform': transform,
        'width': width,
        'height': height
    })
    # reproject data
    with rasterio.open(outfile, 'w', **kwargs) as dst:
        for i in range(1, src.count + 1):
            # strech image data
            # green band
            if i == 2:
                source = gamma(bytscl(src.read(i), maxValue=155), 1.4)
            # other band
            else:
                source = gamma(bytscl(src.read(i), maxValue=160), 1.3)
            reproject(
                    source=source,
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=resample_method)


def read_extent_from_INSPIRE_file(s2_safe_file):
    """Get product extnet from meta data INSPIRE.xml.

    # TODO not complete.

    """
    with ZipFile(s2_safe_file) as myzip:
        for m in myzip.namelist():
            if m.endswith('INSPIRE.xml'):
                tree = ET.parse(myzip.open(m))

                gmd = '{http://www.isotc211.org/2005/gmd}'
                gco = '{http://www.isotc211.org/2005/gco}'
                root = tree.getroot()
                t = root.find('./{0}identificationInfo/{0}MD_DataIdentific'
                              'ation/{0}abstract/{1}CharacterString'.format(
                                      gmd, gco))
                coords = t.text.split(' ')
                while '' in coords:
                    coords.remove('')

                coords = [float(c) for c in coords]


def unzip_band_meta_image(z, s2_unzipdir):
    """Extract data from S2 SAFE file and orgnized like the amazon
    S2 dataset does. With directory name like 'S2_49QFC_20180609_5901',
    data like 'B01.jp2', 'B12.jp2'.

    """
    namelist = z.split('_')
    outdir = (os.path.join(s2_unzipdir, 'S2_' + namelist[-2][1:] + '_' +
                           namelist[-5][:8] + '_' + namelist[-1][11:15]))

    if not os.path.exists(outdir):
        os.mkdir(outdir)

    with ZipFile(z) as myzip:
        for m in myzip.namelist():
            if m.endswith('.jp2') and 'B' in m:
                outfile = myzip.extract(m, path=outdir)

                shutil.move(outfile, os.path.join(
                        outdir, os.path.basename(outfile)[-7:]))

            elif m.endswith('MTD_TL.xml') or m.endswith('MSK_CLOUDS_B00.gml'):
                outfile = myzip.extract(m, path=outdir)

                shutil.move(outfile, os.path.join(
                        outdir, os.path.basename(outfile)))
    rmdirs(outdir)


def ds2tif(ds, out_tif):
    """datasets to tif"""
    kargs = ds.meta.copy()
    kargs.update({
            "driver": "GTIFF",
            "nodata": src.nodata,
            "compress": 'lzw',
            })

    # write out dataset by dataset
    with rasterio.open(out_tif, 'w', BIGTIFF='YES', **kargs) as dst:
        dst.write(ds.read())

if __name__ == '__main__':

    s2_zipdir = r'E:\S2\原始'
    s2_unzipdir = r'I:\sentinel\unzip'
    pci_dir = r'E:\S2\PCI'
    
    datasets = []

    s2_ziplist = glob.glob(os.path.join(s2_zipdir, 'S2*L2A_20210730*_T48*.zip'))
    print('total {} files. '.format(len(s2_ziplist)))
    for i, z in enumerate(s2_ziplist):
        print('extracting and processing number : {}'.format(i+1))
        # unzip_band_meta_image(z, s2_unzipdir)

        outfile = os.path.join(pci_dir,
                                os.path.basename(z).replace('.zip', '.tif'))
        print(z)
        src = read_pci_from_zipfile(z)
        # ds2tif(src, outfile)
        # unzip_pci_image(z,pci_dir)
        # process_pci_dataset(src, outfile, resolution=1e-4)
        # 金字塔
        # os.system("gdaladdo -ro {}".format(outfile))
        
        datasets.append(src)

        # src.close()
    mosaic_file = r"E:\S2\PCI\s2_0730_T48_1.tif"
    merge_one_by_one(datasets, mosaic_file)
    # 金字塔
    os.system("gdaladdo -ro {}".format(mosaic_file))
