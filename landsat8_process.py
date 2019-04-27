# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 10:09:23 2018
Landsat 8 data process program including:
    * Extract Cloud Mask
    * PanSharpen
    * Pencent Linear Strech
    * Resample

@author: Administrator
"""
import numpy as np

import rasterio
from rasterio.enums import Resampling
from rio_pansharpen import worker

import concurrent
import os

from image_process import gamma, bytscl, resample


def panSharpen_L8(src_paths, dst_path, weight=0.2, dst_dtype='uint8',
                  customwindow=1024, jobs=4, verbosity=False,
                  half_window=False, creation_opts=False):
    """PanSharpen Landsat 8 data using rio_pansharpen module. see
        https://github.com/mapbox/rio-pansharpen

    Note:
        Modified the _pad_window(wnd, pad) function to be compatible with
                latest version of rasterio.

    Args:
        src_paths (list) : Input Landsat 8 file paths
                [pan_path, r_path, g_path, b_path].
        dst_path (str): Pansharpened filename.
        weight (float): Weight of blue band , default 0.2 .
        dst_dtype (numpy dtype): Default 'uint8'.
        customwindow (integer): Window size.
        jobs (integer): Processors used.
        verbosity (bool): Whether or not to show the verbosity, default False.
        half_window (bool): Whether or not to use half window, default,False.
                Be careful using this method, may cause strip betwine windows.
        creation_opts (dict): Creation options to update the write profile.

    """
    worker.calculate_landsat_pansharpen(src_paths, dst_path, dst_dtype,
                                        weight, verbosity, jobs, half_window,
                                        customwindow=customwindow,
                                        out_alpha=False,
                                        creation_opts=creation_opts)


def extract_cloud_mask_from_QA(qa_band, out_tiff):
    """Extract cloud mask from  QA band(cloud,cloud shadow,cirrus with high
    confidence).
    See: https://landsat.usgs.gov/collectionqualityband

    Args:
        qa_band (str): Landsat 8 QA band.
        out_tiff (str): Output cloud mask file with geotiff format.
                    The raster value ranges in (0,1,2,3).
                    1: cloud with high confidence.
                    2: cloud shadow with high confidence.
                    3: cirrus with high confidence.
                    0: other area.

    """
    # read data
    with rasterio.Env(RIO_IGNORE_CREATION_KWDS=True):
        with rasterio.open(qa_band) as ds:
            a = ds.read(1)
            meta = ds.meta.copy()
    # pixcel values to be extracted
    cloud_high_conf = [2800, 2804, 2808, 2812, 6896, 6900, 6904, 6908]
    cloud_shadow_high = [2976, 2980, 2984, 2988, 3008, 3012, 3016, 3020,
                         7072, 7076, 7080, 7084, 7104, 7108, 7112, 7116]
    cirrus_confidence_high = [6816, 6820, 6824, 6828, 6848, 6852, 6856,
                              6860, 6896, 6900, 6904, 6908, 7072, 7076,
                              7080, 7084, 7104, 7108, 7112, 7116, 7840,
                              7844, 7848, 7852, 7872, 7876, 7880, 7884]
    # mask
    cloud_high_conf_mask = np.isin(a, cloud_high_conf)
    cloud_shadow_high_mask = np.isin(a, cloud_shadow_high)
    cirrus_confidence_high_mask = np.isin(a, cirrus_confidence_high)

    out_mask = np.zeros((meta['height'], meta['width']), dtype='uint8')

    out_mask[cirrus_confidence_high_mask] = 3
    out_mask[cloud_shadow_high_mask] = 2
    out_mask[cloud_high_conf_mask] = 1

    meta.update(dtype=rasterio.uint8)
    # write out
    with rasterio.open(out_tiff, 'w', **meta) as dst:
        dst.write(out_mask.astype(rasterio.uint8), 1)


def windowed_process(mask_ds, pan_ds, out, strechparams,
                     windowsize=1024, num_workers=4):
    """ Apply cloud mask and linear strech to panSharpend data .
    Split raster file into windows and then process for the purpose of
    parallel processing and cotrolling memory use.

    Args:
        mask_ds (rasterio dataset): Input cloud mask array.
        pan_ds (rasterio dataset): PanSharpened data.
        out (str): Output raster file with geotiff format.
        strechparams (list): Linear strech params for each band
                [(minvalue_b1,maxvalue_b1),(minvalue_b2,maxvalue_b2)...]
        windowsize (int): Lines or columns of each window, default 1024.
        num_workers (int): Max worker when using concurrent processing,
        default 4.

    """
    # open data with tiles
    profile = pan_ds.profile
    profile.update(blockxsize=windowsize, blockysize=windowsize, tiled=True)
    with rasterio.open(out, 'w', **profile) as dst:

        windows = [window for ij, window in dst.block_windows()]
        print(len(windows))

        # process using concurrent
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=num_workers
        ) as executor:
            # {result:(window,band_number)}
            futuresToExec = {}
            for window in windows:
                maskArr = mask_ds.read(1, window=window)
                for i in range(1, dst.count + 1):
                    imageArr = pan_ds.read(i, window=window)

                    futuresToExec[
                            executor.submit(apply_mask_and_then_strech,
                                            maskArr, imageArr,
                                            strechparams[i - 1][1],
                                            strechparams[i - 1][0])] = (
                                                                    window, i)
            # make sure all windows completed
            for future in concurrent.futures.as_completed(futuresToExec):
                index = futuresToExec[future]
                streched = future.result()
                dst.write(streched, window=index[0], indexes=index[1])


def apply_mask_and_then_strech(maskArr, imageArr, maxValue, minValue):
    """Apply cloud mask,linear strech and gamma strech image.

    Args:
        maskArr (numpy ndarray): Cloud mask data array.
        imageArr (numpy ndarray): Image data array.
        maxValue, minValue (float): Linear strech parameters.

    Returns:
        numpy ndarray: Processed data.

    """
    assert maskArr.shape == imageArr.shape
    masked = np.ma.filled(np.ma.masked_where(maskArr > 0, imageArr), 0)

    return gamma(bytscl(masked, maxValue=maxValue, minValue=minValue), 1.5)


def get_strech_value_for_masked_img(mask_ds, pan_ds, left_percent=2,
                                    right_percent=2):
    """Calculate the linear strech parameters (percent strech values outside
    the cloud area).

    Args:
        mask_ds (rasterio dataset): Input cloud mask array.
        pan_ds (rasterio dataset): PanSharpened data.
        left_percent, right_percent (float): Strech percent.

    Returns:
        list: linear strech params for each band
                    [(minvalue_b1,maxvalue_b1),(minvalue_b2,maxvalue_b2)...]

    """
    r = []
    maskArr = mask_ds.read(1)
    for i in range(1, pan_ds.count + 1):
        imageArr = pan_ds.read(i)
        masked = np.ma.filled(np.ma.masked_where(maskArr > 0, imageArr), 0)
        r.append(get_percentile(masked, left_percent, right_percent))
    return r


def get_percentile(inArr, left_percent, right_percent, nodata=0):
    """ Calculate the percentile value of designated percent .

    Args:
        inArr (numpy ndarray): Input array.
        left_percent, right_percent (float): The designated percent.

    Returns:
        tuple: The min and max value associated with the strech percent.

    Raise:
        ValueError: If Input strech percent great than 100 or
                    The max value less than the min value.

    """
    if left_percent >= 100 or right_percent >= 100:
        raise ValueError('Input strech percent must be less than 100.')
    a = inArr[inArr != nodata]
    minValue = np.percentile(a, left_percent, interpolation="nearest")
    maxValue = np.percentile(a, 100 - right_percent, interpolation="nearest")
    if maxValue <= minValue:
        raise ValueError('The max value should not be less than the min.')
    return (minValue, maxValue)


if __name__ == '__main__':

    para_dir = r'I:\landsat\untar'

    for d in os.listdir(para_dir):
        l8_dir = os.path.join(para_dir, d)
        print(l8_dir)

        # l8_dir = r'I:\landsat\untar\LC08_L1TP_122043_20180722_20180731_01_T1'
        l8_base_name = l8_dir.split(os.sep)[-1]


# ==================分辨率合成、云掩模、重采样======================================
        cld_msk_dir = r'I:\landsat\cld_msk'
        result_dir = r'I:\landsat\result'
        temp_dir = r'D:\temp'

        pan_path = os.path.join(l8_dir, l8_base_name + '_B8.TIF')
        r_path = os.path.join(l8_dir, l8_base_name + '_B4.TIF')
        g_path = os.path.join(l8_dir, l8_base_name + '_B3.TIF')
        b_path = os.path.join(l8_dir, l8_base_name + '_B2.TIF')
        qa_band = os.path.join(l8_dir, l8_base_name + '_BQA.TIF')

        src_paths = [pan_path, r_path, g_path, b_path]

        # test tempfile
        # import tempfile
        # tmpfile = tempfile.NamedTemporaryFile()
        # result
        cloud_mask_30m = os.path.join(
            cld_msk_dir, l8_base_name + '_cloud_mask.tif')
        final_result_10m = os.path.join(result_dir, l8_base_name + '_10m.tif')

        # intermediate
        pan_15m = temp_dir + os.sep + "pan15.tif"
        cloud_mask_resample_15m = temp_dir + os.sep + 'cld_resample5.tif'
        masked_paned_result_15m = temp_dir + os.sep + 'masked_paned_result.tif'

        # pansharpen
        print('pansharpening...')
        panSharpen_L8(src_paths, pan_15m, weight=0.2, dst_dtype='uint8',
                      customwindow=1024, jobs=4, verbosity=False,
                      half_window=False, creation_opts=False)

        # cloud mask
        print('cal cloud mask ...')
        extract_cloud_mask_from_QA(qa_band, cloud_mask_30m)

        # 对云数据重采样，应用于pansharpen后结果 强制重采样后数据行列数一致
        print('resample cloud mask ...')
        with rasterio.open(pan_path) as ds:
            width = ds.width
            height = ds.height

        resample(
            cloud_mask_30m,
            cloud_mask_resample_15m,
            15,
            width,
            height,
            method=Resampling.nearest)

        print('apply cloud mask and strech pansharpened image ...')
        pan_ds = rasterio.open(pan_15m)
        mask_ds = rasterio.open(cloud_mask_resample_15m)

        print('   get strech params..')
        s = get_strech_value_for_masked_img(mask_ds, pan_ds, 1, 2.5)
        print('   process..')
        windowed_process(
            mask_ds,
            pan_ds,
            masked_paned_result_15m,
            s,
            windowsize=1024,
            num_workers=4)

        mask_ds.close()
        pan_ds.close()

        # 最后重采样10m
        print('finally resample image to 10 meter ...')
        resample(
            masked_paned_result_15m,
            final_result_10m,
            10,
            method=Resampling.bilinear)
