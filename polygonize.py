# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 15:46:24 2018

@author: Administrator
"""

import os
import numpy as np
import rasterio
from rasterio import features

from fiona.crs import from_epsg
import fiona


def fill_empty_area(ds):
    """假设原影像数据空值为0，但内部很少空值，对3个波段数据取交集作为掩膜，然后填充内部的
    少量空值（可能是由于某一个波段数据为0引起，也可能是小范围的空数据）.

    Note:
        不通用，而且只用于只有”一块完整数据”的影像，最初目的是计算数据有效范围，后来基本没
    用到，暂时保留.

    Args:
        ds (rasterio dataset): 读入的栅格数据集.

    Returns:
        numpy ndarray: 填充后数组.

    """
    if ds.count == 3:
        # get mask of zero
        r_arr = src.read(1)
        g_arr = src.read(2)

        mask = np.logical_and(r_arr, g_arr)
        del r_arr, g_arr

        b_arr = src.read(3)
        mask = (np.logical_and(mask, b_arr)).astype('uint8')
        del b_arr

        # fill inside line by line
        # 用于只有一个有效区域
        # TODO applay to many areas not adjacent.
        nrow, ncol = mask.shape
        for r in range(nrow):
            if r % 5000 == 0:
                print(r)

            temp = np.where(mask[r] == 1)[0]
            if temp.size != 0:
                mask[r, temp.min(): temp.max()] = 1
        return mask


def mask_to_shp(src, out_shp, mask_value=0, driver='ESRI Shapefile'):
    """ 将掩膜数据矢量化（输出为shapefile格式）.

    Note:
        * 输出shapefile字段及属性写死在代码；
        * 最好只用于掩膜值为特定值的数据，如果将每一个像素矢量化会非常慢;

    Args:
        src (rasterio dataset): 输入的包含掩膜值的栅格数据集.
        out_shp (str): 输出shapefile文件.
        mask_value (float): 掩膜值，默认0.
        driver (str): 输出矢量类型, 默认 'ESRI Shapefile'.

    """
    if src.count != 1:
        return

    if os.path.isfile(out_shp):
        fiona.remove(out_shp, driver=driver)

    import mask
    # 分窗口读取，用于占用内存比较大数组
    windows = mask.window_list(10000, src.width, src.height)

    # 直接用于PS处理后每一层的蒙版导出结果
    for idx_window, window in enumerate(windows):
        print("process {} of total {}\n".format(idx_window + 1, len(windows)))

        # read data of current window
        mask_data = src.read(1, window=window)

        # transfrom of current window
        transform = rasterio.windows.transform(window, src.transform)

        # get features of the same value from an array
        none_zero = features.shapes(mask_data, mask=mask_data == mask_value,
                                    connectivity=4, transform=transform)
        mask_data = None

        # get record
        results = (
                {'properties': {'date': '20181007'}, 'geometry': s}
                for i, (s, v) in enumerate(none_zero))

        # shapefile schema
        source_schema = {
                'geometry': 'Polygon',
                'properties': {
                        'date': 'str:20'}
                }

        # write to shapefile using fiona
        if results is not None:
            for r in results:
                # TODO merge result features by order
                try:
                    with fiona.open(out_shp, 'a', driver=driver,
                                    crs=from_epsg(src.crs.to_epsg()),
                                    schema=source_schema) as c:
                        c.write(r)
                except OSError:
                    with fiona.open(out_shp, 'w', driver=driver,
                                    crs=from_epsg(src.crs.to_epsg()),
                                    schema=source_schema) as c:
                        c.write(r)


if __name__ == '__main__':

    in_r = 'H:/未标题-1.tif'
    out_shp = "d:/testt3.shp"
    driver = 'ESRI Shapefile'

    with rasterio.open(in_r) as src:
        mask_to_shp(src, out_shp)
