# -*- coding: utf-8 -*-
"""
哨兵2数据真彩色合成处理，包括:
    绿波段增强；
    应用云掩膜；
    图像增强并合成（百分比拉伸、gamma拉伸）；
    镶嵌同一天数据
不足:
    应用于早期从亚马逊云下载数据，文件夹组织方式需一致
    拉伸参数需要反复测试,但是同一天或临近日期数据可以用相同参数
"""

import os
from io_utils import read_raster_gdal, read_json
from image_process import new_green_band, zoom, apply_mask
from image_process import gamma, bytscl, composite_band, pilImage
import numpy as np

import logging
# import image_process.pilImage as im
logger = logging.getLogger(__name__)

logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler(r'F:\SENTINEL\处理\params.log')
fh.setLevel(logging.DEBUG)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.ERROR)
# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s '
                              '- %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(fh)
logger.addHandler(ch)


def get_new_RGB_Arr(blueBand, greenBand, redBand, irBand,
                    cldMask=None, maskValue=0):
    """采用gdal读取输入数据，利用近红外波段增强绿光波段，如果有云掩膜则对每个波段应用云掩膜.

    Args:
        blueBand, greenBand, redBand, irBand (str): 输入数据，对应B2|B3|B4|B8波段.
        cldMask (numpy ndarray): 云掩膜数据，默认None,不去云.
        maskValue: 云掩膜中的掩膜值，默认0.

    Returns:
        list: 红,绿,蓝波段数组,地理变换及投影信息列表.

    """
    bandInfo = read_raster_gdal(blueBand)
    bArr = bandInfo[0]
    geoTrans = bandInfo[1]
    projInfo = bandInfo[2]
    gArr = read_raster_gdal(greenBand)[0]

    rArr = read_raster_gdal(redBand)[0]
    irArr = read_raster_gdal(irBand)[0]

    gg = new_green_band(gArr, irArr, 0.07)

    del gArr, irArr, bandInfo

    if cldMask is not None:
        # 来自fmask算法，分辨率为20米
        cldArr = read_raster_gdal(cldMask)[0]
        if cldArr.shape != rArr.shape:
            print("云掩膜重采样")
            cldArr[np.where(cldArr == 0)] = 5
            cldArr[np.logical_or(cldArr == 2, cldArr == 3)] = 0
            cldArr[np.where(cldArr > 0)] = 1

            cldArr = zoom(cldArr, [rArr.shape[0]/cldArr.shape[0],
                                   rArr.shape[1]/cldArr.shape[1]])

        rArr = apply_mask(rArr, cldArr, maskValue=maskValue, fillvalue=0)
        gg = apply_mask(gg, cldArr, maskValue=maskValue, fillvalue=0)
        bArr = apply_mask(bArr, cldArr, maskValue=maskValue, fillvalue=0)
        del cldArr
    return [rArr, gg, bArr, geoTrans, projInfo]


def get_input_params(root, cloud_mask_file=None):
    """获取计算参数

    Args:
        root(str): 数据目录.
        cloud_mask_file (str): 云掩膜数据名称，与其它数据在同一目录，默认None.

    Returns:
        list: 参数列表，包括蓝光波段、绿光波段、红光波段、近红外波段、云掩膜完整路径及
            输出数据的名称.

    """
    base = root + os.sep
    tileInfo = base + 'tileInfo.json'
    blueBand = base + 'B02.jp2'
    greenBand = base + 'B03.jp2'
    redBand = base + 'B04.jp2'
    irBand = base + 'B08.jp2'
    if cloud_mask_file is not None:
        cldMask = base + cloud_mask_file
    else:
        cldMask = None

    # 读原数据
    # jsonf = read_json(tileInfo)
    # outName = jsonf['productName']
    outName = root.split(os.sep)[-1]
    return [blueBand, greenBand, redBand, irBand, cldMask, outName]


def get_strech_params(sentinelDir, tiletodefineArgs, cloud_mask_file=None,
                      strech_value=[2, 2, 2, 2, 2, 2]):
    """ 选取合适数据,计算各波段线性拉伸参数,如果有云掩膜,只利用云区以外区域计算

    Args:
        sentinelDir (str): 哨兵2数据父文件夹,哨兵2数据在该文件夹下以文件夹形式存放.
        tiletodefineArgs (str): 用来选取合适数据做拉伸计算.
        cloud_mask_file (str): 云掩膜数据文件.
        strech_value (list or float): 红、绿、蓝波段分别需要拉伸最大及最小百分比,
                默认[2,2,2,2,2,2].

    Returns:
        list: 红、绿、蓝3个波段最小最大拉伸取值.

    """
    if isinstance(strech_value, list):
        strech_list = strech_value
        assert len(strech_value) == 6
        strech_list = [strech_value[0], 100 - strech_value[1], strech_value[2],
                       100 - strech_value[3], strech_value[4],
                       100 - strech_value[5]]
    else:
        # strech_value is a single number
        strech_list = [strech_value, 100-strech_value, strech_value,
                       100-strech_value, strech_value, 100-strech_value, ]
    rleft, rright, gleft, gright, bleft, bright = strech_list
    logger.info("Calculae strech params using tile ({0}) with cloud "
                "mask ({1})".format(tiletodefineArgs, cloud_mask_file))
    for root, dirs, files in os.walk(sentinelDir):
        if tiletodefineArgs in root:
            print(root)

            blueBand, greenBand, redBand, irBand, cldMask, outName = \
                get_input_params(root, cloud_mask_file)

            if tiletodefineArgs in outName:
                # 20180309T030539_N0206_R075_T49QEE
                rArr, gg, bArr, geoTrans, projInfo = get_new_RGB_Arr(
                        blueBand, greenBand, redBand,
                        irBand, cldMask=cldMask)

                rArr = rArr[rArr != 0]
                minValueR = np.percentile(rArr, rleft, interpolation="nearest")
                maxValueR = np.percentile(rArr, rright,
                                          interpolation="nearest")
                gg = gg[gg != 0]
                minValueG = np.percentile(gg, gleft, interpolation="nearest")
                maxValueG = np.percentile(gg, gright, interpolation="nearest")
                bArr = bArr[bArr != 0]
                minValueB = np.percentile(bArr, bleft, interpolation="nearest")
                maxValueB = np.percentile(bArr, bright,
                                          interpolation="nearest")

                del bArr, rArr, gg
                break

    logger.info("min R Value:{0} max R value: {1}".format(
            minValueR, maxValueR))
    logger.info("min G Value:{0} max G value: {1}".format(
            minValueG, maxValueG))
    logger.info("min B Value:{0} max B value: {1}".format(
            minValueB, maxValueB))
    return [minValueR, maxValueR, minValueG, maxValueG, minValueB, maxValueB]


def process_stretch(sentinelDir, outDir, datetoprocess, cloud_mask_files,
                    strechRGBrange, test=False):
    """ 对数据进行拉伸增强处理.

    Args:
        sentinelDir (str): 哨兵2数据父文件夹,哨兵2数据在该文件夹下以文件夹形式存放.
        outDir (str): 处理后结果存放目录.
        datetoprocess (str): 日期，处理同一天数据.
        cloud_mask_files (str): 云掩膜数据文件.
        strechRGBrange (list): 红、绿、蓝3个波段最小最大拉伸取值.
        test (bool): 是否测试,测试一景数据看效果后再处理其它,默认False.

    Returns:
        list: 成功处理后的文件列表方便后续处理.

    """
    outfilelist = []
    minValueR, maxValueR, minValueG, maxValueG, minValueB, maxValueB = (
            strechRGBrange)
    for root, dirs, files in os.walk(sentinelDir):
        if datetoprocess in root:
            blueBand, greenBand, redBand, irBand, cldMask, outName = \
                get_input_params(root, cloud_mask_file)

            if datetoprocess in outName:
                print(outName)
                outfile = outDir + outName + '.tif'
                rArr, gg, bArr, geoTrans, projInfo = get_new_RGB_Arr(
                        blueBand, greenBand, redBand,
                        irBand, cldMask=cldMask)

                b1 = gamma(bytscl(bArr, maxValue=maxValueB,
                                  minValue=minValueB), 1.5)
                b2 = gamma(bytscl(gg, maxValue=maxValueG,
                                  minValue=minValueG), 1.5)
                b3 = gamma(bytscl(rArr, maxValue=maxValueR,
                                  minValue=minValueR), 1.5)

                composite_band([b3, b2, b1], outfile, geoTrans, projInfo)

                # 缩略图
                im = pilImage([b3, b2, b1])
                im.saveThumb(outDir + outName + '.jpg')
                del bArr, rArr, gg, b1, b2, b3
                outfilelist.append(outfile)
                if test:
                    break
    return outfilelist


if __name__ == '__main__':
    outDir = 'I:/sentinel/处理/20181029/'
    if not os.path.exists(outDir):
        os.mkdir(outDir)

    sentinelDir = r'I:\sentinel\unzip'

    continueProcessAfterCalStrechParams = 1  # True or False
    strech_value = [1.5, 1, 0.5, 1, 0.001, 1]
    # 用来计算拉伸参数
    tiletodefineArgs = '49QFE_20181007'
    # 处理该日期数据
    datetoprocess = tiletodefineArgs[-8:]
    datetoprocess = '20181029'

# ==云掩膜数据，不需要或没有取None==================================================
#     cloud_mask_file = None
#     cloud_mask_file = 'cld_mask_gml.tif'
#     cloud_mask_file = 'cloud.img'
# =============================================================================

    cloud_mask_file = 'cloud.img'
    cloud_mask_file = None
    test = False

    print("计算拉伸参数")
    strechRGBrange = get_strech_params(sentinelDir, tiletodefineArgs,
                                       cloud_mask_file,
                                       strech_value=strech_value)

    minValueR, maxValueR, minValueG, maxValueG, minValueB, maxValueB = (
            strechRGBrange)
#    minValueR = 335
    minValueG = 588
    minValueB = 600
#    maxValueR = 2313
#    maxValueG = 2221
#    maxValueB = 2480
#    strechRGBrange = [minValueR,maxValueR,minValueG,
#                      maxValueG,minValueB,maxValueB]
    print("min R Value:{0} max R value: {1}".format(minValueR, maxValueR))
    print("min G Value:{0} max G value: {1}".format(minValueG, maxValueG))
    print("min B Value:{0} max B value: {1}".format(minValueB, maxValueB))

    # cloud_mask_file = 'cld_mask_gml.tif'
    # cloud_mask_file = 'cld_mask_gml.tif'
    # cloud_mask_file = None
    # minValueR,maxValueR,minValueG,maxValueG,minValueB, maxValueB = (
    #      200 ,1800,300 ,1600,700 ,1500)

    if continueProcessAfterCalStrechParams:

        print("拉伸处理")
        resultlist = process_stretch(sentinelDir, outDir, datetoprocess,
                                     cloud_mask_file, strechRGBrange,
                                     test=test)

        if len(resultlist) > 1:
            # 投影一致
            projlist = [r.split('_')[-2][:3] for r in resultlist]

            assert(projlist.count(projlist[0]) == len(projlist))
            print('镶嵌')
            outMosaic = os.path.join(outDir,
                                     'mosaic_{0}.tif'.format(datetoprocess))

            import rasterio
            from mosaic import merge_rio
            src_files_to_mosaic = [rasterio.open(f) for f in resultlist]
            merge_rio(src_files_to_mosaic, outMosaic, res=10)

            del src_files_to_mosaic

            for f in resultlist:
                os.remove(f)
                if os.path.isfile(f.replace('.tif', '.jpg')):
                    os.remove(f.replace('.tif', '.jpg'))
