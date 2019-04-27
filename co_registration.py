# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 10:39:51 2019

测试几何校正工具
See https://gitext.gfz-potsdam.de/danschef/arosics
https://www.mdpi.com/2072-4292/9/7/676

未测试
https://github.com/andrestumpf/coregis_public

"""


if __name__ == '__main__':

    # Global coregistration test
    from geoarray import GeoArray
    from arosics import COREG

    im_reference = r'I:\sentinel\unzip\T49QEE_20180930T030541_B08.jp2'
    im_target = r'I:\sentinel\unzip\T49QEE_20181005T030549_B08.jp2'
    # im_target = r'D:\T50QKM_geo.tif'
    # get a sample numpy array with corresponding geoinformation as reference
    geoArr = GeoArray(im_reference)

    ref_ndarray = geoArr[:]       # numpy.ndarray with shape (10980, 10980)
    ref_gt = geoArr.geotransform  # GDAL geotransform
    ref_prj = geoArr.projection   # projection as WKT string

    # get a sample numpy array with corresponding geoinformation as target
    geoArr = GeoArray(im_target)

    tgt_ndarray = geoArr[:]       # numpy.ndarray with shape (10980, 10980)
    tgt_gt = geoArr.geotransform  # GDAL geotransform
    tgt_prj = geoArr.projection   # projection as WKT string

    # pass an instance of GeoArray to COREG and calculate spatial shifts
    geoArr_reference = GeoArray(ref_ndarray, ref_gt, ref_prj)
    geoArr_target = GeoArray(tgt_ndarray, tgt_gt, tgt_prj)

    CR = COREG(geoArr_reference, geoArr_target,
               path_out='D:/T49QEE_20181005_correct.tif',
               wp=(595951, 2480095), ws=(256, 256), v=True,
               max_iter=20, max_shift=100)
    CR.calculate_spatial_shifts()
    CR.correct_shifts()

    # Local coregistration test
    # from arosics import COREG_LOCAL

    # kwargs = {
    #     'grid_res'     : 200,
    #     'window_size'  : (64,64),
    #     'path_out'     : 'auto',
    #     'projectDir'   : 'D:/temp',
    #     'q'            : False,
    #     'CPUs'         : 1,
    #     'max_points'   : 500,
    #     'max_iter'     : 10,
    #     'max_shift'    : 100
    # }

    # CRL = COREG_LOCAL(im_reference,im_target,**kwargs)
    # CRL.correct_shifts()
