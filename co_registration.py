# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 10:39:51 2019

未测试
https://github.com/andrestumpf/coregis_public

"""
import numpy as np

def arosics_test():
    """
    测试几何校正工具
    See https://gitext.gfz-potsdam.de/danschef/arosics
    https://www.mdpi.com/2072-4292/9/7/676

    """
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


def hl_matrix_generate(input_x, input_y, n, dims):
    """生成最小二乘系数矩阵."""
    if n == 0:
        Hl_array = np.ones([dims, 1])
        return Hl_array
    elif n > 0:
        Hl_array = np.zeros([dims, 1])
        for i in range(n+1):
            Hl_array = np.c_[Hl_array, (input_x**(n-i))*(input_y**i)]
        Hl_array = np.delete(Hl_array, 0, axis=1)
        Hl_array = np.c_[Hl_array,
                         hl_matrix_generate(input_x, input_y, n-1, dims)]
    return Hl_array


class Residual_Cal(object):
    """仿射变换残差计算方法。
    """
    def __init__(self, input_x, input_y, output, n):
        if input_x.size != input_y.size:
            raise AttributeError("Parameter input_x and input_y must have "
                                 "the same size.")
        self.dims = input_x.size
        self.input_x = input_x
        self.input_y = input_y
        self.output = output
        self.n = n
        # 系数阵
        self.hl_array = hl_matrix_generate(input_x, input_y, n, self.dims)

    def cal_coef(self):
        """最小二乘法计算系数."""
        from scipy.linalg import lstsq
        p, res, rnk, s = lstsq(self.hl_array, self.output)
        return p
    
    def get_result(self):
        """计算结果."""
        self.result = self.hl_array.dot(self.cal_coef())
        self.res_array = self.output - self.result


if __name__ == '__main__':
    input_x = [108.769928,108.697541,109.004523,108.852212,108.822135,
               108.733109,108.86987,108.783179,108.986501,108.982744,
               108.947673,108.737097,108.681963,108.891484,108.969123,
               108.929151,108.823685,108.8027,108.924168,108.960853,
               108.917661,108.821568,108.907735,108.931008,108.848093,
               108.84511,108.77136]
    input_y = [34.331761,34.161562,34.30035,34.242175,34.119152,34.24899,
               34.308182,34.213829,34.257844,34.20163,34.095594,34.352723,
               34.156097,34.105698,34.155842,34.309908,34.331847,34.198671,
               34.133773,34.292245,34.272465,34.284469,34.165846,34.240433,
               34.187739,34.160596,34.287835]
    output_data_x = [108.770105,108.697708,109.004658,108.852356,108.822225,
                     108.733306,108.870051,108.783337,108.986602,108.98266,
                     108.947641,108.737291,108.682097,108.891509,108.968881,
                     108.929342,108.82395,108.802887,108.92401,108.961049,
                     108.91788,108.821826,108.907731,108.93113,108.848099,
                     108.84519,108.771583]

    output_data_y = [34.330539,34.160329,34.299143,34.240982,34.11793,34.247742,
                     34.306968,34.2126,34.256617,34.200443,34.09439,34.351514,
                     34.154873,34.104497,34.154684,34.308762,34.330631,34.197451,
                     34.132611,34.291022,34.271174,34.283254,34.164629,34.239213,
                     34.186531,34.159415,34.286617]
    input_x = np.array(input_x)
    input_y = np.array(input_y)
    output = np.array(output_data_y)
    rc = Residual_Cal(input_x, input_y, output, 1)
    rc.get_result()
    print(rc.res_array)