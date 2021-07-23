# -*- coding: utf-8 -*-

from skimage.morphology import square
from skimage.filters import median

import numpy as np
import rasterio
# import pyrtools as pt
import cv2 as cv


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


def open_img(img, band_idx=1):
    with rasterio.open(img) as src:
        return src.read(band_idx), src.meta.copy()


def write_raster(arr, out, kargs):

    shape = arr.shape
    if len(shape) != 2:
        raise ValueError("input array must be a 2-d array.")
    kargs.update({'width': shape[1],
                  'height': shape[0],
                  'dtype': arr.dtype,
                  'count': 1,
                  })
    with rasterio.open(out, 'w', **kargs) as dst:
        dst.write(arr, 1)


def generate_steerable_pyramid(input_arr, levels=3, order=0):
    """
    Args:
        levels: pyramid levels.
        order: filters {0,1,3,5}(sp0_filter,sp1_filter,etc..)

    Returns:
        generator: pyramids and (lowpass highpass) band.

    """
    import pyrtools as pt
    pyr = pt.pyramids.SteerablePyramidSpace(input_arr, height=levels, order=order)
    print(f"num_steerable_pyramids: {levels}")
    print(f"num_orientations: {pyr.num_orientations}")

    for s in range(pyr.num_scales):
        yield pyr.pyr_coeffs[(s,0)].astype('float32')

    yield pyr.pyr_coeffs['residual_lowpass'].astype('float32')
    yield pyr.pyr_coeffs['residual_highpass'].astype('float32')


def bytscl_std(input_arr, n=3):
    """标准差拉伸"""
    mean, std = input_arr.mean(), input_arr.std()
    return bytscl(input_arr, mean+n*std, mean-n*std)

def cal_keypoint_and_descriptor(input_arr):
    """SIFTkeypoint及descriptor"""
    gray = bytscl_std(input_arr, n=3)
    # gray = bytscl(input_arr)
    sift = cv.SIFT_create()
    # sift = cv.ORB_create()
    return sift.detectAndCompute(gray, None)

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
        return self.output - self.result


if __name__ == '__main__':
    pan = r'D:\work\算法\registration\data\pan.tif'
    ref = r'D:\work\算法\registration\data\ref.tif'
    pan = '/mnt/data/pan.tif'
    ref = '/mnt/data/ref.tif'

    # ori_arr, ori_meta = open_img(pan, band_idx=1)
    # ref_arr, ref_meta = open_img(ref, band_idx=1)
    # print(f"origin image shape: {ori_arr.shape}")
    # print(f"reference image shape: {ref_arr.shape}")

    # med_ori = median(ori_arr, square(3))
    # med_ref = median(ref_arr, square(3))

    # # write_raster(med_ori, r'D:\temp11\test_regis\med_ori3.tif', ori_meta)
    # # write_raster(med_ref, r'D:\temp11\test_regis\med_ref3.tif', ref_meta)

    # # Steerable Pyramid Transform(SPT)
    # # 3层金字塔, 单向滤波算子
    # print("Steerable Pyramid Transform")
    # spts_ori = generate_steerable_pyramid(med_ori)
    # spts_ref = generate_steerable_pyramid(med_ref)

    # for i, p in enumerate(spts_ori):
    #     write_raster(p, f'/mnt/temp/spt/spt_ori_{i}.tif', ori_meta)
    # for i, p in enumerate(spts_ref):
    #     write_raster(p, f'/mnt/temp/spt/spt_ref_{i}.tif', ref_meta)

    ori = 'D:/work/算法/registration/temp/spt/spt_ori_2.tif'
    ref = 'D:/work/算法/registration/temp/spt/spt_ref_2.tif'

    img_ori = '../data/pan.png'
    img_ref = '../data/ref.tif'
    img1 = cv.imread(img_ori, cv.IMREAD_GRAYSCALE)
    img2 = cv.imread(img_ref, cv.IMREAD_GRAYSCALE)

    # img1 = median(img1, square(3))
    # img2 = median(img2, square(3))
    # img = cv.imread(ref0)

    ori_arr, ori_meta = open_img(ori, band_idx=1)
    ref_arr, ref_meta = open_img(ref, band_idx=1)

    gray_ori = bytscl_std(ori_arr, n=3)
    gray_ref = bytscl_std(ref_arr, n=3)

    kp_ori, des_ori = cal_keypoint_and_descriptor(img1)
    kp_ref, des_ref = cal_keypoint_and_descriptor(img2)

    img=cv.drawKeypoints(img1, kp_ori, img1, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv.imwrite('sift_keypoints_pan.jpg', img)

    img=cv.drawKeypoints(img2, kp_ref, img2, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv.imwrite('sift_keypoints_ref.jpg', img)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary

    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des_ori,des_ref, k=2)

    # matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)
    # matches = matcher.knnMatch(des_ori, des_ref, 2)

    # BFMatcher with default params
    # bf = cv.BFMatcher()
    # matches = bf.knnMatch(des_ori, des_ref, k=2)

    # good = [m1 for (m1, m2) in matches if m1.distance * 1.5 < m2.distance]
    good = [m1 for (m1, m2) in matches if m1.distance < 0.7*m2.distance]

    good_matches = sorted(good, key = lambda x:x.distance)

    src_pts_list = [ kp_ori[m.queryIdx].pt for m in good_matches ]
    dst_pts_list = [ kp_ref[m.trainIdx].pt for m in good_matches ]


    # 排除多对一匹配点
    # 根据距离阈值排除较远的点
    affine_ori = ori_meta['transform']
    affine_ref = ref_meta['transform']

    coor_ori = np.float32([affine_ori * p for p in src_pts_list])
    coor_ref = np.float32([affine_ref * p for p in dst_pts_list])

    diff = coor_ori - coor_ref
    dis = np.sqrt(np.square(diff[:, 0]) + np.square(diff[:, 1]))
    mask = dis < 0.01
    matchesMask1 = mask*1
    print(matchesMask1.sum())

    MIN_MATCH_COUNT = 50
    if matchesMask1.sum() > MIN_MATCH_COUNT:
        # RANSAC OPENCV官方
        src_pts = np.float32(src_pts_list).reshape(-1,1,2)
        dst_pts = np.float32(dst_pts_list).reshape(-1,1,2)
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,1.0)
        # M, mask = cv.findHomography(src_pts, dst_pts, cv.LMEDS)

        mask = mask[:, 0] * matchesMask1.astype('uint8')
        print(mask.sum())
        # mask = matchesMask1 * mask
    matchesMask = (mask*1).ravel().tolist()
    # matchesMask = list(mask*1)

    mask = np.bool8(mask)
    print(mask.sum())

    gcps = np.hstack((coor_ori[mask], coor_ref[mask]))

    RC = Residual_Cal(gcps[:, 0], gcps[:, 1], gcps[:, 2], 2)
    res_x = RC.get_result()
    RC = Residual_Cal(gcps[:, 0], gcps[:, 1], gcps[:, 3], 2)
    res_y = RC.get_result()
    res = np.sqrt(res_x ** 2 + res_y ** 2)
    print(f"Mean residual of x: {np.abs(res_x).mean()}")
    print(f"Mean residual of y: {np.abs(res_y).mean()}")
    print(f"Mean residual: {res.mean()}")

    print("Writing gcp files of ArcMap format.")
    np.savetxt('d:/gcp_ori.txt', gcps,
               fmt='%10.6f', delimiter='\t')

    # draw matches
    med = cv.drawMatches(img1,kp_ori,img2,kp_ref, good_matches, None,
                         matchesThickness=2,
                         matchColor = (0,255,0),
                         singlePointColor = None,
                         matchesMask = matchesMask,
                         flags = 2)

    cv.imwrite("match.jpg", med)
    
    import os
    # warp
    gcps_gdal = np.hstack((np.float32(src_pts_list)[mask], coor_ref[mask]))
    arg_gcp = ''
    for gcp in gcps_gdal:
        arg_gcp += ' '.join(['-gcp',
                             '{:.2f}'.format(gcp[0]),
                             '{:.2f}'.format(gcp[1]),
                             '{:.6f}'.format(gcp[2]),
                             '{:.6f}'.format(gcp[3]),
                            ]) + ' '
    command_translate = f'gdal_translate -ot UInt16 -q -r bilinear {arg_gcp}{img_ori} ../temp/trans.vrt'
    os.system(command_translate)
    # gdalwarp -order 2 -refine_gcps 3 50 -r bilinear -t_srs "EPSG:4326" -overwrite ../temp/trans.vrt ../temp/corrected.tif
        