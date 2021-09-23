# -*- coding: utf-8 -*-
import os
import math

from skimage.morphology import square
from skimage.filters import median

import numpy as np
import rasterio
from rasterio.windows import Window
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
    if image.dtype != 'uint8':
        raise ValueError("data type must be uint8")

    norm = image/256.
    norm **= 1.0 / gamma
    return (norm * 255).astype('uint8')


def auto_gamma(image, mean_v=0.45, nodata=None):
    """自动获取gamma值,"""
    dims = image.shape
    if len(dims) > 2 or image.dtype != 'uint8':
        raise ValueError()
    img = image[::2, ::2].astype('float32')
    if nodata is not None:
        img[img == nodata] = np.nan
    gammav = np.log10(mean_v)/np.log10(np.nanmean(img)/256)
    return 1/gammav


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


def bytscl_std(input_arr, n=3, nodata=None):
    """标准差拉伸"""
    if nodata is None:
        mean, std = input_arr.mean(), input_arr.std()
        
    else:
        masked = np.ma.masked_where(input_arr == nodata, input_arr)
        mean, std = np.ma.mean(masked), np.ma.std(masked)

    return bytscl(input_arr, mean+n*std, mean-n*std, nodata)


def cal_keypoint_and_descriptor(gray):
    """SIFTkeypoint及descriptor"""
    # gray = bytscl_std(input_arr, n=3)
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


def window_list(size_x, size_y, width, height):
    """Split data into windows (square).

    Args:
        size_x, size_y (int): Window size.
        width, height (int): Data width and height.

    Returns:
        list: Rasterio Window object.

    """
    import math
    n_windows_x = math.ceil(width*1./size_x)
    n_windows_y = math.ceil(height*1./size_y)

    x_last_width = width % size_x
    y_last_width = height % size_y

    x_last_width = size_x if x_last_width == 0 else x_last_width
    y_last_width = size_y if y_last_width == 0 else y_last_width

    window_list = []
    for iy in range(n_windows_y):
        for ix in range(n_windows_x):
            # complete windows
            if ix != n_windows_x - 1 and iy != n_windows_y - 1:
                window_list.append(Window(
                        size_x * ix, size_y * iy,
                        size_x, size_y))
            # windows at the last column
            elif ix == n_windows_x - 1 and iy != n_windows_y - 1:
                window_list.append(Window(
                        size_x * ix, size_y * iy,
                        x_last_width, size_y))
            # window at the last row
            elif ix != n_windows_x - 1 and iy == n_windows_y - 1:
                window_list.append(Window(
                        size_x * ix, size_y * iy,
                        size_x, y_last_width))
            # window both in the last row and column
            else:
                window_list.append(Window(
                        size_x * ix, size_y * iy,
                        x_last_width, y_last_width))

    return window_list


def rgb2gray(rgb, r_weight=0.2125, g_weight=0.7154, b_weight=0.0721):
    """Y = 0.2125 R + 0.7154 G + 0.0721 B
    """
    assert(rgb.shape[0] == 3)
    return (rgb[0] * r_weight + rgb[1] * g_weight + rgb[2] * b_weight).astype('uint8')



if __name__ == '__main__':
    pan = r'D:\work\data\影像样例\GF2\GF2_PMS1_E108.9_N34.2_20181026_L1A0003549596\GF2_PMS1_E108.9_N34.2_20181026_L1A0003549596-PAN1.tiff'
    ref = r'E:\google\xian_18_google84.img'

    # ref_img = cv.imread(ref, cv.IMREAD_GRAYSCALE)
    ds = rasterio.open(ref)
    ref_trans = ds.transform
    # arr_ref, meta = open_img(ref, band_idx=1)
    
    # 预处理
    rpc_trans_out = r"D:\work\算法\registration\data\rpc_trans_out.tif"
    # RPC变换
    rpc_trans = f"gdalwarp -rpc {pan} {rpc_trans_out}"
    # os.system(rpc_trans)
    
    import glob
    gcp_files = glob.glob("d:/gcp_ori_*.txt")
    contents = ""
    for f in gcp_files:
        with open(f) as in_f:
            contents += in_f.read()
    gcps = np.fromstring(contents, sep='\t')
    gcps = gcps.reshape((int(gcps.size/4), 4))
    
    src_pts1 = gcps[:, :2].reshape(-1,1,2)
    dst_pts1 = gcps[:, 2:].reshape(-1,1,2)
    
    M, mask = cv.findHomography(src_pts1, dst_pts1, cv.RANSAC, 1e-5)
    n = mask.sum()
    mask = np.repeat(mask, 4, 1)
    gcps = gcps[mask == 1].reshape(n, 4)
    
    np.savetxt(f'd:/gcps.txt', gcps,
                        fmt='%10.6f', delimiter='\t')
    
    
    
    RC = Residual_Cal(gcps[:, 0], gcps[:, 1], gcps[:, 2], 2)
    res_x = RC.get_result()
    RC = Residual_Cal(gcps[:, 0], gcps[:, 1], gcps[:, 3], 2)
    res_y = RC.get_result()
    res = np.sqrt(res_x ** 2 + res_y ** 2)
    print(f"Mean residual of x: {np.abs(res_x).mean()}")
    print(f"Mean residual of y: {np.abs(res_y).mean()}")
    print(f"Mean residual: {res.mean()}")
    
    
    
    import sys
    sys.exit(0)
    
    # 分块处理
    blocks = 10
    buffer = 150  # 像素
    # arr, meta = open_img(rpc_trans_out, band_idx=1)
    with rasterio.open(rpc_trans_out) as src:
        width, height = src.width, src.height
        block_width = math.ceil(width/blocks)
        block_height = math.ceil(height/blocks)
        # 分块
        windows = window_list(block_width, block_height, width, height)
        # 分块处理
        for idx, window in enumerate(windows):
            print("current number:{}".format(idx))

            kwargs = src.meta.copy()
            
            # 预处理 拉伸变换
            arr_ori = src.read(window=window)[0]
            arr_ori = bytscl_std(arr_ori, n=3, nodata=0)

            if arr_ori.sum() == 0:
                print("Full of zero, pass!")
                continue

            # 对应参考数据的窗口,设置一定的缓冲距离
            window_trans = rasterio.windows.transform(window, src.transform)
            res = window_trans.a
            buffer_dis = buffer * res
            # 左上角在参考数据位置
            ul_col, ul_row = ~ref_trans * (window_trans.c - buffer_dis, window_trans.f + buffer_dis)
            # 右下角在参考数据位置
            lr_col, lr_row = ~ref_trans * (window_trans.c + window.width * res + buffer_dis, window_trans.f - window.height * res - buffer_dis)
            
            window_ref = Window.from_slices((int(ul_row), int(lr_row)), (int(ul_col), int(lr_col)))
            
            # 参考数据数组
            trans_ref = rasterio.windows.transform(window_ref, ref_trans)
            arr_ref = ds.read(window = window_ref)
            arr_ref = rgb2gray(arr_ref)
            arr_ref = bytscl_std(arr_ref, n=2.5, nodata=0)
            arr_ref = gamma(arr_ref, gamma=auto_gamma(arr_ref, mean_v=0.5, nodata=0))
            
            try:
                kp_ori, des_ori = cal_keypoint_and_descriptor(arr_ori)
                kp_ref, des_ref = cal_keypoint_and_descriptor(arr_ref)
            except Exception as e:
                print(e)
                continue
        
            # img=cv.drawKeypoints(img1, kp_ori, img1, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            # cv.imwrite('sift_keypoints_pan.jpg', img)
        
            # img=cv.drawKeypoints(img2, kp_ref, img2, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            # cv.imwrite('sift_keypoints_ref.jpg', img)
        
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
            print(f"{len(src_pts_list)} good matched!")
        
        
            # 排除多对一匹配点
            # 根据距离阈值排除较远的点
            # affine_ori = ori_meta['transform']
            # affine_ref = ref_meta['transform']
        
            coor_ori = np.float32([window_trans * p for p in src_pts_list])
            coor_ref = np.float32([trans_ref * p for p in dst_pts_list])
        
            diff = coor_ori - coor_ref
            dis = np.sqrt(np.square(diff[:, 0]) + np.square(diff[:, 1]))
            mask = dis < 0.01
            matchesMask1 = mask*1
            # print(matchesMask1.sum())
            n_matched = matchesMask1.sum()
            print(f"{n_matched} matched after distance filter!")
            
        
            MIN_MATCH_COUNT = 50
            if n_matched > MIN_MATCH_COUNT:
                # RANSAC OPENCV官方
                src_pts = np.float32(src_pts_list).reshape(-1,1,2)
                dst_pts = np.float32(dst_pts_list).reshape(-1,1,2)
                M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 2.0)
                # M, mask = cv.findHomography(src_pts, dst_pts, cv.LMEDS)
        
                mask = (mask[:, 0] * matchesMask1).astype('uint8')
                # mask = matchesMask1 * mask
            matchesMask = (mask*1).ravel().tolist()
            # matchesMask = list(mask*1)
        
            mask = np.bool8(mask)
            print("{} matched left after ransac.".format(mask.sum()))
        
            gcps = np.hstack((coor_ori[mask], coor_ref[mask]))
            
            # RC = Residual_Cal(gcps[:, 0], gcps[:, 1], gcps[:, 2], 2)
            # res_x = RC.get_result()
            # RC = Residual_Cal(gcps[:, 0], gcps[:, 1], gcps[:, 3], 2)
            # res_y = RC.get_result()
            # res = np.sqrt(res_x ** 2 + res_y ** 2)
            # print(f"Mean residual of x: {np.abs(res_x).mean()}")
            # print(f"Mean residual of y: {np.abs(res_y).mean()}")
            # print(f"Mean residual: {res.mean()}")
        
            # print("Writing gcp files of ArcMap format.")
            np.savetxt(f'd:/gcp_ori_{idx}.txt', gcps,
                        fmt='%10.6f', delimiter='\t')
        
            # draw matches
            med = cv.drawMatches(arr_ori,kp_ori,arr_ref,kp_ref, good_matches, None,
                                  matchesThickness=2,
                                  matchColor = (0,255,0),
                                  singlePointColor = None,
                                  matchesMask = matchesMask,
                                  flags = 2)
        
            cv.imwrite(f"match_block_{idx}.jpg", med)


            # 写出对应目前窗口的参考窗口数据, 如果需要验证
            kwargs.update({
                'height': window_ref.height,
                'width': window_ref.width,
                'transform': rasterio.windows.transform(window_ref, ref_trans),
                'dtype': 'uint8',
                'count': 1,
                })
            # with rasterio.open('d:/temp11/ref_window_gray_stretch_gamma1.tif', 'w', **kwargs) as dst:
            #     dst.write(arr_ref, 1)
            # break
            
            
            
            # 写出blocks 如果需要验证
            kwargs.update({
                'height': window.height,
                'width': window.width,
                'transform': window_trans,
                'dtype': 'uint8',
                })
            # TODO using memery or tempfile to store temp data
            out = r'D:\work\算法\registration\data\blocks\win_{}.tif'.format(str(idx + 1))
            # with rasterio.open(out, 'w', **kwargs) as dst:
            #     dst.write(arr_ori)
            # break



    # import os
    # # warp
    # gcps_gdal = np.hstack((np.float32(src_pts_list)[mask], coor_ref[mask]))
    # arg_gcp = ''
    # for gcp in gcps_gdal:
    #     arg_gcp += ' '.join(['-gcp',
    #                          '{:.2f}'.format(gcp[0]),
    #                          '{:.2f}'.format(gcp[1]),
    #                          '{:.6f}'.format(gcp[2]),
    #                          '{:.6f}'.format(gcp[3]),
    #                         ]) + ' '
    # trans_file = '../temp/trans.vrt'
    # warped_file = '../temp/corrected.tif'
    # command_translate = f'gdal_translate -q -r bilinear {arg_gcp}{img_ori} {trans_file}'
    # command_warp = f'gdalwarp -order 2 -refine_gcps 3 50 -r bilinear -t_srs "EPSG:4326" -overwrite {trans_file} {warped_file}'
    # os.system(command_translate)
    # os.system(command_warp)
    # gdalwarp -order 2 -refine_gcps 3 50 -r bilinear -t_srs "EPSG:4326" -overwrite ../temp/trans.vrt ../temp/corrected.tif
        