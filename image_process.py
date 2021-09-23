# -*- coding: utf-8 -*-
"""一些栅格数据常用处理工具

"""
import math
import numpy as np
from io_utils import read_raster_gdal, read_text

try:
    import gdal
except ImportError:
    from osgeo import gdal
except ImportError:
    print('ignore if not using gdal module')
    pass
try:
    from osgeo import ogr
except ImportError:
    print('ignore if not using ogr module')
    pass

try:
    from PIL import Image as Pil
except ImportError:
    print('ignore if not using PIL module')
    pass

try:
    from affine import Affine
except ImportError:
    print('ignore if not using affine module')
    pass

try:
    import rasterio
    from rasterio.warp import reproject
    from rasterio.enums import Resampling
except ImportError:
    print('ignore if not using rasterio module')
    pass


class pilImage(object):
    """利用PIL显示、保存图像或缩略图.

    Attributes:
        data (numpy ndarray or list): 单通道二维数组或列表形式的RGB数组[R,G,B].
        img (pil image object): PIL对象

    """
    def __init__(self, data):
        """利用数组构建PIL Image对象.

        Args:
            data (numpy ndarray or list): 单通道二维数组或列表形式的RGB数组[R,G,B].

        Raises:
            ValueError: If input format wrong.
        """

        self.data = data
        if isinstance(data, list):
            if len(data) == 3:
                R = Pil.fromarray(np.asanyarray(data[0]), 'L')
                G = Pil.fromarray(np.asanyarray(data[1]), 'L')
                B = Pil.fromarray(np.asanyarray(data[2]), 'L')
                self.img = Pil.merge("RGB", (R, G, B))
        elif isinstance(data, np.ndarray):
            if data.ndim == 2:
                self.img = Pil.fromarray(np.asanyarray(data), 'L')
        else:
            raise ValueError('Wrong input format,please input either numpy \
                              ndarray or RGB list of ndarray.')

    def show(self):
        """Show image."""
        self.img.show()

    def save(self, location):
        """Save image.

        Args:
            location (str): The target image file.
        """
        self.img.save(location, "JPEG")

    def thumbnail(self, size=(1024, 1024)):
        """Construct a thumbnail object.

        Args:
            size (tuple): Size of the thumbnail, default (1024, 1024).
        """
        self.img.thumbnail(size)

    def showThumb(self, size=(1024, 1024)):
        """Show thumbnail."""
        self.thumbnail(size)
        self.show()

    def saveThumb(self, location, size=(1024, 1024)):
        """Save thumbnail."""
        self.thumbnail(size)
        self.save(location)


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


def percentile_v(argArry, percent=2, leftPercent=None,
                 rightPercent=None, nodata=None):
    if percent is not None:
        leftPercent = percent
        rightPercent = percent
    elif (leftPercent is None or rightPercent is None):
        raise ValueError('Wrong parameter! Both left and right percent '
                         'should be set.')

    if len(argArry.shape) == 2:
        _arr = argArry[::2, ::2]
        retArry = _arr[_arr != nodata]
    else:
        retArry = argArry[argArry != nodata]

    minValue = np.percentile(retArry, leftPercent, interpolation="nearest")
    maxValue = np.percentile(retArry, 100 - rightPercent,
                             interpolation="nearest")
    
    return maxValue, minValue


def linear_stretch(argArry, percent=2, leftPercent=None,
                   rightPercent=None, nodata=None):
    """指定百分比对数据进行线性拉伸处理.

    Args:
        argArry (numpy ndarray): 输入图像数组.
        percent (float): 最大最小部分不参与拉伸的百分比.
        leftPercent (float):  左侧（小）不参与拉伸的百分比.
        rightPercent (float):  右侧（大）不参与拉伸的百分比.
        nodata (same as input array): 空值，默认None，计算时排除.

    Returns:
        numpy ndarray: 拉伸后八位无符号整型数组(0-255).

    Raises:
        ValueError: If only one of the leftPercent or the rightPercent is set.

    """
    maxValue, minValue = percentile_v(argArry, percent, leftPercent,
                                      rightPercent, nodata)
    return bytscl(argArry, maxValue=maxValue, minValue=minValue, nodata=nodata)


def bytscl_std(input_arr, n=3):
    """标准差拉伸"""
    mean, std = input_arr.mean(), input_arr.std()
    return bytscl(input_arr, mean+n*std, mean-n*std)


def cira_strech(band_data):
    """ Logarithmic stretch adapted to human vision.

    Note:
        Applicable only for visible channels.

    Args:
        band_data (numpy ndarray): 输入数组,可见光波段反射率数据(一般0-1).

    Returns:
        numpy ndarray (float32): 拉伸后数组.

    """
    with np.errstate(invalid='ignore', divide='ignore'):
        band_data = np.ma.masked_equal(band_data, 0)

        log_root = np.log10(0.0223)
        denom = (1.0 - log_root) * 0.75
        band_data = np.log10(band_data)
        band_data -= log_root
        band_data /= denom

        return np.ma.filled(band_data, 0).astype('float32').clip(0)


def composite_band(bandList, outTiff, trans, proj, dataType=gdal.GDT_Byte):
    """ 利用gdal进行波段组合操作，并将结果保存为GTiff格式.

    Note:
        很少再用,可以直接用rasterio包读写.

    Args:
        bandList (list): List of band arrays.
        outTiff (str): Filename of output tifffile.
        trans (tuple): GDAL geotransform infomation.
        proj (object): Projection used by gdal.
        dataType: Datatype supported by gdal, default gdal.GDT_Byte(default).
            GDT_Unknown 	Unknown or unspecified type
            GDT_Byte 	   Eight bit unsigned integer
            GDT_UInt16 	   Sixteen bit unsigned integer
            GDT_Int16 	   Sixteen bit signed integer
            GDT_UInt32 	   Thirty two bit unsigned integer
            GDT_Int32 	   Thirty two bit signed integer
            GDT_Float32 	Thirty two bit floating point
            GDT_Float64 	Sixty four bit floating point
            GDT_CInt16 	   Complex Int16
            GDT_CInt32 	   Complex Int32
            GDT_CFloat32 	Complex Float32
            GDT_CFloat64 	Complex Float64

    """
    rows, cols = bandList[0].shape
    # 创建输出
    outdriver = gdal.GetDriverByName("GTiff")
    outdata = outdriver.Create(outTiff, cols, rows, len(bandList), dataType)
    # 设置数据基本信息
    outdata.SetGeoTransform(trans)
    outdata.SetProjection(proj)

    # 写入波段信息
    for i in range(len(bandList)):
        outdata.GetRasterBand(i+1).WriteArray(bandList[i])
    # 刷到硬盘
    outdata = None


def new_green_band(g, ir, ratio=0.1):
    """采用绿通道和近红外通道像元重新组合生成新的绿通道.

    Args:
        g (numpy ndarray): 绿色通道值.
        ir (numpy ndarray): 近红外通道值.
        ratio (float): number of range 0-1,近红外通道所采取的比例.

    Returns:
        numpy ndarray: 合成后的绿通道数组.

    """
    newBand = g * (1 - ratio) + ir * ratio
    return newBand.astype(g.dtype)


def rasterize(ogrData, raster_fn, ref_image=None, pixel_size=None,
              outputSRS=None):
    """矢量数据栅格化.

    Args:
        ogrData (ogr支持矢量格式): 输入数据.
        raster_fn (str):  The target raster file.
        ref_image (str):  参考数据，如果存在，从中读取相关参数.
        pixel_size (float): 分辨率，如果有参考影像,与参考数据一致，如果没有参考影像
                    并且没有指定,则默认为x或y方向范围较小的值除以250.
        outputSRS (SpatialReference object): 输出栅格投影，
                    默认与参考影像或输入矢量数据一致.

    Returns:
        str: Output raster file.Mask value is 0,other value is 1.

    """
    outputBounds = []

    if ref_image is not None:
        inDS = read_raster_gdal(ref_image)
        geoMatrix = inDS[1]
        rows, cols = inDS[0].shape
        extent = get_extent(geoMatrix, cols, rows)
        pixel_size = min(geoMatrix[1], abs(geoMatrix[5]))
        outputBounds = [extent[0], extent[1], extent[2], extent[3]]
        if outputSRS is None:
            outputSRS = inDS[2]

    else:
        # Open the data source and read in the extent
        source_ds = ogr.Open(ogrData)
        source_layer = source_ds.GetLayer()
        x_min, x_max, y_min, y_max = source_layer.GetExtent()
        outputBounds = [x_min, y_min, x_max, y_max]
        if outputSRS is None:
            outputSRS = source_layer.GetSpatialRef()

        if pixel_size is None:
            pixel_size = min(y_max - y_min, x_max - x_min) / 250.

    gdal.Rasterize(raster_fn, ogrData, format="GTiff", initValues=1,
                   burnValues=0, xRes=pixel_size, yRes=pixel_size,
                   outputBounds=outputBounds, outputType=gdal.GDT_Byte,
                   outputSRS=outputSRS)

    return raster_fn


def apply_mask(imageArr, maskArr, maskValue=0, fillvalue=0):
    """应用掩膜至指定数组.

    Args:
        imageArr (numpy ndarray): 需要掩膜的数组.
        maskArr (numpy ndarray):  掩膜数组.
        maskValue (float): 掩膜数组中掩膜对应的值.
        fillvalue (float): 掩膜后需填充的值.

    Returns:
        numpy ndarray: 应用掩膜后的数组.

    Raises:
        ValueError: If imageArr and maskArr have different size.

    """
    if imageArr.shape != maskArr.shape:
        raise ValueError('Image and Mask data must have the same dimentions.')

    masked = np.ma.masked_where(maskArr == maskValue, imageArr)
    return np.ma.filled(masked, fillvalue)


def world2Pixel(geoMatrix, x, y):
    """ 应用gdal几何变换信息,计算x,y坐标所在行列号.

    Args:
        geoMatrix (tuple):  gdal geomatrix (gdal.GetGeoTransform()).
        x, y (float): 地理坐标.

    Returns:
        tuple: (列号,行号).

    """
    ulX = geoMatrix[0]
    ulY = geoMatrix[3]
    xDist = geoMatrix[1]
    yDist = geoMatrix[5]
    pixel = int((x - ulX) / xDist)
    line = int((ulY - y) / yDist)
    return (pixel, line)


def get_extent(geoMatrix, cols, rows):
    """ 应用gdal几何变换信息及数据维度，计算数据范围.

    Args:
        geoMatrix (tuple): gdal geomatrix (gdal.GetGeoTransform()).
        cols (int): 列数.
        rows (int): 行数.

    Returns:
        tuple: 数据范围(x_min,y_min,x_max,y_max).

    """
    upx, xres, xskew, upy, yskew, yres = geoMatrix

    llx = upx + 0*xres + rows*xskew
    lly = upy + 0*yskew + rows*yres

    urx = upx + cols*xres + 0*xskew
    ury = upy + cols*yskew + 0*yres

    return (llx, lly, urx, ury)


def get_gml_src_proj(path_gml):
    """ 从哨兵2 L1C数据自带gml数据中读取投影信息.

    Args:
        path_gml (str): gml文件名称.

    Returns:
        str: 数据投影（EPSG编号）.

    """
    import xml.etree.ElementTree as ET
    text = read_text(path_gml)
    text = text.split('xmlns:gml=')[-1]
    gml = text.split('"')[1]

    tree = ET.parse(path_gml)

    node = tree.find('.//{'+gml+'}Envelope')
    try:
        srs_string = node.get('srsName')
        epsg_no = srs_string.split(':')[-1]
        epsg_code = 'EPSG:{}'.format(epsg_no)
    except Exception:
        print('Warning, gml has no SRS information.')
        epsg_code = None

    return epsg_code


def lonlat2Raster(arr, outTiff, res, dtype='float32', fillvalue=-1.0):
    """指定经纬度范围生成栅格.

    Args:
        arr (numpy ndarray): n x 3数组 第一列经度 第二列 纬度 第三列 为对应值.
        outTiff (str): 输出tiff格式数据名称.
        res (float): 分辨率.
        dtype (numpy dtype): 数据类型, default float32.
        fillvalue (float): 填充值,default -1 .

    """
    lon_max, lat_max = np.amax(arr[:, :2], axis=0)
    lon_min, lat_min = np.amin(arr[:, :2], axis=0)
    # 处理跨越180度经线的问题
    if lon_max - lon_min > 180:

        arr1 = arr[np.where(arr[:, 0] > 0)[0], :]
        arr2 = arr[np.where(arr[:, 0] < 0)[0], :]

        # 分2块
        lon_max1 = np.amax(arr1[:, 0], axis=0)
        lon_min1 = np.amin(arr1[:, 0], axis=0)
        # print(lon_max1,lon_min1)
        outTiff1 = outTiff[:-4] + '_part1.tif'

        lonlatRange2Raster(arr1, lon_min1, lat_min, lon_max1, lat_max,
                           outTiff1, res, dtype=dtype, fillvalue=fillvalue)

        lon_max2 = np.amax(arr2[:, 0], axis=0)
        lon_min2 = np.amin(arr2[:, 0], axis=0)
        outTiff2 = outTiff[:-4] + '_part2.tif'
        lonlatRange2Raster(arr2, lon_min2, lat_min, lon_max2, lat_max,
                           outTiff2, res, dtype=dtype, fillvalue=fillvalue)

    else:
        lonlatRange2Raster(arr, lon_min, lat_min, lon_max, lat_max, outTiff,
                           res, dtype=dtype, fillvalue=fillvalue)


def lonlatRange2Raster(arr, lon_min, lat_min, lon_max, lat_max, outTiff,
                       res, dtype='float32', fillvalue=-1.):
    """根据指定经纬度范围生成栅格.

    TODO: 添加插值

    Args:
        arr (numpy ndarray): n x 3数组 第一列经度 第二列 纬度 第三列 为对应值.
        lon_min, lat_min, lon_max, lat_max (float): 给定经纬度范围.
        outTiff (str): 输出tiff格式数据名称.
        res (float): 分辨率.
        dtype (numpy dtype): 数据类型, default float32.
        fillvalue (float): 填充值,default -1 .

    """

    n_row = math.ceil((lat_max - lat_min)/res)
    n_col = math.ceil((lon_max - lon_min)/res)

    while lon_min + n_col*res > 180:
        n_col -= 1

    imgArr = np.full((n_row, n_col), fillvalue, dtype=dtype)

    fwd = Affine.from_gdal(lon_min, res, 0, lat_max, 0, -res)

    for r in range(arr.shape[0]):
        col, row = [math.floor(a) for a in (~fwd * arr[:, :2][r, :])]
        imgArr[row, col] = arr[r, 2]

    with rasterio.open(outTiff, 'w', driver='GTiff', height=n_row,
                       width=n_col, count=1, dtype=dtype,
                       crs='+proj=latlong', transform=fwd) as dst:
        dst.write(imgArr, 1)


def zoom_array(imgArr, nzoom=[1, 1], order=3, mode='constant', cval=0.0,
               prefilter=True):
    """利用scipy zoom 函数按比例缩放数组,相当于重采样。见：
        https://docs.scipy.org/doc/scipy-0.16.1/reference/generated/scipy.ndimage.interpolation.zoom.html

    Args:
        imgArr (numpy ndarray):  Input array.
        zoom (float or sequence): The zoom factor along the axes.
        order (int): The order of the spline interpolation, default is 3.
        mode (str): Points outside the boundaries of the input are filled
                according to the given mode (‘constant’, ‘nearest’,
                ‘reflect’ or ‘wrap’). Default is ‘constant’.
        cval (float) : Value used for points outside the boundaries of the
                input if mode='constant'.
        prefilter (bool): The parameter prefilter determines if the input is
                pre-filtered with spline_filter before interpolation

    Returns:
        numpy ndarray: The zoomed input.

    """
    from scipy.ndimage.interpolation import zoom
    return zoom(imgArr, nzoom, order=order, mode=mode, cval=cval,
                prefilter=prefilter)


def resample(srcfile, dstfile, new_res, width=0, height=0,
             method=Resampling.bilinear):
    """用rasterio reproject做栅格数据重采样.

    Args:
        srcfile (str): 输入栅格数据.
        dstfile (str): 重采样后栅格数据.
        new_res (float): 重采样分辨率.
        width, height (int): 数据列数及行数，默认自动计算.
        resampling method (rasterio Resampleing method): One of the following:
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

    """
    src = rasterio.open(srcfile)
    meta = src.meta.copy()
    res = src.res[0]

    ratio = res*1./new_res

    # adjust the new affine transform to the 150% smaller cell size
    aff = src.transform

    newaff = Affine(new_res, aff.b, aff.c,
                    aff.d, -new_res, aff.f)

    if width == 0 or height == 0:
        width = int(src.width * ratio)
        height = int(src.height * ratio)

    meta.update({
            'transform': newaff,
            'width': width,
            'height': height
            })

    with rasterio.open(dstfile, 'w', **meta) as dst:
        for i in range(1, src.count + 1):
            reproject(
                source=rasterio.band(src, i),
                destination=rasterio.band(dst, i),
                src_transform=aff,
                src_crs=src.crs,
                dst_transform=newaff,
                dst_crs=src.crs,
                resampling=method)


def change_dtype(in_raster, out_raster, dst_dtype='float32'):
    """不改变其它信息的情况下,改变数据类型."""
    with rasterio.open(in_raster) as src:
        kargs = src.meta.copy()
        kargs.update({'dtype': dst_dtype,})

        with rasterio.open(out_raster, 'w', **kargs) as dst:
            dst.write(src.read())


def pre_process_thumbs(r_arr, g_arr, b_arr, nodata=None):
    """输入原始RGB值,返回2%线性拉伸及自动gamma校正后的RGB值."""
    # 计算线性拉伸最大最小值,采用2%拉伸
    r_p_max, r_p_min = percentile_v(r_arr, nodata=nodata)
    g_p_max, g_p_min = percentile_v(g_arr, nodata=nodata)
    b_p_max, b_p_min = percentile_v(b_arr, nodata=nodata)
    # 如果2%处数值相等,极有可能是nodata
    if r_p_max == g_p_max == b_p_max and nodata is None:
        print("The max value may be the nodata value")
        nodata = r_p_max
        r_p_max, r_p_min = percentile_v(r_arr, nodata=nodata)
        g_p_max, g_p_min = percentile_v(g_arr, nodata=nodata)
        b_p_max, b_p_min = percentile_v(b_arr, nodata=nodata)
    elif r_p_min == g_p_min == b_p_min and nodata is None:
        print("The min value may be the nodata value")
        nodata = r_p_min
        r_p_max, r_p_min = percentile_v(r_arr, nodata=nodata)
        g_p_max, g_p_min = percentile_v(g_arr, nodata=nodata)
        b_p_max, b_p_min = percentile_v(b_arr, nodata=nodata)
    # 线性拉伸
    stretched_r = bytscl(r_arr, r_p_max, r_p_min, nodata)
    stretched_g = bytscl(g_arr, g_p_max, g_p_min, nodata)
    stretched_b = bytscl(b_arr, b_p_max, b_p_min, nodata)
    
    # 计算gamma校正用到的gamma值
    gamma_v_r = aoto_gamma(stretched_r, nodata=nodata)
    gamma_v_g = aoto_gamma(stretched_g, nodata=nodata)
    gamma_v_b = aoto_gamma(stretched_b, nodata=nodata)
    g = (gamma_v_r + gamma_v_g + gamma_v_b) / 3
    print(f"Use gamma value: {g:.3f}")
    return [gamma(stretched_r, g), gamma(stretched_g, g),
            gamma(stretched_b, g)]


def create_thumbnail_rs(image, out_thumbs='thumbs.png', size=500):
    """遥感影像缩略图生产, 预处理采用2%线性拉伸及自动gamma校正, 缩略图保持原来长宽比,
    短边为指定像素大小.
        * 单波段: 灰度图;
        * 3波段: 默认采用1、2、3波段作为RGB波段;
        * 4波段及以上: 默认采用3、2、1波段作为RGB波段;
    """
    with rasterio.open(image) as src:
        meta = src.meta.copy()
        width, height = meta['width'], meta['height']
        nodata = meta['nodata']
        bands = meta['count']
        # 短边等于size
        ratio = height / width
        if width > height:
            size1 = size
            size0 = int(size / ratio)
        else:
            size0 = size
            size1 = int(size * ratio)
        # 如果数据过大时,比如大于10倍size, 读取view的最大尺寸
        x_view_size, y_view_size = size1 * 10, size0 * 10
        if bands == 1:
            if height > x_view_size or width > y_view_size:
                print("The input data too large, read view of the data array.")
                data_array = src.read(1, out_shape=(x_view_size, y_view_size))
            else:
                data_array = src.read(1)
            print("Pre-processing the bands data..")
            stretched = linear_stretch(data_array, 2, nodata=nodata)
            gamma_trans = gamma(stretched, aoto_gamma(stretched, nodata=nodata))
            print("Creating thumbnails..")
            IMG = pilImage(gamma_trans)
            IMG.saveThumb(out_thumbs, (size0, size1))
        elif bands == 3:
            print("Use band_1, band_2, band_3 as the RGB bands respectly.")
            if height > x_view_size or width > y_view_size:
                print("The input data too large, read view of the data array.")
                r = src.read(1, out_shape=(x_view_size, y_view_size))
                g = src.read(2, out_shape=(x_view_size, y_view_size))
                b = src.read(3, out_shape=(x_view_size, y_view_size))
            else:
                r, g, b = src.read(1), src.read(2), src.read(3)
            rgb_list = pre_process_thumbs(r, g, b, nodata)
            IMG = pilImage(rgb_list)
            IMG.saveThumb(out_thumbs, (size0, size1))
        elif bands > 3:
            print("Use band_3, band_2, band_1 as the RGB bands respectly.")
            if height > x_view_size or width > y_view_size:
                print("The input data too large, read view of the data array.")
                r = src.read(3, out_shape=(x_view_size, y_view_size))
                g = src.read(2, out_shape=(x_view_size, y_view_size))
                b = src.read(1, out_shape=(x_view_size, y_view_size))
            else:
                r, g, b = src.read(1), src.read(2), src.read(3)
            print("Pre-processing the bands data..")
            rgb_list = pre_process_thumbs(r, g, b, nodata)
            print("Creating thumbnails..")
            IMG = pilImage(rgb_list)
            IMG.saveThumb(out_thumbs, (size0, size1))


if __name__ == '__main__':
    ref_image = r'F:\SENTINEL\download\down0702\S2_49RFJ_20180627_0\B04.jp2'
    mask = r'D:\testtt33355555333333344433.tif'
    # outputSRS = get_gml_src_proj(gml)
    shp = 'D:/Export_Output.shp'
    # source_ds = ogr.Open(shp)
    # rasterize(shp,mask)
    src = r'D:\temp11\test_osm\YMSS.tif'
    dst = r'D:\temp11\test_osm\YMSS1.tif'
    # change_dtype(src, dst)
    img = r'D:\temp11\pleiades_test\pleiades_test_MS.tif'
    size = 500
    # create_thumbnail_rs(img, "d:/temp11/thumbs_5.png")
    import time
    st = time.time()
    in_raster = r"D:\temp11\cd_1m.tif"
    out_raster = r"D:\temp\演示数据\cd_1m.tif"
    with rasterio.open(in_raster) as src:
        kargs = src.meta.copy()
        kargs.update({'nodata': 0,
                      'count': 3})
        print(kargs)
        
        arr = src.read(1)
        arr[arr==255] = 0
        mask = arr == 1
        # 22 181 255
        # 
        with rasterio.open(out_raster, 'w', **kargs) as dst:
            arr[mask] = 255
            dst.write(arr, 1)
            arr[mask] = 1
            dst.write(arr, 2)
            arr[mask] = 1
            dst.write(arr, 3)
    print(time.time() - st)