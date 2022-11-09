# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 15:26:20 2018

一些与矢量操作有关的工具

@author: Administrator
"""

import os
from math import ceil
from osgeo import ogr
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon, mapping, shape
from shapely.validation import make_valid
from shapely.ops import transform
from fiona.crs import from_epsg

import rasterio
from rasterio import features

import pyproj

from xml.etree import ElementTree


def get_features_from_shp(shpfile):
    """Get shapely geometry features from shapefile.

    Args:
        shpfile (str): The input shapefile name.
    """

    data = gpd.read_file(shpfile)
    return [mapping(data.geometry[i]) for i in range(data.geometry.count())]


def checkHJ(shpfile, xmlDir, outShpFile):
    """检查环境卫星数据元数据，读取数据范围，从中找出与目标区域相交的卫星数据

    Args:
        shpfile (str) : 目标区域，shapefile文件，经纬度投影.
        xmlDir (str): 包含环境卫星元数据的文件目录.
        outShpFile (str): 输出与目标区域相交的卫星数据范围.

    """

    # 读取目标区域shapefile文件，获取几何信息
    data = gpd.read_file(shpfile)
    geo_info = data['geometry']
    geo_info_collection = MultiPolygon([geo_info[i] for i in range(
            geo_info.count())])

    # xmls = glob.glob(os.path.join(xmlDir,'*.XML'))
    xmls = os.listdir(xmlDir)
    # 初始化几个列表作为数据属性表，如：轨道号，文件名称（）
    poly_list = []
    path_list = []
    row_list = []
    satPath_list = []
    satRow_list = []
    fn_list = []
    # 循环读取元数据
    for i in range(len(xmls)):
        # xml = r'E:\gongzuo\HJ1B\HJ1B-CCD1-4-51-20171010-L20003239956
        # \HJ1B-CCD1-4-51-20171010-L20003239956.XML'
        xml = os.path.join(xmlDir, xmls[i])

        file_base_name = xmls[i][:-4]

        # 直接读取提示encoding错误，因此用字符串的方式读取xml文件
        with open(xml, 'r') as file:
            xml_str = file.read()

        root = ElementTree.fromstring(xml_str.replace('encoding="GB2312"', ''))

        data = root.find('data')
        # 提取所需要数据，轨道号以及4个角点坐标
        scenePath = int(data.find('scenePath').text)
        sceneRow = int(data.find('sceneRow').text)

        satPath = int(data.find('satPath').text)
        satRow = int(data.find('satRow').text)

        dataUpperLeftLat = float(data.find('dataUpperLeftLat').text)
        dataUpperLeftLong = float(data.find('dataUpperLeftLong').text)

        dataUpperRightLat = float(data.find('dataUpperRightLat').text)
        dataUpperRightLong = float(data.find('dataUpperRightLong').text)

        dataLowerLeftLat = float(data.find('dataLowerLeftLat').text)
        dataLowerLeftLong = float(data.find('dataLowerLeftLong').text)

        dataLowerRightLat = float(data.find('dataLowerRightLat').text)
        dataLowerRightLong = float(data.find('dataLowerRightLong').text)

        # 构建数据范围
        poly = Polygon([(dataUpperLeftLong, dataUpperLeftLat),
                        (dataLowerLeftLong, dataLowerLeftLat),
                        (dataLowerRightLong, dataLowerRightLat),
                        (dataUpperRightLong, dataUpperRightLat)])

        # 检查数据范围是否与省界相交  ，将与省界相交数据写出
        if poly.intersects(geo_info_collection):
            # print(poly)
            poly_list.append(poly)
            path_list.append(scenePath)
            row_list.append(sceneRow)
            satPath_list.append(satPath)
            satRow_list.append(satRow)
            fn_list.append(file_base_name)

    listToShp(outShpFile, poly_list, epsg=4326, path=path_list,
              row=row_list, satpath=satPath_list,
              satrow=satRow_list, filename=fn_list)


def listToShp(outShpFile, geometry_list, epsg=4326, encoding='gbk',
              schema=None, **kwargs):
    """利用geoPandas生成新数据并导出shp文件.

    Args:
        outShpFile (str): 输出shape文件.
        geometry_list (list): 文件几何属性，每一个元素代表一条记录.
        epsg (str or int): 投影EPSG编号，默认4326.
        encoding (str): 字符编码，默认GBK.
        kwargs (dict) : 属性字段，字段名称及对应的属性列表，长度与几何列表一致.

    Raises:
        ValueError: If geometry_list is empty
                or kwargs value is not a list or a tuple
                or length of kwargs value is not equal to the geometry_list.

    """

    if not geometry_list:
        raise ValueError('Geometry must not be empty.')
    # 空数据
    newdata = gpd.GeoDataFrame()
    # 初始化投影和几何信息
    newdata['geometry'] = None
    newdata.crs = None if epsg is None else from_epsg(epsg)

    # 添加数据
    for key in kwargs:
        # 保证输入是列表 并且长度一致
        if not isinstance(kwargs[key], (list, tuple)):
            raise ValueError('Value of kwargs must be a list or tuple.')

        if len(geometry_list) != len(kwargs[key]):
            raise ValueError('Kwargs value list must have the \
                             same length of the geometry_list.')

        newdata[key] = kwargs[key]

    newdata['geometry'] = geometry_list
    # 写出成shp文件
    newdata.to_file(outShpFile, encoding=encoding, schema=schema)


def extent2grid(outputGridfn,xmin,xmax,ymin,ymax,gridHeight,gridWidth, SpatialReference=None):
    """Generate grid(shapefile)
    
    Usage:
    output_shp = "/mnt/out.shp"
    ref_raster = "/mnt/ref_raster.tif"
    ds = gdal.Open(ref_raster)
    # SpatialReference
    sr = ds.GetSpatialRef()
    grid_width_pixel = grid_height_pixel = 10000
    # extent
    geotransform = ds.GetGeoTransform()
    xmin, res_x, _, ymax, _, res_y = geotransform
    
    cols, rows = (ds.RasterXSize, ds.RasterYSize)
    
    buffer_x = cols * res_x * 0.05
    buffer_y = abs(rows * res_y * 0.05)
    
    xmin, xmax = (xmin-buffer_x, xmin + cols * res_x * 1.05)
    ymin, ymax = (ymax + rows * res_y * 1.05, ymax + buffer_y)

    # grid width and height
    gridHeight, gridWidth = (int(grid_width_pixel) * res_x, 
                             abs(int(grid_height_pixel) * res_y))
    
    # 检查/创建输出文件所在目录
    dirname = os.path.dirname(output_shp)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    extent2grid(output_shp, xmin, xmax, ymin, ymax, gridHeight, gridWidth, sr)
    
    """
    xmin = float(xmin)
    xmax = float(xmax)
    ymin = float(ymin)
    ymax = float(ymax)
    gridWidth = float(gridWidth)
    gridHeight = float(gridHeight)

    # get rows
    rows = ceil((ymax-ymin)/gridHeight)
    # get columns
    cols = ceil((xmax-xmin)/gridWidth)

    # start grid cell envelope
    ringXleftOrigin = xmin
    ringXrightOrigin = xmin + gridWidth
    ringYtopOrigin = ymax
    ringYbottomOrigin = ymax-gridHeight

    # create output file
    outDriver = ogr.GetDriverByName('ESRI Shapefile')
    if os.path.exists(outputGridfn):
        os.remove(outputGridfn)
    outDataSource = outDriver.CreateDataSource(outputGridfn)
    outLayer = outDataSource.CreateLayer(outputGridfn,SpatialReference, geom_type=ogr.wkbPolygon)
    outLayer.CreateField(ogr.FieldDefn('NewMapNo',ogr.OFTString))
    featureDefn = outLayer.GetLayerDefn()

    # create grid cells
    countcols = 0
    while countcols < cols:
        countcols += 1

        # reset envelope for rows
        ringYtop = ringYtopOrigin
        ringYbottom =ringYbottomOrigin
        countrows = 0

        while countrows < rows:
            countrows += 1
            ring = ogr.Geometry(ogr.wkbLinearRing)
            ring.AddPoint(ringXleftOrigin, ringYtop)
            ring.AddPoint(ringXrightOrigin, ringYtop)
            ring.AddPoint(ringXrightOrigin, ringYbottom)
            ring.AddPoint(ringXleftOrigin, ringYbottom)
            ring.AddPoint(ringXleftOrigin, ringYtop)
            poly = ogr.Geometry(ogr.wkbPolygon)
            poly.AddGeometry(ring)

            # add new geom to layer
            outFeature = ogr.Feature(featureDefn)
            outFeature.SetGeometry(poly)
            outFeature.SetField("NewMapNo", f"grid_{countcols:04d}_{countrows:04d}")
            outLayer.CreateFeature(outFeature)
            outFeature.Destroy

            # new envelope for next poly
            ringYtop = ringYtop - gridHeight
            ringYbottom = ringYbottom - gridHeight

        # new envelope for next poly
        ringXleftOrigin = ringXleftOrigin + gridWidth
        ringXrightOrigin = ringXrightOrigin + gridWidth

    # Close DataSources
    outDataSource.Destroy()


def create_shp_from_shapely_geometry(geometry, out_shp='out.shp'):
    """shapely geometry转shapefile"""
    # Now convert it to a shapefile with OGR    
    driver = ogr.GetDriverByName('Esri Shapefile')
    ds = driver.CreateDataSource(out_shp)
    layer = ds.CreateLayer('', None, ogr.wkbPolygon)
    # Add one attribute
    layer.CreateField(ogr.FieldDefn('id', ogr.OFTInteger))
    defn = layer.GetLayerDefn()
    
    if not geometry.geom_type.startswith('Multi'):
        geometry = [geometry]
    
    ## If there are multiple geometries, put the "for" loop here
    for i, poly in enumerate(geometry):
    
        # Create a new feature (attribute and geometry)
        feat = ogr.Feature(defn)
        feat.SetField('id', i)
        
        # Make a geometry, from Shapely object
        geom = ogr.CreateGeometryFromWkb(poly.wkb)
        feat.SetGeometry(geom)
        
        layer.CreateFeature(feat)
        feat = geom = None  # destroy these
        
    # Save and close everything
    ds = layer = feat = geom = None


def get_geometry_masks(in_raster, have_rpc=False, do_sieve=False, sieve_size=10):
    """从卫星原始数据获取影像实际有效范围"""
    with rasterio.open(in_raster) as src:
        # 如果数据没有空值，并且不是原始影像，返回四至范围
        if src.nodata is None and not have_rpc:
            x1, y1, x2, y2 = src.bounds
            print("  Reference image donot have nodata value, return image extent.")
            return Polygon([(x1, y1), (x1, y2), (x2, y2), (x2, y1), (x1, y1)])

        if not have_rpc:
            # print("Read data masks..")
            arr = src.read_masks()
        else:
            # print("Read data values..")
            arr = src.read()
        print("  Get masks..")
        mask = arr.any(axis=0)  # 任意波段中值不为0
        if do_sieve:
            print("  Sieve small parts..")
            mask = features.sieve(mask.astype('uint8'), size=sieve_size)
        arr = None

        # get features of the same value from an array
        print("  Get features of the mask..")
        none_zero = features.shapes(mask.astype('uint8'), mask,
                                    connectivity=4, 
                                    transform=src.transform)
    # 获取geometry
    print("  Convert to shapely geometry and return..")
    polygons = []
    for p, _ in none_zero:
        polygons.append(shape(p))
    if len(polygons) == 0: 
        return
    elif len(polygons) == 1:
        return polygons[0]
    else:
        return MultiPolygon(polygons)


def cal_area_intersection(ori_img, ref_img, out_txt="/tmp/area.txt"):
    """Get valid extent of two images, intersection and write the area to 
    text."""
    print("Warp origin image through rpc model..")
    gdal.Warp('/tmp/temp.vrt', ori_img,  rpc=True)
    print("Get mask of origin image...")
    geo_sat = make_valid(get_geometry_masks('/tmp/temp.vrt', have_rpc=True))
    print("Get mask of reference image...")
    geo_ref = make_valid(get_geometry_masks(ref_img, have_rpc=False, 
                                            do_sieve=True, sieve_size=10))
    # 坐标转换
    dst_crs = pyproj.CRS(rasterio.open(ref_img).crs)
    if dst_crs.is_projected:
        print("Transform to WGS1984...")
        wgs84 = pyproj.CRS('EPSG:4326')
        project = pyproj.Transformer.from_crs(dst_crs, wgs84, always_xy=True).transform
        geo_ref = make_valid(transform(project, geo_ref))
    print("Intersect.....")
    if geo_sat is not None and geo_ref is not None:
        inter_area = geo_sat.intersection(geo_ref).area
        print("Intersect area: ", inter_area)
        dirname = os.path.dirname(out_txt)
        if not os.path.isdir(dirname): os.makedirs(dirname)
        with open(out_txt, 'w') as ofh:
            ofh.write(str(inter_area) + '\n')
            ofh.write(os.path.normpath(ori_img) + '\n')
            ofh.write(os.path.normpath(ref_img))

if __name__ == '__main__':
    
    xmin,xmax,ymin,ymax = (36570535, 36575818, 3797003, 3800370)
    gridHeight,gridWidth = (2000, 2000)
    outputGridfn = "d:/out.shp"
    extent2grid(outputGridfn,xmin,xmax,ymin,ymax,gridHeight,gridWidth)
    
    

    # from rtree import index
    # idx = index.Index()


    # 读取省界数据，得到几何信息（投影需一致，经纬度）
    # gdsheng = r'D:\test\test_fishnet1.shp'
    # in_r = r'D:\work\data\影像样例\610124.tif'
    # import rasterio
    # ds = rasterio.open(in_r)
    # # a = (get_features_from_shp(gdsheng))

    # data = gpd.read_file(gdsheng)
    # for i, g in enumerate(data.geometry):
    #     idx.insert(i, g.bounds)

    # # 根据栅格数据四至范围找到相交的分幅
    # ids = list(idx.intersection(tuple(ds.bounds)))
    # from mask import extract_by_mask_rio_ds
    # for i in ids:
    #     print(i)
    #     extract_by_mask_rio_ds([data.geometry[i]], ds,
    #                            'd:/temp/' + data.loc[i, 'name'] + '.tif',
    #                            nodata=0)


    # xml_dir = r'F:\HJ\hjxml_0823'
    # outshp = 'd:/bbbbb.shp'
    # checkHJ(gdsheng,xml_dir,outshp)
    
