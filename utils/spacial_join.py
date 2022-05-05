#!/usr/bin/python
# coding:utf-8

# ---------------------------------------------------------------------------
# @Author     : limeng.meng
# @Date       : 2020-12-23
# @File       : spacial_join.py
# @Description: 求与输入数据的空间交集
#
# 修改:
#   2021-04-01: 添加高分7数据范围方法
# ---------------------------------------------------------------------------
import os
from argparse import ArgumentParser
import xml.etree.ElementTree as ET

import rasterio
import rasterio.warp
from rtree import index

from .rsiutils import get_image_file
from .rpc_trans import cal_boundary_coords

def build_index(ref_doms, trans_bounds=True, file_idx=None):
    """利用输入的数据建立空间索引(RTREE).
        * 如果投影为空 采用默认范围建立索引；
        * 如果trans_bounds为False 采用默认范围建立索引；
        * 在trans_bounds为True并且投影不为空的情况下进行边界转换后(经纬度)建立索引.
    
    """
    if file_idx is None:
        idx = index.Index()
    elif isinstance(file_idx, str):  # 文件类型索引
        dir_name = os.path.split(file_idx)[0]
        if not os.path.isdir(dir_name):
            os.makedirs(dir_name)
        idx = index.Rtree(file_idx)

    def build_index():
        for i, r in enumerate(ref_doms):
            ds = rasterio.open(r)
            idx.insert(i, tuple(ds.bounds))
        return idx
    
    def build_index_trans():
        for i, r in enumerate(ref_doms):
            ds = rasterio.open(r)
            source_bd = ds.bounds
            transed_bd = rasterio.warp.transform_bounds(ds.crs, 'EPSG:4326',
                                                        source_bd.left, source_bd.bottom,
                                                        source_bd.right, source_bd.top)
            idx.insert(i, transed_bd)
        return idx

    ds0 = rasterio.open(ref_doms[0])
    crs0 = ds0.crs
    ds0.close()

    if crs0 is None:  # 投影为空
        print("Source crs cannot be read, use default bounds.")
        return build_index()

    if not trans_bounds:  # 不需要转换边界
        return build_index()

    # 需要进行转换并且投影不为空
    if trans_bounds and crs0 is not None:
        return build_index_trans()


def get_bounds_from_xml(input_file, is_gf7='false'):
    """从元数据中获取范围(left, bottom, right, top)."""
    is_gf7 = 'true' if input_file.lower().startswith('gf7') else 'false'
    suffix = '.tif' if is_gf7 == 'true' else '.tiff'
    meta_xml = input_file.replace(suffix, '.xml')
    fn = os.path.basename(input_file)
    if fn.upper().startswith('GF6_WFV'):
        meta_xml = os.path.join(os.path.dirname(input_file), fn.split('-')[0] + '.xml')
    tree = ET.parse(meta_xml)
    root = tree.getroot()
    def get_coordinate(pos):
        """lambda x: float(root.find(x).text)"""
        return float(root.find(pos).text)
    if is_gf7 == 'true':
        lats = []
        lons = []
        for n in root.iter():
            if n.tag == 'ProductGeographicRange':
                for ele in n:
                    # 考虑未经变换的数据,左上角右下角不一定是最大最小坐标
                    if ele.tag == 'LeftTopPoint':
                        lons.append(float(ele.find('Longtitude').text))
                        lats.append(float(ele.find('Latitude').text))
                    elif ele.tag == 'RightTopPoint':
                        lons.append(float(ele.find('Longtitude').text))
                        lats.append(float(ele.find('Latitude').text))
                    elif ele.tag == 'RightBottomPoint':
                        lons.append(float(ele.find('Longtitude').text))
                        lats.append(float(ele.find('Latitude').text))
                    elif ele.tag == 'LeftBottomPoint':
                        lons.append(float(ele.find('Longtitude').text))
                        lats.append(float(ele.find('Latitude').text))
    else:
        lats = list(map(get_coordinate, ['TopLeftLatitude', 'TopRightLatitude',
                                        'BottomRightLatitude', 'BottomLeftLatitude']))
        lons = list(map(get_coordinate, ['TopLeftLongitude', 'TopRightLongitude',
                                        'BottomRightLongitude', 'BottomLeftLongitude']))
    return (min(lons), min(lats), max(lons), max(lats))

def get_bounds_from_rpc(input_file):
    """利用RPC模型获取影像范围"""
    lons, lats = cal_boundary_coords(input_file)
    return (min(lons), min(lats), max(lons), max(lats))


def get_intersect_files(ref_doms, in_r, is_gf7='false'):
    """利用参考DOM构建空间索引，从输入数据对应的元数据中读到范围，得到与输入数据范围有交集的参考DOM.
    Note: 
        is_gf7参数不再用
    """
    # ref_doms = get_image_file(ref_dom_dir)
    idx = build_index(ref_doms)
    try:
        bound = get_bounds_from_xml(in_r, is_gf7)
    except:
        print("Something wrong parsing xml file, try calculate boundary from rpc.")
        bound = get_bounds_from_rpc(in_r)

    ids = list(idx.intersection(bound))
    return [ref_doms[i] for i in ids]


def get_intersect_files_from_file_index(ref_doms, in_r, idx_file, is_gf7='false'):
    """利用已经生成的空间索引，从输入数据对应的元数据中读到范围，得到与输入数据范围有交集的参考DOM."""
    # ref_doms = get_image_file(ref_dom_dir)
    idx = index.Rtree(idx_file)
    bound = get_bounds_from_xml(in_r, is_gf7)
    ids = list(idx.intersection(bound))
    return [ref_doms[i] for i in ids]


def get_shp_file_bounds(input_shp_file):
    """获取shapefile范围,输出为(xmin,ymin,xmax,ymax)的形式"""
    from osgeo import ogr
    ds = ogr.Open(input_shp_file)
    lyr = ds.GetLayerByIndex(0)
    extent = lyr.GetExtent()
    return (extent[0], extent[2], extent[1], extent[3])


def get_intersect_images_from_shp(images, input_shp_file):
    """从输入影像列表中获取与输入shapefile有交集的影像"""
    idx = build_index(images)
    bound = get_shp_file_bounds(input_shp_file)
    ids = list(idx.intersection(bound))
    return [images[i] for i in ids]


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--ref_dom_dir', required=True,
                       help="参考文件目录")
    parser.add_argument('--input_raster', required=True,
                       help="输入栅格文件")
    parser.add_argument('--is_gf7', default='false', choices=['true', 'false'],
                       help='是否是高分7数据的标签')

    args = parser.parse_args()

    ref_doms = get_image_file(args.ref_dom_dir)
    dom_idx = get_intersect_files(ref_doms, args.input_raster, args.is_gf7)

    # print(get_bounds_from_xml(r'D:\work\data\影像样例\GF7_DLC_E129.3_N45.9_20200429_L1L0000091709-BWDMUX\GF7_DLC_E129.3_N45.9_20200429_L1L0000091709-BWDMUX.tif', is_gf7='true'))

    # print(get_bounds_from_xml(r'D:\work\data\影像样例\GF6_WFV_E104.9_N35.8_20201107_L1A1120048697\GF6_WFV_E104.9_N35.8_20201107_L1A1120048697-2.tiff'))

    # print([dom_idx])
    print()