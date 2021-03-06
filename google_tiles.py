# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 17:23:30 2020

@author: DELL
"""
import os
import glob
import random
import math
import concurrent.futures as future

from pyproj import Transformer

from download_utils import download_one_by_requests_basic
from io_utils import write_text


def lonlat2xy(lon_deg, lat_deg, zoom):
    """由经纬度计算当前缩放级别下对应瓦片的x,y坐标,方法来自:
    https://wiki.openstreetmap.org/wiki/Slippy_map_tilenames"""
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return (xtile, ytile)


def xy2lonlat(xtile, ytile, zoom):
    """计算指定瓦片左上角点的经纬度,方法同样来自osm网站."""
    n = 2.0 ** zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_deg = math.degrees(lat_rad)
    return (lon_deg, lat_deg)


def get_res_mercator(zoom):
    """计算WebMercator当前缩放级别下的图像分辨率"""
    return 20037508.3427892 * 2 / 256 / 2**zoom


class Google_Tiles_Downloader(object):
    """下载指定区域谷歌地图卫星影像,瓦片地址:
    http://mt{0-3}.google.cn/vt/lyrs=m&hl=zh-CN&gl=cn&x={x}&y={y}&z={z}
    其中：
    lyrs = 类型
        h = roads only 仅限道路
        m = standard roadmap 标准路线图
        p = terrain 带标签的地形图
        r = somehow altered roadmap 某种改变的路线图
        s = satellite only 仅限卫星
        t = terrain only 仅限地形
        y = hybrid 带标签的卫星图
    gl = 坐标系
        CN = 中国火星坐标系
    hl = 地图文字语言
        zh-CN = 中文
        en-US = 英文
    x, y, z: 瓦片坐标及缩放级别

    Attributes:
        is_cn: 是否采样国内火星坐标系

    """

    def __init__(self, lon1, lat1, lon2, lat2, zoom=10, out_dir='gtiles',
                 is_cn=False):
        self.x0, self.y0 = lonlat2xy(lon1, lat1, zoom)
        self.x1, self.y1 = lonlat2xy(lon2, lat2, zoom)
        self.zoom = zoom
        self.out_dir = out_dir
        self.is_cn = is_cn
        self.error_urls = os.path.join(out_dir, 'error_urls.txt')

    def construct_url(self, x, y):
        """构建下载url."""
        server_id = random.choice(range(4))
        base_url = 'http://mt{}.google.cn/vt/lyrs=s&'.format(server_id)
        tile_xyz = 'x={0}&y={1}&z={2}'.format(x, y, self.zoom)
        if self.is_cn:
            url = base_url + 'hl=zh-CN&gl=cn&' + tile_xyz
        else:
            url = base_url + tile_xyz
        return url

    def get_world_file(self, x, y):
        """创建坐标文件."""
        # 左上角经纬度
        ul_lon, ul_lat = xy2lonlat(x, y, self.zoom)
        # 转换为WebMercator坐标
        transformer = Transformer.from_crs(4326, 3857, always_xy=True)
        ul_x, ul_y = transformer.transform(ul_lon, ul_lat)
        # 计算当前缩放级别下分辨率
        res = get_res_mercator(self.zoom)
        return ("{0:16.7f}\n{1:16.7f}\n{2:16.7f}\n{3:16.7f}\n{4:16.7f}"
                "\n{5:16.7f}".format(res, 0, 0, -res, ul_x, ul_y))

    def construct_out_file(self, x, y):
        """构建输出文件,文件存储为{z/x/y.png}的形式."""
        out_png = os.path.join(self.out_dir, str(self.zoom),
                               str(x), str(y) + '.png')
        out_pgw = out_png.replace('.png', '.pgw')
        return out_png, out_pgw

    def download_one(self, x, y):
        """下载指定瓦片数据,保存为png格式,并生成对应的pgw."""
        out_png, out_pgw = self.construct_out_file(x, y)
        url = self.construct_url(x, y)

        download_one_by_requests_basic(url, out_png)

        if os.path.exists(out_png):
            write_text(out_pgw, self.get_world_file(x, y))
        else:
            print("Download error. The png file does not exist.")
            write_text(self.error_urls, url + '\n', add=True)

    def download_tiles(self):
        """循环每一个瓦片进行下载."""
        for x in range(self.x0, self.x1 + 1):
            for y in range(self.y0, self.y1 + 1):
                self.download_one(x, y)

    def download_tiles_parallel(self, max_workers=4):
        """批量下载."""
        with future.ThreadPoolExecutor(max_workers=max_workers) as executor:
            for x in range(self.x0, self.x1 + 1):
                for y in range(self.y0, self.y1 + 1):
                    executor.submit(self.download_one, x, y)


if __name__ == '__main__':
    lon1, lat1 = 112.1, 38.25
    lon2, lat2 = 112.7, 37.7
    z = 16

    cn = False

    tile_dir = r'D:\test\google_tiles'

    # import time
    # st = time.time()
    # 下载
    # gtd = Google_Tiles_Downloader(lon1, lat1, lon2, lat2, zoom=z,
    #                               out_dir=tile_dir, is_cn=cn)
    # gtd.download_tiles_parallel(5)

    # 拼接, 定义投影
    mosaic_file = r'D:\test\google_tiles\m_z16.tif'

    import rasterio
    from mosaic import merge_rio
    tile_files = glob.glob(os.path.join(tile_dir, str(z), '*/*.png'))

    merge_rio([rasterio.open(f) for f in tile_files], mosaic_file,
              crs='EPSG:3857')

    # 投影转换
    project_file = r'D:\test\google_tiles\p_z16.tif'
    from projections import reproject_rio
    reproject_rio(mosaic_file, project_file, dst_crs='EPSG:4326')


    # gtd.download_tiles()
    # print(time.time() - st)