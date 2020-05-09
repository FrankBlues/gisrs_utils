# -*- coding: utf-8 -*-
"""Download Landsat 8 data from AWS.
"""
import concurrent.futures
import os
from download_utils import download_one_by_urllib_using_range_header
from io_utils import read_json

try:
    from usgs import api
except ImportError:
    print("The usgs module not found!")
    pass


class Downloader_Landsat8(object):
    """ 利用USGS提供API查询Landsat 8, 从亚马逊云下载.
    See: https://github.com/kapadia/usgs and
         https://docs.opendata.aws/landsat-pds/readme.html

    Attributes:
        path, row (int or str): Landsat wrs2 path and row.
        pr (str): wrs2 pathrow number.
        times (list): 需要下载数据起始日期列表(iso format),列表第一个为起始日期,最后
                一个为结束日期.
        aws_l8_url (str): Landsat 8 亚马逊云下载地址.
        api_key (str): USGS api key.
        _dataset (str): Dataset supported by USGS, here 'SENTINEL_2A'.
        _node (str): USGS download node, here 'EE' which means EarthExplorer.
        _centroidLonlat (dict): Centroid coordinates of input tiles.

    """
    def __init__(self, path, row, times):
        """ 初始化

        """
        if int(path) > 151 or int(path) < 113:
            raise ValueError("Wrong path.")
        if int(row) < 23 or int(row) > 49:
            raise ValueError("Wrong row.")
        self.path = "{:03d}".format(path)
        self.row = "{:03d}".format(row)
        self.pr = self.path + self.row
        self.times = times

        # login
        try:
            api.login('menglimeng', '302112aA', save=True, catalogId='EE')
        except Exception:
            print('Cannot login to usgs earthexplorer right now.')
        self.api_key = api._get_api_key(None)

        self.aws_l8_url = 'https://landsat-pds.s3.amazonaws.com/c1/L8/'
        self._dataset = 'LANDSAT_8_C1'
        # self._dataset = 'LANDSAT_ETM_C1'
        self._node = 'EE'

        # centroid coordinate of each tile
        self._centroidLonlat = read_json(os.path.join(
                os.path.dirname(__file__), 'aux_data',
                'wrs2_centroid_china.json'))

    def __repr__(self):
        return ("Query Landsat 8 data between {0} and {1} from usgs, "
                "download from AWS.".format(self.times[0], self.times[-1]))

    def _query_by_point(self, lon, lat, date_start, date_end):
        """ Query by point using usgs api.

        Args:
            lon, lat (float): Coordinates.
            date_start, date_end (str): The start and end date.

        Returns:
            query object.

        """
        return api.search(self._dataset, self._node, lat=lat, lng=lon,
                          distance=100,
                          # ll={ "longitude": 108.963791 ,
                          # "latitude": 19.845140},
                          # ur={ "longitude": 110.266751 ,
                          # "latitude": 20.831747},
                          start_date=date_start, end_date=date_end,
                          api_key=self.api_key,
                          )

    def query(self):
        """Make the query.

        Returns:
            iterator: Tuple containing the tile and the search result
                associated with the tile.

        """

        lat = self._centroidLonlat[self.pr]['centroid_lat']
        lon = self._centroidLonlat[self.pr]['centroid_lon']
        search_result = self._query_by_point(lon, lat,
                                             self.times[0], self.times[-1])
        total_hits = search_result['data']['totalHits']
        if total_hits == 0:
            print("No data found.")
            return (None, None)
        else:
            print("Total {} found!".format(total_hits))
            return (self.pr, search_result['data']['results'])

    def meta_data(self, entityId):
        """Get metadata.

        Args:
            entityId (str): Landsat8 product Id.

        """
        return api.metadata(self._dataset, self._node, [entityId],
                            api_key=self.api_key)

    def get_download_url(self, cloud=100):
        """Get the download urls.

        Args:
            cloud (float): The max cloud coverage allowed.

        Returns:
            generator: Tuples containing tile, date and download urls.

        """
        print("Querying requested product...")
        _pr, _result = self.query()

        def index_url(product_name):
            return (self.aws_l8_url + self.path + '/' + self.row + '/'
                    + product_name + '/index.html')
        for i, result in enumerate(_result):
            print("Product num: {}:".format(i + 1))
            displayId = result['displayId']
            acquisitionDate = result['acquisitionDate']
            entityId = result['entityId']
            # May find more than one for each tile, so make sure to
            # download the one matching the tile name.
            url = index_url(displayId)
            # print(url)
            from download_utils import open_request
            print("  Get the product url on AWS, Try first! ")
            res = open_request(url)
            # 如果在亚马逊云未找到产品（一般是T2）,则测试RT产品是否存在
            if isinstance(res, int):
                print("  Not found, trying RT product.")
                namelist = displayId.split('_')
                rt_name = displayId.replace(namelist[4],
                                            namelist[3]).replace('T2', 'RT')
                url = index_url(rt_name)
                res = open_request(url)
                if isinstance(res, int):
                    print("  Product of date {0} and pr {1} not found.".format(
                            acquisitionDate,
                            self.pr))
                    url = None

            meta = self.meta_data(entityId)
            cloud_cover = meta['data'][0]['metadataFields'][17]['value']
            if float(cloud_cover) > cloud:
                print("  Cloud cover: {}, too many!".format(cloud_cover))
            elif url is not None:
                yield url

    def download(self,
                 band_list=['B1', 'B2', 'B3', 'B4', 'B5',
                            'B6', 'B7', 'B8', 'B9', 'B10', 'B11', 'BQA'],
                 meta=['MTL.txt'],
                 cloud=90):
        """Download.

        Args:
            band_list (list): Landsat 8 band to be downloaded.
            meta (list): Landsat 8 metadata file suffixes.
            cloud (float): The max cloud coverage allowed.

        """
        download_num = 0
        urls = self.get_download_url(cloud)
        # print(urls)
        for url in urls:
            if url is not None:
                download_num += 1
                print(url)
                url_base = url.rstrip('index.html')
                product = url.split('/')[-2]
                print(" Downloading {}..".format(product))
                for b in band_list:
                    url_b = url_base + product + '_' + b + '.TIF'
                    # download via wget.
                    # print(url_b)
                    # cmd = "wget {}".format(url_b)
                    # os.system(cmd)
                for m in meta:
                    url_m = url_base + product + '_' + m
                    # cmd = "wget {}".format(url_m)
                    # os.system(cmd)
                    # print(url_m)
        print("{} downloaded.".format(download_num))


def download_one_scene_from_AWS_LC8(landsat_scene, landsat_bands, landsat_meta,
                                    out_dir, part=20):
    """Download Landsat 8 data from AWS(one scene).

    Args:
        landsat_scene (str): Scene name to be downloaded, i.e
                            'LC08_L1TP_122044_20180722_20180731_01_T1'.
        landsat_bands (list): Band list to be downloaded, can be any of
              ['B1','B2','B3','B4','B5','B6','B7','B8','B9','B10','B11','BQA'].
        landsat_meta (list): Metadata to be downloaded, i.e ['MTL.txt'].
        out_dir (str): The target directory.
        part (int): Number of parts to be splitted when downloading.

    """

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    url_list_onescene = []
    para_url = 'https://landsat-pds.s3.amazonaws.com/c1/L8/'

    for b in landsat_bands:
        url_oneband = para_url+landsat_scene[10:13] + '/' + \
            landsat_scene[13:16] + '/' + landsat_scene + '/' + \
            landsat_scene + '_' + b + '.TIF'
        url_list_onescene.append(url_oneband)

    url_mtl = para_url+landsat_scene[10:13] + '/' + landsat_scene[13:16] + \
        '/' + landsat_scene + '/' + landsat_scene + '_' + landsat_meta[0]
    url_list_onescene.append(url_mtl)

    print(url_list_onescene)

    out = [os.path.join(out_dir, url.split('/')[-1])
           for url in url_list_onescene]

    for i in range(len(out)):
        if os.path.exists(out[i]):
            print('The file is already downloaded.')
            continue

        download_one_by_urllib_using_range_header(url_list_onescene[i],
                                                  out[i], part)


def main_lc8():
    """Main procejure to download the Landsat 8 data.
    """
    para_dir = r'I:\landsat\untar'
    landsat_list = ['LC08_L1TP_118039_20180726_20180731_01_T1']

    # in ['B1','B2','B3','B4','B5','B6','B7','B8','B9','B10','B11','BQA']
    landsatBands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B8', 'BQA']
    landsatMeta = ['MTL.txt']

    # download_one_scene_from_AWS_LC8(landsat_list[0],landsatBands,
    #                                 landsatMeta,para_dir,part = 20)
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        for l in landsat_list:
            outDir = os.path.join(para_dir, l)
            executor.submit(download_one_scene_from_AWS_LC8, l,
                            landsatBands, landsatMeta, outDir, part=20)


if __name__ == '__main__':
    # main_lc8()
    from time_utils import get_dates_in_range
    times = get_dates_in_range('2012-01-01', '2012-12-31')
    downer_l8 = Downloader_Landsat8(132, 43, times)

    search_result = downer_l8.query()
    # res = downer_l8.get_download_url()
    # downer_l8.download()
