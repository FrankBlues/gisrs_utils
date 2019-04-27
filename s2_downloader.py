# -*- coding: utf-8 -*-
"""Download sentinel2 data or previews from different source.

Note:
    在linux下用scihub下载;在windows下用code-de生成下载列表用下载工具下载.

Todos:
    * 将一些用户名密码,基础网址等固定参数写入配置文件

"""

import os
import sys
import shutil
import concurrent.futures
import tempfile

from download_utils import download_one_by_requests_basic, construct_request
from download_utils import download_one_by_urllib_basic
from download_utils import download_one_by_urllib_using_range_header
from time_utils import iso_to_datetime, one_hour, datetime_to_iso
from io_utils import read_json, write_csv_list, read_xml, read_csv

try:
    from usgs import api
except ImportError:
    print("The usgs module not found!")
    pass

try:
    from sentinelhub import AwsTileRequest  # , AwsProductRequest,  AwsTile
    from sentinelhub.download import AwsDownloadFailedException
except ImportError:
    print("sentinelhub not used any more.")
    pass


def process_preview_image(thumbsDir, jsons, max_cloud_cover=100):
    """根据元数据（json）整理从sentinel-hub下载的缩略图，按照云量给文件命名输出到指定目录.

    Args:
        thumbsDir (str): 缩略图输出文件夹.
        jsons (list or iterator): json文件(全路径)列表或迭代器.
        max_cloud_cover (float): 最大允许云覆盖比例（0-100）,默认100.

    """
    if not os.path.exists(thumbsDir):
        os.mkdir(thumbsDir)

    for j in jsons:
        print(j)

        try:
            jsonf = read_json(j)
        except Exception as e:
            print(e)
            continue
        productName = jsonf['productName']
        cloudyPixelPercentage = jsonf['cloudyPixelPercentage']

        date = productName[11:19]
        tile = productName[39:44]
        # print(date,tile)

        dirname = os.path.dirname(j)

        if cloudyPixelPercentage > max_cloud_cover:
            shutil.rmtree(dirname)
        else:
            thumbName = "S2_C{0:02d}_T{1}_D{2}.jpg".format(
                int(cloudyPixelPercentage), tile, date)

            try:
                shutil.copy(os.path.join(dirname, 'preview.jpg'),
                            os.path.join(thumbsDir, thumbName))
            except Exception as e:
                print(e)
                print(os.path.join(dirname, 'preview.jpg'))


def download_tile_date_to_csv(thumbsDir, out_csv):
    """将挑选出来的需要下载数据的编号及日期写入CSV文件.

    Args:
        thumbsDir (str): 缩略图所在文件夹.
        out_csv (str): 输出CSV文件.

    """
    if os.path.exists(out_csv):
        os.remove(out_csv)
    for l in os.listdir(thumbsDir):
        line = [l[:-4], l[8:13], l[15:23]]
        write_csv_list(out_csv, line, True, ',')


class Downloader_metas_usgs_ee(object):
    """ 利用USGS提供API查询下载哨兵2数据缩略图.
    See: https://github.com/kapadia/usgs

    Note:
        可能是账号权限原因不能下载数据.

    Attributes:
        tiles (list): 需要下载的哨兵2数据MGRS编号列表.
        times (list): 需要下载数据日期列表(iso format ).
        api_key (str): USGS api key.
        _dataset (str): Dataset supported by USGS, here 'SENTINEL_2A'.
        _node (str): USGS download node, here 'EE' which means EarthExplorer.
        _centroidLonlat (dict): Centroid coordinates of input tiles.

    """
    def __init__(self, tiles, times):
        """ 初始化

        Args:
            tiles (list): 需要下载的哨兵2数据MGRS编号列表.
            times (list): 需要下载数据日期列表(iso format ).
        """

        self.tiles = tiles
        self.times = times

        # login
        try:
            api.login('menglimeng', '302112aA', save=True, catalogId='EE')
        except Exception:
            print('Cannot login to usgs earthexplorer right now.')
        self.api_key = api._get_api_key(None)

        self._dataset = 'SENTINEL_2A'
        self._node = 'EE'

        # centroid coordinate of each tile
        self._centroidLonlat = read_json(os.path.join(
                os.path.dirname(__file__), 'aux_data', 'mgrs_centroid.json'))

    def __repr__(self):
        return ("Query sentinel2 data between {0} and {1} from usgs "
                "earthexplorer.".format(self.times[0], self.times[-1]))

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
        for t in self.tiles:
            # print("Query tile {0} between date "
            #       "{1} and {2}..".format(t, self.times[0], self.times[-1]))
            # centroid coordinate
            lat = self._centroidLonlat[t]['centroid_lat']
            lon = self._centroidLonlat[t]['centroid_lon']
            search_result = self._query_by_point(lon, lat,
                                                 self.times[0], self.times[-1])

            total_hits = search_result['data']['totalHits']
            if total_hits == 0:
                print("No data found.")
                continue
            else:
                # print("Total {} datasets found. ".format(total_hits))
                # parse result
                yield (t, search_result['data']['results'])

    def download_preview(self, out_dir=tempfile.gettempdir(), download=True):
        """ Download preview with cloud coverage in the filename.

        Args:
            out_dir (str): The target directory,default system tempdir.
            download (bool): Whether or not to download the preview.

        """
        results_all = self.query()
        for y in results_all:
            t, result_tile = y
            for result in result_tile:
                entityId = result['entityId']
                browseUrl = result['browseUrl']
                displayId = result['displayId']

                acquisitionDate = result['acquisitionDate']
                # May find more than one for each tile, so make sure to
                # download the one matching the tile name.
                if displayId.split('_')[1][1:] != t:
                    continue
                # yield result['downloadUrl']
                if download:
                    # get cloud cover from metadata
                    meta = api.metadata(self._dataset, self._node, [entityId],
                                        api_key=self.api_key)
                    cloud_cover = meta['data'][0]['metadataFields'][4]['value']
                    # download
                    # print("Downloading preview from {}.".format(browseUrl))
                    download_one_by_requests_basic(
                            browseUrl,
                            os.path.join(out_dir,
                                         "S2_CC{0:02.0f}_T{1}_D{2}.jpg".format(
                                                 float(cloud_cover),
                                                 t,
                                                 acquisitionDate.replace('_',
                                                                         ''))
                                         )
                            )

    def get_download_url(self):
        """Get the download urls.

        Returns:
            generator: Tuples containing tile, date and download urls.

        """
        results_all = self.query()
        for y in results_all:
            t, result_tile = y
            for result in result_tile:
                displayId = result['displayId']
                acquisitionDate = result['acquisitionDate']
                # May find more than one for each tile, so make sure to
                # download the one matching the tile name.
                if displayId.split('_')[1][1:] != t:
                    continue
                yield (t, acquisitionDate, result['downloadUrl'])


class Downloader_metas_sentinel_hub(object):
    """ 通过sentinel-hub下载缩略图及元文件.

    Usage:
        meta_downer = Downloader_metas(tiles,times,data_folder)
        meta_downer.down()
        or
        meta_downer.down_multiThread()

    """
    def __init__(self, tiles, times, out_dir, aws_index=0,
                 metafiles=['tileInfo.json', 'preview.jpg'], works=2):
        """ 初始化.

        Args:
            tiles (list): 需要下载的哨兵2数据MGRS编号列表.
            times (list): 需要下载数据日期列表(iso format ).
            out_dir (str): 下载内容存放路径.
            aws_index (int): 同一天轨道重叠区域可能会有多个数据,数据索引,0或1,默认0.
            metafiles (list): 下载元数据内容,默认['tileInfo.json', 'preview.jpg'].
            works (int): 并发下载时的并发数,默认2.

        """
        self.tiles = tiles
        self.times = times
        self.out_dir = out_dir
        self.aws_index = aws_index
        self.metafiles = metafiles
        self.works = works
        self.meta_url = 'https://roda.sentinel-hub.com/sentinel-s2-l1c/tiles/'

    def down(self):
        """ 下载主程序
        """
        if not os.path.exists(data_folder):
            os.mkdir(data_folder)

        for tile in self.tiles:
            for date in self.times:
                print("Downloading tile:{0},date:{1}".format(tile, date))
                ymd = date.split('-')
                for meta in self.metafiles:
                    # 构建下载URL
                    url = (self.meta_url + tile[:2] + '/' + tile[2:3] + '/' +
                           tile[3:] + '/' + ymd[0] + '/' +
                           str(int(ymd[1])) + '/' + str(int(ymd[2])) +
                           '/' + str(self.aws_index) + '/' + meta)
                    out = os.path.join(data_folder, tile + ',' +
                                       date + ',' + str(self.aws_index), meta)

                    if os.path.exists(out):
                        print('Already exists!')
                        continue
                    else:

                        try:
                            download_one_by_requests_basic(url, out,
                                                           verify=False)
                            # downloadOneByUrllibBasic(url,out)
                        except Exception as e:

                            print("Raise exception while downloading from "
                                  "{}.".format(url))
                            print(e)
                            continue

    def down_multiThread(self):
        """ 并发下载程序
        """
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.works) \
                as executor:
            executor.submit(self.down())


class Downloader_peps(object):
    """ 利用哨兵数据法国镜像（PEPS）,查询、下载（2018年12月后不可用）哨兵2数据,查询基于哨兵2
    数据MGRS 100km网格编号及日期查询当日数据.

    Usage:
        peps_downer = Downloader_peps("49QCD", "2018-11-22")
        peps_downer.down("d:/temp")

    """
    base_query_url = 'https://peps.cnes.fr/resto/api/collections/S2ST/search.json?'
    search_result_json = os.path.join(tempfile.gettempdir(), 'out.json')
    username = None
    password = None

    def __init__(self, tile, date, max_cloud_cover=100):
        """ Initial with tile id, date, max cloud coverage.

        Args:
            tile (str): MGRS tile id, i.e '49QCC'.
            date (str): Date (iso format ), i.e '2018-11-11'.
            max_cloud_cover (float): Max cloud coverage.

        """
        self.tile = tile
        if len(date) != 10:
            raise ValueError("Date format must like '2018-11-11'")
        self.date = date
        self.max_cloud_cover = max_cloud_cover

    @classmethod
    def set_username_and_pass(cls, username='dream15320@gmail.com'):
        """设置用户名密码，密码输入时不用明文显示"""
        import getpass
        cls.username = username
        print("your username is {}\n".format(username))
        cls.password = getpass.getpass(
            prompt="Password (will not be displayed): ")

    @staticmethod
    def get_query_url_S2ST(tile, date):
        """根据网格编号和日期构建数据查询URL
        """
        in_date = iso_to_datetime(date)
        onehour = one_hour()
        # 北京时间
        startDate = datetime_to_iso(
            (in_date - 8*onehour), only_date=False)
        completionDate = datetime_to_iso(
            (in_date + 15*onehour), only_date=False)

        search_condition = ('completionDate={0}&&startDate={1}&tileid={2}&'
                            'processingLevel=LEVEL1C'.format(
                                    completionDate, startDate, tile))
        return Downloader_peps.base_query_url + search_condition

    @staticmethod
    def queryOneTile(tile, date, search_result_json):
        """查询数据（返回json结果）
        """
        user_agent = ("Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/"
                      "537.36 (KHTML, like Gecko) Chrome/67.0.3396.99 Safari"
                      "/537.36")

        headers = {"User-Agent": "{0}".format(user_agent)}

        search_url = Downloader_peps.get_query_url_S2ST(tile, date)
        print("query url: {}\n".format(search_url))
        request = construct_request(search_url, headers=headers)
        download_one_by_urllib_basic(request, search_result_json)

        return search_result_json

    @staticmethod
    def parse_catalog(search_json_file):
        """解析查询后的json数据，返回需要的下载信息，下载地址、云量等.

        Args:
            search_result_json (str): Json file the query result.

        Returns:
            tuple: download url dict and cloud coverage dict.
                The dict key is the product name.

        """
        data = read_json(search_json_file)

        if 'ErrorCode' in data:
            print(data['ErrorMessage'])
            sys.exit(-2)

        # Sort data
        # download_dict = {}
        # storage_dict = {}
        url_dict = {}
        cloudC_dict = {}
        if len(data["features"]) > 0:
            for i in range(len(data["features"])):
                prod = data["features"][i]["properties"]["productIdentifier"]
                # feature_id = data["features"][i]["id"]
                try:
                    # storage = data["features"][i]["properties"]["storage"]
                    # ["mode"]
                    # platform  =data["features"][i]["properties"]["platform"]
                    # recup du numero d'orbite
                    # orbitN=data["features"][i]["properties"]["orbitNumber"]
                    cloudCover = data["features"][i]["properties"][
                            "cloudCover"]
                    down_url = data["features"][i]["properties"][
                            "services"]["download"]["url"]

                    # download_dict[prod] = feature_id
                    # storage_dict[prod] = storage
                    url_dict[prod] = down_url
                    cloudC_dict[prod] = cloudCover

                except Exception:
                    pass
        else:
            print(">>> no product corresponds to selection criteria")
            # sys.exit(-1)
        return (url_dict, cloudC_dict)

    @staticmethod
    def construct_request(url, user, passw, **kargs):
        """ 构建请求 添加Authorization 和 User-Agent 请求头
        """
        import base64
        # Authorization header
        try:
            # python2
            user_pass = base64.b64encode(
                bytes(user+":"+passw))
        except TypeError:
            # python3
            user_pass = base64.b64encode(
                bytes(user+":"+passw, "utf-8"))
            user_pass = user_pass.decode("utf-8")
        # user_agent header
        user_agent = ("Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/"
                      "537.36 (KHTML, like Gecko) Chrome/69.0.3497.81 Safari"
                      "/537.36")

        headers = {"User-Agent": "{}".format(user_agent)}
        headers.update({
            "Authorization": "Basic {0}".format(user_pass)
        })

        if kargs:
            headers.update(kargs)
        return construct_request(url, headers=headers)

    def down(self, outdir):
        """下载"""
        cloudC_dict = self.getCloudCover()
        if cloudC_dict is not None:
            for prod in cloudC_dict:
                if float(cloudC_dict[prod]) > self.max_cloud_cover:
                    print("云量大于{} \n".format(self.max_cloud_cover))
                    return
                else:
                    # download
                    if not os.path.isdir(outdir):
                        print("目标目录不存在，创建")
                        os.mkdir(outdir)

                    url_dict = self.getDownloadUrl()
                    url = url_dict[prod]
                    print(" download url: {} \n".format(url))
                    if Downloader_peps.username is None \
                            or Downloader_peps.password is None:
                        self.set_username_and_pass()

                    request = self.construct_request(
                            url,
                            Downloader_peps.username,
                            Downloader_peps.password)

                    download_one_by_urllib_basic(
                        request, os.path.join(outdir, prod, prod + '.zip'))
                    # download_one_by_urllib_using_range_header(
                    #         request, os.path.join(outdir, prod, '.zip'),50)

    def query(self):
        """查询，以字典类型返回查询到的下载地址，云覆盖"""
        out_json = self.queryOneTile(
            self.tile, self.date, Downloader_peps.search_result_json)
        return self.parse_catalog(out_json)

    def getDownloadUrl(self):
        """获取解析后的URL列表"""
        url_dict = self.query()[0]
        if len(url_dict.items()) != 0:
            return url_dict
        return None

    def getCloudCover(self):
        """获取查询到数据的云覆盖（0-100）列表
        """

        cloudC_dict = self.query()[1]
        if len(cloudC_dict.items()) != 0:
            return cloudC_dict
        return None

    def getDownloadUrlForXL(self):
        """生成迅雷软件可用下载URL列表
         'https://username@password@download_url'

        """
        if Downloader_peps.username is None \
                or Downloader_peps.password is None:
            self.set_username_and_pass()

        url_dict = self.getDownloadUrl()
        return ([url_dict[key][:8] + Downloader_peps.username + ':' +
                Downloader_peps.password + '@' +
                url_dict[key][8:] for key in url_dict])


class Downloader_scihub(Downloader_peps):
    """ 在PEPS查询基础上，根据名称查询获取scihub下载地址，并下载

    Usage:
        scihub_d = Downloader_scihub("49QCC", "2018-12-04")
        scihub_d.query_by_name()
        scihub_d.down('d:/temp')

    """
    scihub_query_baseUrl = "https://scihub.copernicus.eu/dhus/search?q="
    scihub_user = "mellem"
    scihub_pass = "302112aa"

    def __init__(self, tile, date, max_cloud_cover=100):
        """ initial same like class Downloader_peps
        """
        Downloader_peps.__init__(self, tile, date, max_cloud_cover=100)

    def query_by_name(self):
        """ 简单地根据产品名称查询哨兵2数据，查询结果以xml格式存储"""
        queryfiles = []
        peps_down_url_dict = self.getDownloadUrl()

        if peps_down_url_dict is not None:
            cc = 0
            for prod in peps_down_url_dict:
                query_url = (self.scihub_query_baseUrl +
                             "filename:{}*".format(prod))
                # wget --no-check-certificate --user=scihub_user
                # --password=scihub_pass
                # --output-document=query_results.txt query_url
                query_request = self.construct_request(query_url,
                                                       self.scihub_user,
                                                       self.scihub_pass)
                out_xml = os.path.join(tempfile.gettempdir(), 'query' +
                                       str(cc) + '.xml')
                cc += 1
                download_one_by_urllib_basic(query_request, out_xml)
                queryfiles.append(out_xml)

            return queryfiles
        else:
            return None

    @staticmethod
    def parse_query_result(xml="d:/query.xml"):
        """ 解析查询到的结果（xml 文件） ，返回数据产品名称、下载地址、云量"""
        tree = read_xml(xml)
        root = tree.getroot()

        basename = '{http://www.w3.org/2005/Atom}'

        entrys = root.findall("./{}entry".format(basename))

        if len(entrys) == 1:
            for entry in entrys:
                # product name
                prod_name = entry.find("./{}title".format(basename)).text
                print(prod_name)

                # download link
                down_link = entry.find("./{}link".format(
                        basename)).attrib['href']

                for ii in entry.findall("./{}double".format(basename)):
                    if ii.attrib['name'] == 'cloudcoverpercentage':
                        cloud_cover = float(ii.text)
            return (prod_name, down_link[:-6] + '\\' + down_link[-6:],
                    cloud_cover)

        else:
            print("not found or find more than 1.")
            return None

    def down(self, outdir):
        """ 利用wget实现命令行方式下载，在linux下测试可用"""

        queryfiles = self.query_by_name()
        if queryfiles is not None:
            for queryfile in queryfiles:
                prod_name, down_url, cloud_cover = self.parse_query_result(
                        queryfile)
                if cloud_cover < self.max_cloud_cover:
                    print("download url : {} \n".format(down_url))
                    # linux
                    # --content-disposition 采用原数据文件名，不用在指定输出名称（-O）
                    cmd = ('wget --content-disposition --continue --user={0} '
                           '--password={1} -P {2} "{3}"'.format(
                                   self.scihub_user, self.scihub_pass,
                                   outdir, down_url))
                    print(cmd)
                    os.system(cmd)


class Downloader_code_de(Downloader_peps):
    """ 在PEPS查询基础上，根据名称获取哨兵数据德国镜像code-de.org下载地址，并下载.

    Note:
        目前测试(20181206)下载有速度限制，较慢，一般生成下载列表用工具下载.

    Usage:
        code_downer = Downloader_code_de("49QFC", "2018-11-16")
        code_downer.down("d:/temp")
        print(code_downer.getDownloadUrl_with_user_pass())
    """

    code_de_download_baseUrl = "https://code-de.org/download/"
    code_de_user = "menglimeng"
    code_de_pass = "2kct84rm4Dz57iG"

    def __init__(self, tile, date, max_cloud_cover=100):
        """ initial same like class Downloader_peps
        """
        Downloader_peps.__init__(self, tile, date, max_cloud_cover=100)

    def getDownloadUrls_code_de(self):
        """Get downloadurl
        https://code-de.org/download/S2A_MSIL1C_20181129T031051_N0207_R075_T49QEE_20181129T060514.SAFE.zip
        """
        cloud_dict = self.getCloudCover()
        if cloud_dict is None:
            return None

        urllist = []
        for prod in cloud_dict:
            cloud_cover = float(cloud_dict[prod])
            if cloud_cover < self.max_cloud_cover:
                urllist.append("{0}{1}.SAFE.zip".format(
                        self.code_de_download_baseUrl, prod))
            else:
                continue

        if len(urllist) >= 1:
            return urllist
        else:
            return None

    def getDownloadUrl_with_user_pass(self):
        """Get downloadurl.
        With the format 'https://username@password@download_url'

        """
        urls = self.getDownloadUrls_code_de()
        if urls is not None:
            return ["{0}{1}:{2}@{3}".format("https://", self.code_de_user,
                    self.code_de_pass, url[8:]) for url in urls]
        return None

    def down(self, outDir):
        """Download"""

        dlist = self.getDownloadUrls_code_de()
        if dlist is not None:
            for url in dlist:
                print("download url: {} \n".format(url))
                request = self.construct_request(url,
                                                 self.code_de_user,
                                                 self.code_de_pass)
                download_one_by_urllib_basic(request, os.path.join(
                        outDir,
                        url.split('/')[-1]))


@DeprecationWarning
def getFullAWSDownloadList_S2_oneTile(tile, date, index='0', bands=[],
                                      metafiles=[]):
    """构建需要下载文件(单景)的下载列表(亚马逊云)，包含需要下载的波段和元数据.

    Args:
        tile (str): 需要下载数据的tile,MGRS格式，如'49QCE'.
        date (str): 需要下载数据的日期，ISO 8601格式,如'2018-06-02'.
        index (int): 索引，一般'0'或'1'.
        bands (list): 下载波段列表，['B01','B02','B03','B04','B05','B06',
                  'B07','B08','B8A','B09','B10','B11','B12'].
        metafiles (list): 下载元数据列表，['tileInfo.json', 'preview.jpg',
                  'qi/MSK_CLOUDS_B00.gml','metadata.xml'].

    Returns:
        list: 当前数据的下载列表.

    """
    baseSentinelURL = 'http://sentinel-s2-l1c.s3.amazonaws.com/tiles/'

    dateList = date.split('-')
    baseURL_oneTile = (baseSentinelURL + tile[:2] + '/'+tile[2:3] + '/' +
                       tile[3:] + '/' + date[:4] + '/' +
                       str(int(dateList[1])) + '/' + str(int(dateList[2])) +
                       '/' + index + '/')

    urls_neededBands = [baseURL_oneTile + band + '.jp2' for band in bands]
    urls_neededMetas = [baseURL_oneTile + meta for meta in metafiles]

    return urls_neededBands + urls_neededMetas


@DeprecationWarning
def downloadOneSentinel2FromAWS(tile, date, outDir,
                                bands=['B02', 'B03', 'B04', 'B08'],
                                index='0', part=20):
    """采用分包的方法从亚马逊云下载一景哨兵2数据，元数据列表固定.

    Args:
        tile (str): 需要下载数据的tile,MGRS格式，如'49QCE'.
        date (str): 需要下载数据的日期，ISO 8601格式,如'2018-06-02'.
        outDir (str): 数据存放目录.
        index (int): 索引，一般'0'或'1'.
        bands (list): 下载波段列表，['B01','B02','B03','B04','B05','B06',
                  'B07','B08','B8A','B09','B10','B11','B12'].
        part (int): 分包数.

    """
    if not os.path.exists(outDir):
        os.mkdir(outDir)

    metafiles = ['tileInfo.json', 'preview.jpg',
                 'qi/MSK_CLOUDS_B00.gml', 'metadata.xml']
    urlist = getFullAWSDownloadList_S2_oneTile(
        tile, date, index, bands, metafiles)

    print(urlist)
    out = [os.path.join(outDir, url.split('/')[-1]) for url in urlist]

    for i in range(len(out)):
        if os.path.exists(out[i]):
            print('已经存在')
            continue
        print(urlist[i])
        download_one_by_urllib_using_range_header(urlist[i], out[i], part)


@DeprecationWarning
def main_sentinel2():
    """哨兵2数据下载主程序,采用多线程方式下载多个数据
    下载tile及date从输入csv文件中解析，csv结构：
        S2_C75_T49QCD_D20180602,49QCD,20180602
        S2_C15_T49QCD_D20180617,49QCD,20180617

    """
    bands = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06',
             'B07', 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12']
    paraDir = r'F:\SENTINEL\鄂尔多斯\download'
    downlistcsvfile = r'F:\SENTINEL\鄂尔多斯\鄂前旗2017_48SXH.csv'
    index = '0'
    downlistcsvfile = r'F:\SENTINEL\广东0801_down.csv'
    paraDir = r'F:\SENTINEL\download\down0801'
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        for oneline in read_csv(downlistcsvfile, delimiter=','):
            tile = oneline[1]
            date = oneline[2]
            isodate = date[:4] + '-' + date[4:6] + '-' + date[-2:]
            outdirname = 'S2_' + tile + '_' + date+'_'+index
            print(tile, date)
            outDir = os.path.join(paraDir, outdirname)
            executor.submit(downloadOneSentinel2FromAWS,
                            tile, isodate, outDir, bands)


if __name__ == '__main__':
    tileFile = r'F:\SENTINEL\广东\SENTINEL_MGRS.txt'
    data_folder = r'F:\SENTINEL\缩略图20190121_0214'
    thumbsDir = r"F:\SENTINEL\缩略图20190121_0214\整理"
    # 整理后需要下载的TILE编号及日期
    downlistcsvfile = "F:/SENTINEL/s2down_20190121_0214.csv"
    # 下载列表 https://username@password@download_url
    downlist = 'F:/SENTINEL/downlist_0121_0214_de.downlist'

    # peps_downer = Downloader_peps("49QCD", "2018-11-22")
    # peps_downer.down("d:/temp")

    # scihub_d = Downloader_scihub("49QCC", "2018-12-04")
    # scihub_d.query_by_name()
    # scihub_d.down('d:/temp')

    # code_downer = Downloader_code_de("49QFC", "2018-11-16")
    # code_downer.down("d:/temp")
    # print(code_downer.getDownloadUrl_with_user_pass())

    import warnings
    warnings.filterwarnings("ignore")

    # 1. 下载缩略图
    with open(tileFile) as file:
        t = file.readlines()

    # tiles = [l[:5] for l in t]
    tiles = ['47QLG']
    # time 格式 '2018-01-01'
    from time_utils import get_dates_in_range
    times = get_dates_in_range('2019-03-02', '2019-03-24')
    meta_downer = Downloader_metas_usgs_ee(tiles, times)
    # print(meta_downer)
    # search_result = meta_downer.get_download_url()
    # for i in search_result:
    #     t, acquisitionDate, url = i
    #     print(i)

    meta_downer.download_preview(r'D:\temp')

    # meta_downer = Downloader_metas_sentinel_hub(tiles, times,
    #                                             data_folder, aws_index=1)
    # meta_downer.down()
    # meta_downer.down_multiThread()
    import glob
    # 2. 整理缩略图
    jsons = glob.glob(os.path.join(data_folder + "/*/tileInfo.json"))
    # process_preview_image(thumbsDir,jsons)
    # 3. 根据缩略图生成需要下载的日期及分块列表
    # download_tile_date_to_csv(thumbsDir, downlistcsvfile)

    # 4. 生成下载列表
    # if os.path.exists(downlist):
    #     os.remove(downlist)
    # c = 1
    # print("生成下载列表")
    # for oneline in read_csv(downlistcsvfile, delimiter=','):

    #     tile = oneline[1]

    #     date = oneline[2]
    #     isodate = date[:4] + '-' + date[4:6] + '-' + date[-2:]
    #     print(c)
    #     # for peps
    #     # peps_downer = Downloader_peps(tile, isodate)
    #     # urlforXL = peps_downer.getDownloadUrlForXL()

    #     # for code-de
    #     code_downer = Downloader_code_de(tile, isodate)
    #     urlforXL = code_downer.getDownloadUrl_with_user_pass()

    #     c += 1
    #     print(urlforXL)
    #     if urlforXL is not None:
    #         for url in urlforXL:
    #             with open(downlist, 'a') as dl:
    #                 dl.write(url + '\n')
