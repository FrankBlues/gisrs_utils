# -*- coding: utf-8 -*-
import os
import sys
import shutil
import glob
import concurrent.futures
import base64
import tempfile
import getpass
import download_utils
import time_utils
import io_utils

try:
    from sentinelhub import AwsTileRequest  # , AwsProductRequest,  AwsTile
    from sentinelhub.download import AwsDownloadFailedException
except ImportError:
    print("sentinelhub not used any more.")
    pass


def process_preview_image(thumbsDir, jsons, max_cloud_cover=100):
    """根据元数据（json）整理下载的缩略图，按照云量给文件命名输出到指定目录
    """

    if not os.path.exists(thumbsDir):
        os.mkdir(thumbsDir)

    for j in jsons:
        print(j)

        try:
            jsonf = io_utils.read_json(j)
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
    """将挑选出来的需要下载数据的编号及日期写入CSV文件
    """
    if os.path.exists(out_csv):
        os.remove(out_csv)
    for l in os.listdir(thumbsDir):
        line = [l[:-4], l[8:13], l[15:23]]
        io_utils.write_csv_list(out_csv, line, True, ',')


class Downloader_metas:
    """通过sentinel-hub下载缩略图及元文件

    Usage:
        meta_downer = Downloader_metas(tiles,times,data_folder)
        meta_downer.down() 
        or
        meta_downer.down_multiThread()
    """
    meta_url = 'https://roda.sentinel-hub.com/sentinel-s2-l1c/tiles/'

    def __init__(self, tiles, times, out_dir, aws_index=0, metafiles=['tileInfo.json', 'preview.jpg'], works=2):
        self.tiles = tiles
        self.times = times
        self.out_dir = out_dir
        self.aws_index = aws_index
        self.metafiles = metafiles
        self.works = works

    def down(self):
        if not os.path.exists(data_folder):
            os.mkdir(data_folder)

        for tile in self.tiles:
            for date in self.times:
                ymd = date.split('-')
                for meta in self.metafiles:
                    url = self.meta_url + tile[:2] + '/' + tile[2:3] + '/' + tile[3:] + '/' + ymd[0] + '/' + \
                        str(int(ymd[1])) + '/' + str(int(ymd[2])) + \
                        '/' + str(self.aws_index) + '/' + meta
                    out = os.path.join(data_folder, tile + ',' +
                                       date + ',' + str(self.aws_index), meta)

                    if os.path.exists(out):
                        print('已经下载')
                        continue
                    else:

                        try:
                            download_utils.downloadOneByRequestsBasic(url, out)
                            # downloadOneByUrllibBasic(url,out)
                        except Exception as e:

                            print(url)
                            print(e)
                            continue

    def down_multiThread(self):
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.works) as executor:
            executor.submit(self.down)


class Downloader_peps:
    """ 利用哨兵数据法国镜像（PEPS），查询、下载（2018年12月后不可用）哨兵2数据
    其中： 查询基于哨兵2数据MGRD 100km网格编号及日期查询当日数据
    
    """

    base_query_url = 'https://peps.cnes.fr/resto/api/collections/S2ST/search.json?'
    search_result_json = os.path.join(tempfile.gettempdir(), 'out.json')
    username = None
    password = None

    def __init__(self, tile, date,max_cloud_cover = 100):
        """ initial with tile id, date, max cloud coverage
        params:
            tile: str MGRD id , like '49QCC'
            date: str date (iso format ), like '2018-11-11'
            max_cloud_cover : float max cloud coverage
        """
        self.tile = tile
        if len(date) != 10:
            raise ValueError("date format must like '2018-11-11'")
        self.date = date
        self.max_cloud_cover = max_cloud_cover

    @classmethod
    def set_username_and_pass(cls, username='dream15320@gmail.com'):
        """设置用户名密码，密码输入时不用明文显示
        """
        cls.username = username
        print("your username is {}\n".format(username))
        cls.password = getpass.getpass(
            prompt="Password (will not be displayed): ")

    @staticmethod
    def get_query_url_S2ST(tile, date):
        """根据网格编号和日期构建数据查询URL
        """
        in_date = time_utils.iso_to_datetime(date)
        onehour = time_utils.one_hour()
        # 北京时间
        startDate = time_utils.datetime_to_iso(
            (in_date - 8*onehour), only_date=False)
        completionDate = time_utils.datetime_to_iso(
            (in_date + 15*onehour), only_date=False)

        search_condition = 'completionDate={0}&&startDate={1}&tileid={2}'.format(
            completionDate, startDate, tile)

        return Downloader_peps.base_query_url + search_condition

    @staticmethod
    def queryOneTile(tile, date, search_result_json):
        """查询数据（返回json结果）
        """
        user_agent = "Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/67.0.3396.99 Safari/537.36"

        headers = {"User-Agent": "{0}".format(user_agent)}

        search_url = Downloader_peps.get_query_url_S2ST(tile, date)
        print("query url: {}\n".format(search_url))
        request = download_utils.constructRequest(search_url, headers=headers)
        download_utils.downloadOneByUrllibBasic(request, search_result_json)

        return search_result_json

    @staticmethod
    def parse_catalog(search_json_file):
        """解析查询后的json数据，返回下载地址
        params : 
            search_result_json: json file the query result
        return:
            tuple(dict, dict)
            download url dict and cloud coverage dict
            dict key is the product name
        """
        data = io_utils.read_json(search_json_file)

        if 'ErrorCode' in data:
            print(data['ErrorMessage'])
            sys.exit(-2)

        # Sort data
        #download_dict = {}
        #storage_dict = {}
        url_dict = {}
        cloudC_dict = {}
        if len(data["features"]) > 0:
            for i in range(len(data["features"])):
                prod = data["features"][i]["properties"]["productIdentifier"]
                #print(prod, data["features"][i]["properties"]["storage"]["mode"])
                #feature_id = data["features"][i]["id"]
                try:
                    #storage = data["features"][i]["properties"]["storage"]["mode"]
                    #platform  =data["features"][i]["properties"]["platform"]
                    # recup du numero d'orbite
                    # orbitN=data["features"][i]["properties"]["orbitNumber"]
                    cloudCover = data["features"][i]["properties"]["cloudCover"]
                    down_url = data["features"][i]["properties"]["services"]["download"]["url"]

                    #download_dict[prod] = feature_id
                    #storage_dict[prod] = storage
                    url_dict[prod] = down_url
                    cloudC_dict[prod] = cloudCover

                except:
                    pass
        else:
            print(">>> no product corresponds to selection criteria")
            # sys.exit(-1)
        return (url_dict,cloudC_dict)
    
    @staticmethod
    def construct_request(url,user,passw,**kargs):
        """ 构建请求 添加Authorization 和 User-Agent 请求头
        """
        
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
        user_agent = "Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/69.0.3497.81 Safari/537.36"

        headers = {"User-Agent": "{}".format(user_agent)}
        headers.update({
            "Authorization": "Basic {0}".format(user_pass)
        })
        
        if kargs:
            headers.update(kargs)
        return download_utils.constructRequest(url, headers=headers)
        
        
    def down(self, outdir):
        """下载
        """
        cloudC_dict = self.getCloudCover()
        if cloudC_dict is not None:
            for prod in cloudC_dict:
                if float(cloudC_dict[prod]) > self.max_cloud_cover:
                    print("云量大于{} \n".format(self.max_cloud_cover))
                    return
                else:
                    #download
                    if not os.path.isdir(outdir):
                        print("目标目录不存在，创建")
                        os.mkdir(outdir)

                    url_dict = self.getDownloadUrl()
                    url = url_dict[prod]
                    print(" download url: {} \n".format(url))
                    if Downloader_peps.username is None or Downloader_peps.password is None:
                        self.set_username_and_pass()
                        
                    request = self.construct_request(url,Downloader_peps.username,Downloader_peps.password)

                    download_utils.downloadOneByUrllibBasic(
                        request, os.path.join(outdir, prod, prod + '.zip'))
                    #download_utils.downloadOneByUrllibUsingRangeHeader(request, os.path.join(outdir, prod, '.zip'),50)

    def query(self):
        """查询，以字典类型返回查询到的下载地址，云覆盖
        """
        out_json = self.queryOneTile(
            self.tile, self.date, Downloader_peps.search_result_json)
        return self.parse_catalog(out_json)


    def getDownloadUrl(self):
        """获取解析后的URL列表
        """
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
         https://username@password@download_url
        """
        if Downloader_peps.username is None or Downloader_peps.password is None:
            self.set_username_and_pass()

        url_dict = self.getDownloadUrl()
        return [url_dict[key][:8] + Downloader_peps.username + ':' + Downloader_peps.password + '@' + url_dict[key][8:] for key in url_dict]

class Downloader_scihub(Downloader_peps):
    """ 在PEPS查询基础上，根据名称查询获取scihub下载地址，并下载
    """
    scihub_query_baseUrl = "https://scihub.copernicus.eu/dhus/search?q="
    scihub_user = "mellem"
    scihub_pass = "302112aa"
    
    def __init__(self, tile, date,max_cloud_cover = 100):
        """ initial same like class Downloader_peps 
        """
        Downloader_peps.__init__(self, tile,  date,max_cloud_cover = 100)
    
    def query_by_name(self):
        """ 简单地根据产品名称查询哨兵2数据，查询结果以xml格式存储
        """
        queryfiles = []
        peps_down_url_dict = self.getDownloadUrl()
        
        if peps_down_url_dict is not None:
            cc = 0
            for prod in peps_down_url_dict:
                query_url = self.scihub_query_baseUrl + "filename:{}*".format(prod)
                #wget --no-check-certificate --user=scihub_user --password=scihub_pass --output-document=query_results.txt query_url
                query_request = self.construct_request(query_url,self.scihub_user,self.scihub_pass)
                out_xml = os.path.join(tempfile.gettempdir(), 'query'+ str(cc) +'.xml')
                cc += 1
                download_utils.downloadOneByUrllibBasic(query_request, out_xml)
                queryfiles.append(out_xml)
                
            return queryfiles
        else:
            return None
    
    @staticmethod           
    def parse_query_result(xml = "d:/query.xml"):
        """ 解析查询到的结果（xml 文件） ，返回数据产品名称、下载地址、云量
        """
        tree = io_utils.read_xml(xml)
        root = tree.getroot()
        
        basename = '{http://www.w3.org/2005/Atom}'
        
        entrys = root.findall("./{}entry".format(basename))
        
        if len(entrys) == 1:
            for entry in entrys:
                #product name
                prod_name = entry.find("./{}title".format(basename)).text
                print(prod_name)
                
                #download link
                down_link = entry.find("./{}link".format(basename)).attrib['href']
                
                for ii in entry.findall("./{}double".format(basename)):
                    if ii.attrib['name'] == 'cloudcoverpercentage':
                        cloud_cover = float(ii.text)
            return (prod_name,down_link[:-6] + '\\' + down_link[-6:],cloud_cover)

        else:
            print("not found or find more than 1.")
            return None
    
    def down(self,outdir):
        """ 利用wget实现命令行方式下载，在linux下测试可用
        """
        
        queryfiles = self.query_by_name()
        if queryfiles is not None:
            for queryfile in queryfiles:
                prod_name,down_url,cloud_cover = self.parse_query_result(queryfile)
                if cloud_cover < self.max_cloud_cover:
                    print( "download url : {} \n".format(down_url))
                    #linux
                    #--content-disposition 采用原数据文件名，不用在指定输出名称（-O）
                    cmd = 'wget --content-disposition --continue --user={0} --password={1} -P {2} "{3}"'.format(self.scihub_user,self.scihub_pass, outdir,down_url)
                    print(cmd)
                    os.system(cmd)

class Downloader_code_de(Downloader_peps):
    """ 在PEPS查询基础上，根据名称获取哨兵数据德国镜像code-de.org下载地址，并下载
        目前测试(20181206)下载有速度限制，较慢
    """
    code_de_download_baseUrl = "https://code-de.org/download/"
    
    code_de_user = "menglimeng"
    code_de_pass = "2kct84rm4Dz57iG"
    
    def __init__(self, tile, date,max_cloud_cover = 100):
        """ initial same like class Downloader_peps 
        """
        Downloader_peps.__init__(self, tile,  date,max_cloud_cover = 100)
    
    def getDownloadUrls_code_de(self):
        """ get downloadurl
        https://code-de.org/download/S2A_MSIL1C_20181129T031051_N0207_R075_T49QEE_20181129T060514.SAFE.zip
        """
        cloud_dict = self.getCloudCover()
        if cloud_dict is None:
            return None
        
        urllist = []
        for prod in cloud_dict:
            cloud_cover = float(cloud_dict[prod])
            if cloud_cover < self.max_cloud_cover:
                urllist.append("{0}{1}.SAFE.zip".format(self.code_de_download_baseUrl,prod))
            else:
                continue
            
        if len(urllist) >=1:
            return urllist
        else:
            return None
    
    def getDownloadUrl_with_user_pass(self):
        """get downloadurl forms lick "https://username@password@download_url""
        """
        
        urls = self.getDownloadUrls_code_de()
        if urls is not None:
            return ["{0}{1}:{2}@{3}".format("https://", self.code_de_user,
                    self.code_de_pass,url[8:]) for url in urls]
        return None

    def down(self,outDir):
        """ download
        """
        
        dlist = self.getDownloadUrls_code_de()
        if dlist is not None:
            for url in dlist:
                print( "download url: {} \n".format(url))
                request = self.construct_request(url,self.code_de_user,self.code_de_pass)
                download_utils.downloadOneByUrllibBasic(request,os.path.join(outDir,url.split('/')[-1]))

@DeprecationWarning
def getFullAWSDownloadList_S2_oneTile(tile, date, index='0', bands=[], metafiles=[]):
    """构建需要下载文件(单景)的下载列表(亚马逊云)，包含需要下载的波段和元数据
    参数：
        tile:需要下载数据的tile,MGRS格式，如'49QCE'
        date:需要下载数据的日期，ISO 8601格式,如'2018-06-02'
        index:索引，一般'0'或'1'
        bands:下载波段列表，['B01','B02','B03','B04','B05','B06','B07','B08','B8A','B09','B10','B11','B12']
        metafiles:下载元数据列表，['tileInfo.json', 'preview.jpg','qi/MSK_CLOUDS_B00.gml','metadata.xml']
    返回：
        当前数据的下载列表
    """

    baseSentinelURL = 'http://sentinel-s2-l1c.s3.amazonaws.com/tiles/'

    dateList = date.split('-')
    baseURL_oneTile = baseSentinelURL + tile[:2] + '/'+tile[2:3] + '/' + tile[3:] + '/' + date[:4] + '/' + \
        str(int(dateList[1])) + '/' + str(int(dateList[2])) + '/' + index + '/'

    urls_neededBands = [baseURL_oneTile + band + '.jp2' for band in bands]
    urls_neededMetas = [baseURL_oneTile + meta for meta in metafiles]

    return urls_neededBands + urls_neededMetas

@DeprecationWarning
def downloadOneSentinel2FromAWS(tile, date, outDir, bands=['B02', 'B03', 'B04', 'B08'], index='0', part=20):
    """采用分包的方法从亚马逊云下载一景哨兵2数据，元数据列表固定
    参数：
        tile:需要下载数据的tile,MGRS格式，如'49QCE'
        date:需要下载数据的日期，ISO 8601格式,如'2018-06-02'
        outDir：数据存放目录
        index:索引，一般'0'或'1'
        bands:下载波段列表，['B01','B02','B03','B04','B05','B06','B07','B08','B8A','B09','B10','B11','B12']
        part:分包数

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
        download_utils.downloadOneByUrllibUsingRangeHeader(urlist[i], out[i], part)

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
        for oneline in io_utils.read_csv(downlistcsvfile, delimiter=','):
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
    data_folder = r'F:\SENTINEL\缩略图201811_'
    thumbsDir = r"F:\SENTINEL\缩略图201811_\整理"
    # 整理后需要下载的TILE编号及日期
    downlistcsvfile = "d:/s2down.csv"
    # 下载列表 https://username@password@download_url
    downlist = 'd:/downlist_1116_1130_de.downlist'
    
    # peps_downer = Downloader_peps("49QCD", "2018-11-22")
    # peps_downer.down("d:/temp")
    
    #scihub_d = Downloader_scihub("49QCC", "2018-12-04")
    #scihub_d.query_by_name()
    #scihub_d.down('d:/temp')
    
    #code_downer = Downloader_code_de("49QCC", "2018-12-04")
    #code_downer.down("d:/temp")
    #print(code_downer.getDownloadUrl_with_user_pass())
    

    # 1. 下载缩略图
    with open(tileFile) as file:
        t = file.readlines()

    tiles = [l[:5] for l in t]
    #tiles = ['49QGC','49QFC','49QEC']
    # time 格式 '2018-01-01'
    times = time_utils.get_dates_in_range('2018-11-16', '2018-11-16')
    # meta_downer = Downloader_metas(tiles, times, data_folder)
    # meta_downer.down_multiThread()

    # 2. 整理缩略图
    #jsons = glob.glob(os.path.join(data_folder + "/*/tileInfo.json"))
    # process_preview_image(thumbsDir,jsons)
    # 3. 根据缩略图生成需要下载的日期及分块列表
    #download_tile_date_to_csv(thumbsDir, "d:/s2down.csv")

    # 4. 生成下载列表
    if os.path.exists(downlist):
        os.remove(downlist)
    c = 1
    print("生成下载列表")
    for oneline in io_utils.read_csv(downlistcsvfile, delimiter=','):

        tile = oneline[1]

        date = oneline[2]
        isodate = date[:4] + '-' + date[4:6] + '-' + date[-2:]
        print(c)
        # for peps
        #peps_downer = Downloader_peps(tile, isodate)
        #urlforXL = peps_downer.getDownloadUrlForXL()
        
        # for code-de 
        code_downer = Downloader_code_de(tile, isodate)
        urlforXL = code_downer.getDownloadUrl_with_user_pass()

        c += 1
        print(urlforXL)
        if urlforXL is not None:
            for url in urlforXL:
                with open(downlist, 'a') as dl:
                    dl.write(url + '\n')



