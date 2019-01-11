# -*- coding: utf-8 -*-

import os
import concurrent.futures

import base64

import requests
try:

    # Python 2.x Libs
    from urllib2 import build_opener, install_opener, Request, urlopen, HTTPError
    from urllib2 import URLError, HTTPSHandler,  HTTPHandler, HTTPCookieProcessor

    from cookielib import MozillaCookieJar
    from StringIO import StringIO

except ImportError as e:

    # Python 3.x Libs
    from urllib.request import build_opener, install_opener, Request, urlopen
    from urllib.request import HTTPHandler, HTTPSHandler, HTTPCookieProcessor
    from urllib.error import HTTPError, URLError

    from http.cookiejar import MozillaCookieJar
    from io import StringIO


def validOutFile(out):
    """验证下载输出文件
    1.如果是文件目录，报错退出
    2.如果文件目录不存在则创建
    """
    if os.path.isdir(out):
        raise ValueError('outfile is a directory!')

    outdir = os.path.dirname(out)
    if len(outdir) != 0:
        if not os.path.exists(outdir):
            os.mkdir(outdir)


def downloadOneByRequestsBasic(url, out):
    """download by requests lib simplely
    """
    
    r = requests.get(url)
    if r.status_code == 200:
        validOutFile(out)
        with open(out, 'wb') as fd:
            fd.write(r.content)
            
def downloadOneByRequestsBasic_simple_auth(url, out,user,passw):
    """download by requests lib simplely
    """
    
    r = requests.get(url,auth=(user,passw))
    if r.status_code == 200:
        validOutFile(out)
        with open(out, 'wb') as fd:
            fd.write(r.content)

def downloadOneByRequestsIterChunk(url, out, chunk_size=512):
    """download by requests lib by chunk
    """
    validOutFile(out)
    r = requests.get(url, stream=True)
    with open(out, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)


def openRequest(req, retries=3):
    """利用urllib 发送请求 返回响应内容
    参数：
        req:请求内容（url地址或request对象）
        retries:重试次数

    返回：
        响应内容
    """

    try:
        return urlopen(req)
    except URLError as e:
        print(e)
        if retries > 0:
            retries -= 1
            openRequest(req, retries)
        else:
            raise e


def downloadOneByUrllibBasic(url, out):
    """测试urllib包最基础下载方法
    参数：
        url:下载地址
        out:输出文件名称
    """
    r = openRequest(url)

    validOutFile(out)
    with open(out, 'wb') as fd:
        fd.write(r.read())


def getTotalSize(response):
    """从响应中获取内容长度
    """
    try:
        file_size = response.info().getheader('Content-Length').strip()
    except AttributeError:
        try:
            file_size = response.getheader('Content-Length').strip()
        except AttributeError:
            print("> Problem getting size")
            return None

    return int(file_size)


def downloadOneByUrllibUsingRangeHeader(req, out, part=5):
    """将要下载文件根据返回内容的长度，平均分成若干份后采用异步方式下载
    缺点：需要所有分块下载完成后再写入磁盘
    参数：
        url:下载地址
        out:输出文件名称
    """
    # 获取响应内容总长度
    res = openRequest(req)
    lens = getTotalSize(res)

    # 不分包
    if part == 1 or len is None:
        downloadOneByUrllibBasic(req, out)
        return
    # 设置最小包大小为100K
    if lens/part < 100000:
        part = int(lens/100000)

    if part == 0 or part == 1:
        downloadOneByUrllibBasic(req, out)
        return
    # 分包
    startR = [0] + [int(lens/part*(i+1)) + 1 for i in range(part-1)]
    endR = [int(lens/part*(i+1)) for i in range(part-1)] + [lens]
    ranges = [(startR[i], endR[i]) for i in range(part)]

    # 请求列表
    reqs = []
    parts = {}
    for i in range(part):
        # req = urllib.request.Request(req)
        req.headers['Range'] = 'bytes={0}-{1}'.format(
            str(ranges[i][0]), str(ranges[i][1]))
        reqs.append(req)
        # 索引，写入文件时的顺序
        parts[req] = i  # 'part{:02d}'.format(i)
    # 下载
    result = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=part) as executor:
        futuresToReq = {executor.submit(
            openRequest, req): parts[req] for req in reqs}

        for future in concurrent.futures.as_completed(futuresToReq):
            index = futuresToReq[future]
            result[index] = future.result().read()
    # 写入
    validOutFile(out)
    if os.path.exists(out):
        os.remove(out)
    with open(out, 'ab') as fd:
        for i in range(part):
            fd.write(result[i])

def constructRequest(url,headers):
    """ 添加请求头，构建urllib请求
    params:
        url : str
        headers : dict request headers
    returns: urllib Request Object
    
    """
    
    return Request(url, headers=headers)

def construct_request_basic_auth(url,user,passw,**kargs):
    """ 构建包含基本认证的请求，添加Authorization 和User-Agent请求头及其它
    params:
        url : str
        user: str username
        passw: str password
        kargs: dict other headers
    returns:
        urllib Request Object
    usage:
        headers_other = {
                    "Host" : 'scihub.copernicus.eu',
                    "Upgrade-Insecure-Requests" : 1,
                    "Connection": 'keep-alive',
                    "Accept": 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8'
                    }
        construct_request(down_url,scihub_user,scihub_pass,**headers_other)
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
        
    return constructRequest(url, headers=headers)