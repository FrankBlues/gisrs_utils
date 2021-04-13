# -*- coding: utf-8 -*-
"""基于urllib及requests包的一些基础下载工具.

"""

import os
import concurrent.futures
import base64
import ssl
try:
    # Python 2.x Libs
    from urllib2 import build_opener, install_opener, Request, urlopen
    from urllib2 import HTTPError, URLError, HTTPSHandler
    from urllib2 import HTTPHandler, HTTPCookieProcessor
    from cookielib import MozillaCookieJar
    from StringIO import StringIO

except ImportError:
    # Python 3.x Libs
    from urllib.request import build_opener, install_opener, Request, urlopen
    from urllib.request import HTTPHandler, HTTPSHandler, HTTPCookieProcessor
    from urllib.error import HTTPError, URLError
    from http.cookiejar import MozillaCookieJar
    from io import StringIO


def valid_out_file(out):
    """验证下载输出文件: 1.如果是文件目录，报错退出; 2.如果文件目录不存在则创建.

    Args:
        out (str): The input file.

    Raises:
        ValueError: If input file is a directory.
    """

    if os.path.isdir(out):
        raise ValueError('Outfile is a directory!')

    out_dir = os.path.dirname(out)
    if out_dir:
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)


def download_one_by_requests_basic(url, out, verify=True):
    """Download one file by requests lib simplely.

    Args:
        url (str): The download url.
        out (str): The target file.
        verify (bool): Whether or not need SSL verify. Default True.
    """

    import requests
    r = requests.get(url, verify=verify)
    if r.status_code == 200:
        valid_out_file(out)
        with open(out, 'wb') as fd:
            fd.write(r.content)
    else:
        print(url)


def download_one_by_requests_basic_simple_auth(url, out, user, passw):
    """Download one file by requests lib with simple authorization.

    Args:
        url (str): The download url.
        out (str): The target file.
        user, passw (str): The user name and password for authorization.
    """

    import requests
    r = requests.get(url, auth=(user, passw))
    if r.status_code == 200:
        valid_out_file(out)
        with open(out, 'wb') as fd:
            fd.write(r.content)


def download_one_by_requests_iter_chunk(url, out, chunk_size=512):
    """Download one file by requests lib using chunks.

    Args:
        url (str): The download url.
        out (str): The target file.
        chunk_size (int): Size of each chunk in KB.
    """

    import requests
    valid_out_file(out)
    r = requests.get(url, stream=True)
    with open(out, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)


def open_request(req, retries=3):
    """利用urllib 发送请求 返回响应内容

    Args：
        req (str or urllib Request object):请求内容.
        retries (int): 重试次数

    Returns：
        HTTPResponse object for HTTP and HTTPS URLs.
        HTTPError code if HTTPError occurs.

    Raises:
        URLError: If SSL verify failed and retry times more than allowed.
    """

    try:
        return urlopen(req)
    except HTTPError as e:
        # print(e)
        return e.code
    except URLError as e:
        print(e)
        if 'SSL:CERTIFICATE_VERIFY_FAILED' in e:
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            return urlopen(req, context=ctx)
        # if retries > 0:
        #     retries -= 1
        #     open_request(req, retries)
        else:
            raise e


def download_one_by_urllib_basic(url, out):
    """Download one file using urllib.

    Args:
        url (str): The download url.
        out (str): The target file.
    """

    r = open_request(url)
    valid_out_file(out)
    with open(out, 'wb') as fd:
        fd.write(r.read())


def get_total_size(response):
    """从响应中获取内容长度.

    Args:
        response (Response object): 响应内容.

    Returns:
        int: Length of the response content.

    Raises:
        AttributeError: If failed to get the Content-Length header.
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


def download_one_by_urllib_using_range_header(req, out, part=5):
    """将要下载文件根据返回内容的长度，平均分成若干份后采用异步方式下载
    缺点：需要所有分块下载完成后再写入磁盘

    Args:
        req (str or urllib Request object):请求内容.
        out (str): The target file.
    """

    # 获取响应内容总长度
    res = open_request(req)
    lens = get_total_size(res)

    # 不分包
    if part == 1 or len is None:
        download_one_by_urllib_basic(req, out)
        return
    # 设置最小包大小为100K
    if lens/part < 100000:
        part = int(lens/100000)

    if part == 0 or part == 1:
        download_one_by_urllib_basic(req, out)
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
            open_request, req): parts[req] for req in reqs}

        for future in concurrent.futures.as_completed(futuresToReq):
            index = futuresToReq[future]
            result[index] = future.result().read()
    # 写入
    valid_out_file(out)
    if os.path.exists(out):
        os.remove(out)
    with open(out, 'ab') as fd:
        for i in range(part):
            fd.write(result[i])


def construct_request(url, headers):
    """添加请求头，构建urllib请求.

    Args:
        url (str) : The URL.
        headers (dict): The request headers.

    Returns:
        urllib Request Object.

    """

    return Request(url, headers=headers)


def construct_request_basic_auth(url, user, passw, **kargs):
    """ 构建包含基本认证的请求，添加Authorization 和User-Agent请求头及其它请求头.

    Args:
        url (str) : The URL.
        user, passw (str): The user name and password for authorization.
        kargs (dict): Other headers.

    Returns:
        urllib Request Object.

    Example:
        >>> headers_other = {
                    "Host" : 'scihub.copernicus.eu',
                    "Upgrade-Insecure-Requests" : 1,
                    "Connection": 'keep-alive',
                    "Accept": 'text/html,application/xhtml+xml,application/
                    xml;q=0.9,image/webp,image/apng,*/*;q=0.8'
                    }
        >>> construct_request_basic_auth(down_url,scihub_user,scihub_pass,
        **headers_other)
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
    user_agent = ("Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.3"
                  "6 (KHTML, like Gecko) Chrome/69.0.3497.81 Safari/537.36")

    headers = {"User-Agent": "{}".format(user_agent)}
    headers.update({
        "Authorization": "Basic {0}".format(user_pass)
    })

    if kargs:
        headers.update(kargs)

    return construct_request(url, headers=headers)


if __name__ == '__main__':
    
    root_uri = 'https://scihub.copernicus.eu/dhus/odata/v1/'
    
    tile = '50SMJ'
    # 50SLJ 50SMJ
    # 50SLH 50SMH
    import requests
    con = (f"Products?$format=json&"
           "$filter=year(IngestionDate) eq 2021 and "
           "month(IngestionDate) eq 4 and "
           "startswith(Name,'S2') and "
           "substringof('50SLJ',Name) and "
           "substringof('L1C',Name)&"
           "$orderby=IngestionDate desc")
    r = requests.get(root_uri + con,
                     auth=('mellem', '302112aa'))
    if r.status_code == 200:
        content = r.json()
        results = content['d']['results']
        print(f'{len(results)} found!')
        FLAG = 1
        for result in results:
            product_id = result['Id']
            name = result['Name']
            product_url = root_uri + f"Products('{product_id}')/$value" + '_' + str(FLAG)
            preview_url = root_uri + f"Products('{product_id}')/Products('Quicklook')/$value"
            print(product_url)
            r1 = requests.get(preview_url, auth=('mellem', '302112aa'))
            download_one_by_requests_basic_simple_auth(preview_url, f'e:/S2/{name}_{FLAG}.jpg', 'mellem', '302112aa')
            FLAG += 1
