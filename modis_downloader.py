# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 11:07:19 2018

Query and Download modis data from NASA website:
https://ladsweb.modaps.eosdis.nasa.gov
@author: Administrator
"""

import os
import concurrent
import re
from download_utils import open_request
from download_utils import download_one_by_urllib_using_range_header

BASEURL = 'https://ladsweb.modaps.eosdis.nasa.gov'


def urllist_to_txt(search_result, downlist_file, checkExist=True):
    """ Write list to txt file.

    Args:
        search_result (list): List of urls.
        downlist_file (str): Target text file.
        checkExist (bool): Whether or not to check the target file existing.

    """
    # if checkExist:
    #     if os.path.exists(downlist_file):
    #         os.remove(downlist_file)
    with open(downlist_file, 'a') as f:
        for u in search_result:
            f.write(BASEURL + u + '\n')


def gen_url_list_for_MODIS_MYD04_3K(year, dn, downlist_file):
    """Query MYD04_03 data, generate url list and write out to file.

    Args:
        year (int or str): Year with format('YYYY').
        dn (int or str): Day of year.
        downlist_file (str): File where url list write to.

    """
    # query
    url = ('https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/61/MYD04'
           '_3K/{year}/{dn}/?process=ftpAsHttp&path=allData%2f61%2fMYD04_3K'
           '%2f{year}%2f{dn}'.format(year=str(year), dn=str(dn)))

    p = re.compile(('/archive/allData/61/MYD04_3K/{year}/{dn}/MYD04_3K.A'
                    '{year}{dn}'.format(year=str(year), dn=str(dn)) +
                    '.{24}hdf'))
    res_content = open_request(url, retries=5)
    search_result = p.findall(str(res_content.read()))
    urllist_to_txt(search_result, downlist_file)


def gen_url_list_for_MODIS_11A1(year, dn, downlist_file, myd=False):
    """Query MODIS_11A1 data,generate url list and write out.

    Args:
        year (int or str): Year with format('YYYY').
        dn (int or str): Day of year.
        downlist_file (str): File where url list write to.
        myd (bool): Whether or not to download MYD11A1 data,
            defalut False, meaning query MOD11A1 data.

    Usage:
        for i in range(305, 318):
            MODIS_11A1(2018, i, "d:/myd11a1.downlist", myd=True)

    """
    url = ('https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/6/MOD11A'
           '1/{year}/{dn}/?process=ftpAsHttp&path=allData%2f6%2fMOD11A1%2f{'
           'year}%2f{dn}'.format(year=str(year), dn=str(dn)))
    if myd:
        url = url.replace("MOD11A1", "MYD11A1")

    match_url_part1 = ('/archive/allData/6/MOD11A1/{year}/{dn}/MOD11A1.A{year'
                       '}{dn}.'.format(year=str(year), dn=str(dn)))
    if myd:
        match_url_part1 = match_url_part1.replace("MOD11A1", "MYD11A1")
    p = re.compile(match_url_part1 + 'h2[6-7]v04.{19}hdf')
    res_content = open_request(url, retries=5)
    search_result = p.findall(str(res_content))
    urllist_to_txt(search_result, downlist_file)


def down_from_list(downlist, out_dir):
    """Download file from downlist.

    Args:
        downlist (str): File contain the url list.
        out_dir (str): Directory where to download to.

    Usage:
        down_from_list("d:/mod11a1.downlist","I:/MODIS/MOD11A1")

    """

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        with open(downlist) as d:
            dlist = d.readlines()

        for d in dlist:
            url = d.rstrip('\n')
            fn = url.split('/')[-1]
            print(fn)
            out = os.path.join(out_dir, fn)
            if not os.path.exists(out):
                executor.submit(download_one_by_urllib_using_range_header,
                                url, out, 30)


if __name__ == '__main__':
    for i in range(308, 362):
        # name = 'd:/myd04_2018{}.txt'.format(i)
        name = 'd:/myd04_2018308_362.txt'.format(i)
        gen_url_list_for_MODIS_MYD04_3K(2018, i, name)
    # down_from_list("d:/myd04_2018275.txt",r'H:\MODIS\MYD04_3K\2018275')

    datelist = ['2017.01.01', '2017.01.17', '2017.02.02', '2017.02.18',
                '2017.03.06', '2017.03.22', '2017.04.07', '2017.04.23',
                '2017.05.09', '2017.05.25', '2017.06.10', '2017.06.26',
                '2017.07.12', '2017.07.28', '2017.08.13', '2017.08.29',
                '2017.09.14', '2017.09.30', '2017.10.16', '2017.11.01',
                '2017.11.17', '2017.12.03', '2017.12.19']

    # AQUA ndvi 250m
    modurl = 'https://e4ftl01.cr.usgs.gov/MOLA/MYD13Q1.006/'
    datelist = ['2017.01.09', '2017.01.25', '2017.02.10', '2017.02.26',
                '2017.03.14', '2017.03.30', '2017.04.15', '2017.05.01',
                '2017.05.17', '2017.06.02', '2017.06.18', '2017.07.04',
                '2017.07.20', '2017.08.05', '2017.08.21', '2017.09.06',
                '2017.09.22', '2017.10.08', '2017.10.24', '2017.11.09',
                '2017.11.25', '2017.12.11', '2017.12.27', '2018.01.09',
                '2018.01.25', '2018.02.10', '2018.02.26', '2018.03.14',
                '2018.03.30', '2018.04.15', '2018.05.01', '2018.05.17',
                '2018.06.02', '2018.06.18', '2018.07.04', '2018.07.20',
                '2018.08.05', ]

    # url = ('https://e4ftl01.cr.usgs.gov/VIIRS/VNP09GA.001/2018.11.25/VNP09'
    #        'GA.A2018329.h28v06.001.2018330095339.h5')
    # out = 'd:/VNP09GA.A2018329.h28v06.001.2018330095339.h5'
    # from download_utils import construct_request_basic_auth
    # request = construct_request_basic_auth(url,'menglimeng','302112aA')
    # download_one_by_urllib_using_range_header(request, out, 15)
