# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 17:23:30 2020

@author: DELL
"""

import numpy as np


def lon2pix(lon, z):
    return (lon + 180) * (256 * 2**(-z)) / 360


def pix2lon(pix_x, z):
    return pix_x * 360 / (256 * 2**(-z)) - 180


def lat2pix(lat, z):
    siny = np.sin(np.deg2rad(lat))
    y = np.log((1 + siny)/(1-siny))
    return 128 * 2**(-z) * (1 - y/(2 * np.pi))


def pix2lat(pix_y, z):
    y = 2 * np.pi * (1 - pix_y / (128 * 2**(-z)))
    ey = np.exp(y)
    siny = (ey - 1)/(ey + 1)
    return np.rad2deg(np.arcsin(siny))


def lonlat2xy(lon, lat, z):
    n = 2 ** (z - 1)
    rad_lat = np.deg2rad(lat)
    x = n * (lon/180 + 1)
    y = n * (1 - (np.log(np.tan(rad_lat) + 1/np.cos(rad_lat))) / np.pi)
    return int(x), int(y)


if __name__ == '__main__':
    lon1, lat1 = 112.1, 38.25
    lon2, lat2 = 112.7, 37.7
    z = 10
    
    zn = False

    sx, sy = lonlat2xy(lon1, lat1, z)
    ex, ey = lonlat2xy(lon2, lat2, z)

    # print(lonlat2xy(lon, lat, z))
    # print(lon2pix(lon, z))
    # print(lat2pix(lat, z))
    # import requests
    # res = requests.get('http://mt0.google.cn/vt/lyrs=s&hl=zh-CN&gl=cn&x=1&y=0&z=1')

    for x in range(sx, ey + 1):
        for y in range(sy, ey + 1):
            base_url = 'http://mt0.google.cn/vt/lyrs=s&'
            xyz = 'x={0}&y={1}&z={2}'.format(x, y, z)
            if zn:
                url = base_url + 'hl=zh-CN&gl=cn&' + xyz
            else:
                url = base_url + xyz