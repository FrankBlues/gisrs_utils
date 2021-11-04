# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 16:04:50 2018

@author: Administrator
"""

import math
import numpy as np
import rasterio

import time


def dist(n):
    """Based on IDL dist function. Returns a rectangular array in which the
    value of each element is proportional to its frequency. This array may
    be used for a variety of purposes, including frequency-domain filtering.

    Args:
        n (int): Dimensions.

    Returns:
        numpy ndarray: Dist array.

    Examples:
        >>> dist(3)
        array([[ 0.        ,  1.        ,  1.        ],
               [ 1.        ,  1.41421356,  1.41421356],
               [ 1.        ,  1.41421356,  1.41421356]])
        >>> dist(4)
        array([[ 0.        ,  1.        ,  2.        ,  1.        ],
               [ 1.        ,  1.41421356,  2.23606798,  1.41421356],
               [ 2.        ,  2.23606798,  2.82842712,  2.23606798],
               [ 1.        ,  1.41421356,  2.23606798,  1.41421356]])

    """
    x = np.arange(n * 2).reshape(2, n)
    x[1] = n - x[0]

    x = (np.amin(x, axis=0)) ** 2
    a = np.empty((n, n))

    for i in range(math.floor(n/2) + 1):
        y = np.sqrt(x + i**2.)
        a[i] = y
        if i != 0:
            a[n - i] = y

    return a


def structElement_circle(radius):
    """ 构建圆形结构要素，用于形态学计算（侵蚀、膨胀、开闭运算等）.

    Args:
        radius (int): Radius of the round element, in pixel.

    Returns:
        numpy ndarray: Element filled with 1, in uint8 format.

    Examples:
        >>> structElement_circle(2)
        [[0 0 1 0 0]
         [0 1 1 1 0]
         [1 1 1 1 1]
         [0 1 1 1 0]
         [0 0 1 0 0]]

    """
    d = dist(2 * radius + 1)
    # r = np.roll(np.roll(d,radius,axis=1),radius,axis=0)
    r = np.roll(d, radius, axis=(0, 1))
    return ((r <= radius) * 1).astype('uint8')


class timer(object):
    """A timer class.

    Examples:
        >>> with timer():
        ...     your code

    """
    def __enter__(self):
        self.start = time.time()

    def __exit__(self, type, value, traceback):
        self.end = time.time()
        print('Time spent : {0:.2f} seconds'.format((self.end - self.start)))


def dn2date(dn):
    """Convert day of a year to the date.

    Args:
        dn (str or int): Day of year with the format yyyyddd, i.e 2018093

    Returns:
        str: The date with the format YYYYMMDD.

    """
    idays_nonleap = [0, 31, 59, 90, 120, 151, 181, 212,
                     243, 273, 304, 334, 365]
    idays_leap = [0, 31, 60, 91, 121, 152, 182, 213, 244,
                  274, 305, 335, 366]

    daystr = str(dn)
    yr = int(daystr[:4])

    if (yr % 4 == 0 and yr % 100 != 0) or yr % 400 == 0:
        idays = idays_leap
    else:
        idays = idays_nonleap

    idx = 0
    d = 0
    for i in idays:
        idx += 1

        day = int(daystr[4:])

        if day <= idays[idx]:
            d = day - idays[idx-1]
            break

    m = "{0:02d}".format(idx)  # m = "0" + str(idx) if idx < 10 else str(idx)
    d = "{0:02d}".format(d)  # d = "0" + str(d) if d < 10 else str(d)
    print("date:" + str(yr) + m + d)
    return str(yr) + m + d


class RasterCreate(object):

    def __init__(self, cols=10, rows=10, count=1, dtype='uint8'):
        self.cols = cols
        self.rows = rows
        self.count = count
        self.dtype = dtype

    def range_raster(self, out_raster):
        data = np.arange(self.cols*self.rows).reshape(self.rows, self.cols).astype(self.dtype)
        kargs = {
            "height": self.rows,
            "width": self.cols,
            "count": self.count,
            "driver": "GTiff",
            "dtype": self.dtype,
            }
        with rasterio.open(out_raster, 'w', **kargs) as dst:
            for i in range(self.count):
                dst.write(data, i+1)


if __name__ == '__main__':
    print(structElement_circle(2))
    
    rc = RasterCreate()
    rc.range_raster("d:/temp11/range100.tif")
