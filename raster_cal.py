# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 10:10:19 2022

@author: DELL
"""
import os
import numpy as np

import rasterio


def log10():
  return np.log10


def cal(img, fun, out):
  with rasterio.open(img) as src:
    kargs = src.meta.copy()
    arr = src.read()
    result = fun(arr).astype(kargs.get('dtype'))
    with rasterio.open(out, 'w', **kargs) as dst:
      result[np.logical_or(np.isnan(result), np.isinf(result))] = 0

      dst.write(result)
    
if __name__ == '__main__':
  
  img = r'E:\S1\S1A_IW_GRDH_1SDV_20210722T101333_20210722T101358_038889_0496BE_2C03_Cal_TF_TC.data\Gamma0_VV.img'
  out = img.replace('.img', '_log10.img')
  cal(img, log10(), out)