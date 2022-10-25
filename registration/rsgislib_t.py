# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 17:03:49 2022

@author: DELL
"""
import rsgislib
from rsgislib import imageregistration
reference = r'D:\temp11\DOM\resample\refdom.tif'
floating = r'D:\temp11\DATA\resample\GF2_PMS1_E110.0_N34.0_20200727_L1A0004953380-PAN11.tif'
pixelGap = 600
threshold = 0.4
window = 50
search = 70
stddevRef = 2
stddevFloat = 2
subpixelresolution = 4
metric = imageregistration.METRIC_CORELATION
outputType = imageregistration.TYPE_RSGIS_IMG2MAP
output = r'D:\temp11\gcp555.txt'
# imageregistration.basic_registration(reference, floating, output, pixelGap, 
#                                       threshold, window, search, stddevRef, 
#                                       stddevFloat, subpixelresolution, metric, 
#                                       outputType)


# dist_threshold = 90
# max_n_iters = 10
# move_chng_thres = 90
# p_smooth = 2
# imageregistration.single_layer_registration(reference, floating, output, pixelGap,
#                                             threshold, window, search, stddevRef, 
#                                             stddevFloat, subpixelresolution, 
#                                             dist_threshold, max_n_iters, move_chng_thres,
#                                             p_smooth, metric, outputType)


imageregistration.gcp_to_gdal(floating, output, 'd:/aaaaa.tif', 'GTiff', rsgislib.TYPE_16UINT)




