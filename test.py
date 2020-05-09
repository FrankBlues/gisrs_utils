# -*- coding: utf-8 -*-

import io_utils
import os
import warnings

from image_process import cira_strech, new_green_band, bytscl
import glob
import gdal
import numpy as np
import rasterio
from affine import Affine

from netCDF4 import Dataset

#import image_process.pilImage as im
warnings.filterwarnings("ignore")
def get_raster_arr(r):
    with rasterio.open(r) as src:
        return src.read(1)


if __name__ == '__main__':
    
    r = r'E:\work\海冰\code\result\tif\SeaIce_NP_20171008_20171015_20200115151817.tif'
    e = r'D:\temp\New_Shapefile.shp'
    from mask import extract_by_mask
    from mosaic import merge_rio
    from rasterio.io import MemoryFile
#    with open('tests/data/RGB.byte.tif', 'rb') as f, MemoryFile(f) as memfile:
#    with memfile.open() as src:
#        pprint.pprint(src.profile)

    extract_by_mask(e, r, 'd:/temp/test1.tif', nodata=0)
    with rasterio.open('d:/temp/test1.tif') as src:
        kargs = src.meta.copy()
        arr = src.read(1)
        arr[arr==2] = 1
        with rasterio.open('d:/temp/test2.tif', 'w', **kargs) as dst:
            dst.write(arr, 1)
    import datetime
    nowtime = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    merge_rio([rasterio.open(f) for f in ['d:/temp/test2.tif', r]],
               r'E:\work\海冰\code\result\tif\SeaIce_NP_20171008_20171015_{}.tif'.format(nowtime),
               res=None, nodata=0)


    # outDir = 'D:/test/SENTINEL/s/'

    # r = r'D:\temp\Extract_tif12.tif'

    # r = r'D:\test\杭州\FY4\FY4A-_AGRI--_N_REGI_1047E_L1C_TCC-_MULT_GLL_20181213053834_20181213054250_1000M_V0001.JPG'

    # nc = r'D:\NC_H08_20180622_0500_R21_FLDK.06001_06001.nc'

    # r = r'D:\气溶胶\201811\raster'

    # rs = glob.glob(os.path.join(r,'*.tif'))

    # r0 = r'D:\气溶胶\201812\raster\gf5_aot_20181201.tif'
    # r = r'D:\气溶胶\201812\raster\gf5_aot_20181228.tif'


    # d10m = r'I:\sentinel\S2A_MSIL2A_20181106T025911_N0207_R032_T49QHF_20181106T060119.SAFE\GRANULE\L2A_T49QHF_A017618_20181106T030458\IMG_DATA\R10m'
    # r = os.path.join(d10m,'T49QHF_20181106T025911_B04_10m.jp2')
    # g = os.path.join(d10m,'T49QHF_20181106T025911_B03_10m.jp2')
    # b = os.path.join(d10m,'T49QHF_20181106T025911_B02_10m.jp2')
    # ir = os.path.join(d10m,'T49QHF_20181106T025911_B08_10m.jp2')

    # r = r'H:\temp\S2_20181029_T50.tif'
#    csv = 'D:/wrs2_centroid_china.csv'
#    from io_utils import read_csv, write_json
#    c = read_csv(csv,delimiter=',')[1:]
#    dic = {}
#    for pr in c:
#        dic[pr[0]] = {"centroid_lat": float(pr[2]),"centroid_lon": float(pr[1])
#                }
#    write_json('d:/wrs2_centroid_china.json', dic)
#    
#
#    r = r'D:\temp\GF5_DPC_20190202_003929_L10000011720_L2A.h5'
    # import h5py

    # f = h5py.File(r, 'r')
    # ks = list(f.keys())
    # ['Aod_fine', 'Aod_tot', 'Cloud_Indicator', 'Col', 'Latitude', 'Longitude', 'Mod', 'Row']
    # for ds in list(f.keys()):
    #     arr = f[ds][:]
    #     nodata = arr[0,0]
    #     valid = arr[arr != nodata]
    #     print("{0} valid count: {1}.".format(ds,valid.sum()))
    
    
    # aod_fine = f['Aod_fine'][:]  # -99999
    # aod_tot = f['Aod_tot'][:]  # -99999
    # Cloud_Indicator = f['Cloud_Indicator'][:]  # 65280
    # Col = f['Col'][:]  # 65280
    # Latitude = f['Latitude'][:]  # -99999
    # Longitude = f['Longitude'][:]  # -99999  
    # Mod = f['Mod'][:]  # -99999
    # Row = f['Row'][:]  # 65280

    # c = aod_tot != -99999.
    # print(c.sum())
    # aod_tot[ds == -99999.] = np.nan
    # import matplotlib.pyplot as plt
    # plt.imshow(ds)









# ================解压指定哨兵2数据===================================
#     from SAFE_process import unzip_files_in_safe_endswith
#
#     zipdir = 'I:/sentinel/'
#     zipfs = glob.glob(zipdir + 'S2*T49QEE*.zip')
#     for z in zipfs:
#         unzip_files_in_safe_endswith(z,'I:/sentinel/unzip','_B08.jp2')
# =============================================================================





# =============add compress====================================================
#     with rasterio.open(r) as src:
#         kargs = src.meta.copy()
#         kargs.update({
#                 'compress' : 'lzw'
#                 })
#         with rasterio.open(r'H:\temp\S2_20181029_T50_lzw.tif','w',**kargs) as dst:
#             dst.write(src.read())
# =============================================================================






# ==============set nodata=================================================
#     with rasterio.open(r0) as r0:
#         nodata = r0.nodata
#         with rasterio.open(r) as src:
#             kargs = src.meta.copy()
#             kargs.update({
#                     'nodata' : nodata
#                     })
#             with rasterio.open('d:/test_nodata.tif','w',**kargs) as dst:
#                 dst.write(src.read())
# =============================================================================



# ================简单处理投影拼接过程中产生的条带===================================
#     m = r'D:\test\杭州\S_NPP\VNP09GA.A2018329.h28v0506_M3_Resample_clip.tif'
#     m_destrip = r'D:\test\杭州\S_NPP\VNP09GA.A2018329.h28v0506_M3_Resample_clip_destrip.tif'
#
#     ds_m = rasterio.open(m)
#
#     height = ds_m.height
#     width = ds_m.width
#
#     arr_m = ds_m.read(1)
#
#     strip_line_idx = np.where(arr_m[:,width // 2] == -28672)
#
#     # d[547] = d[546] + (d[549] - d[546])/4
#     # d[548] = d[546] + (d[549] - d[546])/4 * 2
#
#     quads = (arr_m[549,:] - arr_m[546,:]) // 4
#
#     arr_m[547,:] = arr_m[546,:] + quads
#     arr_m[548,:] = arr_m[546,:] + quads * 2
#
#     kargs = ds_m.meta.copy()
#
#     with rasterio.open(m_destrip,'w',**kargs) as dst:
#         dst.write(arr_m,1)
#     ds_m.close()
# =============================================================================






# ============日本官方葵花8NC等经纬度数据转TIF======================================
#     f = Dataset(nc,'r')
#     albedo_04 = f.variables["albedo_04"][:]
#     geotransform = (80, 0.02, 0.0, 60, 0.0, -0.02)
#
#     kargs = {
#             'crs':'epsg:4326',
#             'transform':Affine.from_gdal(*geotransform),
#             'driver' : 'GTiFF',
#             'count':1,
#             'width':6001,
#             'height':6001,
#             'dtype': 'float32',
#             'nodata':-32768
#             }
#
#     with rasterio.open('d:/h8_nc.tif','w',**kargs) as dst:
#         dst.write(albedo_04,1)
# =============================================================================

# =======根据左上角坐标系分辨率给图片添加地理信息======================================
    #根据左上角坐标系分辨率给图片添加地理信息
    # with rasterio.open(r) as src:
    #     kargs = src.meta.copy()
    #     geotransform = (62.2, 0.01, 0.0, 54, 0.0, -0.01)
    #     kargs.update({
    #             'crs':'epsg:4326',
    #             'transform':Affine.from_gdal(*geotransform),
    #             'driver' : 'GTiFF'

    #             })
    #     with rasterio.open('d:/fy4_test.tif','w',**kargs) as dst:
    #         dst.write(src.read())
# =============================================================================

    # n = r'D:\NC_H08_20181006_0250_L2ARP021_FLDK.02401_02401.nc'
    # f = Dataset(n,'r')
    # aot = f.variables["AOT"][:].data
    # lats = f.variables["latitude"][:]
    # lons = f.variables["longitude"][:]

    # geotransform = (lons[0], 0.05, 0.0, lats[0], 0.0, -0.05)

    # aot[aot == -32768] = 0
    # #aot = aot * 2e-4

    # kargs = {
    #         'driver':'GTIFF',
    #         'count':1,
    #         'width':2401,
    #         'height':2401,
    #         'crs': 'epsg:4326',
    #         'dtype': 'float32',
    #         'transform': Affine.from_gdal(*geotransform),
    #         'nodata':0

    #         }
    # with rasterio.open('d:/aot_h08_20181006_0250_.tif','w',**kargs) as dst:
    #     dst.write(aot,1)


    #16bit to 8bit  set null value 255 with origin value 255 to 254
    # with rasterio.open(r) as src:
    #     kargs = src.meta.copy()
    #     print(kargs)
    #     kargs.update({
    #             'nodata':255,
    #             'dtype':'uint8'
    #             })
    #     print(src.read(1).max())
    #     with rasterio.open('d:/GF1_WFV1_E119_5_N29_7_20181123_L1A0003619434.tif','w',**kargs) as dst:

    #         for i in range(1,src.count+1):
    #             band_arr = src.read(i)
    #             band_arr[band_arr == 255] = 254
    #             band_arr[band_arr > 255] = 255
    #             dst.write(band_arr.astype('uint8'),i)




