# -*- coding: utf-8 -*-

import io_utils
import os
import json
import warnings

from image_process import cira_strech, new_green_band, bytscl
import glob
from osgeo import gdal
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
    
    json_f = r'D:\RadiometricCorrectionParameter.json'
    with open(json_f) as fh:
        dics = json.load(fh)

    sat = ['GF1']
    sensor = ['WFV1', 'WFV2', 'WFV3', 'WFV4']
    year = ['2020']
    param = ['gain', 'offset']
    
    dics['SatelliteID']['GF1']['WFV1'].update({'2020':{'gain': [0.19319,0.16041,0.12796,0.13405],
                                                       'offset': [0.0, 0.0, 0.0, 0.0]}})
    dics['SatelliteID']['GF1']['WFV1'].update({'ESUN':[1968.63,1849.19,1571.46,1079]})
    dics['SatelliteID']['GF1']['WFV2'].update({'2020':{'gain': [0.2057,0.1648,0.126,0.1187],
                                                       'offset': [0.0, 0.0, 0.0, 0.0]}})
    dics['SatelliteID']['GF1']['WFV2'].update({'ESUN':[1955.11,1847.22,1569.45,1087.87]})
    dics['SatelliteID']['GF1']['WFV3'].update({'2020':{'gain': [0.2106,0.1825,0.1346,0.1187],
                                                       'offset': [0.0, 0.0, 0.0, 0.0]}})
    dics['SatelliteID']['GF1']['WFV3'].update({'ESUN':[1956.62,1840.46,1541.45,1084.06]})
    dics['SatelliteID']['GF1']['WFV4'].update({'2020':{'gain': [0.2522,0.2029,0.1528,0.1031],
                                                       'offset': [0.0, 0.0, 0.0, 0.0]}})
    dics['SatelliteID']['GF1']['WFV4'].update({'ESUN':[1968.08,1841.53,1540.8,1069.6]})

    dics['SatelliteID']['GF1']['PMS1'].update({'ESUN':{'MSS': [1945.29,1854.1,1542.9,1080.77],
                                                       'PAN': [1371.79]}})
    dics['SatelliteID']['GF1']['PMS2'].update({'ESUN':{'MSS': [1945.63,1853.83,1543.9,1081.89],
                                                       'PAN': [1376.37]}})

    dics['SatelliteID']['GF1B']['PMS'].update({'2020':{'MSS':{'gain': [0.0757,0.0618,0.0545,0.0572],   
                                                              'offset': [0.0, 0.0, 0.0, 0.0]},
                                                       'PAN':{'gain': [0.0687],
                                                              'offset': [0.0]}
                                                       }})
    dics['SatelliteID']['GF1B']['PMS'].update({'ESUN':{'MSS': [1945.29,1854.1,1542.9,1080.77],
                                                       'PAN': [1371.79]}})
    dics['SatelliteID']['GF1C']['PMS'].update({'2020':{'MSS':{'gain': [0.0758,0.0657,0.0543,0.0564],   
                                                              'offset': [0.0, 0.0, 0.0, 0.0]},
                                                       'PAN':{'gain': [0.0709],
                                                              'offset': [0.0]}
                                                       }})
    dics['SatelliteID']['GF1C']['PMS'].update({'ESUN':{'MSS': [1931.98,1848.95,1535.31,1063.84],
                                                       'PAN': [1383.91]}})
    dics['SatelliteID']['GF1D']['PMS'].update({'2020':{'MSS':{'gain': [0.0738,0.0656,0.059,0.0585],   
                                                              'offset': [0.0, 0.0, 0.0, 0.0]},
                                                       'PAN':{'gain': [0.0715],
                                                              'offset': [0.0]}
                                                       }})
    dics['SatelliteID']['GF1D']['PMS'].update({'ESUN':{'MSS': [1935.67,1849.93,1556.04,1076.39],
                                                       'PAN': [1395.3]}})

    dics['SatelliteID']['GF2']['PMS1'].update({'2020':{'MSS':{'gain': [0.1378,0.1778,0.17,0.1858],   
                                                              'offset': [0.0, 0.0, 0.0, 0.0]},
                                                       'PAN':{'gain': [0.1817],
                                                              'offset': [0.0]}
                                                       }})
    dics['SatelliteID']['GF2']['PMS1'].update({'ESUN':{'MSS': [1941.76,1853.73,1541.79,1086.47],
                                                       'PAN': [1364.26]}})
    dics['SatelliteID']['GF2']['PMS2'].update({'2020':{'MSS':{'gain': [0.1752,0.1919,0.1804,0.1968],   
                                                              'offset': [0.0, 0.0, 0.0, 0.0]},
                                                       'PAN':{'gain': [0.2025],
                                                              'offset': [0.0]}
                                                       }})
    dics['SatelliteID']['GF2']['PMS2'].update({'ESUN':{'MSS': [1941.22,1853.61,1541.7,1086.53],
                                                       'PAN': [1362.16]}})
    dics['SatelliteID']['GF4'] = {}
    dics['SatelliteID']['GF4']['PMS 2,6,4,6,6'] = {'2020':{'MSS':{'gain':[0.9767,1.0278,0.809,0.5738],'offset':[0.0,0.0,0.0,0.0]},'PAN':{'gain':[0.5329],'offset':[0.0]}}}
    dics['SatelliteID']['GF4']['PMS 4,16,12,16,16'] = {'2020':{'MSS':{'gain':[0.3728,0.3833,0.331,0.2363],'offset':[0.0,0.0,0.0,0.0]},'PAN':{'gain':[0.3293],'offset':[0.0]}}}
    dics['SatelliteID']['GF4']['PMS 6,20,16,20,20'] = {'2020':{'MSS':{'gain':[0.349,0.2719,0.2988,0.2082],'offset':[0.0,0.0,0.0,0.0]},'PAN':{'gain':[0.1733],'offset':[0.0]}}}
    dics['SatelliteID']['GF4']['PMS 6,40,30,40,40'] = {'2020':{'MSS':{'gain':[0.1395,0.1312,0.1203,0.083],'offset':[0.0,0.0,0.0,0.0]},'PAN':{'gain':[0.1725],'offset':[0.0]}}}
    dics['SatelliteID']['GF4']['PMS 8,30,20,30,30'] = {'2020':{'MSS':{'gain':[0.1858,0.2013,0.158,0.1087],'offset':[0.0,0.0,0.0,0.0]},'PAN':{'gain':[0.1266],'offset':[0.0]}}}
    
    
    dics['SatelliteID']['GF4'].update({'PMS':{'ESUN':{'MSS': [1929.556,1839.332,1578.125,1104.766],
                                                       'PAN': [1609.798]}}})
    
    dics['SatelliteID']['GF6'] = {}
    dics['SatelliteID']['GF6']['WFV'] = {'2020':{'gain':[0.0675,0.0552,0.0513,0.0314,0.0519,0.0454,0.0718,0.0596],
                                                 'offset':[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]}}
    dics['SatelliteID']['GF6']['PMS'] = {'2020':{'MSS':{'gain':[0.082,0.0645,0.0489,0.0286],'offset':[0.0,0.0,0.0,0.0]},'PAN':{'gain':[0.0537],'offset':[0.0]}}}
    
    dics['SatelliteID']['GF6']['PMS'].update({'ESUN':{'MSS': [1945.5,1832.38,1558.18,1090.77],
                                                       'PAN': [1497.79]}})
    dics['SatelliteID']['GF6']['WFV'].update({'ESUN':[1952.16,1847.43,1554.76,1074.06,1412,1267.39,1792.64,1736.92]})
    
    dics['SatelliteID']['ZY02C'] = {}
    dics['SatelliteID']['ZY02C']['PMS'] = {'2020':{'MSS':{'gain':[0.733,0.687,0.6252],'offset':[0.0,0.0,0.0]},'PAN':{'gain':[0.6738],'offset':[0.0]}}}
    dics['SatelliteID']['ZY02C']['PMS'].update({'ESUN':{'MSS': [1841.174,1541.93,1086.47],
                                                       'PAN': [1459.70]}})
    
    
    dics['SatelliteID']['ZY302'] = {'2020':{'MSS':{'gain':[0.1787,0.1925,0.2099,0.1798],'offset':[0.0,0.0,0.0,0.0]},'PAN':{'gain':[0.202],'offset':[0.0]}}}
    dics['SatelliteID']['ZY302'].update({'ESUN':{'MSS': [1939.512,1852.459,1552.06,1086.777],
                                                          'BWD': [1504.50],
                                                          'NAD': [1498.36],
                                                          'FWD': [1516.33]}})
    
    dics['SatelliteID']['GFDM'] = {}
    dics['SatelliteID']['GFDM']['2020'] = {'MSS':{'gain':[0.062696,0.07657,0.052356,0.074683,0.105211,0.145286,0.087972,0.064702],
                                                              'offset':[-3.730408,-4.970138,-4.273298,-4.836445,-8.518515,-14.559728,-8.4777,-3.961042],
                                                              '增益模式':[1,1,1,1,8,8,8,6],
                                                              '积分级数':[24,16,18,8,8,4,4,2]},
                                            'PAN':{'gain':[0.071225],
                                                   'offset':[-4.358974],
                                                   '增益模式':[1],
                                                   '积分级数':[32]}
                                                       }
    dics['SatelliteID']['GFDM'].update({'ESUN':{'MSS': [1950.28,1850.84,1663.46,1093.16,1798.07,1724.71,1396.97,886.55],
                                                       'PAN': [1453.90]}})
    
    dics['SatelliteID']['CB04A'] = {}
    dics['SatelliteID']['CB04A']['MUX'] = {'2020':{'gain':[0.97347,1.09124,1.07622,0.87356],
                                                 'offset':[0.0,0.0,0.0,0.0],
                                                 '增益模式':[2,2,2,2],
                                                 '积分级数':[1,1,1,1]},
                                                }
    dics['SatelliteID']['CB04A']['WFI'] = {'2020':{'gain':[0.27127,0.29409,0.2671,0.1851],
                                                        'offset':[0.0,0.0,0.0,0.0],
                                                        '增益模式':[1,1,1,1],
                                                        '积分级数':[1,1,1,1]},
                                                       }
    dics['SatelliteID']['CB04A']['WPM'] = {'2020':{'MSS':{'gain':[0.22724,0.2099,0.15579,0.16928],
                                                              'offset':[0.0,0.0,0.0,0.0],
                                                              '增益模式':[2,3,4,2],
                                                              '积分级数':[2,2,2,2]},
                                                       'PAN':{'gain':[0.16899],
                                                              'offset':[0.0],
                                                              '增益模式':[3],
                                                              '积分级数':[2]}
                                                       }}
    dics['SatelliteID']['CB04A']['WPM'].update({'ESUN':{'MSS': [1940.15,1847.73,1541.19,1081.36],
                                                        'PAN': [1428.13]}})
    dics['SatelliteID']['CB04A']['MUX'].update({'ESUN': [1935.16,1844.53,1573.24,1080.76]})
    dics['SatelliteID']['CB04A']['WFI'].update({'ESUN': [1951.38,1845.78,1590.35,1085.1]})
    # dics['SatelliteID']['CB04A'].update({'WPM':{'ESUN':{'MSS': [1940.15,1847.73,1541.19,1081.36],
    #                                                     'PAN': [1428.13]}},
    #                                      'MUX':{'ESUN': [1935.16,1844.53,1573.24,1080.76]},
    #                                      'WFI':{'ESUN': [1951.38,1845.78,1590.35,1085.1]}})
    dics['SatelliteID']['ZY303'] = {}
    dics['SatelliteID'].update({'ZY303':{'2020':{'FWD':{'gain':[0.23034],
                                                        'offset':[-2.99839],
                                                        '增益模式':[3],
                                                        '积分级数':[12]},
                                                 'NAD':{'gain':[0.20796],
                                                        'offset':[-2.67428],
                                                        '增益模式':[1],
                                                        '积分级数':[24]},
                                                 'BWD':{'gain':[0.23895],
                                                        'offset':[-2.75249],
                                                        '增益模式':[3],
                                                        '积分级数':[12]},
                                                 'MUX':{'gain':[0.20223,0.19506,0.21429,0.21654],
                                                        'offset':[0.0,0.0,0.0,0.0],
                                                        '增益模式':[4,2,4,3],
                                                        '积分级数':[8,8,4,2]},
                                                 }}})
    dics['SatelliteID']['ZY303'].update({'ESUN':{'MSS': [1865.23,1905.01,1716.34,1203.04],
                                                          'BWD': [1604.62],
                                                          'NAD': [1607.13],
                                                          'FWD': [1603.09]}})
    
    dics['SatelliteID']['HJ1A'] = {}
    dics['SatelliteID']['HJ1A']['CCD1'] = {'2019':{'gain':[0.6488,0.6318,1.005,0.9983],
                                                        'offset':[7.325,6.0737,3.6132,1.9028],
                                                       }}
    dics['SatelliteID']['HJ1A']['CCD2'] = {'2020':{'gain':[1.320492,1.345698,0.829058,0.773135],
                                                        'offset':[4.6344,4.0982,3.736,0.7385],
                                                       }}
    dics['SatelliteID']['HJ1A']['CCD1'].update({'ESUN':[1915.06,1825.64,1542.7,1069.58]})
    dics['SatelliteID']['HJ1A']['CCD2'].update({'ESUN':[1930.78,1831.46,1550.25,1073.66]})
    
    # dics['SatelliteID']['HJ1A'].update({'CCD1':{'ESUN':[1915.06,1825.64,1542.7,1069.58]},
    #                                     'CCD2':{'ESUN':[1930.78,1831.46,1550.25,1073.66]}})

    dics['SatelliteID']['GF7'] = {}
    dics['SatelliteID'].update({'GF7':{'2020':{'FWD':{'gain':[0.07886],
                                                      'offset':[-1.99373 ],
                                                      '增益模式':[12],
                                                      '积分级数':[32]},
                                               'BWD':{'gain':[0.08032],
                                                      'offset':[-2.00017],
                                                      '增益模式':[2],
                                                      '积分级数':[32]},
                                               'MUX':{'B1_1': {'gain':[0.65856],
                                                               'offset':[-1.03733],
                                                               '增益模式':[1],
                                                               '积分级数':[32]},
                                                      'B1_2': {'gain':[0.08628],
                                                               'offset':[-1.03733],
                                                               '增益模式':[1],
                                                               '积分级数':[24]},
                                                      'B2_1': {'gain':[0.07315],
                                                               'offset':[-1.75698],
                                                               '增益模式':[2],
                                                               '积分级数':[16]},
                                                      'B2_2': {'gain':[0.09395],
                                                               'offset':[-1.75698],
                                                               '增益模式':[1],
                                                               '积分级数':[16]},
                                                      'B3': {'gain':[0.07339],
                                                             'offset':[-1.91726],
                                                             '增益模式':[2],
                                                             '积分级数':[12]},
                                                      'B4_1': {'gain':[0.06985],
                                                               'offset':[-1.81477],
                                                               '增益模式':[1],
                                                               '积分级数':[8]},
                                                      'B4_2': {'gain':[0.09087],
                                                               'offset':[-1.81477],
                                                               '增益模式':[3],
                                                               '积分级数':[4]},
                                                      }
                                                 }}})

    dics['SatelliteID']['GF7'].update({'ESUN':{'MUX': [1929.44,1843.61,1554.83,1081.34],
                                                      'BWD': [1414.92],
                                                      'FWD': [1394.19]}})

    dics['SatelliteID'].update({'ZY02D':{'BEFORE_ADJUST':{'2020':{'MSS':{'gain':[0.05126,0.0436,0.04049,0.04429,0.05636,0.03908,0.04844,0.02811],
                                                                         'offset':[-2.81333,-2.61122,-2.28339,-2.61762,-2.84846,-1.12028,-1.66111,-0.87143],
                                                                         '增益模式':[4,2,3,3,3,2,3,3],
                                                                         '积分级数':[1,2,1,1,4,3,2,3]},
                                                                  'PAN':{'gain':[0.0447],
                                                                         'offset':[-2.41865],
                                                                         '增益模式':[4],
                                                                         '积分级数':[1]}
                                                       }},
                                         'AFTER_ADJUST':{'2020':{'MSS':{'gain':[0.07644,0.06103,0.05031,0.05638,0.06953,0.05636,0.05838,0.03493],
                                                                         'offset':[-3.25182,-3.38396,-2.63118,-3.23643,-2.8924,-2.21431,-1.59299,-0.89641],
                                                                         '增益模式':[2,4,2,2,2,4,2,2],
                                                                         '积分级数':[1,1,1,1,4,2,2,3]},
                                                                  'PAN':{'gain':[0.06693],
                                                                         'offset':[-2.58546],
                                                                         '增益模式':[2],
                                                                         '积分级数':[1]}
                                                       }}}})
    
    dics['SatelliteID']['ZY02D'].update({'ESUN':{'MSS': [1941.75,1838.56,1543.71,1069.6,1763.65,1706.09,1323.62,857.34],
                                                      'PAN': [1399.57]
                                                      }})
    
    json_f_new = r'D:\RadiometricCalibrationParameter.json'
    with open(json_f_new, 'w') as fh:
        json.dump(dics, fh, indent="  ")
        # dics = json.load(fh)

    
    # tie_files = glob.glob(os.path.join(r"D:\temp11\tie", '*.tie'))
    # first_file = tie_files[0]
    # out_file = os.path.join(os.path.dirname(first_file), 'bundle1.tie')
    # if os.path.exists(out_file):
    #     os.remove(out_file)
    # n_line = 0
    # for i, t in enumerate(tie_files):
    #     with open(t) as fh:
    #         first_line = fh.readline()
            
    #         if i == 0:
    #             ties = fh.readlines()
    #         else:
    #             line_num = n_line
    #             for l in fh.readlines():
    #                 ties.append("{:5d}".format(line_num) + l[5:])
    #                 line_num += 1
    #         n_line += int(first_line)

    # with open(out_file, 'a') as fo:
    #     fo.write('{:10d}\n'.format(n_line))
    #     fo.writelines(ties)
        
                
        
    
    
    

#    with open('tests/data/RGB.byte.tif', 'rb') as f, MemoryFile(f) as memfile:
#    with memfile.open() as src:
#        pprint.pprint(src.profile)

    # extract_by_mask(e, r, 'd:/temp/test1.tif', nodata=0)
    # with rasterio.open('d:/temp/test1.tif') as src:
    #     kargs = src.meta.copy()
    #     arr = src.read(1)
    #     arr[arr==2] = 1
    #     with rasterio.open('d:/temp/test2.tif', 'w', **kargs) as dst:
    #         dst.write(arr, 1)
    # import datetime
    # nowtime = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    # merge_rio([rasterio.open(f) for f in ['d:/temp/test2.tif', r]],
    #            r'E:\work\海冰\code\result\tif\SeaIce_NP_20171008_20171015_{}.tif'.format(nowtime),
    #            res=None, nodata=0)


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




