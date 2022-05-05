#!/usr/bin/python
# coding:utf-8

import os
import datetime
import xml.etree.ElementTree as ET

class SatMeta(object):
    """解析卫星元数据"""
    sat_id = None
    sensor = None

    datetime = None

    solar_zenith = None
    solar_azimuth = None
    sat_zenith = None
    sat_azimuth = None

    ul_lat = None  # Upper Left
    ul_long = None
    ur_lat = None  # Upper Right
    ur_long = None
    lr_lat = None  # Lower Right
    lr_long = None
    ll_lat = None  # Lower Left
    ll_long = None

    def __init__(self, meta_file, sat_type=None):
        self.meta_file = meta_file
        self.sat_type = sat_type

        if sat_type is None:  # 根据名称解析
            m_file_name = os.path.basename(self.meta_file)
            if m_file_name.lower().startswith('gf2') or m_file_name.lower().startswith('gf1'):
                self.parse_gf2_meta()
        else:
            if sat_type.lower().startswith('gf2') or sat_type.lower().startswith('gf1'):
                self.parse_gf2_meta()

    def parse_gf2_meta(self):
        """parse GF2 or GF1 satellite params"""
        tree = ET.parse(self.meta_file)
        root = tree.getroot()
        # coords = list(map(get_element_text,
        #                   [root]*8,
        #                   ['TopLeftLatitude', 'TopRightLatitude',
        #                    'BottomRightLatitude', 'BottomLeftLatitude',
        #                    'TopLeftLongitude', 'TopRightLongitude', 
        #                    'BottomRightLongitude', 'BottomLeftLongitude']
        #                   ))
        # [self.ul_lat, self.ur_lat, self.lr_lat, self.ll_lat,
        #  self.ul_long, self.ur_long, self.lr_long, self.ll_long] = coords

        for ele in root.iter():
            if ele.tag == 'SatelliteID':
                self.sat_id = ele.text
            elif ele.tag == 'SensorID':
                self.sensor = ele.text
            elif ele.tag == 'ReceiveTime':
                self.datetime = ele.text
            elif ele.tag == 'SolarAzimuth':
                self.solar_azimuth = ele.text
            elif ele.tag == 'SolarZenith':
                self.solar_zenith = ele.text
            elif ele.tag == 'SatelliteAzimuth':
                self.sat_azimuth = ele.text
            elif ele.tag == 'SatelliteZenith':
                self.sat_zenith = ele.text
            elif ele.tag == 'TopLeftLatitude':
                self.ul_lat = ele.text
            elif ele.tag == 'TopLeftLongitude':
                self.ul_long = ele.text
            elif ele.tag == 'TopRightLatitude':
                self.ur_lat = ele.text
            elif ele.tag == 'TopRightLongitude':
                self.ur_long = ele.text
            elif ele.tag == 'BottomRightLatitude':
                self.lr_lat = ele.text
            elif ele.tag == 'BottomRightLongitude':
                self.lr_long = ele.text
            elif ele.tag == 'BottomLeftLatitude':
                self.ll_lat = ele.text
            elif ele.tag == 'BottomLeftLongitude':
                self.ll_long = ele.text


class XQSatSpectralSettingParser(object):
    """解析XQRadiometricCorrectionSetting.xml文件,获取辐射定标及大气校正参数."""
    # 定标参数
    gain = []
    bias = []
    esun = []
    # 大气校正参数
    sr_min = []
    sr_max = []
    srf_path = None

    def __init__(self, setting_file, sat_id=None, sensor_type=None, is_atmos_correction=True):
        sats = ['GF1', 'GF1B', 'GF1C', 'GF1D', 'GF2', 'GF6', 'ZY3_01',
                'ZY3_02', 'ZY3_03', 'GFDM', 'GF7', 'GF4', 'CBERS_04A', 'CBERS_04',
                'HJ_1B', 'HJ_1A', 'ZY1_02D', 'ZY1_02C']
    
        sensors = ['GF1_MSS1', 'GF1_MSS2', 'GF1_WFV1', 'GF1_WFV2', 'GF1_WFV3', 'GF1_WFV4',
                   'GF2_MSS1', 'GF2_MSS2',
                   'GF6_MSS', 'GF6_WFV',
                   'CBERS_04A_WPM', 'CBERS_04A_MUX', 'CBERS_04A_WFI',
                   'CBERS_04_P5M', 'CBERS_04_MUX', 'CBERS_04_WFI', 'CBERS_04_P10',
                   'HJ_1B_CCD1', 'HJ_1B_CCD2',
                   'HJ_1A_CCD1', 'HJ_1A_CCD2']
        self.setting_file = setting_file
        self.sat_id = sat_id
        self.sensor_type = sensor_type
        self.is_atmos_correction =is_atmos_correction

        if sensor_type is not None and sensor_type in sensors:
            self.parse_sensor()  # 解析卫星下一层传感器参数
        if sensor_type is None and sat_id in sats:
            self.parse_only_sat()  # 解析卫星名称下
    
    def parse_sensor(self):
        self.parse_settings(self.sensor_type)

    def parse_only_sat(self):
        self.parse_settings(self.sat_id)

    def parse_settings(self, element):
        """解析xml"""
        tree = ET.parse(self.setting_file)
        root = tree.getroot()
        for ele in root.iter():
            if ele.tag == element:
                # srf file name
                self.srf_path = ele.find('SRFName').text

                # spectral range
                sr_ele = ele.find('SpectralRange')
                for e in sr_ele.iter():
                    if e.tag == 'SRMin':
                        self.sr_min.append(e.text)
                    elif e.tag == 'SRMax':
                        self.sr_max.append(e.text)

                # esun
                esun_ele = ele.find('ESUNS')
                for e in esun_ele.iter():
                    if e.tag == 'esun':
                        self.esun.append(e.text)
                
                # gain bias
                cur_year = datetime.datetime.now().year
                # 匹配最新年份
                gb_tag = f"Year{cur_year}"
                while ele.find(gb_tag) is None:
                    cur_year -= 1
                    gb_tag = f"Year{cur_year}"
                
                print(gb_tag)

                # f"Year{cur_year}"
                gb_ele = ele.find(gb_tag)
                for e in gb_ele.iter():
                    if e.tag == 'gain':
                        self.gain.append(e.text)
                    elif e.tag == 'bias':
                        self.bias.append(e.text)      

def get_element_text(node, element):
    """lambda x: float(root.find(x).text)"""
    return float(node.find(element).text)

if __name__ == '__main__':
    meta = SatMeta(r"D:\work\data\影像样例\GF2\GF2_PMS1_E108.9_N34.2_20181026_L1A0003549596\GF2_PMS1_E108.9_N34.2_20181026_L1A0003549596-MSS1.xml",
        None)

    # print(meta.ul_lat, meta.ll_lat, meta.sat_id, meta.sat_zenith, meta.datetime)
    s = XQSatSpectralSettingParser(r'D:\temp11\XQRadiometricCorrectionSetting.xml',
        'ZY1_02C', None)
    print(s.sr_min, s.sr_max, s.srf_path, s.bias, s.esun, s.gain)